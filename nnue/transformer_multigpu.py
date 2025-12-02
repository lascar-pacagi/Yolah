from tqdm import tqdm
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.functional import softmax
import os
import sys
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

sys.path.append("../server")
from yolah import Yolah, Move, Square

torch.set_float32_matmul_precision('high')

def bit_not(n, numbits=64):
    return (1 << numbits) - 1 - n

def bitboard64_to_list(n):
    b = [int(digit) for digit in bin(n)[2:]]
    return [0]*(64 - len(b)) + b

class GameDataset(Dataset):
    def __init__(self, games_dir, max_workers=15, use_processes=False):
        self.inputs = []
        self.outputs = []
        files = glob.glob(games_dir + "/games*")
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        inputs = []
        with executor_class(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_file, filename) for filename in files]
            for future in futures:
                file_inputs, file_outputs = future.result()
                inputs.extend(file_inputs)
                self.outputs.extend(file_outputs)

        sum = 0
        for (nb_positions, r, moves) in inputs:
            sum += nb_positions
            self.inputs.append((sum, r, moves))
        self.size = sum
        mem = self.get_memory_usage()
        print(f"Inputs:  {mem['inputs_mb']:.2f} MB")
        print(f"Outputs: {mem['outputs_mb']:.2f} MB")
        print(f"Total:   {mem['total_mb']:.2f} MB")

    def _process_file(self, filename):
        inputs = []
        outputs = []
        print(filename)
        with open(filename, 'rb') as f:
            idx = 0
            data = f.read()
            print(len(data))
            while idx < len(data):
                nb_moves = int(data[idx])
                nb_random_moves = int(data[idx + 1])
                if nb_random_moves == nb_moves:
                    idx += 2 + 2 * nb_moves + 2
                    continue
                black_score = int(data[idx + 2 + 2 * nb_moves])
                white_score = int(data[idx + 2 + 2 * nb_moves + 1])
                res = 0
                if black_score == white_score:
                    res = 1
                if white_score > black_score:
                    res = 2
                inputs.append((nb_moves + 1, nb_random_moves, data[idx+2:idx+2+2*nb_moves]))
                outputs.append(res)
                idx += 2 + 2 * nb_moves + 2
        return inputs, outputs

    @staticmethod
    def encode_yolah(yolah):
        """
        Encode the board as tokens:
        - For each square i (0-63): token = i * 4 + state
          where state is: 0=vacant, 1=black, 2=white, 3=empty
          This gives tokens in range [0, 255] for squares
        - Turn token: 256=black's turn, 257=white's turn
        Total vocabulary: 64 * 4 + 2 = 258 tokens
        Sequence length: 65 (64 squares + 1 turn token)
        """
        black_bb = bitboard64_to_list(yolah.black)
        white_bb = bitboard64_to_list(yolah.white)
        empty_bb = bitboard64_to_list(yolah.empty)

        # Create tokens for each square
        # Token = square_index * 4 + state
        tokens = []
        for i in range(64):
            if black_bb[i]:
                state = 1  # black piece
            elif white_bb[i]:
                state = 2  # white piece
            elif empty_bb[i]:
                state = 3  # empty (hole)
            else:
                state = 0  # vacant

            token = i * 4 + state
            tokens.append(token)

        # Add turn token (256=black's turn, 257=white's turn)
        turn_token = 256 if (yolah.nb_plies() & 1) == 0 else 257
        tokens.append(turn_token)

        return torch.tensor(tokens, dtype=torch.long)

    @staticmethod
    def encode(moves):
        yolah = Yolah()
        for sq1, sq2 in zip(moves[0::2], moves[1::2]):
            m = Move(Square(int(sq1)), Square(int(sq2)))
            yolah.play(m)
        return GameDataset.encode_yolah(yolah)

    def get_infos(self):
        return self.infos

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        lo = 0
        hi = len(self.inputs)
        while lo < hi:
            m = lo + (hi - lo) // 2
            n, _, _ = self.inputs[m]
            if n <= idx:
                lo = m + 1
            else:
                hi = m
        n = self.inputs[lo - 1][0] if lo > 0 else 0
        _, r, moves = self.inputs[lo]
        return GameDataset.encode(moves[: 2 * (r + idx - n)]), torch.tensor(self.outputs[lo], dtype=torch.long)

    def get_memory_usage(self):
        """Returns memory usage in bytes and MB"""
        inputs_bytes = 0
        for sum_val, r, moves in self.inputs:
            inputs_bytes += sys.getsizeof(sum_val) + sys.getsizeof(r) + len(moves)

        outputs_bytes = sys.getsizeof(self.outputs[0]) * len(self.outputs)
        total_bytes = inputs_bytes + outputs_bytes

        return {
            'inputs_mb': inputs_bytes / (1024**2),
            'outputs_mb': outputs_bytes / (1024**2),
            'total_mb': total_bytes / (1024**2),
            'inputs_bytes': inputs_bytes,
            'outputs_bytes': outputs_bytes,
            'total_bytes': total_bytes
        }


class TransformerNet(nn.Module):
    """
    Transformer-based network for Yolah board evaluation.

    Architecture:
    - Token embedding: Maps each token (square+state or turn) to d_model dimensions
      - Square tokens: 0-255 (64 squares × 4 states)
      - Turn tokens: 256-257 (black turn, white turn)
      - Total vocab: 258
    - Transformer encoder: Multiple self-attention layers (no positional encoding needed)
    - Output head: Projects to 3 classes (black win, draw, white win)

    Note: No positional encoding is needed because each token already encodes both
    the square position and its state.
    """
    def __init__(self,
                 vocab_size=258,    # 64 squares * 4 states + 2 turn states
                 d_model=512,       # embedding dimension
                 nhead=8,           # number of attention heads
                 num_layers=8,      # number of transformer layers
                 dim_feedforward=1024, # feedforward network dimension
                 dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Single embedding layer for all tokens (squares and turn)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.fc_out = nn.Linear(d_model, 3)  # 3 classes: black win, draw, white win

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 65) with token indices
               - x[:, 0:64] are square tokens (range 0-255)
                 Token = square_index * 4 + state
                 where state: 0=vacant, 1=black, 2=white, 3=empty
               - x[:, 64] is turn token (256=black turn, 257=white turn)

        Returns:
            Logits of shape (batch_size, 3)
        """
        # Embed all tokens using single embedding layer
        x = self.token_embedding(x)  # (batch_size, 65, d_model)

        # Pass through transformer encoder (no mask needed - full attention)
        # No positional encoding needed since token position = square position
        x = self.transformer_encoder(x)  # (batch_size, 65, d_model)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Output layer
        x = self.fc_out(x)  # (batch_size, 3)

        return x


NB_EPOCHS=50
#MODEL_PATH="./"
MODEL_PATH="/mnt/"
MODEL_NAME="transformer_d512_h8_l8"
LAST_MODEL=f"{MODEL_PATH}{MODEL_NAME}.pt"
GAME_DIR="./data"

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def dataloader_ddp(trainset, valset, batch_size):
    sampler_train = DistributedSampler(trainset)
    sampler_val = DistributedSampler(valset, shuffle=False)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train,
        num_workers=2, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, sampler=sampler_val,
        num_workers=0, pin_memory=True
    )
    return train_loader, sampler_train, val_loader, sampler_val

class TrainerDDP:
    def __init__(self, gpu_id, model, train_loader, sampler_train, val_loader, sampler_val, save_every=5):
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.sampler_train = sampler_train
        self.val_loader = val_loader
        self.sampler_val = sampler_val
        self.save_every = save_every
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.model = torch.compile(self.model)
        self.stream = torch.cuda.Stream(device=gpu_id)

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0
        accuracy = 0
        for (X, y) in tqdm(self.train_loader):
            n += len(X)
            with torch.cuda.stream(self.stream):
                X = X.to(self.gpu_id, non_blocking=True)
                y = y.to(self.gpu_id, non_blocking=True)
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            accuracy += sum(torch.argmax(logits, dim=1) == y).item()

        self.scheduler.step()
        if self.gpu_id == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            print('epoch {} train loss: {} train accuracy: {} lr: {:.6f}'.format(
                epoch + 1, running_loss / n, accuracy / n, current_lr), flush=True)

    def _validate(self, epoch):
        self.model.eval()
        n = 0
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for (X, y) in tqdm(self.val_loader):
                n += len(X)
                with torch.cuda.stream(self.stream):
                    X = X.to(self.gpu_id, non_blocking=True)
                    y = y.to(self.gpu_id, non_blocking=True)
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)

                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                val_loss += loss.item()
                val_accuracy += sum(torch.argmax(logits, dim=1) == y).item()
        if self.gpu_id == 0:
            print('epoch {} val loss: {} val accuracy: {}'.format(
                epoch + 1, val_loss / n, val_accuracy / n), flush=True)
        self.model.train()

    def train(self, nb_epochs):
        self.model.train()
        for epoch in range(nb_epochs):
            self.sampler_train.set_epoch(epoch)
            self._run_epoch(epoch)
            self._validate(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(nb_epochs - 1)

def main(rank, world_size, batch_size, dataset):
    ddp_setup(rank, world_size)
    print(rank)
    if rank == 0:
        print(len(dataset), flush=True)

    # Split dataset into train and validation (90/10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    if rank == 0:
        print(f'Train size: {train_size}, Val size: {val_size}', flush=True)

    train_loader, sampler_train, val_loader, sampler_val = dataloader_ddp(trainset, valset, batch_size)

    # Create transformer model
    net = TransformerNet(
        vocab_size=258,         # 64 squares * 4 states + 2 turn states
        d_model=512,            # embedding dimension
        nhead=8,                # attention heads
        num_layers=8,           # transformer layers
        dim_feedforward=1024,   # FFN dimension
        dropout=0.1
    )

    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        print(net, flush=True)
        # Count parameters
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}', flush=True)
        print(f'Trainable parameters: {trainable_params:,}', flush=True)

    trainer = TrainerDDP(rank, net, train_loader, sampler_train, val_loader, sampler_val)
    trainer.train(NB_EPOCHS)
    destroy_process_group()

if __name__ == "__main__":
    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    dataset = GameDataset(GAME_DIR)
    mp.spawn(main, args=(world_size, 512, dataset), nprocs=world_size)
