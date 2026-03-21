from tqdm import tqdm
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import itertools

sys.path.append("../server")
from yolah import Yolah, Move, Square

torch.set_float32_matmul_precision('high')

# Raw board encoding: 64 black + 64 white + 64 empty + 1 turn = 193
BOARD_SIZE = 64 + 64 + 64 + 1
# Features: 44 uint8 values
NB_FEATURES = 44
# Combined input size
INPUT_SIZE = BOARD_SIZE + NB_FEATURES  # 237
# Each features record: 44 uint8 features + 1 uint8 result
RECORD_SIZE = NB_FEATURES + 1  # 45 bytes


def bitboard64_to_list(n):
    b = [int(digit) for digit in bin(n)[2:]]
    return [0] * (64 - len(b)) + b


def encode_yolah(yolah):
    res = list(itertools.chain.from_iterable([
        bitboard64_to_list(yolah.black),
        bitboard64_to_list(yolah.white),
        bitboard64_to_list(yolah.empty),
        [Yolah.WHITE_PLAYER if yolah.nb_plies() & 1 else Yolah.BLACK_PLAYER]
    ]))
    return torch.tensor(res, dtype=torch.float32)


def encode_moves(moves):
    yolah = Yolah()
    for sq1, sq2 in zip(moves[0::2], moves[1::2]):
        m = Move(Square(int(sq1)), Square(int(sq2)))
        yolah.play(m)
    return encode_yolah(yolah)


class CombinedDataset(Dataset):
    """
    Loads pairs of (game_file, features_file) from data_dir.
    Each features record: 44 uint8 features + 1 uint8 result (45 bytes).
    Each game record produces nb_moves+1-nb_random_moves positions, each with
    a corresponding features record.

    Assumes features files are named identically to game files but with
    '.features.txt' suffix (e.g., games_0001.features.txt for games_0001).
    """
    def __init__(self, data_dir):
        features_files = sorted(glob.glob(data_dir + "/*.features.txt"))
        if not features_files:
            raise FileNotFoundError(f"No .features.txt files found in {data_dir}")

        self.entries = []   # (cumulative_end, nb_random_moves, moves_bytes, mmap_idx, feat_start)
        self.mmaps = []
        total = 0

        for feat_path in features_files:
            game_path = feat_path.replace('.features.txt', '')
            if not os.path.isfile(game_path):
                print(f"Warning: game file not found for {feat_path}, skipping", flush=True)
                continue

            feat_size = os.path.getsize(feat_path)
            nb_records = feat_size // RECORD_SIZE
            mm = np.memmap(feat_path, dtype=np.uint8, mode='r', shape=(nb_records, RECORD_SIZE))
            mmap_idx = len(self.mmaps)
            self.mmaps.append(mm)

            feat_idx = 0
            with open(game_path, 'rb') as f:
                data = f.read()
            idx = 0
            while idx < len(data) and feat_idx < nb_records:
                nb_moves = int(data[idx])
                nb_random_moves = int(data[idx + 1])
                nb_positions = nb_moves + 1 - nb_random_moves
                moves = data[idx + 2: idx + 2 + 2 * nb_moves]
                total += nb_positions
                self.entries.append((total, nb_random_moves, moves, mmap_idx, feat_idx))
                feat_idx += nb_positions
                idx += 2 + 2 * nb_moves + 2

            print(f"{feat_path}: {feat_idx} records", flush=True)

        self.size = total
        print(f"Total: {total} combined records", flush=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Binary search to find the game entry containing this index
        lo, hi = 0, len(self.entries)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.entries[mid][0] <= idx:
                lo = mid
            else:
                hi = mid
        cum_end, nb_random_moves, moves, mmap_idx, feat_start = self.entries[lo]
        base = self.entries[lo - 1][0] if lo > 0 else 0
        pos_idx = idx - base  # position index within this game

        # Board encoding: replay moves up to position pos_idx + nb_random_moves
        board_tensor = encode_moves(moves[: 2 * (nb_random_moves + pos_idx)])

        # Features encoding from memmap
        record = self.mmaps[mmap_idx][feat_start + pos_idx]
        feat_tensor = torch.tensor(record[:NB_FEATURES], dtype=torch.float32) / 255.0
        label = int(record[NB_FEATURES])

        combined = torch.cat([board_tensor, feat_tensor], dim=0)
        return combined, torch.tensor(label, dtype=torch.long)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # BatchNorm only on the features slice — raw board is already binary {0,1}
        self.bn = nn.BatchNorm1d(NB_FEATURES)
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        board = x[:, :BOARD_SIZE]
        feats = self.bn(x[:, BOARD_SIZE:])
        x = torch.cat([board, feats], dim=1)
        x = torch.clamp(self.fc1(x), min=0.0, max=1.0)
        x = torch.clamp(self.fc2(x), min=0.0, max=1.0)
        x = torch.clamp(self.fc3(x), min=0.0, max=1.0)
        return self.fc4(x)

    def clip(self):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
            fc.weight.data.clamp_(-127/64, 127/64)
            fc.bias.data.clamp_(-127/64, 127/64)


class NetNoBN(nn.Module):
    """Same architecture as Net but without BatchNorm — for C++ inference.
    At inference, the input is simply cat([board_193, features_44]) without
    any normalization; the BN transform has been folded into fc1.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), min=0.0, max=1.0)
        x = torch.clamp(self.fc2(x), min=0.0, max=1.0)
        x = torch.clamp(self.fc3(x), min=0.0, max=1.0)
        return self.fc4(x)


def fold_batchnorm_partial(bn, fc, col_start, col_end):
    """
    Folds a BatchNorm1d layer into a subset of columns [col_start:col_end]
    of a Linear layer in-place, using the BN's frozen running statistics.

    This handles the case where BN is applied to a slice of the input before
    it is concatenated with other features and passed through fc.

    Must be called with the model in eval mode (model.train(False)).
    """
    scale = bn.weight / (bn.running_var + bn.eps).sqrt()  # γ / √(σ²+ε)
    shift = bn.bias - bn.running_mean * scale              # β - μ·scale
    # Bias update must use the original feature-column weights, before scaling
    fc.bias.data  += fc.weight.data[:, col_start:col_end] @ shift
    fc.weight.data[:, col_start:col_end] *= scale.unsqueeze(0)


def export_net(net):
    """
    Returns a NetNoBN with BN folded into the feature columns of fc1,
    ready for C++ export.  Does not modify the original net.

    After export, the network expects a flat 237-d input:
      cat([black_64, white_64, empty_64, turn_1, features_44])
    with features already divided by 255 (as during training).
    """
    net.train(False)
    exported = NetNoBN()
    for attr in ['fc1', 'fc2', 'fc3', 'fc4']:
        src = getattr(net, attr)
        dst = getattr(exported, attr)
        dst.weight.data = src.weight.data.clone()
        dst.bias.data   = src.bias.data.clone()
    fold_batchnorm_partial(net.bn, exported.fc1, BOARD_SIZE, INPUT_SIZE)
    return exported


NB_EPOCHS = 300
MODEL_PATH = "/mnt/"
MODEL_NAME = "combined_256x64x32x3"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"
DATA_DIR = "./data"


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65434"
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        self.loss_fn = nn.CrossEntropyLoss()
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
            self.model.module.clip()
        self.scheduler.step()
        if self.gpu_id == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            print('epoch {} train loss: {} train accuracy: {} lr: {:.6f}'.format(
                epoch + 1, running_loss / n, accuracy / n, current_lr), flush=True)

    def _validate(self, epoch):
        self.model.train(False)
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
    if rank == 0:
        print(len(dataset), flush=True)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    if rank == 0:
        print(f'Train size: {train_size}, Val size: {val_size}', flush=True)

    train_loader, sampler_train, val_loader, sampler_val = dataloader_ddp(trainset, valset, batch_size)
    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        print(net, flush=True)
    trainer = TrainerDDP(rank, net, train_loader, sampler_train, val_loader, sampler_val)
    trainer.train(NB_EPOCHS)
    destroy_process_group()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    dataset = CombinedDataset(DATA_DIR)
    mp.spawn(main, args=(world_size, 512 * 4, dataset), nprocs=world_size)
