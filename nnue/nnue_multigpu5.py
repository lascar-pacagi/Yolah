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
        res = list(itertools.chain.from_iterable([
                    bitboard64_to_list(yolah.black), 
                    bitboard64_to_list(yolah.white), 
                    bitboard64_to_list(yolah.empty),                    
                    [Yolah.WHITE_PLAYER if yolah.nb_plies() & 1 else Yolah.BLACK_PLAYER]]))
        return torch.tensor(res, dtype=torch.float32)

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
        #print(n, moves, lo, r)
        return GameDataset.encode(moves[: 2 * (r + idx - n)]), torch.tensor(self.outputs[lo], dtype=torch.long)

    def get_memory_usage(self):
        """Returns memory usage in bytes and MB"""
        inputs_bytes = 0
        for sum_val, r, moves in self.inputs:
            inputs_bytes += sys.getsizeof(sum_val) + sys.getsizeof(r) + len(moves)

        # outputs are now Python ints (28 bytes each with overhead)
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

# black positions + white positions + empty positions + turn
INPUT_SIZE = 64 + 64 + 64 + 1

class Net(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, l1_size=1024, l2_size=64, l3_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc2(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc3(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return self.fc4(x)#softmax(self.fc4(x), dim=1)#self.fc4(x)
    
    def clip(self):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
            fc.weight.data.clamp_(-127/64, 127/64)
            fc.bias.data.clamp_(-127/64, 127/64)

NB_EPOCHS=300
MODEL_PATH="./"
#MODEL_PATH="/mnt/"
MODEL_NAME="nnue_1024x64x32x3_2"
LAST_MODEL=f"{MODEL_PATH}{MODEL_NAME}.pt"
GAME_DIR="./data"

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def dataloader_ddp(trainset, valset, batch_size):
    sampler_train = DistributedSampler(trainset)
    sampler_val = DistributedSampler(valset, shuffle=False)
    # train_loader = DataLoader(
    #     trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train,
    #     num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=False#, collate_fn=custom_collate
    # )
    # val_loader = DataLoader(
    #     valset, batch_size=batch_size, shuffle=False, sampler=sampler_val,
    #     num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=False#, collate_fn=custom_collate
    # )
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
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # Learning rate scheduler: exponential decay
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.model = torch.compile(self.model)
        # Create CUDA stream for asynchronous data transfer
        self.stream = torch.cuda.Stream(device=gpu_id)
        
    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0
        accuracy = 0
        for (X, y) in tqdm(self.train_loader):
            n += len(X)
            # X = X.to(self.gpu_id)
            # y = y.to(self.gpu_id)
            # Use CUDA stream for asynchronous data transfer
            with torch.cuda.stream(self.stream):
                X = X.to(self.gpu_id, non_blocking=True)
                y = y.to(self.gpu_id, non_blocking=True)
            # Ensure transfer is complete before computation
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)

            self.optimizer.zero_grad()
            logits = self.model(X)
            #y = y.to(self.gpu_id, dtype=torch.long)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            accuracy += sum(torch.argmax(logits, dim=1) == y).item()
            self.model.module.clip()
        # Step the scheduler after each epoch
        self.scheduler.step()
        if self.gpu_id == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            print('epoch {} train loss: {} train accuracy: {} lr: {:.6f}'.format(epoch + 1, running_loss / n, accuracy / n, current_lr), flush=True)

    def _validate(self, epoch):
        self.model.eval()
        n = 0
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for (X, y) in tqdm(self.val_loader):
                n += len(X)
                # X = X.to(self.gpu_id)
                # y = y.to(self.gpu_id)
                # Use CUDA stream for asynchronous data transfer
                with torch.cuda.stream(self.stream):
                    X = X.to(self.gpu_id, non_blocking=True)
                    y = y.to(self.gpu_id, non_blocking=True)
                # # Ensure transfer is complete before computation
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)

                logits = self.model(X)
                #y = y.to(self.gpu_id, dtype=torch.long)
                loss = self.loss_fn(logits, y)
                val_loss += loss.item()
                val_accuracy += sum(torch.argmax(logits, dim=1) == y).item()
        if self.gpu_id == 0:
            print('epoch {} val loss: {} val accuracy: {}'.format(epoch + 1, val_loss / n, val_accuracy / n), flush=True)
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
    dataset = GameDataset(GAME_DIR)
    mp.spawn(main, args=(world_size, 512 * 2, dataset), nprocs=world_size)
