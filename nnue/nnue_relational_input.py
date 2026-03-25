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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

sys.path.append("../server")
from yolah import Yolah, Move, Square

torch.set_float32_matmul_precision('high')

# ── Relational input encoding ──────────────────────────────────────────────────
#
# For every ordered pair of squares (i, j) ∈ [0,64)², we encode the *joint*
# state of both squares as a one-hot vector of length NB_TYPES².
#
# Square types:
#   BLACK = 0   — square occupied by a black piece
#   WHITE = 1   — square occupied by a white piece
#   FREE  = 2   — empty square (reachable)
#   HOLE  = 3   — off-board cell (neither black, white, nor empty)
#
# Feature index for pair (i, j):
#   (i * 64 + j) * 16  +  sq_type[i] * 4  +  sq_type[j]
#
# Total inputs: 64 × 64 × 16 + 1 (turn) = 65,537
# Active inputs per position: 64 × 64 = 4,096  (one per pair — very sparse)

NB_TYPES   = 4
INPUT_SIZE = 64 * 64 * NB_TYPES * NB_TYPES + 1   # 65,537

# Precomputed constants for vectorised encoding (module-level, built once)
_BITS = np.uint64(1) << np.arange(64, dtype=np.uint64)
_BASE = (
    np.arange(64, dtype=np.int64)[:, None] * 64 +
    np.arange(64, dtype=np.int64)[None, :]
) * (NB_TYPES * NB_TYPES)    # shape (64, 64)


def encode_relational(yolah) -> torch.Tensor:
    """
    Encode a Yolah position as a sparse 65,537-d binary float32 tensor.

    For each ordered pair of squares (i, j), exactly one of the 16 type-pair
    features is set to 1.  The last element encodes the turn.
    """
    is_black = (_BITS & np.uint64(yolah.black)).astype(bool)   # (64,)
    is_white = (_BITS & np.uint64(yolah.white)).astype(bool)
    is_empty = (_BITS & np.uint64(yolah.empty)).astype(bool)

    sq_type = np.where(is_black, 0,
               np.where(is_white, 1,
               np.where(is_empty, 2, 3))).astype(np.int64)    # (64,)

    # Pair type index in [0, 15]: sq_type[i]*4 + sq_type[j]
    pair_type = sq_type[:, None] * NB_TYPES + sq_type[None, :]  # (64, 64)
    indices   = (_BASE + pair_type).ravel()                      # (4096,)

    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    x[indices] = 1.0
    x[-1] = 1.0 if yolah.nb_plies() & 1 else 0.0
    return torch.from_numpy(x)


# ── Dataset ────────────────────────────────────────────────────────────────────
class RelationalGameDataset(Dataset):
    def __init__(self, games_dir, max_workers=15, use_processes=False):
        self.inputs  = []
        self.outputs = []
        files = glob.glob(games_dir + "/games*")
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        raw_inputs = []
        with executor_class(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_file, f) for f in files]
            for future in futures:
                file_inputs, file_outputs = future.result()
                raw_inputs.extend(file_inputs)
                self.outputs.extend(file_outputs)

        total = 0
        for nb_positions, r, moves in raw_inputs:
            total += nb_positions
            self.inputs.append((total, r, moves))
        self.size = total
        print(f"Total positions: {self.size:,}", flush=True)

    @staticmethod
    def _process_file(filename):
        inputs  = []
        outputs = []
        print(filename)
        with open(filename, 'rb') as f:
            data = f.read()
        print(len(data))
        idx = 0
        while idx < len(data):
            nb_moves        = int(data[idx])
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
            inputs.append((nb_moves + 1, nb_random_moves,
                            data[idx + 2: idx + 2 + 2 * nb_moves]))
            outputs.append(res)
            idx += 2 + 2 * nb_moves + 2
        return inputs, outputs

    @staticmethod
    def encode(moves) -> torch.Tensor:
        yolah = Yolah()
        for sq1, sq2 in zip(moves[0::2], moves[1::2]):
            yolah.play(Move(Square(int(sq1)), Square(int(sq2))))
        return encode_relational(yolah)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        lo, hi = 0, len(self.inputs)
        while lo < hi:
            m = lo + (hi - lo) // 2
            if self.inputs[m][0] <= idx:
                lo = m + 1
            else:
                hi = m
        n = self.inputs[lo - 1][0] if lo > 0 else 0
        _, r, moves = self.inputs[lo]
        return (self.encode(moves[: 2 * (r + idx - n)]),
                torch.tensor(self.outputs[lo], dtype=torch.long))


# ── Network ────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, l1_size=1024, l2_size=64, l3_size=32):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 3)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        x = torch.clamp(self.fc3(x), 0.0, 1.0)
        return self.fc4(x)

    def clip(self):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
            fc.weight.data.clamp_(-127/64, 127/64)
            fc.bias.data.clamp_(-127/64, 127/64)


# ── Training ───────────────────────────────────────────────────────────────────
NB_EPOCHS  = 300
MODEL_PATH = "/mnt/"
MODEL_NAME = "nnue_relational_1024x64x32x3"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"
GAME_DIR   = "./data"


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65436"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def dataloader_ddp(trainset, valset, batch_size):
    sampler_train = DistributedSampler(trainset)
    sampler_val   = DistributedSampler(valset, shuffle=False)
    train_loader  = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train,
        num_workers=2, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, sampler=sampler_val,
        num_workers=0, pin_memory=True
    )
    return train_loader, sampler_train, val_loader, sampler_val


class TrainerDDP:
    def __init__(self, gpu_id, model, train_loader, sampler_train,
                 val_loader, sampler_val, save_every=5):
        self.gpu_id        = gpu_id
        self.model         = model.to(gpu_id)
        self.train_loader  = train_loader
        self.sampler_train = sampler_train
        self.val_loader    = val_loader
        self.sampler_val   = sampler_val
        self.save_every    = save_every
        self.optimizer     = torch.optim.Adam(self.model.parameters(), lr=0.001,
                                              weight_decay=0)
        self.loss_fn       = torch.nn.CrossEntropyLoss()
        self.scheduler     = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                     gamma=0.99)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model  = DDP(self.model, device_ids=[gpu_id])
        self.model  = torch.compile(self.model)
        self.stream = torch.cuda.Stream(device=gpu_id)

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(),
                   f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n, running_loss, accuracy = 0, 0.0, 0
        for X, y in tqdm(self.train_loader):
            n += len(X)
            with torch.cuda.stream(self.stream):
                X = X.to(self.gpu_id, non_blocking=True)
                y = y.to(self.gpu_id, non_blocking=True)
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss   = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            accuracy     += (torch.argmax(logits, dim=1) == y).sum().item()
            self.model.module.clip()
        self.scheduler.step()
        if self.gpu_id == 0:
            lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch {epoch+1} train loss: {running_loss/n:.4f} '
                  f'train acc: {accuracy/n:.4f} lr: {lr:.6f}', flush=True)

    def _validate(self, epoch):
        self.model.train(False)
        n, val_loss, accuracy = 0, 0.0, 0
        with torch.no_grad():
            for X, y in tqdm(self.val_loader):
                n += len(X)
                with torch.cuda.stream(self.stream):
                    X = X.to(self.gpu_id, non_blocking=True)
                    y = y.to(self.gpu_id, non_blocking=True)
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
                logits = self.model(X)
                val_loss += self.loss_fn(logits, y).item()
                accuracy += (torch.argmax(logits, dim=1) == y).sum().item()
        if self.gpu_id == 0:
            print(f'epoch {epoch+1} val loss: {val_loss/n:.4f} '
                  f'val acc: {accuracy/n:.4f}', flush=True)
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
    val_size   = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    if rank == 0:
        print(f'Train: {train_size:,}  Val: {val_size:,}', flush=True)

    train_loader, sampler_train, val_loader, sampler_val = \
        dataloader_ddp(trainset, valset, batch_size)

    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        nb_params = sum(p.numel() for p in net.parameters())
        print(net, flush=True)
        print(f'Parameters: {nb_params:,}', flush=True)
        print(f'INPUT_SIZE: {INPUT_SIZE:,}  '
              f'(64×64×16 pairs + turn, {INPUT_SIZE*4/1024:.1f} KB per sample)',
              flush=True)

    trainer = TrainerDDP(rank, net, train_loader, sampler_train,
                         val_loader, sampler_val)
    trainer.train(NB_EPOCHS)
    destroy_process_group()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    dataset = RelationalGameDataset(GAME_DIR)
    mp.spawn(main, args=(world_size, 512 * 4, dataset), nprocs=world_size)
