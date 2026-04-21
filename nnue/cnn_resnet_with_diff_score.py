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

# ── Board encoding ─────────────────────────────────────────────────────────────
#
# The position is encoded as a (6, 8, 8) tensor with six channels:
#   0 — black pieces       (binary 8×8 map)
#   1 — white pieces       (binary 8×8 map)
#   2 — empty squares      (binary 8×8 map)
#   3 — turn               (all-zeros = black to play, all-ones = white to play)
#   4 — black certain-win  (all-ones if black has a proven win, else all-zeros)
#   5 — white certain-win  (all-ones if white has a proven win, else all-zeros)
#
# Certain-win reasoning (see nnue_with_diff_score.py). Yolah alternates plies;
# "cannot move" is permanent. In normal alternating play:
#     delta = black_score - white_score
#     effective_delta = delta           if black to move
#                     = delta - 1       if white to move
# effective_delta >= 1  ⇒ black is provably winning (white must already be
#                         stuck, otherwise delta would still be 0 on black's
#                         turn).
# effective_delta <= -1 ⇒ white is provably winning (symmetric).
# Computed per position from live yolah state — NO label leakage.


def _bitboard_to_plane(n: int) -> np.ndarray:
    b = np.zeros(64, dtype=np.float32)
    for i in range(64):
        if n & (1 << (63 - i)):
            b[i] = 1.0
    return b.reshape(8, 8)


def encode_cnn(yolah) -> torch.Tensor:
    """Return a (6, 8, 8) float32 tensor encoding the position."""
    black = _bitboard_to_plane(yolah.black)
    white = _bitboard_to_plane(yolah.white)
    empty = _bitboard_to_plane(yolah.empty)
    black_to_move = (yolah.nb_plies() & 1) == 0
    turn  = np.full((8, 8), 0.0 if black_to_move else 1.0, dtype=np.float32)

    delta = yolah.black_score - yolah.white_score
    effective_delta = delta - (0 if black_to_move else 1)
    black_win = np.full((8, 8), 1.0 if effective_delta >=  1 else 0.0, dtype=np.float32)
    white_win = np.full((8, 8), 1.0 if effective_delta <= -1 else 0.0, dtype=np.float32)

    return torch.from_numpy(
        np.stack([black, white, empty, turn, black_win, white_win]))  # (6, 8, 8)


# ── Dataset ────────────────────────────────────────────────────────────────────
class GameDataset(Dataset):
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
        print(filename, flush=True)
        with open(filename, 'rb') as f:
            data = f.read()
        print(len(data), flush=True)
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
        return encode_cnn(yolah)

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
class ResBlock(nn.Module):
    """
    Pre-activation residual block (ResNet-v2 style).

    Data flow:
                          ┌─────────────────────────────────┐  (skip / identity)
      x ─► BN ─► ReLU ─► Conv ─► BN ─► ReLU ─► Conv ─► (+) ─► output
    """
    def __init__(self, channels: int):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.conv1(torch.relu(self.bn1(x)))
        x = self.conv2(torch.relu(self.bn2(x)))
        return x + residual


class Net(nn.Module):
    """
    ResNet for Yolah position evaluation with certain-win features.

    Input:  (B, 6, 8, 8) — black / white / empty / turn / black-win / white-win
    Output: (B, 3)        — logits for black-win / draw / white-win
    """
    def __init__(self, channels: int = 256, nb_blocks: int = 30, fc_size: int = 256):
        super().__init__()

        # Stem: project 6 input planes into the C-wide feature space.
        self.input_conv = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, padding=1, bias=False),  # (B,6,8,8)→(B,C,8,8)
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(nb_blocks)])
        self.output_bn  = nn.BatchNorm2d(channels)

        self.head = nn.Sequential(
            nn.Linear(channels, fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_size, 3),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = torch.relu(self.output_bn(x))
        x = x.mean(dim=(2, 3))
        return self.head(x)

    def clip(self):
        pass


# ── Training ───────────────────────────────────────────────────────────────────
NB_EPOCHS  = 3
MODEL_PATH = "/mnt/"
MODEL_NAME = "cnn_resnet_256x30_with_diff_score"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"
GAME_DIR   = "./data"


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65437"
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
                 val_loader, sampler_val, save_every=1):
        self.gpu_id        = gpu_id
        self.model         = model.to(gpu_id)
        self.train_loader  = train_loader
        self.sampler_train = sampler_train
        self.val_loader    = val_loader
        self.sampler_val   = sampler_val
        self.save_every    = save_every
        self.optimizer     = torch.optim.Adam(self.model.parameters(), lr=0.001,
                                              weight_decay=1e-4)
        self.loss_fn       = nn.CrossEntropyLoss()
        self.scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
                                 self.optimizer, T_max=NB_EPOCHS, eta_min=1e-5)
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

    trainer = TrainerDDP(rank, net, train_loader, sampler_train,
                         val_loader, sampler_val)
    trainer.train(NB_EPOCHS)
    destroy_process_group()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    dataset = GameDataset(GAME_DIR)
    mp.spawn(main, args=(world_size, 512, dataset), nprocs=world_size)
