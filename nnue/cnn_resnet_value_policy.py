from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append("../server")
from yolah import Yolah, Move, Square

torch.set_float32_matmul_precision('high')

# ── Board encoding ─────────────────────────────────────────────────────────────
#
# The position is encoded as a (4, 8, 8) tensor with four channels:
#   0 — black pieces  (binary 8×8 map)
#   1 — white pieces  (binary 8×8 map)
#   2 — empty squares (binary 8×8 map)
#   3 — turn          (all-zeros = black to play, all-ones = white to play)
#
# The score difference is implicitly recoverable from piece counts (initial
# pieces minus remaining), so the network can learn it from the trunk if it
# turns out to matter.


# ── Action encoding ────────────────────────────────────────────────────────────
#
# A Yolah move is (from_sq, to_sq) with each square in 0..63. We use a flat
# AlphaZero-style policy head with NUM_ACTIONS = 64 * 64 = 4096 logits:
#
#     action_idx = from_sq * 64 + to_sq
#
# Illegal moves (~3/4 of the space) are not masked at training time — the
# softmax naturally pushes their probability mass toward zero because they
# are never the supervised target. At inference time a legal-move mask
# should be applied before renormalising the distribution.
#
# "Pass" (Move.none() in yolah.py) maps to (SQ_A1, SQ_A1) → action_idx 0.
# A1→A1 is never a real move, so this slot is unambiguous.

NUM_ACTIONS = 64 * 64
POLICY_IGNORE_INDEX = -1  # used for terminal positions with no "next" move


# ── Value encoding ─────────────────────────────────────────────────────────────
#
# value_target ∈ {-1, 0, +1} is the game outcome from the CURRENT player's
# perspective:
#     +1  current player won the game
#      0  draw
#     -1  current player lost the game
#
# The network's value output uses tanh, so v ∈ (-1, +1). Loss is MSE(v, z).
# To recover a winning probability for the current player: P(win) = (v + 1) / 2.


def _bitboard_to_plane(n: int) -> np.ndarray:
    b = np.zeros(64, dtype=np.float32)
    for i in range(64):
        if n & (1 << (63 - i)):
            b[i] = 1.0
    return b.reshape(8, 8)


def encode_cnn(yolah) -> torch.Tensor:
    """Return a (4, 8, 8) float32 tensor encoding the position."""
    black = _bitboard_to_plane(yolah.black)
    white = _bitboard_to_plane(yolah.white)
    empty = _bitboard_to_plane(yolah.empty)
    black_to_move = (yolah.nb_plies() & 1) == 0
    turn  = np.full((8, 8), 0.0 if black_to_move else 1.0, dtype=np.float32)

    return torch.from_numpy(
        np.stack([black, white, empty, turn]))  # (4, 8, 8)


# ── Dataset ────────────────────────────────────────────────────────────────────
# Backed by the 4-plane memory-mapped files produced by preprocess.py.
IN_CHANNELS = 4


class GameDataset(Dataset):
    """
    Memory-mapped pre-encoded position dataset.

    Each sample returns:
        state         : (4, 8, 8) float32 — board encoding from the cache
        value_target  : ()        float32 — z ∈ {-1, 0, +1} from current
                                            player's perspective
        policy_target : ()        int64   — from*64 + to, or
                                            POLICY_IGNORE_INDEX for terminal
                                            positions
    """
    def __init__(self, cache_dir: str):
        meta_path = os.path.join(cache_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        n_planes = meta["positions"]["shape"][1]
        if n_planes != IN_CHANNELS:
            raise ValueError(
                f"Cache has {n_planes} planes but training expects {IN_CHANNELS}. "
                "Re-run preprocess.py.")

        # mmap mode='r' shares one OS page cache across all DataLoader workers
        self.positions = np.memmap(
            os.path.join(cache_dir, meta["positions"]["path"]),
            dtype=np.uint8, mode='r',
            shape=tuple(meta["positions"]["shape"]))
        self.values = np.memmap(
            os.path.join(cache_dir, meta["values"]["path"]),
            dtype=np.int8, mode='r',
            shape=tuple(meta["values"]["shape"]))
        self.policies = np.memmap(
            os.path.join(cache_dir, meta["policies"]["path"]),
            dtype=np.int16, mode='r',
            shape=tuple(meta["policies"]["shape"]))
        self.size = int(meta["n_positions"])
        print(f"Memmap dataset: {self.size:,} positions, {IN_CHANNELS} planes",
              flush=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # np.ascontiguousarray forces a copy out of the memmap so the tensor
        # owns its memory (safer with pin_memory + multi-worker prefetch).
        state = np.ascontiguousarray(self.positions[idx], dtype=np.float32)
        return (
            torch.from_numpy(state),
            torch.tensor(float(self.values[idx]), dtype=torch.float32),
            torch.tensor(int(self.policies[idx]), dtype=torch.long),
        )


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
    ResNet for Yolah with TWO heads (AlphaZero style).

      Trunk : (B, 4, 8, 8) ──► input conv ──► N × ResBlock ──► BN ──► ReLU
              → shared feature map  (B, C, 8, 8)

      Value head : 1×1 conv → 1 channel → flatten(64) → FC → ReLU → FC → tanh
                   → v ∈ (-1, +1), scalar per position
                   Interpretation: expected outcome from current player's POV.
                   P(current player wins) = (v + 1) / 2.

      Policy head: 1×1 conv → 2 channels → flatten(128) → FC → 4096 logits
                   Action index = from_sq * 64 + to_sq.

    Forward returns (value, policy_logits).
    """
    def __init__(self, channels: int = 256, nb_blocks: int = 30,
                 value_fc_size: int = 256, num_actions: int = NUM_ACTIONS):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────
        self.input_conv = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ── Shared residual trunk ────────────────────────────────────────
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(nb_blocks)])
        self.output_bn  = nn.BatchNorm2d(channels)

        # ── Value head (AlphaZero style) ─────────────────────────────────
        # 1×1 conv collapses C channels to a single 8×8 map, then FC layers
        # produce a scalar squashed through tanh.
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(64, value_fc_size)
        self.value_fc2  = nn.Linear(value_fc_size, 1)

        # ── Policy head (AlphaZero style) ────────────────────────────────
        # 1×1 conv to 2 channels, flatten, project to 4096 action logits.
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 64, num_actions)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = torch.relu(self.output_bn(x))   # shared (B, C, 8, 8) features

        # Value head: (B,C,8,8) → (B,1,8,8) → (B,64) → (B,fc) → (B,1) → (B,)
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        # Policy head: (B,C,8,8) → (B,2,8,8) → (B,128) → (B, NUM_ACTIONS)
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        p = self.policy_fc(p)

        return v, p

    def clip(self):
        pass


# ── Training ───────────────────────────────────────────────────────────────────
NB_EPOCHS  = 3
MODEL_PATH = "/mnt/"
MODEL_NAME = "cnn_resnet_256x30_value_policy"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"
CACHE_DIR  = os.environ.get("YOLAH_CACHE_DIR", "/cache")
NUM_WORKERS = int(os.environ.get("YOLAH_DATALOADER_WORKERS", "8"))

# Combined-loss weight: total = VALUE_LOSS_WEIGHT * MSE(v,z) + CE(p, action)
# AlphaZero uses 1.0 for both terms; tweak if value/policy learning rates diverge.
VALUE_LOSS_WEIGHT  = 1.0
POLICY_LOSS_WEIGHT = 1.0


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65437"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def dataloader_ddp(trainset, valset, batch_size):
    sampler_train = DistributedSampler(trainset)
    sampler_val   = DistributedSampler(valset, shuffle=False)
    train_loader  = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4,
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, sampler=sampler_val,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4,
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
        # Value loss: MSE on tanh output vs target z ∈ {-1, 0, +1}.
        self.value_loss_fn  = nn.MSELoss()
        # Policy loss: CE over 4096 logits, ignoring terminal positions.
        self.policy_loss_fn = nn.CrossEntropyLoss(ignore_index=POLICY_IGNORE_INDEX)
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

    def _compute_loss(self, value_pred, policy_logits, value_target, policy_target):
        v_loss = self.value_loss_fn(value_pred, value_target)
        p_loss = self.policy_loss_fn(policy_logits, policy_target)
        return (VALUE_LOSS_WEIGHT  * v_loss
              + POLICY_LOSS_WEIGHT * p_loss, v_loss, p_loss)

    @staticmethod
    def _policy_correct(policy_logits, policy_target):
        # Only count positions where the policy target is valid (not ignore_index).
        mask = policy_target != POLICY_IGNORE_INDEX
        if mask.sum().item() == 0:
            return 0, 0
        preds = torch.argmax(policy_logits, dim=1)
        correct = ((preds == policy_target) & mask).sum().item()
        return correct, int(mask.sum().item())

    def _run_epoch(self, epoch):
        n = 0
        v_correct = 0  # sign-match counter (predicted v has same sign as target)
        p_correct = 0
        p_total   = 0  # excludes terminal positions
        v_running, p_running = 0.0, 0.0
        for X, v_target, p_target in tqdm(self.train_loader):
            bs = len(X)
            n += bs
            with torch.cuda.stream(self.stream):
                X        = X.to(self.gpu_id, non_blocking=True)
                v_target = v_target.to(self.gpu_id, non_blocking=True)
                p_target = p_target.to(self.gpu_id, non_blocking=True)
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
            self.optimizer.zero_grad()
            v_pred, p_logits = self.model(X)
            loss, v_loss, p_loss = self._compute_loss(v_pred, p_logits,
                                                     v_target, p_target)
            loss.backward()
            self.optimizer.step()
            v_running += v_loss.item() * bs
            p_running += p_loss.item() * bs
            v_correct += (torch.sign(v_pred) == torch.sign(v_target)).sum().item()
            pc, pt = self._policy_correct(p_logits, p_target)
            p_correct += pc
            p_total   += pt
            self.model.module.clip()
        self.scheduler.step()
        if self.gpu_id == 0:
            lr = self.optimizer.param_groups[0]['lr']
            p_acc = (p_correct / p_total) if p_total > 0 else 0.0
            print(f'epoch {epoch+1} '
                  f'train value_mse: {v_running/n:.4f} '
                  f'value_sign_acc: {v_correct/n:.4f} '
                  f'policy_ce: {p_running/n:.4f} '
                  f'policy_acc: {p_acc:.4f} '
                  f'lr: {lr:.6f}', flush=True)

    def _validate(self, epoch):
        self.model.train(False)
        n = 0
        v_correct = 0
        p_correct = 0
        p_total   = 0
        v_running, p_running = 0.0, 0.0
        with torch.no_grad():
            for X, v_target, p_target in tqdm(self.val_loader):
                bs = len(X)
                n += bs
                with torch.cuda.stream(self.stream):
                    X        = X.to(self.gpu_id, non_blocking=True)
                    v_target = v_target.to(self.gpu_id, non_blocking=True)
                    p_target = p_target.to(self.gpu_id, non_blocking=True)
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
                v_pred, p_logits = self.model(X)
                _, v_loss, p_loss = self._compute_loss(v_pred, p_logits,
                                                      v_target, p_target)
                v_running += v_loss.item() * bs
                p_running += p_loss.item() * bs
                v_correct += (torch.sign(v_pred) == torch.sign(v_target)).sum().item()
                pc, pt = self._policy_correct(p_logits, p_target)
                p_correct += pc
                p_total   += pt
        if self.gpu_id == 0:
            p_acc = (p_correct / p_total) if p_total > 0 else 0.0
            print(f'epoch {epoch+1} '
                  f'val value_mse: {v_running/n:.4f} '
                  f'value_sign_acc: {v_correct/n:.4f} '
                  f'policy_ce: {p_running/n:.4f} '
                  f'policy_acc: {p_acc:.4f}', flush=True)
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
    dataset = GameDataset(CACHE_DIR)
    mp.spawn(main, args=(world_size, 512, dataset), nprocs=world_size)
