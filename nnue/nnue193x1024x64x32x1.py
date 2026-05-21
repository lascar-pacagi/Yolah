"""
nnue193x1024x64x32x1.py — multi-GPU NNUE training (193 → 1024 → 64 → 32 → 1)
adapted from nnue_multigpu5.py with memory-mapped data loading and an
AlphaZero-style value head.

Input is the flat 193-element NNUE encoding:
    bytes  0–63   : black piece bitboard, unpacked MSB-first
    bytes 64–127  : white piece bitboard
    bytes 128–191 : empty (scored) squares
    byte  192     : turn (0 = black to play, 1 = white to play)

Target is a single scalar value ∈ {-1, 0, +1} from the CURRENT player's
perspective:
    +1  current player won
     0  draw
    -1  current player lost

The network outputs one tanh-squashed value v ∈ (-1, +1); loss is MSE.
P(current player wins) can be recovered as (v + 1) / 2 if needed.

Run preprocess_nnue.py *once* to produce the encoded cache (inputs.u8 +
values.i8 + meta.json). This script then memory-maps it and trains.
"""
from tqdm import tqdm
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.append("../server")
from yolah import Yolah, Move, Square

torch.set_float32_matmul_precision('high')

# DataLoader workers default to sharing tensors through /dev/shm, which on
# SLURM + Singularity is typically capped to ~64 MB by the cgroup and
# triggers a SIGBUS the moment the first batch is handed off. Switching
# to the file_system strategy moves IPC to regular temp files, removing
# the /dev/shm dependency entirely.
torch.multiprocessing.set_sharing_strategy('file_system')


# black positions + white positions + empty positions + turn
INPUT_SIZE = 64 + 64 + 64 + 1


# ── Dataset ────────────────────────────────────────────────────────────────────
# Backed by the memory-mapped files produced by preprocess_nnue.py.
class GameDataset(Dataset):
    """
    Memory-mapped pre-encoded position dataset, optionally restricted to a
    contiguous [start, end) slice of the cache (used for the train/val split).

    Cheap to serialize across a process boundary: __getstate__ emits only
    (cache_dir, start, end), never the np.memmap arrays. A torch.mp.spawn
    child inherits the 'spawn' start method, so the DataLoader *spawns* its
    workers and serializes the dataset into each one — a raw np.memmap would
    serialize BY VALUE (copying the whole backing file into every worker).
    __setstate__ re-opens the memmap instead: a lazy mmap(), page cache shared.

    Each sample returns:
        state : (193,) float32 — flat NNUE input (binary + turn bit)
        value : ()     float32 — z ∈ {-1, 0, +1} from current player's POV
    """
    def __init__(self, cache_dir: str, start: int = 0, end: int = None):
        self.cache_dir = cache_dir
        self.start = start
        self._end = end
        self._open()

    def _open(self):
        meta_path = os.path.join(self.cache_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        if meta["inputs"]["shape"][1] != INPUT_SIZE:
            raise ValueError(
                f"Cache input width {meta['inputs']['shape'][1]} "
                f"!= expected {INPUT_SIZE}")

        # mmap mode='r' shares one OS page cache across all processes
        self.inputs = np.memmap(
            os.path.join(self.cache_dir, meta["inputs"]["path"]),
            dtype=np.uint8, mode='r',
            shape=tuple(meta["inputs"]["shape"]))
        self.values = np.memmap(
            os.path.join(self.cache_dir, meta["values"]["path"]),
            dtype=np.int8, mode='r',
            shape=tuple(meta["values"]["shape"]))
        self.n_positions = int(meta["n_positions"])
        if self._end is None:
            self._end = self.n_positions

    # Serialize only the recipe — see class docstring.
    def __getstate__(self):
        return {"cache_dir": self.cache_dir, "start": self.start, "end": self._end}

    def __setstate__(self, state):
        self.cache_dir = state["cache_dir"]
        self.start     = state["start"]
        self._end      = state["end"]
        self._open()

    def __len__(self):
        return self._end - self.start

    def __getitem__(self, idx):
        i = self.start + idx
        # ascontiguousarray forces a copy out of the memmap so the tensor
        # owns its memory (safer with pin_memory + multi-worker prefetch).
        state = np.ascontiguousarray(self.inputs[i], dtype=np.float32)
        return (
            torch.from_numpy(state),
            torch.tensor(float(self.values[i]), dtype=torch.float32),
        )


# ── Network ────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """
    NNUE feature transformer (fc1–fc3) + scalar value head (fc4 → tanh).

    The first three layers keep the NNUE-style clamp([0,1]) activations and
    int8-friendly weight clipping (|w| ≤ 127/64) so the trained body remains
    quantizable. The final 1-d value head is left unclamped — clamping it to
    ±127/64 with a 32-dim input would let the pre-tanh sum reach ~63, where
    tanh saturates and dtanh/dx ≈ 0 (no gradient).
    """
    def __init__(self, input_size=INPUT_SIZE, l1_size=1024, l2_size=64, l3_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc2(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc3(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        # Value head: single scalar squashed to (-1, +1) via tanh.
        return torch.tanh(self.fc4(x)).squeeze(-1)

    def clip(self):
        # Only clamp the quantizable trunk — leave the value head free.
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.weight.data.clamp_(-127 / 64, 127 / 64)
            fc.bias.data.clamp_(-127 / 64, 127 / 64)


# ── Training config ────────────────────────────────────────────────────────────
NB_EPOCHS  = 30
MODEL_PATH = "/mnt/"
MODEL_NAME = "nnue_193x1024x64x32x1"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"

CACHE_DIR   = os.environ.get("YOLAH_CACHE_DIR", "/cache")
NUM_WORKERS = int(os.environ.get("YOLAH_DATALOADER_WORKERS", "8"))


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"
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
                 val_loader, sampler_val, save_every=5):
        self.gpu_id        = gpu_id
        self.model         = model.to(self.gpu_id)
        self.train_loader  = train_loader
        self.sampler_train = sampler_train
        self.val_loader    = val_loader
        self.sampler_val   = sampler_val
        self.save_every    = save_every
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.001, weight_decay=0)
        # Value head: MSE between tanh-squashed output and z ∈ {-1, 0, +1}.
        self.loss_fn   = torch.nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                            self.optimizer, gamma=0.99)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model  = DDP(self.model, device_ids=[self.gpu_id])
        self.model  = torch.compile(self.model)
        self.stream = torch.cuda.Stream(device=gpu_id)

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(),
                   f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0.0
        sign_correct = 0
        for (X, y) in tqdm(self.train_loader):
            bs = len(X)
            n += bs
            with torch.cuda.stream(self.stream):
                X = X.to(self.gpu_id, non_blocking=True)
                y = y.to(self.gpu_id, non_blocking=True)
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)

            self.optimizer.zero_grad()
            v_pred = self.model(X)
            loss   = self.loss_fn(v_pred, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * bs
            # Sign-match: does the prediction agree with the outcome direction?
            sign_correct += (torch.sign(v_pred) == torch.sign(y)).sum().item()
            self.model.module.clip()
        self.scheduler.step()
        if self.gpu_id == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            print('epoch {} train mse: {:.4f} train sign-acc: {:.4f} lr: {:.6f}'.format(
                epoch + 1, running_loss / n, sign_correct / n, current_lr),
                flush=True)

    def _validate(self, epoch):
        self.model.train(False)
        n = 0
        val_loss     = 0.0
        sign_correct = 0
        with torch.no_grad():
            for (X, y) in tqdm(self.val_loader):
                bs = len(X)
                n += bs
                with torch.cuda.stream(self.stream):
                    X = X.to(self.gpu_id, non_blocking=True)
                    y = y.to(self.gpu_id, non_blocking=True)
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)

                v_pred = self.model(X)
                loss   = self.loss_fn(v_pred, y)
                val_loss     += loss.item() * bs
                sign_correct += (torch.sign(v_pred) == torch.sign(y)).sum().item()
        if self.gpu_id == 0:
            print('epoch {} val mse: {:.4f} val sign-acc: {:.4f}'.format(
                epoch + 1, val_loss / n, sign_correct / n), flush=True)
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


def main(rank, world_size, batch_size, cache_dir):
    ddp_setup(rank, world_size)
    print(rank)

    # Build the dataset views HERE, inside each spawned process — never pass a
    # dataset object through mp.spawn or into a DataLoader worker by value.
    #
    # Train/val is a CONTIGUOUS split, not random_split: a Subset's index list
    # would be a multi-GB Python list serialized into every spawned DataLoader
    # worker. Positions are stored game-by-game, so a contiguous cut keeps
    # whole games on one side — less leakage than a per-position random split.
    n_total  = GameDataset(cache_dir).n_positions
    n_train  = int(0.95 * n_total)
    trainset = GameDataset(cache_dir, 0, n_train)
    valset   = GameDataset(cache_dir, n_train, n_total)

    if rank == 0:
        print(f'Dataset: {n_total} positions -> '
              f'train {n_train}, val {n_total - n_train}', flush=True)

    train_loader, sampler_train, val_loader, sampler_val = \
        dataloader_ddp(trainset, valset, batch_size)
    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        print(net, flush=True)
    trainer = TrainerDDP(rank, net, train_loader, sampler_train,
                         val_loader, sampler_val)
    trainer.train(NB_EPOCHS)
    destroy_process_group()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    # Pass the cache directory (a string), not the dataset object — see main().
    mp.spawn(main, args=(world_size, 512, CACHE_DIR), nprocs=world_size)
