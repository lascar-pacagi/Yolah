"""
features_net119x256x64x1.py — multi-GPU training of the features network
(119 inputs → 256 → 64 → 1) with AlphaZero-style value head.

Input is the 119-byte feature vector produced by `YolahFeatures::set_features`
(see ../player/yolah_features.h). Each on-disk record is:

    bytes 0 .. NB_FEATURES-1   : feature bytes (uint8, treated as v/255)
    byte  NB_FEATURES          : 3-class outcome label
        0 = black wins
        1 = draw
        2 = white wins

At training time the 3-class label + TURN feature are combined on the fly
into a current-player value ∈ {-1, 0, +1}, exactly as in the CNN/NNUE
value heads.

The on-disk format IS the cache — no separate preprocess step is needed.
The bottleneck of the original `features_network.py` is `num_workers=0`,
which this file fixes (8 persistent workers + prefetch_factor=4).
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
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob

torch.set_float32_matmul_precision('high')

# DataLoader workers default to sharing tensors through /dev/shm, which on
# SLURM + Singularity is typically capped to ~64 MB by the cgroup and
# triggers a SIGBUS the moment the first batch is handed off. Switching
# to the file_system strategy moves IPC to regular temp files.
torch.multiprocessing.set_sharing_strategy('file_system')

# ── Feature layout ────────────────────────────────────────────────────────────
# NB_FEATURES MUST match `YolahFeatures::NB_FEATURES` in player/yolah_features.h.
# If your header was regenerated with a different feature count, update this
# constant (and the data files) to match.
NB_FEATURES  = 119
RECORD_SIZE  = NB_FEATURES + 1   # last byte is the 3-class outcome label

# TURN is the final feature in the YolahFeatures enum, so it lives at the last
# index of the feature vector. 0 = black to move, 1 = white to move (uint8).
# Adjust if your feature layout puts TURN elsewhere.
TURN_INDEX   = NB_FEATURES - 1


# ── Dataset ───────────────────────────────────────────────────────────────────
class FeaturesDataset(Dataset):
    """
    Memory-mapped features dataset returning (features, value) per sample,
    optionally restricted to a contiguous [start, end) slice (train/val split).

    Cheap to serialize across a process boundary: __getstate__ emits only
    (data_dir, max_records, start, end), never the np.memmap arrays. A
    torch.multiprocessing.spawn child inherits the 'spawn' start method, so the
    DataLoader *spawns* its workers and serializes the dataset into each one —
    a raw np.memmap would serialize BY VALUE (copying the backing files into
    every worker). __setstate__ re-opens the memmaps: lazy mmap()s, shared
    OS page cache.

    state : (119,) float32 — feature_byte / 255
    value : ()     float32 — z ∈ {-1, 0, +1} from CURRENT player's POV,
                              derived from the 3-class label + TURN feature.
    """
    def __init__(self, data_dir: str, max_records: int = 10**12,
                 start: int = 0, end: int = None):
        self.data_dir    = data_dir
        self.max_records = max_records
        self.start       = start
        self._end        = end
        self._open()
        print(f"FeaturesDataset: {self.n_records:,} records across "
              f"{len(self.mmaps)} files", flush=True)

    def _open(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.features.txt")))
        if not files:
            raise FileNotFoundError(f"No .features.txt files in {self.data_dir}")

        self.mmaps      = []
        self.cumulative = []  # cumulative[i] = total records before file i
        total = 0
        for path in files:
            nb_records = os.path.getsize(path) // RECORD_SIZE
            mm = np.memmap(path, dtype=np.uint8, mode='r',
                           shape=(nb_records, RECORD_SIZE))
            self.cumulative.append(total)
            self.mmaps.append(mm)
            total += nb_records
            if total >= self.max_records:
                break

        self.n_records = total
        if self._end is None:
            self._end = total

    # Serialize only the recipe — see class docstring.
    def __getstate__(self):
        return {"data_dir": self.data_dir, "max_records": self.max_records,
                "start": self.start, "end": self._end}

    def __setstate__(self, state):
        self.data_dir    = state["data_dir"]
        self.max_records = state["max_records"]
        self.start       = state["start"]
        self._end        = state["end"]
        self._open()

    def __len__(self):
        return self._end - self.start

    def __getitem__(self, idx):
        i = self.start + idx
        # Binary search for the file containing global record i
        lo, hi = 0, len(self.cumulative)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.cumulative[mid] <= i:
                lo = mid
            else:
                hi = mid
        local_idx = i - self.cumulative[lo]
        record = self.mmaps[lo][local_idx]            # (RECORD_SIZE,) uint8

        features = torch.from_numpy(
            np.ascontiguousarray(record[:NB_FEATURES], dtype=np.float32) / 255.0)

        # 3-class label  ->  z_black  ->  z_current
        label = int(record[NB_FEATURES])
        z_black = 1 if label == 0 else (-1 if label == 2 else 0)
        # TURN feature: 0 = black to play, 1 = white to play (raw byte).
        black_to_move = (int(record[TURN_INDEX]) == 0)
        z_current = z_black if black_to_move else -z_black

        return features, torch.tensor(float(z_current), dtype=torch.float32)


# ── Network ───────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """
    Features network with NNUE-style clamped trunk + scalar value head.

    fc1, fc2 keep the [0, 1] clamp activations and int8-friendly weight
    clipping (|w| ≤ 127/64), so the trunk stays quantizable for C++ inference.
    fc3 is a 64 → 1 value head whose output is passed through tanh — its
    weights are left UNCLAMPED to avoid tanh saturation (a clamped 64-d input
    sum could otherwise reach ±127, where dtanh/dx ≈ 0 and gradients vanish).
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NB_FEATURES, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), min=0.0, max=1.0)
        x = torch.clamp(self.fc2(x), min=0.0, max=1.0)
        return torch.tanh(self.fc3(x)).squeeze(-1)

    def clip(self):
        # Only clamp the quantizable trunk — leave the value head free.
        for fc in [self.fc1, self.fc2]:
            fc.weight.data.clamp_(-127 / 64, 127 / 64)
            fc.bias.data.clamp_(-127 / 64, 127 / 64)


# ── Training config ───────────────────────────────────────────────────────────
NB_EPOCHS  = 200
MODEL_PATH = "/mnt/"
MODEL_NAME = "features_119x256x64x1"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"

DATA_DIR    = os.environ.get("YOLAH_FEATURES_DIR", "/cache")
NUM_WORKERS = int(os.environ.get("YOLAH_DATALOADER_WORKERS", "8"))


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65433"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def dataloader_ddp(trainset, valset, batch_size):
    sampler_train = DistributedSampler(trainset)
    sampler_val   = DistributedSampler(valset, shuffle=False)
    train_loader = DataLoader(
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
        self.model         = model.to(self.gpu_id)
        self.train_loader  = train_loader
        self.sampler_train = sampler_train
        self.val_loader    = val_loader
        self.sampler_val   = sampler_val
        self.save_every    = save_every
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.001, weight_decay=0)
        # Value head: MSE between tanh-squashed output and z ∈ {-1, 0, +1}.
        self.loss_fn   = nn.MSELoss()
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
            sign_correct += (torch.sign(v_pred) == torch.sign(y)).sum().item()
            self.model.module.clip()
        self.scheduler.step()
        if self.gpu_id == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            print('epoch {} train mse: {:.4f} train sign-acc: {:.4f} lr: {:.6f}'
                  .format(epoch + 1, running_loss / n, sign_correct / n,
                          current_lr), flush=True)

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


def main(rank, world_size, batch_size, data_dir):
    ddp_setup(rank, world_size)

    # Build the dataset views HERE, inside each spawned process — never pass a
    # dataset object through mp.spawn or into a DataLoader worker by value.
    #
    # Train/val is a CONTIGUOUS split, not random_split: a Subset's index list
    # would be a large Python list serialized into every spawned DataLoader
    # worker. Records are written game-by-game, so a contiguous cut keeps whole
    # games on one side — less leakage than a per-record random split.
    n_total  = FeaturesDataset(data_dir).n_records
    n_train  = int(0.95 * n_total)
    trainset = FeaturesDataset(data_dir, start=0, end=n_train)
    valset   = FeaturesDataset(data_dir, start=n_train, end=n_total)

    if rank == 0:
        print(f'Dataset: {n_total} records -> '
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
    # Pass the data directory (a string), not the dataset object — see main().
    mp.spawn(main, args=(world_size, 256, DATA_DIR), nprocs=world_size)
