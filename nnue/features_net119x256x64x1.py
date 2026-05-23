"""
features_net119x256x64x1.py — multi-GPU training of the 119-input features
network (119 → 256 → 64 → 1) with an AlphaZero-style scalar value head.

Data path (matches cnn_resnet_value_policy_chunked.py)
─────────────────────────────────────────────────────
The cache produced by preprocess_features.py is a flat memmap of N records:
    features.u8 : (N, 119) uint8   — feature bytes (treated as v / 255)
    values.i8   : (N,)     int8    — z ∈ {-1, 0, +1} from current player's POV
    meta.json   : sidecar with shapes / counts

This script uses a `ChunkedShuffleLoader`:
  • one background producer thread per rank reads a CONTIGUOUS chunk
    (~500 MB) sequentially into RAM, shuffles WITHIN the chunk, slices it
    into batches, pins each batch, and pushes onto a bounded queue;
  • the main thread pulls pre-pinned batches and overlaps the next batch's
    H2D copy with the current batch's compute via a side CUDA stream;
  • DDP is configured with `gradient_as_bucket_view=True` and
    `static_graph=True` to avoid the periodic ~50 s NCCL all-reduce stall
    that plagued the DataLoader variant on multi-GPU runs.

fc1 and fc2 keep NNUE-style clamp([0,1]) activations + int8 weight clipping
(|w| ≤ 127/64) so the trunk is quantizable; fc3 (64 → 1 value head, tanh) is
left unclamped to avoid tanh saturation.
"""
from tqdm import tqdm
import torch
from torch import nn
import torch.multiprocessing as mp           # mp.spawn launches one process per GPU
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import gc                                    # disabled inside main() to avoid GC pauses
import json
import random                                # epoch-level chunk-order shuffle
import threading                             # the background producer thread
import queue as queue_mod                    # bounded hand-off queue (the buffer)
import numpy as np

torch.set_float32_matmul_precision('high')

# The chunked loader does its own threaded I/O, but keep file_system sharing
# in case any other torch IPC sneaks in (SLURM+Singularity caps /dev/shm).
torch.multiprocessing.set_sharing_strategy('file_system')

# ── Feature layout (MUST match YolahFeatures in player/yolah_features.h) ───
NB_FEATURES = 119


# ── Chunked-shuffle double-buffered loader ──────────────────────────────────
#
# CHUNK_SIZE — number of positions read per chunk.
#   4194304 = 2048 * 2048. Two deliberate properties:
#     • an exact multiple of any plausible batch size → no ragged tail;
#     • at 119 bytes/position it is ~500 MB of uint8 — small enough for a
#       few chunks to coexist in RAM, large enough that a chunk spans many
#       games so the within-chunk shuffle decorrelates batches well.
CHUNK_SIZE = 2048 * 2048

# BATCH_QUEUE_DEPTH — how many fully-built, pinned batches the producer may
# stay ahead of the GPU. Each batch (bs=256) is ~120 KB float32 X + 1 KB y
# → 32 batches ≈ 4 MB of pinned RAM per rank: negligible.
BATCH_QUEUE_DEPTH = 32


class ChunkedShuffleLoader:
    """
    Streaming data loader over the (features.u8, values.i8) memmap pair.

    Iterating yields (X, y) batches with X pinned for async H2D copy:
        X : (bs, 119) float32 in [0, 1]    (features divided by 255)
        y : (bs,)     float32 ∈ {-1, 0, +1}

    Distributed training (DDP)
    --------------------------
    With `world_size` GPUs, rank r owns chunks r, r+world_size, … so the
    shards are disjoint. Remainder chunks and the trailing partial chunk are
    dropped to guarantee EVERY rank yields the same number of batches —
    essential because the per-step all-reduce would otherwise hang on a rank
    that ran fewer steps.
    """

    def __init__(self, cache_dir, lo, hi, batch_size, rank, world_size,
                 chunk_size=CHUNK_SIZE, shuffle=True, pin_memory=True,
                 queue_depth=BATCH_QUEUE_DEPTH):
        self.cache_dir   = cache_dir
        self.batch_size  = batch_size
        self.rank        = rank
        self.world_size  = world_size
        self.chunk_size  = chunk_size
        self.shuffle     = shuffle
        self.pin_memory  = pin_memory
        self.queue_depth = queue_depth
        self.epoch       = 0
        self._open()

        # Whole non-overlapping chunks within this loader's [lo, hi) shard.
        all_starts = list(range(lo, hi - chunk_size + 1, chunk_size))

        # Floor-divide so every rank gets the SAME number of chunks; up to
        # (world_size - 1) leftover chunks are dropped on purpose.
        per_rank = len(all_starts) // world_size
        self.my_chunks = all_starts[rank : per_rank * world_size : world_size]

        self.batches_per_chunk = chunk_size // batch_size
        self.n_batches = len(self.my_chunks) * self.batches_per_chunk

    def _open(self):
        """
        Lazy mmap()s into virtual address space — no actual disk read here.
        Pages are demand-paged by the OS when the producer thread touches them,
        and the page cache is shared across processes on the node.
        """
        with open(os.path.join(self.cache_dir, "meta.json")) as f:
            meta = json.load(f)

        n_features_cols = meta["features"]["shape"][1]
        if n_features_cols != NB_FEATURES:
            raise ValueError(
                f"Cache feature width {n_features_cols} != expected {NB_FEATURES}. "
                "Re-run preprocess_features.py.")

        n = int(meta["n_positions"])
        self.features = np.memmap(
            os.path.join(self.cache_dir, meta["features"]["path"]),
            dtype=np.uint8, mode='r', shape=tuple(meta["features"]["shape"]))
        self.values = np.memmap(
            os.path.join(self.cache_dir, meta["values"]["path"]),
            dtype=np.int8, mode='r', shape=(n,))

    def set_epoch(self, epoch):
        """Reseed shuffling for a new epoch."""
        self.epoch = epoch

    def __len__(self):
        """Number of batches per epoch — lets tqdm show a correct total."""
        return self.n_batches

    def _producer(self, chunk_order, q):
        """
        Background thread. For each chunk in this epoch's shuffled order:
            1. read the whole chunk sequentially from the memmap into RAM;
            2. permute positions within the chunk (epoch+offset-seeded);
            3. slice into batches, cast dtypes, pin, push to `q`.

        Why a thread (not a process): numpy bulk memcpy/astype and torch's
        pin_memory all release the Python GIL, so this thread genuinely runs
        at the same time as the main thread's GPU calls.
        """
        bs = self.batch_size
        try:
            for chunk_id in chunk_order:
                start = self.my_chunks[chunk_id]
                end   = start + self.chunk_size

                # Sequential disk read — np.array() forces a contiguous copy.
                feat = np.array(self.features[start:end])    # (C, 119) uint8
                val  = np.array(self.values[start:end])      # (C,)     int8

                if self.shuffle:
                    rng  = np.random.default_rng(
                        (self.epoch * 1_000_003 + start) & 0xFFFFFFFF)
                    perm = rng.permutation(self.chunk_size)
                else:
                    perm = np.arange(self.chunk_size)

                for b in range(self.batches_per_chunk):
                    idx = perm[b * bs : (b + 1) * bs]
                    # Features are stored as raw uint8 bytes; the trained net
                    # expects values in [0, 1] (see YolahFeatures encoding),
                    # so we divide by 255 here once per batch.
                    X = torch.from_numpy(
                            feat[idx].astype(np.float32) / 255.0)
                    y = torch.from_numpy(val[idx].astype(np.float32))
                    if self.pin_memory:
                        X, y = X.pin_memory(), y.pin_memory()
                    q.put((X, y))                           # blocks if full
        except Exception as e:                              # pragma: no cover
            print(f"[ChunkedShuffleLoader] producer error: {e}", flush=True)
        finally:
            q.put(None)                                     # sentinel

    def __iter__(self):
        """Spawn the producer thread and yield its batches for one epoch."""
        order = list(range(len(self.my_chunks)))
        if self.shuffle:
            random.Random(self.epoch).shuffle(order)

        q = queue_mod.Queue(maxsize=self.queue_depth)
        t = threading.Thread(target=self._producer, args=(order, q), daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join()


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

CACHE_DIR = os.environ.get("YOLAH_FEATURES_DIR", "/cache")


def ddp_setup(rank, world_size):
    """All ranks live on one node → rendezvous over localhost; NCCL is the backend."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65433"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class TrainerDDP:
    """Owns one GPU's model replica, optimizer, and the train/validate loops."""

    def __init__(self, gpu_id, model, train_loader, val_loader, save_every=1):
        self.gpu_id       = gpu_id
        self.model        = model.to(gpu_id)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.save_every   = save_every

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.001, weight_decay=0)
        # Value head: MSE between tanh-squashed output and z ∈ {-1, 0, +1}.
        self.loss_fn   = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                            self.optimizer, gamma=0.99)

        # Mixed precision. bf16 needs Ampere+ tensor cores (cc major ≥ 8);
        # on Turing (cc 7.5, e.g. Quadro RTX 8000) bf16 would be emulated
        # and slow, so fall back to fp16 + GradScaler there.
        major, _ = torch.cuda.get_device_capability(gpu_id)
        self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=(self.amp_dtype == torch.float16))

        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        # gradient_as_bucket_view + static_graph: fix the periodic ~50 s NCCL
        # all-reduce stall observed on multi-GPU DDP training. The autograd
        # graph here is identical every step (no conditionals, no unused
        # params), so static_graph is sound.
        self.model = DDP(self.model, device_ids=[gpu_id],
                         gradient_as_bucket_view=True, static_graph=True)
        self.model = torch.compile(self.model)
        # Side CUDA stream for prefetching the next batch's H2D copy.
        self.stream = torch.cuda.Stream(device=gpu_id)

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(),
                   f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _h2d_async(self, batch_cpu):
        """Issue an async H2D for (X, y) on self.stream; return GPU tensors."""
        X_cpu, y_cpu = batch_cpu
        with torch.cuda.stream(self.stream):
            X = X_cpu.to(self.gpu_id, non_blocking=True)
            y = y_cpu.to(self.gpu_id, non_blocking=True)
        return (X, y)

    def _run_epoch(self, epoch):
        """One training pass with overlapping H2D + compute (see _h2d_async)."""
        n = 0
        running_loss = 0.0
        sign_correct = 0
        it = 0

        loader_iter = iter(self.train_loader)
        cpu_cur = next(loader_iter, None)
        if cpu_cur is None:
            return
        gpu_cur = self._h2d_async(cpu_cur)

        pbar = tqdm(total=len(self.train_loader))
        while True:
            # Make the default stream wait until the current batch's H2D is
            # complete on self.stream. Stream dependency only — host does
            # not block here.
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
            X, y = gpu_cur
            bs = len(X)
            n += bs

            # Issue the NEXT iteration's H2D on the side stream now, so it
            # runs concurrently with this iteration's compute below.
            cpu_next = next(loader_iter, None)
            if cpu_next is not None:
                gpu_next = self._h2d_async(cpu_next)

            self.optimizer.zero_grad()
            with torch.autocast('cuda', dtype=self.amp_dtype):
                v_pred = self.model(X)
                loss   = self.loss_fn(v_pred, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * bs
            sign_correct += (torch.sign(v_pred) == torch.sign(y)).sum().item()
            self.model.module.clip()        # int8 trunk weight clamp

            # Manual GC under disabled auto-GC: one controlled pause every
            # 1000 iters instead of unpredictable Gen-2 stalls.
            it += 1
            if it % 1000 == 0:
                gc.collect()

            pbar.update(1)
            if cpu_next is None:
                break
            cpu_cur = cpu_next
            gpu_cur = gpu_next

        pbar.close()
        self.scheduler.step()
        if self.gpu_id == 0:
            lr = self.optimizer.param_groups[0]['lr']
            print('epoch {} train mse: {:.4f} train sign-acc: {:.4f} lr: {:.6f}'
                  .format(epoch + 1, running_loss / n, sign_correct / n, lr),
                  flush=True)

    def _validate(self, epoch):
        """One validation pass — no gradients, same prefetch pattern."""
        self.model.train(False)
        n = 0
        val_loss     = 0.0
        sign_correct = 0
        with torch.no_grad():
            loader_iter = iter(self.val_loader)
            cpu_cur = next(loader_iter, None)
            if cpu_cur is None:
                self.model.train()
                return
            gpu_cur = self._h2d_async(cpu_cur)
            pbar = tqdm(total=len(self.val_loader))
            while True:
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
                X, y = gpu_cur
                bs = len(X)
                n += bs

                cpu_next = next(loader_iter, None)
                if cpu_next is not None:
                    gpu_next = self._h2d_async(cpu_next)

                with torch.autocast('cuda', dtype=self.amp_dtype):
                    v_pred = self.model(X)
                    loss   = self.loss_fn(v_pred, y)
                val_loss     += loss.item() * bs
                sign_correct += (torch.sign(v_pred) == torch.sign(y)).sum().item()

                pbar.update(1)
                if cpu_next is None:
                    break
                cpu_cur = cpu_next
                gpu_cur = gpu_next
            pbar.close()
        if self.gpu_id == 0:
            print('epoch {} val mse: {:.4f} val sign-acc: {:.4f}'.format(
                epoch + 1, val_loss / n, sign_correct / n), flush=True)
        self.model.train()

    def train(self, nb_epochs):
        self.model.train()
        for epoch in range(nb_epochs):
            self.train_loader.set_epoch(epoch)
            self._run_epoch(epoch)
            self._validate(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(nb_epochs - 1)


def main(rank, world_size, batch_size, cache_dir):
    """Per-process entry point — mp.spawn runs one copy of this on each GPU."""
    ddp_setup(rank, world_size)

    # Read the dataset size from meta.json and take a contiguous 95/5 split.
    # Positions are stored game-by-game, so a contiguous cut keeps whole games
    # on one side — less leakage than a per-position random split.
    with open(os.path.join(cache_dir, "meta.json")) as f:
        n_total = int(json.load(f)["n_positions"])
    n_train = int(0.95 * n_total)

    train_loader = ChunkedShuffleLoader(cache_dir, 0, n_train, batch_size,
                                        rank, world_size, shuffle=True)
    val_loader   = ChunkedShuffleLoader(cache_dir, n_train, n_total, batch_size,
                                        rank, world_size, shuffle=False)

    if rank == 0:
        print(f'Dataset: {n_total:,} positions -> '
              f'train {n_train:,}  val {n_total - n_train:,}', flush=True)
        print(f'Batches/rank/epoch: train {len(train_loader):,}  '
              f'val {len(val_loader):,}  (chunk_size={CHUNK_SIZE:,})', flush=True)

    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        nb_params = sum(p.numel() for p in net.parameters())
        print(net, flush=True)
        print(f'Parameters: {nb_params:,}', flush=True)

    # Disable Python's cyclic GC — refcount still frees everything not in a
    # cycle, and _run_epoch mops up cycles with gc.collect() every 1000 iters.
    gc.disable()

    trainer = TrainerDDP(rank, net, train_loader, val_loader)
    trainer.train(NB_EPOCHS)
    destroy_process_group()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    # Pass the cache directory (a string), not the dataset/loader — these are
    # built inside `main()` so the spawn-pickled args stay tiny.
    mp.spawn(main, args=(world_size, 256, CACHE_DIR), nprocs=world_size)
