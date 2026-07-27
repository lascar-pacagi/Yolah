"""
combined_net311x1024x64x32x1.py — multi-GPU training of the COMBINED network
((193 nnue board-input ⊕ 118 features, turn-deduplicated) → 1024 → 64 → 32 → 1)
with an AlphaZero-style scalar value head.

Rationale — "do the features help at EQUAL hidden width?" control
────────────────────────────────────────────────────────────────
This is the L1 = 1024 sibling of combined_net311x512x64x32x1.py. It holds the
hidden width equal to the pure-nnue (193x1024x64x32x1) and only ADDS the 118
features to the input, so comparing it against the pure-nnue isolates the feature
contribution at MATCHED capacity:
  • combined_311x1024  >  nnue_193x1024  → even at width 1024 the net cannot
    synthesize these features itself → a genuine capacity-bounded info gain.
  • combined_311x1024  ≈  nnue_193x1024  → at this width the features are
    redundant (the body already learns them) → they were only a convergence aid.
(The 512 sibling instead tests whether a *smaller* body can match the pure-nnue
thanks to the features.) Param count ≈ 387k — the wider 311 input into the same
1024 — vs the pure-nnue's ≈ 266k. NOTE: this trades away NNUE's incremental-update
speed (the 118 features do not update cheaply per move), so it is aimed at eval
QUALITY, not node rates.

Turn deduplication
──────────────────
The turn appears in BOTH source representations: the nnue turn byte (input index
192, fed as 0/1) and the feature TURN (the 119th feature = current_player()).
They are redundant, so preprocess_combined.py drops the FEATURE turn when it
WRITES the cache: features.u8 is 118-wide and the cache row IS the model input,
193 + 118 = 311. (The alignment guard still uses the full 119 features from the
.features.txt records at preprocess time, so dropping the column does not weaken
it.) The loader here simply concatenates — no column dropping needed.

Data path — the ALIGNED combined cache (preprocess_combined.py)
──────────────────────────────────────────────────────────────
preprocess_combined.py produces ONE row-aligned cache where row i of every
array describes the SAME board position:
    inputs.u8   : (N, 193) uint8   — board bitboards + turn (binary 0/1 bytes)
    features.u8 : (N, 118) uint8   — feature bytes WITHOUT the redundant turn
                                     (treated as v / 255)
    values.i8   : (N,)     int8    — z ∈ {-1, 0, +1} from current player's POV
    meta.json   : sidecar with shapes / counts

The two existing per-network caches (cache_nnue193, cache_features119) are NOT
row-aligned (different file ordering + different position sets), so they cannot
be concatenated row-by-row — hence the dedicated aligned preprocessor.

This script reuses the proven chunked-shuffle loader: one background producer
thread per rank reads CONTIGUOUS chunks of BOTH memmaps sequentially, applies
the SAME within-chunk permutation to each (keeping rows paired), builds the
311-d input as concat([nnue_bits (193), features (118) / 255]), pins each batch,
and pushes onto a bounded queue; the main thread overlaps the
next batch's H2D copy with the current batch's compute via a side CUDA stream.
DDP uses gradient_as_bucket_view=True + static_graph=True to avoid the periodic
~50 s NCCL all-reduce stall seen on the DataLoader variant.

fc1–fc3 keep NNUE-style clamp([0,1]) activations + int8 weight clipping
(|w| ≤ 127/64) so the trunk stays quantizable; the value head (fc4 → tanh) is
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

sys.path.append("../server")
from yolah import Yolah, Move, Square        # only needed for live-inference encoders

torch.set_float32_matmul_precision('high')

# The chunked loader does its own threaded I/O, but keep file_system sharing
# in case any other torch IPC sneaks in (SLURM+Singularity caps /dev/shm).
torch.multiprocessing.set_sharing_strategy('file_system')


# ── Input layout (the cache MUST match preprocess_combined.py) ─────────────
# preprocess_combined.py already dropped the redundant FEATURE turn at write
# time, so the cache stores 193 nnue inputs + 118 features and the cache row IS
# the model input: 193 + 118 = 311. The single turn signal kept is the nnue turn
# byte at index 192.
INPUT_SIZE_NNUE = 64 + 64 + 64 + 1           # 193 — black | white | empty | turn
NB_FEATURES     = 118                          # features stored in the cache (no TURN)
INPUT_SIZE      = INPUT_SIZE_NNUE + NB_FEATURES   # 311


# ── Chunked-shuffle double-buffered loader ──────────────────────────────────
#
# CHUNK_SIZE — number of positions read per chunk.
#   4194304 = 2048 * 2048. Two deliberate properties:
#     • an exact multiple of any plausible batch size → no ragged tail;
#     • at 193+118 = 311 bytes/position on disk it is ~1.3 GB of uint8
#       (inputs+features) — small enough for a few chunks to coexist in RAM,
#       large enough that a chunk spans many games so the within-chunk shuffle
#       decorrelates batches.
CHUNK_SIZE = 2048 * 2048

# BATCH_QUEUE_DEPTH — how many fully-built, pinned batches the producer may stay
# ahead of the GPU. Each batch (bs=1024) is ~1.3 MB float32 X + 4 KB y → 32
# batches ≈ 40 MB of pinned RAM per rank: negligible.
BATCH_QUEUE_DEPTH = 32


class ChunkedShuffleLoader:
    """
    Streaming loader over the ALIGNED (inputs.u8, features.u8, values.i8) triple.

    Iterating yields (X, y) batches with X pinned for async H2D copy:
        X : (bs, 311) float32   — concat([nnue_bits (193, 0/1),
                                          features (118) / 255])
        y : (bs,)     float32   ∈ {-1, 0, +1}

    The SAME within-chunk permutation is applied to inputs and features so the
    193-board and 118-features in each row stay paired (they already describe the
    same position by construction of the aligned cache).

    Distributed training (DDP)
    --------------------------
    With `world_size` GPUs, rank r owns chunks r, r+world_size, … so the shards
    are disjoint. Remainder chunks and the trailing partial chunk are dropped to
    guarantee EVERY rank yields the same number of batches — otherwise the
    per-step all-reduce would hang on a rank that ran fewer steps.
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

        # CHUNK_SIZE chosen so this divides cleanly for plausible batch sizes.
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

        n_inputs_cols = meta["inputs"]["shape"][1]
        if n_inputs_cols != INPUT_SIZE_NNUE:
            raise ValueError(
                f"Cache input width {n_inputs_cols} != expected {INPUT_SIZE_NNUE}. "
                "Re-run preprocess_combined.py.")
        n_feat_cols = meta["features"]["shape"][1]
        if n_feat_cols != NB_FEATURES:
            raise ValueError(
                f"Cache feature width {n_feat_cols} != expected {NB_FEATURES}. "
                "Re-run preprocess_combined.py.")
        if meta["inputs"]["shape"][0] != meta["features"]["shape"][0]:
            raise ValueError(
                "inputs and features row counts differ "
                f"({meta['inputs']['shape'][0]} vs {meta['features']['shape'][0]}) — "
                "the cache is not aligned. Re-run preprocess_combined.py.")

        n = int(meta["n_positions"])
        self.inputs = np.memmap(
            os.path.join(self.cache_dir, meta["inputs"]["path"]),
            dtype=np.uint8, mode='r', shape=tuple(meta["inputs"]["shape"]))
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
            1. read the whole chunk of inputs AND features sequentially into RAM;
            2. permute positions within the chunk (epoch+offset-seeded), SAME
               permutation for both arrays so rows stay paired;
            3. build the 311-d input concat([nnue_bits, features/255]), cast,
               pin, push to `q`.

        Why a thread (not a process): numpy bulk memcpy/astype and torch's
        pin_memory all release the Python GIL, so this thread genuinely runs at
        the same time as the main thread's GPU calls.
        """
        bs = self.batch_size
        try:
            for chunk_id in chunk_order:
                start = self.my_chunks[chunk_id]
                end   = start + self.chunk_size

                # Sequential disk reads — np.array() forces a contiguous copy.
                inp  = np.array(self.inputs[start:end])     # (C, 193) uint8 {0,1}
                feat = np.array(self.features[start:end])   # (C, 118) uint8
                val  = np.array(self.values[start:end])     # (C,)     int8

                if self.shuffle:
                    rng  = np.random.default_rng(
                        (self.epoch * 1_000_003 + start) & 0xFFFFFFFF)
                    perm = rng.permutation(self.chunk_size)
                else:
                    perm = np.arange(self.chunk_size)

                for b in range(self.batches_per_chunk):
                    idx = perm[b * bs : (b + 1) * bs]
                    # nnue bits are already 0/1 → kept as-is.
                    x_nnue = inp[idx].astype(np.float32)
                    # Features are raw bytes the net expects in [0, 1] → /255.
                    # The redundant TURN was dropped at preprocess time, so the
                    # cache is already 118-wide; concatenate directly.
                    x_feat = feat[idx].astype(np.float32) / 255.0
                    X = torch.from_numpy(
                            np.concatenate([x_nnue, x_feat], axis=1))   # (bs, 311)
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


# ── Network ────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """
    Combined feature transformer (fc1–fc3) + scalar value head (fc4 → tanh).

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
# Override at runtime with YOLAH_NB_EPOCHS — lets you change epoch count without
# rebuilding the .sif (the .sh's `singularity exec --env ...` passes it through).
NB_EPOCHS  = int(os.environ.get("YOLAH_NB_EPOCHS", "100"))
MODEL_PATH = "/mnt/"
MODEL_NAME = "combined_311x1024x64x32x1"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"

CACHE_DIR  = os.environ.get("YOLAH_COMBINED_DIR", "/cache")


def ddp_setup(rank, world_size):
    """All ranks live on one node → rendezvous over localhost; NCCL is the backend."""
    os.environ["MASTER_ADDR"] = "localhost"
    # Distinct port from the nnue (65432) and features (65433) trainers so a
    # combined run can coexist with them on the same node.
    os.environ["MASTER_PORT"] = "65435"
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
        running_loss   = 0.0
        # Three value-accuracy variants tracked in parallel:
        #   sign_correct   — torch.sign(v) == torch.sign(y); draws always count
        #                    as wrong since sign(tanh(...)) is essentially never
        #                    0 → ceiling = 1 - P(draw).
        #   bucket_correct — bucket(v) ∈ {-1, 0, +1} via |v| < 0.33; draws
        #                    correctly predicted when v_pred sits near 0.
        #   signed_correct / n_signed — sign-acc over non-draw rows only;
        #                    excludes y=0 from both numerator and denominator.
        sign_correct   = 0
        bucket_correct = 0
        signed_correct = 0
        n_signed       = 0
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

            running_loss   += loss.item() * bs
            sign_correct   += (torch.sign(v_pred) == torch.sign(y)).sum().item()
            # 3-way bucketed accuracy: bucket v_pred to {-1, 0, +1} via |v|<0.33.
            # A v_pred near 0 now correctly predicts a draw.
            bucket = torch.where(v_pred >  0.33,  1.0,
                     torch.where(v_pred < -0.33, -1.0, 0.0))
            bucket_correct += (bucket == y).sum().item()
            # Sign-acc on non-draws only (mask out y == 0).
            mask = y != 0
            if mask.any():
                signed_correct += (torch.sign(v_pred[mask]) == y[mask]).sum().item()
                n_signed       += int(mask.sum().item())
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
            signed_acc = signed_correct / max(n_signed, 1)
            print('epoch {} train mse: {:.4f} sign-acc: {:.4f} bucket-acc: {:.4f} '
                  'signed-acc: {:.4f} lr: {:.6f}'.format(
                      epoch + 1, running_loss / n, sign_correct / n,
                      bucket_correct / n, signed_acc, lr),
                  flush=True)

    def _validate(self, epoch):
        """One validation pass — no gradients, same prefetch pattern."""
        self.model.train(False)
        n = 0
        val_loss       = 0.0
        sign_correct   = 0
        bucket_correct = 0
        signed_correct = 0
        n_signed       = 0
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
                val_loss       += loss.item() * bs
                sign_correct   += (torch.sign(v_pred) == torch.sign(y)).sum().item()
                bucket = torch.where(v_pred >  0.33,  1.0,
                         torch.where(v_pred < -0.33, -1.0, 0.0))
                bucket_correct += (bucket == y).sum().item()
                mask = y != 0
                if mask.any():
                    signed_correct += (torch.sign(v_pred[mask]) == y[mask]).sum().item()
                    n_signed       += int(mask.sum().item())

                pbar.update(1)
                if cpu_next is None:
                    break
                cpu_cur = cpu_next
                gpu_cur = gpu_next
            pbar.close()
        if self.gpu_id == 0:
            signed_acc = signed_correct / max(n_signed, 1)
            print('epoch {} val mse: {:.4f} sign-acc: {:.4f} bucket-acc: {:.4f} '
                  'signed-acc: {:.4f}'.format(
                      epoch + 1, val_loss / n, sign_correct / n,
                      bucket_correct / n, signed_acc),
                  flush=True)
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
        # weights_only=True: the checkpoint is a plain tensor state_dict, so
        # refuse to unpickle arbitrary objects (also the torch>=2.6 default).
        net.load_state_dict(
            torch.load(LAST_MODEL, map_location='cpu', weights_only=True))
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
    # world_size honours CUDA_VISIBLE_DEVICES (the SLURM convention) when set,
    # falling back to device_count() only for local dev. Avoids the cluster
    # pitfall where Singularity --nv exposes all physical GPUs and mp.spawn
    # over device_count() then races for one the job does not own → NCCL hang.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    world_size = (len([x for x in cvd.split(",") if x.strip()])
                  if cvd else torch.cuda.device_count())
    print(world_size, flush=True)
    # Pass the cache directory (a string), not the dataset/loader — these are
    # built inside `main()` so the spawn-pickled args stay tiny.
    mp.spawn(main, args=(world_size, 1024, CACHE_DIR), nprocs=world_size)
