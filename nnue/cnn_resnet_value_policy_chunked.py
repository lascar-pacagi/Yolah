"""
cnn_resnet_value_policy_chunked.py
==================================
AlphaZero-style two-headed ResNet for the game of Yolah, trained with a
custom double-buffered "chunked-shuffle" data loader.

It is functionally the same network as cnn_resnet_value_policy.py — the ONLY
difference is the data pipeline:

  cnn_resnet_value_policy.py        torch DataLoader + 8 worker processes,
                                    random-access reads into the memmap cache.
  cnn_resnet_value_policy_chunked.py (this file)  one background thread per
                                    rank, large sequential chunk reads.

Why the chunked loader exists
─────────────────────────────
The cache (produced by preprocess.py) is a ~253 GB file of encoded positions.
A standard DataLoader samples positions uniformly at random, which becomes
millions of random ~256-byte reads scattered across that file. On an
HDD-backed filesystem random reads are seek-bound and collapse throughput;
the only rescue is pre-loading the whole 253 GB into RAM page cache.

This loader instead:
  • reads large CONTIGUOUS chunks (~1 GB) *sequentially* — sequential reads
    are fast even on spinning disks, so no RAM pre-warm is needed;
  • shuffles positions WITHIN a chunk and reshuffles the chunk ORDER each
    epoch. This is an approximate shuffle, but a chunk holds ~4M positions
    (~80k games), so any one batch still mixes thousands of games — gradients
    stay decorrelated;
  • does the chunk read + batch construction in ONE background thread while
    the main thread runs the GPU. A bounded queue between them is the
    rotating "ping-pong" buffer. This overlap works because numpy's memcpy /
    dtype-convert and CUDA's pin-memory all release the Python GIL, so the
    helper thread genuinely runs concurrently with GPU compute;
  • hands the main loop batch tensors that are already pinned, so the
    host→GPU copy can be asynchronous.

Everything below the loader — the network, the loss, mixed precision,
channels-last memory format, DistributedDataParallel — is standard and
identical to the sibling script.
"""

# ── Imports ─────────────────────────────────────────────────────────────────
from tqdm import tqdm                  # progress bar around the training loop
import torch
from torch import nn
import torch.multiprocessing as mp     # mp.spawn launches one process per GPU
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import gc                              # disabled inside main() to avoid GC pauses
import json                            # the cache's meta.json sidecar
import time                            # DIAGNOSTIC: chunk-load timing
import random                          # epoch-level chunk-order shuffle
import threading                       # the background producer thread
import queue as queue_mod              # bounded hand-off queue (the buffer)
import numpy as np

# The position encoder needs the Yolah game module, which lives in ../server.
sys.path.append("../server")
from yolah import Yolah, Move, Square

# Allow TF32 matmuls on Ampere+ (a speed/precision trade that is safe here).
torch.set_float32_matmul_precision('high')


# ── Board encoding ──────────────────────────────────────────────────────────
#
# Every position is a (4, 8, 8) float tensor — four 8×8 "planes":
#   plane 0 — black pieces  : 1 where a black stone sits, else 0
#   plane 1 — white pieces  : 1 where a white stone sits, else 0
#   plane 2 — empty squares : 1 where a square has been vacated (scored), else 0
#   plane 3 — turn          : a constant plane, all-0 if black is to move,
#                             all-1 if white is to move
# (preprocess.py writes this same layout to disk as uint8.)

# Policy action space: a Yolah move is (from_square, to_square), each in 0..63.
# We flatten it to a single index  action = from*64 + to  → 64*64 = 4096 logits.
NUM_ACTIONS = 64 * 64

# A terminal position has no "next move" to predict. Its policy target is set
# to this sentinel, and CrossEntropyLoss(ignore_index=...) skips those rows.
POLICY_IGNORE_INDEX = -1

# Value target z ∈ {-1, 0, +1}, from the CURRENT player's point of view:
#   +1  current player went on to win
#    0  draw
#   -1  current player lost
# The network's value head ends in tanh, so its output v ∈ (-1, +1); the
# value loss is MSE(v, z).

# Number of input planes — must match what preprocess.py wrote (a guard in the
# loader checks this against meta.json).
IN_CHANNELS = 4


def _bitboard_to_plane(n: int) -> np.ndarray:
    """
    Expand a 64-bit Yolah bitboard into an 8×8 float32 plane.

    The flattened plane is filled MSB-first: array cell i receives bit
    (63 - i) of `n`, so cell 0 holds bit 63 and cell 63 holds bit 0.

    Note this is the reverse of Yolah's own square numbering — in yolah.py
    square s is bit s, i.e. square 0 is the LEAST-significant bit — so the
    plane ends up in reverse-square order. That is harmless: a CNN does not
    care about the absolute orientation of the 8×8 plane. What matters is
    that this matches preprocess.py's bb_to_plane (np.unpackbits on a
    big-endian word, which yields the same MSB-first order), so the inference
    encoder and the cached training tensors agree. Do not "fix" the order
    here without regenerating the cache.

    Used only by encode_cnn() for inference; training reads pre-encoded
    planes straight from the cache.
    """
    b = np.zeros(64, dtype=np.float32)
    for i in range(64):
        if n & (1 << (63 - i)):
            b[i] = 1.0
    return b.reshape(8, 8)


def encode_cnn(yolah) -> torch.Tensor:
    """
    Encode a live Yolah position into the (4, 8, 8) float32 tensor the network
    expects. This is the INFERENCE-time encoder (e.g. for MCTS); training does
    not call it — preprocess.py already baked these tensors into the cache.
    """
    black = _bitboard_to_plane(yolah.black)
    white = _bitboard_to_plane(yolah.white)
    empty = _bitboard_to_plane(yolah.empty)
    # Black moves on even plies (0, 2, 4, ...), white on odd plies.
    black_to_move = (yolah.nb_plies() & 1) == 0
    turn  = np.full((8, 8), 0.0 if black_to_move else 1.0, dtype=np.float32)
    return torch.from_numpy(np.stack([black, white, empty, turn]))


# ── Chunked-shuffle double-buffered loader ──────────────────────────────────
#
# CHUNK_SIZE — number of positions read per chunk.
#   4194304 = 2048 * 2048. Two deliberate properties:
#     • it is an exact multiple of the batch size (2048) → every chunk yields
#       exactly 2048 whole batches, no ragged tail;
#     • at 256 bytes/position it is ~1 GB of uint8 — small enough that a few
#       chunks fit comfortably in RAM, large enough to span ~80k games so the
#       within-chunk shuffle decorrelates batches well.
CHUNK_SIZE = 2048 * 2048

# BATCH_QUEUE_DEPTH — how many fully-built, pinned batches the producer thread
# may run ahead of the GPU. This is the depth of the ping-pong buffer: bigger
# = more slack to absorb a slow chunk read, but more pinned RAM held at once
# (each batch ≈ 2 MB → 32 batches ≈ 64 MB per rank).
BATCH_QUEUE_DEPTH = 32


class ChunkedShuffleLoader:
    """
    Streaming data loader over the memmap cache produced by preprocess.py.

    It is a drop-in iterable for the trainer:
        • iterating it yields (state, value_target, policy_target) tensors,
          one batch at a time, already pinned for async host→GPU copy;
        • len(loader) is the number of batches in one epoch;
        • set_epoch(e) reseeds the shuffling so each epoch differs.

    Distributed training (DDP):
        With `world_size` GPUs, rank r reads a DISJOINT, strided subset of the
        chunks (chunks r, r+world_size, r+2*world_size, ...). Remainder chunks
        and any partial trailing batch are dropped so EVERY rank yields exactly
        the same number of batches — this is essential: each backward pass does
        a collective all-reduce, and if one rank ran fewer steps the others
        would hang waiting on it.
    """

    def __init__(self, cache_dir, lo, hi, batch_size, rank, world_size,
                 chunk_size=CHUNK_SIZE, shuffle=True, pin_memory=True,
                 queue_depth=BATCH_QUEUE_DEPTH):
        # cache_dir   — directory holding positions.u8 / values.i8 /
        #               policies.i16 / meta.json
        # [lo, hi)    — the slice of the global dataset this loader covers
        #               (the caller passes the train slice or the val slice)
        # batch_size  — positions per batch, PER GPU
        # rank/world_size — this process's DDP rank and the total #GPUs
        # shuffle     — True for training, False for validation
        # pin_memory  — pin produced batches (page-locked RAM → async H2D copy)
        self.cache_dir   = cache_dir
        self.batch_size  = batch_size
        self.rank        = rank
        self.world_size  = world_size
        self.chunk_size  = chunk_size
        self.shuffle     = shuffle
        self.pin_memory  = pin_memory
        self.queue_depth = queue_depth
        self.epoch       = 0
        self._open()     # open the memmaps

        # Tile the half-open range [lo, hi) into whole, non-overlapping chunks.
        # `all_starts` is the list of every chunk's starting position. The
        # `hi - chunk_size + 1` upper bound discards a final partial chunk.
        all_starts = list(range(lo, hi - chunk_size + 1, chunk_size))

        # Shard the chunks across ranks. `per_rank` is floor-divided so every
        # rank gets the SAME count; up to (world_size - 1) leftover chunks are
        # dropped. Rank r then takes a strided slice starting at index r.
        per_rank = len(all_starts) // world_size
        self.my_chunks = all_starts[rank : per_rank * world_size : world_size]

        # With CHUNK_SIZE an exact multiple of batch_size this division is
        # clean; otherwise the trailing partial batch of each chunk is dropped.
        self.batches_per_chunk = chunk_size // batch_size

        # Total batches this rank yields per epoch — identical across ranks.
        self.n_batches = len(self.my_chunks) * self.batches_per_chunk

    def _open(self):
        """
        Open the three cache files as read-only memory maps.

        np.memmap does not read any data here — it only maps the file into the
        process's virtual address space. Bytes are paged in lazily by the OS as
        the producer thread actually touches them, and the page cache is shared
        by every process on the node (so 3 ranks do not triple RAM use).
        """
        with open(os.path.join(self.cache_dir, "meta.json")) as f:
            meta = json.load(f)

        # Fail loudly on a stale cache (e.g. a 6-plane cache from an older
        # preprocess.py) rather than silently mis-reading bytes.
        n_planes = meta["positions"]["shape"][1]
        if n_planes != IN_CHANNELS:
            raise ValueError(
                f"Cache has {n_planes} planes but training expects {IN_CHANNELS}. "
                "Re-run preprocess.py.")

        n = int(meta["n_positions"])
        # positions.u8  : (N, 4, 8, 8) uint8  — the board planes
        self.positions = np.memmap(
            os.path.join(self.cache_dir, meta["positions"]["path"]),
            dtype=np.uint8, mode='r', shape=tuple(meta["positions"]["shape"]))
        # values.i8     : (N,)         int8   — value target z ∈ {-1, 0, +1}
        self.values = np.memmap(
            os.path.join(self.cache_dir, meta["values"]["path"]),
            dtype=np.int8, mode='r', shape=(n,))
        # policies.i16  : (N,)         int16  — action index, or -1 if terminal
        self.policies = np.memmap(
            os.path.join(self.cache_dir, meta["policies"]["path"]),
            dtype=np.int16, mode='r', shape=(n,))

    def set_epoch(self, epoch):
        """Reseed shuffling for a new epoch (chunk order + within-chunk perm)."""
        self.epoch = epoch

    def __len__(self):
        """Number of batches per epoch — lets tqdm show a correct total."""
        return self.n_batches

    def _producer(self, chunk_order, q):
        """
        The background worker thread (one per loader, started by __iter__).

        For each chunk, in the epoch's shuffled order:
          1. read the whole chunk sequentially from the memmap into RAM;
          2. build a permutation that shuffles positions within the chunk;
          3. slice it into batches, convert dtypes, optionally pin, and push
             each batch onto the bounded queue `q`.

        The queue's `maxsize` is the back-pressure mechanism: once it holds
        `queue_depth` batches, `q.put(...)` blocks, so this thread can run at
        most that far ahead of the consumer and never grows memory unbounded.

        Why a thread (not a process) overlaps the GPU: numpy's bulk memcpy /
        astype and torch's pin_memory all drop the GIL for their heavy work,
        and so does every CUDA call in the main thread — so the two threads
        truly run at the same time.
        """
        bs = self.batch_size
        try:
            for chunk_id in chunk_order:
                start = self.my_chunks[chunk_id]
                end   = start + self.chunk_size

                # --- step 1: sequential read of the chunk into RAM ----------
                # np.array(memmap_slice) forces a contiguous copy, i.e. an
                # actual sequential read of this region off disk.
                # DIAGNOSTIC: time this read — it is T_load, the producer's
                # only blocking stall (see the chunk-load timing print below).
                _t0 = time.time()
                pos = np.array(self.positions[start:end])   # (C, 4, 8, 8) uint8
                val = np.array(self.values[start:end])      # (C,)         int8
                pol = np.array(self.policies[start:end])    # (C,)         int16
                t_load = time.time() - _t0

                # --- step 2: within-chunk shuffle ---------------------------
                # Seed depends on both epoch and chunk start, so a given chunk
                # is permuted differently every epoch. `& 0xFFFFFFFF` keeps the
                # seed inside the 32-bit range numpy's Generator expects.
                _t0 = time.time()
                if self.shuffle:
                    rng  = np.random.default_rng(
                        (self.epoch * 1_000_003 + start) & 0xFFFFFFFF)
                    perm = rng.permutation(self.chunk_size)
                else:
                    # Validation: keep natural order — no need to shuffle.
                    perm = np.arange(self.chunk_size)
                t_perm = time.time() - _t0

                # --- step 3: cut the chunk into batches ---------------------
                # DIAGNOSTIC: accumulate where the producer spends its time.
                #   t_gather — fancy-index + dtype convert
                #   t_pin    — pin_memory() calls
                #   t_put    — time BLOCKED in q.put (back-pressure). If t_put
                #              dominates the chunk, the producer is faster than
                #              the GPU (consumer-bound); if t_gather/t_pin
                #              dominate, the producer can't keep up.
                t_gather = t_pin = t_put = 0.0
                for b in range(self.batches_per_chunk):
                    # `idx` are the rows of this chunk that form batch b.
                    idx = perm[b * bs : (b + 1) * bs]
                    # Gather rows and convert to the dtypes the model wants:
                    #   X : float32 board planes
                    #   v : float32 value target
                    #   p : int64   policy target (CrossEntropyLoss needs int64)
                    _t = time.time()
                    X = torch.from_numpy(pos[idx].astype(np.float32))
                    v = torch.from_numpy(val[idx].astype(np.float32))
                    p = torch.from_numpy(pol[idx].astype(np.int64))
                    t_gather += time.time() - _t
                    # Pin into page-locked RAM so the later .to(gpu,
                    # non_blocking=True) can be a true asynchronous DMA copy.
                    if self.pin_memory:
                        _t = time.time()
                        X, v, p = X.pin_memory(), v.pin_memory(), p.pin_memory()
                        t_pin += time.time() - _t
                    # Blocks here if the consumer is behind (queue full).
                    _t = time.time()
                    q.put((X, v, p))
                    t_put += time.time() - _t

                # DIAGNOSTIC: one consolidated line per chunk; the wall-clock
                # stamp lets you line it up against nvidia-smi idle phases.
                # Remove this block once the stall is located.
                print(f"[{time.strftime('%F %T')}] [rank {self.rank}] chunk: "
                      f"read {t_load:5.1f}s shuffle {t_perm:5.2f}s | build "
                      f"gather {t_gather:6.1f}s pin {t_pin:6.1f}s "
                      f"q.put(wait) {t_put:7.1f}s  "
                      f"({self.batches_per_chunk} batches)", flush=True)
        except Exception as e:                              # pragma: no cover
            # A crash in a thread is otherwise silent; surface it.
            print(f"[ChunkedShuffleLoader] producer error: {e}", flush=True)
        finally:
            # Always post the end-of-epoch sentinel, even on error, so the
            # consumer's loop terminates instead of blocking forever.
            q.put(None)

    def __iter__(self):
        """
        Start one epoch: spawn the producer thread and yield its batches.

        Called once per epoch (by the trainer's `for ... in loader`). Each call
        creates a fresh queue + thread, so epochs are independent.
        """
        # Decide the order in which this rank visits its chunks this epoch.
        order = list(range(len(self.my_chunks)))
        if self.shuffle:
            random.Random(self.epoch).shuffle(order)

        # The bounded queue IS the rotating buffer between producer & consumer.
        q = queue_mod.Queue(maxsize=self.queue_depth)
        # daemon=True → the thread will not keep the process alive on exit.
        t = threading.Thread(target=self._producer, args=(order, q), daemon=True)
        t.start()

        # Consume batches until the producer posts the None sentinel.
        # DIAGNOSTIC: per 100-batch window, log avg+MAX q.get wait and MIN
        # observed queue size. Window-max catches bursty stalls that point
        # sampling missed — a 50 s wait *anywhere* in the window now shows up.
        # Remove this block once the stall is located.
        n = 0
        win_wait_sum = 0.0
        win_wait_max = 0.0
        win_q_min    = self.queue_depth
        while True:
            _t = time.time()
            item = q.get()
            wait = time.time() - _t
            qs   = q.qsize()
            n += 1
            win_wait_sum += wait
            win_wait_max  = max(win_wait_max, wait)
            win_q_min     = min(win_q_min, qs)
            if n % 100 == 0:
                print(f"[{time.strftime('%F %T')}] [rank {self.rank}] "
                      f"batch {n}: q.get avg {win_wait_sum / 100:.3f}s  "
                      f"max {win_wait_max:6.2f}s  "
                      f"queue min {win_q_min:2d}/{self.queue_depth}  "
                      f"(last 100)", flush=True)
                win_wait_sum = 0.0
                win_wait_max = 0.0
                win_q_min    = self.queue_depth
            if item is None:
                break
            yield item
        t.join()   # tidy up the (already finished) producer thread


# ── Network ─────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """
    One pre-activation residual block (the "ResNet v2" ordering:
    BatchNorm → ReLU → Conv, twice, then add the input back).

    The skip connection (x + residual) lets gradients flow straight through,
    which is what makes a deep (30-block) stack trainable.
    """
    def __init__(self, channels: int):
        super().__init__()
        # bias=False on the convs because the following BatchNorm has its own
        # learnable shift, making a conv bias redundant.
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        residual = x                                  # save the input
        x = self.conv1(torch.relu(self.bn1(x)))       # BN → ReLU → Conv
        x = self.conv2(torch.relu(self.bn2(x)))       # BN → ReLU → Conv
        return x + residual                           # add the skip connection


class Net(nn.Module):
    """
    AlphaZero-style two-headed network.

      Trunk : (B, 4, 8, 8) ─ input conv ─ N residual blocks ─ BN ─ ReLU
              → a shared (B, C, 8, 8) feature map.

      Value head  : 1×1 conv to 1 channel → flatten(64) → FC → ReLU → FC
                    → tanh → a scalar v ∈ (-1, +1) per position.
                    Interpretation: expected game outcome for the player to
                    move; P(win) = (v + 1) / 2.

      Policy head : 1×1 conv to 2 channels → flatten(128) → FC
                    → 4096 logits, one per (from, to) action.

    forward() returns the pair (value, policy_logits).
    """
    def __init__(self, channels: int = 256, nb_blocks: int = 30,
                 value_fc_size: int = 256, num_actions: int = NUM_ACTIONS):
        super().__init__()

        # Stem: lift the 4 input planes to `channels` feature maps.
        self.input_conv = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Shared trunk: nb_blocks residual blocks, then a final BN.
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(nb_blocks)])
        self.output_bn  = nn.BatchNorm2d(channels)

        # Value head. The 1×1 conv squeezes C channels down to 1, giving a
        # (B, 1, 8, 8) map = 64 numbers, which the FC layers turn into a scalar.
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(64, value_fc_size)
        self.value_fc2  = nn.Linear(value_fc_size, 1)

        # Policy head. 1×1 conv to 2 channels → (B, 2, 8, 8) = 128 numbers →
        # one FC straight to the 4096-way action logits.
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 64, num_actions)

    def forward(self, x):
        # Shared trunk.
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = torch.relu(self.output_bn(x))             # shared (B, C, 8, 8)

        # Value head: (B,C,8,8) → (B,1,8,8) → (B,64) → (B,fc) → (B,1) → (B,)
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)                              # keep batch dim, flatten rest
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1) # tanh → (-1,1); drop last dim

        # Policy head: (B,C,8,8) → (B,2,8,8) → (B,128) → (B, NUM_ACTIONS)
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        p = self.policy_fc(p)                         # raw logits (no softmax)
        return v, p

    def clip(self):
        # No-op: kept so the training loop can call model.clip() uniformly
        # across networks (the NNUE variants clamp weights here for int8
        # quantization; this CNN does not need it).
        pass


# ── Training configuration ──────────────────────────────────────────────────
NB_EPOCHS  = 1                              # passes over the (huge) dataset
MODEL_PATH = "/mnt/"                        # checkpoint dir (bind-mounted)
MODEL_NAME = "cnn_resnet_256x30_value_policy"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt" # if present, resume from it
# Cache directory inside the container (the .sh bind-mounts the host cache
# onto /cache); overridable via env var for local runs.
CACHE_DIR  = os.environ.get("YOLAH_CACHE_DIR", "/cache")

# Total loss = VALUE_LOSS_WEIGHT * MSE(value) + POLICY_LOSS_WEIGHT * CE(policy).
# AlphaZero weights both at 1.0; raise the value weight if value learning lags.
VALUE_LOSS_WEIGHT  = 1.0
POLICY_LOSS_WEIGHT = 1.0


def ddp_setup(rank, world_size):
    """
    Initialise the process group for DistributedDataParallel.

    All ranks live on one node here, so they rendezvous over localhost. NCCL is
    the GPU-to-GPU communication backend used for the gradient all-reduce.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65437"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class TrainerDDP:
    """Owns one GPU's model replica, optimizer, and the train/validate loops."""

    def __init__(self, gpu_id, model, train_loader, val_loader, save_every=1):
        self.gpu_id = gpu_id

        # Move the model to this rank's GPU and convert it to channels-last
        # (NHWC) memory format: cuDNN's NHWC convolution kernels are faster on
        # tensor-core GPUs, and it makes the gradient layout match what DDP's
        # buckets expect (silences the "grad strides" warning).
        self.model = model.to(gpu_id).to(memory_format=torch.channels_last)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.save_every   = save_every

        # Adam with a little weight decay (L2 regularisation).
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001,
                                          weight_decay=1e-4)
        # Value head: regression → mean-squared error against z ∈ {-1, 0, +1}.
        self.value_loss_fn = nn.MSELoss()
        # Policy head: 4096-way classification; ignore_index skips terminal
        # positions whose policy target is POLICY_IGNORE_INDEX.
        self.policy_loss_fn = nn.CrossEntropyLoss(ignore_index=POLICY_IGNORE_INDEX)
        # Cosine schedule: lr eases from 1e-3 down to 1e-5 over the run.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NB_EPOCHS, eta_min=1e-5)

        # Mixed precision. bf16 has wide dynamic range and needs no loss
        # scaling, but only Ampere+ (compute capability major >= 8) has bf16
        # tensor cores. Turing GPUs (e.g. Quadro RTX 8000, cc 7.5) would run
        # bf16 emulated and SLOW — they do, however, have fp16 tensor cores.
        # So: gate strictly on the capability major version.
        major, _ = torch.cuda.get_device_capability(gpu_id)
        self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        # fp16 gradients can underflow → need a GradScaler. bf16 does not, so
        # the scaler is disabled there and its scale/step/update become no-ops.
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=(self.amp_dtype == torch.float16))

        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        # Wrap in DDP: every backward pass all-reduces gradients across ranks.
        # gradient_as_bucket_view: gradients live directly in the all-reduce
        # bucket memory → one less memcpy + one less stream sync per bucket.
        # static_graph: tells DDP the autograd graph is identical every step
        # (true for us — no conditionals, no unused params, no model surgery)
        # so the reducer is built once and DDP can fuse/overlap aggressively.
        # Both are aimed at the periodic ~50 s all-reduce stall observed on
        # multi-GPU runs that vanishes on single-GPU.
        self.model = DDP(self.model, device_ids=[gpu_id],
                         gradient_as_bucket_view=True, static_graph=True)
        # torch.compile JIT-fuses the graph for extra speed (first iterations
        # are slow while it compiles, then much faster).
        self.model = torch.compile(self.model)
        # A side CUDA stream used to prefetch the next batch's H2D copy.
        self.stream = torch.cuda.Stream(device=gpu_id)

    def _save_checkpoint(self, epoch):
        """Save the raw model weights (unwrap DDP via .module)."""
        torch.save(self.model.module.state_dict(),
                   f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _compute_loss(self, value_pred, policy_logits, value_target, policy_target):
        """Combined value + policy loss; also returns the two parts for logging."""
        v_loss = self.value_loss_fn(value_pred, value_target)
        p_loss = self.policy_loss_fn(policy_logits, policy_target)
        return (VALUE_LOSS_WEIGHT  * v_loss
              + POLICY_LOSS_WEIGHT * p_loss, v_loss, p_loss)

    @staticmethod
    def _policy_correct(policy_logits, policy_target):
        """
        Count correct top-1 policy predictions, ignoring terminal positions.
        Returns (n_correct, n_scored) so the caller can accumulate an accuracy.
        """
        mask = policy_target != POLICY_IGNORE_INDEX     # valid (non-terminal) rows
        if mask.sum().item() == 0:
            return 0, 0
        preds = torch.argmax(policy_logits, dim=1)
        correct = ((preds == policy_target) & mask).sum().item()
        return correct, int(mask.sum().item())

    def _h2d_async(self, batch_cpu):
        """
        Issue an asynchronous host->GPU transfer for one (X, v, p) batch on
        self.stream, return the GPU tensors.

        PREFETCH: callers `wait_stream` on the default stream *before* using
        these tensors. Because the H2D runs on the side stream, it can proceed
        concurrently with the previous iteration's compute on the default
        stream — that is the whole point of the prefetch.
        """
        X_cpu, v_cpu, p_cpu = batch_cpu
        with torch.cuda.stream(self.stream):
            X = X_cpu.to(self.gpu_id, non_blocking=True)
            v = v_cpu.to(self.gpu_id, non_blocking=True)
            p = p_cpu.to(self.gpu_id, non_blocking=True)
        return (X, v, p)

    def _run_epoch(self, epoch):
        """One training pass over this rank's shard of the data.

        PREFETCH: while iteration N is computing on the default stream, the
        H2D transfer for iteration N+1 is in flight on self.stream. The body's
        wall-time therefore measures only compute, not transfer, and the
        transfer is hidden behind compute instead of adding to the iteration.
        """
        n = 0                                   # positions seen
        v_correct = 0                           # value sign matches (win/lose)
        p_correct = 0                           # policy top-1 hits
        p_total   = 0                           # non-terminal positions
        v_running, p_running = 0.0, 0.0         # running loss sums
        it = 0                                  # DIAGNOSTIC: iteration counter
        win_sum = 0.0                           # DIAGNOSTIC: body-time sum/window
        win_max = 0.0                           # DIAGNOSTIC: body-time max/window

        # Pre-load the first batch onto the GPU (side stream). We keep the
        # CPU batch reference (cpu_cur) alive across the iteration so its
        # pinned tensors are not freed before the async H2D completes.
        loader_iter = iter(self.train_loader)
        cpu_cur = next(loader_iter, None)
        if cpu_cur is None:
            return
        gpu_cur = self._h2d_async(cpu_cur)

        pbar = tqdm(total=len(self.train_loader))
        while True:
            _t_body = time.time()               # DIAGNOSTIC: time the iteration

            # Make the default stream wait until the current batch's H2D is
            # complete on self.stream. This is just a stream dependency — the
            # host does not block here.
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
            X, v_target, p_target = gpu_cur
            bs = len(X)
            n += bs

            # Issue the NEXT iteration's H2D RIGHT NOW on self.stream. It will
            # run concurrently with this iteration's forward/backward/step on
            # the default stream — that overlap is the whole point.
            cpu_next = next(loader_iter, None)
            if cpu_next is not None:
                gpu_next = self._h2d_async(cpu_next)

            # Match the model's channels-last memory format (sub-ms reformat).
            X = X.contiguous(memory_format=torch.channels_last)

            self.optimizer.zero_grad()
            with torch.autocast('cuda', dtype=self.amp_dtype):
                v_pred, p_logits = self.model(X)
                loss, v_loss, p_loss = self._compute_loss(v_pred, p_logits,
                                                          v_target, p_target)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics (.item() pulls scalars back to the CPU).
            v_running += v_loss.item() * bs
            p_running += p_loss.item() * bs
            v_correct += (torch.sign(v_pred) == torch.sign(v_target)).sum().item()
            pc, pt = self._policy_correct(p_logits, p_target)
            p_correct += pc
            p_total   += pt

            self.model.module.clip()            # no-op for this CNN

            # DIAGNOSTIC: per-iteration body wall-time. With prefetch the body
            # no longer includes the H2D, so this is purely compute.
            it += 1
            _dt = time.time() - _t_body
            win_sum += _dt
            win_max  = max(win_max, _dt)
            if it % 100 == 0 and self.gpu_id == 0:
                print(f"[{time.strftime('%F %T')}] [rank 0] iter {it}: "
                      f"body avg {win_sum / 100:.3f}s  max {win_max:.2f}s  "
                      f"(last 100 iters)", flush=True)
                win_sum = 0.0
                win_max = 0.0
            if it % 1000 == 0:
                _t_gc = time.time()
                gc.collect()
                if self.gpu_id == 0:
                    print(f"[{time.strftime('%F %T')}] [rank 0] "
                          f"gc.collect() took {time.time() - _t_gc:.2f}s",
                          flush=True)

            pbar.update(1)

            # Advance to the next batch, or stop if the loader is exhausted.
            # Re-binding here drops the previous iter's CPU tensors — by now
            # their H2D has long completed (we waited for it at the top of
            # this iter), so it is safe to free them.
            if cpu_next is None:
                break
            cpu_cur = cpu_next
            gpu_cur = gpu_next

        pbar.close()
        self.scheduler.step()                   # advance the lr schedule

        # Only rank 0 logs, so the output is not printed three times.
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
        """One pass over the validation shard — no gradients, no optimizer.
        Uses the same prefetch pattern as _run_epoch.
        """
        self.model.train(False)                 # eval mode (freezes BatchNorm)
        n = 0
        v_correct = 0
        p_correct = 0
        p_total   = 0
        v_running, p_running = 0.0, 0.0
        with torch.no_grad():                   # no autograd graph → less memory
            loader_iter = iter(self.val_loader)
            cpu_cur = next(loader_iter, None)
            if cpu_cur is None:
                self.model.train()
                return
            gpu_cur = self._h2d_async(cpu_cur)
            pbar = tqdm(total=len(self.val_loader))
            while True:
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
                X, v_target, p_target = gpu_cur
                bs = len(X)
                n += bs

                cpu_next = next(loader_iter, None)
                if cpu_next is not None:
                    gpu_next = self._h2d_async(cpu_next)

                X = X.contiguous(memory_format=torch.channels_last)
                with torch.autocast('cuda', dtype=self.amp_dtype):
                    v_pred, p_logits = self.model(X)
                    _, v_loss, p_loss = self._compute_loss(v_pred, p_logits,
                                                           v_target, p_target)
                v_running += v_loss.item() * bs
                p_running += p_loss.item() * bs
                v_correct += (torch.sign(v_pred) == torch.sign(v_target)).sum().item()
                pc, pt = self._policy_correct(p_logits, p_target)
                p_correct += pc
                p_total   += pt

                pbar.update(1)

                if cpu_next is None:
                    break
                cpu_cur = cpu_next
                gpu_cur = gpu_next
            pbar.close()
        if self.gpu_id == 0:
            p_acc = (p_correct / p_total) if p_total > 0 else 0.0
            print(f'epoch {epoch+1} '
                  f'val value_mse: {v_running/n:.4f} '
                  f'value_sign_acc: {v_correct/n:.4f} '
                  f'policy_ce: {p_running/n:.4f} '
                  f'policy_acc: {p_acc:.4f}', flush=True)
        self.model.train()                      # back to train mode

    def train(self, nb_epochs):
        """Top-level loop: for each epoch, train, validate, checkpoint."""
        self.model.train()
        for epoch in range(nb_epochs):
            # Reseed the loader's shuffling so this epoch differs from the last.
            self.train_loader.set_epoch(epoch)
            self._run_epoch(epoch)
            self._validate(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(nb_epochs - 1)    # always save the final model


def main(rank, world_size, batch_size, cache_dir):
    """
    Per-process entry point — mp.spawn runs one copy of this on each GPU,
    passing the GPU's index as `rank`.
    """
    ddp_setup(rank, world_size)

    # Read the dataset size from meta.json and take a contiguous 95/5 split.
    # The split is contiguous (not random) because positions are stored
    # game-by-game, so a contiguous cut keeps whole games on one side — less
    # train/val leakage than a per-position random split.
    #
    # The loaders are built HERE, inside the spawned process — never passed in
    # through mp.spawn. np.memmap is a lazy mmap (virtual address space only),
    # so constructing a loader is cheap and the OS page cache is shared.
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
    # Resume from a previous checkpoint if one exists.
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        nb_params = sum(p.numel() for p in net.parameters())
        print(net, flush=True)
        print(f'Parameters: {nb_params:,}', flush=True)

    # DIAGNOSTIC: disable Python's cyclic GC for the duration of training.
    # Reference counting still frees everything that is not in a reference
    # cycle; we mop up cycles ourselves with gc.collect() every 1000 iters
    # in _run_epoch. This eliminates the unpredictable Gen-2 GC pauses that
    # are the leading suspect for the periodic ~50 s GPU idle.
    gc.disable()

    trainer = TrainerDDP(rank, net, train_loader, val_loader)
    trainer.train(NB_EPOCHS)
    destroy_process_group()                     # clean up the NCCL group


if __name__ == "__main__":
    print(torch.cuda.is_available())
    # One training process per visible GPU.
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    # mp.spawn calls main(rank, world_size, 2048, CACHE_DIR) on each GPU; it
    # prepends `rank` itself. batch_size is PER GPU → effective batch is
    # 2048 * world_size.
    mp.spawn(main, args=(world_size, 2048, CACHE_DIR), nprocs=world_size)
