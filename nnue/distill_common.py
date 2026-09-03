"""
distill_common.py — code shared by the knowledge-distillation scripts:

    distill_teacher_labels.py           label a cache with the ResNet teacher
    distill_boards_for_features.py      board sidecar for the features cache
    nnue193x1024x64x32x1_distill.py     distil into the 193-input NNUE
    features_net119x256x64x1_distill.py distil into the 119-feature network

What is distilled
─────────────────
The teacher is the two-headed ResNet trained by cnn_resnet_value_policy_chunked.py
(value head: tanh ∈ (-1, 1) for the player to move). The students are the small
value-only networks that the C++ engine can run millions of times per second.
Their original training target is the game outcome z ∈ {-1, 0, +1}, a very
noisy label (one sample of a random variable). The teacher's value v_t is a
much less noisy estimate of the same quantity, so a student trained on v_t
learns faster and generalises better — this is the classic "soft target"
argument of Hinton et al. The distillation loss mixes both:

    L = α · MSE(v_s, v_t) + (1 - α) · MSE(v_s, z)

Optionally the teacher label is the average of the teacher's value over the 8
symmetries of the board (test-time augmentation): the game is invariant under
the dihedral group of the square but the CNN is not exactly, so averaging
gives a strictly better label at 8× the labelling cost.

Reported metrics ("training set" vs "training set augmented")
────────────────────────────────────────────────────────────
For every epoch, on the training shard and on the validation shard:
  [vs outcome]  the student against the ground truth z — mse, sign-acc,
                bucket-acc, signed-acc, exactly the metrics of the original
                trainers, so the numbers are comparable with the non-distilled
                runs;
  [vs teacher]  the student against the augmented targets v_t — mse,
                sign-agreement, bucket-agreement;
  [teacher]     the teacher itself against z on the same positions — the
                ceiling the student is being pulled towards.

Cache layout consumed here (see preprocess_nnue.py / preprocess_features.py)
───────────────────────────────────────────────────────────────────────────
    inputs.u8 | features.u8   student inputs                (N, 193) / (N, 119)
    boards.u8                 board rows for the features cache, same layout
                              as inputs.u8 (written by distill_boards_for_features.py)
    values.i8                 z from the current player's point of view (N,)
    teacher_values.f16        v_t as float16 (N,)          ← distill_teacher_labels.py
    teacher_done.u8           one byte per labelling chunk (1 = labelled)
    teacher_meta.json         labelling parameters
"""
import os
import json
import random
import threading
import queue as queue_mod
import numpy as np
import torch

# ── constants shared with the other scripts ────────────────────────────────
BOARD_ROW = 64 + 64 + 64 + 1          # 193: black bits, white bits, empty bits, turn
TEACHER_VALUES = "teacher_values.f16"
TEACHER_DONE   = "teacher_done.u8"
TEACHER_META   = "teacher_meta.json"
LABEL_CHUNK    = 1 << 20              # positions per labelling chunk (1,048,576)


# ── teacher ────────────────────────────────────────────────────────────────
def load_teacher(checkpoint, device):
    """
    Build the ResNet from cnn_resnet_value_policy_chunked.Net and load the
    checkpoint (a state_dict, DDP prefix tolerated). Returns the eval-mode
    model on `device`, in channels-last memory format.
    """
    from cnn_resnet_value_policy_chunked import Net   # imports ../server/yolah.py
    sd = torch.load(checkpoint, map_location="cpu")
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    C = sd["input_conv.0.weight"].shape[0]
    B = sum(1 for k in sd if k.endswith(".conv1.weight"))
    net = Net(channels=C, nb_blocks=B,
              value_fc_size=sd["value_fc1.weight"].shape[0],
              num_actions=sd["policy_fc.weight"].shape[0])
    net.load_state_dict(sd)
    net.eval()
    return net.to(device).to(memory_format=torch.channels_last)


def boards_to_planes(rows):
    """
    (B, 193) uint8 tensor of board rows → (B, 4, 8, 8) float32 planes.

    Row layout (preprocess_nnue.py): bytes 0-63 black, 64-127 white, 128-191
    empty, each MSB-first (np.unpackbits of a big-endian uint64) — which is
    exactly the cell order of the CNN planes written by preprocess.py — and
    byte 192 is the turn (0 black to move, 1 white), broadcast to a plane.
    """
    b = rows.shape[0]
    planes = rows[:, :192].reshape(b, 3, 8, 8).float()
    turn = rows[:, 192].float().view(b, 1, 1, 1).expand(b, 1, 8, 8)
    return torch.cat([planes, turn], dim=1)


def symmetries(planes):
    """
    Yield the 8 images of a (B, 4, 8, 8) batch under the dihedral group of the
    square (4 rotations × optional mirror). The value of a Yolah position is
    invariant under all of them.
    """
    for k in range(4):
        r = torch.rot90(planes, k, dims=(2, 3))
        yield r
        yield torch.flip(r, dims=(3,))


@torch.no_grad()
def teacher_values(net, rows, device, tta=False, amp_dtype=torch.float16):
    """
    Teacher value for a (B, 193) uint8 batch (numpy or tensor), as a float32
    CPU tensor. With tta=True the value is averaged over the 8 symmetries.
    """
    if not torch.is_tensor(rows):
        rows = torch.from_numpy(np.ascontiguousarray(rows))
    rows = rows.to(device, non_blocking=True)
    planes = boards_to_planes(rows)
    views = symmetries(planes) if tta else [planes]
    acc = None
    n = 0
    for x in views:
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            v, _ = net(x)
        v = v.float()
        acc = v if acc is None else acc + v
        n += 1
    return (acc / n).cpu()


# ── teacher label bookkeeping ──────────────────────────────────────────────
def open_teacher_labels(cache_dir, n_positions, mode="r"):
    """memmaps of (teacher_values.f16, teacher_done.u8); returns (values, done)."""
    n_chunks = (n_positions + LABEL_CHUNK - 1) // LABEL_CHUNK
    values = np.memmap(os.path.join(cache_dir, TEACHER_VALUES), dtype=np.float16,
                       mode=mode, shape=(n_positions,))
    done = np.memmap(os.path.join(cache_dir, TEACHER_DONE), dtype=np.uint8,
                     mode=mode, shape=(n_chunks,))
    return values, done


def labelled_ranges(cache_dir, n_positions):
    """
    The list of [lo, hi) position ranges whose teacher labels are complete,
    from teacher_done.u8 (merging consecutive chunks). Raises if no labels.
    """
    done_path = os.path.join(cache_dir, TEACHER_DONE)
    if not os.path.isfile(done_path):
        raise FileNotFoundError(f"{done_path} not found — run distill_teacher_labels.py first")
    _, done = open_teacher_labels(cache_dir, n_positions, mode="r")
    ranges = []
    for c, flag in enumerate(np.asarray(done)):
        if not flag:
            continue
        lo, hi = c * LABEL_CHUNK, min((c + 1) * LABEL_CHUNK, n_positions)
        if ranges and ranges[-1][1] == lo:
            ranges[-1][1] = hi
        else:
            ranges.append([lo, hi])
    if not ranges:
        raise RuntimeError("no labelled chunk in teacher_done.u8")
    return ranges


# ── chunked loader with teacher labels ─────────────────────────────────────
CHUNK_SIZE = 2048 * 2048        # same as the original trainers
BATCH_QUEUE_DEPTH = 32


class DistillLoader:
    """
    Chunked-shuffle double-buffered loader (see the original trainers for the
    rationale) that yields (X, z, v_t) batches:
        X   : (bs, input_size) float32 student input
        z   : (bs,)            float32 game outcome ∈ {-1, 0, +1}
        v_t : (bs,)            float32 teacher value ∈ (-1, +1)

    `inputs` is the memmap of the student inputs (uint8), `scale` the factor
    applied to them (1.0 for the NNUE bits, 1/255 for the feature bytes).
    Only chunks whose teacher labels are complete are used (`ranges`), so a
    partially labelled cache trains on the labelled part.
    """

    def __init__(self, cache_dir, inputs, values, teacher, ranges, lo, hi, batch_size,
                 rank, world_size, scale=1.0, chunk_size=CHUNK_SIZE, shuffle=True,
                 pin_memory=True, queue_depth=BATCH_QUEUE_DEPTH):
        self.inputs, self.values, self.teacher = inputs, values, teacher
        self.batch_size, self.rank, self.world_size = batch_size, rank, world_size
        self.scale, self.chunk_size, self.shuffle = scale, chunk_size, shuffle
        self.pin_memory, self.queue_depth = pin_memory, queue_depth
        self.epoch = 0
        # Whole chunks inside [lo, hi) that are entirely labelled.
        all_starts = []
        for start in range(lo, hi - chunk_size + 1, chunk_size):
            end = start + chunk_size
            if any(r_lo <= start and end <= r_hi for r_lo, r_hi in ranges):
                all_starts.append(start)
        per_rank = len(all_starts) // world_size
        self.my_chunks = all_starts[rank: per_rank * world_size: world_size]
        self.batches_per_chunk = chunk_size // batch_size
        self.n_batches = len(self.my_chunks) * self.batches_per_chunk
        self.n_positions = len(all_starts) * chunk_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_batches

    def _producer(self, chunk_order, q):
        bs = self.batch_size
        try:
            for chunk_id in chunk_order:
                start = self.my_chunks[chunk_id]
                end = start + self.chunk_size
                inp = np.array(self.inputs[start:end])
                val = np.array(self.values[start:end])
                tea = np.array(self.teacher[start:end])
                if self.shuffle:
                    rng = np.random.default_rng((self.epoch * 1_000_003 + start) & 0xFFFFFFFF)
                    perm = rng.permutation(self.chunk_size)
                else:
                    perm = np.arange(self.chunk_size)
                for b in range(self.batches_per_chunk):
                    idx = perm[b * bs: (b + 1) * bs]
                    X = torch.from_numpy(inp[idx].astype(np.float32))
                    if self.scale != 1.0:
                        X *= self.scale
                    z = torch.from_numpy(val[idx].astype(np.float32))
                    vt = torch.from_numpy(tea[idx].astype(np.float32))
                    if self.pin_memory:
                        X, z, vt = X.pin_memory(), z.pin_memory(), vt.pin_memory()
                    q.put((X, z, vt))
        except Exception as e:                              # pragma: no cover
            print(f"[DistillLoader] producer error: {e}", flush=True)
        finally:
            q.put(None)

    def __iter__(self):
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


# ── metrics ────────────────────────────────────────────────────────────────
def bucket(v):
    """Map values to {-1, 0, +1} with the |v| < 0.33 draw band (as in the trainers)."""
    return torch.where(v > 0.33, 1.0, torch.where(v < -0.33, -1.0, 0.0))


class Metrics:
    """
    Accumulates, over an epoch, the three metric groups described in the
    module docstring. Every update takes the batch's student output v_s, the
    outcome z and the teacher value v_t (all 1-d float tensors on the GPU).
    """

    def __init__(self):
        self.n = 0
        self.loss = 0.0
        # student vs outcome
        self.mse_z = 0.0; self.sign_z = 0; self.bucket_z = 0; self.signed_z = 0; self.n_signed = 0
        # student vs teacher
        self.mse_t = 0.0; self.sign_t = 0; self.bucket_t = 0
        # teacher vs outcome
        self.t_mse_z = 0.0; self.t_sign_z = 0; self.t_bucket_z = 0; self.t_signed_z = 0

    @torch.no_grad()
    def update(self, v_s, z, v_t, loss):
        bs = z.numel()
        self.n += bs
        self.loss += float(loss) * bs
        self.mse_z += float(((v_s - z) ** 2).sum())
        self.sign_z += int((torch.sign(v_s) == torch.sign(z)).sum())
        self.bucket_z += int((bucket(v_s) == z).sum())
        mask = z != 0
        self.n_signed += int(mask.sum())
        self.signed_z += int((torch.sign(v_s[mask]) == z[mask]).sum())
        self.mse_t += float(((v_s - v_t) ** 2).sum())
        self.sign_t += int((torch.sign(v_s) == torch.sign(v_t)).sum())
        self.bucket_t += int((bucket(v_s) == bucket(v_t)).sum())
        self.t_mse_z += float(((v_t - z) ** 2).sum())
        self.t_sign_z += int((torch.sign(v_t) == torch.sign(z)).sum())
        self.t_bucket_z += int((bucket(v_t) == z).sum())
        self.t_signed_z += int((torch.sign(v_t[mask]) == z[mask]).sum())

    def report(self, tag):
        if self.n == 0:
            return (f"{tag}: no complete labelled chunk in this shard "
                    f"(shard smaller than the chunk size, or not labelled yet)")
        n = max(self.n, 1)
        ns = max(self.n_signed, 1)
        return (f"{tag} [vs outcome] loss: {self.loss / n:.4f} mse: {self.mse_z / n:.4f} "
                f"sign-acc: {self.sign_z / n:.4f} bucket-acc: {self.bucket_z / n:.4f} "
                f"signed-acc: {self.signed_z / ns:.4f}\n"
                f"{tag} [vs teacher] mse: {self.mse_t / n:.4f} sign-agree: {self.sign_t / n:.4f} "
                f"bucket-agree: {self.bucket_t / n:.4f}\n"
                f"{tag} [teacher   ] mse: {self.t_mse_z / n:.4f} sign-acc: {self.t_sign_z / n:.4f} "
                f"bucket-acc: {self.t_bucket_z / n:.4f} signed-acc: {self.t_signed_z / ns:.4f}")


def read_meta(cache_dir):
    with open(os.path.join(cache_dir, "meta.json")) as f:
        return json.load(f)
