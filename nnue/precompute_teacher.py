"""
precompute_teacher.py — label a student cache with the ResNet teacher's value,
for knowledge distillation.

It runs the trained AlphaZero-style ResNet (cnn_resnet_value_policy_chunked.py's
Net, 256×30, ~36M params) over every position in a student cache and writes the
teacher's scalar value, row-aligned, as:

    <cache_dir>/teacher_value.f16   shape (N,) float16   v_teacher ∈ (-1, +1)

The nnue / features trainers auto-detect this file and switch their value loss
to a blend:

    loss = alpha · MSE(v_student, z) + (1 - alpha) · MSE(v_student, v_teacher)

The teacher value is already in the SAME current-player POV as values.i8 and its
head is a tanh scalar, so distillation is a direct MSE — no temperature, no
perspective flip.

TWO MODES — because the teacher needs the BOARD, not the hand-crafted features
──────────────────────────────────────────────────────────────────────────────
The teacher consumes a (4, 8, 8) plane stack (black|white|empty|turn), built
MSB-first — the exact byte order of the nnue inputs.u8. So:

  (default) inputs.u8 mode  — for caches that store the 193-wide board
      (cache_nnue193, cache_combined311). The teacher planes are a pure reshape
      of inputs.u8; perfectly aligned, no replay.

  --from-games mode         — for the FEATURES cache, which stores only the 119
      hand-crafted features and NO board. We cannot evaluate the teacher from
      lossy features, so we replay the games to reconstruct each board IN THE
      SAME ORDER the features cache was built (sorted by .features.txt name,
      mirroring preprocess_features.py), evaluate the teacher, and write labels
      aligned to that cache. A per-file cross-check (replayed current-player z ==
      the cache's values.i8) PROVES the alignment before anything is trusted.

Usage
─────
    # board-bearing cache (nnue / combined):
    python3 precompute_teacher.py <cache_dir> --model <ckpt.pt>

    # features-only cache (replays games to rebuild the board):
    python3 precompute_teacher.py <cache_dir> --from-games --model <ckpt.pt>

Env:
  YOLAH_TEACHER_MODEL — default checkpoint path (overridden by --model)
  YOLAH_GAME_DIR      — games dir for --from-games (default /nnue/data)
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("/server")     # path inside the .sif
sys.path.append("../server")   # local dev path
# Reuse the teacher's EXACT architecture so it can never drift from training.
from cnn_resnet_value_policy_chunked import Net
# Reuse the EXACT replay used to build the combined cache (mirrors the C++
# generate_features loop), so --from-games reproduces the features cache order.
from preprocess_combined import replay_game_file

INPUT_SIZE_NNUE = 64 + 64 + 64 + 1            # 193 — black | white | empty | turn
GAME_DIR = os.environ.get("YOLAH_GAME_DIR", "/nnue/data")


def inputs_to_planes(inp_u8: np.ndarray) -> np.ndarray:
    """
    (C, 193) uint8 nnue inputs → (C, 4, 8, 8) float32 teacher planes.

    Bytes 0:192 are three 64-bit MSB-first bitboards (black, white, empty) →
    three 8×8 planes; byte 192 is the turn → a constant 4th plane. Matches the
    teacher's training encoding exactly (both use np.unpackbits, big-endian).
    """
    c = inp_u8.shape[0]
    x = np.empty((c, 4, 8, 8), dtype=np.float32)
    x[:, 0:3] = inp_u8[:, :192].reshape(c, 3, 8, 8).astype(np.float32)
    x[:, 3]   = inp_u8[:, 192].astype(np.float32)[:, None, None]   # broadcast 8×8
    return x


def load_teacher(ckpt: str, channels: int, blocks: int, value_fc: int,
                 device: torch.device) -> Net:
    """Instantiate the teacher and load its checkpoint (prefix-robust)."""
    net = Net(channels=channels, nb_blocks=blocks, value_fc_size=value_fc)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    # Strip any DDP/torch.compile prefixes that may have been baked into keys.
    sd = {k.replace('_orig_mod.', '').replace('module.', ''): v
          for k, v in sd.items()}
    net.load_state_dict(sd)            # strict: fail loudly on any mismatch
    net.eval().to(device)
    return net.to(memory_format=torch.channels_last)


@torch.no_grad()
def teacher_values(net, inp_u8: np.ndarray, device, amp_dtype, batch: int,
                   pbar=None) -> np.ndarray:
    """(M, 193) uint8 → (M,) float16 teacher value, GPU-batched."""
    m = inp_u8.shape[0]
    out = np.empty(m, dtype=np.float16)
    for i in range(0, m, batch):
        planes = inputs_to_planes(inp_u8[i:i + batch])           # (b, 4, 8, 8)
        xb = torch.from_numpy(planes).to(
            device, memory_format=torch.channels_last, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            v, _policy = net(xb)                                  # value head only
        out[i:i + xb.shape[0]] = v.float().cpu().numpy().astype(np.float16)
        if pbar is not None:
            pbar.update(xb.shape[0])
    return out


def _features_cache_order(game_dir: str):
    """
    Source game files in the SAME order preprocess_features.py consolidated them
    — i.e. sorted by their .features.txt name. The C++ encoder selects files
    matching ^games((?!.*features.*)) (start with 'games', no 'features' in the
    name), so we mirror that selection here.
    """
    names = [f for f in os.listdir(game_dir)
             if f.startswith("games") and "features" not in f]
    # .features.txt name = replace the final extension with .features.txt
    return sorted(names, key=lambda nm: os.path.splitext(nm)[0] + ".features.txt")


def run_from_inputs(cache_dir, out_path, net, device, amp_dtype, batch, chunk):
    """Board-bearing cache: teacher planes are a reshape of inputs.u8."""
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    n = int(meta["n_positions"])
    if meta["inputs"]["shape"][1] != INPUT_SIZE_NNUE:
        raise ValueError(f"inputs width {meta['inputs']['shape'][1]} != "
                         f"{INPUT_SIZE_NNUE}; this cache has no 193-wide board.")
    inputs = np.memmap(os.path.join(cache_dir, meta["inputs"]["path"]),
                       dtype=np.uint8, mode='r',
                       shape=tuple(meta["inputs"]["shape"]))
    with open(out_path, "wb") as f:
        f.truncate(n * 2)
    out = np.memmap(out_path, dtype=np.float16, mode='r+', shape=(n,))

    pbar = tqdm(total=n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        inp = np.array(inputs[start:end])                        # (C, 193) uint8
        out[start:end] = teacher_values(net, inp, device, amp_dtype, batch, pbar)
    pbar.close()
    out.flush()
    return n, meta


def run_from_games(cache_dir, out_path, net, device, amp_dtype, batch):
    """
    Features-only cache: replay the games to reconstruct the board per row, in
    the cache's own order, and PROVE alignment against values.i8 before writing.
    """
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    n = int(meta["n_positions"])
    values = np.memmap(os.path.join(cache_dir, meta["values"]["path"]),
                       dtype=np.int8, mode='r', shape=(n,))

    order = _features_cache_order(GAME_DIR)
    if not order:
        raise FileNotFoundError(f"No source games_* files in {GAME_DIR}")

    with open(out_path, "wb") as f:
        f.truncate(n * 2)
    out = np.memmap(out_path, dtype=np.float16, mode='r+', shape=(n,))

    cursor = 0
    pbar = tqdm(total=n)
    for nm in order:
        inp, vals_py, _turns = replay_game_file(os.path.join(GAME_DIR, nm))
        m = inp.shape[0]
        if cursor + m > n:
            raise RuntimeError(f"replay overran the cache at {nm} "
                               f"(cursor {cursor} + {m} > {n})")
        # Alignment guard: the replayed current-player z MUST match the cache's
        # stored values for these rows, or the orders disagree.
        if not np.array_equal(vals_py, values[cursor:cursor + m]):
            bad = int(np.argmax(vals_py != values[cursor:cursor + m]))
            raise RuntimeError(
                f"VALUE mismatch at {nm} row {bad} (global {cursor + bad}): "
                f"replay={int(vals_py[bad])} cache={int(values[cursor + bad])}. "
                f"The replay order does not match this features cache.")
        out[cursor:cursor + m] = teacher_values(net, inp, device, amp_dtype,
                                                 batch, pbar)
        cursor += m
    pbar.close()
    if cursor != n:
        raise RuntimeError(f"replayed {cursor} positions but the cache has {n}")
    out.flush()
    return n, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--model", default=os.environ.get(
        "YOLAH_TEACHER_MODEL", "/mnt/cnn_resnet_256x30_value_policy.pt"))
    ap.add_argument("--out", default="teacher_value.f16",
                    help="output filename (relative to cache_dir)")
    ap.add_argument("--from-games", action="store_true",
                    help="replay games to rebuild the board (features-only cache)")
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--blocks",   type=int, default=30)
    ap.add_argument("--value-fc", type=int, default=256, dest="value_fc")
    ap.add_argument("--batch",    type=int, default=4096,
                    help="positions per GPU forward")
    ap.add_argument("--chunk",    type=int, default=1 << 20,
                    help="positions per sequential disk read (inputs.u8 mode)")
    args = ap.parse_args()

    out_path = os.path.join(args.cache_dir, args.out)

    if not torch.cuda.is_available():
        print("WARNING: no CUDA — running the 36M teacher on CPU will be slow.",
              flush=True)
        device, amp_dtype = torch.device("cpu"), torch.float32
    else:
        device = torch.device("cuda:0")
        major, _ = torch.cuda.get_device_capability(0)
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16

    print(f"Cache       : {args.cache_dir}", flush=True)
    print(f"Mode        : {'replay games' if args.from_games else 'inputs.u8'}",
          flush=True)
    print(f"Teacher     : {args.model}  ({args.channels}×{args.blocks})", flush=True)
    print(f"Output      : {out_path}", flush=True)
    print(f"Device      : {device}  amp={amp_dtype}", flush=True)

    net = load_teacher(args.model, args.channels, args.blocks, args.value_fc, device)

    if args.from_games:
        n, meta = run_from_games(args.cache_dir, out_path, net, device,
                                 amp_dtype, args.batch)
    else:
        n, meta = run_from_inputs(args.cache_dir, out_path, net, device,
                                  amp_dtype, args.batch, args.chunk)

    vals = np.asarray(np.memmap(out_path, dtype=np.float16, mode='r',
                                shape=(n,))[:], dtype=np.float32)
    print(f"teacher value: mean {vals.mean():+.4f}  std {vals.std():.4f}  "
          f"min {vals.min():+.4f}  max {vals.max():+.4f}", flush=True)

    meta["teacher_value"] = {"path": args.out, "dtype": "float16", "shape": [n],
                             "model": os.path.basename(args.model),
                             "mode": "from_games" if args.from_games else "inputs"}
    with open(os.path.join(args.cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_path} and updated meta.json", flush=True)


if __name__ == "__main__":
    main()
