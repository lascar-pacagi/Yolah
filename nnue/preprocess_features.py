"""
preprocess_features.py — Offline encoding for the 119-input features network.

Reads:  $YOLAH_GAME_DIR/games_*  (defaults to /nnue/data inside the .sif)
Writes: <cache_dir>/features.u8  shape (N, 119) uint8 — raw feature bytes
        <cache_dir>/values.i8    shape (N,)     int8  — z ∈ {-1, 0, +1} from
                                                        CURRENT player's POV
        <cache_dir>/meta.json    metadata sidecar

Usage:  python3 preprocess_features.py [<cache_dir>]

Env:
  YOLAH_GAME_DIR     — input games directory   (default /nnue/data)
  YOLAH_CACHE_DIR    — output cache dir        (default /cache, override via argv[1])
  YOLAH_ENCODER_BIN  — path to the C++ encoder (default /usr/local/bin/encode_features)

Why a two-phase pipeline:
  Phase A — invoke the C++ `encode_features` binary on the games directory to
            produce per-source-file `.features.txt` records (NB_FEATURES + 1
            bytes per position). The feature math lives in C++ for speed and
            because the Python yolah module does NOT expose set_features.
  Phase B — consolidate every `.features.txt` into a single flat memmap pair
            (features.u8 + values.i8) and write meta.json. This matches the
            layout the CNN / NNUE preprocess scripts produce so the chunked
            loader can sequentially read large contiguous spans.

The intermediate `.features.txt` files are kept under `<cache_dir>/feature_txt/`
in case the user wants to inspect them; pass --cleanup to delete them after
consolidation.
"""
import os
import sys
import glob
import json
import time
import shutil
import subprocess
import numpy as np


# ── Feature layout (MUST match YolahFeatures in player/yolah_features.h) ───
NB_FEATURES = 119
RECORD_SIZE = NB_FEATURES + 1                # last byte is the 3-class outcome label
TURN_INDEX  = NB_FEATURES - 1                # TURN is the final feature


# ── Env / arg setup ────────────────────────────────────────────────────────
GAME_DIR    = os.environ.get("YOLAH_GAME_DIR",    "/nnue/data")
DEFAULT_OUT = os.environ.get("YOLAH_CACHE_DIR",   "/cache")
ENCODER_BIN = os.environ.get("YOLAH_ENCODER_BIN", "/usr/local/bin/encode_features")


def run_encoder(src_dir: str, dst_dir: str) -> None:
    """Invoke the C++ encoder on every games_* file under src_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isfile(ENCODER_BIN) or not os.access(ENCODER_BIN, os.X_OK):
        raise RuntimeError(
            f"Encoder binary missing or not executable: {ENCODER_BIN}\n"
            f"  Set YOLAH_ENCODER_BIN, or rebuild the SIF "
            f"(features_net119x256x64x1.def builds it).")
    cmd = [ENCODER_BIN, src_dir, dst_dir]
    print(f"Running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    # check=True surfaces a non-zero exit immediately. stdout/stderr inherit
    # so the encoder's "Processing: ..." progress flows to our log.
    subprocess.run(cmd, check=True)
    print(f"Encoder finished in {time.time() - t0:.1f}s", flush=True)


def consolidate(src_txt_dir: str, out_dir: str) -> dict:
    """
    Concatenate every <src_txt_dir>/*.features.txt into a flat (features.u8,
    values.i8) pair under out_dir, transforming the 3-class outcome label into
    a current-player z ∈ {-1, 0, +1} once at preprocess time so the trainer
    skips that per-batch math.
    """
    files = sorted(glob.glob(os.path.join(src_txt_dir, "*.features.txt")))
    if not files:
        raise FileNotFoundError(
            f"No *.features.txt files in {src_txt_dir} — encoder produced nothing?")

    # ── Pass 1: count records (cheap, just file sizes) ─────────────────────
    sizes = []
    for path in files:
        sz = os.path.getsize(path)
        if sz % RECORD_SIZE != 0:
            raise ValueError(
                f"{path} is {sz} bytes, not a multiple of {RECORD_SIZE}")
        sizes.append(sz // RECORD_SIZE)
    total = sum(sizes)
    print(f"Consolidating: {len(files)} files, {total:,} positions", flush=True)

    # ── Preallocate the destination memmaps ────────────────────────────────
    features_path = os.path.join(out_dir, "features.u8")
    values_path   = os.path.join(out_dir, "values.i8")

    # truncate() creates a sparse file; the OS fills as we write into it.
    with open(features_path, "wb") as f:
        f.truncate(total * NB_FEATURES)
    with open(values_path, "wb") as f:
        f.truncate(total)

    features = np.memmap(features_path, dtype=np.uint8, mode='r+',
                         shape=(total, NB_FEATURES))
    values   = np.memmap(values_path, dtype=np.int8, mode='r+',
                         shape=(total,))

    # ── Pass 2: copy + outcome→z_current transform ─────────────────────────
    # The .features.txt record is [feature_bytes (119) | label_byte (1)].
    # Vectorized per-file: read whole file, reshape, slice.
    cursor = 0
    t0 = time.time()
    for path, n_records in zip(files, sizes):
        # mmap source read-only; this is just a virtual address — actual reads
        # happen lazily as we touch bytes.
        src = np.memmap(path, dtype=np.uint8, mode='r',
                        shape=(n_records, RECORD_SIZE))

        # Block-copy the feature bytes. .astype is a no-op here (same dtype).
        features[cursor:cursor + n_records] = src[:, :NB_FEATURES]

        # 3-class label → z_black ∈ {+1, 0, -1}
        # label 0 = black wins, 1 = draw, 2 = white wins.
        label   = src[:, NB_FEATURES].astype(np.int16)
        z_black = np.zeros(n_records, dtype=np.int8)
        z_black[label == 0] = +1
        z_black[label == 2] = -1

        # z_current = z_black if it's black's turn, else -z_black.
        # TURN byte: 0 = black to move, 1 = white to move (uint8).
        black_to_move = src[:, TURN_INDEX] == 0
        z_current     = np.where(black_to_move, z_black, -z_black).astype(np.int8)
        values[cursor:cursor + n_records] = z_current

        cursor += n_records
        # Free the source memmap so the OS can reclaim its pages.
        del src

    features.flush()
    values.flush()
    elapsed = time.time() - t0
    print(f"Consolidated {total:,} positions in {elapsed:.1f}s  "
          f"({total / max(elapsed, 1e-6):,.0f} pos/s)", flush=True)

    # Distribution sanity check (cheap on the int8 values array).
    n_win  = int((values == +1).sum())
    n_draw = int((values ==  0).sum())
    n_lose = int((values == -1).sum())
    print(f"Value distribution (current-player POV): "
          f"win {n_win:,}  draw {n_draw:,}  lose {n_lose:,}", flush=True)

    return {
        "n_positions": total,
        "features": {"path": "features.u8",
                     "dtype": "uint8",
                     "shape": [total, NB_FEATURES]},
        "values":   {"path": "values.i8",
                     "dtype": "int8",
                     "shape": [total]},
        "input_size": NB_FEATURES,
        "value_perspective": "current_player",
        "value_range": [-1, 0, 1],
    }


def main():
    args = [a for a in sys.argv[1:] if a != "--cleanup"]
    cleanup = "--cleanup" in sys.argv[1:]
    out_dir = args[0] if args else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    print(f"Source       : {GAME_DIR}",   flush=True)
    print(f"Output dir   : {out_dir}",    flush=True)
    print(f"Encoder      : {ENCODER_BIN}", flush=True)

    # The intermediate per-source .features.txt files. Kept under the cache
    # dir so a re-run after a crash can be resumed without re-encoding.
    feature_txt_dir = os.path.join(out_dir, "feature_txt")

    # ── Phase A: run the C++ encoder if the intermediate dir is empty ──────
    existing = glob.glob(os.path.join(feature_txt_dir, "*.features.txt"))
    if existing:
        print(f"[A/B] {len(existing)} .features.txt files already present "
              f"in {feature_txt_dir}, skipping encoder.", flush=True)
    else:
        print(f"\n[A/B] Encoding games to .features.txt ...", flush=True)
        run_encoder(GAME_DIR, feature_txt_dir)

    # ── Phase B: consolidate into the flat (features.u8, values.i8) cache ─
    print(f"\n[B/B] Consolidating .features.txt → flat memmap ...", flush=True)
    meta = consolidate(feature_txt_dir, out_dir)

    # ── Metadata sidecar ───────────────────────────────────────────────────
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote metadata: {meta_path}", flush=True)

    # Optional cleanup — pass --cleanup to delete the intermediate dir once
    # the flat cache is in place (saves ~2× the cache footprint).
    if cleanup:
        print(f"Removing intermediate dir: {feature_txt_dir}", flush=True)
        shutil.rmtree(feature_txt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
