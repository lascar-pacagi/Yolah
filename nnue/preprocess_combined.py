"""
preprocess_combined.py — Offline encoding for the combined network
(193 nnue board-input ⊕ 118 features → one row-aligned cache, model input 311).

The turn is present in BOTH source representations (nnue byte 192 as 0/1, and the
feature TURN at index 118), so it is redundant. We drop the FEATURE turn at WRITE
time → features.u8 is 118-wide and the cache row is exactly 193 + 118 = 311, i.e.
exactly what the model consumes. The alignment guard still reads the full 119
features straight from the `.features.txt` records (NOT the cache), so dropping
the column from the cache does not weaken it.

Reads:  $YOLAH_GAME_DIR/games_*  (defaults to /nnue/data inside the .sif)
Writes: <cache_dir>/inputs.u8    shape (N, 193) uint8 — board bitboards + turn
        <cache_dir>/features.u8  shape (N, 118) uint8 — features WITHOUT the
                                                        redundant TURN (index 118)
        <cache_dir>/values.i8    shape (N,)     int8  — z ∈ {-1, 0, +1} from the
                                                        CURRENT player's POV
        <cache_dir>/meta.json    metadata sidecar

Usage:  python3 preprocess_combined.py [<cache_dir>]

Env:
  YOLAH_GAME_DIR        — input games directory   (default /nnue/data)
  YOLAH_CACHE_DIR       — output dir              (default /cache, override via argv[1])
  YOLAH_ENCODER_BIN     — path to the C++ encoder (default /usr/local/bin/encode_features)
  YOLAH_PREPROC_NPROC   — replay worker count     (default min(cpu_count, 32))

──────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS — THE ROW-ALIGNMENT PROBLEM
──────────────────────────────────────────────────────────────────────────────
A combined net needs, in EACH training row, the 193-input AND the 119-features
for the SAME board position. The two existing caches (cache_nnue193,
cache_features119) were produced by INDEPENDENT pipelines that enumerate
positions in DIFFERENT orders (and even different position sets):

  • preprocess_nnue.py sorts glob("games*"), so "..._0.symmetries.txt" sorts
    BEFORE "..._0.txt".  The features path consolidates sorted("*.features.txt"),
    where "..._0.features.txt" sorts BEFORE "..._0.symmetries.features.txt".
    → within each original/symmetry pair the two caches order the files OPPOSITELY.
  • preprocess_nnue.py SKIPS fully-random games (nb_random == nb_moves) and emits
    a fixed nb_moves-nb_random+1 plies; the C++ generate_features has NO such
    filter and terminates on game_over() — so per-game counts can differ too.

So concatenating row i of one cache with row i of the other glues board A's
bitboards to board B's features. This script instead builds ONE aligned cache.

──────────────────────────────────────────────────────────────────────────────
HOW ALIGNMENT IS GUARANTEED
──────────────────────────────────────────────────────────────────────────────
The C++ encoder remains the single source of truth for WHICH positions exist:

  Phase A — invoke the baked-in `encode_features` C++ binary to produce, per
            source game file, a `.features.txt` of (119 feature bytes + 1 outcome
            label byte) records. (Same step as preprocess_features.py.)
  Phase B — drive consolidation from the sorted *.features.txt files. For each
            file we replay the SAME source game in Python, mirroring the C++
            generate_features loop EXACTLY (skip-random with game_over guard,
            then emit-then-advance until game_over). We thereby emit the 193
            board-input for each position the C++ emitted features for, IN THE
            SAME ORDER.

Two per-row cross-checks turn "should be aligned" into "provably aligned" — any
single-position drift flips turn parity and trips them immediately:

  1. TURN check  : the C++ feature at index TURN_INDEX (== current_player(),
                   0=black) must equal the turn byte we compute in the replay.
  2. VALUE check : the current-player z we compute from the recorded scores must
                   equal the z derived independently from the C++ outcome label.

If either fails, the worker raises with the offending file + row, aborting the
build rather than writing silently-misaligned data.

Note the ordering: both cross-checks run on the FULL 119-feature records read
from `.features.txt`, BEFORE the write. Only after they pass do we write the 193
inputs and the 118 features (TURN column dropped) to the cache. So the redundant
TURN never reaches the cache, yet the guard still uses it at full strength.
"""
import os
import sys
import glob
import json
import time
import subprocess
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.append("/server")     # path inside the .sif (bind-mounted)
sys.path.append("../server")   # local dev path
from yolah import Yolah, Move, Square


# ── Feature layout (MUST match YolahFeatures in player/yolah_features.h) ────
NB_FEATURES        = 119             # features in each .features.txt record
RECORD_SIZE        = NB_FEATURES + 1  # last byte is the 3-class outcome label
TURN_INDEX         = NB_FEATURES - 1  # 118 — TURN is the final feature, 0 = black
# The redundant TURN (nnue byte 192 already carries it) is dropped at write time:
# the cache stores only the first 118 features, so cache row = 193 + 118 = 311.
NB_FEATURES_STORED = NB_FEATURES - 1  # 118 — features written to the cache

# ── NNUE board-input layout (MUST match preprocess_nnue.py / the trainer) ──
INPUT_SIZE = 64 + 64 + 64 + 1        # 193 — black | white | empty | turn


GAME_DIR    = os.environ.get("YOLAH_GAME_DIR",    "/nnue/data")
DEFAULT_OUT = os.environ.get("YOLAH_CACHE_DIR",   "/cache")
ENCODER_BIN = os.environ.get("YOLAH_ENCODER_BIN", "/usr/local/bin/encode_features")
NUM_PROCS   = int(os.environ.get("YOLAH_PREPROC_NPROC", str(min(cpu_count(), 32))))


# ── Vectorized bitboard → 64 binary bits (≈50× faster than a Python loop) ──
def bb_to_bits(n: int) -> np.ndarray:
    """64-bit int → (64,) uint8 binary array, MSB-first to match the codebase."""
    return np.unpackbits(np.array([n], dtype='>u8').view(np.uint8))


# ── Phase A: run the C++ encoder (skipped if .features.txt already present) ──
def run_encoder(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isfile(ENCODER_BIN) or not os.access(ENCODER_BIN, os.X_OK):
        raise RuntimeError(
            f"Encoder binary missing or not executable: {ENCODER_BIN}\n"
            f"  Set YOLAH_ENCODER_BIN, or rebuild the SIF "
            f"(combined_net312x512x64x32x1.def builds it).")
    cmd = [ENCODER_BIN, src_dir, dst_dir]
    print(f"Running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"Encoder finished in {time.time() - t0:.1f}s", flush=True)


# ── Map a .features.txt back to its source game file ───────────────────────
def source_game_path(features_txt_name: str) -> str:
    """
    games_X.features.txt            -> games_X.txt
    games_X.symmetries.features.txt -> games_X.symmetries.txt
    (The C++ encoder names outputs via replace_extension("features.txt"), which
    only replaces the final ".txt" — so stripping ".features.txt" and re-adding
    ".txt" inverts it exactly.)
    """
    base = features_txt_name[: -len(".features.txt")]
    return os.path.join(GAME_DIR, base + ".txt")


# ── Replay one source game file, mirroring C++ generate_features EXACTLY ────
def replay_game_file(path: str):
    """
    Returns (inputs, values, turns) for every position the C++ encoder would
    have emitted for this file, in the same order:
        inputs : (M, 193) uint8   board bitboards + turn byte
        values : (M,)     int8     current-player z ∈ {-1, 0, +1}
        turns  : (M,)     uint8    0 = black to move, 1 = white to move
    """
    with open(path, "rb") as f:
        data = f.read()

    inputs_list, values_list, turns_list = [], [], []
    idx = 0
    while idx < len(data):
        nb_moves  = data[idx]
        nb_random = data[idx + 1]
        moves     = data[idx + 2 : idx + 2 + 2 * nb_moves]
        bs        = data[idx + 2 + 2 * nb_moves]
        ws        = data[idx + 2 + 2 * nb_moves + 1]
        # Outcome from BLACK's perspective (matches the C++ label inversion):
        #   black wins -> +1, draw -> 0, white wins -> -1.
        z_black = 1 if bs > ws else (-1 if ws > bs else 0)

        y = Yolah()
        i = 0
        # Skip the random opening — check game_over BEFORE each play (mirror C++).
        while i < nb_random and not y.game_over():
            y.play(Move(Square(moves[2 * i]), Square(moves[2 * i + 1])))
            i += 1

        # Emit-then-advance until game_over (mirror C++ `while (true) { ... }`).
        while True:
            black_to_move = (y.nb_plies() & 1) == 0
            row = np.empty(INPUT_SIZE, dtype=np.uint8)
            row[0:64]    = bb_to_bits(y.black)
            row[64:128]  = bb_to_bits(y.white)
            row[128:192] = bb_to_bits(y.empty)
            row[192]     = 0 if black_to_move else 1
            inputs_list.append(row)
            values_list.append(z_black if black_to_move else -z_black)
            turns_list.append(0 if black_to_move else 1)

            if y.game_over():
                break
            if i >= nb_moves:
                # Ran out of recorded moves but the position is not game_over.
                # The C++ would read past the filled moves here; we stop instead
                # and let the per-file count assertion in Phase B flag the file.
                break
            y.play(Move(Square(moves[2 * i]), Square(moves[2 * i + 1])))
            i += 1

        idx += 2 + 2 * nb_moves + 2

    inputs = (np.stack(inputs_list) if inputs_list
              else np.empty((0, INPUT_SIZE), dtype=np.uint8))
    values = np.array(values_list, dtype=np.int8)
    turns  = np.array(turns_list,  dtype=np.uint8)
    return inputs, values, turns


# ── Phase B worker: replay + cross-check + write one file's block ──────────
def encode_one(args):
    (feat_path, src_path, start, total,
     inputs_path, features_path, values_path) = args

    # Authoritative records produced by the C++ encoder for this file.
    src = np.fromfile(feat_path, dtype=np.uint8)
    if src.size % RECORD_SIZE != 0:
        raise ValueError(f"{feat_path}: {src.size} bytes, not a multiple of "
                         f"{RECORD_SIZE}")
    src = src.reshape(-1, RECORD_SIZE)
    n_feat = src.shape[0]
    feat_block = src[:, :NB_FEATURES]
    label      = src[:, NB_FEATURES]

    # Replay the SAME source game file in Python to get the 193-board inputs.
    inputs_block, values_py, turns_py = replay_game_file(src_path)

    # ── Cross-check 1: identical position COUNT ────────────────────────────
    if inputs_block.shape[0] != n_feat:
        raise RuntimeError(
            f"Count mismatch for {os.path.basename(src_path)}: "
            f"replay emitted {inputs_block.shape[0]} positions but the C++ "
            f"features file has {n_feat}. The two encoders disagree on this "
            f"file's positions — alignment cannot be guaranteed.")

    # ── Cross-check 2: per-row TURN agreement (C++ TURN feature == replay) ─
    feat_turn = feat_block[:, TURN_INDEX]
    if not np.array_equal(feat_turn, turns_py):
        bad = int(np.argmax(feat_turn != turns_py))
        raise RuntimeError(
            f"TURN mismatch for {os.path.basename(src_path)} at row {bad}: "
            f"features TURN={int(feat_turn[bad])} replay turn={int(turns_py[bad])}. "
            f"Rows are misaligned.")

    # ── Cross-check 3: per-row VALUE agreement (label-derived == replay) ───
    z_black_lbl = np.where(label == 0, 1, np.where(label == 2, -1, 0)).astype(np.int8)
    black_to_move = (turns_py == 0)
    value_from_label = np.where(black_to_move, z_black_lbl, -z_black_lbl).astype(np.int8)
    if not np.array_equal(value_from_label, values_py):
        bad = int(np.argmax(value_from_label != values_py))
        raise RuntimeError(
            f"VALUE mismatch for {os.path.basename(src_path)} at row {bad}: "
            f"label-derived={int(value_from_label[bad])} "
            f"replay={int(values_py[bad])}. Rows are misaligned.")

    # ── All checks passed → block-write into the shared memmaps ────────────
    # Each worker mmaps the same files; writes go to disjoint [start, start+n)
    # regions (same pattern as preprocess_nnue.py), so no locking is needed.
    # The redundant TURN is dropped here: feat_block[:, :118] omits the final
    # feature (index 118 == TURN), which the nnue turn byte already carries.
    inputs = np.memmap(inputs_path, dtype=np.uint8,
                       mode='r+', shape=(total, INPUT_SIZE))
    features = np.memmap(features_path, dtype=np.uint8,
                         mode='r+', shape=(total, NB_FEATURES_STORED))
    values = np.memmap(values_path, dtype=np.int8,
                       mode='r+', shape=(total,))

    inputs[start:start + n_feat]   = inputs_block
    features[start:start + n_feat] = feat_block[:, :NB_FEATURES_STORED]
    values[start:start + n_feat]   = values_py

    inputs.flush()
    features.flush()
    values.flush()
    return feat_path, n_feat


# Module-level so the Pool workers inherit it (set in main()).
OUT_DIR = DEFAULT_OUT


def main():
    global OUT_DIR
    OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Source       : {GAME_DIR}", flush=True)
    print(f"Output dir   : {OUT_DIR}", flush=True)
    print(f"Encoder      : {ENCODER_BIN}", flush=True)
    print(f"Worker procs : {NUM_PROCS}", flush=True)
    print(f"Layout       : inputs {INPUT_SIZE}  features-stored "
          f"{NB_FEATURES_STORED} (of {NB_FEATURES}, TURN dropped)  "
          f"cache_row = model_input = {INPUT_SIZE + NB_FEATURES_STORED}",
          flush=True)

    feature_txt_dir = os.path.join(OUT_DIR, "feature_txt")

    # ── Phase A: run the C++ encoder if the intermediate dir is empty ──────
    existing = glob.glob(os.path.join(feature_txt_dir, "*.features.txt"))
    if existing:
        print(f"\n[A/B] {len(existing)} .features.txt files already present in "
              f"{feature_txt_dir}, skipping encoder.", flush=True)
    else:
        print(f"\n[A/B] Encoding games to .features.txt ...", flush=True)
        run_encoder(GAME_DIR, feature_txt_dir)

    # ── Phase B: replay + consolidate into the aligned cache ───────────────
    feat_files = sorted(glob.glob(os.path.join(feature_txt_dir, "*.features.txt")))
    if not feat_files:
        print(f"ERROR: no *.features.txt in {feature_txt_dir}", file=sys.stderr)
        sys.exit(1)

    # Pass 1 — per-file record counts from file sizes (cheap, no read).
    print(f"\n[B/B] Counting records and resolving source files ...", flush=True)
    counts, src_paths = [], []
    for fp in feat_files:
        sz = os.path.getsize(fp)
        if sz % RECORD_SIZE != 0:
            raise ValueError(f"{fp}: {sz} bytes, not a multiple of {RECORD_SIZE}")
        counts.append(sz // RECORD_SIZE)
        sp = source_game_path(os.path.basename(fp))
        if not os.path.isfile(sp):
            raise FileNotFoundError(
                f"Source game file missing for {os.path.basename(fp)}: {sp}")
        src_paths.append(sp)

    total = sum(counts)
    starts = [0]
    for c in counts[:-1]:
        starts.append(starts[-1] + c)
    print(f"      {len(feat_files):,} files, {total:,} positions", flush=True)

    # ── Preallocate sparse memmaps (OS fills as written) ───────────────────
    inputs_path   = os.path.join(OUT_DIR, "inputs.u8")
    features_path = os.path.join(OUT_DIR, "features.u8")
    values_path   = os.path.join(OUT_DIR, "values.i8")

    print(f"\nMemmap layout:")
    print(f"  inputs   : {total * INPUT_SIZE / 1e9:7.2f} GB  -> {inputs_path}")
    print(f"  features : {total * NB_FEATURES_STORED / 1e9:7.2f} GB  -> {features_path}")
    print(f"  values   : {total / 1e6:7.2f} MB  -> {values_path}", flush=True)

    for path, sz in [(inputs_path,   total * INPUT_SIZE),
                     (features_path, total * NB_FEATURES_STORED),
                     (values_path,   total)]:
        with open(path, "wb") as f:
            f.truncate(sz)

    # ── Pass 2 — replay + cross-check + write, in parallel over files ──────
    print(f"\n[B/B] Replaying + consolidating "
          f"(per-row TURN/VALUE alignment checks) ...", flush=True)
    t0 = time.time()
    tasks = [(fp, sp, start, total, inputs_path, features_path, values_path)
             for fp, sp, start in zip(feat_files, src_paths, starts)]

    completed = 0
    with Pool(NUM_PROCS) as p:
        for _fp, _n in p.imap_unordered(encode_one, tasks):
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                rate = completed / max(time.time() - t0, 1e-6)
                eta  = (len(tasks) - completed) / max(rate, 1e-6)
                print(f"      {completed:5d}/{len(tasks):d} files  "
                      f"({rate:5.1f} files/s, ETA {eta:6.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {total:,} positions in {elapsed:.1f}s  "
          f"({total / max(elapsed, 1e-6):,.0f} positions/s)", flush=True)

    # Distribution sanity check.
    values = np.memmap(values_path, dtype=np.int8, mode='r', shape=(total,))
    n_win  = int((values == +1).sum())
    n_draw = int((values ==  0).sum())
    n_lose = int((values == -1).sum())
    print(f"Value distribution (current-player POV): "
          f"win {n_win:,}  draw {n_draw:,}  lose {n_lose:,}", flush=True)

    # ── Metadata sidecar ───────────────────────────────────────────────────
    meta = {
        "n_positions": total,
        "inputs":   {"path": "inputs.u8",   "dtype": "uint8",
                     "shape": [total, INPUT_SIZE]},
        "features": {"path": "features.u8", "dtype": "uint8",
                     "shape": [total, NB_FEATURES_STORED]},
        "values":   {"path": "values.i8",   "dtype": "int8",
                     "shape": [total]},
        "input_size_nnue": INPUT_SIZE,                       # 193
        # Features stored in the cache: the redundant TURN (the 119th feature)
        # is dropped at write time, so the cache row IS the model input.
        "n_features":      NB_FEATURES_STORED,              # 118 (TURN dropped)
        "n_features_source": NB_FEATURES,                   # 119 in the .features.txt
        "model_input_size": INPUT_SIZE + NB_FEATURES_STORED,  # 311
        "feature_turn_dropped": True,
        "value_perspective": "current_player",
        "value_range": [-1, 0, 1],
    }
    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
