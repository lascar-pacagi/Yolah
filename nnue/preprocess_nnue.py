"""
preprocess_nnue.py — Offline encoding for the 193-input NNUE.

Reads:  $YOLAH_GAME_DIR/games_*  (defaults to /nnue/data inside the .sif)
Writes: <cache_dir>/inputs.u8   shape (N, 193) uint8
        <cache_dir>/values.i8   shape (N,)     int8  (z ∈ {-1, 0, +1} from
                                                       CURRENT player's POV)
        <cache_dir>/meta.json   metadata sidecar

Usage:  python3 preprocess_nnue.py [<cache_dir>]

Env:
  YOLAH_GAME_DIR        — input games directory  (default /nnue/data)
  YOLAH_CACHE_DIR       — output dir             (default /cache, override via argv[1])
  YOLAH_PREPROC_NPROC   — encoder worker count   (default min(cpu_count, 32))

Encoding layout (matches GameDataset.encode_yolah in nnue_multigpu5.py):
    bytes   0–63   : black pieces  (1 = present)
    bytes  64–127  : white pieces
    bytes 128–191  : empty (already scored) squares
    byte  192      : turn  (0 = black to play, 1 = white to play)

Why this exists — same reasoning as preprocess.py for the CNN: the live
GameDataset replays each game from initial state on every __getitem__,
which is O(M²) per game per epoch. Encoding once and memmap-loading at
training time turns the data path into a memcpy + cast.
"""
import os
import sys
import glob
import json
import time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.append("/server")     # path inside the .sif (bind-mounted)
sys.path.append("../server")   # local dev path
from yolah import Yolah, Move, Square


GAME_DIR    = os.environ.get("YOLAH_GAME_DIR", "/nnue/data")
DEFAULT_OUT = os.environ.get("YOLAH_CACHE_DIR", "/cache")
NUM_PROCS   = int(os.environ.get("YOLAH_PREPROC_NPROC",
                                 str(min(cpu_count(), 32))))

INPUT_SIZE = 64 + 64 + 64 + 1   # 193 — must match nnue193x1024x64x32x1.py


# ── Vectorized bitboard → 64 binary bits (≈50× faster than Python loop) ────
def bb_to_bits(n: int) -> np.ndarray:
    """64-bit int → (64,) uint8 binary array, MSB-first to match the codebase."""
    return np.unpackbits(np.array([n], dtype='>u8').view(np.uint8))


# ── Pass 1: count positions per file (cheap byte-stream walk) ──────────────
def count_positions(path: str):
    with open(path, "rb") as f:
        data = f.read()
    n_games = 0
    n_positions = 0
    idx = 0
    while idx < len(data):
        nb_moves  = data[idx]
        nb_random = data[idx + 1]
        if nb_random != nb_moves:
            n_games += 1
            n_positions += (nb_moves - nb_random + 1)
        idx += 2 + 2 * nb_moves + 2
    return path, n_games, n_positions


# ── Pass 2: encode positions for one file into the global memmap ───────────
def encode_file(args):
    path, start_offset, total, inputs_path, values_path = args

    # Each worker mmaps the same files; writes go to disjoint regions.
    inputs = np.memmap(inputs_path, dtype=np.uint8, mode='r+',
                       shape=(total, INPUT_SIZE))
    values = np.memmap(values_path, dtype=np.int8, mode='r+',
                       shape=(total,))

    with open(path, "rb") as f:
        data = f.read()

    cursor = start_offset
    idx = 0
    while idx < len(data):
        nb_moves  = data[idx]
        nb_random = data[idx + 1]
        if nb_random == nb_moves:
            idx += 2 + 2 * nb_moves + 2
            continue
        moves = data[idx + 2 : idx + 2 + 2 * nb_moves]
        bs    = data[idx + 2 + 2 * nb_moves]
        ws    = data[idx + 2 + 2 * nb_moves + 1]
        # Outcome from BLACK's perspective: +1 black wins, 0 draw, -1 white wins.
        z_black = 1 if bs > ws else (-1 if ws > bs else 0)

        # Sequential replay — fast-forward the random opening once
        y = Yolah()
        for ply in range(nb_random):
            s1, s2 = moves[2 * ply], moves[2 * ply + 1]
            y.play(Move(Square(s1), Square(s2)))

        # Encode every position from ply r to ply nb_moves
        for ply in range(nb_random, nb_moves + 1):
            black_to_move = (y.nb_plies() & 1) == 0
            inputs[cursor, 0:64]    = bb_to_bits(y.black)
            inputs[cursor, 64:128]  = bb_to_bits(y.white)
            inputs[cursor, 128:192] = bb_to_bits(y.empty)
            inputs[cursor, 192]     = 0 if black_to_move else 1
            # Value target from CURRENT player's perspective ∈ {-1, 0, +1}.
            values[cursor] = z_black if black_to_move else -z_black

            if ply < nb_moves:
                s1, s2 = moves[2 * ply], moves[2 * ply + 1]
                y.play(Move(Square(s1), Square(s2)))
            cursor += 1

        idx += 2 + 2 * nb_moves + 2

    inputs.flush()
    values.flush()
    return path, cursor - start_offset


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(GAME_DIR, "games*")))
    if not files:
        print(f"ERROR: no files matched {GAME_DIR}/games*", file=sys.stderr)
        sys.exit(1)

    print(f"Source       : {GAME_DIR}  ({len(files):,} files)", flush=True)
    print(f"Output dir   : {out_dir}", flush=True)
    print(f"Worker procs : {NUM_PROCS}", flush=True)
    print(f"Input size   : {INPUT_SIZE}", flush=True)

    # ── Pass 1: count ──────────────────────────────────────────────────────
    print(f"\n[1/2] Counting positions ...", flush=True)
    t0 = time.time()
    with Pool(NUM_PROCS) as p:
        per_file = p.map(count_positions, files)
    total_pos   = sum(c[2] for c in per_file)
    total_games = sum(c[1] for c in per_file)
    print(f"      {total_pos:,} positions across {total_games:,} games  "
          f"({time.time() - t0:.1f}s)", flush=True)

    # Per-file offsets
    starts = [0]
    for _, _, n in per_file[:-1]:
        starts.append(starts[-1] + n)

    # ── Preallocate sparse memmap files (OS fills as written) ─────────────
    inputs_path = os.path.join(out_dir, "inputs.u8")
    values_path = os.path.join(out_dir, "values.i8")

    n_bytes_inputs = total_pos * INPUT_SIZE
    n_bytes_values = total_pos

    print(f"\nMemmap layout:")
    print(f"  inputs : {n_bytes_inputs / 1e9:7.2f} GB  -> {inputs_path}")
    print(f"  values : {n_bytes_values / 1e6:7.2f} MB  -> {values_path}",
          flush=True)

    for path, sz in [(inputs_path, n_bytes_inputs),
                     (values_path, n_bytes_values)]:
        with open(path, "wb") as f:
            f.truncate(sz)

    # ── Pass 2: encode in parallel ─────────────────────────────────────────
    print(f"\n[2/2] Encoding positions ...", flush=True)
    t0 = time.time()
    args = [(path, start, total_pos, inputs_path, values_path)
            for (path, _, _), start in zip(per_file, starts)]

    completed = 0
    with Pool(NUM_PROCS) as p:
        for path, n_written in p.imap_unordered(encode_file, args):
            completed += 1
            if completed % 50 == 0 or completed == len(files):
                rate = completed / max(time.time() - t0, 1e-6)
                eta  = (len(files) - completed) / max(rate, 1e-6)
                print(f"      {completed:5d}/{len(files):d} files  "
                      f"({rate:5.1f} files/s, ETA {eta:6.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {total_pos:,} positions in {elapsed:.1f}s  "
          f"({total_pos / elapsed:,.0f} positions/s)", flush=True)

    # ── Metadata sidecar ───────────────────────────────────────────────────
    meta = {
        "n_positions": total_pos,
        "n_games":     total_games,
        "inputs": {"path": "inputs.u8",
                   "dtype": "uint8",
                   "shape": [total_pos, INPUT_SIZE]},
        "values": {"path": "values.i8",
                   "dtype": "int8",
                   "shape": [total_pos]},
        "input_size": INPUT_SIZE,
        "value_perspective": "current_player",
        "value_range": [-1, 0, 1],
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
