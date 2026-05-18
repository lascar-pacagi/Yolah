"""
preprocess.py — Offline encoding of all game positions to memory-mapped files.

Reads:  $YOLAH_GAME_DIR/games_*  (defaults to /nnue/data inside the .sif)
Writes: <cache_dir>/positions.u8   shape (N, 4, 8, 8) uint8
        <cache_dir>/values.i8       shape (N,)        int8
        <cache_dir>/policies.i16    shape (N,)        int16
        <cache_dir>/meta.json       metadata sidecar

Usage:  python3 preprocess.py [<cache_dir>]

Env:
  YOLAH_GAME_DIR        — input games directory (default: /nnue/data)
  YOLAH_CACHE_DIR       — output dir (default: /cache, overridable by argv[1])
  YOLAH_PREPROC_NPROC   — encoder worker count (default: min(cpu_count, 32))

Why this exists
───────────────
The live `GameDataset` replays each Yolah game from scratch every time a
position is sampled. For a game of M plies that's O(M²) Python work per
epoch — at 1B positions and num_workers=0 the GPU starves. This script
encodes every position *once*, sequentially within each game (O(M)), and
stores the result as memory-mapped binary files. Training then becomes
a memcpy + cast in `__getitem__`, with multi-worker DataLoaders sharing
the OS page cache (no RAM multiplication per worker).

Layout
──────
Stored as uint8 because every plane is binary or a small categorical:
  plane 0 — black pieces  (0/1)
  plane 1 — white pieces  (0/1)
  plane 2 — empty squares (0/1)
  plane 3 — turn          (0 = black to play, 1 = white to play)
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


# ── Vectorized bitboard → 8×8 binary plane (≈50× faster than Python loop) ──
def bb_to_plane(n: int) -> np.ndarray:
    """64-bit int → (8, 8) uint8 binary plane, MSB-first to match the codebase."""
    return np.unpackbits(np.array([n], dtype='>u8').view(np.uint8)).reshape(8, 8)


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
        if nb_random != nb_moves:           # all-random games are skipped
            n_games += 1
            n_positions += (nb_moves - nb_random + 1)
        idx += 2 + 2 * nb_moves + 2
    return path, n_games, n_positions


# ── Pass 2: encode positions for one file into the global memmap ───────────
def encode_file(args):
    (path, start_offset, total,
     pos_path, val_path, pol_path) = args

    # Each worker mmaps the same files; writes go to disjoint regions so
    # there is no inter-process write contention.
    pos = np.memmap(pos_path, dtype=np.uint8, mode='r+',
                    shape=(total, 4, 8, 8))
    val = np.memmap(val_path, dtype=np.int8, mode='r+',
                    shape=(total,))
    pol = np.memmap(pol_path, dtype=np.int16, mode='r+',
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
        # outcome from BLACK's perspective
        z_black = 1 if bs > ws else (-1 if ws > bs else 0)

        # Sequential replay — fast-forward the random opening
        y = Yolah()
        for ply in range(nb_random):
            s1, s2 = moves[2 * ply], moves[2 * ply + 1]
            y.play(Move(Square(s1), Square(s2)))

        # Encode every position from ply r to ply nb_moves
        for ply in range(nb_random, nb_moves + 1):
            black_to_move = (y.nb_plies() & 1) == 0

            # Write planes directly into the memmap (no temp np.stack)
            pos[cursor, 0] = bb_to_plane(y.black)
            pos[cursor, 1] = bb_to_plane(y.white)
            pos[cursor, 2] = bb_to_plane(y.empty)
            pos[cursor, 3] = 0 if black_to_move else 1

            # Value target from CURRENT player's perspective
            val[cursor] = z_black if black_to_move else -z_black

            # Policy target = action index of the next move played
            if ply < nb_moves:
                s1, s2 = moves[2 * ply], moves[2 * ply + 1]
                pol[cursor] = s1 * 64 + s2
                y.play(Move(Square(s1), Square(s2)))
            else:
                pol[cursor] = -1            # terminal — masked out in CE loss

            cursor += 1

        idx += 2 + 2 * nb_moves + 2

    pos.flush(); val.flush(); pol.flush()
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

    # ── Pass 1: count ──────────────────────────────────────────────────────
    print(f"\n[1/2] Counting positions ...", flush=True)
    t0 = time.time()
    with Pool(NUM_PROCS) as p:
        per_file = p.map(count_positions, files)
    total_pos   = sum(c[2] for c in per_file)
    total_games = sum(c[1] for c in per_file)
    print(f"      {total_pos:,} positions across {total_games:,} games  "
          f"({time.time() - t0:.1f}s)", flush=True)

    # Compute offsets per file
    starts = [0]
    for _, _, n in per_file[:-1]:
        starts.append(starts[-1] + n)

    # ── Preallocate sparse memmap files (OS fills as written) ─────────────
    pos_path = os.path.join(out_dir, "positions.u8")
    val_path = os.path.join(out_dir, "values.i8")
    pol_path = os.path.join(out_dir, "policies.i16")

    n_bytes_pos = total_pos * 4 * 8 * 8
    n_bytes_val = total_pos
    n_bytes_pol = total_pos * 2

    print(f"\nMemmap layout:")
    print(f"  positions : {n_bytes_pos / 1e9:7.2f} GB  -> {pos_path}")
    print(f"  values    : {n_bytes_val / 1e6:7.2f} MB  -> {val_path}")
    print(f"  policies  : {n_bytes_pol / 1e9:7.2f} GB  -> {pol_path}", flush=True)

    for path, sz in [(pos_path, n_bytes_pos),
                     (val_path, n_bytes_val),
                     (pol_path, n_bytes_pol)]:
        with open(path, "wb") as f:
            f.truncate(sz)

    # ── Pass 2: encode in parallel ─────────────────────────────────────────
    print(f"\n[2/2] Encoding positions ...", flush=True)
    t0 = time.time()
    args = [(path, start, total_pos, pos_path, val_path, pol_path)
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
        "positions": {"path": "positions.u8",
                      "dtype": "uint8",
                      "shape": [total_pos, 4, 8, 8]},
        "values":    {"path": "values.i8",
                      "dtype": "int8",
                      "shape": [total_pos]},
        "policies":  {"path": "policies.i16",
                      "dtype": "int16",
                      "shape": [total_pos]},
        "policy_ignore_idx": -1,
        "n_actions": 64 * 64,
        "encoding_planes": ["black", "white", "empty", "turn"],
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
