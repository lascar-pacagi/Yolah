"""
distill_boards_for_features.py — board sidecar for a features cache.

The features cache (preprocess_features.py) stores only the 119 hand-crafted
feature bytes per position — the board itself cannot be recovered from them,
yet the ResNet teacher needs it. This script replays the game files exactly
the way the C++ feature encoder does (YolahFeatures::generate_features) and
writes the board of every position, in the same order, as

    <cache_dir>/boards.u8   (N, 193) uint8 — the inputs.u8 layout of
                            preprocess_nnue.py (black/white/empty bits MSB
                            first, then the turn byte)

so that distill_teacher_labels.py can label the features cache.

Alignment with features.u8 is enforced, not assumed:
  • the position count must equal meta.json's n_positions;
  • the TURN feature (last feature byte) of every position must equal the
    replayed side to move.
Any mismatch aborts before boards.u8 is declared valid.

Replay rules mirrored from the C++ encoder (they differ from preprocess*.py):
  • files: every file in GAME_DIR whose name starts with "games" and does not
    contain "features" (this includes the *.symmetries.txt files), processed
    in the order preprocess_features.py consolidates them — sorted by the
    name of the .features.txt file the encoder writes;
  • games whose moves are all random (nb_random == nb_moves) are NOT skipped
    (preprocess.py / preprocess_nnue.py skip them);
  • the random prefix is played while the game is not over, then every
    position from there to the end of the game is written.

Usage:  python3 distill_boards_for_features.py <cache_dir>
Env:    YOLAH_GAME_DIR      games directory (default /nnue/data)
        YOLAH_PREPROC_NPROC worker processes (default min(cpu_count, 32))
"""
import os
import sys
import glob
import json
import time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.append("/server")
sys.path.append("../server")
from yolah import Yolah, Move, Square      # noqa: E402
from distill_common import BOARD_ROW       # noqa: E402

GAME_DIR  = os.environ.get("YOLAH_GAME_DIR", "/nnue/data")
NUM_PROCS = int(os.environ.get("YOLAH_PREPROC_NPROC", str(min(cpu_count(), 32))))
NB_FEATURES = 119
TURN_INDEX  = NB_FEATURES - 1


def bb_to_bits(n):
    return np.unpackbits(np.array([n], dtype=">u8").view(np.uint8))


def encoder_output_name(path):
    """Name of the .features.txt the C++ encoder writes for `path` (extension replaced)."""
    base = os.path.basename(path)
    stem = base[:base.rfind(".")] if "." in base else base
    return stem + ".features.txt"


def game_files():
    files = [p for p in glob.glob(os.path.join(GAME_DIR, "games*")) if "features" not in os.path.basename(p)]
    return sorted(files, key=encoder_output_name)


def iterate_games(data):
    """Yield (moves, nb_moves, nb_random) for every game record of a file."""
    idx = 0
    while idx < len(data):
        nb_moves, nb_random = data[idx], data[idx + 1]
        moves = data[idx + 2: idx + 2 + 2 * nb_moves]
        yield moves, nb_moves, nb_random
        idx += 2 + 2 * nb_moves + 2


def replay(moves, nb_moves, nb_random):
    """Yield the Yolah positions the C++ encoder writes for one game."""
    y = Yolah()
    i = 0
    while i < nb_random and not y.game_over():
        y.play(Move(Square(moves[2 * i]), Square(moves[2 * i + 1])))
        i += 1
    while True:
        yield y
        if y.game_over():
            break
        if i >= nb_moves:      # defensive: a record that ends before the game is over
            break
        y.play(Move(Square(moves[2 * i]), Square(moves[2 * i + 1])))
        i += 1


def count_positions(path):
    with open(path, "rb") as f:
        data = f.read()
    n = 0
    for moves, nb_moves, nb_random in iterate_games(data):
        n += sum(1 for _ in replay(moves, nb_moves, nb_random))
    return path, n


def encode_file(args):
    path, start, total, boards_path = args
    boards = np.memmap(boards_path, dtype=np.uint8, mode="r+", shape=(total, BOARD_ROW))
    with open(path, "rb") as f:
        data = f.read()
    cursor = start
    for moves, nb_moves, nb_random in iterate_games(data):
        for y in replay(moves, nb_moves, nb_random):
            boards[cursor, 0:64] = bb_to_bits(y.black)
            boards[cursor, 64:128] = bb_to_bits(y.white)
            boards[cursor, 128:192] = bb_to_bits(y.empty)
            boards[cursor, 192] = y.nb_plies() & 1
            cursor += 1
    boards.flush()
    return path, cursor - start


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cache_dir = sys.argv[1]
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    n_positions = int(meta["n_positions"])
    if "features" not in meta:
        sys.exit("ERROR: not a features cache (meta.json has no 'features' entry)")

    files = game_files()
    if not files:
        sys.exit(f"ERROR: no games* files in {GAME_DIR}")
    print(f"Source  : {GAME_DIR} ({len(files)} files)\nCache   : {cache_dir} ({n_positions:,} positions)",
          flush=True)

    print("[1/3] Counting positions (replay rules of the C++ encoder) ...", flush=True)
    t0 = time.time()
    with Pool(NUM_PROCS) as p:
        per_file = p.map(count_positions, files)
    total = sum(n for _, n in per_file)
    print(f"      {total:,} positions ({time.time() - t0:.0f}s)", flush=True)
    if total != n_positions:
        sys.exit(f"ERROR: replay yields {total:,} positions but the cache holds {n_positions:,}. "
                 f"Is YOLAH_GAME_DIR the directory the cache was built from?")

    boards_path = os.path.join(cache_dir, "boards.u8")
    with open(boards_path, "wb") as f:
        f.truncate(total * BOARD_ROW)
    starts = np.concatenate([[0], np.cumsum([n for _, n in per_file])[:-1]]).astype(int)

    print("[2/3] Writing boards ...", flush=True)
    t0 = time.time()
    with Pool(NUM_PROCS) as p:
        for k, (path, n) in enumerate(p.imap_unordered(encode_file,
                [(path, int(s), total, boards_path) for (path, _), s in zip(per_file, starts)])):
            if (k + 1) % 50 == 0 or k + 1 == len(files):
                print(f"      {k + 1}/{len(files)} files", flush=True)
    print(f"      done ({time.time() - t0:.0f}s)", flush=True)

    print("[3/3] Checking alignment with features.u8 (turn byte) ...", flush=True)
    feats = np.memmap(os.path.join(cache_dir, meta["features"]["path"]), dtype=np.uint8, mode="r",
                      shape=tuple(meta["features"]["shape"]))
    boards = np.memmap(boards_path, dtype=np.uint8, mode="r", shape=(total, BOARD_ROW))
    mismatches = 0
    step = 1 << 22
    for lo in range(0, total, step):
        hi = min(lo + step, total)
        mismatches += int((feats[lo:hi, TURN_INDEX] != boards[lo:hi, 192]).sum())
    if mismatches:
        sys.exit(f"ERROR: {mismatches:,} positions whose TURN feature disagrees with the replayed board — "
                 f"boards.u8 is NOT aligned, do not use it")
    meta["boards"] = {"path": "boards.u8", "dtype": "uint8", "shape": [total, BOARD_ROW]}
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("OK: boards.u8 aligned with features.u8; meta.json updated", flush=True)


if __name__ == "__main__":
    main()
