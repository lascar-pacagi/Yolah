"""
az_gen_data.py — the self-play files of the generations loop (nnue/az_gen_loop.sh).

    python3 az_gen_data.py count DIR [--tag TAG]
        rows in DIR/*.bin (whose name contains TAG): one number on stdout
    python3 az_gen_data.py stats DIR [DIR ...] [--max-files 0]
        per directory: games, rows, fast / exact rows, and the DIVERSITY of the
        games: per number of moves played, the number of distinct positions
        among the games (distinct / games: 100 % = no two games share the
        position), and the solver's corrections of z
"""
import argparse
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphazero_learn import read_header, read_samples, SAMPLE_DTYPE  # noqa: E402

FAST, EXACT = 1, 2


def files_of(d, tag=""):
    return sorted(f for f in glob.glob(os.path.join(d, "*.bin")) if tag in os.path.basename(f))


def count(d, tag=""):
    n = 0
    for f in files_of(d, tag):
        try:
            n += read_header(f)[0]
        except (ValueError, OSError):
            pass
    return n


def popcount(x):
    b = np.ascontiguousarray(x, dtype="<u8").view(np.uint8).reshape(-1, 8)
    return np.unpackbits(b, axis=1).sum(1, dtype=np.int64)


def stats(d, max_files):
    files = files_of(d)
    if max_files > 0:
        files = files[-max_files:]
    if not files:
        print(f"{d}: no file")
        return
    games = sum(read_header(f)[1] for f in files)
    rows = np.concatenate([read_samples(f, SAMPLE_DTYPE) for f in files])
    n = len(rows)
    fast = (rows["flags"] & FAST) != 0
    exact = (rows["flags"] & EXACT) != 0
    print(f"== {d}: {len(files)} files, {games:,} games, {n:,} rows ({n / max(1, games):.1f} per game), "
          f"fast {fast.mean():.1%}, exact {exact.mean():.1%}")
    moves = popcount(rows["empty"])                    # holes = moves played (a pass leaves none)
    print(f"{'moves':>5} | {'rows':>9} {'distinct':>9} {'distinct/rows':>13}")
    for m in list(range(0, 21, 2)) + [25, 30, 40]:
        sel = moves == m
        k = int(sel.sum())
        if k == 0:
            continue
        key = np.stack([rows["black"][sel], rows["white"][sel], rows["empty"][sel],
                        rows["turn"][sel].astype(np.uint64)], axis=1)
        u = len(np.unique(key, axis=0))
        print(f"{m:>5} | {k:>9,} {u:>9,} {u / k:>13.1%}")
    print("(the rows of one move count are one per game at most, except passes: distinct/rows = share of the games"
          " that are in a position no other game reached at that move)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count")
    c.add_argument("dir")
    c.add_argument("--tag", default="")
    s = sub.add_parser("stats")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--max-files", type=int, default=0, help="the newest files only (0: all)")
    a = ap.parse_args()
    if a.cmd == "count":
        print(count(a.dir, a.tag))
    else:
        for d in a.dirs:
            stats(d, a.max_files)


if __name__ == "__main__":
    main()
