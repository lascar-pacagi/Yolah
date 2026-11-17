#!/usr/bin/env python3
"""lmr_learn.py — the learnt LMR adjustments of the dev player (step U5, idea 6).

The data: `search_bench run --player CFG --lmr-log FILE` with "lmr log" N in the
configuration logs one reduced fail-low in N, re-searched at full depth:
    bucket free depth rank history tried reduction wrong
(`wrong`: the full-depth search beat alpha, the reduction hid a move worth
searching). A bucket (player/minmax_nnue_dev_player.cpp, lmr_bucket) is a
phase × depth × rank × history × moves of the piece already tried.

The first half of the lines (the first positions) learns, the second half
checks. Per bucket, from the learning half:
    error rate >= --hi (and >= --min-n samples): −1 (one ply less of LMR)
    error rate <= --lo (and >= --min-n-plus samples): +1 (one ply more), with --plus
The table (one "bucket delta" line per non-zero bucket) goes to --out; the
check half tells what the table changes: errors moved to −1 buckets (caught),
and the error rate of the +1 buckets (paid).

    python3 nnue/lmr_learn.py lmr_log.txt --out config/search_study/lmr_table.txt [--plus]
"""
import argparse
import collections

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--out", required=True)
ap.add_argument("--hi", type=float, default=0.05)
ap.add_argument("--lo", type=float, default=0.003)
ap.add_argument("--min-n", type=int, default=300)
ap.add_argument("--min-n-plus", type=int, default=3000)
ap.add_argument("--plus", action="store_true", help="also +1 for the safest buckets")
args = ap.parse_args()

rows = []
with open(args.log) as f:
    for line in f:
        p = line.split()
        if len(p) == 8:
            rows.append((int(p[0]), int(p[7])))
half = len(rows) // 2
learn, check = rows[:half], rows[half:]

def stats(part):
    n = collections.Counter()
    e = collections.Counter()
    for b, w in part:
        n[b] += 1
        e[b] += w
    return n, e

n, e = stats(learn)
delta = {}
for b in n:
    rate = e[b] / n[b]
    if rate >= args.hi and n[b] >= args.min_n:
        delta[b] = -1
    elif args.plus and rate <= args.lo and n[b] >= args.min_n_plus:
        delta[b] = +1

nc, ec = stats(check)
tot_n, tot_e = sum(nc.values()), sum(ec.values())
minus_n = sum(nc[b] for b in delta if delta[b] < 0)
minus_e = sum(ec[b] for b in delta if delta[b] < 0)
plus_n = sum(nc[b] for b in delta if delta[b] > 0)
plus_e = sum(ec[b] for b in delta if delta[b] > 0)
print(f"{len(rows)} samples ({half} to learn), error rate {tot_e / max(tot_n, 1):.4f} on the check half")
print(f"−1 buckets: {sum(1 for d in delta.values() if d < 0)}, {minus_n / max(tot_n, 1):.1%} of the late moves, "
      f"{minus_e / max(tot_e, 1):.1%} of the errors (rate {minus_e / max(minus_n, 1):.3f})")
if args.plus:
    print(f"+1 buckets: {sum(1 for d in delta.values() if d > 0)}, {plus_n / max(tot_n, 1):.1%} of the late moves, "
          f"{plus_e / max(tot_e, 1):.1%} of the errors (rate {plus_e / max(plus_n, 1):.4f})")
with open(args.out, "w") as f:
    for b in sorted(delta):
        f.write(f"{b} {delta[b]}\n")
print(f"written {args.out}")
