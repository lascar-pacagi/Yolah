#!/usr/bin/env python3
"""search_study_summary.py DIR — one line per match of nnue/search_study.sh.

Every DIR/<cand>__vs__<ref>__t<T>.csv (games of yolah_tournament) gives the
candidate's score, Elo and 95 % interval with pentanomial statistics: the two
games of a round (same opening, colours swapped) form a pair, and the variance
is the variance of the pair scores (0, 1/2, …, 2 points), which removes the
opening's own bias. Rounds with a single game played (job stopped) are left out.
"""
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict


def elo(s):
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else ".", "*.csv"))):
    m = re.match(r"(.+)__vs__(.+)__t(\d+)\.csv$", os.path.basename(path))
    if not m:
        continue
    cand, ref, t = m.group(1), m.group(2), int(m.group(3))
    pairs = defaultdict(list)
    for g in csv.DictReader(open(path)):
        r = float(g["result"])                       # black's points
        pts = r if g["black"] == cand else 1.0 - r
        pairs[(g["round"], g["pair"])].append(pts)
    counts = [0] * 5
    for p in pairs.values():
        if len(p) == 2:
            counts[int(round(2 * sum(p)))] += 1
    n = sum(counts)
    if n == 0:
        continue
    s = sum(k * c for k, c in enumerate(counts)) / (4.0 * n)
    var = sum(c * (k / 4.0 - s) ** 2 for k, c in enumerate(counts)) / n
    se = math.sqrt(var / n) if n > 1 else float("inf")
    rows.append((t, ref, cand, 2 * n, s, elo(s), elo(s - 1.96 * se), elo(s + 1.96 * se), counts))

print(f"{'candidate':24s} {'reference':10s} {'µs':>8} {'games':>6} {'score':>6} {'Elo':>7}  95% interval     pentanomial")
for t, ref, cand, g, s, e, lo, hi, counts in sorted(rows):
    print(f"{cand:24s} {ref:10s} {t:8d} {g:6d} {100 * s:5.1f}% {e:+7.1f}  [{lo:+6.1f}, {hi:+6.1f}]  {counts}")
