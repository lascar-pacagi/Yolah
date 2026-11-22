#!/usr/bin/env python3
"""devdata_check.py — what the games of gen_dev_data.sh look like so far.

    python3 nnue/devdata_check.py [~/DevData/games] [--files N]

Reads the task files (40-byte records of test/selfplay_label_main.cpp, kind 0
only: the root positions) and prints:
  • per file: games, positions (→ with the running time of the task, games/h);
  • diversity, per number of random plies r (game g: r = g mod 10):
    distinct openings (the first recorded position: r random plies after the
    start), distinct whole games (opening + every move of the dev player),
    distinct prefixes of the first 10 dev moves, and distinct positions;
  • the labels: z (result for the side to move), the exact results (≤ 22 free
    squares), the search depth, the value.
A game = its records sorted by ply; the random plies themselves are not
recorded, their outcome is the first recorded position.
"""
import os
import sys
from collections import defaultdict

import numpy as np

REC = np.dtype([("b", "<u8"), ("w", "<u8"), ("e", "<u8"), ("ply", "<u2"), ("bs", "u1"), ("ws", "u1"),
                ("value", "<i2"), ("move", "<u2"), ("depth", "u1"), ("kind", "u1"), ("z", "i1"),
                ("exact", "i1"), ("game", "<u4")])

args = [a for a in sys.argv[1:] if not a.startswith("--")]
d = os.path.expanduser(args[0] if args else "~/DevData/games")
max_files = int(sys.argv[sys.argv.index("--files") + 1]) if "--files" in sys.argv else 10**9

files = sorted(f for f in os.listdir(d) if f.endswith(".bin"))[:max_files]
parts = []
print("file       games  positions")
for f in files:
    a = np.fromfile(os.path.join(d, f), dtype=REC)
    a = a[a["kind"] == 0]
    if len(a) == 0:
        continue
    # The last game of a running task may be written in part: drop it.
    a = a[a["game"] != a["game"][-1]]
    print(f"{f}  {len(np.unique(a['game'])):6d}  {len(a):9d}")
    parts.append(a)
a = np.concatenate(parts)
a = a[np.lexsort((a["ply"], a["game"]))]
print(f"\n{len(np.unique(a['game']))} complete games, {len(a)} positions, "
      f"{len(a) / len(np.unique(a['game'])):.1f} per game")

# ── diversity ────────────────────────────────────────────────────────────────
starts = np.flatnonzero(np.r_[True, a["game"][1:] != a["game"][:-1]])
ends = np.r_[starts[1:], len(a)]
openings, wholes, prefixes = defaultdict(set), defaultdict(set), defaultdict(set)
ngames = defaultdict(int)
for s, t in zip(starts, ends):
    r = int(a["game"][s]) % 10
    o = (int(a["b"][s]), int(a["w"][s]), int(a["e"][s]))
    mv = a["move"][s:t].tobytes()
    ngames[r] += 1
    openings[r].add(o)
    wholes[r].add((o, mv))
    prefixes[r].add((o, mv[:20]))         # 10 moves × 2 bytes
pos = np.stack([a["b"], a["w"], a["e"]], axis=1)
r_of = (a["game"] % 10).astype(int)
print("\n r   games  openings  whole games  10-move prefixes  positions (distinct / all)")
for r in sorted(ngames):
    p = pos[r_of == r]
    up = len(np.unique(p, axis=0))
    print(f" {r}  {ngames[r]:6d}  {len(openings[r]):8d}  {len(wholes[r]):11d}  {len(prefixes[r]):16d}"
          f"  {up:9d} / {len(p):9d} ({100 * up / len(p):.1f} %)")
up = len(np.unique(pos, axis=0))
print(f"all positions: {up} distinct / {len(pos)} ({100 * up / len(pos):.1f} %)")

# ── labels ───────────────────────────────────────────────────────────────────
z = a["z"]
print(f"\nz (side to move): win {100 * np.mean(z == 1):.1f} %, draw {100 * np.mean(z == 0):.1f} %, "
      f"loss {100 * np.mean(z == -1):.1f} %")
first = a[starts]
zb = np.where(first["ply"] % 2 == 0, first["z"], -first["z"])   # for black
print(f"games: black wins {100 * np.mean(zb == 1):.1f} %, draws {100 * np.mean(zb == 0):.1f} %, "
      f"white wins {100 * np.mean(zb == -1):.1f} %")
ex = a["exact"]
known = (ex >= -1) & (ex <= 1)
occ = a["b"] | a["w"] | a["e"]          # e = the holes (squares already left)
free = 64 - np.array([bin(int(x)).count("1") for x in occ])
print(f"exact result known: {100 * np.mean(known):.1f} % of positions "
      f"({100 * np.mean(known[free <= 22]):.1f} % of those with ≤ 22 free squares); "
      f"agrees with z: {100 * np.mean(ex[known] == z[known]):.1f} %")
print(f"search depth: mean {a['depth'].mean():.1f}, by ply 0-9 / 10-29 / 30+: "
      f"{a['depth'][a['ply'] < 10].mean():.1f} / {a['depth'][(a['ply'] >= 10) & (a['ply'] < 30)].mean():.1f} / "
      f"{a['depth'][a['ply'] >= 30].mean():.1f}")
v = a["value"].astype(int)
print(f"value: mean |v| {np.abs(v).mean():.0f}, sign agrees with z (z ≠ 0, |v| > 0): "
      f"{100 * np.mean(np.sign(v[(z != 0) & (v != 0)]) == z[(z != 0) & (v != 0)]):.1f} %")
