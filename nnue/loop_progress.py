"""
loop_progress.py LOOP_DIR — the progress of the distillation loop
(nnue/distill_loop.sh), one line per evaluation match:

  NNUE    LOOP_DIR/eval/nnue_r<B>__vs__nnue_r<A>.csv (yolah_tournament games):
          the dev player with the network of round B against round A, Elo and
          95 % interval with pentanomial statistics (as test/search_study_summary.py)
  ResNet  LOOP_DIR/resnet/matches.csv (nnue/alphazero_match.sh): the MCTS player
          with the ResNet of round B (az_<B>) against round A

Writes LOOP_DIR/progress.csv and prints the table.
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


def nnue_rows(loop):
    rows = []
    for path in sorted(glob.glob(os.path.join(loop, "eval", "*.csv"))):
        m = re.match(r"nnue_r(\d+)__vs__nnue_r(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        cand, ref = f"nnue_r{m.group(1)}", f"nnue_r{m.group(2)}"
        pairs = defaultdict(list)
        for g in csv.DictReader(open(path)):
            r = float(g["result"])
            pairs[(g["round"], g["pair"])].append(r if g["black"] == cand else 1.0 - r)
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
        rows.append(dict(player="nnue", round=int(m.group(1)), against=int(m.group(2)), games=2 * n,
                         score=f"{s:.3f}", elo=f"{elo(s):+.1f}", elo_low=f"{elo(s - 1.96 * se):+.1f}",
                         elo_high=f"{elo(s + 1.96 * se):+.1f}"))
    return rows


def resnet_rows(loop):
    path = os.path.join(loop, "resnet", "matches.csv")
    if not os.path.exists(path):
        return []
    rows = []
    for g in csv.DictReader(open(path)):
        a, b = int(g["a"][3:]), int(g["b"][3:])       # az_<step>
        rows.append(dict(player="resnet", round=a, against=b, games=int(g["games"]), score=g["score"],
                         elo=g["elo"], elo_low=g["elo_low95"], elo_high=g["elo_high95"]))
    return rows


def main():
    loop = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = sorted(nnue_rows(loop) + resnet_rows(loop), key=lambda r: (r["player"], r["round"], r["against"]))
    fields = ["player", "round", "against", "games", "score", "elo", "elo_low", "elo_high"]
    with open(os.path.join(loop, "progress.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{'player':7s} {'round':>5} {'vs':>4} {'games':>6} {'score':>6} {'Elo':>7}  95% interval")
    for r in rows:
        print(f"{r['player']:7s} {r['round']:5d} {r['against']:4d} {r['games']:6d} {r['score']:>6} {r['elo']:>7}  "
              f"[{r['elo_low']}, {r['elo_high']}]")


if __name__ == "__main__":
    main()
