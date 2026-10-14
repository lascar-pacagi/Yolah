"""
pattern_stats.py — the local patterns (3×3 around a piece) that go with won
games and with lost ones (test/pattern_stats_main.cpp).

    ./pattern_stats ../data/games pat3.bin --stride 10
    python3 pattern_stats.py pat3.bin [--top 10] [--min-share 0.002]

The 8 symmetric versions of a pattern (rotations, reflections) are one
pattern (merged by pattern_stats, each piece counted once).
For each phase (free squares 40–56, 20–39, 0–19) and for each number of
fixed neighbours (1, 2, 3, 4, 8: the others are ?), the patterns whose
occurrences have the best and the worst result for the piece's owner,
beyond what the colour, the side to move and the number of free squares
already say:

    gain   mean of (owner's result − expected result), result in −1 / 0 / +1:
           +0.10 = the owner scores 5 points of percentage more than expected
    share  pieces matching the pattern (in any orientation) / pieces
    z      gain / its standard error, the occurrences of a game counted as ONE
           (they share the game's result): a conservative significance

Patterns are drawn from the piece's owner's view: P the piece, m one of his
pieces, a an opponent's piece, o a hole, . a free square, # off the board,
? anything.
"""
import sys
import numpy as np

NB, NP, PHASES = 8, 6 ** 8, 3
PHASE_NAMES = ["40–56 free squares (opening)", "20–39 free squares (middle game)", "0–19 free squares (endgame)"]
SYM = "?.oma#"
DR = [1, 1, 1, 0, 0, -1, -1, -1]
DC = [-1, 0, 1, -1, 1, -1, 0, 1]


def draw(pattern):
    d = [(pattern // 6 ** k) % 6 for k in range(NB)]
    cell = {(r, c): SYM[d[k]] for k, (r, c) in enumerate(zip(DR, DC))}
    cell[(0, 0)] = "P"
    return [" ".join(cell[(r, c)] for c in (-1, 0, 1)) for r in (1, 0, -1)]


def main():
    args = sys.argv[1:]
    top, min_share = 10, 0.002
    if "--top" in args:
        top = int(args[args.index("--top") + 1])
    if "--min-share" in args:
        min_share = float(args[args.index("--min-share") + 1])
    raw = open(args[0], "rb").read()
    o = 0
    count = np.frombuffer(raw, np.uint32, PHASES * NP, o).reshape(PHASES, NP).astype(np.float64); o += 4 * PHASES * NP
    sums = np.frombuffer(raw, np.float64, PHASES * NP, o).reshape(PHASES, NP); o += 8 * PHASES * NP
    total = np.frombuffer(raw, np.uint64, PHASES, o).astype(np.float64); o += 8 * PHASES
    total_sq = np.frombuffer(raw, np.float64, PHASES, o); o += 8 * PHASES
    games = float(np.frombuffer(raw, np.uint64, 1, o)[0])
    per_game = total.sum() / games

    digits = np.stack([(np.arange(NP) // 6 ** k) % 6 for k in range(NB)], 1)
    fixed = (digits > 0).sum(1)
    print(f"{int(total.sum())} piece occurrences in {int(games)} games ({per_game:.0f} per game, which share "
          f"their result: z counts a game as one)")
    for ph in range(PHASES):
        c, s = count[ph], sums[ph]          # non-zero only at the canonical patterns
        sd = np.sqrt(total_sq[ph] / total[ph])
        ok = c >= min_share * total[ph]
        gain = np.where(c > 0, s / np.maximum(c, 1), 0)
        # standard error with the occurrences of one game counted as one
        z = gain / (sd / np.sqrt(np.maximum(c / per_game, 1)))
        print(f"\n══ {PHASE_NAMES[ph]}: {int(total[ph])} occurrences ══")
        for k in (1, 2, 3, 4, 8):
            sel = np.nonzero(ok & (fixed == k))[0]
            if len(sel) == 0:
                continue
            for title, order in (("WINNING", np.argsort(-z[sel])), ("LOSING", np.argsort(z[sel]))):
                best = sel[order[:top]]
                best = best[(z[best] > 0) if title == "WINNING" else (z[best] < 0)]
                if len(best) == 0:
                    continue
                print(f"\n  {k} fixed neighbour(s), {title} patterns ({len(sel)} patterns with share ≥ {min_share:.1%}):")
                pics = [draw(p) for p in best]
                for row in range(3):
                    print("    " + "".join(f"{pic[row]:<10}" for pic in pics))
                print("    " + "".join(f"{gain[p]:<+10.3f}" for p in best) + "gain")
                print("    " + "".join(f"{c[p] / total[ph]:<10.1%}" for p in best) + "share")
                print("    " + "".join(f"{z[p]:<10.1f}" for p in best) + "z")


if __name__ == "__main__":
    main()
