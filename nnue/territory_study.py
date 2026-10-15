"""
territory_study.py — what each measure of space (test/territory_study_main.cpp)
says about the rest of the game.

    ./territory_study ../data/games terr.bin --stride 10
    python3 territory_study.py terr.bin

Every measure is taken as a DIFFERENCE, side to move minus the other side
(e.g. mobility = my moves − his moves). Targets: the points I score from here
to the end minus his (FUTURE), and the result (+1 / 0 / −1). To compare
positions at the same stage of the game, targets and measures are centred per
(number of free squares, colour of the side to move): what remains is what the
measure adds to "Black to move with 30 free squares".

Printed per phase: the share of the target's variance each measure explains
alone (R², %), then combinations (least squares), then how often the corrected
flood fill differs from the plain one and, on those positions only, which of
the two predicts better. The plain flood fill and its private part give the
same difference: the squares both sides reach cancel out.
"""
import sys
import numpy as np

NAMES = "FREE COLOUR RESULT FUTURE CURRENT MOB DIRS TRAPPED FF PRIV CORR CORRPRIV INFL " \
        "H_MOB H_DIRS H_TRAPPED H_FF H_PRIV H_CORR H_CORRPRIV H_INFL".split()
F = {n: i for i, n in enumerate(NAMES)}
MEASURES = [("mobility", "MOB"), ("open directions", "DIRS"), ("trapped pieces", "TRAPPED"),
            ("flood fill", "FF"), ("flood fill, private", "PRIV"), ("corrected flood fill", "CORR"),
            ("corrected, private", "CORRPRIV"), ("influence", "INFL")]
COMBOS = [("mobility + directions", ["MOB", "DIRS"]),
                    ("corrected + corrected private", ["CORR", "CORRPRIV"]),
          ("both flood fills (4)", ["FF", "PRIV", "CORR", "CORRPRIV"]),
          ("influence + mobility", ["INFL", "MOB"]),
          ("influence + corrected", ["INFL", "CORR"]),
          ("influence + corrected + mobility", ["INFL", "CORR", "MOB"]),
          ("everything", [m for _, m in MEASURES])]
PHASES = [("opening 40-56", 40, 57), ("middle 20-39", 20, 40), ("endgame 10-19", 10, 20), ("endgame 0-9", 0, 10)]


def centred(a, cols, group):
    """Columns as float64, each centred within its (free, colour) group."""
    x = a[:, cols].astype(np.float64)
    n = np.bincount(group, minlength=130).astype(np.float64)
    for j in range(x.shape[1]):
        m = np.bincount(group, weights=x[:, j], minlength=130) / np.maximum(n, 1)
        x[:, j] -= m[group]
    return x


def r2(x, y):
    """Share of the variance of y explained by a least-squares fit on x (centred)."""
    if x.ndim == 1:
        x = x[:, None]
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    res = y - x @ coef
    return 100 * (1 - res.var() / y.var())


def main():
    a = np.fromfile(sys.argv[1], "<i2").reshape(-1, len(NAMES))
    print(f"{len(a):,} positions")
    for name, lo, hi in PHASES:
        s = a[(a[:, F["FREE"]] >= lo) & (a[:, F["FREE"]] < hi)]
        group = (s[:, F["FREE"]].astype(np.int64) * 2 + s[:, F["COLOUR"]])
        diff = {m: (s[:, F[m]] - s[:, F["H_" + m]]).astype(np.int16) for _, m in MEASURES}
        d = np.stack([diff[m] for _, m in MEASURES], 1)
        x = centred(d, list(range(d.shape[1])), group)
        col = {m: x[:, j] for j, (_, m) in enumerate(MEASURES)}
        y_fut = centred(s, [F["FUTURE"]], group)[:, 0]
        y_res = centred(s, [F["RESULT"]], group)[:, 0]
        print(f"\n=== {name}: {len(s):,} positions; points still to score, mean |margin| "
              f"{np.abs(s[:, F['FUTURE']]).mean():.1f} ===")
        print(f"  {'measure (me − him)':32s} {'R² points %':>11s} {'R² result %':>11s}   mean |diff|")
        for label, m in MEASURES:
            print(f"  {label:32s} {r2(col[m], y_fut):11.1f} {r2(col[m], y_res):11.1f}   {np.abs(diff[m]).mean():8.2f}")
        for label, ms in COMBOS:
            xx = np.stack([col[m] for m in ms], 1)
            print(f"  {label:32s} {r2(xx, y_fut):11.1f} {r2(xx, y_res):11.1f}")
        # Where the corrected flood fill differs from the plain one.
        differs = (s[:, F["CORR"]] != s[:, F["FF"]]) | (s[:, F["H_CORR"]] != s[:, F["H_FF"]])
        over = np.concatenate([s[:, F["FF"]] - s[:, F["CORR"]], s[:, F["H_FF"]] - s[:, F["H_CORR"]]])
        print(f"  corrected ≠ plain for at least one side: {100 * differs.mean():.1f} % of positions; "
              f"when a side is corrected, the plain flood fill overcounts by {over[over > 0].mean():.1f} squares "
              f"(max {over.max()})")
        if differs.sum() > 1000:
            k = differs
            print(f"    on those {k.sum():,} positions: R² points  plain {r2(col['FF'][k], y_fut[k]):.1f} %, "
                  f"corrected {r2(col['CORR'][k], y_fut[k]):.1f} %;  private: plain {r2(col['PRIV'][k], y_fut[k]):.1f} %, "
                  f"corrected {r2(col['CORRPRIV'][k], y_fut[k]):.1f} %")


if __name__ == "__main__":
    main()
