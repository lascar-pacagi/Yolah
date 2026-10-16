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
the two predicts better.

With --weights in territory_study (NNUE column), two more tables:
- what each measure ADDS to the network's own value (the base is a cubic in
  the NNUE value, so that its non-linear scale is not held against it):
  R² base + measure − R² base, in points of %; a large gain = something the
  network does not compute by itself;
- my side and his side separately (not their difference): the weight of each,
  in target units per standard deviation of the measure, alone and on top of
  the network. |his| > |mine| = what I take from him counts more than what I
  keep. The plain flood fill and its private part give the
same difference: the squares both sides reach cancel out.
"""
import sys
import numpy as np

NAMES = "FREE COLOUR RESULT FUTURE CURRENT MOB DIRS TRAPPED FF PRIV CORR CORRPRIV INFL " \
        "H_MOB H_DIRS H_TRAPPED H_FF H_PRIV H_CORR H_CORRPRIV H_INFL NNUE " \
        "Q1 Q2 Q3 QFIRST QW H_Q1 H_Q2 H_Q3 H_QFIRST H_QW QTIE " \
        "WASTE UNDER OWNMIN SPREAD H_WASTE H_UNDER H_OWNMIN H_SPREAD " \
        "KMEAN KMAX QMEAN LATE4 LATE9 GEO H_KMEAN H_KMAX H_QMEAN H_LATE4 H_LATE9 H_GEO".split() + \
        [f"{k}{p}" for k in ("KD", "QD", "QH") for p in range(16)]   # connectivity: connect_study.py
F = {n: i for i, n in enumerate(NAMES)}
MEASURES = [("mobility", "MOB"), ("open directions", "DIRS"), ("trapped pieces", "TRAPPED"),
            ("flood fill", "FF"), ("flood fill, private", "PRIV"), ("corrected flood fill", "CORR"),
            ("corrected, private", "CORRPRIV"), ("influence", "INFL"),
            ("queen: squares in 1 move", "Q1"), ("queen: squares in 2 moves", "Q2"),
            ("queen: squares in 3 moves", "Q3"), ("queen: reached first", "QFIRST"),
            ("queen: weighted 2^-d", "QW"),
            ("zones: wasted squares", "WASTE"), ("zones: pieces under 3", "UNDER"),
            ("zones: least piece", "OWNMIN"), ("zones: spread", "SPREAD"),
            ("cover: mean king distance", "KMEAN"), ("cover: worst square", "KMAX"),
            ("cover: mean queen distance", "QMEAN"), ("cover: late, 4 zones", "LATE4"),
            ("cover: late, 9 zones", "LATE9"), ("cover: piece spread (geom.)", "GEO")]
COMBOS = [("mobility + directions", ["MOB", "DIRS"]),
                    ("corrected + corrected private", ["CORR", "CORRPRIV"]),
          ("both flood fills (4)", ["FF", "PRIV", "CORR", "CORRPRIV"]),
          ("influence + mobility", ["INFL", "MOB"]),
          ("influence + corrected", ["INFL", "CORR"]),
          ("influence + corrected + mobility", ["INFL", "CORR", "MOB"]),
          ("queen curve (1, 2, 3 moves)", ["Q1", "Q2", "Q3"]),
          ("queen, all 5", ["Q1", "Q2", "Q3", "QFIRST", "QW"]),
          ("influence + queen all", ["INFL", "Q1", "Q2", "Q3", "QFIRST", "QW"]),
          ("corrected + queen all", ["CORR", "Q1", "Q2", "Q3", "QFIRST", "QW"]),
          ("corrected + influence + queen all + mobility", ["CORR", "INFL", "Q1", "Q2", "Q3", "QFIRST", "QW", "MOB"]),
          ("zones, all 4", ["WASTE", "UNDER", "OWNMIN", "SPREAD"]),
          ("queen all + zones", ["Q1", "Q2", "Q3", "QFIRST", "QW", "WASTE", "UNDER", "OWNMIN", "SPREAD"]),
          ("corrected + queen all + zones", ["CORR", "Q1", "Q2", "Q3", "QFIRST", "QW", "WASTE", "UNDER", "OWNMIN", "SPREAD"]),
          ("cover, all 6", ["KMEAN", "KMAX", "QMEAN", "LATE4", "LATE9", "GEO"]),
          ("queen all + cover", ["Q1", "Q2", "Q3", "QFIRST", "QW", "KMEAN", "KMAX", "QMEAN", "LATE4", "LATE9", "GEO"]),
          ("corrected + queen all + cover", ["CORR", "Q1", "Q2", "Q3", "QFIRST", "QW", "KMEAN", "KMAX", "QMEAN", "LATE4", "LATE9", "GEO"]),
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
    has_nnue = (a[:, F["NNUE"]] != 0).any()
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
        if has_nnue:
            v = centred(s, [F["NNUE"]], group)[:, 0] / 10000
            raw = s[:, F["NNUE"]].astype(np.float64) / 10000
            base = centred(np.stack([raw, raw ** 2, raw ** 3], 1), [0, 1, 2], group)
            b_fut, b_res = r2(base, y_fut), r2(base, y_res)
            print(f"  NNUE value alone (cubic): R² points {b_fut:.1f} %, result {b_res:.1f} %. Gain when adding:")
            for label, ms in [(l, [m]) for l, m in MEASURES] + COMBOS:
                xx = np.concatenate([base, np.stack([col[m] for m in ms], 1)], 1)
                print(f"    + {label:30s} {r2(xx, y_fut) - b_fut:+6.1f} {r2(xx, y_res) - b_res:+6.1f}")
            print("  my side vs his side, weight per standard deviation (points | result), alone / on top of NNUE:")
            for label, m in [("mobility", "MOB"), ("open directions", "DIRS"), ("corrected flood fill", "CORR"),
                             ("influence", "INFL"), ("queen: squares in 2 moves", "Q2"), ("queen: reached first", "QFIRST"),
                             ("zones: wasted squares", "WASTE"), ("zones: least piece", "OWNMIN"),
                             ("cover: mean king distance", "KMEAN"), ("cover: late, 9 zones", "LATE9"),
                             ("cover: piece spread (geom.)", "GEO")]:
                xs = centred(s, [F[m], F["H_" + m]], group)
                xs /= xs.std(0) + 1e-9
                out = []
                for extra in (None, base):
                    xx = xs if extra is None else np.concatenate([xs, extra], 1)
                    cf, *_ = np.linalg.lstsq(xx, y_fut, rcond=None)
                    cr, *_ = np.linalg.lstsq(xx, y_res, rcond=None)
                    out.append(f"mine {cf[0]:+.3f} his {cf[1]:+.3f} | mine {cr[0]:+.3f} his {cr[1]:+.3f}")
                print(f"    {label:22s} {out[0]}   /   {out[1]}")
        if differs.sum() > 1000:
            k = differs
            print(f"    on those {k.sum():,} positions: R² points  plain {r2(col['FF'][k], y_fut[k]):.1f} %, "
                  f"corrected {r2(col['CORR'][k], y_fut[k]):.1f} %;  private: plain {r2(col['PRIV'][k], y_fut[k]):.1f} %, "
                  f"corrected {r2(col['CORRPRIV'][k], y_fut[k]):.1f} %")


if __name__ == "__main__":
    main()
