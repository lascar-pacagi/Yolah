"""
connect_study.py — does the connectivity between my pieces and his pieces
(test/territory_study_main.cpp, fields KD/QD/QH: 4 × 4 pairs, pieces in the
order of the side to move's network view) say something the network does not
already see, and something the space features of step 2 do not already give?

    ./territory_study ../data/games conn.bin --stride 20 --weights NNUE.quantized.txt
    python3 connect_study.py conn.bin

Same conventions as territory_study.py: targets (points still to score, result)
centred per (free squares, colour). Two bases, both fitted on the first half of
the records and measured on the second half (other games):
  NNUE        a cubic in the network's value
  NNUE+space  the same + the 16 step-2 space features (corrected flood fill,
              queen 1/2/3, reached first, weighted, late zones, mobility; mine
              and his separately, as the network reads them)
Gain = R² (base + set) − R² (base), in points of %, on the held-out half.
Sets:
  bits          16 × connected or not (same block of free squares)
  bits + king   + the king-step distance of the connected pairs
  queen         16 × 1 / (queen moves of my piece to his) + 16 × the other way
  all pairs     bits + king + queen (48)
  summaries     order-free counts: connected pairs, my pieces connected to none
                of his, his to none of mine, Σ 1/king distance, Σ 1/queen
                distance each way, my pieces reaching one of his in 1 move,
                his reaching one of mine in 1 move
  table         the mean residual of each of the 65536 connection patterns
                (any function of the 16 bits: the most the bits alone can give)
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from territory_study import NAMES, centred, PHASES  # noqa: E402

F = {n: i for i, n in enumerate(NAMES)}
SPACE = ["CORR", "Q1", "Q2", "Q3", "QFIRST", "QW", "LATE9", "MOB"]


def fit_gain(base_tr, base_te, x_tr, x_te, y_tr, y_te):
    """R² on the test half of base alone and of base + x (least squares on the train half)."""
    def r2(a_tr, a_te):
        c, *_ = np.linalg.lstsq(a_tr, y_tr, rcond=None)
        return 100 * (1 - (y_te - a_te @ c).var() / y_te.var()), y_te - a_te @ c, y_tr - a_tr @ c
    b, res_te, res_tr = r2(base_tr, base_te)
    if x_tr is None:
        return b, res_tr, res_te
    g, *_ = r2(np.concatenate([base_tr, x_tr], 1), np.concatenate([base_te, x_te], 1))
    return g - b


def table_gain(mask_tr, mask_te, res_tr, res_te, y_te):
    """Gain of the mean residual per connection pattern (shrunk to 0 on rare patterns)."""
    n = np.bincount(mask_tr, minlength=65536).astype(np.float64)
    s = np.bincount(mask_tr, weights=res_tr, minlength=65536)
    t = s / (n + 100)
    pred = t[mask_te]
    c = (pred @ res_te) / max(pred @ pred, 1e-12)     # one scale on top (≈ 1)
    return 100 * (res_te.var() - (res_te - c * pred).var()) / y_te.var()


def main():
    a = np.fromfile(sys.argv[1], "<i2").reshape(-1, len(NAMES))
    print(f"{len(a):,} positions")
    for name, lo, hi in PHASES:
        s = a[(a[:, F["FREE"]] >= lo) & (a[:, F["FREE"]] < hi)]
        g = s[:, F["FREE"]].astype(np.int64) * 2 + s[:, F["COLOUR"]]
        kd = s[:, F["KD0"]:F["KD0"] + 16].astype(np.float64)
        qd = s[:, F["QD0"]:F["QD0"] + 16].astype(np.float64)
        qh = s[:, F["QH0"]:F["QH0"] + 16].astype(np.float64)
        bits = (kd > 0).astype(np.float64)
        inv = lambda d: np.where(d > 0, 1 / np.maximum(d, 1), 0)   # noqa: E731
        sets = {
            "bits": bits,
            "bits + king": np.concatenate([bits, kd], 1),
            "queen": np.concatenate([inv(qd), inv(qh)], 1),
            "all pairs": np.concatenate([bits, kd, inv(qd), inv(qh)], 1),
            "summaries": np.stack([
                bits.sum(1),
                (bits.reshape(-1, 4, 4).sum(2) == 0).sum(1),
                (bits.reshape(-1, 4, 4).sum(1) == 0).sum(1),
                inv(kd).sum(1), inv(qd).sum(1), inv(qh).sum(1),
                (qd.reshape(-1, 4, 4) == 1).any(2).sum(1),
                (qh.reshape(-1, 4, 4) == 1).any(1).sum(1)], 1),
        }
        # pieces really present: rows with fewer than 4 pieces are impossible in Yolah (4 each)
        raw = s[:, F["NNUE"]].astype(np.float64) / 10000
        b_nnue = np.stack([raw, raw ** 2, raw ** 3], 1)
        b_space = np.concatenate([b_nnue, s[:, [F[m] for m in SPACE] + [F["H_" + m] for m in SPACE]]], 1)
        y = {"points": s[:, F["FUTURE"]], "result": s[:, F["RESULT"]]}
        mask = (bits * (1 << np.arange(16))).sum(1).astype(np.int64)

        # centre everything per group, then split in halves (records are grouped by game file)
        def c(x):
            return centred(np.ascontiguousarray(x, dtype=np.float64), list(range(x.shape[1])), g)
        half = len(s) // 2
        tr, te = slice(0, half), slice(half, None)
        print(f"\n=== {name}: {len(s):,} positions; connected pairs on average {bits.sum(1).mean():.1f} / 16, "
              f"all 16 connected {100 * (bits.sum(1) == 16).mean():.0f} %, none {100 * (bits.sum(1) == 0).mean():.0f} %, "
              f"distinct patterns {len(np.unique(mask)):,} ===")
        cs = {k: c(v) for k, v in sets.items()}
        bases = {"NNUE": c(b_nnue), "NNUE+space": c(b_space)}
        print(f"  {'gain over (points | result)':22s} " + "  ".join(f"{b:>19s}" for b in bases))
        lines = {k: [] for k in list(sets) + ["table"]}
        heads = []
        for bn, bx in bases.items():
            row = []
            for tn, yt in y.items():
                yc = c(yt[:, None].astype(np.float64))[:, 0]
                r2b, res_tr, res_te = fit_gain(bx[tr], bx[te], None, None, yc[tr], yc[te])
                row.append(r2b)
                for k, x in cs.items():
                    lines[k].append(fit_gain(bx[tr], bx[te], x[tr], x[te], yc[tr], yc[te]))
                lines["table"].append(table_gain(mask[tr], mask[te], res_tr, res_te, yc[te]))
            heads.append(f"R² base {row[0]:5.1f} | {row[1]:5.1f}")
        print(f"  {'':22s} " + "  ".join(f"{h:>19s}" for h in heads))
        for k, v in lines.items():
            print(f"  + {k:20s} " + "  ".join(f"{v[2 * i]:+8.2f} | {v[2 * i + 1]:+7.2f}" for i in range(len(bases))))


if __name__ == "__main__":
    main()
