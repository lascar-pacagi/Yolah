"""
encircle_study.py — does the encirclement of a piece (the user: "several of our
pieces hinder the moves of one piece", surrounded or nearly) say something the
network and the step-2 space features do not already give?

    ./territory_study ../data/games enc.bin --stride 20 --weights NNUE.quantized.txt
    python3 encircle_study.py enc.bin

Per piece (fields E* of territory_study, "other side" = the side not owning it):
  lib      free neighbours            opp    neighbours = the other side's pieces
  wall     neighbours = holes / edge  own    neighbours = its own side's pieces
  short    directions with 1-2 free squares
  gene     queen moves the other side's pieces take away from it
  contrib  other-side pieces that stop one of its lines first
  att      liberties the other side reaches in one queen move (a plug holds:
           the square it leaves becomes a hole)
  close    lib = 1 and att = 1 (closable in one move)
Versions (gain over NNUE / NNUE + space, held-out half, as piece_order_study.py):
  A  per piece, sorted      both sides, each side's 4 pieces most encircled
                            first (fewest liberties, then most gene): 9 × 8
  B  summaries, no order    per side: Σ gene, Σ contrib, Σ opp, Σ att, Σ lib,
                            Σ short, pieces with ≤ 1 liberty, with 0, closable
  C  A without the threat   A minus att and close: is the present state enough?
  D  neighbours only        A restricted to lib, opp, wall, own
  E  compact                B + the threat per piece, sorted (att, close): 34
  K  connectivity           piece_order_study version 2 (5 per piece, most isolated first): 40
  A + K, E + K              do encirclement and connectivity overlap?
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from territory_study import NAMES, centred, PHASES  # noqa: E402
from connect_study import SPACE  # noqa: E402
from piece_order_study import fit_gain, CAP, inv, first  # noqa: E402

F = {n: i for i, n in enumerate(NAMES)}
KEYS = ["ELIB", "EOPP", "EWALL", "EOWN", "ESHORT", "EGENE", "ECONTRIB", "EATT"]


def connectivity(s):
    """piece_order_study version 2: [n, 40], each side's pieces most isolated first."""
    kd, qd, qh = (s[:, F[k + "0"]:F[k + "0"] + 16].astype(np.float64).reshape(-1, 4, 4) for k in ("KD", "QD", "QH"))
    blk = s[:, F["PBLK0"]:F["PBLK0"] + 8].astype(np.float64)
    con = kd > 0
    mine = np.stack([blk[:, :4], con.sum(2), inv(first(kd, 2)), inv(first(qd, 2)), inv(first(qh, 2))], 2)
    his = np.stack([blk[:, 4:], con.sum(1), inv(first(kd, 1)), inv(first(qh, 1)), inv(first(qd, 1))], 2)
    desc = np.concatenate([mine, his], 1)
    key = desc[:, :, 1] * 100 + desc[:, :, 0]
    order = np.concatenate([np.argsort(key[:, :4], 1, kind="stable"), 4 + np.argsort(key[:, 4:], 1, kind="stable")], 1)
    return desc[np.arange(len(s))[:, None], order].reshape(len(s), -1)


def main():
    a = np.fromfile(sys.argv[1], "<i2").reshape(-1, len(NAMES))
    print(f"{len(a):,} positions")
    for name, lo, hi in PHASES:
        s = a[(a[:, F["FREE"]] >= lo) & (a[:, F["FREE"]] < hi)]
        s = s[::-(-len(s) // CAP)]
        n = len(s)
        g = s[:, F["FREE"]].astype(np.int64) * 2 + s[:, F["COLOUR"]]
        e = np.stack([s[:, F[k + "0"]:F[k + "0"] + 8] for k in KEYS], 2).astype(np.float64)   # [n, 8, 8]
        lib, gene, att = e[:, :, 0], e[:, :, 5], e[:, :, 7]
        close = ((lib == 1) & (att >= 1)).astype(np.float64)
        e = np.concatenate([e, close[:, :, None]], 2)                                          # 9 per piece
        key = lib * 100 - gene
        order = np.concatenate([np.argsort(key[:, :4], 1, kind="stable"),
                                4 + np.argsort(key[:, 4:], 1, kind="stable")], 1)
        es = e[np.arange(n)[:, None], order]
        side = lambda x: [x[:, :4].sum(1), x[:, 4:].sum(1)]   # noqa: E731
        summ = np.stack(side(gene) + side(e[:, :, 6]) + side(e[:, :, 1]) + side(att) + side(lib)
                        + side(e[:, :, 4]) + side((lib <= 1) * 1.0) + side((lib == 0) * 1.0) + side(close), 1)
        sets = {
            "A  per piece, sorted": es.reshape(n, -1),
            "B  summaries, no order": summ,
            "C  A without threat": es[:, :, :7].reshape(n, -1),
            "D  neighbours only": es[:, :, :4].reshape(n, -1),
        }
        sets["E  compact (B + threat)"] = np.concatenate([summ, es[:, :, 7:].reshape(n, -1)], 1)
        k = connectivity(s)
        sets["K  connectivity"] = k
        sets["A + K"] = np.concatenate([sets["A  per piece, sorted"], k], 1)
        sets["E + K"] = np.concatenate([sets["E  compact (B + threat)"], k], 1)
        raw = s[:, F["NNUE"]].astype(np.float64) / 10000
        b_nnue = np.stack([raw, raw ** 2, raw ** 3], 1)
        b_space = np.concatenate([b_nnue, s[:, [F[m] for m in SPACE] + [F["H_" + m] for m in SPACE]]], 1)
        y = {"points": s[:, F["FUTURE"]], "result": s[:, F["RESULT"]]}

        def c(x):
            return centred(np.ascontiguousarray(x, dtype=np.float64), list(range(x.shape[1])), g)
        half = n // 2
        tr, te = slice(0, half), slice(half, None)
        print(f"\n=== {name}: {n:,} positions; per piece: liberties {lib.mean():.2f}, gene {gene.mean():.2f}, "
              f"≤ 1 liberty {100 * (lib <= 1).mean():.0f} %, closable {100 * close.mean():.1f} % ===")
        bases = {"NNUE": c(b_nnue), "NNUE+space": c(b_space)}
        print(f"  {'gain over (points | result)':26s} " + "  ".join(f"{b:>19s}" for b in bases))
        lines = {k: [] for k in sets}
        heads = []
        cs = {k: c(v) for k, v in sets.items()}
        for bx in bases.values():
            row = []
            for yt in y.values():
                yc = c(yt[:, None].astype(np.float64))[:, 0]
                row.append(fit_gain(bx[tr], bx[te], None, None, yc[tr], yc[te])[0])
                for k, x in cs.items():
                    lines[k].append(fit_gain(bx[tr], bx[te], x[tr], x[te], yc[tr], yc[te]))
            heads.append(f"R² base {row[0]:5.1f} | {row[1]:5.1f}")
        print(f"  {'':26s} " + "  ".join(f"{h:>19s}" for h in heads))
        for k, v in lines.items():
            print(f"  + {k:24s} " + "  ".join(f"{v[2 * i]:+8.2f} | {v[2 * i + 1]:+7.2f}" for i in range(len(bases))))


if __name__ == "__main__":
    main()
