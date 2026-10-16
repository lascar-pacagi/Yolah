"""
piece_order_study.py — which identity should a piece have in per-piece inputs?
(the user's question: the 16 pair distances of connect_study.py name the pieces
by their place in the view, which the network cannot follow.)

    ./territory_study ../data/games conn.bin --stride 20 --weights NNUE.quantized.txt
    python3 piece_order_study.py conn.bin

Per piece, five descriptors (my piece i / his piece j):
  blk      free squares it reaches by king steps (PBLK)
  ncon     pieces of the other side in contact (same block: KD > 0)
  kmin     1 / fewest king steps to one of them (0 if none)
  qgo      1 / fewest queen moves to land next to one of them
  qcome    1 / fewest queen moves for one of them to land next to it
Versions (gain over the same two bases as connect_study.py, held-out half):
  1a  pairs, view order        the 16 pairs × (bit, king, queen both ways): connect_study "all pairs"
  1b  pieces, view order       the 5 descriptors of each piece, pieces in view order
  2   pieces, isolation order  the same, each side's pieces sorted most isolated first
                               (fewest contacts, then smallest block)
  3   2 + zone                 + the piece's zone, one-hot on the 3 × 3 grid (3, 2, 3)
  3b  2 + place                + distance to the edge and to the centre (2 numbers)
1a vs 1b: pairs or pieces; 1b vs 2: does a meaningful order help; 2 vs 3: does
the place of the piece add anything to its role.
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from territory_study import NAMES, centred, PHASES  # noqa: E402
from connect_study import SPACE  # noqa: E402

F = {n: i for i, n in enumerate(NAMES)}
CAP = 1_500_000      # positions per phase


def fit_gain(base_tr, base_te, x_tr, x_te, y_tr, y_te):
    """As connect_study.fit_gain (R² on the test half), by normal equations: lstsq ran out of memory."""
    def r2(a_tr, a_te):
        c = np.linalg.solve(a_tr.T @ a_tr + 1e-6 * np.eye(a_tr.shape[1]), a_tr.T @ y_tr)
        return 100 * (1 - (y_te - a_te @ c).var() / y_te.var())
    b = r2(base_tr, base_te)
    if x_tr is None:
        return (b,)
    return r2(np.concatenate([base_tr, x_tr], 1), np.concatenate([base_te, x_te], 1)) - b


def inv(d):
    return np.where(d > 0, 1 / np.maximum(d, 1), 0)


def first(d, axis):
    """Smallest positive value along axis (0 if none)."""
    m = np.where(d > 0, d, 1e9).min(axis)
    return np.where(m < 1e9, m, 0)


def main():
    a = np.fromfile(sys.argv[1], "<i2").reshape(-1, len(NAMES))
    print(f"{len(a):,} positions")
    for name, lo, hi in PHASES:
        s = a[(a[:, F["FREE"]] >= lo) & (a[:, F["FREE"]] < hi)]
        s = s[::-(-len(s) // CAP)]     # memory: every k-th record (still grouped by game file)
        g = s[:, F["FREE"]].astype(np.int64) * 2 + s[:, F["COLOUR"]]
        kd = s[:, F["KD0"]:F["KD0"] + 16].astype(np.float64).reshape(-1, 4, 4)   # [me i, him j]
        qd = s[:, F["QD0"]:F["QD0"] + 16].astype(np.float64).reshape(-1, 4, 4)
        qh = s[:, F["QH0"]:F["QH0"] + 16].astype(np.float64).reshape(-1, 4, 4)
        sq = s[:, F["PSQ0"]:F["PSQ0"] + 8].astype(np.int64)
        blk = s[:, F["PBLK0"]:F["PBLK0"] + 8].astype(np.float64)
        con = kd > 0
        # descriptors [n, 8 pieces, 5]: my 4 then his 4
        mine = np.stack([blk[:, :4], con.sum(2), inv(first(kd, 2)), inv(first(qd, 2)), inv(first(qh, 2))], 2)
        his = np.stack([blk[:, 4:], con.sum(1), inv(first(kd, 1)), inv(first(qh, 1)), inv(first(qd, 1))], 2)
        desc = np.concatenate([mine, his], 1)
        # isolation order per side: fewest contacts, then smallest block (stable: view order on ties)
        key = desc[:, :, 1] * 100 + desc[:, :, 0]
        order = np.concatenate([np.argsort(key[:, :4], 1, kind="stable"),
                                4 + np.argsort(key[:, 4:], 1, kind="stable")], 1)
        rows = np.arange(len(s))[:, None]
        dsort, ssort = desc[rows, order], sq[rows, order]
        r, f = ssort // 8, ssort % 8
        zone = (np.where(r < 3, 0, np.where(r < 5, 1, 2)) * 3 + np.where(f < 3, 0, np.where(f < 5, 1, 2)))
        onehot = (zone[:, :, None] == np.arange(9)).astype(np.float64)
        edge = np.minimum(np.minimum(r, 7 - r), np.minimum(f, 7 - f))
        centre = np.maximum(np.abs(2 * r - 7), np.abs(2 * f - 7))
        n = len(s)
        sets = {
            "1a pairs, view order": np.concatenate([con.reshape(n, 16), kd.reshape(n, 16),
                                                    inv(qd).reshape(n, 16), inv(qh).reshape(n, 16)], 1),
            "1b pieces, view order": desc.reshape(n, -1),
            "2  pieces, isolation": dsort.reshape(n, -1),
            "3  2 + zone (one-hot)": np.concatenate([dsort.reshape(n, -1), onehot.reshape(n, -1)], 1),
            "3b 2 + edge, centre": np.concatenate([dsort.reshape(n, -1), edge, centre], 1),
        }
        raw = s[:, F["NNUE"]].astype(np.float64) / 10000
        b_nnue = np.stack([raw, raw ** 2, raw ** 3], 1)
        b_space = np.concatenate([b_nnue, s[:, [F[m] for m in SPACE] + [F["H_" + m] for m in SPACE]]], 1)
        y = {"points": s[:, F["FUTURE"]], "result": s[:, F["RESULT"]]}

        def c(x):
            return centred(np.ascontiguousarray(x, dtype=np.float64), list(range(x.shape[1])), g)
        half = n // 2
        tr, te = slice(0, half), slice(half, None)
        print(f"\n=== {name}: {n:,} positions; most isolated piece: mine {dsort[:, 0, 1].mean():.2f} contacts, "
              f"block {dsort[:, 0, 0].mean():.1f}; cut off from all his: {100 * (dsort[:, 0, 1] == 0).mean():.0f} % ===")
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
