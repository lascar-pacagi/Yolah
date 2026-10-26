"""
compare_teachers.py OLD_CACHE NEW_CACHE — two teachers' labels of the same
positions (relabel_teacher.sh: NEW_CACHE = hard links to OLD_CACHE's positions,
labelled by another network), on the validation positions of the trainers
(the last 5 %, chunks labelled in both caches):

  - mse of each teacher against the game outcome z, overall and per phase
    (free squares, the phases of nnue_persp_distill.py): the closer to z, the
    better the teacher predicts the result of the game;
  - mean |old − new|, correlation, sign agreement: how much the student's
    target changes.
"""
import json
import os
import sys
import numpy as np

from distill_common import BOARD_ROW, LABEL_CHUNK, open_teacher_labels, read_meta

PHASES = ["opening 40-56", "middle 20-39", "endgame 10-19", "endgame 0-9"]


def main():
    old_dir, new_dir = sys.argv[1], sys.argv[2]
    meta = read_meta(new_dir)
    n = int(meta["n_positions"])
    if int(read_meta(old_dir)["n_positions"]) != n:
        sys.exit("the two caches do not hold the same positions")
    inputs = np.memmap(os.path.join(new_dir, meta["inputs"]["path"]), dtype=np.uint8, mode="r",
                       shape=(n, BOARD_ROW))
    z_all = np.memmap(os.path.join(new_dir, meta["values"]["path"]), dtype=np.int8, mode="r", shape=(n,))
    t_old, d_old = open_teacher_labels(old_dir, n)
    t_new, d_new = open_teacher_labels(new_dir, n)

    k = 4                                    # 0 old vs z, 1 new vs z, 2 |old-new|, 3 count
    acc = np.zeros((len(PHASES), k))
    sxy = sxx = syy = sx = sy = 0.0
    agree = 0
    lo_chunk = int(0.95 * n) // LABEL_CHUNK + 1   # whole chunks inside the validation range
    for c in range(lo_chunk, (n + LABEL_CHUNK - 1) // LABEL_CHUNK):
        if not (d_old[c] and d_new[c]):
            continue
        lo, hi = c * LABEL_CHUNK, min((c + 1) * LABEL_CHUNK, n)
        o = np.asarray(t_old[lo:hi], dtype=np.float64)
        w = np.asarray(t_new[lo:hi], dtype=np.float64)
        z = np.asarray(z_all[lo:hi], dtype=np.float64)
        free = 56 - np.asarray(inputs[lo:hi, 128:192], dtype=np.int32).sum(axis=1)
        ph = (free < 40).astype(int) + (free < 20) + (free < 10)
        for p in range(len(PHASES)):
            m = ph == p
            acc[p] += [((o[m] - z[m]) ** 2).sum(), ((w[m] - z[m]) ** 2).sum(), np.abs(o[m] - w[m]).sum(), m.sum()]
        sx += o.sum(); sy += w.sum(); sxx += (o * o).sum(); syy += (w * w).sum(); sxy += (o * w).sum()
        agree += int((np.sign(o) == np.sign(w)).sum())
    tot = acc.sum(axis=0)
    cnt = tot[3]
    if cnt == 0:
        sys.exit("no validation chunk labelled in both caches")
    cov = sxy / cnt - (sx / cnt) * (sy / cnt)
    corr = cov / np.sqrt((sxx / cnt - (sx / cnt) ** 2) * (syy / cnt - (sy / cnt) ** 2))
    print(f"teachers: old {json.load(open(os.path.join(old_dir, 'teacher_meta.json')))['model']}, "
          f"new {json.load(open(os.path.join(new_dir, 'teacher_meta.json')))['model']}; "
          f"{int(cnt):,} validation positions")
    print(f"  {'':16s} {'mse old vs z':>13s} {'mse new vs z':>13s} {'mean |old-new|':>15s} {'positions':>12s}")
    for name, a in [("all", tot)] + list(zip(PHASES, acc)):
        if a[3]:
            print(f"  {name:16s} {a[0] / a[3]:13.4f} {a[1] / a[3]:13.4f} {a[2] / a[3]:15.4f} {int(a[3]):12,d}")
    print(f"  correlation old / new {corr:.4f}, same sign {agree / cnt:.4f}")


if __name__ == "__main__":
    main()
