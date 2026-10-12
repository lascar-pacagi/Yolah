"""
endgame_truth.py — the teacher, the NNUE and the games' outcomes against the
exact result of the endgame solver (retrograde analysis study: would exact
endgame labels teach the NNUE something the teacher does not?)

    ./endgame_truth CACHE_DIR nnue.quantized.txt OUT.bin --free 14,16,18,20,22 --per 2000
    python3 endgame_truth.py CACHE_DIR OUT.bin

Per number of free squares, for the solved positions:
    result errors   how often the sign of the value (win / loss) differs from
                    the exact result (exact draws: |value| > 0.5 counts as an error)
    mse vs exact    against the exact result as -1 / 0 / +1 (the value's target)
for the teacher (the ResNet the NNUE is distilled from), the NNUE (193,
int8), and the game's outcome (the other training target: was the game
played perfectly from there?). Then the solve times.
"""
import os
import sys
import numpy as np

REC = np.dtype([("index", "<i8"), ("free", "i1"), ("solved", "i1"), ("diff", "<i2"), ("nnue", "<f4"), ("ms", "<f4")])


def main():
    cache, path = sys.argv[1], sys.argv[2]
    r = np.fromfile(path, dtype=REC)
    teacher = np.fromfile(os.path.join(cache, "teacher_values.f16"), dtype=np.float16)[r["index"]].astype(np.float32)
    outcome = np.fromfile(os.path.join(cache, "values.i8"), dtype=np.int8)[r["index"]].astype(np.float32)
    exact = np.sign(r["diff"]).astype(np.float32)

    def wrong(v, e):
        return np.where(e == 0, np.abs(v) > 0.5, np.sign(v) != e)

    print(f"{'free':>4} {'solved':>11} {'win/draw/loss %':>17} {'result errors %':>27} {'mse vs exact':>27}  "
          f"{'solve ms':>16}")
    print(f"{'':>4} {'':>11} {'':>17} {'teacher':>9}{'NNUE':>9}{'game':>9} {'teacher':>9}{'NNUE':>9}{'game':>9}  "
          f"{'median':>8}{'max':>8}")
    tot = {}
    for f in sorted(set(r["free"])):
        m = r["free"] == f
        s = m & (r["solved"] == 1)
        e = exact[s]
        row = []
        for name, v in (("teacher", teacher[s]), ("NNUE", r["nnue"][s]), ("game", outcome[s])):
            row.append((wrong(v, e).mean() * 100, np.mean((v - e) ** 2)))
            t = tot.setdefault(name, [0, 0.0, 0.0])
            t[0] += s.sum(); t[1] += wrong(v, e).sum(); t[2] += ((v - e) ** 2).sum()
        wdl = [np.mean(e == 1) * 100, np.mean(e == 0) * 100, np.mean(e == -1) * 100]
        print(f"{f:>4} {s.sum():>5}/{m.sum():<5} {wdl[0]:5.1f}/{wdl[1]:4.1f}/{wdl[2]:5.1f} "
              + "".join(f"{x[0]:9.1f}" for x in row) + " " + "".join(f"{x[1]:9.4f}" for x in row)
              + f"  {np.median(r['ms'][m]):8.1f}{r['ms'][m].max():8.0f}")
    n = tot["teacher"][0]
    print(f"{'all':>4} {n:>11} {'':>17} " + "".join(f"{tot[k][1] / n * 100:9.1f}" for k in ("teacher", "NNUE", "game"))
          + " " + "".join(f"{tot[k][2] / n:9.4f}" for k in ("teacher", "NNUE", "game")))


if __name__ == "__main__":
    main()
