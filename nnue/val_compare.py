"""
val_compare.py — the validation mse of several NNUE trainings, epoch by epoch,
from their .val.jsonl files (nnue_persp_distill.py), next to the first one.

    python3 val_compare.py REF.val.jsonl OTHER.val.jsonl [...]

Per epoch: the mse against the teacher of the quantized network (as the C++
computes it) for each run, and for the others their gap to REF in %; then the
same per phase for the last epoch they all have.
"""
import json
import os
import sys


def read(path):
    with open(path) as f:
        return {r["epoch"]: r for r in map(json.loads, filter(str.strip, f))}


def name(path):
    n = os.path.basename(path).replace(".val.jsonl", "").replace("nnue_persp_", "").replace("_distill", "")
    return n[:28]


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        sys.exit(__doc__)
    runs = [read(p) for p in paths]
    names = [name(p) for p in paths]
    print(f"{'epoch':>5}  " + "  ".join(f"{n:>28}" for n in names))
    common = []
    for e in sorted(set().union(*runs)):
        ref = runs[0].get(e)
        cells = []
        for i, r in enumerate(runs):
            if e not in r:
                cells.append(f"{'-':>28}")
                continue
            v = r[e]["mse_teacher_quantized"]
            gap = f" ({100 * (v / ref['mse_teacher_quantized'] - 1):+5.1f} %)" if i and ref else ""
            cells.append(f"{v:.4f}{gap}".rjust(28))
        if all(e in r for r in runs):
            common.append(e)
        print(f"{e:>5}  " + "  ".join(cells))
    if not common:
        return
    e = common[-1]
    print(f"\nper phase, epoch {e} (float mse against the teacher)")
    for ph in runs[0][e]["phases"]:
        ref = runs[0][e]["phases"][ph]
        cells = [f"{ref:.4f}".rjust(28)]
        for r in runs[1:]:
            v = r[e]["phases"][ph]
            cells.append(f"{v:.4f} ({100 * (v / ref - 1):+5.1f} %)".rjust(28))
        print(f"{ph:>14}  " + "  ".join(cells))
    print(f"{'sign agree':>14}  " + "  ".join(f"{r[e]['sign_teacher']:.4f}".rjust(28) for r in runs))


if __name__ == "__main__":
    main()
