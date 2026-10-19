"""
ablation.py — backward elimination of the candidate NNUE features (doc/nnue_roadmap.qmd).

The whole network (k8_them, step 1) reads in its second layer a set of feature
GROUPS (nnue/space_features.h, space::Group); each run fine-tunes it for 5
epochs with one set (nnue_persp_distill.sh, YOLAH_SPACE_GROUPS). A round starts
from the groups still kept, trains the kept set minus each group in turn, and
removes the group whose loss costs least, if that cost is small; then again.

    python3 ablation.py plan   --kept 0x1ff [--round 1]     # → ablation_r1.sh (sbatch lines)
    python3 ablation.py report MODEL_ROOT --kept 0x1ff      # after the runs

Runs of a round (the ones already trained are skipped by `plan`):
    kept               the reference of the round (in round 1: all groups)
    kept, seed 1       round 1 only: the same with another data order = noise
    none (g000)        round 1 only: no feature, trained the same: the scale
    kept − g           for each kept group g
Cost of removing g, on the validation mse vs the teacher after the last epoch:
    cost(g) = (mse(kept − g) − mse(kept)) / (mse(none) − mse(all))
i.e. the share of what ALL the features bring that is lost without g. Also per
phase (free squares), and compared with the noise |mse(all, seed 1) − mse(all)|.
`report` proposes the cheapest group if its cost ≤ --tol (default 5 %) and
above the noise neither overall nor in a phase by more than 2 tol; the
decision stays yours (pass the new --kept to `plan`).
"""
import argparse
import json
import os

GROUPS = ["corr", "qdist", "qfirst", "qw", "late9", "mob", "infl", "connect", "encircle"]
COLUMNS = [2, 6, 2, 2, 2, 2, 2, 40, 34]
# cost of the computation per position, both sides (space::compute_groups, one group alone, local
# i7, ns); qdist, qfirst, qw, late9 and mob together ~125 ns (they share the queen levels)
COST_NS = ["150", "shared", "shared", "shared", "shared", "shared", "22", "640", "200"]
EPOCHS = 5
PHASES = ["opening 40-56", "middle 20-39", "endgame 10-19", "endgame 0-9"]
SUBMIT = ("sbatch --export=ALL,YOLAH_PERSP_K=8,YOLAH_PERSP_THEM=1,YOLAH_SPACE_GROUPS={g:#05x},"
          "YOLAH_SEED={seed},YOLAH_NB_EPOCHS={ep},MODEL_ROOT=/home/pgarcia/NNUE/DistillationNNUE/models_ablation"
          " nnue_persp_distill.sh")


def name(groups, seed=0):
    return f"k8_them_g{groups:03x}" + (f"_s{seed}" if seed else "")


def runs(kept, rnd):
    out = [(kept, 0, "reference: kept groups")]
    if rnd == 1:
        out += [(kept, 1, "noise: same, seed 1"), (0, 0, "scale: no feature")]
    out += [(kept & ~(1 << g), 0, f"without {GROUPS[g]}") for g in range(len(GROUPS)) if kept >> g & 1]
    return out


def describe(groups):
    return "+".join(GROUPS[g] for g in range(len(GROUPS)) if groups >> g & 1) or "none"


def plan(args):
    kept = int(args.kept, 0)
    path = f"ablation_r{args.round}.sh"
    root = args.root
    with open(path, "w") as f:
        f.write(f"#!/bin/bash\n# Ablation round {args.round}: kept {kept:#05x} = {describe(kept)}\n"
                f"# from nnue/ on the cluster: bash {path}\nset -e\n")
        for g, seed, why in runs(kept, args.round):
            done = root and os.path.isfile(val_path(root, g, seed))
            f.write(f"\n# {name(g, seed)}: {why}" + (" — already trained" if done else "") + "\n")
            f.write(("# " if done else "") + SUBMIT.format(g=g, seed=seed, ep=EPOCHS) + "\n")
    print(f"→ {path} ({sum(1 for _ in runs(kept, args.round))} runs of {EPOCHS} epochs, ~12 h each on one GPU)")


def val_path(root, groups, seed=0):
    return os.path.join(root, name(groups, seed), f"nnue_persp_{name(groups, seed)}_distill.val.jsonl")


def last(root, groups, seed=0):
    """The validation record of the last epoch (None if missing), and the number of epochs done."""
    p = val_path(root, groups, seed)
    if not os.path.isfile(p):
        return None, 0
    recs = [json.loads(line) for line in open(p) if line.strip()]
    recs = [r for r in recs if r["epoch"] <= EPOCHS]
    return (recs[-1] if recs and recs[-1]["epoch"] == EPOCHS else None), len(recs)


def report(args):
    kept, root, tol = int(args.kept, 0), args.root, args.tol
    full = (1 << len(GROUPS)) - 1
    ref, _ = last(root, kept)
    allg, _ = last(root, full)
    none, _ = last(root, 0)
    noise_run, _ = last(root, full, 1)
    if not (ref and allg and none):
        missing = [name(g) for g, r in ((kept, ref), (full, allg), (0, none)) if not r]
        print(f"not finished yet: {', '.join(missing)}")
        return
    scale = none["mse_teacher"] - allg["mse_teacher"]
    print(f"validation mse vs the teacher after {EPOCHS} epochs (root {root})")
    print(f"  none {none['mse_teacher']:.5f}   all {allg['mse_teacher']:.5f}   "
          f"→ the features bring {scale:+.5f} ({100 * scale / none['mse_teacher']:.2f} % of the mse)")
    pscale = {p: none["phases"][p] - allg["phases"][p] for p in PHASES}
    print("  per phase: " + "  ".join(f"{p} {pscale[p]:+.5f}" for p in PHASES))
    noise = abs(noise_run["mse_teacher"] - allg["mse_teacher"]) / scale if noise_run else None
    print(f"  noise (all, seed 1 vs seed 0): " + (f"{100 * noise:.1f} % of the gain" if noise is not None else "not finished"))
    print(f"\nround: kept {kept:#05x} = {describe(kept)}, mse {ref['mse_teacher']:.5f} "
          f"(cost of the groups already removed: {100 * (ref['mse_teacher'] - allg['mse_teacher']) / scale:+.1f} %)")
    print(f"  {'without':10s} {'cols':>4s} {'ns':>7s} {'mse':>9s} {'cost':>7s}  " + "  ".join(f"{p[:13]:>13s}" for p in PHASES))
    best, running = None, 0
    for g in range(len(GROUPS)):
        if not kept >> g & 1:
            continue
        r, n = last(root, kept & ~(1 << g))
        if not r:
            print(f"  {GROUPS[g]:10s} {COLUMNS[g]:4d} {COST_NS[g]:>7s}   running ({n}/{EPOCHS} epochs)")
            running += 1
            continue
        cost = (r["mse_teacher"] - ref["mse_teacher"]) / scale
        pc = {p: (r["phases"][p] - ref["phases"][p]) / pscale[p] if pscale[p] > 0 else 0.0 for p in PHASES}
        print(f"  {GROUPS[g]:10s} {COLUMNS[g]:4d} {COST_NS[g]:>7s} {r['mse_teacher']:9.5f} {100 * cost:+6.1f} %  "
              + "  ".join(f"{100 * pc[p]:+12.1f}%" for p in PHASES))
        if best is None or cost < best[1]:
            best = (g, cost, max(pc.values()))
    if running:
        print(f"\n{running} run(s) not finished: no decision yet")
    elif best:
        g, cost, worst = best
        ok = cost <= tol and worst <= 2 * tol
        print(f"\ncheapest: {GROUPS[g]} ({100 * cost:+.1f} % overall, worst phase {100 * worst:+.1f} %) → "
              + (f"remove it: python3 ablation.py plan --kept {kept & ~(1 << g):#05x} --round {args.round + 1}"
                 if ok else f"keep everything: no group costs ≤ {100 * tol:.0f} % (phases ≤ {200 * tol:.0f} %)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--kept", default="0x1ff")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--root", default="", help="models root, to skip the runs already trained")
    r = sub.add_parser("report")
    r.add_argument("root")
    r.add_argument("--kept", default="0x1ff")
    r.add_argument("--round", type=int, default=1)
    r.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else report(a)


if __name__ == "__main__":
    main()
