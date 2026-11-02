"""
loop_blend.py — the NNUE's teacher target of one round of the distillation
loop (nnue/distill_loop.sh), after distill_teacher_labels.py has labelled the
round's cache with the round's ResNet.

    python3 loop_blend.py CACHE [--search-weight 0.5]

    teacher = (1 − w) · v_resnet + w · s

s being the cache's search target (search.f16: the exact result near the end,
else the dev player's search value; the old teacher's value on the old
positions). Where the exact result is known (exact.u1: the solver, or a
result the search proved), it replaces the blend: the network should learn the
truth there, not an estimate of it. The ResNet's labels are kept in
teacher_resnet.f16 (the script can be run again with another weight);
teacher_values.f16 then holds the blend, and nnue_persp_distill.py trains on
    α · MSE(v, teacher) + (1 − α) · MSE(v, z)
as before.
"""
import argparse
import json
import os
import shutil
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--search-weight", type=float, default=0.5)
    a = ap.parse_args()
    with open(os.path.join(a.cache, "meta.json")) as f:
        n = int(json.load(f)["n_positions"])
    tv = os.path.join(a.cache, "teacher_values.f16")
    tr = os.path.join(a.cache, "teacher_resnet.f16")
    if not os.path.exists(tr):
        shutil.copyfile(tv, tr)
    resnet = np.memmap(tr, np.float16, "r", shape=(n,))
    search = np.memmap(os.path.join(a.cache, "search.f16"), np.float16, "r", shape=(n,))
    is_exact = np.memmap(os.path.join(a.cache, "exact.u1"), np.uint8, "r", shape=(n,))
    out = np.memmap(tv, np.float16, "r+", shape=(n,))
    w = a.search_weight
    stats = np.zeros(4)
    step = 1 << 24
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        r = resnet[lo:hi].astype(np.float32)
        s = search[lo:hi].astype(np.float32)
        t = (1.0 - w) * r + w * s
        exact = is_exact[lo:hi] == 1
        t[exact] = s[exact]
        out[lo:hi] = t.astype(np.float16)
        stats += [hi - lo, exact.sum(), ((r - s) ** 2).sum(), ((r[exact] - s[exact]) ** 2).sum()]
    out.flush()
    n_all, n_exact = stats[0], max(1.0, stats[1])
    report = {"search_weight": w, "positions": int(n_all), "exact": int(stats[1]),
              "mse_resnet_vs_search": stats[2] / n_all, "mse_resnet_vs_exact": stats[3] / n_exact}
    with open(os.path.join(a.cache, "blend_meta.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
