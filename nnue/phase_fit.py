"""
phase_fit.py — can a cheap function of hand-made features replace the NNUE in
some phases of the game? (the "fast evaluation per phase" study)

    ./phase_fit_dump CACHE_DIR nnue.quantized.txt OUT_DIR --stride 3
    python3 phase_fit.py CACHE_DIR OUT_DIR

For each range of numbers of holes, the error (MSE against the teacher, the
ResNet the NNUE is distilled from; values for the side to move in [-1, 1]) of:
    nnue       the 193 NNUE (int8, as in the minimax)
    mean       a constant (the spread of the values in that phase)
    heur       the old minimax's evaluation (heuristic::eval, GA weights),
               tanh(a · heur + b) fitted per phase
    hf         the 24 heuristic_features, least squares per phase
    yf         the 119 YolahFeatures, least squares per phase
    hf+yf      both
The linear models are fitted on atanh(teacher) (clipped) and read through
tanh, like the NNUE's head, and also fitted on the teacher directly; the
better of the two is reported. Fitted on the first 85 % of the positions,
measured on the last 15 % (the cache is in game order: no game in both).
"""
import os
import sys
import numpy as np

HF_NAMES = ["MATERIAL_DIFF", "SCORE_DIFF", "MOBILITY_DIFF", "FREE_SQUARES", "CENTER_CONTROL", "INNER_CENTER",
            "EXTENDED_CENTER", "EDGE_PIECES", "CORNER_PIECES", "NEAR_CORNER", "BLOCKED_PIECES", "FRONTIER_PIECES",
            "GROUPS_DIFF", "TENSION", "CONNECTIVITY_DIFF", "CONNECTIVITY_SET_DIFF", "ALONE_DIFF", "FIRST_DIFF",
            "INFLUENCE_DIFF", "FREEDOM_LOW", "FREEDOM_MID", "FREEDOM_HIGH", "GAME_PHASE", "TURN"]
# heuristic_features that are not black − white (no sign change for the side to move)
HF_NEUTRAL = {"FREE_SQUARES", "GAME_PHASE", "TURN", "TENSION", "SCORE_DIFF"}
YF_SIDE = 58                       # YolahFeatures: 58 black, 58 white, then CONTACT_WITH_OTHER, FREE, TURN
RANGES = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 34), (35, 41), (42, 48), (49, 56)]


def load(cache, out):
    meta = dict(l.split() for l in open(os.path.join(out, "meta.txt")))
    n = int(meta["n"])
    idx = np.fromfile(os.path.join(out, "index.i64"), dtype=np.int64)
    hf = np.fromfile(os.path.join(out, "hf.f32"), dtype=np.float32).reshape(n, -1)
    yf = np.fromfile(os.path.join(out, "yf.u8"), dtype=np.uint8).reshape(n, -1).astype(np.float32)
    heur = np.fromfile(os.path.join(out, "heur.f32"), dtype=np.float32)
    nnue = np.fromfile(os.path.join(out, "nnue.f32"), dtype=np.float32)
    rows = np.memmap(os.path.join(cache, "inputs.u8"), dtype=np.uint8, mode="r").reshape(-1, 193)
    turn = rows[idx, 192].astype(np.int64)
    holes = rows[idx, 128:192].sum(axis=1).astype(np.int64)
    teacher = np.fromfile(os.path.join(cache, "teacher_values.f16"), dtype=np.float16)[idx].astype(np.float32)
    outcome = np.fromfile(os.path.join(cache, "values.i8"), dtype=np.int8)[idx].astype(np.float32)
    # Side to move's view: black − white features change sign when white is to move.
    sign = np.where(turn == 0, 1.0, -1.0).astype(np.float32)
    for j, name in enumerate(HF_NAMES):
        if name not in HF_NEUTRAL:
            hf[:, j] *= sign
    yf_stm = yf.copy()
    w = turn == 1
    yf_stm[w, :YF_SIDE], yf_stm[w, YF_SIDE:2 * YF_SIDE] = yf[w, YF_SIDE:2 * YF_SIDE], yf[w, :YF_SIDE]
    return dict(n=n, hf=hf, yf=yf_stm, heur=heur * sign, nnue=nnue, holes=holes, teacher=teacher,
                outcome=outcome, meta=meta)


def fit_linear(Xtr, ytr, Xva, ridge=1e-3):
    """Least squares (with a constant), on atanh(y) read through tanh and on y; returns both predictions."""
    def ls(X, y):
        A = np.hstack([X, np.ones((len(X), 1), np.float32)]).astype(np.float64)
        G = A.T @ A
        G += ridge * np.trace(G) / len(G) * np.eye(len(G))
        return np.linalg.solve(G, A.T @ y.astype(np.float64))
    Ava = np.hstack([Xva, np.ones((len(Xva), 1), np.float32)]).astype(np.float64)
    w_raw = ls(Xtr, ytr)
    w_atanh = ls(Xtr, np.arctanh(np.clip(ytr, -0.98, 0.98)))
    return np.clip(Ava @ w_raw, -1, 1), np.tanh(Ava @ w_atanh)


def main():
    cache, out = sys.argv[1], sys.argv[2]
    d = load(cache, out)
    n = d["n"]
    split = int(0.85 * n)
    tr, va = np.arange(n) < split, np.arange(n) >= split
    print(f"{n} positions ({tr.sum()} fit, {va.sum()} measured); cost per position (ns): "
          f"heuristic_features {float(d['meta']['ns_hf']):.0f}, YolahFeatures {float(d['meta']['ns_yf']):.0f}, "
          f"heuristic::eval {float(d['meta']['ns_heur']):.0f}, NNUE value() {float(d['meta']['ns_nnue_value']):.0f}")
    models = ["nnue", "mean", "heur", "hf", "yf", "hf+yf"]
    print(f"\nMSE against the teacher (lower is better), measured positions\n{'holes':>7} {'positions':>9} "
          + " ".join(f"{m:>7}" for m in models) + "   best non-NNUE / NNUE")
    total = {m: 0.0 for m in models}
    for lo, hi in RANGES:
        ph = (d["holes"] >= lo) & (d["holes"] <= hi)
        a, b = tr & ph, va & ph
        if b.sum() < 100:
            continue
        t = d["teacher"]
        err = {}
        err["nnue"] = np.mean((d["nnue"][b] - t[b]) ** 2)
        err["mean"] = np.mean((t[a].mean() - t[b]) ** 2)
        feats = {"heur": d["heur"][:, None] / 1000.0, "hf": d["hf"], "yf": d["yf"],
                 "hf+yf": np.hstack([d["hf"], d["yf"]])}
        for m, X in feats.items():
            p_raw, p_tanh = fit_linear(X[a], t[a], X[b])
            err[m] = min(np.mean((p_raw - t[b]) ** 2), np.mean((p_tanh - t[b]) ** 2))
        for m in models:
            total[m] += err[m] * b.sum()
        best = min(err[m] for m in models if m != "nnue")
        print(f"{lo:>3}–{hi:<3} {b.sum():>9} " + " ".join(f"{err[m]:7.4f}" for m in models)
              + f"   ×{best / err['nnue']:.2f}")
    nv = va.sum()
    print(f"{'all':>7} {nv:>9} " + " ".join(f"{total[m] / nv:7.4f}" for m in models))


if __name__ == "__main__":
    main()
