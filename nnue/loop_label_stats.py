"""
loop_label_stats.py — what the NNUE's targets of one round of the
distillation loop are made of, per number of free squares.

    python3 loop_label_stats.py ROUND_DIR [--max-files 0] [--sample 4000000] [--seed 0]

1. The games (ROUND_DIR/games/*.bin, the records of selfplay_label): per game,
   how many positions of the game and of the search trees, how many with an
   exact value (the solver at most --exact-free free squares, or a result the
   search proved); per bucket of free squares, the share of exact values, for
   the positions of the games and for the tree nodes.
2. The cache (ROUND_DIR/cache, when it is labelled: teacher_resnet.f16), on
   --sample rows drawn in blocks: per bucket, for the positions of the games
   whose value is not exact, how well each part of the NNUE's teacher predicts
   the game's result z: the dev player's search value s (NNUE leaves), the
   ResNet's value r, the blend used as the teacher; and how far s is from r on
   the tree nodes. And the share of each kind of target in the cache (old
   positions, game positions, tree nodes; exact or not).
"""
import argparse
import glob
import json
import os
import numpy as np

RECORD = np.dtype([("black", "<u8"), ("white", "<u8"), ("empty", "<u8"), ("ply", "<u2"),
                   ("black_score", "u1"), ("white_score", "u1"), ("value", "<i2"), ("move", "<u2"),
                   ("depth", "u1"), ("kind", "u1"), ("z", "i1"), ("exact", "i1"), ("game", "<u4")])
EXACT_UNKNOWN = -128
WIN = 30000
ROW = 193
BUCKET = 4


def popcount64(x):
    b = np.ascontiguousarray(x, dtype="<u8").view(np.uint8).reshape(-1, 8)
    return np.unpackbits(b, axis=1).sum(1, dtype=np.int64)


def bucket_label(b):
    return f"{b * BUCKET:2d}-{b * BUCKET + BUCKET - 1:2d}"


def games_part(round_dir, max_files):
    files = sorted(glob.glob(os.path.join(round_dir, "games", "*.bin")))
    if not files:
        print(f"no games in {round_dir}/games\n")
        return
    if max_files > 0:
        files = files[:max_files]
    nb = 64 // BUCKET
    cnt = np.zeros((2, nb), np.int64)       # kind 0 / 1
    ex = np.zeros((2, nb), np.int64)
    per_game = []                           # (path, path exact, tree, tree exact, free at the first exact path position)
    for f in files:
        rec = np.fromfile(f, RECORD)
        free = 64 - popcount64(rec["black"] | rec["white"] | rec["empty"])
        exact = (rec["exact"] != EXACT_UNKNOWN) | (np.abs(rec["value"].astype(np.int32)) > WIN)
        b = np.clip(free // BUCKET, 0, nb - 1)
        for k in (0, 1):
            m = rec["kind"] == k
            cnt[k] += np.bincount(b[m], minlength=nb)
            ex[k] += np.bincount(b[m & exact], minlength=nb)
        order = np.argsort(rec["game"], kind="stable")
        g = rec["game"][order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        ends = np.r_[starts[1:], len(g)]
        kk, ee, ff = rec["kind"][order], exact[order], free[order]
        for s, e in zip(starts, ends):
            k, x, fr = kk[s:e], ee[s:e], ff[s:e]
            p = k == 0
            first = fr[p & x].max() if (p & x).any() else -1
            per_game.append((p.sum(), (p & x).sum(), (~p).sum(), (~p & x).sum(), first))
    pg = np.array(per_game, np.float64)
    n_games = len(pg)
    print(f"== games: {len(files)} files, {n_games:,} games")
    print(f"per game (mean): {pg[:, 0].mean():.1f} positions of the game, {pg[:, 1].mean():.1f} of them exact "
          f"({pg[:, 1].sum() / pg[:, 0].sum():.1%}); {pg[:, 2].mean():.1f} tree nodes, {pg[:, 3].mean():.1f} exact "
          f"({pg[:, 3].sum() / max(1, pg[:, 2].sum()):.1%})")
    first = pg[pg[:, 4] >= 0, 4]
    if len(first):
        q = np.percentile(first, [10, 50, 90])
        print(f"free squares at the first exact position of the game: 10 % {q[0]:.0f}, median {q[1]:.0f}, 90 % {q[2]:.0f}")
    print(f"\n{'free':>6} | {'game pos.':>10} {'exact':>6} | {'tree nodes':>11} {'exact':>6}")
    for b in range(nb - 1, -1, -1):
        if cnt[0, b] + cnt[1, b] == 0:
            continue
        r0 = ex[0, b] / cnt[0, b] if cnt[0, b] else float("nan")
        r1 = ex[1, b] / cnt[1, b] if cnt[1, b] else float("nan")
        print(f"{bucket_label(b):>6} | {cnt[0, b]:>10,} {r0:>6.1%} | {cnt[1, b]:>11,} {r1:>6.1%}")
    print()


def cache_part(round_dir, sample, seed):
    c = os.path.join(round_dir, "cache")
    if not os.path.exists(os.path.join(c, "teacher_resnet.f16")):
        print(f"no labelled cache in {c} (teacher_resnet.f16): run the stages prepare and relabel\n")
        return
    with open(os.path.join(c, "meta.json")) as f:
        n = int(json.load(f)["n_positions"])
    w = None
    bm = os.path.join(c, "blend_meta.json")
    if os.path.exists(bm):
        with open(bm) as f:
            w = json.load(f).get("search_weight")
    mm = lambda name, dt: np.memmap(os.path.join(c, name), dt, "r", shape=(n,))
    inputs = np.memmap(os.path.join(c, "inputs.u8"), np.uint8, "r", shape=(n, ROW))
    kind, exact, zmask = mm("kind.u1", np.uint8), mm("exact.u1", np.uint8), mm("zmask.u1", np.uint8)
    z, s, r, t = mm("values.i8", np.int8), mm("search.f16", np.float16), mm("teacher_resnet.f16", np.float16), mm("teacher_values.f16", np.float16)

    # The shares of the whole cache (small columns: read entirely).
    k_all, x_all = np.asarray(kind), np.asarray(exact)
    print(f"== cache: {n:,} positions" + (f", teacher = {1 - w:g} ResNet + {w:g} search, exact results replace it" if w is not None else ""))
    for k, name in ((2, "old positions (old teacher)"), (0, "positions of the games"), (1, "tree nodes")):
        m = k_all == k
        print(f"  {name:28s} {m.sum():>12,} ({m.mean():5.1%})   exact {(m & (x_all == 1)).sum():>11,} ({(m & (x_all == 1)).mean():5.1%})")

    # A sample of rows, in blocks (the memmaps are read sequentially).
    rng = np.random.default_rng(seed)
    blk = 1 << 16
    nblk = max(1, min(n // blk, sample // blk))
    starts = np.sort(rng.choice(max(1, n // blk), size=nblk, replace=False)) * blk
    nb = 64 // BUCKET
    acc = {}
    def add(key, b, v):
        a = acc.setdefault(key, [np.zeros(nb), np.zeros(nb)])
        a[0] += np.bincount(b, weights=v, minlength=nb)
        a[1] += np.bincount(b, minlength=nb)
    for st in starts:
        sl = slice(int(st), int(min(n, st + blk)))
        free = 64 - inputs[sl, :192].sum(1, dtype=np.int64)
        b = np.clip(free // BUCKET, 0, nb - 1)
        kk, xx, zm = kind[sl], exact[sl], zmask[sl]
        zz, ss, rr, tt = (np.asarray(v[sl], np.float32) for v in (z, s, r, t))
        g = (kk == 0) & (xx == 0) & (zm == 1)          # game positions, value not exact, z known
        for name, v in (("s", ss), ("r", rr), ("t", tt)):
            add("mse_" + name, b[g], (v[g] - zz[g]) ** 2)
            dec = g & (zz != 0)
            add("sign_" + name, b[dec], (np.sign(v[dec]) == zz[dec]).astype(np.float64))
        add("n_g", b[g], np.ones(g.sum()))
        tr = (kk == 1) & (xx == 0)
        add("sr_tree", b[tr], (ss[tr] - rr[tr]) ** 2)
        add("sr_game", b[g], (ss[g] - rr[g]) ** 2)
        ex = (kk != 2) & (xx == 1)
        add("r_exact", b[ex], (rr[ex] - ss[ex]) ** 2)
        add("n_ex", b[ex], np.ones(ex.sum()))
        nx = (kk != 2) & (xx == 0)
        add("n_nx", b[nx], np.ones(nx.sum()))
    f = lambda key, i: acc[key][0][i] / acc[key][1][i] if acc[key][1][i] else float("nan")
    print(f"\nsample of {sum(acc['n_ex'][1]) + sum(acc['n_nx'][1]):,.0f} new positions (games and trees, no old position)")
    print("positions of the games whose value is not exact: how well each target predicts z (mse; sign = right side wins, decisive games)")
    print(f"{'free':>6} {'exact':>6} | {'n':>8} | {'mse s':>6} {'mse r':>6} {'mse t':>6} | {'sign s':>6} {'sign r':>6} {'sign t':>6} | {'(s-r)² game':>11} {'(s-r)² tree':>11} | {'(r-exact)²':>10}")
    for i in range(nb - 1, -1, -1):
        tot = acc["n_ex"][0][i] + acc["n_nx"][0][i]
        if tot == 0:
            continue
        print(f"{bucket_label(i):>6} {acc['n_ex'][0][i] / tot:>6.1%} | {acc['n_g'][0][i]:>8,.0f} | "
              f"{f('mse_s', i):>6.3f} {f('mse_r', i):>6.3f} {f('mse_t', i):>6.3f} | "
              f"{f('sign_s', i):>6.1%} {f('sign_r', i):>6.1%} {f('sign_t', i):>6.1%} | "
              f"{f('sr_game', i):>11.4f} {f('sr_tree', i):>11.4f} | {f('r_exact', i):>10.3f}")
    print("\nexact = share of the new positions with an exact value; s = the dev player's search value (NNUE leaves),"
          " r = ResNet_{r+1}, t = the blend; (r-exact)² = the ResNet's error where the exact result is known")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round_dir")
    ap.add_argument("--max-files", type=int, default=0, help="games files read (0: all)")
    ap.add_argument("--sample", type=int, default=4_000_000, help="cache rows sampled")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    games_part(a.round_dir, a.max_files)
    cache_part(a.round_dir, a.sample, a.seed)


if __name__ == "__main__":
    main()
