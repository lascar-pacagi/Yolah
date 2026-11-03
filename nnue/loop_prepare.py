"""
loop_prepare.py — the training cache of one round of the distillation loop
(nnue/distill_loop.sh), from the games of test/selfplay_label_main.cpp.

    python3 loop_prepare.py ROUND_DIR [--old-cache DIR --old-positions N] [--block 4096] [--seed 0]

Reads ROUND_DIR/games/*.bin (40-byte records) and writes ROUND_DIR/cache/, in
the layout of preprocess_nnue.py, so that distill_teacher_labels.py,
space_features compute-all and nnue_persp_distill.py read it as they are:

    inputs.u8     (N, 193) board rows: black, white, holes (MSB first), turn
    values.i8     (N,)     z, the game's result for the side to move
    meta.json
and for the loop:
    search.f16    (N,)     the search target for the side to move: the exact
                           result (−1, 0, 1) when the solver gave it, else the
                           search value / 30000 (a proven result: ±1)
    policy.u2     (N,)     the move played (a position of a game) or the best
                           move (a tree node), as a policy index from·64 + to
                           (nn::action_index); 65535 = none (a pass, an old
                           position)
    kind.u1       (N,)     0 a position of a game, 1 a node of a search tree,
                           2 an old position (--old-cache)
    depth.u1      (N,)     depth of the search that gave the value (0: old)
    exact.u1      (N,)     1: search.f16 is the exact result (the solver, or a
                           result the search proved)
    zmask.u1      (N,)     1: z is the position's result (a position of the
                           game, an old position, or a solved tree node, whose
                           z is then the exact result); 0: a node of a search
                           tree off the game's path, whose z is the result of
                           another position (the trainers then leave z out)

The tree nodes that are positions of their own game (the game went there: 19 %
of them) are dropped, the game's record (a deeper search) is kept.

--old-cache: N positions of an older cache (inputs.u8, values.i8 and its teacher
labels teacher_values.f16 as the search target), in random blocks of --block
consecutive positions (whole games, mostly), so that the network does not
forget them. The rows of the new games and the old blocks are mixed by blocks
of --block rows: the validation part of the trainers (the last 5 %) holds both.
"""
import argparse
import glob
import json
import os
import numpy as np

RECORD = np.dtype([("black", "<u8"), ("white", "<u8"), ("empty", "<u8"), ("ply", "<u2"),
                   ("black_score", "u1"), ("white_score", "u1"), ("value", "<i2"), ("move", "<u2"),
                   ("depth", "u1"), ("kind", "u1"), ("z", "i1"), ("exact", "i1"), ("game", "<u4")])
assert RECORD.itemsize == 40
EXACT_UNKNOWN = -128
WIN = 30000
ROW = 193
NO_POLICY = 65535


def board_rows(rec):
    """(n, 193) uint8 rows: the 3 bitboards MSB first (byte i = bit 63 − i), then the turn."""
    n = len(rec)
    out = np.empty((n, ROW), np.uint8)
    for k, name in enumerate(("black", "white", "empty")):
        b = rec[name].astype(">u8").view(np.uint8).reshape(n, 8)
        out[:, 64 * k:64 * (k + 1)] = np.unpackbits(b, axis=1)
    out[:, 192] = rec["ply"] & 1
    return out


def search_target(rec):
    v = rec["value"].astype(np.float32)
    t = np.where(np.abs(v) > WIN, np.sign(v), v / WIN)
    exact = rec["exact"] != EXACT_UNKNOWN
    t[exact] = rec["exact"][exact]
    return t.astype(np.float16)


def policy_index(rec):
    m = rec["move"].astype(np.int64)
    frm, to = m & 63, m >> 6
    p = frm * 64 + to
    ok = frm != to                                 # a1→a1: the pass (Move::none)
    return np.where(ok, p, NO_POLICY).astype(np.uint16)


def drop_tree_on_path(rec):
    """Without the tree nodes that are also a position played in the same game."""
    key = np.empty(len(rec), [("game", "<u4"), ("black", "<u8"), ("white", "<u8"), ("empty", "<u8"), ("ply", "<u2")])
    for f in key.dtype.names:
        key[f] = rec[f]
    _, ids = np.unique(key, return_inverse=True)
    ids = ids.ravel()
    played = np.zeros(ids.max() + 1, bool)
    played[ids[rec["kind"] == 0]] = True
    drop = (rec["kind"] == 1) & played[ids]
    print(f"{drop.sum():,} of {(rec['kind'] == 1).sum():,} tree nodes are positions of their game: dropped", flush=True)
    return rec[~drop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round_dir")
    ap.add_argument("--old-cache", default="")
    ap.add_argument("--old-positions", type=int, default=0)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    files = sorted(glob.glob(os.path.join(a.round_dir, "games", "*.bin")))
    if not files:
        raise SystemExit(f"no games in {a.round_dir}/games")
    rec = np.concatenate([np.fromfile(f, RECORD) for f in files])
    rec = drop_tree_on_path(rec)
    n_games = len(np.unique(rec["game"]))
    n_new = len(rec)
    print(f"{len(files)} files, {n_games:,} games, {n_new:,} positions "
          f"({(rec['kind'] == 0).sum():,} of the games, {(rec['kind'] == 1).sum():,} of the trees, "
          f"{(rec['exact'] != EXACT_UNKNOWN).sum():,} solved)", flush=True)

    old = None
    n_old = 0
    if a.old_cache and a.old_positions > 0:
        with open(os.path.join(a.old_cache, "meta.json")) as f:
            om = json.load(f)
        on = int(om["n_positions"])
        old = dict(inputs=np.memmap(os.path.join(a.old_cache, "inputs.u8"), np.uint8, "r", shape=(on, ROW)),
                   values=np.memmap(os.path.join(a.old_cache, "values.i8"), np.int8, "r", shape=(on,)),
                   teacher=np.memmap(os.path.join(a.old_cache, "teacher_values.f16"), np.float16, "r", shape=(on,)))
        nb = min(a.old_positions, on) // a.block
        old_starts = np.sort(rng.choice(on // a.block, size=nb, replace=False)) * a.block
        n_old = nb * a.block
        print(f"old cache {a.old_cache}: {n_old:,} positions in {nb:,} blocks of {a.block}", flush=True)

    # The order of the output: blocks of the new rows and old blocks, shuffled.
    new_blocks = [("new", s) for s in range(0, n_new, a.block)]
    old_blocks = [("old", int(s)) for s in old_starts] if old is not None else []
    blocks = new_blocks + old_blocks
    order = rng.permutation(len(blocks))
    n = n_new + n_old

    out = os.path.join(a.round_dir, "cache")
    os.makedirs(out, exist_ok=True)
    mm = lambda name, dt, shape: np.memmap(os.path.join(out, name), dt, "w+", shape=shape)
    inputs = mm("inputs.u8", np.uint8, (n, ROW))
    values = mm("values.i8", np.int8, (n,))
    search = mm("search.f16", np.float16, (n,))
    policy = mm("policy.u2", np.uint16, (n,))
    kind = mm("kind.u1", np.uint8, (n,))
    depth = mm("depth.u1", np.uint8, (n,))
    exact = mm("exact.u1", np.uint8, (n,))
    zmask = mm("zmask.u1", np.uint8, (n,))
    pos = 0
    for j, b in enumerate(order):
        src, s = blocks[b]
        if src == "new":
            r = rec[s:s + a.block]
            k = len(r)
            inputs[pos:pos + k] = board_rows(r)
            solved = r["exact"] != EXACT_UNKNOWN
            values[pos:pos + k] = np.where((r["kind"] == 1) & solved, r["exact"], r["z"])
            zmask[pos:pos + k] = (r["kind"] == 0) | solved
            search[pos:pos + k] = search_target(r)
            policy[pos:pos + k] = policy_index(r)
            kind[pos:pos + k] = r["kind"]
            depth[pos:pos + k] = r["depth"]
            exact[pos:pos + k] = (r["exact"] != EXACT_UNKNOWN) | (np.abs(r["value"].astype(np.int32)) > WIN)
        else:
            k = a.block
            inputs[pos:pos + k] = old["inputs"][s:s + k]
            values[pos:pos + k] = old["values"][s:s + k]
            search[pos:pos + k] = old["teacher"][s:s + k]
            policy[pos:pos + k] = NO_POLICY
            kind[pos:pos + k] = 2
            depth[pos:pos + k] = 0
            exact[pos:pos + k] = 0
            zmask[pos:pos + k] = 1
        pos += k
        if j % 2000 == 0:
            print(f"  {pos:,} / {n:,}", flush=True)
    assert pos == n
    for m in (inputs, values, search, policy, kind, depth, exact, zmask):
        m.flush()
    meta = {"n_positions": n, "n_games": n_games,
            "inputs": {"path": "inputs.u8", "dtype": "uint8", "shape": [n, ROW]},
            "values": {"path": "values.i8", "dtype": "int8", "shape": [n]},
            "input_size": ROW, "value_perspective": "current_player", "value_range": [-1, 0, 1],
            "loop": {"new_positions": n_new, "old_positions": n_old, "old_cache": a.old_cache,
                     "files": len(files), "block": a.block, "seed": a.seed}}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out}: {n:,} positions", flush=True)


if __name__ == "__main__":
    main()
