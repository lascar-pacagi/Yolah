"""
policies_from_nnue_cache.py — the "move played" target (policies.i16, the
ResNet's policy target) for an NNUE cache of preprocess_nnue.py, WITHOUT
replaying the games: the cache stores every game's positions one after the
other (ply r … end), so the move played in row i is read from rows i and i+1.

    row i+1 of the same game:  one new hole (the from square) and one new piece
                               of the side that moved (the to square), the
                               other side unchanged, the turn flipped
    a pass (Move.none() = a1→a1, index 0): same board, turn flipped
    anything else: row i is the last position of its game → −1 (ignored)

Cell c of the cache is square 63 − c (MSB first), so the action index is
(63 − c_from) · 64 + (63 − c_to), as written by preprocess.py.

Writes <cache>/policies.i16 (int16, one per row; meta.json is left as it is)
and prints the counts. --check N: also compares with a CNN cache of
preprocess.py built from the same games (policies.i16 there).

    python3 policies_from_nnue_cache.py /cache [--check-cnn /cnn_cache]
"""
import argparse
import json
import os
import sys

import numpy as np

BLOCK = 1 << 22   # rows per block (4 M rows ≈ 0.8 GB of inputs)


def policies_of(rows, nxt_first):
    """rows: (n, 193) uint8 block; nxt_first: the row after the block or None → (n,) int16 + counts."""
    n = len(rows)
    nxt = np.empty_like(rows)
    nxt[:-1] = rows[1:]
    if nxt_first is not None:
        nxt[-1] = nxt_first
    black, white, holes, turn = rows[:, 0:64], rows[:, 64:128], rows[:, 128:192], rows[:, 192]
    nblack, nwhite, nholes, nturn = nxt[:, 0:64], nxt[:, 64:128], nxt[:, 128:192], nxt[:, 192]
    flipped = nturn != turn
    if nxt_first is None:
        flipped[-1] = False
    black_moves = (turn == 0)[:, None]
    mover, nmover = np.where(black_moves, black, white), np.where(black_moves, nblack, nwhite)
    other, nother = np.where(black_moves, white, black), np.where(black_moves, nwhite, nblack)

    new_hole = (nholes == 1) & (holes == 0)
    lost_hole = (holes == 1) & (nholes == 0)
    new_piece = (nmover == 1) & (mover == 0)
    left = (mover == 1) & (nmover == 0)
    same_other = (other == nother).all(axis=1)
    n_new_hole, n_new_piece, n_left = new_hole.sum(1), new_piece.sum(1), left.sum(1)

    is_move = (flipped & same_other & ~lost_hole.any(1) & (n_new_hole == 1) & (n_new_piece == 1) & (n_left == 1)
               & (left == new_hole).all(axis=1))           # the piece left the square that became a hole
    is_pass = flipped & (rows[:, :192] == nxt[:, :192]).all(axis=1)

    c_from = new_hole.argmax(1)
    c_to = new_piece.argmax(1)
    pol = np.full(n, -1, dtype=np.int16)
    pol[is_move] = ((63 - c_from[is_move]) * 64 + (63 - c_to[is_move])).astype(np.int16)
    pol[is_pass] = 0
    return pol, int(is_move.sum()), int(is_pass.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("--check-cnn", help="a preprocess.py cache of the same games: compare")
    args = ap.parse_args()
    with open(os.path.join(args.cache, "meta.json")) as f:
        meta = json.load(f)
    n = int(meta["n_positions"])
    inputs = np.memmap(os.path.join(args.cache, meta["inputs"]["path"]), dtype=np.uint8, mode='r',
                       shape=tuple(meta["inputs"]["shape"]))
    out_path = os.path.join(args.cache, "policies.i16")
    tmp = out_path + ".tmp"
    out = np.memmap(tmp, dtype=np.int16, mode='w+', shape=(n,))
    moves = passes = 0
    for s in range(0, n, BLOCK):
        e = min(n, s + BLOCK)
        rows = np.array(inputs[s:e])
        pol, m, p = policies_of(rows, np.array(inputs[e]) if e < n else None)
        out[s:e] = pol
        moves, passes = moves + m, passes + p
        print(f"  {e:,}/{n:,} rows", file=sys.stderr, flush=True)
    out.flush()
    del out
    os.replace(tmp, out_path)
    ends = n - moves - passes
    print(f"{n:,} rows: {moves:,} moves, {passes:,} passes, {ends:,} game ends (−1)"
          f"; games in meta: {meta.get('n_games', '?'):,}" if isinstance(meta.get('n_games'), int) else
          f"{n:,} rows: {moves:,} moves, {passes:,} passes, {ends:,} game ends (−1)")
    if args.check_cnn:
        with open(os.path.join(args.check_cnn, "meta.json")) as f:
            m2 = json.load(f)
        ref = np.memmap(os.path.join(args.check_cnn, m2["policies"]["path"]), dtype=np.int16, mode='r',
                        shape=(int(m2["n_positions"]),))
        mine = np.memmap(out_path, dtype=np.int16, mode='r', shape=(n,))
        assert len(ref) == n, (len(ref), n)
        diff = int((np.asarray(ref) != np.asarray(mine)).sum())
        print(f"compared with {args.check_cnn}: {diff:,} different rows of {n:,}")


if __name__ == "__main__":
    main()
