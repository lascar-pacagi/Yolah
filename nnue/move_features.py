"""
move_features.py — which kinds of moves do strong players choose? (search work, step I)

    python3 move_features.py POSITIONS BEST.csv [BEST2.csv ...] [--label convnet]

POSITIONS: one Yolah JSON per line (search_bench positions); BEST.csv: the move
chosen in each position by a player (search_bench bestmove: idx,move).

For every legal move of every position, candidate "tactical" features are
computed; for each feature (and some combinations):
    freq  = share of all the legal moves that have it,
    best  = share of the chosen moves that have it,
    lift  = best / freq.
A useful signal for move ordering / reductions is RARE (low freq) with a HIGH
lift. Results by game phase (number of free squares) too.

Board: bit s = file + 8·rank (a1 = 0). Free squares = not black, white or hole.
Regions use 8-connectivity (a queen slides diagonally too).
"""
import argparse
import csv
import json
from collections import defaultdict

FULL = (1 << 64) - 1
NOT_A = 0xFEFEFEFEFEFEFEFE
NOT_H = 0x7F7F7F7F7F7F7F7F
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def neighbours(x):
    e = (x << 1) & NOT_A
    w = (x >> 1) & NOT_H
    h = x | e | w
    return ((h << 8) | (h >> 8) | e | w) & FULL


def fill(seed, free):
    """Free squares 8-connected to seed (seed itself must be free)."""
    r = seed & free
    while True:
        n = (r | neighbours(r)) & free
        if n == r:
            return r
        r = n


def popcount(x):
    return bin(x).count("1")


def bits(x):
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def sq_name(s):
    return "abcdefgh"[s & 7] + str((s >> 3) + 1)


def parse_sq(t):
    return "abcdefgh".index(t[0]) + 8 * (int(t[1]) - 1)


def moves_of(pieces, occupied):
    res = []
    for f in bits(pieces):
        fx, fy = f & 7, f >> 3
        for dx, dy in DIRS:
            x, y = fx + dx, fy + dy
            while 0 <= x < 8 and 0 <= y < 8:
                t = x + 8 * y
                if (occupied >> t) & 1:
                    break
                res.append((f, t))
                x += dx
                y += dy
    return res


def local_articulation(t, free):
    cells = [s for s in bits(neighbours(1 << t) & free)]
    parent = {c: c for c in cells}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            if abs((a & 7) - (b & 7)) <= 1 and abs((a >> 3) - (b >> 3)) <= 1:
                parent[find(a)] = find(b)
    return len({find(c) for c in cells}) >= 2


def private_territory(mine, theirs, free):
    """Free squares reachable (through free squares) by my pieces only, by theirs only."""
    rm = fill(neighbours(mine), free)
    rt = fill(neighbours(theirs), free)
    return popcount(rm & ~rt), popcount(rt & ~rm)


def features(f, t, me, opp, holes, opp_reach):
    occupied = me | opp | holes
    free = FULL & ~occupied
    after_free = free & ~(1 << t)
    region = fill(1 << t, free)
    rest = region & ~(1 << t)
    # pieces of the region after the move (sizes of the parts)
    parts = []
    left = rest
    while left:
        p = fill(left & -left, after_free)
        parts.append(popcount(p))
        left &= ~p
    parts.sort(reverse=True)
    exact = len(parts) >= 2
    small = parts[1] if exact else 0           # second largest part
    mine_after = (me & ~(1 << f)) | (1 << t)
    p0, o0 = private_territory(me, opp, free)
    p1, o1 = private_territory(mine_after, opp, after_free)
    d_terr = (p1 - o1) - (p0 - o0)
    return {
        "local": local_articulation(t, free),
        "exact": exact,
        "cut>=3": small >= 3,
        "cut>=6": small >= 6,
        "contested": bool((opp_reach >> t) & 1),
        "dterr>=3": d_terr >= 3,
        "dterr>=6": d_terr >= 6,
        "dterr<=-3": d_terr <= -3,
        "opp_private-": o1 < o0,                 # the opponent's private territory shrinks
        "my_private+": p1 > p0,
        "exact&contested": exact and bool((opp_reach >> t) & 1),
        "cut>=3&contested": small >= 3 and bool((opp_reach >> t) & 1),
        "exact&dterr>=3": exact and d_terr >= 3,
        "cut>=3&dterr>=3": small >= 3 and d_terr >= 3,
    }, d_terr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("positions")
    ap.add_argument("best", nargs="+")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    positions = [json.loads(line) for line in open(args.positions) if line.strip()]
    best = {}
    for path in args.best:
        for r in csv.DictReader(open(path)):
            best[int(r["idx"])] = r["move"]

    phases = [(40, 64), (32, 39), (24, 31), (0, 23)]
    count_all = defaultdict(lambda: defaultdict(int))     # phase → feature → count
    count_best = defaultdict(lambda: defaultdict(int))
    n_all = defaultdict(int)
    n_best = defaultdict(int)
    dterr_best, dterr_all = defaultdict(list), defaultdict(list)
    missing = 0
    for idx, mv in best.items():
        j = positions[idx]
        black, white, holes = int(j["black"]), int(j["white"]), int(j["empty"])
        stm = int(j["ply"]) & 1
        me, opp = (black, white) if stm == 0 else (white, black)
        occupied = black | white | holes
        free_n = popcount(FULL & ~occupied)
        phase = next(p for p in phases if p[0] <= free_n <= p[1])
        legal = moves_of(me, occupied)
        if not legal:
            continue
        opp_reach = 0
        for _, t in moves_of(opp, occupied):
            opp_reach |= 1 << t
        f_best, t_best = parse_sq(mv[:2]), parse_sq(mv[3:5])
        if (f_best, t_best) not in legal:
            missing += 1
            continue
        for f, t in legal:
            feats, d = features(f, t, me, opp, holes, opp_reach)
            n_all[phase] += 1
            dterr_all[phase].append(d)
            for k, v in feats.items():
                count_all[phase][k] += v
            if (f, t) == (f_best, t_best):
                n_best[phase] += 1
                dterr_best[phase].append(d)
                for k, v in feats.items():
                    count_best[phase][k] += v
    if missing:
        print(f"WARNING: {missing} chosen moves not among the generated legal moves")
    print(f"{args.label}: {sum(n_best.values())} positions, {sum(n_all.values())} legal moves")
    keys = list(next(iter(count_all.values())).keys())
    for phase in phases:
        if not n_best[phase]:
            continue
        print(f"\n── {phase[0]}–{phase[1]} free squares: {n_best[phase]} positions, "
              f"{n_all[phase] / n_best[phase]:.1f} moves per position ──")
        print(f"   {'feature':20s} {'freq':>7} {'best':>7} {'lift':>6}")
        for k in keys:
            fa = count_all[phase][k] / n_all[phase]
            fb = count_best[phase][k] / n_best[phase]
            lift = fb / fa if fa > 0 else float("nan")
            print(f"   {k:20s} {100 * fa:6.1f}% {100 * fb:6.1f}% {lift:6.2f}")
        mb = sum(dterr_best[phase]) / len(dterr_best[phase])
        ma = sum(dterr_all[phase]) / len(dterr_all[phase])
        print(f"   mean Δterritory: chosen moves {mb:+.2f}, all moves {ma:+.2f}")


if __name__ == "__main__":
    main()
