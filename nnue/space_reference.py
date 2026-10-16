"""
space_reference.py — every candidate feature of nnue/space_features.h, written
again square by square in plain Python (no bitboards, no shared code), to
check the C++ and to draw the figures of doc/nnue_roadmap.qmd.

    ./space_features dump ../data/games dump.bin --max-positions 20000
    python3 space_reference.py dump.bin          # compares the 92 columns

A position is (us, them, free): three sets of squares 0..63, square = 8 · rank
+ file. `compute_all(us, them, free)` returns the 92 columns of
space::compute_all, in the same order and with the same multipliers.
"""
import sys
import numpy as np

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def rc(sq):
    return divmod(sq, 8)


def around(sq):
    """The (up to 8) neighbours of a square."""
    r, c = rc(sq)
    return {8 * (r + dr) + c + dc for dr, dc in DIRS if 0 <= r + dr < 8 and 0 <= c + dc < 8}


def around_set(squares):
    return set().union(*(around(s) for s in squares)) if squares else set()


def line(sq, d, free):
    """The free squares a queen on sq reaches in direction d."""
    (r, c), out = rc(sq), []
    while True:
        r, c = r + d[0], c + d[1]
        if not (0 <= r < 8 and 0 <= c < 8) or 8 * r + c not in free:
            return out
        out.append(8 * r + c)


def queen_moves(sq, free):
    return {s for d in DIRS for s in line(sq, d, free)}


def queen_levels(pieces, free):
    """levels[k] = free squares first reached after k + 1 queen moves."""
    levels, seen, front = [], set(), set(pieces)
    while True:
        nxt = set().union(*(queen_moves(s, free) for s in front)) - seen if front else set()
        if not nxt:
            return levels
        levels.append(nxt)
        seen |= nxt
        front = nxt


def king_region(pieces, free):
    """Free squares reached by king steps through free squares, by distance."""
    levels, seen, front = [], set(), set(pieces)
    while True:
        nxt = around_set(front) & free - seen
        if not nxt:
            return levels
        levels.append(nxt)
        seen |= nxt
        front = nxt


def blocks(free):
    """Connected components of the free squares (king steps)."""
    out, rest = [], set(free)
    while rest:
        s = rest.pop()
        comp = {s} | set().union(*king_region({s}, free))
        rest -= comp
        out.append(comp)
    return out


def corrected_flood_fill(pieces, free):
    """Each piece enters ONE block touching it: the largest total."""
    bl = blocks(free)
    options = [[i for i, b in enumerate(bl) if around(p) & b] for p in pieces]
    best = 0

    def rec(k, taken):
        nonlocal best
        if k == len(options):
            best = max(best, sum(len(bl[i]) for i in taken))
            return
        rec(k + 1, taken)
        for i in options[k]:
            if i not in taken:
                rec(k + 1, taken | {i})
    rec(0, frozenset())
    return best


def first_level(levels, squares):
    return next((k for k, lv in enumerate(levels) if lv & squares), 99)


def zones9():
    cut = [0, 3, 5, 8]
    return [{8 * r + c for r in range(cut[zr], cut[zr + 1]) for c in range(cut[zc], cut[zc + 1])}
            for zr in range(3) for zc in range(3)]


def compute16(us, them, free):
    out = [0] * 16
    lv = [queen_levels(us, free), queen_levels(them, free)]
    dist = [{s: k for k, l in enumerate(v) for s in l} for v in lv]
    for sd, pcs in enumerate((us, them)):
        o = 8 * sd
        out[o + 0] = corrected_flood_fill(pcs, free)
        for k in range(3):                                   # Q1, Q2, Q3
            out[o + 1 + k] = sum(1 for d in dist[sd].values() if d <= k)
        out[o + 4] = sum(1 for s, d in dist[sd].items() if d < dist[1 - sd].get(s, 99))   # QFIRST
        out[o + 5] = sum(len(l) << (5 - k) for k, l in enumerate(lv[sd][:6])) >> 5          # QW
        out[o + 6] = sum(len(z & free) for z in zones9()                                    # LATE9
                         if z & free and first_level(lv[1 - sd], z & free) < first_level(lv[sd], z & free))
        out[o + 7] = sum(len(queen_moves(p, free)) for p in pcs)                            # MOB
    return [min(v, 64) for v in out]


def influence(us, them, free):
    """As heuristic::influence: both sides spread by king steps at once; squares
    reached together are neutral, and the neutral squares spread too."""
    own, front, neutral, prev = [set(us), set(them)], [set(us), set(them)], set(), None
    while prev != own:
        prev = [set(own[0]), set(own[1])]
        a, b = around_set(front[0]) & free - own[1], around_set(front[1]) & free - own[0]
        neutral |= (around_set(neutral) & free) | (a & b)
        front = [a - neutral, b - neutral]
        own[0] |= front[0]
        own[1] |= front[1]
    return len(own[0] & free), len(own[1] & free)


def inverse64(d):
    return (64 + d // 2) // d if d else 0


def connect_piece(p, opp, free):
    """block, 16·contact, 64/king, 64/go, 64/come."""
    reg = king_region({p}, free)
    seen = set().union(*reg)
    near_opp = around_set(opp)
    king = next((d + 1 for d, l in enumerate(reg) if l & near_opp), 0)
    contact = sum(1 for q in opp if around(q) & seen)
    go = next((k + 1 for k, l in enumerate(queen_levels({p}, free)) if l & near_opp), 0)
    come = next((k + 1 for k, l in enumerate(queen_levels(opp, free)) if l & around(p)), 0)
    return [len(seen), 16 * contact, inverse64(king), inverse64(go), inverse64(come)]


def encircle_piece(p, opp, free):
    """lib, opp, short, gene, contrib, att (dict)."""
    nb = around(p)
    reach1 = set().union(*(queen_moves(q, free) for q in opp))
    e = dict(lib=len(nb & free), opp=len(nb & opp), short=0, gene=0, contrib=0, att=len(nb & free & reach1))
    for d in DIRS:
        real, freed = line(p, d, free), line(p, d, free | opp)
        e["short"] += len(real) in (1, 2)
        e["gene"] += len(freed) - len(real)
        e["contrib"] += any(s in opp for s in freed)
    return e


def compute_all(us, them, free):
    out = compute16(us, them, free) + list(influence(us, them, free))
    conn, enc = [], []
    for mine, opp in ((us, them), (them, us)):
        rows = sorted((connect_piece(p, opp, free) for p in mine), key=lambda x: (x[1], x[0], x[2], x[3], x[4]))
        conn += [v for r in rows for v in r]
        es = [encircle_piece(p, opp, free) for p in mine]
        sums = [sum(e["gene"] for e in es), 4 * sum(e["contrib"] for e in es), 4 * sum(e["opp"] for e in es),
                2 * sum(e["att"] for e in es), 2 * sum(e["lib"] for e in es), 2 * sum(e["short"] for e in es),
                16 * sum(e["lib"] <= 1 for e in es), 16 * sum(e["lib"] == 0 for e in es),
                16 * sum(e["lib"] == 1 and e["att"] > 0 for e in es)]
        es.sort(key=lambda e: (e["lib"], -e["gene"], -e["att"]))
        enc += sums + [v for e in es for v in (8 * e["att"], 64 * (e["lib"] == 1 and e["att"] > 0))]
    return [min(max(v, 0), 64) for v in out + conn + enc]


def squares(bb):
    return {i for i in range(64) if bb >> i & 1}


def main():
    rec = np.dtype([("bb", "<u8", 3), ("f", "u1", 92)])
    data = np.fromfile(sys.argv[1], rec)
    bad = np.zeros(92, np.int64)
    for i, r in enumerate(data):
        us, them, free = (squares(int(b)) for b in r["bb"])
        ref = np.array(compute_all(us, them, free))
        diff = ref != r["f"]
        if diff.any() and bad.sum() == 0:
            print(f"first difference, position {i}: columns {np.where(diff)[0].tolist()}\n"
                  f"  C++ {r['f'][diff].tolist()}\n  ref {ref[diff].tolist()}")
        bad += diff
    print(f"{len(data):,} positions, {int((bad > 0).sum())} columns differ" +
          (f": {dict((int(c), int(bad[c])) for c in np.where(bad)[0])}" if bad.any() else " — C++ = reference"))


if __name__ == "__main__":
    main()
