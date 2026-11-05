"""
root_surprise.py ROOTS.csv — what the root moves of one depth tell about the
next one, from the root log of search_bench (run --rootlog).

    ./search_bench run --player CFG --positions ../test/bench_positions.txt --depth 12 --rootlog roots.csv
    python3 nnue/root_surprise.py roots.csv [--min-depth 5]

Each move of a completed depth has a fail-soft value: exact if it was searched
inside the window it got (alpha < v < beta), an upper bound if v ≤ alpha. All
changes are counted in sigma, the mean |v_d − v_{d−1}| of the adaptive
aspiration windows at that depth.

  1. Bounds: how many moves per depth have an exact value.
  2. Best move changes from d − 1 to d: the rank of the new best move at
     d − 1, and how well each predictor known at d − 1 would have placed it
     among the other moves:
       nodes  the size of its subtree (the current root order)
       gap    (v_best − v_move) / sigma, the closer the better
       rise   (v_{d−1} − v_{d−2}) / sigma of the move, the higher the better
       rise2  (v_{d−1} − v_{d−3}) / sigma (same parity: against the odd/even swing)
       nodes+gap  the sum of the two ranks
  3. Reductions: the new best move at d, and at d − 1 was it rejected by the
     reduced search alone (stage 1) — against the share of such moves.
  4. Surprise of the depth, s = |v_d − v_{d−1}| / sigma: the chance that the
     best move changes at d + 1 and that the first attempt of d + 1 fails.
"""
import csv
import sys
from collections import defaultdict

WIN = 30000


def read(path):
    pos = defaultdict(dict)   # pos → depth → {move: row}
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {k: (r[k] if k == "move" else float(r[k]) if k == "sigma" else int(r[k])) for k in r}
            pos[row["pos"]].setdefault(row["depth"], {})[row["move"]] = row
    return pos


def bound(r):
    return "U" if r["value"] <= r["alpha"] else "L" if r["value"] >= r["beta"] else "E"


def best_of(moves):
    return max(moves.values(), key=lambda r: (r["value"], -r["rank"]))["move"]


def rank_of(target, cands, key, high_first):
    """1-based rank of target among cands by key (ties: the worst place)."""
    t = key(target)
    better = sum(1 for c in cands if (key(c) > t if high_first else key(c) < t))
    ties = sum(1 for c in cands if key(c) == t)
    return better + ties


def quantiles(xs, qs=(0.5, 0.75, 0.9, 0.99)):
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs] if xs else [float("nan")] * len(qs)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    min_depth = int(args[args.index("--min-depth") + 1]) if "--min-depth" in args else 5
    pos = read(args[0])

    nb_bounds = defaultdict(int)
    events = []                       # (pos, d, new best)
    pred = defaultdict(list)          # predictor → ranks of the new best
    first_rank = []                   # rank of the new best at d − 1
    nb_cands = []
    red_new = [0, 0]                  # new best at d: reduced at d, total
    stage1 = [0, 0]                   # new best at d: stage 1 at d − 1, total
    stage1_all = [0, 0]               # all non-best moves at d − 1: stage 1, total
    old_best_drop = []                # the previous best's drop when it lost (≥, upper bound)
    surprise = []                     # (s at d, change at d + 1, first attempt of d + 1 failed)
    same_best_delta = []              # |v_d − v_{d−1}| / sigma when the best move stays

    for p, depths in pos.items():
        for d in sorted(depths):
            cur = depths[d]
            for r in cur.values():
                nb_bounds[bound(r)] += 1
            prev = depths.get(d - 1)
            if d < min_depth or prev is None:
                continue
            iv, ivp = next(iter(cur.values()))["iter_value"], next(iter(prev.values()))["iter_value"]
            sigma = next(iter(cur.values()))["sigma"]
            if abs(iv) >= WIN or abs(ivp) >= WIN or sigma <= 0:
                continue
            b, bp = best_of(cur), best_of(prev)
            s = abs(iv - ivp) / sigma
            nxt = depths.get(d + 1)
            if nxt is not None and abs(next(iter(nxt.values()))["iter_value"]) < WIN:
                surprise.append((s, best_of(nxt) != b, next(iter(nxt.values()))["attempts"] > 1))
            for m, r in prev.items():
                if m != bp:
                    stage1_all[1] += 1
                    stage1_all[0] += r["reduction"] > 0 and r["stage"] == 1
            if b == bp:
                same_best_delta.append(s)
                continue
            if b not in prev:
                continue
            events.append((p, d, b))
            if bp in cur:
                old_best_drop.append((ivp - cur[bp]["value"]) / sigma)
            red_new[1] += 1
            red_new[0] += cur[b]["reduction"] > 0
            stage1[1] += 1
            stage1[0] += prev[b]["reduction"] > 0 and prev[b]["stage"] == 1
            first_rank.append(prev[b]["rank"])
            pp, ppp = depths.get(d - 2, {}), depths.get(d - 3, {})
            cands = [r for m, r in prev.items() if m != bp]
            nb_cands.append(len(cands))
            tgt = prev[b]
            pred["nodes"].append(rank_of(tgt, cands, lambda r: r["nodes"], True))
            pred["gap"].append(rank_of(tgt, cands, lambda r: ivp - r["value"], False))
            rn = {r["move"]: rank_of(r, cands, lambda q: q["nodes"], True) for r in cands}
            rg = {r["move"]: rank_of(r, cands, lambda q: ivp - q["value"], False) for r in cands}
            pred["nodes+gap"].append(rank_of(tgt, cands, lambda r: rn[r["move"]] + rg[r["move"]], False))
            if all(m in pp for m in (r["move"] for r in cands)):
                rise = lambda r: (r["value"] - pp[r["move"]]["value"]) / sigma
                pred["rise"].append(rank_of(tgt, cands, rise, True))
            if all(m in ppp for m in (r["move"] for r in cands)):
                rise2 = lambda r: (r["value"] - ppp[r["move"]]["value"]) / sigma
                pred["rise2"].append(rank_of(tgt, cands, rise2, True))

    tot = sum(nb_bounds.values())
    print(f"root moves logged: {tot}; exact {nb_bounds['E']} ({100 * nb_bounds['E'] / tot:.1f} %), "
          f"upper bound {nb_bounds['U']} ({100 * nb_bounds['U'] / tot:.1f} %), lower bound {nb_bounds['L']}")
    n_iter = len(same_best_delta) + len(events)
    print(f"\ndepths ≥ {min_depth} with a depth before (no game result): {n_iter}; the best move changes: "
          f"{len(events)} ({100 * len(events) / max(1, n_iter):.1f} %)")
    if events:
        print(f"  candidates per change: mean {sum(nb_cands) / len(nb_cands):.1f}")
        fr = defaultdict(int)
        for r in first_rank:
            fr[min(r, 6)] += 1
        print("  rank of the new best in the search order of d − 1 (0 = searched first): "
              + ", ".join(f"{k if k < 6 else '6+'}: {v}" for k, v in sorted(fr.items())))
        print(f"\n  {'predictor':<8} {'n':>5} {'top 1':>7} {'top 3':>7} {'mean rank':>10} {'median':>7}")
        for k in ("nodes", "gap", "nodes+gap", "rise", "rise2"):
            xs = pred[k]
            if not xs:
                continue
            print(f"  {k:<8} {len(xs):>5} {100 * sum(x == 1 for x in xs) / len(xs):>6.1f}% "
                  f"{100 * sum(x <= 3 for x in xs) / len(xs):>6.1f}% {sum(xs) / len(xs):>10.2f} {sorted(xs)[len(xs) // 2]:>7}")
        q = quantiles(old_best_drop)
        print(f"\n  drop of the previous best when it loses, (v_{{d−1}} − v_d) / sigma, at least: "
              f"median {q[0]:.2f}, p75 {q[1]:.2f}, p90 {q[2]:.2f}")
        print(f"  new best at d searched with a reduction at d: {red_new[0]} / {red_new[1]}")
        print(f"  new best at d rejected by the reduced search alone at d − 1: {stage1[0]} / {stage1[1]} "
              f"({100 * stage1[0] / max(1, stage1[1]):.1f} %), all non-best moves: "
              f"{100 * stage1_all[0] / max(1, stage1_all[1]):.1f} %")
    q = quantiles(same_best_delta)
    print(f"\nsurprise s = |v_d − v_{{d−1}}| / sigma when the best move stays: median {q[0]:.2f}, p75 {q[1]:.2f}, "
          f"p90 {q[2]:.2f}, p99 {q[3]:.2f}")
    print(f"\n  {'s at d':<10} {'n':>5} {'best changes at d+1':>20} {'d+1 fails first':>16}")
    edges = [0, 0.5, 1, 2, 4, float("inf")]
    for lo, hi in zip(edges, edges[1:]):
        xs = [x for x in surprise if lo <= x[0] < hi]
        if xs:
            print(f"  {f'[{lo}, {hi})':<10} {len(xs):>5} {100 * sum(x[1] for x in xs) / len(xs):>19.1f}% "
                  f"{100 * sum(x[2] for x in xs) / len(xs):>15.1f}%")


if __name__ == "__main__":
    main()
