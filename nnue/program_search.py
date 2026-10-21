"""
program_search.py — search, in a restricted space of bitboard programs, for cheap
position features (like the influence) that add to what the NNUE already predicts.

    python3 program_search.py POSITIONS.bin OUT_DIR [--n 400000] [--minutes 60] [--workers 8]
                                                    [--base infl,qfirst] [--max-size 6]

POSITIONS.bin: test/program_search_dump (side-to-move view, base network value,
points still to score, result, game number).

The language. A program computes a set of squares (a 64-bit bitboard) from
    M  my pieces      H  his pieces      F  the free squares      X  the holes
with
    N(s)   the 8 neighbours of s                       (8 shifts)
    S(s)   N(s) & F: one king step through free squares
    Q(s)   one queen move from s through free squares  (occluded fill, 8 directions)
    C(s)   the free squares connected to s by king steps (S repeated until it stops)
    a & b, a | b, a - b (and not)
and its feature is the number of squares of the set. Two loop templates add
iteration with a state (at most 12 rounds, stopping when nothing changes):
    GROW[g]   A = M; repeat: A |= g(A)                          (g may read A, M, H, F, X)
    RACE[g]   A = M, B = H; repeat: a = g(A, B) - A, b = g'(B, A) - B,
              squares claimed by both become neutral for good; A |= a, B |= b
              (g' is g with the sides swapped)
with two outputs each: |A & F| ("#") and the squares weighted by the round they
were added in, 32 · 2^-round ("w").
Every program gives two columns: computed for me, then with the sides swapped
(M <-> H) for him, as all the network's features. All operations treat the 8
directions alike, so every feature is mirror invariant.

The score. Target = what the base network gets wrong:
    residual = target − cubic(base value), both centred per (free squares, colour),
the cubic fitted per phase on the train games (half of the games, by game number).
A program's gain = the share of the target's variance (test games) that a
linear fit of its 2 columns (per phase, centred the same way) removes from the
residual, in R² points, for the two targets of section 5.1.5 (points still to
score, result) and four phases (40–56, 20–39, 10–19, 0–9 free squares). The
score is the weighted mean of the 8 gains (--weights, default points 1, result 10:
a result gain is about ten times smaller for the same information, section 5.1.5). --base: known features added to the base
first (infl, qfirst, mob), to look for what they do not already give.

The search. (1) Bottom-up enumeration by size, with observational equivalence:
two programs giving the same columns on 2000 witness positions are the same
program, only the cheapest is kept. (2) Then evolution: mutations and
crossovers of the best programs until the time is up. Every program that enters
the top 10 is printed at once; OUT_DIR/leaderboard.txt (the best 50 and the
gain / cost Pareto front) is rewritten every minute; OUT_DIR/all.tsv keeps
every scored program.

Cost: number of 64-bit operations (shifts count one each), loops multiplied by
their mean number of rounds on the witness positions. Indicative only, the C++
translation will be timed.
"""
import argparse
import hashlib
import heapq
import itertools
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

U = np.uint64
FULL = U(0xFFFFFFFFFFFFFFFF)
FA = U(0x0101010101010101)
FH = U(0x8080808080808080)
NOT_A, NOT_H = ~FA, ~FH
# the 8 directions: (shift, mask after the shift)
DIRS = [(8, FULL), (-8, FULL), (1, NOT_A), (-1, NOT_H), (9, NOT_A), (7, NOT_H), (-7, NOT_A), (-9, NOT_H)]
PHASES = [(40, 56, "40-56"), (20, 39, "20-39"), (10, 19, "10-19"), (0, 9, "0-9")]
TARGETS = ["points", "result"]
REC = np.dtype([('mine', '<u8'), ('his', '<u8'), ('holes', '<u8'), ('value', '<f4'), ('future', 'i1'),
                ('result', 'i1'), ('free', 'u1'), ('colour', 'u1'), ('game', '<u4')])
MAX_ROUNDS = 12
WEIGHTS = np.array([1.0, 10.0])      # points, result
MAX_NODES = 15                       # evolution: programs larger than this are dropped


# ── bitboard primitives (numpy arrays of uint64) ─────────────────────────────
def sh(x, d, m):
    return ((x << U(d)) if d > 0 else (x >> U(-d))) & m


def near(x):
    out = np.zeros_like(x)
    for d, m in DIRS:
        out |= sh(x, d, m)
    return out


def qreach(x, free):
    out = np.zeros_like(x)
    for d, m in DIRS:
        gen, pro = x.copy(), free.copy()
        gen |= pro & sh(gen, d, m)
        pro &= sh(pro, d, m)
        gen |= pro & sh(sh(gen, d, m), d, m)
        pro &= sh(sh(pro, d, m), d, m)
        gen |= pro & sh(sh(sh(sh(gen, d, m), d, m), d, m), d, m)
        out |= sh(gen, d, m) & free
    return out


def closure(x, free):
    c = x & free
    c |= near(x) & free
    while True:
        n = c | (near(c) & free)
        if np.array_equal(n, c):
            return c
        c = n


def popcount(x):
    # bytes view + lookup table: 8 bytes per uint64
    return POP8[x.view(np.uint8)].reshape(-1, 8).sum(axis=1, dtype=np.int32)


POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)

# ── the language ─────────────────────────────────────────────────────────────
# expression = tuple: ('M',) ('H',) ('F',) ('X',) ('A',) ('B',)
#              ('N', e) ('S', e) ('Q', e) ('C', e) ('&', a, b) ('|', a, b) ('-', a, b)
# program    = ('POP', e) | ('GROW', g, out) | ('RACE', g, out), out in {'#', 'w'}
LEAVES = ['M', 'H', 'F', 'X']
UNARY = ['N', 'S', 'Q', 'C']
BINARY = ['&', '|', '-']
OP_COST = {'M': 0, 'H': 0, 'F': 0, 'X': 0, 'A': 0, 'B': 0, 'N': 24, 'S': 25, 'Q': 8 * 22, '&': 1, '|': 1, '-': 2}
SWAP = {'M': 'H', 'H': 'M', 'A': 'B', 'B': 'A'}


def swap_mh(e):
    """M <-> H only (GROW: its single state A is the side's own set)."""
    if len(e) == 1:
        return ({'M': 'H', 'H': 'M'}.get(e[0], e[0]),)
    return (e[0],) + tuple(swap_mh(c) for c in e[1:])


def swap(e):
    if len(e) == 1:
        return (SWAP.get(e[0], e[0]),)
    return (e[0],) + tuple(swap(c) for c in e[1:])


def nodes(e):
    """Number of nodes of a program or an expression."""
    if e[0] in ('POP', 'GROW', 'RACE'):
        return nodes(e[1]) + (0 if e[0] == 'POP' else 1)
    return 1 + sum(nodes(c) for c in e[1:])


def show(e):
    if e[0] == 'POP':
        return f"|{show(e[1])}|"
    if e[0] in ('GROW', 'RACE'):
        return f"{e[0]}{e[2]}[{show(e[1])}]"
    if len(e) == 1:
        return e[0]
    if len(e) == 2:
        return f"{e[0]}({show(e[1])})"
    a, b = (show(c) if len(c) <= 2 else f"({show(c)})" for c in e[1:])
    return f"{a} {e[0]} {b}"


def evaluate(e, env):
    """Set expression → uint64 array. env: leaves (and A, B in loop bodies), 'cache'."""
    if len(e) == 1:
        return env[e[0]]
    cache = env.get('cache')
    if cache is not None and e in cache:
        return cache[e]
    op = e[0]
    if op == 'N':
        v = near(evaluate(e[1], env))
    elif op == 'S':
        v = near(evaluate(e[1], env)) & env['F']
    elif op == 'Q':
        v = qreach(evaluate(e[1], env), env['F'])
    elif op == 'C':
        v = closure(evaluate(e[1], env), env['F'])
    else:
        a, b = evaluate(e[1], env), evaluate(e[2], env)
        v = a & b if op == '&' else a | b if op == '|' else a & ~b
    if cache is not None and not uses_state(e):
        cache[e] = v
    return v


def uses_state(e):
    return e[0] in ('A', 'B') or any(uses_state(c) for c in e[1:] if isinstance(c, tuple))


def expr_cost(e):
    if len(e) == 1:
        return 0
    own = OP_COST.get(e[0], 0)
    if e[0] == 'C':
        own = 26 * 4          # S repeated, ~4 rounds
    return own + sum(expr_cost(c) for c in e[1:])


def run_program(p, env):
    """→ (column for me, column for him, mean rounds)."""
    if p[0] == 'POP':
        e = p[1]
        return popcount(evaluate(e, env)), popcount(evaluate(swap(e), env)), 1.0
    g, out = p[1], p[2]
    F = env['F']
    n = len(F)
    w_me = np.zeros(n, np.int32)
    w_him = np.zeros(n, np.int32)
    rounds = 0
    if p[0] == 'GROW':
        sides = [(env['M'], g, w_me), (env['H'], swap_mh(g), w_him)]   # A stays the growing set
        res = []
        for start, body, w in sides:
            A = start.copy()
            local = dict(env, cache=None)
            for k in range(MAX_ROUNDS):
                local['A'] = A
                new = evaluate(body, local) & ~A
                if not new.any():
                    break
                rounds += 0.5 * float((new != 0).mean())          # mean rounds per position
                if out == 'w':
                    w += popcount(new & F) << (5 - k) if k <= 5 else 0
                A = A | new
            res.append(popcount(A & F) if out == '#' else w)
        return res[0], res[1], max(rounds, 1.0)
    # RACE
    A, B = env['M'].copy(), env['H'].copy()
    neutral = np.zeros_like(A)
    local = dict(env, cache=None)
    gs = swap(g)
    for k in range(MAX_ROUNDS):
        local['A'], local['B'] = A, B
        a = evaluate(g, local) & ~A & ~neutral
        b = evaluate(gs, local) & ~B & ~neutral       # gs: g with A <-> B and M <-> H
        both = a & b
        neutral |= both
        a &= ~both
        b &= ~both
        if not (a.any() or b.any()):
            break
        rounds += float(((a | b) != 0).mean())                 # mean rounds per position
        if out == 'w' and k <= 5:
            w_me += popcount(a & F) << (5 - k)
            w_him += popcount(b & F) << (5 - k)
        A |= a
        B |= b
    if out == '#':
        return popcount(A & F), popcount(B & F), max(rounds, 1.0)
    return w_me, w_him, max(rounds, 1.0)


def program_cost(p, rounds):
    if p[0] == 'POP':
        return 2 * (expr_cost(p[1]) + 4)
    body = expr_cost(p[1]) + 6
    return int(2 * body * rounds + 8)


# ── reference features (to compare, or to add to the base) ───────────────────
def ref_influence(env):
    M, H, F = env['M'], env['H'], env['F']
    own, front = [M.copy(), H.copy()], [M.copy(), H.copy()]
    neutral = np.zeros_like(M)
    prev = [np.zeros_like(M), np.zeros_like(M)]
    while not (np.array_equal(prev[0], own[0]) and np.array_equal(prev[1], own[1])):
        front[0] = near(front[0]) & F & ~own[1]
        front[1] = near(front[1]) & F & ~own[0]
        neutral |= (near(neutral) & F) | (front[0] & front[1])
        front[0] &= ~neutral
        front[1] &= ~neutral
        prev = [own[0].copy(), own[1].copy()]
        own[0] |= front[0]
        own[1] |= front[1]
    return popcount(own[0] & F), popcount(own[1] & F)


def ref_qfirst(env):
    M, H, F = env['M'], env['H'], env['F']
    seen = [np.zeros_like(M), np.zeros_like(M)]
    front = [M, H]
    first = [np.zeros(len(M), np.int32), np.zeros(len(M), np.int32)]
    for _ in range(12):
        lv = [qreach(front[0], F) & ~seen[0], qreach(front[1], F) & ~seen[1]]
        first[0] += popcount(lv[0] & ~lv[1] & ~seen[1])
        first[1] += popcount(lv[1] & ~lv[0] & ~seen[0])
        seen = [seen[0] | lv[0], seen[1] | lv[1]]
        front = lv
        if not (lv[0].any() or lv[1].any()):
            break
    return first[0], first[1]


def ref_mob(env):
    M, H, F = env['M'], env['H'], env['F']
    out = []
    for P in (M, H):
        tot = np.zeros(len(P), np.int32)
        rest = P.copy()
        for _ in range(4):
            low = rest & (~rest + U(1))
            tot += popcount(qreach(low, F))
            rest &= ~low
        out.append(tot)
    return out[0], out[1]


REFS = {'infl': ref_influence, 'qfirst': ref_qfirst, 'mob': ref_mob}


# ── the score ────────────────────────────────────────────────────────────────
class Scorer:
    def __init__(self, r, base_refs):
        self.n = len(r)
        self.free = r['free'].astype(np.int64)
        self.group = self.free * 2 + r['colour']
        self.test = (r['game'] * np.uint32(2654435761) >> np.uint32(16)) & np.uint32(1) == 1
        self.phase = np.full(self.n, -1)
        for i, (lo, hi, _) in enumerate(PHASES):
            self.phase[(self.free >= lo) & (self.free <= hi)] = i
        self.env = env_of(r)
        v = r['value'].astype(np.float64)
        base_cols = [v, v * v, v ** 3]
        for name in base_refs:
            base_cols += [c.astype(np.float64) for c in REFS[name](self.env)]
        self.base = np.stack([self.center(c) for c in base_cols], 1)
        self.resid, self.sst = {}, {}
        for t in TARGETS:
            y = self.center(r['future' if t == 'points' else 'result'].astype(np.float64))
            res = np.empty(self.n)
            sst = np.zeros(len(PHASES))
            for ph in range(len(PHASES)):
                tr, te = (self.phase == ph) & ~self.test, (self.phase == ph) & self.test
                X = self.base[tr]
                coef = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ y[tr])
                res[self.phase == ph] = y[self.phase == ph] - self.base[self.phase == ph] @ coef
                sst[ph] = (y[te] ** 2).sum()
            self.resid[t], self.sst[t] = res, sst

    def center(self, x):
        tr = ~self.test
        s = np.bincount(self.group[tr], x[tr], minlength=130)
        c = np.bincount(self.group[tr], minlength=130)
        return x - (s / np.maximum(c, 1))[self.group]

    def gains(self, me, him):
        """→ array (targets × phases) of R² points gained on the test games."""
        X = np.stack([self.center(me.astype(np.float64)), self.center(him.astype(np.float64))], 1)
        out = np.zeros((len(TARGETS), len(PHASES)))
        for ph in range(len(PHASES)):
            tr, te = (self.phase == ph) & ~self.test, (self.phase == ph) & self.test
            Xtr, Xte = X[tr], X[te]
            A = Xtr.T @ Xtr + 1e-6 * np.eye(2)
            for i, t in enumerate(TARGETS):
                coef = np.linalg.solve(A, Xtr.T @ self.resid[t][tr])
                r = self.resid[t][te]
                out[i, ph] = 100 * ((r ** 2).sum() - ((r - Xte @ coef) ** 2).sum()) / self.sst[t][ph]
        return out


def env_of(r):
    M, H, X = r['mine'].copy(), r['his'].copy(), r['holes'].copy()
    return {'M': M, 'H': H, 'X': X, 'F': ~(M | H | X)}


# ── enumeration ──────────────────────────────────────────────────────────────
def grow_exprs(leaves, max_size):
    """Bottom-up set expressions by size (no equivalence here: see Search.key)."""
    by_size = {1: [(l,) for l in leaves]}
    for s in range(2, max_size + 1):
        cur = []
        for e in by_size[s - 1]:
            for u in UNARY:
                if u in ('S', 'N', 'C') and e[0] == 'C':
                    continue                                    # C absorbs these
                cur.append((u, e))
        for sa in range(1, s - 1):
            sb = s - 1 - sa
            for a in by_size[sa]:
                for b in by_size[sb]:
                    for op in BINARY:
                        if op in '&|' and (sa, a) > (sb, b):
                            continue                            # commutative
                        if a == b:
                            continue
                        cur.append((op, a, b))
        by_size[s] = cur
    return by_size


# worker globals (fork)
W_ENV = W_SCORER = W_WITNESS = None


def init_worker(env_w, scorer):
    global W_WITNESS, W_SCORER
    W_WITNESS, W_SCORER = env_w, scorer


def witness_key(p):
    """Columns on the witness positions → key (None if constant or failing).
    The network sees the columns centred per number of free squares, through a
    linear layer: two programs whose columns differ only by a function of the
    free squares (|S | X| = |S| + 56 − free), by sign or by scale (|F − S| =
    free − |S|) are the same feature. The key is taken after removing these."""
    try:
        me, him, rounds = run_program(p, dict(W_WITNESS, cache={}))
    except Exception:
        return None, 0
    free = popcount(W_WITNESS['F'])
    cols = []
    for c in (me, him):
        c = c.astype(np.float64)
        c = c - (np.bincount(free, c, 65) / np.maximum(np.bincount(free, minlength=65), 1))[free]
        cols.append(c)
    both = np.concatenate(cols)
    m = np.abs(both).max()
    if m < 1e-9:
        return None, 0
    both = both / m
    first = both[np.argmax(np.abs(both) > 1e-6)]
    both = np.round(both * np.sign(first), 6) + 0.0
    h = hashlib.blake2b(both.tobytes(), digest_size=12).digest()
    return h, rounds


def full_score(p):
    me, him, _ = run_program(p, dict(W_SCORER.env, cache={}))
    return W_SCORER.gains(me, him)


class Search:
    def __init__(self, scorer, witness_env, out_dir, workers, minutes):
        self.scorer, self.out_dir = scorer, out_dir
        self.seen = {}           # witness key → (cost, program)
        self.scored = []         # (score, cost, program, gains)
        self.top = []
        self.t0 = time.time()
        self.deadline = self.t0 + 60 * minutes
        self.last_board = 0
        self.pool = Pool(workers, initializer=init_worker, initargs=(witness_env, scorer))
        self.tsv = open(os.path.join(out_dir, "all.tsv"), "w")
        self.tsv.write("score\tcost\t" + "\t".join(f"{t}_{p[2]}" for t in TARGETS for p in PHASES) + "\tprogram\n")

    def time_left(self):
        return time.time() < self.deadline

    def submit(self, programs):
        """Dedupe on the witnesses, then score the new ones."""
        keys = self.pool.map(witness_key, programs, chunksize=16)
        fresh = []
        for p, (k, rounds) in zip(programs, keys):
            if k is None:
                continue
            c = program_cost(p, rounds)
            old = self.seen.get(k)
            if old is None:
                self.seen[k] = (c, p)
                fresh.append((p, c))
            elif c < old[0]:
                self.seen[k] = (c, p)          # same columns, cheaper: replace (not rescored)
                for i, s in enumerate(self.scored):
                    if s[2] == old[1]:
                        self.scored[i] = (s[0], c, p, s[3])
        if not fresh:
            return
        gains = self.pool.map(full_score, [p for p, _ in fresh], chunksize=4)
        for (p, c), g in zip(fresh, gains):
            score = float((g.mean(axis=1) * WEIGHTS).sum() / WEIGHTS.sum())
            self.scored.append((score, c, p, g))
            self.tsv.write(f"{score:.4f}\t{c}\t" + "\t".join(f"{x:.3f}" for x in g.ravel()) + f"\t{show(p)}\n")
            self.maybe_print(score, c, p, g)
        self.tsv.flush()
        if time.time() - self.last_board > 60:
            self.write_board()

    def maybe_print(self, score, c, p, g):
        """Print a program that is the new best, or that no cheaper program beats (Pareto front)."""
        better_cheaper = any(s[1] <= c and s[0] >= score and s[2] != p for s in self.scored)
        best = score >= max(s[0] for s in self.scored)
        if best or not better_cheaper:
            el = time.time() - self.t0
            tag = "MEILLEUR" if best else "front   "
            print(f"[{el / 60:6.1f} min] {tag} score {score:6.3f}  coût {c:5d}  "
                  f"points {' '.join(f'{x:5.2f}' for x in g[0])} | résultat {' '.join(f'{x:5.2f}' for x in g[1])}"
                  f"   {show(p)}", flush=True)

    def write_board(self):
        self.last_board = time.time()
        best = sorted(self.scored, key=lambda s: -s[0])[:50]
        lines = [f"programs scored: {len(self.scored)}, distinct on the witnesses: {len(self.seen)}, "
                 f"{(time.time() - self.t0) / 60:.1f} min",
                 "gains in R² points over the base, phases " + " ".join(p[2] for p in PHASES), ""]
        hdr = f"{'score':>6s} {'coût':>5s}  {'points':^27s} | {'résultat':^27s}  programme"
        lines += ["BEST 50", hdr]
        for s, c, p, g in best:
            lines.append(f"{s:6.3f} {c:5d}  {' '.join(f'{x:6.2f}' for x in g[0])} | "
                         f"{' '.join(f'{x:6.2f}' for x in g[1])}  {show(p)}")
        lines += ["", "PARETO FRONT (no cheaper program scores more)", hdr]
        front, best_score = [], -1e9
        for s, c, p, g in sorted(self.scored, key=lambda s: (s[1], -s[0])):
            if s > best_score + 1e-9:
                best_score = s
                front.append((s, c, p, g))
        for s, c, p, g in front:
            lines.append(f"{s:6.3f} {c:5d}  {' '.join(f'{x:6.2f}' for x in g[0])} | "
                         f"{' '.join(f'{x:6.2f}' for x in g[1])}  {show(p)}")
        with open(os.path.join(self.out_dir, "leaderboard.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # ── evolution ──
    def random_expr(self, leaves, depth):
        if depth == 0 or random.random() < 0.3:
            return (random.choice(leaves),)
        if random.random() < 0.5:
            return (random.choice(UNARY), self.random_expr(leaves, depth - 1))
        return (random.choice(BINARY), self.random_expr(leaves, depth - 1), self.random_expr(leaves, depth - 1))

    def mutate_expr(self, e, leaves):
        r = random.random()
        if r < 0.2:
            return self.random_expr(leaves, 2)
        if r < 0.35:
            return (random.choice(UNARY), e)
        if r < 0.55:
            return (random.choice(BINARY), e, self.random_expr(leaves, 2)) if random.random() < 0.5 else \
                (random.choice(BINARY), self.random_expr(leaves, 2), e)
        if len(e) == 1:
            return (random.choice(leaves),)
        if r < 0.65 and len(e) == 3:
            return (random.choice(BINARY), e[1], e[2])
        if r < 0.72:
            return e[random.randrange(1, len(e))]                  # drop the operator
        i = random.randrange(1, len(e))
        kids = list(e)
        kids[i] = self.mutate_expr(e[i], leaves)
        return tuple(kids)

    def mutate(self, p, others):
        if p[0] == 'POP':
            if random.random() < 0.1:
                return (random.choice(['GROW', 'RACE']), p[1] if not random.random() < 0.5 else ('S', ('A',)),
                        random.choice('#w'))
            if random.random() < 0.15 and others:
                o = random.choice(others)
                if o[0] == 'POP':
                    return ('POP', (random.choice(BINARY), p[1], o[1]))
            return ('POP', self.mutate_expr(p[1], LEAVES))
        leaves = LEAVES + (['A', 'B'] if p[0] == 'RACE' else ['A'])
        r = random.random()
        if r < 0.15:
            return (p[0], p[1], '#' if p[2] == 'w' else 'w')
        if r < 0.25:
            return ('RACE' if p[0] == 'GROW' else 'GROW', p[1] if p[0] == 'GROW' else ('S', ('A',)), p[2])
        g = self.mutate_expr(p[1], leaves)
        if not uses_state(g):
            g = ('&', g, ('S', ('A',)))
        return (p[0], g, p[2])

    def evolve(self, batch=256):
        while self.time_left():
            ranked = sorted(self.scored, key=lambda s: -s[0])
            elite = [s[2] for s in ranked if nodes(s[2]) <= MAX_NODES][:60]
            # also the cheap end of the Pareto front
            cheap = [s[2] for s in sorted(self.scored, key=lambda s: (s[1], -s[0]))[:200] if s[0] > 0][:20]
            parents = elite + cheap
            front, best_score = [], -1e9
            for sc, c, p, g in sorted(self.scored, key=lambda s: (s[1], -s[0])):
                if sc > best_score + 1e-9:
                    best_score = sc
                    front.append(p)
            parents = parents + front
            kids, tries = set(), 0
            while len(kids) < batch and tries < 50 * batch:
                tries += 1
                k = self.mutate(random.choice(parents), parents)
                if nodes(k) <= MAX_NODES:
                    kids.add(k)
            self.submit(list(kids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positions")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base", default="")
    ap.add_argument("--max-size", type=int, default=5, help="enumeration: set expressions up to this size")
    ap.add_argument("--body-size", type=int, default=4, help="enumeration: loop bodies up to this size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="1,10", help="score weights of the points and result gains")
    ap.add_argument("--max-nodes", type=int, default=15, help="evolution: largest program kept")
    a = ap.parse_args()
    random.seed(a.seed)
    global MAX_NODES
    WEIGHTS[:] = [float(x) for x in a.weights.split(",")]
    MAX_NODES = a.max_nodes
    os.makedirs(a.out_dir, exist_ok=True)
    r = np.fromfile(a.positions, dtype=REC)
    rng = np.random.default_rng(a.seed)
    r = r[rng.choice(len(r), min(a.n, len(r)), replace=False)]
    base = [b for b in a.base.split(",") if b]
    print(f"{len(r)} positions, base = NNUE value (cubic){' + ' + ', '.join(base) if base else ''}", flush=True)
    t = time.time()
    scorer = Scorer(r, base)
    print(f"scorer ready ({time.time() - t:.0f} s)", flush=True)
    witness_env = env_of(r[rng.choice(len(r), 2000, replace=False)])

    # references
    print("references (same score):")
    for name, f in REFS.items():
        me, him = f(scorer.env)
        g = scorer.gains(me, him)
        print(f"  {name:7s} score {(g.mean(axis=1) * WEIGHTS).sum() / WEIGHTS.sum():6.3f}  points {' '.join(f'{x:5.2f}' for x in g[0])} | "
              f"résultat {' '.join(f'{x:5.2f}' for x in g[1])}", flush=True)

    s = Search(scorer, witness_env, a.out_dir, a.workers, a.minutes)
    by_size = grow_exprs(LEAVES, a.max_size)
    print("enumeration: |set|", flush=True)
    for sz in range(1, a.max_size + 1):
        progs = [('POP', e) for e in by_size[sz]]
        for i in range(0, len(progs), 2048):
            if not s.time_left():
                break
            s.submit(progs[i:i + 2048])
        print(f"  size {sz}: {len(progs)} programs, {len(s.seen)} distinct so far", flush=True)
    body = grow_exprs(LEAVES + ['A', 'B'], a.body_size)
    print("enumeration: loops", flush=True)
    for sz in range(1, a.body_size + 1):
        progs = []
        for g in body[sz]:
            if not uses_state(g):
                continue
            for out in '#w':
                if 'B' not in str(g):
                    progs.append(('GROW', g, out))
                progs.append(('RACE', g, out))
        for i in range(0, len(progs), 1024):
            if not s.time_left():
                break
            s.submit(progs[i:i + 1024])
        print(f"  body size {sz}: {len(progs)} programs, {len(s.seen)} distinct so far", flush=True)
    print("evolution", flush=True)
    s.evolve()
    s.write_board()
    print("done", flush=True)


if __name__ == "__main__":
    main()
