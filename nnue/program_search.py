"""
program_search.py — search, in a restricted space of bitboard programs, for cheap
position features (like the influence) that add to what the NNUE already predicts.

    python3 program_search.py POSITIONS.bin OUT_DIR [--n 400000] [--minutes 60] [--workers 8] ...

POSITIONS.bin: test/program_search_dump (side-to-move view, base network value,
points still to score, result, game number).

THE LANGUAGE. A program computes a set of squares (a 64-bit bitboard) from
    M  my pieces      H  his pieces      F  the free squares      X  the holes
    L1, L2...  the sets of the features already found (OOPS stages, below)
with
    N(s)   the 8 neighbours of s
    S(s)   N(s) & F: one king step through free squares
    Q(s)   one queen move from s through free squares (occluded fill, 8 directions)
    C(s)   the free squares connected to s by king steps (S repeated until it stops)
    a & b, a | b, a - b (and not)
and its feature is the number of squares of the set. Two loop templates add
iteration with a state (at most 12 rounds, stopping when nothing changes):
    GROW[g]   A = M; repeat: A |= g(A)                          (g reads A, M, H, F, X, L…)
    RACE[g]   A = M, B = H; repeat: a = g(A, B) - A, b = g'(B, A) - B,
              squares claimed by both become neutral for good; A |= a, B |= b
              (g' is g with the sides swapped)
with two outputs each: |A & F| ("#") and the squares weighted by the round they
were added in, 32 · 2^-round ("w"). Every program gives two columns: for me,
then with the sides swapped for him. All operations treat the 8 directions
alike, so every feature is mirror invariant.

THE GAIN. Target = what the base network gets wrong:
    residual = target − cubic(base value) − linear(features already kept),
all centred per (free squares, colour), fitted per phase on the train games
(half of the games, by game number). A program's gain = the share of the
target's variance (test games) that a linear fit of its 2 columns removes from
the residual, in R² points, for the two targets of section 5.1.5 (points still
to score, result) and four phases (40–56, 20–39, 10–19, 0–9 free squares).
score = weighted mean of the 8 gains (--weights, default points 1, result 10:
a result gain is about ten times smaller for the same information).

THE COST, in ns per evaluation (both sides), predicted then measured:
  1. prices: the C++ time of each operation (N, S, Q, a round of C, &, |, -,
     popcount, a loop round, a call), fitted by non-negative least squares on
     the measured times of ~160 calibration programs (OUT_DIR/prices.json);
  2. loops and C: their mean number of rounds per position, measured on the
     witness positions;
  3. a computation graph, not a tree: the program for me and the one for him
     are merged and every distinct sub-expression is paid once (C(X), N(F)…
     are shared by the two sides; the parts of a loop body that do not depend
     on the state are computed once before the loop); what the engine already
     computes (the sets of the kept features and their sub-expressions, --have)
     is free: the cost is the MARGINAL cost;
  4. the programs of the Pareto front and the best by speed prior are
     translated into C++ functions (one line per graph node), compiled, timed
     on 20,000 positions and checked against the Python values, every
     --measure-minutes: the "mes." column.

THE SPEED PRIOR (Schmidhuber 2002: probability ∝ 2^-length / time):
    sp = score − λ · log2(ns) − μ · nodes          (--lam 0.03, --mu 0.002)
ranks the programs for the evolution's parents and the leaderboard.

OOPS STAGES (Schmidhuber's Optimal Ordered Problem Solver: reuse what was
found). Every --stage-minutes, the best program by speed prior (if its score is
at least --min-gain) is frozen: its set becomes a new leaf Lk of the language,
its columns join the base (the search then looks for what it does not give),
its sub-expressions become free, the best programs are rescored against the new
base. OUT_DIR/library.txt and OUT_DIR/features.h (the C++ of the kept
features) list the stages.

THE SEARCH. (1) Bottom-up enumeration by size, with observational equivalence:
two programs giving the same columns on 2000 witness positions (up to a
function of the free squares, sign and scale) are the same feature, only the
cheapest is kept. (2) Evolution: mutations and crossovers of the best programs
(speed prior, score) and of the Pareto front, until the time is up. Printed at
once: every new best score (MEILLEUR), new best speed prior (MEILLEUR SP), new
Pareto point (front). OUT_DIR/leaderboard.txt is rewritten every minute,
OUT_DIR/all.tsv keeps every scored program.
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from multiprocessing import Pool

import numpy as np

U = np.uint64
FULL = U(0xFFFFFFFFFFFFFFFF)
FA = U(0x0101010101010101)
FH = U(0x8080808080808080)
NOT_A, NOT_H = ~FA, ~FH
DIRS = [(8, FULL), (-8, FULL), (1, NOT_A), (-1, NOT_H), (9, NOT_A), (7, NOT_H), (-7, NOT_A), (-9, NOT_H)]
PHASES = [(40, 56, "40-56"), (20, 39, "20-39"), (10, 19, "10-19"), (0, 9, "0-9")]
TARGETS = ["points", "result"]
REC = np.dtype([('mine', '<u8'), ('his', '<u8'), ('holes', '<u8'), ('value', '<f4'), ('future', 'i1'),
                ('result', 'i1'), ('free', 'u1'), ('colour', 'u1'), ('game', '<u4')])
MAX_ROUNDS = 12
WEIGHTS = np.array([1.0, 10.0])
MAX_NODES = 15
LAM, MU = 0.03, 0.002
DEFAULT_PRICES = {'N': 2.5, 'S': 2.6, 'Q': 12.0, 'Cr': 3.0, 'bit': 0.4, 'pop': 0.8, 'round': 1.5, 'call': 1.0}
PRICE_KEYS = list(DEFAULT_PRICES)


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
    """→ (set, mean number of extra rounds per position)."""
    c = (x | near(x)) & free
    rounds = np.ones(len(x))
    while True:
        n = c | (near(c) & free)
        changed = n != c
        if not changed.any():
            return c, float(rounds.mean())
        rounds += changed
        c = n


POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def popcount(x):
    return POP8[x.view(np.uint8)].reshape(-1, 8).sum(axis=1, dtype=np.int32)


# ── the language ─────────────────────────────────────────────────────────────
# expression = tuple: ('M',) ('H',) ('F',) ('X',) ('A',) ('B',) ('L1',) ('L1s',)…
#              ('N', e) ('S', e) ('Q', e) ('C', e) ('&', a, b) ('|', a, b) ('-', a, b)
# program    = ('POP', e) | ('GROW', g, out) | ('RACE', g, out), out in {'#', 'w'}
LEAVES = ['M', 'H', 'F', 'X']
UNARY = ['N', 'S', 'Q', 'C']
BINARY = ['&', '|', '-']
LIBRARY = []            # kept programs (OOPS); leaf L{k+1} = set of LIBRARY[k] for me, L{k+1}s for him
HAVE = set()            # expressions the engine already computes (free)


def lib_leaves():
    return [f"L{k + 1}" for k in range(len(LIBRARY))]


def swap_leaf(name, ab):
    if name in ('M', 'H'):
        return 'H' if name == 'M' else 'M'
    if ab and name in ('A', 'B'):
        return 'B' if name == 'A' else 'A'
    if name.startswith('L'):
        return name[:-1] if name.endswith('s') else name + 's'
    return name


def swap(e, ab=True):
    """The expression for the other side (ab=False: GROW, whose state A stays)."""
    if len(e) == 1:
        return (swap_leaf(e[0], ab),)
    return (e[0],) + tuple(swap(c, ab) for c in e[1:])


def nodes(e):
    if e[0] in ('POP', 'GROW', 'RACE'):
        return nodes(e[1]) + (0 if e[0] == 'POP' else 1)
    return 1 + sum(nodes(c) for c in e[1:])


def uses_state(e):
    return e[0] in ('A', 'B') or any(uses_state(c) for c in e[1:] if isinstance(c, tuple))


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


def parse(text):
    """show() syntax → expression or program (for --have)."""
    text = text.replace(" ", "")
    pop = text.startswith("|") and text.endswith("|") and len(text) > 2
    if pop:
        text = text[1:-1]
    toks = re.findall(r"GROW[#w]|RACE[#w]|L\d+s?|[NSQC](?=\()|[MHFXAB]|[()&|\-\[\]]", text)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        pos[0] += 1
        return toks[pos[0] - 1]

    def term():
        t = take()
        if t == '(':
            e = expr()
            take()
            return e
        if t in UNARY and peek() == '(':
            take()
            e = expr()
            take()
            return (t, e)
        return (t,)

    def expr():
        e = term()
        while peek() in ('&', '|', '-'):
            op = take()
            e = (op, e, term())
        return e

    if toks[0].startswith(('GROW', 'RACE')):
        kind = take()
        take()
        return (kind[:4], expr(), kind[4])
    e = expr()
    return ('POP', e) if pop else e


# ── evaluation ───────────────────────────────────────────────────────────────
def evaluate(e, env):
    """Set expression → uint64 array. env: leaves, 'cache' (or None), 'stats' (C rounds)."""
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
        v, rounds = closure(evaluate(e[1], env), env['F'])
        stats = env.get('stats')
        if stats is not None:
            stats[e] = rounds
    else:
        a, b = evaluate(e[1], env), evaluate(e[2], env)
        v = a & b if op == '&' else a | b if op == '|' else a & ~b
    if cache is not None and not uses_state(e):
        cache[e] = v
    return v


def run_program(p, env):
    """→ (column for me, column for him, info). info: mean number of times the
    loop body runs per position ('me', 'him' for GROW, 'race' for RACE) and the final sets ('set_me', 'set_him')."""
    F = env['F']
    if p[0] == 'POP':
        a, b = evaluate(p[1], env), evaluate(swap(p[1]), env)
        return popcount(a), popcount(b), {'set_me': a, 'set_him': b}
    g, out = p[1], p[2]
    n = len(F)
    info = {}
    local = dict(env)
    if p[0] == 'GROW':
        cols = []
        for side, start, body in (('me', env['M'], g), ('him', env['H'], swap(g, ab=False))):
            A = start.copy()
            w = np.zeros(n, np.int32)
            rounds = np.zeros(n)
            for k in range(MAX_ROUNDS):
                local['A'] = A
                new = evaluate(body, local) & ~A
                active = new != 0
                if not active.any():
                    break
                rounds += active
                if out == 'w' and k <= 5:
                    w += popcount(new & F) << (5 - k)
                A = A | new
            info[side] = float(np.minimum(rounds + 1, MAX_ROUNDS).mean())   # + the round that finds nothing
            info['set_' + side] = A
            cols.append(popcount(A & F) if out == '#' else w)
        return cols[0], cols[1], info
    A, B = env['M'].copy(), env['H'].copy()
    neutral = np.zeros_like(A)
    gs = swap(g)
    w_me, w_him = np.zeros(n, np.int32), np.zeros(n, np.int32)
    rounds = np.zeros(n)
    for k in range(MAX_ROUNDS):
        local['A'], local['B'] = A, B
        a = evaluate(g, local) & ~A & ~neutral
        b = evaluate(gs, local) & ~B & ~neutral
        both = a & b
        neutral |= both
        a &= ~both
        b &= ~both
        active = (a | b) != 0
        if not active.any():
            break
        rounds += active
        if out == 'w' and k <= 5:
            w_me += popcount(a & F) << (5 - k)
            w_him += popcount(b & F) << (5 - k)
        A |= a
        B |= b
    info['race'] = float(np.minimum(rounds + 1, MAX_ROUNDS).mean())     # + the round that finds nothing
    info['set_me'], info['set_him'] = A, B
    if out == '#':
        return popcount(A & F), popcount(B & F), info
    return w_me, w_him, info


def library_env(env):
    """Add the leaves L1, L1s… (sets of the kept programs, for me and for him)."""
    env = dict(env)
    for k, p in enumerate(LIBRARY):
        _, _, info = run_program(p, dict(env, cache={}, stats=None))
        env[f"L{k + 1}"], env[f"L{k + 1}s"] = info['set_me'], info['set_him']
    return env


def env_of(r):
    M, H, X = r['mine'].copy(), r['his'].copy(), r['holes'].copy()
    return {'M': M, 'H': H, 'X': X, 'F': ~(M | H | X)}


# ── the cost: operation counts on the computation graph ──────────────────────
def subexprs(e, acc):
    if len(e) > 1 and e not in acc:
        acc.add(e)
        for c in e[1:]:
            subexprs(c, acc)
    return acc


def add_node(counts, e, m, stats):
    op = e[0]
    if op in ('N', 'S', 'Q'):
        counts[op] += m
    elif op == 'C':
        counts['N'] += m                                         # the first step
        counts['Cr'] += m * stats.get(e, 4.0)
    else:
        counts['bit'] += m


def program_counts(p, info, stats):
    """How many times each priced operation runs per evaluation (both sides),
    on the merged computation graph: every distinct node once (loop nodes that
    depend on the state once per round), nothing below a node of HAVE."""
    counts = dict.fromkeys(PRICE_KEYS, 0.0)
    counts['call'] = 1
    if p[0] == 'POP':
        roots = [(p[1], 1.0), (swap(p[1]), 1.0)]
        loop_bits = loop_pops = rounds_total = 0.0
        final_pops = 2
    elif p[0] == 'GROW':
        rm, rh = info.get('me', 1.0), info.get('him', 1.0)
        roots = [(p[1], rm), (swap(p[1], ab=False), rh)]
        rounds_total = rm + rh
        loop_bits, loop_pops, final_pops = 3, (1 if p[2] == 'w' else 0), (2 if p[2] == '#' else 0)
    else:
        r = info.get('race', 1.0)
        roots = [(p[1], r), (swap(p[1]), r)]
        rounds_total = r
        loop_bits, loop_pops, final_pops = 10, (2 if p[2] == 'w' else 0), (2 if p[2] == '#' else 0)
    seen = set()

    def walk(e, rounds):
        if len(e) == 1 or e in HAVE:
            return
        m = rounds if uses_state(e) else 1.0            # invariant nodes: hoisted, computed once
        key = (e, m) if uses_state(e) else e
        if key in seen:
            return
        seen.add(key)
        add_node(counts, e, m, stats)
        for c in e[1:]:
            walk(c, rounds)

    for e, rounds in roots:
        walk(e, rounds)
    counts['bit'] += loop_bits * rounds_total
    counts['pop'] += loop_pops * rounds_total + final_pops
    counts['round'] += rounds_total
    return counts


def price(counts, prices):
    return sum(counts[k] * prices[k] for k in PRICE_KEYS)


def speed_prior(score, ns, nd):
    return score - LAM * np.log2(max(ns, 1.0)) - MU * nd


# ── C++ translation, timing, check ───────────────────────────────────────────
CPP_PRELUDE = r"""
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
using u64 = uint64_t;
static constexpr u64 NA = ~0x0101010101010101ULL, NH = ~0x8080808080808080ULL;
static inline u64 near_(u64 x) {
    return (x << 8) | (x >> 8) | ((x << 1) & NA) | ((x >> 1) & NH) |
           ((x << 9) & NA) | ((x << 7) & NH) | ((x >> 7) & NA) | ((x >> 9) & NH);
}
template <int D, u64 MASK> static inline u64 shd(u64 x) { return (D > 0 ? (x << D) : (x >> -D)) & MASK; }
template <int D, u64 MASK> static inline u64 slide(u64 gen, u64 F) {
    u64 pro = F;
    gen |= pro & shd<D, MASK>(gen);
    pro &= shd<D, MASK>(pro);
    gen |= pro & shd<D, MASK>(shd<D, MASK>(gen));
    pro &= shd<D, MASK>(shd<D, MASK>(pro));
    gen |= pro & shd<D, MASK>(shd<D, MASK>(shd<D, MASK>(shd<D, MASK>(gen))));
    return shd<D, MASK>(gen) & F;
}
static inline u64 qreach(u64 x, u64 F) {
    return slide<8, ~0ULL>(x, F) | slide<-8, ~0ULL>(x, F) | slide<1, NA>(x, F) | slide<-1, NH>(x, F) |
           slide<9, NA>(x, F) | slide<7, NH>(x, F) | slide<-7, NA>(x, F) | slide<-9, NH>(x, F);
}
static inline u64 closure(u64 x, u64 F) {
    u64 c = (x | near_(x)) & F;
    for (;;) { const u64 n = c | (near_(c) & F); if (n == c) return c; c = n; }
}
static inline int pc(u64 x) { return std::popcount(x); }
"""


class Emitter:
    """One C++ statement per graph node; `names`: expression → variable (CSE)."""
    def __init__(self, leaves):
        self.lines, self.names, self.k = [], dict(leaves), 0

    def var(self, e, names=None, lines=None):
        names = self.names if names is None else names
        lines = self.lines if lines is None else lines
        if len(e) == 1:
            return names[e[0]]
        if e in names:
            return names[e]
        args = [self.var(c, names, lines) for c in e[1:]]
        op = e[0]
        code = {'N': f"near_({args[0]})", 'S': f"near_({args[0]}) & F", 'Q': f"qreach({args[0]}, F)",
                'C': f"closure({args[0]}, F)"}.get(op)
        if code is None:
            code = f"{args[0]} & {args[1]}" if op == '&' else f"{args[0]} | {args[1]}" if op == '|' \
                else f"{args[0]} & ~{args[1]}"
        self.k += 1
        v = f"v{self.k}"
        lines.append(f"const u64 {v} = {code};")
        names[e] = v
        return v


def leaf_names():
    d = {'M': 'M', 'H': 'H', 'F': 'F', 'X': 'X'}
    for k in range(len(LIBRARY)):
        d[f"L{k + 1}"], d[f"L{k + 1}s"] = f"Lm[{k}]", f"Lh[{k}]"
    return d


def cpp_body(p, mode):
    """C++ statements computing the program. mode 'cols': out[0], out[1] (both sides);
    'set': return the set for me (library function)."""
    em = Emitter(leaf_names())
    L = em.lines
    if p[0] == 'POP':
        a = em.var(p[1])
        if mode == 'set':
            L.append(f"return {a};")
            return L
        b = em.var(swap(p[1]))
        L.append(f"out[0] = pc({a}); out[1] = pc({b});")
        return L
    g, out = p[1], p[2]
    bodies = [g, swap(g, ab=False)] if p[0] == 'GROW' else [g, swap(g)]
    if mode == 'set' and p[0] == 'GROW':
        bodies = bodies[:1]
    for body in bodies:                                   # hoisted invariant nodes
        for s in sorted(subexprs(body, set()), key=nodes):
            if not uses_state(s):
                em.var(s)
    if p[0] == 'GROW':
        for (start, i), body in zip((('M', 0), ('H', 1)), bodies):
            L.append(f"{{ u64 A = {start}; int w = 0; (void)w;")
            L.append(f"  for (int k = 0; k < {MAX_ROUNDS}; k++) {{")
            inner = dict(em.names)
            inner['A'] = 'A'
            lines = []
            v = em.var(body, inner, lines)
            L += ["    " + x for x in lines]
            L.append(f"    const u64 nw = {v} & ~A; if (!nw) break;")
            if out == 'w':
                L.append("    if (k <= 5) w += pc(nw & F) << (5 - k);")
            L.append("    A |= nw; }")
            L.append("  return A; }" if mode == 'set' else
                     f"  out[{i}] = {'pc(A & F)' if out == '#' else 'w'}; }}")
        return L
    L.append("u64 A = M, B = H, neu = 0; int wa = 0, wb = 0; (void)wa; (void)wb;")
    L.append(f"for (int k = 0; k < {MAX_ROUNDS}; k++) {{")
    inner = dict(em.names)
    inner['A'], inner['B'] = 'A', 'B'
    lines = []
    va = em.var(bodies[0], inner, lines)
    vb = em.var(bodies[1], inner, lines)
    L += ["  " + x for x in lines]
    L.append(f"  u64 a = {va} & ~A & ~neu, b = {vb} & ~B & ~neu; const u64 both = a & b;")
    L.append("  neu |= both; a &= ~both; b &= ~both; if (!(a | b)) break;")
    if out == 'w':
        L.append("  if (k <= 5) { wa += pc(a & F) << (5 - k); wb += pc(b & F) << (5 - k); }")
    L.append("  A |= a; B |= b; }")
    if mode == 'set':
        L.append("return A;")
    elif out == '#':
        L.append("out[0] = pc(A & F); out[1] = pc(B & F);")
    else:
        L.append("out[0] = wa; out[1] = wb;")
    return L


def cpp_library():
    out = []
    for k, p in enumerate(LIBRARY):
        out.append(f"// L{k + 1} = the set of {show(p)}")
        out.append(f"static inline u64 lib{k + 1}(u64 M, u64 H, u64 X, u64 F, const u64* Lm, const u64* Lh) {{")
        out.append("    (void)M; (void)H; (void)X; (void)F; (void)Lm; (void)Lh;")
        out += ["    " + x for x in cpp_body(p, 'set')]
        out.append("}")
    return "\n".join(out)


def measure_cpp(programs, pos_file, work_dir, target_ns=2e7):
    """Compile the programs into one benchmark, time each (ns per position, both
    sides) → list of (ns, checksum me, checksum him), or None (no compiler)."""
    nl = len(LIBRARY)
    src = [CPP_PRELUDE, cpp_library()]
    for i, p in enumerate(programs):
        src.append(f"// {show(p)}")
        src.append(f"__attribute__((noinline)) static void f{i}(u64 M, u64 H, u64 X, u64 F, const u64* Lm, "
                   f"const u64* Lh, int* out) {{")
        src.append("    (void)M; (void)H; (void)X; (void)F; (void)Lm; (void)Lh;")
        src += ["    " + x for x in cpp_body(p, 'cols')]
        src.append("}")
    src.append("using Fn = void (*)(u64, u64, u64, u64, const u64*, const u64*, int*);")
    src.append(f"static Fn FNS[] = {{{', '.join(f'f{i}' for i in range(len(programs)))}}};")
    lib_init = "".join(f"        Lm[i * NL + {k}] = lib{k + 1}(M, H, X, F[i], &Lm[i * NL], &Lh[i * NL]);\n"
                       f"        Lh[i * NL + {k}] = lib{k + 1}(H, M, X, F[i], &Lh[i * NL], &Lm[i * NL]);\n"
                       for k in range(nl))
    src.append(r"""
int main(int, char** argv) {
    std::ifstream in(argv[1], std::ios::binary);
    in.seekg(0, std::ios::end); const size_t bytes = in.tellg(); in.seekg(0);
    std::vector<u64> d(bytes / 8); in.read((char*)d.data(), bytes);
    const size_t n = d.size() / 3;
    constexpr int NL = """ + str(max(nl, 1)) + r""";
    std::vector<u64> Lm(n * NL), Lh(n * NL), F(n);
    for (size_t i = 0; i < n; i++) {
        const u64 M = d[3 * i], H = d[3 * i + 1], X = d[3 * i + 2];
        F[i] = ~(M | H | X);
        (void)M; (void)H; (void)X;
""" + lib_init + r"""    }
    long long sink = 0;
    for (size_t f = 0; f < sizeof(FNS) / sizeof(FNS[0]); f++) {
        long long s0 = 0, s1 = 0;
        int out[2];
        for (size_t i = 0; i < n; i++) {
            FNS[f](d[3 * i], d[3 * i + 1], d[3 * i + 2], F[i], &Lm[i * NL], &Lh[i * NL], out);
            s0 += out[0]; s1 += out[1];
        }
        double best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            int loops = 0;
            const auto t0 = std::chrono::steady_clock::now();
            double el = 0;
            do {
                for (size_t i = 0; i < n; i++) {
                    FNS[f](d[3 * i], d[3 * i + 1], d[3 * i + 2], F[i], &Lm[i * NL], &Lh[i * NL], out);
                    sink += out[0] - out[1];
                }
                loops++;
                el = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
            } while (el < """ + f"{target_ns / 3:.0f}" + r""");
            best = std::min(best, el / (double(n) * loops));
        }
        std::printf("%zu %.3f %lld %lld\n", f, best, s0, s1);
    }
    std::fprintf(stderr, "%lld\n", sink & 1);
}
""")
    cpp = os.path.join(work_dir, "measure.cpp")
    exe = os.path.join(work_dir, "measure")
    with open(cpp, "w") as f:
        f.write("\n".join(src))
    try:
        subprocess.run(["g++", "-std=c++20", "-O3", "-march=native", "-o", exe, cpp], check=True,
                       capture_output=True, text=True)
        out = subprocess.run([exe, pos_file], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        print("C++ measure failed:", str(getattr(e, 'stderr', e))[:2000], flush=True)
        return None
    res = {}
    for line in out.split("\n"):
        if line.strip():
            i, ns, s0, s1 = line.split()
            res[int(i)] = (float(ns), int(s0), int(s1))
    return [res.get(i) for i in range(len(programs))]


def nnls(A, b, iters=200):
    """Non-negative least squares (small: drop the most negative column until none)."""
    active = np.ones(A.shape[1], bool)
    x = np.zeros(A.shape[1])
    for _ in range(iters):
        sol = np.linalg.lstsq(A[:, active], b, rcond=None)[0]
        if (sol >= 0).all():
            x[:] = 0
            x[active] = sol
            return x
        idx = np.flatnonzero(active)
        active[idx[np.argmin(sol)]] = False
    return x


# ── the score ────────────────────────────────────────────────────────────────
class Scorer:
    def __init__(self, r, base_refs):
        self.n = len(r)
        self.free = r['free'].astype(np.int64)
        self.group = self.free * 2 + r['colour']
        self.test = ((r['game'] * np.uint32(2654435761)) >> np.uint32(16)) & np.uint32(1) == 1
        self.phase = np.full(self.n, -1)
        for i, (lo, hi, _) in enumerate(PHASES):
            self.phase[(self.free >= lo) & (self.free <= hi)] = i
        self.env = env_of(r)
        v = r['value'].astype(np.float64)
        cols = [v, v * v, v ** 3]
        for name in base_refs:
            cols += [c.astype(np.float64) for c in REFS[name](self.env)]
        self.base = np.stack([self.center(c) for c in cols], 1)
        self.y = {t: self.center(r['future' if t == 'points' else 'result'].astype(np.float64)) for t in TARGETS}
        self.refit()

    def add_base(self, me, him):
        new = np.stack([self.center(me.astype(np.float64)), self.center(him.astype(np.float64))], 1)
        self.base = np.concatenate([self.base, new], 1)
        self.refit()

    def refit(self):
        self.resid, self.sst = {}, {}
        for t in TARGETS:
            y = self.y[t]
            res = np.empty(self.n)
            sst = np.zeros(len(PHASES))
            for ph in range(len(PHASES)):
                tr, te, al = (self.phase == ph) & ~self.test, (self.phase == ph) & self.test, self.phase == ph
                X = self.base[tr]
                coef = np.linalg.solve(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ y[tr])
                res[al] = y[al] - self.base[al] @ coef
                sst[ph] = (y[te] ** 2).sum()
            self.resid[t], self.sst[t] = res, sst

    def center(self, x):
        tr = ~self.test
        s = np.bincount(self.group[tr], x[tr], minlength=130)
        c = np.bincount(self.group[tr], minlength=130)
        return x - (s / np.maximum(c, 1))[self.group]

    def gains(self, me, him):
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


def score_of(g):
    return float((g.mean(axis=1) * WEIGHTS).sum() / WEIGHTS.sum())


# ── reference features ───────────────────────────────────────────────────────
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


# ── workers ──────────────────────────────────────────────────────────────────
W = {}


def init_worker(witness_env, scorer_env, harness_env, scorer, library, have, prices):
    LIBRARY[:] = library
    HAVE.clear()
    HAVE.update(have)
    W['witness'] = library_env(witness_env)
    W['full'] = library_env(scorer_env)
    W['harness'] = library_env(harness_env)
    W['scorer'], W['prices'] = scorer, prices


def witness_key(p):
    """Columns on the witness positions → (key, predicted ns). The network sees
    the columns centred per number of free squares, through a linear layer: two
    programs whose columns differ only by a function of the free squares
    (|S | X| = |S| + 56 − free), by sign or by scale (|F − S| = free − |S|) are
    the same feature; the key is taken after removing these."""
    try:
        stats = {}
        me, him, info = run_program(p, dict(W['witness'], cache={}, stats=stats))
    except Exception:
        return None, 0.0
    free = popcount(W['witness']['F'])
    cols = []
    for c in (me, him):
        c = c.astype(np.float64)
        c = c - (np.bincount(free, c, 65) / np.maximum(np.bincount(free, minlength=65), 1))[free]
        cols.append(c)
    both = np.concatenate(cols)
    m = np.abs(both).max()
    if m < 1e-9:
        return None, 0.0
    both = both / m
    first = both[np.argmax(np.abs(both) > 1e-6)]
    both = np.round(both * np.sign(first), 6) + 0.0
    ns = price(program_counts(p, info, stats), W['prices'])
    return hashlib.blake2b(both.tobytes(), digest_size=12).digest(), ns


def standalone_ns(p):
    """Predicted cost with nothing already computed (what the C++ benchmark times)."""
    have = set(HAVE)
    HAVE.clear()
    try:
        stats = {}
        _, _, info = run_program(p, dict(W['witness'], cache={}, stats=stats))
        return price(program_counts(p, info, stats), W['prices'])
    finally:
        HAVE.update(have)


def full_score(p):
    me, him, _ = run_program(p, dict(W['full'], cache={}, stats=None))
    return W['scorer'].gains(me, him)


def columns_full(p):
    me, him, _ = run_program(p, dict(W['full'], cache={}, stats=None))
    return me, him


def columns_harness(p):
    me, him, _ = run_program(p, dict(W['harness'], cache={}, stats=None))
    return me, him


# ── the search ───────────────────────────────────────────────────────────────
class Entry:
    __slots__ = ('p', 'score', 'gains', 'ns', 'nd', 'sp', 'measured')

    def __init__(self, p, gains, ns):
        self.p, self.gains, self.ns, self.nd = p, gains, ns, nodes(p)
        self.score = score_of(gains)
        self.sp = speed_prior(self.score, ns, self.nd)
        self.measured = None


class Search:
    def __init__(self, scorer, witness_env, harness_env, harness_file, out_dir, workers, minutes, prices):
        self.scorer, self.witness_env, self.harness_env = scorer, witness_env, harness_env
        self.harness_file, self.out_dir, self.workers, self.prices = harness_file, out_dir, workers, prices
        self.seen = {}               # witness key → Entry
        self.entries = []
        self.t0 = time.time()
        self.deadline = self.t0 + 60 * minutes
        self.last_board = 0
        self.pool = None
        self.restart_pool()
        self.tsv = open(os.path.join(out_dir, "all.tsv"), "w")
        self.tsv.write("stage\tscore\tsp\tns\t" + "\t".join(f"{t}_{p[2]}" for t in TARGETS for p in PHASES)
                       + "\tprogram\n")
        self.stage = 0
        self.best_score = self.best_sp = -1e9

    def restart_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
        self.pool = Pool(self.workers, initializer=init_worker,
                         initargs=(self.witness_env, self.scorer.env, self.harness_env, self.scorer,
                                   list(LIBRARY), set(HAVE), self.prices))

    def time_left(self):
        return time.time() < self.deadline

    def submit(self, programs):
        keys = self.pool.map(witness_key, programs, chunksize=16)
        fresh, batch_keys = [], set()
        for p, (k, ns) in zip(programs, keys):
            if k is None:
                continue
            old = self.seen.get(k)
            if old is None and k not in batch_keys:
                batch_keys.add(k)
                fresh.append((k, p, ns))
            elif old is not None and ns < old.ns - 1e-9:        # same feature, cheaper program
                old.p, old.ns, old.nd = p, ns, nodes(p)
                old.sp = speed_prior(old.score, ns, old.nd)
                old.measured = None
        if not fresh:
            return
        gains = self.pool.map(full_score, [p for _, p, _ in fresh], chunksize=4)
        new = []
        for (k, p, ns), g in zip(fresh, gains):
            e = Entry(p, g, ns)
            self.seen[k] = e
            self.entries.append(e)
            new.append(e)
            self.tsv.write(f"{self.stage}\t{e.score:.4f}\t{e.sp:.4f}\t{ns:.1f}\t"
                           + "\t".join(f"{x:.3f}" for x in g.ravel()) + f"\t{show(p)}\n")
        self.tsv.flush()
        on_front = {id(e) for e in self.front()}
        for e in new:
            tags = []
            if e.score > self.best_score:
                self.best_score = e.score
                tags.append("MEILLEUR")
            if e.sp > self.best_sp:
                self.best_sp = e.sp
                tags.append("MEILLEUR SP")
            if not tags and id(e) in on_front:
                tags.append("front")
            if tags:
                print(f"[{(time.time() - self.t0) / 60:6.1f} min] {'/'.join(tags):20s} {self.line(e)}", flush=True)
        if time.time() - self.last_board > 60:
            self.write_board()

    def front(self):
        out, best = [], -1e9
        for e in sorted(self.entries, key=lambda e: (e.ns, -e.score)):
            if e.score > best + 1e-9:
                best = e.score
                out.append(e)
        return out

    @staticmethod
    def line(e):
        mes = f"{e.measured:6.1f}" if e.measured is not None else "     -"
        return (f"score {e.score:6.3f} sp {e.sp:6.3f} ns {e.ns:6.1f} mes. {mes} | points "
                f"{' '.join(f'{x:5.2f}' for x in e.gains[0])} | résultat {' '.join(f'{x:5.2f}' for x in e.gains[1])}"
                f" | {show(e.p)}")

    def write_board(self):
        self.last_board = time.time()
        lines = [f"stage {self.stage}, programs scored: {len(self.entries)}, {(time.time() - self.t0) / 60:.1f} min",
                 f"base: NNUE value (cubic){''.join(f' + L{k + 1}' for k in range(len(LIBRARY)))}",
                 "gains in R² points over the base, phases " + " ".join(p[2] for p in PHASES),
                 f"sp = score − {LAM} log2(ns) − {MU} nodes; ns = predicted MARGINAL cost (both sides), "
                 "mes. = measured in C++ alone (it recomputes what the kept features already give)", ""]
        if LIBRARY:
            lines += ["LIBRARY (OOPS stages)"] + [f"  L{k + 1} = {show(p)}" for k, p in enumerate(LIBRARY)] + [""]
        lines += ["BEST 40 BY SPEED PRIOR"] + [self.line(e) for e in sorted(self.entries, key=lambda e: -e.sp)[:40]]
        lines += ["", "BEST 20 BY SCORE"] + [self.line(e) for e in sorted(self.entries, key=lambda e: -e.score)[:20]]
        lines += ["", "PARETO FRONT (no cheaper program scores more)"] + [self.line(e) for e in self.front()]
        with open(os.path.join(self.out_dir, "leaderboard.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # ── step 4: the front and the best, measured in C++ ──
    def measure(self):
        todo = [e for e in self.front() + sorted(self.entries, key=lambda e: -e.sp)[:20] if e.measured is None]
        todo = list({id(e): e for e in todo}.values())[:80]
        if not todo:
            return
        res = measure_cpp([e.p for e in todo], self.harness_file, self.out_dir)
        if res is None:
            return
        cols = self.pool.map(columns_harness, [e.p for e in todo])
        alone = self.pool.map(standalone_ns, [e.p for e in todo])
        bad = 0
        pred, meas = [], []
        for e, r, (me, him), a in zip(todo, res, cols, alone):
            if r is None:
                continue
            e.measured = r[0]
            pred.append(a)
            meas.append(r[0])
            if (r[1], r[2]) != (int(me.sum()), int(him.sum())):
                bad += 1
                print(f"  C++ ≠ Python: {show(e.p)}: {r[1:]} vs {(int(me.sum()), int(him.sum()))}", flush=True)
        pred, meas = np.array(pred), np.array(meas)
        if len(pred) > 2:
            print(f"[{(time.time() - self.t0) / 60:6.1f} min] {len(meas)} programs measured in C++: "
                  f"predicted (alone) / measured median {np.median(pred / meas):.2f}, corr {np.corrcoef(pred, meas)[0, 1]:.2f}, "
                  f"{bad} value mismatches", flush=True)
        self.write_board()

    # ── OOPS stage ──
    def new_stage(self, min_gain):
        cands = [e for e in sorted(self.entries, key=lambda e: -e.sp) if e.score >= min_gain]
        if not cands:
            print("OOPS: no program good enough to freeze", flush=True)
            return False
        e = cands[0]
        me, him = self.pool.apply(columns_full, (e.p,))
        LIBRARY.append(e.p)
        k = len(LIBRARY)
        # what the engine now computes: the new leaf and the parts of the program without state
        body = e.p[1]
        for r in (body, swap(body), swap(body, ab=False)):
            for s in subexprs(r, set()):
                if not uses_state(s):
                    HAVE.add(s)
        self.scorer.add_base(me, him)
        self.stage += 1
        mes = f", mesuré {e.measured:.0f} ns" if e.measured is not None else ""
        print(f"\n===== ÉTAPE {self.stage} : L{k} = {show(e.p)}  (score {e.score:.3f}, {e.ns:.0f} ns{mes}) "
              f"rejoint la base et devient une brique =====\n", flush=True)
        with open(os.path.join(self.out_dir, "library.txt"), "a") as f:
            f.write(f"L{k}\t{show(e.p)}\tscore {e.score:.4f}\tns {e.ns:.1f}\tmeasured {e.measured}\t"
                    f"points {' '.join(f'{x:.2f}' for x in e.gains[0])}\t"
                    f"result {' '.join(f'{x:.2f}' for x in e.gains[1])}\n")
        with open(os.path.join(self.out_dir, "features.h"), "w") as f:
            f.write("// generated by nnue/program_search.py: the sets of the features kept by the OOPS stages\n"
                    "// (feature = popcount of the set, for me: libk(M, H, …), for him: libk(H, M, …))\n"
                    + CPP_PRELUDE + "\n" + cpp_library() + "\n")
        # rescore the best programs against the new base, forget the others
        keep = sorted(self.entries, key=lambda x: -x.sp)[:300] + self.front()
        keep = [x for x in {id(x): x for x in keep}.values() if x is not e]
        self.restart_pool()
        gains = self.pool.map(full_score, [x.p for x in keep], chunksize=4)
        self.entries = []
        self.seen = {key: x for key, x in self.seen.items() if x in keep}
        for x, g in zip(keep, gains):
            x.gains, x.score = g, score_of(g)
            x.sp = speed_prior(x.score, x.ns, x.nd)
            self.entries.append(x)
        self.best_score = max(x.score for x in self.entries)
        self.best_sp = max(x.sp for x in self.entries)
        self.write_board()
        leaf = (f"L{k}",)
        small = [('POP', leaf)] + [('POP', (u, leaf)) for u in UNARY] + \
                [('POP', (op, leaf, (l,))) for op in BINARY for l in LEAVES + ['M', 'H']] + \
                [('POP', ('-', (l,), leaf)) for l in LEAVES]
        self.submit(small)
        return True

    # ── evolution ──
    def leaves_for(self, kind):
        base = LEAVES + lib_leaves()
        return base + (['A', 'B'] if kind == 'RACE' else ['A'] if kind == 'GROW' else [])

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
            return e[random.randrange(1, len(e))]
        i = random.randrange(1, len(e))
        kids = list(e)
        kids[i] = self.mutate_expr(e[i], leaves)
        return tuple(kids)

    def mutate(self, p, others):
        if p[0] == 'POP':
            if random.random() < 0.1:
                return (random.choice(['GROW', 'RACE']), ('S', ('A',)) if random.random() < 0.5 else
                        ('&', p[1], ('S', ('A',))), random.choice('#w'))
            if random.random() < 0.15 and others:
                o = random.choice(others)
                if o[0] == 'POP':
                    return ('POP', (random.choice(BINARY), p[1], o[1]))
            return ('POP', self.mutate_expr(p[1], self.leaves_for('POP')))
        r = random.random()
        if r < 0.15:
            return (p[0], p[1], '#' if p[2] == 'w' else 'w')
        if r < 0.25:
            return ('RACE' if p[0] == 'GROW' else 'GROW', p[1] if p[0] == 'GROW' else ('S', ('A',)), p[2])
        g = self.mutate_expr(p[1], self.leaves_for(p[0]))
        if p[0] == 'GROW' and uses_b(g):
            g = replace_b(g)
        if not uses_state(g):
            g = ('&', g, ('S', ('A',)))
        return (p[0], g, p[2])

    def evolve(self, stage_minutes, measure_minutes, min_gain, batch=256):
        next_stage = time.time() + 60 * stage_minutes
        next_measure = time.time() + 60 * measure_minutes
        while self.time_left():
            if time.time() > next_measure:
                self.measure()
                next_measure = time.time() + 60 * measure_minutes
            if time.time() > next_stage:
                self.measure()
                self.new_stage(min_gain)
                next_stage = time.time() + 60 * stage_minutes
            ok = [e for e in self.entries if e.nd <= MAX_NODES]
            parents = [e.p for e in sorted(ok, key=lambda e: -e.sp)[:50]]
            parents += [e.p for e in sorted(ok, key=lambda e: -e.score)[:20]]
            parents += [e.p for e in self.front() if e.nd <= MAX_NODES]
            kids, tries = set(), 0
            while len(kids) < batch and tries < 50 * batch:
                tries += 1
                k = self.mutate(random.choice(parents), parents)
                if nodes(k) <= MAX_NODES:
                    kids.add(k)
            self.submit(list(kids))


def uses_b(e):
    return e[0] == 'B' or any(uses_b(c) for c in e[1:] if isinstance(c, tuple))


def replace_b(e):
    if len(e) == 1:
        return ('A',) if e[0] == 'B' else e
    return (e[0],) + tuple(replace_b(c) for c in e[1:])


# ── enumeration ──────────────────────────────────────────────────────────────
def grow_exprs(leaves, max_size):
    by_size = {1: [(l,) for l in leaves]}
    for s in range(2, max_size + 1):
        cur = []
        for e in by_size[s - 1]:
            for u in UNARY:
                if u in ('S', 'N', 'C') and e[0] == 'C':
                    continue
                cur.append((u, e))
        for sa in range(1, s - 1):
            sb = s - 1 - sa
            for a in by_size[sa]:
                for b in by_size[sb]:
                    for op in BINARY:
                        if op in '&|' and (sa, a) > (sb, b):
                            continue
                        if a == b:
                            continue
                        cur.append((op, a, b))
        by_size[s] = cur
    return by_size


def loop_programs(body_exprs):
    progs = []
    for g in body_exprs:
        if not uses_state(g):
            continue
        for out in '#w':
            if not uses_b(g):
                progs.append(('GROW', g, out))
            progs.append(('RACE', g, out))
    return progs


def calibrate(harness_file, out_dir, witness_env):
    """Step 1: time ~160 programs in C++, fit the prices by non-negative least squares."""
    by_size = grow_exprs(LEAVES, 4)
    pops = [('POP', e) for s in range(2, 5) for e in by_size[s]]
    body = grow_exprs(LEAVES + ['A', 'B'], 3)
    loops = loop_programs([g for s in range(1, 4) for g in body[s]])
    random.shuffle(pops)
    random.shuffle(loops)
    rows, keep = [], []
    for p in pops[:110] + loops[:50]:
        stats = {}
        try:
            _, _, info = run_program(p, dict(witness_env, cache={}, stats=stats))
        except Exception:
            continue
        rows.append([program_counts(p, info, stats)[k] for k in PRICE_KEYS])
        keep.append(p)
    t = time.time()
    res = measure_cpp(keep, harness_file, out_dir)
    if res is None:
        print("no C++ compiler: default prices", flush=True)
        return dict(DEFAULT_PRICES)
    ok = [i for i, m in enumerate(res) if m is not None]
    A = np.array([rows[i] for i in ok])
    b = np.array([res[i][0] for i in ok])
    x = nnls(A, b)
    prices = {k: float(v) for k, v in zip(PRICE_KEYS, x)}
    pred = A @ x
    print(f"calibration ({time.time() - t:.0f} s, {len(b)} programs timed in C++): prices in ns "
          + ", ".join(f"{k} {v:.2f}" for k, v in prices.items())
          + f"; fit corr {np.corrcoef(pred, b)[0, 1]:.3f}, median error {np.median(np.abs(pred - b) / b) * 100:.0f} %",
          flush=True)
    for k in PRICE_KEYS:                      # an operation the fit dropped keeps a small positive price
        if prices[k] <= 0:
            prices[k] = DEFAULT_PRICES[k] * 0.1
    with open(os.path.join(out_dir, "prices.json"), "w") as f:
        json.dump(prices, f, indent=1)
    return prices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positions")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base", default="", help="reference features added to the base: infl,qfirst,mob")
    ap.add_argument("--have", default="", help="expressions the engine already computes (free), ';'-separated")
    ap.add_argument("--max-size", type=int, default=5, help="enumeration: set expressions up to this size")
    ap.add_argument("--body-size", type=int, default=4, help="enumeration: loop bodies up to this size")
    ap.add_argument("--max-nodes", type=int, default=15, help="evolution: largest program kept")
    ap.add_argument("--weights", default="1,10", help="score weights of the points and result gains")
    ap.add_argument("--lam", type=float, default=0.03, help="speed prior: score per doubling of the time")
    ap.add_argument("--mu", type=float, default=0.002, help="speed prior: score per node")
    ap.add_argument("--stage-minutes", type=float, default=240, help="OOPS: freeze the best program every …")
    ap.add_argument("--min-gain", type=float, default=0.05, help="OOPS: smallest score frozen")
    ap.add_argument("--measure-minutes", type=float, default=30, help="C++ timing of the front every …")
    ap.add_argument("--prices", default="", help="prices.json of an earlier run (skips the calibration)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    global MAX_NODES, LAM, MU
    random.seed(a.seed)
    WEIGHTS[:] = [float(x) for x in a.weights.split(",")]
    MAX_NODES, LAM, MU = a.max_nodes, a.lam, a.mu
    for h in filter(None, a.have.split(";")):
        e = parse(h)
        for s in (e, swap(e)):
            subexprs(s, HAVE)
    os.makedirs(a.out_dir, exist_ok=True)
    r = np.fromfile(a.positions, dtype=REC)
    rng = np.random.default_rng(a.seed)
    r = r[rng.choice(len(r), min(a.n, len(r)), replace=False)]
    base = [b for b in a.base.split(",") if b]
    print(f"{len(r)} positions, base = NNUE value (cubic){' + ' + ', '.join(base) if base else ''}", flush=True)
    scorer = Scorer(r, base)
    witness_env = env_of(r[rng.choice(len(r), 2000, replace=False)])
    hr = r[rng.choice(len(r), 20000, replace=False)]
    harness_file = os.path.join(a.out_dir, "harness_positions.bin")
    np.stack([hr['mine'], hr['his'], hr['holes']], 1).astype('<u8').tofile(harness_file)
    harness_env = env_of(hr)

    prices = json.load(open(a.prices)) if a.prices else calibrate(harness_file, a.out_dir, witness_env)

    print("references (same score):")
    for name, f in REFS.items():
        g = scorer.gains(*f(scorer.env))
        print(f"  {name:7s} score {score_of(g):6.3f}  points {' '.join(f'{x:5.2f}' for x in g[0])} | "
              f"résultat {' '.join(f'{x:5.2f}' for x in g[1])}", flush=True)

    s = Search(scorer, witness_env, harness_env, harness_file, a.out_dir, a.workers, a.minutes, prices)
    by_size = grow_exprs(LEAVES, a.max_size)
    print("enumeration: |set|", flush=True)
    for sz in range(1, a.max_size + 1):
        progs = [('POP', e) for e in by_size[sz]]
        for i in range(0, len(progs), 2048):
            if s.time_left():
                s.submit(progs[i:i + 2048])
        print(f"  size {sz}: {len(progs)} programs, {len(s.entries)} distinct features so far", flush=True)
    body = grow_exprs(LEAVES + ['A', 'B'], a.body_size)
    print("enumeration: loops", flush=True)
    for sz in range(1, a.body_size + 1):
        progs = loop_programs(body[sz])
        for i in range(0, len(progs), 1024):
            if s.time_left():
                s.submit(progs[i:i + 1024])
        print(f"  body size {sz}: {len(progs)} programs, {len(s.entries)} distinct features so far", flush=True)
    s.measure()
    print("evolution", flush=True)
    s.evolve(a.stage_minutes, a.measure_minutes, a.min_gain)
    s.measure()
    s.write_board()
    print("done", flush=True)


if __name__ == "__main__":
    main()
