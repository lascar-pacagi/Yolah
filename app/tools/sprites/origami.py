"""
origami.py — the Origami set: one folded paper figure per AI, from the
simplest fold to the hardest (the strength of the AI is the skill of the
folder):

  random        Bateau      the paper boat everyone has folded
  beginner      Grenouille  a jumping frog
  monte_carlo   Lapin       a rabbit, long ears
  mcts          Renard      the classic fox face on a body
  basic_minmax  Papillon    a butterfly, wings beating
  nnue193       Grue        the crane, orizuru
  features      Cerf        a stag, branching antlers
  resnet        Paon        a peacock, its tail spread in a fan
  dev           Lion        a lion, a mane of folded rays
  alphazero     Dragon      a dragon of red and gold paper, wings spread

Every piece is a convex polyhedron (sd_hull: the intersection of its face
planes), so the faces stay flat and the edges sharp, like folded paper. The
stronger the AI, the richer the paper: plain, then printed, then gold foil.
The human is never part of a set.
"""
import itertools
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, vec, sd_sphere, fbm
from models import base, BASE_TOP, TAU, glow
from heroic import HEAD


def sd_hull(points):
    """The convex hull of a few points: max over its faces of the signed
    distance to the face's plane (a lower bound of the true distance: safe
    for sphere tracing, and the faces come out perfectly flat)."""
    P = [torch.tensor(p, dtype=torch.float32) for p in points]
    planes = []
    for i, j, k in itertools.combinations(range(len(P)), 3):
        n = torch.linalg.cross(P[j] - P[i], P[k] - P[i])
        ln = float(n.norm())
        if ln < 1e-9:
            continue
        n = n / ln
        for nn in (n, -n):
            d = [float(torch.dot(nn, q - P[i])) for q in P]
            if max(d) <= 1e-6:
                if not any(float(torch.dot(nn, m)) > 0.9999 and abs(float(torch.dot(nn, P[i])) - o) < 1e-5 for m, o in planes):
                    planes.append((nn, float(torch.dot(nn, P[i]))))
    N = vec(*[0, 0, 0])
    Nm = torch.stack([m for m, _ in planes]).to(N.device)
    O = torch.tensor([o for _, o in planes], device=N.device)
    # far from the hull, the planes alone underestimate the distance badly (shadow
    # rays would stop short): a bounding sphere, also a lower bound, takes over
    Pc = torch.stack(P)
    ctr = Pc.mean(0)
    R = float((Pc - ctr).norm(dim=1).max())
    ctr = ctr.to(N.device)

    def f(p):
        return torch.maximum((p @ Nm.T - O).max(dim=-1).values, (p - ctr).norm(dim=-1) - R)
    return f


def sheet(pts, th=0.006):
    """A thin folded sheet: a flat polygon given a thickness along its normal."""
    a, b, c = (torch.tensor(q, dtype=torch.float32) for q in pts[:3])
    n = torch.linalg.cross(b - a, c - a)
    n = (n / n.norm() * th).tolist()
    return sd_hull([tuple(q[i] + s_ * n[i] for i in range(3)) for q in pts for s_ in (-1, 1)])


def paper(c, print_=None, gold=False):
    """Matte paper with fibres; print_: a second colour printed in small
    motifs (chiyogami); gold: gold foil (metallic)."""
    if gold:
        return Mat(c, metal=0.9, rough=0.3, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.85 + 0.3 * fbm(pl * 60, 2))[:, None]})

    def pat(pl, pw, n):
        a = srgb(c) * (0.93 + 0.1 * fbm(pl * 200, 2))[:, None]
        if print_:
            dots = ((torch.sin(pl[:, 0] * 260) * torch.sin(pl[:, 1] * 260) * torch.sin(pl[:, 2] * 260)) > 0.55).float()[:, None]
            a = a * (1 - dots) + srgb(print_) * dots
        return {"albedo": a}
    return Mat(c, rough=0.92, sss=0.15, pattern=pat)


def root_of(anim, t, lift=0.0, hop=0.05):
    shake = 0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    j = hop * abs(math.sin(TAU * t)) if anim in ("move", "win") else 0.006 * math.sin(TAU * t)
    spin = TAU * t if anim == "win" else 0.0
    return RY(HEAD + shake + 0.0 * spin) * T(0, BASE_TOP + lift + j, 0)


def gold_ring(sc, t, k=4.0):
    for i in range(10):
        a = TAU * (i / 10 + t * 0.2)
        sc.add(sheet([(0, 0, 0), (0.03, 0.02, 0), (0.0, 0.05, 0)], 0.002), paper("#e0b860", gold=True),
               T(0.46 * math.cos(a), BASE_TOP + 0.1 + 0.05 * math.sin(TAU * t + i), 0.46 * math.sin(a)) * RY(-a))


# ─── 1 · Bateau ─────────────────────────────────────────────────────────────
def boat(anim, t, side):
    sc = Scene()
    base(sc, side)
    rock = 0.12 * math.sin(TAU * t) + (0.3 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)
    root = root_of(anim, t) * RZ(rock * 0.4)
    m = paper("#f4efe4")
    hull_ = sd_hull([(-0.32, 0.12, -0.1), (0.32, 0.12, -0.1), (-0.32, 0.12, 0.1), (0.32, 0.12, 0.1),
                     (-0.2, 0.0, -0.06), (0.2, 0.0, -0.06), (-0.2, 0.0, 0.06), (0.2, 0.0, 0.06)])
    sc.add(hull_, m, root)
    sc.add(sheet([(-0.22, 0.12, 0), (0.22, 0.12, 0), (0, 0.42, 0)], 0.01), paper("#f4efe4"), root)     # the sail, a folded triangle
    sc.add(sheet([(-0.2, 0.13, 0.012), (0.0, 0.13, 0.012), (0, 0.4, 0.012)], 0.004), paper("#e8e0d0"), root)
    return sc


# ─── 2 · Grenouille ─────────────────────────────────────────────────────────
def frog(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t, hop=0.12)
    m = paper("#5aa04a")
    sc.add(sd_hull([(-0.14, 0.02, -0.16), (0.14, 0.02, -0.16), (-0.12, 0.02, 0.14), (0.12, 0.02, 0.14), (0, 0.2, -0.02), (0, 0.12, 0.18)]), m, root)
    for sgn in (-1, 1):
        sc.add(sheet([(0.1 * sgn, 0.02, -0.12), (0.26 * sgn, 0.0, -0.2), (0.22 * sgn, 0.06, -0.02)], 0.008), paper("#4a8a3a"), root)   # back legs
        sc.add(sheet([(0.08 * sgn, 0.02, 0.1), (0.2 * sgn, 0.0, 0.2), (0.12 * sgn, 0.06, 0.16)], 0.008), paper("#4a8a3a"), root)        # front legs
        sc.add(sd_hull([(0.04 * sgn, 0.16, 0.12), (0.09 * sgn, 0.16, 0.1), (0.065 * sgn, 0.22, 0.11), (0.065 * sgn, 0.17, 0.15)]), paper("#e8e0c0"), root)
        sc.add(sd_sphere(0.012), Mat("#141414", rough=0.2), root * T(0.065 * sgn, 0.185, 0.135))
    return sc


# ─── 3 · Lapin ──────────────────────────────────────────────────────────────
def rabbit(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t, hop=0.1)
    m = paper("#f2ece2")
    sc.add(sd_hull([(-0.12, 0.0, -0.18), (0.12, 0.0, -0.18), (-0.1, 0.0, 0.1), (0.1, 0.0, 0.1), (0, 0.24, -0.1), (0, 0.16, 0.08)]), m, root)
    head = root * T(0, 0.24, 0.1)
    sc.add(sd_hull([(-0.07, 0.0, -0.04), (0.07, 0.0, -0.04), (0, 0.1, -0.02), (0, -0.02, 0.1), (0, 0.04, 0.04)]), m, head)
    twitch = 0.15 * math.sin(TAU * t * 2)
    for sgn in (-1, 1):
        ear = head * T(0.035 * sgn, 0.07, -0.02) * RZ(-0.15 * sgn + twitch * sgn) * RX(-0.2)
        sc.add(sheet([(-0.03, 0, 0), (0.03, 0, 0), (0.0, 0.26, -0.02)], 0.006), m, ear)
        sc.add(sheet([(-0.015, 0.02, 0.008), (0.015, 0.02, 0.008), (0.0, 0.2, -0.01)], 0.003), paper("#f0b8b0"), ear)
        sc.add(sd_sphere(0.01), Mat("#141414", rough=0.2), head * T(0.04 * sgn, 0.04, 0.03))
    sc.add(sd_hull([(-0.04, 0.06, -0.18), (0.04, 0.06, -0.18), (0, 0.12, -0.2), (0, 0.06, -0.24)]), paper("#ffffff"), root)   # the tail
    return sc


# ─── 4 · Renard ─────────────────────────────────────────────────────────────
def fox(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t)
    m, w = paper("#e8762a"), paper("#f6efe2")
    sc.add(sd_hull([(-0.12, 0.0, -0.18), (0.12, 0.0, -0.18), (-0.1, 0.0, 0.12), (0.1, 0.0, 0.12), (0, 0.32, 0.0), (0, 0.2, -0.16)]), m, root)
    sc.add(sheet([(-0.08, 0.02, 0.12), (0.08, 0.02, 0.12), (0, 0.22, 0.06)], 0.006), w, root)            # the white chest
    head = root * T(0, 0.32, 0.04) * RY(0.3 * math.sin(TAU * t) if anim == "idle" else 0)
    sc.add(sd_hull([(-0.12, 0.0, 0.0), (0.12, 0.0, 0.0), (0, -0.08, 0.14), (0, 0.04, 0.06), (0, 0.0, -0.06)]), m, head)   # the classic face
    sc.add(sheet([(-0.1, -0.005, 0.02), (0, -0.075, 0.13), (0, -0.02, 0.05)], 0.004), w, head)
    sc.add(sheet([(0.1, -0.005, 0.02), (0, -0.075, 0.13), (0, -0.02, 0.05)], 0.004), w, head)
    for sgn in (-1, 1):
        sc.add(sd_hull([(0.05 * sgn, 0.0, 0.0), (0.12 * sgn, 0.0, -0.01), (0.1 * sgn, 0.14, -0.02), (0.09 * sgn, 0.02, -0.05)]), m, head)
        sc.add(sd_sphere(0.012), Mat("#141414", rough=0.2), head * T(0.04 * sgn, 0.01, 0.07))
    sc.add(sd_sphere(0.012), Mat("#141414", rough=0.2), head * T(0, -0.075, 0.14))
    sw = 0.3 * math.sin(TAU * t)
    sc.add(sd_hull([(-0.06, 0.04, -0.16), (0.06, 0.04, -0.16), (0.04 + sw * 0.1, 0.18, -0.36), (0, 0.0, -0.2), (0.0, 0.1, -0.4 + 0.02 * sw)]), m, root)
    return sc


# ─── 5 · Papillon ───────────────────────────────────────────────────────────
def butterfly(anim, t, side):
    sc = Scene()
    base(sc, side)
    flap = 0.5 + 0.5 * math.sin(TAU * t * (3 if anim in ("move", "win") else 1))
    root = root_of(anim, t, lift=0.2 + 0.05 * math.sin(TAU * t), hop=0.08)
    a, b = paper("#3a6ac8", "#f2c84a"), paper("#2a4a98", "#f2f0e8")
    sc.add(sd_hull([(0, 0.0, -0.14), (0, 0.0, 0.12), (0.015, 0.02, 0), (-0.015, 0.02, 0), (0, -0.02, 0)]), paper("#2a2a30"), root)
    for sgn in (-1, 1):
        wing = root * RZ(sgn * (0.2 + 1.0 * flap))
        sc.add(sheet([(0, 0, -0.02), (0, 0, 0.12), (0.28 * sgn, 0, 0.2), (0.3 * sgn, 0, 0.02)], 0.004), a, wing)
        sc.add(sheet([(0, 0, -0.02), (0, 0, -0.12), (0.2 * sgn, 0, -0.2), (0.22 * sgn, 0, -0.04)], 0.004), b, wing)
        sc.add(sheet([(0.01 * sgn, 0.01, 0.12), (0.015 * sgn, 0.012, 0.13), (0.06 * sgn, 0.08, 0.22)], 0.003), paper("#2a2a30"), root)   # antennae
    return sc


# ─── 6 · Grue ───────────────────────────────────────────────────────────────
def crane(anim, t, side):
    sc = Scene()
    base(sc, side)
    flap = 0.3 * math.sin(TAU * t * (2 if anim in ("move", "win") else 0.5))
    root = root_of(anim, t, lift=0.06)
    m, m2 = paper("#f2f0ea", "#c8382e"), paper("#e8e4da")
    sc.add(sd_hull([(0, 0.0, -0.08), (0, 0.0, 0.08), (0.07, 0.08, 0), (-0.07, 0.08, 0), (0, 0.16, 0)]), m, root)     # the body, a diamond
    sc.add(sheet([(0, 0.06, -0.04), (0, 0.06, 0.04), (0, 0.42, -0.24)], 0.006), m2, root)      # the tail
    nk = root
    sc.add(sheet([(0, 0.06, -0.03), (0, 0.06, 0.05), (0, 0.4, 0.22)], 0.006), m2, nk)          # the neck
    sc.add(sheet([(0, 0.4, 0.22), (0, 0.37, 0.22), (0, 0.33, 0.32)], 0.006), m2, nk)           # the head, folded down
    for sgn in (-1, 1):
        wing = root * T(0, 0.12, 0) * RZ(sgn * (0.6 + flap))
        sc.add(sheet([(0, 0, -0.1), (0, 0, 0.1), (0.36 * sgn, 0.0, 0.0)], 0.005), m, wing)
    return sc


# ─── 7 · Cerf ───────────────────────────────────────────────────────────────
def stag(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t)
    m, d = paper("#a8784a", "#d8b888"), paper("#7a5232")
    L = 0.24
    body = root * T(0, L, 0)
    sc.add(sd_hull([(-0.08, 0.0, -0.2), (0.08, 0.0, -0.2), (-0.08, 0.0, 0.14), (0.08, 0.0, 0.14), (0, 0.12, -0.16), (0, 0.14, 0.1),
                    (0, -0.06, 0.0)]), m, body)
    for sx, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        sw = 0.3 * math.sin(TAU * t * 2 + (sx * sz > 0) * math.pi) if anim == "move" else 0.0
        leg = body * T(0.05 * sx, 0, 0.11 * sz) * RX(sw)
        sc.add(sd_hull([(-0.02, 0.0, -0.02), (0.02, 0.0, -0.02), (0.0, 0.0, 0.025), (0.0, -L, 0.0), (0.008, -L, 0.0)]), d, leg)
    nk = body * T(0, 0.1, 0.12) * RX(-0.3 if anim == "win" else 0.0)
    sc.add(sd_hull([(-0.04, 0.0, -0.03), (0.04, 0.0, -0.03), (0, 0.0, 0.04), (0, 0.2, 0.06), (0.02, 0.18, 0.04), (-0.02, 0.18, 0.04)]), m, nk)
    head = nk * T(0, 0.2, 0.06)
    sc.add(sd_hull([(-0.035, 0.0, -0.02), (0.035, 0.0, -0.02), (0, 0.04, 0.0), (0, -0.03, 0.14)]), m, head)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.009), Mat("#141414", rough=0.2), head * T(0.03 * sgn, 0.01, 0.03))
        ant = head * T(0.03 * sgn, 0.03, -0.01) * RZ(-0.35 * sgn)
        for (a0, a1) in (((0, 0, 0), (0, 0.18, -0.03)), ((0, 0.07, -0.01), (0.06 * sgn, 0.13, 0.02)), ((0, 0.12, -0.02), (0.07 * sgn, 0.2, -0.03)),
                         ((0, 0.16, -0.025), (-0.03 * sgn, 0.24, -0.02))):
            sc.add(sheet([a0, (a0[0] + 0.012, a0[1], a0[2]), a1], 0.005), paper("#e8dcc0"), ant)
    return sc


# ─── 8 · Paon ───────────────────────────────────────────────────────────────
def peacock(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t)
    blue = paper("#1e4aa8", "#3ac8b0")
    sc.add(sd_hull([(-0.08, 0.02, -0.08), (0.08, 0.02, -0.08), (0, 0.02, 0.1), (0, 0.22, 0.02), (0, 0.1, -0.12)]), blue, root)
    nk = root * T(0, 0.2, 0.04)
    sc.add(sd_hull([(-0.03, 0.0, -0.02), (0.03, 0.0, -0.02), (0, 0.0, 0.03), (0, 0.18, 0.04)]), blue, nk)
    head = nk * T(0, 0.18, 0.04)
    sc.add(sd_hull([(-0.025, 0.0, -0.02), (0.025, 0.0, -0.02), (0, 0.03, 0), (0, -0.01, 0.07)]), blue, head)
    for k in range(3):   # the crest
        sc.add(sheet([(0, 0.02, 0), (0.004, 0.02, 0), ((k - 1) * 0.025, 0.09, -0.01)], 0.003), paper("#e0b860", gold=True), head)
    spread = 1.0 if anim == "win" else 0.75 + 0.1 * math.sin(TAU * t)
    n = 13
    for i in range(n):   # the fan: feathers of green and gold paper, an eye on each
        a = (i / (n - 1) - 0.5) * math.pi * spread
        fx = root * T(0, 0.12, -0.1) * RZ(a) * RX(-0.35)
        col = paper("#1e8a6a", "#e0b860") if i % 2 else paper("#2a9a7a")
        sc.add(sheet([(-0.035, 0, 0), (0.035, 0, 0), (0.0, 0.5, 0)], 0.004), col, fx)
        sc.add(sheet([(-0.02, 0.38, 0.006), (0.02, 0.38, 0.006), (0.0, 0.46, 0.006)], 0.003), paper("#1e3a9a"), fx)
    gold_ring(sc, t, 3.0)
    return sc


# ─── 9 · Lion ───────────────────────────────────────────────────────────────
def lion(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t)
    m, mane = paper("#d8a04a"), paper("#a8602a", "#e0b860")
    L = 0.2
    body = root * T(0, L, 0)
    sc.add(sd_hull([(-0.1, 0.0, -0.22), (0.1, 0.0, -0.22), (-0.11, 0.0, 0.14), (0.11, 0.0, 0.14), (0, 0.16, -0.18), (0, 0.2, 0.1), (0, -0.07, 0.0)]), m, body)
    for sx, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        sw = 0.3 * math.sin(TAU * t * 2 + (sx * sz > 0) * math.pi) if anim == "move" else 0.0
        leg = body * T(0.07 * sx, 0, 0.12 * sz) * RX(sw)
        sc.add(sd_hull([(-0.035, 0.0, -0.03), (0.035, 0.0, -0.03), (0.0, 0.0, 0.035), (-0.02, -L, 0.01), (0.02, -L, 0.01), (0, -L, 0.04)]), m, leg)
    roar = anim in ("win", "blocked")
    head = body * T(0, 0.2, 0.17) * RX(-0.25 if roar else 0.0)
    for i in range(14):   # the mane: rays of folded paper around the head
        a = TAU * i / 14
        sc.add(sheet([(0, 0, 0), (0.05, 0, 0), (0.17 * math.cos(a), 0.17 * math.sin(a), -0.04)], 0.005), mane, head * RZ(a * 0.0) * T(-0.025 * math.cos(a), -0.025 * math.sin(a), -0.03))
    sc.add(sd_hull([(-0.08, -0.06, 0.0), (0.08, -0.06, 0.0), (-0.07, 0.07, 0.0), (0.07, 0.07, 0.0), (0, 0.0, 0.12), (0, -0.07, 0.09)]), m, head)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.012), Mat("#2a1a10", rough=0.2), head * T(0.035 * sgn, 0.025, 0.07))
    if roar:
        sc.add(sd_hull([(-0.03, -0.06, 0.08), (0.03, -0.06, 0.08), (0, -0.1, 0.07)]), Mat("#5a1a14", rough=0.6), head)
    sw = 0.3 * math.sin(TAU * t)
    sc.add(sheet([(0, 0.1, -0.2), (0, 0.12, -0.2), (0.05 * sw, 0.3, -0.38)], 0.006), m, body)
    sc.add(sd_hull([(0.05 * sw - 0.02, 0.29, -0.37), (0.05 * sw + 0.02, 0.29, -0.37), (0.05 * sw, 0.36, -0.4), (0.05 * sw, 0.3, -0.42)]), mane, body)
    gold_ring(sc, t, 4.0)
    sc.add(sd_sphere(0.0001), glow("#e0b860", 0.0), T(0, 0, 0))
    return sc


# ─── 10 · Dragon ────────────────────────────────────────────────────────────
def dragon(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t)
    red, gold_ = paper("#c8282a", "#e0b860"), paper("#e0b860", gold=True)
    flap = 0.35 * math.sin(TAU * t * (2 if anim in ("move", "win") else 0.7))
    # the body: segments along a curve rising from the tail to the neck
    path = [(-0.0, 0.05, -0.38), (0.12, 0.08, -0.2), (0.0, 0.2, 0.0), (-0.06, 0.38, 0.1), (0.0, 0.56, 0.16)]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        r0, r1 = 0.03 + 0.03 * i, 0.03 + 0.03 * (i + 1) if i < 2 else 0.06 - 0.01 * (i - 2)
        pts = []
        for (c, r) in ((a, r0), (b, r1)):
            pts += [(c[0] + r, c[1], c[2]), (c[0] - r, c[1], c[2]), (c[0], c[1] + r, c[2]), (c[0], c[1] - r * 0.6, c[2])]
        sc.add(sd_hull(pts), red, root)
        sc.add(sheet([(a[0], a[1] + r0, a[2]), (b[0], b[1] + r1, b[2]), ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.08, (a[2] + b[2]) / 2)], 0.004), gold_, root)   # spines
    hd = path[-1]
    head = root * T(*hd) * RX(-0.3 if anim == "win" else 0.1)
    sc.add(sd_hull([(-0.06, -0.03, -0.04), (0.06, -0.03, -0.04), (-0.05, 0.05, -0.03), (0.05, 0.05, -0.03), (0, -0.02, 0.18), (0, 0.03, 0.15)]), red, head)
    for sgn in (-1, 1):
        sc.add(sheet([(0.03 * sgn, 0.04, -0.03), (0.05 * sgn, 0.04, -0.02), (0.08 * sgn, 0.18, -0.12)], 0.005), gold_, head)   # horns
        sc.add(sd_sphere(0.012), glow("#ffd23a", 4.0), head * T(0.04 * sgn, 0.035, 0.06))
        wing = root * T(0.0, 0.36, 0.06) * RY(0.25 * sgn) * RZ(sgn * (0.5 + flap))
        for k in range(3):   # the wing: three folded panels
            sc.add(sheet([(0, 0, 0), (0.42 * sgn, 0.12 - 0.07 * k, -0.05 * k), (0.38 * sgn, 0.02 - 0.1 * k, -0.12 - 0.05 * k)], 0.004),
                   red if k % 2 == 0 else paper("#a81e22"), wing)
        sc.add(sheet([(0, 0, 0), (0.01 * sgn, 0.01, 0), (0.44 * sgn, 0.14, -0.02)], 0.006), gold_, wing)
        sc.add(sheet([(0.06 * sgn, 0.2, 0.0), (0.1 * sgn, 0.2, 0.02), (0.12 * sgn, 0.02, 0.06)], 0.006), red, root)     # front legs
    if anim == "win":
        for k in range(3):
            sc.add(sheet([(0, 0, 0), (0.04, 0, 0), (0.0, 0.0, 0.16 + 0.05 * k)], 0.004), glow("#ffa02a", 6.0 - k), head * T(0, -0.02, 0.18) * RZ(k))
    gold_ring(sc, t, 5.0)
    sc.add(sd_sphere(0.0001), glow("#e0b860", 0.0), T(0, 0, 0))
    return sc


SCALE = {"random": 1.5, "beginner": 1.58, "monte_carlo": 1.65, "mcts": 1.72, "basic_minmax": 1.78, "nnue193": 1.82,
         "features": 1.88, "resnet": 1.8, "dev": 1.85, "alphazero": 1.85}   # the folds bigger and bigger (the peacock and the dragon touch the frame above that)
CAST = {"random": ("Bateau en papier", boat), "beginner": ("Grenouille", frog), "monte_carlo": ("Lapin", rabbit),
        "mcts": ("Renard", fox), "basic_minmax": ("Papillon", butterfly), "nnue193": ("Grue", crane),
        "features": ("Cerf", stag), "resnet": ("Paon", peacock), "dev": ("Lion", lion), "alphazero": ("Dragon de papier", dragon)}
ORIGAMI = {"ori_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"ori_" + k: n for k, (n, _) in CAST.items()}
