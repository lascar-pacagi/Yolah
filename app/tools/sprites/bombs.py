"""
bombs.py — the Bombes set: one explosive per AI, from the weakest to the most
powerful, bigger and bigger (cartoon objects, no real device):

  random        Pétard               a small red firecracker, its fuse fizzing
  beginner      Fusée d'artifice     a striped rocket on its stick
  monte_carlo   Bâton de dynamite    one red stick, a paper band, a lit fuse
  mcts          Paquet de dynamite   three sticks tied with rope
  basic_minmax  Grenade              olive, grooved, its lever and ring
  nnue193       Bombe à mèche        the round black cartoon bomb, a curly fuse
  features      Mine marine          a rusty sphere with its horns
  resnet        Bombe d'avion        a finned bomb lying on its wooden cradle
  dev           Missile              white and red, on its launch pad
  alphazero     Bombe nucléaire      a huge fat bomb, the radiation sign, a green glow

The app blows up the square left with an explosion of the same strength
(break fx "blast"). The human is never part of a set.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm
from models import base, BASE_TOP, TAU, glow
from heroic import chain, bezier, HEAD

ROPE = Mat("#b89a6a", rough=0.9)
STEEL = Mat("#8a8e94", metal=0.8, rough=0.35)
DARK_STEEL = Mat("#3a3c40", metal=0.7, rough=0.4)


def painted(c, rough=0.45, grime=0.25, k=30):
    def pat(pl, pw, n):
        return {"albedo": srgb(c) * (1 - grime * fbm(pl * k, 3))[:, None]}
    return Mat(c, rough=rough, clearcoat=0.3, pattern=pat)


def bands(c, band, y0, y1, rough=0.5):
    """A colour with a band (local y between y0 and y1) of another one: labels, stripes."""
    def pat(pl, pw, n):
        b = ((pl[:, 1] > y0) & (pl[:, 1] < y1)).float()[:, None]
        return {"albedo": srgb(c) * (1 - b) + srgb(band) * b}
    return Mat(c, rough=rough, pattern=pat)


def fuse(sc, pts, t, anim, r=0.008, spark=1.0):
    """A fuse along pts, burning at its end: a glowing tip and sparks flickering around it."""
    chain(sc, pts, [r] * len(pts), ROPE, T(), blend=0.004)
    tip = pts[-1]
    k = 1.6 if anim == "win" else 1.0
    sc.add(sd_sphere(r * 1.6), glow("#ffd060", 6.0 * k), T(*tip))
    for i in range(6):
        a = TAU * (i / 6) + t * 17 + i * 1.3
        u = (t * 9 + i * 0.37) % 1
        d = 0.03 * spark * k * (0.4 + u)
        p = (tip[0] + d * math.cos(a), tip[1] + d * (0.6 + 0.8 * math.sin(a * 1.7)), tip[2] + d * math.sin(a))
        sc.add(sd_sphere(0.006 * spark * (1 - u) + 0.002), glow("#ffe08a" if i % 2 else "#ff9a3a", 7.0), T(*p))


def bob(anim, t):
    """The pose of the object: a hop when it moves, a shake when blocked, a wobble when it wins."""
    if anim == "move":
        return T(0, 0.05 * abs(math.sin(TAU * t)), 0), 0.0
    if anim == "blocked":
        return T(0, 0, 0), 0.12 * math.sin(TAU * 3 * t)
    if anim == "win":
        return T(0, 0.03 * abs(math.sin(TAU * 2 * t)), 0), 0.08 * math.sin(TAU * 2 * t)
    return T(0, 0.004 * math.sin(TAU * t), 0), 0.0


def stand(anim, t):
    lift, tilt = bob(anim, t)
    return RY(HEAD) * lift * T(0, BASE_TOP, 0) * RZ(tilt)


# ─── 1 · Pétard ─────────────────────────────────────────────────────────────
def firecracker(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    sc.add(sd_cylinder(0.09, 0.045, 0.01), bands("#c8282a", "#f2d24a", 0.02, 0.045), S * T(0, 0.09, 0))
    sc.add(sd_cone(0.02, 0.045, 0.012, 0.004), Mat("#c8282a", rough=0.6), S * T(0, 0.2, 0))      # the twisted paper end
    p0 = S.point(0, 0.22, 0).tolist()
    fuse(sc, [tuple(p0), (p0[0] + 0.01, p0[1] + 0.04, p0[2]), (p0[0] + 0.03, p0[1] + 0.07, p0[2] + 0.01)], t, anim, 0.006, 0.7)
    return sc


# ─── 2 · Fusée d'artifice ───────────────────────────────────────────────────
def rocket(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)

    def stripes():
        def pat(pl, pw, n):
            s_ = (torch.sin(pl[:, 1] * 60) > 0).float()[:, None]
            return {"albedo": srgb("#2a5ac8") * s_ + srgb("#f2f0e8") * (1 - s_)}
        return Mat("#2a5ac8", rough=0.45, clearcoat=0.4, pattern=pat)
    sc.add(sd_cylinder(0.17, 0.055, 0.01), stripes(), S * T(0.04, 0.23, 0))
    sc.add(sd_cone(0.07, 0.057, 0.003, 0.002), Mat("#e8b82a", metal=0.4, rough=0.3), S * T(0.04, 0.47, 0))   # the cone
    sc.add(sd_capsule((0, 0, 0), (0, 0.5, 0), 0.009), Mat("#c8a070", rough=0.8), S * T(-0.03, 0.0, 0))   # the stick
    p0 = S.point(0.03, 0.06, 0).tolist()
    fuse(sc, [tuple(p0), (p0[0] + 0.03, p0[1] - 0.02, p0[2] + 0.02), (p0[0] + 0.06, p0[1] - 0.03, p0[2] + 0.03)], t, anim, 0.005, 0.8)
    for k in range(3):   # stars of the box on the base
        a = TAU * k / 3 + 0.7
        sc.add(sd_sphere(0.012), glow(["#ff5a7a", "#7ad8ff", "#ffe060"][k], 3.0 + 2 * math.sin(TAU * t + k)),
               T(0.22 * math.cos(a), BASE_TOP + 0.012, 0.22 * math.sin(a)))
    return sc


# ─── 3 · Bâton de dynamite ──────────────────────────────────────────────────
def dyn_stick(sc, xf, h=0.16, r=0.045):
    sc.add(sd_cylinder(h, r, 0.008), bands("#b8241e", "#e8dcc0", -0.05, 0.05, rough=0.6), xf)
    sc.add(sd_cylinder(0.004, r * 0.8, 0.002), Mat("#d8c8a0", rough=0.7), xf * T(0, h, 0))


def dynamite(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    dyn_stick(sc, S * T(0, 0.17, 0), 0.16, 0.05)
    p0 = S.point(0, 0.34, 0).tolist()
    fuse(sc, bezier(tuple(p0), (p0[0], p0[1] + 0.06, p0[2]), (p0[0] + 0.06, p0[1] + 0.08, p0[2]), (p0[0] + 0.08, p0[1] + 0.04, p0[2] + 0.02), 5), t, anim, 0.007)
    return sc


# ─── 4 · Paquet de dynamite ─────────────────────────────────────────────────
def bundle(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    for (x, z) in ((-0.055, 0.03), (0.055, 0.03), (0.0, -0.06)):
        dyn_stick(sc, S * T(x, 0.19, z), 0.18, 0.05)
    for y in (0.1, 0.28):
        sc.add(sd_torus(0.1, 0.012), ROPE, S * T(0, y, -0.0))
    p0 = S.point(0.0, 0.38, -0.06).tolist()
    fuse(sc, bezier(tuple(p0), (p0[0], p0[1] + 0.08, p0[2]), (p0[0] + 0.08, p0[1] + 0.1, p0[2] + 0.02), (p0[0] + 0.12, p0[1] + 0.05, p0[2] + 0.04), 6), t, anim, 0.008, 1.2)
    return sc


# ─── 5 · Grenade ────────────────────────────────────────────────────────────
def grenade(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)

    def grooves():
        def pat(pl, pw, n):
            a = torch.atan2(pl[:, 0], pl[:, 2])
            g = ((torch.sin(a * 8).abs() < 0.12) | (torch.sin(pl[:, 1] * 55).abs() < 0.15)).float()[:, None]
            return {"albedo": srgb("#4e5a2e") * (1 - 0.55 * g) * (0.9 + 0.15 * fbm(pl * 30, 2))[:, None]}
        return Mat("#4e5a2e", rough=0.6, pattern=pat)
    sc.add(sd_ellipsoid(0.12, 0.15, 0.12), grooves(), S * T(0, 0.15, 0))
    sc.add(sd_cylinder(0.03, 0.035, 0.006), STEEL, S * T(0, 0.31, 0))
    sc.add(sd_box(0.012, 0.11, 0.02, 0.005), STEEL, S * T(0.07, 0.24, 0) * RZ(-0.25))               # the lever
    pull = 0.04 * math.sin(TAU * t * 2) if anim == "win" else 0.0
    sc.add(sd_torus(0.035, 0.005), STEEL, S * T(-0.05 - pull, 0.33, 0) * RZ(math.pi / 2))          # the ring of the pin
    return sc


# ─── 6 · Bombe à mèche ──────────────────────────────────────────────────────
def round_bomb(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    sc.add(sd_sphere(0.19), Mat("#1e2026", metal=0.3, rough=0.25, clearcoat=0.8), S * T(0, 0.2, 0))
    sc.add(sd_cylinder(0.025, 0.06, 0.008), DARK_STEEL, S * T(0, 0.39, 0))
    p0 = S.point(0, 0.42, 0).tolist()
    fuse(sc, bezier(tuple(p0), (p0[0] + 0.0, p0[1] + 0.07, p0[2]), (p0[0] + 0.09, p0[1] + 0.1, p0[2] - 0.02), (p0[0] + 0.1, p0[1] + 0.03, p0[2] + 0.02), 7), t, anim, 0.01, 1.4)
    return sc


# ─── 7 · Mine marine ────────────────────────────────────────────────────────
def sea_mine(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    rust = Mat("#4a4038", metal=0.5, rough=0.6,
               pattern=lambda pl, pw, n: {"albedo": srgb("#3e3a36") * (1 - 0.6 * (fbm(pl * 14, 3) > 0.6).float())[:, None]
                                          + srgb("#8a4a22") * (0.6 * (fbm(pl * 14, 3) > 0.6).float())[:, None]})
    c = S * T(0, 0.28, 0)
    sc.add(sd_sphere(0.21), rust, c)
    sc.add(sd_torus(0.21, 0.012), DARK_STEEL, c)                                                       # the seam
    dirs = [(0, 1, 0), (0.94, 0.33, 0), (-0.94, 0.33, 0), (0, 0.33, 0.94), (0, 0.33, -0.94), (0.6, -0.4, 0.6), (-0.6, -0.4, 0.6),
            (0.6, -0.4, -0.6), (-0.6, -0.4, -0.6), (0.66, 0.66, 0.33), (-0.66, 0.66, -0.33)]
    for d in dirs:   # the horns
        n = math.sqrt(sum(v * v for v in d))
        d = tuple(v / n for v in d)
        a = tuple(0.19 * v for v in d)
        b = tuple(0.27 * v for v in d)
        sc.add(sd_capsule(a, b, 0.018, 0.014), DARK_STEEL, c, blend=0.01)
        sc.add(sd_sphere(0.019), STEEL, c * T(*b))
    sc.add(sd_cylinder(0.04, 0.05, 0.01), DARK_STEEL, S * T(0, 0.04, 0))                                # its foot
    if anim == "win":
        sc.add(sd_sphere(0.02), glow("#ff3a2a", 5.0 + 4 * math.sin(TAU * t * 4)), c * T(0, 0.0, 0.21))
    return sc


# ─── 8 · Bombe d'avion ──────────────────────────────────────────────────────
def aerial_bomb(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    wood = Mat("#8a6a44", rough=0.85, pattern=lambda pl, pw, n: {"albedo": srgb("#8a6a44") * (0.8 + 0.3 * torch.sin(pl[:, 0] * 200 + fbm(pl * 8, 2) * 6))[:, None]})
    for x in (-0.16, 0.16):   # the cradle: two wooden chocks under the body
        sc.add(sd_box(0.035, 0.05, 0.13, 0.006), wood, S * T(x, 0.05, 0))
    body = S * T(0, 0.22, 0) * RY(math.pi / 2) * RX(math.pi / 2)   # its axis along x
    olive = bands("#454c2a", "#e8c22a", 0.2, 0.235, rough=0.45)
    sc.add(sd_ellipsoid(0.15, 0.36, 0.15), olive, body)
    sc.add(sd_cone(0.05, 0.03, 0.012, 0.004), STEEL, body * T(0, 0.38, 0))                           # the nose fuse
    sc.add(sd_cone(0.1, 0.11, 0.06, 0.006), olive, body * T(0, -0.36, 0), blend=0.02)
    for k in range(4):   # the fins in a box
        a = k * math.pi / 2 + math.pi / 4
        sc.add(sd_box(0.004, 0.08, 0.1, 0.002), DARK_STEEL, body * T(0, -0.48, 0) * RY(a) * T(0, 0, 0.0))
    sc.add(sd_box(0.12, 0.004, 0.12, 0.002), DARK_STEEL, body * T(0, -0.43, 0), op="add")
    return sc


# ─── 9 · Missile ────────────────────────────────────────────────────────────
def missile(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    sc.add(sd_cylinder(0.03, 0.2, 0.01), DARK_STEEL, S * T(0, 0.03, 0))                                # the pad
    for k in range(4):
        a = k * math.pi / 2
        sc.add(sd_box(0.012, 0.08, 0.012, 0.003), STEEL, S * T(0.15 * math.cos(a), 0.12, 0.15 * math.sin(a)))
    body = S * T(0, 0.12, 0)
    white = bands("#eceae4", "#1e1e22", 0.42, 0.45, rough=0.35)
    sc.add(sd_cylinder(0.27, 0.065, 0.01), white, body * T(0, 0.3, 0))
    sc.add(sd_cone(0.12, 0.065, 0.004, 0.003), Mat("#c8282a", rough=0.35, clearcoat=0.6), body * T(0, 0.69, 0))
    for k in range(4):   # fins at the tail and in the middle
        a = k * math.pi / 2 + 0.4
        sc.add(sd_box(0.004, 0.06, 0.07, 0.002), Mat("#c8282a", rough=0.4), body * RY(a) * T(0, 0.08, 0.1))
        sc.add(sd_box(0.003, 0.03, 0.035, 0.002), Mat("#eceae4", rough=0.4), body * RY(a + math.pi / 4) * T(0, 0.38, 0.08))
    if anim in ("win", "move"):
        f = 0.8 + 0.2 * math.sin(TAU * t * 6)
        sc.add(sd_cone(0.08 * f, 0.05, 0.005, 0.0), glow("#ffb040", 6.0), body * T(0, -0.02, 0) * RX(math.pi))
    return sc


# ─── 10 · Bombe nucléaire ───────────────────────────────────────────────────
def nuke(anim, t, side):
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    glow_k = 2.5 + 1.5 * math.sin(TAU * t) + (3.0 if anim == "win" else 0.0)
    sc.add(sd_torus(0.5, 0.012), glow("#7aff6a", glow_k), T(0, BASE_TOP + 0.005, 0))
    for z in (-0.18, 0.18):   # its cradle
        sc.add(sd_box(0.2, 0.05, 0.035, 0.008), DARK_STEEL, S * T(0, 0.05, z))

    def body_mat():
        def pat(pl, pw, n):
            a = srgb("#a8903a") * (0.85 + 0.2 * fbm(pl * 12, 3))[:, None]
            a = a * (1 - 0.6 * ((pl[:, 2].abs() < 0.012)).float()[:, None])          # the seam round the middle
            # the radiation sign on its side (the local x face): a black disc, three yellow blades, a yellow centre
            u, v = pl[:, 1], pl[:, 2]
            r = torch.sqrt(u * u + v * v)
            th = torch.atan2(v, u)
            blade = ((torch.cos(3 * th) > 0.5) & (r > 0.035) & (r < 0.11)).float()
            sign = ((r < 0.13) & (pl[:, 0] < 0)).float()
            ylw = (blade + (r < 0.025).float()).clamp(0, 1) * sign
            a = a * (1 - sign[:, None]) + srgb("#141414") * sign[:, None]
            a = a * (1 - ylw[:, None]) + srgb("#f2d020") * ylw[:, None]
            return {"albedo": a}
        return Mat("#c8a83a", rough=0.4, clearcoat=0.5, pattern=pat)
    body = S * T(0, 0.3, 0) * RY(math.pi / 2)
    sc.add(sd_ellipsoid(0.24, 0.25, 0.42), body_mat(), body)
    sc.add(sd_box(0.17, 0.17, 0.08, 0.02), DARK_STEEL, body * T(0, 0, -0.5))                         # the box tail
    sc.add(sd_box(0.15, 0.15, 0.06, 0.02), Mat("#141414"), body * T(0, 0, -0.5), op="sub")
    for k in range(4):
        a = k * math.pi / 2
        sc.add(sd_box(0.004, 0.16, 0.08, 0.002), DARK_STEEL, body * T(0, 0, -0.5) * RZ(a + math.pi / 4))
    for k in range(5):   # bolts round the middle
        a = TAU * k / 10 - 0.9
        sc.add(sd_sphere(0.012), STEEL, body * T(0.245 * math.cos(a) * 0.98, 0.25 * math.sin(a), 0.02))
    return sc


SCALE = {"random": 1.15, "beginner": 1.1, "monte_carlo": 1.15, "mcts": 1.15, "basic_minmax": 1.15, "nnue193": 1.15,
         "features": 1.15, "resnet": 1.1, "dev": 1.1, "alphazero": 1.1}
CAST = {"random": ("Pétard", firecracker), "beginner": ("Fusée d'artifice", rocket), "monte_carlo": ("Bâton de dynamite", dynamite),
        "mcts": ("Paquet de dynamite", bundle), "basic_minmax": ("Grenade", grenade), "nnue193": ("Bombe à mèche", round_bomb),
        "features": ("Mine marine", sea_mine), "resnet": ("Bombe d'avion", aerial_bomb), "dev": ("Missile", missile),
        "alphazero": ("Bombe nucléaire", nuke)}
BOMBS = {"bomb_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"bomb_" + k: n for k, (n, _) in CAST.items()}
