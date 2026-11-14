"""
shonen.py — the Manga set, fourth version: one character per AI, rendered like
painted anime figures (PVC statues: saturated paint, varnish, cloth, fur), not
in the toon mode. From a lone wolf to a dragon god, bigger and more powerful:

  random        Ōkami       a lean grey wolf, scarred, a sacred rope round its neck
  beginner      Shinobi     a ninja crouched to spring, a long red scarf, a kunai
  monte_carlo   Kitsune     a white fox spirit, red markings, three tails of blue foxfire
  mcts          Samouraï    red lacquered armour laced in black, a golden crescent, katana on guard
  basic_minmax  Oni         a huge red ogre, ivory horns, a studded iron club, a tiger skin
  minmax        Tengu       red face, the long nose, crow wings, the feather fan, the ascetic's robe
  mcts_nnue     Nue         monkey face, tanuki body, tiger legs, a snake tail, a black thunder cloud
  nnue193       Onmyōji     a sorcerer in white, a lacquered hat, paper talismans, a star seal of light
  features      Raijū       a great thunder wolf, a mane of lightning, arcs around it
  resnet        Hōō         the phoenix, wings of red and gold feathers, long eyed plumes, embers
  dev           Kaigen      a martial artist at full power, golden hair, an energy sphere, shards of rock lifted
  alphazero     Tenryū      the celestial dragon coiled round a pillar of light, a pearl in its claw

Power shows as flames of aura, lightning and rocks torn from the ground, more
and more for the stronger ones. The human is never part of a set.
"""
import math
import random
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, sd_plane_cut, fbm
from models import base, BASE_TOP, TAU, glow, gold
from legends import cloth, iron, silver, wood, skin, base_pose, held
from heroic import adult, chain, bezier, scales, HEAD
from arctic import quadruped, fur, EYE
from kaiju import lightning
from origami import sd_hull


# ─── materials ──────────────────────────────────────────────────────────────
def lacquer(c, rough=0.22):
    return Mat(c, rough=rough, clearcoat=0.9, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.9 + 0.12 * fbm(pl * 20, 2))[:, None]})


def hair_mat(c, tip=None):
    """Glossy painted hair, a little lighter towards the tips (local y)."""
    tc = tip or c

    def pat(pl, pw, n):
        u = torch.clamp(pl[:, 1] * 4 + 0.5, 0, 1)[:, None]
        return {"albedo": srgb(c) * (1 - u) + srgb(tc) * u}
    return Mat(c, rough=0.3, clearcoat=0.7, pattern=pat)


def thin_ellipsoid(rx, ry, rz):
    """A flat ellipsoid bounded by its sphere (shadow rays would stop short far from it)."""
    f, R = sd_ellipsoid(rx, ry, rz), max(rx, ry, rz)
    return lambda p: torch.maximum(f(p), torch.linalg.norm(p, dim=-1) - R)


STONE = Mat("#6e6660", rough=0.85, pattern=lambda pl, pw, n: {"albedo": srgb("#6e6660") * (0.7 + 0.45 * fbm(pl * 30, 3))[:, None]})
INKM = Mat("#14121a", rough=0.4)


# ─── faces and hair ─────────────────────────────────────────────────────────
def anime_face(sc, x, r, sk, eye_c="#3a6ad0", brow_c="#1a1418", expr="calm", glow_eyes=0.0):
    """An adult anime face: narrow chin, small nose, long almond eyes with a
    glossy iris, angled brows. expr: calm, fierce, shout."""
    sc.add(sd_ellipsoid(r * 0.88, r * 1.0, r * 0.95), sk, x)
    sc.add(sd_ellipsoid(r * 0.55, r * 0.52, r * 0.62), sk, x * T(0, -r * 0.5, r * 0.28), blend=r * 0.28)    # the chin
    sc.add(sd_capsule((0, r * 0.1, 0), (0, -r * 0.12, r * 0.07), r * 0.06, r * 0.08), sk, x * T(0, -r * 0.12, r * 0.88), blend=r * 0.05)
    tilt = {"calm": 0.1, "fierce": 0.4, "shout": 0.5}.get(expr, 0.1)
    for sgn in (-1, 1):
        ec = x * T(r * 0.33 * sgn, r * 0.02, r * 0.8) * RY(0.3 * sgn) * RZ(-tilt * 0.4 * sgn)
        sc.add(thin_ellipsoid(r * 0.2, r * (0.09 if expr != "calm" else 0.11), r * 0.06), Mat("#f4f0ea", rough=0.25), ec)
        sc.add(sd_sphere(r * 0.085), glow(eye_c, glow_eyes) if glow_eyes > 0 else Mat(eye_c, rough=0.1, clearcoat=1.0), ec * T(r * 0.02 * sgn, 0, r * 0.02))
        sc.add(sd_sphere(r * 0.04), INKM, ec * T(r * 0.02 * sgn, 0, r * 0.07))
        sc.add(sd_capsule((-r * 0.22, 0, 0), (r * 0.22, 0, 0), r * 0.025), INKM, ec * T(0, r * 0.1, r * 0.03))      # the lash line
        sc.add(sd_capsule((-r * 0.18, 0, 0), (r * 0.18, 0, 0), r * 0.035), Mat(brow_c, rough=0.4),
               x * T(r * 0.33 * sgn, r * 0.26, r * 0.84) * RY(0.3 * sgn) * RZ(-tilt * sgn))
    mo = x * T(0, -r * 0.45, r * 0.86)
    if expr == "shout":
        sc.add(sd_ellipsoid(r * 0.2, r * 0.14, r * 0.08), Mat("#5a1420", rough=0.5), mo)
        sc.add(sd_box(r * 0.16, r * 0.025, r * 0.05, r * 0.01), Mat("#ffffff", rough=0.3), mo * T(0, r * 0.1, r * 0.02))
    else:
        sc.add(sd_capsule((-r * 0.12, 0, 0), (r * 0.12, -r * 0.01 * (expr == "fierce"), 0), r * 0.018), Mat("#7a3a3a", rough=0.5), mo * T(0, 0, r * 0.04))


def spiky_hair(sc, x, r, mat, n=14, length=1.0, up=0.6, seed=0, bangs=5, cap=True):
    """Hair in long tapered spikes swept up and back, a full cap under them, bangs over the brow."""
    rng = random.Random(seed)
    if cap:
        sc.add(sd_plane_cut(sd_ellipsoid(r * 0.95, r * 1.06, r * 1.02), (0, -0.45, 1), r * 0.35), mat, x * T(0, r * 0.08, -r * 0.03))
    for i in range(n):
        th = -2.5 + 5.0 * i / (n - 1) + rng.uniform(-0.15, 0.15)       # around the head, from the left over the back to the right
        ph = up + rng.uniform(-0.25, 0.35)
        L = r * (0.9 + 0.7 * rng.random()) * length
        d = (math.sin(th) * math.cos(ph), math.sin(ph), -math.cos(th) * math.cos(ph) * 0.9 - 0.25)
        nd = math.sqrt(sum(v * v for v in d))
        d = tuple(v / nd for v in d)
        a = (d[0] * r * 0.7, d[1] * r * 0.7 + r * 0.15, d[2] * r * 0.7)
        b = (a[0] + d[0] * L, a[1] + d[1] * L, a[2] + d[2] * L)
        sc.add(sd_capsule(a, b, r * (0.26 + 0.06 * rng.random()), r * 0.02), mat, x, blend=r * 0.06)
    for k in range(bangs):   # bangs falling over the brow, swept
        u = (k - (bangs - 1) / 2) / max(1, bangs - 1)
        a = (u * r * 1.2, r * 0.75, r * 0.65)
        b = (u * r * 1.5 + r * 0.15, r * 0.25, r * 0.98)
        sc.add(sd_capsule(a, b, r * 0.17, r * 0.02), mat, x, blend=r * 0.05)


# ─── powers ─────────────────────────────────────────────────────────────────
def flame_aura(sc, t, col, n, h, rad, k=2.2, y0=0.0, xf=None, seed=0):
    """The aura: a ring of light on the base, motes of energy rising round the
    figure and fading, and for the strongest a few long thin threads of light."""
    xf = xf or RY(HEAD)
    rng = random.Random(seed)
    sc.add(sd_torus(rad * 1.05, 0.006), glow(col, k * 1.3), xf * T(0, BASE_TOP + y0 + 0.004, 0))
    for i in range(n):
        a = TAU * i / n + rng.uniform(-0.3, 0.3)
        rr = rad * rng.uniform(0.75, 1.05)
        ph = (t + rng.random()) % 1
        y = BASE_TOP + y0 + h * 1.3 * ph
        fade = math.sin(math.pi * ph)
        sc.add(sd_sphere(0.008 * (0.5 + fade)), glow(col, k * 1.5 * fade + 0.2), xf * T(rr * math.cos(a + ph * 0.6), y, rr * math.sin(a + ph * 0.6)))
    if h >= 0.4:
        for i in range(n // 4):
            a = TAU * (i + 0.5) / (n // 4) + 0.3 * math.sin(TAU * t + i)
            ph = (t * 0.7 + i * 0.29) % 1
            y_lo = BASE_TOP + y0 + 0.05 + h * 0.5 * ph
            sc.add(sd_capsule((rad * math.cos(a), y_lo, rad * math.sin(a)), (rad * 0.85 * math.cos(a + 0.2), y_lo + h * 0.45, rad * 0.85 * math.sin(a + 0.2)), 0.003, 0.001),
                   glow(col, k * 1.2 * math.sin(math.pi * ph) + 0.1), xf)


def shards(sc, n, rad, h0, t, xf, seed=0, size=0.045):
    """Shards of rock torn from the ground and floating: faceted, with dust under them."""
    rng = random.Random(seed)
    for i in range(n):
        pts = []
        for _ in range(7):
            v = (rng.uniform(-1, 1), rng.uniform(-0.7, 0.7), rng.uniform(-1, 1))
            nv = math.sqrt(sum(c * c for c in v)) or 1
            s_ = size * rng.uniform(0.6, 1.2)
            pts.append(tuple(c / nv * s_ for c in v))
        a = TAU * (i / n + t * 0.15) + seed
        y = h0 + 0.25 * ((i * 0.37) % 1) + 0.04 * math.sin(TAU * (t + i * 0.3))
        q = (rad * math.cos(a), BASE_TOP + y, rad * math.sin(a))
        sc.add(sd_hull(pts), STONE, xf * T(*q) * RY(a * 2 + t) * RX(i + t * 2))
        for j in range(2):   # crumbs falling
            sc.add(sd_sphere(0.006), STONE, xf * T(q[0] + 0.02 * (j - 0.5), q[1] - 0.05 - 0.08 * ((t * 2 + j * 0.5 + i * 0.2) % 1), q[2]))


def sacred_rope(sc, xf, R, t):
    """A shimenawa: a twisted rope with white zigzag paper (shide) hanging."""
    sc.add(sd_torus(R, R * 0.16), Mat("#d8c08a", rough=0.85, pattern=lambda pl, pw, n: {
        "albedo": srgb("#d8c08a") * (0.75 + 0.3 * torch.sin(torch.atan2(pl[:, 2], pl[:, 0]) * 30 + pl[:, 1] * 200))[:, None]}), xf)
    for k in range(3):
        a = -0.6 + 0.6 * k
        p = xf * T(R * math.sin(a) * 1.05, -R * 0.1, R * math.cos(a) * 1.05) * RY(a)
        for j in range(3):
            sc.add(sd_box(R * 0.14, R * 0.12, R * 0.01, R * 0.005), Mat("#f6f4ee", rough=0.6), p * T(R * 0.06 * (-1) ** j, -R * (0.15 + 0.22 * j), 0))


# ─── 1 · Ōkami ──────────────────────────────────────────────────────────────
def okami(anim, t, side):
    sc = Scene()
    base(sc, side)
    fm, bm = fur("#565a64", 150, 0.7), fur("#b4b8c0", 150, 0.6)
    root, head = quadruped(sc, anim, t, fm, bm, s=1.05, leg=1.05, bulk=0.95, snout=1.15, ears=1.15)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.012), Mat("#f2c23a", rough=0.1, clearcoat=1.0), head * T(0.032 * sgn, 0.02, 0.055))
    sc.add(sd_capsule((0, 0, 0), (0.0, -0.045, 0.03), 0.0045), Mat("#a85050", rough=0.6), head * T(0.034, 0.05, 0.045))   # a scar over the eye
    tail = bezier((0, 0.03, -0.24), (0, 0.06, -0.34), (0.02, -0.02, -0.42), (0.04, -0.1, -0.44), 6)
    chain(sc, tail, [0.035, 0.045, 0.048, 0.042, 0.032, 0.014], fm, root, blend=0.02)
    sacred_rope(sc, root * T(0, 0.08, 0.2) * RX(-0.9), 0.07, t)
    return sc


# ─── 2 · Shinobi ────────────────────────────────────────────────────────────
def shinobi(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.lean += 0.08
    p.step = 0.6 if anim != "move" else p.step
    p.r = [1.5, 0.35, 1.7]           # the kunai raised before the face
    p.l = [-0.5, 0.45, 0.4]          # the other arm swept back
    if anim == "win":
        p.r = [2.6, 0.2, 0.15]
    navy = cloth("#1c2236", 0.8)
    wrap = cloth("#2a3048", 0.8)
    m = {"skin": skin("#e2bea2"), "top": navy, "legs": navy, "boots": wrap, "arms": navy, "hands": wrap, "neck": navy}
    rig = adult(sc, p, m, H=1.0, bulk=0.9)
    x, r = rig["head"], rig["hr"]
    sc.add(sd_ellipsoid(r * 1.04, r * 1.08, r * 1.04), navy, x)                                              # the hood
    sc.add(sd_box(r * 0.75, r * 0.15, r * 0.2, r * 0.07), skin("#e2bea2"), x * T(0, r * 0.12, r * 0.86))    # the eye slit
    for sgn in (-1, 1):
        sc.add(thin_ellipsoid(r * 0.15, r * 0.05, r * 0.03), INKM, x * T(r * 0.3 * sgn, r * 0.13, r * 1.04) * RZ(-0.3 * sgn))
    sc.add(sd_box(r * 0.62, r * 0.12, r * 0.05, r * 0.03), silver(0.25), x * T(0, r * 0.48, r * 0.92) * RX(-0.25))   # the forehead plate
    red = cloth("#c41e2a", 0.7)
    sc.add(sd_torus(0.055, 0.02), red, rig["torso"] * T(0, 0.31, 0.0))                                      # the scarf, two long tails flowing
    for k in range(2):
        pts = []
        for i in range(7):
            u = i / 6
            pts.append((-0.05 - 0.3 * u, 0.3 - 0.1 * u - 0.03 * k + 0.03 * math.sin(TAU * (t * 2 - u + k * 0.3)), -0.06 - 0.1 * u - 0.03 * k))
        chain(sc, pts, [0.016] * 7, red, rig["torso"], blend=0.006)
    sc.add(sd_torus(0.085, 0.014), red, rig["waist"])
    kn = held(rig["r"], -1.3 if anim != "win" else -math.pi)
    sc.add(sd_capsule((0, -0.04, 0), (0, 0.03, 0), 0.008), wrap, kn)
    sc.add(sd_torus(0.012, 0.004), silver(), kn * T(0, -0.055, 0))
    sc.add(sd_cone(0.09, 0.022, 0.001, 0.001), silver(0.12), kn * T(0, 0.125, 0))
    for k in range(2):   # two shuriken stuck in the base
        a = 2.0 + k * 1.2
        sh = T(0.33 * math.cos(a), BASE_TOP + 0.02, 0.33 * math.sin(a)) * RX(0.5) * RY(a)
        for j in range(4):
            sc.add(sd_box(0.006, 0.03, 0.002, 0.001), silver(0.2), sh * RZ(j * math.pi / 2) * T(0, 0.02, 0))
    return sc


# ─── 3 · Kitsune ────────────────────────────────────────────────────────────
def kitsune(anim, t, side):
    sc = Scene()
    base(sc, side)
    fm, bm = fur("#f2f0ea", 150, 0.4), fur("#ffffff", 150, 0.3)
    root, head = quadruped(sc, anim, t, fm, bm, s=1.05, leg=1.1, bulk=0.85, snout=1.3, ears=1.45)
    red = Mat("#d81e3a", rough=0.4)
    for sgn in (-1, 1):   # kumadori markings and narrow eyes
        sc.add(sd_capsule((0, 0, 0), (0.03 * sgn, 0.035, -0.025), 0.0045), red, head * T(0.02 * sgn, 0.035, 0.055))
        sc.add(thin_ellipsoid(0.014, 0.004, 0.006), Mat("#d8202a", rough=0.1, emissive="#ff3a3a", emit=0.6), head * T(0.03 * sgn, 0.02, 0.058) * RZ(-0.45 * sgn))
    sc.add(sd_sphere(0.008), red, head * T(0, 0.055, 0.03))
    for k, a in enumerate((-0.75, 0.0, 0.75)):   # three bushy tails, curling up, tipped with foxfire
        sw = 0.15 * math.sin(TAU * (t + k * 0.3))
        tl = root * T(0, 0.03, -0.2) * RZ(-a * 0.8 + sw) * RX(-1.0)
        pts = bezier((0, 0, 0), (0, 0.12, -0.04), (0, 0.24, 0.0), (0, 0.32, 0.07), 6)
        chain(sc, pts, [0.03, 0.05, 0.055, 0.05, 0.04, 0.02], fm, tl, blend=0.025)
        for j in range(4):
            sc.add(sd_cone(0.04 + 0.015 * math.sin(TAU * t * 3 + k + j), 0.012, 0.001, 0.001), glow("#4ac8ff", 2.5),
                   tl * T(0.01 * (j - 1.5), 0.36, 0.08) * RZ(0.3 * (j - 1.5)) * T(0, 0.03, 0))
    for k in range(3):   # kitsunebi: wisps of foxfire around
        a = TAU * (t * 0.5 + k / 3)
        c = RY(HEAD) * T(0.34 * math.cos(a), BASE_TOP + 0.3 + 0.06 * math.sin(2 * a), 0.34 * math.sin(a))
        sc.add(sd_sphere(0.018), glow("#7ad8ff", 3.0), c)
        sc.add(sd_cone(0.03, 0.014, 0.001, 0.001), glow("#4ac8ff", 2.0), c * T(0, 0.03, 0))
    return sc


# ─── 4 · Samouraï ───────────────────────────────────────────────────────────
def samurai(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.step = 0.35 if anim != "move" else p.step
    p.r = [1.0, 0.1, 0.9]
    p.l = [1.0, -0.15, 1.0]          # both hands on the katana, on guard
    if anim == "win":
        p.r = [2.7, 0.1, 0.2]
        p.l = [2.7, -0.05, 0.2]

    def lamellar(c):
        def pat(pl, pw, n):
            band = (torch.sin(pl[:, 1] * 110) > 0.75).float()[:, None]
            lace = ((torch.sin(pl[:, 1] * 110) > 0.2) & (torch.sin(pl[:, 0] * 160) > 0.85)).float()[:, None]
            a = srgb(c) * (1 - 0.45 * band)
            return {"albedo": a * (1 - lace) + srgb("#141218") * lace}
        return Mat(c, rough=0.25, clearcoat=0.9, pattern=pat)
    lac = lamellar("#a8141e")
    blk = cloth("#1a1a20", 0.7)
    trim = gold()
    m = {"skin": skin("#e2bea2"), "top": lac, "legs": cloth("#2a2430"), "boots": blk, "arms": blk, "fore": lacquer("#1e1e24"), "hands": blk}
    rig = adult(sc, p, m, H=1.04, bulk=1.15, head=1.0)
    for sgn in (-1, 1):   # sode: big shoulder plates in rows
        for k in range(4):
            sc.add(sd_box(0.072, 0.012, 0.065, 0.006), lac, rig["torso"] * T(0.2 * sgn, 0.31 - 0.032 * k, 0) * RZ(-0.4 * sgn))
    for k in range(4):    # kusazuri: the skirt of plates
        sc.add(sd_cone(0.035, 0.115 + 0.012 * k, 0.105 + 0.012 * k, 0.006), lac, rig["waist"] * T(0, -0.07 - 0.032 * k, 0))
    sc.add(sd_torus(0.095, 0.01), trim, rig["waist"] * T(0, -0.03, 0))
    x, r = rig["head"], rig["hr"]
    anime_face(sc, x, r, skin("#e2bea2"), "#3a2a20", expr="fierce" if anim != "idle" else "calm")
    sc.add(sd_box(r * 0.75, r * 0.4, r * 0.3, r * 0.15), lacquer("#1e1e24"), x * T(0, -r * 0.5, r * 0.72))    # menpō, the half mask
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.18, r * 1.05, r * 1.18), (0, -1, 0), -r * 0.08), lacquer("#1e1e24"), x * T(0, r * 0.22, -r * 0.05))   # the kabuto
    for k in range(3):   # the neck guard in three rows
        sc.add(sd_cone(r * 0.12, r * (1.25 + 0.15 * k), r * (1.15 + 0.15 * k), r * 0.05), lac, x * T(0, r * (0.15 - 0.22 * k), -r * 0.1))
    for sgn in (-1, 1):   # the golden crescent
        sc.add(sd_capsule((0, 0, 0), (r * 1.2 * sgn, r * 1.3, r * 0.15), r * 0.13, r * 0.025), trim, x * T(0, r * 0.75, r * 0.8))
    kt = held(rig["r"], -math.pi if anim == "win" else -1.0, -0.1)
    sc.add(sd_capsule((0, -0.13, 0), (0, 0.03, 0), 0.012), Mat("#2a1a2a", rough=0.6, pattern=lambda pl, pw, n: {
        "albedo": srgb("#2a1a2a") * (0.6 + 0.6 * (torch.sin(pl[:, 1] * 300 + pl[:, 0] * 300) > 0).float())[:, None]}), kt)   # the wrapped hilt
    sc.add(sd_cylinder(0.006, 0.028, 0.002), trim, kt * T(0, 0.04, 0))
    sc.add(sd_box(0.011, 0.27, 0.003, 0.002), silver(0.08), kt * T(0, 0.31, 0) * RZ(-0.04))
    if anim == "win":
        sc.add(sd_box(0.004, 0.25, 0.0035, 0.001), glow("#ffffff", 3.0), kt * T(0.006, 0.31, 0) * RZ(-0.04))
    return sc


# ─── 5 · Oni ────────────────────────────────────────────────────────────────
def oni(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.6, 0.4, 0.8]
    p.l = [0.3, 0.5, 0.6]
    if anim == "win":
        p.r = [2.6, 0.25, 0.2]
        p.l = [1.8, 0.7, 0.4]
    red = Mat("#c4302a", rough=0.5, sss=0.3, clearcoat=0.3, pattern=lambda pl, pw, n: {"albedo": srgb("#c4302a") * (0.82 + 0.25 * fbm(pl * 25, 3))[:, None]})
    tiger = Mat("#e8a020", rough=0.8, pattern=lambda pl, pw, n: {
        "albedo": torch.where((torch.sin(pl[:, 0] * 70 + pl[:, 1] * 25 + fbm(pl * 6, 2) * 4) > 0.5)[:, None],
                              srgb("#1a1410").expand_as(pl), srgb("#e8a020").expand_as(pl))})
    m = {"skin": red, "top": red, "legs": tiger, "boots": red}
    rig = adult(sc, p, m, H=1.18, bulk=1.7, head=1.05, hunch=0.18)
    for sgn in (-1, 1):   # muscles: pectorals and abdomen
        sc.add(sd_ellipsoid(0.08, 0.07, 0.035), red, rig["chest"] * T(0.06 * sgn, 0.0, -0.005), blend=0.02)
    x, r = rig["head"], rig["hr"]
    sc.add(sd_ellipsoid(r, r, r), red, x)
    sc.add(sd_ellipsoid(r * 0.95, r * 0.55, r * 0.8), red, x * T(0, -r * 0.5, r * 0.25), blend=r * 0.3)
    sc.add(sd_ellipsoid(r * 0.95, r * 0.2, r * 0.32), red, x * T(0, r * 0.25, r * 0.76), blend=r * 0.15)     # a heavy brow
    sc.add(sd_ellipsoid(r * 0.25, r * 0.2, r * 0.2), red, x * T(0, -r * 0.08, r * 0.98), blend=r * 0.1)
    roar = anim in ("win", "blocked")
    for sgn in (-1, 1):
        sc.add(sd_sphere(r * 0.12), Mat("#ffe02a", rough=0.1, emissive="#ffd02a", emit=1.2), x * T(r * 0.35 * sgn, r * 0.08, r * 0.84))
        sc.add(sd_cone(r * 0.25, r * 0.08, r * 0.01, 0.002), Mat("#f6f0e0", rough=0.3), x * T(r * 0.3 * sgn, -r * 0.62, r * 0.92) * T(0, r * 0.2, 0))   # fangs up
        hp = bezier((0, 0, 0), (r * 0.25 * sgn, r * 0.5, 0), (r * 0.45 * sgn, r * 1.0, -r * 0.2), (r * 0.35 * sgn, r * 1.5, -r * 0.05), 6)
        chain(sc, hp, [r * 0.22, r * 0.19, r * 0.15, r * 0.11, r * 0.07, r * 0.015], Mat("#efe6cc", rough=0.35, clearcoat=0.5), x * T(r * 0.42 * sgn, r * 0.6, 0), blend=0.003)
    sc.add(sd_ellipsoid(r * 0.5, r * (0.25 if roar else 0.12), r * 0.2), Mat("#3a0a0a", rough=0.6), x * T(0, -r * 0.55, r * 0.86))
    spiky_hair(sc, x, r, hair_mat("#f2f2f2", "#d8d8e0"), n=12, length=1.1, up=0.3, seed=5, bangs=0, cap=False)
    kb = held(rig["r"], -math.pi if anim == "win" else -0.9)
    sc.add(sd_capsule((0, -0.06, 0), (0, 0.58, 0), 0.026, 0.065), iron("#2e3036", 0.4), kb)
    for k in range(6):
        for j in range(5):
            a = TAU * j / 5 + k * 0.6
            rr = 0.026 + (0.065 - 0.026) * (0.25 + 0.065 * k + 0.06) / 0.64
            sc.add(sd_cone(0.028, 0.014, 0.001, 0.001), silver(0.3), kb * T(rr * math.cos(a), 0.2 + 0.065 * k, rr * math.sin(a)) * RY(-a) * RZ(-math.pi / 2) * T(0, 0.012, 0))
    flame_aura(sc, t, "#ff6a2a", 9, 0.25, 0.3, 1.6, seed=5)
    return sc


# ─── 5b · Tengu (minmax) ────────────────────────────────────────────────────
def tengu(anim, t, side):
    """A daitengu, the great mountain goblin: a red face with the very long
    nose, fierce eyes, a white mane and beard, the small black cap (tokin) of
    the mountain ascetics, their robe with the big pompons (ochre and white),
    black crow wings, a fan of feathers that raises the wind, one-toothed
    geta."""
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.9, 0.3, 1.1]            # the fan raised before the chest
    p.l = [0.4, 0.4, 0.8]
    if anim == "win":
        p.r = [2.6, 0.25, 0.15]
    red = Mat("#c42a22", rough=0.4, sss=0.3, clearcoat=0.4)
    ochre = cloth("#d89a3a", 0.75)
    white = cloth("#f2f0ea", 0.75)
    m = {"skin": red, "top": white, "legs": ochre, "boots": lacquer("#1a1418"), "arms": white}
    rig = adult(sc, p, m, H=1.08, bulk=1.15, head=1.05)
    sc.add(sd_cone(0.2, 0.16, 0.09, 0.02), ochre, rig["waist"] * T(0, -0.16, 0))                     # hakama
    for k, y in enumerate((0.27, 0.17, 0.07)):   # the pompons of the yamabushi down the chest
        for sgn in (-1, 1):
            sc.add(sd_sphere(0.022), fur("#f2f0ea", 140, 0.3) if k % 2 else Mat("#e8a020", rough=0.9), rig["torso"] * T(0.05 * sgn, y, 0.1))
    x, r = rig["head"], rig["hr"]
    sc.add(sd_ellipsoid(r * 0.9, r * 1.0, r * 0.95), red, x)
    sc.add(sd_ellipsoid(r * 0.95, r * 0.22, r * 0.32), red, x * T(0, r * 0.25, r * 0.74), blend=r * 0.12)     # the brow
    sc.add(sd_capsule((0, 0, 0), (0, -r * 0.15, r * 1.25), r * 0.17, r * 0.1), red, x * T(0, r * 0.02, r * 0.85), blend=r * 0.05)   # the long nose
    for sgn in (-1, 1):
        sc.add(sd_sphere(r * 0.1), Mat("#ffd02a", rough=0.1, emissive="#ffb02a", emit=0.8), x * T(r * 0.34 * sgn, r * 0.08, r * 0.83))
        sc.add(sd_sphere(r * 0.045), INKM, x * T(r * 0.34 * sgn, r * 0.08, r * 0.92))
        sc.add(sd_capsule((-r * 0.2, 0, 0), (r * 0.2, 0, 0), r * 0.07), fur("#f2f0ea", 140, 0.3), x * T(r * 0.34 * sgn, r * 0.3, r * 0.8) * RZ(-0.45 * sgn))   # white brows
    mane = fur("#f2f0ea", 120, 0.35)
    sc.add(sd_ellipsoid(r * 1.05, r * 0.9, r * 0.9), mane, x * T(0, r * 0.1, -r * 0.35), blend=r * 0.1)
    sc.add(sd_ellipsoid(r * 0.75, r * 0.9, r * 0.45), mane, x * T(0, -r * 0.85, r * 0.4) * RX(0.3), blend=r * 0.15)   # the beard
    sc.add(sd_cylinder(r * 0.28, r * 0.26, r * 0.05), lacquer("#16141a"), x * T(0, r * 1.0, r * 0.3) * RX(0.2))   # the tokin
    for sgn in (-1, 1):   # the crow wings, black feathers
        root = rig["torso"] * T(0.07 * sgn, 0.26, -0.1) * RY(-0.4 * sgn)
        open_ = 0.6 if anim == "win" else 0.15 * math.sin(TAU * t)
        for i in range(9):
            u = i / 8
            a = 0.3 + 1.2 * u + open_
            L = 0.38 * (0.55 + 0.45 * math.sin(math.pi * (0.25 + 0.6 * u)))
            sc.add(thin_ellipsoid(0.022, L * 0.5, 0.006), Mat("#16161e", rough=0.3, clearcoat=0.6), root * RZ(-a * sgn) * T(0, L * 0.5, 0))
    fan = held(rig["r"], -1.6 if anim != "win" else -math.pi)
    sc.add(sd_capsule((0, -0.04, 0), (0, 0.06, 0), 0.01), lacquer("#8a1a14"), fan)
    for i in range(9):   # the feather fan (hauchiwa)
        a = (i / 8 - 0.5) * 1.6
        sc.add(thin_ellipsoid(0.02, 0.07, 0.004), Mat("#f4f0e4", rough=0.6), fan * T(0, 0.06, 0) * RZ(a) * T(0, 0.07, 0))
    if anim == "win":   # the wind it raises: pale streaks curling round
        for i in range(5):
            a = TAU * (t + i / 5)
            sc.add(sd_torus(0.34, 0.004), glow("#cfefff", 2.0), RY(a) * T(0, BASE_TOP + 0.2 + 0.1 * i, 0) * RX(0.3))
    flame_aura(sc, t, "#7ad8ff", 8, 0.22, 0.3, 1.4, seed=11)
    return sc


# ─── 5c · Nue (mcts_nnue) ───────────────────────────────────────────────────
def nue(anim, t, side):
    """The nue of the Heike Monogatari: the face of a monkey, the body of a
    tanuki, the legs of a tiger, a snake for a tail; it comes in a black cloud
    of thunder over the palace at night."""
    sc = Scene()
    base(sc, side)
    fm = fur("#7a5a3a", 120, 0.45)
    bm = fur("#c8a878", 120, 0.4)
    tiger = Mat("#e8a020", rough=0.8, pattern=lambda pl, pw, n: {
        "albedo": torch.where((torch.sin(pl[:, 1] * 90 + pl[:, 0] * 30 + fbm(pl * 8, 2) * 4) > 0.45)[:, None],
                              srgb("#1a1410").expand_as(pl), srgb("#e8a020").expand_as(pl))})
    root, head = quadruped(sc, anim, t, fm, bm, s=1.3, leg=1.0, bulk=1.15, snout=0.35, ears=0.6, ear_round=True)
    s_ = 1.3
    gait = TAU * t * (2 if anim == "move" else 0.5)
    for i, (sx, sz) in enumerate(((-1, 1), (1, 1), (-1, -1), (1, -1))):   # tiger legs over the tanuki's
        ph = gait + (math.pi if i in (1, 2) else 0)
        sw = 0.4 * math.sin(ph) if anim == "move" else 0.0
        hip = root * T(0.065 * s_ * sx * 1.15, -0.02 * s_, (0.13 if sz > 0 else -0.15) * s_) * RX(sw)
        L = 0.23 * s_
        sc.add(sd_capsule((0, -0.04 * s_, 0), (0, -L, 0.01 * s_), 0.034 * s_, 0.024 * s_), tiger, hip, blend=0.01)
        for c in (-1, 0, 1):
            sc.add(sd_cone(0.014, 0.005, 0.001, 0.001), Mat("#f2ead8", rough=0.3), hip * T(0.012 * c, -L - 0.01, 0.045) * RX(math.pi / 2 + 0.5))
    # the monkey face: a bare reddish face, a ruff of fur round it
    face = Mat("#b8503e", rough=0.5, sss=0.4)
    sc.add(sd_ellipsoid(0.055, 0.06, 0.03), face, head * T(0, 0.0, 0.06), blend=0.015)
    sc.add(sd_ellipsoid(0.035, 0.025, 0.025), face, head * T(0, -0.035, 0.075), blend=0.01)
    sc.add(sd_torus(0.065, 0.02), fur("#5a4028", 120, 0.4), head * T(0, 0, 0.04) * RX(math.pi / 2))
    roar = True        # it always snarls
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.013), Mat("#ffd02a", rough=0.1, emissive="#ffc02a", emit=1.5 if anim == "win" else 0.9), head * T(0.025 * sgn, 0.012, 0.085))
        sc.add(sd_sphere(0.006), INKM, head * T(0.025 * sgn, 0.012, 0.097))
        sc.add(sd_capsule((-0.016, 0, 0), (0.016, 0, 0), 0.006), Mat("#3a2418", rough=0.6), head * T(0.025 * sgn, 0.03, 0.088) * RZ(0.5 * sgn))   # angry brows
    if roar:
        sc.add(sd_ellipsoid(0.025, 0.015, 0.01), Mat("#3a0a0a", rough=0.6), head * T(0, -0.04, 0.095))
        for sgn in (-1, 1):
            sc.add(sd_cone(0.014, 0.004, 0.001, 0.001), Mat("#f6f0e0", rough=0.3), head * T(0.012 * sgn, -0.03, 0.1) * RX(math.pi))
    # the snake tail, rising and turning to look forward
    sway = 0.06 * math.sin(TAU * t)
    pts = bezier((0, 0.03, -0.28), (sway, 0.16, -0.46), (sway * 2, 0.34, -0.42), (sway, 0.4, -0.28), 9)
    snake = scales("#3a6a3a", "#c8c87a") if False else Mat("#3a6a3a", rough=0.3, clearcoat=0.7, pattern=lambda pl, pw, n: {
        "albedo": srgb("#3a6a3a") * (0.8 + 0.3 * (torch.sin(pl[:, 1] * 160) * torch.sin(pl[:, 0] * 160 + pl[:, 2] * 160) > 0).float())[:, None]})
    chain(sc, pts, [0.036 - 0.002 * i for i in range(9)], snake, root, blend=0.02)
    sh = root * T(*pts[-1]) * RX(-0.3)
    sc.add(sd_ellipsoid(0.03, 0.022, 0.045), snake, sh * T(0, 0, 0.02), blend=0.01)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.007), Mat("#ff3a1a", rough=0.1, emissive="#ff3a1a", emit=1.0), sh * T(0.02 * sgn, 0.012, 0.035))
    sc.add(sd_capsule((0, -0.005, 0.06), (0, -0.01, 0.085), 0.002), Mat("#c81e2a", rough=0.4), sh)          # the forked tongue
    # the black cloud of thunder under it, a few sparks
    cloud = Mat("#1a1622", rough=0.9, pattern=lambda pl, pw, n: {"albedo": srgb("#2a2434") * (0.6 + 0.6 * fbm(pl * 12, 3))[:, None]})
    for k in range(7):
        a = TAU * k / 7 + t * 0.6
        sc.add(sd_sphere(0.07 + 0.02 * math.sin(k * 1.7)), cloud, T(0.3 * math.cos(a), BASE_TOP + 0.04, 0.3 * math.sin(a)))
    for i in range(2 + (3 if anim == "win" else 0)):
        a = TAU * i / 3 + t * 3
        lightning(sc, (0.3 * math.cos(a), 0.06, 0.3 * math.sin(a)), (0.32 * math.cos(a + 0.4), 0.45, 0.32 * math.sin(a + 0.4)), i + 3, t, "#c8a0ff", 4.0, 0.004, T(0, BASE_TOP, 0))
    flame_aura(sc, t, "#a87aff", 9, 0.25, 0.32, 1.5, seed=12)
    return sc


# ─── 6 · Onmyōji ────────────────────────────────────────────────────────────
def onmyoji(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [1.3, 0.2, 1.0]            # two fingers raised in a seal
    p.l = [0.5, 0.45, 0.9]
    if anim == "win":
        p.r = [2.6, 0.2, 0.15]
        p.l = [2.0, 0.8, 0.3]
    white = cloth("#f4f2ec", 0.75)
    m = {"skin": skin("#ecd0b8"), "top": white, "legs": cloth("#6a2a7a"), "boots": cloth("#1a1a20"), "arms": white}
    rig = adult(sc, p, m, H=1.05, bulk=1.0, head=1.0)
    sc.add(sd_cone(0.2, 0.17, 0.09, 0.02), cloth("#6a2a7a"), rig["waist"] * T(0, -0.16, 0))                  # hakama
    for i, sgn in enumerate((-1, 1)):   # wide sleeves hanging under the forearms
        sc.add(thin_ellipsoid(0.02, 0.1, 0.06), white, rig["l" if sgn < 0 else "r"] * T(0, 0.06, -0.04) * RX(0.3), blend=0.01)
    sc.add(sd_box(0.12, 0.2, 0.01, 0.01), white, rig["back"] * T(0, -0.14, -0.02) * RX(0.08))
    x, r = rig["head"], rig["hr"]
    anime_face(sc, x, r, skin("#ecd0b8"), "#9a5ad8", "#16141c", expr="calm" if anim == "idle" else "fierce")
    sc.add(sd_plane_cut(sd_ellipsoid(r * 0.95, r * 1.05, r * 1.02), (0, -0.45, 1), r * 0.3), hair_mat("#16141c"), x * T(0, r * 0.08, -r * 0.03))
    sc.add(sd_ellipsoid(r * 0.55, r * 1.3, r * 0.3), hair_mat("#16141c"), x * T(0, -r * 0.9, -r * 0.65) * RX(0.15))   # long hair tied behind
    sc.add(sd_box(r * 0.5, r * 0.85, r * 0.5, r * 0.3), lacquer("#16141c"), x * T(0, r * 1.35, -r * 0.2) * RX(-0.35))   # the eboshi
    for i in range(6):    # paper talismans (ofuda) orbiting, red ink
        a = TAU * (t * 0.6 + i / 6)
        q = rig["body"] * T(0.36 * math.cos(a), BASE_TOP + 0.38 + 0.14 * math.sin(2 * a + i), 0.36 * math.sin(a)) * RY(-a + math.pi / 2) * RZ(0.2 * math.sin(TAU * t + i))
        sc.add(sd_box(0.022, 0.055, 0.0015, 0.001), Mat("#f2e8c8", rough=0.7), q)
        sc.add(sd_box(0.006, 0.04, 0.002, 0.001), Mat("#c81e2a", rough=0.6), q)
    seal = RY(TAU * t * 0.2) * T(0, BASE_TOP + 0.004, 0)
    for k in range(5):    # the five-pointed seal of light on the base
        a0, a1 = TAU * k / 5, TAU * (k + 2) / 5
        sc.add(sd_capsule((0.4 * math.cos(a0), 0, 0.4 * math.sin(a0)), (0.4 * math.cos(a1), 0, 0.4 * math.sin(a1)), 0.005), glow("#c89aff", 3.0), seal)
    sc.add(sd_torus(0.42, 0.005), glow("#c89aff", 3.0), T(0, BASE_TOP + 0.004, 0))
    flame_aura(sc, t, "#b88aff", 8, 0.22, 0.3, 1.4, seed=6)
    return sc


# ─── 7 · Raijū ──────────────────────────────────────────────────────────────
def raiju(anim, t, side):
    sc = Scene()
    base(sc, side)
    fm, bm = fur("#28305e", 140, 0.6), fur("#b8c2e8", 140, 0.5)
    root, head = quadruped(sc, anim, t, fm, bm, s=1.35, leg=1.05, bulk=1.05, snout=1.1, ears=1.2)
    for sgn in (-1, 1):
        sc.add(thin_ellipsoid(0.018, 0.006, 0.008), Mat("#fff04a", rough=0.1, emissive="#ffe84a", emit=1.5), head * T(0.04 * sgn, 0.027, 0.072) * RZ(-0.4 * sgn))
        sc.add(sd_cone(0.03, 0.01, 0.001, 0.001), Mat("#ffffff", rough=0.3), head * T(0.022 * sgn, -0.06, 0.15) * RX(math.pi))
    for k in range(11):    # a mane of lightning: jagged glowing spikes along the neck and back
        u = k / 10
        z = 0.28 - 0.42 * u
        y = 0.12 + 0.05 * math.sin(math.pi * u)
        for sgn in (-1, 1):
            sp = root * T(0.03 * sgn, y, z) * RZ(-0.5 * sgn) * RX(-0.6)
            sc.add(sd_capsule((0, 0, 0), (0.015 * sgn, 0.06, -0.01), 0.008, 0.002), glow("#ffe84a", 2.2), sp)
            sc.add(sd_capsule((0.015 * sgn, 0.06, -0.01), (-0.005 * sgn, 0.11, -0.03), 0.006, 0.001), glow("#ffe84a", 2.2), sp)
    tail = bezier((0, 0.03, -0.3), (0, 0.12, -0.42), (0, 0.26, -0.45), (0.03, 0.36, -0.38), 6)
    for i, (a, b) in enumerate(zip(tail[:-1], tail[1:])):   # a zigzag tail, like a bolt
        sc.add(sd_capsule(a, (b[0] + 0.03 * (-1) ** i, b[1], b[2]), 0.035 * (1 - 0.15 * i), 0.03 * (1 - 0.15 * i)), fm, root, blend=0.01)
    for i in range(3 + (3 if anim == "win" else 0)):
        a = TAU * i / 3
        lightning(sc, (0.0, 0.15, 0.0), (0.45 * math.cos(a + t * 2), 0.5 + 0.2 * math.sin(a), 0.45 * math.sin(a + t * 2)), i + 1, t, "#fff07a", 4.0, 0.005, root)
    flame_aura(sc, t, "#ffe84a", 10, 0.3, 0.34, 1.6, seed=7)
    return sc


# ─── 8 · Hōō ────────────────────────────────────────────────────────────────
def feather_wing(sc, xf, span, flap, n=11):
    """A wing of long feathers fanned out, red at the root, gold at the tips."""
    def fmat(i):
        u = i / (n - 1)
        c = ["#c81e1a", "#e0401a", "#f07a1e", "#f8a82a", "#ffd04a"][min(4, int(u * 5))]
        return Mat(c, rough=0.35, clearcoat=0.5, sss=0.3)
    for i in range(n):
        u = i / (n - 1)
        a = -0.2 + 1.5 * u + 0.4 * flap
        L = span * (0.55 + 0.45 * math.sin(math.pi * (0.25 + 0.6 * u)))
        sc.add(thin_ellipsoid(0.024, L * 0.5, 0.006), fmat(i), xf * RZ(-a) * T(0, L * 0.5, 0))
    for i in range(6):   # coverts over the root
        a = -0.1 + 1.3 * i / 5 + 0.4 * flap
        sc.add(thin_ellipsoid(0.03, span * 0.18, 0.008), Mat("#a8141a", rough=0.4, clearcoat=0.5), xf * RZ(-a) * T(0, span * 0.17, 0.006))


def hoo(anim, t, side):
    sc = Scene()
    base(sc, side)
    flap = math.sin(TAU * t * (2 if anim in ("move", "win") else 1))
    hover = 0.04 * math.sin(TAU * t)
    root = RY(HEAD) * T(0, BASE_TOP + 0.42 + hover, 0)
    red = Mat("#c8281e", rough=0.4, clearcoat=0.5, sss=0.3)
    gold_ = Mat("#ffb02a", rough=0.35, clearcoat=0.6)
    sc.add(sd_ellipsoid(0.09, 0.16, 0.1), red, root * RX(-0.35))
    sc.add(sd_ellipsoid(0.065, 0.12, 0.05), gold_, root * T(0, -0.01, 0.06) * RX(-0.35), blend=0.02)
    neck = bezier((0, 0.1, 0.03), (0, 0.2, 0.04), (0, 0.26, 0.1), (0, 0.3, 0.13), 5)
    chain(sc, neck, [0.055, 0.047, 0.042, 0.038, 0.038], red, root, blend=0.02)
    head = root * T(0, 0.32, 0.15)
    sc.add(sd_ellipsoid(0.048, 0.044, 0.058), red, head)
    sc.add(sd_cone(0.045, 0.02, 0.002, 0.002), gold(), head * T(0, -0.012, 0.07) * RX(math.pi / 2 + 0.45))   # a hooked beak
    for sgn in (-1, 1):
        sc.add(thin_ellipsoid(0.014, 0.005, 0.006), Mat("#fff6c0", rough=0.1, emissive="#ffe08a", emit=1.2), head * T(0.03 * sgn, 0.012, 0.038) * RZ(-0.4 * sgn))
    for k in range(5):    # a crest of long feathers
        sc.add(thin_ellipsoid(0.01, 0.07 - 0.008 * k, 0.004), gold_, head * T(0, 0.03, -0.02 - 0.012 * k) * RX(-0.6 - 0.22 * k) * T(0, 0.06, 0))
    for sgn in (-1, 1):
        feather_wing(sc, root * T(0.07 * sgn, 0.08, -0.03) * RY(0.25 * sgn) * RZ(-0.9 * sgn) * (RY(math.pi) if sgn < 0 else T()), 0.55, flap)
    for k in range(5):    # the tail: long plumes with an eye at the end, curling
        a = -0.5 + 0.25 * k
        pl = bezier((0, 0, 0), (0.12 * math.sin(a), -0.15, -0.15), (0.26 * math.sin(a), -0.3, -0.1), (0.34 * math.sin(a), -0.36, 0.06), 6)
        chain(sc, pl, [0.014, 0.012, 0.011, 0.01, 0.009, 0.008], red, root * T(0, -0.13, -0.06), blend=0.004)
        e = pl[-1]
        sc.add(thin_ellipsoid(0.03, 0.045, 0.008), gold_, root * T(0, -0.13, -0.06) * T(*e))
        sc.add(thin_ellipsoid(0.014, 0.02, 0.01), Mat("#2a6ad0", rough=0.2, clearcoat=0.8), root * T(0, -0.13, -0.06) * T(e[0], e[1], e[2] + 0.002))
    for k in range(7):   # embers rising
        a = TAU * (t + k / 7)
        sc.add(sd_sphere(0.009), glow("#ffc04a", 4.0), RY(HEAD) * T(0.38 * math.cos(a), BASE_TOP + 0.05 + 0.75 * ((t + k / 7) % 1), 0.38 * math.sin(a)))
    flame_aura(sc, t, "#ff7a2a", 12, 0.3, 0.32, 1.8, seed=8)
    return sc


# ─── 9 · Kaigen ─────────────────────────────────────────────────────────────
def kaigen(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.step = 0.45 if anim != "move" else p.step
    p.lean += 0.1
    p.l = [0.95, 0.35, 1.55]         # an energy sphere charged at the hip
    p.r = [0.95, 0.35, 1.55]
    if anim == "win":
        p.l = [2.65, 0.3, 0.1]
        p.r = [2.65, 0.3, 0.1]
    sk = skin("#dcae8a", 0.35)
    gi = cloth("#e86a1a", 0.75)
    m = {"skin": sk, "top": sk, "legs": gi, "boots": cloth("#1e2a5a"), "arms": sk, "hands": sk}
    rig = adult(sc, p, m, H=1.08, bulk=1.4, head=0.95)
    for sgn in (-1, 1):   # muscles: pectorals, abdomen, the gi torn and hanging at the waist
        sc.add(sd_ellipsoid(0.07, 0.065, 0.03), sk, rig["chest"] * T(0.05 * sgn, 0.005, -0.004), blend=0.02)
        for j in range(3):
            sc.add(sd_ellipsoid(0.026, 0.02, 0.012), sk, rig["torso"] * T(0.028 * sgn, 0.17 - 0.035 * j, 0.072), blend=0.01)
        sc.add(sd_torus(0.026, 0.008), cloth("#1e2a5a"), rig["l" if sgn < 0 else "r"] * T(0, 0.05, 0) * RX(math.pi / 2))   # wristbands
    sc.add(sd_torus(0.098, 0.016), cloth("#1e2a5a"), rig["waist"])
    sc.add(sd_box(0.11, 0.05, 0.08, 0.03), gi, rig["waist"] * T(0, -0.03, -0.02), blend=0.02)
    x, r = rig["head"], rig["hr"]
    anime_face(sc, x, r, sk, "#2ad8c8", "#d8a018", expr="shout" if anim in ("win", "blocked") else "fierce", glow_eyes=0.0)
    spiky_hair(sc, x, r, hair_mat("#ffc81e", "#fff0a0"), n=16, length=1.4, up=0.75, seed=9, bangs=4)
    ball = rig["torso"] * T(0, 0.04, 0.24) if anim != "win" else rig["body"] * T(0, 1.42, 0.05)
    k = 1.0 + 0.12 * math.sin(TAU * t * 4) + (0.8 if anim == "win" else 0)
    sc.add(sd_sphere(0.05 * k), glow("#ffffff", 6.0), ball)
    sc.add(sd_sphere(0.075 * k), Mat("#7ac8ff", rough=0.1, emissive="#5ab8ff", emit=1.6), ball)
    sc.add(sd_torus(0.1 * k, 0.005), glow("#bfe8ff", 3.0), ball * RX(TAU * t) * RZ(0.5))
    for i in range(3):
        a = TAU * i / 3 + t * 3
        lightning(sc, (0.0, 0.5, 0.0), (0.42 * math.cos(a), 0.2 + 0.6 * ((i * 0.37 + t) % 1), 0.42 * math.sin(a)), i + 3, t, "#fff0a0", 4.0, 0.005, rig["body"])
    shards(sc, 6, 0.42, 0.1, t, RY(HEAD), seed=3)
    flame_aura(sc, t, "#ffd23a", 14, 0.55, 0.3, 1.8, seed=9)
    return sc


# ─── 10 · Tenryū ────────────────────────────────────────────────────────────
def tenryu(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = RY(HEAD)
    sc.add(sd_cylinder(0.65, 0.045, 0.0), glow("#fff4c8", 2.6 + 1.0 * math.sin(TAU * t)), root * T(0, BASE_TOP + 0.65, 0))   # the pillar of light
    sc.add(sd_cylinder(0.65, 0.07, 0.0), Mat("#fff4c8", rough=0.2, emissive="#ffe8a0", emit=0.25), root * T(0, BASE_TOP + 0.65, 0))
    scale = scales("#1e7a5a", "#e8d8a8", k=110, rough=0.3, metal=0.25)
    belly = Mat("#e8d8a8", rough=0.45, pattern=lambda pl, pw, n: {"albedo": srgb("#e8d8a8") * (0.8 + 0.25 * (torch.sin(pl[:, 1] * 160) > 0).float())[:, None]})
    trim = gold()
    n = 26
    pts, radii = [], []
    for i in range(n):
        u = i / (n - 1)
        a = TAU * (u * 2.2 + t * (0.25 if anim != "blocked" else 0.05))
        rr = 0.3 * (1 - 0.35 * u)
        pts.append((rr * math.cos(a), BASE_TOP + 0.06 + u * 1.02, rr * math.sin(a)))
        radii.append(0.068 * (0.45 + 0.75 * math.sin(math.pi * min(1.0, u * 1.25 + 0.08))))
    chain(sc, pts, radii, scale, root, blend=0.03)
    for i in range(n - 1):
        sc.add(sd_sphere(radii[i] * 0.75), belly, root * T(pts[i][0] * 0.84, pts[i][1] - radii[i] * 0.3, pts[i][2] * 0.84), blend=0.02)
        if i % 2 == 0:   # a red mane along the back
            sc.add(sd_cone(0.05, 0.02, 0.001, 0.001), Mat("#d8301e", rough=0.5), root * T(pts[i][0] * 1.08, pts[i][1] + radii[i] * 0.8, pts[i][2] * 1.08) * RZ(0.4))
    for i in (6, 14):     # four clawed legs, one holding the pearl
        for sgn in (-1, 1):
            leg = root * T(*pts[i])
            sc.add(sd_capsule((0, 0, 0), (0.1 * sgn, -0.05, 0.07), 0.024, 0.014), scale, leg, blend=0.01)
            for k in range(3):
                sc.add(sd_cone(0.025, 0.008, 0.001, 0.001), Mat("#f2ead8", rough=0.4), leg * T(0.1 * sgn + 0.01 * (k - 1), -0.06, 0.08) * RX(math.pi * 0.8))
    pearl = root * T(*pts[14]) * T(0.13, -0.05, 0.1)
    sc.add(sd_sphere(0.035), Mat("#e8f4ff", rough=0.05, clearcoat=1.0, emissive="#bfe0ff", emit=1.5 + 0.5 * math.sin(TAU * t)), pearl)
    hp = pts[-1]
    head = root * T(hp[0], hp[1] + 0.07, hp[2]) * RY(-0.6 + 0.2 * math.sin(TAU * t)) * RX(0.15)
    sc.add(sd_ellipsoid(0.1, 0.075, 0.12), scale, head)
    sc.add(sd_box(0.065, 0.042, 0.1, 0.034), scale, head * T(0, -0.01, 0.13), blend=0.03)        # a long snout
    jaw = 0.5 if anim == "win" else 0.15
    sc.add(sd_box(0.055, 0.018, 0.095, 0.014), belly, head * T(0, -0.05, 0.05) * RX(jaw) * T(0, 0, 0.08))
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.06, 0.02, 0.055), scale, head * T(0.05 * sgn, 0.05, 0.05) * RZ(-0.35 * sgn), blend=0.01)   # brows
        sc.add(thin_ellipsoid(0.02, 0.008, 0.008), Mat("#ff3a2a", rough=0.1, emissive="#ff3a2a", emit=1.5), head * T(0.055 * sgn, 0.032, 0.1) * RZ(-0.4 * sgn))
        for k in range(3):
            sc.add(sd_cone(0.022, 0.007, 0.001, 0.001), Mat("#ffffff", rough=0.3), head * T(0.04 * sgn, -0.045, 0.12 + 0.035 * k) * RX(math.pi))
        sc.add(sd_capsule((0, 0, 0), (0.14 * sgn, 0.22, -0.2), 0.016, 0.004), trim, head * T(0.05 * sgn, 0.07, -0.05))   # antlers
        sc.add(sd_capsule((0.07 * sgn, 0.11, -0.1), (0.13 * sgn, 0.18, -0.05), 0.008, 0.003), trim, head * T(0.05 * sgn, 0.07, -0.05))
        wh = bezier((0, 0, 0), (0.1 * sgn, -0.04, 0.05), (0.2 * sgn, -0.1, 0.0), (0.26 * sgn, -0.2, -0.05), 6)   # long whiskers
        chain(sc, wh, [0.006, 0.005, 0.004, 0.003, 0.002, 0.001], trim, head * T(0.05 * sgn, -0.03, 0.2), blend=0.002)
        for k in range(4):   # the mane behind the head
            sc.add(sd_cone(0.08, 0.02, 0.001, 0.001), Mat("#d8301e", rough=0.5), head * T(0.06 * sgn, 0.02 - 0.03 * k, -0.08) * RZ(-1.2 * sgn) * RX(0.6) * T(0, 0.06, 0))
    if anim == "win":
        sc.add(sd_cone(0.2, 0.06, 0.02, 0.01), glow("#fff4c8", 4.0), head * T(0, -0.04, 0.4) * RX(math.pi / 2))
    sc.add(sd_torus(0.4, 0.006), glow("#ffe88a", 3.0), root * T(0, BASE_TOP + 0.95, 0) * RX(0.35) * RY(TAU * t))
    shards(sc, 7, 0.48, 0.15, t, root, seed=11, size=0.05)
    for i in range(3):
        a = TAU * i / 3 + t * 2
        lightning(sc, (0.0, 1.15, 0.0), (0.5 * math.cos(a), 0.3, 0.5 * math.sin(a)), i + 7, t, "#fff0a0", 4.0, 0.005, root)
    flame_aura(sc, t, "#ffd24a", 16, 0.6, 0.42, 1.8, seed=10)
    return sc


SCALE = {"random": 1.2, "beginner": 1.2, "monte_carlo": 1.2, "mcts": 1.15, "basic_minmax": 1.05, "minmax": 1.15, "mcts_nnue": 1.1, "nnue193": 1.15,
         "features": 1.05, "resnet": 1.1, "dev": 1.15, "alphazero": 1.0}
CAST = {"random": ("Ōkami", okami), "beginner": ("Shinobi", shinobi), "monte_carlo": ("Kitsune", kitsune),
        "mcts": ("Samouraï", samurai), "basic_minmax": ("Oni", oni), "minmax": ("Tengu", tengu), "mcts_nnue": ("Nue", nue),
        "nnue193": ("Onmyōji", onmyoji),
        "features": ("Raijū", raiju), "resnet": ("Hōō", hoo), "dev": ("Kaigen", kaigen), "alphazero": ("Tenryū", tenryu)}
SHONEN = {"manga_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"manga_" + k: n for k, (n, _) in CAST.items()}
