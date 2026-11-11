"""
kaiju.py — the Manga set, second version: creatures and fighters, original
designs in the spirit of manga monsters and over-powered heroes, one per AI,
each one bigger and more spectacular than the previous one:

  Pomu (a fluff ball) · Pyrolo (a fire salamander) · Kiko (a fox spirit with
  three glowing tails) · Mokko (a jade golem) · Raijū (a thunder wolf) ·
  Mizuchi (a water serpent dragon) · Tora (a tiger-headed swordsman) · Hōō (a
  fire bird) · Ryū (a fighter at full power, an energy sphere in his hands) ·
  Tenryū (a celestial dragon god coiled around a pillar of light)

and the user as a manga fighter. Power shows as size, gems on the base, an
aura, then lightning, floating rocks, a pillar of light. Rendered in the toon
mode (render.py): flat bands and ink lines.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, sd_plane_cut
from models import base, BASE_TOP, TAU, wave, glow
from manga import flat, anime_eye, gems, aura, human_manga, INK
from legends import humanoid, base_pose, held

HEAD = math.radians(28)


# ─── pieces of creatures ────────────────────────────────────────────────────
def eye(sc, xf, w, h, iris, closed=False, angry=False):
    if closed:
        sc.add(sd_capsule((-w, -h * 0.1, 0), (0, h * 0.3, 0), w * 0.18), flat(INK), xf)
        sc.add(sd_capsule((0, h * 0.3, 0), (w, -h * 0.1, 0), w * 0.18), flat(INK), xf)
        return
    sc.add(sd_ellipsoid(w, h, w * 0.35), anime_eye(iris, w, h), xf)
    if angry:
        sc.add(sd_capsule((-w * 1.1, h * 1.0, 0), (w * 1.1, h * 0.55, 0), w * 0.16), flat(INK), xf * T(0, 0, w * 0.15))


def chain(sc, pts, radii, mat, xf, blend=0.02):
    """A body along points (a tail, a serpent): capsules between them."""
    for (a, b), (ra, rb) in zip(zip(pts[:-1], pts[1:]), zip(radii[:-1], radii[1:])):
        sc.add(sd_capsule(a, b, ra, rb), mat, xf, blend=blend)


def lightning(sc, a, b, seed, t, col="#bfe8ff", k=9.0, w=0.008, xf=None):
    """A jagged bolt from a to b, changing every frame."""
    xf = xf or T()
    n = 6
    pts = []
    for i in range(n + 1):
        u = i / n
        j = 0 if i in (0, n) else 0.06 * math.sin(seed * 12.9 + i * 7.3 + math.floor(t * 12) * 3.1)
        pts.append(tuple(a[c] + (b[c] - a[c]) * u + (j if c != 1 else j * 0.5) for c in range(3)))
    for p, q in zip(pts[:-1], pts[1:]):
        sc.add(sd_capsule(p, q, w), glow(col, k), xf)


def rocks(sc, n, rad, h0, t, xf, seed=0):
    """Rocks floating around, lifted by the power."""
    for i in range(n):
        a = TAU * (i / n + t * 0.25) + seed
        y = h0 + 0.08 * math.sin(TAU * (t + i * 0.3))
        q = (rad * math.cos(a), y, rad * math.sin(a))
        s_ = 0.035 + 0.02 * ((i * 7) % 3)
        sc.add(sd_ellipsoid(s_, s_ * 0.7, s_ * 0.9), flat("#6a5a50"), xf * T(*q) * RY(a) * RX(0.5 * i))
        sc.add(sd_sphere(s_ * 0.5), glow("#9fdcff", 2.0), xf * T(q[0], q[1] - s_, q[2]))


def pillar(sc, h, col, k, xf, r=0.09):
    sc.add(sd_cylinder(h * 0.5, r, 0.0), glow(col, k), xf * T(0, BASE_TOP + h * 0.5, 0))


def wings(sc, xf, span, flap, mat, n=7, glow_col=None):
    for sgn in (-1, 1):
        w = xf * RZ(-sgn * (0.5 + 0.5 * flap)) * RY(sgn * 0.3)
        for k in range(n):
            a = 0.2 + 1.1 * k / (n - 1)
            ln = span * (0.6 + 0.4 * math.sin(math.pi * k / (n - 1)))
            tip = (sgn * ln * math.cos(a - 0.6), ln * math.sin(a - 0.6) + 0.05, -0.02 * k)
            sc.add(sd_capsule((0, 0, 0), tip, 0.03, 0.008), mat if glow_col is None or k % 2 else glow(glow_col, 4.0), w)


# ─── 1 · Pomu, a fluff ball ─────────────────────────────────────────────────
def pomu(anim, t, side):
    sc = Scene()
    base(sc, side)
    bounce = 0.03 * abs(math.sin(TAU * t * 2)) if anim in ("idle", "win") else 0.08 * math.sin(math.pi * t) if anim == "move" else 0.0
    sq = 1 + 0.08 * math.sin(TAU * t * 2)
    root = RY(HEAD + (0.3 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)) * T(0, BASE_TOP + 0.2 + bounce, 0)
    fur = Mat("#f8d8e8", rough=0.8, pattern=lambda pl, pw, n: {"albedo": srgb("#f8d8e8")[None].expand(pl.shape[0], 3)})
    sc.add(lambda p: sd_ellipsoid(0.2 * sq, 0.19 / sq, 0.2 * sq)(p) + 0.012 * torch.sin(p[:, 0] * 80) * torch.sin(p[:, 1] * 80), fur, root)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.05, 0.035, 0.06), flat("#e8a8c8"), root * T(0.09 * sgn, -0.18, 0.04))       # feet
        eye(sc, root * T(0.075 * sgn, 0.03, 0.18) * RY(0.3 * sgn), 0.045, 0.06, "#2a2a5a", closed=anim == "win",
            angry=False)
        sc.add(sd_ellipsoid(0.03, 0.015, 0.01), flat("#ff8aa8"), root * T(0.13 * sgn, -0.04, 0.16) * RY(0.6 * sgn))   # blush
    sc.add(sd_capsule((0, 0.17, 0), (0.02, 0.25, 0), 0.01), flat("#4aa84a"), root)                          # the sprout
    sc.add(sd_ellipsoid(0.05, 0.012, 0.025), flat("#6ad86a"), root * T(0.05, 0.26, 0) * RZ(0.4))
    if anim == "blocked":   # dizzy
        for k in range(3):
            a = TAU * (t * 2 + k / 3)
            sc.add(sd_sphere(0.015), glow("#ffe060", 4), root * T(0.12 * math.cos(a), 0.3, 0.12 * math.sin(a)))
    if anim == "win":
        for k in range(5):
            a = TAU * (t + k / 5)
            sc.add(sd_sphere(0.012), glow("#fff0a0", 6), root * T(0.28 * math.cos(a), 0.15 + 0.08 * math.sin(2 * a), 0.28 * math.sin(a)))
    gems(sc, 1)
    return sc


# ─── 2 · Pyrolo, a fire salamander ──────────────────────────────────────────
def quadruped(sc, anim, t, body_c, belly_c, size=1.0, head_size=1.0, legs=0.12, tail=None):
    """A four-legged creature: body, legs with a trot, a head; returns (root, head)."""
    gait = TAU * t * (2 if anim == "move" else 0.5)
    lift = 0.03 * size * abs(math.sin(gait)) if anim == "move" else 0.0
    root = RY(HEAD + (0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)) * T(0, BASE_TOP + legs * size + lift, 0)
    bm = flat(body_c)
    sc.add(sd_ellipsoid(0.12 * size, 0.09 * size, 0.2 * size), bm, root)
    sc.add(sd_ellipsoid(0.1 * size, 0.06 * size, 0.17 * size), flat(belly_c), root * T(0, -0.035 * size, 0.01))
    for i, (sx, sz) in enumerate(((-1, 1), (1, 1), (-1, -1), (1, -1))):
        ph = gait + (math.pi if (i in (1, 2)) else 0)
        swing = 0.35 * math.sin(ph) if anim == "move" else 0.0
        hip = root * T(0.08 * size * sx, -0.02 * size, 0.12 * size * sz) * RX(swing)
        sc.add(sd_capsule((0, 0, 0), (0, -legs * size, 0.01), 0.035 * size, 0.028 * size), bm, hip)
        sc.add(sd_ellipsoid(0.035 * size, 0.018 * size, 0.045 * size), bm, hip * T(0, -legs * size, 0.02 * size))
    head = root * T(0, 0.08 * size, 0.2 * size) * RX(-0.25 + (0.2 if anim == "win" else 0))
    return root, head


def pyrolo(anim, t, side):
    sc = Scene()
    base(sc, side)
    root, head = quadruped(sc, anim, t, "#e8562a", "#ffd07a", size=1.15, legs=0.09)
    hm = flat("#e8562a")
    sc.add(sd_ellipsoid(0.13, 0.11, 0.13), hm, head)
    sc.add(sd_ellipsoid(0.1, 0.06, 0.08), flat("#ffd07a"), head * T(0, -0.04, 0.06))
    for sgn in (-1, 1):
        eye(sc, head * T(0.06 * sgn, 0.04, 0.1) * RY(0.35 * sgn), 0.035, 0.045, "#2a7ad8", closed=anim == "win", angry=anim == "move")
    if anim == "win":
        sc.add(sd_ellipsoid(0.05, 0.03, 0.02), flat("#7a1a28"), head * T(0, -0.05, 0.12))
    # the tail curling up, a flame at its tip
    pts = [(0, 0, -0.2), (0, 0.04, -0.3), (0, 0.12, -0.36), (0, 0.22, -0.34), (0, 0.29, -0.28)]
    chain(sc, pts, [0.05, 0.04, 0.032, 0.025, 0.02], hm, root)
    fk = 1.0 + 0.2 * math.sin(TAU * t * 4) + (0.6 if anim == "win" else 0)
    sc.add(sd_cone(0.06 * fk, 0.05, 0.004, 0.003), glow("#ffb02a", 9.0), root * T(0, 0.36 + 0.03 * fk, -0.28) * RZ(0.15 * math.sin(TAU * t * 3)))
    sc.add(sd_cone(0.035 * fk, 0.028, 0.002, 0.002), glow("#fff2a0", 12.0), root * T(0, 0.34 + 0.02 * fk, -0.27))
    gems(sc, 1)
    return sc


# ─── 3 · Kiko, a fox spirit with three tails ────────────────────────────────
def kiko(anim, t, side):
    sc = Scene()
    base(sc, side)
    root, head = quadruped(sc, anim, t, "#f2f0ea", "#ffffff", size=1.2, legs=0.13)
    fm = flat("#f2f0ea")
    sc.add(sd_ellipsoid(0.11, 0.1, 0.12), fm, head)
    sc.add(sd_cone(0.06, 0.05, 0.012, 0.01), fm, head * T(0, -0.02, 0.13) * RX(math.pi / 2))      # the snout
    sc.add(sd_sphere(0.014), flat(INK), head * T(0, -0.02, 0.2))
    for sgn in (-1, 1):
        sc.add(sd_cone(0.07, 0.045, 0.006, 0.004), fm, head * T(0.07 * sgn, 0.12, -0.02) * RZ(-0.25 * sgn))   # ears
        sc.add(sd_cone(0.04, 0.025, 0.004, 0.003), flat("#e84a5a"), head * T(0.07 * sgn, 0.12, 0.0) * RZ(-0.25 * sgn))
        eye(sc, head * T(0.055 * sgn, 0.03, 0.09) * RY(0.4 * sgn) * RZ(0.3 * sgn), 0.03, 0.022, "#e8a82a", closed=anim == "win", angry=anim != "idle")
        sc.add(sd_capsule((0, 0, 0), (0.03 * sgn, 0.02, 0), 0.006), flat("#e84a5a"), head * T(0.07 * sgn, 0.07, 0.07))   # markings
    # three tails, each ending in a blue flame
    for k, a in enumerate((-0.9, 0.0, 0.9)):
        sw = 0.15 * math.sin(TAU * (t + k * 0.3))
        tail = root * T(0, 0.03, -0.18) * RZ(-a * 0.8 + sw) * RX(-1.3)
        pts = [(0, 0, 0), (0, 0.1, -0.04), (0, 0.2, -0.04), (0, 0.28, 0.02)]
        chain(sc, pts, [0.035, 0.05, 0.045, 0.025], fm, tail, blend=0.02)
        sc.add(sd_ellipsoid(0.03, 0.05, 0.03), flat("#6ad8ff"), tail * T(0, 0.3, 0.04))
        sc.add(sd_sphere(0.035 + 0.008 * math.sin(TAU * t * 3 + k)), glow("#6ad8ff", 7.0), tail * T(0, 0.35, 0.06))
    for k in range(3):   # fox fire wisps
        a = TAU * (t + k / 3)
        sc.add(sd_sphere(0.022), glow("#6ad8ff", 6.0), root * T(0.3 * math.cos(a), 0.25 + 0.05 * math.sin(2 * a), 0.3 * math.sin(a)))
    gems(sc, 2)
    return sc


# ─── 4 · Mokko, a jade golem ────────────────────────────────────────────────
def mokko(anim, t, side):
    sc = Scene()
    base(sc, side)
    stone = Mat("#4a8a6a", rough=0.7)
    moss = flat("#3a6a2a")
    rune = "#7affc8"
    stomp = math.sin(math.pi * t) if anim == "move" else 0
    root = RY(HEAD + (0.15 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)) * T(0, BASE_TOP + 0.03 * stomp, 0)
    for sgn in (-1, 1):
        sc.add(sd_box(0.07, 0.09, 0.08, 0.025), stone, root * T(0.1 * sgn, 0.09, 0.0) * RX(0.3 * stomp * sgn))
    body = root * T(0, 0.34, 0)
    sc.add(sd_box(0.2, 0.17, 0.15, 0.06), stone, body)
    sc.add(sd_box(0.21, 0.05, 0.16, 0.03), moss, body * T(0, 0.14, 0))
    for k, (x, y) in enumerate(((-0.08, 0.04), (0.06, -0.05), (0.0, 0.08))):
        sc.add(sd_box(0.03, 0.01, 0.01, 0.004), glow(rune, 4.0 + 2 * wave(t, 2)), body * T(x, y, 0.155) * RZ(k))
    for sgn in (-1, 1):   # big arms down to the ground
        arm = body * T(0.27 * sgn, 0.08, 0) * RZ(0.15 * sgn) * RX(-0.3 if anim != "win" else -2.5)
        sc.add(sd_box(0.07, 0.15, 0.07, 0.03), stone, arm * T(0, -0.12, 0))
        sc.add(sd_box(0.08, 0.06, 0.08, 0.03), stone, arm * T(0, -0.3, 0.02))
    hd = body * T(0, 0.25, 0.03)
    sc.add(sd_box(0.11, 0.08, 0.1, 0.035), stone, hd)
    for sgn in (-1, 1):
        sc.add(sd_box(0.025, 0.012, 0.01, 0.004), glow(rune, 8.0 if anim == "win" else 5.0), hd * T(0.045 * sgn, 0.0, 0.1))
    for k, (x, z, h_) in enumerate(((-0.1, -0.08, 0.12), (0.05, -0.1, 0.16), (0.12, -0.04, 0.1))):   # crystals on the back
        sc.add(sd_cone(h_ * 0.5, 0.03, 0.002, 0.002), Mat("#9affd8", rough=0.1, emissive="#5affb8", emit=1.2),
               body * T(x, 0.17 + h_ * 0.4, z) * RZ(0.3 * x * 5) * RX(-0.3))
    aura(sc, 0.15, "#7affc8", t, root)
    gems(sc, 2)
    return sc


# ─── 5 · Raijū, a thunder wolf ──────────────────────────────────────────────
def raiju(anim, t, side):
    sc = Scene()
    base(sc, side)
    root, head = quadruped(sc, anim, t, "#3a4a8a", "#d8e0f8", size=1.45, legs=0.12)
    fm = flat("#3a4a8a")
    sc.add(sd_ellipsoid(0.11, 0.1, 0.12), fm, head)
    sc.add(sd_cone(0.07, 0.055, 0.02, 0.01), fm, head * T(0, -0.03, 0.13) * RX(math.pi / 2))
    sc.add(sd_sphere(0.016), flat(INK), head * T(0, -0.03, 0.21))
    for sgn in (-1, 1):
        sc.add(sd_cone(0.07, 0.04, 0.006, 0.004), fm, head * T(0.07 * sgn, 0.12, -0.03) * RZ(-0.3 * sgn) * RX(-0.3))
        eye(sc, head * T(0.055 * sgn, 0.035, 0.09) * RY(0.4 * sgn) * RZ(0.25 * sgn), 0.028, 0.02, "#ffe02a", angry=True)
        sc.add(sd_cone(0.025, 0.012, 0.002, 0.002), flat("#ffffff"), head * T(0.03 * sgn, -0.075, 0.17) * RX(math.pi))   # fangs
    for k in range(7):   # spiky mane, yellow, crackling
        a = -1.2 + 0.4 * k
        sc.add(sd_cone(0.08, 0.03, 0.003, 0.002), glow("#ffe02a", 2.5), root * T(0.08 * math.sin(a), 0.1, 0.05 + 0.06 * math.cos(a)) * RZ(-0.6 * math.sin(a)) * RX(-0.5) * T(0, 0.07, 0))
    pts = [(0, 0.02, -0.19), (0, 0.08, -0.3), (0, 0.17, -0.36), (0.02, 0.24, -0.33)]
    chain(sc, pts, [0.045, 0.04, 0.035, 0.015], fm, root)
    for i in range(3 + (3 if anim == "win" else 0)):
        a = TAU * i / 3
        lightning(sc, (0.0, 0.1, 0.0), (0.4 * math.cos(a + t * 2), 0.5 + 0.2 * math.sin(a), 0.4 * math.sin(a + t * 2)), i + 1, t, "#ffe85a", 8.0,
                  0.006, root)
    aura(sc, 0.35, "#ffe85a", t, RY(HEAD))
    gems(sc, 3)
    return sc


# ─── 6 · Mizuchi, a water serpent dragon ────────────────────────────────────
def mizuchi(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = RY(HEAD + (0.2 * math.sin(TAU * 3 * t) if anim == "blocked" else 0))
    sk = flat("#2a8ab8")
    belly = flat("#bff0ff")
    rise = 0.1 * math.sin(math.pi * t) if anim == "move" else 0.0
    pts, radii = [], []
    n = 16
    for i in range(n):
        u = i / (n - 1)
        a = TAU * (u * 1.4 + t * 0.25)
        rr = 0.26 * (1 - 0.55 * u)
        pts.append((rr * math.cos(a), BASE_TOP + 0.05 + u * (0.75 + rise), rr * math.sin(a)))
        radii.append(0.07 * (0.6 + 0.6 * math.sin(math.pi * min(1.0, u * 1.4 + 0.15))))
    chain(sc, pts, radii, sk, root, blend=0.03)
    for i in range(2, n - 2, 2):   # a fin along the back
        sc.add(sd_cone(0.05, 0.03, 0.002, 0.002), flat("#6ad8ff"), root * T(pts[i][0], pts[i][1] + radii[i] * 0.9, pts[i][2]))
    hp = pts[-1]
    head = root * T(hp[0], hp[1] + 0.05, hp[2]) * RY(-0.4) * RX(0.2)
    sc.add(sd_ellipsoid(0.1, 0.08, 0.13), sk, head)
    sc.add(sd_ellipsoid(0.07, 0.04, 0.1), belly, head * T(0, -0.04, 0.05))
    for sgn in (-1, 1):
        eye(sc, head * T(0.055 * sgn, 0.03, 0.08) * RY(0.4 * sgn) * RZ(0.25 * sgn), 0.03, 0.022, "#ff4a8a", closed=anim == "win", angry=anim != "idle")
        sc.add(sd_capsule((0, 0, 0), (0.1 * sgn, 0.12, -0.12), 0.012, 0.004), flat("#e8f8ff"), head * T(0.04 * sgn, 0.06, -0.03))   # horns
        sc.add(sd_capsule((0, 0, 0), (0.12 * sgn, -0.06, 0.05), 0.006, 0.002), flat("#6ad8ff"), head * T(0.05 * sgn, -0.03, 0.1))   # whiskers
    for k in range(4):   # water orbs orbiting
        a = TAU * (t + k / 4)
        sc.add(sd_sphere(0.04), Mat("#bfefff", glass=True), root * T(0.38 * math.cos(a), 0.45 + 0.1 * math.sin(2 * a), 0.38 * math.sin(a)))
        sc.add(sd_sphere(0.02), glow("#6ad8ff", 3.0), root * T(0.38 * math.cos(a), 0.45 + 0.1 * math.sin(2 * a), 0.38 * math.sin(a)))
    aura(sc, 0.45, "#6ad8ff", t, RY(HEAD))
    gems(sc, 3)
    return sc


# ─── 7 · Tora, a tiger-headed swordsman ─────────────────────────────────────
def tora(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [1.0, 0.2, 0.3] if anim != "win" else [2.5, 0.2, 0.1]
    p.l = [0.6, 0.4, 1.2]
    orange = flat("#e8862a")
    m = {"skin": orange, "top": flat("#2a2a3a"), "legs": flat("#2a2a3a"), "boots": flat("#1a1a20"), "arms": orange}
    rig = humanoid(sc, p, m, w=1.15, h=1.0, head_r=0.18)
    sc.add(sd_box(0.15, 0.03, 0.12, 0.02), flat("#c82a2a"), rig["torso"] * T(0, 0.06, 0))
    for k in range(2):   # a scarf flowing
        sc.add(sd_box(0.03, 0.12, 0.008, 0.006), flat("#c82a2a"), rig["back"] * T(0.05 * k, -0.02, -0.03) * RX(0.6 + 0.2 * math.sin(TAU * (t + k * 0.3))) * T(0, -0.1, 0))
    x = rig["head"] * RX(-0.1)
    r = 0.18
    sc.add(sd_ellipsoid(r, r * 0.95, r), orange, x)
    sc.add(sd_ellipsoid(r * 0.6, r * 0.45, r * 0.5), flat("#ffffff"), x * T(0, -r * 0.35, r * 0.6))
    sc.add(sd_sphere(r * 0.12), flat(INK), x * T(0, -r * 0.2, r * 1.0))
    for k in range(3):   # stripes
        for sgn in (-1, 1):
            sc.add(sd_capsule((0, 0, 0), (r * 0.25 * sgn, -r * 0.05, 0), r * 0.05), flat(INK), x * T(r * 0.6 * sgn, r * (0.45 - 0.2 * k), r * 0.75))
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(r * 0.25, r * 0.3, r * 0.12), orange, x * T(r * 0.75 * sgn, r * 0.75, 0))
        eye(sc, x * T(r * 0.33 * sgn, r * 0.18, r * 0.85) * RY(0.35 * sgn) * RZ(0.3 * sgn), r * 0.17, r * 0.12, "#7aff6a", angry=True,
            closed=anim == "win")
    # a giant blade
    sw = held(rig["r"], -math.pi if anim == "win" else -1.2)
    sc.add(sd_capsule((0, -0.07, 0), (0, 0.04, 0), 0.016), flat("#1a1a2a"), sw)
    sc.add(sd_box(0.06, 0.012, 0.025, 0.006), flat("#d8b040", 0.4, 1.0), sw * T(0, 0.05, 0))
    sc.add(sd_box(0.035, 0.36, 0.006, 0.004), flat("#e8f4ff", 0.15, 1.0), sw * T(0, 0.42, 0))
    sc.add(sd_box(0.006, 0.34, 0.007, 0.002), glow("#7aff6a", 3.0), sw * T(0.02, 0.42, 0))
    aura(sc, 0.55, "#7aff6a", t, rig["body"])
    gems(sc, 4)
    return sc


# ─── 8 · Hōō, a fire bird ───────────────────────────────────────────────────
def hoo(anim, t, side):
    sc = Scene()
    base(sc, side)
    flap = math.sin(TAU * t * (2 if anim in ("move", "win") else 1))
    hover = 0.05 * math.sin(TAU * t)
    root = RY(HEAD) * T(0, BASE_TOP + 0.38 + hover, 0)
    red = flat("#e8402a")
    gold_ = flat("#ffb02a")
    sc.add(sd_ellipsoid(0.13, 0.17, 0.12), red, root * RX(-0.3))
    sc.add(sd_ellipsoid(0.09, 0.12, 0.06), gold_, root * T(0, -0.02, 0.07) * RX(-0.3))
    head = root * T(0, 0.2, 0.05)
    sc.add(sd_sphere(0.09), red, head)
    sc.add(sd_cone(0.045, 0.03, 0.004, 0.003), flat("#ffd04a"), head * T(0, -0.01, 0.11) * RX(math.pi / 2 + 0.3))   # beak
    for sgn in (-1, 1):
        eye(sc, head * T(0.045 * sgn, 0.02, 0.07) * RY(0.45 * sgn) * RZ(0.3 * sgn), 0.024, 0.018, "#ffe85a", angry=True, closed=anim == "win")
    for k in range(5):   # a crest of flames
        sc.add(sd_cone(0.06 + 0.02 * (k % 2), 0.025, 0.002, 0.002), glow("#ffb02a", 6.0),
               head * T(0, 0.06, -0.03 - 0.02 * k) * RX(-0.6 - 0.15 * k) * T(0, 0.06, 0))
    wings(sc, root * T(0, 0.08, -0.05), 0.42, flap, red, 8, "#ffb02a")
    for k in range(5):   # the tail: long burning feathers
        a = -0.5 + 0.25 * k
        sc.add(sd_capsule((0, 0, 0), (0.25 * math.sin(a), -0.32, -0.15), 0.025, 0.004), glow("#ff7a2a", 5.0), root * T(0, -0.12, -0.08))
    for k in range(4):   # embers
        a = TAU * (t + k / 4)
        sc.add(sd_sphere(0.012), glow("#ffd04a", 8.0), root * T(0.35 * math.cos(a), -0.2 + 0.5 * ((t + k / 4) % 1), 0.35 * math.sin(a)))
    aura(sc, 0.65, "#ff7a2a", t, RY(HEAD))
    gems(sc, 4)
    return sc


# ─── 9 · Ryū, a fighter at full power ───────────────────────────────────────
def ryu(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    # charging an energy sphere between his hands, at the hip
    p.l = [0.9, 0.35, 1.6]
    p.r = [0.9, 0.35, 1.6]
    if anim == "win":
        p.l = [2.6, 0.3, 0.1]
        p.r = [2.6, 0.3, 0.1]
    m = {"skin": flat("#f2c8a8"), "top": flat("#1e2a5a"), "legs": flat("#e8862a"), "boots": flat("#1a1a20"), "arms": flat("#f2c8a8")}
    rig = humanoid(sc, p, m, w=1.25, h=1.05, head_r=0.18, limb=1.25)
    sc.add(sd_torus(0.15, 0.025), flat("#1e2a5a"), rig["torso"] * T(0, 0.07, 0))
    from manga import anime_head, spikes_ring
    hair = {"c": "#e8302a", "spikes": spikes_ring(13, 0.85, 1.1, 0.28, jitter=0.25) + spikes_ring(7, 0.35, 0.9, 0.28, -1.0, 1.0, 0.2)}
    anime_head(sc, rig["head"] * RX(-0.15), 0.18, "#f2c8a8", hair, "#2ad0ff", "fierce" if anim != "blocked" else "worried")
    ball_c = rig["torso"] * T(0, 0.05, 0.25) if anim != "win" else rig["body"] * T(0, 1.25, 0.05)
    k = 1.0 + 0.15 * math.sin(TAU * t * 4) + (0.8 if anim == "win" else 0)
    sc.add(sd_sphere(0.07 * k), glow("#8ad8ff", 12.0), ball_c)
    sc.add(sd_torus(0.1 * k, 0.006), glow("#ffffff", 8.0), ball_c * RX(TAU * t) * RZ(0.5))
    for i in range(4):
        a = TAU * i / 4 + t * 3
        lightning(sc, (0.0, 0.5, 0.0), (0.45 * math.cos(a), 0.2 + 0.7 * ((i * 0.37 + t) % 1), 0.45 * math.sin(a)), i + 3, t, "#bfe8ff", 9.0, 0.006, rig["body"])
    rocks(sc, 5, 0.42, 0.25, t, RY(HEAD))
    aura(sc, 0.9, "#4ab0ff", t, rig["body"])
    gems(sc, 5)
    return sc


# ─── 10 · Tenryū, a celestial dragon god ────────────────────────────────────
def tenryu(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = RY(HEAD)
    pillar(sc, 1.3, "#fff4c8", 4.0 + 2.0 * wave(t, 2), root, r=0.05)
    scale = Mat("#3a8a6a", rough=0.35, metal=0.3)
    gold_ = flat("#ffc83a", 0.3, 1.0)
    n = 22
    pts, radii = [], []
    for i in range(n):
        u = i / (n - 1)
        a = TAU * (u * 2.2 + t * (0.4 if anim != "blocked" else 0.1))
        rr = 0.3 * (1 - 0.35 * u)
        pts.append((rr * math.cos(a), BASE_TOP + 0.08 + u * 1.05, rr * math.sin(a)))
        radii.append(0.075 * (0.55 + 0.7 * math.sin(math.pi * min(1.0, u * 1.3 + 0.1))))
    chain(sc, pts, radii, scale, root, blend=0.035)
    for i in range(0, n - 1):      # a pale belly on the inner side of the coil
        sc.add(sd_sphere(radii[i] * 0.8), flat("#f2e8c8"), root * T(pts[i][0] * 0.82, pts[i][1] - radii[i] * 0.25, pts[i][2] * 0.82), blend=0.02)
    for i in range(1, n - 2, 2):   # a golden mane along the back
        sc.add(sd_cone(0.05, 0.028, 0.002, 0.002), gold_, root * T(pts[i][0], pts[i][1] + radii[i] * 0.95, pts[i][2]))
    for i in (5, 12):              # little arms with claws
        for sgn in (-1, 1):
            sc.add(sd_capsule((0, 0, 0), (0.08 * sgn, -0.06, 0.06), 0.02, 0.012), scale, root * T(*pts[i]))
    hp = pts[-1]
    head = root * T(hp[0], hp[1] + 0.08, hp[2]) * RY(-0.6 + 0.2 * math.sin(TAU * t)) * RX(0.15)
    sc.add(sd_ellipsoid(0.14, 0.1, 0.18), scale, head)
    sc.add(sd_ellipsoid(0.11, 0.055, 0.15), flat("#f2e8c8"), head * T(0, -0.055, 0.08))
    sc.add(sd_ellipsoid(0.06, 0.03, 0.04), flat("#f2e8c8"), head * T(0, 0.04, 0.17))
    for sgn in (-1, 1):
        eye(sc, head * T(0.065 * sgn, 0.035, 0.1) * RY(0.4 * sgn) * RZ(0.3 * sgn), 0.035, 0.024, "#ff2a5a", angry=True)
        sc.add(sd_capsule((0, 0, 0), (0.12 * sgn, 0.2, -0.16), 0.016, 0.004), gold_, head * T(0.05 * sgn, 0.07, -0.05))   # antlers
        sc.add(sd_capsule((0.06 * sgn, 0.1, -0.1), (0.11 * sgn, 0.16, -0.06), 0.008, 0.003), gold_, head * T(0.05 * sgn, 0.07, -0.05))
        sc.add(sd_capsule((0, 0, 0), (0.2 * sgn, -0.12, 0.1), 0.007, 0.002), gold_, head * T(0.06 * sgn, -0.04, 0.14))   # whiskers
    if anim == "win":
        sc.add(sd_cone(0.2, 0.06, 0.03, 0.01), glow("#fff4c8", 10.0), head * T(0, -0.02, 0.35) * RX(math.pi / 2))      # a breath of light
    # a crown of light: rings and a pearl
    sc.add(sd_torus(0.4, 0.008), glow("#ffe88a", 6.0), root * T(0, BASE_TOP + 1.0, 0) * RX(0.35) * RY(TAU * t))
    sc.add(sd_sphere(0.06), glow("#ffffff", 14.0), root * T(0, BASE_TOP + 1.25 + 0.03 * math.sin(TAU * t), 0))
    rocks(sc, 6, 0.5, 0.35, t, root, seed=1.0)
    for i in range(3):
        a = TAU * i / 3 + t * 2
        lightning(sc, (0.0, 1.2, 0.0), (0.5 * math.cos(a), 0.3, 0.5 * math.sin(a)), i + 7, t, "#fff0a0", 9.0, 0.006, root)
    aura(sc, 1.0, "#ffd24a", t, root)
    gems(sc, 5)
    return sc


# ─── the user, a fighter in a guard, an aura ────────────────────────────────
def fighter_user(anim, t, side):
    sc = human_manga(anim, t, side)
    aura(sc, 0.4, "#9fdcff", t, RY(HEAD))
    return sc


SCALE = {"random": 1.5, "beginner": 1.35, "monte_carlo": 1.25, "mcts": 1.2, "basic_minmax": 1.15, "nnue193": 1.2,
         "features": 1.25, "resnet": 1.25, "dev": 1.3, "alphazero": 1.0, "human": 1.25}
CAST = {"random": ("Pomu", pomu), "beginner": ("Pyrolo", pyrolo), "monte_carlo": ("Kiko", kiko), "mcts": ("Mokko", mokko),
        "basic_minmax": ("Raijū", raiju), "nnue193": ("Mizuchi", mizuchi), "features": ("Tora", tora), "resnet": ("Hōō", hoo),
        "dev": ("Ryū", ryu), "alphazero": ("Tenryū", tenryu), "human": ("Vous (manga)", fighter_user)}
MANGA = {"manga_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"manga_" + k: n for k, (n, _) in CAST.items()}
