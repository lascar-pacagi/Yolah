"""
dinos.py — the Dinosaures set: the growth of a tyrannosaur, one stage per AI,
small at first and more and more terrifying:

  random        Œuf                 an egg in a nest of ferns, trembling
  beginner      Éclosion            the shell splits, a snout comes out
  monte_carlo   Nouveau-né          a piece of shell still on its head
  mcts          Petit               covered in down, hopping
  basic_minmax  Juvénile            long legs, feathers, quick
  nnue193       Adolescent          the first real teeth, stripes
  features      Jeune adulte        hunting, head low, green scales
  resnet        Adulte              the whole tyrannosaur, brown-grey, massive
  dev           Vieux mâle balafré  scars, one foot on bones
  alphazero     Roi tyran           near black and red, roaring, eyes and cracks of lava

Every stage has its own size, silhouette and colours, so that two of them
never look alike. An original design (no film's look). The human is never part
of a set.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm
from models import base, BASE_TOP, TAU, glow
from heroic import chain, bezier, scales, HEAD
from arctic import fur, EYE, IVORY


def hide(c, k=60, stripes=None, scar=False):
    def pat(pl, pw, n):
        a = srgb(c) * (0.7 + 0.45 * fbm(pl * k, 3))[:, None]
        cell = (fbm(pl * k * 2.2, 2) > 0.55).float()[:, None]
        a = a * (1 - 0.18 * cell)
        if stripes:
            st = (torch.sin(pl[:, 2] * 40 + fbm(pl * 8, 2) * 4) > 0.55).float()[:, None]
            a = a * (1 - st) + srgb(stripes) * st
        if scar:
            sc_ = (torch.sin(pl[:, 2] * 25 + pl[:, 1] * 30) > 0.97).float()[:, None] * (fbm(pl * 6, 2) > 0.5).float()[:, None]
            a = a * (1 - sc_) + srgb("#c8a898") * sc_
        return {"albedo": a}
    return Mat(c, rough=0.75, sss=0.2, pattern=pat)


def theropod(sc, anim, t, s, skin, belly, head_k=1.0, leg_k=1.0, teeth=True, down=None, roar=0.0, low=0.0, eye_m=None, arm_k=1.0):
    """A tyrannosaur: legs, body, tail, short arms, neck and a big head with a jaw that opens.
    s: size; head_k: head size (babies: big); roar: jaw open 0..1; low: head lowered (hunting)."""
    walk = anim == "move"
    gait = TAU * t * 2
    L = 0.3 * s * leg_k
    hip_y = BASE_TOP + L + (0.02 * s * abs(math.sin(gait)) if walk else 0.0)
    breathe = 0.01 * s * math.sin(TAU * t)
    yaw = HEAD + 1.2 + (0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0)   # three-quarter profile: the tail goes sideways, not away
    root = RY(yaw) * T(0, hip_y, 0.08 * s) * RX(0.12 - low * 0.25)
    sc.add(sd_ellipsoid(0.12 * s, 0.13 * s + breathe, 0.24 * s), skin, root)
    sc.add(sd_ellipsoid(0.11 * s, 0.12 * s + breathe, 0.12 * s), skin, root * T(0, 0.04 * s, 0.17 * s), blend=0.05 * s)   # the chest
    sc.add(sd_ellipsoid(0.08 * s, 0.07 * s, 0.18 * s), belly, root * T(0, -0.06 * s, 0.06 * s), blend=0.03 * s)
    if down:
        sc.add(sd_ellipsoid(0.13 * s, 0.14 * s, 0.25 * s), down, root * T(0, 0.01 * s, 0.02 * s), blend=0.02 * s)
    # the tail, swaying
    sw = 0.08 * s * math.sin(TAU * t) + (0.1 * s * math.sin(gait) if walk else 0.0)
    tail = bezier((0, 0.02 * s, -0.18 * s), (sw * 0.3, 0.04 * s, -0.36 * s), (sw, 0.02 * s, -0.52 * s), (sw * 1.6, -0.02 * s, -0.68 * s), 8)
    chain(sc, tail, [0.1 * s * (1 - i / 8) ** 1.1 + 0.006 for i in range(8)], skin, root, blend=0.03 * s)
    # the legs: thigh, shin, metatarsus, three clawed toes
    for sgn, ph in ((-1, 0.0), (1, math.pi)):
        sw_ = 0.45 * math.sin(gait + ph) if walk else 0.0
        hip = RY(yaw) * T(0.09 * s * sgn, hip_y, 0.08 * s) * RX(sw_)
        knee = (0, -L * 0.45, 0.08 * s)
        ankle = (0, -L * 0.82, -0.04 * s)
        foot = (0, -L, 0.03 * s)
        sc.add(sd_ellipsoid(0.07 * s, 0.13 * s, 0.1 * s), skin, hip * T(0, -0.03 * s, 0.02 * s), blend=0.04 * s)
        chain(sc, [(0, 0, 0), knee, ankle, foot], [0.06 * s, 0.045 * s, 0.03 * s, 0.025 * s], skin, hip, blend=0.02 * s)
        for k in (-1, 0, 1):
            toe = hip * T(*foot) * RY(0.35 * k)
            sc.add(sd_capsule((0, 0, 0), (0, -0.005 * s, 0.08 * s), 0.018 * s, 0.012 * s), skin, toe, blend=0.01 * s)
            sc.add(sd_cone(0.03 * s, 0.01 * s, 0.001, 0.001), Mat("#2a2420", rough=0.4), toe * T(0, -0.005 * s, 0.085 * s) * RX(math.pi / 2 + 0.3))
    # the arms: small, two fingers
    for sgn in (-1, 1):
        sh = root * T(0.08 * s * sgn, -0.02 * s, 0.22 * s) * RX(0.6 + 0.2 * math.sin(TAU * t))
        a1 = 0.08 * s * arm_k
        chain(sc, [(0, 0, 0), (0.01 * sgn * s, -a1, 0.02 * s), (0.0, -a1 * 1.3, a1 * 0.6)], [0.022 * s, 0.016 * s, 0.01 * s], skin, sh, blend=0.01 * s)
    # the neck and the head
    up = roar * 0.45 - low * 0.3
    nk = root * T(0, 0.08 * s, 0.24 * s) * RX(-0.35 - up)
    sc.add(sd_capsule((0, 0, 0), (0, 0.14 * s, 0.06 * s), 0.075 * s, 0.065 * s), skin, nk, blend=0.04 * s)
    hs = s * head_k
    head = nk * T(0, 0.15 * s, 0.07 * s) * RX(0.6 + up * 0.4) * RY(0.35 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)
    sc.add(sd_ellipsoid(0.07 * hs, 0.08 * hs, 0.1 * hs), skin, head, blend=0.03 * s)
    sc.add(sd_box(0.055 * hs, 0.045 * hs, 0.11 * hs, 0.035 * hs), skin, head * T(0, 0.0, 0.12 * hs), blend=0.04 * hs)    # the snout
    jaw = head * T(0, -0.035 * hs, -0.02 * hs) * RX(0.08 + 0.55 * roar)
    sc.add(sd_box(0.05 * hs, 0.022 * hs, 0.12 * hs, 0.018 * hs), skin, jaw * T(0, -0.01 * hs, 0.13 * hs), blend=0.02 * hs)
    if roar > 0.1:
        sc.add(sd_box(0.04 * hs, 0.012 * hs, 0.1 * hs, 0.01 * hs), Mat("#6a1a1e", rough=0.5), jaw * T(0, 0.012 * hs, 0.12 * hs))
    if teeth:
        for i in range(6):
            z = 0.04 * hs + i * 0.032 * hs
            for sgn in (-1, 1):
                sc.add(sd_cone(0.025 * hs, 0.007 * hs, 0.001, 0.001), IVORY, head * T(0.042 * hs * sgn, -0.04 * hs, z) * RX(math.pi))
                sc.add(sd_cone(0.02 * hs, 0.006 * hs, 0.001, 0.001), IVORY, jaw * T(0.038 * hs * sgn, 0.005 * hs, z + 0.02 * hs))
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.016 * hs), eye_m or EYE, head * T(0.055 * hs * sgn, 0.03 * hs, 0.05 * hs))
        sc.add(sd_ellipsoid(0.03 * hs, 0.012 * hs, 0.04 * hs), skin, head * T(0.05 * hs * sgn, 0.05 * hs, 0.05 * hs), blend=0.01 * hs)   # brow
        sc.add(sd_sphere(0.008 * hs), Mat("#1a1414", rough=0.5), head * T(0.03 * hs * sgn, 0.025 * hs, 0.225 * hs))   # nostrils
    return root, head


def nest(sc):
    for i in range(14):   # twigs and fern leaves around
        a = TAU * i / 14
        sc.add(sd_capsule((0, 0, 0), (0.12, 0.03, 0.0), 0.012), Mat("#5a4028", rough=0.9), T(0.2 * math.cos(a), BASE_TOP + 0.02, 0.2 * math.sin(a)) * RY(-a + 1.2))
    sc.add(sd_torus(0.2, 0.05), fur("#5a6a2e", 80, 0.8), T(0, BASE_TOP + 0.03, 0))


def egg_mat():
    def pat(pl, pw, n):
        sp = (fbm(pl * 40, 2) > 0.62).float()[:, None]
        return {"albedo": srgb("#e8dcc0") * (0.9 + 0.15 * fbm(pl * 10, 2))[:, None] * (1 - sp) + srgb("#8a6a4a") * sp}
    return Mat("#e8dcc0", rough=0.5, clearcoat=0.3, pattern=pat)


# ─── 1 · Œuf ────────────────────────────────────────────────────────────────
def egg(anim, t, side):
    sc = Scene()
    base(sc, side)
    nest(sc)
    wob = 0.12 * math.sin(TAU * t * (4 if anim in ("win", "blocked", "move") else 1)) * (1 if anim != "idle" else 0.4)
    sc.add(sd_ellipsoid(0.12, 0.16, 0.12), egg_mat(), T(0, BASE_TOP + 0.15, 0) * RZ(wob) * T(0, 0.02, 0))
    return sc


# ─── 2 · Éclosion ───────────────────────────────────────────────────────────
def hatching(anim, t, side):
    sc = Scene()
    base(sc, side)
    nest(sc)
    em = egg_mat()
    b = T(0, BASE_TOP + 0.13, 0) * RY(HEAD)
    sc.add(sd_ellipsoid(0.13, 0.12, 0.13), em, b)                                # the lower shell
    sc.add(sd_ellipsoid(0.11, 0.11, 0.11), Mat("#2a2018"), b * T(0, 0.08, 0))     # its dark inside (open at the top)
    pop = 0.05 + 0.04 * abs(math.sin(TAU * t * (2 if anim != "idle" else 0.5)))
    skin = hide("#7a8a4a", 120)
    head = b * T(0, 0.1 + pop, 0.02) * RX(-0.4 if anim == "win" else -0.15)
    sc.add(sd_ellipsoid(0.06, 0.06, 0.07), skin, head)
    sc.add(sd_box(0.04, 0.03, 0.06, 0.025), skin, head * T(0, -0.005, 0.06), blend=0.02)
    if anim == "win":
        sc.add(sd_box(0.03, 0.01, 0.05, 0.008), Mat("#a83a3a"), head * T(0, -0.03, 0.07))
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.017), EYE, head * T(0.04 * sgn, 0.02, 0.04))
    sc.add(sd_ellipsoid(0.08, 0.06, 0.08), em, head * T(0, 0.06, -0.01) * RZ(0.4))   # the cap of shell on its head
    for k in range(4):   # chips of shell on the nest
        a = TAU * k / 4 + 0.5
        sc.add(sd_box(0.03, 0.006, 0.02, 0.004), em, T(0.16 * math.cos(a), BASE_TOP + 0.06, 0.16 * math.sin(a)) * RY(a) * RZ(0.4))
    return sc


# ─── 3 · Nouveau-né ─────────────────────────────────────────────────────────
def newborn(anim, t, side):
    sc = Scene()
    base(sc, side)
    skin = hide("#8a9a52", 120)
    root, head = theropod(sc, anim, t, 0.45, skin, hide("#c8c890", 120), head_k=1.7, leg_k=0.9, teeth=False,
                          roar=0.6 if anim == "win" else 0.0)
    sc.add(sd_ellipsoid(0.07, 0.05, 0.07), egg_mat(), head * T(0, 0.12, -0.01) * RZ(0.3))   # a piece of shell still on its head
    return sc


# ─── 4 · Petit ──────────────────────────────────────────────────────────────
def chick(anim, t, side):
    sc = Scene()
    base(sc, side)
    skin = hide("#7a6a4a", 100)
    down = fur("#b89a6a", 160, 0.8)
    theropod(sc, anim, t, 0.55, skin, hide("#d8c8a0", 100), head_k=1.45, leg_k=1.0, teeth=False, down=down,
             roar=0.6 if anim == "win" else 0.0)
    return sc


# ─── 5 · Juvénile ───────────────────────────────────────────────────────────
def juvenile(anim, t, side):
    sc = Scene()
    base(sc, side)
    skin = hide("#7a3a24", 90)
    down = fur("#b0582e", 140, 0.6)
    theropod(sc, anim, t, 0.68, skin, hide("#c8b890", 90), head_k=1.2, leg_k=1.35, teeth=True, down=down,
             roar=0.8 if anim == "win" else 0.0)
    return sc


# ─── 6 · Adolescent ─────────────────────────────────────────────────────────
def adolescent(anim, t, side):
    sc = Scene()
    base(sc, side)
    skin = hide("#7a5a3a", 70, stripes="#2a1a12")
    theropod(sc, anim, t, 0.82, skin, hide("#c8a878", 70), head_k=1.1, leg_k=1.2, roar=0.9 if anim == "win" else 0.15)
    return sc


# ─── 7 · Jeune adulte ───────────────────────────────────────────────────────
def young_adult(anim, t, side):
    sc = Scene()
    base(sc, side)
    skin = hide("#4a6a3a", 60, stripes="#2a3a20")
    theropod(sc, anim, t, 0.95, skin, hide("#a8a878", 60), head_k=1.05, leg_k=1.05, low=0.0 if anim == "win" else 0.8,
             roar=0.9 if anim == "win" else 0.25, eye_m=Mat("#e8c23a", rough=0.2, emissive="#e8c23a", emit=0.5))
    return sc


# ─── 8 · Adulte ─────────────────────────────────────────────────────────────
def adult_rex(anim, t, side):
    sc = Scene()
    base(sc, side)
    skin = hide("#5e5236", 50, stripes="#3e3424")
    theropod(sc, anim, t, 1.08, skin, hide("#b0a078", 50), head_k=1.1, leg_k=0.95, arm_k=0.8, roar=1.0 if anim == "win" else 0.2,
             eye_m=Mat("#e8a23a", rough=0.2, emissive="#e8a23a", emit=0.6))
    return sc


# ─── 9 · Vieux mâle balafré ─────────────────────────────────────────────────
def scarred(anim, t, side):
    sc = Scene()
    base(sc, side)
    bone = Mat("#e2d6bc", rough=0.6)
    for k in range(5):   # ribs and a skull under its foot
        sc.add(sd_torus(0.07, 0.008), bone, T(0.18, BASE_TOP + 0.02, -0.05 + 0.035 * k) * RX(math.pi / 2) * RZ(0.3))
    sc.add(sd_ellipsoid(0.06, 0.04, 0.08), bone, T(-0.2, BASE_TOP + 0.03, 0.18) * RY(0.6))
    skin = hide("#46484a", 45, scar=True)
    theropod(sc, anim, t, 1.15, skin, hide("#8a7a68", 45), head_k=1.15, leg_k=0.92, arm_k=0.7, roar=1.0 if anim == "win" else 0.35,
             eye_m=Mat("#ff8a2a", rough=0.2, emissive="#ff8a2a", emit=1.2))
    return sc


# ─── 10 · Roi tyran ─────────────────────────────────────────────────────────
def tyrant(anim, t, side):
    sc = Scene()
    base(sc, side)
    for k in range(6):   # cracks of lava across the base
        a = TAU * k / 6 + 0.3
        sc.add(sd_box(0.2, 0.004, 0.008, 0.002), glow("#ff6a1a", 5.0 + 2 * math.sin(TAU * t + k)), T(0.22 * math.cos(a), BASE_TOP + 0.002, 0.22 * math.sin(a)) * RY(-a))
    skin = hide("#2a2224", 40, stripes="#5a1a14")
    theropod(sc, anim, t, 1.22, skin, hide("#5a4038", 40), head_k=1.18, leg_k=0.9, arm_k=0.7, roar=1.0 if anim in ("win", "blocked") else 0.55,
             eye_m=glow("#ff3a1a", 6.0))
    for k in range(5):   # embers rising
        a = TAU * (t + k / 5)
        sc.add(sd_sphere(0.012), glow("#ffa02a", 7.0), T(0.42 * math.cos(a), BASE_TOP + 0.1 + 0.7 * ((t + k / 5) % 1), 0.42 * math.sin(a)))
    return sc


SCALE = {"random": 1.3, "beginner": 1.3, "monte_carlo": 1.25, "mcts": 1.2, "basic_minmax": 1.15, "nnue193": 1.1,
         "features": 1.1, "resnet": 1.1, "dev": 1.1, "alphazero": 1.1}   # the growth shows: about 2.4 times longer from the newborn to the tyrant
CAST = {"random": ("Œuf", egg), "beginner": ("Éclosion", hatching), "monte_carlo": ("Nouveau-né", newborn),
        "mcts": ("Petit", chick), "basic_minmax": ("Juvénile", juvenile), "nnue193": ("Adolescent", adolescent),
        "features": ("Jeune adulte", young_adult), "resnet": ("Adulte", adult_rex), "dev": ("Vieux mâle balafré", scarred),
        "alphazero": ("Roi tyran", tyrant)}
DINOS = {"dino_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"dino_" + k: n for k, (n, _) in CAST.items()}
