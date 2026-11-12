"""
arctic.py — the Banquise set: one creature (or one explorer) per AI, from the
seal to the woolly mammoth, natural proportions, no chibi:

  random        Phoque              a grey seal lying on the ice, raising its head
  beginner      Manchot empereur    upright, black and white, gold at the neck
  monte_carlo   Renard polaire      a white fox, a bushy tail
  mcts          Harfang des neiges  a snowy owl on a block of ice, golden eyes
  basic_minmax  Morse               a heavy walrus, long tusks, whiskers
  nnue193       Loup arctique       a white wolf, lean and tall
  features      Exploratrice polaire  a red parka with a fur hood, goggles, an ice axe
  resnet        Orque               a killer whale leaping out of a hole in the ice
  dev           Ours blanc          a great polar bear, roaring
  alphazero     Mammouth laineux    shaggy, curved tusks, the trunk raised

Power shows as frost crystals turning around the figure (more, brighter), then
a ring of aurora light on the base. The human is never part of a set.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm
from models import base, BASE_TOP, TAU, wave, glow
from legends import base_pose, held, skin
from heroic import adult, chain, bezier, HEAD


def fur(c, k=140, spread=0.6):
    return Mat(c, rough=0.95, sss=0.2, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (1 - spread / 2 + spread * fbm(pl * k, 2))[:, None]})


def spotted(c, spot="#3a3a40"):
    def pat(pl, pw, n):
        s_ = (fbm(pl * 22, 3) > 0.6).float()[:, None]
        return {"albedo": srgb(c) * (0.85 + 0.25 * fbm(pl * 60, 2))[:, None] * (1 - s_) + srgb(spot) * s_}
    return Mat(c, rough=0.45, clearcoat=0.4, pattern=pat)


def hide(c, k=40):
    return Mat(c, rough=0.7, sss=0.3, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.75 + 0.35 * fbm(pl * k, 3))[:, None]})


EYE = Mat("#0a0a0c", rough=0.08, clearcoat=1.0)
NOSE = Mat("#141214", rough=0.3, clearcoat=0.6)
IVORY = Mat("#efe6d0", rough=0.35, clearcoat=0.5)
ICE = Mat("#a8d8f0", rough=0.12, clearcoat=1.0, sss=0.5,
          pattern=lambda pl, pw, n: {"albedo": srgb("#a8d8f0") * (0.8 + 0.4 * fbm(pl * 25, 3))[:, None]})


def frost(sc, n, t, r=0.42, h=0.35, k=4.0):
    """n frost crystals turning around the figure (the power)."""
    for i in range(n):
        a = TAU * (t * 0.5 + i / n)
        y = BASE_TOP + h + 0.12 * math.sin(TAU * t + i * 1.7)
        xf = T(r * math.cos(a), y, r * math.sin(a)) * RY(HEAD) * RZ(TAU * t * 0.3 + i)   # a snowflake facing the camera
        for j in range(3):
            sc.add(sd_box(0.045, 0.006, 0.004, 0.002), glow("#cfeeff", k), xf * RZ(j * math.pi / 3))


def aurora_ring(sc, t):
    sc.add(sd_torus(0.5, 0.012), glow("#5cffc8", 6.0 + 2 * math.sin(TAU * t)), T(0, BASE_TOP + 0.005, 0))


def root_of(anim, t, lift=0.0):
    shake = 0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    hop = 0.04 * abs(math.sin(TAU * t)) if anim == "move" else (0.03 * abs(math.sin(TAU * t)) if anim == "win" else 0.0)
    return RY(HEAD + shake) * T(0, BASE_TOP + lift + hop, 0)


# ─── a quadruped (fox, wolf, bear, mammoth) ─────────────────────────────────
def quadruped(sc, anim, t, fm, bm, s=1.0, leg=1.0, bulk=1.0, snout=1.0, ears=1.0, ear_round=False, neck=1.0):
    gait = TAU * t * (2 if anim == "move" else 0.5)
    lift = 0.03 * s * abs(math.sin(gait)) if anim == "move" else 0.0
    L = 0.23 * s * leg
    root = RY(HEAD + (0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)) * T(0, BASE_TOP + L + 0.02 * s + lift, 0)
    sc.add(sd_ellipsoid(0.09 * s * bulk, 0.09 * s * bulk, 0.21 * s), fm, root * T(0, 0, -0.03 * s))
    sc.add(sd_ellipsoid(0.105 * s * bulk, 0.115 * s * bulk, 0.12 * s), fm, root * T(0, 0.02 * s, 0.1 * s), blend=0.04 * s)
    sc.add(sd_ellipsoid(0.07 * s * bulk, 0.07 * s, 0.1 * s), bm, root * T(0, -0.035 * s * bulk, 0.1 * s), blend=0.03 * s)
    for i, (sx, sz) in enumerate(((-1, 1), (1, 1), (-1, -1), (1, -1))):
        ph = gait + (math.pi if i in (1, 2) else 0)
        sw = 0.4 * math.sin(ph) if anim == "move" else 0.0
        hip = root * T(0.065 * s * sx * bulk, -0.02 * s, (0.13 if sz > 0 else -0.15) * s) * RX(sw)
        r0 = 0.032 * s * bulk
        if sz > 0:
            sc.add(sd_capsule((0, 0, 0), (0, -L, 0.01 * s), r0, r0 * 0.6), fm, hip, blend=0.02 * s)
        else:
            sc.add(sd_ellipsoid(0.05 * s * bulk, 0.085 * s, 0.065 * s), fm, hip * T(0, -0.02 * s, 0), blend=0.025 * s)
            sc.add(sd_capsule((0, -0.06 * s, 0.02 * s), (0, -L * 0.6, -0.04 * s), r0 * 0.9, r0 * 0.6), fm, hip, blend=0.01 * s)
            sc.add(sd_capsule((0, -L * 0.6, -0.04 * s), (0, -L, -0.01 * s), r0 * 0.6, r0 * 0.55), fm, hip, blend=0.01 * s)
        sc.add(sd_ellipsoid(0.026 * s * bulk, 0.013 * s, 0.034 * s), fm, hip * T(0, -L - 0.005 * s, 0.015 * s))
    howl = anim == "win"
    nk = root * T(0, 0.06 * s, 0.17 * s)
    head = nk * T(0, 0.1 * s * neck, 0.07 * s) * RX(-0.6 if howl else (0.15 if anim == "blocked" else 0.0))
    sc.add(sd_capsule((0, 0, 0), (0, 0.09 * s * neck, 0.06 * s), 0.062 * s * bulk, 0.05 * s), fm, nk, blend=0.03 * s)
    sc.add(sd_ellipsoid(0.06 * s, 0.056 * s, 0.066 * s), fm, head)
    sc.add(sd_capsule((0, -0.01 * s, 0.03 * s), (0, -0.025 * s, 0.13 * s * snout), 0.036 * s, 0.02 * s), fm, head, blend=0.02 * s)
    sc.add(sd_sphere(0.014 * s), NOSE, head * T(0, -0.02 * s, 0.134 * s * snout + 0.012 * s))
    if howl:
        sc.add(sd_box(0.016 * s, 0.008 * s, 0.05 * s, 0.004 * s), Mat("#4a0a14", rough=0.5), head * T(0, -0.045 * s, 0.08 * s * snout))
    for sgn in (-1, 1):
        if ear_round:
            sc.add(sd_ellipsoid(0.022 * s * ears, 0.02 * s * ears, 0.01 * s), fm, head * T(0.04 * sgn * s, 0.05 * s, -0.01 * s), blend=0.01 * s)
        else:
            sc.add(sd_cone(0.055 * s * ears, 0.026 * s, 0.003 * s, 0.002), fm, head * T(0.035 * sgn * s, 0.065 * s, -0.01 * s) * RZ(-0.2 * sgn) * T(0, 0.02 * s, 0))
        sc.add(sd_sphere(0.009 * s), EYE, head * T(0.03 * sgn * s, 0.018 * s, 0.05 * s))
    return root, head


# ─── 1 · Phoque ─────────────────────────────────────────────────────────────
def seal(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t)
    m = spotted("#8a8e96", "#4a4e56")
    breathe = 0.01 * math.sin(TAU * t)
    sc.add(sd_ellipsoid(0.15, 0.11 + breathe, 0.34), m, root * T(0, 0.11, -0.06))
    up = 0.35 if anim == "win" else 0.15 + 0.05 * math.sin(TAU * t)
    nk = root * T(0, 0.15, 0.2) * RX(-up)
    sc.add(sd_capsule((0, 0, 0), (0, 0.06, 0.08), 0.1, 0.085), m, nk, blend=0.04)
    head = nk * T(0, 0.08, 0.1) * RY(0.3 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)
    sc.add(sd_sphere(0.09), m, head, blend=0.03)
    sc.add(sd_ellipsoid(0.055, 0.045, 0.05), m, head * T(0, -0.02, 0.07), blend=0.02)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.024), EYE, head * T(0.042 * sgn, 0.02, 0.065))
        sc.add(sd_sphere(0.022), m, head * T(0.02 * sgn, -0.025, 0.105), blend=0.01)   # whisker pads
        for k in range(3):
            sc.add(sd_capsule((0, 0, 0), (0.07 * sgn, 0.01 * (k - 1), 0.01), 0.0016), Mat("#e8e4dc", rough=0.4),
                   head * T(0.03 * sgn, -0.025 - 0.006 * k, 0.11))
        flip = 0.6 * math.sin(TAU * t * 4) if anim == "win" else 0.0
        sc.add(sd_ellipsoid(0.07, 0.015, 0.04), m, root * T(0.14 * sgn, 0.05, 0.12) * RZ(-0.5 * sgn - flip * sgn))
        sc.add(sd_ellipsoid(0.05, 0.012, 0.08), m, root * T(0.05 * sgn, 0.06, -0.42) * RY(0.4 * sgn))
    sc.add(sd_sphere(0.012), NOSE, head * T(0, 0.0, 0.12))
    return sc


# ─── 2 · Manchot empereur ───────────────────────────────────────────────────
def penguin(anim, t, side):
    sc = Scene()
    base(sc, side)
    waddle = 0.12 * math.sin(TAU * t * 2) if anim == "move" else 0.03 * math.sin(TAU * t)
    root = root_of(anim, t) * RZ(waddle)
    black = Mat("#14161c", rough=0.4, clearcoat=0.5)
    white = Mat("#f2f0ea", rough=0.6)
    gold_ = Mat("#f2b83a", rough=0.4)
    sc.add(sd_ellipsoid(0.15, 0.3, 0.14), black, root * T(0, 0.33, -0.01))
    sc.add(sd_ellipsoid(0.125, 0.26, 0.1), white, root * T(0, 0.3, 0.045), blend=0.02)
    sc.add(sd_ellipsoid(0.09, 0.06, 0.06), Mat("#f8e8b0", rough=0.5), root * T(0, 0.5, 0.08), blend=0.03)   # pale gold chest
    up = -0.5 if anim == "win" else 0.0
    head = root * T(0, 0.66, 0.0) * RX(up)
    sc.add(sd_capsule((0, -0.1, 0), (0, 0, 0), 0.09, 0.085), black, head, blend=0.04)
    sc.add(sd_sphere(0.085), black, head, blend=0.02)
    sc.add(sd_cone(0.1, 0.022, 0.004, 0.003), black, head * T(0, -0.01, 0.07) * RX(math.pi / 2 + 0.25) * T(0, 0.05, 0))
    sc.add(sd_ellipsoid(0.012, 0.006, 0.04), Mat("#f08a3a", rough=0.4), head * T(0, -0.025, 0.13), blend=0.005)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.025, 0.045, 0.02), gold_, head * T(0.07 * sgn, -0.04, 0.02) * RZ(0.4 * sgn), blend=0.02)
        sc.add(sd_sphere(0.014), EYE, head * T(0.055 * sgn, 0.015, 0.05))
        flap = (1.1 + 0.3 * math.sin(TAU * t * 4)) if anim == "win" else 0.15 + 0.1 * math.sin(TAU * t)
        sc.add(sd_ellipsoid(0.03, 0.16, 0.06), black, root * T(0.15 * sgn, 0.42, 0) * RZ(flap * sgn) * T(0, -0.12, 0))
        sc.add(sd_ellipsoid(0.04, 0.015, 0.06), Mat("#3a3436", rough=0.6), root * T(0.05 * sgn, 0.01, 0.05))
    return sc


# ─── 3 · Renard polaire ─────────────────────────────────────────────────────
def arctic_fox(anim, t, side):
    sc = Scene()
    base(sc, side)
    fm, bm = fur("#f4f6f8"), fur("#ffffff")
    root, head = quadruped(sc, anim, t, fm, bm, s=0.9, leg=0.85, snout=1.1, ears=0.9)
    sw = 0.3 * math.sin(TAU * t)
    tail = bezier((0, 0.0, -0.2), (0.05, 0.02, -0.33), (0.1 + 0.05 * sw, 0.12, -0.36), (0.12 + 0.05 * sw, 0.2, -0.3), 6)
    chain(sc, tail, [0.04, 0.06, 0.07, 0.065, 0.05, 0.02], fm, root, blend=0.02)
    frost(sc, 1, t, k=3.0)
    return sc


# ─── 4 · Harfang des neiges ─────────────────────────────────────────────────
def snowy_owl(anim, t, side):
    sc = Scene()
    base(sc, side)
    sc.add(sd_box(0.16, 0.06, 0.13, 0.02), ICE, T(0, BASE_TOP + 0.06, 0) * RY(0.3))          # a block of ice
    root = root_of(anim, t, 0.12)
    def pat(pl, pw, n):
        s_ = ((fbm(pl * 30, 2) > 0.66) & (pl[:, 1] < 0.4)).float()[:, None]
        return {"albedo": srgb("#f6f6f2") * (1 - s_ * 0.85)}
    plum = Mat("#f6f6f2", rough=0.9, sss=0.2, pattern=pat)
    sc.add(sd_ellipsoid(0.15, 0.21, 0.14), plum, root * T(0, 0.22, 0))
    turn = 0.9 * math.sin(TAU * t) if anim == "idle" else (0.4 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0)
    head = root * T(0, 0.48, 0.01) * RY(turn * 0.5)
    sc.add(sd_sphere(0.12), plum, head, blend=0.05)
    sc.add(sd_ellipsoid(0.1, 0.085, 0.03), Mat("#ffffff", rough=0.9), head * T(0, 0.0, 0.095), blend=0.02)   # the facial disc
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.026), Mat("#f2c22a", rough=0.2, emissive="#f2c22a", emit=0.6), head * T(0.042 * sgn, 0.015, 0.105))
        sc.add(sd_sphere(0.013), EYE, head * T(0.042 * sgn, 0.015, 0.125))
        spread = 1.4 + 0.3 * math.sin(TAU * t * 3) if anim == "win" else 0.1
        w = root * T(0.13 * sgn, 0.3, -0.01) * RZ(spread * sgn)
        sc.add(sd_ellipsoid(0.04, 0.2, 0.12), plum, w * T(0, -0.13, 0) * RX(0.15))
        sc.add(sd_capsule((0, 0, 0), (0, -0.04, 0.03), 0.012), Mat("#2a2a2a", rough=0.4), root * T(0.05 * sgn, 0.03, 0.06))
    sc.add(sd_cone(0.035, 0.014, 0.001, 0.001), Mat("#2a2420", rough=0.3), head * T(0, -0.02, 0.12) * RX(math.pi / 2 + 0.6))
    frost(sc, 2, t, k=3.5)
    return sc


# ─── 5 · Morse ──────────────────────────────────────────────────────────────
def walrus(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = root_of(anim, t)
    m = hide("#7e5644", 30)
    breathe = 0.012 * math.sin(TAU * t)
    sc.add(sd_ellipsoid(0.24, 0.17 + breathe, 0.34), m, root * T(0, 0.17, -0.08))
    up = 0.35 if anim == "win" else 0.1
    nk = root * T(0, 0.24, 0.12) * RX(-up)
    sc.add(sd_ellipsoid(0.19, 0.16, 0.15), m, nk, blend=0.08)
    head = nk * T(0, 0.12, 0.11) * RY(0.3 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)
    sc.add(sd_ellipsoid(0.11, 0.1, 0.1), m, head, blend=0.05)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.055), Mat("#b88a76", rough=0.8), head * T(0.04 * sgn, -0.04, 0.08), blend=0.02)   # the whisker pad
        for k in range(4):
            sc.add(sd_capsule((0, 0, 0), (0.05 * sgn, -0.01 * k, 0.03), 0.002), Mat("#f2e8d8", rough=0.4),
                   head * T(0.05 * sgn, -0.03 - 0.012 * k, 0.12))
        sc.add(sd_sphere(0.013), EYE, head * T(0.06 * sgn, 0.04, 0.06))
        tusk = bezier((0, 0, 0), (0, -0.08, 0.02), (0, -0.16, 0.0), (0.01 * sgn, -0.24, -0.03), 4)
        chain(sc, tusk, [0.018, 0.016, 0.012, 0.004], IVORY, head * T(0.035 * sgn, -0.07, 0.1), blend=0.004)
        flip = 0.5 * math.sin(TAU * t * 4) if anim == "win" else 0.0
        sc.add(sd_ellipsoid(0.09, 0.02, 0.06), m, root * T(0.22 * sgn, 0.05, 0.12) * RZ(-0.3 * sgn - flip * sgn))
        sc.add(sd_ellipsoid(0.06, 0.015, 0.1), m, root * T(0.08 * sgn, 0.05, -0.44) * RY(0.4 * sgn))
    frost(sc, 2, t, k=4.0)
    return sc


# ─── 6 · Loup arctique ──────────────────────────────────────────────────────
def arctic_wolf(anim, t, side):
    sc = Scene()
    base(sc, side)
    fm, bm = fur("#e8eaee"), fur("#fafbfc")
    root, head = quadruped(sc, anim, t, fm, bm, s=1.1, leg=1.05, snout=1.05)
    sw = 0.15 * math.sin(TAU * t)
    tail = bezier((0, 0.02, -0.23), (0, 0.0, -0.33), (0.02 + sw * 0.1, -0.1, -0.36), (0.03 + sw * 0.1, -0.17, -0.33), 5)
    chain(sc, tail, [0.035, 0.045, 0.04, 0.035, 0.015], fm, root, blend=0.015)
    frost(sc, 3, t, k=4.5)
    return sc


# ─── 7 · Exploratrice polaire ───────────────────────────────────────────────
def explorer(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.6, 0.3, 0.9] if anim != "win" else [2.6, 0.2, 0.2]
    p.l = [0.3, 0.3, 0.7]
    parka = fur("#c8302a", 60, 0.25)
    trim = fur("#e8dcc8", 160, 0.8)
    m = {"skin": skin("#e2b89a"), "top": parka, "legs": Mat("#2a2e3a", rough=0.8), "boots": Mat("#3a2a22", rough=0.7),
         "arms": parka, "hands": Mat("#2a2a30", rough=0.8)}
    rig = adult(sc, p, m, H=1.0, bulk=1.15)
    x, r = rig["head"], rig["hr"]
    sc.add(sd_ellipsoid(r * 0.86, r * 1.02, r * 0.92), skin("#e2b89a"), x)
    sc.add(sd_ellipsoid(r * 1.12, r * 1.22, r * 1.0), parka, x * T(0, r * 0.12, -r * 0.5))          # the hood, behind the face
    sc.add(sd_torus(r * 1.0, r * 0.24), trim, x * T(0, r * 0.08, r * 0.15) * RX(math.pi / 2))     # its fur ruff, framing the face
    sc.add(sd_box(r * 0.8, r * 0.17, r * 0.12, r * 0.06), Mat("#1a2a3a", metal=0.6, rough=0.1, clearcoat=1.0), x * T(0, r * 0.25, r * 0.8))   # goggles
    sc.add(sd_box(r * 0.9, r * 0.04, r * 0.05, r * 0.02), Mat("#3a2a22", rough=0.6), x * T(0, r * 0.25, r * 0.7))
    sc.add(sd_torus(0.1, 0.03), trim, rig["waist"] * T(0, -0.04, 0) * RX(0.0))
    sc.add(sd_box(0.11, 0.14, 0.06, 0.02), Mat("#4a5a3a", rough=0.8), rig["back"] * T(0, -0.02, -0.04))   # the backpack
    sc.add(sd_cylinder(0.11, 0.035), Mat("#8a7a5a", rough=0.8), rig["back"] * T(0, 0.14, -0.06) * RZ(math.pi / 2))   # a bedroll
    axe = held(rig["r"], -1.4 if anim != "win" else -math.pi)
    sc.add(sd_capsule((0, -0.12, 0), (0, 0.2, 0), 0.01), Mat("#5a4030", rough=0.6), axe)
    sc.add(sd_box(0.08, 0.012, 0.008, 0.003), Mat("#9aa4b0", metal=0.9, rough=0.3), axe * T(0.02, 0.2, 0) * RZ(0.15))
    frost(sc, 3, t, k=5.0)
    return sc


# ─── 8 · Orque ──────────────────────────────────────────────────────────────
def orca(anim, t, side):
    sc = Scene()
    base(sc, side)
    sc.add(sd_cylinder(0.006, 0.36), Mat("#04121e", rough=0.05, clearcoat=1.0), T(0, BASE_TOP + 0.004, 0))   # the hole: dark water
    sc.add(sd_torus(0.37, 0.03), ICE, T(0, BASE_TOP + 0.01, 0))
    black = Mat("#101216", rough=0.25, clearcoat=0.8)
    white = Mat("#f2f2ee", rough=0.3, clearcoat=0.8)
    k = 0.5 + 0.5 * math.sin(TAU * t) if anim != "win" else 1.0
    rise = 0.1 + 0.12 * k
    yaw = RY(HEAD + (0.3 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0))
    # the body: an arc leaping sideways out of the water, the head up
    up = 0.08 * k
    pts = bezier((-0.22, BASE_TOP - 0.04, 0), (-0.2, BASE_TOP + 0.3 + up, 0), (0.05, BASE_TOP + 0.55 + up, 0), (0.24, BASE_TOP + 0.5 + up, 0), 8)
    radii = [0.05, 0.08, 0.11, 0.13, 0.135, 0.12, 0.09, 0.05]
    chain(sc, pts, radii, black, yaw, blend=0.04)
    for i in (3, 4, 5):   # the white belly, on the underside of the arc
        sc.add(sd_ellipsoid(0.08, 0.06, 0.09), white, yaw * T(pts[i][0] + 0.03, pts[i][1] - radii[i] * 0.55, pts[i][2]), blend=0.03)
    hd = pts[-1]
    sc.add(sd_ellipsoid(0.06, 0.035, 0.06), white, yaw * T(hd[0] - 0.01, hd[1] - 0.035, hd[2]), blend=0.02)   # the white jaw
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.045, 0.022, 0.012), white, yaw * T(pts[6][0], pts[6][1] + 0.03, 0.095 * sgn), blend=0.005)   # eye patches
        sc.add(sd_ellipsoid(0.1, 0.012, 0.05), black, yaw * T(pts[5][0], pts[5][1] - 0.06, 0.12 * sgn) * RX(0.5 * sgn))   # pectoral fins
    sc.add(sd_cone(0.17, 0.06, 0.004, 0.003), black, yaw * T(pts[4][0] - 0.02, pts[4][1] + 0.1, 0) * RZ(0.35))   # the dorsal fin
    for sgn in (-1, 1):   # the tail flukes, at the water
        sc.add(sd_ellipsoid(0.03, 0.012, 0.09), black, yaw * T(-0.22, BASE_TOP + 0.02, 0.06 * sgn) * RY(0.4 * sgn))
    for k_ in range(6):   # spray
        a = TAU * (k_ / 6 + t * 0.3)
        sc.add(sd_sphere(0.022), Mat("#e8f6ff", rough=0.2, clearcoat=1.0), T(0.3 * math.cos(a), BASE_TOP + 0.05 + 0.06 * abs(math.sin(TAU * t + k_)), 0.3 * math.sin(a)))
    frost(sc, 4, t, r=0.46, k=5.0)
    return sc


# ─── 9 · Ours blanc ─────────────────────────────────────────────────────────
def polar_bear(anim, t, side):
    sc = Scene()
    base(sc, side)
    fm, bm = fur("#f2eee2", 110, 0.4), fur("#f8f4ea", 110, 0.4)
    root, head = quadruped(sc, anim, t, fm, bm, s=1.35, leg=0.85, bulk=1.4, snout=0.8, ears=0.7, ear_round=True, neck=0.9)
    sc.add(sd_ellipsoid(0.12, 0.08, 0.1), fm, root * T(0, 0.1, -0.05), blend=0.06)    # the shoulders' hump
    frost(sc, 5, t, r=0.48, k=5.5)
    aurora_ring(sc, t)
    return sc


# ─── 10 · Mammouth laineux ──────────────────────────────────────────────────
def mammoth(anim, t, side):
    sc = Scene()
    base(sc, side)
    hair = fur("#6a4a2e", 90, 0.7)
    dark = fur("#4a3220", 90, 0.6)
    gait = TAU * t * (2 if anim == "move" else 0.5)
    L = 0.2
    root = RY(HEAD + (0.2 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)) * T(0, BASE_TOP + L + 0.03, 0)
    sc.add(sd_ellipsoid(0.2, 0.18, 0.3), hair, root * T(0, 0.1, -0.03))
    sc.add(sd_ellipsoid(0.16, 0.14, 0.14), hair, root * T(0, 0.24, 0.08), blend=0.08)     # the hump
    for i, (sx, sz) in enumerate(((-1, 1), (1, 1), (-1, -1), (1, -1))):
        sw = 0.25 * math.sin(gait + (math.pi if i in (1, 2) else 0)) if anim == "move" else 0.0
        hip = root * T(0.12 * sx, 0.0, 0.17 * sz) * RX(sw)
        sc.add(sd_capsule((0, 0.05, 0), (0, -L, 0), 0.06, 0.055), dark, hip, blend=0.03)
        sc.add(sd_cylinder(0.015, 0.06, 0.01), Mat("#2a2018", rough=0.7), hip * T(0, -L, 0))
    for k in range(14):   # the long hair hanging along the flanks
        a = TAU * k / 14
        sc.add(sd_capsule((0, 0, 0), (0, -0.14, 0), 0.03, 0.012), hair, root * T(0.19 * math.cos(a), 0.0, 0.27 * math.sin(a)), blend=0.03)
    trunk_up = anim == "win"
    head = root * T(0, 0.24, 0.27) * RX(-0.3 if trunk_up else 0.05)
    sc.add(sd_ellipsoid(0.12, 0.14, 0.11), hair, head, blend=0.06)
    sc.add(sd_sphere(0.08), hair, head * T(0, 0.1, -0.02), blend=0.05)                    # the domed skull
    sw = 0.08 * math.sin(TAU * t)
    tp = bezier((0, -0.04, 0.08), (0, -0.2, 0.16), (0, -0.32 + (0.3 if trunk_up else 0), 0.14 + sw), (0, -0.3 + (0.45 if trunk_up else 0), 0.24 + sw), 7)
    chain(sc, tp, [0.05, 0.045, 0.04, 0.035, 0.03, 0.026, 0.022], dark, head, blend=0.02)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.06, 0.08, 0.02), dark, head * T(0.11 * sgn, 0.0, -0.04) * RY(0.6 * sgn), blend=0.02)   # small ears
        sc.add(sd_sphere(0.012), EYE, head * T(0.075 * sgn, 0.03, 0.07))
        tusk = bezier((0, 0, 0), (0.06 * sgn, -0.12, 0.12), (0.16 * sgn, -0.05, 0.26), (0.1 * sgn, 0.08, 0.32), 7)
        chain(sc, tusk, [0.026, 0.025, 0.022, 0.019, 0.015, 0.01, 0.004], IVORY, head * T(0.05 * sgn, -0.06, 0.06), blend=0.006)
    frost(sc, 7, t, r=0.5, h=0.45, k=6.5)
    aurora_ring(sc, t)
    return sc


SCALE = {"random": 1.3, "beginner": 1.2, "monte_carlo": 1.3, "mcts": 1.2, "basic_minmax": 1.15, "nnue193": 1.15,
         "features": 1.25, "resnet": 1.15, "dev": 1.05, "alphazero": 1.05}
CAST = {"random": ("Phoque", seal), "beginner": ("Manchot empereur", penguin), "monte_carlo": ("Renard polaire", arctic_fox),
        "mcts": ("Harfang des neiges", snowy_owl), "basic_minmax": ("Morse", walrus), "nnue193": ("Loup arctique", arctic_wolf),
        "features": ("Exploratrice polaire", explorer), "resnet": ("Ours blanc", polar_bear), "dev": ("Orque", orca),
        "alphazero": ("Mammouth laineux", mammoth)}
ARCTIC = {"ice_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"ice_" + k: n for k, (n, _) in CAST.items()}
