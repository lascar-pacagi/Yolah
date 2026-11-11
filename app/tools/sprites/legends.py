"""
legends.py — the "Légendes" set (assets/sprites/sets.json): the orc hordes
(Black) against the alliance of elves and dwarves (White), four units each,
original designs in the fantasy tradition. Built on one humanoid rig
(humanoid): legs, torso, two jointed arms, a head, posed by the animation;
each unit adds its face, armour and weapon to the hands.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, sd_plane_cut, fbm, length
from models import base, BASE_TOP, HEADING, TAU, wave, glow, gold, armor, eye_mat, smoothstep

# ─── materials ──────────────────────────────────────────────────────────────
def leather(c="#5a3a24"):
    return Mat(c, rough=0.7, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.75 + 0.4 * fbm(pl * 30, 3))[:, None]})


def cloth(c, rough=0.85):
    return Mat(c, rough=rough, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.82 + 0.25 * fbm(pl * 45, 2))[:, None]})


def iron(c="#4a4d52", rough=0.45):
    return Mat(c, metal=0.9, rough=rough, pattern=lambda pl, pw, n: {
        "albedo": srgb(c) * (0.7 + 0.5 * fbm(pl * 25, 3))[:, None], "rough": rough + 0.2 * fbm(pl * 40, 2)})


def silver(rough=0.2):
    return Mat("#d6dbe2", metal=1.0, rough=rough)


def wood(c="#6b4a2b"):
    return Mat(c, rough=0.6, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.7 + 0.5 * (0.5 + 0.5 * torch.sin(pl[:, 1] * 120 + fbm(pl * 8) * 6)))[:, None]})


def skin(c, sss=0.4):
    return Mat(c, rough=0.55, sss=sss, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.85 + 0.25 * fbm(pl * 20, 3))[:, None]})


# ─── poses ──────────────────────────────────────────────────────────────────
class Pose:
    """Joint angles (radians) of the rig for one frame."""
    def __init__(self):
        self.bob = self.lean = self.shake = self.yaw = self.pitch = self.step = self.jump = 0.0
        # arms: shoulder pitch (forward +), spread (out +), elbow bend (+)
        self.l = [0.15, 0.25, 0.4]
        self.r = [0.15, 0.25, 0.4]


def base_pose(anim, t):
    p = Pose()
    if anim == "idle":
        p.bob = 0.008 * wave(t)
        p.yaw = 0.15 * wave(t, 1, 0.2)
    elif anim == "move":
        p.step = math.sin(math.pi * t)
        p.lean = 0.2 * math.sin(math.pi * t)
        p.bob = 0.04 * math.sin(math.pi * t)
        p.l[0] = 0.15 - 0.5 * p.step
        p.r[0] = 0.15 + 0.5 * p.step
    elif anim == "blocked":
        p.shake = 0.06 * math.sin(TAU * 3 * t)
        p.yaw = 0.4 * math.sin(TAU * 2 * t)
        p.pitch = 0.15
    elif anim == "win":
        p.jump = 0.06 * abs(math.sin(TAU * t))
        p.pitch = -0.25
        p.r = [2.45 + 0.12 * wave(t, 2), 0.2, 0.15]
        p.l = [0.5, 0.5, 1.0]
    return p


# ─── the rig ────────────────────────────────────────────────────────────────
def humanoid(sc, p, m, w=1.0, h=1.0, head_r=0.16, limb=1.0, belly=1.0):
    """Adds legs, torso, arms and neck; returns the transforms of the head,
    the hands (local −y along the forearm), the chest and the back.
    m: materials {"skin", "top", "legs", "boots", "arms"}."""
    y0 = BASE_TOP
    body = RY(HEADING + p.shake) * T(0, p.bob + p.jump, 0)
    hip_y = y0 + 0.24 * h
    for sgn in (-1, 1):
        s_ = p.step * sgn * 0.6
        hip = body * T(0.075 * w * sgn, hip_y, 0)
        th = hip * RX(-0.6 * s_)
        sc.add(sd_capsule((0, 0, 0), (0, -0.11 * h, 0.01), 0.05 * w * limb), m["legs"], th)
        kn = th * T(0, -0.11 * h, 0.01) * RX(0.5 * max(0.0, s_))
        sc.add(sd_capsule((0, 0, 0), (0, -0.09 * h, 0), 0.045 * w * limb, 0.05 * w * limb), m["boots"], kn)
        sc.add(sd_box(0.05 * w * limb, 0.03, 0.08, 0.025), m["boots"], kn * T(0, -0.11 * h, 0.025))
    torso = body * T(0, hip_y, 0) * RX(p.lean)
    sc.add(sd_ellipsoid(0.13 * w * belly, 0.09 * h, 0.1 * w * belly), m["legs"], torso * T(0, 0.03 * h, 0))
    sc.add(sd_ellipsoid(0.15 * w, 0.14 * h, 0.115 * w), m["top"], torso * T(0, 0.17 * h, 0), blend=0.04)
    hands = []
    for sgn, (pitch, spread, elbow) in ((-1, p.l), (1, p.r)):
        sh = torso * T(0.16 * w * sgn, 0.24 * h, 0) * RZ(spread * sgn) * RX(-pitch)
        sc.add(sd_sphere(0.055 * w * limb), m["top"], sh)
        sc.add(sd_capsule((0, 0, 0), (0, -0.1 * h, 0), 0.04 * w * limb), m["arms"], sh)
        el = sh * T(0, -0.1 * h, 0) * RX(-elbow)
        sc.add(sd_capsule((0, 0, 0), (0, -0.09 * h, 0), 0.037 * w * limb, 0.04 * w * limb), m["arms"], el)
        hand = el * T(0, -0.11 * h, 0)
        sc.add(sd_sphere(0.04 * w * limb), m["skin"], hand)
        hands.append(hand)
    sc.add(sd_capsule((0, 0.25 * h, 0), (0, 0.3 * h, 0), 0.045 * w), m["skin"], torso)
    head = torso * T(0, 0.31 * h + head_r * 0.85, 0.01) * RY(p.yaw) * RX(p.pitch - 0.3)
    return {"head": head, "l": hands[0], "r": hands[1], "torso": torso, "chest": torso * T(0, 0.17 * h, 0.1 * w),
            "back": torso * T(0, 0.2 * h, -0.11 * w), "body": body}


def held(hand, ax=-1.35, az=0.0):
    """A frame in the hand for a weapon: its +y points out of the fist, up and
    forward for ax about −1.3 (the hand's −y runs along the forearm)."""
    return hand * RX(-ax) * RZ(az)


# ─── faces ──────────────────────────────────────────────────────────────────
def orc_head(sc, x, r, sk, anim, t, helmet=None):
    sc.add(sd_ellipsoid(r * 1.05, r * 0.95, r), sk, x)
    sc.add(sd_ellipsoid(r * 0.9, r * 0.55, r * 0.75), sk, x * T(0, -r * 0.45, r * 0.25), blend=0.04)    # big jaw
    sc.add(sd_ellipsoid(r * 0.85, r * 0.22, r * 0.3), sk, x * T(0, r * 0.25, r * 0.75), blend=0.03)    # brow ridge
    sc.add(sd_ellipsoid(r * 0.25, r * 0.18, r * 0.2), sk, x * T(0, -r * 0.02, r * 0.95), blend=0.03)  # flat nose
    angry = anim in ("blocked", "win")
    for sgn in (-1, 1):
        sc.add(sd_sphere(r * 0.13), glow("#ff9a1a", 5.0 if angry else 3.0), x * T(r * 0.36 * sgn, r * 0.1, r * 0.86))
        sc.add(sd_cone(r * 0.12, r * 0.07, r * 0.015, 0.003), Mat("#efe6cf", rough=0.4), x * T(r * 0.35 * sgn, -r * 0.42, r * 0.9) * RZ(-0.2 * sgn))
        ear = x * T(r * 0.98 * sgn, r * 0.1, -r * 0.05) * RZ(-1.2 * sgn) * RX(0.2)
        sc.add(sd_cone(r * 0.35, r * 0.18, r * 0.02, 0.01), sk, ear * T(0, r * 0.3, 0))
    if anim in ("blocked", "win"):   # a roar
        sc.add(sd_ellipsoid(r * 0.45, r * 0.2, r * 0.3), Mat("#3a0c0c", rough=0.6), x * T(0, -r * 0.5, r * 0.75))
    if helmet:
        helmet(sc, x, r)


def fair_head(sc, x, r, sk, hair_c, anim, t, ears=True, beard=None):
    """An elf's (or a dwarf's) face: eyes, brows, nose, mouth; long ears."""
    sc.add(sd_ellipsoid(r, r * 1.05, r * 0.97), sk, x)
    sc.add(sd_ellipsoid(r * 0.7, r * 0.55, r * 0.7), sk, x * T(0, -r * 0.45, r * 0.25), blend=0.06)
    sc.add(sd_capsule((0, 0.02, 0), (0, -0.03, 0.02), r * 0.1, r * 0.13), sk, x * T(0, -r * 0.05, r * 0.95), blend=0.02)
    look = 0.25 * math.sin(TAU * t) if anim == "idle" else 0.0
    blink = anim == "idle" and abs(t - 0.6) < 0.04
    for sgn in (-1, 1):
        ec = x * T(r * 0.36 * sgn, r * 0.12, r * 0.78)
        if anim == "win" or blink:
            sc.add(sd_capsule((-r * 0.14, 0, 0), (r * 0.14, 0, 0), r * 0.035), Mat("#2a1a12", rough=0.6), ec * T(0, 0, r * 0.12))
        else:
            sc.add(sd_sphere(r * 0.2), eye_mat((look, 0.0, 1.0), iris="#3f7f5f" if ears else "#4a6a8a"), ec)
        bi = 0.06 * r if anim == "blocked" else 0.0
        sc.add(sd_capsule((0, bi, 0), (r * 0.3 * sgn, -bi * 0.4, -r * 0.05), r * 0.05), Mat(hair_c, rough=0.7),
               x * T(r * 0.2 * sgn, r * 0.42, r * 0.85))
        if ears:
            sc.add(sd_cone(r * 0.4, r * 0.16, r * 0.02, 0.01), sk, x * T(r * 0.95 * sgn, r * 0.15, -r * 0.05) * RZ(-1.15 * sgn) * RX(0.25) * T(0, r * 0.35, 0))
    mo = x * T(0, -r * 0.45, r * 0.86)
    if anim == "win":
        sc.add(sd_ellipsoid(r * 0.25, r * 0.12, r * 0.15), Mat("#5a1a1a", rough=0.6), mo)
    elif anim == "blocked":
        sc.add(sd_capsule((-r * 0.18, -r * 0.03, 0), (r * 0.18, -r * 0.03, 0), r * 0.03), Mat("#8a4a40", rough=0.5), mo * T(0, 0.01, 0))
    else:
        sc.add(sd_capsule((-r * 0.18, r * 0.02, 0), (0, -r * 0.03, 0), r * 0.025), Mat("#9a5a50", rough=0.5), mo)
        sc.add(sd_capsule((0, -r * 0.03, 0), (r * 0.18, r * 0.02, 0), r * 0.025), Mat("#9a5a50", rough=0.5), mo)
    if beard:
        beard(sc, x, r)


ORC_SKIN = "#6d8b46"


def orc_mats(top="#4a4d52"):
    sk = skin(ORC_SKIN, 0.3)
    return sk, {"skin": sk, "top": iron(top), "legs": leather("#3e2a1c"), "boots": leather("#2a1d14"), "arms": sk}


# ─── the orcs (Black) ───────────────────────────────────────────────────────
def orc_grunt(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.l = [0.6, 0.35, 1.2]                       # the shield held in front
    if anim == "move":
        p.r[0] = 0.15 + 0.6 * p.step
    if anim == "win":
        p.r = [2.45, 0.25, 0.15]
    sk, m = orc_mats()
    rig = humanoid(sc, p, m, w=1.12, h=1.0, head_r=0.17, belly=1.1)
    red = Mat("#8a1f1a", rough=0.6)
    sc.add(sd_box(0.16, 0.025, 0.12, 0.02), red, rig["torso"] * T(0, 0.06, 0.0))          # a belt-sash
    sc.add(sd_box(0.03, 0.05, 0.02, 0.01), iron("#7a6a40"), rig["torso"] * T(0, 0.06, 0.12))

    def helmet(sc, x, r):
        sc.add(sd_plane_cut(sd_ellipsoid(r * 1.12, r * 1.0, r * 1.08), (0, -1, 0), -r * 0.15), iron("#5e626a", 0.35), x * T(0, r * 0.12, -r * 0.02))
        for sgn in (-1, 1):
            sc.add(sd_capsule((0, 0, 0), (r * 0.5 * sgn, r * 0.7, -r * 0.2), r * 0.13, r * 0.03), Mat("#e8dcc0", rough=0.5),
                   x * T(r * 0.75 * sgn, r * 0.55, 0))
    orc_head(sc, rig["head"], 0.17, sk, anim, t, helmet)
    # round wooden shield with an iron rim and boss, on the left forearm
    sh = rig["l"] * RX(-1.5) * T(0, 0.0, 0.06)
    sc.add(sd_cylinder(0.02, 0.17, 0.01), wood("#5a3c22"), sh * RX(math.pi / 2))
    sc.add(sd_torus(0.17, 0.016), iron(), sh * RX(math.pi / 2) * T(0, 0, 0))
    sc.add(sd_sphere(0.045), iron("#6a6a6a", 0.3), sh * T(0, 0, 0.02))
    sc.add(sd_box(0.15, 0.02, 0.01, 0.005), Mat("#8a1f1a", rough=0.6), sh * T(0, 0, 0.022) * RZ(0.8))
    # a cleaver
    ax = held(rig["r"], -math.pi if anim == "win" else -1.35)
    sc.add(sd_capsule((0, -0.04, 0), (0, 0.26, 0), 0.018), wood(), ax)
    sc.add(sd_box(0.1, 0.12, 0.01, 0.008), iron("#6a6e76", 0.25), ax * T(0.09, 0.21, 0))
    sc.add(sd_box(0.004, 0.11, 0.011, 0.002), Mat("#d8dce2", metal=1.0, rough=0.15), ax * T(0.19, 0.21, 0))
    return sc


def orc_shaman(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.lean += 0.15                              # hunched
    p.r = [0.5, 0.3, 0.9]
    if anim == "win":
        p.r = [2.45, 0.2, 0.15]
    sk, m = orc_mats()
    robe = cloth("#5a2a1e")
    m.update({"top": robe, "legs": robe, "arms": robe})
    rig = humanoid(sc, p, m, w=0.95, h=1.0, head_r=0.16)
    sc.add(sd_cone(0.14, 0.17, 0.11, 0.02), robe, rig["torso"] * T(0, -0.06, 0))          # robe skirt
    for k in range(7):                                                                    # bone necklace
        a = -0.9 + 0.3 * k
        sc.add(sd_capsule((0, 0, 0), (0, -0.03, 0.01), 0.008), Mat("#efe6cf", rough=0.5),
               rig["torso"] * T(0.12 * math.sin(a), 0.26, 0.1 * math.cos(a)))

    def hood(sc, x, r):
        sc.add(sd_plane_cut(sd_ellipsoid(r * 1.25, r * 1.2, r * 1.2), (0, 0, 1), r * 0.55), robe, x * T(0, r * 0.15, -r * 0.1))
    orc_head(sc, rig["head"], 0.16, sk, anim, t, hood)
    st = held(rig["r"], -math.pi if anim == "win" else -1.6)
    sc.add(sd_capsule((0, -0.22, 0), (0, 0.3, 0), 0.013), wood("#4a3420"), st)
    sc.add(sd_sphere(0.045), Mat("#efe6cf", rough=0.45), st * T(0, 0.34, 0))           # a skull
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.011), Mat("#100808", rough=0.6), st * T(0.016 * sgn, 0.345, 0.038))
    k = 6.0 + 3.0 * wave(t, 2) + (8.0 if anim == "win" else 0.0)
    sc.add(sd_sphere(0.03), glow("#5dff6a", k), st * T(0, 0.4, 0))
    if anim in ("win", "idle"):
        for i in range(3):
            a = TAU * (t * 2 + i / 3)
            sc.add(sd_sphere(0.012), glow("#5dff6a", 6.0), st * T(0.07 * math.cos(a), 0.4 + 0.05 * math.sin(a * 2), 0.07 * math.sin(a)))
    return sc


def orc_brute(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.4, 0.3, 0.6]
    p.l = [0.2, 0.45, 0.5]
    if anim == "win":
        p.r = [2.45, 0.3, 0.15]
        p.l = [2.9, -0.3, 0.2]
    sk, m = orc_mats("#3a3c40")
    m["skin"] = skin("#5f7a3e", 0.3)
    m["arms"] = m["skin"]
    rig = humanoid(sc, p, m, w=1.45, h=1.05, head_r=0.13, limb=1.3, belly=1.25)
    for sgn in (-1, 1):                                                                   # spiked pauldrons
        pad = rig["torso"] * T(0.23 * sgn, 0.27, 0)
        sc.add(sd_ellipsoid(0.11, 0.07, 0.1), iron("#3a3c40"), pad)
        for k in range(3):
            sc.add(sd_cone(0.03, 0.018, 0.002, 0.002), iron("#8a8a8a", 0.3), pad * T(0.05 * sgn * (k - 1) * 0.6, 0.06, 0.03 * (k - 1)) * RZ(-0.4 * sgn))

    def helm(sc, x, r):
        sc.add(sd_plane_cut(sd_ellipsoid(r * 1.15, r * 1.0, r * 1.1), (0, -1, 0), -r * 0.2), iron("#565a62", 0.35), x * T(0, r * 0.15, 0))
        sc.add(sd_box(r * 0.08, r * 0.5, r * 0.08, r * 0.03), iron("#565a62", 0.35), x * T(0, -r * 0.1, r * 1.0))
    orc_head(sc, rig["head"], 0.13, m["skin"], anim, t, helm)
    cl = held(rig["r"], -math.pi if anim == "win" else -1.1)
    sc.add(sd_capsule((0, -0.02, 0), (0, 0.36, 0), 0.03, 0.085), wood("#4a3220"), cl)
    for k in range(8):
        a = TAU * k / 8
        sc.add(sd_cone(0.035, 0.018, 0.002, 0.002), iron("#a8a8a8", 0.3),
               cl * T(0.08 * math.cos(a), 0.29 + 0.04 * (k % 2), 0.08 * math.sin(a)) * RY(-a) * RZ(-math.pi / 2) * T(0, 0.02, 0))
    return sc


def orc_archer(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    draw_k = 0.5 + 0.5 * wave(t) if anim == "idle" else (1.0 if anim == "win" else 0.3)
    p.l = [1.45, 0.15, 0.05]                      # the bow arm forward
    p.r = [1.3, -0.2, 1.4 + 0.5 * draw_k]         # drawing the string
    if anim == "win":
        p.l = [2.7, 0.2, 0.1]
    sk, m = orc_mats()
    m["top"] = leather("#4a3424")
    rig = humanoid(sc, p, m, w=0.95, h=1.0, head_r=0.155)

    def bandana(sc, x, r):
        sc.add(sd_plane_cut(sd_ellipsoid(r * 1.08, r * 0.95, r * 1.05), (0, -1, 0), -r * 0.3), Mat("#2a2420", rough=0.8), x * T(0, r * 0.12, -r * 0.02))
        sc.add(sd_capsule((0, 0, 0), (-0.03, -0.1, -0.08), 0.02, 0.008), Mat("#2a2420", rough=0.8), x * T(0, r * 0.4, -r * 0.9))
    orc_head(sc, rig["head"], 0.155, sk, anim, t, bandana)
    bw = rig["l"] * RX(-1.5)
    for k in range(6):                                                               # the bow: an arc of capsules
        a0, a1 = -1.0 + 0.4 * k, -1.0 + 0.4 * (k + 1)
        q0 = (0.0, 0.2 * math.sin(a0), 0.06 * math.cos(a0) - 0.06)
        q1 = (0.0, 0.2 * math.sin(a1), 0.06 * math.cos(a1) - 0.06)
        sc.add(sd_capsule(q0, q1, 0.012), wood("#3a2818"), bw)
    sc.add(sd_capsule((0, -0.17, -0.03), (0, 0.17, -0.03), 0.003), Mat("#d8d0c0", rough=0.6), bw * T(0, 0, -0.05 * draw_k))
    sc.add(sd_box(0.05, 0.11, 0.04, 0.03), leather("#3a2618"), rig["back"] * RZ(0.3))       # quiver
    for k in range(3):
        sc.add(sd_capsule((0, 0, 0), (0, 0.06, 0), 0.006), Mat("#c84a2a", rough=0.6), rig["back"] * RZ(0.3) * T(-0.02 + 0.02 * k, 0.11, 0))
    return sc


# ─── the alliance (White) ───────────────────────────────────────────────────
ELF_SKIN = "#f1d2bd"


def elf_knight(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.l = [0.55, 0.3, 1.2]
    p.r = [0.35, 0.25, 0.6]
    if anim == "win":
        p.r = [2.45, 0.2, 0.15]
    plate = Mat("#e3e7ee", metal=1.0, rough=0.18)
    trim = gold()
    sk = skin(ELF_SKIN, 0.5)
    m = {"skin": sk, "top": plate, "legs": plate, "boots": plate, "arms": plate}
    rig = humanoid(sc, p, m, w=1.05, h=1.02, head_r=0.16)
    sc.add(sd_torus(0.14, 0.012), trim, rig["torso"] * T(0, 0.08, 0))
    sc.add(sd_cone(0.1, 0.15, 0.12, 0.02), cloth("#2f6aa8"), rig["torso"] * T(0, -0.04, 0))     # tabard
    cape = cloth("#2f6aa8")
    sc.add(sd_box(0.13, 0.2, 0.015, 0.012), cape, rig["back"] * T(0, -0.12, -0.02) * RX(0.15 + 0.1 * p.lean))
    r = 0.16
    x = rig["head"]
    fair_head(sc, x, r, sk, "#d9c48a", anim, t, ears=True)
    # an open helm with a leaf crest (not hiding the face)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.12, r * 1.1, r * 1.1), (0, -0.4, 1), r * 0.25), plate, x * T(0, r * 0.12, -r * 0.05))
    sc.add(sd_ellipsoid(r * 0.12, r * 0.55, r * 0.6), trim, x * T(0, r * 1.05, -r * 0.1) * RX(-0.4))
    # a kite shield with a gold tree, a long sword
    sh = rig["l"] * RX(-1.5) * T(0, 0, 0.06)
    sc.add(sd_box(0.13, 0.17, 0.015, 0.06), Mat("#2f6aa8", rough=0.4, clearcoat=0.8), sh)
    sc.add(sd_box(0.135, 0.175, 0.012, 0.065), trim, sh * T(0, 0, -0.006))
    sc.add(sd_capsule((0, -0.08, 0), (0, 0.08, 0), 0.012), trim, sh * T(0, 0, 0.018))
    for sgn in (-1, 1):
        sc.add(sd_capsule((0, 0.02, 0), (0.06 * sgn, 0.08, 0), 0.008), trim, sh * T(0, 0, 0.018))
    sw = held(rig["r"], -math.pi if anim == "win" else -1.25)
    sc.add(sd_capsule((0, -0.07, 0), (0, 0.04, 0), 0.014), Mat("#3a2a1a", rough=0.6), sw)
    sc.add(sd_sphere(0.022), trim, sw * T(0, -0.08, 0))
    sc.add(sd_box(0.085, 0.013, 0.016, 0.006), trim, sw * T(0, 0.05, 0))
    sc.add(sd_box(0.02, 0.28, 0.005, 0.004), silver(0.08), sw * T(0, 0.33, 0))
    if anim == "win":
        sc.add(sd_box(0.008, 0.25, 0.002, 0.001), glow("#bfe8ff", 4.0), sw * T(0, 0.33, 0.005))
    return sc


def elf_archer(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    draw_k = 0.5 + 0.5 * wave(t) if anim == "idle" else (1.0 if anim == "win" else 0.3)
    p.l = [1.45, 0.15, 0.05]
    p.r = [1.3, -0.2, 1.4 + 0.5 * draw_k]
    if anim == "win":
        p.l = [2.7, 0.2, 0.1]
    green = cloth("#3f6b3a")
    sk = skin(ELF_SKIN, 0.5)
    m = {"skin": sk, "top": leather("#6a4a2a"), "legs": cloth("#4a3a2a"), "boots": leather("#3a2a1a"), "arms": green}
    rig = humanoid(sc, p, m, w=0.9, h=1.05, head_r=0.155)
    sc.add(sd_box(0.14, 0.2, 0.015, 0.012), green, rig["back"] * T(0, -0.1, -0.02) * RX(0.1))   # cloak
    r = 0.155
    x = rig["head"]
    fair_head(sc, x, r, sk, "#e6d08a", anim, t)
    # long golden hair under a green hood
    sc.add(sd_ellipsoid(r * 0.9, r * 1.3, r * 0.5), Mat("#e6c46a", rough=0.4, clearcoat=0.5), x * T(0, -r * 0.4, -r * 0.65))
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.2, r * 1.15, r * 1.15), (0, 0, 1), r * 0.5), green, x * T(0, r * 0.15, -r * 0.08))
    bw = rig["l"] * RX(-1.5)
    for k in range(8):
        a0, a1 = -1.2 + 0.3 * k, -1.2 + 0.3 * (k + 1)
        q0 = (0.0, 0.26 * math.sin(a0), 0.07 * math.cos(a0) - 0.07)
        q1 = (0.0, 0.26 * math.sin(a1), 0.07 * math.cos(a1) - 0.07)
        sc.add(sd_capsule(q0, q1, 0.01), Mat("#e8d9b0", rough=0.35, clearcoat=0.6), bw)
    sc.add(sd_capsule((0, -0.24, -0.04), (0, 0.24, -0.04), 0.003), Mat("#f0ece0", rough=0.5), bw * T(0, 0, -0.05 * draw_k))
    sc.add(sd_capsule((0, 0, -0.05 * draw_k - 0.04), (0, 0, 0.2), 0.005), wood("#8a6a40"), bw)      # the arrow
    sc.add(sd_cone(0.02, 0.012, 0.001, 0.001), silver(), bw * T(0, 0, 0.21) * RX(math.pi / 2))
    sc.add(sd_box(0.045, 0.11, 0.04, 0.03), leather("#6a4a2a"), rig["back"] * RZ(-0.3))
    for k in range(3):
        sc.add(sd_capsule((0, 0, 0), (0, 0.06, 0), 0.006), Mat("#f2f0ea", rough=0.6), rig["back"] * RZ(-0.3) * T(-0.02 + 0.02 * k, 0.11, 0))
    return sc


def elf_mage(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.5, 0.3, 0.9]
    p.l = [0.6, 0.4, 1.4]
    if anim == "win":
        p.r = [2.45, 0.2, 0.15]
        p.l = [2.0, 0.8, 0.3]
    robe = cloth("#f2f1ec", 0.7)
    sk = skin(ELF_SKIN, 0.5)
    m = {"skin": sk, "top": robe, "legs": robe, "boots": cloth("#c8c4b8"), "arms": robe}
    rig = humanoid(sc, p, m, w=0.95, h=1.05, head_r=0.155)
    sc.add(sd_cone(0.15, 0.17, 0.1, 0.02), robe, rig["torso"] * T(0, -0.06, 0))
    sc.add(sd_torus(0.105, 0.01), gold(), rig["torso"] * T(0, 0.08, 0))
    sc.add(sd_box(0.012, 0.13, 0.005, 0.003), gold(), rig["torso"] * T(0, 0.0, 0.135) * RX(-0.12))
    r = 0.155
    x = rig["head"]
    fair_head(sc, x, r, sk, "#c8ccd4", anim, t)
    hair = Mat("#e6e8ee", rough=0.35, clearcoat=0.6)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.08, r * 1.08, r * 1.06), (0, -0.3, 1), r * 0.35), hair, x * T(0, r * 0.12, -r * 0.03))
    sc.add(sd_ellipsoid(r * 0.95, r * 1.5, r * 0.45), hair, x * T(0, -r * 0.55, -r * 0.6))
    sc.add(sd_torus(r * 0.98, r * 0.06), gold(), x * T(0, r * 0.35, 0) * RX(0.15))        # a circlet
    sc.add(sd_sphere(r * 0.1), glow("#7fd8ff", 4.0), x * T(0, r * 0.4, r * 0.95))
    st = held(rig["r"], -math.pi if anim == "win" else -1.6)
    sc.add(sd_capsule((0, -0.22, 0), (0, 0.3, 0), 0.012), Mat("#e8e2d0", rough=0.35, clearcoat=0.6), st)
    for k in range(3):
        a = TAU * k / 3
        sc.add(sd_capsule((0, 0.28, 0), (0.035 * math.cos(a), 0.36, 0.035 * math.sin(a)), 0.006), gold(), st)
    k = 7.0 + 3.0 * wave(t, 2) + (10.0 if anim == "win" else 0.0)
    sc.add(sd_ellipsoid(0.024, 0.04, 0.024), glow("#7fd8ff", k), st * T(0, 0.36, 0))
    if anim in ("idle", "win"):
        for i in range(4):
            a = TAU * (t + i / 4)
            sc.add(sd_sphere(0.009), glow("#bfeaff", 7.0), rig["l"] * T(0.06 * math.cos(a), 0.05 + 0.03 * math.sin(2 * a), 0.06 * math.sin(a)))
    return sc


def dwarf(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.l = [0.55, 0.3, 1.2]
    p.r = [0.6, 0.3, 0.9]
    if anim == "win":
        p.r = [2.45, 0.25, 0.15]
    mail = Mat("#9aa0a8", metal=0.9, rough=0.4, pattern=lambda pl, pw, n: {
        "albedo": srgb("#9aa0a8") * (0.7 + 0.5 * (torch.sin(pl[:, 0] * 260) * torch.sin(pl[:, 1] * 260) > 0).float())[:, None]})
    sk = skin("#e8b394", 0.5)
    m = {"skin": sk, "top": mail, "legs": cloth("#6a3a2a"), "boots": leather("#3a2618"), "arms": mail}
    rig = humanoid(sc, p, m, w=1.25, h=0.8, head_r=0.17, limb=1.2, belly=1.25)
    sc.add(sd_torus(0.15, 0.022), leather("#4a2e1c"), rig["torso"] * T(0, 0.06, 0))
    sc.add(sd_box(0.04, 0.03, 0.02, 0.01), gold(), rig["torso"] * T(0, 0.06, 0.15))
    r = 0.17
    x = rig["head"]
    beard_m = Mat("#c0552a", rough=0.6, pattern=lambda pl, pw, n: {"albedo": srgb("#c0552a") * (0.7 + 0.5 * (0.5 + 0.5 * torch.sin(pl[:, 0] * 160 + fbm(pl * 10) * 4)))[:, None]})

    def beard(sc, x, r):
        sc.add(sd_ellipsoid(r * 0.85, r * 0.75, r * 0.5), beard_m, x * T(0, -r * 0.7, r * 0.55), blend=0.02)
        for sgn in (-1, 1):   # two braids
            sc.add(sd_capsule((0, 0, 0), (0.02 * sgn, -0.14, 0.03), r * 0.16, r * 0.1), beard_m, x * T(r * 0.3 * sgn, -r * 1.1, r * 0.6))
            sc.add(sd_torus(r * 0.15, r * 0.04), gold(), x * T(r * 0.3 * sgn + 0.01 * sgn, -r * 1.65, r * 0.62))
        sc.add(sd_ellipsoid(r * 0.5, r * 0.12, r * 0.15), beard_m, x * T(0, -r * 0.3, r * 0.92))   # moustache
    fair_head(sc, x, r, sk, "#c0552a", anim, t, ears=False, beard=beard)
    helm = Mat("#b7bcc4", metal=1.0, rough=0.25)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.1, r * 1.0, r * 1.08), (0, -1, 0), -r * 0.25), helm, x * T(0, r * 0.2, -r * 0.02))
    sc.add(sd_torus(r * 1.02, r * 0.07), gold(), x * T(0, r * 0.25, 0))
    sc.add(sd_box(r * 0.07, r * 0.35, r * 0.06, r * 0.03), helm, x * T(0, r * 0.05, r * 1.0))      # nose guard
    # a round shield and a war hammer
    sh = rig["l"] * RX(-1.5) * T(0, 0, 0.06)
    sc.add(sd_cylinder(0.02, 0.15, 0.01), wood("#7a4a28"), sh * RX(math.pi / 2))
    sc.add(sd_torus(0.15, 0.015), gold(), sh * RX(math.pi / 2))
    sc.add(sd_sphere(0.04), gold(), sh * T(0, 0, 0.02))
    hm = held(rig["r"], -math.pi if anim == "win" else -1.2)
    sc.add(sd_capsule((0, -0.04, 0), (0, 0.3, 0), 0.016), wood("#5a3a20"), hm)
    sc.add(sd_box(0.1, 0.055, 0.055, 0.012), Mat("#9aa0a8", metal=1.0, rough=0.25), hm * T(0, 0.31, 0))
    sc.add(sd_box(0.102, 0.01, 0.057, 0.004), gold(), hm * T(0, 0.31, 0))
    return sc


SCALE = {"orc_brute": 1.2, "dwarf": 1.35}
LEGENDS = {"orc_grunt": orc_grunt, "orc_shaman": orc_shaman, "orc_brute": orc_brute, "orc_archer": orc_archer,
           "elf_knight": elf_knight, "elf_archer": elf_archer, "elf_mage": elf_mage, "dwarf": dwarf}
NAMES = {"orc_grunt": "Guerrier orque", "orc_shaman": "Chaman orque", "orc_brute": "Brute orque", "orc_archer": "Archer orque",
         "elf_knight": "Chevalier elfe", "elf_archer": "Archère elfe", "elf_mage": "Mage elfe", "dwarf": "Nain"}
