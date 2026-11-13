"""
dinos.py — the Dinosaures set: one species per AI, from the most peaceful to
the most ferocious, bigger and bigger:

  random        Brachiosaure      a gentle long neck, grazing high
  beginner      Parasaurolophus   a duck bill and a long tube crest, calling
  monte_carlo   Stégosaure        plates along the back, spikes on the tail
  mcts          Ankylosaure       a tank of armour, a club at the end of the tail
  basic_minmax  Tricératops       a frill and three horns, charging
  nnue193       Dilophosaure      the first hunter: slender, two thin crests
  features      Quetzalcoatlus    the giant flying reptile, wings raised in threat
  resnet        Allosaure         a big hunter, horns above the eyes, stripes
  dev           Spinosaure        a sail on the back, a long crocodile snout
  alphazero     Tyrannosaure      the king: a huge head, roaring

Shapes after the fossils (no film's look: no frill on the dilophosaur). Countershaded skins (dark back, light belly). Modelled at
unit size; SCALE makes the stronger ones bigger. The human is never part of a set.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm
from models import base, BASE_TOP, TAU, glow
from heroic import chain, bezier, HEAD
from arctic import fur, EYE, IVORY
from origami import sheet

YAW = HEAD + 0.6          # a three-quarter profile: the head to the right, the tail behind to the left
DIR = (math.sin(YAW), 0.0, math.cos(YAW))
CLAW = Mat("#2a2420", rough=0.35)
MOUTH = Mat("#6a1e22", rough=0.5)
PUPIL = Mat("#050404", rough=0.1, clearcoat=1.0)


def iris(c):
    """A reptile's eye: wet, barely lit from inside (no cartoon glow)."""
    return Mat(c, rough=0.12, clearcoat=1.0, emissive=c, emit=0.12)


def reptile_eye(sc, xf, x, y, z, r, sgn, mat, lid=None):
    """An eye half sunk in its socket: the iris, a vertical slit pupil looking
    sideways, and (lid) a bony brow hooding its upper half."""
    sc.add(sd_sphere(r), mat, xf * T(x, y, z))
    sc.add(sd_ellipsoid(0.24 * r, 0.8 * r, 0.32 * r), PUPIL, xf * T(x + sgn * 0.86 * r, y, z))
    if lid is not None:
        sc.add(sd_ellipsoid(1.7 * r, 0.75 * r, 3.4 * r), lid, xf * T(x + sgn * 0.1 * r, y + 0.95 * r, z + 0.3 * r), blend=0.9 * r)


def skin(back, belly, k=70, stripes=None, freq=26.0, spots=None, rough=0.72):
    """Countershaded: the back colour on top, the belly colour below; fine
    mottling; stripes across the body (on the back and flanks only) or spots."""
    cb, cl = srgb(back), srgb(belly)

    def pat(pl, pw, n):
        up = torch.clamp((n[:, 1] + 0.55) / 0.5, 0, 1)[:, None]
        a = cl * (1 - up) + cb * up
        a = a * (0.84 + 0.28 * fbm(pw * k, 3))[:, None]
        a = a * (1 - 0.12 * (fbm(pw * k * 2.5, 2) > 0.58).float())[:, None]    # scales
        along = pw[:, 0] * DIR[0] + pw[:, 2] * DIR[2]
        if stripes:
            st = (torch.sin(along * freq + fbm(pw * 6, 2) * 3) > 0.45).float()[:, None] * up
            a = a * (1 - 0.85 * st) + srgb(stripes) * 0.85 * st
        if spots:
            sp = (fbm(pw * 14, 2) > 0.62).float()[:, None] * up
            a = a * (1 - sp) + srgb(spots) * sp
        return {"albedo": a * 0.78}
    return Mat(back, rough=rough, sss=0.25, pattern=pat)


def sd_thin(rx, ry, rz):
    """A thin ellipsoid (plates, crests, the sail): its distance bound is very low
    far from it (shadow rays would stop short and draw fans of shadow); the
    bounding sphere, also a lower bound, takes over there."""
    f, R = sd_ellipsoid(rx, ry, rz), max(rx, ry, rz)
    return lambda p: torch.maximum(f(p), torch.linalg.norm(p, dim=-1) - R)


def flat(c, rough=0.6):
    def pat(pl, pw, n):
        return {"albedo": srgb(c) * (0.85 + 0.25 * fbm(pw * 50, 2))[:, None]}
    return Mat(c, rough=rough, pattern=pat)


def at(xf, p):
    """The frame xf moved to its local point p (keeps the orientation of the yaw frame only)."""
    q = xf.point(*p).tolist()
    return T(*q)


# ─── the theropods (two legs) ───────────────────────────────────────────────
def theropod(sc, anim, t, m, belly=None, leg=1.0, bulk=1.0, tail=1.0, neck=1.0, head=1.0, snout=1.0, width=1.0,
             arms="long", teeth=True, roar=0.0, crouch=0.0, eye=None, pitch=0.18):
    walk = anim == "move"
    gait = TAU * t * 2
    L = 0.36 * leg
    hip_y = BASE_TOP + L * (1 - 0.18 * crouch) - 0.01 + (0.025 * abs(math.sin(gait)) if walk else 0.0)
    breathe = 0.008 * math.sin(TAU * t)
    base_x = RY(YAW) * T(0, hip_y, 0.05)
    root = base_x * RX(-pitch + 0.15 * crouch)
    bm = belly or m
    # the body: pelvis, belly, chest
    sc.add(sd_ellipsoid(0.095 * bulk, 0.115 * bulk + breathe, 0.2), m, root * T(0, 0.0, 0.04))
    sc.add(sd_ellipsoid(0.09 * bulk, 0.11 * bulk + breathe, 0.12), m, root * T(0, -0.01, 0.2), blend=0.07)
    sc.add(sd_ellipsoid(0.07 * bulk, 0.07, 0.15), bm, root * T(0, -0.055 * bulk, 0.12), blend=0.05)
    # the tail, swaying
    sw = 0.07 * math.sin(TAU * t) + (0.09 * math.sin(gait) if walk else 0.0)
    pts = bezier((0, 0.02, -0.08), (sw * 0.2, 0.04, -0.34 * tail), (sw * 0.6, 0.03, -0.6 * tail), (sw, 0.0, -0.86 * tail), 9)
    chain(sc, pts, [0.092 * bulk * (1 - i / 9) ** 1.25 + 0.008 for i in range(9)], m, root, blend=0.03)
    # the legs: a muscled thigh, shin, metatarsus, three toes
    tk = L / 0.36
    for sgn, ph in ((-1, 0.0), (1, math.pi)):
        sw_ = 0.42 * math.sin(gait + ph) if walk else 0.0
        hip = RY(YAW) * T(0.08 * bulk * sgn, hip_y - 0.02, 0.05) * RX(sw_)
        knee = (0.012 * sgn, -0.15 * tk * (1 - 0.25 * crouch), 0.09 * tk + 0.05 * crouch)
        ankle = (0.006 * sgn, -0.29 * tk * (1 - 0.15 * crouch), -0.05 * tk)
        foot = (0.0, -(hip_y - 0.02 - BASE_TOP) + 0.018, 0.0)
        sc.add(sd_ellipsoid(0.06 * bulk, 0.12 * tk, 0.09 * tk), m, hip * T(0, -0.05 * tk, 0.035) * RX(-0.35), blend=0.05)
        chain(sc, [(0, -0.04, 0.02), knee, ankle, foot], [0.05 * bulk, 0.036 * bulk, 0.026, 0.021], m, hip, blend=0.025)
        for k in (-1, 0, 1):
            toe = hip * T(*foot) * RY(0.38 * k)
            sc.add(sd_capsule((0, 0, 0), (0, -0.006, 0.075), 0.017, 0.011), m, toe, blend=0.01)
            sc.add(sd_cone(0.024, 0.009, 0.001, 0.001), CLAW, toe * T(0, -0.008, 0.08) * RX(math.pi / 2 + 0.35))
    # the arms
    for sgn in (-1, 1):
        flap = 0.15 * math.sin(TAU * t) if anim != "win" else 0.5 * math.sin(TAU * 2 * t)
        sh = root * T(0.07 * bulk * sgn, -0.03, 0.27) * RX(0.5 + flap)
        if arms == "tiny":
            chain(sc, [(0, 0, 0), (0.012 * sgn, -0.045, 0.01), (0.006 * sgn, -0.06, 0.05)], [0.022, 0.016, 0.011], m, sh, blend=0.012)
            for f in (-1, 1):
                sc.add(sd_cone(0.012, 0.005, 0.001, 0.001), CLAW, sh * T(0.006 * sgn + 0.006 * f, -0.06, 0.065) * RX(math.pi / 2))
        else:
            hand = (0.02 * sgn, -0.13, 0.1)
            chain(sc, [(0, 0, 0), (0.025 * sgn, -0.08, 0.02), hand], [0.026, 0.019, 0.013], m, sh, blend=0.015)
            for f in (-1, 0, 1):
                sc.add(sd_capsule(hand, (hand[0] + 0.012 * f, hand[1] - 0.025, hand[2] + 0.03), 0.008, 0.005), m, sh, blend=0.005)
                sc.add(sd_cone(0.016, 0.006, 0.001, 0.001), CLAW, sh * T(hand[0] + 0.012 * f, hand[1] - 0.03, hand[2] + 0.04) * RX(2.2))
    # the neck and the head
    up = roar * 0.45 - crouch * 0.35
    nk = root * T(0, 0.03, 0.29) * RX(-0.75 - up)
    sc.add(sd_capsule((0, 0, 0), (0, 0, 0.17 * neck), 0.07 * bulk, 0.055 * bulk * max(head, 0.8)), m, nk, blend=0.05)
    hs = head
    shake = 0.35 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    hd = nk * T(0, 0, 0.17 * neck) * RX(0.75 + pitch + up * 0.55 + 0.12) * RY(shake)
    sc.add(sd_ellipsoid(0.058 * hs * width, 0.068 * hs, 0.08 * hs), m, hd * T(0, 0.02 * hs, 0.0), blend=0.03)
    sc.add(sd_ellipsoid(0.044 * hs * width, 0.048 * hs, 0.12 * hs * snout), m, hd * T(0, 0.0, 0.1 * hs * snout), blend=0.035)
    jaw = hd * T(0, -0.035 * hs, 0.0) * RX(0.05 + 0.65 * roar)
    sc.add(sd_ellipsoid(0.04 * hs * width, 0.022 * hs, 0.115 * hs * snout), bm, jaw * T(0, -0.012 * hs, 0.1 * hs * snout), blend=0.02)
    if roar > 0.15:
        sc.add(sd_ellipsoid(0.03 * hs * width, 0.03 * hs * roar, 0.1 * hs * snout), MOUTH, hd * T(0, -0.04 * hs, 0.1 * hs * snout) * RX(0.3 * roar))
    if teeth:
        n = int(6 * snout)
        for i in range(n):
            z = (0.035 + i * 0.03) * hs
            for sgn in (-1, 1):
                sc.add(sd_cone(0.02 * hs, 0.006 * hs, 0.001, 0.001), IVORY, hd * T(0.036 * hs * width * sgn, -0.044 * hs, z) * RX(math.pi))
                if roar > 0.15:
                    sc.add(sd_cone(0.016 * hs, 0.005 * hs, 0.001, 0.001), IVORY, jaw * T(0.033 * hs * width * sgn, 0.004 * hs, z + 0.015 * hs))
    for sgn in (-1, 1):
        reptile_eye(sc, hd, 0.045 * hs * width * sgn, 0.042 * hs, 0.03 * hs, 0.0105 * hs, sgn, eye or iris("#a8742a"), lid=m)
        sc.add(sd_ellipsoid(0.004 * hs, 0.0035 * hs, 0.008 * hs), Mat("#1a1414", rough=0.8), hd * T(0.02 * hs * width * sgn, 0.03 * hs, 0.205 * hs * snout))
    return root, hd, nk


# ─── the quadrupeds ─────────────────────────────────────────────────────────
def quad(sc, anim, t, m, belly=None, Lf=0.24, Lb=0.28, bw=0.13, bh=0.13, bl=0.24, leg_r=0.045,
         neck=((0, 0.05, 0.0), (0, 0.08, 0.1), (0, 0.06, 0.18)), neck_r=(0.07, 0.05), head_pitch=0.2,
         tail=((0, 0, 0), (0, 0.01, -0.25), (0, -0.02, -0.45), (0, -0.04, -0.62)), tail_r=0.09, head_up=0.0):
    walk = anim == "move"
    gait = TAU * t * 2
    bob = 0.012 * abs(math.sin(gait)) if walk else 0.004 * math.sin(TAU * t)
    zf, zb = bl * 0.62, -bl * 0.55
    yf, yb = BASE_TOP + Lf + bh * 0.45, BASE_TOP + Lb + bh * 0.45
    pitch = math.atan2(yf - yb, zf - zb)
    yc = (yf * zb - yb * zf) / (zb - zf)                         # the body's height at z = 0 on the line hip–shoulder
    root = RY(YAW) * T(0, yc + bob, 0) * RX(-pitch)
    bm = belly or m
    sc.add(sd_ellipsoid(bw, bh, bl), m, root)
    sc.add(sd_ellipsoid(bw * 0.8, bh * 0.6, bl * 0.8), bm, root * T(0, -bh * 0.35, 0), blend=0.05)
    # the legs: pillars with a knee, round feet
    for i, (sx, front) in enumerate(((-1, 1), (1, 1), (-1, 0), (1, 0))):
        ph = gait + (math.pi if i in (1, 2) else 0)
        swing = 0.05 * math.sin(ph) if walk else 0.0
        lift = 0.02 * max(0.0, math.sin(ph)) if walk else 0.0
        z = zf if front else zb
        top = root.point(bw * 0.62 * sx, -bh * 0.3, z * 0.95).tolist()
        yaw_inv = RY(YAW)
        # in the yaw frame: from the top (on the body) down to the ground
        lx = RY(-YAW).point(*top).tolist()
        hip = RY(YAW) * T(lx[0], lx[1], lx[2])
        h = lx[1] - BASE_TOP
        knee = (0.0, -h * 0.5, (0.025 if front else -0.03))
        foot = (0.0, -h + 0.02 + lift, swing)
        r0 = leg_r * (1.0 if front else 1.15)
        sc.add(sd_ellipsoid(r0 * 1.5, h * 0.35, r0 * 1.8), m, hip * T(0, -h * 0.12, 0), blend=0.04)
        chain(sc, [(0, 0, 0), knee, foot], [r0, r0 * 0.8, r0 * 0.75], m, hip, blend=0.02)
        sc.add(sd_cylinder(0.012, r0 * 1.05, 0.008), m, hip * T(0, -h + 0.012 + lift, swing), blend=0.012)
        for k in (-1, 0, 1):
            sc.add(sd_sphere(r0 * 0.3), Mat("#3a342c", rough=0.6), hip * T(r0 * 0.6 * k, -h + 0.01 + lift, swing + r0 * 0.85))
    # the tail
    sw = 0.06 * math.sin(TAU * t) + (0.07 * math.sin(gait) if walk else 0.0)
    p0, p1, p2, p3 = tail
    tp = bezier((p0[0], p0[1], zb * 0.9 + p0[2]), (sw * 0.3, p1[1], zb + p1[2]), (sw * 0.7, p2[1], zb + p2[2]), (sw, p3[1], zb + p3[2]), 9)
    chain(sc, tp, [tail_r * (1 - i / 9) ** 1.2 + 0.008 for i in range(9)], m, root, blend=0.03)
    # the neck and the head (the head frame keeps only the yaw: its pitch is head_pitch)
    q0, q1, q2 = neck
    sway = 0.02 * math.sin(TAU * t) + head_up * 0.0
    npts = bezier((0, bh * 0.25, zf * 0.9), (0, q0[1] + bh * 0.25, zf + q0[2] + 0.04), (sway, q1[1], zf + q1[2]), (sway, q2[1], zf + q2[2]), 6)
    chain(sc, npts, [neck_r[0] + (neck_r[1] - neck_r[0]) * i / 5 for i in range(6)], m, root, blend=0.03)
    end = RY(-YAW).point(*root.point(*npts[-1]).tolist()).tolist()
    shake = 0.35 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    hd = RY(YAW) * T(*end) * RX(head_pitch) * RY(shake)
    return root, hd, tp


# ─── 1 · Brachiosaure ───────────────────────────────────────────────────────
def brachiosaurus(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#566a5c", "#b8b698", 60, spots="#3e4e44")
    win = anim == "win"
    reach = 0.06 * math.sin(TAU * t * 2) if win else 0.02 * math.sin(TAU * t)
    root, hd, _ = quad(sc, anim, t, m, Lf=0.36, Lb=0.27, bw=0.12, bh=0.13, bl=0.22, leg_r=0.042,
                       neck=((0, 0.18, 0.04), (0, 0.42 + reach, 0.04), (0, 0.6 + reach, 0.1)), neck_r=(0.075, 0.036),
                       head_pitch=0.35 - (0.3 if win else 0.0), tail=((0, 0, 0), (0, 0.0, -0.2), (0, -0.04, -0.36), (0, -0.08, -0.5)),
                       tail_r=0.07)
    sc.add(sd_ellipsoid(0.04, 0.038, 0.055), m, hd * T(0, 0.0, 0.02), blend=0.02)
    sc.add(sd_ellipsoid(0.03, 0.026, 0.05), m, hd * T(0, -0.012, 0.06), blend=0.02)
    sc.add(sd_ellipsoid(0.026, 0.024, 0.03), m, hd * T(0, 0.03, 0.03), blend=0.015)        # the bump of the nostrils
    chew = 0.008 * abs(math.sin(TAU * t * 4)) if anim == "idle" else 0.0
    sc.add(sd_ellipsoid(0.026, 0.01, 0.045), m, hd * T(0, -0.032 - chew, 0.06), blend=0.01)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.011), EYE, hd * T(0.034 * sgn, 0.012, 0.03))
    if anim == "idle":   # a twig of leaves in the mouth
        sc.add(sd_capsule((0, 0, 0), (0.08, -0.02, 0.0), 0.004), Mat("#5a4028", rough=0.9), hd * T(-0.04, -0.03, 0.1))
        for k in range(3):
            sc.add(sd_thin(0.02, 0.004, 0.012), flat("#5a8a3a"), hd * T(-0.03 + 0.025 * k, -0.03 - 0.006 * k, 0.11) * RZ(0.5 * k))
    return sc


# ─── 2 · Parasaurolophus ────────────────────────────────────────────────────
def parasaurolophus(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#9a6a3c", "#e2d2a8", 70, stripes="#5a3a22", freq=30)
    win = anim == "win"
    root, hd, _ = quad(sc, anim, t, m, Lf=0.21, Lb=0.33, bw=0.11, bh=0.12, bl=0.22, leg_r=0.035,
                       neck=((0, 0.06, 0.04), (0, 0.14, 0.08), (0, 0.18 + (0.06 if win else 0), 0.12)), neck_r=(0.06, 0.045),
                       head_pitch=0.25 - (0.7 if win else 0.0), tail_r=0.08)
    sc.add(sd_ellipsoid(0.048, 0.05, 0.065), m, hd * T(0, 0.01, 0.0))
    sc.add(sd_ellipsoid(0.04, 0.03, 0.08), m, hd * T(0, -0.01, 0.07), blend=0.03)
    sc.add(sd_thin(0.045, 0.012, 0.045), flat("#3a3028"), hd * T(0, -0.022, 0.12), blend=0.01)     # the duck bill
    crest = Mat("#c8642e", rough=0.55)
    pts = bezier((0, 0.04, 0.02), (0, 0.1, -0.05), (0, 0.13, -0.15), (0, 0.12, -0.27), 6)
    chain(sc, pts, [0.024, 0.026, 0.026, 0.024, 0.022, 0.018], crest, hd, blend=0.01)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.012), EYE, hd * T(0.04 * sgn, 0.025, 0.03))
    return sc


# ─── 3 · Stégosaure ─────────────────────────────────────────────────────────
def stegosaurus(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#6a7a48", "#d0c89a", 70, spots="#4a5a32")
    root, hd, tp = quad(sc, anim, t, m, Lf=0.17, Lb=0.3, bw=0.13, bh=0.16, bl=0.26, leg_r=0.045,
                        neck=((0, -0.02, 0.04), (0, -0.05, 0.1), (0, -0.07, 0.15)), neck_r=(0.07, 0.04),
                        head_pitch=0.35 if anim != "win" else -0.2,
                        tail=((0, 0, 0), (0, 0.06, -0.2), (0, 0.06, -0.38), (0, 0.04 + (0.1 if anim == "win" else 0), -0.55)), tail_r=0.09)
    sc.add(sd_ellipsoid(0.034, 0.03, 0.06), m, hd)
    sc.add(sd_ellipsoid(0.024, 0.02, 0.05), m, hd * T(0, -0.01, 0.05), blend=0.02)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.009), EYE, hd * T(0.026 * sgn, 0.012, 0.02))
    # the plates: two alternating rows, the biggest above the hips
    def plate_mat():
        def pat(pl, pw, n):
            e = torch.clamp(pl[:, 1] / 0.1, 0, 1)[:, None]
            return {"albedo": srgb("#7a8a52") * (1 - e) + srgb("#c0582e") * e}
        return Mat("#a05a2e", rough=0.55, pattern=pat)
    pm = plate_mat()
    for i in range(9):
        u = i / 8
        z = 0.26 * (1 - u) - 0.28 * u
        h = 0.045 + 0.075 * math.sin(math.pi * (0.15 + 0.75 * u))
        sx = 0.025 if i % 2 else -0.025
        y = 0.14 + 0.03 * math.sin(math.pi * u) - 0.02 * u
        sc.add(sd_thin(0.008, h, h * 0.8), pm, root * T(sx, y + h * 0.55, z) * RZ(-sx * 4))
    # the thagomizer: four spikes at the end of the tail
    tip = tp[-2]
    for sx in (-1, 1):
        for dz in (0.0, 0.06):
            sc.add(sd_cone(0.07, 0.016, 0.002, 0.002), IVORY, root * T(tip[0] + 0.02 * sx, tip[1] + 0.02, tip[2] + dz) * RZ(-sx * 1.0) * RX(-0.4) * T(0, 0.06, 0))
    return sc


# ─── 4 · Ankylosaure ────────────────────────────────────────────────────────
def ankylosaurus(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#5e5040", "#b8a880", 60)
    root, hd, tp = quad(sc, anim, t, m, Lf=0.14, Lb=0.16, bw=0.2, bh=0.12, bl=0.28, leg_r=0.05,
                        neck=((0, -0.02, 0.03), (0, -0.03, 0.06), (0, -0.03, 0.09)), neck_r=(0.08, 0.06),
                        head_pitch=0.15 if anim != "win" else -0.15,
                        tail=((0, 0, 0), (0.0, 0.0, -0.2), (0.0, 0.0, -0.4), (0.0, 0.0, -0.58)), tail_r=0.06)
    armour = Mat("#8a7658", rough=0.6)
    for i in range(6):   # rows of osteoderms on the back
        z = -0.22 + i * 0.085
        for j in range(-2, 3):
            x = j * 0.065
            y = 0.12 * math.sqrt(max(0.0, 1 - (x / 0.21) ** 2 - (z / 0.3) ** 2)) + 0.005
            sc.add(sd_cone(0.018, 0.026, 0.008, 0.004), armour, root * T(x, y, z) * RZ(-x * 3.5))
    for sgn in (-1, 1):   # spikes along the flanks
        for i in range(5):
            z = -0.2 + i * 0.1
            sc.add(sd_cone(0.04, 0.02, 0.002, 0.002), IVORY, root * T(0.19 * sgn, 0.0, z) * RZ(-sgn * 1.6) * T(0, 0.03, 0))
    # the head: wide, with horns at its back corners
    sc.add(sd_ellipsoid(0.075, 0.045, 0.07), m, hd * T(0, 0.01, 0.02), blend=0.02)
    sc.add(sd_ellipsoid(0.05, 0.035, 0.05), m, hd * T(0, -0.005, 0.07), blend=0.02)
    for sgn in (-1, 1):
        sc.add(sd_cone(0.03, 0.016, 0.002, 0.002), IVORY, hd * T(0.065 * sgn, 0.03, -0.03) * RZ(-sgn * 1.2) * RX(0.6))
        sc.add(sd_sphere(0.011), EYE, hd * T(0.05 * sgn, 0.025, 0.04))
    # the club: two big lobes of bone at the end of the tail
    tip = tp[-1]
    swing = 0.12 * math.sin(TAU * t * 2) if anim in ("win", "move") else 0.03 * math.sin(TAU * t)
    club = root * T(tip[0] + swing, tip[1] + 0.01, tip[2] - 0.03)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.055, 0.04, 0.05), armour, club * T(0.035 * sgn, 0, 0), blend=0.02)
    return sc


# ─── 5 · Tricératops ────────────────────────────────────────────────────────
def triceratops(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#5a4836", "#b8a07a", 60, spots="#3e3024")
    charge = anim == "win"
    root, hd, _ = quad(sc, anim, t, m, Lf=0.22, Lb=0.25, bw=0.15, bh=0.15, bl=0.25, leg_r=0.05,
                       neck=((0, 0.0, 0.04), (0, 0.0, 0.07), (0, -0.01 - (0.04 if charge else 0), 0.1)), neck_r=(0.1, 0.08),
                       head_pitch=0.15 + (0.35 if charge else 0.0) + (0.15 * math.sin(TAU * t * 2) if anim == "move" else 0.0),
                       tail_r=0.08, tail=((0, 0, 0), (0, 0.0, -0.18), (0, -0.04, -0.3), (0, -0.08, -0.42)))
    sc.add(sd_ellipsoid(0.07, 0.07, 0.09), m, hd * T(0, 0.0, 0.03))
    sc.add(sd_ellipsoid(0.05, 0.055, 0.07), m, hd * T(0, -0.02, 0.1), blend=0.03)
    sc.add(sd_cone(0.04, 0.03, 0.004, 0.003), flat("#2e2a26"), hd * T(0, -0.05, 0.16) * RX(2.3))          # the beak
    # the frill, rising behind the head, a dark edge with bumps
    frill = skin("#9a6a44", "#9a6a44", 50, spots="#5a3a24")
    fr = hd * T(0, 0.08, -0.04) * RX(-0.55)
    sc.add(sd_thin(0.17, 0.15, 0.02), frill, fr * T(0, 0.04, 0), blend=0.02)
    for k in range(11):
        a = math.pi * (0.05 + 0.9 * k / 10)
        sc.add(sd_cone(0.022, 0.016, 0.002, 0.002), IVORY, fr * T(0.17 * math.cos(a), 0.04 + 0.15 * math.sin(a), 0) * RZ(a - math.pi / 2))
    # three horns: two long ones over the eyes, a short one on the nose
    for sgn in (-1, 1):
        sc.add(sd_cone(0.1, 0.022, 0.002, 0.003), IVORY, hd * T(0.04 * sgn, 0.06, 0.06) * RX(1.05) * RZ(-0.15 * sgn) * T(0, 0.09, 0))
        sc.add(sd_sphere(0.012), EYE, hd * T(0.056 * sgn, 0.025, 0.06))
    sc.add(sd_cone(0.035, 0.016, 0.002, 0.002), IVORY, hd * T(0, 0.02, 0.15) * RX(0.6) * T(0, 0.03, 0))
    return sc


# ─── 6 · Dilophosaure ───────────────────────────────────────────────────────
def dilophosaurus(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#8a8a4a", "#e0d8a8", 80, stripes="#4a3a22", freq=34)
    roar = 0.85 if anim == "win" else 0.1
    root, hd, nk = theropod(sc, anim, t, m, leg=1.05, bulk=0.8, tail=1.0, neck=1.3, head=0.95, snout=1.15, width=0.85,
                            roar=roar, eye=iris("#c0922a"))
    crest = Mat("#a8603a", rough=0.5, sss=0.5)
    for sgn in (-1, 1):   # two thin parallel crests along the top of the skull and the snout, above the eyes
        sc.add(sd_thin(0.003, 0.03, 0.13), crest, hd * T(0.016 * sgn, 0.072, 0.08) * RX(0.16) * RZ(0.1 * sgn), blend=0.004)
    return sc


# ─── 7 · Quetzalcoatlus ─────────────────────────────────────────────────────
def quetzalcoatlus(anim, t, side):
    """Standing on its hind legs, wings raised to look bigger, the long neck up,
    the huge beak; the membranes translucent, the body furred (pycnofibres)."""
    sc = Scene()
    base(sc, side)
    walk, win = anim == "move", anim == "win"
    B = BASE_TOP
    m = fur("#4e423a", 170, 0.5)
    pale = skin("#d8d0c0", "#efe8da", 60, rough=0.6)
    bone = Mat("#d8c8a0", rough=0.4, clearcoat=0.3)

    def memb_pat(pl, pw, n):
        v = (torch.sin(pl[:, 0] * 70 + 4 * fbm(pl * 6, 2)) > 0.93).float()[:, None]
        return {"albedo": srgb("#5e3a2c") * (0.85 + 0.3 * fbm(pl * 30, 2))[:, None] * (1 - 0.3 * v)}
    memb = Mat("#7a4632", rough=0.6, sss=0.7, pattern=memb_pat)
    crest = Mat("#b8402a", rough=0.5, sss=0.5)
    hop = 0.05 * math.sin(math.pi * t) if walk else 0.0
    flap = 0.45 + 0.45 * math.sin(TAU * 2 * t) if win else 0.1 * math.sin(TAU * t)
    Y = RY(YAW) * T(0, hop, 0)
    # the body, its front raised
    sc.add(sd_ellipsoid(0.075, 0.085, 0.13), m, Y * T(0, B + 0.3, 0.0) * RX(-0.7))
    # the hind legs, short, three toes
    for sgn in (-1, 1):
        foot = (0.065 * sgn, B + 0.014 - hop, 0.0)
        chain(sc, [(0.055 * sgn, B + 0.25, -0.05), (0.07 * sgn, B + 0.15, 0.02), (0.066 * sgn, B + 0.06, -0.04), foot],
              [0.034, 0.024, 0.015, 0.012], m, Y, blend=0.02)
        for k in (-1, 0, 1):
            toe = Y * T(*foot) * RY(0.35 * k)
            sc.add(sd_capsule((0, 0, 0), (0, -0.004, 0.05), 0.011, 0.007), m, toe, blend=0.008)
            sc.add(sd_cone(0.016, 0.006, 0.001, 0.001), CLAW, toe * T(0, -0.006, 0.055) * RX(math.pi / 2 + 0.3))
    # the wings: arm, wrist with three small clawed fingers, the long fourth finger; membranes to the leg
    for sgn in (-1, 1):
        sh = (0.065 * sgn, B + 0.37, 0.07)
        el = (0.15 * sgn, B + 0.46 + 0.04 * flap, 0.03)
        wr = (0.24 * sgn, B + 0.56 + 0.08 * flap, 0.07)
        tip = ((0.36 + 0.06 * flap) * sgn, B + 0.8 + 0.12 * flap, -0.16)
        knee = (0.07 * sgn, B + 0.16, 0.0)
        chain(sc, [sh, el, wr], [0.03, 0.022, 0.017], m, Y, blend=0.015)
        sc.add(sd_capsule(wr, tip, 0.012, 0.004), pale, Y, blend=0.006)
        for f in (-1, 0, 1):
            fx = (wr[0] + 0.012 * f * sgn, wr[1] + 0.01, wr[2] + 0.035)
            sc.add(sd_capsule(wr, fx, 0.006, 0.004), m, Y, blend=0.004)
            sc.add(sd_cone(0.012, 0.005, 0.001, 0.001), CLAW, Y * T(*fx) * RX(math.pi / 2 + 0.6))
        # the membrane: a fan from the shoulder to a trailing edge bowed inward (tip → knee), slightly cupped
        sc.add(sheet([sh, wr, tip], 0.0025), memb, Y)
        prev = tip
        for i in range(1, 6):
            u = i / 5
            q = [tip[k] + (knee[k] - tip[k]) * u for k in range(3)]
            bow = 0.09 * math.sin(math.pi * u)
            q = (q[0] - sgn * bow * 0.9, q[1] + bow * 0.5, q[2] - 0.05 * math.sin(math.pi * u))
            sc.add(sheet([sh, prev, q], 0.0025), memb, Y)
            prev = q
        sc.add(sheet([sh, (0.05 * sgn, B + 0.44, 0.12), wr], 0.0025), memb, Y)
    # the neck, long and stiff, and the head
    sway = 0.015 * math.sin(TAU * t)
    up = 0.06 if win else 0.0
    npts = bezier((0, B + 0.38, 0.06), (0, B + 0.5, 0.1), (sway, B + 0.62 + up, 0.12), (sway, B + 0.7 + up, 0.19), 7)
    chain(sc, npts, [0.042 - 0.003 * i for i in range(7)], pale, Y, blend=0.02)
    shake = 0.35 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    hd = Y * T(*npts[-1]) * RY(shake) * RX(0.45 - (0.5 if win else 0.0))
    sc.add(sd_ellipsoid(0.03, 0.04, 0.06), pale, hd * T(0, 0.0, 0.02), blend=0.02)
    gape = 0.25 if win else 0.0
    sc.add(sd_cone(0.14, 0.03, 0.003), bone, hd * T(0, 0.004, 0.2) * RX(math.pi / 2), blend=0.01)
    sc.add(sd_cone(0.13, 0.022, 0.003), bone, hd * T(0, -0.016, 0.06) * RX(gape) * T(0, 0, 0.13) * RX(math.pi / 2 + 0.03), blend=0.006)
    sc.add(sheet([(0, 0.02, 0.06), (0, 0.035, -0.02), (0, 0.13, -0.09)], 0.004), crest, hd, blend=0.01)    # a blade crest
    for sgn in (-1, 1):
        reptile_eye(sc, hd, 0.026 * sgn, 0.014, 0.035, 0.009, sgn, iris("#b0702a"), lid=pale)
    return sc


# ─── 8 · Allosaure ──────────────────────────────────────────────────────────
def allosaurus(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#a0683a", "#e0c8a0", 60, stripes="#4a2a1a", freq=24)
    roar = 0.95 if anim == "win" else 0.2
    root, hd, nk = theropod(sc, anim, t, m, leg=1.0, bulk=1.0, tail=1.0, neck=1.05, head=1.1, snout=1.1, width=0.9,
                            roar=roar, eye=iris("#b87a28"))
    horn = Mat("#c8603a", rough=0.5)
    for sgn in (-1, 1):   # the small horns in front of the eyes, a ridge along the snout
        sc.add(sd_cone(0.03, 0.02, 0.003, 0.003), horn, hd * T(0.042 * sgn, 0.085, 0.06) * RX(-0.2))
        sc.add(sd_capsule((0, 0, 0), (0, -0.01, 0.1), 0.008), horn, hd * T(0.02 * sgn, 0.06, 0.1), blend=0.006)
    return sc


# ─── 9 · Spinosaure ─────────────────────────────────────────────────────────
def spinosaurus(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#3e5458", "#c8c0a0", 60, spots="#2a3a3e")
    roar = 0.9 if anim == "win" else 0.15
    root, hd, nk = theropod(sc, anim, t, m, leg=0.95, bulk=1.05, tail=1.1, neck=1.25, head=1.1, snout=1.6, width=0.65,
                            roar=roar, pitch=0.25, eye=iris("#c0922a"))
    # the sail: a high membrane along the back, red at the top, crossed by its spines
    def sail_mat():
        def pat(pl, pw, n):
            h = torch.clamp((pl[:, 1] + 0.05) / 0.3, 0, 1)[:, None]
            a = srgb("#3e5458") * (1 - h) + srgb("#c8402a") * h
            sp = (torch.sin(pl[:, 2] * 90) > 0.8).float()[:, None]
            return {"albedo": a * (1 - 0.35 * sp)}
        return Mat("#c8402a", rough=0.5, sss=0.5, pattern=pat)
    sm = sail_mat()
    sc.add(sd_thin(0.012, 0.26, 0.36), sm, root * T(0, 0.14, -0.02), blend=0.02)
    sc.add(sd_capsule((0, 0, 0), (0, 0, 0.12), 0.012, 0.006), m, hd * T(0, 0.05, 0.1), blend=0.01)    # a ridge on the snout
    return sc


# ─── 10 · Tyrannosaure ──────────────────────────────────────────────────────
def tyrannosaurus(anim, t, side):
    sc = Scene()
    base(sc, side)
    m = skin("#3a3430", "#a08a68", 55, stripes="#5a2e1e", freq=20)
    roar = 1.0 if anim in ("win", "blocked") else 0.3
    root, hd, nk = theropod(sc, anim, t, m, leg=1.0, bulk=1.25, tail=1.0, neck=0.8, head=1.45, snout=1.0, width=1.15,
                            arms="tiny", roar=roar, eye=iris("#c8741c"))
    return sc


SCALE = {"random": 1.0, "beginner": 1.0, "monte_carlo": 1.0, "mcts": 1.05, "basic_minmax": 1.08, "nnue193": 1.05,
         "features": 1.08, "resnet": 1.12, "dev": 1.12, "alphazero": 1.15}
CAST = {"random": ("Brachiosaure", brachiosaurus), "beginner": ("Parasaurolophus", parasaurolophus),
        "monte_carlo": ("Stégosaure", stegosaurus), "mcts": ("Ankylosaure", ankylosaurus),
        "basic_minmax": ("Tricératops", triceratops), "nnue193": ("Dilophosaure", dilophosaurus),
        "features": ("Quetzalcoatlus", quetzalcoatlus), "resnet": ("Allosaure", allosaurus), "dev": ("Spinosaure", spinosaurus),
        "alphazero": ("Tyrannosaure", tyrannosaurus)}
DINOS = {"dino_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"dino_" + k: n for k, (n, _) in CAST.items()}
