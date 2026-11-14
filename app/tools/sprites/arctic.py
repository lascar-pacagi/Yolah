"""
arctic.py — the Banquise set: one animal of the ice per AI, from the seal to
the woolly mammoth, after photographs (proportions, the joints of the legs,
the heads) with fur as a displacement of the surface:

  random        Phoque gris         hauled out on the ice, arched, mottled
  beginner      Manchot empereur    upright, the golden ear patches
  monte_carlo   Renard polaire      the winter coat, a huge tail
  mcts          Harfang des neiges  on a block of ice, yellow eyes, the head turning
  basic_minmax  Morse               a bull, wrinkled skin, tusks, bristles
  minmax        Renne               the great antlers, the pale mane, broad hooves
  mcts_nnue     Narval              surfacing, dappled grey, the spiral tusk
  nnue193       Loup arctique       tall and lean, a ruff
  features      Bœuf musqué         the skirt of long hair, the horns' boss
  resnet        Ours blanc          long neck, small head carried low, huge paws
  dev           Orque               breaching through a hole in the ice
  alphazero     Mammouth laineux    the domed head, the hump, spiralling tusks

The quadruped of the first version (quadruped) stays for the Manga set.
The human is never part of a set.
"""
import math
import torch
from render import X as X_, Scene, Mat, T, RX, RY, RZ, srgb, vec, sd_plane_cut, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm
from models import base, BASE_TOP, TAU, wave, glow
from legends import base_pose, held, skin
from heroic import adult, chain, bezier, HEAD
from origami import sheet, sd_hull


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


# ════════════════════════════════════════════════════════════════════════════
# The Banquise set, second version: anatomy after photographs (proportions of
# the body, the joints of the legs, the head), fur as a displacement of the
# surface (short and dense, or long and hanging), no frost crystals turning.
# ════════════════════════════════════════════════════════════════════════════
YAW = HEAD + 0.72                  # three-quarter profile: the head to the right


def furry(f, amp, k=70.0, hang=0.0):
    """A surface with fur: the distance pushed out by strands of noise
    (stretched along y when the hair hangs, longer at the bottom)."""
    sk = vec(k, k * (1.0 - 0.75 * hang), k)

    def g(q):
        n = fbm(q * sk, 3)
        a = amp * (1.0 + hang * (-q[:, 1] / 0.1).clamp(0, 2.5)) if hang > 0 else amp
        return f(q) - a * n
    return g


def pelt(c, light=None, k=110.0, spread=0.35, strands=0.0, dark=None):
    """Fur's colour: tufts of lighter and darker hair; `light`: the colour it
    turns to on the belly and the legs (by the normal); strands: hanging locks."""
    a0, a1 = srgb(c), srgb(light or c)
    d0 = srgb(dark) if dark else None

    def pat(pl, pw, n):
        v = fbm(pl * k, 3)
        tone = 1 - spread / 2 + spread * v
        if strands > 0:
            tone = tone * (1 - strands * 0.5 * (0.5 + 0.5 * torch.sin(pl[:, 0] * 260 + pl[:, 2] * 230 + 9 * fbm(pl * vec(20, 4, 20), 2))))
        up = ((n[:, 1] + 0.3) / 0.9).clamp(0, 1)[:, None]
        a = a1 * (1 - up) + a0 * up
        if d0 is not None:
            a = a * (1 - 0.6 * (v > 0.62).float()[:, None]) + d0 * 0.6 * (v > 0.62).float()[:, None]
        return {"albedo": a * tone[:, None]}
    return Mat(c, rough=0.92, sss=0.25, pattern=pat)


def frame_of(anim, t, lift=0.0):
    shake = 0.22 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    return RY(YAW + shake) * T(0, BASE_TOP + lift, 0)


def segs(sc, m, X, parts, amp, k, blend, hang=0.0):
    """The trunk as blended ellipsoids along the spine: (z, y, rx, ry, rz)."""
    for z, y, rx, ry, rz in parts:
        sc.add(furry(sd_ellipsoid(rx, ry, rz), amp, k, hang), m, X * T(0, y, z), blend=blend)


def limb(sc, m, X, pts, radii, amp, k, blend, hang=0.0):
    for (a, b), (ra, rb) in zip(zip(pts[:-1], pts[1:]), zip(radii[:-1], radii[1:])):
        sc.add(furry(sd_capsule(a, b, ra, rb), amp, k, hang), m, X, blend=blend)


def swing_pts(pts, ang):
    """Points of a leg turned about the x axis through its top (the gait)."""
    x0, y0, z0 = pts[0]
    c, s_ = math.cos(ang), math.sin(ang)
    return [(x, y0 + (y - y0) * c - (z - z0) * s_, z0 + (y - y0) * s_ + (z - z0) * c) for x, y, z in pts]


def paw(sc, m, X, p, size, claws=4, claw_m=None, toe=0.0):
    sc.add(sd_ellipsoid(size * 0.85, size * 0.42, size * 1.15), m, X * T(*p), blend=size * 0.4)
    if claw_m is not None:
        for k in range(claws):
            u = (k - (claws - 1) / 2) / max(claws - 1, 1)
            sc.add(sd_cone(size * 0.22, size * 0.09, 0.001, 0.001), claw_m,
                   X * T(p[0] + u * size * 1.1, p[1] - size * 0.25, p[2] + size * 1.15) * RX(math.pi / 2 + 0.5))


def gait(anim, t):
    """Leg swings (front left, front right, hind left, hind right) and the body's bob."""
    if anim != "move":
        return (0.0, 0.0, 0.0, 0.0), 0.0
    g = TAU * t
    a = 0.38
    return (a * math.sin(g), a * math.sin(g + math.pi), a * math.sin(g + math.pi), a * math.sin(g)), 0.012 * abs(math.sin(2 * g))


def eye(sc, X, p, r, sgn, iris="#2a1a10"):
    """A mammal's eye: dark, wet, a lid above."""
    sc.add(sd_sphere(r), Mat(iris, rough=0.05, clearcoat=1.0), X * T(*p))
    sc.add(sd_sphere(r * 0.35), Mat("#ffffff", emissive="#ffffff", emit=0.6), X * T(p[0] + sgn * r * 0.55, p[1] + r * 0.45, p[2] + r * 0.5))


# ─── 1 · Phoque gris ────────────────────────────────────────────────────────
def seal(anim, t, side):
    """A grey seal hauled out on the ice: a long spindle body arched like a
    banana (head and hind flippers up), mottled, a long dog-like muzzle, big
    dark eyes, whiskers, front flippers with claws."""
    sc = Scene()
    base(sc, side)
    X = frame_of(anim, t)
    skin_m = Mat("#7e848c", rough=0.5, clearcoat=0.35, pattern=lambda pl, pw, n: {
        "albedo": srgb("#5e646c") * (0.85 + 0.2 * fbm(pl * 60, 2))[:, None] * (1 - 0.45 * (fbm(pl * 26 + 3.1, 3) > 0.6).float())[:, None]
                  * (0.75 + 0.35 * ((n[:, 1] + 0.4) / 1.2).clamp(0, 1))[:, None] + srgb("#d8d6cc") * 0.12 * (n[:, 1] < -0.4).float()[:, None]})
    breathe = 0.008 * math.sin(TAU * t)
    up = (0.35 + 0.1 * math.sin(TAU * 2 * t)) if anim == "win" else 0.18 + 0.04 * math.sin(TAU * t)
    tail_up = 0.12 + (0.12 * math.sin(TAU * t) if anim == "move" else 0.04 * math.sin(TAU * t))
    body = [(-0.3, 0.07 + tail_up * 0.4, 0.06, 0.05, 0.08), (-0.18, 0.085, 0.1, 0.08 + breathe, 0.12), (-0.02, 0.1, 0.13, 0.1 + breathe, 0.13),
            (0.13, 0.11, 0.115, 0.095, 0.1)]
    segs(sc, skin_m, X, body, 0.0, 1.0, 0.06)
    nk = X * T(0, 0.13, 0.2) * RX(-up)
    sc.add(sd_capsule((0, 0, 0), (0, 0.06, 0.08), 0.085, 0.065), skin_m, nk, blend=0.05)
    hd = nk * T(0, 0.075, 0.1) * RX(up * 0.7 - 0.05)
    sc.add(sd_ellipsoid(0.058, 0.055, 0.062), skin_m, hd, blend=0.03)
    sc.add(sd_capsule((0, -0.01, 0.03), (0, -0.018, 0.105), 0.038, 0.03), skin_m, hd, blend=0.025)             # the long muzzle
    nose = Mat("#2a2a2e", rough=0.35, clearcoat=0.6)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.008, 0.012, 0.006), nose, hd * T(0.012 * sgn, -0.008, 0.132) * RZ(0.4 * sgn))
        eye(sc, hd, (0.034 * sgn, 0.018, 0.04), 0.014, sgn, "#0e0a08")
        for k in range(5):   # whiskers
            sc.add(sd_capsule((0, 0, 0), (0.06 * sgn, -0.01 - 0.008 * k, 0.03 - 0.008 * k), 0.0012, 0.0005), Mat("#e8e4dc", rough=0.4),
                   hd * T(0.028 * sgn, -0.03, 0.105))
    # front flippers on the ice, the claws; hind flippers raised together
    for sgn in (-1, 1):
        fl = X * T(0.11 * sgn, 0.04, 0.12) * RZ(0.5 * sgn) * RX(0.3)
        sc.add(sd_ellipsoid(0.03, 0.06, 0.016), skin_m, fl * T(0, -0.03, 0.01), blend=0.02)
        for k in range(4):
            sc.add(sd_cone(0.008, 0.003, 0.0005, 0.0005), Mat("#2a2420", rough=0.4), fl * T(-0.012 + 0.008 * k, -0.085, 0.02) * RX(2.5))
        hf = X * T(0.025 * sgn, 0.09 + tail_up * 0.6, -0.4) * RY(0.35 * sgn) * RX(-0.9 - tail_up)
        sc.add(sd_ellipsoid(0.035, 0.012, 0.06), skin_m, hf, blend=0.015)
    return sc


# ─── 2 · Manchot empereur ───────────────────────────────────────────────────
def penguin(anim, t, side):
    """Upright, the head and back black, the belly white turning pale yellow
    under the throat, the ear patch golden orange, the long bill with an
    orange stripe, flippers held out a little, the feet pink-black."""
    sc = Scene()
    base(sc, side)
    waddle = 0.08 * math.sin(TAU * t * 2) if anim == "move" else 0.02 * math.sin(TAU * t)
    X = frame_of(anim, t, 0.0) * RY(-0.55) * RZ(waddle)

    def plum(pl, pw, n):
        # local frame of the body: black where the normal faces back/up the sides, white on the front
        front = ((n @ torch.tensor([math.sin(YAW - 0.55), 0.0, math.cos(YAW - 0.55)], device=n.device)) + 0.05) / 0.35
        w = front.clamp(0, 1)[:, None]
        yel = ((pw[:, 1] - (BASE_TOP + 0.5)) / 0.12).clamp(0, 1)[:, None]
        white = srgb("#f4f2ea") * (1 - 0.6 * yel) + srgb("#f2dc8a") * 0.6 * yel
        black = srgb("#16181e") * (0.85 + 0.3 * fbm(pl * 140, 2))[:, None]
        return {"albedo": black * (1 - w) + white * w * (0.92 + 0.1 * fbm(pl * 160, 2))[:, None]}
    pm = Mat("#16181e", rough=0.55, clearcoat=0.25, pattern=plum)
    sc.add(furry(sd_ellipsoid(0.16, 0.27, 0.14), 0.002, 160), pm, X * T(0, 0.29, 0))
    sc.add(furry(sd_ellipsoid(0.13, 0.12, 0.12), 0.002, 160), pm, X * T(0, 0.5, 0.01), blend=0.08)
    hd = X * T(0, 0.64, 0.02) * RX(-0.25 if anim == "win" else 0.08 * math.sin(TAU * t))
    black = Mat("#14161c", rough=0.5, clearcoat=0.25)
    sc.add(sd_ellipsoid(0.075, 0.075, 0.085), black, hd, blend=0.05)
    sc.add(sd_capsule((0, -0.005, 0.06), (0, -0.03, 0.17), 0.022, 0.006), black, hd, blend=0.01)               # the bill
    sc.add(sd_capsule((0, -0.02, 0.07), (0, -0.034, 0.15), 0.012, 0.004), Mat("#f08a3a", rough=0.4, clearcoat=0.5), hd * T(0, -0.002, 0), blend=0.004)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.022, 0.045, 0.03), Mat("#f2b030", rough=0.5), hd * T(0.062 * sgn, -0.03, -0.01) * RZ(0.3 * sgn), blend=0.02)   # ear patch
        sc.add(sd_sphere(0.009), Mat("#1a1210", rough=0.05, clearcoat=1.0), hd * T(0.05 * sgn, 0.012, 0.04))
        flap = (0.5 + 0.4 * math.sin(TAU * 2 * t)) if anim == "win" else 0.18 + 0.05 * math.sin(TAU * t)
        fl = X * T(0.145 * sgn, 0.42, -0.01) * RZ(flap * sgn)
        sc.add(sd_ellipsoid(0.018, 0.13, 0.05), black, fl * T(0, -0.11, 0), blend=0.015)
        ft = X * T(0.06 * sgn, 0.012, 0.06) * RY(0.25 * sgn)
        sc.add(sd_ellipsoid(0.035, 0.012, 0.05), Mat("#2a2226", rough=0.6), ft)
        for k in range(3):
            sc.add(sd_cone(0.01, 0.004, 0.0005, 0.0005), Mat("#141214", rough=0.4), ft * T(-0.02 + 0.02 * k, -0.004, 0.052) * RX(math.pi / 2))
    return sc


# ─── 3 · Renard polaire ─────────────────────────────────────────────────────
def canid(sc, anim, t, m, s=1.0, leg=1.0, muzzle=1.0, ear=1.0, ear_round=0.0, tail=1.0, fluff=1.0, howl=False, ruff=0.0):
    """A fox or a wolf: deep chest, tucked belly, long legs with the hock set
    back, the head with a stop, a tapering muzzle, pointed ears, a bushy tail."""
    (fl, fr, hl, hr), bob = gait(anim, t)
    X = frame_of(anim, t, bob)
    amp, k = 0.006 * s * fluff, 60.0 / s
    sh_y, hip_y = 0.33 * s * leg, 0.33 * s * leg
    trunk = [(-0.17 * s, hip_y + 0.0 * s, 0.065 * s, 0.07 * s, 0.09 * s), (-0.04 * s, hip_y - 0.01 * s, 0.07 * s, 0.07 * s, 0.1 * s),
             (0.09 * s, sh_y - 0.02 * s, 0.08 * s, 0.095 * s, 0.1 * s), (0.17 * s, sh_y + 0.02 * s, 0.07 * s, 0.075 * s, 0.07 * s)]
    segs(sc, m, X, trunk, amp, k, 0.05 * s)
    if ruff > 0:
        sc.add(furry(sd_ellipsoid(0.085 * s, 0.09 * s, 0.07 * s), amp * 2.2, k * 0.7, 0.4), m, X * T(0, sh_y + 0.03 * s, 0.2 * s), blend=0.04 * s)
    # legs: front straight under the shoulder, hind with the thigh and the hock
    for sgn, sw in ((-1, fl), (1, fr)):
        pts = swing_pts([(0.05 * s * sgn, sh_y - 0.02 * s, 0.15 * s), (0.055 * s * sgn, sh_y * 0.55, 0.13 * s),
                         (0.05 * s * sgn, sh_y * 0.14, 0.15 * s), (0.05 * s * sgn, 0.02 * s, 0.165 * s)], sw)
        limb(sc, m, X, pts, [0.04 * s, 0.024 * s, 0.017 * s, 0.016 * s], amp * 0.6, k, 0.02 * s)
        paw(sc, m, X, (pts[-1][0], 0.014 * s, pts[-1][2] + 0.012 * s), 0.022 * s)
    for sgn, sw in ((-1, hl), (1, hr)):
        pts = swing_pts([(0.05 * s * sgn, hip_y, -0.17 * s), (0.055 * s * sgn, hip_y * 0.55, -0.11 * s),
                         (0.05 * s * sgn, hip_y * 0.24, -0.2 * s), (0.05 * s * sgn, 0.02 * s, -0.17 * s)], sw)
        sc.add(furry(sd_ellipsoid(0.05 * s, 0.085 * s, 0.07 * s), amp, k), m, X * T(pts[0][0], hip_y * 0.8, -0.15 * s), blend=0.03 * s)
        limb(sc, m, X, pts, [0.04 * s, 0.026 * s, 0.017 * s, 0.016 * s], amp * 0.6, k, 0.02 * s)
        paw(sc, m, X, (pts[-1][0], 0.014 * s, pts[-1][2] + 0.012 * s), 0.022 * s)
    # the tail: bushy, hanging in a curve
    wag = 0.08 * math.sin(TAU * t * (3 if anim == "win" else 1))
    tp = bezier((0, hip_y + 0.01 * s, -0.24 * s), (wag * s, hip_y - 0.02 * s, -0.36 * s * tail), (wag * 1.5 * s, hip_y - 0.12 * s * tail, -0.42 * s * tail),
                (wag * 2 * s, hip_y - 0.2 * s * tail, -0.4 * s * tail), 6)
    for i, p in enumerate(tp):
        r = (0.022 + 0.018 * math.sin(math.pi * (i + 1) / 7) * fluff) * s
        sc.add(furry(sd_sphere(r), amp * 1.3, k * 0.8), m, X * T(*p), blend=0.025 * s)
    # neck and head
    lift = 0.35 if howl else (0.1 if anim == "blocked" else 0.0)
    nk = X * T(0, sh_y + 0.02 * s, 0.19 * s) * RX(-0.7 - lift)
    sc.add(furry(sd_capsule((0, 0, 0), (0, 0, 0.12 * s), 0.058 * s, 0.045 * s), amp, k), m, nk, blend=0.04 * s)
    hd = nk * T(0, 0, 0.13 * s) * RX(0.75 + lift * 0.4 - (0.6 if howl else 0.0))
    sc.add(furry(sd_ellipsoid(0.05 * s, 0.045 * s, 0.055 * s), amp * 0.7, k), m, hd, blend=0.03 * s)
    for sgn in (-1, 1):   # the cheeks
        sc.add(furry(sd_sphere(0.03 * s), amp, k), m, hd * T(0.03 * s * sgn, -0.02 * s, 0.01 * s), blend=0.02 * s)
    sc.add(sd_capsule((0, -0.006 * s, 0.035 * s), (0, -0.016 * s, 0.1 * s * muzzle), 0.027 * s, 0.014 * s), m, hd, blend=0.02 * s)
    jaw_open = 0.35 if howl else 0.0
    sc.add(sd_capsule((0, -0.028 * s, 0.03 * s), (0, -0.032 * s - jaw_open * 0.05 * s, 0.09 * s * muzzle), 0.016 * s, 0.009 * s), m, hd, blend=0.01 * s)
    sc.add(sd_ellipsoid(0.012 * s, 0.009 * s, 0.009 * s), Mat("#141214", rough=0.3, clearcoat=0.6), hd * T(0, -0.008 * s, 0.103 * s * muzzle))
    for sgn in (-1, 1):
        eye(sc, hd, (0.026 * s * sgn, 0.014 * s, 0.04 * s), 0.008 * s, sgn, "#4a3010")
        ex = hd * T(0.03 * s * sgn, 0.04 * s, -0.005 * s) * RZ(-0.25 * sgn) * RX(-0.15)
        h = 0.06 * s * ear
        if ear_round > 0:
            sc.add(furry(sd_ellipsoid(0.022 * s * ear, h * 0.45, 0.01 * s), amp * 0.6, k), m, ex * T(0, h * 0.3, 0), blend=0.01 * s)
        else:
            sc.add(sd_cone(h * 0.5, 0.024 * s * ear, 0.002 * s, 0.002 * s), m, ex * T(0, h * 0.45, 0), blend=0.008 * s)
            sc.add(sd_cone(h * 0.4, 0.016 * s * ear, 0.001 * s, 0.001 * s), Mat("#c8b8a8", rough=0.9), ex * T(0, h * 0.42, 0.006 * s))
    return X, hd


def arctic_fox(anim, t, side):
    """The white fox in winter coat: small, thick fur, short legs and muzzle,
    small rounded ears, a huge bushy tail."""
    sc = Scene()
    base(sc, side)
    m = pelt("#f2f0ea", "#e2ded4", k=120, spread=0.25)
    canid(sc, anim, t, m, s=0.85, leg=0.82, muzzle=0.85, ear=0.75, ear_round=1.0, tail=1.1, fluff=1.8, howl=anim == "win", ruff=0.6)
    return sc


# ─── 4 · Harfang des neiges ─────────────────────────────────────────────────
def snowy_owl(anim, t, side):
    """The male snowy owl, nearly all white with a few dark flecks, a round
    head with a facial disc, yellow eyes, the black beak half hidden in the
    feathers, feathered feet gripping a rock of ice."""
    sc = Scene()
    base(sc, side)
    X = frame_of(anim, t) * RY(-0.5)
    rock = [(-0.15, 0.0, -0.12), (0.14, 0.0, -0.13), (0.16, 0.0, 0.1), (-0.12, 0.0, 0.13), (-0.11, 0.13, -0.07), (0.1, 0.15, -0.09),
            (0.12, 0.12, 0.06), (-0.08, 0.14, 0.09), (0.0, 0.16, 0.0)]
    sc.add(sd_hull(rock), ICE, X * T(0, 0.0, 0))                                                  # a broken block of ice

    def plum(pl, pw, n):
        fleck = (fbm(pl * vec(70, 140, 70), 2) > 0.7).float() * (pw[:, 1] < BASE_TOP + 0.48).float()
        a = srgb("#f6f4ee") * (0.9 + 0.1 * fbm(pl * 200, 2))[:, None]
        return {"albedo": a * (1 - 0.75 * fleck)[:, None] + srgb("#3a3430") * 0.75 * fleck[:, None]}
    pm = Mat("#f6f4ee", rough=0.85, sss=0.3, pattern=plum)
    flap = anim == "win"
    sc.add(furry(sd_ellipsoid(0.12, 0.15, 0.11), 0.004, 120), pm, X * T(0, 0.3, 0) * RX(0.12))
    sc.add(furry(sd_ellipsoid(0.1, 0.1, 0.1), 0.004, 120), pm, X * T(0, 0.37, 0.025), blend=0.05)
    turn = 0.9 * math.sin(TAU * t) if anim == "idle" else (0.6 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0)
    hd = X * T(0, 0.49, 0.02) * RY(turn)
    sc.add(furry(sd_ellipsoid(0.1, 0.085, 0.09), 0.004, 130), pm, hd)
    sc.add(sd_ellipsoid(0.075, 0.06, 0.02), Mat("#fbfaf6", rough=0.9), hd * T(0, -0.005, 0.075), blend=0.03)           # the facial disc
    for sgn in (-1, 1):
        ec = hd * T(0.032 * sgn, 0.01, 0.082)
        sc.add(sd_sphere(0.016), Mat("#f2c020", rough=0.05, clearcoat=1.0, emissive="#f2c020", emit=0.1), ec)
        sc.add(sd_sphere(0.008), Mat("#050404", rough=0.05, clearcoat=1.0), ec * T(0, 0, 0.011))
        sc.add(sd_plane_cut(sd_sphere(0.019), (0, -1, 0.3), -0.004), pm, ec, blend=0.003)                              # the lid, a fierce look
    sc.add(sd_cone(0.012, 0.009, 0.001, 0.001), Mat("#1a1614", rough=0.3), hd * T(0, -0.022, 0.092) * RX(2.4))
    # the wings folded along the body (spread to win), the tail below
    for sgn in (-1, 1):
        if flap:
            a = 0.9 + 0.5 * math.sin(TAU * 2 * t)
            w = X * T(0.1 * sgn, 0.38, -0.01) * RZ(a * sgn)
            sc.add(furry(sd_ellipsoid(0.03, 0.2, 0.11), 0.003, 120), pm, w * T(0, 0.17, -0.03))
        else:
            sc.add(furry(sd_ellipsoid(0.035, 0.15, 0.09), 0.003, 120), pm, X * T(0.09 * sgn, 0.3, -0.03) * RZ(-0.08 * sgn) * RX(0.2), blend=0.02)
        ft = X * T(0.04 * sgn, 0.16, 0.06)
        sc.add(furry(sd_ellipsoid(0.03, 0.03, 0.035), 0.004, 120), pm, ft, blend=0.02)
        for k in range(3):
            sc.add(sd_cone(0.016, 0.005, 0.0005, 0.0005), Mat("#1a1614", rough=0.3), ft * T(-0.015 + 0.015 * k, -0.02, 0.03) * RX(2.3))
    sc.add(sd_ellipsoid(0.06, 0.012, 0.09), pm, X * T(0, 0.17, -0.11) * RX(-0.6))
    return sc


# ─── 5 · Morse ──────────────────────────────────────────────────────────────
def walrus(anim, t, side):
    """A bull walrus: an enormous body of wrinkled pinkish-brown skin, the
    head small for it, the muzzle broad with a pad of stiff bristles, long
    ivory tusks, small bloodshot eyes; propped on the front flippers."""
    sc = Scene()
    base(sc, side)
    X = frame_of(anim, t)
    wr = Mat("#5e4030", rough=0.7, sss=0.05, pattern=lambda pl, pw, n: {
        "albedo": srgb("#5e4030") * (0.8 + 0.3 * fbm(pl * 30, 3))[:, None] * (1 - 0.25 * (torch.sin(pl[:, 1] * 180 + 14 * fbm(pl * 9, 3)) > 0.8).float())[:, None]})

    def wrinkled(f):
        return lambda q: f(q) - 0.0025 * torch.sin(q[:, 1] * 180 + 14 * fbm(q * 9, 3)) - 0.004 * fbm(q * 25, 2)
    breathe = 0.01 * math.sin(TAU * t)
    up = 0.45 if anim == "win" else 0.2 + 0.03 * math.sin(TAU * t)
    sc.add(wrinkled(sd_ellipsoid(0.17, 0.13 + breathe, 0.24)), wr, X * T(0, 0.15, -0.07))
    sc.add(wrinkled(sd_ellipsoid(0.15, 0.15, 0.13)), wr, X * T(0, 0.21, 0.1), blend=0.08)
    sc.add(wrinkled(sd_ellipsoid(0.08, 0.06, 0.14)), wr, X * T(0, 0.08, -0.3), blend=0.06)
    hd = X * T(0, 0.3, 0.2) * RX(-up)
    sc.add(wrinkled(sd_ellipsoid(0.095, 0.085, 0.085)), wr, hd, blend=0.06)
    sc.add(sd_ellipsoid(0.09, 0.06, 0.06), wr, hd * T(0, -0.045, 0.065), blend=0.03)                             # the muzzle pad
    bristle = Mat("#e8dcc0", rough=0.6)
    for i in range(7):
        for sgn in (-1, 1):
            sc.add(sd_capsule((0, 0, 0), (0.004 * sgn, -0.018, 0.006), 0.0018, 0.0012), bristle, hd * T(sgn * (0.012 + 0.01 * i), -0.045 - 0.003 * (i % 2), 0.1 - 0.004 * i))
    for sgn in (-1, 1):
        sc.add(sd_capsule((0, 0, 0), (0.008 * sgn, -0.21, 0.025), 0.013, 0.006), IVORY, hd * T(0.026 * sgn, -0.07, 0.07))
        eye(sc, hd, (0.06 * sgn, 0.025, 0.04), 0.008, sgn, "#3a1410")
        fl = X * T(0.15 * sgn, 0.1, 0.12) * RZ(0.35 * sgn)
        sc.add(wrinkled(sd_ellipsoid(0.04, 0.1, 0.05)), wr, fl * T(0, -0.04, 0), blend=0.03)
        sc.add(sd_ellipsoid(0.05, 0.012, 0.07), wr, X * T(0.17 * sgn, 0.012, 0.16) * RY(0.4 * sgn), blend=0.02)
        sc.add(sd_ellipsoid(0.04, 0.012, 0.08), wr, X * T(0.06 * sgn, 0.03, -0.42) * RY(0.3 * sgn) * RX(-0.2), blend=0.02)
    return sc


# ─── 6 · Loup arctique ──────────────────────────────────────────────────────
def arctic_wolf(anim, t, side):
    """The white wolf of the high Arctic: tall and lean, long legs, a broad
    head with a long muzzle and short pointed ears, a thick ruff, a bushy tail."""
    sc = Scene()
    base(sc, side)
    m = pelt("#ece6da", "#d8d0c0", k=90, spread=0.3)
    canid(sc, anim, t, m, s=1.12, leg=1.05, muzzle=1.15, ear=0.8, tail=1.0, fluff=1.1, howl=anim == "win", ruff=1.0)
    return sc


# ─── 7 · Bœuf musqué ────────────────────────────────────────────────────────
def muskox(anim, t, side):
    """A musk ox: a hump over the shoulders, a skirt of long dark brown hair
    hanging almost to the ground, a pale saddle, the head low with the horns'
    boss and the horns sweeping down then out and up, pale stockings."""
    sc = Scene()
    base(sc, side)
    (fl, fr, hl, hr), bob = gait(anim, t)
    X = frame_of(anim, t, bob)
    hair = pelt("#3a2a1e", "#2a1e16", k=70, spread=0.3, strands=0.6)
    saddle = pelt("#a89070", "#7a6448", k=70, spread=0.3)
    amp, k = 0.012, 40.0
    trunk = [(-0.2, 0.29, 0.12, 0.12, 0.12), (-0.04, 0.29, 0.135, 0.14, 0.15), (0.12, 0.31, 0.14, 0.15, 0.12), (0.16, 0.39, 0.09, 0.07, 0.09)]
    segs(sc, hair, X, trunk, amp, k, 0.07, hang=0.8)
    # the skirt of hair, hanging from the flanks
    # the skirt of long hair hanging from the flanks and the belly almost to the hooves
    sc.add(furry(sd_ellipsoid(0.135, 0.14, 0.26), 0.022, 26, hang=1.0), hair, X * T(0, 0.2, -0.03), blend=0.08)
    legs_m = pelt("#d8ccb4", "#c8b89a", k=90, spread=0.25)
    for sgn, sw, front in ((-1, fl, 1), (1, fr, 1), (-1, hl, 0), (1, hr, 0)):
        z = 0.12 if front else -0.18
        pts = swing_pts([(0.075 * sgn, 0.2, z), (0.075 * sgn, 0.1, z + (0.0 if front else -0.02)), (0.075 * sgn, 0.02, z + 0.01)], sw)
        limb(sc, legs_m, X, pts, [0.04, 0.028, 0.026], 0.003, 90, 0.015)
        sc.add(sd_ellipsoid(0.03, 0.014, 0.035), Mat("#1a1612", rough=0.4, clearcoat=0.5), X * T(pts[-1][0], 0.013, pts[-1][2] + 0.01))   # the hoof
    low = 0.2 if anim == "win" else (0.05 if anim == "blocked" else 0.0)
    hd = X * T(0, 0.3 - low * 0.3, 0.29) * RX(0.5 + low) * T(0, 0, 0) 
    S_ = 1.35
    sc.add(furry(sd_ellipsoid(0.07 * S_, 0.08 * S_, 0.1 * S_), 0.01, 40, hang=0.7), hair, hd, blend=0.05)
    sc.add(sd_ellipsoid(0.05 * S_, 0.045 * S_, 0.05 * S_), pelt("#4a3a2c", k=90), hd * T(0, -0.02 * S_, 0.1 * S_), blend=0.03)       # the muzzle
    sc.add(sd_ellipsoid(0.038 * S_, 0.022 * S_, 0.02 * S_), Mat("#2a2420", rough=0.4, clearcoat=0.4), hd * T(0, -0.02 * S_, 0.145 * S_))
    horn = Mat("#4e4234", rough=0.4, clearcoat=0.6, pattern=lambda pl, pw, n: {"albedo": srgb("#5a4c3c") * (0.7 + 0.35 * fbm(pl * 50, 2))[:, None]})
    sc.add(sd_ellipsoid(0.075 * S_, 0.025 * S_, 0.045 * S_), horn, hd * T(0, 0.075 * S_, 0.0), blend=0.01)                    # the boss
    for sgn in (-1, 1):
        pts = bezier((0.04 * sgn, 0.075, 0.0), (0.1 * sgn, 0.06, -0.01), (0.1 * sgn, -0.035, 0.02), (0.13 * sgn, -0.015, 0.07), 6)
        pts = [(x * S_, y * S_, z * S_) for x, y, z in pts]
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            sc.add(sd_capsule(a, b, (0.017 * (1 - i / 6) + 0.003) * S_, (0.017 * (1 - (i + 1) / 6) + 0.003) * S_), horn, hd, blend=0.005)
        eye(sc, hd, (0.062 * sgn * S_, 0.005, 0.05 * S_), 0.01, sgn)
    return sc


# ─── 8 · Ours blanc ─────────────────────────────────────────────────────────
def polar_bear(anim, t, side):
    """A great male polar bear: long neck and small head carried low, a
    straight long muzzle with a black nose, small round ears, the rump as high
    as the shoulders, massive forelegs, huge furred paws with black claws."""
    sc = Scene()
    base(sc, side)
    (fl, fr, hl, hr), bob = gait(anim, t)
    X = frame_of(anim, t, bob)
    m = pelt("#f0e8d6", "#e0d2b8", k=80, spread=0.22)
    amp, k = 0.007, 55.0
    trunk = [(-0.2, 0.31, 0.11, 0.12, 0.11), (-0.06, 0.29, 0.13, 0.13, 0.14), (0.08, 0.29, 0.125, 0.135, 0.12), (0.16, 0.32, 0.1, 0.1, 0.08)]
    segs(sc, m, X, trunk, amp, k, 0.06)
    claw = Mat("#141214", rough=0.3, clearcoat=0.5)
    for sgn, sw in ((-1, fl), (1, fr)):
        pts = swing_pts([(0.075 * sgn, 0.28, 0.14), (0.08 * sgn, 0.15, 0.12), (0.075 * sgn, 0.06, 0.14), (0.075 * sgn, 0.025, 0.155)], sw)
        sc.add(furry(sd_ellipsoid(0.06, 0.1, 0.07), amp, k), m, X * T(pts[0][0], 0.25, 0.13), blend=0.04)          # shoulder and arm
        limb(sc, m, X, pts, [0.06, 0.048, 0.04, 0.038], amp, k, 0.025)
        paw(sc, m, X, (pts[-1][0], 0.02, pts[-1][2] + 0.02), 0.042, 5, claw)
    for sgn, sw in ((-1, hl), (1, hr)):
        pts = swing_pts([(0.075 * sgn, 0.3, -0.2), (0.08 * sgn, 0.17, -0.12), (0.075 * sgn, 0.07, -0.21), (0.075 * sgn, 0.025, -0.19)], sw)
        sc.add(furry(sd_ellipsoid(0.07, 0.11, 0.09), amp, k), m, X * T(pts[0][0], 0.24, -0.16), blend=0.04)         # the thigh
        limb(sc, m, X, pts, [0.06, 0.045, 0.036, 0.036], amp, k, 0.025)
        paw(sc, m, X, (pts[-1][0], 0.02, pts[-1][2] + 0.03), 0.042, 5, claw)
    sc.add(furry(sd_sphere(0.03), amp, k), m, X * T(0, 0.32, -0.32), blend=0.03)                                     # the tail
    roar = anim == "win"
    low = 0.2 if anim == "blocked" else 0.0
    nk = X * T(0, 0.31, 0.17) * RX(0.15 - (0.5 if roar else 0.0) + low)
    sc.add(furry(sd_capsule((0, 0, 0), (0, -0.02, 0.15), 0.085, 0.06), amp, k), m, nk, blend=0.05)
    hd = nk * T(0, -0.02, 0.17) * RX(0.15 + (0.35 if roar else 0.0))
    sc.add(furry(sd_ellipsoid(0.055, 0.055, 0.065), amp * 0.7, k), m, hd, blend=0.03)
    sc.add(furry(sd_capsule((0, -0.005, 0.03), (0, -0.018, 0.11), 0.038, 0.024), amp * 0.4, k), m, hd, blend=0.02)       # the long muzzle
    gape = 0.3 if roar else 0.0
    jaw = hd * T(0, -0.03, 0.03) * RX(gape)
    sc.add(sd_capsule((0, 0, 0), (0, -0.008, 0.075), 0.025, 0.017), m, jaw, blend=0.012)
    if roar:
        sc.add(sd_ellipsoid(0.022, 0.015, 0.05), Mat("#5a1a24", rough=0.5), hd * T(0, -0.04, 0.07) * RX(0.15))
        for sgn in (-1, 1):
            sc.add(sd_cone(0.012, 0.005, 0.0005, 0.0005), IVORY, hd * T(0.016 * sgn, -0.03, 0.1) * RX(math.pi))
            sc.add(sd_cone(0.012, 0.005, 0.0005, 0.0005), IVORY, jaw * T(0.014 * sgn, 0.012, 0.065))
    sc.add(sd_ellipsoid(0.018, 0.013, 0.012), Mat("#121014", rough=0.3, clearcoat=0.6), hd * T(0, -0.012, 0.125))
    sc.add(sd_capsule((-0.02, 0, 0), (0.02, 0, 0), 0.004), Mat("#121014", rough=0.4), hd * T(0, -0.037, 0.105))      # the black lips
    for sgn in (-1, 1):
        eye(sc, hd, (0.033 * sgn, 0.018, 0.045), 0.008, sgn, "#100a08")
        sc.add(furry(sd_ellipsoid(0.019, 0.017, 0.009), amp * 0.4, k), m, hd * T(0.04 * sgn, 0.045, -0.015) * RZ(-0.3 * sgn), blend=0.01)
    return sc


# ─── 9 · Orque ──────────────────────────────────────────────────────────────
def along(p, tng):
    """A frame at p with z along the tangent and y towards the back (up)."""
    z = torch.nn.functional.normalize(vec(*tng), dim=0)
    y = vec(0, 1, 0) - z * z[1]
    y = torch.nn.functional.normalize(y + vec(0, 0, 1e-4), dim=0)
    x = torch.linalg.cross(y, z)
    return X_(torch.stack([x, y, z], 1), vec(*p))


def orca_mat(u0, u1, half):
    """The killer whale's pattern by the position u along the body (0 tail → 1 snout)
    and the angle around it, in the frame of one segment."""
    def pat(pl, pw, n):
        u = u0 + (pl[:, 2] / (2 * half) + 0.5).clamp(0, 1) * (u1 - u0)
        r = torch.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2) + 1e-6
        up = pl[:, 1] / r
        belly = ((up < -0.35) & (u > 0.42) & (u < 0.97)).float()
        flank = ((up < 0.05 + 0.25 * torch.sin(((u - 0.18) / 0.26).clamp(0, 1) * math.pi)) & (u > 0.18) & (u < 0.44) & (up < 0.2)).float()
        flank = flank * (up < -0.1 + 0.4 * torch.sin(((u - 0.18) / 0.26).clamp(0, 1) * math.pi)).float()
        patch = ((((u - 0.84) / 0.06) ** 2 + ((up - 0.35) / 0.16) ** 2) < 1).float()
        saddle = ((((u - 0.5) / 0.08) ** 2 + ((up - 0.95) / 0.18) ** 2) < 1).float()
        white = (belly + flank + patch).clamp(0, 1)
        black = srgb("#0b0c0f") * (1 - saddle)[:, None] + srgb("#6e737c") * saddle[:, None]
        return {"albedo": black * (1 - white)[:, None] + srgb("#f0f0ec") * white[:, None]}
    return Mat("#0b0c0f", rough=0.2, clearcoat=0.9, pattern=pat)


def orca(anim, t, side):
    """A killer whale breaching through a hole in the ice, the body arched,
    the head up: black, the white eye patch, chin and belly, the white patch
    sweeping up the rear flank, the grey saddle behind the tall dorsal fin."""
    sc = Scene()
    base(sc, side)
    # the hole: dark water, foam, slabs of broken ice
    sc.add(sd_cylinder(0.005, 0.36, 0.004), Mat("#0a1622", rough=0.04, clearcoat=1.0), T(0, BASE_TOP + 0.004, 0))
    foam = Mat("#f4f8fc", rough=0.6, sss=0.5)
    sc.add(lambda q: sd_torus(0.2, 0.03)(q) - 0.012 * fbm(q * 40, 2), foam, T(0, BASE_TOP + 0.01, 0))
    for k in range(8):
        a = TAU * k / 8 + 0.3
        rr = 0.42 + 0.02 * math.sin(k * 2.7)
        pts = [(math.cos(a + d) * (rr + e), 0.0, math.sin(a + d) * (rr + e)) for d, e in ((-0.32, -0.06), (0.3, -0.05), (0.28, 0.1), (-0.25, 0.12))]
        sc.add(sheet([(x * 0.85, y, z * 0.85) for x, y, z in pts], 0.012 + 0.005 * (k % 3)), ICE, T(0, BASE_TOP + 0.022, 0) * RZ(0.06 * math.sin(k * 1.9)) * RX(0.05 * math.cos(k * 2.3)))
    rise = {"win": 0.06 * math.sin(TAU * t), "move": 0.05 * math.sin(math.pi * t)}.get(anim, 0.012 * math.sin(TAU * t))
    roll = 0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    Xr = RY(YAW) * T(0, BASE_TOP + rise, 0) * RZ(roll)
    # the spine: from the flukes under the water up to the snout
    spine = bezier((0, -0.35, -0.3), (0, 0.05, -0.2), (0, 0.42, -0.05), (0, 0.6, 0.22), 10)
    radii = [0.035, 0.06, 0.09, 0.115, 0.125, 0.12, 0.11, 0.095, 0.075, 0.045]
    for i in range(len(spine) - 1):
        p0, p1 = spine[i], spine[i + 1]
        mid = tuple((p0[k] + p1[k]) / 2 for k in range(3))
        tg = tuple(p1[k] - p0[k] for k in range(3))
        half = 0.5 * math.sqrt(sum(v * v for v in tg))
        r = (radii[i] + radii[i + 1]) / 2
        sc.add(sd_capsule((0, 0, -half), (0, 0, half), radii[i], radii[i + 1]), orca_mat(i / 9, (i + 1) / 9, half), Xr * along(mid, tg), blend=0.03)
    # the head: rounded melon at the snout
    hp, ht = spine[-1], tuple(spine[-1][k] - spine[-2][k] for k in range(3))
    sc.add(sd_ellipsoid(0.05, 0.045, 0.06), orca_mat(0.92, 1.0, 0.06), Xr * along(hp, ht), blend=0.04)
    # the dorsal fin on the back at the widest point, the paddles, the flukes
    fp, ft = spine[5], tuple(spine[6][k] - spine[4][k] for k in range(3))
    black = Mat("#0b0c0f", rough=0.2, clearcoat=0.9)
    sc.add(sd_thin_fin(0.2, 0.11, 0.014), black, Xr * along(fp, ft) * T(0, 0.1, 0))
    pp, pt = spine[7], tuple(spine[8][k] - spine[6][k] for k in range(3))
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.014, 0.05, 0.075), black, Xr * along(pp, pt) * T(0.09 * sgn, -0.07, 0) * RZ(-0.9 * sgn) * T(0, -0.04, 0) * RX(-0.3))
    # the spray: droplets thrown up around the body
    drop = Mat("#ffffff", rough=0.15, clearcoat=1.0, sss=0.6, emissive="#e8f6ff", emit=0.15)
    for k in range(22):
        a = TAU * k / 22 + 0.7 * k
        h = 0.04 + 0.22 * ((k * 0.37 + t) % 1) * (0.5 + 0.5 * math.sin(k * 1.3))
        rr = 0.14 + 0.1 * ((k * 0.61) % 1)
        sc.add(sd_sphere(0.008 + 0.006 * (k % 3)), drop, T(rr * math.cos(a), BASE_TOP + h, rr * math.sin(a)))
    return sc


def sd_thin_fin(h, w, th):
    """A dorsal fin: a curved triangle, thin, its tip swept back."""
    def f(p):
        y = p[:, 1].clamp(0, h)
        u = y / h
        half = w * (1 - u) ** 1.2 * 0.5
        cz = -0.35 * w * u ** 1.5
        dz = (p[:, 2] - cz).abs() - half
        dx = p[:, 0].abs() - th * (1 - 0.7 * u)
        dy = torch.maximum(-p[:, 1], p[:, 1] - h)
        d = torch.maximum(torch.maximum(dz, dx), dy)
        return torch.maximum(d * 0.7, torch.linalg.norm(p - vec(0, h * 0.5, -0.1 * w), dim=-1) - h)
    return f


# ─── 10 · Mammouth laineux ──────────────────────────────────────────────────
def mammoth(anim, t, side):
    """A woolly mammoth after the reconstructions: the high domed head, the
    hump of fat over the shoulders and the back sloping down to the hips,
    long reddish-brown hair hanging to the knees, small ears, a short tail,
    columns for legs, the trunk ending in two lips, long tusks spiralling
    up and inwards."""
    sc = Scene()
    base(sc, side)
    (fl, fr, hl, hr), bob = gait(anim, t)
    X = frame_of(anim, t, bob)
    hair = pelt("#4a2e1c", "#36200f", k=60, spread=0.35, strands=0.7)
    amp, k = 0.014, 32.0
    trunk = [(-0.2, 0.4, 0.11, 0.11, 0.11), (-0.06, 0.42, 0.13, 0.13, 0.14), (0.08, 0.45, 0.13, 0.15, 0.12), (0.12, 0.55, 0.08, 0.07, 0.09)]
    segs(sc, hair, X, trunk, amp, k, 0.08, hang=0.9)
    legs_m = pelt("#4a3020", "#3a2618", k=60, spread=0.35, strands=0.5)
    for sgn, sw, front in ((-1, fl, 1), (1, fr, 1), (-1, hl, 0), (1, hr, 0)):
        z = 0.1 if front else -0.18
        top = 0.36 if front else 0.34
        pts = swing_pts([(0.08 * sgn, top, z), (0.08 * sgn, 0.16, z + 0.01), (0.08 * sgn, 0.03, z)], sw * 0.6)
        limb(sc, legs_m, X, pts, [0.058, 0.046, 0.046], 0.008, 40, 0.02, hang=0.4)
        sc.add(sd_cylinder(0.014, 0.05, 0.008), Mat("#2a2420", rough=0.6), X * T(pts[-1][0], 0.014, pts[-1][2]))           # the foot
    sc.add(furry(sd_capsule((0, 0, 0), (0, -0.1, -0.04), 0.015, 0.008), 0.01, 40, hang=0.8), hair, X * T(0, 0.38, -0.3))  # the short tail
    raise_ = 0.6 if anim == "win" else 0.08 * math.sin(TAU * t)
    hd = X * T(0, 0.55, 0.21) * RX(0.25 - (0.2 if anim == "win" else 0.0))
    sc.add(furry(sd_ellipsoid(0.1, 0.12, 0.1), 0.01, 36, hang=0.5), hair, hd, blend=0.06)
    sc.add(furry(sd_ellipsoid(0.07, 0.07, 0.07), 0.01, 36), hair, hd * T(0, 0.1, -0.02), blend=0.05)                 # the dome
    skin_m = Mat("#3a2a20", rough=0.7, pattern=lambda pl, pw, n: {"albedo": srgb("#3e2c22") * (0.8 + 0.3 * fbm(pl * 50, 2))[:, None]})
    # the trunk: down then curling up at the tip (raised to trumpet when it wins)
    tp = bezier((0, -0.04, 0.08), (0, -0.2, 0.15), (0, -0.33 + raise_ * 0.3, 0.14 + raise_ * 0.15), (0, -0.33 + raise_ * 0.5, 0.21 + raise_ * 0.1), 9)
    for i, (a, b) in enumerate(zip(tp[:-1], tp[1:])):
        ra, rb = 0.045 * (1 - i / 9) + 0.014, 0.045 * (1 - (i + 1) / 9) + 0.014
        sc.add(lambda q, a=a, b=b, ra=ra, rb=rb: sd_capsule(a, b, ra, rb)(q) - 0.002 * torch.sin(q[:, 1] * 320), skin_m, hd, blend=0.012)
    sc.add(furry(sd_capsule(tp[0], tp[3], 0.05, 0.035), 0.009, 40, hang=0.6), hair, hd, blend=0.02)                     # hair down the trunk
    # the tusks: out, down, forward, then curving up and inwards
    for sgn in (-1, 1):
        pts = bezier((0.045 * sgn, -0.06, 0.07), (0.16 * sgn, -0.3, 0.2), (0.12 * sgn, -0.18, 0.45), (0.02 * sgn, 0.02, 0.4), 12)
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            sc.add(sd_capsule(a, b, 0.024 * (1 - i / 13) + 0.005, 0.024 * (1 - (i + 1) / 13) + 0.005), IVORY, hd, blend=0.004)
        eye(sc, hd, (0.085 * sgn, 0.0, 0.06), 0.009, sgn)
        sc.add(furry(sd_ellipsoid(0.015, 0.04, 0.03), 0.006, 40), hair, hd * T(0.095 * sgn, 0.02, -0.04), blend=0.01)    # the small ears
    return sc


# ─── 5b · Renne (minmax) ────────────────────────────────────────────────────
def reindeer(anim, t, side):
    """A reindeer bull in early winter: a compact deer on long legs, the coat
    dark grey-brown with a pale neck and a long creamy mane under the throat,
    a white rump, broad hooves, a heavy furred muzzle, and the great antlers:
    two long beams sweeping back then forward, the tines along them, one
    flattened brow tine (the shovel) over the face."""
    sc = Scene()
    base(sc, side)
    (fl, fr, hl, hr), bob = gait(anim, t)
    X = frame_of(anim, t, bob)
    coat = pelt("#5a4a3c", "#7a6a58", k=90, spread=0.3)
    pale = pelt("#e6dccb", "#d6cab4", k=90, spread=0.25)
    amp, k = 0.005, 60.0
    trunk = [(-0.16, 0.3, 0.075, 0.08, 0.09), (-0.03, 0.3, 0.085, 0.09, 0.11), (0.1, 0.31, 0.085, 0.095, 0.09)]
    segs(sc, coat, X, trunk, amp, k, 0.05)
    sc.add(furry(sd_ellipsoid(0.07, 0.07, 0.04), amp, k), pale, X * T(0, 0.31, -0.23), blend=0.03)       # the white rump
    sc.add(furry(sd_sphere(0.02), amp, k), pale, X * T(0, 0.33, -0.26), blend=0.015)                      # the short tail
    leg_m = pelt("#4e4236", "#5e5040", k=90, spread=0.25)
    hoof = Mat("#1c1814", rough=0.4, clearcoat=0.5)
    for sgn, sw, front in ((-1, fl, 1), (1, fr, 1), (-1, hl, 0), (1, hr, 0)):
        z = 0.12 if front else -0.16
        if front:
            pts = [(0.05 * sgn, 0.27, z), (0.05 * sgn, 0.15, z - 0.01), (0.048 * sgn, 0.06, z + 0.005), (0.048 * sgn, 0.02, z + 0.01)]
            radii = [0.04, 0.022, 0.015, 0.014]
        else:
            pts = [(0.05 * sgn, 0.28, z), (0.052 * sgn, 0.17, z + 0.03), (0.05 * sgn, 0.1, z - 0.04), (0.048 * sgn, 0.02, z - 0.02)]
            radii = [0.05, 0.026, 0.016, 0.014]
            sc.add(furry(sd_ellipsoid(0.045, 0.07, 0.06), amp, k), coat, X * T(0.05 * sgn, 0.26, z - 0.01), blend=0.03)   # the haunch
        pts = swing_pts(pts, sw)
        limb(sc, leg_m, X, pts, radii, amp * 0.5, k, 0.015)
        for d in (-1, 1):   # the two broad toes of the hoof
            sc.add(sd_ellipsoid(0.012, 0.012, 0.022), hoof, X * T(pts[-1][0] + d * 0.009, 0.012, pts[-1][2] + 0.012) * RY(0.12 * d))
    # the neck, carried forward and up, the mane hanging under it
    low = 0.25 if anim == "blocked" else 0.0
    up = 0.25 if anim == "win" else 0.04 * math.sin(TAU * t)
    nk = X * T(0, 0.33, 0.15) * RX(-0.75 - up + low)
    sc.add(furry(sd_capsule((0, 0, 0), (0, 0, 0.13), 0.055, 0.04), amp, k), pale, nk, blend=0.04)
    sc.add(furry(sd_ellipsoid(0.04, 0.06, 0.08), 0.012, 40, hang=1.0), pale, nk * T(0, -0.04, 0.06), blend=0.03)   # the throat mane
    hd = nk * T(0, 0, 0.14) * RX(1.15 + up * 0.6 - low)
    sc.add(furry(sd_ellipsoid(0.05, 0.052, 0.06), amp * 0.6, k), coat, hd, blend=0.025)
    sc.add(furry(sd_capsule((0, -0.01, 0.03), (0, -0.026, 0.115), 0.038, 0.03), amp * 0.5, k), coat, hd, blend=0.02)   # the long furred muzzle
    sc.add(sd_ellipsoid(0.026, 0.016, 0.014), pelt("#3a3028", k=120, spread=0.2), hd * T(0, -0.016, 0.138), blend=0.008)  # the hairy nose
    for sgn in (-1, 1):
        eye(sc, hd, (0.04 * sgn, 0.014, 0.035), 0.009, sgn, "#1a120c")
        sc.add(sd_ellipsoid(0.009, 0.026, 0.006), leg_m, hd * T(0.045 * sgn, 0.025, -0.02) * RZ(-1.2 * sgn) * RX(-0.3) * T(0, 0.02, 0), blend=0.004)
    # the antlers
    horn = Mat("#c8b494", rough=0.45, clearcoat=0.3, pattern=lambda pl, pw, n: {"albedo": srgb("#cbb898") * (0.75 + 0.3 * fbm(pl * 60, 2))[:, None]})
    for sgn in (-1, 1):
        beam = bezier((0.025 * sgn, 0.035, -0.005), (0.11 * sgn, 0.13, -0.11), (0.15 * sgn, 0.3, -0.03), (0.09 * sgn, 0.36, 0.08), 10)
        for i, (a, b) in enumerate(zip(beam[:-1], beam[1:])):
            sc.add(sd_capsule(a, b, 0.011 - 0.0006 * i, 0.011 - 0.0006 * (i + 1)), horn, hd, blend=0.004)
        for i, ln in ((3, 0.07), (5, 0.08), (7, 0.06), (8, 0.05)):   # the tines, pointing forward and up
            a = beam[i]
            b = (a[0] + 0.01 * sgn, a[1] + ln * 0.75, a[2] + ln * 0.75)
            sc.add(sd_capsule(a, b, 0.007, 0.003), horn, hd, blend=0.003)
        if sgn < 0:   # the shovel: one brow tine, flattened, over the face
            sc.add(sd_ellipsoid(0.006, 0.035, 0.03), horn, hd * T(0.02 * sgn, 0.07, 0.05) * RX(0.6), blend=0.006)
            sc.add(sd_capsule(beam[1], (0.02 * sgn, 0.065, 0.04), 0.008, 0.006), horn, hd, blend=0.004)
        else:
            sc.add(sd_capsule(beam[1], (0.04, 0.09, 0.05), 0.007, 0.003), horn, hd, blend=0.003)
    return sc


# ─── 5c · Narval (mcts_nnue) ────────────────────────────────────────────────
def narwhal_mat(u0, u1, half):
    """The narwhal's skin: grey dappled with dark spots on the back, paler
    and almost white on the belly and the flanks."""
    def pat(pl, pw, n):
        r = torch.sqrt(pl[:, 0] ** 2 + pl[:, 1] ** 2) + 1e-6
        up = pl[:, 1] / r
        spots = (fbm(pw * 38, 2) > 0.55).float() * ((up + 0.2) / 0.8).clamp(0, 1)
        light = srgb("#dcdcd8") * (1 - ((up + 0.4) / 1.2).clamp(0, 1))[:, None] + srgb("#7c8088") * ((up + 0.4) / 1.2).clamp(0, 1)[:, None]
        return {"albedo": light * (1 - 0.75 * spots)[:, None] + srgb("#2a2c32") * 0.75 * spots[:, None]}
    return Mat("#8a8e94", rough=0.25, clearcoat=0.85, pattern=pat)


def narwhal(anim, t, side):
    """A narwhal surfacing through a hole in the ice, the head out of the
    water: a small round head, no dorsal fin but a low ridge, the grey dappled
    back, the pale belly, and the tusk: a canine grown straight out of the
    upper lip, long, twisted in a left-handed spiral, ivory."""
    sc = Scene()
    base(sc, side)
    sc.add(sd_cylinder(0.005, 0.36, 0.004), Mat("#0a1622", rough=0.04, clearcoat=1.0), T(0, BASE_TOP + 0.004, 0))
    foam = Mat("#f4f8fc", rough=0.6, sss=0.5)
    sc.add(lambda q: sd_torus(0.17, 0.026)(q) - 0.01 * fbm(q * 40, 2), foam, T(0, BASE_TOP + 0.01, 0))
    for k in range(7):
        a = TAU * k / 7 + 0.9
        rr = 0.42 + 0.02 * math.sin(k * 3.1)
        pts = [(math.cos(a + d) * (rr + e), 0.0, math.sin(a + d) * (rr + e)) for d, e in ((-0.36, -0.06), (0.34, -0.05), (0.3, 0.1), (-0.28, 0.12))]
        sc.add(sheet([(x * 0.85, y, z * 0.85) for x, y, z in pts], 0.012 + 0.005 * (k % 3)), ICE, T(0, BASE_TOP + 0.022, 0) * RZ(0.06 * math.sin(k * 2.1)) * RX(0.05 * math.cos(k * 1.7)))
    rise = {"win": 0.05 * math.sin(TAU * t), "move": 0.05 * math.sin(math.pi * t)}.get(anim, 0.012 * math.sin(TAU * t))
    roll = 0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    tilt = -0.25 if anim == "win" else 0.0
    Xr = RY(YAW) * T(0, BASE_TOP + rise, 0) * RZ(roll) * RX(tilt)
    spine = bezier((0, -0.3, -0.26), (0, 0.0, -0.16), (0, 0.28, -0.02), (0, 0.4, 0.14), 9)
    radii = [0.03, 0.055, 0.08, 0.1, 0.108, 0.104, 0.094, 0.08, 0.065]
    for i in range(len(spine) - 1):
        p0, p1 = spine[i], spine[i + 1]
        mid = tuple((p0[k] + p1[k]) / 2 for k in range(3))
        tg = tuple(p1[k] - p0[k] for k in range(3))
        half = 0.5 * math.sqrt(sum(v * v for v in tg))
        sc.add(sd_capsule((0, 0, -half), (0, 0, half), radii[i], radii[i + 1]), narwhal_mat(i / 8, (i + 1) / 8, half), Xr * along(mid, tg), blend=0.03)
    hp, ht = spine[-1], tuple(spine[-1][k] - spine[-2][k] for k in range(3))
    H = Xr * along(hp, ht)
    sc.add(sd_ellipsoid(0.066, 0.06, 0.07), narwhal_mat(0.9, 1.0, 0.07), H, blend=0.04)          # the round head (melon)
    # the low dorsal ridge instead of a fin
    for i in (4, 5, 6):
        pr, tr = spine[i], tuple(spine[i + 1][k] - spine[i - 1][k] for k in range(3))
        sc.add(sd_ellipsoid(0.012, 0.02, 0.05), narwhal_mat(0.5, 0.6, 0.05), Xr * along(pr, tr) * T(0, radii[i] - 0.005, 0), blend=0.02)
    # the flippers
    pp, pt = spine[6], tuple(spine[7][k] - spine[5][k] for k in range(3))
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.008, 0.022, 0.04), narwhal_mat(0.6, 0.7, 0.04),
               Xr * along(pp, pt) * T(0.08 * sgn, -0.06, 0) * RZ(-1.1 * sgn) * T(0, -0.02, 0) * RX(-0.3))
    # the eyes, small, low on the head
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.007), EYE, H * T(0.055 * sgn, -0.012, 0.02))
    # the tusk: from the left of the upper lip, straight, a left-handed spiral
    L = 0.42
    tusk_m = Mat("#efe6d0", rough=0.35, clearcoat=0.5,
                 pattern=lambda pl, pw, n: {"albedo": srgb("#e8dcc0") * (0.82 + 0.18 * torch.sin(torch.atan2(pl[:, 0], pl[:, 1]) * 2 + pl[:, 2] * 140))[:, None]})

    def tusk(q):
        z = q[:, 2].clamp(0, L)
        r = 0.014 * (1 - z / L) + 0.002
        ang = torch.atan2(q[:, 0], q[:, 1]) - z * 140 / 2
        rr = torch.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2)
        d = torch.sqrt((rr - r).clamp_min(0) ** 2 + (q[:, 2] - z) ** 2) + torch.minimum(rr - r, torch.zeros_like(rr)).clamp_max(0) * 0
        d = torch.where((q[:, 2] >= 0) & (q[:, 2] <= L), rr - r, d)
        return d * 0.8 - 0.0015 * torch.sin(ang * 2)
    sc.add(tusk, tusk_m, H * T(-0.02, -0.01, 0.05) * RX(-0.05))
    # the spray
    drop = Mat("#ffffff", rough=0.15, clearcoat=1.0, sss=0.6, emissive="#e8f6ff", emit=0.15)
    for k in range(16):
        a = TAU * k / 16 + 0.7 * k
        h = 0.03 + 0.16 * ((k * 0.37 + t) % 1) * (0.5 + 0.5 * math.sin(k * 1.3))
        rr = 0.12 + 0.08 * ((k * 0.61) % 1)
        sc.add(sd_sphere(0.007 + 0.005 * (k % 3)), drop, T(rr * math.cos(a), BASE_TOP + h, rr * math.sin(a)))
    return sc


SCALE = {"random": 1.4, "beginner": 1.3, "monte_carlo": 1.45, "mcts": 1.4, "basic_minmax": 1.35, "minmax": 1.4, "mcts_nnue": 1.45, "nnue193": 1.4,
         "features": 1.4, "resnet": 1.45, "dev": 1.5, "alphazero": 1.45}
CAST = {"random": ("Phoque gris", seal), "beginner": ("Manchot empereur", penguin), "monte_carlo": ("Renard polaire", arctic_fox),
        "mcts": ("Harfang des neiges", snowy_owl), "basic_minmax": ("Morse", walrus), "minmax": ("Renne", reindeer), "mcts_nnue": ("Narval", narwhal),
        "nnue193": ("Loup arctique", arctic_wolf),
        "features": ("Bœuf musqué", muskox), "resnet": ("Ours blanc", polar_bear), "dev": ("Orque", orca),
        "alphazero": ("Mammouth laineux", mammoth)}
ARCTIC = {"ice_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"ice_" + k: n for k, (n, _) in CAST.items()}
