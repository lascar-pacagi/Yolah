"""
models.py — the characters (robots of the AIs, the human) as functions
model(anim, t, side) → Scene, rendered by render.py.

Every piece is a miniature on a round base: the base tells the side (black:
obsidian and gold, white: porcelain and gold), the figure tells the player.
The figures are slightly turned to the right (HEADING); the app mirrors them
when they move left. Proportions are "chibi": big heads, so that the faces
and the eyes read at the size of a square.
"""
import math
from render import (KEY, Scene, Mat, X, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule,
                    sd_cylinder, sd_torus, sd_cone, sd_plane_cut, fbm, vnoise, length)
import torch

TAU = 2 * math.pi
HEADING = math.radians(28)       # the figure turns this much to the right
BASE_TOP = 0.11                  # height of the top of the base


def wave(t, k=1, ph=0.0):
    return math.sin(TAU * (k * t + ph))


# ─── the base ───────────────────────────────────────────────────────────────
def gold():
    return Mat("#d9a441", metal=1.0, rough=0.28)


def base(sc, side):
    if side == 0:      # black: obsidian
        body = Mat("#16181e", metal=0.2, rough=0.22, clearcoat=1.0,
                   pattern=lambda pl, pw, n: {"albedo": srgb("#16181e") * (0.8 + 0.5 * fbm(pl * 9)[:, None])})
    else:              # white: porcelain with a faint vein
        body = Mat("#ece8e1", metal=0.0, rough=0.25, clearcoat=1.0,
                   pattern=lambda pl, pw, n: {"albedo": srgb("#ece8e1") * (0.9 + 0.12 * fbm(pl * 6)[:, None])})
    sc.add(sd_cylinder(0.045, 0.56, 0.035), body, T(0, 0.045, 0))
    sc.add(sd_cylinder(0.035, 0.50, 0.03), body, T(0, 0.09, 0))
    sc.add(sd_torus(0.535, 0.012), gold(), T(0, 0.088, 0))
    sc.add(sd_torus(0.49, 0.008), gold(), T(0, 0.124, 0))


# ─── materials ──────────────────────────────────────────────────────────────
def armor(hexcol, rough=0.38, lines=7.0, wear=0.35):
    base_c = srgb(hexcol)

    def pat(pl, pw, n):
        g = torch.frac(pl[:, 1] * lines + 0.5)
        groove = ((g < 0.05) | (g > 0.95)).float()
        w = fbm(pl * 14.0, 3)
        worn = ((w - 0.62).clamp_min(0) * 6 * wear).clamp(0, 1)
        a = base_c * (1 - 0.65 * groove)[:, None]
        a = a * (1 - worn[:, None]) + srgb("#b9bec6") * worn[:, None]
        r = torch.full_like(w, rough) + 0.3 * groove - 0.15 * worn
        return {"albedo": a, "rough": r}
    return Mat(hexcol, metal=0.85, rough=rough, pattern=pat)


def glow(hexcol, k=6.0):
    return Mat(hexcol, metal=0.0, rough=0.2, emissive=hexcol, emit=k * 0.4)


# ─── NNUE dev: the Sentinel, a heavy war robot ──────────────────────────────
def painted(hexcol, paint="#c8361f", axis=0, lo=0.35, hi=0.65, rough=0.4):
    """armor with a band of war paint across local axis between lo and hi (fractions of ±1)."""
    a0 = armor(hexcol, rough=rough, lines=5.0)
    pc = srgb(paint)

    def pat(pl, pw, n):
        o = a0.pattern(pl, pw, n)
        x = pl[:, axis] / 0.2
        band = ((x > lo) & (x < hi)).float()
        chip = (fbm(pl * 22.0, 3) > 0.58).float()
        b = band * (1 - chip)
        o["albedo"] = o["albedo"] * (1 - b[:, None]) + pc * b[:, None]
        o["rough"] = o["rough"] * (1 - b) + 0.45 * b
        return o
    return Mat(hexcol, metal=0.8, rough=rough, pattern=pat)


def sentinel(anim, t, side):
    sc = Scene()
    base(sc, side)
    plate = armor("#5d6572", rough=0.34)
    dark = armor("#2a2e36", rough=0.5, lines=12, wear=0.15)
    paint = painted("#5d6572", axis=0, lo=0.15, hi=0.55)
    black = Mat("#07080a", metal=0.4, rough=0.12, clearcoat=1.0)
    red = "#ff2414"
    bob = lean = head_yaw = head_pitch = shake = step = 0.0
    visor, core = 7.0, 5.0
    arm_l, arm_r = 0.1, 0.25           # shoulder pitch (forward +)
    elbow_l, elbow_r = -0.55, -0.9
    sparks = []
    if anim == "idle":
        bob = 0.010 * wave(t)
        head_yaw = 0.2 * wave(t, 1, 0.1)
        visor = 6.0 + 3.0 * (0.5 + 0.5 * wave(t, 2))
        core = 4.0 + 2.5 * (0.5 + 0.5 * wave(t, 1, 0.25))
    elif anim == "move":
        step = math.sin(math.pi * t)
        lean = 0.25 * math.sin(math.pi * t)
        bob = 0.05 * math.sin(math.pi * t)
        arm_l, arm_r = 0.1 - 0.6 * step, 0.25 + 0.3 * step
        visor = 11.0
    elif anim == "blocked":
        shake = 0.06 * math.sin(TAU * 3 * t)
        head_yaw = 0.5 * math.sin(TAU * 2 * t)
        visor = 1.5 + 9.0 * (1.0 if (int(t * 12) % 3) else 0.1)
        core = 1.5
        for k in range(5):
            ph = (t * 2 + k * 0.23) % 1
            sparks.append((0.3 + 0.12 * math.cos(k * 2.1) + 0.1 * ph, 0.8 + 0.3 * ph - 0.25 * ph * ph,
                           0.06 * math.sin(k * 1.7), 0.02 * (1 - ph)))
    elif anim == "win":
        bob = 0.05 * abs(math.sin(TAU * t))
        arm_r, elbow_r = 2.6 + 0.2 * wave(t), -0.1       # cannon to the sky
        arm_l, elbow_l = 0.4, -1.6
        head_pitch = -0.28
        visor = 16.0 + 5.0 * wave(t, 2)
        core = 12.0 + 4.0 * wave(t, 2)
    body = RY(HEADING + shake) * T(0, bob, 0)
    y0 = BASE_TOP
    # legs: short, massive
    for sgn in (-1, 1):
        s_ = step * sgn * 0.6
        hip = body * T(0.13 * sgn, y0 + 0.30, 0)
        thigh = hip * RX(-0.5 * s_)
        sc.add(sd_capsule((0, 0, 0), (0, -0.09, 0.02), 0.08), dark, thigh)
        knee = thigh * T(0, -0.1, 0.03)
        sc.add(sd_box(0.07, 0.06, 0.05, 0.03), plate, knee * T(0, 0, 0.04))
        shin = knee * RX(0.4 * max(0.0, s_))
        sc.add(sd_capsule((0, 0, 0), (0, -0.07, -0.01), 0.085, 0.10), plate, shin)
        foot = shin * T(0, -0.1, 0.03)
        sc.add(sd_box(0.10, 0.045, 0.15, 0.035), dark, foot)
        sc.add(sd_box(0.08, 0.014, 0.03, 0.01), glow(red, 2.5), foot * T(0, -0.005, 0.16))
    torso = body * T(0, y0 + 0.36, 0) * RX(lean) * T(0, 0.24, 0)
    # pelvis, abdomen, chest (a V)
    sc.add(sd_box(0.16, 0.06, 0.12, 0.05), dark, torso * T(0, -0.22, 0))
    sc.add(sd_box(0.15, 0.08, 0.12, 0.05), dark, torso * T(0, -0.12, 0.0))
    sc.add(sd_box(0.29, 0.13, 0.18, 0.09), plate, torso * T(0, 0.04, 0), blend=0.03)
    sc.add(sd_box(0.12, 0.07, 0.03, 0.03), dark, torso * T(0, 0.02, 0.175))
    sc.add(sd_sphere(0.05), glow(red, core), torso * T(0, 0.02, 0.19))
    sc.add(sd_torus(0.062, 0.014), plate, torso * T(0, 0.02, 0.19) * RX(math.pi / 2))
    for k in range(3):
        for sgn in (-1, 1):
            sc.add(sd_box(0.05, 0.009, 0.02, 0.006), dark, torso * T(0.17 * sgn, 0.09 - 0.04 * k, 0.17))
    # backpack, exhausts
    sc.add(sd_box(0.18, 0.14, 0.08, 0.04), dark, torso * T(0, 0.04, -0.2))
    for sgn in (-1, 1):
        sc.add(sd_cylinder(0.08, 0.04, 0.012), plate, torso * T(0.1 * sgn, 0.18, -0.24))
        if anim in ("move", "win"):
            sc.add(sd_sphere(0.032), glow("#ff8a3a", 7.0), torso * T(0.1 * sgn, 0.27, -0.24))
    # arms
    for sgn, pitch, elbow in ((-1, arm_l, elbow_l), (1, arm_r, elbow_r)):
        sh = torso * T(0.36 * sgn, 0.1, 0) * RZ(0.1 * sgn) * RX(-pitch)
        pad = sh * T(0.03 * sgn, 0.07, 0) * RZ(-0.35 * sgn)
        sc.add(sd_box(0.15, 0.08, 0.15, 0.06), paint, pad)
        sc.add(sd_box(0.13, 0.03, 0.13, 0.025), dark, pad * T(0, -0.08, 0))
        sc.add(sd_capsule((0, 0, 0), (0, -0.14, 0), 0.06), dark, sh)
        el = sh * T(0, -0.15, 0) * RX(-elbow)
        sc.add(sd_sphere(0.065), plate, el)
        if sgn > 0:      # the cannon arm
            sc.add(sd_cylinder(0.13, 0.085, 0.025), plate, el * T(0, -0.14, 0))
            sc.add(sd_box(0.03, 0.09, 0.05, 0.015), dark, el * T(0.08, -0.12, 0))
            sc.add(sd_cylinder(0.06, 0.055, 0.012), dark, el * T(0, -0.31, 0))
            sc.add(sd_cylinder(0.025, 0.04, 0.006), glow(red, visor * 0.7), el * T(0, -0.37, 0))
            for k in range(3):
                sc.add(sd_torus(0.088, 0.009), dark, el * T(0, -0.05 - 0.065 * k, 0))
        else:            # the fist
            sc.add(sd_capsule((0, 0, 0), (0, -0.12, 0), 0.065, 0.075), plate, el)
            sc.add(sd_box(0.085, 0.075, 0.08, 0.04), dark, el * T(0, -0.21, 0.01))
            for k in range(4):
                sc.add(sd_box(0.012, 0.03, 0.02, 0.008), plate, el * T(-0.06 + 0.04 * k, -0.25, 0.08))
    # head: a big helmet, brow ridge, one red slit, jaw grill, swept fins
    head = torso * T(0, 0.27, 0.02) * RY(head_yaw) * RX(head_pitch)
    sc.add(sd_box(0.2, 0.15, 0.19, 0.11), plate, head * T(0, 0.09, 0))
    sc.add(sd_box(0.21, 0.035, 0.07, 0.03), dark, head * T(0, 0.13, 0.15) * RX(0.35))
    sc.add(sd_box(0.17, 0.04, 0.04, 0.03), black, head * T(0, 0.065, 0.165))
    sc.add(sd_capsule((-0.13, 0, 0), (0.13, 0, 0), 0.02), glow(red, visor), head * T(0, 0.065, 0.2))
    sc.add(sd_box(0.12, 0.06, 0.08, 0.04), dark, head * T(0, -0.04, 0.1))
    for k in range(5):
        sc.add(sd_box(0.008, 0.035, 0.01, 0.004), black, head * T(-0.06 + 0.03 * k, -0.045, 0.18))
    for sgn in (-1, 1):
        sc.add(sd_box(0.03, 0.09, 0.03, 0.02), plate, head * T(0.17 * sgn, 0.23, -0.06) * RX(-0.7) * RZ(0.3 * sgn))
        sc.add(sd_cylinder(0.035, 0.05, 0.015), dark, head * T(0.2 * sgn, 0.06, -0.01) * RZ(math.pi / 2))
    for (x, y, z, r) in sparks:
        if r > 0.002:
            sc.add(sd_sphere(r), glow("#ffd27a", 14.0), body * T(x, y0 + y, z))
    return sc


# ─── the human: an expressive face (a bust) ─────────────────────────────────
# A portrait of the user, stylised (photo, 2026-11-11): very short buzzed hair
# with a high, receding hairline, a high forehead with faint lines, a marked
# brow, light grey-blue eyes, a longer face with a clear jaw, a salt-and-pepper
# stubble beard and moustache, ears that show; a crew-neck T-shirt, black for
# Black and white for White.
SKIN = "#e4c6b4"


def smoothstep(a, b, x):
    k = ((x - a) / (b - a)).clamp(0, 1)
    return k * k * (3 - 2 * k)


def hairline(q):
    """Height of the hairline over the head-local point q (high at the front,
    higher at the temples, low at the back)."""
    front = smoothstep(-0.05, 0.22, q[:, 2])
    temple = torch.exp(-((q[:, 0].abs() - 0.11) / 0.05) ** 2)          # the receding corners (M shape)
    return 0.2 * front + 0.1 * (q[:, 0] / 0.2) ** 2 * front + 0.06 * temple * front - 0.13 * (1 - front)


# The player's own face (ref/human_face.jpg, a drawing from their photo),
# projected from the front onto the head: wrinkles, brows, stubble, hairline.
# Head-local (x, y) → photo pixels: the pupils (±0.085, 0.032) on the drawing's
# pupils, 1288 px per unit.
_FACE = None


def face_detail(q):
    """Luminance of the drawing at head-local points q (1 = paper), and the
    weight of the projection (front of the face only)."""
    global _FACE
    if _FACE is None:
        import os
        from PIL import Image
        import numpy as np
        p = os.path.join(os.path.dirname(__file__), "ref", "human_face.jpg")
        im = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        _FACE = torch.tensor(im, device=q.device)[None, None]
    H, W = _FACE.shape[-2:]
    px = 666.5 + q[:, 0] * 1288.0
    py = 370.5 - (q[:, 1] - 0.032) * 1288.0
    g = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1)[None, None]
    L = torch.nn.functional.grid_sample(_FACE, g, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]
    w = smoothstep(0.02, 0.12, q[:, 2]) * smoothstep(0.17, 0.13, q[:, 0].abs()) * smoothstep(0.22, 0.17, q[:, 1])
    # not over the 3D eyes and lips (they are modelled and animated)
    for sgn in (-1, 1):
        e = ((q[:, 0] - 0.085 * sgn) / 0.05) ** 2 + ((q[:, 1] - 0.032) / 0.03) ** 2
        w = w * smoothstep(0.7, 1.2, e)
    m = (q[:, 0] / 0.07) ** 2 + ((q[:, 1] + 0.16) / 0.018) ** 2
    w = w * (0.4 + 0.6 * smoothstep(0.6, 1.3, m))
    return L, w


def skin_mat(head=None):
    """Skin; with `head` (the head's transform): the stubble beard and moustache,
    and faint lines on the forehead, placed in the head's coordinates."""
    base_c = srgb(SKIN)

    def pat(pl, pw, n):
        q = head.to_local(pw) if head is not None else pl
        cheek = torch.exp(-(((q[:, 0].abs() - 0.14) / 0.06) ** 2 + ((q[:, 1] + 0.03) / 0.05) ** 2)) * (q[:, 2] > 0).float()
        mott = fbm(pl * 30.0, 3)
        a = base_c * (0.94 + 0.12 * mott)[:, None]
        a = a * (1 - 0.12 * cheek[:, None]) + srgb("#d9887a") * 0.12 * cheek[:, None]
        rough = 0.42 + 0.1 * mott
        if head is not None:
            ax = q[:, 0].abs()
            bline = -0.09 - 0.3 * (0.17 - ax).clamp_min(0)            # the beard's upper edge on the cheeks
            beard = smoothstep(0.012, -0.012, q[:, 1] - bline) * (q[:, 2] > -0.06).float()
            stache = smoothstep(-0.095, -0.108, q[:, 1]) * smoothstep(0.08, 0.06, ax) * (q[:, 2] > 0.2).float()
            beard = torch.maximum(beard, stache)
            side = smoothstep(0.16, 0.2, ax) * (q[:, 1] < 0.06).float() * (q[:, 2] > -0.07).float() * (q[:, 2] < 0.12).float()
            beard = torch.maximum(beard, side * smoothstep(0.02, -0.03, q[:, 1]) * 0.7)
            beard = beard * smoothstep(-0.31, -0.26, q[:, 1])             # fades under the jaw
            speck = fbm(q * 180.0, 2)
            grey = (fbm(q * 220.0 + 7.0, 2) > 0.68).float()
            stubble = srgb("#5a4e46") * (1 - 0.65 * grey[:, None]) + srgb("#b4ada6") * 0.65 * grey[:, None]
            w = (beard * (0.45 + 0.3 * speck))[:, None]
            a = a * (1 - w) + stubble * w
            rough = rough + 0.3 * beard
            fore = smoothstep(0.11, 0.14, q[:, 1]) * smoothstep(0.27, 0.22, q[:, 1]) * (q[:, 2] > 0.1).float()
            lines = (torch.sin(q[:, 1] * 160 + 1.3 + 3 * q[:, 0] ** 2 * 40) > 0.88).float() * fore * smoothstep(0.13, 0.05, q[:, 0].abs())
            a = a * (1 - 0.32 * lines[:, None])
            # crow's feet at the outer corners of the eyes
            crow = smoothstep(0.13, 0.15, q[:, 0].abs()) * smoothstep(0.19, 0.17, q[:, 0].abs()) * ((q[:, 1] - 0.03).abs() < 0.035).float()
            a = a * (1 - 0.15 * crow[:, None] * (torch.sin(q[:, 1] * 300) > 0.6).float()[:, None])
            L, wf = face_detail(q)
            ink = ((1 - L) * 1.5).clamp(0, 1) ** 0.85
            # the drawing's lines and hatching: darker, a touch cooler (stubble), on the skin
            tone = a * (1 - 0.85 * ink[:, None]) + srgb("#3a322e") * 0.25 * ink[:, None]
            a = a * (1 - wf[:, None]) + tone * wf[:, None]
        return {"albedo": a, "rough": rough}
    return Mat(SKIN, rough=0.45, sss=0.55, pattern=pat)


def eye_mat(gaze, iris="#3f7fa0", ri=0.6, rp=0.24):
    """An eyeball looking along `gaze` (local): sclera, iris with fibres, pupil."""
    g = torch.nn.functional.normalize(vec(*gaze), dim=0)
    ic = srgb(iris)

    def pat(pl, pw, n):
        d = torch.nn.functional.normalize(pl, dim=-1)
        c = (d @ g).clamp(-1, 1)
        ang = torch.acos(c)                       # 0 at the centre of the iris
        # ri, rp: iris and pupil radii (radians)
        # angle around the gaze, for the fibres
        ref = torch.nn.functional.normalize(torch.linalg.cross(g.expand_as(d), vec(0, 1, 0).expand_as(d), dim=-1) + 1e-6, dim=-1)
        fib = torch.sin(torch.atan2((torch.linalg.cross(ref, d, dim=-1) * g).sum(-1), (ref * d).sum(-1)) * 40) * 0.5 + 0.5
        sclera = srgb("#f4efe9") * (1 - 0.25 * (ang / 1.6).clamp(0, 1) ** 3)[:, None]
        k = (ang / ri).clamp(0, 1)
        irisc = ic * (0.55 + 0.6 * (1 - k) + 0.25 * fib * k)[:, None]
        irisc = irisc * (1 - 0.7 * ((k > 0.85).float() * (k - 0.85) / 0.15))[:, None]     # limbal ring
        a = torch.where((ang < ri)[:, None], irisc, sclera)
        a = torch.where((ang < rp)[:, None], srgb("#050505").expand_as(a), a)
        # catchlights: the key light and a small second one, as painted in films
        hk = torch.nn.functional.normalize(KEY + vec(0, 0.53, 0.85), dim=0)
        h2 = torch.nn.functional.normalize(vec(0.5, 0.2, 1.0) + vec(0, 0.53, 0.85), dim=0)
        spot = ((n @ hk) > 0.975).float() + 0.6 * ((n @ h2) > 0.99).float()
        em = vec(1, 1, 1) * (spot.clamp(0, 1) * 2.5)[:, None]
        return {"albedo": a, "rough": torch.full_like(ang, 0.06), "emissive": em}
    return Mat("#ffffff", rough=0.05, clearcoat=1.5, pattern=pat)


def sd_lid(r, th, open_):
    """An eyelid: a spherical shell of radius r around the eye, kept above a plane
    through the eye's centre tilted by open_ (radians: 0 = closed at the middle,
    + opens upward)."""
    def f(p):
        shell = (length(p) - (r + th)).abs() - th
        n_y, n_z = math.cos(open_), math.sin(open_)
        plane = -(p[:, 1] * n_y - p[:, 2] * n_z)     # keeps the side above the plane
        return torch.maximum(shell, plane)
    return f


def sd_lower_lid(r, th, open_):
    def f(p):
        shell = (length(p) - (r + th)).abs() - th
        n_y, n_z = math.cos(open_), math.sin(open_)
        plane = (p[:, 1] * n_y - p[:, 2] * n_z)     # keeps the side below the plane
        return torch.maximum(shell, plane)
    return f


def lid_mat(edge, upper=True):
    """skin, with a dark lash line along the lid's edge (eye-local coordinates)."""
    sk = skin_mat()

    def pat(pl, pw, n):
        o = sk.pattern(pl, pw, n)
        el = torch.atan2(pl[:, 1], pl[:, 2])
        k = ((el - edge).abs() < (0.07 if upper else 0.05)).float() * (pl[:, 2] > 0).float() * (0.85 if upper else 0.4)
        dark = srgb("#3a2820") if upper else srgb("#9a6a58")
        o["albedo"] = o["albedo"] * (1 - k[:, None]) + dark * k[:, None]
        return o
    return Mat(SKIN, rough=0.45, sss=0.4, pattern=pat)


def human(anim, t, side, outfit="tee"):
    """outfit: "tee" (a crew-neck T-shirt) or "legendes" (a tunic, a cloak
    held by a brooch, a leather strap: the human in the Légendes set)."""
    sc = Scene()
    base(sc, side)
    tee_c = "#1b1c20" if side == 0 else "#ebe9e4"
    tee = Mat(tee_c, rough=0.75, pattern=lambda pl, pw, n: {"albedo": srgb(tee_c) * (0.88 + 0.2 * fbm(pl * 60, 2))[:, None]})
    rib = Mat(tee_c, rough=0.8, pattern=lambda pl, pw, n: {"albedo": srgb(tee_c) * (0.8 + 0.25 * (torch.sin(torch.atan2(pl[:, 2], pl[:, 0]) * 70) > 0).float())[:, None]})
    lip = Mat("#b48676", rough=0.5, sss=0.5, clearcoat=0.15)
    mouth_in = Mat("#4a1418", rough=0.6)
    teeth = Mat("#f3efe6", rough=0.25, clearcoat=0.6)
    brow = Mat("#5a4c42", rough=0.8)
    # expression parameters
    yaw, pitch, roll, bob = 0.0, 0.0, 0.0, 0.0
    gaze = (0.1, 0.0, 1.0)
    lid_up, lid_low = 0.5, -0.55        # opening angles: open, almond-shaped
    brow_in, brow_out = 0.0, 0.0        # heights of the inner / outer ends of the brows
    smile, open_m = 0.1, 0.0
    sweat = None
    happy_eyes = False
    if anim == "idle":
        bob = 0.008 * wave(t)
        yaw = 0.07 * wave(t, 1, 0.3)
        roll = 0.025 * wave(t, 1)
        gx = 0.35 * math.sin(TAU * t)
        gaze = (0.1 + gx, 0.05, 1.0)
        b = max(0.0, 1 - abs(t - 0.5) / 0.04)        # one blink
        lid_up = 0.5 - 1.0 * b
        lid_low = -0.55 + 0.3 * b
    elif anim == "move":
        pitch = 0.12 * math.sin(math.pi * t)
        yaw = 0.25
        gaze = (0.6, -0.05, 1.0)
        brow_in, brow_out = -0.02, 0.0
        lid_up = 0.3
        smile = 0.05
        bob = 0.03 * math.sin(math.pi * t)
    elif anim == "blocked":
        yaw = 0.18 * math.sin(TAU * 2 * t)
        pitch = 0.1
        gaze = (0.0, -0.35, 1.0)
        brow_in, brow_out = 0.035, -0.015
        lid_up = 0.4
        smile = -0.55
        sweat = (t % 1.0)
    elif anim == "win":
        bob = 0.04 * abs(math.sin(TAU * t))
        roll = 0.08 * wave(t)
        pitch = -0.1
        happy_eyes = True
        brow_in, brow_out = 0.03, 0.03
        smile, open_m = 0.8, 0.85
    y0 = BASE_TOP
    root = RY(HEADING * 0.3) * T(0, bob, 0)
    # bust: shoulders and chest in a T-shirt, a ribbed crew neck, the neck
    if outfit == "legendes":
        tunic_c, cloak_c = ("#4a1c1e", "#18161a") if side == 0 else ("#3a5a34", "#e4dccb")
        tunic = Mat(tunic_c, rough=0.85, pattern=lambda pl, pw, n: {"albedo": srgb(tunic_c) * (0.8 + 0.3 * fbm(pl * 70, 2))[:, None]})
        cloak = Mat(cloak_c, rough=0.9, pattern=lambda pl, pw, n: {"albedo": srgb(cloak_c) * (0.82 + 0.25 * fbm(pl * 40, 3))[:, None]})
        sc.add(sd_ellipsoid(0.35, 0.2, 0.21), tunic, root * T(0, y0 + 0.16, 0))
        sc.add(sd_ellipsoid(0.27, 0.17, 0.17), tunic, root * T(0, y0 + 0.28, 0), blend=0.06)
        # the cloak draped on the shoulders and down the back, its collar, a brooch, a strap
        for sgn in (-1, 1):
            sc.add(sd_ellipsoid(0.15, 0.085, 0.17), cloak, root * T(0.2 * sgn, y0 + 0.3, -0.03) * RZ(-0.35 * sgn), blend=0.02)
        sc.add(sd_ellipsoid(0.33, 0.22, 0.09), cloak, root * T(0, y0 + 0.2, -0.16), blend=0.03)
        sc.add(sd_torus(0.11, 0.032), cloak, root * T(0, y0 + 0.41, -0.015) * RX(0.25))
        sc.add(sd_cylinder(0.008, 0.032, 0.004), gold(), root * T(0.12, y0 + 0.37, 0.12) * RX(1.1))
        sc.add(sd_sphere(0.013), glow("#ff4a3a" if side == 0 else "#4ad0ff", 3.0), root * T(0.12, y0 + 0.375, 0.128))
        sc.add(sd_box(0.022, 0.19, 0.012, 0.008), Mat("#4a301e", rough=0.6), root * T(-0.03, y0 + 0.24, 0.18) * RX(-0.35) * RZ(0.55))
    else:
        sc.add(sd_ellipsoid(0.37, 0.2, 0.22), tee, root * T(0, y0 + 0.16, 0))
        sc.add(sd_ellipsoid(0.28, 0.17, 0.18), tee, root * T(0, y0 + 0.28, 0), blend=0.06)
        sc.add(sd_torus(0.098, 0.022), rib, root * T(0, y0 + 0.415, 0.0) * RX(0.18))
    sc.add(sd_capsule((0, 0, 0), (0, 0.15, 0.01), 0.083), skin_mat(), root * T(0, y0 + 0.38, 0))
    head = root * T(0, y0 + 0.75, 0.02) * RY(yaw) * RX(pitch - 0.42) * RZ(roll)   # faces the camera
    skin = skin_mat(head)
    # head: cranium, jaw, chin, brow ridge, cheekbones, nose, ears
    sc.add(sd_ellipsoid(0.226, 0.285, 0.254), skin, head * T(0, 0.012, 0))
    sc.add(sd_ellipsoid(0.145, 0.2, 0.17), skin, head * T(0, -0.13, 0.075), blend=0.07)
    for sgn in (-1, 1):   # the jaw's angles, firm
        sc.add(sd_ellipsoid(0.04, 0.05, 0.06), skin, head * T(0.105 * sgn, -0.225, 0.07), blend=0.05)
    sc.add(sd_ellipsoid(0.065, 0.055, 0.06), skin, head * T(0, -0.305, 0.165), blend=0.05)       # the chin
    sc.add(sd_ellipsoid(0.16, 0.026, 0.05), skin, head * T(0, 0.085, 0.205), blend=0.04)          # brow ridge
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.05, 0.03, 0.04), skin, head * T(0.12 * sgn, -0.035, 0.19), blend=0.04)   # high cheekbones
        ear = head * T(0.228 * sgn, -0.0, -0.02) * RY(0.85 * sgn)                    # ears standing out
        sc.add(sd_ellipsoid(0.052, 0.088, 0.024), skin, ear, blend=0.012)
        sc.add(sd_ellipsoid(0.03, 0.058, 0.018), skin, ear * T(0, 0, 0.02), op="sub", blend=0.01)
    # the nose: long, straight, narrow
    sc.add(sd_capsule((0, 0.075, 0.0), (0, -0.06, 0.04), 0.018, 0.026), skin, head * T(0, -0.01, 0.245), blend=0.02)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.014), skin, head * T(0.024 * sgn, -0.07, 0.265), blend=0.012)
    # eye sockets and eyes: smaller, deep-set under the brow ridge
    er = 0.042
    for sgn in (-1, 1):
        ec = head * T(0.085 * sgn, 0.032, 0.2)
        sc.add(sd_sphere(er + 0.013), skin, ec * T(0, 0.004, 0.004), op="sub", blend=0.022)

        if happy_eyes:   # closed, curved happily
            sc.add(sd_lid(er, 0.006, -0.1), skin, ec, blend=0.01)
            sc.add(sd_lower_lid(er, 0.006, 0.1), skin, ec, blend=0.01)
            sc.add(sd_capsule((-0.042, -0.005, 0), (0, 0.011, 0), 0.007), Mat("#2c1a12", rough=0.6), ec * T(0, 0, er + 0.01))
            sc.add(sd_capsule((0, 0.011, 0), (0.042, -0.005, 0), 0.007), Mat("#2c1a12", rough=0.6), ec * T(0, 0, er + 0.01))
        else:
            sc.add(sd_sphere(er), eye_mat(gaze, iris="#6c7c86", ri=0.48, rp=0.2), ec)
            sc.add(sd_lid(er, 0.005, lid_up), lid_mat(lid_up), ec, blend=0.005)
            sc.add(sd_lower_lid(er, 0.006, lid_low), lid_mat(lid_low, False), ec, blend=0.006)
        # brows: straight, under the brow ridge
        bi, bo = brow_in, brow_out
        sc.add(sd_capsule((0.0, 0.0 + bi, 0.0), (0.05 * sgn, 0.008 + (bi + bo) / 2, -0.012), 0.0055, 0.0055), brow,
               head * T(0.03 * sgn, 0.084, 0.262))
        sc.add(sd_capsule((0.05 * sgn, 0.008 + (bi + bo) / 2, -0.012), (0.088 * sgn, -0.008 + bo, -0.042), 0.0055, 0.0035), brow,
               head * T(0.03 * sgn, 0.084, 0.262))
    # mouth: lips along a curve (smile > 0: corners up), the mouth open by open_m
    mw = 0.064
    pts = []
    for k in range(7):
        x = -mw + 2 * mw * k / 6
        y = smile * 0.06 * (x / mw) ** 2 - smile * 0.015
        pts.append((x, y))
    mouth = head * T(0, -0.16, 0.25)
    if open_m > 0.05:
        sc.add(sd_ellipsoid(mw * 0.95, 0.025 + 0.05 * open_m, 0.06), mouth_in, mouth * T(0, -0.015 * open_m + smile * 0.01, -0.01))
        sc.add(sd_ellipsoid(mw * 1.05, 0.028 + 0.05 * open_m, 0.07), skin, mouth * T(0, -0.015 * open_m + smile * 0.01, 0.0), op="sub", blend=0.012)
        if smile > 0.3:
            sc.add(sd_box(mw * 0.72, 0.014, 0.02, 0.008), teeth, mouth * T(0, 0.016 + smile * 0.008, 0.012))
            sc.add(sd_ellipsoid(mw * 0.6, 0.02, 0.04), Mat("#c8565a", rough=0.4, sss=0.4), mouth * T(0, -0.03 * open_m, -0.005))
    gap = 0.003 + 0.045 * open_m
    for (x0, y0_), (x1, y1) in zip(pts[:-1], pts[1:]):
        z0, z1 = -0.022 * (x0 / mw) ** 2, -0.022 * (x1 / mw) ** 2
        w0, w1 = 1 - 0.55 * (x0 / mw) ** 2, 1 - 0.55 * (x1 / mw) ** 2       # thinner at the corners
        sc.add(sd_capsule((x0, y0_ + gap, z0), (x1, y1 + gap, z1), 0.0045 * w0, 0.0045 * w1), lip, mouth, blend=0.006)
        sc.add(sd_capsule((x0, y0_ - gap - 0.004, z0), (x1, y1 - gap - 0.004, z1), 0.0068 * w0, 0.0068 * w1), lip, mouth, blend=0.006)
    if open_m <= 0.05:
        for (x0, y0_), (x1, y1) in zip(pts[:-1], pts[1:]):
            sc.add(sd_capsule((x0, y0_, 0.0), (x1, y1, 0.0), 0.003), mouth_in, mouth * T(0, 0, 0.004))
    # buzzed hair: a thin shell over the skull, above the hairline, a stubble texture
    def buzz_pat(pl, pw, n):
        q = head.to_local(pw)
        edge = smoothstep(0.0, 0.035, q[:, 1] - hairline(q))          # thinner near the hairline
        speck = fbm(q * 200.0, 2)
        grey = (fbm(q * 240.0 + 3.0, 2) > 0.6).float()
        hc = srgb("#040302") * (1 - 0.35 * grey[:, None]) + srgb("#2c2926") * 0.35 * grey[:, None]
        crown = torch.exp(-((q[:, 2] - 0.05) / 0.12) ** 2 - (q[:, 0] / 0.1) ** 2) * (q[:, 1] > 0.2).float()   # thinner on top
        k = (edge * (0.94 + 0.05 * speck) * (1 - 0.08 * crown))[:, None]
        a = srgb(SKIN) * 0.9 * (1 - k) + hc * k
        L, wf = face_detail(q)
        drawn = srgb(SKIN) * 0.85 * L[:, None] ** 1.6 + hc * (1 - L[:, None])
        a = a * (1 - wf[:, None]) + drawn * wf[:, None]
        return {"albedo": a, "rough": 0.7 + 0.25 * edge}

    def buzz(p):
        shell = sd_ellipsoid(0.236, 0.295, 0.264)(p - vec(0, 0.012, 0))
        return torch.maximum(shell, -(p[:, 1] - hairline(p)) * 0.5)
    sc.add(buzz, Mat("#3d342e", rough=0.8, pattern=buzz_pat), head)
    if sweat is not None:
        ph = sweat
        sc.add(sd_ellipsoid(0.018, 0.028, 0.018), Mat("#9ad8ff", rough=0.05, clearcoat=2.0, metal=0.0),
               head * T(0.23, 0.12 - 0.16 * ph, 0.12))
    return sc


# ─── shared bits ────────────────────────────────────────────────────────────
def paint(hexcol, rough=0.35, metal=0.0, clearcoat=0.8, noise=0.08):
    c = srgb(hexcol)
    return Mat(hexcol, metal=metal, rough=rough, clearcoat=clearcoat,
               pattern=lambda pl, pw, n: {"albedo": c * (1 - noise + 2 * noise * fbm(pl * 18.0, 2))[:, None]})


def chrome(rough=0.12):
    return Mat("#d8dde4", metal=1.0, rough=rough)


def rivets(sc, mat, xf, pts, r=0.012):
    for q in pts:
        sc.add(sd_sphere(r), mat, xf * T(*q))


def screen_eyes(sc, xf, col, k, w=0.035, h=0.045, dx=0.06, look=0.0, closed=False, happy=False):
    """Two glowing eyes on a face (xf: the face's centre, z out of the face)."""
    for sgn in (-1, 1):
        e = xf * T(dx * sgn + look, 0, 0)
        if happy:
            sc.add(sd_capsule((-w, -0.008, 0), (0, 0.012, 0), 0.008), glow(col, k), e)
            sc.add(sd_capsule((0, 0.012, 0), (w, -0.008, 0), 0.008), glow(col, k), e)
        elif closed:
            sc.add(sd_capsule((-w, 0, 0), (w, 0, 0), 0.007), glow(col, k), e)
        else:
            sc.add(sd_ellipsoid(w, h, 0.012), glow(col, k), e)


# ─── Hasard: a wind-up tin toy ──────────────────────────────────────────────
def tin_toy(anim, t, side):
    sc = Scene()
    base(sc, side)
    tin = paint("#c63b2b", rough=0.3, metal=0.6, clearcoat=1.0)
    cream = paint("#efe1bd", rough=0.35, metal=0.3)
    silver = chrome(0.2)
    key_a = TAU * t * (2 if anim != "blocked" else 0.5)
    wob = 0.06 * wave(t, 2)
    hop = 0.0
    spin = 0.0
    arm = 0.3
    spring = 0.0
    eye_spin = TAU * t
    if anim == "move":
        hop = 0.07 * abs(math.sin(TAU * t))
        wob = 0.15 * wave(t, 2)
        arm = 0.6 * wave(t, 2)
    elif anim == "blocked":
        wob = 0.03 * wave(t, 6)
        spring = 0.12 + 0.05 * wave(t, 3)
        eye_spin = TAU * t * 4
    elif anim == "win":
        spin = TAU * t
        hop = 0.05 * abs(math.sin(TAU * 2 * t))
        arm = 2.4
    body = RY(HEADING + spin) * T(0, hop, 0) * RZ(wob)
    y0 = BASE_TOP
    # little feet
    for sgn in (-1, 1):
        sc.add(sd_box(0.07, 0.035, 0.1, 0.03), silver, body * T(0.1 * sgn, y0 + 0.04, 0.02))
        sc.add(sd_cylinder(0.04, 0.035, 0.01), silver, body * T(0.1 * sgn, y0 + 0.1, 0))
    # body: a tin can, cream band, rivets
    tb = body * T(0, y0 + 0.33, 0)
    sc.add(sd_cylinder(0.17, 0.2, 0.04), tin, tb)
    sc.add(sd_cylinder(0.045, 0.205, 0.01), cream, tb * T(0, -0.02, 0))
    for k in range(10):
        a = TAU * k / 10
        sc.add(sd_sphere(0.012), silver, tb * T(0.205 * math.cos(a), 0.11, 0.205 * math.sin(a)))
    # a dial on the chest
    sc.add(sd_cylinder(0.012, 0.06, 0.006), cream, tb * T(0, 0.06, 0.2) * RX(math.pi / 2))
    sc.add(sd_box(0.004, 0.04, 0.004, 0.002), Mat("#222", rough=0.4), tb * T(0, 0.06, 0.215) * RZ(TAU * t * 3))
    # wind-up key on the back
    kb = tb * T(0, 0.02, -0.22) * RX(math.pi / 2) * RY(key_a)
    sc.add(sd_cylinder(0.05, 0.015), silver, kb * T(0, 0.05, 0))
    for sgn in (-1, 1):
        sc.add(sd_torus(0.045, 0.015), silver, kb * T(0.06 * sgn, 0.12, 0) * RZ(math.pi / 2))
    # arms: pipes with pincers
    for sgn in (-1, 1):
        sh = tb * T(0.21 * sgn, 0.06, 0) * RX(-arm * (1 if sgn > 0 else (0.6 if anim != "move" else -1))) * RZ(0.4 * sgn)
        sc.add(sd_capsule((0, 0, 0), (0, -0.16, 0.02), 0.03), silver, sh)
        sc.add(sd_torus(0.04, 0.014), tin, sh * T(0, -0.19, 0.02) * RZ(math.pi / 2))
    # head: a dome with big glass eyes
    hd = tb * T(0, 0.24, 0)
    sc.add(sd_sphere(0.16), cream, hd * T(0, 0.04, 0), blend=0.0)
    sc.add(sd_cylinder(0.03, 0.17, 0.02), tin, hd * T(0, -0.04, 0))
    for sgn in (-1, 1):
        e = hd * T(0.075 * sgn, 0.07, 0.12)
        sc.add(sd_cylinder(0.02, 0.065, 0.01), silver, e * RX(math.pi / 2))
        # a spiral pupil under a glass dome
        sp = Mat("#f4f0e6", rough=0.3, pattern=lambda pl, pw, n, ph=eye_spin * sgn: {
            "albedo": srgb("#f4f0e6") * (1 - 0.95 * (torch.sin(length(pl[:, :2]) * 160 - torch.atan2(pl[:, 1], pl[:, 0]) + ph) > 0.3).float()
                                         * (length(pl[:, :2]) < 0.05).float())[:, None]})
        sc.add(sd_cylinder(0.006, 0.055, 0.003), sp, e * T(0, 0, 0.022) * RX(math.pi / 2))
        sc.add(sd_sphere(0.058), Mat("#ffffff", glass=True), e * T(0, 0, 0.0))
    # antenna or the spring popping out
    sc.add(sd_capsule((0, 0, 0), (0, 0.1 + spring, 0), 0.01), silver, hd * T(0, 0.19, 0))
    sc.add(sd_sphere(0.03), glow("#ffcf3a", 4.0 + 3.0 * (0.5 + 0.5 * wave(t, 2))), hd * T(0, 0.3 + spring, 0))
    if spring > 0:
        for k in range(6):
            sc.add(sd_torus(0.025, 0.006), silver, hd * T(0, 0.2 + spring * k / 6, 0))
    return sc


# ─── Débutant: a little robot on a wheel ────────────────────────────────────
def roller(anim, t, side):
    sc = Scene()
    base(sc, side)
    shell = paint("#f3f5f2", rough=0.25, clearcoat=1.0)
    mint = paint("#5cc79a", rough=0.3, clearcoat=1.0)
    dark = Mat("#1e2226", rough=0.5)
    lean = 0.06 * wave(t)
    roll = 0.0
    hop = 0.0
    blink = 1.0 if (0.48 < t < 0.53) else 0.0
    face = "open"
    bulb = 3.0 + 4.0 * (1.0 if wave(t, 2) > 0 else 0.0)
    arms = 0.2
    if anim == "move":
        lean = 0.35 * math.sin(math.pi * t)
        roll = TAU * t * 1.5
    elif anim == "blocked":
        lean = 0.15 * wave(t, 3)
        face = "x"
        bulb = 2.0 + 8.0 * (1.0 if int(t * 12) % 2 else 0.0)
        arms = 1.2 + 0.3 * wave(t, 3)
    elif anim == "win":
        hop = 0.1 * abs(math.sin(TAU * t))
        face = "happy"
        bulb = 10.0
        arms = 2.5 + 0.3 * wave(t, 2)
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)
    wheel = root * T(0, y0 + 0.09, 0) * RZ(math.pi / 2) * RY(roll)
    sc.add(sd_torus(0.065, 0.03), dark, wheel)
    sc.add(sd_cylinder(0.025, 0.06, 0.01), mint, wheel)
    body = root * T(0, y0 + 0.09, 0) * RX(lean) * T(0, 0.27, 0)
    sc.add(sd_capsule((0, -0.25, 0), (0, -0.12, 0), 0.035), dark, body)
    sc.add(sd_ellipsoid(0.24, 0.27, 0.23), shell, body)
    sc.add(sd_ellipsoid(0.245, 0.06, 0.235), mint, body * T(0, -0.08, 0))
    # face screen
    sc.add(sd_ellipsoid(0.17, 0.12, 0.05), Mat("#0d1b1e", rough=0.08, clearcoat=1.5), body * T(0, 0.06, 0.2))
    fc = body * T(0, 0.06, 0.25)
    if face == "x":
        for sgn in (-1, 1):
            for d in (-1, 1):
                sc.add(sd_capsule((-0.025, -0.025 * d, 0), (0.025, 0.025 * d, 0), 0.008), glow("#ff5a5a", 5), fc * T(0.06 * sgn, 0, 0))
    else:
        screen_eyes(sc, fc, "#5ff3c4", 6.0, w=0.03, h=0.045 * (1 - blink) + 0.006, dx=0.06, happy=(face == "happy"))
    # arms
    for sgn in (-1, 1):
        sh = body * T(0.24 * sgn, -0.02, 0) * RZ(arms * sgn) 
        sc.add(sd_capsule((0, 0, 0), (0.12 * sgn, -0.05, 0), 0.025), mint, sh)
        sc.add(sd_sphere(0.04), shell, sh * T(0.14 * sgn, -0.06, 0))
    # antenna with a bulb
    sc.add(sd_capsule((0, 0, 0), (0.03, 0.14, 0), 0.01), dark, body * T(0, 0.26, 0))
    sc.add(sd_sphere(0.035), glow("#ff7a59", bulb), body * T(0.03, 0.42, 0))
    return sc


# ─── Monte Carlo: a robot with a die for a head ─────────────────────────────
def dicer(anim, t, side):
    sc = Scene()
    base(sc, side)
    felt = paint("#1f6f4e", rough=0.6, clearcoat=0.2)
    gold_ = gold()
    die = Mat("#f7f3ea", rough=0.18, clearcoat=1.2)
    tilt = 0.12 * wave(t)
    roll = 0.0
    hop = 0.0
    pip = "#30d0ff"
    pk = 4.0 + 2.0 * (0.5 + 0.5 * wave(t, 2))
    arm = 0.2
    if anim == "move":
        roll = math.pi / 2 * (t * t * (3 - 2 * t))      # the head rolls a quarter turn
        hop = 0.08 * math.sin(math.pi * t)
    elif anim == "blocked":
        tilt = 0.25 * wave(t, 3)
        pip = "#ff4040"
        pk = 2.0 + 7.0 * (1.0 if int(t * 12) % 2 else 0.0)
        arm = 1.0
    elif anim == "win":
        roll = TAU * t * 2
        hop = 0.08 * abs(math.sin(TAU * 2 * t))
        pip = "#ffd84a"
        pk = 9.0
        arm = 2.5
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)
    for sgn in (-1, 1):
        sc.add(sd_capsule((0, 0, 0), (0, 0.12, 0), 0.04), gold_, root * T(0.08 * sgn, y0 + 0.04, 0))
        sc.add(sd_box(0.06, 0.03, 0.09, 0.025), felt, root * T(0.08 * sgn, y0 + 0.03, 0.02))
    body = root * T(0, y0 + 0.3, 0)
    sc.add(sd_box(0.16, 0.13, 0.13, 0.06), felt, body)
    sc.add(sd_box(0.165, 0.02, 0.135, 0.015), gold_, body * T(0, 0.07, 0))
    sc.add(sd_cylinder(0.01, 0.05, 0.005), gold_, body * T(0, -0.01, 0.13) * RX(math.pi / 2))   # a chip
    for sgn in (-1, 1):
        sh = body * T(0.18 * sgn, 0.05, 0) * RZ(arm * sgn * 0.6) * RX(-0.3)
        sc.add(sd_capsule((0, 0, 0), (0.0, -0.15, 0.03), 0.032), gold_, sh)
        sc.add(sd_sphere(0.045), felt, sh * T(0, -0.17, 0.03))
    # the die: rounded cube, pips on every face; the front face's pips glow (eyes)
    hd = body * T(0, 0.31, 0) * RZ(tilt) * RX(-roll)
    sc.add(sd_box(0.17, 0.17, 0.17, 0.05), die, hd)
    faces = {  # normal → pips (u, v) in the face
        (0, 0, 1): [(-0.07, 0.04), (0.07, 0.04)],
        (0, 1, 0): [(-0.08, -0.08), (0, 0), (0.08, 0.08)],
        (1, 0, 0): [(-0.08, -0.08), (0.08, 0.08), (-0.08, 0.08), (0.08, -0.08)],
        (-1, 0, 0): [(0, 0)],
        (0, 0, -1): [(-0.08, -0.08), (0.08, 0.08), (-0.08, 0.08), (0.08, -0.08), (0, 0)],
        (0, -1, 0): [(-0.08, -0.08), (-0.08, 0), (-0.08, 0.08), (0.08, -0.08), (0.08, 0), (0.08, 0.08)],
    }
    for nrm_, pips in faces.items():
        nx, ny, nz = nrm_
        for (u, v) in pips:
            if nz:
                q = (u, v, 0.17 * nz)
            elif ny:
                q = (u, 0.17 * ny, v)
            else:
                q = (0.17 * nx, v, u)
            front = nrm_ == (0, 0, 1)
            sc.add(sd_sphere(0.032 if front else 0.026), glow(pip, pk) if front else Mat("#15161a", rough=0.3), hd * T(*q))
    return sc


# ─── MCTS: a robot whose antennas are a search tree ─────────────────────────
def tree_bot(anim, t, side):
    sc = Scene()
    base(sc, side)
    bronze = Mat("#b07a45", metal=1.0, rough=0.3,
                 pattern=lambda pl, pw, n: {"albedo": srgb("#b07a45") * (0.8 + 0.35 * fbm(pl * 16))[:, None]})
    teal = paint("#2a8a8a", rough=0.35, metal=0.3)
    dark = Mat("#1d2024", rough=0.5)
    sway = 0.08 * wave(t)
    hop = 0.0
    lit = lambda i: 0.5 + 0.5 * math.sin(TAU * (t * 2 - i * 0.13))
    col = "#7dffb0"
    kmul = 1.0
    if anim == "move":
        sway = 0.3 * math.sin(math.pi * t)
        hop = 0.05 * math.sin(math.pi * t)
    elif anim == "blocked":
        col = "#ff6a4a"
        lit = lambda i: 1.0 if (int(t * 16) + i) % 3 == 0 else 0.1
        sway = 0.05 * wave(t, 4)
    elif anim == "win":
        lit = lambda i: 1.0
        kmul = 1.8
        hop = 0.05 * abs(math.sin(TAU * t))
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)
    for sgn in (-1, 1):
        sc.add(sd_cylinder(0.05, 0.06, 0.02), bronze, root * T(0.09 * sgn, y0 + 0.05, 0))
    body = root * T(0, y0 + 0.28, 0)
    sc.add(sd_cylinder(0.15, 0.17, 0.06), teal, body)
    sc.add(sd_torus(0.172, 0.018), bronze, body * T(0, 0.1, 0))
    sc.add(sd_torus(0.172, 0.018), bronze, body * T(0, -0.1, 0))
    for sgn in (-1, 1):
        sh = body * T(0.19 * sgn, 0.04, 0) * RZ(0.5 * sgn) * RX(-0.3 + (0.6 * wave(t) if anim == "move" else 0))
        sc.add(sd_capsule((0, 0, 0), (0, -0.14, 0), 0.03), bronze, sh)
        sc.add(sd_sphere(0.04), teal, sh * T(0, -0.16, 0))
    hd = body * T(0, 0.25, 0)
    sc.add(sd_sphere(0.14), bronze, hd)
    sc.add(sd_box(0.11, 0.035, 0.05, 0.03), dark, hd * T(0, 0.0, 0.11))
    screen_eyes(sc, hd * T(0, 0.0, 0.155), col, 6.0, w=0.022, h=0.022, dx=0.05)
    # the tree: trunk, two levels of branches, glowing nodes
    tr = hd * T(0, 0.13, 0) * RZ(sway)
    sc.add(sd_capsule((0, 0, 0), (0, 0.12, 0), 0.016), bronze, tr)
    n = 0
    sc.add(sd_sphere(0.03), glow(col, 6 * kmul * lit(n)), tr * T(0, 0.13, 0)); n += 1
    for a in (-0.7, 0.0, 0.7):
        b1 = tr * T(0, 0.13, 0) * RZ(a) * RX(0.2 * a)
        sc.add(sd_capsule((0, 0, 0), (0, 0.12, 0), 0.011), bronze, b1)
        sc.add(sd_sphere(0.024), glow(col, 6 * kmul * lit(n)), b1 * T(0, 0.12, 0)); n += 1
        for a2 in (-0.45, 0.45):
            b2 = b1 * T(0, 0.12, 0) * RZ(a2) * RX(0.3 * a2)
            sc.add(sd_capsule((0, 0, 0), (0, 0.08, 0), 0.008), bronze, b2)
            sc.add(sd_sphere(0.018), glow(col, 6 * kmul * lit(n)), b2 * T(0, 0.08, 0)); n += 1
    return sc


# ─── Minimax basique: a retro robot of the fifties ──────────────────────────
def retro(anim, t, side):
    sc = Scene()
    base(sc, side)
    steel = chrome(0.16)
    red = paint("#c7322c", rough=0.3, metal=0.5, clearcoat=1.0)
    dark = Mat("#202225", rough=0.5)
    blink = lambda i: 1.0 if (int(t * 8) + i) % 3 else 0.2
    hop = 0.0
    arm = (0.0, 0.0)
    smoke = []
    track = t
    if anim == "move":
        hop = 0.02 * abs(math.sin(TAU * 2 * t))
        arm = (0.5 * wave(t), -0.5 * wave(t))
        track = t * 3
    elif anim == "blocked":
        blink = lambda i: 1.0 if int(t * 12) % 2 else 0.0
        for k in range(3):
            ph = (t + k / 3) % 1
            smoke.append((0.0 + 0.05 * k, 0.75 + 0.25 * ph, 0.04 * ph + 0.03))
    elif anim == "win":
        blink = lambda i: 1.0
        arm = (2.6, 2.6)
        hop = 0.04 * abs(math.sin(TAU * t))
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)
    # tracks
    for sgn in (-1, 1):
        tk = root * T(0.12 * sgn, y0 + 0.07, 0)
        sc.add(sd_box(0.055, 0.06, 0.15, 0.05), dark, tk)
        for k in range(5):
            z = -0.14 + 0.07 * ((k + track * 2) % 5)
            sc.add(sd_box(0.058, 0.008, 0.012, 0.004), steel, tk * T(0, 0.06, z))
    body = root * T(0, y0 + 0.3, 0)
    sc.add(sd_box(0.17, 0.15, 0.13, 0.04), steel, body)
    sc.add(sd_box(0.12, 0.09, 0.02, 0.015), dark, body * T(0, 0.01, 0.13))
    for i, (x, y, c) in enumerate([(-0.07, 0.05, "#ff3b30"), (-0.02, 0.05, "#ffcc00"), (0.03, 0.05, "#34c759"), (0.08, 0.05, "#0a84ff")]):
        sc.add(sd_sphere(0.016), glow(c, 5.0 * blink(i)), body * T(x, y, 0.15))
    for sgn in (-1, 1):   # two dials
        sc.add(sd_cylinder(0.008, 0.03, 0.004), Mat("#f2efe6", rough=0.3), body * T(0.05 * sgn, -0.035, 0.15) * RX(math.pi / 2))
        sc.add(sd_box(0.003, 0.022, 0.003, 0.001), dark, body * T(0.05 * sgn, -0.035, 0.158) * RZ(TAU * t * sgn))
    # accordion arms with claws
    for sgn, a in ((-1, arm[0]), (1, arm[1])):
        sh = body * T(0.18 * sgn, 0.08, 0) * RX(-a) * RZ(0.15 * sgn)
        for k in range(5):
            sc.add(sd_torus(0.028, 0.011), steel, sh * T(0, -0.035 * k, 0))
        sc.add(sd_capsule((0, -0.18, 0), (0.025 * sgn, -0.24, 0.02), 0.014), red, sh)
        sc.add(sd_capsule((0, -0.18, 0), (-0.025 * sgn, -0.24, 0.02), 0.014), red, sh)
    # head: a box, round glass ears, a grille mouth, two antenna balls
    hd = body * T(0, 0.27, 0) * RY(0.12 * wave(t))
    sc.add(sd_box(0.14, 0.11, 0.12, 0.035), steel, hd)
    sc.add(sd_box(0.1, 0.035, 0.02, 0.015), dark, hd * T(0, 0.03, 0.115))
    screen_eyes(sc, hd * T(0, 0.03, 0.13), "#ffd36a", 6.0 * (0.4 if anim == "blocked" and int(t * 12) % 2 else 1), w=0.025, h=0.016, dx=0.045,
                happy=(anim == "win"))
    for k in range(4):
        sc.add(sd_box(0.07, 0.004, 0.01, 0.003), dark, hd * T(0, -0.04 - 0.016 * k, 0.12))
    for sgn in (-1, 1):
        sc.add(sd_cylinder(0.03, 0.05, 0.015), red, hd * T(0.15 * sgn, 0.0, 0) * RZ(math.pi / 2))
        sc.add(sd_capsule((0, 0, 0), (0.02 * sgn, 0.12, 0), 0.008), steel, hd * T(0.07 * sgn, 0.11, 0))
        sc.add(sd_sphere(0.022), red, hd * T(0.07 * sgn + 0.02 * sgn, 0.24, 0))
    for (x, y, r) in smoke:
        sc.add(sd_sphere(r), Mat("#9a9a9a", rough=0.9), root * T(x, y0 + y, 0))
    return sc


# ─── Minimax NNUE (old): a robot with a cathode-ray screen head ─────────────
def crt(anim, t, side):
    sc = Scene()
    base(sc, side)
    beige = paint("#d9cfb4", rough=0.45, noise=0.05)
    dark = Mat("#2a2a2a", rough=0.5)
    mode = anim
    phase = t

    def screen_pat(pl, pw, n):
        x, y = pl[:, 0] / 0.13, pl[:, 1] / 0.1
        scan = 0.75 + 0.25 * torch.sin((y + phase * 4) * 40) ** 2
        g = srgb("#46ff8a")
        if mode == "blocked":     # static
            v = vnoise(torch.stack([x * 40, y * 40, torch.full_like(x, phase * 97.0)], -1))
            return {"albedo": vec(0, 0, 0).expand_as(pl), "emissive": (vec(1, 1, 1) * v[:, None] * 1.2)}
        if mode == "win":         # a big smile
            smile = ((torch.sqrt(x * x + (y + 0.1) ** 2) - 0.55).abs() < 0.08) & (y < -0.1)
            eyes = (((x.abs() - 0.35) ** 2 + (y - 0.25) ** 2) < 0.012)
            on = (smile | eyes).float()
            return {"albedo": vec(0, 0, 0).expand_as(pl), "emissive": g * (on * scan * 2.5)[:, None]}
        # a small network: three layers of nodes, links between them
        on = torch.zeros_like(x)
        layers = [(-0.6, [-0.5, 0.0, 0.5]), (0.0, [-0.6, -0.2, 0.2, 0.6]), (0.6, [0.0])]
        for lx, ys in layers:
            for yy in ys:
                on = torch.maximum(on, (((x - lx) ** 2 + (y - yy) ** 2) < 0.012).float())
        for (ax, ays), (bx, bys) in zip(layers[:-1], layers[1:]):
            for ya in ays:
                for yb in bys:
                    hh = ((x - ax) / (bx - ax)).clamp(0, 1)
                    yy = ya + (yb - ya) * hh
                    on = torch.maximum(on, ((y - yy).abs() < 0.025).float() * ((x > ax) & (x < bx)).float() * 0.45)
        pulse = 0.6 + 0.4 * torch.sin((x - phase * 2) * 6)
        return {"albedo": vec(0, 0, 0).expand_as(pl), "emissive": g * (0.08 + on * pulse * 2.2)[:, None] * scan[:, None]}
    screen = Mat("#05140a", rough=0.05, clearcoat=1.5, pattern=screen_pat)
    hop = 0.0
    arm = 0.0
    if anim == "move":
        hop = 0.04 * abs(math.sin(TAU * t))
        arm = 0.6 * wave(t)
    elif anim == "win":
        arm = 2.3
        hop = 0.04 * abs(math.sin(TAU * 2 * t))
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)
    for sgn in (-1, 1):
        sc.add(sd_capsule((0, 0, 0), (0, 0.13, 0), 0.035), dark, root * T(0.07 * sgn, y0 + 0.04, 0))
        sc.add(sd_box(0.05, 0.025, 0.08, 0.02), beige, root * T(0.07 * sgn, y0 + 0.03, 0.02))
    body = root * T(0, y0 + 0.28, 0)
    sc.add(sd_box(0.15, 0.1, 0.11, 0.03), beige, body)
    for k in range(4):        # a keyboard on the chest
        for j in range(3):
            sc.add(sd_box(0.018, 0.008, 0.012, 0.004), Mat("#8a8270", rough=0.5), body * T(-0.075 + 0.05 * k, -0.04 + 0.03 * j, 0.11))
    sc.add(sd_box(0.03, 0.01, 0.005, 0.003), glow("#ffb020", 4.0), body * T(0.09, 0.07, 0.11))
    for sgn in (-1, 1):
        sh = body * T(0.17 * sgn, 0.05, 0) * RX(-arm * (1 if sgn > 0 else (1 if anim == "win" else -1))) * RZ(0.2 * sgn)
        sc.add(sd_capsule((0, 0, 0), (0, -0.15, 0), 0.028), dark, sh)
        sc.add(sd_box(0.035, 0.035, 0.035, 0.015), beige, sh * T(0, -0.18, 0))
    hd = body * T(0, 0.25, -0.02) * RX(-0.1)
    sc.add(sd_box(0.19, 0.15, 0.15, 0.03), beige, hd)
    sc.add(sd_box(0.14, 0.1, 0.1, 0.03), beige, hd * T(0, 0, -0.15))
    sc.add(sd_box(0.15, 0.11, 0.02, 0.025), dark, hd * T(0, 0.01, 0.145))
    sc.add(sd_ellipsoid(0.13, 0.095, 0.03), screen, hd * T(0, 0.01, 0.15))
    sc.add(sd_cylinder(0.01, 0.012, 0.004), glow("#ff3b30", 4.0), hd * T(0.14, -0.12, 0.15) * RX(math.pi / 2))
    return sc


# ─── Minimax Features: a brass geometer with a lens eye ─────────────────────
def geometer(anim, t, side):
    sc = Scene()
    base(sc, side)
    brass = Mat("#c9a14a", metal=1.0, rough=0.25,
                pattern=lambda pl, pw, n: {"albedo": srgb("#c9a14a") * (0.85 + 0.25 * fbm(pl * 20))[:, None]})
    teal = paint("#1f5c66", rough=0.4, metal=0.2)
    dark = Mat("#1a1c1f", rough=0.4)
    focus = 0.5 + 0.5 * wave(t)
    hop = 0.0
    hat = 0.0
    arm = 0.4 * wave(t)
    lens_col, lk = "#62e0ff", 4.0 + 2.0 * focus
    if anim == "move":
        hop = 0.05 * abs(math.sin(TAU * t))
    elif anim == "blocked":
        lens_col, lk = "#ff5f4a", 2.0 + 6.0 * (1.0 if int(t * 12) % 2 else 0.0)
        arm = 0.2 * wave(t, 4)
    elif anim == "win":
        hat = 0.18 * math.sin(math.pi * min(1.0, t * 2))
        arm = 1.5
        lk = 9.0
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)
    # three thin legs (a tripod)
    for a in (0.0, 2.1, 4.2):
        sc.add(sd_capsule((0, 0, 0), (0.12 * math.sin(a), -0.2, 0.12 * math.cos(a)), 0.016), brass, root * T(0, y0 + 0.24, 0))
        sc.add(sd_sphere(0.025), dark, root * T(0.12 * math.sin(a), y0 + 0.04, 0.12 * math.cos(a)))
    body = root * T(0, y0 + 0.33, 0)
    sc.add(sd_ellipsoid(0.13, 0.12, 0.11), teal, body)
    sc.add(sd_torus(0.125, 0.014), brass, body)
    sc.add(sd_cylinder(0.06, 0.02, 0.005), brass, body * T(0, 0.14, 0))
    # an arm holding a pair of compasses
    sh = body * T(0.13, 0.03, 0) * RZ(-0.6 - arm) * RX(-0.4)
    sc.add(sd_capsule((0, 0, 0), (0.12, -0.04, 0), 0.018), brass, sh)
    cp = sh * T(0.13, -0.04, 0) * RZ(-1.2)
    sc.add(sd_capsule((0, 0, 0), (0.04, -0.14, 0), 0.008), chrome(0.15), cp)
    sc.add(sd_capsule((0, 0, 0), (-0.04, -0.14, 0), 0.008), chrome(0.15), cp)
    sc.add(sd_sphere(0.016), brass, cp)
    sh2 = body * T(-0.13, 0.03, 0) * RZ(0.8) * RX(0.2)
    sc.add(sd_capsule((0, 0, 0), (-0.11, -0.06, 0), 0.018), brass, sh2)
    # head: a sphere, the big lens with its rings, a small top hat
    hd = body * T(0, 0.25, 0)
    sc.add(sd_sphere(0.13), brass, hd)
    le = hd * T(0.0, 0.0, 0.12)
    sc.add(sd_cylinder(0.035 + 0.02 * focus, 0.085, 0.015), dark, le * RX(math.pi / 2))
    sc.add(sd_torus(0.085, 0.014), brass, le * T(0, 0, 0.03) * RX(math.pi / 2))
    sc.add(sd_sphere(0.04), glow(lens_col, lk), le * T(0, 0, 0.02))
    sc.add(sd_sphere(0.075), Mat("#cfefff", glass=True), le * T(0, 0, 0.035))
    sc.add(sd_sphere(0.03), Mat("#cfefff", glass=True), hd * T(-0.09, 0.05, 0.08))     # a small second lens
    hatx = hd * T(0, 0.12 + hat, 0) * RZ(-0.15 - hat)
    sc.add(sd_cylinder(0.008, 0.09, 0.004), dark, hatx)
    sc.add(sd_cylinder(0.06, 0.06, 0.01), dark, hatx * T(0, 0.06, 0))
    sc.add(sd_cylinder(0.012, 0.062, 0.003), brass, hatx * T(0, 0.02, 0))
    return sc


# ─── MCTS ResNet: a sleek android with a glowing brain under glass ──────────
def android(anim, t, side):
    sc = Scene()
    base(sc, side)
    ceramic = Mat("#f2f3f5", rough=0.18, clearcoat=1.5)
    joint = Mat("#2b2f38", metal=0.6, rough=0.3)
    accent = Mat("#7c5cff", metal=0.0, rough=0.3)
    br_col, bk = "#b48cff", 4.0 + 2.5 * (0.5 + 0.5 * wave(t, 2))
    yaw = 0.15 * wave(t)
    lean = 0.0
    hop = 0.0
    arm_open = 0.15
    if anim == "move":
        lean = 0.2 * math.sin(math.pi * t)
        hop = 0.06 * math.sin(math.pi * t)
    elif anim == "blocked":
        br_col, bk = "#ff4a6a", 2.0 + 7.0 * (1.0 if int(t * 12) % 3 else 0.0)
        yaw = 0.35 * wave(t, 2)
    elif anim == "win":
        arm_open = 1.4
        bk = 10.0 + 3 * wave(t, 2)
        hop = 0.06 * abs(math.sin(TAU * t))
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)

    def brain_pat(pl, pw, n):
        folds = torch.sin(pl[:, 0] * 90 + torch.sin(pl[:, 1] * 60) * 2) * torch.sin(pl[:, 2] * 80 + pl[:, 1] * 40)
        v = (0.5 + 0.5 * folds).clamp(0, 1)
        return {"emissive": srgb(br_col) * (bk * 0.4 * (0.35 + 0.9 * v))[:, None]}
    brain = Mat("#3a1f6a", rough=0.4, pattern=brain_pat)
    # legs: slim, smooth
    for sgn in (-1, 1):
        hip = root * T(0.07 * sgn, y0 + 0.3, 0)
        sc.add(sd_capsule((0, 0, 0), (0, -0.13, 0.01), 0.04, 0.03), ceramic, hip)
        sc.add(sd_sphere(0.03), joint, hip * T(0, -0.14, 0.01))
        sc.add(sd_capsule((0, -0.14, 0.01), (0, -0.25, 0), 0.03, 0.035), ceramic, hip)
        sc.add(sd_ellipsoid(0.035, 0.02, 0.06), joint, hip * T(0, -0.28, 0.02))
    body = root * T(0, y0 + 0.4, 0) * RX(lean)
    sc.add(sd_ellipsoid(0.13, 0.15, 0.1), ceramic, body * T(0, 0.08, 0))
    sc.add(sd_ellipsoid(0.08, 0.07, 0.07), joint, body * T(0, -0.06, 0))
    sc.add(sd_capsule((-0.06, 0.12, 0.09), (0.06, 0.12, 0.09), 0.008), glow(br_col, bk * 0.6), body)
    sc.add(sd_ellipsoid(0.03, 0.05, 0.01), accent, body * T(0, 0.04, 0.1))
    for sgn in (-1, 1):
        sh = body * T(0.15 * sgn, 0.17, 0) * RZ(arm_open * sgn) * RX(-0.1)
        sc.add(sd_sphere(0.04), joint, sh)
        sc.add(sd_capsule((0, 0, 0), (0.0, -0.15, 0.0), 0.032, 0.026), ceramic, sh)
        sc.add(sd_capsule((0, -0.16, 0), (0.0, -0.27, 0.04), 0.024, 0.02), ceramic, sh)
        sc.add(sd_sphere(0.026), joint, sh * T(0, -0.29, 0.05))
    # head: a smooth face, a glass dome on top with the brain
    hd = body * T(0, 0.3, 0.0) * RY(yaw)
    sc.add(sd_capsule((0, -0.06, 0), (0, -0.02, 0), 0.035), joint, body * T(0, 0.27, 0))
    sc.add(sd_plane_cut(sd_ellipsoid(0.12, 0.13, 0.12), (0, 1, 0), 0.0), ceramic, hd)
    sc.add(sd_box(0.085, 0.022, 0.02, 0.02), Mat("#0c0e14", rough=0.05, clearcoat=1.5), hd * T(0, -0.02, 0.105))
    screen_eyes(sc, hd * T(0, -0.02, 0.122), br_col, 7.0, w=0.022, h=0.01, dx=0.04, happy=(anim == "win"))
    sc.add(sd_torus(0.118, 0.01), joint, hd)
    sc.add(sd_ellipsoid(0.09, 0.07, 0.09), brain, hd * T(0, 0.06, 0))
    sc.add(sd_plane_cut(sd_ellipsoid(0.115, 0.14, 0.115), (0, -1, 0), 0.0), Mat("#e8f0ff", glass=True), hd)
    return sc


# ─── AlphaZero: an energy core in orbiting rings ────────────────────────────
def core(anim, t, side):
    sc = Scene()
    base(sc, side)
    obsidian = Mat("#0d0f16", metal=0.5, rough=0.12, clearcoat=1.5)
    platinum = Mat("#e6e8ee", metal=1.0, rough=0.15)
    spin = TAU * t
    tilt = 0.0
    k_core = 9.0 + 3.0 * wave(t, 2)
    ring_r = 1.0
    jitter = 0.0
    col = "#9fdcff"
    hover = 0.03 * wave(t)
    if anim == "move":
        tilt = 0.4 * math.sin(math.pi * t)
        spin = TAU * t * 2
    elif anim == "blocked":
        jitter = 0.08
        col = "#ff6a7a"
        k_core = 3.0 + 6.0 * (1.0 if int(t * 12) % 2 else 0.0)
    elif anim == "win":
        ring_r = 1.0 + 0.25 * (0.5 + 0.5 * wave(t, 2))
        k_core = 18.0
        spin = TAU * t * 2
    y0 = BASE_TOP
    # a pedestal and a beam of light
    sc.add(sd_cone(0.06, 0.16, 0.08, 0.01), obsidian, T(0, y0 + 0.06, 0))
    sc.add(sd_cylinder(0.012, 0.085, 0.004), glow(col, 4.0), T(0, y0 + 0.125, 0))
    sc.add(sd_cylinder(0.2, 0.02, 0.0), glow(col, 1.5), T(0, y0 + 0.33, 0))
    c = T(0, y0 + 0.55 + hover, 0) * RY(spin * 0.25)
    # the core: a bright sphere in a faceted dark crystal cage
    sc.add(sd_sphere(0.085), glow("#ffffff", k_core * 0.7), c)
    sc.add(sd_sphere(0.11), Mat("#bfe6ff", glass=True), c)
    for k in range(6):
        a = TAU * k / 6
        sc.add(sd_box(0.012, 0.13, 0.012, 0.006), obsidian, c * RY(a) * T(0.12, 0, 0) * RZ(0.15))
    # three rings in orbit, each tilted differently, with beads of light
    for i, (ax, az, sp) in enumerate(((1.2, 0.0, 1), (0.3, 1.1, -1), (-0.9, 0.5, 1))):
        j = jitter * math.sin(TAU * (t * 6 + i * 0.3))
        rx = c * RX(ax + tilt + j) * RZ(az) * RY(spin * sp)
        R = (0.2 + 0.05 * i) * ring_r
        sc.add(sd_torus(R, 0.012), platinum, rx)
        for b in range(3):
            a = TAU * b / 3
            sc.add(sd_sphere(0.02), glow(col, 8.0), rx * T(R * math.cos(a), 0, R * math.sin(a)))
    return sc


# ─── Minimax: a stopwatch robot with four arms (time per move, lazy SMP) ────
def chrono(anim, t, side):
    """The search by the clock: the head is a stopwatch whose hand sweeps the
    time of the move; four arms, the threads that search together; a stack of
    lit rings on the chest, the depths reached (iterative deepening)."""
    sc = Scene()
    base(sc, side)
    steel = chrome(0.18)
    navy = paint("#24407a", rough=0.3, metal=0.4)
    orange = paint("#e8782a", rough=0.3, metal=0.2)
    dark = Mat("#1d2024", rough=0.5)
    hop, arm = 0.0, 0.0
    depth = int(t * 6) % 6 + 1
    if anim == "move":
        hop = 0.04 * abs(math.sin(TAU * t))
        arm = 0.5 * wave(t)
    elif anim == "blocked":
        depth = 1 + int(t * 12) % 2
    elif anim == "win":
        hop = 0.04 * abs(math.sin(TAU * 2 * t))
        arm = 2.4
        depth = 6
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)
    for sgn in (-1, 1):   # legs
        sc.add(sd_capsule((0, 0, 0), (0, 0.13, 0), 0.03), steel, root * T(0.07 * sgn, y0 + 0.04, 0))
        sc.add(sd_box(0.045, 0.025, 0.075, 0.02), navy, root * T(0.07 * sgn, y0 + 0.03, 0.02))
    body = root * T(0, y0 + 0.28, 0)
    sc.add(sd_cylinder(0.12, 0.13, 0.05), navy, body)
    for k in range(6):    # the depths reached: rings lit from the bottom
        on = k < depth
        sc.add(sd_torus(0.132, 0.008), glow("#5ad8ff", 5.0) if on else dark, body * T(0, -0.08 + 0.032 * k, 0))
    for i, (sgn, y) in enumerate(((-1, 0.07), (1, 0.07), (-1, -0.03), (1, -0.03))):   # four arms, the threads
        a = arm if y > 0 else arm * 0.5
        sh = body * T(0.14 * sgn, y, 0) * RX(-a - 0.25 * math.sin(TAU * t + i)) * RZ((0.35 if y > 0 else 0.7) * sgn)
        sc.add(sd_sphere(0.028), steel, sh)
        sc.add(sd_capsule((0, 0, 0), (0, -0.12, 0), 0.016), steel, sh)
        sc.add(sd_sphere(0.024), orange, sh * T(0, -0.14, 0))
    # the stopwatch head: a chrome case, a white dial, the sweeping hand, the crown on top
    hd = body * T(0, 0.27, 0) * RY(0.1 * wave(t))
    sc.add(sd_cylinder(0.04, 0.15, 0.03), steel, hd * RX(math.pi / 2))
    sc.add(sd_cylinder(0.005, 0.13, 0.003), Mat("#f4f2ec", rough=0.3), hd * T(0, 0, 0.04) * RX(math.pi / 2))
    for k in range(12):
        a = TAU * k / 12
        sc.add(sd_box(0.003, 0.012, 0.003, 0.001), dark, hd * T(0.11 * math.sin(a), 0.11 * math.cos(a), 0.046) * RZ(-a))
    sweep = -TAU * t * (3 if anim == "win" else 1)
    sc.add(sd_box(0.004, 0.05, 0.003, 0.0015), orange, hd * T(0, 0, 0.05) * RZ(sweep) * T(0, 0.045, 0))
    sc.add(sd_sphere(0.012), orange, hd * T(0, 0, 0.05))
    screen_eyes(sc, hd * T(0, -0.045, 0.05), "#5ad8ff", 5.0 * (0.3 if anim == "blocked" and int(t * 12) % 2 else 1.0),
                w=0.018, h=0.012, dx=0.04, happy=(anim == "win"))
    sc.add(sd_cylinder(0.02, 0.022, 0.006), steel, hd * T(0, 0.17, 0))
    sc.add(sd_cylinder(0.008, 0.03, 0.004), orange, hd * T(0, 0.195, 0))
    for sgn in (-1, 1):
        sc.add(sd_cylinder(0.012, 0.014, 0.004), steel, hd * T(0.11 * sgn, 0.13, 0) * RZ(-0.7 * sgn))
    return sc


# ─── MCTS NNUE: the tree robot grown up, a neural core judging its leaves ───
def sylve(anim, t, side):
    """The tree search, now with a judge: a silver robot carrying a tree of
    branches whose leaves light up one after the other; each lit leaf sends a
    pulse down to the neural core in its chest (stacked glowing layers) and
    three lamps (black wins, draw, white wins) give the verdict."""
    sc = Scene()
    base(sc, side)
    silver_ = chrome(0.2)
    green = paint("#2a7a5a", rough=0.3, metal=0.4)
    dark = Mat("#1d2024", rough=0.5)
    hop, sway = 0.0, 0.06 * wave(t)
    col = "#a8ff7a"
    if anim == "move":
        hop = 0.05 * math.sin(math.pi * t)
        sway = 0.25 * math.sin(math.pi * t)
    elif anim == "blocked":
        col = "#ff6a4a"
        sway = 0.05 * wave(t, 4)
    elif anim == "win":
        hop = 0.05 * abs(math.sin(TAU * t))
    y0 = BASE_TOP
    root = RY(HEADING) * T(0, hop, 0)
    sc.add(sd_cylinder(0.04, 0.13, 0.02), silver_, root * T(0, y0 + 0.04, 0))           # a round foot
    body = root * T(0, y0 + 0.26, 0)
    sc.add(sd_box(0.13, 0.14, 0.1, 0.05), green, body)
    core = body * T(0, 0.02, 0.1)
    sc.add(sd_box(0.08, 0.09, 0.012, 0.01), dark, core)
    for k, w in enumerate((0.065, 0.045, 0.03)):   # the network's layers, narrower and narrower
        p_ = 0.5 + 0.5 * math.sin(TAU * (t * 2 - k * 0.2))
        sc.add(sd_box(w, 0.008, 0.004, 0.003), glow("#ffd24a", 2.0 + 5.0 * p_), core * T(0, 0.05 - 0.035 * k, 0.012))
    verdict = int(t * 3) % 3 if anim != "win" else 0
    for k, c in enumerate(("#f2f2f2", "#9a9aa8", "#3a3a48")):
        sc.add(sd_sphere(0.012), glow(c if k != 2 else "#5a6aff", 6.0 if k == verdict else 0.6), core * T(-0.04 + 0.04 * k, -0.07, 0.012))
    for sgn in (-1, 1):
        sh = body * T(0.16 * sgn, 0.06, 0) * RZ(0.45 * sgn) * RX(-0.3 - (2.2 if anim == "win" else 0.0))
        sc.add(sd_capsule((0, 0, 0), (0, -0.14, 0), 0.025), silver_, sh)
        sc.add(sd_sphere(0.034), green, sh * T(0, -0.16, 0))
    hd = body * T(0, 0.22, 0)
    sc.add(sd_ellipsoid(0.12, 0.09, 0.1), silver_, hd)
    sc.add(sd_box(0.09, 0.028, 0.04, 0.025), dark, hd * T(0, 0.0, 0.07))
    screen_eyes(sc, hd * T(0, 0.0, 0.11), col, 6.0, w=0.02, h=0.02, dx=0.045, happy=(anim == "win"))
    # the tree, wider than the MCTS robot's, a leaf lit in turn and its pulse running down
    tr = hd * T(0, 0.08, 0) * RZ(sway)
    sc.add(sd_capsule((0, 0, 0), (0, 0.1, 0), 0.016), silver_, tr)
    leaves = []
    for a in (-0.8, -0.27, 0.27, 0.8):
        b1 = tr * T(0, 0.1, 0) * RZ(a) * RX(0.2 * a)
        sc.add(sd_capsule((0, 0, 0), (0, 0.11, 0), 0.01), silver_, b1)
        for a2 in (-0.4, 0.4):
            b2 = b1 * T(0, 0.11, 0) * RZ(a2) * RX(0.3 * a2)
            sc.add(sd_capsule((0, 0, 0), (0, 0.07, 0), 0.007), silver_, b2)
            leaves.append(b2 * T(0, 0.07, 0))
    lit = int(t * len(leaves) * (2 if anim == "win" else 1)) % len(leaves)
    for i, lf in enumerate(leaves):
        on = anim == "win" or i == lit
        sc.add(sd_sphere(0.02), glow(col, 8.0 if on else 1.2), lf)
    u = (t * len(leaves)) % 1     # the pulse going down the trunk to the core
    sc.add(sd_sphere(0.014), glow("#ffd24a", 6.0), tr * T(0, 0.1 * (1 - u), 0))
    return sc


SCALE = {"tin_toy": 1.3, "roller": 1.2, "dicer": 1.2, "tree_bot": 1.15, "retro": 1.35, "crt": 1.25, "geometer": 1.35,
         "android": 1.3, "core": 1.3, "chrono": 1.3, "sylve": 1.3}
BASE_PARTS = 4        # the parts of base(): never scaled


def scaled(name, f):
    k = SCALE.get(name, 1.0)
    if k == 1.0:
        return f
    return lambda anim, t, side: f(anim, t, side).scale_from(BASE_PARTS, k, (0, BASE_TOP, 0))


_MODELS = {"sentinel": sentinel, "human": human, "tin_toy": tin_toy, "roller": roller, "dicer": dicer, "tree_bot": tree_bot,
          "retro": retro, "crt": crt, "geometer": geometer, "android": android, "core": core,
          "chrono": chrono, "sylve": sylve}
MODELS = {k: scaled(k, f) for k, f in _MODELS.items()}


# the "Légendes" set (legends.py)
def _add_legends():
    import legends
    MODELS["human_legendes"] = lambda anim, t, side: human(anim, t, side, "legendes")
    for k, f in legends.LEGENDS.items():
        SCALE.setdefault(k, legends.SCALE.get(k, 1.3))
        MODELS[k] = scaled(k, f)

