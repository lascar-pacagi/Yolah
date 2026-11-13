"""
generals.py — the "Généraux" set: the ten best generals of Ethan Arsht's
statistical ranking (WAR, 3,580 battles, 2020; "all in good fun" in its
author's words), one per AI, the strongest AI the best general — Napoleon
for AlphaZero — and the user as a marshal.

Likeness at the size of a square comes from silhouettes and attributes more
than from faces: each one has the hat, the clothes, the beard and the object
people know him by (Napoleon's bicorne worn sideways and his hand in his
waistcoat, Caesar's laurel, Wellington's bicorne worn front to back and his
nose, Shingen's horned helmet with a white mane and his war fan, Khalid's
turban over mail, Hannibal's crested helmet and lost eye, Grant's beard and
cigar, Frederick's tricorne and cane, Zhukov's cap and medals, Alexander's
curls and diadem). Gold stars on the base tell the rank.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, sd_plane_cut, fbm, length
from models import base, BASE_TOP, TAU, wave, glow, gold, eye_mat
from legends import base_pose, held, leather, cloth, iron, silver, wood, skin
from heroic import adult

GOLD = gold


def flat(c, rough=0.6, metal=0.0):
    return Mat(c, rough=rough, metal=metal)


# ─── faces ──────────────────────────────────────────────────────────────────
def face(sc, x, r, sk_c, anim, t, hair_c="#3a2a20", nose=1.0, brow_c=None, eye_patch=False, iris="#5a6a7a", cheeks=1.0):
    """A man's face for a general: skull, jaw, ears, eyes with lids, brows, nose, mouth."""
    sk = skin(sk_c, 0.45)
    sc.add(sd_ellipsoid(r, r * 1.06, r * 0.98), sk, x)
    sc.add(sd_ellipsoid(r * 0.74 * cheeks, r * 0.6, r * 0.72), sk, x * T(0, -r * 0.45, r * 0.22), blend=r * 0.3)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(r * 0.16, r * 0.26, r * 0.1), sk, x * T(r * 0.95 * sgn, -r * 0.02, -r * 0.05), blend=r * 0.05)
    sc.add(sd_capsule((0, r * 0.15, 0), (0, -r * 0.18, r * 0.12 * nose), r * 0.1 * nose, r * 0.14 * nose), sk, x * T(0, -r * 0.08, r * 0.9), blend=r * 0.08)
    look = 0.2 * math.sin(TAU * t) if anim == "idle" else 0.0
    closed = anim == "win" or (anim == "idle" and abs(t - 0.6) < 0.04)
    bc = brow_c or hair_c
    for sgn in (-1, 1):
        ec = x * T(r * 0.35 * sgn, r * 0.12, r * 0.8)
        if eye_patch and sgn < 0:
            sc.add(sd_ellipsoid(r * 0.22, r * 0.18, r * 0.06), flat("#141414", 0.5), ec * T(0, 0, r * 0.12))
            sc.add(sd_capsule((-r * 0.9, r * 0.5, -r * 0.6), (r * 0.9, -r * 0.1, r * 0.05), r * 0.025), flat("#141414", 0.5), ec)
        elif closed:
            sc.add(sd_capsule((-r * 0.13, 0, 0), (r * 0.13, 0, 0), r * 0.03), flat("#2a1a12", 0.6), ec * T(0, 0, r * 0.12))
        else:
            sc.add(sd_sphere(r * 0.17), eye_mat((look, 0.0, 1.0), iris=iris), ec)
            sc.add(sd_plane_cut(sd_sphere(r * 0.185), (0, -1, 0.2), -r * 0.05), sk, ec, blend=r * 0.02)     # heavy upper lid
        tilt = {"blocked": 0.25, "move": -0.2}.get(anim, -0.05)
        sc.add(sd_capsule((0, 0, 0), (r * 0.3 * sgn, -tilt * r * 0.4 * sgn * sgn, -r * 0.05), r * 0.05), flat(bc, 0.7),
               x * T(r * 0.18 * sgn, r * 0.4, r * 0.86))
    mo = x * T(0, -r * 0.5, r * 0.84)
    if anim == "win":
        sc.add(sd_ellipsoid(r * 0.22, r * 0.1, r * 0.12), flat("#5a1a1a", 0.6), mo)
    elif anim == "blocked":
        sc.add(sd_capsule((-r * 0.17, -r * 0.02, 0), (r * 0.17, -r * 0.02, 0), r * 0.025), flat("#7a3a34", 0.5), mo * RZ(0.1))
    else:
        sc.add(sd_capsule((-r * 0.17, 0, 0), (r * 0.17, 0, 0), r * 0.022), flat("#8a4a42", 0.5), mo)


def hair_cap(sc, x, r, c, front=0.35, back=-0.6, bulk=1.06, curls=False):
    hm = Mat(c, rough=0.5, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.75 + 0.4 * fbm(pl * 60, 2))[:, None]})

    def capf(p):
        ell = sd_ellipsoid(r * bulk, r * bulk, r * bulk)(p)
        fz = ((p[:, 2] / r - 0.05) * 2.0).clamp(0, 1)
        edge = r * front * fz + r * back * (1 - fz)
        return torch.maximum(ell, (edge - p[:, 1]) * 0.7)
    sc.add(capf, hm, x * T(0, r * 0.03, -r * 0.02))
    if curls:
        for k in range(16):
            a = k * 2.4
            e = 0.2 + 0.5 * (k % 4) / 3
            q = (math.sin(a) * math.cos(e) * r, math.sin(e) * r + r * 0.15, math.cos(a) * math.cos(e) * r * 0.9 - r * 0.1)
            sc.add(sd_sphere(r * 0.22), hm, x * T(*q), blend=r * 0.08)
    return hm


def beard(sc, x, r, c, full=True, moustache=True, sideburns=False):
    bm = Mat(c, rough=0.7, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.7 + 0.5 * fbm(pl * 90, 2))[:, None]})
    if full:
        sc.add(sd_plane_cut(sd_ellipsoid(r * 0.78, r * 0.62, r * 0.7), (0, 1, 0.25), -r * 0.12), bm, x * T(0, -r * 0.5, r * 0.25), blend=r * 0.04)
    if moustache:
        sc.add(sd_ellipsoid(r * 0.3, r * 0.08, r * 0.1), bm, x * T(0, -r * 0.38, r * 0.88))
    if sideburns:
        for sgn in (-1, 1):
            sc.add(sd_ellipsoid(r * 0.1, r * 0.3, r * 0.15), bm, x * T(r * 0.85 * sgn, -r * 0.2, r * 0.1))


# ─── hats ───────────────────────────────────────────────────────────────────
def bicorne(sc, x, r, sideways=True, trim=True, cockade=True):
    """A half moon standing on the head: along x (worn sideways, Napoleon) or
    along z (worn front to back, Wellington)."""
    felt = Mat("#14141a", rough=0.75, pattern=lambda pl, pw, n: {"albedo": srgb("#14141a") * (0.8 + 0.4 * fbm(pl * 50, 2))[:, None]})
    rot = RY(0) if sideways else RY(math.pi / 2)
    h = x * T(0, r * 0.82, -r * 0.05) * rot
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.75, r * 0.85, r * 0.42), (0, -1, 0), 0.0), felt, h)
    sc.add(sd_ellipsoid(r * 0.95, r * 0.25, r * 0.75), felt, h * T(0, r * 0.05, 0))
    if trim:
        sc.add(sd_plane_cut(sd_torus(r * 1.0, r * 0.03), (0, -1, 0), 0.0), gold(), h * RX(math.pi / 2) * T(0, 0, 0) * RZ(0) if False else h * T(0, 0, r * 0.0), op="add") if False else None
    if cockade:
        cz = r * 0.4 if sideways else r * 0.0
        cx = 0.0 if sideways else r * 0.0
        c0 = h * T(r * 0.55 if sideways else 0, r * 0.45, r * 0.38 if sideways else 0) * (RY(0) if sideways else RY(math.pi / 2))
        sc.add(sd_cylinder(r * 0.02, r * 0.15, r * 0.01), flat("#c8242a"), c0 * RX(math.pi / 2))
        sc.add(sd_cylinder(r * 0.025, r * 0.1, r * 0.01), flat("#f2f0ea"), c0 * RX(math.pi / 2) * T(0, r * 0.01, 0))
        sc.add(sd_cylinder(r * 0.03, r * 0.05, r * 0.01), flat("#1e3a8a"), c0 * RX(math.pi / 2) * T(0, r * 0.02, 0))


def tricorne(sc, x, r):
    felt = Mat("#14141a", rough=0.75)
    h = x * T(0, r * 0.75, 0)
    sc.add(sd_cylinder(r * 0.25, r * 0.85, r * 0.1), felt, h * T(0, r * 0.15, 0))
    for k in range(3):
        a = k * TAU / 3 + math.pi / 3
        b = h * RY(a) * T(0, r * 0.1, r * 0.95) * RX(-1.0)
        sc.add(sd_box(r * 0.75, r * 0.32, r * 0.04, r * 0.03), felt, b)
        sc.add(sd_box(r * 0.75, r * 0.03, r * 0.045, r * 0.02), gold(), b * T(0, r * 0.32, 0))


def slouch_hat(sc, x, r):
    felt = Mat("#1c1a1a", rough=0.85, pattern=lambda pl, pw, n: {"albedo": srgb("#1c1a1a") * (0.8 + 0.4 * fbm(pl * 40, 2))[:, None]})
    h = x * T(0, r * 0.7, 0)
    sc.add(sd_cylinder(r * 0.35, r * 0.8, r * 0.2), felt, h * T(0, r * 0.25, 0))
    sc.add(lambda p: sd_cylinder(r * 0.03, r * 1.45, r * 0.03)(p) + 0.15 * (p[:, 0] ** 2 + p[:, 2] ** 2) / r, felt, h)
    sc.add(sd_torus(r * 0.82, r * 0.05), flat("#2a2420"), h * T(0, r * 0.05, 0))


def marshal_cap(sc, x, r, star=True):
    cap = Mat("#6a6a58", rough=0.7)
    h = x * T(0, r * 0.75, r * 0.02) * RX(-0.12)
    sc.add(sd_cylinder(r * 0.08, r * 1.15, r * 0.05), cap, h * T(0, r * 0.32, 0))
    sc.add(sd_cylinder(r * 0.2, r * 0.9, r * 0.05), cap, h * T(0, r * 0.12, 0))
    sc.add(sd_cylinder(r * 0.08, r * 0.92, r * 0.03), flat("#a8242a"), h * T(0, r * 0.05, 0))
    sc.add(sd_plane_cut(sd_ellipsoid(r * 0.8, r * 0.05, r * 0.55), (0, 0, -1), 0.0), flat("#141414", 0.25), h * T(0, -r * 0.05, r * 0.55) * RX(0.25))
    sc.add(sd_box(r * 0.4, r * 0.025, r * 0.02, r * 0.01), gold(), h * T(0, r * 0.0, r * 0.92))
    if star:
        sc.add(sd_sphere(r * 0.09), glow("#ff4a3a", 1.0), h * T(0, r * 0.16, r * 0.92))


def laurel(sc, x, r):
    leaf = Mat("#4a7a2a", rough=0.45, clearcoat=0.5)
    h = x * T(0, r * 0.45, -r * 0.02) * RX(-0.25)
    for k in range(18):
        a = -2.6 + 5.2 * k / 17
        sgn = 1 if k % 2 else -1
        sc.add(sd_ellipsoid(r * 0.12, r * 0.05, r * 0.06), leaf,
               h * T(math.sin(a) * r * 0.97, 0.0, math.cos(a) * r * 0.97) * RY(a + math.pi / 2) * RZ(0.4 * sgn))


def kabuto(sc, x, r):
    lac = Mat("#1a1414", metal=0.3, rough=0.25, clearcoat=1.0)
    h = x * T(0, r * 0.25, 0)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.12, r * 1.05, r * 1.1), (0, -1, 0), 0.0), lac, h)
    for k in range(4):    # the flared neck guard, in plates
        sc.add(sd_plane_cut(sd_torus(r * (1.15 + 0.08 * k), r * 0.05), (0, 0, 1), r * 0.3), flat("#b8242a", 0.4), h * T(0, -r * (0.05 + 0.12 * k), 0))
    sc.add(sd_box(r * 0.08, r * 0.3, r * 0.05, r * 0.03), gold(), h * T(0, r * 0.25, r * 1.05) * RX(0.3))
    for sgn in (-1, 1):      # the horns (kuwagata)
        sc.add(sd_capsule((0, 0, 0), (r * 0.45 * sgn, r * 0.85, r * 0.15), r * 0.07, r * 0.03), gold(), h * T(r * 0.15 * sgn, r * 0.35, r * 1.0))
    mane = Mat("#f2f0ea", rough=0.85, pattern=lambda pl, pw, n: {"albedo": srgb("#f2f0ea") * (0.75 + 0.35 * (0.5 + 0.5 * torch.sin(pl[:, 0] * 260)))[:, None]})
    sc.add(lambda p: sd_ellipsoid(r * 1.3, r * 1.45, r * 1.1)(p) + 0.02 * torch.sin(p[:, 0] * 120), mane, h * T(0, -r * 0.4, -r * 0.55))


def turban(sc, x, r, c="#f0ece0"):
    tm = Mat(c, rough=0.8, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.8 + 0.25 * (0.5 + 0.5 * torch.sin((pl[:, 1] * 3 + pl[:, 0]) * 120)))[:, None]})
    mail = Mat("#8a9098", metal=0.9, rough=0.4, pattern=lambda pl, pw, n: {"albedo": srgb("#8a9098") * (0.6 + 0.6 * (torch.sin(pl[:, 0] * 300) * torch.sin(pl[:, 1] * 300) > 0).float())[:, None]})
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.12, r * 1.2, r * 1.12), (0, 0.15, 1), r * 0.55), mail, x * T(0, -r * 0.05, -r * 0.05))
    h = x * T(0, r * 0.45, 0)
    for k in range(3):
        sc.add(sd_torus(r * (1.0 - 0.08 * k), r * 0.17), tm, h * T(0, r * 0.2 * k, 0) * RZ(0.12 * (k - 1)))
    sc.add(sd_cone(r * 0.25, r * 0.35, r * 0.05, r * 0.03), iron("#7a7a7a", 0.3), h * T(0, r * 0.65, 0))


def crested_helm(sc, x, r):
    bronze = Mat("#b07a3a", metal=1.0, rough=0.3)
    h = x * T(0, r * 0.2, 0)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.12, r * 1.08, r * 1.12), (0, -1, 0), 0.0), bronze, h)
    for sgn in (-1, 1):
        sc.add(sd_box(r * 0.05, r * 0.4, r * 0.3, r * 0.03), bronze, h * T(r * 0.95 * sgn, -r * 0.35, r * 0.25))
    crest = Mat("#a8242a", rough=0.85, pattern=lambda pl, pw, n: {"albedo": srgb("#a8242a") * (0.7 + 0.4 * (0.5 + 0.5 * torch.sin(pl[:, 2] * 200)))[:, None]})
    sc.add(sd_box(r * 0.1, r * 0.35, r * 1.15, r * 0.08), crest, h * T(0, r * 1.2, -r * 0.15))


def diadem(sc, x, r):
    sc.add(sd_torus(r * 1.0, r * 0.04), gold(), x * T(0, r * 0.35, 0) * RX(-0.15))
    sc.add(sd_ellipsoid(r * 0.08, r * 0.12, r * 0.04), glow("#ffd24a", 1.5), x * T(0, r * 0.4, r * 0.98))


# ─── uniforms ───────────────────────────────────────────────────────────────
def coat_mats(coat, facing=None, trousers="#f0ece0", boots="#141414"):
    return {"skin": skin("#e8b49a"), "top": cloth(coat, 0.75), "legs": cloth(trousers, 0.8), "boots": Mat(boots, rough=0.25, clearcoat=0.8),
            "arms": cloth(coat, 0.75)}


def epaulettes(sc, rig, fringe=True):
    for sgn in (-1, 1):
        e = rig["torso"] * T(0.15 * rig["bulk"] * sgn, 0.3 * rig["H"] + 0.012, 0)
        sc.add(sd_ellipsoid(0.055, 0.016, 0.045), gold(), e)
        if fringe:
            for k in range(5):
                sc.add(sd_capsule((0, 0, 0), (0.006 * sgn, -0.03, 0), 0.005), gold(), e * T(0.048 * sgn, -0.006, -0.032 + 0.016 * k))


def coat_tails(sc, rig, c, ln=0.18):
    sc.add(sd_plane_cut(sd_cone(ln * 0.5, 0.125, 0.105, 0.01), (0, 0, 1), 0.03), cloth(c, 0.75), rig["torso"] * T(0, 0.04 - ln * 0.5, 0))


def buttons(sc, rig, n=5, c=None):
    for k in range(n):
        sc.add(sd_sphere(0.007), c or gold(), rig["chest"] * T(0, 0.07 - 0.03 * k, 0.004))


def stars(sc, n):
    """Rank: n gold stars in a row on the front of the base."""
    for k in range(n):
        a = math.pi / 2 + (k - (n - 1) / 2) * 0.17
        sc.add(sd_cone(0.016, 0.04, 0.006, 0.004), Mat("#ffd24a", metal=1.0, rough=0.2, emissive="#ffb030", emit=0.6),
               T(0.5 * math.cos(a) * 0.98, 0.11, 0.5 * math.sin(a) * 0.98) * RX(-1.2))


# ─── the generals ───────────────────────────────────────────────────────────
def general(anim, t, side, g):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    if "pose" in g:
        g["pose"](p, anim, t)
    rig = adult(sc, p, g["mats"], H=g.get("h", 1.0) * 1.04, bulk=g.get("w", 1.0) * 1.05, head=1.02)
    r = rig["hr"]
    x = rig["head"]
    face(sc, x, r, g.get("skin", "#e8b49a"), anim, t, g.get("hair", "#3a2a20"), g.get("nose", 1.0), eye_patch=g.get("patch", False),
         iris=g.get("iris", "#5a6a7a"), cheeks=g.get("cheeks", 1.0))
    if "head" in g:
        g["head"](sc, x, r, anim, t)
    if "body" in g:
        g["body"](sc, rig, anim, t)
    stars(sc, g["stars"])
    return sc


def pose_hand_in_waistcoat(p, anim, t):
    if anim != "win":
        p.r = [0.9, -0.55, 2.0]          # the right hand in the waistcoat
    p.l = [0.1, 0.12, 0.6] if anim != "win" else [2.45, 0.25, 0.15]


def napoleon_body(sc, rig, anim, t):
    # the grey redingote over the green coat, white waistcoat, a medal; the eagle standard at the back
    sc.add(sd_box(0.05, 0.1, 0.012, 0.012), cloth("#f2f0ea", 0.7), rig["chest"] * T(0, -0.01, -0.004))
    buttons(sc, rig, 4)
    sc.add(sd_sphere(0.011), glow("#c8242a", 0.6), rig["chest"] * T(-0.06, 0.05, 0.0))
    coat_tails(sc, rig, "#6a6c70", 0.24)
    for sgn in (-1, 1):
        sc.add(sd_box(0.025, 0.13, 0.012, 0.01), cloth("#6a6c70", 0.75), rig["chest"] * T(0.07 * sgn, -0.02, -0.006) * RZ(0.1 * sgn))
    pole = rig["back"] * T(0.08, 0.0, -0.04)
    sc.add(sd_capsule((0, -0.4, 0), (0, 0.16, 0), 0.007), wood("#3a2a1a"), pole)
    eagle = pole * T(0, 0.19, 0)
    sc.add(sd_ellipsoid(0.03, 0.04, 0.025), gold(), eagle)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.05, 0.03, 0.01), gold(), eagle * T(0.04 * sgn, 0.02, 0) * RZ(-0.5 * sgn))
    if anim == "win":
        sc.add(sd_capsule((0, -0.02, 0), (0, 0.16, 0), 0.016), Mat("#2a2420", rough=0.3, metal=0.5), held(rig["l"], -math.pi))    # spyglass


def caesar_body(sc, rig, anim, t):
    cuirass = Mat("#e0d8c0", metal=0.8, rough=0.3)
    sc.add(sd_ellipsoid(0.142, 0.12, 0.092), cuirass, rig["torso"] * T(0, 0.21, 0.005))
    sc.add(sd_ellipsoid(0.11, 0.08, 0.085), cuirass, rig["torso"] * T(0, 0.1, 0.003), blend=0.03)
    sc.add(sd_box(0.13, 0.24, 0.012, 0.012), cloth("#a8242a", 0.8), rig["back"] * T(0, -0.2, -0.02) * RX(0.1))
    for k in range(9):
        a = -1.6 + 0.4 * k
        sc.add(sd_box(0.018, 0.05, 0.006, 0.004), leather("#5a3a1a"), rig["torso"] * T(0.11 * math.sin(a), -0.015, 0.085 * math.cos(a)) * RY(a))
    sw = held(rig["r"], -math.pi if anim == "win" else -1.3)
    sc.add(sd_box(0.016, 0.13, 0.004, 0.003), silver(0.15), sw * T(0, 0.17, 0))
    sc.add(sd_box(0.035, 0.008, 0.012, 0.004), gold(), sw * T(0, 0.04, 0))


def wellington_body(sc, rig, anim, t):
    sc.add(sd_ellipsoid(0.035, 0.03, 0.025), cloth("#f2f0ea", 0.7), rig["torso"] * T(0, 0.31, 0.07))          # the cravat
    buttons(sc, rig, 5)
    coat_tails(sc, rig, "#1c2236", 0.22)
    sc.add(sd_box(0.13, 0.16, 0.012, 0.012), cloth("#1c2236", 0.8), rig["back"] * T(0, -0.12, -0.02) * RX(0.1))   # a short cloak
    sg = held(rig["r"], -math.pi if anim == "win" else -1.6)
    sc.add(sd_capsule((0, -0.02, 0), (0, 0.14, 0), 0.014, 0.018), Mat("#c8a050", metal=1.0, rough=0.3), sg)       # a telescope


def shingen_body(sc, rig, anim, t):
    red = Mat("#a8242a", metal=0.2, rough=0.3, clearcoat=1.0,
              pattern=lambda pl, pw, n: {"albedo": srgb("#a8242a") * (0.75 + 0.3 * (torch.frac(pl[:, 1] * 45) > 0.15).float())[:, None]})
    sc.add(sd_ellipsoid(0.15, 0.13, 0.1), red, rig["torso"] * T(0, 0.2, 0))
    sc.add(sd_cone(0.07, 0.15, 0.12, 0.02), red, rig["torso"] * T(0, -0.02, 0))
    for sgn in (-1, 1):
        sc.add(sd_box(0.065, 0.05, 0.06, 0.015), red, rig["torso"] * T(0.19 * sgn, 0.29, 0) * RZ(-0.45 * sgn))
    fan = held(rig["r"], -math.pi if anim == "win" else -1.4)
    sc.add(sd_capsule((0, -0.02, 0), (0, 0.08, 0), 0.01), wood("#2a1a10"), fan)
    sc.add(sd_cylinder(0.006, 0.075, 0.003), Mat("#1a1a1a", rough=0.4, metal=0.6), fan * T(0, 0.15, 0) * RX(math.pi / 2))
    sc.add(sd_sphere(0.03), glow("#ffd24a", 0.6), fan * T(0, 0.15, 0.008))


def khalid_body(sc, rig, anim, t):
    robe = cloth("#2a5a3a", 0.85)
    sc.add(sd_cone(0.17, 0.15, 0.11, 0.02), robe, rig["torso"] * T(0, -0.1, 0))
    sc.add(sd_box(0.13, 0.26, 0.012, 0.012), cloth("#f0ece0", 0.85), rig["back"] * T(0, -0.22, -0.02) * RX(0.1))
    sc.add(sd_torus(0.105, 0.016), cloth("#c8a050", 0.6), rig["torso"] * T(0, 0.1, 0))
    sw = held(rig["r"], -math.pi if anim == "win" else -1.3)
    for k in range(5):          # a curved sabre
        a0, a1 = k * 0.12, (k + 1) * 0.12
        sc.add(sd_capsule((0.0 + 0.25 * (1 - math.cos(a0)) * 0.6, 0.06 + 0.25 * math.sin(a0) * 1.0, 0),
                          (0.0 + 0.25 * (1 - math.cos(a1)) * 0.6, 0.06 + 0.25 * math.sin(a1) * 1.0, 0), 0.012, 0.01), silver(0.15), sw)
    sc.add(sd_box(0.035, 0.008, 0.012, 0.004), gold(), sw * T(0, 0.05, 0))


def hannibal_body(sc, rig, anim, t):
    bronze = Mat("#b07a3a", metal=1.0, rough=0.3)
    sc.add(sd_ellipsoid(0.142, 0.12, 0.092), bronze, rig["torso"] * T(0, 0.21, 0.005))
    sc.add(sd_box(0.13, 0.24, 0.012, 0.012), cloth("#5a2a6a", 0.8), rig["back"] * T(0, -0.2, -0.02) * RX(0.1))
    sc.add(sd_cone(0.06, 0.13, 0.11, 0.02), cloth("#f0ece0", 0.8), rig["torso"] * T(0, -0.02, 0))
    sh = rig["l"] * RX(-1.5) * T(0, 0, 0.06)
    sc.add(sd_cylinder(0.02, 0.14, 0.01), bronze, sh * RX(math.pi / 2))
    sc.add(sd_sphere(0.04), bronze, sh * T(0, 0, 0.02))
    sp = held(rig["r"], -math.pi if anim == "win" else -1.5)
    sc.add(sd_capsule((0, -0.15, 0), (0, 0.32, 0), 0.008), wood("#4a3420"), sp)
    sc.add(sd_cone(0.04, 0.02, 0.002, 0.002), silver(0.2), sp * T(0, 0.36, 0))


def grant_body(sc, rig, anim, t):
    buttons(sc, rig, 6)
    coat_tails(sc, rig, "#1c2440", 0.26)
    for sgn in (-1, 1):
        sc.add(sd_box(0.04, 0.008, 0.025, 0.004), Mat("#c8a050", metal=0.8, rough=0.3), rig["torso"] * T(0.15 * sgn, 0.315, 0))
    # the cigar, a wisp of smoke
    x, r = rig["head"], rig["hr"]
    sc.add(sd_capsule((0, 0, 0), (r * 0.5, -r * 0.08, r * 0.25), r * 0.08), Mat("#5a3a1a", rough=0.7), x * T(r * 0.2, -r * 0.5, r * 0.88))
    sc.add(sd_sphere(r * 0.07), glow("#ff6a2a", 4.0), x * T(r * 0.72, -r * 0.6, r * 1.12))
    for k in range(3):
        ph = (t + k / 3) % 1
        sc.add(sd_sphere(r * (0.08 + 0.15 * ph)), Mat("#c8c8c8", rough=0.9), x * T(r * (0.75 + 0.2 * ph), r * (-0.4 + 1.4 * ph), r * 1.12))


def frederick_body(sc, rig, anim, t):
    sc.add(sd_box(0.045, 0.1, 0.012, 0.012), cloth("#e8d070", 0.7), rig["chest"] * T(0, -0.01, -0.004))       # the yellow waistcoat
    buttons(sc, rig, 4)
    coat_tails(sc, rig, "#1c2a5a", 0.24)
    for sgn in (-1, 1):
        sc.add(sd_box(0.025, 0.12, 0.012, 0.01), cloth("#a8242a", 0.75), rig["chest"] * T(0.068 * sgn, -0.02, -0.006) * RZ(0.1 * sgn))
    sc.add(sd_sphere(0.014), glow("#ffa040", 0.6), rig["chest"] * T(-0.06, 0.04, 0.0))           # the Black Eagle's star, gilded
    cn = held(rig["r"], -math.pi if anim == "win" else -2.6)
    sc.add(sd_capsule((0, -0.02, 0), (0, 0.3, 0), 0.009), wood("#3a2a1a"), cn)
    sc.add(sd_sphere(0.018), gold(), cn * T(0, -0.02, 0))


def zhukov_body(sc, rig, anim, t):
    for sgn in (-1, 1):
        sc.add(sd_box(0.05, 0.008, 0.028, 0.005), gold(), rig["torso"] * T(0.15 * sgn, 0.315, 0))     # shoulder boards
    for row in range(4):     # rows of medals and ribbons
        for k in range(4):
            col = ["#c8242a", "#e8c040", "#2a6ac8", "#2a8a4a"][(row + k) % 4]
            sc.add(sd_box(0.009, 0.006, 0.003, 0.0015), flat(col), rig["chest"] * T(-0.075 + 0.018 * k, 0.06 - 0.016 * row, 0.0))
    for k in range(3):
        sc.add(sd_cone(0.006, 0.012, 0.003, 0.002), gold(), rig["chest"] * T(0.03 + 0.024 * k, 0.06, 0.004) * RX(-1.4))
    buttons(sc, rig, 4)


def alexander_body(sc, rig, anim, t):
    bronze = Mat("#c8903a", metal=1.0, rough=0.25)
    sc.add(sd_ellipsoid(0.142, 0.12, 0.092), bronze, rig["torso"] * T(0, 0.21, 0.005))
    sc.add(sd_ellipsoid(0.11, 0.08, 0.085), bronze, rig["torso"] * T(0, 0.1, 0.003), blend=0.03)
    sc.add(sd_box(0.13, 0.24, 0.012, 0.012), cloth("#a8242a", 0.8), rig["back"] * T(0, -0.2, -0.02) * RX(0.1))
    for k in range(9):
        a = -1.6 + 0.4 * k
        sc.add(sd_box(0.018, 0.05, 0.006, 0.004), cloth("#f0ece0", 0.8), rig["torso"] * T(0.11 * math.sin(a), -0.015, 0.085 * math.cos(a)) * RY(a))
    sw = held(rig["r"], -math.pi if anim == "win" else -1.3)
    sc.add(sd_box(0.018, 0.14, 0.004, 0.003), silver(0.15), sw * T(0, 0.18, 0))
    sc.add(sd_box(0.035, 0.008, 0.012, 0.004), gold(), sw * T(0, 0.04, 0))


def pose_still(p, anim, t):
    if anim not in ("win",):
        p.l = [0.15, 0.2, 0.5]


G = {
    "alphazero": dict(name="Napoléon", stars=10, mats=coat_mats("#2a4a2a", trousers="#f0ece0"), w=0.95, h=0.92,
                      hair="#2a1e16", nose=0.95, iris="#5a6a7a", pose=pose_hand_in_waistcoat,
                      head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#2a1e16", front=0.15, back=-0.5), bicorne(sc, x, r, True)),
                      body=lambda sc, rig, a, t: (napoleon_body(sc, rig, a, t), epaulettes(sc, rig))),
    "dev": dict(name="Jules César", stars=9, mats=coat_mats("#a8242a", trousers="#e8b49a", boots="#5a3a1a"), w=0.95,
                hair="#3a2e26", nose=1.15, cheeks=0.85,
                head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#3a2e26", front=0.55, back=-0.4, bulk=1.03), laurel(sc, x, r)),
                body=caesar_body),
    "resnet": dict(name="Wellington", stars=8, mats=coat_mats("#1c2236", trousers="#e8e2d0"), w=0.95, h=1.05, hair="#8a8278", nose=1.45,
                   cheeks=0.85, head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#8a8278", front=0.25, back=-0.5), bicorne(sc, x, r, False, cockade=False)),
                   body=wellington_body),
    "features": dict(name="Takeda Shingen", stars=7, mats=coat_mats("#2a2a2a", trousers="#2a2a2a", boots="#1a1a1a"), w=1.1, hair="#1a1414",
                     head=lambda sc, x, r, a, t: (beard(sc, x, r, "#1a1414", full=True, moustache=True), kabuto(sc, x, r)),
                     body=shingen_body),
    "nnue193": dict(name="Khalid ibn al-Walid", stars=6, mats=coat_mats("#2a5a3a", trousers="#e8e2d0", boots="#5a3a1a"), skin="#c8946a", hair="#1a1210",
                    head=lambda sc, x, r, a, t: (beard(sc, x, r, "#1a1210"), turban(sc, x, r)), body=khalid_body),
    "basic_minmax": dict(name="Hannibal", stars=5, mats=coat_mats("#5a2a6a", trousers="#e8b49a", boots="#5a3a1a"), skin="#c8946a", hair="#1a1210", patch=True,
                         head=lambda sc, x, r, a, t: (beard(sc, x, r, "#1a1210"), crested_helm(sc, x, r)), body=hannibal_body),
    "mcts": dict(name="Ulysses Grant", stars=4, mats=coat_mats("#1c2440", trousers="#1c2440"), hair="#5a3a22",
                 head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#5a3a22", front=0.3), beard(sc, x, r, "#5a3a22"), slouch_hat(sc, x, r)),
                 body=grant_body),
    "monte_carlo": dict(name="Frédéric le Grand", stars=3, mats=coat_mats("#1c2a5a", trousers="#e8d070"), hair="#e8e4dc", nose=1.2, cheeks=0.8,
                        head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#e8e4dc", front=0.25, back=-0.7, bulk=1.1),
                                                     sc.add(sd_capsule((0, 0, 0), (0, -r * 0.7, -r * 0.25), r * 0.18, r * 0.09), flat("#1a1a1a"), x * T(0, -r * 0.1, -r * 1.05)),
                                                     tricorne(sc, x, r)),
                        body=frederick_body),
    "beginner": dict(name="Gueorgui Joukov", stars=2, mats=coat_mats("#5a5a46", trousers="#2a2a36"), w=1.12, hair="#6a5a4a", cheeks=1.15,
                     head=lambda sc, x, r, a, t: marshal_cap(sc, x, r), body=zhukov_body),
    "random": dict(name="Alexandre le Grand", stars=1, mats=coat_mats("#a8242a", trousers="#e8b49a", boots="#5a3a1a"), w=0.92, hair="#c8a050", iris="#4a7a8a",
                   head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#c8a050", front=0.4, back=-0.7, bulk=1.05, curls=True), diadem(sc, x, r)),
                   body=alexander_body),
}


def marshal_user(anim, t, side):
    """The user as a marshal: the buzz cut and the stubble of the portrait,
    a black or white coat by side, gold epaulettes, a bicorne."""
    import models
    coat = "#1a1a20" if side == 0 else "#ece8e0"
    g = dict(name="Vous", stars=0, mats=coat_mats(coat, trousers="#f0ece0" if side == 0 else "#2a2a36"), hair="#3d342e",
             head=lambda sc, x, r, a, t2: (hair_cap(sc, x, r, "#3d342e", front=0.45, back=-0.4, bulk=1.02),
                                            sc.add(sd_plane_cut(sd_ellipsoid(r * 0.78, r * 0.6, r * 0.7), (0, 1, 0.25), -r * 0.12),
                                                   Mat("#a8907e", rough=0.8), x * T(0, -r * 0.5, r * 0.24), blend=r * 0.03),
                                            bicorne(sc, x, r, True)),
             body=lambda sc, rig, a, t2: (epaulettes(sc, rig), buttons(sc, rig, 5), coat_tails(sc, rig, coat, 0.22)), iris="#7f97a6")
    return general(anim, t, side, g)


ORDER = ["random", "beginner", "monte_carlo", "mcts", "basic_minmax", "nnue193", "features", "resnet", "dev", "alphazero"]
GENERALS = {"gen_" + k: (lambda k: lambda anim, t, side: general(anim, t, side, G[k]))(k) for k in ORDER}
GENERALS["gen_human"] = marshal_user
NAMES = {"gen_" + k: G[k]["name"] for k in ORDER}
NAMES["gen_human"] = "Vous (maréchal)"
