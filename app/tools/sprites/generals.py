"""
generals.py — the "Généraux" set: the ten best generals of Ethan Arsht's
statistical ranking (WAR, 3,580 battles, 2020; "all in good fun" in its
author's words) and two great captains outside it (Genghis Khan, Vercingetorix,
for the AIs added later), one per AI, the strongest AI the best general — Napoleon
for AlphaZero — and the user as a marshal.

Likeness at the size of a square comes from silhouettes and attributes more
than from faces: each one has the hat, the clothes, the beard and the object
people know him by (Napoleon's bicorne worn sideways and his hand in his
waistcoat, Caesar's laurel, Wellington's bicorne worn front to back and his
nose, Shingen's horned helmet with a white mane and his war fan, Khalid's
turban over mail, Hannibal's crested helmet and lost eye, Grant's beard and
cigar, Frederick's tricorne and cane, Zhukov's cap and medals, Alexander's
curls and diadem). Gold stars over the name plaque tell the rank (STARS).
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, sd_plane_cut, fbm, length
from models import base, BASE_TOP, TAU, wave, glow, gold, eye_mat
from legends import base_pose, held, leather, cloth, iron, silver, wood, skin
from heroic import HEAD

GOLD = gold


def flat(c, rough=0.6, metal=0.0):
    return Mat(c, rough=rough, metal=metal)


# ─── faces ──────────────────────────────────────────────────────────────────
def aged_skin(c, age=0.0):
    """Skin with pores and, with age, lines on the forehead and at the corners of the eyes."""
    base_c = srgb(c)

    def pat(pl, pw, n):
        a = base_c * (0.88 + 0.2 * fbm(pl * 120, 2))[:, None]
        if age > 0:
            lines = (torch.sin(pl[:, 1] * 900) > 0.8).float() * (pl[:, 2] > 0.02).float() * (pl[:, 1] > 0.012).float()
            a = a * (1 - 0.18 * age * lines)[:, None]
        return {"albedo": a}
    return Mat(c, rough=0.5, sss=0.45, pattern=pat)


def face(sc, x, r, sk_c, anim, t, hair_c="#3a2a20", nose=1.0, brow_c=None, eye_patch=False, iris="#5a6a7a", cheeks=1.0,
         jaw=1.0, long=1.0, aquiline=0.0, brow=1.0, age=0.0, lips="#a86a5e"):
    """A man's face built like a sculptor's: cranium, the mass of the face, the
    jaw and chin, cheekbones, the brow ridge over sunk eyes with lids, a nose
    with its bridge, tip and wings, two lips, ears. jaw: width of the jaw;
    long: length of the face; aquiline: the bump of the nose; age: lines."""
    sk = aged_skin(sk_c, age)
    sc.add(sd_ellipsoid(r * 0.95, r * 1.04, r * 1.0), sk, x * T(0, r * 0.1, -r * 0.06))                           # cranium
    sc.add(sd_ellipsoid(r * 0.7 * cheeks, r * 0.72 * long, r * 0.68), sk, x * T(0, -r * 0.28 * long, r * 0.22), blend=r * 0.25)
    sc.add(sd_ellipsoid(r * 0.5 * jaw, r * 0.3, r * 0.42), sk, x * T(0, -r * 0.8 * long, r * 0.36), blend=r * 0.22)   # jaw, chin
    sc.add(sd_ellipsoid(r * 0.16, r * 0.14, r * 0.1), sk, x * T(0, -r * 0.92 * long, r * 0.7), blend=r * 0.1)
    for sgn in (-1, 1):
        sc.add(sd_sphere(r * 0.2), sk, x * T(r * 0.5 * cheeks * sgn, -r * 0.1, r * 0.6), blend=r * 0.15)                 # cheekbones
        sc.add(sd_ellipsoid(r * 0.13, r * 0.25, r * 0.08), sk, x * T(r * 0.93 * sgn, -r * 0.05, -r * 0.05) * RY(0.3 * sgn), blend=r * 0.05)
    sc.add(sd_capsule((-r * 0.42, 0, 0), (r * 0.42, 0, 0), r * 0.085 * brow), sk, x * T(0, r * 0.3, r * 0.8), blend=r * 0.12)  # brow ridge
    # the nose
    nz = r * (0.9 + 0.1 * nose)
    sc.add(sd_capsule((0, r * 0.25, r * 0.82), (0, -r * 0.2, nz + r * 0.14 * nose), r * 0.06, r * 0.085 * nose), sk, x, blend=r * 0.06)
    if aquiline > 0:
        sc.add(sd_sphere(r * 0.075 * nose), sk, x * T(0, r * 0.07, nz + r * 0.03 + r * 0.03 * aquiline), blend=r * 0.05)
    for sgn in (-1, 1):
        sc.add(sd_sphere(r * 0.06 * nose), sk, x * T(r * 0.09 * sgn, -r * 0.22, nz + r * 0.04), blend=r * 0.04)
    # the eyes, sunk under the brow; lids
    look = 0.2 * math.sin(TAU * t) if anim == "idle" else 0.0
    closed = anim == "win" or (anim == "idle" and abs(t - 0.6) < 0.04)
    bc = brow_c or hair_c
    for sgn in (-1, 1):
        ec = x * T(r * 0.31 * sgn, r * 0.12, r * 0.7)
        if eye_patch and sgn < 0:
            sc.add(sd_ellipsoid(r * 0.2, r * 0.17, r * 0.06), flat("#141414", 0.5), ec * T(0, 0, r * 0.14))
            sc.add(sd_capsule((-r * 0.9, r * 0.5, -r * 0.6), (r * 0.9, -r * 0.1, r * 0.08), r * 0.022), flat("#141414", 0.5), ec)
        elif closed:
            sc.add(sd_sphere(r * 0.13), sk, ec, blend=r * 0.03)
        else:
            sc.add(sd_sphere(r * 0.12), eye_mat((look, 0.0, 1.0), iris=iris), ec)
            sc.add(sd_plane_cut(sd_sphere(r * 0.135), (0, -1, 0.15), -r * 0.035), sk, ec, blend=r * 0.015)    # upper lid
            sc.add(sd_plane_cut(sd_sphere(r * 0.128), (0, 1, 0.1), -r * 0.075), sk, ec, blend=r * 0.015)      # lower lid
        tilt = {"blocked": 0.25, "move": -0.2}.get(anim, -0.08)
        sc.add(sd_capsule((0, 0, 0), (r * 0.3 * sgn, -tilt * r * 0.4, -r * 0.06), r * 0.04, r * 0.025), flat(bc, 0.8),
               x * T(r * 0.12 * sgn, r * 0.36, r * 0.86))
    # the mouth
    mo = x * T(0, -r * 0.47 * long - r * 0.05, r * 0.8)
    lm = Mat(lips, rough=0.45, sss=0.4)
    if anim == "win":
        sc.add(sd_ellipsoid(r * 0.18, r * 0.09, r * 0.1), flat("#3a1414", 0.6), mo)
    sc.add(sd_ellipsoid(r * 0.19, r * 0.035, r * 0.06), lm, mo * T(0, r * 0.035, r * 0.02), blend=r * 0.02)
    sc.add(sd_ellipsoid(r * 0.16, r * 0.045, r * 0.06), lm, mo * T(0, -r * 0.04 - (r * 0.06 if anim == "win" else 0), r * 0.01), blend=r * 0.02)


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
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.9, r * 1.1, r * 0.38), (0, -1, 0), 0.0), felt, h)
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
    sc.add(lambda p: sd_ellipsoid(r * 0.95, r * 2.0, r * 0.7)(p) + 0.015 * torch.sin(p[:, 0] * 150), mane, h * T(0, -r * 0.9, -r * 0.75) * RX(0.15))   # the white mane down the back


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


# ─── cloth, metal, orders ───────────────────────────────────────────────────
def wool(c, rough=0.85):
    """Broadcloth: a fine weave, and soft vertical folds."""
    base_c = srgb(c)

    def pat(pl, pw, n):
        fold = torch.sin(pl[:, 0] * 95 + pl[:, 1] * 12 + 5 * fbm(pl * 7, 2))
        return {"albedo": base_c * (0.86 + 0.12 * fbm(pl * 70, 2) + 0.07 * fold)[:, None]}
    return Mat(c, rough=rough, pattern=pat)


def lace():
    """Gold lace and embroidery: metal threads."""
    return Mat("#d8b050", metal=1.0, rough=0.35, pattern=lambda pl, pw, n: {
        "albedo": srgb("#d8b050") * (0.75 + 0.35 * (0.5 + 0.5 * torch.sin((pl[:, 0] + pl[:, 1]) * 700)))[:, None]})


def boot_leather(c="#121214"):
    return Mat(c, rough=0.22, clearcoat=0.9, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.85 + 0.25 * fbm(pl * 40, 2))[:, None]})


def order_star(sc, xf, R, mat, centre=None, points=8):
    """A breast star of an order: rays of thin plates around a round centre."""
    for k in range(points // 2):
        sc.add(sd_box(R, R * 0.16, R * 0.08, R * 0.05), mat, xf * RZ(math.pi * k / (points // 2)))
    if centre:
        sc.add(sd_cylinder(R * 0.12, R * 0.38, R * 0.05), centre, xf * RX(math.pi / 2) * T(0, R * 0.06, 0))


def medal(sc, xf, ribbon, metal):
    """A medal hanging from its ribbon."""
    sc.add(sd_box(0.008, 0.006, 0.002, 0.0015), flat(ribbon, 0.7), xf)
    sc.add(sd_cylinder(0.0025, 0.007, 0.001), metal, xf * T(0, -0.013, 0.002) * RX(math.pi / 2))


# ─── the rig: a man of eight heads, slim limbs ─────────────────────────────
def officer(sc, p, d, H=1.0):
    """Legs (breeches or trousers, boots or sandals), the body under the coat,
    sleeves with cuffs, hands, neck; and the coat's parts: standing collar,
    lapels or plastron, the waistcoat, the skirts (tails, frock, robe or
    pteruges), a sash, a belt, epaulettes, buttons. d: the dress (colours and
    styles). Returns the frames (head, hands, torso, chest, back) and the head radius."""
    w = d.get("w", 1.0)
    coat = wool(d["coat"])
    legs = wool(d.get("legs", "#f0ece0"), 0.8)
    boots = boot_leather(d.get("boots", "#121214"))
    sk = aged_skin(d.get("skin", "#e2b096"), 0.0)
    y0 = BASE_TOP
    body = RY(HEAD + p.shake) * T(0, p.bob + p.jump, 0)
    thigh, shin = 0.24 * H, 0.225 * H
    hip_y = y0 + thigh + shin + 0.03 * H
    style = d.get("boot", "tall")
    for sgn in (-1, 1):
        s_ = p.step * sgn * 0.6
        th = body * T(0.056 * w * sgn, hip_y, 0) * RX(-0.6 * s_)
        sc.add(sd_capsule((0, 0, 0), (0, -thigh, 0.01), 0.047 * w, 0.032), legs, th)
        kn = th * T(0, -thigh, 0.01) * RX(0.6 * max(0.0, s_))
        lower = boots if style == "tall" else (legs if style in ("shoe", "trousers") else sk)
        sc.add(sd_capsule((0, 0, 0), (0, -shin, -0.004), 0.033, 0.024), lower, kn)
        sc.add(sd_ellipsoid(0.03, 0.055 * H, 0.028), lower, kn * T(0, -shin * 0.3, -0.01), blend=0.02)          # calf
        if style == "tall":   # the top of the boot, cut higher in front
            sc.add(sd_plane_cut(sd_cylinder(0.02, 0.038, 0.004), (0, -0.35, -1), 0.0), boots, kn * T(0, 0.01, 0.0))
        if style == "sandal":
            for k in range(3):
                sc.add(sd_torus(0.027 - 0.002 * k, 0.003), leather("#4a2e18"), kn * T(0, -shin * (0.55 + 0.17 * k), 0))
        foot = kn * T(0, -shin - 0.01, 0.026)
        fm = sk if style == "sandal" else boots
        sc.add(sd_box(0.026, 0.017, 0.058, 0.014), fm, foot)
        if style == "sandal":
            sc.add(sd_box(0.03, 0.004, 0.064, 0.003), leather("#4a2e18"), foot * T(0, -0.016, 0))
        else:
            sc.add(sd_box(0.028, 0.005, 0.062, 0.004), Mat("#0a0a0a", rough=0.5), foot * T(0, -0.015, 0))        # the sole
    torso = body * T(0, hip_y, 0) * RX(p.lean)
    # a man's trunk: narrow hips, a straight waist, a broad flat chest wider at the shoulders
    sc.add(sd_ellipsoid(0.084 * w, 0.068 * H, 0.062), legs, torso * T(0, 0.02 * H, 0))
    sc.add(sd_box(0.08 * w, 0.07 * H, 0.056, 0.045), coat, torso * T(0, 0.13 * H, 0.0), blend=0.03)
    sc.add(lambda q: sd_box(0.105 * w, 0.085 * H, 0.06, 0.05)(q * vec(1.0 - 0.6 * 0, 1.0, 1.0)) - 0.012 * (q[:, 1] / (0.1 * H)).clamp(-1, 1),
           coat, torso * T(0, 0.235 * H, 0.004), blend=0.035)
    sh_y = 0.3 * H
    cuff = wool(d.get("cuffs", d["coat"]))
    hands = []
    for sgn, (pitch, spread, elbow) in ((-1, p.l), (1, p.r)):
        sc.add(sd_sphere(0.042 * w), coat, torso * T(0.12 * w * sgn, sh_y - 0.01, 0), blend=0.02)
        up, fo = 0.175 * H, 0.16 * H
        sh = torso * T(0.132 * w * sgn, sh_y - 0.012, 0) * RZ(spread * sgn) * RX(-pitch)
        bare = d.get("sleeve", "coat") == "bare"
        sc.add(sd_capsule((0, 0, 0), (0, -up, 0), 0.032 if bare else 0.034, 0.026 if bare else 0.028), sk if bare else coat, sh)
        if bare:   # the short sleeve of the tunic
            sc.add(sd_capsule((0, 0.01, 0), (0, -up * 0.4, 0), 0.04, 0.035), wool(d.get("tunic", "#f0ece0")), sh)
        el = sh * T(0, -up, 0) * RX(-elbow)
        sc.add(sd_capsule((0, 0, 0), (0, -fo, 0), 0.026 if bare else 0.028, 0.021 if bare else 0.024), sk if bare else coat, el, blend=0.012)
        if d.get("sleeve", "coat") == "coat":
            sc.add(sd_cylinder(0.018, 0.028, 0.004), cuff, el * T(0, -fo + 0.02, 0))
        hand = el * T(0, -fo - 0.024, 0)
        sc.add(sd_ellipsoid(0.021, 0.028, 0.014), sk, hand)
        sc.add(sd_capsule((0, 0.01, 0.01), (0.012 * sgn, -0.012, 0.018), 0.007), sk, hand, blend=0.004)       # the thumb
        hands.append(hand)
    nk = 0.055 * H
    hr = 0.058 * H
    hy = max(sh_y + nk + hr * 0.95, 0.335 * H + hr * 0.75)
    sc.add(sd_capsule((0, sh_y - 0.03, 0), (0, hy - hr * 0.7, 0.01), 0.027, 0.024), sk, torso)
    hx = torso * T(0, hy, 0.012) * RY(p.yaw) * RX(p.pitch - 0.1)
    rig = {"head": hx, "hr": hr, "l": hands[0], "r": hands[1], "torso": torso, "body": body,
           "chest": torso * T(0, 0.235 * H, 0.078 * 1.0), "back": torso * T(0, 0.24 * H, -0.078),
           "waist": torso * T(0, 0.1 * H, 0), "hip_y": hip_y, "H": H, "bulk": w, "sh_y": sh_y, "coat": d["coat"]}
    # the coat's parts
    if "collar" in d:
        sc.add(sd_plane_cut(sd_cylinder(0.022, 0.036, 0.006), (0, 0, 1), 0.026), wool(d["collar"]), torso * T(0, sh_y + 0.028, 0.004))
        if d.get("collar_lace"):
            sc.add(sd_plane_cut(sd_torus(0.036, 0.004), (0, 0, 1), 0.026), lace(), torso * T(0, sh_y + 0.046, 0.004))
    if "vest" in d:
        sc.add(sd_box(0.042 * w, 0.075 * H, 0.01, 0.008), wool(d["vest"], 0.75), torso * T(0, 0.17 * H, 0.068))
    if "lapels" in d:   # the plastron: a shell over the chest, wider at the top
        lw = d.get("lapel_w", 0.058)
        shell = sd_box(0.11 * w, 0.09 * H, 0.066, 0.052)
        front = lambda q: torch.maximum(torch.maximum(q[:, 0].abs() - lw * (0.75 + 0.25 * (q[:, 1] / (0.1 * H)).clamp(-1, 1)), q[:, 1].abs() - 0.085 * H), 0.03 - q[:, 2])
        sc.add(lambda q: torch.maximum(shell(q), front(q)), wool(d["lapels"]), torso * T(0, 0.235 * H, 0.006))
    for bx in d.get("buttons", ()):
        for k in range(d.get("nbuttons", 6)):
            sc.add(sd_sphere(0.0065), d.get("button_mat") or gold(), torso * T(bx * w, (0.29 - 0.034 * k) * H, 0.082 - 0.004 * k))
    sk_ = d.get("skirt")
    if sk_ == "tails":
        for sgn in (-1, 1):
            sc.add(sd_box(0.04, 0.13 * H, 0.008, 0.008), coat, torso * T(0.05 * sgn, -0.06 * H, -0.065) * RZ(0.06 * sgn) * RX(0.12))
            if "lapels" in d:
                sc.add(sd_box(0.012, 0.11 * H, 0.009, 0.006), wool(d["lapels"]), torso * T(0.075 * sgn, -0.06 * H, -0.062) * RX(0.12))
    elif sk_ in ("frock", "robe", "redingote"):
        ln = {"frock": 0.24, "redingote": 0.3, "robe": 0.44}[sk_] * H
        f = sd_cone(ln * 0.5, 0.1 * w + ln * 0.12, 0.092 * w, 0.006)
        shell = lambda q, f=f: torch.maximum(f(q), -(f(q) + 0.008))
        cut = 0.04 if sk_ != "robe" else 1.0
        sc.add(sd_plane_cut(shell, (0, 0, 1), cut), coat, torso * T(0, 0.08 * H - ln * 0.5, 0))
    elif sk_ == "pteruges":
        for k in range(11):
            a = -2.4 + 0.48 * k
            sc.add(sd_box(0.016, 0.05, 0.004, 0.003), leather(d.get("pteruges", "#e8dcc0")),
                   torso * T(0.1 * math.sin(a), -0.03 * H, 0.075 * math.cos(a)) * RY(a) * RX(-0.12))
        sc.add(sd_cone(0.06 * H, 0.106, 0.092, 0.01), wool(d.get("tunic", "#f0ece0")), torso * T(0, -0.05 * H, 0))
    if "sash" in d:   # over the right shoulder to the left hip
        ell = lambda q: sd_torus(0.1, 0.009)(q * vec(1.0, 1.0, 1.42)) / 1.42
        sc.add(sd_plane_cut(ell, (0, 1, 0), 0.08), wool(d["sash"], 0.5), torso * T(0, 0.2 * H, 0.004) * RZ(0.72))
    if "belt" in d:
        ell = lambda q: sd_torus(0.088 * w, 0.011)(q * vec(1.0, 1.0, 1.3)) / 1.3
        sc.add(ell, d["belt"] if isinstance(d["belt"], Mat) else wool(d["belt"], 0.5), torso * T(0, 0.075 * H, 0.004))
    if d.get("epaulettes"):
        for sgn in (-1, 1):
            e = torso * T(0.115 * w * sgn, sh_y + 0.025, 0)
            sc.add(sd_ellipsoid(0.045, 0.013, 0.035), lace(), e)
            if d["epaulettes"] == "fringe":
                for k in range(6):
                    sc.add(sd_capsule((0, 0, 0), (0.004 * sgn, -0.035, 0), 0.0045), lace(), e * T(0.04 * sgn, -0.004, -0.026 + 0.0105 * k))
    if d.get("boards"):   # Soviet shoulder boards
        for sgn in (-1, 1):
            sc.add(sd_box(0.05, 0.006, 0.024, 0.004), lace(), torso * T(0.095 * w * sgn, sh_y + 0.022, 0))
    return rig


def sword_hilt(sc, rig, side=-1):
    """A sword at the hip: the scabbard and the guard."""
    x = rig["torso"] * T(0.09 * side, 0.05, -0.02) * RX(0.5)
    sc.add(sd_capsule((0, 0, 0), (0, -0.32, 0), 0.01, 0.008), boot_leather("#141414"), x)
    sc.add(sd_capsule((0, 0, 0), (0, 0.06, 0), 0.006), gold(), x)
    sc.add(sd_torus(0.018, 0.003), gold(), x * T(0, 0.01, 0) * RX(math.pi / 2))


# ─── the generals ───────────────────────────────────────────────────────────
def general(anim, t, side, g):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    if "pose" in g:
        g["pose"](p, anim, t)
    d = dict(g["dress"])
    d.setdefault("skin", g.get("skin", "#e2b096"))
    rig = officer(sc, p, d, H=g.get("h", 1.0))
    r, x = rig["hr"], rig["head"]
    face(sc, x, r, d["skin"], anim, t, g.get("hair", "#3a2a20"), g.get("nose", 1.0), eye_patch=g.get("patch", False),
         iris=g.get("iris", "#5a6a7a"), cheeks=g.get("cheeks", 1.0), jaw=g.get("jaw", 1.0), long=g.get("long", 1.0),
         aquiline=g.get("aquiline", 0.0), brow=g.get("brow", 1.0), age=g.get("age", 0.0))
    if "head" in g:
        g["head"](sc, x, r, anim, t)
    if "body" in g:
        g["body"](sc, rig, anim, t)
    return sc


def pose_hand_in_waistcoat(p, anim, t):
    if anim != "win":
        p.r = [0.55, -0.6, 1.65]          # the right hand in the waistcoat
    p.l = [0.1, 0.1, 0.4] if anim != "win" else [2.45, 0.25, 0.15]


def pose_still(p, anim, t):
    if anim not in ("win",):
        p.l = [0.15, 0.12, 0.5]


def napoleon_body(sc, rig, anim, t):
    # after David (1812): the Legion of Honour's star and cross, the sword at the hip
    order_star(sc, rig["chest"] * T(-0.055, -0.03, 0.006), 0.022, silver(0.25), flat("#c8242a", 0.4))
    sc.add(sd_box(0.006, 0.006, 0.002), flat("#c8242a", 0.6), rig["chest"] * T(-0.035, 0.035, 0.008))
    sc.add(sd_cylinder(0.002, 0.008, 0.001), silver(0.25), rig["chest"] * T(-0.035, 0.022, 0.009) * RX(math.pi / 2))
    sword_hilt(sc, rig, -1)
    if anim == "win":
        sc.add(sd_capsule((0, -0.02, 0), (0, 0.16, 0), 0.014), Mat("#2a2420", rough=0.3, metal=0.5), held(rig["l"], -math.pi))   # spyglass


def caesar_body(sc, rig, anim, t):
    # a muscled cuirass, the general's red cloak over the left shoulder, the gladius
    cuirass = Mat("#b08a4a", metal=1.0, rough=0.3)
    sc.add(sd_box(0.113, 0.095, 0.07, 0.055), cuirass, rig["torso"] * T(0, 0.23, 0.004))
    sc.add(sd_box(0.088, 0.07, 0.064, 0.045), cuirass, rig["torso"] * T(0, 0.12, 0.002), blend=0.03)
    for sgn in (-1, 1):   # the modelled pectorals
        sc.add(sd_ellipsoid(0.04, 0.022, 0.012), cuirass, rig["torso"] * T(0.042 * sgn, 0.26, 0.066), blend=0.015)
    cloak = wool("#9a1e24", 0.8)
    sc.add(sd_box(0.12, 0.25, 0.01, 0.01), cloak, rig["back"] * T(0, -0.21, -0.016) * RX(0.08))
    sc.add(sd_ellipsoid(0.06, 0.03, 0.08), cloak, rig["torso"] * T(-0.1, 0.31, 0.0), blend=0.02)
    sc.add(sd_box(0.03, 0.15, 0.012, 0.01), cloak, rig["torso"] * T(-0.105, 0.18, 0.06) * RZ(0.1))
    sw = held(rig["r"], -math.pi if anim == "win" else -1.3)
    sc.add(sd_box(0.014, 0.12, 0.003, 0.0025), silver(0.15), sw * T(0, 0.16, 0))
    sc.add(sd_box(0.03, 0.007, 0.012, 0.004), gold(), sw * T(0, 0.04, 0))


def wellington_body(sc, rig, anim, t):
    # after Lawrence (1815): the scarlet coat laced with gold, the sash, the stars of the Bath and the Garter
    order_star(sc, rig["chest"] * T(-0.06, -0.04, 0.006), 0.02, silver(0.25), flat("#2a3a8a", 0.4))
    order_star(sc, rig["chest"] * T(-0.06, -0.085, 0.004), 0.016, silver(0.25), flat("#c8242a", 0.4))
    for k in range(5):   # gold frogging across the chest
        sc.add(sd_box(0.05, 0.003, 0.003, 0.002), lace(), rig["chest"] * T(0.0, 0.05 - 0.028 * k, 0.012))
    sc.add(sd_ellipsoid(0.022, 0.016, 0.014), wool("#f2f0ea", 0.7), rig["torso"] * T(0, rig["sh_y"] + 0.03, 0.035))     # the stock
    sg = held(rig["r"], -math.pi if anim == "win" else -1.6)
    sc.add(sd_capsule((0, -0.02, 0), (0, 0.14, 0), 0.012, 0.016), Mat("#c8a050", metal=1.0, rough=0.3), sg)
    sword_hilt(sc, rig, -1)


def shingen_body(sc, rig, anim, t):
    # lacquered lamellar armour laced in red, shoulder guards (sode), the war fan (gunbai) with the Takeda diamonds
    lam = Mat("#7a1a1e", metal=0.2, rough=0.28, clearcoat=1.0,
              pattern=lambda pl, pw, n: {"albedo": srgb("#7a1a1e") * (0.7 + 0.35 * (torch.frac(pl[:, 1] * 55) > 0.18).float())[:, None]
                                         * (0.8 + 0.25 * (torch.sin(pl[:, 0] * 420) > -0.6).float())[:, None]})
    sc.add(sd_box(0.118, 0.11, 0.075, 0.05), lam, rig["torso"] * T(0, 0.2, 0.004))
    for k in range(5):   # the tassets (kusazuri) in lacquered plates
        a = -2.0 + 1.0 * k
        sc.add(sd_box(0.05, 0.075, 0.008, 0.005), lam, rig["torso"] * T(0.1 * math.sin(a), -0.05, 0.075 * math.cos(a)) * RY(a) * RX(-0.15))
    for sgn in (-1, 1):   # wide hakama trousers
        sc.add(sd_cone(0.17, 0.065, 0.045, 0.01), wool("#2a2230"), rig["body"] * T(0.06 * sgn, rig["hip_y"] - 0.2, 0))
    for sgn in (-1, 1):
        sc.add(sd_box(0.055, 0.06, 0.05, 0.012), lam, rig["torso"] * T(0.16 * sgn, 0.25, 0) * RZ(-0.35 * sgn))
    sc.add(sd_ellipsoid(0.07, 0.012, 0.055), Mat("#141414", rough=0.3, clearcoat=1.0), rig["torso"] * T(0, rig["sh_y"] + 0.02, 0))
    fan = held(rig["r"], -math.pi if anim == "win" else -1.4)
    sc.add(sd_capsule((0, -0.03, 0), (0, 0.08, 0), 0.008), wood("#2a1a10"), fan)
    blade = fan * T(0, 0.15, 0) * RX(math.pi / 2)
    sc.add(sd_cylinder(0.004, 0.07, 0.002), Mat("#141414", rough=0.35, metal=0.4, clearcoat=0.6), blade)
    for k in range(4):   # the four diamonds of the Takeda (takeda-bishi)
        dx, dz = (0.018 if k % 2 else -0.018), (0.014 if k >= 2 else -0.014)
        sc.add(sd_box(0.012, 0.004, 0.009, 0.001), gold(), blade * T(dx, 0.002, dz) * RY(math.pi / 4))


def khalid_body(sc, rig, anim, t):
    mail = Mat("#8a9098", metal=0.9, rough=0.4, pattern=lambda pl, pw, n: {
        "albedo": srgb("#8a9098") * (0.55 + 0.6 * (torch.sin(pl[:, 0] * 420) * torch.sin(pl[:, 1] * 420) > 0).float())[:, None]})
    sc.add(sd_ellipsoid(0.124, 0.12, 0.08), mail, rig["torso"] * T(0, 0.19, 0.005))
    sc.add(sd_cone(0.1, 0.13, 0.105, 0.01), mail, rig["torso"] * T(0, -0.02, 0))
    sc.add(sd_box(0.12, 0.26, 0.01, 0.01), wool("#f0ece0", 0.85), rig["back"] * T(0, -0.22, -0.016) * RX(0.08))
    sw = held(rig["r"], -math.pi if anim == "win" else -1.3)
    for k in range(5):          # a curved sword
        a0, a1 = k * 0.1, (k + 1) * 0.1
        sc.add(sd_capsule((0.15 * (1 - math.cos(a0)), 0.06 + 0.25 * math.sin(a0), 0),
                          (0.15 * (1 - math.cos(a1)), 0.06 + 0.25 * math.sin(a1), 0), 0.01, 0.008), silver(0.15), sw)
    sc.add(sd_box(0.03, 0.007, 0.012, 0.004), gold(), sw * T(0, 0.05, 0))


def hannibal_body(sc, rig, anim, t):
    bronze = Mat("#a8763a", metal=1.0, rough=0.3)
    sc.add(sd_box(0.113, 0.095, 0.07, 0.055), bronze, rig["torso"] * T(0, 0.23, 0.004))
    sc.add(sd_box(0.088, 0.07, 0.064, 0.045), bronze, rig["torso"] * T(0, 0.12, 0.002), blend=0.03)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.04, 0.022, 0.012), bronze, rig["torso"] * T(0.042 * sgn, 0.26, 0.066), blend=0.015)
    sc.add(sd_box(0.12, 0.25, 0.01, 0.01), wool("#4a1e5a", 0.8), rig["back"] * T(0, -0.21, -0.016) * RX(0.08))
    sh = rig["body"] * T(-0.19, rig["hip_y"] + 0.12, 0.11) * RY(-0.6)     # the round shield on the left arm, facing forward
    sc.add(sd_cylinder(0.012, 0.13, 0.006), bronze, sh * RX(math.pi / 2))
    sc.add(sd_torus(0.126, 0.009), bronze, sh * RX(math.pi / 2) * T(0, 0.012, 0))
    sc.add(sd_sphere(0.03), bronze, sh * T(0, 0, 0.012))
    sp = held(rig["r"], -math.pi if anim == "win" else -1.5)
    sc.add(sd_capsule((0, -0.15, 0), (0, 0.34, 0), 0.007), wood("#4a3420"), sp)
    sc.add(sd_cone(0.04, 0.018, 0.002, 0.002), silver(0.2), sp * T(0, 0.38, 0))


def grant_body(sc, rig, anim, t):
    for sgn in (-1, 1):   # the shoulder straps of a lieutenant general, three stars
        st = rig["torso"] * T(0.1 * sgn, rig["sh_y"] + 0.02, 0)
        sc.add(sd_box(0.045, 0.004, 0.016, 0.003), Mat("#1a1e30", rough=0.7), st)
        sc.add(sd_box(0.047, 0.005, 0.018, 0.002), lace(), st * T(0, -0.001, 0))
    x, r = rig["head"], rig["hr"]
    sc.add(sd_capsule((0, 0, 0), (r * 0.5, -r * 0.08, r * 0.25), r * 0.07), Mat("#5a3a1a", rough=0.7), x * T(r * 0.15, -r * 0.52, r * 0.86))
    sc.add(sd_sphere(r * 0.06), glow("#ff6a2a", 4.0), x * T(r * 0.66, -r * 0.6, r * 1.1))
    for k in range(3):
        ph = (t + k / 3) % 1
        sc.add(sd_sphere(r * (0.07 + 0.14 * ph)), Mat("#c8c8c8", rough=0.9), x * T(r * (0.7 + 0.2 * ph), r * (-0.4 + 1.4 * ph), r * 1.1))


def frederick_body(sc, rig, anim, t):
    # the Black Eagle: its orange sash and silver star; the cane
    order_star(sc, rig["chest"] * T(-0.055, -0.03, 0.006), 0.022, silver(0.25), flat("#e88a2a", 0.4))
    cn = held(rig["r"], -math.pi if anim == "win" else -2.6)
    sc.add(sd_capsule((0, -0.02, 0), (0, 0.3, 0), 0.008), wood("#3a2a1a"), cn)
    sc.add(sd_sphere(0.016), gold(), cn * T(0, -0.02, 0))
    sword_hilt(sc, rig, -1)


def zhukov_body(sc, rig, anim, t):
    gold_m = Mat("#e8c050", metal=1.0, rough=0.25)
    for k in range(4):        # the four gold stars of Hero of the Soviet Union
        sc.add(sd_cone(0.003, 0.009, 0.004, 0.001), gold_m, rig["chest"] * T(-0.07 + 0.017 * k, 0.07, 0.006) * RX(-1.45))
    cols = ["#c8242a", "#e8c040", "#2a6ac8", "#2a8a4a", "#8a2a6a", "#e8e0d0"]
    for row in range(5):      # rows of ribbons and medals on both sides
        for k in range(4):
            for sgn in (-1, 1):
                xf = rig["chest"] * T(sgn * (0.034 + 0.017 * k), 0.045 - 0.015 * row, 0.002 - 0.002 * row)
                if row < 2:
                    sc.add(sd_box(0.0075, 0.005, 0.002, 0.0012), flat(cols[(row * 3 + k + (sgn > 0)) % 6], 0.7), xf)
                else:
                    medal(sc, xf, cols[(row + k) % 6], gold_m if (row + k) % 2 else silver(0.3))
    sc.add(sd_box(0.012, 0.012, 0.004, 0.002), Mat("#f0f4ff", metal=0.6, rough=0.1), rig["torso"] * T(0, rig["sh_y"] + 0.012, 0.045) * RZ(math.pi / 4))   # the Marshal's star


def alexander_body(sc, rig, anim, t):
    # after the mosaic: a linen corslet (linothorax) with the Gorgon's head, a red cloak, a kopis
    lin = wool("#e8dcbc", 0.8)
    sc.add(sd_box(0.112, 0.105, 0.07, 0.04), lin, rig["torso"] * T(0, 0.2, 0.004))
    sc.add(sd_box(0.114, 0.008, 0.072, 0.006), Mat("#a8242a", rough=0.6), rig["torso"] * T(0, 0.11, 0.004))
    for sgn in (-1, 1):   # the shoulder flaps of the corslet
        sc.add(sd_box(0.035, 0.006, 0.06, 0.004), lin, rig["torso"] * T(0.075 * sgn, rig["sh_y"] + 0.012, 0.0))
    gorgon = rig["chest"] * T(0, 0.0, 0.01)
    sc.add(sd_sphere(0.018), Mat("#c8a050", metal=1.0, rough=0.3), gorgon)
    for k in range(8):
        a = TAU * k / 8
        sc.add(sd_capsule((0, 0, 0), (0.028 * math.cos(a), 0.028 * math.sin(a), 0), 0.004, 0.002), Mat("#c8a050", metal=1.0, rough=0.3), gorgon)
    sc.add(sd_box(0.12, 0.24, 0.01, 0.01), wool("#a8242a", 0.8), rig["back"] * T(0, -0.2, -0.016) * RX(0.08))
    sw = held(rig["r"], -math.pi if anim == "win" else -1.3)
    sc.add(sd_box(0.016, 0.12, 0.003, 0.0025), silver(0.15), sw * T(0.006, 0.16, 0) * RZ(-0.08))
    sc.add(sd_box(0.03, 0.007, 0.012, 0.004), gold(), sw * T(0, 0.04, 0))


def mongol_hat(sc, x, r):
    """A Mongol hat: a rounded crown of dark blue silk, a wide upturned brim of
    brown fur, a red tassel and a knob on top."""
    furm = Mat("#6a4a2e", rough=0.95, pattern=lambda pl, pw, n: {"albedo": srgb("#6a4a2e") * (0.6 + 0.6 * fbm(pl * 140, 2))[:, None]})
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.0, r * 1.05, r * 1.0), (0, -1, 0), 0.0), wool("#1e2a5a", 0.6), x * T(0, r * 0.35, 0))
    sc.add(sd_torus(r * 1.02, r * 0.2), furm, x * T(0, r * 0.38, 0) * RX(-0.1))
    sc.add(sd_sphere(r * 0.12), gold(), x * T(0, r * 1.38, 0))
    sc.add(sd_cone(r * 0.35, r * 0.2, r * 0.02, 0.003), Mat("#b8241e", rough=0.8), x * T(0, r * 1.35, 0) * RX(math.pi) * T(0, r * 0.3, 0))


def winged_helm(sc, x, r):
    """The winged helmet of the 19th-century statues of Vercingetorix: a
    rounded iron cap, a bronze band, a white wing on each side."""
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.08, r * 1.02, r * 1.08), (0, -1, 0), -r * 0.05), iron("#7a7e86", 0.3), x * T(0, r * 0.22, 0))
    sc.add(sd_torus(r * 1.05, r * 0.07), Mat("#b07a3a", metal=1.0, rough=0.3), x * T(0, r * 0.2, 0))
    feather = Mat("#f2f0ea", rough=0.6)
    for sgn in (-1, 1):
        w = x * T(r * 1.0 * sgn, r * 0.4, -r * 0.05) * RZ(-0.6 * sgn)
        for k in range(6):
            a = 0.1 + 0.25 * k
            L = r * (1.7 - 0.18 * k)
            sc.add(sd_ellipsoid(r * 0.1, L * 0.5, r * 0.24), feather, w * RX(-a) * T(0, L * 0.5, 0), blend=r * 0.05)


def genghis_body(sc, rig, anim, t):
    # the deel: the robe crossed over to the right, a gold trim, a wide sash; a curved sabre and a composite bow on the back
    sw = held(rig["r"], -math.pi if anim == "win" else -1.3)
    for k in range(5):
        a0, a1 = k * 0.08, (k + 1) * 0.08
        sc.add(sd_capsule((0.15 * (1 - math.cos(a0)), 0.06 + 0.24 * math.sin(a0), 0),
                          (0.15 * (1 - math.cos(a1)), 0.06 + 0.24 * math.sin(a1), 0), 0.009, 0.007), silver(0.15), sw)
    sc.add(sd_box(0.028, 0.006, 0.012, 0.004), gold(), sw * T(0, 0.05, 0))


def vercingetorix_body(sc, rig, anim, t):
    # a tunic belted with bronze, a long cloak pinned on the shoulder with a fibula, a gold torc, a long Gaulish sword
    sc.add(sd_box(0.13, 0.28, 0.01, 0.01), wool("#2a4a32", 0.8), rig["back"] * T(0, -0.22, -0.018) * RX(0.08))   # the cloak
    sc.add(sd_sphere(0.016), gold(), rig["torso"] * T(-0.09, rig["sh_y"] - 0.01, 0.06))                        # the fibula
    sw = held(rig["r"], -math.pi if anim == "win" else -1.25)
    sc.add(sd_box(0.016, 0.24, 0.004, 0.003), silver(0.18), sw * T(0, 0.3, 0))
    sc.add(sd_box(0.03, 0.007, 0.012, 0.004), Mat("#b07a3a", metal=1.0, rough=0.3), sw * T(0, 0.05, 0))


def queue_wig(sc, x, r):
    """Frederick's powdered wig: rolls over the ears, the queue tied in black."""
    wig = Mat("#ece8e2", rough=0.7, pattern=lambda pl, pw, n: {"albedo": srgb("#ece8e2") * (0.82 + 0.25 * fbm(pl * 90, 2))[:, None]})
    hair_cap(sc, x, r, "#ece8e2", front=0.35, back=-0.7, bulk=1.06)
    for sgn in (-1, 1):
        sc.add(sd_capsule((0, 0, -r * 0.35), (0, 0, r * 0.25), r * 0.17), wig, x * T(r * 0.95 * sgn, r * 0.05, 0))
    sc.add(sd_capsule((0, 0, 0), (0, -r * 0.75, -r * 0.2), r * 0.15, r * 0.09), flat("#141414"), x * T(0, -r * 0.15, -r * 1.02))


def anastole(sc, x, r, c):
    """Alexander's hair: thick, tousled, the forelock swept up from the middle of the brow, sideburns."""
    hm = hair_cap(sc, x, r, c, front=0.45, back=-0.75, bulk=1.1, curls=True)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(r * 0.22, r * 0.3, r * 0.14), hm, x * T(r * 0.18 * sgn, r * 0.78, r * 0.62) * RZ(-0.5 * sgn), blend=r * 0.06)
        sc.add(sd_ellipsoid(r * 0.1, r * 0.4, r * 0.18), hm, x * T(r * 0.86 * sgn, -r * 0.25, r * 0.2), blend=r * 0.05)


G = {
    "alphazero": dict(name="Napoléon", plate="Napoléon", hair="#3a2a1e", nose=1.0, iris="#6a7a8a", jaw=1.05, cheeks=1.05, pose=pose_still,
                      dress=dict(coat="#1e2a5c", lapels="#f2f0ea", collar="#1e2a5c", cuffs="#b8242a", vest="#f2f0ea", legs="#f2f0ea",
                                 skirt="tails", epaulettes="fringe", buttons=(-0.03, 0.03), nbuttons=6),
                      head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#3a2a1e", front=0.4, back=-0.15, bulk=1.02), bicorne(sc, x, r, True)),
                      body=napoleon_body),
    "dev": dict(name="Jules César", plate="César", hair="#4a3e34", nose=1.15, cheeks=0.82, long=1.12, jaw=0.9, aquiline=0.6, age=1.0, brow=1.2,
                dress=dict(coat="#9a1e24", legs="#e8b49a", boot="sandal", skirt="pteruges", pteruges="#7a4a24", tunic="#e8e0d0", sleeve="bare"),
                head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#4a3e34", front=0.6, back=-0.45, bulk=1.02), laurel(sc, x, r)),
                body=caesar_body),
    "resnet": dict(name="Wellington", plate="Wellington", hair="#3a2a20", nose=1.4, aquiline=1.0, cheeks=0.88, long=1.08, jaw=0.95,
                   dress=dict(coat="#a81e24", collar="#1c2440", collar_lace=True, cuffs="#1c2440", lapels="#1c2440", lapel_w=0.03, legs="#ece6d6",
                              skirt="tails", epaulettes="fringe", sash="#c8242a", buttons=(0.035,), nbuttons=7),
                   head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#3a2a20", front=0.3, back=-0.5, curls=True, bulk=1.04),
                                                bicorne(sc, x, r, False, cockade=False)),
                   body=wellington_body),
    "features": dict(name="Takeda Shingen", plate="Takeda Shingen", hair="#1a1414", jaw=1.1, cheeks=1.08, skin="#d8a882",
                     dress=dict(coat="#2a1a1a", legs="#2a2230", boot="shoe", cuffs="#7a1a1e", w=1.05),
                     head=lambda sc, x, r, a, t: (beard(sc, x, r, "#1a1414", full=True, moustache=True), kabuto(sc, x, r)),
                     body=shingen_body),
    "nnue193": dict(name="Khalid ibn al-Walid", plate="Khalid", skin="#c08c64", hair="#1a1210", jaw=1.05,
                    dress=dict(coat="#2a4a36", legs="#e8e2d0", boots="#4a3020", skirt="robe", belt="#c8a050"),
                    head=lambda sc, x, r, a, t: (beard(sc, x, r, "#1a1210"), turban(sc, x, r)), body=khalid_body),
    "mcts_nnue": dict(name="Vercingétorix", plate="Vercingétorix", hair="#8a5a2a", nose=1.1, long=1.05, jaw=1.05,
                      dress=dict(coat="#8a3a24", legs="#a88a4a", skirt="frock", boots="#3a2416", boot="shoe", sleeve="coat",
                                 belt=Mat("#b07a3a", metal=1.0, rough=0.3)),
                      head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#8a5a2a", front=0.4, back=-1.2, bulk=1.1),
                                                   sc.add(sd_ellipsoid(r * 0.75, r * 1.0, r * 0.45), Mat("#8a5a2a", rough=0.6), x * T(0, -r * 0.6, -r * 0.55)),
                                                   sc.add(sd_capsule((-r * 0.45, -r * 0.75, r * 0.85), (r * 0.45, -r * 0.75, r * 0.85), r * 0.1), Mat("#8a5a2a", rough=0.7), x),
                                                   [sc.add(sd_capsule((r * 0.45 * sg, -r * 0.75, r * 0.85), (r * 0.5 * sg, -r * 1.15, r * 0.7), r * 0.09, r * 0.04),
                                                           Mat("#8a5a2a", rough=0.7), x) for sg in (-1, 1)],
                                                   winged_helm(sc, x, r)),
                      body=vercingetorix_body),
    "minmax": dict(name="Gengis Khan", plate="Gengis Khan", hair="#2a2420", skin="#d8a87a", jaw=1.1, cheeks=1.15, age=0.6,
                   dress=dict(coat="#1e2a5a", legs="#1e2a5a", collar="#c8a040", skirt="robe", boots="#3a2416", w=1.05, belt="#c8a040"),
                   head=lambda sc, x, r, a, t: (beard(sc, x, r, "#8a8480", full=False, moustache=True),
                                                [sc.add(sd_capsule((r * 0.15 * sg, -r * 0.4, r * 0.9), (r * 0.42 * sg, -r * 0.85, r * 0.75), r * 0.05, r * 0.025),
                                                        Mat("#8a8480", rough=0.7), x) for sg in (-1, 1)],
                                                sc.add(sd_ellipsoid(r * 0.2, r * 0.32, r * 0.14), Mat("#8a8480", rough=0.7), x * T(0, -r * 0.85, r * 0.7)),
                                                mongol_hat(sc, x, r)),
                   body=genghis_body),
    "basic_minmax": dict(name="Hannibal", plate="Hannibal", skin="#c8946a", hair="#2a1e16", patch=True, nose=1.1, long=1.05,
                         dress=dict(coat="#e8e0d0", legs="#c8946a", boot="sandal", skirt="pteruges", pteruges="#e8dcc0", tunic="#e8e0d0", sleeve="bare"),
                         head=lambda sc, x, r, a, t: (beard(sc, x, r, "#2a1e16", full=True, moustache=True), crested_helm(sc, x, r)),
                         body=hannibal_body),
    "mcts": dict(name="Ulysses Grant", plate="Grant", hair="#5a3e28", jaw=1.0, age=0.4,
                 dress=dict(coat="#1a2038", legs="#1a2038", collar="#1a2038", skirt="frock", buttons=(-0.035, 0.035), nbuttons=9,
                            button_mat=Mat("#c8a050", metal=1.0, rough=0.3)),
                 head=lambda sc, x, r, a, t: (hair_cap(sc, x, r, "#5a3e28", front=0.3), beard(sc, x, r, "#5a3e28"), slouch_hat(sc, x, r)),
                 body=grant_body),
    "monte_carlo": dict(name="Frédéric le Grand", plate="Frédéric II", hair="#ece8e2", nose=1.2, cheeks=0.8, long=1.06, iris="#4a7aa8", age=0.8,
                        dress=dict(coat="#1e3270", collar="#1e3270", cuffs="#a8242a", lapels="#a8242a", lapel_w=0.045, vest="#e8d070", legs="#e8d070",
                                   skirt="tails", sash="#e88a2a", buttons=(-0.04, 0.04), nbuttons=5, button_mat=silver(0.25)),
                        head=lambda sc, x, r, a, t: (queue_wig(sc, x, r), tricorne(sc, x, r)),
                        body=frederick_body),
    "beginner": dict(name="Gueorgui Joukov", plate="Joukov", hair="#9a948c", jaw=1.25, cheeks=1.15, brow=1.2, w=1.08, age=0.5,
                     dress=dict(coat="#55583e", legs="#3a3e4a", collar="#55583e", collar_lace=True, cuffs="#55583e", boards=True,
                                belt=Mat("#6a4a2a", rough=0.4, clearcoat=0.5), buttons=(0.0,), nbuttons=5, w=1.08),
                     head=lambda sc, x, r, a, t: marshal_cap(sc, x, r), body=zhukov_body),
    "random": dict(name="Alexandre le Grand", plate="Alexandre", hair="#4a3020", iris="#5a4a3a", long=0.98, nose=1.05,
                   dress=dict(coat="#e8dcbc", legs="#e2b096", boot="sandal", skirt="pteruges", pteruges="#e8dcc0", tunic="#a8242a", sleeve="bare"),
                   head=lambda sc, x, r, a, t: anastole(sc, x, r, "#4a3020"),
                   body=alexander_body),
}


def marshal_user(anim, t, side):
    """The user as a marshal: the buzz cut and the stubble of the portrait,
    a black or white coat by side, gold epaulettes, a bicorne."""
    coat = "#1a1a20" if side == 0 else "#ece8e0"
    g = dict(name="Vous", hair="#3d342e", iris="#7f97a6",
             dress=dict(coat=coat, lapels="#c8a050" if side == 0 else "#1a1a20", legs="#f0ece0" if side == 0 else "#2a2a36",
                        skirt="tails", epaulettes="fringe", buttons=(0.0,), nbuttons=6),
             head=lambda sc, x, r, a, t2: (hair_cap(sc, x, r, "#3d342e", front=0.45, back=-0.4, bulk=1.02),
                                            sc.add(sd_plane_cut(sd_ellipsoid(r * 0.78, r * 0.6, r * 0.7), (0, 1, 0.25), -r * 0.12),
                                                   Mat("#a8907e", rough=0.8), x * T(0, -r * 0.5, r * 0.24), blend=r * 0.03),
                                            bicorne(sc, x, r, True)))
    return general(anim, t, side, g)


ORDER = ["random", "beginner", "monte_carlo", "mcts", "basic_minmax", "minmax", "mcts_nnue", "nnue193", "features", "resnet", "dev", "alphazero"]
GENERALS = {"gen_" + k: (lambda k: lambda anim, t, side: general(anim, t, side, G[k]))(k) for k in ORDER}
GENERALS["gen_human"] = marshal_user
NAMES = {"gen_" + k: G[k]["name"] for k in ORDER}
NAMES["gen_human"] = "Vous (maréchal)"
PLATES = {"gen_" + k: G[k]["plate"] for k in ORDER}
# the rank, as gold stars over the plaque (drawn by the application, "stars" in the
# drawing's file): 1 to 5 by the place in the set, five for Napoleon alone
RANKS = ["random", "beginner", "monte_carlo", "mcts", "basic_minmax", "minmax", "mcts_nnue", "nnue193", "features", "resnet", "dev", "alphazero"]
STARS = {"gen_" + k: 1 + i * 4 // (len(RANKS) - 1) for i, k in enumerate(RANKS)}
