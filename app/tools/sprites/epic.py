"""
epic.py — the Légendes set, second version: one character per AI, in adult
proportions (heroic.adult), from a goblin thief to an ancient dragon:

  random        Gobelin chapardeur      a hunched goblin, a rusty knife, a stolen sack
  beginner      Orque éclaireur         a lean orc, a fur mantle, a spear
  monte_carlo   Rôdeuse elfe            a hooded elf ranger drawing her longbow
  mcts          Guerrier orque          black iron, a horned helm, cleaver and shield
  basic_minmax  Troll des collines      a hunched hill troll dragging a tree-trunk club
  nnue193       Chevalier elfe          silver plate, a blue cloak, a winged helm, a long sword
  features      Seigneur de guerre orque  spiked black armour, a red cloak, a great axe
  resnet        Troll de guerre         a giant troll in bolted plates, a rune maul
  dev           Archimage elfe          white and gold robes, levitating, a staff of light, runes
  alphazero     Dragon ancien           wings spread, horns, fire in the throat

The human is never part of a set: always the user's own figurine.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, sd_plane_cut, fbm
from models import base, BASE_TOP, TAU, wave, glow, gold
from legends import leather, cloth, iron, silver, wood, skin, base_pose, held
from heroic import adult, chain, bezier, membrane_wing, scales, HEAD


def fur(c):
    return Mat(c, rough=0.95, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.6 + 0.6 * fbm(pl * 140, 2))[:, None]})


def warty(c):
    return Mat(c, rough=0.8, sss=0.15, pattern=lambda pl, pw, n: {
        "albedo": srgb(c) * (0.7 + 0.45 * fbm(pl * 35, 4))[:, None] * (1 - 0.25 * (fbm(pl * 90, 2) > 0.62).float())[:, None]})


def rust(c="#5a4a40"):
    return Mat(c, metal=0.7, rough=0.6, pattern=lambda pl, pw, n: {
        "albedo": torch.where((fbm(pl * 30, 3) > 0.64)[:, None], srgb("#5a3424").expand_as(pl), srgb(c).expand_as(pl)),
        "rough": 0.5 + 0.4 * fbm(pl * 50, 2)})


# ─── faces ──────────────────────────────────────────────────────────────────
def orc_face(sc, x, r, sk, anim, eye="#ff5a1a", tusk=1.0):
    sc.add(sd_ellipsoid(r, r * 1.0, r * 1.05), sk, x)
    sc.add(sd_ellipsoid(r * 0.98, r * 0.58, r * 0.85), sk, x * T(0, -r * 0.55, r * 0.22), blend=r * 0.3)     # a heavy jaw
    sc.add(sd_ellipsoid(r * 0.92, r * 0.2, r * 0.35), sk, x * T(0, r * 0.22, r * 0.76), blend=r * 0.15)     # brow ridge
    sc.add(sd_ellipsoid(r * 0.3, r * 0.2, r * 0.25), sk, x * T(0, -r * 0.08, r * 0.98), blend=r * 0.1)     # a flat nose
    roar = anim in ("win", "blocked")
    for sgn in (-1, 1):
        sc.add(sd_sphere(r * 0.12), glow(eye, 4.0 if roar else 2.5), x * T(r * 0.35 * sgn, r * 0.07, r * 0.84))
        sc.add(sd_cone(r * 0.2 * tusk, r * 0.09 * tusk, r * 0.012, 0.003), Mat("#e8dcc0", rough=0.4),
               x * T(r * 0.36 * sgn, -r * 0.5, r * 0.9) * RZ(-0.15 * sgn) * T(0, r * 0.12 * tusk, 0))
        sc.add(sd_cone(r * 0.32, r * 0.16, r * 0.015, 0.006), sk, x * T(r * 0.95 * sgn, r * 0.1, -r * 0.1) * RZ(-1.3 * sgn) * RX(0.35) * T(0, r * 0.28, 0))
    if roar:
        sc.add(sd_ellipsoid(r * 0.42, r * 0.2, r * 0.3), Mat("#2a0808", rough=0.6), x * T(0, -r * 0.55, r * 0.8))


def elf_face(sc, x, r, sk, hair_c, eye="#6ad8a8", hair="long", anim="idle"):
    sc.add(sd_ellipsoid(r * 0.86, r * 1.05, r * 0.95), sk, x)
    sc.add(sd_ellipsoid(r * 0.55, r * 0.5, r * 0.6), sk, x * T(0, -r * 0.48, r * 0.26), blend=r * 0.25)    # a narrow chin
    sc.add(sd_capsule((0, r * 0.12, 0), (0, -r * 0.15, r * 0.08), r * 0.08, r * 0.1), sk, x * T(0, -r * 0.05, r * 0.9), blend=r * 0.05)
    hm = Mat(hair_c, rough=0.35, clearcoat=0.5)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(r * 0.17, r * 0.065, r * 0.05), Mat("#16141c", rough=0.3), x * T(r * 0.32 * sgn, r * 0.08, r * 0.83) * RZ(0.15 * sgn))
        sc.add(sd_sphere(r * 0.055), glow(eye, 2.0), x * T(r * 0.32 * sgn, r * 0.08, r * 0.87))
        sc.add(sd_capsule((-r * 0.15, 0, 0), (r * 0.15, 0, 0), r * 0.03), hm, x * T(r * 0.32 * sgn, r * 0.24, r * 0.82) * RZ(-0.25 * sgn))
        sc.add(sd_cone(r * 0.6, r * 0.13, r * 0.01, 0.005), sk, x * T(r * 0.82 * sgn, r * 0.1, -r * 0.05) * RZ(-1.15 * sgn) * RX(0.55) * T(0, r * 0.5, 0))
    if hair:
        sc.add(sd_plane_cut(sd_ellipsoid(r * 0.95, r * 1.12, r * 1.04), (0, -0.35, 1), r * 0.3), hm, x * T(0, r * 0.08, -r * 0.04))
        if hair == "long":
            sc.add(sd_ellipsoid(r * 0.85, r * 1.7, r * 0.42), hm, x * T(0, -r * 1.0, -r * 0.6) * RX(0.15))


def troll_face(sc, x, r, sk, anim, eye="#ffb02a"):
    sc.add(sd_ellipsoid(r * 1.05, r * 0.85, r * 1.0), sk, x)
    sc.add(sd_ellipsoid(r * 1.15, r * 0.7, r * 1.0), sk, x * T(0, -r * 0.62, r * 0.25), blend=r * 0.35)    # an underbite
    sc.add(sd_ellipsoid(r * 1.0, r * 0.25, r * 0.4), sk, x * T(0, r * 0.25, r * 0.7), blend=r * 0.2)
    sc.add(sd_capsule((0, 0, 0), (0, -r * 0.45, r * 0.25), r * 0.2, r * 0.26), sk, x * T(0, r * 0.05, r * 0.95), blend=r * 0.1)   # a long nose
    for sgn in (-1, 1):
        sc.add(sd_sphere(r * 0.09), glow(eye, 2.5), x * T(r * 0.33 * sgn, r * 0.08, r * 0.86))
        sc.add(sd_cone(r * 0.32, r * 0.12, r * 0.02, 0.005), Mat("#d8ccaa", rough=0.5), x * T(r * 0.55 * sgn, -r * 0.62, r * 0.95) * RZ(-0.3 * sgn) * T(0, r * 0.25, 0))
        sc.add(sd_ellipsoid(r * 0.15, r * 0.3, r * 0.08), sk, x * T(r * 1.05 * sgn, -r * 0.05, -r * 0.05) * RZ(0.5 * sgn))
    for k, (px, py, pz) in enumerate(((0.5, 0.5, 0.6), (-0.6, 0.35, 0.65), (0.2, -0.3, 1.05), (-0.7, -0.45, 0.8))):
        sc.add(sd_sphere(r * 0.08), sk, x * T(r * px, r * py, r * pz), blend=r * 0.04)
    if anim in ("win", "blocked"):
        sc.add(sd_ellipsoid(r * 0.6, r * 0.25, r * 0.3), Mat("#2a0e0a", rough=0.6), x * T(0, -r * 0.62, r * 0.98))


def goblin_face(sc, x, r, sk, anim):
    sc.add(sd_ellipsoid(r, r * 0.92, r * 0.95), sk, x)
    sc.add(sd_ellipsoid(r * 0.7, r * 0.45, r * 0.7), sk, x * T(0, -r * 0.48, r * 0.3), blend=r * 0.3)
    sc.add(sd_capsule((0, 0, 0), (0, -r * 0.35, r * 0.45), r * 0.16, r * 0.07), sk, x * T(0, r * 0.05, r * 0.9), blend=r * 0.06)   # a hooked nose
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(r * 0.15, r * 0.1, r * 0.06), glow("#ffd42a", 3.0), x * T(r * 0.36 * sgn, r * 0.12, r * 0.84) * RZ(-0.35 * sgn))
        sc.add(sd_capsule((-r * 0.2, 0, 0), (r * 0.2, 0, 0), r * 0.05), sk, x * T(r * 0.36 * sgn, r * 0.28, r * 0.8) * RZ(-0.5 * sgn), blend=r * 0.05)
        sc.add(sd_cone(r * 1.0, r * 0.3, r * 0.02, 0.01), sk, x * T(r * 0.9 * sgn, r * 0.15, -r * 0.05) * RZ(-1.35 * sgn) * RX(0.2) * T(0, r * 0.85, 0))
    grin = r * (0.2 if anim == "win" else 0.1)
    sc.add(sd_ellipsoid(r * 0.45, grin, r * 0.2), Mat("#2a0e0a", rough=0.6), x * T(0, -r * 0.5, r * 0.88))
    sc.add(sd_box(r * 0.38, r * 0.035, r * 0.05, r * 0.01), Mat("#e0d8b0", rough=0.5), x * T(0, -r * 0.5 + grin * 0.6, r * 0.95))


# ─── 1 · the goblin ─────────────────────────────────────────────────────────
def goblin(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.9, 0.3, 0.9]
    p.l = [0.3, 0.35, 1.4]
    if anim == "win":
        p.r = [2.5, 0.3, 0.2]
        p.jump = 0.05 * abs(math.sin(TAU * t * 2))
    sk = warty("#7a8a46")
    rag = cloth("#5a4a36")
    m = {"skin": sk, "top": leather("#4a3a2a"), "legs": rag, "boots": sk, "belly": rag}
    rig = adult(sc, p, m, H=0.74, bulk=0.82, head=1.5, hunch=0.35, arm=1.15)
    goblin_face(sc, rig["head"], rig["hr"], sk, anim)
    # the stolen sack on the back, a rope across the chest
    sc.add(sd_ellipsoid(0.09, 0.1, 0.07), cloth("#8a7450"), rig["back"] * T(0.03, -0.12, -0.05))
    sc.add(sd_torus(0.1, 0.008), Mat("#6a5a3a", rough=0.9), rig["torso"] * T(0, 0.17, 0.0) * RZ(0.6))
    for k in range(3):
        sc.add(sd_sphere(0.012), gold(), rig["back"] * T(0.05 - 0.03 * k, -0.02 + 0.01 * k, -0.03))
    kn = held(rig["r"], -math.pi if anim == "win" else -1.2)
    sc.add(sd_capsule((0, -0.04, 0), (0, 0.03, 0), 0.012), wood("#3a2a1a"), kn)
    sc.add(sd_cone(0.08, 0.022, 0.003, 0.002), rust("#6a6a6a"), kn * T(0, 0.11, 0) * RZ(0.15))
    return sc


# ─── 2 · the orc scout ──────────────────────────────────────────────────────
ORC = "#5d7038"


def orc_scout(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.55, 0.25, 0.9]
    p.l = [0.35, 0.35, 0.7]
    if anim == "win":
        p.r = [2.6, 0.2, 0.1]
    sk = skin(ORC, 0.3)
    m = {"skin": sk, "top": leather("#4a3424"), "legs": leather("#3a2a1e"), "boots": leather("#24180f"), "arms": sk}
    rig = adult(sc, p, m, H=0.98, bulk=1.05, head=1.05)
    for sgn in (-1, 1):   # a fur mantle on the shoulders
        sc.add(sd_ellipsoid(0.1, 0.06, 0.1), fur("#6a5a48"), rig["torso"] * T(0.09 * sgn, 0.29, -0.01), blend=0.03)
    for k in range(5):    # a necklace of teeth
        a = -0.6 + 0.3 * k
        sc.add(sd_cone(0.02, 0.008, 0.001, 0.001), Mat("#e8dcc0", rough=0.5), rig["torso"] * T(0.08 * math.sin(a), 0.23, 0.09 * math.cos(a)) * RX(math.pi))
    x, r = rig["head"], rig["hr"]
    orc_face(sc, x, r, sk, anim)
    sc.add(sd_capsule((0, r * 0.9, -r * 0.2), (0, r * 1.3, -r * 0.6), r * 0.18, r * 0.08), Mat("#1a1612", rough=0.6), x)   # a topknot
    sp = held(rig["r"], -math.pi if anim == "win" else -1.45)
    sc.add(sd_capsule((0, -0.3, 0), (0, 0.5, 0), 0.011), wood("#4a3420"), sp)
    sc.add(sd_cone(0.1, 0.03, 0.002, 0.002), rust("#7a7a7a"), sp * T(0, 0.58, 0))
    sc.add(sd_box(0.02, 0.01, 0.004, 0.002), Mat("#8a1f1a", rough=0.7), sp * T(0.02, 0.48, 0) * RZ(-0.4))
    return sc


# ─── 3 · the elf ranger ─────────────────────────────────────────────────────
ELF = "#efd2bc"


def elf_ranger(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    draw_k = 0.5 + 0.5 * wave(t) if anim == "idle" else (1.0 if anim == "win" else 0.3)
    p.l = [1.45, 0.12, 0.05]
    p.r = [1.35, -0.25, 1.45 + 0.45 * draw_k]
    if anim == "win":
        p.l = [2.75, 0.2, 0.1]
    green = cloth("#355a32")
    sk = skin(ELF, 0.5)
    m = {"skin": sk, "top": leather("#5a4128"), "legs": cloth("#3e3326"), "boots": leather("#2e2216"), "arms": green, "hands": leather("#4a3424")}
    rig = adult(sc, p, m, H=1.05, bulk=0.85, head=0.95)
    sc.add(sd_box(0.12, 0.28, 0.014, 0.012), green, rig["back"] * T(0, -0.17, -0.025) * RX(0.12 + 0.1 * p.lean))      # the cloak
    sc.add(sd_torus(0.085, 0.008), leather("#2e2216"), rig["waist"])
    x, r = rig["head"], rig["hr"]
    elf_face(sc, x, r, sk, "#e2c070", anim=anim)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.25, r * 1.25, r * 1.2), (0, -0.2, 1), r * 0.45), green, x * T(0, r * 0.12, -r * 0.12))   # the hood
    bw = rig["l"] * RX(-1.5)
    for k in range(10):
        a0, a1 = -1.25 + 0.25 * k, -1.25 + 0.25 * (k + 1)
        q0 = (0.0, 0.36 * math.sin(a0), 0.09 * math.cos(a0) - 0.09)
        q1 = (0.0, 0.36 * math.sin(a1), 0.09 * math.cos(a1) - 0.09)
        sc.add(sd_capsule(q0, q1, 0.009), Mat("#d8c8a0", rough=0.35, clearcoat=0.6), bw)
    sc.add(sd_capsule((0, -0.34, -0.035), (0, 0.34, -0.035), 0.0025), Mat("#f0ece0", rough=0.5), bw * T(0, 0, -0.07 * draw_k))
    sc.add(sd_capsule((0, 0, -0.07 * draw_k - 0.04), (0, 0, 0.26), 0.004), wood("#8a6a40"), bw)
    sc.add(sd_cone(0.022, 0.011, 0.001, 0.001), silver(), bw * T(0, 0, 0.27) * RX(math.pi / 2))
    sc.add(sd_box(0.04, 0.13, 0.035, 0.025), leather("#5a4128"), rig["back"] * T(0.04, 0, -0.03) * RZ(-0.35))
    for k in range(4):
        sc.add(sd_capsule((0, 0, 0), (0, 0.07, 0), 0.005), Mat("#f2f0ea", rough=0.6), rig["back"] * T(0.04, 0, -0.03) * RZ(-0.35) * T(-0.024 + 0.016 * k, 0.13, 0))
    return sc


# ─── 4 · the orc warrior ────────────────────────────────────────────────────
def orc_warrior(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.l = [0.65, 0.35, 1.25]
    p.r = [0.45, 0.3, 0.7]
    if anim == "win":
        p.r = [2.55, 0.25, 0.15]
    sk = skin(ORC, 0.3)
    blk = iron("#2e3034", 0.4)
    m = {"skin": sk, "top": blk, "legs": leather("#2a1d14"), "boots": blk, "arms": sk, "fore": blk}
    rig = adult(sc, p, m, H=1.02, bulk=1.3, head=1.1)
    for sgn in (-1, 1):
        pad = rig["torso"] * T(0.17 * sgn, 0.3, 0)
        sc.add(sd_plane_cut(sd_ellipsoid(0.09, 0.07, 0.09), (0, -1, 0), 0.0), blk, pad)
        for k in range(2):
            sc.add(sd_cone(0.045, 0.016, 0.002, 0.002), iron("#9a9a9a", 0.3), pad * T(0.03 * sgn, 0.04, 0.035 * (k - 0.5) * 2) * RZ(-0.5 * sgn) * T(0, 0.03, 0))
    sc.add(sd_box(0.11, 0.06, 0.03, 0.01), Mat("#6a1410", rough=0.6), rig["waist"] * T(0, -0.1, 0.06))                 # a war skirt
    x, r = rig["head"], rig["hr"]
    orc_face(sc, x, r, sk, anim)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.1, r * 1.0, r * 1.1), (0, -1, 0), -r * 0.05), blk, x * T(0, r * 0.15, -r * 0.03))
    for sgn in (-1, 1):   # horns
        hp = bezier((0, 0, 0), (r * 0.6 * sgn, r * 0.1, 0), (r * 1.1 * sgn, r * 0.6, 0), (r * 1.0 * sgn, r * 1.3, -r * 0.2), 5)
        chain(sc, hp, [r * 0.2, r * 0.17, r * 0.12, r * 0.08, r * 0.02], Mat("#d8ccb0", rough=0.5), x * T(r * 0.7 * sgn, r * 0.45, 0), blend=0.005)
    sh = rig["l"] * RX(-1.5) * T(0, 0, 0.07)
    sc.add(sd_cylinder(0.018, 0.17, 0.01), wood("#3e2a18"), sh * RX(math.pi / 2))
    sc.add(sd_torus(0.17, 0.016), blk, sh * RX(math.pi / 2))
    sc.add(sd_box(0.12, 0.02, 0.004, 0.003), Mat("#9a1a12", rough=0.6), sh * T(0, 0, 0.02) * RZ(0.8))
    sc.add(sd_box(0.12, 0.02, 0.004, 0.003), Mat("#9a1a12", rough=0.6), sh * T(0, 0, 0.02) * RZ(-0.8))
    sc.add(sd_sphere(0.04), iron("#6a6a6a", 0.3), sh * T(0, 0, 0.02))
    ax = held(rig["r"], -math.pi if anim == "win" else -1.3)
    sc.add(sd_capsule((0, -0.05, 0), (0, 0.22, 0), 0.016), wood("#3a2818"), ax)
    sc.add(sd_box(0.08, 0.13, 0.008, 0.006), iron("#5a5e66", 0.25), ax * T(0.07, 0.2, 0))
    sc.add(sd_box(0.004, 0.12, 0.009, 0.002), Mat("#d8dce2", metal=1.0, rough=0.15), ax * T(0.15, 0.2, 0))
    return sc


# ─── 5 · the hill troll ─────────────────────────────────────────────────────
def hill_troll(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.15, 0.25, 0.25]        # dragging the club
    p.l = [0.35, 0.3, 0.6]
    if anim == "win":
        p.r = [2.6, 0.3, 0.25]
        p.l = [1.6, 0.6, 0.6]
    sk = warty("#4c5844")
    m = {"skin": sk, "top": sk, "legs": cloth("#4a3a2a"), "boots": sk}
    rig = adult(sc, p, m, H=1.2, bulk=1.75, head=1.0, hunch=0.5, arm=1.35, neck=0.5, leg=0.7)
    sc.add(sd_ellipsoid(0.15, 0.13, 0.13), sk, rig["waist"] * T(0, 0.02, 0.06), blend=0.05)                         # a belly
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.07, 0.035, 0.07), Mat("#3e5a2a", rough=0.95), rig["torso"] * T(0.16 * sgn, 0.36, 0), blend=0.02)   # moss
    troll_face(sc, rig["head"], rig["hr"], sk, anim)
    cl = held(rig["r"], -math.pi if anim == "win" else 0.35)
    sc.add(sd_capsule((0, -0.05, 0), (0, 0.5, 0), 0.03, 0.075), wood("#4a3420"), cl)
    for k in range(4):
        sc.add(sd_ellipsoid(0.035, 0.025, 0.035), wood("#3a2818"), cl * T(0.06 * math.cos(k * 2.1), 0.25 + 0.07 * k, 0.06 * math.sin(k * 2.1)))
    return sc


# ─── 6 · the elf knight ─────────────────────────────────────────────────────
def elf_knight(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.l = [0.6, 0.3, 1.25]
    p.r = [0.4, 0.25, 0.55]
    if anim == "win":
        p.r = [2.55, 0.2, 0.12]
    plate = Mat("#dfe4ec", metal=1.0, rough=0.16)
    trim = gold()
    sk = skin(ELF, 0.5)
    blue = cloth("#24508a")
    m = {"skin": sk, "top": plate, "legs": plate, "boots": plate, "arms": plate, "hands": plate, "neck": plate}
    rig = adult(sc, p, m, H=1.06, bulk=1.05, head=0.95)
    sc.add(sd_torus(0.1, 0.01), trim, rig["waist"] * T(0, -0.02, 0))
    sc.add(sd_cone(0.1, 0.12, 0.08, 0.01), blue, rig["waist"] * T(0, -0.12, 0))
    sc.add(sd_box(0.15, 0.33, 0.012, 0.01), blue, rig["back"] * T(0, -0.2, -0.03) * RX(0.1 + 0.1 * p.lean))
    for sgn in (-1, 1):
        sc.add(sd_plane_cut(sd_ellipsoid(0.075, 0.06, 0.075), (0, -1, 0), 0.0), plate, rig["torso"] * T(0.16 * sgn, 0.3, 0))
        sc.add(sd_torus(0.07, 0.006), trim, rig["torso"] * T(0.16 * sgn, 0.3, 0))
    x, r = rig["head"], rig["hr"]
    elf_face(sc, x, r, sk, "#d9c48a", hair=None, anim=anim)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.12, r * 1.15, r * 1.12), (0, -0.4, 1), r * 0.25), plate, x * T(0, r * 0.12, -r * 0.05))
    for sgn in (-1, 1):   # wings on the helm
        for k in range(3):
            sc.add(sd_ellipsoid(r * 0.08, r * (0.7 - 0.15 * k), r * 0.22), trim, x * T(r * 0.95 * sgn, r * (0.5 + 0.1 * k), -r * (0.3 + 0.2 * k)) * RZ(-0.5 * sgn) * RX(-0.6))
    sh = rig["l"] * RX(-1.5) * T(0, 0, 0.065)
    sc.add(sd_box(0.12, 0.19, 0.014, 0.05), Mat("#24508a", rough=0.35, clearcoat=0.8), sh)
    sc.add(sd_box(0.125, 0.195, 0.011, 0.055), trim, sh * T(0, 0, -0.005))
    sc.add(sd_capsule((0, -0.1, 0), (0, 0.1, 0), 0.01), trim, sh * T(0, 0, 0.017))
    for sgn in (-1, 1):
        sc.add(sd_capsule((0, 0.03, 0), (0.06 * sgn, 0.1, 0), 0.007), trim, sh * T(0, 0, 0.017))
    sw = held(rig["r"], -math.pi if anim == "win" else -1.25)
    sc.add(sd_capsule((0, -0.08, 0), (0, 0.04, 0), 0.013), Mat("#2a2018", rough=0.6), sw)
    sc.add(sd_box(0.08, 0.012, 0.016, 0.005), trim, sw * T(0, 0.05, 0))
    sc.add(sd_box(0.018, 0.32, 0.004, 0.003), silver(0.06), sw * T(0, 0.38, 0))
    if anim == "win":
        sc.add(sd_box(0.007, 0.3, 0.002, 0.001), glow("#bfe8ff", 4.0), sw * T(0, 0.38, 0.005))
    return sc


# ─── 7 · the orc warlord ────────────────────────────────────────────────────
def orc_warlord(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.l = [0.9, -0.1, 1.2]           # both hands on the great axe
    p.r = [0.7, 0.05, 1.0]
    if anim == "win":
        p.l = [2.6, 0.1, 0.2]
        p.r = [2.6, 0.1, 0.2]
    sk = skin("#4e6232", 0.3)
    blk = iron("#232428", 0.35)
    m = {"skin": sk, "top": blk, "legs": blk, "boots": blk, "arms": blk, "neck": sk}
    rig = adult(sc, p, m, H=1.08, bulk=1.5, head=1.15)
    red = cloth("#7a1210")
    sc.add(sd_box(0.19, 0.36, 0.014, 0.012), red, rig["back"] * T(0, -0.22, -0.04) * RX(0.08 + 0.1 * p.lean))
    for sgn in (-1, 1):
        pad = rig["torso"] * T(0.2 * sgn, 0.31, 0)
        sc.add(sd_plane_cut(sd_ellipsoid(0.11, 0.085, 0.11), (0, -1, 0), -0.01), blk, pad)
        for k in range(3):
            sc.add(sd_cone(0.07, 0.02, 0.002, 0.002), iron("#a8a8a8", 0.25), pad * T(0.04 * sgn, 0.05, 0.05 * (k - 1)) * RZ(-0.4 * sgn) * T(0, 0.05, 0))
    sc.add(sd_sphere(0.04), Mat("#e8dcc0", rough=0.5), rig["waist"] * T(0.06, -0.04, 0.09))   # a skull on the belt
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.011), Mat("#100808", rough=0.6), rig["waist"] * T(0.06 + 0.015 * sgn, -0.035, 0.125))
    x, r = rig["head"], rig["hr"]
    orc_face(sc, x, r, sk, anim, eye="#ff2a1a", tusk=1.3)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.12, r * 1.0, r * 1.12), (0, -1, 0), -r * 0.05), blk, x * T(0, r * 0.18, -r * 0.03))
    for k in range(5):    # a crown of spikes
        a = -1.0 + 0.5 * k
        sc.add(sd_cone(r * 0.45, r * 0.1, r * 0.01, 0.002), iron("#a8a8a8", 0.25), x * T(r * 0.9 * math.sin(a), r * 0.75, r * 0.9 * math.cos(a) - r * 0.2) * RZ(-0.4 * math.sin(a)) * T(0, r * 0.35, 0))
    ax = held(rig["r"], -math.pi if anim == "win" else -1.0, 0.2)
    sc.add(sd_capsule((0, -0.25, 0), (0, 0.45, 0), 0.017), wood("#2a1d12"), ax)
    for sgn in (-1, 1):   # a double-bitted head
        sc.add(sd_plane_cut(sd_cylinder(0.008, 0.13, 0.004), (0, 0, -sgn), 0.0), iron("#5a5e66", 0.2), ax * T(0, 0.38, 0) * RX(math.pi / 2) * RZ(math.pi / 2))
    sc.add(sd_box(0.03, 0.06, 0.03, 0.01), blk, ax * T(0, 0.38, 0))
    return sc


# ─── 8 · the war troll ──────────────────────────────────────────────────────
def war_troll(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.5, 0.35, 0.9]
    p.l = [0.4, 0.4, 0.5]
    if anim == "win":
        p.r = [2.6, 0.25, 0.25]
        p.l = [2.0, 0.6, 0.5]
    sk = warty("#424c4e")
    plate = rust("#4a4c50")
    m = {"skin": sk, "top": sk, "legs": cloth("#3a2a20"), "boots": plate, "arms": sk}
    rig = adult(sc, p, m, H=1.3, bulk=1.9, head=0.95, hunch=0.42, arm=1.3, neck=0.4, leg=0.72)
    rune = glow("#ff8a2a", 3.0 + 1.5 * wave(t, 2) + (3.0 if anim == "win" else 0))
    sc.add(sd_box(0.17, 0.12, 0.05, 0.03), plate, rig["chest"] * T(0, 0.0, 0.0))            # a breastplate
    for k in range(4):
        sc.add(sd_sphere(0.012), iron("#8a8a8a", 0.3), rig["chest"] * T(-0.13 + 0.087 * k, 0.09, 0.05))
    sc.add(sd_box(0.06, 0.008, 0.004, 0.002), rune, rig["chest"] * T(0, 0, 0.052))
    sc.add(sd_box(0.008, 0.06, 0.004, 0.002), rune, rig["chest"] * T(0, 0, 0.052))
    for sgn in (-1, 1):
        pad = rig["torso"] * T(0.27 * sgn, 0.4, 0)
        sc.add(sd_plane_cut(sd_ellipsoid(0.13, 0.09, 0.12), (0, -1, 0), -0.01), plate, pad)
    for k in range(6):    # chains on the belly
        sc.add(sd_torus(0.016, 0.005), iron("#6a6a6a", 0.4), rig["waist"] * T(-0.12 + 0.05 * k, -0.05 - 0.02 * math.sin(k), 0.13) * RX(math.pi / 2 * (k % 2)))
    troll_face(sc, rig["head"], rig["hr"], sk, anim, eye="#ff6a1a")
    x, r = rig["head"], rig["hr"]
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.15, r * 0.95, r * 1.1), (0, -1, 0), -r * 0.1), plate, x * T(0, r * 0.2, -r * 0.05))
    mm = held(rig["r"], -math.pi if anim == "win" else -1.0)
    sc.add(sd_capsule((0, -0.12, 0), (0, 0.5, 0), 0.022), wood("#2a1d12"), mm)
    sc.add(sd_box(0.15, 0.09, 0.09, 0.025), Mat("#5a5a5e", rough=0.8), mm * T(0, 0.55, 0))
    sc.add(sd_box(0.152, 0.015, 0.092, 0.005), plate, mm * T(0, 0.5, 0))
    sc.add(sd_box(0.152, 0.015, 0.092, 0.005), plate, mm * T(0, 0.6, 0))
    sc.add(sd_box(0.06, 0.012, 0.002, 0.002), rune, mm * T(0, 0.55, 0.092))
    return sc


# ─── 9 · the elf archmage ───────────────────────────────────────────────────
def archmage(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    lev = 0.06 + 0.02 * math.sin(TAU * t)
    p.bob += lev
    p.r = [0.6, 0.3, 0.8]
    p.l = [0.9, 0.55, 1.2]
    if anim == "win":
        p.r = [2.6, 0.2, 0.15]
        p.l = [2.2, 0.8, 0.3]
    robe = cloth("#f2f0ea", 0.7)
    sk = skin(ELF, 0.5)
    m = {"skin": sk, "top": robe, "legs": robe, "boots": robe, "arms": robe}
    rig = adult(sc, p, m, H=1.06, bulk=0.92, head=0.95)
    sc.add(sd_cone(0.26, 0.2, 0.09, 0.02), robe, rig["waist"] * T(0, -0.2, 0))                  # the robe down to the feet
    sc.add(sd_torus(0.19, 0.012), gold(), rig["waist"] * T(0, -0.45, 0))
    sc.add(sd_torus(0.088, 0.01), gold(), rig["waist"])
    sc.add(sd_box(0.012, 0.22, 0.004, 0.003), gold(), rig["waist"] * T(0, -0.2, 0.14) * RX(-0.18))
    for sgn in (-1, 1):   # a high collar
        sc.add(sd_box(0.07, 0.07, 0.006, 0.004), Mat("#e8e2d0", rough=0.5), rig["torso"] * T(0.07 * sgn, 0.36, -0.03) * RY(0.6 * sgn) * RX(-0.2))
    x, r = rig["head"], rig["hr"]
    elf_face(sc, x, r, sk, "#e8eaf0", eye="#8adfff", anim=anim)
    sc.add(sd_torus(r * 0.92, r * 0.06), gold(), x * T(0, r * 0.35, 0) * RX(0.15))
    sc.add(sd_sphere(r * 0.12), glow("#8adfff", 5.0), x * T(0, r * 0.42, r * 0.9))
    st = held(rig["r"], -math.pi if anim == "win" else -1.55)
    sc.add(sd_capsule((0, -0.4, 0), (0, 0.42, 0), 0.011), Mat("#e8e2d0", rough=0.3, clearcoat=0.6), st)
    for k in range(4):
        a = TAU * k / 4
        sc.add(sd_capsule((0, 0.38, 0), (0.05 * math.cos(a), 0.5, 0.05 * math.sin(a)), 0.006), gold(), st)
    kk = 8.0 + 3.0 * wave(t, 2) + (10.0 if anim == "win" else 0.0)
    sc.add(sd_ellipsoid(0.028, 0.05, 0.028), glow("#9ae4ff", kk), st * T(0, 0.5, 0))
    for i in range(3):    # runes orbiting
        a = TAU * (t + i / 3)
        q = rig["body"] * T(0.38 * math.cos(a), BASE_TOP + 0.45 + 0.1 * math.sin(2 * a + i), 0.38 * math.sin(a)) * RY(-a)
        sc.add(sd_torus(0.03, 0.004), glow("#9ae4ff", 6.0), q)
        sc.add(sd_box(0.004, 0.022, 0.003, 0.001), glow("#9ae4ff", 6.0), q)
    sc.add(sd_torus(0.42, 0.006), glow("#9ae4ff", 4.0), RY(TAU * t * 0.5) * T(0, BASE_TOP + 0.01, 0))
    return sc


# ─── 10 · the ancient dragon ────────────────────────────────────────────────
def dragon(anim, t, side):
    sc = Scene()
    base(sc, side)
    flap = math.sin(TAU * t * (2 if anim in ("move", "win") else 1))
    rise = 0.04 * math.sin(math.pi * t) if anim == "move" else 0.0
    root = RY(HEAD + (0.2 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)) * T(0, BASE_TOP + rise, 0)
    hide = scales("#5a1a16", k=80.0)
    dark = scales("#2e0e0e", k=110.0)
    belly = Mat("#b8925a", rough=0.6, pattern=lambda pl, pw, n: {
        "albedo": srgb("#b8925a") * (0.7 + 0.4 * (torch.sin(pl[:, 2] * 160 + pl[:, 1] * 160) > -0.6).float())[:, None]})
    horn = Mat("#2a2420", rough=0.35, clearcoat=0.5)
    # the body, crouched
    sc.add(sd_ellipsoid(0.21, 0.18, 0.3), hide, root * T(0, 0.35, -0.03) * RX(-0.15))
    sc.add(sd_ellipsoid(0.16, 0.12, 0.25), belly, root * T(0, 0.28, 0.02) * RX(-0.15), blend=0.03)
    for (sx, sz, front) in ((-1, 1, True), (1, 1, True), (-1, -1, False), (1, -1, False)):
        hip = root * T(0.16 * sx, 0.33, 0.16 * sz)
        step = 0.3 * math.sin(TAU * t * 2 + (0 if sx * sz > 0 else math.pi)) if anim == "move" else 0.0
        if front:
            sc.add(sd_capsule((0, 0, 0), (0.03 * sx, -0.14, 0.06 + step * 0.1), 0.065, 0.045), hide, hip)
            sc.add(sd_capsule((0.03 * sx, -0.14, 0.06), (0.04 * sx, -0.31, 0.1), 0.045, 0.036), hide, hip)
            paw = hip * T(0.04 * sx, -0.31, 0.12)
        else:
            sc.add(sd_ellipsoid(0.1, 0.13, 0.12), hide, hip * T(0.02 * sx, -0.04, 0), blend=0.03)
            sc.add(sd_capsule((0.03 * sx, -0.08, 0.04), (0.04 * sx, -0.2, -0.05), 0.045, 0.03), hide, hip)
            sc.add(sd_capsule((0.04 * sx, -0.2, -0.05), (0.04 * sx, -0.31, 0.02), 0.03, 0.026), hide, hip)
            paw = hip * T(0.04 * sx, -0.31, 0.04)
        sc.add(sd_ellipsoid(0.04, 0.02, 0.05), hide, paw)
        for k in range(3):
            sc.add(sd_cone(0.03, 0.009, 0.001, 0.001), horn, paw * T(0.02 * (k - 1), -0.005, 0.05) * RX(math.pi / 2) * T(0, 0.015, 0))
    # the neck, rising and arching forward, then the head
    lift = 0.06 * wave(t) if anim == "idle" else (0.1 if anim == "win" else 0.0)
    neck = bezier((0, 0.42, 0.2), (0, 0.6, 0.34), (0, 0.78 + lift, 0.3), (0, 0.8 + lift, 0.42), 8)
    chain(sc, neck, [0.1 - 0.006 * i for i in range(8)], hide, root, blend=0.03)
    for i in range(1, 8):
        sc.add(sd_cone(0.04, 0.016, 0.001, 0.001), horn, root * T(neck[i][0], neck[i][1] + 0.05, neck[i][2] - 0.04) * RX(-0.6))
    hp = neck[-1]
    hx = root * T(hp[0], hp[1] + 0.02, hp[2] + 0.04) * RY(0.15 * math.sin(TAU * t) if anim == "idle" else 0) * RX(0.35 if anim != "win" else -0.2) * T(0, 0, 0) 
    sc.add(sd_ellipsoid(0.075, 0.065, 0.09), hide, hx)
    sc.add(sd_box(0.045, 0.03, 0.085, 0.025), hide, hx * T(0, -0.005, 0.11), blend=0.02)                        # the snout
    jaw = 0.05 + (0.45 if anim == "win" else 0.12 * max(0.0, wave(t, 1)) if anim == "idle" else 0.25 if anim == "blocked" else 0.05)
    jx = hx * T(0, -0.035, 0.02) * RX(jaw)
    sc.add(sd_box(0.04, 0.016, 0.085, 0.012), hide, jx * T(0, -0.01, 0.085))
    sc.add(sd_box(0.035, 0.006, 0.08, 0.004), glow("#ff6a1a", 3.0 + 3.0 * jaw), hx * T(0, -0.035, 0.1) * RX(jaw * 0.5))   # fire in the throat
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.02, 0.01, 0.012), glow("#ffc82a", 6.0), hx * T(0.05 * sgn, 0.025, 0.06) * RY(0.3 * sgn))
        sc.add(sd_ellipsoid(0.05, 0.016, 0.05), hide, hx * T(0.04 * sgn, 0.04, 0.04) * RZ(-0.3 * sgn), blend=0.01)   # brow ridges
        hp_ = bezier((0, 0, 0), (0.03 * sgn, 0.03, -0.08), (0.06 * sgn, 0.08, -0.16), (0.08 * sgn, 0.06, -0.26), 5)
        chain(sc, hp_, [0.022, 0.018, 0.013, 0.008, 0.002], horn, hx * T(0.045 * sgn, 0.05, -0.03), blend=0.004)
        hp2 = bezier((0, 0, 0), (0.04 * sgn, -0.01, -0.05), (0.07 * sgn, -0.02, -0.09), (0.09 * sgn, -0.04, -0.13), 4)
        chain(sc, hp2, [0.014, 0.01, 0.006, 0.002], horn, hx * T(0.06 * sgn, 0.0, -0.03), blend=0.003)
        for k in range(3):
            sc.add(sd_cone(0.014, 0.005, 0.001, 0.001), Mat("#f2ead8", rough=0.4), hx * T(0.03 * sgn, -0.04, 0.12 + 0.03 * k) * RX(math.pi))
    if anim == "win":    # the breath
        for k in range(3):
            sc.add(sd_cone(0.18 + 0.06 * k, 0.05 + 0.03 * k, 0.012, 0.01), glow("#ff7a1a" if k else "#ffe08a", 10.0 - 2 * k),
                   hx * T(0, -0.05, 0.3 + 0.08 * k) * RX(math.pi / 2 + 0.25))
    # the tail, curling around the base, spikes on top
    tail = bezier((0, 0.32, -0.28), (0.05, 0.2, -0.55), (0.45, 0.12, -0.45), (0.5, 0.13, 0.05 + 0.04 * math.sin(TAU * t)), 10)
    chain(sc, tail, [0.075 * (1 - 0.09 * i) for i in range(10)], hide, root, blend=0.03)
    for i in range(1, 9, 2):
        sc.add(sd_cone(0.04, 0.014, 0.001, 0.001), horn, root * T(tail[i][0], tail[i][1] + 0.06 * (1 - 0.08 * i), tail[i][2]))
    sc.add(sd_cone(0.06, 0.03, 0.002, 0.002), horn, root * T(*tail[-1]) * RY(0.3) * RX(math.pi / 2) * T(0, 0.04, 0))
    # the wings, spread and beating
    for sgn in (-1, 1):
        w = root * T(0.12 * sgn, 0.5, 0.05) * RY(0.35 * sgn if sgn > 0 else math.pi - 0.35) * RZ(0.25 * flap) * RX(-0.25)
        membrane_wing(sc, w, 0.78, -0.35, 1.3, 4, dark, Mat("#3a0e0c", rough=0.75, sss=0.6), thick=0.006, depth=0.22)
    # embers rising
    for k in range(5):
        a = TAU * (t + k / 5)
        sc.add(sd_sphere(0.01), glow("#ffb02a", 8.0), root * T(0.4 * math.cos(a), 0.2 + 0.8 * ((t + k / 5) % 1), 0.4 * math.sin(a)))
    return sc


SCALE = {"random": 1.4, "beginner": 1.3, "monte_carlo": 1.25, "mcts": 1.3, "basic_minmax": 1.25, "nnue193": 1.3,
         "features": 1.3, "resnet": 1.25, "dev": 1.25, "alphazero": 1.15}
CAST = {"random": ("Gobelin chapardeur", goblin), "beginner": ("Orque éclaireur", orc_scout),
        "monte_carlo": ("Rôdeuse elfe", elf_ranger), "mcts": ("Guerrier orque", orc_warrior),
        "basic_minmax": ("Troll des collines", hill_troll), "nnue193": ("Chevalier elfe", elf_knight),
        "features": ("Seigneur de guerre orque", orc_warlord), "resnet": ("Troll de guerre", war_troll),
        "dev": ("Archimage elfe", archmage), "alphazero": ("Dragon ancien", dragon)}
EPIC = {"leg_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"leg_" + k: n for k, (n, _) in CAST.items()}
