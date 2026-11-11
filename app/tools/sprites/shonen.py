"""
shonen.py — the Manga set, third version: one character per AI, adult
proportions and fierce looks (no chibi heads, no big round eyes), in the toon
mode (flat bands, ink lines). From a lone wolf to a dragon god:

  random        Ōkami       a lean grey wolf, scarred
  beginner      Shinobi     a masked ninja, a red scarf, kunai
  monte_carlo   Kitsune     a white fox spirit, three tails of blue fire
  mcts          Samouraï    red lacquered armour, a crescent helm, a katana
  basic_minmax  Oni         a red ogre, horns, an iron club
  nnue193       Onmyōji     a sorcerer in white, a tall black hat, talismans, a seal of light
  features      Raijū       a great thunder wolf, a mane of lightning
  resnet        Hōō         the phoenix, wings of fire
  dev           Kaigen      a martial artist at full power, an energy sphere, rocks lifted
  alphazero     Tenryū      the celestial dragon coiled around a pillar of light

Power shows as gems on the base, then an aura, lightning, floating rocks, a pillar.
The human is never part of a set: always the user's own figurine.
"""
import math
from render import Scene, Mat, T, RX, RY, RZ, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, sd_torus, \
    sd_cone, sd_plane_cut
from models import base, BASE_TOP, TAU, wave, glow
from manga import flat, gems, aura, anime_head, spikes_ring, INK
from legends import base_pose, held
from heroic import adult, chain, bezier, HEAD
from kaiju import lightning, rocks, pillar, wings


def slit_eyes(sc, x, w, h, col, k=3.0, tilt=0.35, dx=1.0):
    """Narrow glowing eyes, slanted: the eyes of a beast or a demon."""
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(w, h, w * 0.4), glow(col, k), x * T(dx * sgn, 0, 0) * RY(0.35 * sgn) * RZ(-tilt * sgn))


# ─── a canine (wolf, fox, thunder wolf) ─────────────────────────────────────
def canine(sc, anim, t, fur_c, belly_c, s=1.0, snout=1.0, ears=1.0):
    gait = TAU * t * (2 if anim == "move" else 0.5)
    lift = 0.03 * s * abs(math.sin(gait)) if anim == "move" else 0.0
    root = RY(HEAD + (0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0)) * T(0, BASE_TOP + 0.25 * s + lift, 0)
    fm, bm = flat(fur_c), flat(belly_c)
    sc.add(sd_ellipsoid(0.085 * s, 0.085 * s, 0.2 * s), fm, root * T(0, 0, -0.03 * s))
    sc.add(sd_ellipsoid(0.1 * s, 0.11 * s, 0.11 * s), fm, root * T(0, 0.02 * s, 0.1 * s), blend=0.04 * s)       # a deep chest
    sc.add(sd_ellipsoid(0.07 * s, 0.07 * s, 0.09 * s), bm, root * T(0, -0.03 * s, 0.13 * s), blend=0.02 * s)
    for i, (sx, sz) in enumerate(((-1, 1), (1, 1), (-1, -1), (1, -1))):
        ph = gait + (math.pi if i in (1, 2) else 0)
        sw = 0.4 * math.sin(ph) if anim == "move" else 0.0
        hip = root * T(0.06 * s * sx, -0.02 * s, (0.12 if sz > 0 else -0.15) * s) * RX(sw)
        if sz > 0:
            sc.add(sd_capsule((0, 0, 0), (0, -0.23 * s, 0.01 * s), 0.03 * s, 0.017 * s), fm, hip)
        else:
            sc.add(sd_ellipsoid(0.045 * s, 0.08 * s, 0.06 * s), fm, hip * T(0, -0.02 * s, 0), blend=0.02 * s)
            sc.add(sd_capsule((0, -0.06 * s, 0.02 * s), (0, -0.14 * s, -0.04 * s), 0.028 * s, 0.018 * s), fm, hip)
            sc.add(sd_capsule((0, -0.14 * s, -0.04 * s), (0, -0.23 * s, -0.01 * s), 0.018 * s, 0.016 * s), fm, hip)
        sc.add(sd_ellipsoid(0.022 * s, 0.012 * s, 0.03 * s), fm, hip * T(0, -0.235 * s, 0.015 * s))
    howl = anim == "win"
    nk = root * T(0, 0.06 * s, 0.16 * s)
    head = nk * T(0, 0.1 * s, 0.07 * s) * RX(-0.7 if howl else (0.15 if anim == "blocked" else 0.0))
    sc.add(sd_capsule((0, 0, 0), (0, 0.09 * s, 0.06 * s), 0.06 * s, 0.05 * s), fm, nk, blend=0.03 * s)
    sc.add(sd_ellipsoid(0.058 * s, 0.055 * s, 0.065 * s), fm, head)
    sc.add(sd_capsule((0, -0.01 * s, 0.03 * s), (0, -0.025 * s, 0.13 * s * snout), 0.034 * s, 0.018 * s), fm, head, blend=0.02 * s)   # the muzzle
    sc.add(sd_sphere(0.013 * s), flat(INK), head * T(0, -0.02 * s, 0.135 * s * snout + 0.012 * s))
    if howl or anim == "blocked":
        sc.add(sd_box(0.015 * s, 0.008 * s, 0.05 * s, 0.004 * s), flat("#5a0a14"), head * T(0, -0.045 * s, 0.08 * s * snout))
    for sgn in (-1, 1):
        sc.add(sd_cone(0.06 * s * ears, 0.026 * s, 0.003 * s, 0.002), fm, head * T(0.035 * sgn, 0.07 * s, -0.01 * s) * RZ(-0.2 * sgn) * T(0, 0.02 * s, 0))
        sc.add(sd_ellipsoid(0.018 * s, 0.006 * s, 0.006 * s), flat(INK), head * T(0.032 * sgn * s, 0.018 * s, 0.05 * s) * RY(0.4 * sgn) * RZ(-0.35 * sgn))
    return root, head, fm


# ─── 1 · Ōkami ──────────────────────────────────────────────────────────────
def okami(anim, t, side):
    sc = Scene()
    base(sc, side)
    root, head, fm = canine(sc, anim, t, "#8a8e98", "#d8dae0", s=1.0)
    slit_eyes(sc, head * T(0, 0.02, 0.055), 0.012, 0.005, "#ffd23a", 2.5, dx=0.03)
    sc.add(sd_capsule((0, 0, 0), (0.0, -0.03, 0.03), 0.004), flat("#c86a6a"), head * T(0.03, 0.04, 0.05))   # a scar over the eye
    tail = bezier((0, 0.02, -0.21), (0, 0.0, -0.3), (0.02, -0.1, -0.33), (0.03, -0.16, -0.3), 5)
    chain(sc, tail, [0.03, 0.04, 0.035, 0.03, 0.012], fm, root, blend=0.015)
    gems(sc, 1)
    return sc


# ─── 2 · Shinobi ────────────────────────────────────────────────────────────
def shinobi(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.lean += 0.12
    p.r = [0.9, 0.45, 1.3]
    p.l = [0.25, 0.4, 0.6]
    if anim == "win":
        p.r = [2.6, 0.2, 0.15]
    navy = flat("#1e2436")
    m = {"skin": flat("#e8c4a8"), "top": navy, "legs": navy, "boots": flat("#14161e"), "arms": navy, "hands": flat("#14161e")}
    rig = adult(sc, p, m, H=1.0, bulk=0.95)
    x, r = rig["head"], rig["hr"]
    sc.add(sd_ellipsoid(r, r * 1.05, r), navy, x)                                            # the hood
    sc.add(sd_box(r * 0.8, r * 0.16, r * 0.1, r * 0.05), flat("#e8c4a8"), x * T(0, r * 0.15, r * 0.88))   # the eye slit
    slit_eyes(sc, x * T(0, r * 0.15, r * 0.97), r * 0.14, r * 0.05, "#3a1a1a", 0.0, 0.25, dx=r * 0.32)
    sc.add(sd_box(r * 0.7, r * 0.06, r * 0.06, r * 0.02), flat("#9aa0aa", 0.3, 1.0), x * T(0, r * 0.5, r * 0.85))   # a forehead plate
    red = flat("#c81e2a")
    sc.add(sd_torus(0.05, 0.016), red, rig["torso"] * T(0, 0.31, 0.01))                     # the scarf, flowing behind
    for k in range(2):
        f = 0.6 + 0.25 * math.sin(TAU * (t * 2 + k * 0.3))
        sc.add(sd_box(0.022, 0.13, 0.005, 0.004), red, rig["torso"] * T(0.02 * k, 0.3, -0.06) * RX(1.3 * f) * RZ(0.3 - 0.4 * k) * T(0, -0.12, 0))
    sc.add(sd_torus(0.08, 0.012), red, rig["waist"])
    kn = held(rig["r"], -math.pi if anim == "win" else -1.5)
    sc.add(sd_capsule((0, -0.04, 0), (0, 0.03, 0), 0.008), flat("#14161e"), kn)
    sc.add(sd_torus(0.012, 0.004), flat("#9aa0aa", 0.3, 1.0), kn * T(0, -0.055, 0))
    sc.add(sd_cone(0.06, 0.016, 0.001, 0.001), flat("#c8ced8", 0.2, 1.0), kn * T(0, 0.09, 0))
    sc.add(sd_box(0.07, 0.035, 0.004, 0.002), flat("#8a6a3a"), rig["back"] * T(0, 0.02, -0.02) * RZ(0.7))   # a sheathed short blade
    gems(sc, 1)
    return sc


# ─── 3 · Kitsune ────────────────────────────────────────────────────────────
def kitsune(anim, t, side):
    sc = Scene()
    base(sc, side)
    root, head, fm = canine(sc, anim, t, "#f4f2ec", "#ffffff", s=1.05, snout=1.15, ears=1.3)
    slit_eyes(sc, head * T(0, 0.022, 0.058), 0.014, 0.005, "#ff3a5a", 3.0, 0.45, dx=0.03)
    for sgn in (-1, 1):   # red markings
        sc.add(sd_capsule((0, 0, 0), (0.025 * sgn, 0.03, -0.02), 0.005), flat("#d81e3a"), head * T(0.02 * sgn, 0.04, 0.05))
    for k, a in enumerate((-0.8, 0.0, 0.8)):
        sw = 0.15 * math.sin(TAU * (t + k * 0.3))
        tl = root * T(0, 0.03, -0.2) * RZ(-a * 0.7 + sw) * RX(-1.1)
        pts = [(0, 0, 0), (0, 0.1, -0.03), (0, 0.2, -0.02), (0, 0.29, 0.03)]
        chain(sc, pts, [0.03, 0.045, 0.04, 0.02], fm, tl, blend=0.02)
        sc.add(sd_cone(0.06 + 0.01 * math.sin(TAU * t * 3 + k), 0.03, 0.003, 0.002), glow("#4ac8ff", 7.0), tl * T(0, 0.33, 0.04))
    for k in range(3):
        a = TAU * (t + k / 3)
        sc.add(sd_sphere(0.02), glow("#4ac8ff", 6.0), root * T(0.32 * math.cos(a), 0.2 + 0.05 * math.sin(2 * a), 0.32 * math.sin(a)))
    gems(sc, 2)
    return sc


# ─── 4 · Samouraï ───────────────────────────────────────────────────────────
def samurai(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [1.0, 0.1, 0.9]
    p.l = [1.0, -0.15, 1.0]          # both hands on the katana, in guard
    if anim == "win":
        p.r = [2.7, 0.1, 0.2]
        p.l = [2.7, -0.05, 0.2]
    lac = flat("#a8141e", 0.35)
    blk = flat("#1a1a20")
    gold_ = flat("#e8b030", 0.3, 1.0)
    m = {"skin": flat("#e8c4a8"), "top": lac, "legs": blk, "boots": blk, "arms": blk, "fore": lac, "hands": blk}
    rig = adult(sc, p, m, H=1.02, bulk=1.15, head=1.05)
    for sgn in (-1, 1):   # sode: shoulder plates in rows
        for k in range(3):
            sc.add(sd_box(0.07, 0.012, 0.06, 0.006), lac, rig["torso"] * T(0.2 * sgn, 0.3 - 0.03 * k, 0) * RZ(-0.35 * sgn))
    for k in range(3):    # kusazuri: the skirt of plates
        sc.add(sd_cone(0.04, 0.11 + 0.01 * k, 0.1 + 0.01 * k, 0.006), lac, rig["waist"] * T(0, -0.08 - 0.035 * k, 0))
    sc.add(sd_torus(0.09, 0.01), gold_, rig["waist"] * T(0, -0.03, 0))
    x, r = rig["head"], rig["hr"]
    sc.add(sd_ellipsoid(r, r, r), flat("#e8c4a8"), x)
    sc.add(sd_box(r * 0.75, r * 0.45, r * 0.3, r * 0.15), flat("#2a2a30", 0.4), x * T(0, -r * 0.45, r * 0.75))   # menpō, the mask
    sc.add(sd_box(r * 0.4, r * 0.04, r * 0.05, r * 0.01), flat("#f2f2f2"), x * T(0, -r * 0.5, r * 1.04))
    slit_eyes(sc, x * T(0, r * 0.12, r * 0.92), r * 0.15, r * 0.05, "#1a1010", 0.0, 0.3, dx=r * 0.33)
    sc.add(sd_plane_cut(sd_ellipsoid(r * 1.2, r * 1.05, r * 1.2), (0, -1, 0), -r * 0.05), blk, x * T(0, r * 0.2, -r * 0.05))   # kabuto
    sc.add(sd_cone(r * 0.25, r * 1.5, r * 1.3, r * 0.05), blk, x * T(0, r * 0.05, -r * 0.1))         # the neck guard
    for sgn in (-1, 1):   # the golden crescent
        sc.add(sd_capsule((0, 0, 0), (r * 1.1 * sgn, r * 1.2, r * 0.1), r * 0.12, r * 0.03), gold_, x * T(0, r * 0.7, r * 0.75))
    kt = held(rig["r"], -math.pi if anim == "win" else -1.0, -0.1)
    sc.add(sd_capsule((0, -0.12, 0), (0, 0.03, 0), 0.012), flat("#2a1a2a"), kt)
    sc.add(sd_cylinder(0.006, 0.026, 0.002), gold_, kt * T(0, 0.04, 0))
    sc.add(lambda q: sd_box(0.011, 0.25, 0.003, 0.002)(q), flat("#e8f0f8", 0.15, 1.0), kt * T(0, 0.29, 0) * RZ(-0.05))
    if anim == "win":
        sc.add(sd_box(0.004, 0.24, 0.0035, 0.001), glow("#ffffff", 4.0), kt * T(0.006, 0.29, 0) * RZ(-0.05))
    gems(sc, 2)
    return sc


# ─── 5 · Oni ────────────────────────────────────────────────────────────────
def oni(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [0.6, 0.35, 0.8]
    p.l = [0.3, 0.45, 0.6]
    if anim == "win":
        p.r = [2.6, 0.25, 0.2]
        p.l = [1.8, 0.7, 0.4]
    red = flat("#c8322a")
    tiger = Mat("#e8a020", rough=0.6, pattern=lambda pl, pw, n: {
        "albedo": __import__("torch").where(((pl[:, 0] * 60 + pl[:, 1] * 30).sin() > 0.55)[:, None],
                                            __import__("render").srgb(INK).expand_as(pl), __import__("render").srgb("#e8a020").expand_as(pl))})
    m = {"skin": red, "top": red, "legs": tiger, "boots": red}
    rig = adult(sc, p, m, H=1.15, bulk=1.65, head=1.05, hunch=0.15)
    x, r = rig["head"], rig["hr"]
    sc.add(sd_ellipsoid(r, r * 1.0, r * 1.0), red, x)
    sc.add(sd_ellipsoid(r * 0.95, r * 0.55, r * 0.8), red, x * T(0, -r * 0.5, r * 0.25), blend=r * 0.3)
    sc.add(sd_ellipsoid(r * 0.9, r * 0.2, r * 0.3), red, x * T(0, r * 0.25, r * 0.78), blend=r * 0.15)
    slit_eyes(sc, x * T(0, r * 0.08, r * 0.9), r * 0.16, r * 0.07, "#ffe02a", 4.0, 0.5, dx=r * 0.35)
    sc.add(sd_ellipsoid(r * 0.55, r * (0.28 if anim != "idle" else 0.16), r * 0.2), flat("#3a0a0a"), x * T(0, -r * 0.55, r * 0.85))
    for sgn in (-1, 1):
        sc.add(sd_cone(r * 0.3, r * 0.08, r * 0.01, 0.002), flat("#f8f4e8"), x * T(r * 0.3 * sgn, -r * 0.65, r * 0.95) * T(0, r * 0.2, 0))   # fangs
        hp = bezier((0, 0, 0), (r * 0.3 * sgn, r * 0.5, 0), (r * 0.5 * sgn, r * 1.0, -r * 0.2), (r * 0.4 * sgn, r * 1.5, -r * 0.1), 5)
        chain(sc, hp, [r * 0.22, r * 0.18, r * 0.13, r * 0.08, r * 0.015], flat("#f2ead0"), x * T(r * 0.45 * sgn, r * 0.6, 0), blend=0.003)
    for k in range(9):    # wild white hair
        a = -2.4 + 0.6 * k
        sc.add(sd_cone(r * 0.9, r * 0.25, r * 0.02, 0.003), flat("#f2f2f2"), x * T(r * 0.85 * math.sin(a), r * 0.4, r * 0.85 * math.cos(a) - r * 0.1) * RY(a) * RX(-1.9) * T(0, r * 0.5, 0))
    kb = held(rig["r"], -math.pi if anim == "win" else -0.9)
    sc.add(sd_capsule((0, -0.05, 0), (0, 0.55, 0), 0.025, 0.06), flat("#2a2a32", 0.4, 0.8), kb)
    for k in range(5):
        for j in range(4):
            a = TAU * j / 4 + k * 0.6
            sc.add(sd_cone(0.03, 0.014, 0.001, 0.001), flat("#c8ccd4", 0.3, 1.0), kb * T(0.05 * math.cos(a) * (0.6 + 0.1 * k), 0.25 + 0.06 * k, 0.05 * math.sin(a) * (0.6 + 0.1 * k)) * RY(-a) * RZ(-math.pi / 2) * T(0, 0.012, 0))
    aura(sc, 0.15, "#ff6a2a", t, rig["body"])
    gems(sc, 3)
    return sc


# ─── 6 · Onmyōji ────────────────────────────────────────────────────────────
def onmyoji(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.r = [1.3, 0.2, 0.9]            # two fingers raised in a seal
    p.l = [0.4, 0.5, 0.8]
    if anim == "win":
        p.r = [2.6, 0.2, 0.15]
        p.l = [2.0, 0.8, 0.3]
    white = flat("#f6f4ee")
    m = {"skin": flat("#ecd0b8"), "top": white, "legs": flat("#7a2a8a"), "boots": flat("#1a1a20"), "arms": white}
    rig = adult(sc, p, m, H=1.04, bulk=1.0, head=1.0)
    sc.add(sd_cone(0.2, 0.17, 0.09, 0.02), flat("#7a2a8a"), rig["waist"] * T(0, -0.16, 0))       # hakama
    for sgn in (-1, 1):   # wide sleeves
        sc.add(sd_ellipsoid(0.06, 0.11, 0.05), white, rig["shoulder"][0 if sgn < 0 else 1] * T(0.04 * sgn, -0.12, 0), blend=0.02)
    x, r = rig["head"], rig["hr"]
    hair = {"c": "#16141c", "spikes": spikes_ring(5, -0.3, 0.9, 0.25, 2.3, 4.0)}
    anime_head(sc, x, r, "#ecd0b8", hair, "#b05ad8", "calm" if anim == "idle" else "determined")
    sc.add(sd_box(r * 0.55, r * 0.9, r * 0.5, r * 0.25), flat("#16141c", 0.3), x * T(0, r * 1.25, -r * 0.15) * RX(-0.3))   # eboshi
    for i in range(5):    # paper talismans orbiting
        a = TAU * (t * 0.7 + i / 5)
        q = rig["body"] * T(0.36 * math.cos(a), BASE_TOP + 0.4 + 0.12 * math.sin(2 * a + i), 0.36 * math.sin(a)) * RY(-a + math.pi / 2) * RZ(0.2 * math.sin(TAU * t + i))
        sc.add(sd_box(0.022, 0.05, 0.002, 0.001), flat("#f2e8c8"), q)
        sc.add(sd_box(0.006, 0.035, 0.0025, 0.001), glow("#d81e3a", 2.0), q)
    for k in range(5):    # a seal of light on the base: a star in a ring
        a0, a1 = TAU * k / 5, TAU * (k + 2) / 5
        r0 = 0.4
        sc.add(sd_capsule((r0 * math.cos(a0), 0, r0 * math.sin(a0)), (r0 * math.cos(a1), 0, r0 * math.sin(a1)), 0.006),
               glow("#c89aff", 4.0 + 2.0 * wave(t, 2)), RY(TAU * t * 0.25) * T(0, BASE_TOP + 0.005, 0))
    sc.add(sd_torus(0.42, 0.006), glow("#c89aff", 4.0), T(0, BASE_TOP + 0.005, 0))
    aura(sc, 0.3, "#c89aff", t, rig["body"])
    gems(sc, 3)
    return sc


# ─── 7 · Raijū ──────────────────────────────────────────────────────────────
def raiju(anim, t, side):
    sc = Scene()
    base(sc, side)
    root, head, fm = canine(sc, anim, t, "#2a3466", "#bfc8e8", s=1.45, ears=1.2)
    slit_eyes(sc, head * T(0, 0.03, 0.08), 0.017, 0.006, "#fff04a", 5.0, 0.45, dx=0.045)
    for sgn in (-1, 1):
        sc.add(sd_cone(0.03, 0.009, 0.001, 0.001), flat("#ffffff"), head * T(0.025 * sgn, -0.06, 0.15) * RX(math.pi))
    for k in range(9):    # a mane of lightning
        a = -1.4 + 0.35 * k
        sc.add(sd_cone(0.12, 0.03, 0.002, 0.002), glow("#ffe84a", 3.0), root * T(0.1 * math.sin(a), 0.14, 0.12 + 0.08 * math.cos(a)) * RZ(-0.7 * math.sin(a)) * RX(-0.7) * T(0, 0.08, 0))
    tail = bezier((0, 0.02, -0.29), (0, 0.1, -0.42), (0, 0.25, -0.45), (0.03, 0.35, -0.38), 6)
    for i, (a, b) in enumerate(zip(tail[:-1], tail[1:])):   # a zigzag tail, like a bolt
        sc.add(sd_capsule(a, (b[0] + 0.03 * (-1) ** i, b[1], b[2]), 0.035 * (1 - 0.15 * i), 0.03 * (1 - 0.15 * i)), fm, root, blend=0.01)
    for i in range(3 + (3 if anim == "win" else 0)):
        a = TAU * i / 3
        lightning(sc, (0.0, 0.15, 0.0), (0.45 * math.cos(a + t * 2), 0.55 + 0.2 * math.sin(a), 0.45 * math.sin(a + t * 2)), i + 1, t, "#ffe85a", 8.0, 0.006, root)
    aura(sc, 0.45, "#ffe85a", t, RY(HEAD))
    gems(sc, 4)
    return sc


# ─── 8 · Hōō ────────────────────────────────────────────────────────────────
def hoo(anim, t, side):
    sc = Scene()
    base(sc, side)
    flap = math.sin(TAU * t * (2 if anim in ("move", "win") else 1))
    hover = 0.05 * math.sin(TAU * t)
    root = RY(HEAD) * T(0, BASE_TOP + 0.45 + hover, 0)
    red, gold_ = flat("#d8321e"), flat("#ffb02a")
    sc.add(sd_ellipsoid(0.1, 0.17, 0.1), red, root * RX(-0.35))
    sc.add(sd_ellipsoid(0.07, 0.12, 0.05), gold_, root * T(0, 0.0, 0.06) * RX(-0.35))
    neck = bezier((0, 0.1, 0.03), (0, 0.2, 0.04), (0, 0.26, 0.1), (0, 0.3, 0.13), 5)
    chain(sc, neck, [0.06, 0.05, 0.045, 0.04, 0.04], red, root, blend=0.02)
    head = root * T(0, 0.32, 0.15)
    sc.add(sd_ellipsoid(0.05, 0.045, 0.06), red, head)
    sc.add(sd_cone(0.05, 0.022, 0.002, 0.002), flat("#ffd84a", 0.3, 0.8), head * T(0, -0.01, 0.075) * RX(math.pi / 2 + 0.4))   # a hooked beak
    slit_eyes(sc, head * T(0, 0.012, 0.04), 0.013, 0.005, "#fff8c0", 5.0, 0.5, dx=0.03)
    for k in range(5):    # a crest of flames
        sc.add(sd_cone(0.08 + 0.03 * (k % 2), 0.02, 0.002, 0.002), glow("#ffb02a", 6.0), head * T(0, 0.03, -0.02 - 0.015 * k) * RX(-0.7 - 0.2 * k) * T(0, 0.06, 0))
    wings(sc, root * T(0, 0.1, -0.04), 0.62, flap, red, 9, "#ffb02a")
    for k in range(7):    # the tail: long burning plumes, curling
        a = -0.6 + 0.2 * k
        pl = bezier((0, 0, 0), (0.15 * math.sin(a), -0.15, -0.15), (0.3 * math.sin(a), -0.3, -0.1), (0.38 * math.sin(a), -0.38, 0.05), 5)
        chain(sc, pl, [0.02, 0.018, 0.014, 0.01, 0.004], glow("#ff6a1a" if k % 2 else "#ffc84a", 5.0), root * T(0, -0.14, -0.06), blend=0.005)
    for k in range(5):
        a = TAU * (t + k / 5)
        sc.add(sd_sphere(0.012), glow("#ffd04a", 8.0), root * T(0.4 * math.cos(a), -0.3 + 0.6 * ((t + k / 5) % 1), 0.4 * math.sin(a)))
    aura(sc, 0.65, "#ff7a2a", t, RY(HEAD))
    gems(sc, 4)
    return sc


# ─── 9 · Kaigen ─────────────────────────────────────────────────────────────
def kaigen(anim, t, side):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    p.l = [0.95, 0.35, 1.55]         # an energy sphere charged at the hip
    p.r = [0.95, 0.35, 1.55]
    if anim == "win":
        p.l = [2.65, 0.3, 0.1]
        p.r = [2.65, 0.3, 0.1]
    sk = flat("#e0b090")
    gi = flat("#1c1c24")
    m = {"skin": sk, "top": sk, "legs": gi, "boots": flat("#14141a"), "arms": sk}
    rig = adult(sc, p, m, H=1.08, bulk=1.4, head=1.0)
    sc.add(sd_ellipsoid(0.07, 0.08, 0.03), sk, rig["chest"] * T(0.05, 0.01, 0.0), blend=0.02)     # pectorals
    sc.add(sd_ellipsoid(0.07, 0.08, 0.03), sk, rig["chest"] * T(-0.05, 0.01, 0.0), blend=0.02)
    for sgn in (-1, 1):   # the torn gi: a vest open on the chest
        sc.add(sd_box(0.05, 0.11, 0.07, 0.02), gi, rig["torso"] * T(0.11 * sgn, 0.2, 0.0) * RZ(0.1 * sgn))
    sc.add(sd_torus(0.095, 0.014), flat("#e8e8e8"), rig["waist"])
    x, r = rig["head"], rig["hr"]
    hair = {"c": "#eef2ff", "spikes": spikes_ring(11, 0.9, 1.6, 0.3, jitter=0.25) + spikes_ring(7, 0.3, 1.2, 0.3, -1.2, 1.2, 0.2)}
    anime_head(sc, x * RX(-0.1), r, "#e0b090", hair, "#9a5aff", "fierce" if anim != "blocked" else "angry")
    ball = rig["torso"] * T(0, 0.05, 0.25) if anim != "win" else rig["body"] * T(0, 1.4, 0.05)
    k = 1.0 + 0.15 * math.sin(TAU * t * 4) + (0.8 if anim == "win" else 0)
    sc.add(sd_sphere(0.075 * k), glow("#c89aff", 12.0), ball)
    sc.add(sd_torus(0.11 * k, 0.006), glow("#ffffff", 8.0), ball * RX(TAU * t) * RZ(0.5))
    for i in range(4):
        a = TAU * i / 4 + t * 3
        lightning(sc, (0.0, 0.55, 0.0), (0.45 * math.cos(a), 0.2 + 0.7 * ((i * 0.37 + t) % 1), 0.45 * math.sin(a)), i + 3, t, "#e0c8ff", 9.0, 0.006, rig["body"])
    rocks(sc, 5, 0.45, 0.25, t, RY(HEAD))
    aura(sc, 0.9, "#9a5aff", t, rig["body"])
    gems(sc, 5)
    return sc


# ─── 10 · Tenryū ────────────────────────────────────────────────────────────
def tenryu(anim, t, side):
    sc = Scene()
    base(sc, side)
    root = RY(HEAD)
    pillar(sc, 1.3, "#fff4c8", 4.0 + 2.0 * wave(t, 2), root, r=0.05)
    scale = Mat("#2a7a5a", rough=0.35, metal=0.3)
    belly = flat("#e8d8a8")
    gold_ = flat("#ffc83a", 0.3, 1.0)
    n = 22
    pts, radii = [], []
    for i in range(n):
        u = i / (n - 1)
        a = TAU * (u * 2.2 + t * (0.4 if anim != "blocked" else 0.1))
        rr = 0.3 * (1 - 0.35 * u)
        pts.append((rr * math.cos(a), BASE_TOP + 0.08 + u * 1.05, rr * math.sin(a)))
        radii.append(0.07 * (0.55 + 0.7 * math.sin(math.pi * min(1.0, u * 1.3 + 0.1))))
    chain(sc, pts, radii, scale, root, blend=0.035)
    for i in range(0, n - 1):
        sc.add(sd_sphere(radii[i] * 0.8), belly, root * T(pts[i][0] * 0.82, pts[i][1] - radii[i] * 0.25, pts[i][2] * 0.82), blend=0.02)
    for i in range(1, n - 2, 2):
        sc.add(sd_cone(0.05, 0.025, 0.002, 0.002), gold_, root * T(pts[i][0], pts[i][1] + radii[i] * 0.95, pts[i][2]))
    for i in (5, 12):     # arms with claws
        for sgn in (-1, 1):
            sc.add(sd_capsule((0, 0, 0), (0.09 * sgn, -0.06, 0.06), 0.022, 0.012), scale, root * T(*pts[i]))
            for k in range(3):
                sc.add(sd_cone(0.025, 0.007, 0.001, 0.001), flat("#f2ead8"), root * T(*pts[i]) * T(0.09 * sgn + 0.01 * (k - 1), -0.07, 0.07) * RX(math.pi * 0.8))
    hp = pts[-1]
    head = root * T(hp[0], hp[1] + 0.08, hp[2]) * RY(-0.6 + 0.2 * math.sin(TAU * t)) * RX(0.15)
    sc.add(sd_ellipsoid(0.11, 0.08, 0.13), scale, head)
    sc.add(sd_box(0.07, 0.045, 0.1, 0.035), scale, head * T(0, -0.01, 0.13), blend=0.03)        # a long snout
    jaw = 0.5 if anim == "win" else 0.15
    sc.add(sd_box(0.06, 0.02, 0.1, 0.015), belly, head * T(0, -0.055, 0.05) * RX(jaw) * T(0, 0, 0.08))
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.07, 0.02, 0.06), scale, head * T(0.05 * sgn, 0.05, 0.05) * RZ(-0.35 * sgn), blend=0.01)   # brows
        slit_eyes(sc, head * T(0, 0.035, 0.1), 0.02, 0.007, "#ff2a3a", 6.0, 0.45, dx=0.06)
        for k in range(3):
            sc.add(sd_cone(0.025, 0.008, 0.001, 0.001), flat("#ffffff"), head * T(0.04 * sgn, -0.05, 0.12 + 0.035 * k) * RX(math.pi))
        sc.add(sd_capsule((0, 0, 0), (0.14 * sgn, 0.22, -0.2), 0.018, 0.004), gold_, head * T(0.05 * sgn, 0.07, -0.05))   # antlers
        sc.add(sd_capsule((0.07 * sgn, 0.11, -0.1), (0.13 * sgn, 0.18, -0.06), 0.009, 0.003), gold_, head * T(0.05 * sgn, 0.07, -0.05))
        sc.add(sd_capsule((0, 0, 0), (0.24 * sgn, -0.14, 0.12), 0.007, 0.002), gold_, head * T(0.06 * sgn, -0.03, 0.2))   # whiskers
    if anim == "win":
        sc.add(sd_cone(0.22, 0.07, 0.03, 0.01), glow("#fff4c8", 10.0), head * T(0, -0.04, 0.42) * RX(math.pi / 2))
    sc.add(sd_torus(0.4, 0.008), glow("#ffe88a", 6.0), root * T(0, BASE_TOP + 1.0, 0) * RX(0.35) * RY(TAU * t))
    sc.add(sd_sphere(0.06), glow("#ffffff", 14.0), root * T(0, BASE_TOP + 1.25 + 0.03 * math.sin(TAU * t), 0))
    rocks(sc, 6, 0.5, 0.35, t, root, seed=1.0)
    for i in range(3):
        a = TAU * i / 3 + t * 2
        lightning(sc, (0.0, 1.2, 0.0), (0.5 * math.cos(a), 0.3, 0.5 * math.sin(a)), i + 7, t, "#fff0a0", 9.0, 0.006, root)
    aura(sc, 1.0, "#ffd24a", t, root)
    gems(sc, 5)
    return sc


SCALE = {"random": 1.4, "beginner": 1.3, "monte_carlo": 1.35, "mcts": 1.3, "basic_minmax": 1.2, "nnue193": 1.3,
         "features": 1.15, "resnet": 1.15, "dev": 1.25, "alphazero": 1.0}
CAST = {"random": ("Ōkami", okami), "beginner": ("Shinobi", shinobi), "monte_carlo": ("Kitsune", kitsune),
        "mcts": ("Samouraï", samurai), "basic_minmax": ("Oni", oni), "nnue193": ("Onmyōji", onmyoji),
        "features": ("Raijū", raiju), "resnet": ("Hōō", hoo), "dev": ("Kaigen", kaigen), "alphazero": ("Tenryū", tenryu)}
SHONEN = {"manga_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"manga_" + k: n for k, (n, _) in CAST.items()}
