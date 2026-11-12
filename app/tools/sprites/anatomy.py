"""
anatomy.py — the Laboratoire set: a body rebuilt layer by layer, one stage per
AI, until it becomes a being of light (white and gold, an original design: no
blue skin, no symbol on the forehead):

  random        Système nerveux    the brain, the spinal cord, the nerves, glowing cyan
  beginner      Circulation        + the heart beating, arteries red, veins blue
  monte_carlo   Squelette          + the bones
  mcts          Organes            + lungs, liver, stomach, intestines
  basic_minmax  Écorché            + the muscles
  nnue193       Peau               the skin spreading in patches over the muscles
  features      Corps achevé       smooth, eyes closed, like a cast
  resnet        Éveil              veins of light under the skin, the eyes light up
  dev           Lévitation         the skin glows, rings of light turn around it, it floats
  alphazero     Être de lumière    a radiant body of white and gold, a halo, floating high

The layers add up (each stage keeps the ones before, inside); the seven first
stand on a scanner (a cyan ring sweeping up and down). One skeleton of joints
(joints) for every layer, so that they fit into one another.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm
from models import base, BASE_TOP, TAU, glow
from heroic import chain, bezier, HEAD


def joints(anim, t, lift=0.0, arms_up=False):
    """The joints in the body's frame (y up from the base's top), and the root transform."""
    breathe = 0.004 * math.sin(TAU * t)
    sway = 0.03 * math.sin(TAU * t) if anim == "idle" else 0.0
    hy = 0.48
    J = {"pelvis": (0, hy, 0), "chest": (0, hy + 0.22 + breathe, 0.01), "neck": (0, hy + 0.34, -0.005), "head": (0, hy + 0.43, 0.01)}
    J["spine"] = [(0, hy - 0.02 + k * 0.035, -0.03 + 0.012 * math.sin(k * 0.6)) for k in range(11)]
    up = arms_up or anim == "win"
    for sgn, side in ((-1, "l"), (1, "r")):
        sh = (0.15 * sgn, hy + 0.29, 0)
        sp = 2.5 if up else 0.32 + (0.15 * math.sin(TAU * t * 2) if anim == "blocked" else 0.0)
        el = (sh[0] + sgn * 0.17 * math.sin(sp), sh[1] - 0.17 * math.cos(sp), 0.01)
        wr = (el[0] + sgn * 0.155 * math.sin(sp * 1.05), el[1] - 0.155 * math.cos(sp * 1.05), 0.03)
        step = 0.2 * math.sin(TAU * t * 2) * sgn if anim == "move" else 0.0
        hp = (0.065 * sgn, hy - 0.03, 0)
        kn = (0.07 * sgn, hy - 0.265, 0.01 + step * 0.3)
        an = (0.07 * sgn, 0.03, -0.01 + step * 0.4)
        to = (0.075 * sgn, 0.012, 0.09 + step * 0.4)
        J[side] = {"sh": sh, "el": el, "wr": wr, "hp": hp, "kn": kn, "an": an, "to": to}
    shake = 0.25 * math.sin(TAU * 3 * t) if anim == "blocked" else 0.0
    hop = 0.03 * abs(math.sin(TAU * t)) if anim in ("move", "win") else 0.0
    root = RY(HEAD + shake) * T(0, BASE_TOP + lift + hop, 0) * RZ(sway * 0.1)
    return J, root


def lerp(a, b, u):
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))


def off(p, dx=0.0, dy=0.0, dz=0.0):
    return (p[0] + dx, p[1] + dy, p[2] + dz)


# ─── the layers ─────────────────────────────────────────────────────────────
def nerves(sc, J, root, k=5.0):
    m = glow("#7ae8ff", k)
    brain = Mat("#bff4ff", rough=0.4, emissive="#7ae8ff", emit=k * 0.3,
                pattern=lambda pl, pw, n: {"albedo": srgb("#bff4ff") * (0.6 + 0.6 * (torch.sin(fbm(pl * 60, 3) * 30) > 0).float())[:, None]})
    sc.add(sd_ellipsoid(0.052, 0.045, 0.06), brain, root * T(*off(J["head"], 0, 0.012, -0.005)))
    sc.add(sd_ellipsoid(0.028, 0.02, 0.022), brain, root * T(*off(J["head"], 0, -0.02, -0.04)))
    chain(sc, [off(J["head"], 0, -0.03, -0.03)] + J["spine"][::-1], [0.009] * 12, m, root, blend=0.004)
    for side in ("l", "r"):
        L = J[side]
        sgn = 1 if side == "r" else -1
        chain(sc, [off(J["spine"][9], 0.02 * sgn), off(L["sh"], -0.02 * sgn, -0.02), L["el"], L["wr"]], [0.006, 0.005, 0.004, 0.003], m, root, blend=0.003)
        for f in range(4):   # into the fingers
            sc.add(sd_capsule(L["wr"], off(L["wr"], sgn * 0.01 * (f - 1.5), -0.05, 0.01 * f), 0.0022), m, root)
        chain(sc, [off(J["spine"][0], 0.01 * sgn), L["hp"], L["kn"], L["an"], L["to"]], [0.007, 0.006, 0.005, 0.004, 0.003], m, root, blend=0.003)
        for i in range(2, 9, 2):   # intercostal nerves, around the chest
            s0 = J["spine"][i]
            arc = bezier(s0, off(s0, 0.09 * sgn, 0.0, 0.0), off(s0, 0.1 * sgn, -0.01, 0.07), off(s0, 0.02 * sgn, -0.02, 0.1), 5)
            chain(sc, arc, [0.003] * 5, m, root)


def vessels(sc, J, root, t):
    art, vein = glow("#ff3a3a", 3.5), glow("#4a6aff", 3.0)
    beat = 1 + 0.12 * max(0.0, math.sin(TAU * t * 2)) ** 4
    heart = off(J["chest"], 0.02, -0.01, 0.03)
    sc.add(sd_ellipsoid(0.03 * beat, 0.04 * beat, 0.03 * beat), Mat("#c8282a", rough=0.4, sss=0.6, emissive="#ff3a3a", emit=1.5), root * T(*heart) * RZ(0.4))
    aorta = bezier(heart, off(heart, 0, 0.06, -0.02), off(J["spine"][7], 0, 0, 0.02), off(J["spine"][0], 0, 0, 0.02), 6)
    chain(sc, aorta, [0.009] * 6, art, root, blend=0.004)
    for sgn in (-1, 1):
        chain(sc, [heart, off(J["neck"], 0.02 * sgn, 0, 0.02), off(J["head"], 0.04 * sgn, -0.02, 0.02)], [0.006, 0.005, 0.004], art, root)
        chain(sc, [off(heart, -0.02), off(J["neck"], 0.03 * sgn, 0, 0.0), off(J["head"], 0.05 * sgn, 0.0, -0.01)], [0.006, 0.005, 0.004], vein, root)
    for side in ("l", "r"):
        L = J[side]
        sgn = 1 if side == "r" else -1
        chain(sc, [heart, off(L["sh"], 0, -0.01, 0.01), off(L["el"], 0, 0, 0.012), off(L["wr"], 0, 0, 0.01)], [0.006, 0.005, 0.004, 0.003], art, root, blend=0.002)
        chain(sc, [off(heart, -0.01), off(L["sh"], 0, -0.03, -0.01), off(L["el"], 0, 0, -0.012), off(L["wr"], 0, 0, -0.01)], [0.006, 0.005, 0.004, 0.003], vein, root, blend=0.002)
        chain(sc, [off(J["spine"][0], 0, 0, 0.02), off(L["hp"], 0, 0, 0.02), off(L["kn"], 0, 0, 0.015), off(L["an"], 0, 0, 0.01)], [0.008, 0.006, 0.005, 0.004], art, root, blend=0.002)
        chain(sc, [off(J["spine"][0], 0.01 * sgn, 0, -0.01), off(L["hp"], 0.01 * sgn, 0, -0.015), off(L["kn"], 0.008 * sgn, 0, -0.015), off(L["an"], 0.006 * sgn, 0, -0.012)],
              [0.008, 0.006, 0.005, 0.004], vein, root, blend=0.002)


BONE = Mat("#ece2cc", rough=0.45, sss=0.3, pattern=lambda pl, pw, n: {"albedo": srgb("#ece2cc") * (0.85 + 0.2 * fbm(pl * 50, 2))[:, None]})


def bones(sc, J, root):
    hd = J["head"]
    sc.add(sd_ellipsoid(0.058, 0.062, 0.066), BONE, root * T(*hd))
    sc.add(sd_box(0.04, 0.025, 0.04, 0.015), BONE, root * T(*off(hd, 0, -0.05, 0.03)), blend=0.015)   # the jaw
    for sgn in (-1, 1):   # the eye sockets
        sc.add(sd_sphere(0.017), BONE, root * T(*off(hd, 0.024 * sgn, 0.0, 0.058)), op="sub", blend=0.004)
    for p in J["spine"]:
        sc.add(sd_cylinder(0.008, 0.016, 0.004), BONE, root * T(*p))
    sc.add(sd_capsule(off(J["chest"], 0, 0.08, 0.075), off(J["chest"], 0, -0.06, 0.08), 0.008), BONE, root)   # the sternum
    for i in range(3, 10):   # the ribs
        s0 = J["spine"][i]
        w = 0.07 + 0.025 * math.sin(math.pi * (i - 2) / 8)
        for sgn in (-1, 1):
            rib = bezier(s0, off(s0, w * sgn, -0.01, -0.01), off(s0, (w + 0.01) * sgn, -0.03, 0.08), off(J["chest"], 0.015 * sgn, (i - 6) * 0.02 - 0.0, 0.08), 6)
            chain(sc, rib, [0.006] * 6, BONE, root)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.055, 0.045, 0.025), BONE, root * T(*off(J["pelvis"], 0.05 * sgn, 0.0, -0.005)) * RY(0.5 * sgn) * RZ(0.4 * sgn))   # the pelvis
        L = J["r" if sgn > 0 else "l"]
        sc.add(sd_capsule(off(J["neck"], 0.01 * sgn, -0.04, 0.03), off(L["sh"], 0, 0, 0.02), 0.007), BONE, root)   # the clavicles
        for a, b, r in ((L["sh"], L["el"], 0.011), (L["el"], L["wr"], 0.009), (L["hp"], L["kn"], 0.014), (L["kn"], L["an"], 0.012)):
            sc.add(sd_capsule(a, b, r), BONE, root)
            sc.add(sd_sphere(r * 1.6), BONE, root * T(*b), blend=0.006)
        for f in range(4):
            sc.add(sd_capsule(L["wr"], off(L["wr"], sgn * 0.01 * (f - 1.5), -0.045, 0.01 * f), 0.004), BONE, root)
        sc.add(sd_capsule(L["an"], L["to"], 0.01), BONE, root)


def organs(sc, J, root):
    lung = Mat("#d88a8a", rough=0.6, sss=0.7)
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(0.045, 0.075, 0.045), lung, root * T(*off(J["chest"], 0.045 * sgn, 0.02, 0.02)))
    sc.add(sd_ellipsoid(0.06, 0.035, 0.045), Mat("#7a2a24", rough=0.5, sss=0.5), root * T(*off(J["chest"], -0.03, -0.075, 0.03)) * RZ(0.3))   # the liver
    sc.add(sd_ellipsoid(0.035, 0.03, 0.03), Mat("#d8a08a", rough=0.6, sss=0.5), root * T(*off(J["chest"], 0.04, -0.08, 0.04)))           # the stomach
    coil = [off(J["pelvis"], 0.045 * math.cos(a), 0.06 + 0.008 * a - 0.04, 0.035 + 0.03 * math.sin(a)) for a in [k * 0.6 for k in range(16)]]
    chain(sc, coil, [0.013] * 16, Mat("#e0a89a", rough=0.6, sss=0.6), root, blend=0.006)


def body_volume(sc, J, root, m, blend=0.02, face=None):
    """The muscles / the skin: the same volumes (blend small: muscle groups apart; large: a smooth skin)."""
    hd = J["head"]
    sc.add(sd_ellipsoid(0.06, 0.07, 0.068), m, root * T(*hd))
    sc.add(sd_ellipsoid(0.045, 0.04, 0.045), m, root * T(*off(hd, 0, -0.04, 0.025)), blend=blend * 1.5)
    sc.add(sd_capsule(off(J["neck"], 0, -0.04, 0), off(hd, 0, -0.04, -0.01), 0.032), m, root, blend=blend)
    sc.add(sd_ellipsoid(0.13, 0.1, 0.075), m, root * T(*off(J["chest"], 0, 0.02, 0)), blend=blend)
    sc.add(sd_ellipsoid(0.1, 0.11, 0.065), m, root * T(*off(J["pelvis"], 0, 0.1, 0.005)), blend=blend)
    sc.add(sd_ellipsoid(0.11, 0.075, 0.07), m, root * T(*off(J["pelvis"], 0, 0.0, 0)), blend=blend)
    for side in ("l", "r"):
        L = J[side]
        sgn = 1 if side == "r" else -1
        sc.add(sd_ellipsoid(0.055, 0.045, 0.02), m, root * T(*off(J["chest"], 0.055 * sgn, 0.04, 0.06)), blend=blend)   # pectoral
        sc.add(sd_sphere(0.045), m, root * T(*off(L["sh"], 0, -0.01, 0)), blend=blend)                                  # deltoid
        sc.add(sd_capsule(L["sh"], L["el"], 0.036, 0.028), m, root, blend=blend)
        sc.add(sd_capsule(L["el"], L["wr"], 0.03, 0.02), m, root, blend=blend)
        sc.add(sd_ellipsoid(0.02, 0.03, 0.012), m, root * T(*off(L["wr"], 0, -0.03, 0.005)), blend=blend * 0.5)          # hand
        sc.add(sd_ellipsoid(0.06, 0.05, 0.055), m, root * T(*off(L["hp"], 0, -0.01, -0.025)), blend=blend)            # gluteal
        sc.add(sd_capsule(L["hp"], L["kn"], 0.055, 0.04), m, root, blend=blend)
        sc.add(sd_capsule(L["kn"], L["an"], 0.04, 0.026), m, root, blend=blend)
        sc.add(sd_ellipsoid(0.034, 0.07, 0.04), m, root * T(*lerp(L["kn"], L["an"], 0.3)) * T(0, 0, -0.012), blend=blend)   # calf
        sc.add(sd_capsule(L["an"], L["to"], 0.026, 0.02), m, root, blend=blend)
    if face:   # a simple face: a nose, the brow, closed (or glowing) eyes
        sc.add(sd_ellipsoid(0.01, 0.02, 0.012), m, root * T(*off(hd, 0, -0.01, 0.068)), blend=0.01)
        sc.add(sd_ellipsoid(0.045, 0.01, 0.015), m, root * T(*off(hd, 0, 0.018, 0.055)), blend=0.01)
        for sgn in (-1, 1):
            sc.add(sd_ellipsoid(0.013, 0.003, 0.006), face, root * T(*off(hd, 0.022 * sgn, 0.004, 0.064)))


def muscle_mat():
    def pat(pl, pw, n):
        fib = (0.72 + 0.28 * torch.sin((pl[:, 0] + pl[:, 2]) * 300 + fbm(pl * 12, 2) * 8))[:, None]   # fibres along the limbs
        return {"albedo": srgb("#a8302a") * fib * (0.85 + 0.25 * fbm(pl * 30, 2))[:, None]}
    return Mat("#a8302a", rough=0.45, sss=0.5, clearcoat=0.4, pattern=pat)


SKIN = "#d8c4b4"


def skin_mat(cover=1.0, veins=0.0, glow_k=0.0):
    """cover: the share of the body the skin covers (patches growing from the noise);
    veins: veins of gold light under it; glow_k: the whole skin emitting light."""
    def pat(pl, pw, n):
        a = srgb(SKIN) * (0.92 + 0.12 * fbm(pl * 40, 2))[:, None]
        out = {}
        if cover < 1.0:
            m = (fbm(pl * 9, 3) < cover).float()[:, None]
            mus = srgb("#a8302a") * (0.72 + 0.28 * torch.sin((pl[:, 0] + pl[:, 2]) * 300))[:, None]
            a = a * m + mus * (1 - m)
        if veins > 0:
            v = (torch.abs(torch.sin(fbm(pl * 14, 3) * 18)) < 0.12).float()[:, None]
            out["emissive"] = srgb("#ffd27a") * v * veins * 3
            a = a * (1 - 0.3 * v) + srgb("#ffe8b0") * 0.3 * v
        out["albedo"] = a
        return out
    if glow_k > 0:
        return Mat("#fff2d8", rough=0.3, sss=0.8, emissive="#ffe0a0", emit=glow_k,
                   pattern=lambda pl, pw, n: {"albedo": srgb("#fff2d8") * (0.9 + 0.15 * fbm(pl * 30, 2))[:, None]})
    return Mat(SKIN, rough=0.5, sss=0.6, pattern=pat)


def scanner(sc, t):
    h = BASE_TOP + 0.05 + 0.85 * (0.5 - 0.5 * math.cos(TAU * t))
    sc.add(sd_torus(0.32, 0.006), glow("#5ad8ff", 6.0), T(0, h, 0))
    sc.add(sd_torus(0.5, 0.01), glow("#5ad8ff", 4.0), T(0, BASE_TOP + 0.004, 0))


def light_rings(sc, t, n, k, r=0.34, y0=0.3):
    for i in range(n):
        a = TAU * (t + i / n)
        sc.add(sd_torus(r + 0.04 * i, 0.007), glow("#ffe0a0", k), T(0, BASE_TOP + y0 + 0.2 * i, 0) * RX(0.5 * math.sin(a)) * RZ(0.5 * math.cos(a)))


# ─── the stages ─────────────────────────────────────────────────────────────
def stage(level):
    def f(anim, t, side):
        sc = Scene()
        base(sc, side)
        lift = {8: 0.0, 9: 0.08 + 0.02 * math.sin(TAU * t), 10: 0.14 + 0.03 * math.sin(TAU * t)}.get(level, 0.0)
        J, root = joints(anim, t, lift, arms_up=level == 10)
        if level <= 7:
            scanner(sc, t)
        if level <= 4:
            nerves(sc, J, root, 5.0 if level == 1 else 3.0)
        if 2 <= level <= 4:
            vessels(sc, J, root, t)
        if 3 <= level <= 4:
            bones(sc, J, root)
        if level == 4:
            organs(sc, J, root)
        if level == 5:
            body_volume(sc, J, root, muscle_mat(), blend=0.008)
        if level == 6:
            body_volume(sc, J, root, skin_mat(cover=0.5 + 0.1 * math.sin(TAU * t)), blend=0.02, face=Mat("#3a2a24"))
        if level == 7:
            body_volume(sc, J, root, skin_mat(), blend=0.025, face=Mat("#6a5448", rough=0.6))
        if level == 8:
            awake = 0.5 + 0.5 * math.sin(TAU * t)
            body_volume(sc, J, root, skin_mat(veins=0.6 + 0.4 * awake), blend=0.025, face=glow("#ffd27a", 4.0 + 3.0 * awake))
        if level == 9:
            body_volume(sc, J, root, skin_mat(glow_k=0.8), blend=0.025, face=glow("#ffffff", 8.0))
            light_rings(sc, t, 2, 5.0)
        if level == 10:
            body_volume(sc, J, root, skin_mat(glow_k=2.5), blend=0.03, face=glow("#ffffff", 10.0))
            hd = J["head"]
            sc.add(sd_torus(0.1, 0.008), glow("#ffe0a0", 9.0), root * T(*off(hd, 0, 0.02, -0.06)))      # the halo
            for k in range(10):   # rays around it
                a = TAU * k / 10 + t
                sc.add(sd_capsule((0, 0, 0), (0.18 * math.cos(a), 0.18 * math.sin(a), 0), 0.004, 0.001), glow("#fff2c8", 6.0),
                       root * T(*off(hd, 0, 0.02, -0.07)))
            light_rings(sc, t, 3, 7.0, r=0.38, y0=0.15)
            for k in range(6):   # sparks rising
                a = TAU * (t + k / 6)
                sc.add(sd_sphere(0.01), glow("#fff2c8", 8.0), T(0.4 * math.cos(a), BASE_TOP + 0.1 + 0.8 * ((t + k / 6) % 1), 0.4 * math.sin(a)))
        return sc
    return f


SCALE = {k: 1.1 for k in ("random", "beginner", "monte_carlo", "mcts", "basic_minmax", "nnue193", "features", "resnet", "dev", "alphazero")}
CAST = {"random": ("Système nerveux", stage(1)), "beginner": ("Circulation", stage(2)), "monte_carlo": ("Squelette", stage(3)),
        "mcts": ("Organes", stage(4)), "basic_minmax": ("Écorché", stage(5)), "nnue193": ("Peau", stage(6)),
        "features": ("Corps achevé", stage(7)), "resnet": ("Éveil", stage(8)), "dev": ("Lévitation", stage(9)),
        "alphazero": ("Être de lumière", stage(10))}
ANATOMY = {"lab_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"lab_" + k: n for k, (n, _) in CAST.items()}
