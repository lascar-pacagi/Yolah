"""
anatomy.py — the Laboratoire set: a body rebuilt layer by layer, one stage per
AI, until it becomes a being of light (white and gold, an original design: no
blue skin, no symbol on the forehead):

  random        Système nerveux    the brain, the spinal cord, the nerves, glowing cyan
  beginner      Circulation        + the heart beating, arteries red, veins blue
  monte_carlo   Squelette          + the bones
  mcts          Organes            + lungs, liver, stomach, intestines
  basic_minmax  Écorché            + the muscles
  minmax        Fascias            the muscles under the deep fascia, a thin glistening sheath
  mcts_nnue     Hypoderme          the yellow lobules of subcutaneous fat spreading over them
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
from render import X, F, DEV
import random


def joints(anim, t, lift=0.0, arms_up=None, spread=0.32):
    """The joints in the body's frame (y up from the base's top), and the root transform."""
    breathe = 0.004 * math.sin(TAU * t)
    sway = 0.03 * math.sin(TAU * t) if anim == "idle" else 0.0
    hy = 0.48
    J = {"pelvis": (0, hy, 0), "chest": (0, hy + 0.22 + breathe, 0.01), "neck": (0, hy + 0.34, -0.005), "head": (0, hy + 0.43, 0.01)}
    J["spine"] = [(0, hy - 0.02 + k * 0.035, -0.03 + 0.012 * math.sin(k * 0.6)) for k in range(11)]
    up = anim == "win" if arms_up is None else arms_up
    for sgn, side in ((-1, "l"), (1, "r")):
        sh = (0.15 * sgn, hy + 0.29, 0)
        sp = 2.5 if up else spread + (0.15 * math.sin(TAU * t * 2) if anim == "blocked" else 0.0)
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


# ─── helpers ────────────────────────────────────────────────────────────────
def along(a, b):
    """A frame at a whose local y goes to b (so that patterns run along a bone, a nerve, a muscle); and the length."""
    av, bv = torch.tensor(a, dtype=F, device=DEV), torch.tensor(b, dtype=F, device=DEV)
    d = bv - av
    n = float(torch.linalg.norm(d))
    y = d / max(n, 1e-6)
    ref = torch.tensor((0.0, 0.0, 1.0) if abs(float(y[2])) < 0.9 else (1.0, 0.0, 0.0), dtype=F, device=DEV)
    x = torch.linalg.cross(y, ref)
    x = x / torch.linalg.norm(x)
    z = torch.linalg.cross(x, y)
    return X(torch.stack([x, y, z], dim=1), t=av), n


def sd_thin(rx, ry, rz):
    """A flat ellipsoid bounded by its sphere (shadow rays would stop short far from it)."""
    f, R = sd_ellipsoid(rx, ry, rz), max(rx, ry, rz)
    return lambda p: torch.maximum(f(p), torch.linalg.norm(p, dim=-1) - R)


def twigs(sc, pts, m, root, n, seed, length=0.03, r=0.0016):
    """Fine branches along a nerve (pts), deterministic."""
    rng = random.Random(seed)
    for _ in range(n):
        i = rng.randrange(len(pts) - 1)
        u = rng.random()
        p = lerp(pts[i], pts[i + 1], u)
        d = (rng.uniform(-1, 1), rng.uniform(-1, 0.3), rng.uniform(-1, 1))
        L = length * rng.uniform(0.6, 1.2)
        nd = math.sqrt(sum(c * c for c in d)) or 1.0
        q = off(p, d[0] / nd * L, d[1] / nd * L, d[2] / nd * L)
        sc.add(sd_capsule(p, q, r, r * 0.5), m, root)


# ─── the layers ─────────────────────────────────────────────────────────────
def nerves(sc, J, root, k=3.0):
    """The central nervous system and the main nerves, with their branches."""
    m, fine = glow("#7ae8ff", k), glow("#7ae8ff", k * 0.6)
    brain = Mat("#bff4ff", rough=0.4, emissive="#7ae8ff", emit=k * 0.3,
                pattern=lambda pl, pw, n: {"albedo": srgb("#bff4ff") * (0.55 + 0.6 * (torch.sin(fbm(pl * 70, 3) * 28) > 0).float())[:, None]})
    hd = J["head"]
    sc.add(sd_ellipsoid(0.05, 0.042, 0.06), brain, root * T(*off(hd, 0, 0.014, -0.004)))      # the cerebrum, split in two hemispheres
    sc.add(sd_box(0.0025, 0.05, 0.07), brain, root * T(*off(hd, 0, 0.05, -0.004)), op="sub", blend=0.004)
    for sgn in (-1, 1):   # the cerebellum, the brainstem
        sc.add(sd_ellipsoid(0.02, 0.016, 0.02), brain, root * T(*off(hd, 0.017 * sgn, -0.024, -0.04)), blend=0.004)
    stem = [off(hd, 0, -0.01, -0.02), off(hd, 0, -0.045, -0.028)]
    sc.add(sd_capsule(stem[0], stem[1], 0.009, 0.007), brain, root, blend=0.004)
    cord = [stem[1]] + J["spine"][::-1]
    chain(sc, cord, [0.006 - 0.002 * i / len(cord) for i in range(len(cord))], m, root, blend=0.003)
    s0 = J["spine"][0]
    for k_ in range(5):   # the cauda equina
        sc.add(sd_capsule(s0, off(s0, 0.01 * (k_ - 2), -0.06, -0.005), 0.002), fine, root)
    for sgn in (-1, 1):
        # cranial nerves to the face: forehead, cheek, jaw
        ear = off(hd, 0.042 * sgn, -0.012, 0.005)
        for tgt in (off(hd, 0.03 * sgn, 0.035, 0.055), off(hd, 0.035 * sgn, -0.01, 0.06), off(hd, 0.025 * sgn, -0.05, 0.05)):
            chain(sc, bezier(stem[0], ear, lerp(ear, tgt, 0.5), tgt, 5), [0.0026] * 5, fine, root)
        L = J["r" if sgn > 0 else "l"]
        sh, el, wr = L["sh"], L["el"], L["wr"]
        # the brachial plexus: three roots join at the shoulder, then radial, median, ulnar nerves
        plex = off(sh, -0.035 * sgn, -0.03, 0.0)
        for i in (8, 9, 10):
            sc.add(sd_capsule(off(J["spine"][i], 0.008 * sgn), plex, 0.003, 0.0035), m, root, blend=0.002)
        for dz, dx, seed in ((-0.012, 0.004, 1), (0.012, 0.0, 2), (0.0, -0.01, 3)):
            pts = [plex, off(sh, dx * sgn, -0.03, dz), off(el, dx * sgn, 0, dz), off(wr, dx * sgn, -0.005, dz * 0.5)]
            chain(sc, pts, [0.0035, 0.003, 0.0026, 0.002], m, root, blend=0.002)
            twigs(sc, pts[1:], fine, root, 5, seed * 7 + sgn, 0.028)
        for f in range(5):   # into the fingers
            tip = off(wr, sgn * 0.011 * (f - 2) + (0.012 * sgn if f == 0 else 0), -0.075 + 0.012 * abs(f - 2), 0.012 + 0.004 * f)
            sc.add(sd_capsule(off(wr, 0, -0.012, 0.008), tip, 0.0022, 0.0012), fine, root)
        # intercostal nerves, along the ribs
        for i in range(4, 10, 2):
            for pts in [rib_curve(J, i, sgn, 0.985)]:
                chain(sc, pts, [0.002] * len(pts), fine, root)
        # the femoral nerve in front of the thigh, the sciatic behind, splitting at the knee
        hp, kn, an, to = L["hp"], L["kn"], L["an"], L["to"]
        fem = [off(J["spine"][2], 0.012 * sgn, 0, 0.01), off(hp, -0.005 * sgn, 0.0, 0.035), off(lerp(hp, kn, 0.5), 0, 0, 0.035), off(kn, 0, 0.01, 0.025)]
        chain(sc, fem, [0.0035, 0.003, 0.0026, 0.002], m, root, blend=0.002)
        twigs(sc, fem[1:], fine, root, 6, 11 + sgn, 0.03)
        sci = [off(J["spine"][0], 0.01 * sgn, -0.02, -0.01), off(hp, 0, -0.03, -0.035), off(lerp(hp, kn, 0.6), 0, 0, -0.03), off(kn, 0, 0.02, -0.025)]
        chain(sc, sci, [0.005, 0.0045, 0.004, 0.0035], m, root, blend=0.002)
        tib = [sci[-1], off(lerp(kn, an, 0.5), 0, 0, -0.025), off(an, -0.01 * sgn, 0.0, -0.02), off(lerp(an, to, 0.6), 0, -0.008, 0.0)]
        per = [sci[-1], off(kn, 0.025 * sgn, -0.02, -0.005), off(lerp(kn, an, 0.5), 0.018 * sgn, 0, 0.02), off(lerp(an, to, 0.5), 0.01 * sgn, 0.008, 0.01)]
        for pts, seed in ((tib, 21), (per, 31)):
            chain(sc, pts, [0.003, 0.0026, 0.0022, 0.0018], m, root, blend=0.002)
            twigs(sc, pts, fine, root, 5, seed + sgn, 0.026)
        twigs(sc, sci[1:], fine, root, 4, 41 + sgn, 0.03)
        for f in range(5):
            sc.add(sd_capsule(off(lerp(an, to, 0.5), 0, -0.005, 0), off(to, sgn * 0.012 * (f - 2), -0.006, 0.02), 0.0016, 0.001), fine, root)


def vessels(sc, J, root, t, k=1.0):
    art, vein = glow("#ff3a3a", 3.0 * k), glow("#4a6aff", 2.6 * k)
    beat = 1 + 0.12 * max(0.0, math.sin(TAU * t * 2)) ** 4
    heart = off(J["chest"], 0.02, -0.01, 0.03)
    hm = Mat("#c8282a", rough=0.4, sss=0.6, emissive="#ff3a3a", emit=1.2)
    sc.add(sd_ellipsoid(0.03 * beat, 0.04 * beat, 0.03 * beat), hm, root * T(*heart) * RZ(0.4))
    sc.add(sd_capsule(off(heart, -0.005, 0.02, 0), off(heart, -0.012, 0.06, -0.01), 0.011), hm, root, blend=0.006)   # the aortic arch
    aorta = bezier(off(heart, -0.012, 0.06, -0.01), off(heart, -0.02, 0.07, -0.04), off(J["spine"][7], 0.008, 0, 0.022), off(J["spine"][0], 0.008, 0, 0.022), 6)
    chain(sc, aorta, [0.009, 0.008, 0.008, 0.007, 0.007, 0.006], art, root, blend=0.004)
    vena = bezier(off(heart, 0.012, 0.0, -0.01), off(J["spine"][6], -0.012, 0, 0.022), off(J["spine"][3], -0.012, 0, 0.022), off(J["spine"][0], -0.012, 0, 0.022), 6)
    chain(sc, vena, [0.009] * 6, vein, root, blend=0.004)
    for sgn in (-1, 1):
        chain(sc, [off(heart, -0.01, 0.06, -0.01), off(J["neck"], 0.015 * sgn, 0, 0.02), off(J["head"], 0.035 * sgn, -0.04, 0.03), off(J["head"], 0.04 * sgn, 0.03, 0.03)],
              [0.006, 0.005, 0.004, 0.003], art, root)
        chain(sc, [off(heart, 0.0, 0.03), off(J["neck"], 0.028 * sgn, 0, 0.0), off(J["head"], 0.048 * sgn, -0.03, -0.01), off(J["head"], 0.045 * sgn, 0.03, -0.02)],
              [0.006, 0.005, 0.004, 0.003], vein, root)
        L = J["r" if sgn > 0 else "l"]
        a_pts = [off(heart, -0.01, 0.06, -0.01), off(L["sh"], 0, -0.02, 0.012), off(L["el"], 0, 0, 0.013), off(L["wr"], 0, 0, 0.01)]
        v_pts = [off(heart, 0.0, 0.03), off(L["sh"], 0, -0.035, -0.01), off(L["el"], 0, 0, -0.013), off(L["wr"], 0, 0, -0.01)]
        chain(sc, a_pts, [0.006, 0.005, 0.004, 0.003], art, root, blend=0.002)
        chain(sc, v_pts, [0.006, 0.005, 0.004, 0.003], vein, root, blend=0.002)
        twigs(sc, a_pts[1:], art, root, 4, 51 + sgn, 0.025, 0.0014)
        twigs(sc, v_pts[1:], vein, root, 4, 61 + sgn, 0.025, 0.0014)
        la = [off(J["spine"][0], 0.008, 0, 0.022), off(L["hp"], 0, 0, 0.022), off(L["kn"], 0, 0, -0.018), off(L["an"], 0, 0, 0.012), off(L["to"], 0, 0.005, 0)]
        lv = [off(J["spine"][0], -0.012, 0, 0.022), off(L["hp"], 0.01 * sgn, 0, 0.012), off(L["kn"], 0.008 * sgn, 0, -0.022), off(L["an"], 0.006 * sgn, 0, -0.012), off(L["to"], 0.008 * sgn, 0.005, -0.01)]
        chain(sc, la, [0.008, 0.006, 0.005, 0.004, 0.003], art, root, blend=0.002)
        chain(sc, lv, [0.008, 0.006, 0.005, 0.004, 0.003], vein, root, blend=0.002)
        twigs(sc, la[1:], art, root, 5, 71 + sgn, 0.03, 0.0015)
        twigs(sc, lv[1:], vein, root, 5, 81 + sgn, 0.03, 0.0015)


BONE = Mat("#ece2cc", rough=0.45, sss=0.3, pattern=lambda pl, pw, n: {"albedo": srgb("#ece2cc") * (0.82 + 0.22 * fbm(pl * 50, 2))[:, None]})
CART = Mat("#cfdcd8", rough=0.35, sss=0.6)          # cartilage
HOLE = Mat("#2a2220", rough=0.8)


def rib_curve(J, i, sgn, k=1.0):
    """Rib i (spine index), side sgn: from the vertebra, out and back, round the side, to the front, falling."""
    s0 = J["spine"][i]
    u = (i - 3) / 7
    w = (0.055 + 0.055 * math.sin(math.pi * min(1.0, 0.25 + 0.8 * (1 - u)))) * k
    fy = s0[1] - 0.05 - 0.02 * (1 - u)
    return bezier(off(s0, 0.012 * sgn, 0, 0), off(s0, w * 0.85 * sgn, 0.005, -0.03 * k), off(s0, w * 1.15 * sgn, -0.03, 0.07 * k),
                  (0.035 * sgn * k, fy, 0.075 * k), 8)


def bone(sc, a, b, r, root, ends=1.5):
    sc.add(sd_capsule(a, b, r), BONE, root)
    for p in (a, b):
        sc.add(sd_sphere(r * ends), BONE, root * T(*p), blend=r * 0.8)


def bones(sc, J, root):
    hd = J["head"]
    # the skull: cranium, cheekbones, upper jaw, eye sockets, nose, teeth, mandible
    sc.add(sd_ellipsoid(0.055, 0.058, 0.065), BONE, root * T(*off(hd, 0, 0.014, -0.006)))
    sc.add(sd_ellipsoid(0.043, 0.03, 0.035), BONE, root * T(*off(hd, 0, -0.022, 0.03)), blend=0.012)
    for sgn in (-1, 1):
        sc.add(sd_sphere(0.012), BONE, root * T(*off(hd, 0.04 * sgn, -0.012, 0.04)), blend=0.008)
        sc.add(sd_sphere(0.017), BONE, root * T(*off(hd, 0.021 * sgn, 0.004, 0.058)), op="sub", blend=0.004)
        sc.add(sd_sphere(0.013), HOLE, root * T(*off(hd, 0.021 * sgn, 0.004, 0.05)))
        sc.add(sd_capsule(off(hd, 0.044 * sgn, -0.02, -0.012), off(hd, 0.036 * sgn, -0.062, 0.03), 0.008), BONE, root, blend=0.004)   # the ramus
    sc.add(sd_cone(0.012, 0.008, 0.002, 0.002), HOLE, root * T(*off(hd, 0, -0.022, 0.062)) * RX(math.pi))      # the nasal cavity
    jaw = bezier(off(hd, -0.036, -0.062, 0.03), off(hd, -0.02, -0.075, 0.06), off(hd, 0.02, -0.075, 0.06), off(hd, 0.036, -0.062, 0.03), 6)
    chain(sc, jaw, [0.008, 0.009, 0.01, 0.01, 0.009, 0.008], BONE, root, blend=0.005)
    for k in range(8):
        x = -0.021 + 0.006 * k
        for y in (-0.044, -0.054):
            sc.add(sd_box(0.0026, 0.0045, 0.003, 0.001), Mat("#f6f0e2", rough=0.3), root * T(*off(hd, x, y, 0.058 - 0.012 * (x / 0.021) ** 2)))
    # the spine: vertebrae with their processes, up through the neck
    neck = [lerp(J["spine"][-1], off(hd, 0, -0.05, -0.03), u) for u in (0.25, 0.5, 0.75)]
    for i, p in enumerate(J["spine"] + neck):
        r = 0.018 if i < 3 else (0.015 if i < 11 else 0.011)
        sc.add(sd_cylinder(0.008, r, 0.004), BONE, root * T(*p))
        sc.add(sd_capsule(off(p, 0, 0, -0.012), off(p, 0, -0.01, -0.032), 0.004), BONE, root, blend=0.003)
        sc.add(sd_capsule(off(p, -0.024, 0, -0.008), off(p, 0.024, 0, -0.008), 0.0035), BONE, root, blend=0.003)
    # the ribs, their cartilage to the sternum
    st_top, st_bot = off(J["chest"], 0, 0.085, 0.082), off(J["chest"], 0, -0.045, 0.088)
    sc.add(sd_box(0.012, 0.065, 0.004, 0.003), BONE, root * T(*lerp(st_top, st_bot, 0.5)) * RX(0.08))
    for i in range(3, 11):
        for sgn in (-1, 1):
            pts = rib_curve(J, i, sgn)
            chain(sc, pts, [0.0058] * len(pts), BONE, root, blend=0.002)
            end = pts[-1]
            if i >= 5:
                to = (0.012 * sgn, min(end[1] + 0.02, st_top[1] - 0.005 * (10 - i)), st_top[2])
                to = (to[0], max(to[1], st_bot[1]), to[2])
                sc.add(sd_capsule(end, to, 0.005), CART, root, blend=0.002)
    for sgn in (-1, 1):
        L = J["r" if sgn > 0 else "l"]
        sh, el, wr = L["sh"], L["el"], L["wr"]
        # the clavicle (an S), the scapula (flat, on the back)
        chain(sc, bezier(off(st_top, 0.01 * sgn), off(st_top, 0.05 * sgn, 0.012, -0.0), off(sh, -0.04 * sgn, 0.012, 0.025), off(sh, 0, 0.01, 0.0), 5),
              [0.006] * 5, BONE, root, blend=0.003)
        sc.add(sd_thin(0.04, 0.055, 0.006), BONE, root * T(*off(sh, -0.05 * sgn, -0.05, -0.05)) * RY(-0.5 * sgn) * RZ(0.2 * sgn))
        # the arm: humerus, radius and ulna, carpus, five fingers
        sc.add(sd_sphere(0.016), BONE, root * T(*sh))
        bone(sc, sh, el, 0.009, root, 1.4)
        bone(sc, off(el, 0, 0, 0.006), off(wr, 0, 0, 0.006), 0.0055, root, 1.3)
        bone(sc, off(el, 0, 0, -0.006), off(wr, 0, 0, -0.006), 0.0055, root, 1.3)
        sc.add(sd_ellipsoid(0.012, 0.009, 0.007), BONE, root * T(*off(wr, 0, -0.012, 0.004)))
        for f in range(5):
            th = f == 0
            k0 = off(wr, sgn * (0.012 if th else 0.008 * (f - 2.5)), -0.018, 0.008 + 0.004 * f)
            dx = sgn * (0.018 if th else 0.004 * (f - 2.5))
            dy = -0.03 if th else -0.032 + 0.004 * abs(f - 2.5)
            pts = [k0, off(k0, dx * 0.6, dy * 0.8, 0.006), off(k0, dx, dy * 1.4, 0.01), off(k0, dx * 1.2, dy * 1.8, 0.014)]
            chain(sc, pts, [0.0034, 0.003, 0.0026, 0.002], BONE, root, blend=0.001)
        # the pelvis: iliac wings, the sacrum, the pubis with its holes
        P = J["pelvis"]
        sc.add(sd_thin(0.052, 0.045, 0.008), BONE, root * T(*off(P, 0.062 * sgn, 0.035, -0.006)) * RY(0.55 * sgn) * RZ(0.35 * sgn))
        sc.add(sd_torus(0.018, 0.006), BONE, root * T(*off(P, 0.03 * sgn, -0.03, 0.022)) * RX(math.pi / 2 - 0.3) * RZ(0.3 * sgn))
        sc.add(sd_capsule(off(P, 0.06 * sgn, 0.0, 0.0), off(P, 0.012 * sgn, -0.035, 0.035), 0.008), BONE, root, blend=0.004)
        # the leg: femur (head, neck, shaft, condyles), patella, tibia and fibula, the foot
        hp, kn, an, to = L["hp"], L["kn"], L["an"], L["to"]
        tro = off(hp, 0.028 * sgn, -0.01, 0)
        sc.add(sd_sphere(0.017), BONE, root * T(*off(hp, -0.008 * sgn, 0.008, 0)))
        sc.add(sd_capsule(off(hp, -0.008 * sgn, 0.008, 0), tro, 0.009), BONE, root, blend=0.004)
        sc.add(sd_sphere(0.014), BONE, root * T(*tro), blend=0.004)
        bone(sc, tro, off(kn, 0, 0.012, 0), 0.0095, root, 1.0)
        for d in (-1, 1):
            sc.add(sd_sphere(0.012), BONE, root * T(*off(kn, 0.01 * d, 0.012, -0.002)), blend=0.005)
        sc.add(sd_ellipsoid(0.011, 0.014, 0.006), BONE, root * T(*off(kn, 0, 0.008, 0.02)))
        bone(sc, off(kn, -0.003 * sgn, -0.008, 0), off(an, -0.003 * sgn, 0.008, 0.0), 0.009, root, 1.4)
        bone(sc, off(kn, 0.016 * sgn, -0.014, -0.006), off(an, 0.014 * sgn, 0.0, -0.004), 0.0045, root, 1.4)
        sc.add(sd_sphere(0.013), BONE, root * T(*off(an, 0, -0.008, -0.022)))                   # the heel
        for f in range(5):
            b0 = off(an, sgn * 0.006 * (f - 2), -0.012, 0.01)
            tip = off(to, sgn * 0.009 * (f - 2) - sgn * 0.004, -0.005, 0.03 - 0.004 * abs(f - 1))
            chain(sc, [b0, lerp(b0, tip, 0.65), tip], [0.004, 0.0034, 0.0028], BONE, root, blend=0.001)


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


def muscle_mat(c="#a8302a"):
    def pat(pl, pw, n):
        fib = (0.7 + 0.3 * torch.sin(pl[:, 0] * 420 + fbm(pl * 14, 2) * 6))[:, None]   # fibres along the local y (each muscle's axis)
        return {"albedo": srgb(c) * fib * (0.85 + 0.25 * fbm(pl * 30, 2))[:, None]}
    return Mat(c, rough=0.4, sss=0.5, clearcoat=0.5, pattern=pat)


TENDON = Mat("#e8e0d4", rough=0.3, sss=0.4, clearcoat=0.4,
             pattern=lambda pl, pw, n: {"albedo": srgb("#e8e0d4") * (0.85 + 0.15 * torch.sin(pl[:, 0] * 500))[:, None]})


def muscles(sc, J, root):
    """The écorché: the muscles one by one, each with its fibres along its axis, white tendons."""
    M, D = muscle_mat(), muscle_mat("#7a1e1c")

    def mu(a, b, rx, rz=None, m=None, blend=0.004, ry=None):
        f, n = along(a, b)
        sc.add(sd_thin(rx, ry or n / 2, rz or rx), m or M, root * f * T(0, n / 2, 0), blend=blend)

    def tendon(a, b, r):
        f, n = along(a, b)
        sc.add(sd_capsule((0, 0, 0), (0, n, 0), r, r * 0.8), TENDON, root * f, blend=0.003)
    hd, ch, P = J["head"], J["chest"], J["pelvis"]
    # a dark core so that no gap shows through
    sc.add(sd_ellipsoid(0.11, 0.1, 0.062), D, root * T(*off(ch, 0, 0.01, 0)))
    sc.add(sd_ellipsoid(0.085, 0.13, 0.055), D, root * T(*off(P, 0, 0.09, 0)), blend=0.02)
    sc.add(sd_ellipsoid(0.1, 0.07, 0.06), D, root * T(*off(P, 0, 0.0, 0)), blend=0.02)
    # the head: cranium under the temporal muscles, the eyes, masseters, the teeth
    sc.add(sd_ellipsoid(0.056, 0.06, 0.066), D, root * T(*off(hd, 0, 0.012, -0.004)))
    sc.add(sd_ellipsoid(0.04, 0.035, 0.04), D, root * T(*off(hd, 0, -0.035, 0.028)), blend=0.012)
    for sgn in (-1, 1):
        mu(off(hd, 0.046 * sgn, 0.035, -0.01), off(hd, 0.044 * sgn, -0.012, 0.015), 0.008, 0.024)
        mu(off(hd, 0.042 * sgn, -0.02, 0.02), off(hd, 0.036 * sgn, -0.065, 0.035), 0.01, 0.014)
        sc.add(sd_sphere(0.0105), Mat("#f2eee8", rough=0.2, clearcoat=0.8), root * T(*off(hd, 0.021 * sgn, 0.004, 0.056)))
        sc.add(sd_sphere(0.005), Mat("#3a4a5a", rough=0.2), root * T(*off(hd, 0.021 * sgn, 0.004, 0.0655)))
    sc.add(sd_box(0.022, 0.006, 0.004, 0.002), Mat("#f6f0e2", rough=0.3), root * T(*off(hd, 0, -0.05, 0.062)))
    # the neck: sternocleidomastoids; the trapezius
    for sgn in (-1, 1):
        mu(off(hd, 0.038 * sgn, -0.035, -0.01), off(ch, 0.012 * sgn, 0.09, 0.06), 0.011)
        mu(off(J["neck"], 0.01 * sgn, 0.03, -0.03), off(J[("r" if sgn > 0 else "l")]["sh"], -0.012 * sgn, 0.0, -0.015), 0.016, 0.01)
    # the torso: pectorals, the abdominal blocks, obliques, serratus, latissimus
    for sgn in (-1, 1):
        L = J["r" if sgn > 0 else "l"]
        mu(off(ch, 0.012 * sgn, 0.05, 0.075), off(L["sh"], -0.01 * sgn, -0.02, 0.025), 0.03, 0.012)
        mu(off(ch, 0.012 * sgn, 0.0, 0.075), off(L["sh"], -0.012 * sgn, -0.035, 0.02), 0.03, 0.012)
        for j in range(4):
            sc.add(sd_ellipsoid(0.02, 0.02 if j else 0.03, 0.01), M, root * T(*off(P, 0.023 * sgn, 0.17 - 0.042 * j - (0.01 if j == 3 else 0), 0.06 - 0.004 * j)), blend=0.002)
        mu(off(ch, 0.1 * sgn, -0.05, 0.035), off(P, 0.075 * sgn, 0.03, 0.035), 0.025, 0.022)
        for j in range(3):
            mu(off(ch, 0.1 * sgn, 0.02 - 0.03 * j, 0.045), off(ch, 0.075 * sgn, 0.0 - 0.03 * j, 0.07), 0.009, 0.006)
        mu(off(ch, 0.05 * sgn, -0.1, -0.045), off(L["sh"], -0.02 * sgn, -0.05, -0.025), 0.04, 0.014)
    sc.add(sd_box(0.003, 0.09, 0.004, 0.002), TENDON, root * T(*off(P, 0, 0.105, 0.064)))       # the linea alba
    for sgn in (-1, 1):
        L = J["r" if sgn > 0 else "l"]
        sh, el, wr, hp, kn, an, to = L["sh"], L["el"], L["wr"], L["hp"], L["kn"], L["an"], L["to"]
        # the arm: deltoid, biceps, triceps, forearm flexors and extensors with their tendons, the hand
        mu(off(sh, 0, 0.02, 0), lerp(sh, el, 0.5), 0.032, 0.036)
        mu(off(lerp(sh, el, 0.2), 0, 0, 0.014), off(lerp(sh, el, 0.88), 0, 0, 0.012), 0.019, 0.02)
        mu(off(lerp(sh, el, 0.15), 0, 0, -0.014), off(lerp(sh, el, 0.95), 0, 0, -0.012), 0.02, 0.02)
        mu(off(el, 0.006 * sgn, 0.0, 0.008), off(lerp(el, wr, 0.65), 0, 0, 0.008), 0.017, 0.016)
        mu(off(el, -0.004 * sgn, 0.0, -0.008), off(lerp(el, wr, 0.65), 0, 0, -0.008), 0.016, 0.015)
        tendon(off(lerp(el, wr, 0.6), 0, 0, 0.008), off(wr, 0, 0, 0.006), 0.006)
        tendon(off(lerp(el, wr, 0.6), 0, 0, -0.008), off(wr, 0, 0, -0.006), 0.006)
        sc.add(sd_ellipsoid(0.02, 0.026, 0.01), M, root * T(*off(wr, 0, -0.026, 0.006)), blend=0.004)
        for f in range(5):
            th = f == 0
            k0 = off(wr, sgn * (0.012 if th else 0.008 * (f - 2.5)), -0.04, 0.008 + 0.004 * f)
            tendon(k0, off(k0, sgn * (0.012 if th else 0.003 * (f - 2.5)), -0.03, 0.01), 0.0035)
        # the hip and the thigh: gluteus, quadriceps (three heads), sartorius, adductors, hamstrings
        mu(off(hp, 0.0, 0.02, -0.04), off(hp, 0.01 * sgn, -0.07, -0.035), 0.05, 0.035)
        mu(off(hp, 0.03 * sgn, -0.01, 0.005), off(kn, 0.016 * sgn, 0.03, 0.0), 0.028, 0.03)
        mu(off(hp, 0.0, -0.015, 0.03), off(kn, 0.0, 0.035, 0.02), 0.024, 0.024)
        mu(off(lerp(hp, kn, 0.45), -0.02 * sgn, 0, 0.012), off(kn, -0.016 * sgn, 0.03, 0.012), 0.022, 0.022)
        mu(off(hp, 0.035 * sgn, 0.03, 0.035), off(kn, -0.025 * sgn, 0.0, 0.0), 0.008, 0.007, blend=0.002)
        mu(off(hp, -0.03 * sgn, -0.02, 0.0), off(lerp(hp, kn, 0.7), -0.02 * sgn, 0, 0.0), 0.024, 0.026)
        mu(off(hp, 0.0, -0.05, -0.03), off(kn, 0.0, 0.03, -0.022), 0.026, 0.022)
        sc.add(sd_ellipsoid(0.012, 0.014, 0.007), TENDON, root * T(*off(kn, 0, 0.008, 0.024)))
        tendon(off(kn, 0, 0.0, 0.026), off(kn, 0, -0.035, 0.02), 0.006)
        # the leg: gastrocnemius (two heads), soleus, tibialis anterior, the Achilles tendon, the foot
        for d in (-1, 1):
            mu(off(kn, 0.012 * d, -0.01, -0.018), off(lerp(kn, an, 0.55), 0.008 * d, 0, -0.02), 0.017, 0.02)
        mu(off(lerp(kn, an, 0.2), 0, 0, -0.012), off(lerp(kn, an, 0.8), 0, 0, -0.012), 0.022, 0.016)
        mu(off(kn, 0.012 * sgn, -0.03, 0.012), off(lerp(kn, an, 0.75), 0.006 * sgn, 0, 0.015), 0.012, 0.012)
        tendon(off(lerp(kn, an, 0.6), 0, 0, -0.02), off(an, 0, -0.01, -0.022), 0.006)
        sc.add(sd_capsule(off(kn, 0, -0.01, 0), off(an, 0, 0.01, 0), 0.016, 0.012), D, root, blend=0.004)
        mu(off(an, 0, -0.01, -0.02), off(to, 0, -0.002, 0.015), 0.022, 0.016, ry=None)
        for f in range(5):
            tendon(off(lerp(an, to, 0.3), sgn * 0.004 * (f - 2), 0.008, 0.01), off(to, sgn * 0.009 * (f - 2), 0.0, 0.03), 0.0028)


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


def fascia_mat():
    """The deep fascia: a thin silvery-white sheath of collagen over the
    muscles, which show through it, pink, with the grain of their fibres."""
    def pat(pl, pw, n):
        fib = (0.72 + 0.28 * torch.sin((pl[:, 0] + pl[:, 2]) * 300))[:, None]
        mus = srgb("#b8463e") * fib
        sheen = (0.2 + 0.15 * fbm(pl * 30, 2))[:, None]
        return {"albedo": mus * (1 - sheen) + srgb("#eef0f4") * sheen}
    return Mat("#d8b8b8", rough=0.2, sss=0.4, clearcoat=0.9, pattern=pat)


def fat_mat(cover=0.5):
    """The subcutaneous fat: pale yellow lobules (cells in clusters, small
    vessels between them), over the muscles where it does not cover yet."""
    def pat(pl, pw, n):
        lob = fbm(pl * 70, 2)
        a = srgb("#e8c46a") * (0.8 + 0.3 * lob)[:, None]
        a = a * (1 - 0.35 * (lob < 0.32).float()[:, None]) + srgb("#c86a4a") * 0.35 * (lob < 0.32).float()[:, None]
        m = (fbm(pl * 9, 3) < cover).float()[:, None]
        mus = srgb("#a8302a") * (0.72 + 0.28 * torch.sin((pl[:, 0] + pl[:, 2]) * 300))[:, None]
        return {"albedo": a * m + mus * (1 - m)}
    return Mat("#e8c46a", rough=0.35, sss=0.6, clearcoat=0.4, pattern=pat)


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
        J, root = joints(anim, t, lift, **({"arms_up": False, "spread": 0.55} if level == 10 else {}))
        if level <= 7:
            scanner(sc, t)
        if level <= 4:
            nerves(sc, J, root, 3.0 if level == 1 else (2.0 if level == 2 else 1.0))
        if 2 <= level <= 4:
            vessels(sc, J, root, t, 1.0 if level == 2 else 0.45)
        if 3 <= level <= 4:
            bones(sc, J, root)
        if level == 4:
            organs(sc, J, root)
        if level in (5, 5.3, 5.6):
            muscles(sc, J, root)
        if level == 5.3:     # the deep fascia: a thin glistening sheath over every muscle
            body_volume(sc, J, root, fascia_mat(), blend=0.02, face=Mat("#3a2a24"))
        if level == 5.6:     # the hypodermis: yellow lobules of fat spreading over the muscles
            body_volume(sc, J, root, fat_mat(cover=0.55 + 0.08 * math.sin(TAU * t)), blend=0.02, face=Mat("#3a2a24"))
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


SCALE = {k: 1.3 for k in ("random", "beginner", "monte_carlo", "mcts", "basic_minmax", "minmax", "mcts_nnue", "nnue193", "features", "resnet", "dev", "alphazero")}
CAST = {"random": ("Système nerveux", stage(1)), "beginner": ("Circulation", stage(2)), "monte_carlo": ("Squelette", stage(3)),
        "mcts": ("Organes", stage(4)), "basic_minmax": ("Écorché", stage(5)), "minmax": ("Fascias", stage(5.3)), "mcts_nnue": ("Hypoderme", stage(5.6)),
        "nnue193": ("Peau", stage(6)),
        "features": ("Corps achevé", stage(7)), "resnet": ("Éveil", stage(8)), "dev": ("Lévitation", stage(9)),
        "alphazero": ("Être de lumière", stage(10))}
ANATOMY = {"lab_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"lab_" + k: n for k, (n, _) in CAST.items()}
