"""
heroic.py — an adult rig (heroic proportions: the head about a sixth of the
height, long legs, a muscled torso, jointed arms) and the creatures that do
not stand on two legs (a dragon with membrane wings), shared by the Légendes
set (epic.py) and the Manga set (shonen.py).

The rig takes legends.Pose (bob, lean, shake, yaw, pitch, step, jump; arms
l, r = [shoulder pitch, spread, elbow]) and returns the frames of the head,
the hands (−y along the forearm: legends.held gives a weapon frame), the
torso, the chest and the back.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, sd_plane_cut, fbm
from models import base, BASE_TOP, TAU, wave, glow
from legends import Pose, base_pose, held

HEAD = math.radians(28)          # the figure turns a little towards the camera


def adult(sc, p, m, H=1.0, bulk=1.0, head=1.0, hunch=0.0, arm=1.0, leg=1.0, neck=1.0):
    """Adds legs, pelvis, waist, chest, shoulders, arms and neck. m: materials
    {"skin", "top", "legs", "boots"} and optionally "belly", "arms", "fore",
    "hands", "neck". Returns the frames and the head radius."""
    g = lambda k, d: m.get(k, m[d])
    y0 = BASE_TOP
    body = RY(HEAD + p.shake) * T(0, p.bob + p.jump, 0)
    thigh, shin = 0.235 * H * leg, 0.215 * H * leg
    hip_y = y0 + thigh + shin + 0.03 * H
    for sgn in (-1, 1):
        s_ = p.step * sgn * 0.6
        hip = body * T(0.062 * bulk * sgn, hip_y, 0)
        th = hip * RX(-0.6 * s_)
        sc.add(sd_capsule((0, 0, 0), (0, -thigh, 0.01), 0.05 * bulk, 0.036 * bulk), m["legs"], th)
        kn = th * T(0, -thigh, 0.01) * RX(0.6 * max(0.0, s_))
        sc.add(sd_capsule((0, 0, 0), (0, -shin, -0.005), 0.036 * bulk, 0.026 * bulk), m["boots"], kn)
        sc.add(sd_ellipsoid(0.032 * bulk, 0.05 * H, 0.03 * bulk), m["boots"], kn * T(0, -shin * 0.3, -0.012), blend=0.02)   # calf
        sc.add(sd_box(0.03 * bulk, 0.02, 0.062 * bulk, 0.016), m["boots"], kn * T(0, -shin - 0.008, 0.028))
    torso = body * T(0, hip_y, 0) * RX(p.lean + hunch)
    sc.add(sd_ellipsoid(0.105 * bulk, 0.075 * H, 0.075 * bulk), m["legs"], torso * T(0, 0.02 * H, 0))
    sc.add(sd_ellipsoid(0.088 * bulk, 0.09 * H, 0.064 * bulk), g("belly", "top"), torso * T(0, 0.12 * H, 0.004), blend=0.03)
    sc.add(sd_ellipsoid(0.135 * bulk, 0.1 * H, 0.082 * bulk), m["top"], torso * T(0, 0.235 * H, 0.006), blend=0.035)
    sh_y = 0.3 * H
    hands = []
    for sgn, (pitch, spread, elbow) in ((-1, p.l), (1, p.r)):
        sc.add(sd_sphere(0.052 * bulk), m["top"], torso * T(0.135 * bulk * sgn, sh_y - 0.01, 0), blend=0.02)   # deltoid
        up, fo = 0.17 * H * arm, 0.155 * H * arm
        sh = torso * T(0.15 * bulk * sgn, sh_y - 0.01, 0) * RZ(spread * sgn) * RX(-pitch)
        sc.add(sd_capsule((0, 0, 0), (0, -up, 0), 0.04 * bulk, 0.031 * bulk), g("arms", "skin"), sh)
        sc.add(sd_ellipsoid(0.036 * bulk, 0.06 * H * arm, 0.036 * bulk), g("arms", "skin"), sh * T(0, -up * 0.45, 0.012), blend=0.015)
        el = sh * T(0, -up, 0) * RX(-elbow)
        sc.add(sd_capsule((0, 0, 0), (0, -fo, 0), 0.033 * bulk, 0.023 * bulk), g("fore", "arms") if "fore" in m or "arms" in m else m["skin"], el)
        hand = el * T(0, -fo - 0.022 * bulk, 0)
        sc.add(sd_ellipsoid(0.027 * bulk, 0.034 * bulk, 0.023 * bulk), g("hands", "skin"), hand)
        hands.append(hand)
    nk = 0.06 * H * neck
    hr = 0.066 * H * head
    # the head above the chest (whose top is at 0.335 H), pushed forward and down by the hunch
    hy = max(sh_y + nk + hr * 0.95, 0.335 * H + hr * 0.75) - hunch * 0.12 * H
    hz = 0.016 + hunch * (0.082 * bulk + hr * 0.4) * 1.3
    sc.add(sd_capsule((0, sh_y - 0.03, 0), (0, hy - hr * 0.7, hz * 0.8), 0.034 * bulk * (0.8 + 0.2 * neck), 0.03 * bulk), g("neck", "skin"), torso)
    hx = torso * T(0, hy, hz) * RX(-hunch * 0.9) * RY(p.yaw) * RX(p.pitch - 0.1)
    return {"head": hx, "hr": hr, "l": hands[0], "r": hands[1], "torso": torso, "body": body,
            "chest": torso * T(0, 0.235 * H, 0.085 * bulk), "back": torso * T(0, 0.24 * H, -0.085 * bulk),
            "waist": torso * T(0, 0.1 * H, 0), "shoulder": (torso * T(-0.15 * bulk, sh_y, 0), torso * T(0.15 * bulk, sh_y, 0)),
            "hip_y": hip_y, "H": H, "bulk": bulk}


def chain(sc, pts, radii, mat, xf, blend=0.02):
    for (a, b), (ra, rb) in zip(zip(pts[:-1], pts[1:]), zip(radii[:-1], radii[1:])):
        sc.add(sd_capsule(a, b, ra, rb), mat, xf, blend=blend)


def bezier(p0, p1, p2, p3, n):
    out = []
    for i in range(n):
        u = i / (n - 1)
        a, b, c, d = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
        out.append(tuple(a * p0[k] + b * p1[k] + c * p2[k] + d * p3[k] for k in range(3)))
    return out


def sd_fan(R, th0, th1, n, depth, thick):
    """A membrane in the local xy plane: the sector th0..th1 of radius R, its
    edge scalloped between n fingers; thickness 2·thick along z."""
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = torch.sqrt(x * x + y * y) + 1e-6
        th = torch.atan2(y, x)
        u = ((th - th0) / (th1 - th0)).clamp(0, 1)
        Rt = R * (1 - depth * torch.sin(math.pi * u * n).abs())
        d2 = torch.maximum(r - Rt, torch.maximum(th0 - th, th - th1) * r)
        return torch.maximum(d2, z.abs() - thick)
    return f


def membrane_wing(sc, xf, R, th0, th1, n, bone, skin_m, thick=0.006, depth=0.28):
    """A bat/dragon wing in frame xf (x outwards): an arm bone, n finger bones, the membrane."""
    sc.add(sd_fan(R * 0.97, th0, th1, n, depth, thick), skin_m, xf)
    for k in range(n + 1):
        a = th0 + (th1 - th0) * k / n
        ln = R * (1.02 if k == n else 0.98)
        sc.add(sd_capsule((0, 0, 0), (ln * math.cos(a), ln * math.sin(a), 0), 0.022 if k == n else 0.012, 0.004), bone, xf)
    sc.add(sd_cone(0.04, 0.012, 0.001, 0.001), bone, xf * T(R * math.cos(th1), R * math.sin(th1), 0) * RZ(th1 - math.pi / 2) * T(0, 0.03, 0))


def scales(c, belly=None, k=90.0, rough=0.45, metal=0.15):
    """Overlapping scales: rows of rounded cells, darker in the gaps."""
    base_c = srgb(c)

    def pat(pl, pw, n):
        u, v = pl[:, 0] * k, pl[:, 2] * k + pl[:, 1] * k * 0.6
        row = torch.floor(v)
        cu = u + 0.5 * (row % 2)
        fu, fv = cu - torch.floor(cu) - 0.5, v - row - 0.5
        cell = (fu * fu * 2.5 + (fv + 0.2) ** 2).clamp(0, 1)
        shade = 1.15 - 0.75 * cell
        return {"albedo": base_c * (shade * (0.85 + 0.3 * fbm(pl * 12, 2)))[:, None], "rough": rough + 0.3 * cell}
    return Mat(c, metal=metal, rough=rough, pattern=pat)
