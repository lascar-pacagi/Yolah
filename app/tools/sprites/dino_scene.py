"""
dino_scene.py — the Dinosaures' background in 3D (scenery.py): a valley of
the Cretaceous at dusk.

  the volcano   a broad cone at the back, lava running down its flank and
                glowing at the crater, a plume of ash leaning in the wind
  the valley    a lake reflecting the sky, banks of mud and stones, rolling
                hills covered with forest
  the plants    tree ferns and cycads (trunks, crowns of fronds), tall
                araucarias, ferns on the ground
  the light     a low orange sun through the haze, light shafts, the lava's
                red glow on the smoke

    cd app/tools/sprites && python3 dino_scene.py [W H out]
"""
import math
import os
import random
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_capsule, sd_cone, sd_ellipsoid, sd_box, fbm
from scenery import Camera, render_view, gradient_sky

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


def terrain(p):
    x, z = p[:, 0], p[:, 2]
    q = torch.stack([x * 0.012, x * 0, z * 0.012], 1)
    h = 26 * (fbm(q, 5) - 0.35) + 2.5 * fbm(p * 0.15, 3)
    # the valley floor flattens towards the lake in the middle
    lake = torch.exp(-((x / 60) ** 2 + ((z + 70) / 45) ** 2))
    h = h * (1 - 0.9 * lake) - 3.0 * lake
    return (p[:, 1] - h) * 0.55


def ground_mat():
    def pat(pl, pw, n):
        slope = (1 - n[:, 1]).clamp(0, 1)
        moss = srgb("#3a5a22") * (0.6 + 0.5 * fbm(pw * 0.6, 3))[:, None]
        rock = srgb("#5a4a3a") * (0.6 + 0.5 * fbm(pw * 1.3, 3))[:, None]
        mud = srgb("#3a2e22")
        wet = ((pw[:, 1] + 2.6) / 1.2).clamp(0, 1)[:, None]
        a = moss * (1 - slope[:, None] * 2).clamp(0, 1) + rock * (slope[:, None] * 2).clamp(0, 1)
        return {"albedo": mud * (1 - wet) + a * wet}
    return Mat("#3a5a22", rough=0.9, pattern=pat)


def volcano(p):
    r = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
    az = torch.atan2(p[:, 0], p[:, 2])
    gully = (torch.sin(az * 23 + 3 * fbm(p * 0.01, 2)).abs() - 0.5) * 4 * (r / 200).clamp(0, 1)
    h = 260 - r * 0.62 + 14 * fbm(p * 0.02, 4) + gully
    crater = torch.exp(-(r / 30) ** 2) * 40
    return (p[:, 1] - (h - crater).clamp(max=230)) * 0.45


def volcano_mat():
    def pat(pl, pw, n):
        x, y, z = pl[:, 0], pl[:, 1], pl[:, 2]
        a = torch.lerp(srgb("#3a322c"), srgb("#6a5a4a"), fbm(pl * 0.04, 4)[:, None]) * (0.7 + 0.5 * fbm(pl * 0.3, 3))[:, None]
        # lava streams: narrow channels down the slope (by the azimuth), brighter near the top
        az = torch.atan2(x, z)
        ch = (torch.sin(az * 5 + 4 * fbm(pl * 0.01, 3)) - 0.985).clamp_min(0) / 0.015
        ch = ch * (az.abs() < 1.0).float()
        heat = (y / 230).clamp(0, 1)
        top = torch.exp(-((torch.sqrt(x * x + z * z) - 22) / 10) ** 2) * (y > 200).float()
        em = srgb("#ff4a10") * (ch * (0.6 + 2.5 * heat) + top * 6.0)[:, None] * (0.6 + 0.8 * fbm(pl * 0.08, 2))[:, None]
        return {"albedo": a * (1 - ch)[:, None], "emissive": em}
    return Mat("#2a2420", rough=0.85, pattern=pat)


def plume_mat():
    return Mat("#4a3a34", rough=1.0, sss=0.9, pattern=lambda pl, pw, n: {
        "albedo": srgb("#3e3430") * (0.6 + 0.6 * fbm(pw * 0.02, 3))[:, None],
        "emissive": srgb("#ff5a1a") * (0.6 * ((260 - pw[:, 1]).clamp(0, 120) / 120) ** 2)[:, None]})


def lake_mat():
    return Mat("#0a1418", rough=0.03, clearcoat=1.0, pattern=lambda pl, pw, n: {"albedo": srgb("#0a1418").expand(pw.shape[0], 3)})


def bark(c="#4a3a2a"):
    return Mat(c, rough=0.9, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.6 + 0.6 * (0.5 + 0.5 * torch.sin(pw[:, 1] * 9 + 3 * fbm(pw * 2, 2))))[:, None]})


def frond_mat(c="#3e6a26"):
    return Mat(c, rough=0.6, sss=0.7, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.6 + 0.6 * fbm(pw * 3, 2))[:, None]})


def tree_fern(sc, x, y, z, h, rnd, fm, bm):
    lean = rnd.uniform(-0.15, 0.15)
    top = (x + lean * h, y + h, z)
    k = h / 8
    sc.add(lambda q: sd_capsule((x, y - 1, z), top, 0.32 * k, 0.24 * k)(q) - 0.05 * k * fbm(q * 6, 2), bm)
    n = 13
    for i in range(n):
        a = math.tau * i / n + rnd.uniform(-0.25, 0.25)
        L = h * rnd.uniform(0.5, 0.65)
        up = rnd.uniform(0.15, 0.4)
        ca, sa = math.cos(a), math.sin(a)
        # an arching frond: up, out, then drooping; the leaflets as a flat, notched blade
        pts = [top]
        for s_ in (0.33, 0.66, 1.0):
            pts.append((top[0] + ca * L * s_, top[1] + L * (up * s_ * 2.2 - 0.9 * s_ * s_), top[2] + sa * L * s_))
        for j in range(3):
            w0, w1 = (0.55, 0.45, 0.3)[j] * k, (0.45, 0.3, 0.06)[j] * k
            sc.add(lambda q, a_=pts[j], b_=pts[j + 1], w0=w0, w1=w1: torch.maximum(
                       sd_capsule(a_, b_, w0, w1)(q) - 0.25 * k * (torch.sin(q[:, 0] * 9 / k + q[:, 2] * 9 / k) > 0.3).float() * 0,
                       (q[:, 1] - (a_[1] + b_[1]) / 2 - (q[:, 0] - (a_[0] + b_[0]) / 2) * 0).abs() - 0.6 * k) - 0.06 * k * fbm(q * (5 / k), 2),
                   fm, blend=0.08 * k)


def araucaria(sc, x, y, z, h, rnd, fm, bm):
    sc.add(sd_capsule((x, y - 1, z), (x, y + h, z), 0.012 * h, 0.006 * h), bm)
    # whorls of horizontal branches near the top, each ending in a dense tuft: the umbrella crown
    for k in range(5):
        yy = y + h * (0.62 + 0.08 * k)
        r = h * (0.27 - 0.035 * k)
        m = 7 - k
        for i in range(m):
            a = math.tau * i / m + rnd.uniform(-0.3, 0.3) + k
            tip = (x + math.cos(a) * r, yy + rnd.uniform(-0.03, 0.04) * h, z + math.sin(a) * r)
            sc.add(sd_capsule((x, yy, z), tip, 0.006 * h), bm)
            sc.add(lambda q, s_=0.07 * h: sd_ellipsoid(s_ * 1.5, s_ * 0.32, s_ * 1.5)(q) - 0.2 * s_ * fbm(q * (4 / s_), 2), fm,
                   T(*tip), blend=0.02 * h)


def forest(p):
    """The forest on the hills as a canopy: a lumpy layer above the terrain (distant trees)."""
    x, z = p[:, 0], p[:, 2]
    q = torch.stack([x * 0.012, x * 0, z * 0.012], 1)
    h = 26 * (fbm(q, 5) - 0.35) + 2.5 * fbm(p * 0.15, 3)
    lake = torch.exp(-((x / 60) ** 2 + ((z + 70) / 45) ** 2))
    h = h * (1 - 0.9 * lake) - 3.0 * lake
    canopy = h + 9 * (fbm(p * 0.09, 3) - 0.2) * (1 - lake * 1.4).clamp(0, 1) * ((-z - 60) / 60).clamp(0, 1) * (fbm(q * 3, 2) > 0.45).float()
    return (p[:, 1] - canopy) * 0.5


def scene():
    sc = Scene()
    sc.add(terrain, ground_mat())
    sc.add(forest, frond_mat("#2e4a1e"))
    sc.add(sd_box(80, 0.1, 60), lake_mat(), T(0, -2.7, -70))
    sc.add(volcano, volcano_mat(), T(40, -10, -620))
    pm = plume_mat()
    for k in range(9):          # the ash plume, leaning left
        sc.add(lambda q, k=k: sd_ellipsoid(30 + 12 * k, 22 + 8 * k, 26 + 10 * k)(q) - (8 + 3 * k) * fbm(q * 0.03 + k, 3), pm,
               T(40 - 26 * k, 245 + 34 * k, -620 - 10 * k), blend=10)
    rnd = random.Random(11)
    fm, bm = frond_mat(), bark()
    spots = [(-22, -18, 9), (-30, -26, 11), (24, -20, 8), (31, -30, 12), (-14, -40, 7), (18, -44, 8), (-44, -60, 13), (46, -58, 12)]
    spots += [(rnd.uniform(-90, 90), rnd.uniform(-160, -30), rnd.uniform(7, 13)) for _ in range(14)]
    spots = [s_ for s_ in spots if abs(s_[0]) > 35 or s_[1] > -30 or s_[1] < -120]
    for (x, z, h) in spots:
        tree_fern(sc, x, terrain_h(x, z), z, h, rnd, fm, bm)
    for (x, z, h) in ((-60, -110, 34), (-48, -130, 30), (70, -120, 38), (58, -150, 32)):
        araucaria(sc, x, terrain_h(x, z), z, h, rnd, frond_mat("#2a4a20"), bark("#3a2a1e"))
    return sc


def terrain_h(x, z):
    p = vec(x, 0, z)[None]
    return float(-terrain(p)[0] / 0.55)


def render(W=2560, H=1440, ss=1):
    cam = Camera((0, 13.0, 18), (8, 12.0, -200), fov=60)
    sun = (-0.55, 0.12, -0.8)
    sky = gradient_sky("#24345e", "#f0a060", ground="#2a1a10", sun_dir=sun, sun_col="#ffc070", sun_size=0.9993)
    img = render_view(scene(), W, H, cam, sky, sun_dir=sun, sun_col="#ffa050", sun_soft=0.05,
                      ambient=("#6a6a8a", "#2a2010"), fog=(0.0011, "#9a7a78", 0.012, 0.0), shafts=0.3, exposure=1.1, steps=240,
                      tmax=1400, ao_scale=0.8, bloom=1.0, dof=(30.0, 0.3), vignette=0.38, grain=0.015,
                      grade=("#3a3a6a", "#ffe0b0"), ss=ss)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "dinosaures.jpg")
    Image.fromarray(render(W, H)).save(out, quality=93)
    print("saved", out)
