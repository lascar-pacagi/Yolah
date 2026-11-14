"""
ice_scene.py — the Banquise's background in 3D (scenery.py): the sea ice at
night under the aurora.

  the sky       deep blue night, the stars, the aurora: curtains of green
                light with purple tops, folding across the sky, fine
                vertical rays
  the ice       a plain of sea ice: snow drifts, pressure ridges of broken
                blocks, open leads of black water reflecting the sky
  icebergs      tabular and broken ones, their ice glowing faintly blue
  mountains     snow-covered peaks on the horizon
  the light     the moon low, cold; the aurora's green on the snow; mist

    cd app/tools/sprites && python3 ice_scene.py [W H out]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, norm, srgb, sd_sphere, sd_box, sd_ellipsoid, sd_cone, fbm, hash3
from scenery import Camera, render_view
from origami import sd_hull

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


def aurora_sky():
    zen, hor = srgb("#02040c"), srgb("#0a1c2e")
    green, violet = srgb("#3affa0"), srgb("#9a4aff")

    def sky(d):
        y = d[:, 1].clamp(-1, 1)
        el = torch.asin(y)
        az = torch.atan2(d[:, 0], -d[:, 2])
        up = y.clamp(0, 1)
        c = hor * (1 - up[:, None] ** 0.4) + zen * up[:, None] ** 0.4
        # stars
        cell = torch.floor(torch.stack([az * 260, el * 260, az * 0], 1))
        h = hash3(cell)
        star = (h > 0.995).float() * (y > 0.02).float() * (0.4 + 3.0 * (h - 0.995) / 0.005)
        c = c + star[:, None] * srgb("#e8f0ff") * 0.8
        # the aurora: two folded curtains; the base of each at a height that wanders with the azimuth
        aur = torch.zeros_like(y)
        top = torch.zeros_like(y)
        for k, (a0, amp, base, w) in enumerate(((0.0, 0.9, 0.2, 1.0), (1.9, 0.7, 0.34, 0.65))):
            fold = base + amp * torch.sin(az * 2.2 + a0) * 0.3 + 0.06 * fbm(torch.stack([az * 3 + k, az * 0, az * 0 + k * 7], 1), 3)
            rel = el - fold
            body = torch.exp(-(rel.clamp_min(0) / (0.16 * w))) * (rel > -0.08).float() * torch.exp(-((rel.clamp_max(0)) / 0.025) ** 2)
            rays = 0.55 + 0.45 * torch.sin(az * 90 + 14 * fbm(torch.stack([az * 4, el * 0.8, az * 0 + k], 1), 3)) * fbm(torch.stack([az * 20, el * 2, az * 0 + k], 1), 2)
            bright = 0.5 + 0.8 * fbm(torch.stack([az * 1.7 + k * 3, az * 0, az * 0], 1), 3)
            aur = aur + body * rays * bright * w
            top = top + (rel / 0.4).clamp(0, 1) * body * w
        c = c + green * (aur * (1 - top))[:, None] * 0.9 + violet * (aur * top)[:, None] * 0.7
        below = (y < 0).float()[:, None]
        return c * (1 - below) + hor * 0.5 * below
    return sky


def snow():
    def pat(pl, pw, n):
        a = srgb("#e8eef6") * (0.85 + 0.15 * fbm(pw * 3, 3))[:, None]
        sparkle = (hash3(torch.floor(pw * 120)) > 0.997).float()[:, None] * 2.0
        return {"albedo": a, "emissive": srgb("#ffffff") * sparkle * 0.05}
    return Mat("#e8eef6", rough=0.75, sss=0.6, pattern=pat)


def ice_m():
    return Mat("#8ac8e8", rough=0.12, clearcoat=1.0, sss=0.8, pattern=lambda pl, pw, n: {
        "albedo": srgb("#9ad2ee") * (0.75 + 0.35 * fbm(pw * 2, 3))[:, None], "emissive": srgb("#2a7ab0") * 0.05})


def sea_ice(p):
    """The ice plain: drifts, pressure ridges along a few lines, and leads of open water cut into it."""
    x, z = p[:, 0], p[:, 2]
    q2 = torch.stack([x * 0.15, x * 0, z * 0.15], 1)
    h = 0.12 * fbm(q2, 4) + 0.03 * fbm(p * 1.5, 3)
    # ridges: ridged noise along wandering lines
    r = 1 - (2 * fbm(torch.stack([x * 0.05 + 3, x * 0, z * 0.05], 1), 3) - 1).abs()
    h = h + 0.7 * (r ** 10) * (0.5 + 0.6 * fbm(p * 0.8, 2))
    # leads: where a low-frequency noise is near 0.5 the ice is open
    lead = (fbm(torch.stack([x * 0.012 + 9, x * 0, z * 0.01], 1), 3) - 0.5).abs()
    cut = ((0.012 - lead) / 0.006).clamp(0, 1)
    h = h * (1 - cut) - 0.25 * cut
    return (p[:, 1] - h) * 0.6


def water():
    return Mat("#020408", rough=0.02, clearcoat=1.0, metal=0.0, pattern=lambda pl, pw, n: {
        "albedo": srgb("#020408").expand(pw.shape[0], 3)})


def scene():
    sc = Scene()
    sc.add(sea_ice, snow())
    sc.add(sd_box(400, 0.1, 400), water(), T(0, -0.12, -100))
    ice = ice_m()
    # icebergs: hulls of random points, displaced
    import random
    rnd = random.Random(5)
    for (cx, cz, s) in ((-30, -90, 16), (26, -70, 10), (70, -170, 30), (-110, -230, 34), (10, -42, 4), (-16, -34, 3)):
        pts = []
        for _ in range(16):
            a = rnd.uniform(0, math.tau)
            r = rnd.uniform(0.6, 1.0) * s
            pts.append((math.cos(a) * r, rnd.uniform(-0.3, 1.0) * s * (0.6 if s > 10 else 1.0), math.sin(a) * r * 0.7))
        pts += [(0, -0.4 * s, 0)]
        hull = sd_hull(pts)
        sc.add(lambda q, h=hull, s=s: h(q) - 0.04 * s * fbm(q * (3.0 / s), 3), ice, T(cx, 0, cz))
        sc.add(lambda q, h=hull, s=s: torch.maximum(h(q - vec(0, 0.06 * s, 0)) - 0.03 * s, 0.15 * s - q[:, 1] + 0.2 * s * fbm(q * (2.0 / s), 2)),
               snow(), T(cx, 0, cz))
    # mountains on the horizon
    for (cx, cz, h, r) in ((-500, -1100, 150, 260), (-150, -1250, 230, 300), (260, -1150, 180, 280), (650, -1050, 130, 230)):
        sc.add(lambda q, h=h, r=r: (torch.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2) * h / r + q[:, 1] - h) * 0.5
               - 18 * fbm(q * 0.012, 4), snow(), T(cx, 0, cz))
    return sc


def render(W=2560, H=1440, ss=1):
    cam = Camera((0, 3.2, 12), (0, 6.0, -60), fov=62)
    sky = aurora_sky()
    img = render_view(scene(), W, H, cam, sky, sun_dir=(0.6, 0.18, -0.75), sun_col="#a8c4ff", sun_soft=0.04,
                      ambient=("#3a7a86", "#1a2a3a"), fog=(0.006, "#0e2236", 0.02, 0.0), exposure=1.3, steps=220,
                      tmax=1600, ao_scale=0.6, bloom=0.9, dof=(60.0, 0.25), vignette=0.35, grain=0.015,
                      grade=("#2a5a8a", "#d8fff0"), ss=ss)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "banquise.jpg")
    Image.fromarray(render(W, H)).save(out, quality=93)
    print("saved", out)
