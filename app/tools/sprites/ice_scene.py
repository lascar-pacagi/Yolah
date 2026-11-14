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
from scenery import Camera, render_view, Volume
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
        star = (h > 0.993).float() * (y > 0.02).float() * (0.3 + 3.0 * (h - 0.993) / 0.007)
        tint = torch.lerp(srgb("#ffd8b0"), srgb("#c8dcff"), hash3(cell + 3.3)[:, None])
        c = c + star[:, None] * tint * 0.8
        # the Milky Way: a soft band across the sky, dust lanes, a haze of faint stars
        mw_n = norm(vec(0.35, 0.55, 0.76))
        band = torch.exp(-((d @ mw_n) / 0.16) ** 2) * (y > 0).float()
        dust = fbm(d * 9 + 2, 4)
        faint = (hash3(torch.floor(torch.stack([az * 700, el * 700, az * 0], 1))) > 0.97).float()
        c = c + srgb("#4a5470")[None] * (band * (0.25 + 0.5 * dust) * (1 - 0.6 * (fbm(d * 22, 3) > 0.6).float()) * 0.35)[:, None]
        c = c + srgb("#c8d4ff")[None] * (band * faint * 0.35)[:, None]
        # the moon, low in the north-east, and its halo
        md = norm(vec(0.62, 0.2, -0.76))
        cm = (d @ md)
        c = c + srgb("#e8ecf4") * ((cm > 0.99985).float() * 6.0 * (0.8 + 0.2 * fbm(d * 400, 3)))[:, None]
        c = c + srgb("#8aa0c8") * (cm.clamp(0, 1) ** 400 * 0.6 + cm.clamp(0, 1) ** 30 * 0.08)[:, None]
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
    def pat(pl, pw, n):
        layers = fbm(torch.stack([pw[:, 1] * 1.4, pw[:, 0] * 0.02, pw[:, 2] * 0.02], 1), 3)
        a = torch.lerp(srgb("#5aa6d6"), srgb("#d8f0fa"), (layers * 1.3 - 0.15).clamp(0, 1)[:, None]) * (0.8 + 0.25 * fbm(pw * 2, 3))[:, None]
        glow = torch.exp(-(pw[:, 1].clamp_min(0)) / 1.5)
        return {"albedo": a, "emissive": srgb("#1a7ad0") * (0.03 + 0.12 * glow)[:, None]}
    return Mat("#8ac8e8", rough=0.18, clearcoat=0.8, sss=0.9, pattern=pat)


def fract_(x):
    return x - torch.floor(x)


def ridged(p, octaves=5):
    return 1 - (2 * fbm(p, octaves) - 1).abs()


def range_h(x, z):
    """The mountains on the horizon: ridged noise, sharp crests, valleys between."""
    q = torch.stack([x * 0.0016, x * 0 + 5, z * 0.0016], 1)
    m = ridged(q, 5) ** 2.2 * 300 + 50 * fbm(q * 4, 3)
    far = ((-z - 1400) / 500).clamp(0, 1)
    return m * (fbm(torch.stack([x * 0.0006, x * 0, z * 0.0006], 1), 2) * 1.6).clamp(0, 1) * far - 40


def mountains(p):
    return torch.maximum((p[:, 1] - range_h(p[:, 0], p[:, 2])) * 0.3, p[:, 2] + 1300)


def mountain_mat():
    def pat(pl, pw, n):
        steep = (1 - n[:, 1]).clamp(0, 1)
        rock = srgb("#2a2c32") * (0.6 + 0.5 * fbm(pw * 0.05, 3))[:, None]
        snowc = srgb("#e8eef6") * (0.85 + 0.15 * fbm(pw * 0.02, 3))[:, None]
        k = ((steep - 0.45 + 0.25 * (fbm(pw * 0.03, 3) - 0.5)) * 6).clamp(0, 1)[:, None]
        return {"albedo": snowc * (1 - k) + rock * k}
    return Mat("#e8eef6", rough=0.8, pattern=pat)


def drift_mist(p):
    """Frost smoke over the leads and blowing snow, close to the ice."""
    n = fbm(p * vec(0.04, 0.2, 0.04), 4)
    return torch.exp(-(p[:, 1] + 0.2).clamp_min(0) / 1.6) * ((n - 0.35) * 2).clamp_min(0) * 0.06


def sea_ice(p):
    """The ice plain: drifts, pressure ridges along a few lines, and leads of open water cut into it."""
    x, z = p[:, 0], p[:, 2]
    q2 = torch.stack([x * 0.15, x * 0, z * 0.15], 1)
    h = 0.12 * fbm(q2, 4) + 0.03 * fbm(p * 1.5, 3)
    # ridges: ridged noise along wandering lines
    r = 1 - (2 * fbm(torch.stack([x * 0.05 + 3, x * 0, z * 0.05], 1), 3) - 1).abs()
    h = h + 0.7 * (r ** 10) * (0.5 + 0.6 * fbm(p * 0.8, 2))
    # sastrugi: sharp wind-carved ridges in the snow, near the eye
    near = torch.exp(-((z + 10) / 45) ** 2)
    w = x * 0.9 + z * 0.35 + 2.5 * fbm(torch.stack([x * 0.08, x * 0, z * 0.08], 1), 3)
    h = h + near * 0.07 * (1 - (2 * fract_(w * 0.6) - 1).abs()) ** 3 * (0.4 + fbm(p * 0.3 + 4, 2))
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
    # icebergs: tabular ones (flat tops, sheer walls carved by the sea, a wave-cut notch at the waterline)
    # and weathered ones (domes and pinnacles), their walls glowing faintly blue
    import random
    rnd = random.Random(5)
    for (cx, cz, sx, sz, hgt, kind) in ((-38, -80, 20, 11, 11, "tab"), (30, -64, 9, 7, 9, "pin"), (85, -160, 48, 24, 20, "tab"),
                                         (-120, -210, 40, 28, 26, "pin"), (12, -40, 4, 3, 2.4, "tab"), (-18, -32, 3, 2.2, 3.2, "pin")):
        if kind == "tab":
            def berg(q, sx=sx, sz=sz, hgt=hgt):
                w = q + vec(1, 0, 1) * (fbm(q * (0.6 / sz), 3)[:, None] - 0.5) * sz * 0.35
                body = sd_box(sx, hgt, sz, 0.15 * hgt)(w - vec(0, 0, 0))
                walls = 0.06 * sz * (fbm(torch.stack([q[:, 0] * (1.5 / sz), q[:, 1] * (0.4 / hgt), q[:, 2] * (1.5 / sz)], 1), 4) - 0.5)
                notch = 0.25 * hgt * torch.exp(-((q[:, 1] - 0.3) / (0.12 * hgt)) ** 2)
                return body - walls + notch
        else:
            def berg(q, sx=sx, sz=sz, hgt=hgt):
                d = sd_ellipsoid(sx, hgt * 1.6, sz)(q)
                spire = sd_ellipsoid(sx * 0.35, hgt * 2.4, sz * 0.4)(q - vec(sx * 0.3, 0, 0))
                d = torch.minimum(d, spire)
                r = 1 - (2 * fbm(q * (1.2 / sz), 4) - 1).abs()
                return d - 0.18 * sz * r + 0.05 * sz
        sc.add(berg, ice, T(cx, 0, cz))
        sc.add(lambda q, b=berg, hgt=hgt: torch.maximum(b(q - vec(0, 0.04 * hgt, 0)) - 0.04 * hgt, 0.85 * hgt - q[:, 1] + 0.2 * hgt * fbm(q * (1.0 / hgt), 2)),
               snow(), T(cx, 0, cz))
    # a pressure ridge in the foreground: slabs of sea ice pushed up and tilted, crossing the view
    def slabs(q):
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        line = z + 16 + 6 * torch.sin(x * 0.05) + 3 * fbm(torch.stack([x * 0.03, x * 0, z * 0], 1), 2) * 4
        i = torch.round(x / 1.7)
        h1, h2, h3 = hash3(torch.stack([i, i * 0 + 2, i * 0], 1)), hash3(torch.stack([i, i * 0 + 7, i * 0], 1)), hash3(torch.stack([i, i * 0 + 9, i * 0], 1))
        lx = x - i * 1.7
        a1, a2 = (h1 - 0.5) * 1.6, (h2 - 0.5) * 0.9
        ca, sa, cb, sb = torch.cos(a1), torch.sin(a1), torch.cos(a2), torch.sin(a2)
        ly = y - 0.2 - 0.5 * h3
        lz = line + (h3 - 0.5) * 1.2
        # rotate (lz, ly) by a1 (tilt), (lx, ly) by a2 (roll)
        z1, y1 = lz * ca - ly * sa, lz * sa + ly * ca
        x2, y2 = lx * cb - y1 * sb, lx * sb + y1 * cb
        w = torch.stack([x2, y2, z1], 1)
        d = sd_box(0.75, 0.2, 0.9, 0.04)(w) - 0.05 * fbm(q * 2.5, 2)
        return torch.maximum(d * 0.6, (x.abs() - 70))
    sc.add(slabs, ice)
    sc.add(mountains, mountain_mat())
    return sc


def render(W=2560, H=1440, ss=1):
    cam = Camera((0, 3.2, 12), (0, 6.0, -60), fov=62)
    sky = aurora_sky()
    vols = [Volume(drift_mist, (-300, -1, -400), (300, 8, 14), albedo="#d8e4f0", g=0.5, steps=32, light_steps=2, light_dist=8)]
    img = render_view(scene(), W, H, cam, sky, sun_dir=(0.62, 0.2, -0.76), sun_col="#a8c4ff", sun_soft=0.03, volumes=vols,
                      ambient=("#3a7a86", "#1a2a3a"), fog=(0.006, "#0e2236", 0.02, 0.0), exposure=1.3, steps=220,
                      tmax=4000, ao_scale=0.6, bloom=0.9, dof=(60.0, 0.25), vignette=0.35, grain=0.015,
                      grade=("#2a5a8a", "#d8fff0"), ss=ss)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "banquise.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
