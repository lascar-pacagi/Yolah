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
MOON = (-0.72, 0.3, 0.42)


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
        md = norm(vec(*MOON))
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
        # wind-scoured patches where the old blue ice shows through, crusts of older, greyer snow
        bare = sstep(0.62, 0.68, fbm(torch.stack([pw[:, 0] * 0.12 + 2, pw[:, 1] * 0, pw[:, 2] * 0.2], 1), 4)) * (pw[:, 1] < 0.35).float()
        a = torch.lerp(a, srgb("#7aa4c4") * (0.8 + 0.3 * fbm(pw * 6, 2))[:, None], (bare * 0.85)[:, None])
        a = a * (1 - 0.12 * sstep(0.55, 0.7, fbm(pw * 0.5 + 9, 3)))[:, None]
        sparkle = (hash3(torch.floor(pw * 120)) > 0.997).float()[:, None] * 2.0
        return {"albedo": a, "emissive": srgb("#ffffff") * sparkle * 0.05}
    return Mat("#e8eef6", rough=0.75, sss=0.6, pattern=pat)


def ice_m():
    def pat(pl, pw, n):
        layers = fbm(torch.stack([pw[:, 1] * 1.4, pw[:, 0] * 0.02, pw[:, 2] * 0.02], 1), 3)
        a = torch.lerp(srgb("#5aa6d6"), srgb("#d8f0fa"), (layers * 1.3 - 0.15).clamp(0, 1)[:, None]) * (0.8 + 0.25 * fbm(pw * 2, 3))[:, None]
        glow = torch.exp(-(pw[:, 1].clamp_min(0)) / 3.0)
        stripe = sstep(0.66, 0.7, fbm(torch.stack([pw[:, 1] * 1.8 + pw[:, 0] * 0.01, pw[:, 0] * 0.02, pw[:, 2] * 0.02], 1), 3))
        a = torch.lerp(a, srgb("#2a6aa8"), (0.5 * stripe)[:, None])
        a = torch.lerp(a, srgb("#eaf6fc"), (0.35 * sstep(0.4, 0.8, fbm(pw * vec(0.6, 0.08, 0.6), 3)))[:, None])
        side = (1 - n[:, 1].abs()).clamp(0, 1)
        return {"albedo": a, "emissive": srgb("#1a7ad0") * ((0.05 + 0.2 * glow + 0.15 * stripe) * side + 0.02)[:, None]}
    return Mat("#8ac8e8", rough=0.35, clearcoat=0.3, sss=0.9, pattern=pat)


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
    base = torch.exp(-(p[:, 1] + 0.2).clamp_min(0) / 1.6) * ((n - 0.35) * 2).clamp_min(0) * 0.06
    # frost smoke: the open water steams into the cold air, wisps rising and leaning with the wind
    ld = lead_line(p[:, 0] - 0.4 * p[:, 1], p[:, 2])
    wisp = ((fbm(p * vec(0.15, 0.35, 0.15) + 5, 4) - 0.4) * 2.5).clamp_min(0)
    smoke = torch.exp(-(ld.clamp_min(0) / 2.0) ** 2) * torch.exp(-p[:, 1].clamp_min(0) / 2.5) * wisp * 0.25
    return base + smoke


def ridge_line(x, z):
    """Signed distance (roughly, along z) to the pressure ridge crossing the foreground."""
    return z + 16 + 6 * torch.sin(x * 0.05) + 12 * (fbm(torch.stack([x * 0.03, x * 0, x * 0 + 1], 1), 2) - 0.5)


def lead_line(x, z):
    """The open lead: a long winding crack of black water across the middle distance."""
    w = 1.6 + 1.6 * fbm(torch.stack([x * 0.04 + 4, x * 0, x * 0 + 3], 1), 3)
    c = z + 5 + 3 * torch.sin(x * 0.05 + 1.0) + 14 * (fbm(torch.stack([x * 0.015, x * 0 + 2, x * 0], 1), 3) - 0.5)
    return c.abs() - w * (0.4 + 0.8 * sstep(-120, -20, x) * sstep(140, 40, x) + 0.4)


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def sea_ice(p):
    """The ice plain: drifts, a snow berm along the pressure ridge, sastrugi, and one long lead cut into it."""
    x, z = p[:, 0], p[:, 2]
    q2 = torch.stack([x * 0.15, x * 0, z * 0.15], 1)
    h = 0.12 * fbm(q2, 4) + 0.03 * fbm(p * 1.5, 3) + 0.25 * fbm(torch.stack([x * 0.02, x * 0 + 8, z * 0.02], 1), 3)
    # floes: old plates of ice slightly raised, their edges softened by snow
    fl = fbm(torch.stack([x * 0.06 + 1, x * 0 + 4, z * 0.06], 1), 3)
    h = h + 0.18 * sstep(0.48, 0.56, fl)
    # the berm of snow and rubble along the pressure ridge
    rl = ridge_line(x, z)
    h = h + (0.55 + 0.4 * fbm(p * 0.3, 2)) * torch.exp(-(rl / 2.6) ** 2)
    # sastrugi: sharp wind-carved ridges in the snow, near the eye
    near = torch.exp(-((z + 10) / 45) ** 2)
    w = x * 0.9 + z * 0.35 + 2.5 * fbm(torch.stack([x * 0.08, x * 0, z * 0.08], 1), 3)
    h = h + near * 0.07 * (1 - (2 * fract_(w * 0.6) - 1).abs()) ** 3 * (0.4 + fbm(p * 0.3 + 4, 2))
    # the lead: its banks drop to the water, a rim of broken ice along them
    ld = lead_line(x, z)
    rim = 0.12 * torch.exp(-((ld - 0.6) / 0.5) ** 2) * fbm(p * 1.2 + 3, 2)
    cut = ((0.3 - ld) / 0.5).clamp(0, 1)
    # pans of broken ice drifting in the lead
    pan = sstep(0.6, 0.63, fbm(torch.stack([x * 0.5 + 7, x * 0 + 1, z * 0.7], 1), 3))
    h = (h + rim) * (1 - cut) + (-0.3 * (1 - pan) + 0.04 * pan) * cut
    return (p[:, 1] - h) * 0.6


def tab_berg(q, sx, sz, hgt, seed):
    """A tabular iceberg: an irregular outline (coves, broken corners), a calved terrace on one side,
    walls fluted by melt streams, a wave-cut notch at the waterline, a gently undulating top."""
    x, y, z = q[:, 0], q[:, 1], q[:, 2]

    def outline(cx, s_, k):
        u, v = (x - cx) / (sx * s_), z / (sz * s_)
        rho = torch.sqrt(u * u + v * v) + 1e-6
        cu, cv = u / rho, v / rho
        rr = 1 + 0.45 * (fbm(torch.stack([cu * 1.3 + seed + k, cv * 1.3, y * 0 + k], 1), 4) - 0.5) \
            + 0.08 * (fbm(torch.stack([cu * 5 + seed, cv * 5 + k, y * 0], 1), 3) - 0.5)
        return (rho - rr) * min(sx, sz) * s_ * 0.55
    # vertical grooves cut by meltwater, and large facets where slabs have calved off
    flute = 0.6 * ridged(torch.stack([x * 0.9, y * 0.04, z * 0.9], 1), 3) ** 2 \
        + 0.05 * min(sx, sz) * fbm(torch.stack([x * (3.0 / sx), y * (0.6 / hgt), z * (3.0 / sz)], 1) + seed, 3)
    top = y - hgt * (1 + 0.05 * (fbm(q * (0.6 / sx), 3) - 0.5))
    k = 0.05 * hgt
    smax = lambda a, b: 0.5 * (a + b) + torch.sqrt(((a - b) * 0.5) ** 2 + k * k * 0.25)
    main = smax(outline(sx * 0.12, 0.88, 0.0) + flute, top)
    terrace = smax(outline(-sx * 0.05, 1.0, 3.0) + flute, y - hgt * 0.42 * (1 + 0.1 * fbm(q * (1.0 / sx), 2)))
    d = torch.minimum(main, terrace)
    notch = 0.06 * hgt * torch.exp(-((y - 0.25) / (0.08 * hgt)) ** 2)
    return torch.maximum(d + notch, -y - 2.0) * 0.8


def water():
    return Mat("#2a3644", rough=0.04, clearcoat=1.0, metal=1.0)


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
                                         (-120, -210, 40, 28, 26, "pin"), (16, -44, 3.5, 2.6, 3.0, "pin"), (-18, -32, 3, 2.2, 3.2, "pin")):
        if kind == "tab":
            def berg(q, sx=sx, sz=sz, hgt=hgt, sd=rnd.random() * 10):
                return tab_berg(q, sx, sz, hgt, sd)
        else:
            def berg(q, sx=sx, sz=sz, hgt=hgt):
                d = sd_ellipsoid(sx, hgt * 1.1, sz)(q)
                spire = sd_ellipsoid(sx * 0.3, hgt * 2.6, sz * 0.35)(q - vec(sx * 0.3, 0, 0))
                spire2 = sd_ellipsoid(sx * 0.22, hgt * 1.8, sz * 0.3)(q - vec(-sx * 0.35, 0, sz * 0.2))
                d = torch.minimum(torch.minimum(d, spire), spire2)
                r = ridged(q * (1.0 / sz), 5)
                return (d - 0.22 * sz * r ** 2 + 0.06 * sz) * 0.8
        sc.add(berg, ice, T(cx, 0, cz))
        sc.add(lambda q, b=berg, hgt=hgt: torch.maximum(b(q - vec(0, 0.04 * hgt, 0)) - 0.04 * hgt, 0.85 * hgt - q[:, 1] + 0.2 * hgt * fbm(q * (1.0 / hgt), 2)),
               snow(), T(cx, 0, cz))
    # a pressure ridge in the foreground: slabs of sea ice pushed up and tilted, crossing the view
    def slabs(q):
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        rl = ridge_line(x, z)
        rise = (0.25 + 1.1 * fbm(torch.stack([x * 0.04, x * 0 + 3, x * 0], 1), 3)).clamp_min(0)
        d = torch.full_like(x, 1e9)
        for row, (cell, zo, seed) in enumerate(((1.5, 0.0, 0.0), (1.15, 1.3, 31.0), (1.9, -1.4, 57.0))):
            for off in (-1, 0, 1):
                i = torch.floor(x / cell) + off
                hs = [hash3(torch.stack([i, i * 0 + seed + k, i * 0], 1)) for k in range(7)]
                cx = (i + 0.5 + (hs[0] - 0.5) * 0.5) * cell
                sx_, sz_, th = 0.45 + 0.6 * hs[1], 0.6 + 0.9 * hs[2], 0.12 + 0.18 * hs[3]
                a1, a2, a3 = (hs[4] - 0.5) * 2.6, (hs[5] - 0.5) * 1.2, (hs[6] - 0.5) * 1.4
                lx = x - cx
                ly = y - (0.2 + 0.9 * hs[3]) * rise * (1.0 - 0.3 * row)
                lz = rl + zo + (hs[6] - 0.5) * 1.0
                ca, sa, cb, sb, cc_, sc_ = torch.cos(a1), torch.sin(a1), torch.cos(a2), torch.sin(a2), torch.cos(a3), torch.sin(a3)
                z1, y1 = lz * ca - ly * sa, lz * sa + ly * ca
                x2, y2 = lx * cb - y1 * sb, lx * sb + y1 * cb
                x3, z3 = x2 * cc_ - z1 * sc_, x2 * sc_ + z1 * cc_
                w = torch.stack([x3, y2, z3], 1)
                b = sd_box(1.0, 1.0, 1.0, 0.0)(w / torch.stack([sx_, th, sz_], 1)) * torch.minimum(torch.minimum(sx_, th), sz_)
                d = torch.minimum(d, b)
        d = d - 0.04 * fbm(q * 2.5, 2)
        return torch.maximum(d * 0.6, (x.abs() - 70))
    sc.add(slabs, ice)
    # big blocks pushed up near the viewer, framing the lead, half buried in drifted snow
    for (bx, bz, sx_, sy_, sz_, ax, ay, az_) in ((-11.5, 1.5, 1.6, 0.35, 1.2, 0.5, 0.3, 0.35), (-9.2, 3.2, 0.9, 0.25, 0.8, -0.3, 1.1, 0.6),
                                                  (12.5, 0.5, 1.8, 0.4, 1.3, -0.45, -0.4, -0.3), (10.0, 2.8, 0.8, 0.22, 0.7, 0.6, 0.9, -0.5),
                                                  (14.5, -2.5, 1.2, 0.3, 1.0, 0.3, 0.2, 0.7)):
        blk = lambda q, sx_=sx_, sy_=sy_, sz_=sz_: (sd_box(sx_, sy_, sz_, 0.06)(q) - 0.08 * fbm(q * 1.6, 3)) * 0.8
        sc.add(blk, ice, T(bx, 0.2, bz) * RY(ay) * RX(ax) * RZ(az_))
        sc.add(lambda q, b=blk: torch.maximum(b(q) - 0.05, -(q[:, 1] - 0.1 - 0.1 * fbm(q * 2, 2))) , snow(), T(bx, 0.2, bz) * RY(ay) * RX(ax) * RZ(az_))
    sc.add(mountains, mountain_mat())
    return sc


def render(W=2560, H=1440, ss=1):
    cam = Camera((0, 4.6, 12), (0, 4.6, -60), fov=62)
    sky = aurora_sky()
    vols = [Volume(drift_mist, (-300, -1, -400), (300, 8, 14), albedo="#d8e4f0", g=0.5, steps=32, light_steps=2, light_dist=8)]
    img = render_view(scene(), W, H, cam, sky, sun_dir=MOON, sun_col="#a8c4ff", sun_soft=0.03, volumes=vols,
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
