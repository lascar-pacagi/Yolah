"""
ice_scene.py — the Banquise background in 3D (scenery.py): a polar sea at night,
icebergs adrift under the aurora.

  the aurora    curtains of light in perspective: each a folded ribbon hanging
                100 to 300 km up, marched through its height (green at the
                foot, red and violet above), fine rays along it; reflected by
                the sea
  the sky       the stars (colours, sizes), the Milky Way and its dust lanes
  the icebergs  long tabular bergs on the horizon, a great dry-dock berg (two
                peaks, a saddle, flat calving faces), a nearer berg pierced by
                an arch, bergy bits; snow on their tops, walls of blue ice with
                darker bands, crevasses glowing deep blue, a wet waterline
  the sea       calm black water mirroring the sky, floes and pancake ice
                drifting near the eye, frost smoke lying on the water
  the light     the moon behind the viewer, the aurora's green in the shadows

    cd app/tools/sprites && python3 ice_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, vec, norm, srgb, sd_ellipsoid, fbm, hash3
from scenery import Camera, render_view, Volume

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
MOON = (-0.55, 0.32, 0.77)


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def ridged(p, o=5):
    return 1 - (2 * fbm(p, o) - 1).abs()


def P3(a, b, c):
    return torch.stack([a, b, c], 1)


# ─── the sky ────────────────────────────────────────────────────────────────
#  curtains: (distance to the north in km, fold amplitude km, fold wavelength km, brightness, phase)
CURTAINS = [(170.0, 35.0, 210.0, 1.0, 0.0), (300.0, 60.0, 330.0, 0.9, 2.1), (520.0, 90.0, 420.0, 0.8, 4.0), (800.0, 120.0, 600.0, 0.8, 5.5)]


def aurora(d):
    """The aurora's radiance along the directions d: emission summed through the curtains' height."""
    y = d[:, 1].clamp_min(0.012)
    out = torch.zeros_like(d)
    green, red, violet = srgb("#43ff8c"), srgb("#ff3a4a"), srgb("#8a4aff")
    n = 64
    jit = hash3(torch.floor(d * 4000.0))                      # a random offset per ray: no banding between the slices
    for k in range(n):
        h = 95 + (k + jit) * (230 / n)                        # km
        t = h / y
        x, z = d[:, 0] * t, -d[:, 2] * t                      # the ground point under the sample, km (z: north)
        far = (t < 2600).float()
        e = torch.zeros_like(y)
        for (z0, amp, lam, b, ph) in CURTAINS:
            zc = z0 + amp * torch.sin(x / lam * 6.283 + ph) + 0.4 * amp * torch.sin(x / lam * 15.1 + 2 * ph) \
                + 30 * (fbm(P3(x * 0.004 + ph, x * 0, x * 0), 3) - 0.5)
            dist = (z - zc).abs()
            sheet = torch.exp(-(dist / 7.0) ** 2) + 0.06 * torch.exp(-(dist / 30.0) ** 2)
            # rays: brightness along the curtain, in fine vertical striations
            along = x + 0.3 * (z - zc)
            rays = (0.15 + 1.6 * fbm(P3(along * 0.05 + ph * 3, ph + 0.0 * along, along * 0), 3) ** 3) \
                * (0.45 + 0.55 * ridged(P3(along * 0.6 + ph, along * 0 + ph * 2, along * 0), 2) ** 3)
            # the foot of the curtain: sharp, wavering in height; above it the glow fades
            foot = 100 + 14 * fbm(P3(x * 0.01 + ph, x * 0 + 2, x * 0), 2)
            prof = sstep(foot - 3, foot + 4, h) * torch.exp(-((h - foot).clamp_min(0)) / 55)
            e = e + sheet * rays * prof * b
        hs = h
        col = torch.lerp(green, red * 0.7, sstep(170, 250, hs)[:, None])
        col = torch.lerp(col, violet, (0.5 * sstep(230, 300, hs))[:, None])
        out = out + col * (e * far)[:, None] * (230 / n) * 0.11
    return out * (d[:, 1] > 0).float()[:, None]


def night_sky(d):
    y = d[:, 1].clamp(-1, 1)
    el = torch.asin(y)
    az = torch.atan2(d[:, 0], -d[:, 2])
    up = y.clamp(0, 1)
    c = srgb("#071424") * (1 - up[:, None] ** 0.35) + srgb("#01030a") * up[:, None] ** 0.35
    # stars: three layers, colours
    for (res, thr, gain) in ((900, 0.982, 0.25), (380, 0.992, 0.9), (150, 0.996, 3.0)):
        cell = torch.floor(torch.stack([az * res / math.pi, el * res / math.pi, az * 0 + res], 1))
        h = hash3(cell)
        fx = (az * res / math.pi) - cell[:, 0] - hash3(cell + 1.7)
        fy = (el * res / math.pi) - cell[:, 1] - hash3(cell + 4.1)
        spot = torch.exp(-(fx * fx + fy * fy) / (0.03 if res > 300 else 0.008))
        temp = hash3(cell + 9.3)
        col = torch.where((temp < 0.2)[:, None], srgb("#ffc29a"), torch.where((temp > 0.8)[:, None], srgb("#b0c8ff"), srgb("#fff6ec")))
        c = c + col * ((h > thr).float() * spot * gain * (0.4 + (h - thr) / (1 - thr)) * sstep(0.0, 0.08, y))[:, None]
    # the Milky Way: a band, its dust lanes
    mw = norm(vec(0.42, 0.5, 0.76))
    k = d @ mw
    band = torch.exp(-(k / 0.18) ** 2) * sstep(0.0, 0.15, y)
    lanes = ((fbm(d * 8 + 3, 5) - 0.45) * 3).clamp(0, 1) * torch.exp(-(k / 0.06) ** 2)
    c = c + srgb("#3a4260") * (band * (0.3 + 0.6 * fbm(d * 5, 4)) * (1 - 0.7 * lanes) * 0.35)[:, None]
    c = c + aurora(d)
    return torch.where((y < 0)[:, None], srgb("#030810").expand_as(c), c)


# ─── ice ────────────────────────────────────────────────────────────────────
def ice_mat():
    def pat(pl, pw, n):
        x, y, z = pw[:, 0], pw[:, 1], pw[:, 2]
        # walls: white compact snow-ice, bluer where the ice is old and clear, dark bands of marine ice
        bands = sstep(0.66, 0.7, fbm(P3(y * 0.6, x * 0.004, z * 0.004), 3))
        clear = sstep(0.4, 0.75, fbm(pw * 0.03 + 4, 4))
        a = torch.lerp(srgb("#d6ecf6"), srgb("#7ab8e0"), (0.3 * clear)[:, None])
        a = torch.lerp(a, srgb("#2a7ab0"), (0.4 * bands)[:, None])
        a = a * (0.85 + 0.2 * fbm(pw * 0.6, 3))[:, None]
        # snow on what faces up
        snow = sstep(0.7, 0.9, n[:, 1] + 0.1 * (fbm(pw * 0.2, 2) - 0.5))
        a = torch.lerp(a, srgb("#eef4fa") * (0.9 + 0.1 * fbm(pw * 2, 2))[:, None], snow[:, None])
        # the waterline: wet, darker
        a = a * (1 - 0.35 * sstep(1.6, 0.2, y))[:, None]
        # crevasses and the blue light through the ice (the walls glow faintly)
        crev = sstep(0.72, 0.85, ridged(P3(x * 0.08, y * 0.02, z * 0.08), 3)) * (1 - snow)
        a = torch.lerp(a, srgb("#0a4a8a"), (0.7 * crev)[:, None])
        side = (1 - snow) * (1 - n[:, 1].abs()).clamp(0, 1)
        em = srgb("#1a78d8") * (0.04 * side + 0.12 * crev + 0.05 * clear * side)[:, None]
        return {"albedo": a, "emissive": em, "rough": 0.35 + 0.4 * snow}
    return Mat("#d6ecf6", rough=0.4, sss=0.9, clearcoat=0.2, pattern=pat)


def facets(q, planes):
    """Cut a body by planes (n, c): the flat faces left where slabs of ice calved off."""
    d = None
    for (nx, ny, nz, c) in planes:
        k = q[:, 0] * nx + q[:, 1] * ny + q[:, 2] * nz - c
        d = k if d is None else torch.maximum(d, k)
    return d


def drydock(cx, cz, s):
    """The great berg: two peaks and a saddle between them, flat calving faces, fluted walls."""
    def f(p):
        q = (p - vec(cx, 0, cz)) / s
        a = sd_ellipsoid(1.0, 1.15, 0.8)(q - vec(-0.75, 0, 0))
        b = sd_ellipsoid(0.85, 1.6, 0.75)(q - vec(0.8, 0, 0.1))
        saddle = sd_ellipsoid(1.9, 0.45, 0.75)(q)
        body = torch.minimum(torch.minimum(a, b), saddle)
        cut = facets(q, [(0.25, 0.3, 0.92, 0.62), (-0.6, 0.35, 0.72, 0.7), (0.85, 0.2, 0.5, 1.35), (-0.9, 0.1, 0.4, 1.45),
                         (0.1, 0.95, 0.3, 1.35), (0.5, 0.86, 0.1, 1.38)])
        body = torch.maximum(body, cut)
        flute = 0.035 * ridged(P3(q[:, 0] * 7, q[:, 1] * 0.6, q[:, 2] * 7), 3) + 0.03 * fbm(q * 5, 3)
        scallop = 0.025 * (1 - ridged(q * 12 + 3, 2))           # the dimples melt carves in the walls
        return (body + flute + scallop) * s * 0.7
    return f


def arch_berg(cx, cz, s):
    """A nearer berg, worn by the sea into an arch."""
    def f(p):
        q = (p - vec(cx, 0, cz)) / s
        body = sd_ellipsoid(1.6, 0.95, 0.75)(q - vec(0, 0.05, 0))
        body = torch.minimum(body, sd_ellipsoid(0.6, 1.5, 0.55)(q - vec(-1.0, 0, 0.05)))
        body = torch.maximum(body, facets(q, [(0.15, 0.4, 0.9, 0.55), (0.95, 0.3, 0.1, 1.3), (-0.3, 0.95, 0.1, 0.95)]))
        hole = sd_ellipsoid(0.45, 0.5, 2.0)(q - vec(0.35, 0.1, 0))
        body = torch.maximum(body, -hole)
        body = body + 0.06 * ridged(P3(q[:, 0] * 7, q[:, 1] * 1.2, q[:, 2] * 7), 3) + 0.03 * fbm(q * 6, 2)
        return body * s * 0.7
    return f


def tabular(cx, cz, sx, sz, h):
    """A long tabular berg: an irregular outline, sheer fluted walls, a flat snowy top."""
    def f(p):
        x, y, z = p[:, 0] - cx, p[:, 1], p[:, 2] - cz
        u, v = x / sx, z / sz
        r = torch.sqrt(u * u + v * v) + 1e-6
        rr = 1 + 0.25 * (fbm(P3(u / r * 1.4 + 3, v / r * 1.4, u * 0), 4) - 0.5)
        side = (r - rr) * min(sx, sz) * 0.5
        side = side + 0.8 * ridged(P3(x * 0.35, y * 0.03, z * 0.35), 3)
        top = y - h * (1 + 0.04 * fbm(P3(x * 0.02, z * 0.02, x * 0), 2))
        return torch.maximum(torch.maximum(side, top), -y - 4) * 0.8
    return f


def bits(p):
    """Bergy bits and growlers scattered on the water (2×2 nearest cells)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    cell = 30.0
    d = torch.full_like(x, 1e9)
    i0, k0 = torch.floor(x / cell - 0.5), torch.floor(z / cell - 0.5)
    for di in (0, 1):
        for dk in (0, 1):
            i, k = i0 + di, k0 + dk
            idv = P3(i, k, i * 0 + 2)
            cx = (i + 0.5 + (hash3(idv) - 0.5) * 0.4) * cell
            cz = (k + 0.5 + (hash3(idv + 1.3) - 0.5) * 0.4) * cell
            r = 1.0 + 3.0 * hash3(idv + 2.2) ** 2
            keep = (hash3(idv + 5.5) < 0.35).float()
            q = P3((x - cx) / r, y / (r * (0.5 + 0.6 * hash3(idv + 7.1))), (z - cz) / r)
            b = (torch.sqrt((q * q).sum(1)) - 1) * r * 0.6
            b = b - 0.25 * r * ridged(P3(x, y, z) / r[:, None] + idv * 0.3, 3)
            d = torch.minimum(d, torch.where(keep > 0, b * 0.7, torch.full_like(b, 1e9)))
    bound = torch.maximum(y - 9, -y - 3)
    return torch.maximum(d.clamp_max(cell * 0.3), bound)


def floes(p):
    """Floes and pancake ice near the eye: flat plates at the waterline, their rims raised."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    cell = 7.0
    d = torch.full_like(x, 1e9)
    i0, k0 = torch.floor(x / cell - 0.5), torch.floor(z / cell - 0.5)
    for di in (0, 1):
        for dk in (0, 1):
            i, k = i0 + di, k0 + dk
            idv = P3(i, k, i * 0 + 9)
            cx = (i + 0.5 + (hash3(idv) - 0.5) * 0.5) * cell
            cz = (k + 0.5 + (hash3(idv + 1.3) - 0.5) * 0.5) * cell
            r = 1.2 + 2.0 * hash3(idv + 2.2)
            dens = 0.8 * sstep(0.5, 0.62, fbm(P3(cx * 0.02, cz * 0.02, cx * 0 + 3), 3))
            keep = (hash3(idv + 5.5) < dens).float()
            # an angular plate: the intersection of a few half planes turned at random
            lx, lz = x - cx, z - cz
            rad = torch.sqrt(lx * lx + lz * lz)
            edge = torch.full_like(x, -1e9)
            for j in range(5):
                a = hash3(idv + 11.1 * (j + 1)) * 6.283
                edge = torch.maximum(edge, lx * torch.cos(a) + lz * torch.sin(a) - r * (0.6 + 0.5 * hash3(idv + 3.7 * (j + 1))))
            edge = edge + 0.15 * (fbm(P3(x * 1.5, z * 1.5, idv[:, 0] * 0.1), 2) - 0.5)
            plate = torch.maximum(edge, (y - 0.12).abs() - 0.16 - 0.06 * fbm(P3(x * 2, z * 2, y * 0), 2))
            d = torch.minimum(d, torch.where(keep > 0, plate * 0.8, torch.full_like(plate, 1e9)))
    region = torch.maximum(z - 20, -z - 160)
    bound = torch.maximum(y - 1.0, -y - 1.0)
    return torch.maximum(torch.maximum(d.clamp_max(cell * 0.3), bound), region)


def sea(p):
    return p[:, 1] + 0.02 * fbm(p * vec(0.3, 0, 0.1), 2)


def sea_mat():
    def pat(pl, pw, n):
        return {"albedo": srgb("#0a1420").expand(pw.shape[0], 3), "rough": 0.03 + 0.05 * fbm(pw * 0.02, 2)}
    return Mat("#0a1420", rough=0.04, metal=1.0, pattern=pat)


def mountains(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    q = P3(x * 0.0007, x * 0 + 5, z * 0.0007)
    h = ridged(q, 6) ** 2.2 * 420 + 60 * fbm(q * 5, 3) - 120
    h = h * sstep(-4500, -6000, z)
    return torch.maximum((y - h) * 0.3, z + 4500)


def mountain_mat():
    def pat(pl, pw, n):
        steep = (1 - n[:, 1]).clamp(0, 1)
        rock = srgb("#22252c") * (0.6 + 0.5 * fbm(pw * 0.02, 3))[:, None]
        snowc = srgb("#dfe8f2") * (0.85 + 0.15 * fbm(pw * 0.01, 3))[:, None]
        k = ((steep - 0.5 + 0.3 * (fbm(pw * 0.01, 3) - 0.5)) * 5).clamp(0, 1)[:, None]
        return {"albedo": snowc * (1 - k) + rock * k}
    return Mat("#dfe8f2", rough=0.8, pattern=pat)


def frost_smoke(p):
    n = fbm(p * vec(0.02, 0.15, 0.02), 4)
    return torch.exp(-p[:, 1].clamp_min(0) / 3.0) * ((n - 0.45) * 2.5).clamp_min(0) * 0.03


def scene():
    sc = Scene()
    ice = ice_mat()
    sc.add(sea, sea_mat())
    sc.add(drydock(70, -300, 55), ice)
    sc.add(arch_berg(-150, -190, 36), ice)
    sc.add(tabular(330, -720, 170, 60, 38), ice)
    sc.add(tabular(-700, -1600, 420, 90, 55), ice)
    sc.add(tabular(900, -2300, 300, 70, 45), ice)
    sc.add(bits, ice)
    sc.add(floes, ice)
    sc.add(mountains, mountain_mat())
    return sc


NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    cam = Camera((0, 6.0, 30), (0, 30, -400), fov=62)
    vols = [Volume(frost_smoke, (-600, 0, -900), (600, 14, 25), albedo="#d8e4f0", g=0.5, steps=40, light_steps=2, light_dist=10)]
    img = render_view(scene(), W, H, cam, night_sky, sun_dir=MOON, sun_col="#8aa4d8", sun_soft=0.02,
                      ambient=("#2a6a5a", "#0a1622"), exposure=1.5, steps=320, tmax=9000, ao_scale=2.0, bloom=1.0,
                      dof=None, vignette=0.35, grain=0.012, grade=("#0a2a4a", "#d8fff0"), ss=ss,
                      volumes=[] if NOVOL else vols, fog=(0.0002, "#0a1a2a", 0.002, 0.0))
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "banquise.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
