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
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_capsule, sd_cone, sd_ellipsoid, sd_box, fbm, hash3
from scenery import Camera, render_view, gradient_sky, Atmosphere, Volume

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


def ridged(p, octaves=4):
    """Ridged noise: sharp crests (1 − |2n − 1|), the look of eroded rock."""
    return 1 - (2 * fbm(p, octaves) - 1).abs()


def volcano(p):
    r = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
    az = torch.atan2(p[:, 0], p[:, 2])
    warp = 2.5 * fbm(p * 0.006, 3)
    gully = (torch.sin(az * 17 + warp).abs() ** 0.5 - 0.6) * 9 * (r / 160).clamp(0, 1)
    gully = gully + (torch.sin(az * 41 + 2 * warp).abs() ** 0.5 - 0.6) * 3 * (r / 160).clamp(0, 1)
    h = 260 - r * 0.62 + 16 * fbm(p * 0.015, 4) + gully + 6 * ridged(p * 0.03, 3)
    crater = torch.exp(-(r / 32) ** 2) * 45
    return (p[:, 1] - (h - crater).clamp(max=232)) * 0.35


def volcano_mat():
    def pat(pl, pw, n):
        x, y, z = pl[:, 0], pl[:, 1], pl[:, 2]
        r = torch.sqrt(x * x + z * z)
        rock = torch.lerp(srgb("#2a2420"), srgb("#5e5246"), fbm(pl * 0.04, 4)[:, None]) * (0.6 + 0.6 * ridged(pl * 0.25, 3))[:, None]
        ash = srgb("#8a8078") * (0.7 + 0.3 * fbm(pl * 0.1, 3))[:, None]
        a = torch.lerp(rock, ash, (n[:, 1].clamp(0, 1) ** 3 * 0.6)[:, None])            # ash settles on the gentler slopes
        # lava flows: branching channels down the slope, crust with glowing cracks
        az = torch.atan2(x, z) + 0.35 * fbm(pl * 0.008, 3)
        flow = torch.zeros_like(r)
        for (c, w, k) in ((0.15, 0.06, 1.0), (-0.35, 0.04, 0.8), (0.55, 0.035, 0.7)):
            wig = c + 0.08 * torch.sin(r * 0.05 + c * 9) + 0.04 * fbm(pl * 0.03 + c, 2)
            branch = torch.exp(-((az - wig) / (w * (0.6 + r / 300))) ** 2) * k
            flow = torch.maximum(flow, branch * sstep(320, 120, r))
        flow = (flow - 0.15).clamp(0, 1) / 0.85
        heat = sstep(80, 230, y)
        crack = (fbm(pl * 0.5, 3) > 0.52).float() * 0.8 + 0.2
        crust = srgb("#14100e")
        a = torch.lerp(a, crust, flow[:, None])
        top = torch.exp(-((r - 24) / 9) ** 2) * (y > 205).float()
        em = srgb("#ff4a10") * (flow * crack * (0.4 + 3.0 * heat) + top * 7.0)[:, None] * (0.6 + 0.8 * fbm(pl * 0.08, 2))[:, None]
        return {"albedo": a, "emissive": em}
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


def scatter(cell, rmax, jit, region, make, seed, keep, top=2.5, ext=1.3):
    """One object per cell (domain repetition), standing on the terrain; see bomb_scene.scatter."""
    margin = cell / 2 - jit - rmax * ext          # the widest reach of an object: rmax · ext
    assert margin > 0.1 * cell, margin
    x0, x1, z0, z1 = region

    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        out = torch.maximum(torch.maximum(x0 - x, x - x1), torch.maximum(z0 - z, z - z1))
        ix, iz = torch.floor(x / cell), torch.floor(z / cell)
        id3 = torch.stack([ix, ix * 0 + seed, iz], 1)
        h1, h2, h3 = hash3(id3), hash3(id3 + 3.7), hash3(id3 + 11.3)
        cx = (ix + 0.5) * cell + (h1 - 0.5) * 2 * jit
        cz = (iz + 0.5) * cell + (h2 - 0.5) * 2 * jit
        g = -terrain(torch.stack([cx, cx * 0, cz], 1)) / 0.55
        r = rmax * (0.5 + 0.5 * h3)
        d = make(torch.stack([x - cx, y - g, z - cz], 1), r, h3)
        edge = cell / 2 - torch.maximum((x - (ix + 0.5) * cell).abs(), (z - (iz + 0.5) * cell).abs())
        bound = torch.maximum(edge + margin, y - g - top * rmax - 1.0)
        d = torch.where(hash3(id3 + 5.1) < keep, torch.minimum(d, bound), bound)
        return torch.maximum(d, out)
    return f


def fern_clump(loc, r, h):
    q = loc / torch.stack([r * 1.3, r * 0.45, r * 1.3], 1)
    d = (torch.sqrt((q * q).sum(-1)) - 1) * r * 0.4
    fr = fbm(torch.stack([loc[:, 0] * (9 / r), loc[:, 1] * (3 / r), loc[:, 2] * (9 / r)], 1) + h[:, None] * 5, 2)
    return d + (0.5 - fr) * r * 0.35


def far_tree(loc, r, h):
    """A distant araucaria: a bare trunk and a flat umbrella crown."""
    hgt = r * 3.2
    trunk = torch.sqrt(loc[:, 0] ** 2 + loc[:, 2] ** 2) - r * 0.05
    trunk = torch.maximum(trunk, torch.maximum(loc[:, 1] - hgt, -loc[:, 1]))
    c = loc - torch.stack([r * 0, hgt + r * 0 - r * 0.15, r * 0], 1)
    q = c / torch.stack([r * 0.9, r * 0.35, r * 0.9], 1)
    crown = (torch.sqrt((q * q).sum(-1)) - 1) * r * 0.3 - 0.12 * r * fbm(loc * (3 / r[:, None]) + h[:, None], 2)
    return torch.minimum(trunk, crown)


def rock(loc, r, h):
    q = loc / torch.stack([r * 1.2, r * 0.7, r], 1)
    return (torch.sqrt((q * q).sum(-1)) - 1) * r * 0.6 - 0.2 * r * fbm(loc * (3 / r[:, None]) + h[:, None] * 9, 3)


def scene():
    sc = Scene()
    sc.add(terrain, ground_mat())
    sc.add(forest, frond_mat("#2e4a1e"))
    sc.add(sd_box(80, 0.1, 60), lake_mat(), T(0, -2.1, -70))
    sc.add(volcano, volcano_mat(), T(40, -10, -620))
    # ferns covering the ground near the eye, stones, a forest of araucarias on the far hills
    sc.add(scatter(3.6, 0.85, 0.3, (-70.0, 70.0, -70.0, 12.0), fern_clump, 1.0, 0.9, top=1.0), frond_mat("#3a6224"))
    sc.add(scatter(5.0, 0.6, 1.2, (-80.0, 80.0, -90.0, 10.0), rock, 2.0, 0.35), Mat("#5a5048", rough=0.85, pattern=lambda pl, pw, n: {
        "albedo": torch.lerp(srgb("#3a322c"), srgb("#7a6e60"), fbm(pw * 1.5, 3)[:, None])}))
    sc.add(scatter(20.0, 6.0, 2.0, (-600.0, 600.0, -480.0, -150.0), far_tree, 3.0, 0.55, top=4.2, ext=0.95), frond_mat("#2a4a20"))
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


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


VX, VY, VZ = 40.0, 220.0, -620.0           # the crater (world)


def ash_plume(p):
    """The eruption column: rising from the crater, bending with the wind, spreading into an umbrella."""
    x, y, z = p[:, 0] - VX, p[:, 1] - VY, p[:, 2] - VZ
    H = 420.0
    u = (y / H).clamp(0, 1.3)
    rx = x + 420 * u * u                        # the wind blows to the left
    r = torch.sqrt(rx * rx + (z + 40 * u) ** 2)
    rad = 40 + 130 * u + 220 * sstep(0.75, 1.05, u)
    top = sstep(1.15, 0.95, u)
    shape = (1 - r / rad) * top * (y > -10).float()
    n = fbm(p * 0.008 + vec(0, -0.15, 0), 5) + 0.5 * (fbm(p * 0.03 + 3, 3) - 0.5)
    return (shape + 1.0 * (n - 0.5)).clamp_min(0) * 0.1


def ash_glow(p, s):
    y = p[:, 1] - VY
    heat = torch.exp(-y.clamp_min(0) / 60.0)
    return srgb("#ff5a1a") * (heat * 2.5 * fbm(p * 0.04, 3) ** 2)[:, None]


def cumulus(p):
    y = p[:, 1]
    q = p * vec(0.0008, 0.0015, 0.0008)
    layer = sstep(1200, 1450, y) * sstep(2600, 1900, y)
    n = fbm(q + vec(2, 0, 7), 6)
    cover = fbm(p * 0.00012 + 4, 3)
    return ((n - 0.5 + 0.35 * (cover - 0.5)) * 5).clamp_min(0) * layer * 0.008


def lake_mist(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    near = torch.exp(-((x / 90) ** 2 + ((z + 70) / 70) ** 2))
    n = fbm(p * vec(0.03, 0.12, 0.03), 4)
    return torch.exp(-(y + 2.6).clamp_min(0) / 3.0) * near * (0.4 + n) * 0.02


NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    cam = Camera((0, 13.0, 18), (8, 12.0, -200), fov=60)
    el, az = math.radians(6.5), math.radians(-34)
    sun = (math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el))
    atmo = Atmosphere(sun, sun_power=24.0, mie=70e-6, mie_g=0.78, ground="#3a4a26")
    vols = [Volume(ash_plume, (VX - 900, VY - 20, VZ - 500), (VX + 300, VY + 620, VZ + 400), albedo="#6a625c", g=0.45,
                   emission=ash_glow, steps=64, light_steps=6, light_dist=300),
            Volume(cumulus, (-20000, 1200, -30000), (20000, 2600, 500), albedo="#ffffff", g=0.65, steps=40, light_steps=5, light_dist=500),
            Volume(lake_mist, (-200, -4, -200), (200, 14, 40), albedo="#f0e8e0", g=0.6, steps=32, light_steps=2, light_dist=10)]
    img = render_view(scene(), W, H, cam, None, sun_dir=sun, atmo=atmo, aerial=(0.0005, 0.004, 0.0), sun_soft=0.04,
                      exposure=1.15, steps=300, tmax=4000, ao_scale=0.8, bloom=0.9, dof=(30.0, 0.15), vignette=0.32, grain=0.012,
                      grade=("#3a3a6a", "#ffe0b0"), ss=ss, volumes=[] if NOVOL else vols)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "dinosaures.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
