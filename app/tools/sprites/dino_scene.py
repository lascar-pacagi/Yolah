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
        moss = torch.lerp(srgb("#2a4418"), srgb("#5a6a2a"), fbm(pw * 0.25 + 3, 3)[:, None]) * (0.6 + 0.5 * fbm(pw * 0.6, 3))[:, None]
        litter = sstep(0.55, 0.65, fbm(pw * 1.7 + 8, 2))
        moss = torch.lerp(moss, srgb("#5a4228"), (0.5 * litter)[:, None])
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


def fern_mat():
    """Fronds: pinnae (stripes across the frond), lighter young tips, some fronds browned."""
    def pat(pl, pw, n):
        k = 0.5 + 0.5 * torch.sin(pw[:, 0] * 23 + pw[:, 2] * 19 + 2 * fbm(pw * 2, 2))
        c = torch.lerp(srgb("#2a4a1a"), srgb("#6a9a34"), (0.4 * k + 0.6 * fbm(pw * 0.8, 2))[:, None])
        brown = sstep(0.62, 0.7, fbm(pw * 0.35 + 5, 3))
        c = torch.lerp(c, srgb("#7a5a2a"), (0.7 * brown)[:, None])
        return {"albedo": c}
    return Mat("#3a6224", rough=0.6, sss=0.7, pattern=pat)


def bark(c="#4a3a2a"):
    return Mat(c, rough=0.9, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.6 + 0.6 * (0.5 + 0.5 * torch.sin(pw[:, 1] * 9 + 3 * fbm(pw * 2, 2))))[:, None]})


def frond_mat(c="#3e6a26"):
    return Mat(c, rough=0.6, sss=0.7, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.6 + 0.6 * fbm(pw * 3, 2))[:, None]})


def fronds(top, n, L, up, rnd, pinnae=22, width=0.11, droop=0.9):
    """A rosette of pinnate fronds springing from `top`: each arches out and droops, its leaflets
    shown by a serrated outline; one SDF for the whole crown."""
    spec = []
    for i in range(n):
        a = math.tau * i / n + rnd.uniform(-0.25, 0.25)
        spec.append((math.cos(a), math.sin(a), L * rnd.uniform(0.8, 1.1), up * rnd.uniform(0.7, 1.3), rnd.uniform(-0.15, 0.15)))
    tx, ty, tz = top

    def f(p):
        qx, qy, qz = p[:, 0] - tx, p[:, 1] - ty, p[:, 2] - tz
        d = torch.full_like(qx, 1e9)
        for (ca, sa, Li, ui, tw) in spec:
            u = qx * ca + qz * sa
            v = -qx * sa + qz * ca
            t = (u / Li).clamp(0, 1)
            yr = Li * (ui * t * 2.2 - droop * t * t)
            s_ = pinnae * t - torch.floor(pinnae * t)
            teeth = 0.55 + 0.45 * (2 * s_ - 1).abs()                    # the leaflets: a serrated edge
            w = width * Li * torch.sin(math.pi * t.clamp(0.02, 0.999)).clamp_min(0) ** 0.6 * teeth
            ydev = qy - yr + 0.5 * v * v / Li - tw * v                     # leaflets droop either side of the rachis
            blade = torch.maximum(torch.maximum(v.abs() - w, ydev.abs() - 0.012 * Li), torch.maximum(-u, u - Li))
            d = torch.minimum(d, blade)
        return d * 0.5
    return f


def frond_mat2(c1="#2e5a1e", c2="#7aa83a"):
    def pat(pl, pw, n):
        k = fbm(pw * 0.6, 3)
        a = torch.lerp(srgb(c1), srgb(c2), (0.5 * k + 0.5 * n[:, 1].clamp(0, 1))[:, None]) * (0.7 + 0.4 * fbm(pw * 6, 2))[:, None]
        brown = sstep(0.68, 0.75, fbm(pw * 0.3 + 5, 3))
        return {"albedo": torch.lerp(a, srgb("#8a6a32"), (0.8 * brown)[:, None])}
    return Mat(c1, rough=0.6, sss=0.8, pattern=pat)


def tree_fern(sc, x, y, z, h, rnd, fm, bm):
    lean = rnd.uniform(-0.15, 0.15)
    top = (x + lean * h, y + h, z)
    k = h / 8
    # the trunk: shaggy with the bases of old fronds
    sc.add(lambda q: sd_capsule((x, y - 1, z), top, 0.32 * k, 0.26 * k)(q) - 0.06 * k * ridged(q * vec(5 / k, 2 / k, 5 / k), 2), bm)
    sc.add(fronds(top, 14, h * 0.62, 0.3, rnd, 24), fm)


def cycad(sc, x, y, z, h, rnd, fm, bm):
    """A cycad: a stout scaly trunk and a crown of stiff, straight fronds."""
    top = (x, y + h, z)
    sc.add(lambda q: sd_capsule((x, y - 0.5, z), top, 0.28 * h, 0.22 * h)(q) - 0.04 * h * ridged(q * (8 / h), 2), bm)
    sc.add(fronds(top, 16, h * 1.3, 0.55, rnd, 30, 0.09, 0.4), fm)


def horsetails(cx, cz, r, n, rnd):
    """A clump of horsetails: thin jointed stems, whorls of needles at the joints (one SDF)."""
    stems = [(cx + rnd.uniform(-r, r), cz + rnd.uniform(-r, r), rnd.uniform(1.0, 2.2), rnd.uniform(-0.08, 0.08)) for _ in range(n)]

    def f(p):
        d = torch.full_like(p[:, 0], 1e9)
        for (x, z, h, lean) in stems:
            g = TH.get((x, z))
            q = p - vec(x, g, z)
            qx = q[:, 0] - lean * q[:, 1]
            rr = torch.sqrt(qx * qx + q[:, 2] ** 2)
            seg = q[:, 1] * 5 - torch.floor(q[:, 1] * 5)
            stem = torch.maximum(rr - 0.025 - 0.012 * sstep(0.85, 1.0, seg), torch.maximum(q[:, 1] - h, -q[:, 1]))
            whorl = torch.maximum(torch.maximum(rr - 0.22 * (1 - q[:, 1] / h).clamp(0.2, 1), (seg - 0.92).abs() - 0.03),
                                  torch.maximum(q[:, 1] - h * 0.9, 0.2 - q[:, 1]))
            d = torch.minimum(d, torch.minimum(stem, whorl * 0.7))
        return d
    return f


class _TH(dict):
    def get(self, k):
        if k not in self:
            self[k] = terrain_h(*k)
        return self[k]


TH = _TH()


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


def forest_canopy(x, z):
    """Heights of the crowns over the hills: umbrellas of araucarias, conifer spires, round crowns (2×2 nearest cells)."""
    cell = 9.0
    c = torch.zeros_like(x)
    kind = torch.zeros_like(x)
    i0, k0 = torch.floor(x / cell - 0.5), torch.floor(z / cell - 0.5)
    for di in (0, 1):
        for dk in (0, 1):
            i, k = i0 + di, k0 + dk
            idv = torch.stack([i, k, i * 0 + 7], 1)
            cx = (i + 0.5 + (hash3(idv) - 0.5) * 0.6) * cell
            cz = (k + 0.5 + (hash3(idv + 1.3) - 0.5) * 0.6) * cell
            t = hash3(idv + 2.2)
            rad = cell * (0.35 + 0.25 * hash3(idv + 3.3))
            H = 8 + 14 * hash3(idv + 4.4)
            u = torch.sqrt((x - cx) ** 2 + (z - cz) ** 2) / rad
            umb = H * sstep(1.0, 0.75, u) * (0.85 + 0.15 * (1 - u).clamp_min(0))          # flat-topped umbrella
            con = H * 1.3 * (1 - u).clamp_min(0) ** 1.2                                   # spire
            rnd_ = H * 0.8 * (1 - u * u).clamp_min(0) ** 0.6                               # round
            h = torch.where(t < 0.35, umb, torch.where(t < 0.7, con, rnd_)) * (hash3(idv + 6.6) < 0.85).float()
            win = h > c
            c = torch.where(win, h, c)
            kind = torch.where(win, t, kind)
    return c * (0.85 + 0.3 * fbm(torch.stack([x * 0.4, z * 0.4, x * 0 + 2], 1), 2)), kind


def forest(p):
    """The forest on the hills, beyond the lake: a canopy of crowns over the ground."""
    x, z = p[:, 0], p[:, 2]
    q = torch.stack([x * 0.012, x * 0, z * 0.012], 1)
    h = 26 * (fbm(q, 5) - 0.35) + 2.5 * fbm(p * 0.15, 3)
    lake = torch.exp(-((x / 60) ** 2 + ((z + 70) / 45) ** 2))
    h = h * (1 - 0.9 * lake) - 3.0 * lake
    mask = (1 - lake * 1.4).clamp(0, 1) * ((-z - 60) / 60).clamp(0, 1) * sstep(0.4, 0.5, fbm(q * 3, 2))
    c, _ = forest_canopy(x, z)
    return (p[:, 1] - h - c * mask + 0.5) * 0.35


def forest_mat():
    def pat(pl, pw, n):
        _, kind = forest_canopy(pw[:, 0], pw[:, 2])
        a = torch.where((kind < 0.35)[:, None], srgb("#2a4220").expand_as(pw), torch.where((kind < 0.7)[:, None], srgb("#1e3220").expand_as(pw), srgb("#46602a").expand_as(pw)))
        return {"albedo": a * (0.6 + 0.6 * fbm(pw * 0.8, 3))[:, None] * (0.75 + 0.35 * n[:, 1].clamp(0, 1))[:, None]}
    return Mat("#2e4a1e", rough=0.8, sss=0.6, pattern=pat)


def scatter(cell, rmax, jit, region, make, seed, keep, top=2.5, ext=1.3, rot=0.0, clumps=0.0):
    """One object per cell (domain repetition), standing on the terrain; see bomb_scene.scatter.
    rot turns the grid (two scatters with different angles hide each other's rows); clumps makes
    the density vary (thickets and clearings) instead of a uniform keep."""
    margin = cell / 2 - jit - rmax * ext          # the widest reach of an object: rmax · ext
    assert margin > 0.1 * cell, margin
    x0, x1, z0, z1 = region

    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        out = torch.maximum(torch.maximum(x0 - x, x - x1), torch.maximum(z0 - z, z - z1))
        if rot:
            c_, s_ = math.cos(rot), math.sin(rot)
            x, z = c_ * x - s_ * z, s_ * x + c_ * z
        ix, iz = torch.floor(x / cell), torch.floor(z / cell)
        id3 = torch.stack([ix, ix * 0 + seed, iz], 1)
        h1, h2, h3 = hash3(id3), hash3(id3 + 3.7), hash3(id3 + 11.3)
        cx = (ix + 0.5) * cell + (h1 - 0.5) * 2 * jit
        cz = (iz + 0.5) * cell + (h2 - 0.5) * 2 * jit
        if rot:      # the cell's centre back in the world, for the terrain's height
            wx, wz = c_ * cx + s_ * cz, -s_ * cx + c_ * cz
        else:
            wx, wz = cx, cz
        g = -terrain(torch.stack([wx, wx * 0, wz], 1)) / 0.55
        r = rmax * (0.5 + 0.5 * h3)
        d = make(torch.stack([x - cx, y - g, z - cz], 1), r, h3)
        edge = cell / 2 - torch.maximum((x - (ix + 0.5) * cell).abs(), (z - (iz + 0.5) * cell).abs())
        bound = torch.maximum(edge + margin, y - g - top * rmax - 1.0)
        kp = keep + clumps * (fbm(torch.stack([wx * 0.05, wz * 0.05, wx * 0 + seed], 1), 3) - 0.5) * 2
        d = torch.where(hash3(id3 + 5.1) < kp, torch.minimum(d, bound), bound)
        return torch.maximum(d, out)
    return f


def fern_clump(loc, r, h):
    """A rosette of fronds arching out from the centre and drooping at their tips."""
    x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
    R = r * 1.35
    rho = torch.sqrt(x * x + z * z) + 1e-5
    n = 7.0
    a = torch.atan2(z, x) * (n / math.tau) + h * 13.0
    da = (a - torch.round(a)) * (math.tau / n)           # the angle to the nearest frond's axis
    tilt = 0.85 + 0.5 * hash3(torch.stack([torch.round(a), h * 31, h * 0], 1))
    u = (rho / R).clamp(0, 1)
    yf = R * tilt * (0.75 * u - 0.7 * u * u)             # the frond's arch
    lat = rho * torch.sin(da)
    w = 0.16 * R * (1 - u) ** 0.7 * torch.sqrt(u + 0.02) * 2.2
    d = torch.maximum(torch.maximum(lat.abs() - w, (y - yf).abs() - 0.03 * R), rho - R)
    return d * 0.6


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
    sc.add(forest, forest_mat())
    sc.add(sd_box(80, 0.1, 60), lake_mat(), T(0, -2.1, -70))
    sc.add(volcano, volcano_mat(), T(40, -10, -620))
    # ferns covering the ground near the eye, stones, a forest of araucarias on the far hills
    ferns = fern_mat()
    sc.add(scatter(5.0, 1.15, 0.35, (-70.0, 70.0, -70.0, 12.0), fern_clump, 1.0, 0.95, top=1.0, ext=1.4, rot=0.47, clumps=0.4), ferns)
    sc.add(scatter(6.6, 1.5, 0.5, (-80.0, 80.0, -80.0, 12.0), fern_clump, 4.0, 0.85, top=1.0, ext=1.4, rot=-0.93, clumps=0.5), ferns)
    sc.add(scatter(2.6, 0.55, 0.2, (-50.0, 50.0, -45.0, 12.0), fern_clump, 7.0, 0.8, top=1.0, ext=1.4, rot=1.31, clumps=0.5), ferns)
    sc.add(scatter(1.8, 0.38, 0.12, (-40.0, 40.0, -40.0, 12.0), fern_clump, 9.0, 0.7, top=1.0, ext=1.4, rot=2.1, clumps=0.4), ferns)
    sc.add(scatter(5.0, 0.6, 1.2, (-80.0, 80.0, -90.0, 10.0), rock, 2.0, 0.35), Mat("#5a5048", rough=0.85, pattern=lambda pl, pw, n: {
        "albedo": torch.lerp(srgb("#3a322c"), srgb("#7a6e60"), fbm(pw * 1.5, 3)[:, None])}))
    sc.add(scatter(20.0, 6.0, 2.0, (-600.0, 600.0, -480.0, -150.0), far_tree, 3.0, 0.55, top=4.2, ext=0.95), frond_mat("#2a4a20"))
    rnd = random.Random(11)
    fm, bm = frond_mat2(), bark()
    spots = [(-22, -18, 9), (-30, -26, 11), (24, -20, 8), (31, -30, 12), (-14, -40, 7), (18, -44, 8), (-44, -60, 13), (46, -58, 12)]
    spots += [(rnd.uniform(-90, 90), rnd.uniform(-160, -30), rnd.uniform(7, 13)) for _ in range(14)]
    spots = [s_ for s_ in spots if abs(s_[0]) > 35 or s_[1] > -30 or s_[1] < -120]
    for (x, z, h) in spots:
        tree_fern(sc, x, terrain_h(x, z), z, h, rnd, fm, bm)
    for (x, z, h) in ((-12, -8, 1.4), (15, -6, 1.2), (-38, -40, 1.8), (40, -36, 1.6), (-6, -30, 1.1)):
        cycad(sc, x, terrain_h(x, z), z, h, rnd, frond_mat2("#3a5a1e", "#8aa840"), bark("#5a4a32"))
    for (x, z, r, n) in ((-30, -48, 2.5, 14), (24, -52, 2.0, 12), (-8, -46, 1.5, 9)):
        sc.add(horsetails(x, z, r, n, rnd), Mat("#5a7a2a", rough=0.6, pattern=lambda pl, pw, n: {"albedo": torch.lerp(srgb("#3a5a1e"), srgb("#8aa042"), fbm(pw * 3, 2)[:, None])}))
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
