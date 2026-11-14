"""
manga_scene.py — the Manga background in 3D (scenery.py): a celestial realm
above the clouds, in the bright style of anime skies (the sanctuary itself
is drawn by the app under the board: assets/backgrounds/manga_sanctuaire.png).

  the islands   floating islands: rock tapering downwards (strata, roots,
                hanging vines), a lawn on top, trees with round crowns,
                waterfalls pouring off their edges into the void, their spray
  the clouds    a sea of cumulus below (volumes lit by the high sun, bright
                tops, blue shadows), towering clouds on the horizon
  the sky       a deep clear blue (physical sky, little haze), a pale ringed
                planet risen in daylight, the sun with its glare

    cd app/tools/sprites && python3 manga_scene.py [W H out [ss]]
"""
import math
import os
import random
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, norm, srgb, sd_sphere, sd_ellipsoid, sd_capsule, sd_cone, sd_box, fbm, hash3
from scenery import Camera, render_view, Atmosphere, Volume

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
_EL, _AZ = math.radians(42), math.radians(115)
SUN = (math.sin(_AZ) * math.cos(_EL), math.sin(_EL), -math.cos(_AZ) * math.cos(_EL))


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def ridged(p, o=4):
    return 1 - (2 * fbm(p, o) - 1).abs()


ISLANDS = [  # x, y (top), z, radius
    (-210.0, 40.0, -420.0, 70.0), (160.0, 95.0, -620.0, 55.0), (-60.0, 150.0, -900.0, 40.0),
    (330.0, 20.0, -380.0, 45.0), (-420.0, 120.0, -760.0, 50.0), (60.0, -10.0, -260.0, 26.0),
    (520.0, 160.0, -1100.0, 60.0), (-300.0, 230.0, -1300.0, 35.0),
    (250.0, -40.0, -300.0, 22.0), (-120.0, 70.0, -560.0, 18.0), (-560.0, 30.0, -520.0, 48.0), (40.0, 210.0, -1500.0, 50.0),
    (380.0, 260.0, -1700.0, 30.0), (-140.0, -60.0, -330.0, 12.0)]


def island_sdf(R):
    def f(q):
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        r0 = torch.sqrt(x * x + z * z) + 1e-6
        cu, cv = x / r0, z / r0
        # an irregular outline: bays and headlands
        r = r0 / (1 + 0.3 * (fbm(torch.stack([cu * 1.4 + R, cv * 1.4, cu * 0], 1), 3) - 0.5))
        # the top: a rolling lawn with a knoll; below: rock tapering in hanging lobes, lumpy
        top = y - (0.06 * R * (1 - (r / R) ** 2).clamp_min(0) + 0.08 * R * fbm(q * (1.6 / R) + 2, 3) * (1 - r / R).clamp_min(0)
                   + 0.02 * R * fbm(q * (5 / R), 2))
        depth = 1.6 * R * (0.8 + 0.5 * fbm(torch.stack([x * (1.5 / R), x * 0 + 1, z * (1.5 / R)], 1), 3))
        cone = r - R * (1 + y / depth).clamp(0, 1) ** 0.6 - 0.14 * R * ridged(q * vec(2.5 / R, 1.0 / R, 2.5 / R), 4)
        # ledges: the harder layers stand out
        lay = (y + 0.12 * x) * (5 / R) + 1.5 * fbm(q * (1.5 / R), 3)
        cone = cone - 0.02 * R * sstep(0.0, 0.25, lay - torch.floor(lay))
        d = torch.maximum(cone * 0.6, top)
        return torch.maximum(d, -y - depth * 1.2) * 0.8
    return f


def island_mat(R):
    def pat(pl, pw, n):
        y = pl[:, 1]
        layer = (y + 0.12 * pl[:, 0]) * (5 / R) + 1.5 * fbm(pl * (1.5 / R), 3)
        k = layer - torch.floor(layer)
        band = torch.floor(layer)
        pal = torch.stack([srgb(c) for c in ("#c8a882", "#b8946e", "#d8bc96", "#a88a6a", "#ccac86", "#bc9c7a")])
        rock = pal[(hash3(torch.stack([band, band * 0 + R, band * 0], 1)) * 6).long().clamp(0, 5)]
        rock = rock * (0.75 + 0.4 * fbm(pl * (5 / R), 4))[:, None] * (0.9 + 0.1 * sstep(0.0, 0.15, k))[:, None]
        crack = sstep(0.75, 0.9, ridged(torch.stack([pl[:, 0] * (6 / R), y * (1.5 / R), pl[:, 2] * (6 / R)], 1), 3))
        rock = rock * (1 - 0.45 * crack)[:, None]
        L = vec(*SUN)
        lit = sstep(-0.05, 0.3, (n * L).sum(-1))
        grass = torch.lerp(torch.lerp(srgb("#2e6a3a"), srgb("#4a8a3a"), fbm(pl * (3 / R), 2)[:, None]),
                           torch.lerp(srgb("#7ab840"), srgb("#c0e060"), fbm(pl * (2 / R) + 3, 3)[:, None]), lit[:, None])
        grass = grass * (0.85 + 0.25 * fbm(pl * (20 / R), 2))[:, None]
        fl = (hash3(torch.floor(pl * (90 / R))) > 0.985).float()
        grass = torch.lerp(grass, torch.where((hash3(torch.floor(pl * (90 / R)) + 3) > 0.5)[:, None], srgb("#fff8e0"), srgb("#ffd040")), fl[:, None])
        roots = sstep(0.7, 0.85, ridged(torch.stack([pl[:, 0] * (10 / R), y * (2 / R), pl[:, 2] * (10 / R)], 1), 2))
        soil = torch.lerp(srgb("#5a3e2a"), srgb("#2a1e16"), roots[:, None])
        up = sstep(0.55, 0.8, n[:, 1])[:, None]
        edge = sstep(-0.08 * R, -0.01 * R, y)[:, None]
        a = torch.lerp(rock, soil, edge * 0.6)
        a = torch.lerp(a, grass, up * edge)
        # vines hanging from the rim
        vine = ((fbm(torch.stack([pl[:, 0] * (8 / R), y * 0.02, pl[:, 2] * (8 / R)], 1), 2) - 0.62) * 6).clamp(0, 1) * sstep(-0.5 * R, -0.05 * R, y)
        a = torch.lerp(a, srgb("#3a6a26"), vine[:, None])
        return {"albedo": a}
    return Mat("#8a7a6a", rough=0.85, sss=0.25, pattern=pat)


def smin_(a, b, k):
    h = (0.5 + 0.5 * (b - a) / k).clamp(0, 1)
    return torch.lerp(b, a, h) - k * h * (1 - h)


def crowns(R, seed):
    """Round tree crowns over an island's lawn (domain repetition, the 3×3 neighbouring cells
    searched so that no artificial bound is needed: far islands stay clean)."""
    cell = R * 0.28
    hmax = 0.08 * R + cell * 1.4

    def tree(q, ix, iz):
        x, z = q[:, 0], q[:, 2]
        id3 = torch.stack([ix, ix * 0 + seed, iz], 1)
        cx = (ix + 0.5) * cell + (hash3(id3) - 0.5) * cell * 0.2
        cz = (iz + 0.5) * cell + (hash3(id3 + 3.1) - 0.5) * cell * 0.2
        keep = (hash3(id3 + 7.7) < 0.62 + 0.3 * (fbm(torch.stack([cx * (2 / R), cz * (2 / R), cx * 0 + seed], 1), 2) - 0.5)) \
            & (torch.sqrt(cx * cx + cz * cz) < R * 0.72)
        rr = cell * (0.15 + 0.12 * hash3(id3 + 5.5))
        conifer = (hash3(id3 + 2.2) < 0.2).float()
        sy = 1.0 + 1.3 * conifer                          # conifers: tall narrow crowns
        rr = rr * (1 - 0.25 * conifer)
        base = 0.06 * R * (1 - (cx * cx + cz * cz) / (R * R)).clamp_min(0) + 0.02 * R
        yq = q[:, 1] - base
        loc = torch.stack([x - cx, (yq - rr * (1.6 + 1.2 * conifer)) / sy, z - cz], 1)
        lump = loc / rr[:, None]
        crown = (torch.sqrt((loc * loc).sum(-1)) - rr) - 0.12 * rr * fbm(lump * 2.5 + id3 * 0.37, 2)
        # round anime trees: a few smaller puffs around the main one
        for j, (ox, oy, oz) in enumerate(((0.6, -0.35, 0.3), (-0.55, -0.3, -0.35), (0.1, 0.45, -0.55), (-0.3, 0.5, 0.45),
                                          (0.55, 0.2, -0.4), (-0.6, -0.1, 0.5))):
            jx = (hash3(idv3 := id3 + 1.7 * (j + 1)) - 0.5) * 0.4
            pf = torch.sqrt((lump[:, 0] - ox - jx) ** 2 + (lump[:, 1] - oy) ** 2 + (lump[:, 2] - oz + jx) ** 2) * rr - rr * (0.5 + 0.2 * hash3(idv3 + 0.3))
            crown = smin_(crown, pf, rr * 0.15)
        trunk = torch.maximum(torch.sqrt((x - cx) ** 2 + (z - cz) ** 2) - rr * 0.12, torch.maximum(yq - rr * 1.5, -yq - 0.1 * R))
        d = torch.minimum(crown, trunk) * 0.6
        return torch.where(keep, d, torch.full_like(d, 1e9))

    def f(q):
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        slab = torch.maximum(torch.maximum(-y - 2, y - hmax), torch.sqrt(x * x + z * z) - R)
        # the 2×2 cells nearest to the point (a tree reaches at most ~0.48 cell from its cell's centre)
        fx, fz = x / cell, z / cell
        ix0, iz0 = torch.floor(fx - 0.5), torch.floor(fz - 0.5)
        d = torch.full_like(x, 1e9)
        for dx in (0, 1):
            for dz in (0, 1):
                d = torch.minimum(d, tree(q, ix0 + dx, iz0 + dz))
        # dense (no indexing: the distance is replayed from a CUDA graph)
        return torch.maximum(torch.where(slab < cell * 0.5, d.clamp_max(cell * 0.5), slab), slab)
    return f


def leaves():
    L = vec(*SUN)

    def pat(pl, pw, n):
        nl = (n * L).sum(-1) + 0.25 * (fbm(pw * 0.6, 2) - 0.5)
        lit = sstep(-0.05, 0.25, nl)
        shade = torch.lerp(srgb("#1e4a3a"), srgb("#2e6a3a"), fbm(pw * 0.1, 2)[:, None])
        light = torch.lerp(srgb("#6aa83a"), srgb("#b8e060"), fbm(pw * 0.12 + 4, 2)[:, None])
        a = torch.lerp(shade, light, lit[:, None])
        # leaf clumps: small darker hollows, a few brighter leaves catching the light
        a = a * (0.75 + 0.35 * fbm(pw * 1.5, 3))[:, None]
        return {"albedo": a}
    return Mat("#3e8a2a", rough=0.7, sss=0.6, pattern=pat)


FALLS = []      # (x0, y0, z0, ang, length, width): filled by scene()


def fall_path(u, ang):
    return 0.06 * u + 0.0012 * u * u


def falls_density(fx):
    x0, y0, z0, ang, length, width = fx
    ca, sa = math.cos(ang), math.sin(ang)

    def dens(p):
        u = (y0 - p[:, 1]).clamp_min(0)
        out = fall_path(u, ang)
        lx, lz = p[:, 0] - x0 - ca * out, p[:, 2] - z0 - sa * out
        across = -sa * lx + ca * lz
        along = ca * lx + sa * lz
        w = width * (1 + u / length * 1.2)
        th = 0.6 + u * 0.12
        streak = fbm(torch.stack([across * (2.5 / width), p[:, 1] * 0.025, along * 0.2], 1), 4)
        core = torch.exp(-(across / w) ** 8) * torch.exp(-(along / th) ** 2) * (0.3 + 1.4 * streak)
        # the fall breaks into mist: its sheet fades, a cloud of spray grows around it
        k = (u / length).clamp(0, 1.4)
        mist = torch.exp(-(across / (w * 1.6)) ** 2 - (along / (th * 6 + 6)) ** 2) * fbm(p * 0.05 + 3, 4) * k
        dens = core * (1 - k).clamp_min(0) * 0.9 + mist * 0.12
        return dens * (p[:, 1] < y0 + 0.5).float()
    return dens


def fall_box(fx):
    x0, y0, z0, ang, length, width = fx
    ex = fall_path(length * 1.3, ang)
    xs = [x0, x0 + math.cos(ang) * ex]
    zs = [z0, z0 + math.sin(ang) * ex]
    m = width * 3.5 + 12
    return (min(xs) - m, y0 - length * 1.35, min(zs) - m), (max(xs) + m, y0 + 1, max(zs) + m)


def waterfall(x, y, z, R, ang, length):
    """A sheet of water leaving the rim and falling, fanning out (a thin ribbon)."""
    ex, ez = x + math.cos(ang) * R * 0.95, z + math.sin(ang) * R * 0.95

    def f(q):
        u = (-q[:, 1]).clamp(0, length)
        out = 0.06 * u + 0.0012 * u * u                 # the arc away from the rim
        lx = q[:, 0] - math.cos(ang) * out
        lz = q[:, 2] - math.sin(ang) * out
        across = (-math.sin(ang) * lx + math.cos(ang) * lz).abs() - (1.4 + 0.04 * u) * (1 - u / length * 0.7)
        thick = (math.cos(ang) * lx + math.sin(ang) * lz).abs() - 0.25
        return torch.maximum(torch.maximum(across, thick), torch.maximum(q[:, 1], -q[:, 1] - length)) * 0.8

    def pat(pl, pw, n):
        streak = fbm(torch.stack([pl[:, 0] * 0.8, pl[:, 1] * 0.06, pl[:, 2] * 0.8], 1), 3)
        a = torch.lerp(srgb("#9ad8f0"), srgb("#ffffff"), streak[:, None])
        return {"albedo": a, "emissive": srgb("#ffffff") * (0.15 * streak)[:, None]}
    return f, Mat("#ffffff", rough=0.3, sss=0.8, pattern=pat), T(ex, y - 1.0, ez)


def scene():
    sc = Scene()
    rnd = random.Random(4)
    for k, (x, y, z, R) in enumerate(ISLANDS):
        sc.add(island_sdf(R), island_mat(R), T(x, y, z))
        sc.add(crowns(R, k + 1.0), leaves(), T(x, y, z))
        if R > 40:
            ang = rnd.uniform(1.2, 2.0)
            f, m, X = waterfall(x, y, z, R, ang, R * 0.25)
            sc.add(f, m, X)
            ex, ez = x + math.cos(ang) * R * 0.95, z + math.sin(ang) * R * 0.95
            FALLS.append((ex, y - 1.0, ez, ang, R * 1.4, 0.05 * R))
    # a red torii on the near large island, looking out over the void
    tx, ty, tz = ISLANDS[0][0] + 30, ISLANDS[0][1] + 2.5, ISLANDS[0][2] + 42
    lac = Mat("#c8281e", rough=0.4, clearcoat=0.4)
    blk = Mat("#1a1414", rough=0.5)
    for sx_ in (-3.2, 3.2):
        sc.add(sd_capsule((tx + sx_, ty - 1, tz), (tx + sx_ * 0.95, ty + 8.2, tz), 0.42), lac)
    sc.add(sd_box(4.6, 0.32, 0.42, 0.05), lac, T(tx, ty + 6.6, tz))
    sc.add(lambda q: sd_box(5.6, 0.3, 0.6, 0.08)(q - vec(0, 1, 0) * (0.025 * q[:, 0:1] ** 2)) * 0.8, blk, T(tx, ty + 8.5, tz))
    return sc


def puffs(x, z, cell, seed, amp, keep=1.0):
    """A heightfield of rounded cloud heads on a jittered grid (the 2×2 nearest cells)."""
    acc = torch.zeros_like(x)
    i0, k0 = torch.floor(x / cell - 0.5), torch.floor(z / cell - 0.5)
    for di in (0, 1):
        for dk in (0, 1):
            i, k = i0 + di, k0 + dk
            idv = torch.stack([i, k, i * 0 + seed], 1)
            cx = (i + 0.5 + (hash3(idv) - 0.5) * 0.7) * cell
            cz = (k + 0.5 + (hash3(idv + 1.3) - 0.5) * 0.7) * cell
            rr = cell * (0.55 + 0.35 * hash3(idv + 2.1))
            u = ((x - cx) ** 2 + (z - cz) ** 2) / (rr * rr)
            on = (hash3(idv + 5.7) < keep).float()
            hi = on * amp * (0.4 + 0.6 * hash3(idv + 3.3)) * (1 - u).clamp_min(0) ** 0.75
            acc = torch.maximum(acc, hi)
    return acc


def cloud_top(x, z):
    big = fbm(torch.stack([x * 0.0006, z * 0.0006, x * 0 + 4], 1), 3)
    heads = puffs(x, z, 320.0, 1.0, 120, 0.7)
    on_head = sstep(10, 60, heads)                       # the smaller puffs grow on the heads, the valleys stay smooth
    return -330 + 60 * sstep(0.3, 0.7, big) + heads + on_head * (puffs(x, z, 110.0, 2.0, 40) + puffs(x, z, 38.0, 3.0, 12)) \
        + (1 - on_head) * puffs(x, z, 70.0, 4.0, 12, 0.6)


def cloud_sea(p):
    x, z = p[:, 0], p[:, 2]
    top = cloud_top(x, z) + 5 * fbm(p * 0.05, 3)
    return (p[:, 1] - top) * 0.3


def cloud_mat():
    L = vec(*SUN)

    def pat(pl, pw, n):
        nl = (n * L).sum(-1)
        # the anime look: white where the sun strikes, a soft lavender-blue in the shade, warm at the edge of light
        lit = sstep(-0.15, 0.35, nl)
        a = torch.lerp(srgb("#8a9cd0"), srgb("#ffffff"), lit[:, None])
        a = torch.lerp(a, srgb("#ffe8d0"), (sstep(-0.15, 0.05, nl) * sstep(0.3, 0.05, nl) * 0.6)[:, None])
        return {"albedo": a * (0.72 + 0.08 * fbm(pw * 0.05, 2))[:, None]}
    return Mat("#ffffff", rough=1.0, sss=0.5, pattern=pat)


def wisps(p):
    """Thin veils of mist lying over the sea of clouds, drifting between the islands."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    top = -250.0
    n = fbm(p * vec(0.004, 0.02, 0.004) + 3, 4)
    return torch.exp(-((y - top - 25) / 25) ** 2) * ((n - 0.5) * 3).clamp_min(0) * 0.004


def fluff(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    above = y - cloud_top(x, z)
    n = fbm(p * 0.04 + 2, 4)
    return torch.exp(-(above.clamp_min(0) / 6.0)) * (above > -4).float() * ((n - 0.35) * 2).clamp_min(0) * 0.012


def sea_of_clouds(p):
    y = p[:, 1]
    q = p * vec(0.0022, 0.004, 0.0022)
    # billows: cumulus heads rising out of the layer, flatter valleys between them
    heads = fbm(torch.stack([p[:, 0] * 0.004, p[:, 0] * 0 + 2, p[:, 2] * 0.004], 1), 3)
    top = -170 + 90 * sstep(0.4, 0.75, heads) + 60 * (fbm(p * 0.0025 + 5, 3) - 0.4)
    layer = sstep(top, top - 18, y) * sstep(-420, -300, y)
    n = fbm(q + vec(1, 0, 3), 4)
    return ((n - 0.44) * 6).clamp_min(0) * layer * 0.3


def towers(p):
    """Towering cumulus on the horizon."""
    y = p[:, 1]
    q = p * vec(0.0009, 0.0011, 0.0009)
    base = sstep(-200, 0, y) * sstep(1800, 900, y)
    col = fbm(torch.stack([p[:, 0] * 0.0004, p[:, 0] * 0, p[:, 2] * 0.0004], 1) + 2, 3)
    n = fbm(q + vec(4, 0, 2), 6)
    return ((n - 0.62 + (col - 0.5) * 0.8) * 4).clamp_min(0) * base * 0.01


def spray(p):
    """Mist at the foot of the waterfalls and around the islands' undersides."""
    d = torch.full((p.shape[0],), 0.0, device=p.device)
    for (x, y, z, R) in ISLANDS:
        if R > 40:
            r2 = ((p[:, 0] - x) ** 2 + (p[:, 2] - z) ** 2) / (R * 1.6) ** 2 + ((p[:, 1] - (y - R * 2.5)) / (R * 1.2)) ** 2
            d = d + torch.exp(-r2 * 2)
    return d * (fbm(p * 0.02, 3) + 0.2) * 0.02


def sky_planet(base_sky):
    pd = norm(vec(-0.42, 0.34, -0.84))
    u = norm(torch.linalg.cross(pd, vec(0, 1, 0)))
    v = torch.linalg.cross(u, pd)

    def sky(d):
        c = base_sky(d)
        x, y = (d @ u), (d @ v)
        r = torch.sqrt(x * x + y * y) / 0.11
        front = (d @ pd > 0.9).float()
        disc = (r < 1).float() * front
        bands = torch.sin(y / 0.11 * 9 + 2 * fbm(d * 30, 3)) * 0.5 + 0.5
        pc = torch.lerp(srgb("#e8c8b0"), srgb("#f8f0e8"), bands[:, None])
        lit = (0.55 + 0.45 * (x / 0.11 + 0.3).clamp(-1, 1))[:, None]
        # the ring: an ellipse around the disc
        rx, ry = x / 0.2, y / 0.05
        rr = torch.sqrt(rx * rx + ry * ry)
        ring = ((rr > 0.75) & (rr < 1.0)).float() * front * ((r >= 1) | (y < 0)).float() * (0.6 + 0.4 * torch.sin(rr * 60))
        k = 0.55                                         # seen through the blue air: pale
        c = c * (1 - disc[:, None] * k) + pc * (disc[:, None] * k) * lit * 2.2
        c = c + srgb("#f0e0d0") * (ring * 0.6)[:, None]
        return c
    return sky


NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    sun = SUN
    atmo = Atmosphere(sun, sun_power=22.0, mie=6e-6, mie_g=0.8, ground="#d8dce8", rayleigh=1.25)
    cam = Camera((0.0, 60.0, 0.0), (0, 40, -600), fov=62)
    sc = scene()
    sc.add(cloud_sea, cloud_mat())
    vols = [Volume(fluff, (-2500, -340, -4000), (2500, -100, 100), albedo="#ffffff", g=0.6, steps=128, light_steps=3, light_dist=40, ambient=1.0),
            Volume(towers, (-12000, -200, -14000), (12000, 1800, -2500), albedo="#ffffff", g=0.6, steps=40, light_steps=5, light_dist=700),
            Volume(spray, (-500, -250, -1200), (600, 120, -200), albedo="#ffffff", g=0.5, steps=40, light_steps=2, light_dist=40)]
    for fx in FALLS:
        lo, hi = fall_box(fx)
        vols.append(Volume(falls_density(fx), lo, hi, albedo="#f4faff", g=0.4, steps=64, light_steps=3, light_dist=20, ambient=1.2))
    img = render_view(sc, W, H, cam, sky_planet(atmo.sky), sun_dir=sun, atmo=atmo, aerial=(0.00008, 0.001, 0.0), sun_soft=0.03,
                      exposure=0.75, steps=480, tmax=16000, ao_scale=2.0, bloom=0.7, dof=None, vignette=0.2, grain=0.008,
                      grade=("#3a6ad0", "#fffaf2"), ss=ss, volumes=[] if NOVOL else vols)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "manga.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
