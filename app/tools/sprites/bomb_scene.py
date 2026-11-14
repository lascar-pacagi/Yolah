"""
bomb_scene.py — the Bombes' background in 3D (scenery.py): the firing range
of a bomb squad in the desert, at the golden hour, a few minutes after a
detonation.

  the land      a basin of desert pavement (sand, gravel, pebbles), dry washes,
                a dirt road with its ruts running to the mesas, craters with
                their scorched rims, creosote bushes and stones scattered
  the mesas     buttes and mesas of layered sandstone kilometres away: red,
                orange and cream strata, cliffs with ledges, talus slopes
  the range     a concrete bunker half buried in its berm (observation slit,
                steel door, sandbags), a steel watch tower, a fence of posts
                and barbed wire, warning signs, a line of telegraph poles,
                a burnt-out tank as a target, oil drums and tyres
  the air       a physical sky (Rayleigh, Mie, ozone) with the sun low, the
                mushroom of smoke and dust of the last blast lit by the sun,
                a smaller fire still burning, a layer of altocumulus, dust
                hanging over the ground

    cd app/tools/sprites && python3 bomb_scene.py [W H out [ss]]
"""
import math
import os
import random
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_capsule, sd_cone, sd_ellipsoid, sd_box, sd_cylinder, sd_torus, fbm, hash3, smin
from scenery import Camera, render_view, Atmosphere, Volume

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
D = vec(0).device


def fract(x):
    return x - torch.floor(x)


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# ─── the land ───────────────────────────────────────────────────────────────
CRATERS = [(-14.0, -48.0, 4.5), (9.0, -95.0, 7.0), (-38.0, -150.0, 9.0), (30.0, -40.0, 3.0), (55.0, -230.0, 11.0),
           (-70.0, -260.0, 8.0), (-5.0, -330.0, 12.0), (95.0, -150.0, 6.0), (-120.0, -120.0, 7.0)]


def road_x(z):
    return 4.0 * torch.sin(z / 90.0) + 0.012 * z


def land_h(x, z):
    q = torch.stack([x * 0.0016, x * 0, z * 0.0016], 1)
    h = 18 * (fbm(q, 5) - 0.45)                                          # the basin's swell
    h = h + 2.2 * (fbm(torch.stack([x * 0.012, x * 0, z * 0.012], 1), 4) - 0.5)
    h = h + 0.35 * (fbm(torch.stack([x * 0.09, x * 0, z * 0.09], 1), 3) - 0.5)
    # a dry wash meandering across
    wx = x - (60 * torch.sin(z / 140.0) - 90)
    h = h - 1.6 * torch.exp(-(wx / 9.0) ** 2)
    # the road: levelled, slightly sunk, two ruts
    r = x - road_x(z)
    near = torch.exp(-(r / 4.0) ** 2)
    corridor = torch.exp(-(r / 30.0) ** 2) * sstep(-300, -100, z)
    h = h + 0.05 * (fbm(torch.stack([x * 0.9, x * 0, z * 0.9], 1), 3) - 0.5) * (1 - near)
    h = h * (1 - 0.7 * corridor) * (1 - 0.6 * near) - 0.15 * near - 0.03 * (torch.exp(-((r - 0.9) / 0.25) ** 2) + torch.exp(-((r + 0.9) / 0.25) ** 2))
    for (cx, cz, R) in CRATERS:
        d = torch.sqrt((x - cx) ** 2 + (z - cz) ** 2) / R
        h = h - 0.42 * R * torch.exp(-d * d * 2.2) + 0.12 * R * torch.exp(-((d - 1.05) / 0.28) ** 2)
    return h


def land_at(x, z):
    return float(land_h(torch.tensor([float(x)], device=D), torch.tensor([float(z)], device=D))[0])


def land(p):
    return (p[:, 1] - land_h(p[:, 0], p[:, 2])) * 0.62


def ground_mat():
    def pat(pl, pw, n):
        x, z = pw[:, 0], pw[:, 2]
        sand = torch.lerp(srgb("#c49a72"), srgb("#a8764e"), fbm(pw * 0.02, 4)[:, None])
        sand = sand * (0.82 + 0.3 * fbm(pw * 0.35, 4))[:, None]
        # desert pavement: dark varnished pebbles packed on the sand
        cell = torch.floor(pw * 22.0)
        hsh = hash3(cell * vec(1, 0, 1))
        peb = ((hsh > 0.6).float() * (fbm(pw * 0.06 + 4, 3) > 0.42).float())[:, None]
        pcol = torch.lerp(srgb("#7a5a44"), srgb("#c8a684"), hash3(cell * vec(1, 0, 1) + 7)[:, None])
        a = sand * (1 - peb * 0.6) + pcol * peb * 0.6
        # gravel fans darker, the wash's floor paler and finer
        wx = x - (60 * torch.sin(z / 140.0) - 90)
        wash = torch.exp(-(wx / 7.0) ** 2)[:, None]
        a = torch.lerp(a, srgb("#d2b090") * (0.9 + 0.15 * fbm(pw * 0.5, 2))[:, None], wash * 0.8)
        # the road: compacted, pale, with darker ruts
        r = x - road_x(z)
        road = sstep(4.2, 2.6, r.abs())[:, None]
        rut = (torch.exp(-((r - 0.9) / 0.22) ** 2) + torch.exp(-((r + 0.9) / 0.22) ** 2))[:, None]
        a = torch.lerp(a, srgb("#c8a888") * (0.85 + 0.2 * fbm(pw * 1.5, 3))[:, None], road * 0.85)
        a = a * (1 - 0.12 * rut * road)
        # craters: scorched floors, ejecta of darker earth around
        for (cx, cz, R) in CRATERS:
            d = torch.sqrt((x - cx) ** 2 + (z - cz) ** 2) / R
            burn = (torch.exp(-d * d * 1.6) * (0.75 + 0.4 * fbm(pw * 0.8, 3)))[:, None].clamp(0, 1)
            ej = (torch.exp(-((d - 1.4) / 0.6) ** 2) * (fbm(pw * 1.2 + 3, 3) > 0.45).float())[:, None]
            a = torch.lerp(a, srgb("#2a2420") * (0.7 + 0.5 * fbm(pw * 3, 2))[:, None], burn * 0.85)
            a = torch.lerp(a, srgb("#7a5a44"), ej * 0.5)
        rough = 0.98 - 0.06 * peb[:, 0]
        return {"albedo": a, "rough": rough}
    return Mat("#c49a72", rough=0.92, pattern=pat)


# ─── the mesas ──────────────────────────────────────────────────────────────
def mesa_field(x, z):
    """> 0 inside the plan of the buttes (blobs of noise far away)."""
    w = torch.stack([x * 0.0021, x * 0 + 1.7, z * 0.0021], 1)
    wx = x + 260 * (fbm(w, 3) - 0.5)
    wz = z + 260 * (fbm(w + 5.2, 3) - 0.5)
    q = torch.stack([wx * 0.0013, x * 0 + 3.3, wz * 0.0013], 1)
    n = fbm(q, 3) + 0.1 * fbm(q * 7, 3)
    far = sstep(-900, -1500, z)
    return (n - 0.66) * far - 0.5 * (1 - far)


def mesas(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    f = mesa_field(x, z)
    e = 6.0
    gx = (mesa_field(x + e, z) - f) / e
    gz = (mesa_field(x, z + e) - f) / e
    g = torch.sqrt(gx * gx + gz * gz).clamp_min(2e-5)
    dist = -f / g                                       # metres to the cliff's foot line (in plan)
    far = (dist > 300).float()
    # cliffs: hard layers stand out as ledges, gullies cut down the faces
    layer = 2.4 * (fbm(torch.stack([y * 0.035, y * 0, y * 0 + 7], 1), 3) - 0.5)
    groove = 9.0 * (fbm(torch.stack([x * 0.02, y * 0.012, z * 0.02], 1), 4) - 0.5) + 1.5 * (fbm(torch.stack([x * 0.08, y * 0.05, z * 0.08], 1), 3) - 0.5)
    rough = (1 - far) * (layer + groove) * sstep(0, 40, y)
    top = 170 + 200 * fbm(torch.stack([x * 0.0004, x * 0, z * 0.0004], 1) + 9, 3)
    # benches: the upper cliff set back from the lower one (softer beds eroded between)
    bench = 0.62 + 0.12 * (fbm(torch.stack([x * 0.002, x * 0, z * 0.002], 1) + 3, 2) - 0.5)
    inset = (35 + 60 * fbm(torch.stack([x * 0.003, x * 0, z * 0.003], 1) + 8, 2)) * sstep(bench * top - 4, bench * top + 4, y)
    batter = 0.18 * (y - 0.5 * top)                     # the faces lean back a little
    # narrow buttes stay low: the summit rises from the rim inwards (no towers)
    top = torch.minimum(top, 70 + 2.2 * (-dist).clamp_min(0))
    cliff = torch.maximum(dist + rough + inset + batter, y - top)
    # the rim rounded a little, crumbling
    cliff = torch.maximum(cliff, (y - top + 6) * 0.7 + dist * 0.7 - 4)
    # talus: aprons of fallen rock around the feet, 45 % of the height, slope ~35°
    talus = y - (0.52 * top - 0.65 * dist.clamp_min(0))
    talus = talus - (1 - far) * (4 * fbm(p * 0.03, 3) + 1.5 * fbm(p * 0.15, 3))
    d = torch.minimum(cliff, talus * 0.8)
    return torch.maximum(d * 0.55, -y - 50)


def mesa_mat():
    def pat(pl, pw, n):
        x, z = pw[:, 0], pw[:, 2]
        y = pw[:, 1] + 5 * fbm(pw * 0.008, 3)
        top = 170 + 200 * fbm(torch.stack([x * 0.0004, x * 0, z * 0.0004], 1) + 9, 3)
        # the red sandstone of the cliffs, beds of slightly different tones
        bed = fbm(torch.stack([y * 0.05, y * 0, y * 0 + 3], 1), 2)
        a = torch.lerp(srgb("#7e3420"), srgb("#a8502c"), bed[:, None])
        a = a * (0.8 + 0.3 * fbm(pw * 0.02, 4))[:, None] * (0.9 + 0.12 * fbm(pw * 0.25, 3))[:, None]
        # the caprock: a paler, harder layer at the top; the dark red shale of the slopes below
        cap = sstep(top - 30, top - 18, y)[:, None]
        a = torch.lerp(a, srgb("#b88a68") * (0.8 + 0.25 * fbm(pw * 0.05, 3))[:, None], cap)
        shale = sstep(0.47 * top, 0.4 * top, y)[:, None]
        a = torch.lerp(a, srgb("#7a3a26") * (0.8 + 0.3 * fbm(pw * 0.08, 3))[:, None], shale)
        # desert varnish: dark streaks down the cliff faces
        steep = (1 - n[:, 1].abs()).clamp(0, 1)[:, None]
        streak = (fbm(torch.stack([x * 0.05, pw[:, 1] * 0.008, z * 0.05], 1), 3) - 0.52).clamp(0, 0.2)[:, None] * 4
        a = a * (1 - 0.3 * streak * steep)
        return {"albedo": a}
    return Mat("#b2522e", rough=0.9, pattern=pat)


# ─── scattered stones and bushes (one shape each, by cells) ──────────────────
def scatter(cell, rmax, jit, region, make, seed, keep):
    """Domain repetition on the ground: in each cell (size `cell`) one object
    (from make(local, r, h)) at a jittered spot, kept with probability keep.
    The bound keeps the march safe: a neighbour's object is at least
    `margin` inside its cell."""
    margin = cell / 2 - jit - rmax
    x0, x1, z0, z1 = region

    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        out = torch.maximum(torch.maximum(x0 - x, x - x1), torch.maximum(z0 - z, z - z1))
        ix, iz = torch.floor(x / cell), torch.floor(z / cell)
        id3 = torch.stack([ix, ix * 0 + seed, iz], 1)
        h1, h2, h3 = hash3(id3), hash3(id3 + 3.7), hash3(id3 + 11.3)
        cx = (ix + 0.5) * cell + (h1 - 0.5) * 2 * jit
        cz = (iz + 0.5) * cell + (h2 - 0.5) * 2 * jit
        ground = land_h(cx, cz)
        r = rmax * (0.35 + 0.65 * h3)
        loc = torch.stack([x - cx, y - ground, z - cz], 1)
        d = make(loc, r, h3)
        edge = cell / 2 - torch.maximum((x - (ix + 0.5) * cell).abs(), (z - (iz + 0.5) * cell).abs())
        bound = torch.maximum(edge + margin, y - ground - 2.5 * rmax - 1.5)
        d = torch.where(hash3(id3 + 5.1) < keep, torch.minimum(d, bound), bound)
        return torch.maximum(d, out)
    return f


def stone(loc, r, h):
    q = loc / torch.stack([r * 1.2, r * 0.6, r], 1)
    d = (torch.sqrt((q * q).sum(-1)) - 1) * r * 0.6
    return d - 0.18 * r * fbm(loc * (3 / r[:, None]) + h[:, None] * 9, 3)


def bush(loc, r, h):
    q = loc - torch.stack([r * 0, r * 0.6, r * 0], 1)
    q = q / torch.stack([r * 1.1, r * 0.8, r * 1.1], 1)
    d = (torch.sqrt((q * q).sum(-1)) - 1) * r * 0.75
    return d - 0.35 * r * fbm(loc * (2.5 / r[:, None]) + h[:, None] * 13, 3) + 0.12 * r


def stone_mat():
    return Mat("#6a4a38", rough=0.75, pattern=lambda pl, pw, n: {
        "albedo": torch.lerp(srgb("#7a5a44"), srgb("#b8906a"), fbm(pw * 2.5, 3)[:, None]) * (0.8 + 0.3 * fbm(pw * 9, 2))[:, None]})


def bush_mat():
    return Mat("#5a5a34", rough=0.85, sss=0.5, pattern=lambda pl, pw, n: {
        "albedo": torch.lerp(srgb("#5a5c38"), srgb("#a09a6a"), fbm(pw * 1.5, 3)[:, None]) * (0.55 + 0.6 * fbm(pw * 14, 2))[:, None]})


# ─── the range ──────────────────────────────────────────────────────────────
def concrete(c="#a49c90"):
    def pat(pl, pw, n):
        a = srgb(c) * (0.72 + 0.35 * fbm(pw * 0.7, 4))[:, None]
        # rain stains running down, form-tie holes, the joints of the pours
        stain = (fbm(torch.stack([pw[:, 0] * 1.5, pw[:, 1] * 0.15, pw[:, 2] * 1.5], 1), 3) - 0.4).clamp(0, 0.4)[:, None] * 1.4
        a = a * (1 - stain * 0.6)
        tie = ((fract(pw[:, 0] * 1.0 + pw[:, 2] * 1.0) - 0.5).abs() < 0.025).float() * ((fract(pw[:, 1] * 1.25) - 0.5).abs() < 0.03).float()
        a = a * (1 - 0.5 * tie[:, None])
        joint = (fract(pw[:, 1] / 1.2) < 0.012).float()[:, None]
        a = a * (1 - 0.25 * joint)
        # soot on the face towards the range
        return {"albedo": a}
    return Mat(c, rough=0.88, pattern=pat)


def rust(c="#6a3a22", k=1.0):
    def pat(pl, pw, n):
        base = torch.lerp(srgb("#3a2a22"), srgb(c), fbm(pw * 2.2, 4)[:, None])
        bright = (fbm(pw * 6 + 3, 3) > 0.6).float()[:, None]
        a = torch.lerp(base, srgb("#a8562a"), bright * 0.6) * (0.75 + 0.35 * fbm(pw * 20, 2))[:, None]
        burnt = (fbm(pw * 0.8 + 5, 3) > 0.55).float()[:, None] * k
        a = torch.lerp(a, srgb("#1e1a18"), burnt * 0.7)
        return {"albedo": a, "rough": 0.7 + 0.25 * fbm(pw * 5, 2)}
    return Mat(c, metal=0.35, rough=0.75, pattern=pat)


def paint(c, rough=0.55, wear=0.3):
    def pat(pl, pw, n):
        chip = (fbm(pw * 7, 3) > 1 - wear * 0.6).float()[:, None]
        a = srgb(c) * (0.8 + 0.25 * fbm(pw * 1.5, 3))[:, None]
        a = torch.lerp(a, srgb("#5a3a26"), chip * 0.8)
        return {"albedo": a}
    return Mat(c, rough=rough, metal=0.2, pattern=pat)


def bunker(sc, x, z, rot):
    b = land_at(x, z)
    X = T(x, b, z) * RY(rot)
    con = concrete()
    # the earth berm piled over the back and sides
    sc.add(lambda q: sd_ellipsoid(10.5, 5.0, 8.5)(q) - 0.8 * fbm(q * 0.4, 3), ground_mat(), X * T(0, -0.9, 3.4), blend=1.0)
    # the face: a thick wall, chamfered, with the slit and a door
    sc.add(sd_box(6.5, 1.9, 1.1, 0.12), con, X * T(0, 1.4, -3.2))
    sc.add(sd_box(7.0, 0.35, 4.6, 0.1), con, X * T(0, 3.35, -0.2))                      # the roof slab
    sc.add(sd_box(4.2, 0.22, 1.6), Mat("#0c0a0a", rough=0.9), X * T(-1.0, 2.25, -3.6), op="sub")    # the observation slit
    sc.add(sd_box(4.0, 0.03, 0.4), paint("#2a2a2a", wear=0.6), X * T(-1.0, 2.04, -4.25))
    sc.add(sd_box(0.75, 1.05, 0.08, 0.02), rust("#4a4a3a", 0.3), X * T(4.4, 1.05, -4.33))   # steel door
    for yy in (0.4, 1.7):
        sc.add(sd_box(0.08, 0.06, 0.05), rust("#3a3a32", 0.2), X * T(3.75, yy, -4.42))
    # warning paint: yellow and black chevrons along the roof edge
    def chev(pl, pw, n):
        s = fract((pl[:, 0] + pl[:, 1]) * 0.9)
        a = torch.where((s < 0.5)[:, None], srgb("#d8a822"), srgb("#1a1a1a")) * (0.7 + 0.4 * fbm(pw * 3, 2))[:, None]
        return {"albedo": torch.lerp(a, srgb("#8a8478"), (fbm(pw * 6, 3) > 0.62).float()[:, None])}
    sc.add(sd_box(7.02, 0.18, 0.05), Mat("#d8a822", rough=0.7, pattern=chev), X * T(0, 3.35, -4.82))
    # sandbags stacked in front of the door
    sb = Mat("#a88a62", rough=0.95, pattern=lambda pl, pw, n: {"albedo": srgb("#a4865e") * (0.7 + 0.35 * fbm(pw * 4, 3))[:, None]})
    for row in range(3):
        for k in range(5 - row):
            px = 2.2 + k * 0.62 + row * 0.31
            sc.add(sd_ellipsoid(0.32, 0.14, 0.2), sb, X * T(px, 0.14 + row * 0.25, -5.2) * RY(0.05 * ((k * 7 + row) % 3 - 1)), blend=0.03)
    # an antenna and a lamp on the roof
    sc.add(sd_capsule((-5.5, 3.6, 1.0), (-5.5, 8.5, 1.0), 0.03, 0.015), Mat("#3a3a3a", metal=0.8, rough=0.4), X)
    sc.add(sd_sphere(0.12), Mat("#ff3a1a", rough=0.3, emissive="#ff3018", emit=6.0), X * T(-5.5, 8.6, 1.0))


def tower(sc, x, z, rot):
    b = land_at(x, z)
    X = T(x, b, z) * RY(rot)
    steel = paint("#7a7468", rough=0.6, wear=0.5)
    H = 11.0
    for sx in (-1, 1):
        for sz in (-1, 1):
            sc.add(sd_capsule((sx * 2.2, -0.5, sz * 2.2), (sx * 1.4, H, sz * 1.4), 0.09), steel, X)
    # cross braces on each face, three bays
    for k in range(3):
        y0, y1 = k * H / 3, (k + 1) * H / 3
        w0, w1 = 2.2 - 0.8 * y0 / H, 2.2 - 0.8 * y1 / H
        for face in range(4):
            R = X * RY(face * math.pi / 2)
            sc.add(sd_capsule((-w0, y0, w0), (w1, y1, w1), 0.04), steel, R)
            sc.add(sd_capsule((w0, y0, w0), (-w1, y1, w1), 0.04), steel, R)
            sc.add(sd_capsule((-w1, y1, w1), (w1, y1, w1), 0.05), steel, R)
    # the cabin: walls up to the sill, windows all round, a roof with eaves
    wood = Mat("#6a5a48", rough=0.8, pattern=lambda pl, pw, n: {"albedo": srgb("#6e5c48") * (0.6 + 0.5 * (0.5 + 0.5 * torch.sin(pw[:, 1] * 40 + 4 * fbm(pw * 2, 2))))[:, None]})
    sc.add(sd_box(1.8, 0.1, 1.8), steel, X * T(0, H + 0.1, 0))
    sc.add(lambda q: torch.maximum(sd_box(1.6, 0.6, 1.6)(q), -sd_box(1.5, 0.7, 1.5)(q)), wood, X * T(0, H + 0.8, 0))
    glass = Mat("#1a2228", rough=0.05, metal=0.0, clearcoat=1.0)
    sc.add(sd_box(1.52, 0.5, 1.52), glass, X * T(0, H + 1.85, 0))
    for sx in (-1, 1):
        for sz in (-1, 1):
            sc.add(sd_box(0.06, 0.55, 0.06), wood, X * T(sx * 1.55, H + 1.85, sz * 1.55))
    sc.add(lambda q: sd_cone(0.7, 2.6, 0.2)(q), paint("#5a2a20", 0.7, 0.4), X * T(0, H + 3.05, 0))
    # the ladder
    for k in range(int(H / 0.4)):
        sc.add(sd_capsule((-0.3, k * 0.4, -2.3 + k * 0.02), (0.3, k * 0.4, -2.3 + k * 0.02), 0.025), steel, X)
    sc.add(sd_capsule((-0.3, 0, -2.3), (-0.3, H, -1.6), 0.03), steel, X)
    sc.add(sd_capsule((0.3, 0, -2.3), (0.3, H, -1.6), 0.03), steel, X)
    # a siren and a flag on the roof
    sc.add(sd_capsule((0, H + 3.6, 0), (0, H + 6.5, 0), 0.03), steel, X)
    def flag_f(q):
        w = q.clone()
        w[:, 2] = q[:, 2] - 0.12 * torch.sin(q[:, 0] * 3.5 + 1) * (q[:, 0] / 1.2).clamp(0, 1)
        return sd_box(0.6, 0.35, 0.015)(w - vec(0.6, 0, 0)) * 0.8
    sc.add(flag_f, Mat("#d02018", rough=0.8, sss=0.5, pattern=lambda pl, pw, n: {"albedo": srgb("#c81c14") * (0.75 + 0.3 * fbm(pw * 4, 2))[:, None]}),
           X * T(0.03, H + 6.1, 0) * RY(0.6))


def fence(sc, x0, x1, z, rnd):
    """Steel T-posts every 3 m (one shape for all) and three strands of barbed wire."""
    post = paint("#5a5248", 0.6, 0.6)
    step = 3.0

    def posts(q):
        x = q[:, 0]
        i = torch.round(x / step).clamp(x0 / step, x1 / step)
        lx = x - i * step + 0.06 * torch.sin(i * 3.1)
        g = land_h(i * step, q[:, 2] * 0 + z)
        loc = torch.stack([lx, q[:, 1] - g, q[:, 2] - z - 0.05 * torch.sin(i * 1.7)], 1)
        lean = 0.04 * torch.sin(i * 7.7)
        loc = torch.stack([loc[:, 0] - lean * loc[:, 1], loc[:, 1], loc[:, 2]], 1)
        return sd_box(0.035, 0.75, 0.035)(loc - vec(0, 0.65, 0))
    sc.add(posts, post)
    wire = Mat("#4a4440", metal=0.7, rough=0.5)

    def strands(q):
        x = q[:, 0].clamp(x0, x1)
        i = torch.floor(x / step)
        u = fract(x / step)
        g0 = land_h(i * step, q[:, 2] * 0 + z)
        g1 = land_h((i + 1) * step, q[:, 2] * 0 + z)
        g = g0 * (1 - u) + g1 * u
        sag = 0.06 * 4 * u * (1 - u)
        d = torch.full_like(x, 1e9)
        for hgt in (0.45, 0.85, 1.25):
            dy = q[:, 1] - (g + hgt - sag)
            dz = q[:, 2] - z
            dd = torch.sqrt(dy * dy + dz * dz + (q[:, 0] - x) ** 2) - 0.008
            # barbs: little knots every 12 cm
            near_cam = (q[:, 0].abs() < 30).float()
            kn = torch.sqrt(dy * dy + dz * dz + ((fract(q[:, 0] / 0.12) - 0.5) * 0.12) ** 2) - 0.022
            d = torch.minimum(d, torch.minimum(dd, torch.maximum(kn, dd - 0.02) + (1 - near_cam) * 1.0))
        return d
    sc.add(strands, wire, shadow=False)


def sign(sc, x, z, rot, text_col="#c81c14"):
    b = land_at(x, z)
    X = T(x, b, z) * RY(rot)
    wood = Mat("#5a4632", rough=0.85)
    sc.add(sd_box(0.05, 0.9, 0.05), wood, X * T(-0.35, 0.9, 0))
    sc.add(sd_box(0.05, 0.9, 0.05), wood, X * T(0.35, 0.9, 0))

    def pat(pl, pw, n):
        u, v = pl[:, 0], pl[:, 1]
        a = srgb("#e0b428").expand(u.shape[0], 3).clone()
        band = (v > 0.12).float()[:, None]
        a = torch.lerp(a, srgb(text_col), band)
        # a black bomb with its fuse on the yellow part (a circle and a stroke)
        r = torch.sqrt((u + 0.0) ** 2 + (v + 0.1) ** 2)
        a = torch.where(((r < 0.11) & (v < 0.1))[:, None], srgb("#141414"), a)
        a = torch.where((((u - 0.1).abs() < 0.015) & (v > -0.02) & (v < 0.08))[:, None], srgb("#141414"), a)
        # letters suggested: white dashes on the red band
        dash = ((fract(u * 7.5) < 0.6) & ((v - 0.22).abs() < 0.04) & (u.abs() < 0.4)).float()[:, None]
        a = torch.lerp(a, srgb("#f0ece0"), dash)
        a = a * (0.7 + 0.35 * fbm(pw * 6, 3))[:, None]
        hole = (hash3(torch.floor(pl * 30)) > 0.985).float()[:, None]                 # bullet holes
        return {"albedo": a * (1 - 0.8 * hole)}
    sc.add(sd_box(0.5, 0.34, 0.02), Mat("#e0b428", rough=0.6, metal=0.3, pattern=pat), X * T(0, 1.55, 0.06) * RZ(0.04))


def poles(sc, x0, z0, x1, z1, n):
    wood = Mat("#4a3a2c", rough=0.85, pattern=lambda pl, pw, n: {"albedo": srgb("#4e3e2e") * (0.6 + 0.5 * fbm(torch.stack([pw[:, 0] * 8, pw[:, 1] * 0.5, pw[:, 2] * 8], 1), 3))[:, None]})
    tops = []
    for k in range(n):
        u = k / (n - 1)
        x, z = x0 + (x1 - x0) * u, z0 + (z1 - z0) * u
        b = land_at(x, z)
        sc.add(sd_capsule((x, b - 0.5, z), (x, b + 8.5, z), 0.14, 0.11), wood)
        sc.add(sd_box(1.0, 0.07, 0.07), wood, T(x, b + 8.0, z) * RY(math.atan2(x1 - x0, z1 - z0)))
        tops.append((x, b + 8.0, z))
    wire = Mat("#2a2622", metal=0.6, rough=0.5)
    for (a, c) in zip(tops[:-1], tops[1:]):
        L = math.dist(a, c)
        for off in (-0.8, 0.8):
            ang = math.atan2(c[0] - a[0], c[2] - a[2])
            ox, oz = math.cos(ang) * off, -math.sin(ang) * off
            mid = ((a[0] + c[0]) / 2 + ox, (a[1] + c[1]) / 2 - L * 0.012, (a[2] + c[2]) / 2 + oz)
            sc.add(sd_capsule((a[0] + ox, a[1] + 0.1, a[2] + oz), mid, 0.012), wire, shadow=False)
            sc.add(sd_capsule(mid, (c[0] + ox, c[1] + 0.1, c[2] + oz), 0.012), wire, shadow=False)


def tank(sc, x, z, rot):
    b = land_at(x, z)
    X = T(x, b, z) * RY(rot) * RZ(0.06)
    r = rust("#6a4a2e")
    sc.add(lambda q: torch.maximum(sd_box(1.7, 0.55, 3.4, 0.1)(q), (q[:, 1] * 0.7 + q[:, 2].abs() * 0.4) - 1.6), r, X * T(0, 1.05, 0))
    for sx in (-1, 1):       # the tracks and road wheels
        sc.add(sd_box(0.32, 0.5, 3.3, 0.25), rust("#3a3028", 1.0), X * T(sx * 1.75, 0.5, 0))
        for k in range(6):
            sc.add(sd_cylinder(0.08, 0.32), rust("#4a3a2c"), X * T(sx * 2.08, 0.42, -2.5 + k) * RZ(math.pi / 2))
    sc.add(lambda q: sd_ellipsoid(1.25, 0.5, 1.6)(q) - 0.02 * fbm(q * 8, 2), r, X * T(0, 1.85, -0.3) * RY(0.7))
    sc.add(sd_capsule((0, 0, 0), (0, 0, 3.3), 0.1, 0.085), r, X * T(0, 1.95, -0.3) * RY(0.7 + 0.35) * RX(0.12))
    # a hole blown through the hull side, soot round it
    sc.add(sd_sphere(0.38), Mat("#0c0a0a", rough=1.0), X * T(1.7, 1.1, 0.6), op="sub")


def drums(sc, x, z):
    b = land_at(x, z)
    cols = ["#2a4a6a", "#8a2a1e", "#3a5a2a", "#a87a22"]
    rnd = random.Random(5)
    for k, (dx, dz, lie) in enumerate(((0, 0, 0), (0.65, 0.2, 0), (0.3, 0.7, 0), (1.4, -0.6, 1))):
        m = paint(cols[k % 4], 0.5, 0.65)
        if lie:
            sc.add(sd_cylinder(0.44, 0.3, 0.02), m, T(x + dx, b + 0.3, z + dz) * RY(0.8) * RZ(math.pi / 2))
        else:
            sc.add(sd_cylinder(0.44, 0.3, 0.02), m, T(x + dx, b + 0.44, z + dz))
            for hy in (-0.15, 0.15):
                sc.add(sd_torus(0.305, 0.012), m, T(x + dx, b + 0.44 + hy, z + dz))
    # a stack of tyres
    tyre = Mat("#141414", rough=0.85)
    for k in range(4):
        sc.add(sd_torus(0.36, 0.13), tyre, T(x - 1.6 + 0.04 * k, b + 0.13 + k * 0.25, z + 0.4))


# ─── the air: smoke, fire, dust and clouds ──────────────────────────────────
def mushroom(cx, cz, base, H, R, dens=0.02):
    """The smoke of a big detonation: a column rising into a rolling cap, a skirt of dust at the foot."""
    def f(p):
        x, y, z = p[:, 0] - cx, p[:, 1] - base, p[:, 2] - cz
        u = (y / H).clamp(0, 1.2)
        lean = 0.18 * H * u * u
        rx = x - lean
        r = torch.sqrt(rx * rx + z * z)
        stem_r = R * (0.18 + 0.12 * u)
        stem = (1 - r / stem_r) * (y < H * 0.85).float()
        cap_y = (y - H * 0.88) / (R * 0.38)
        cap = 1 - torch.sqrt((r / R) ** 2 + cap_y * cap_y)
        torus = 1 - torch.sqrt((torch.sqrt(rx * rx + z * z) - R * 0.55) ** 2 + (y - H * 0.84) ** 2) / (R * 0.32)
        skirt = (1 - torch.sqrt((r / (R * 1.6)) ** 2 + (y / (H * 0.12)) ** 2))
        shape = torch.maximum(torch.maximum(stem, cap), torch.maximum(torus, skirt * 0.9))
        q = p * (3.0 / R)
        n = fbm(q + vec(0, -0.2, 0), 5) + 0.5 * (fbm(q * 3.1 + 7, 3) - 0.5)
        d = (shape + 1.1 * (n - 0.5) - 0.05).clamp_min(0) * 3.5
        return d * dens
    lo = (cx - R * 1.8, base - 2, cz - R * 1.8)
    hi = (cx + R * 1.8 + 0.25 * H, base + H + R * 0.6, cz + R * 1.8)
    return f, lo, hi


def fire_plume(cx, cz, base, H, R):
    def f(p):
        x, y, z = p[:, 0] - cx, p[:, 1] - base, p[:, 2] - cz
        u = (y / H).clamp(0, 1)
        rx = x - 0.5 * H * u * u
        r = torch.sqrt(rx * rx + z * z)
        rr = R * (0.3 + 1.2 * u)
        shape = 1 - r / rr - (y < 0).float() * 4 - (y > H).float() * 4
        n = fbm(p * (2.5 / R) + vec(0, -0.4, 0), 5)
        return (shape + 0.8 * (n - 0.5)).clamp_min(0) * 0.08

    def em(p, s):
        y = p[:, 1] - base
        heat = torch.exp(-y / (R * 0.8)).clamp(0, 1)
        flick = fbm(p * (4 / R), 3)
        return srgb("#ff6a18") * (heat * heat * 40 * flick * flick)[:, None]
    return f, (cx - R * 1.6, base - 1, cz - R * 1.6), (cx + R * 1.6 + 0.5 * H, base + H, cz + R * 1.6), em


def clouds(p):
    y = p[:, 1]
    q = p * vec(0.00042, 0.0011, 0.00042)
    layer = sstep(2400, 2700, y) * sstep(3500, 3100, y)
    n = fbm(q + vec(5, 0, 2), 6)
    cover = fbm(p * 0.00008 + 2, 3)
    return ((n - 0.62 + 0.35 * (cover - 0.5)) * 6).clamp_min(0) * layer * 0.004


def dust(p):
    y = p[:, 1]
    n = fbm(p * vec(0.004, 0.02, 0.004), 4)
    return torch.exp(-(y.clamp_min(-5) + 5) / 18) * (0.6 + 0.8 * n) * 0.0007


# ─── the scene ──────────────────────────────────────────────────────────────
def scene():
    sc = Scene()
    sc.shadow_scene = Scene()       # the same without the thin wires (no stripes of shadow)
    _add = sc.add

    def add(sdf, mat, xf=None, blend=0.0, op="add", shadow=True):
        _add(sdf, mat, xf, blend, op)
        if shadow:
            sc.shadow_scene.add(sdf, mat, xf, blend, op)
        return sc
    sc.add = add
    sc.add(land, ground_mat())
    sc.add(mesas, mesa_mat())
    sc.add(scatter(2.4, 0.22, 0.75, (-90.0, 90.0, -140.0, 10.0), stone, 1.0, 0.25), stone_mat())
    sc.add(scatter(0.9, 0.07, 0.3, (-30.0, 30.0, -30.0, 10.0), stone, 3.0, 0.3), stone_mat())
    sc.add(scatter(6.0, 0.7, 1.8, (-200.0, 200.0, -420.0, 10.0), bush, 2.0, 0.42), bush_mat())
    rnd = random.Random(7)
    bunker(sc, -26.0, -38.0, 0.45 + 2.3)
    tower(sc, 34.0, -78.0, -0.3)
    fence(sc, -120.0, -9.0, -22.0, rnd)
    fence(sc, 9.0, 130.0, -22.0, rnd)
    sc.add(sd_box(0.08, 1.2, 0.08), paint("#d8d0c0", 0.6, 0.3), T(-8.0, land_at(-8.0, -22.0) + 0.6, -22.0))
    sc.add(sd_box(0.08, 1.2, 0.08), paint("#d8d0c0", 0.6, 0.3), T(8.0, land_at(8.0, -22.0) + 0.6, -22.0))
    sign(sc, -10.5, -21.4, 0.15)
    sign(sc, 18.0, -21.6, -0.2, "#1a1a1a")
    poles(sc, 70.0, -30.0, 220.0, -900.0, 9)
    tank(sc, 22.0, -175.0, 0.9)
    drums(sc, -14.0, -30.0)
    return sc


NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    el, az = math.radians(5.5), math.radians(-38)
    sun = (math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el))
    atmo = Atmosphere(sun, sun_power=24.0, mie=32e-6, mie_g=0.76, ground="#b08a66")
    cam = Camera((0.6, land_at(0.6, 6.0) + 3.4, 6.0), (6.0, 14.0, -1000.0), fov=52)
    vols = []
    f, lo, hi = mushroom(-520.0, -2100.0, -6.0, 700.0, 260.0)
    vols.append(Volume(f, lo, hi, albedo="#c8b8a8", g=0.55, steps=64, light_steps=6))
    f, lo, hi, em = fire_plume(240.0, -560.0, land_at(240.0, -560.0), 170.0, 30.0)
    vols.append(Volume(f, lo, hi, albedo="#6a625c", g=0.4, emission=em, steps=48, light_steps=5))
    vols.append(Volume(clouds, (-30000, 2400, -40000), (30000, 3500, 200), albedo="#ffffff", g=0.65, steps=40, light_steps=5, light_dist=900))
    vols.append(Volume(dust, (-3000, -20, -3000), (3000, 60, 20), albedo="#d8b898", g=0.6, steps=32, light_steps=1, light_dist=40))
    img = render_view(scene(), W, H, cam, None, sun_dir=sun, atmo=atmo, aerial=(0.00018, 0.0012, 0.0), sun_soft=0.03,
                      lights=[((-25.0, land_at(-26, -38) + 8.9, -38.6), "#ff3018", 1.5)],
                      exposure=1.15, steps=420, tmax=9000, ao_scale=0.35, bloom=0.7, dof=(40.0, 0.12), vignette=0.32,
                      grain=0.012, grade=("#3a4a6a", "#ffe2b8"), ss=ss, volumes=[] if NOVOL else vols, vol_scale=0.5)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "bombes.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
