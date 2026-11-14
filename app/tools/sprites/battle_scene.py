"""
battle_scene.py — the Généraux' background in 3D (scenery.py): a battlefield
of the Napoleonic wars at dawn, before the battle.

  the land      rolling fields of ripe wheat and pasture, hedgerows, rows of
                poplars, a walled farm with its barn and a tower on the ridge
  the guns      a battery in the foreground: two field guns on their carriages,
                a pyramid of cannonballs, a limber, a drum
  the camp      tents on the slope, flags on their poles
  the air       the low sun through the morning mist, smoke drifting from the
                bivouac fires, light shafts

    cd app/tools/sprites && python3 battle_scene.py [W H out]
"""
import math
import os
import random
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_capsule, sd_cone, sd_ellipsoid, sd_box, sd_cylinder, sd_torus, fbm, hash3, smin
from scenery import Camera, render_view, gradient_sky, Atmosphere, Volume

def fract(x):
    """x - floor(x): in [0, 1) also for negative x (torch.frac keeps the sign)."""
    return x - torch.floor(x)


ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


def land_h(x, z):
    q = torch.stack([x * 0.008, x * 0, z * 0.008], 1)
    h = 14 * (fbm(q, 4) - 0.4) + 1.2 * fbm(torch.stack([x * 0.08, x * 0, z * 0.08], 1), 3)
    # the battery's ridge in the foreground, the plain falling away below it
    ridge = torch.sigmoid((z + 34) / 7)
    h = h * (1 - 0.8 * ridge) + 16 * ridge - 6
    # the meadow's lumps and tussocks on the ridge (near the eye)
    near = torch.sigmoid((z + 40) / 4)
    return h + near * (0.25 * (fbm(torch.stack([x * 0.6, x * 0, z * 0.6], 1), 3) - 0.5) + 0.08 * (fbm(torch.stack([x * 3.0, x * 0 + 1, z * 3.0], 1), 2) - 0.5))


def land(p):
    return (p[:, 1] - land_h(p[:, 0], p[:, 2])) * 0.6


def fields():
    def pat(pl, pw, n):
        x, z = pw[:, 0], pw[:, 2]
        # a patchwork of fields: wheat (gold, furrows), pasture (green), ploughed (brown)
        cell = torch.floor(torch.stack([(x + 7 * fbm(pw * 0.01, 2)) / 60, x * 0, (z + 5 * fbm(pw * 0.01 + 3, 2)) / 45], 1))
        kind = (fbm(cell * 1.7, 1) * 3).floor().clamp(0, 2)
        furrow = 0.85 + 0.15 * torch.sin((x * 0.7 + z * 0.3) * 6)
        wheat = srgb("#c8a04a") * (0.75 + 0.35 * fbm(pw * 0.4, 3))[:, None] * furrow[:, None]
        grass = srgb("#5a7a32") * (0.7 + 0.4 * fbm(pw * 0.3, 3))[:, None]
        plough = srgb("#6a4a30") * (0.75 + 0.3 * fbm(pw * 0.5, 3))[:, None] * furrow[:, None]
        a = torch.where((kind == 0)[:, None], wheat, torch.where((kind == 1)[:, None], grass, plough))
        # the ridge: a meadow, trampled paths, wild flowers
        blades = fbm(torch.stack([pw[:, 0] * 22, pw[:, 1] * 2, pw[:, 2] * 22], 1), 2)
        meadow = torch.lerp(srgb("#2e4a1a"), srgb("#6e8a3a"), fbm(pw * 0.5, 4)[:, None]) * (0.6 + 0.5 * blades)[:, None] * (0.85 + 0.25 * fbm(pw * 3, 2))[:, None]
        dry = srgb("#9a8a52") * (0.7 + 0.4 * fbm(pw * 0.8, 3))[:, None]
        meadow = torch.lerp(meadow, dry, (fbm(pw * 0.15 + 5, 3) - 0.45).clamp(0, 0.3)[:, None] * 2.5)
        fl = (fbm(pw * 6 + 9, 2) > 0.82).float()[:, None]
        meadow = meadow * (1 - fl) + srgb("#d8c8e0") * fl * 0.8
        w = torch.sigmoid((z + 46) / 3)[:, None]
        return {"albedo": a * (1 - w) + meadow * w}
    return Mat("#c8a04a", rough=0.9, pattern=pat)


def hedges(p):
    """Hedgerows along the field borders: lumpy walls of leaves."""
    x, z = p[:, 0], p[:, 2]
    gx = (fract((x + 7 * fbm(p * 0.01, 2)) / 60) - 0.5).abs() * 60
    gz = (fract((z + 5 * fbm(p * 0.01 + 3, 2)) / 45) - 0.5).abs() * 45
    d2 = torch.minimum(30 - gx, 22.5 - gz)
    y = p[:, 1] - land_h(x, z)
    shape = torch.maximum(d2 - 0.7, (y - 1.0 - 0.6 * fbm(p * 0.3, 2)).abs() - 1.1)
    shape = torch.maximum(shape, z + 55)          # none on the ridge
    return shape * 0.5 - 0.25 * fbm(p * 0.6, 2) + 0.1


def leaves(c="#3a5228"):
    return Mat(c, rough=0.8, sss=0.6, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.55 + 0.6 * fbm(pw * 0.8, 3))[:, None]})


def stone(c="#b8ac94"):
    return Mat(c, rough=0.85, pattern=lambda pl, pw, n: {
        "albedo": srgb(c) * (0.7 + 0.35 * fbm(pw * 2, 3))[:, None] * (1 - 0.25 * ((fract(pw[:, 1] * 3) < 0.07) | (fract(pw[:, 0] * 1.5 + torch.floor(pw[:, 1] * 3) * 0.5) < 0.05)).float())[:, None]})


def tiles_roof(c="#8a3a2a"):
    return Mat(c, rough=0.75, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.65 + 0.4 * fbm(pw * 1.5, 3))[:, None] * (0.8 + 0.2 * (fract(pw[:, 1] * 6) < 0.5).float())[:, None]})


def farm(sc, cx, cz, rot):
    base = float(land_h(torch.tensor([cx], device=vec(0).device), torch.tensor([cz], device=vec(0).device))[0])
    st = stone()
    R = RY(rot)
    X = T(cx, base, cz) * R
    sc.add(sd_box(9, 4.5, 5), st, X * T(0, 4.5, 0))
    sc.add(lambda q: roof_f(q, 9.6, 5.6, 4.5), tiles_roof(), X * T(0, 9.0, 0))
    sc.add(sd_box(6, 3.5, 4), st, X * T(-14, 3.5, 6))
    sc.add(lambda q: roof_f(q, 4.6, 6.5, 3.5), tiles_roof("#6a3a2a"), X * T(-14, 7.0, 6) * RY(math.pi / 2))
    sc.add(sd_box(2.5, 9, 2.5), st, X * T(10, 9, -3))                      # the tower
    sc.add(sd_cone(3.0, 3.2, 0.1), Mat("#3a3e44", rough=0.6, metal=0.4), X * T(10, 21, -3))
    # the walls of the yard
    for a, b in (((-22, 0, -10), (16, 0, -10)), ((16, 0, -10), (16, 0, 18)), ((-22, 0, 18), (16, 0, 18)), ((-22, 0, -10), (-22, 0, 18))):
        mid = tuple((a[k] + b[k]) / 2 for k in range(3))
        hx, hz = abs(b[0] - a[0]) / 2 + 0.4, abs(b[2] - a[2]) / 2 + 0.4
        sc.add(sd_box(hx, 1.4, hz), st, X * T(mid[0], 1.0, mid[2]))
    # windows: dark recesses
    sc.add(lambda q: torch.maximum(torch.maximum(torch.maximum((fract(q[:, 0] / 3.0) - 0.5).abs() * 3.0 - 0.5, (q[:, 1] - 5.5).abs() - 0.9), (q[:, 2] - 5).abs() - 0.3), q[:, 0].abs() - 7.5),
           Mat("#1a1612", rough=0.5), X * T(0, 0, 0), op="sub")


def roof_f(q, w, d, h):
    a = math.atan2(h, w)
    s1 = q[:, 0].abs() * math.sin(a) + q[:, 1] * math.cos(a) - h * math.cos(a)
    return torch.maximum(torch.maximum(s1, -q[:, 1]), q[:, 2].abs() - d)


def poplars(sc, x0, z0, x1, z1, n, rnd):
    lm = leaves("#2e4422")
    for k in range(n):
        u = k / (n - 1)
        x, z = x0 + (x1 - x0) * u + rnd.uniform(-2, 2), z0 + (z1 - z0) * u
        b = float(land_h(torch.tensor([x], device=vec(0).device), torch.tensor([z], device=vec(0).device))[0])
        h = rnd.uniform(16, 22)
        sc.add(lambda q, h=h: sd_ellipsoid(2.0, h * 0.5, 2.0)(q) - 0.7 * fbm(q * 0.7, 2), lm, T(x, b + h * 0.55, z))
        sc.add(sd_capsule((x, b, z), (x, b + h * 0.3, z), 0.35), Mat("#3a2e24", rough=0.9))


def cannon(sc, x, z, rot, h):
    bronze = Mat("#8a6a3a", metal=1.0, rough=0.35, pattern=lambda pl, pw, n: {"albedo": srgb("#7a5e34") * (0.7 + 0.4 * fbm(pw * 3, 3))[:, None]})
    wood = Mat("#5a4430", rough=0.7, pattern=lambda pl, pw, n: {"albedo": srgb("#5a4430") * (0.7 + 0.4 * (0.5 + 0.5 * torch.sin(pw[:, 0] * 14 + 4 * fbm(pw * 2, 2))))[:, None]})
    iron = Mat("#2a2a2e", metal=0.9, rough=0.45)
    X = T(x, h, z) * RY(rot)
    sc.add(sd_capsule((0, 1.15, -0.6), (0, 1.25, 1.6), 0.24, 0.16), bronze, X)        # the barrel
    sc.add(sd_sphere(0.2), bronze, X * T(0, 1.13, -0.75))
    sc.add(sd_torus(0.19, 0.05), bronze, X * T(0, 1.25, 1.6) * RX(math.pi / 2))
    sc.add(sd_box(0.22, 0.2, 1.6, 0.04), wood, X * T(0, 0.75, -1.2) * RX(-0.35))       # the trail
    for sgn in (-1, 1):
        w = X * T(sgn * 0.62, 0.68, 0.0) * RZ(math.pi / 2)
        sc.add(sd_torus(0.6, 0.06), iron, w)
        sc.add(sd_cylinder(0.16, 0.12), wood, w)
        for k in range(7):
            a = math.tau * k / 7
            sc.add(sd_capsule((0, 0, 0), (0.58 * math.cos(a), 0, 0.58 * math.sin(a)), 0.035), wood, w)
    sc.add(sd_capsule((-0.62, 0.68, 0), (0.62, 0.68, 0), 0.06), iron, X)


def flag(sc, x, y, z, cols, rot=0.0):
    sc.add(sd_capsule((x, y, z), (x, y + 7, z), 0.06), Mat("#4a3a2a", rough=0.6))
    def pat(pl, pw, n):
        u = (pl[:, 0] / 2.4).clamp(0, 0.999)
        k = (u * len(cols)).floor().long()
        c = torch.stack([srgb(c) for c in cols])[k]
        return {"albedo": c * (0.8 + 0.25 * fbm(pw * 2, 2))[:, None]}
    m = Mat(cols[0], rough=0.8, sss=0.5, pattern=pat)
    def cloth_f(q):
        w = q.clone()
        w[:, 2] = q[:, 2] - 0.18 * torch.sin(q[:, 0] * 2.4) * (q[:, 0] / 2.4).clamp(0, 1)
        return sd_box(1.2, 0.8, 0.03)(w - vec(1.2, 0, 0)) * 0.8
    sc.add(cloth_f, m, T(x + 0.06, y + 6.1, z) * RY(rot) * RZ(-0.05))


def smoke_mat():
    return Mat("#d8d4cc", rough=1.0, sss=1.0, pattern=lambda pl, pw, n: {"albedo": srgb("#cfc8bc") * (0.8 + 0.3 * fbm(pw * 0.15, 3))[:, None]})


def battalion(sc, cx, cz, rot, files, hb, coat_c="#1e3270", flag_cols=("#1e3a8a", "#f2f0ea", "#c8242a"), shako=True):
    """A battalion in line: `files` × 3 men by domain repetition (one shape for all)."""
    S, R = 0.75, 0.95                                   # spacing along the line, between ranks
    def men(q):
        i = torch.round(q[:, 0] / S).clamp(-(files // 2), files // 2 - 1)
        j = torch.round(q[:, 2] / R).clamp(-1, 1)
        l = torch.stack([q[:, 0] - i * S + 0.08 * torch.sin(i * 7.3 + j * 3.1), q[:, 1], q[:, 2] - j * R], 1)
        body = sd_capsule((0, 0.15, 0), (0, 1.35, 0), 0.2, 0.16)(l)
        head = sd_sphere(0.12)(l - vec(0, 1.6, 0))
        shako = sd_cylinder(0.13, 0.12)(l - vec(0, 1.8, 0))
        musket = sd_capsule((0.22, 0.6, 0.05), (0.24, 2.25, 0.08), 0.03)(l)
        return torch.minimum(torch.minimum(torch.minimum(body, head), shako), musket)
    def pat(pl, pw, n):
        y = pl[:, 1]
        coat, trou, black, skin = srgb(coat_c), srgb("#e6e0d0"), srgb("#141418"), srgb("#c89878")
        c = torch.where((y < 0.75)[:, None], trou, coat.expand(y.shape[0], 3))
        c = torch.where(((y > 1.12) & (y < 1.32))[:, None], torch.where(((pl[:, 0] / S) - torch.round(pl[:, 0] / S)).abs()[:, None] < 0.2, trou, coat), c)
        c = torch.where((y > 1.45)[:, None], skin.expand(y.shape[0], 3), c)
        c = torch.where((y > 1.67)[:, None], black.expand(y.shape[0], 3), c)
        return {"albedo": c}
    sc.add(men, Mat("#1e3270", rough=0.7, pattern=pat), T(cx, hb(cx, cz), cz) * RY(rot))
    flag(sc, cx + 0.4, hb(cx, cz) + 0.2, cz - 0.6, list(flag_cols), rot)


def horse_sdf(l, rider=True):
    """A horse (and its rider) standing, facing −z, in local coordinates (feet at y = 0)."""
    x, y, z = l[:, 0], l[:, 1], l[:, 2]
    body = sd_ellipsoid(0.32, 0.36, 0.85)(l - vec(0, 1.25, 0))
    neck = sd_capsule((0, 1.4, -0.65), (0, 1.95, -1.0), 0.17, 0.12)(l)
    head = sd_capsule((0, 1.98, -1.02), (0, 1.62, -1.35), 0.11, 0.07)(l)
    ears = sd_capsule((0, 2.05, -1.0), (0, 2.2, -0.98), 0.03)(l)
    d = smin(smin(body, neck, 0.12), smin(head, ears, 0.03), 0.06)
    for (lx, lz, ph) in ((-0.17, -0.6, 0.0), (0.17, -0.6, 0.15), (-0.17, 0.62, 0.1), (0.17, 0.62, -0.1)):
        leg = sd_capsule((lx, 1.1, lz), (lx, 0.45, lz + ph), 0.085, 0.055)(l)
        shin = sd_capsule((lx, 0.45, lz + ph), (lx, 0.04, lz + ph * 0.5), 0.05, 0.045)(l)
        d = smin(d, torch.minimum(leg, shin), 0.05)
    tail = sd_capsule((0, 1.4, 0.82), (0, 0.8, 1.0), 0.07, 0.03)(l)
    d = torch.minimum(d, tail)
    if rider:
        torso = sd_capsule((0, 1.75, 0.05), (0, 2.35, 0.0), 0.17, 0.14)(l)
        thighs = sd_capsule((-0.2, 1.62, -0.05), (0.2, 1.62, -0.05), 0.1)(l)
        legs = torch.minimum(sd_capsule((-0.26, 1.6, -0.1), (-0.3, 1.05, -0.05), 0.07)(l), sd_capsule((0.26, 1.6, -0.1), (0.3, 1.05, -0.05), 0.07)(l))
        hd = sd_sphere(0.11)(l - vec(0, 2.55, 0))
        helm = sd_ellipsoid(0.13, 0.16, 0.14)(l - vec(0, 2.66, 0.0))
        crest = sd_capsule((0, 2.8, -0.08), (0, 2.75, 0.22), 0.025)(l)
        d = torch.minimum(d, torch.minimum(torch.minimum(torso, thighs), torch.minimum(legs, torch.minimum(torch.minimum(hd, helm), crest))))
    return d


def horse_mat(coat="#4a2e1e", uniform="#1e3270", helm="#c8ccd2"):
    def pat(pl, pw, n):
        y = pl[:, 1]
        horse = srgb(coat) * (0.75 + 0.3 * fbm(pw * 4, 2))[:, None]
        legs_dark = (y < 0.5).float()[:, None]
        horse = horse * (1 - 0.45 * legs_dark)
        rid = (y > 1.55).float()[:, None] * (pl[:, 2].abs() < 0.4).float()[:, None]
        r = torch.where((y > 2.42)[:, None], srgb("#d8a888"), srgb(uniform).expand(y.shape[0], 3))
        r = torch.where((y > 2.6)[:, None], srgb(helm), r)
        r = torch.where(((y < 1.7) & (y > 1.0))[:, None], srgb("#e8e2d2"), r)      # breeches
        r = torch.where(((y < 1.3) & (y > 1.0))[:, None], srgb("#141414"), r)       # boots
        return {"albedo": torch.lerp(horse, r, rid)}
    return Mat(coat, rough=0.55, pattern=pat)


def squadron(sc, cx, cz, rot, files, hb):
    """A squadron of cuirassiers in two ranks (one shape, repeated)."""
    S, R = 1.15, 2.6
    def f(q):
        i = torch.round(q[:, 0] / S).clamp(-(files // 2), files // 2 - 1)
        j = torch.round(q[:, 2] / R).clamp(0, 1)
        l = torch.stack([q[:, 0] - i * S, q[:, 1], q[:, 2] - j * R + 0.15 * torch.sin(i * 5.1)], 1)
        return horse_sdf(l)
    sc.add(f, horse_mat("#3a2418", "#1e3270", "#d0d4da"), T(cx, hb(cx, cz), cz) * RY(rot))


def tuft(loc, r, h):
    """A tuft of grass: thin blades (noise stretched upwards) in a cone."""
    x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
    rr = torch.sqrt(x * x + z * z)
    hgt = r * 1.6
    cone = torch.maximum(rr - r * (1.0 - 0.6 * (y / hgt).clamp(0, 1)) * 0.9, torch.maximum(y - hgt, -y))
    blades = fbm(torch.stack([x * (14 / r), y * (1.2 / r), z * (14 / r)], 1) + h[:, None] * 7, 2)
    return cone * 0.6 + (blades - 0.45) * r * 0.5


def scatter(height, cell, rmax, jit, region, make, seed, keep):
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
        ground = height(cx, cz)
        r = rmax * (0.45 + 0.55 * h3)
        loc = torch.stack([x - cx, y - ground, z - cz], 1)
        d = make(loc, r, h3)
        edge = cell / 2 - torch.maximum((x - (ix + 0.5) * cell).abs(), (z - (iz + 0.5) * cell).abs())
        bound = torch.maximum(edge + margin, y - ground - 2.5 * rmax - 0.5)
        d = torch.where(hash3(id3 + 5.1) < keep, torch.minimum(d, bound), bound)
        return torch.maximum(d, out)
    return f


def village(sc, cx, cz, hb, rnd):
    """A village: houses of stone with tiled roofs round a church and its steeple."""
    st, wall2 = stone("#c8bca4"), stone("#b0a488")
    for k in range(14):
        a = rnd.uniform(0, math.tau)
        rr = rnd.uniform(8, 38)
        x, z = cx + math.cos(a) * rr, cz + math.sin(a) * rr * 0.7
        w, d_, h = rnd.uniform(3, 6), rnd.uniform(3, 4.5), rnd.uniform(2.6, 4.2)
        b = hb(x, z)
        R = T(x, b, z) * RY(rnd.uniform(-0.4, 0.4) + (0.0 if k % 2 else math.pi / 2))
        sc.add(sd_box(w, h, d_), st if k % 3 else wall2, R * T(0, h, 0))
        sc.add(lambda q, w=w, d_=d_: roof_f(q, w + 0.5, d_ + 0.4, 2.6), tiles_roof(["#8a3a2a", "#7a4030", "#9a4a32"][k % 3]), R * T(0, 2 * h, 0))
    b = hb(cx, cz)
    sc.add(sd_box(4.5, 6, 11), st, T(cx, b + 6, cz))
    sc.add(lambda q: roof_f(q, 5.2, 11.5, 4.0), tiles_roof("#5a4a44"), T(cx, b + 12, cz))
    sc.add(sd_box(3.2, 11, 3.2), st, T(cx, b + 11, cz - 13))
    sc.add(lambda q: sd_cone(4.5, 3.6, 0.05)(q), Mat("#3a4048", metal=0.4, rough=0.5), T(cx, b + 26.5, cz - 13))


def caisson(sc, x, z, rot, hb):
    b = hb(x, z)
    X = T(x, b, z) * RY(rot)
    wood = Mat("#3a4a32", rough=0.7, pattern=lambda pl, pw, n: {"albedo": srgb("#3e4e34") * (0.7 + 0.35 * fbm(pw * 3, 3))[:, None]})
    iron = Mat("#2a2a2e", metal=0.9, rough=0.45)
    sc.add(sd_box(0.6, 0.35, 0.9, 0.05), wood, X * T(0, 1.25, 0))
    sc.add(lambda q: roof_f(q, 0.65, 0.95, 0.18), wood, X * T(0, 1.6, 0))
    for sx in (-1, 1):
        w = X * T(sx * 0.75, 0.68, 0.0) * RZ(math.pi / 2)
        sc.add(sd_torus(0.62, 0.05), iron, w)
        sc.add(sd_cylinder(0.12, 0.1), wood, w)
        for k in range(8):
            a = math.tau * k / 8
            sc.add(sd_capsule((0, 0, 0), (0.6 * math.cos(a), 0, 0.6 * math.sin(a)), 0.03), wood, w)
    sc.add(sd_capsule((0, 0.75, -0.9), (0, 0.15, -3.0), 0.05), wood, X)


# ─── the bivouac ────────────────────────────────────────────────────────────
D0 = vec(0).device


_GMEMO = {}


def ground_at(x, z):
    """The land's height at (x, z), memoised: the SDFs below call it with constants (no GPU sync per call)."""
    k = (round(float(x), 3), round(float(z), 3))
    if k not in _GMEMO:
        _GMEMO[k] = float(land_h(torch.tensor([float(x)], device=D0), torch.tensor([float(z)], device=D0))[0])
    return _GMEMO[k]


TCX, TCZ = 5.0, 10.0                 # the tent grid: one tent per cell, streets between the rows
CAMP_REGIONS = [(-95.0, -12.0, -100.0, -36.0), (12.0, 95.0, -100.0, -36.0), (-60.0, -30.0, -32.0, -24.0)]


def _camp_cell(p, seed):
    x, z = p[:, 0], p[:, 2]
    ix, iz = torch.floor(x / TCX), torch.floor(z / TCZ)
    idv = torch.stack([ix, iz, ix * 0 + seed], 1)
    cx = (ix + 0.5) * TCX + (hash3(idv) - 0.5) * 0.6
    cz = (iz + 0.5) * TCZ
    inside = torch.zeros_like(x, dtype=torch.bool)
    for (x0, x1, z0, z1) in CAMP_REGIONS:
        inside = inside | ((cx > x0) & (cx < x1) & (cz > z0) & (cz < z1))
    keep = inside & (hash3(idv + 5.5) < 0.88)
    g = land_h(cx, cz)
    edge = torch.minimum(TCX / 2 - (x - (ix + 0.5) * TCX).abs(), TCZ / 2 - (z - (iz + 0.5) * TCZ).abs())
    # away from the camp the distance to its regions bounds the step (no false hits far off)
    away = torch.full_like(x, 1e9)
    for (x0, x1, z0, z1) in CAMP_REGIONS:
        away = torch.minimum(away, torch.maximum(torch.maximum(x0 - 3 - x, x - x1 - 3), torch.maximum(z0 - 6 - z, z - z1 - 6)))
    edge = torch.maximum(edge, away)
    return idv, cx, cz, g, keep, edge


def tents(p):
    """The soldiers' tents: ridge tents of canvas, their doors on the street, in rows on the slope and the plain."""
    idv, cx, cz, g, keep, edge = _camp_cell(p, 1.0)
    a = (hash3(idv + 2.2) - 0.5) * 0.25
    lx, lz = p[:, 0] - cx, p[:, 2] - cz
    lx, lz = torch.cos(a) * lx - torch.sin(a) * lz, torch.sin(a) * lx + torch.cos(a) * lz
    ly = p[:, 1] - g
    w, d, h = 1.05, 1.3, 1.35
    # the canvas sags between the poles, a little ridge rope at the top
    sag = 0.06 * (1 - (lz / d) ** 2) * (lx.abs() / w)
    roof = (lx.abs() * h + ly * w - h * w) / math.hypot(h, w) + sag
    t = torch.maximum(torch.maximum(roof, -ly), lz.abs() - d)
    pole = torch.maximum(torch.sqrt(lx * lx + (lz.abs() - d - 0.05) ** 2) - 0.04, torch.maximum(ly - h - 0.25, -ly))
    t = torch.minimum(t, pole)
    bound = torch.maximum(edge + 0.5, ly - 2.0)
    return torch.where(keep, torch.minimum(t * 0.8, bound), bound)


def canvas_mat():
    def pat(pl, pw, n):
        a = torch.lerp(srgb("#cfc4a8"), srgb("#ece4d0"), fbm(pw * 0.8, 3)[:, None]) * (0.85 + 0.2 * fbm(pw * 6, 2))[:, None]
        dirt = sstep(0.4, 0.9, 1 - (pw[:, 1] - land_h(pw[:, 0], pw[:, 2])).clamp(0, 1))
        a = a * (1 - 0.3 * dirt)[:, None]
        # the door: a dark slit on the street side
        idv = torch.stack([torch.floor(pw[:, 0] / TCX), torch.floor(pw[:, 2] / TCZ), pw[:, 0] * 0 + 1.0], 1)
        lx = pw[:, 0] - ((idv[:, 0] + 0.5) * TCX + (hash3(idv) - 0.5) * 0.6)
        door = (lx.abs() < 0.28).float() * (n[:, 2] > 0.6).float() * (pw[:, 1] - land_h(pw[:, 0], pw[:, 2]) < 1.0).float()
        return {"albedo": a * (1 - 0.85 * door)[:, None]}
    return Mat("#e2d8c2", rough=0.85, sss=0.5, pattern=pat)


def muskets(p):
    """Muskets piled in front of each tent: three leaning together, bayonets crossed."""
    idv, cx, cz, g, keep, edge = _camp_cell(p, 1.0)
    keep = keep & (hash3(idv + 8.8) < 0.8)
    lx, ly, lz = p[:, 0] - cx - 0.4, p[:, 1] - g, p[:, 2] - cz - 3.0
    q = torch.stack([lx, ly, lz], 1)
    d = torch.full_like(lx, 1e9)
    for k in range(3):
        ang = k * math.tau / 3 + 0.3
        b = vec(0.42 * math.cos(ang), 0.0, 0.42 * math.sin(ang))
        L = math.sqrt(0.42 ** 2 + 1.55 ** 2)
        u = vec(-0.42 * math.cos(ang) / L, 1.55 / L, -0.42 * math.sin(ang) / L)
        t = ((q - b) @ u).clamp(0, L * 1.22)
        rad = torch.where(t > L * 1.02, torch.full_like(t, 0.012), torch.where(t < 0.35, torch.full_like(t, 0.045), torch.full_like(t, 0.028)))
        d = torch.minimum(d, torch.linalg.norm(q - b - u * t[:, None], dim=1) - rad)
    bound = torch.maximum(edge + 0.5, ly - 2.2)
    return torch.where(keep, torch.minimum(d, bound), bound)


def campers(p):
    idv, cx, cz, g, keep, edge = _camp_cell(p, 1.0)
    keep = keep & (hash3(idv + 12.1) < 0.45)
    ox = (hash3(idv + 13.3) - 0.5) * 3.0
    oz = 2.2 + hash3(idv + 14.7) * 2.5
    l = torch.stack([p[:, 0] - cx - ox, p[:, 1] - g, p[:, 2] - cz - oz], 1)
    body = sd_capsule((0, 0.25, 0), (0, 1.3, 0), 0.24, 0.17)(l)
    legs = sd_capsule((0, 0.0, 0), (0, 0.4, 0), 0.12)(l)
    head = sd_sphere(0.12)(l - vec(0, 1.55, 0))
    hat = sd_cylinder(0.13, 0.12)(l - vec(0, 1.75, 0))
    d = torch.minimum(torch.minimum(body, legs), torch.minimum(head, hat))
    bound = torch.maximum(edge + 0.5, l[:, 1] - 2.2)
    return torch.where(keep, torch.minimum(d, bound), bound)


def camper_mat():
    def pat(pl, pw, n):
        idv = torch.stack([torch.floor(pw[:, 0] / TCX), torch.floor(pw[:, 2] / TCZ), pw[:, 0] * 0 + 1.0], 1)
        k = hash3(idv + 15.5)
        pal = torch.stack([srgb(c) for c in ("#1e3270", "#3e4a62", "#5a5a58", "#e6e0d0")])
        a = pal[(k * 4).long().clamp(0, 3)]
        h = pw[:, 1] - land_h(pw[:, 0], pw[:, 2])
        a = torch.where((h > 1.43)[:, None], srgb("#c8987a").expand_as(a), a)
        a = torch.where((h > 1.62)[:, None], srgb("#141416").expand_as(a), a)
        a = torch.where((h < 0.35)[:, None], srgb("#2a2420").expand_as(a), a)
        return {"albedo": a * (0.8 + 0.3 * fbm(pw * 5, 2))[:, None]}
    return Mat("#1e3270", rough=0.8, pattern=pat)


def musket_mat():
    def pat(pl, pw, n):
        h = pw[:, 1] - land_h(pw[:, 0], pw[:, 2])
        wood = srgb("#5a3a22") * (0.8 + 0.3 * fbm(pw * 8, 2))[:, None]
        steel = srgb("#b8bcc4")
        return {"albedo": torch.where((h > 1.15)[:, None], steel.expand_as(wood), wood), "rough": torch.where(h > 1.15, 0.3, 0.7)}
    return Mat("#5a3a22", rough=0.6, pattern=pat)


# the fires of the bivouac: (x, z, soldiers around it)
FIRES = [(-5.0, -10.0, 5), (9.0, -19.0, 4), (-21.0, -24.0, 5), (22.0, -9.0, 3), (1.0, -29.0, 4), (33.0, -25.0, 4),
         (-30.0, -50.0, 0), (35.0, -45.0, 0), (-55.0, -70.0, 0), (55.0, -66.0, 0), (-20.0, -88.0, 0), (28.0, -92.0, 0),
         (-75.0, -48.0, 0), (75.0, -80.0, 0)]


def fires(p):
    """Each fire: a ring of stones, logs, the embers; a tripod with its cauldron over it."""
    d = torch.full_like(p[:, 0], 1e9)
    for (fx, fz, _) in FIRES:
        g = ground_at(fx, fz)
        q = p - vec(fx, g, fz)
        r = torch.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2)
        ring = torch.sqrt((r - 0.75) ** 2 + (q[:, 1] - 0.08) ** 2) - 0.14 - 0.04 * fbm(q * 6, 2)
        embers = torch.maximum(r - 0.6, (q[:, 1] - 0.08).abs() - 0.12) - 0.06 * fbm(q * 5, 2)
        logs = torch.minimum(sd_capsule((-0.5, 0.15, -0.2), (0.45, 0.3, 0.15), 0.08)(q), sd_capsule((0.1, 0.15, -0.5), (-0.1, 0.32, 0.45), 0.08)(q))
        tri = torch.full_like(r, 1e9)
        for k in range(3):
            a = k * math.tau / 3
            tri = torch.minimum(tri, sd_capsule((0.9 * math.cos(a), 0, 0.9 * math.sin(a)), (0, 1.55, 0), 0.03)(q))
        pot = torch.maximum(sd_sphere(0.32)(q - vec(0, 0.85, 0)), q[:, 1] - 1.05)
        chain = sd_capsule((0, 1.05, 0), (0, 1.55, 0), 0.012)(q)
        d = torch.minimum(d, torch.minimum(torch.minimum(torch.minimum(ring, embers), logs), torch.minimum(torch.minimum(tri, pot), chain)))
    return d


def fire_mat():
    def pat(pl, pw, n):
        em = torch.zeros_like(pw)
        a = srgb("#3a3430").expand_as(pw).clone() * (0.6 + 0.6 * fbm(pw * 4, 2))[:, None]
        for (fx, fz, _) in FIRES:
            g = ground_at(fx, fz)
            q = pw - vec(fx, g, fz)
            r = torch.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2)
            hot = sstep(0.65, 0.2, r) * sstep(0.5, 0.1, q[:, 1])
            em = em + srgb("#ff6a1a") * (hot * 6.0 * (0.4 + fbm(pw * 8, 2) ** 2 * 2))[:, None]
        return {"albedo": a, "emissive": em}
    return Mat("#3a3430", rough=0.8, pattern=pat)


def soldiers_list():
    rnd = random.Random(17)
    out = []
    for (fx, fz, n) in FIRES:
        for k in range(n):
            a = k * math.tau / max(n, 1) + rnd.uniform(-0.3, 0.3)
            rr = rnd.uniform(1.3, 1.9)
            x, z = fx + math.cos(a) * rr, fz + math.sin(a) * rr
            pose = rnd.choice(("sit", "sit", "stand", "crouch"))
            out.append((x, z, a + math.pi, pose, rnd.choice(("#3e4a62", "#5a5a58", "#6a6252", "#1e3270")), rnd.random() < 0.6))
    for (x, z) in ((-12.0, -4.0), (16.0, -27.0), (-30.0, -31.0), (5.0, -36.0)):      # sentries and men walking
        out.append((x, z, rnd.uniform(0, math.tau), "stand", "#1e3270", True))
    return out


SOLDIERS = soldiers_list()


def soldiers(p):
    d = torch.full_like(p[:, 0], 1e9)
    for (x, z, a, pose, _, sh) in SOLDIERS:
        g = ground_at(x, z)
        q = p - vec(x, g, z)
        c, s_ = math.cos(a), math.sin(a)
        l = torch.stack([c * q[:, 0] + s_ * q[:, 2], q[:, 1], -s_ * q[:, 0] + c * q[:, 2]], 1)     # local x: facing the fire
        if pose == "stand":
            body = sd_capsule((0, 0.25, 0), (0, 1.3, 0), 0.24, 0.17)(l)        # the greatcoat, wider at the hem
            legs = sd_capsule((0, 0.0, 0), (0, 0.4, 0), 0.12)(l)
            hy = 1.55
        elif pose == "crouch":
            body = sd_capsule((0, 0.45, 0), (0.15, 1.0, 0), 0.24, 0.17)(l)
            legs = sd_capsule((0, 0.05, 0), (0.25, 0.45, 0), 0.12)(l)
            hy = 1.22
        else:
            body = sd_capsule((-0.05, 0.35, 0), (0.05, 0.95, 0), 0.22, 0.16)(l)
            legs = torch.minimum(sd_capsule((0.0, 0.32, -0.12), (0.45, 0.35, -0.12), 0.09)(l), sd_capsule((0.0, 0.32, 0.12), (0.45, 0.35, 0.12), 0.09)(l))
            legs = torch.minimum(legs, torch.minimum(sd_capsule((0.45, 0.35, -0.12), (0.5, 0.0, -0.12), 0.07)(l), sd_capsule((0.45, 0.35, 0.12), (0.5, 0.0, 0.12), 0.07)(l)))
            legs = torch.minimum(legs, sd_box(0.2, 0.15, 0.2)(l - vec(-0.05, 0.15, 0)))       # a pack he sits on
            hy = 1.18
        arms = sd_capsule((0, hy - 0.3, -0.2), (0.35, hy - 0.6, -0.12), 0.07)(l)
        arms = torch.minimum(arms, sd_capsule((0, hy - 0.3, 0.2), (0.35, hy - 0.6, 0.12), 0.07)(l))
        head = sd_sphere(0.12)(l - vec(0.02, hy, 0))
        hat = sd_cylinder(0.13, 0.12)(l - vec(0.02, hy + 0.2, 0)) if sh else sd_ellipsoid(0.17, 0.08, 0.13)(l - vec(0.02, hy + 0.1, 0))
        d = torch.minimum(d, smin(torch.minimum(torch.minimum(body, legs), arms), torch.minimum(head, hat), 0.04))
    return d


def soldier_mat():
    def pat(pl, pw, n):
        a = srgb("#3e4a62").expand_as(pw).clone()
        hgt = torch.zeros_like(pw[:, 0])
        best = torch.full_like(pw[:, 0], 1e9)
        for (x, z, ang, pose, coat, sh) in SOLDIERS:
            dd = (pw[:, 0] - x) ** 2 + (pw[:, 2] - z) ** 2
            m = dd < best
            best = torch.where(m, dd, best)
            a = torch.where(m[:, None], srgb(coat).expand_as(a), a)
            top = {"stand": 1.55, "crouch": 1.22, "sit": 1.18}[pose]
            hgt = torch.where(m, (pw[:, 1] - ground_at(x, z)) / top, hgt)
        a = a * (0.75 + 0.3 * fbm(pw * 6, 2))[:, None]
        a = torch.where((hgt > 0.93)[:, None], srgb("#c8987a").expand_as(a), a)            # faces
        a = torch.where((hgt > 1.06)[:, None], srgb("#141416").expand_as(a), a)            # shakos, bicornes
        a = torch.where((hgt < 0.22)[:, None], srgb("#2a2420").expand_as(a), a)            # gaiters, boots
        return {"albedo": a}
    return Mat("#3e4a62", rough=0.8, pattern=pat)


def marquee(sc, x, z, rot):
    """The general's marquee: a large wall tent with a striped valance, its flap open."""
    g = ground_at(x, z)
    X = T(x, g, z) * RY(rot)
    def walls(q):
        box = sd_box(3.2, 1.2, 2.4, 0.05)(q - vec(0, 1.2, 0))
        door = sd_box(0.7, 1.0, 0.5)(q - vec(0, 1.0, 2.4))
        return torch.maximum(box, -door)
    def roof(q):
        return roof_f(q - vec(0, 2.35, 0), 3.5, 2.6, 1.6) - 0.02 * fbm(q * 3, 2)
    def pat(pl, pw, n):
        a = srgb("#ece4d2") * (0.85 + 0.2 * fbm(pw * 2, 2))[:, None]
        stripe = ((pl[:, 0] * 1.6 - torch.floor(pl[:, 0] * 1.6)) < 0.5).float() * (pl[:, 1] > 2.05).float() * (pl[:, 1] < 2.45).float()
        return {"albedo": torch.lerp(a, srgb("#1e3a8a"), stripe[:, None])}
    m = Mat("#ece4d2", rough=0.85, sss=0.5, pattern=pat)
    sc.add(walls, m, X)
    sc.add(roof, m, X)
    sc.add(sd_box(0.8, 0.4, 0.5), Mat("#4a3020", rough=0.6), X * T(1.0, 0.4, 3.6))          # a camp table outside, a chest
    sc.add(sd_box(0.5, 0.3, 0.35), Mat("#2a3a5a", rough=0.6), X * T(-1.6, 0.3, 3.2))


def tree(cx, cz, h, rnd, kind="oak"):
    """A tree: a trunk, main branches, and a crown of leafy clusters (each lumpy, leaves at two scales)."""
    g = ground_at(cx, cz)
    clusters = []
    if kind == "poplar":
        for k in range(9):
            t = 0.25 + 0.75 * k / 8
            clusters.append((rnd.uniform(-0.4, 0.4), h * t, rnd.uniform(-0.4, 0.4), h * 0.11 * (1.2 - 0.6 * abs(t - 0.55) * 2)))
        trunk = (0.0, 0.0, 0.0, 0.0, h * 0.85, 0.0, h * 0.025)
    else:
        for k in range(14):
            a = rnd.uniform(0, math.tau)
            rr = rnd.uniform(0, h * 0.32)
            clusters.append((math.cos(a) * rr, h * rnd.uniform(0.5, 0.85) - rr * 0.3, math.sin(a) * rr, h * rnd.uniform(0.14, 0.22)))
        trunk = (0.0, 0.0, 0.0, 0.0, h * 0.55, 0.0, h * 0.04)
    branches = [(trunk[3] * 0, h * 0.4, 0.0, c[0] * 0.8, c[1] * 0.95, c[2] * 0.8) for c in clusters[::3]]

    def wood(p):
        q = p - vec(cx, g, cz)
        d = sd_capsule(trunk[:3], trunk[3:6], trunk[6], trunk[6] * 0.5)(q)
        for b in branches:
            d = torch.minimum(d, sd_capsule(b[:3], b[3:], trunk[6] * 0.45, trunk[6] * 0.15)(q))
        return d

    def crown(p):
        q = p - vec(cx, g, cz)
        d = torch.full_like(q[:, 0], 1e9)
        for (x, y, z, r) in clusters:
            d = smin(d, sd_ellipsoid(r, r * 0.8, r)(q - vec(x, y, z)), r * 0.3)
        d = d - h * 0.05 * fbm(q * (6.0 / h), 3) - h * 0.012 * fbm(q * (30.0 / h), 2)
        return d * 0.6
    return wood, crown


def foliage(c1="#26381a", c2="#6a7a32"):
    def pat(pl, pw, n):
        k = fbm(pw * 0.9, 3)
        a = torch.lerp(srgb(c1), srgb(c2), (0.55 * k + 0.45 * n[:, 1].clamp(0, 1))[:, None]) * (0.6 + 0.5 * fbm(pw * 4, 2))[:, None]
        return {"albedo": a}
    return Mat(c1, rough=0.8, sss=0.8, pattern=pat)


def wagon(sc, x, z, rot):
    g = ground_at(x, z)
    X = T(x, g, z) * RY(rot)
    wood = Mat("#4a3a28", rough=0.75, pattern=lambda pl, pw, n: {"albedo": srgb("#4a3a28") * (0.7 + 0.4 * fbm(pw * 4, 2))[:, None]})
    sc.add(sd_box(0.9, 0.35, 2.0, 0.04), wood, X * T(0, 1.15, 0))
    sc.add(lambda q: torch.maximum(torch.maximum((torch.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2) - 1.05).abs() - 0.03, -q[:, 1]), q[:, 2].abs() - 1.9),
           canvas_mat(), X * T(0, 1.45, 0))
    for sx in (-1, 1):
        for sz in (-1.3, 1.3):
            sc.add(sd_torus(0.55, 0.06), wood, X * T(sx * 1.0, 0.6, sz) * RZ(math.pi / 2))
    sc.add(sd_capsule((0, 0.8, -2.0), (0, 0.4, -4.0), 0.05), wood, X)


def horse_line(p):
    """Horses tethered along a picket rope, heads to the rope."""
    d = torch.full_like(p[:, 0], 1e9)
    for k in range(7):
        x, z = 38.0 + k * 1.6, -14.0 + 0.3 * math.sin(k * 2.1)
        g = ground_at(x, z)
        q = p - vec(x, g, z)
        a = 0.08 * math.sin(k * 3.7)
        c, s_ = math.cos(a), math.sin(a)
        l = torch.stack([c * q[:, 0] + s_ * q[:, 2], q[:, 1], -s_ * q[:, 0] + c * q[:, 2]], 1)
        d = torch.minimum(d, horse_sdf(l, rider=False))
    rope = sd_capsule((37.0, ground_at(37, -15.6) + 1.3, -15.6), (49.0, ground_at(49, -15.6) + 1.3, -15.6), 0.02)(p)
    return torch.minimum(d, rope)


def scene():
    sc = Scene()
    sc.add(land, fields())
    sc.add(hedges, leaves())
    sc.add(woods, leaves("#2e4224"))
    rnd = random.Random(3)
    farm(sc, -40.0, -190.0, 0.3)
    hb = lambda x, z: ground_at(x, z)
    # trees: a row of poplars along the road, oaks on the ridge's flanks
    fol, bark = foliage(), Mat("#3a2e24", rough=0.9, pattern=lambda pl, pw, n: {"albedo": srgb("#3a2e24") * (0.6 + 0.5 * fbm(pw * torch.tensor([6.0, 0.8, 6.0], device=D0), 3))[:, None]})
    trees = [tree(-160 + 17 * k + rnd.uniform(-2, 2), -260 - 3 * k, rnd.uniform(17, 22), rnd, "poplar") for k in range(16)]
    trees += [tree(x, z, h, rnd) for (x, z, h) in ((-46.0, -14.0, 13.0), (-58.0, -22.0, 15.0), (62.0, -6.0, 14.0), (74.0, -24.0, 16.0),
                                                   (-110.0, -60.0, 14.0), (120.0, -70.0, 15.0), (90.0, -150.0, 13.0))]
    for (w, c) in trees:
        sc.add(w, bark)
        sc.add(c, fol)
    # the bivouac: streets of tents, muskets piled before them, fires with their cauldrons, the men round them
    sc.add(tents, canvas_mat())
    sc.add(muskets, musket_mat())
    sc.add(campers, camper_mat())
    sc.add(fires, fire_mat())
    sc.add(soldiers, soldier_mat())
    marquee(sc, -16.0, -15.0, 0.35)
    flag(sc, -12.5, hb(-12.5, -11.0), -11.0, ["#1e3a8a", "#f2f0ea", "#c8242a"], 0.2)
    sc.add(horse_line, horse_mat("#5a3a24"))
    for (x, z, r) in ((-27.0, -27.0, 0.4), (-32.0, -22.0, 0.2), (-24.0, -19.0, 0.9)):
        wagon(sc, x, z, r)
    # the artillery park on the right: guns in a row, their caissons behind
    for k, x in enumerate((9.0, 14.0, 19.0)):
        cannon(sc, x, -15.0 - k * 1.2, -0.3, hb(x, -15.0 - k * 1.2))
    caisson(sc, 16.0, -10.0, 0.6, hb)
    iron = Mat("#1e1e22", metal=0.9, rough=0.4)
    b0 = hb(5.0, -11.0)
    for i in range(4):
        for j in range(4 - i):
            for k in range(4 - i):
                sc.add(sd_sphere(0.2), iron, T(5.0 + (j - (3 - i) / 2) * 0.4, b0 + 0.2 + i * 0.33, -11.0 + (k - (3 - i) / 2) * 0.4))
    # far off, the army forming up on the plain; across the valley, the enemy
    battalion(sc, 12.0, -128.0, -0.08, 36, hb)
    battalion(sc, -95.0, -160.0, 0.2, 40, hb)
    squadron(sc, 70.0, -120.0, -0.35, 18, hb)
    squadron(sc, 105.0, -175.0, -0.3, 18, hb)
    village(sc, -230.0, -420.0, hb, random.Random(9))
    for (x, z, r) in ((-60.0, -265.0, 0.1), (-5.0, -280.0, 0.05), (50.0, -270.0, -0.05), (100.0, -290.0, -0.1)):
        battalion(sc, x, z, r + math.pi, 40, hb, "#a8221c", ("#c8242a", "#f2f0ea", "#1e3a8a"))
    return sc


def woods(p):
    """Woods on the far hills: a canopy of round crowns over the land (heightfield)."""
    x, z = p[:, 0], p[:, 2]
    g = land_h(x, z)
    m = (fbm(torch.stack([x * 0.004, x * 0 + 2, z * 0.004], 1), 4) - 0.55) * 8
    m = m.clamp(0, 1) * sstep(-260, -420, z)
    crowns = 14 * m + 3.5 * (fbm(p * 0.12, 3) - 0.5) * m + 2.0 * (fbm(p * 0.45, 2) - 0.5) * m
    return torch.maximum((p[:, 1] - (g + crowns)) * 0.45, (0.02 - m) * 40)


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def smoke_bank(x0, x1, z0, base, H):
    """A long low bank of musket smoke along a firing line, torn by the wind."""
    def f(p):
        x, y, z = p[:, 0], p[:, 1] - base, p[:, 2] - z0
        along = sstep(x0 - 10, x0 + 25, x) * sstep(x1 + 10, x1 - 25, x)
        prof = torch.exp(-(z / 14.0) ** 2) * torch.exp(-(y.clamp_min(0) / H)) * (y > -2).float()
        n = fbm(p * vec(0.05, 0.08, 0.05) + vec(0, -0.2, 0), 5) + 0.4 * (fbm(p * 0.18 + 5, 3) - 0.5)
        return (prof * along * 1.9 + (n - 0.6) * 2.2).clamp_min(0) * 0.12 * along
    return f, (x0 - 15, base - 3, z0 - 40), (x1 + 15, base + 3 * H, z0 + 40)


def fire_column(cx, cz, base, H, R):
    def f(p):
        x, y, z = p[:, 0] - cx, p[:, 1] - base, p[:, 2] - cz
        u = (y / H).clamp(0, 1)
        rx = x - 0.45 * H * u * u
        r = torch.sqrt(rx * rx + z * z)
        shape = 1 - r / (R * (0.5 + 1.8 * u)) - (y < 0).float() * 4 - (y > H).float() * 4
        n = fbm(p * (2.0 / R) + vec(0, -0.4, 0), 5)
        return (shape + 0.8 * (n - 0.5)).clamp_min(0) * 0.12

    def em(p, s):
        y = p[:, 1] - base
        heat = torch.exp(-y / (R * 0.9)).clamp(0, 1)
        return srgb("#ff6a18") * (heat * heat * 30 * fbm(p * (4 / R), 3) ** 2)[:, None]
    return f, (cx - R * 2, base - 1, cz - R * 2), (cx + R * 2 + 0.45 * H, base + H, cz + R * 2), em


def cumulus(p):
    y = p[:, 1]
    q = p * vec(0.0009, 0.0016, 0.0009)
    layer = sstep(900, 1100, y) * sstep(2200, 1500, y)
    n = fbm(q + vec(2, 0, 7), 6)
    cover = fbm(p * 0.00012 + 4, 3)
    return ((n - 0.46 + 0.4 * (cover - 0.5)) * 5).clamp_min(0) * layer * 0.01


def mist(p):
    """Morning mist lying in the low ground of the plain."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    g = land_h(x, z)
    lowness = (2.0 - g).clamp(0, 8) / 8
    n = fbm(p * vec(0.01, 0.05, 0.01), 4)
    return torch.exp(-(y - g).clamp_min(0) / 4.0) * lowness * (0.4 + n) * 0.012 * sstep(-30, -80, z)


def smoke_column(cx, cz, base, H, R, drift=(0.25, 0.0)):
    def f(p):
        x, y, z = p[:, 0] - cx, p[:, 1] - base, p[:, 2] - cz
        u = (y / H).clamp(0, 1)
        rx = x - drift[0] * H * u * u
        rz = z - drift[1] * H * u * u
        r = torch.sqrt(rx * rx + rz * rz)
        rr = R * (0.4 + 2.2 * u)
        shape = 1 - r / rr - (y < 0).float() * 4 - (y > H).float() * 4
        n = fbm(p * (1.6 / R) + vec(0, -0.3, 0), 4)
        return (shape + 0.9 * (n - 0.5)).clamp_min(0) * 0.25 * (1 - u * 0.8)
    return f, (cx - R * 3, base - 0.5, cz - R * 3), (cx + R * 3 + drift[0] * H, base + H, cz + R * 3 + abs(drift[1]) * H)


def gun_smoke(cx, cy, cz, R):
    """The white cloud of black powder in front of a gun that has just fired."""
    def f(p):
        q = (p - vec(cx, cy, cz)) / vec(R * 1.6, R * 0.8, R)
        r = torch.sqrt((q * q).sum(-1))
        n = fbm(p * (1.8 / R), 5) + 0.5 * (fbm(p * (5.0 / R) + 3, 3) - 0.5)
        return (1 - r + 1.3 * (n - 0.5)).clamp_min(0) * 0.6
    return f, (cx - R * 2.2, cy - R * 1.2, cz - R * 1.6), (cx + R * 2.2, cy + R * 1.4, cz + R * 1.6)


NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    g = float(land_h(torch.tensor([0.0], device=vec(0).device), torch.tensor([8.0], device=vec(0).device))[0])
    cam = Camera((0, g + 9.0, 8), (-6, g - 30.0, -200), fov=60)
    el, az = math.radians(7.0), math.radians(-55)
    sun = (math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el))
    atmo = Atmosphere(sun, sun_power=24.0, mie=45e-6, mie_g=0.78, ground="#5a6a3a")
    hb = lambda x, z: float(land_h(torch.tensor([float(x)], device=vec(0).device), torch.tensor([float(z)], device=vec(0).device))[0])
    vols = [Volume(cumulus, (-20000, 900, -30000), (20000, 2200, 500), albedo="#ffffff", g=0.65, steps=40, light_steps=5, light_dist=500)]
    vols.append(Volume(mist, (-1500, -12, -2500), (1500, 25, -30), albedo="#f0ece4", g=0.6, steps=36, light_steps=2, light_dist=20))
    for (fx, fz, _) in FIRES:
        f, lo, hi = smoke_column(fx, fz, hb(fx, fz) + 0.6, 20.0, 0.8, (0.7, -0.12))
        vols.append(Volume(f, lo, hi, albedo="#cfcac2", g=0.5, steps=40, light_steps=3))
    # the farm's chimney, the village's
    for (x, z) in ((-40.0, -190.0), (-230.0, -420.0)):
        f, lo, hi = smoke_column(x, z, hb(x, z) + 9.0, 60.0, 1.2, (0.3, -0.05))
        vols.append(Volume(f, lo, hi, albedo="#b8b2aa", g=0.5, steps=40, light_steps=3))
    lights = [((fx, hb(fx, fz) + 0.6, fz), "#ff7a2a", -2.2) for (fx, fz, n) in FIRES if n > 0]
    img = render_view(scene(), W, H, cam, None, sun_dir=sun, atmo=atmo, lights=lights, aerial=(0.0009, 0.004, 0.0), sun_soft=0.04,
                      exposure=1.35, steps=360, tmax=5000, ao_scale=0.6, bloom=0.6, dof=(22.0, 0.15), vignette=0.32, grain=0.012,
                      grade=("#4a5a7a", "#ffe6c0"), ss=ss, volumes=[] if NOVOL else vols)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "generaux.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
