"""
legend_scene.py — the Légendes background in 3D (scenery.py): a hidden valley
of legend in autumn, at the end of the afternoon (an original place).

  the gorge     a horseshoe of pale cliffs, hundreds of metres high, their
                faces fluted by vertical joints and banded by ledges where
                bushes cling; the plateau above them wooded
  waterfalls    ribbons of water falling the whole height of the cliffs, wet
                dark rock around them, clouds of spray at their feet; a
                cascade of several tiers beside the village
  the village   on a rocky spur in the middle of the valley: pale stone halls
                with steep slate roofs, slender towers, a terrace on arcades,
                warm windows
  the forest    a dense canopy of round trees in autumn colours (greens, gold,
                orange, crimson) and dark firs, covering the slopes
  the light     a low sun from the right raking the cliffs, golden haze and
                mist lying in the gorge

    cd app/tools/sprites && python3 legend_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, vec, srgb, fbm, hash3
from scenery import Camera, render_view, Atmosphere, Volume

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
D = vec(0).device


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def ridged(p, o=5):
    return 1 - (2 * fbm(p, o) - 1).abs()


def P2(x, z, k=0.0):
    return torch.stack([x, z, x * 0 + k], 1)


def T1(v):
    return torch.tensor([float(v)], device=D)


# ─── the cliffs ─────────────────────────────────────────────────────────────
def cliff_z(x):
    """The foot of the cliffs in plan: a horseshoe open towards the eye."""
    return -800 + 0.0013 * (x - 40) ** 2


def cliff_top(x, z):
    return 380 + 70 * fbm(P2(x * 0.003, z * 0.003, 3), 3)


def cliff(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    s = 0.0026 * (x - 40)
    d = (z - cliff_z(x)) / torch.sqrt(1 + s * s)
    # vertical joints (columns), ledges every few tens of metres, buttresses
    cols = 14 * ridged(torch.stack([x * 0.025, y * 0.003, z * 0.025], 1), 4) ** 1.5 * (0.4 + fbm(P2(x * 0.01, z * 0.01, 2), 2))
    # bedding: the layers of rock stand out in steps, some recessed, with overhangs
    bed = 10 * fbm(torch.stack([x * 0.004, y * 0.015, z * 0.004], 1), 3) + 5 * sstep(0.55, 0.8, fbm(torch.stack([x * 0.008, y * 0.03, z * 0.008], 1), 2))
    ledges = bed
    butt = 45 * sstep(0.5, 0.8, fbm(P2(x * 0.006, z * 0.006, 7), 3))
    d = d - cols - ledges - butt - 6 * fbm(p * 0.05, 3)
    return torch.maximum(d * 0.5, y - cliff_top(x, z))


def autumn(key, region):
    """A tree's colour from its key; the region (a slow noise) gathers the colours in patches."""
    k = (key * 0.7 + region * 0.6) % 1.0
    pal = torch.stack([srgb(c) for c in ("#3e5a22", "#4e6a26", "#6a7a2a", "#9a8a2e", "#c0902e", "#c86a26", "#a8401e", "#7a2a1e",
                                          "#3a5224", "#56702a")])
    idx = (k * pal.shape[0]).long().clamp(0, pal.shape[0] - 1)
    return pal[idx]


def cliff_mat():
    def pat(pl, pw, n):
        x, y, z = pw[:, 0], pw[:, 1], pw[:, 2]
        lime = torch.lerp(srgb("#988a76"), srgb("#d2c4ac"), fbm(torch.stack([x * 0.02, y * 0.004, z * 0.02], 1), 4)[:, None])
        lime = lime * (0.75 + 0.35 * fbm(pw * 0.15, 3))[:, None]
        # streaks of dark seepage down the faces, orange iron stains
        seep = sstep(0.55, 0.75, fbm(torch.stack([x * 0.05, y * 0.002, z * 0.05], 1), 3))
        a = torch.lerp(lime, srgb("#4a4a3e"), (0.6 * seep)[:, None])
        a = torch.lerp(a, srgb("#b07a4a"), (0.35 * sstep(0.62, 0.75, fbm(pw * 0.02 + 4, 3)))[:, None])
        # bushes and trees on ledges (faces turned upwards) and over the cliff's top
        up = torch.maximum(sstep(0.2, 0.45, n[:, 1]) * sstep(0.25, 0.4, fbm(pw * 0.08 + 2, 3)),
                           sstep(0.6, 0.68, fbm(torch.stack([x * 0.012, y * 0.006, z * 0.012], 1) + 5, 4)) * sstep(0.45, 0.6, fbm(pw * 0.15, 2)))
        veg = autumn(fbm(pw * 0.09 + 7, 3) * 2.0, fbm(pw * 0.004, 2)) * (0.55 + 0.6 * fbm(pw * 0.5, 3))[:, None]
        up = torch.maximum(up, sstep(0.55, 0.75, n[:, 1]))          # whatever is flat enough is overgrown
        a = torch.lerp(a, veg, up[:, None])
        # the wet rock behind and around each waterfall
        wet = torch.zeros_like(x)
        for (fx, w) in FALLS:
            wet = torch.maximum(wet, torch.exp(-((x - fx) / (w * 3.0)) ** 2))
        a = a * (1 - 0.45 * wet)[:, None]
        return {"albedo": a, "rough": 0.85 - 0.5 * wet}
    return Mat("#c8bca8", rough=0.85, pattern=pat)


# ─── the valley's floor and slopes ──────────────────────────────────────────
SPUR = (30.0, -430.0, 95.0, 72.0)          # the village's spur: centre x, z, radius, height


def ground_h(x, z):
    h = 8 * fbm(P2(x * 0.01, z * 0.01, 1), 4) + 3 * fbm(P2(x * 0.05, z * 0.05, 2), 2)
    # slopes rising on both sides, a wooded hill near the eye on each side (they frame the view)
    h = h + 330 * sstep(-120, -560, x) * (0.7 + 0.5 * fbm(P2(x * 0.004, z * 0.004, 4), 3))
    h = h + 260 * sstep(230, 520, x) * (0.7 + 0.5 * fbm(P2(x * 0.004, z * 0.004, 5), 3))
    h = h + 110 * torch.exp(-(((x + 300) / 170) ** 2 + ((z + 220) / 150) ** 2))
    h = h + 90 * torch.exp(-(((x - 300) / 150) ** 2 + ((z + 220) / 130) ** 2))
    # scree at the foot of the cliffs
    h = h + 70 * torch.exp(-(z - cliff_z(x)).clamp_min(0) / 45)
    # the spur: steep sides, a rolling top
    r = torch.sqrt((x - SPUR[0]) ** 2 + ((z - SPUR[1]) * 1.2) ** 2) / SPUR[2]
    r = r / (1 + 0.4 * (fbm(P2(x * 0.012, z * 0.012, 6), 3) - 0.5))
    spur = SPUR[3] * sstep(0.8, 0.74, r) + 6 * fbm(P2(x * 0.03, z * 0.03, 8), 2)
    h = torch.maximum(h, spur)
    # the river's bed
    rx = x - (120 + 60 * torch.sin(z * 0.006))
    h = h - 4 * torch.exp(-(rx / 14) ** 2)
    return h


def land_at(x, z):
    return float(ground_h(T1(x), T1(z))[0])


def in_village(x, z):
    return torch.sqrt((x - SPUR[0]) ** 2 + (z - SPUR[1]) ** 2) < 66


def canopy(x, z):
    """The forest: crowns over the ground, on the 2×2 nearest cells of a jittered grid; returns the
    canopy's height over the ground and each point's tree (a key for its colour, and fir or not)."""
    cell = 11.0
    dens = sstep(0.18, 0.3, fbm(P2(x * 0.006, z * 0.006, 11), 3))           # woods and glades
    rs = torch.sqrt((x - SPUR[0]) ** 2 + ((z - SPUR[1]) * 1.2) ** 2) / SPUR[2]
    rs = rs / (1 + 0.4 * (fbm(P2(x * 0.012, z * 0.012, 6), 3) - 0.5))
    cliffs = 1 - sstep(0.7, 0.76, rs) * sstep(0.95, 0.88, rs)           # the spur's rocky sides stay bare
    rx = x - (120 + 60 * torch.sin(z * 0.006))
    mask = sstep(12, 20, rx.abs()) * dens * cliffs * (1 - in_village(x, z).float()) * sstep(-45, -80, z) * (z > cliff_z(x) + 8).float()
    c = torch.zeros_like(x)
    key = torch.zeros_like(x)
    fir = torch.zeros_like(x)
    i0, k0 = torch.floor(x / cell - 0.5), torch.floor(z / cell - 0.5)
    pfir = 0.2 + 0.5 * sstep(-160, -70, z) + 0.3 * sstep(0.5, 0.7, fbm(P2(x * 0.004, z * 0.004, 13), 2))
    for di in (0, 1):
        for dk in (0, 1):
            i, k = i0 + di, k0 + dk
            idv = torch.stack([i, k, i * 0 + 5], 1)
            cx = (i + 0.5 + (hash3(idv) - 0.5) * 0.6) * cell
            cz = (k + 0.5 + (hash3(idv + 1.3) - 0.5) * 0.6) * cell
            isf = (hash3(idv + 8.1) < pfir).float()
            rad = (5.5 + 3.5 * hash3(idv + 4.4)) * (1 - 0.3 * isf)
            H = rad * (1.4 + 0.4 * hash3(idv + 2.9)) * (1 + 1.6 * isf)
            u = torch.sqrt((x - cx) ** 2 + (z - cz) ** 2) / rad
            dome = (1 - u * u).clamp_min(0) ** 0.6
            cone = (1 - u).clamp_min(0) ** 1.1
            h = H * torch.lerp(dome, cone, isf) * (hash3(idv + 6.6) < 0.9).float()
            win = h > c
            c = torch.where(win, h, c)
            key = torch.where(win, hash3(idv + 9.7), key)
            fir = torch.where(win, isf, fir)
    # leaves and needles break the outline
    c = c * (0.72 + 0.5 * fbm(P2(x * 0.7, z * 0.7, 4), 3))
    return c * mask, key, fir


def spur_rock(p):
    """The spur's sides: a crag of the cliffs' pale rock, jointed and stepped, standing out of the woods."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = torch.sqrt((x - SPUR[0]) ** 2 + ((z - SPUR[1]) * 1.2) ** 2)
    r = r / (1 + 0.4 * (fbm(P2(x * 0.012, z * 0.012, 6), 3) - 0.5))
    disp = 7 * ridged(torch.stack([x * 0.05, y * 0.012, z * 0.05], 1), 4) + 4 * fbm(torch.stack([x * 0.02, y * 0.09, z * 0.02], 1), 2)
    r = r - 16 * (fbm(p * 0.012 + 3, 3) - 0.5)
    side = (r - SPUR[2] * (0.84 + 0.3 * sstep(SPUR[3] - 10, 0, y)) - disp) * 0.5
    top = y - (SPUR[3] - 2 + 4 * fbm(P2(x * 0.03, z * 0.03, 8), 2) - 22 * sstep(SPUR[2] * 0.6, SPUR[2] * 1.0, r))
    return torch.maximum(side, top * 0.7)


def land(p):
    x, z = p[:, 0], p[:, 2]
    c, _, _ = canopy(x, z)
    return (p[:, 1] - ground_h(x, z) - c) * 0.3


def land_mat():
    def pat(pl, pw, n):
        x, y, z = pw[:, 0], pw[:, 1], pw[:, 2]
        g = ground_h(x, z)
        c, key, fir = canopy(x, z)
        steep = (1 - n[:, 1]).clamp(0, 1)
        grass = torch.lerp(srgb("#4a5a26"), srgb("#8a7a3a"), fbm(pw * 0.03, 3)[:, None]) * (0.7 + 0.4 * fbm(pw * 0.4, 2))[:, None]
        rock = torch.lerp(srgb("#8a8070"), srgb("#c8bca8"), fbm(pw * 0.05, 3)[:, None])
        a = torch.lerp(grass, rock, sstep(0.45, 0.7, steep)[:, None])
        # leaf litter under the trees
        a = torch.lerp(a, srgb("#8a5a2a"), (0.4 * sstep(0.55, 0.7, fbm(pw * 0.08, 3)))[:, None])
        # the trees: the autumn palette, firs dark, the crowns lighter at the top (sky light, sun)
        top = sstep(0.2, 1.0, (y - g) / (c + 1e-3))
        leaf = autumn(key, fbm(pw * 0.003, 2)) * (0.65 + 0.5 * fbm(pw * 0.7, 2))[:, None]
        firc = torch.lerp(srgb("#16261a"), srgb("#2a3a22"), key[:, None]) * (0.7 + 0.5 * fbm(pw * 0.9, 2))[:, None]
        clump = 0.6 + 0.6 * fbm(pw * 1.6, 2)                    # clumps of leaves in light and shade
        tree = torch.lerp(leaf, firc, fir[:, None]) * ((0.45 + 0.75 * top) * clump)[:, None]
        a = torch.lerp(a, tree, (sstep(0.6, 1.6, y - g) * sstep(0.3, 1.0, c))[:, None])
        return {"albedo": a}
    return Mat("#4a5a26", rough=0.9, sss=0.3, pattern=pat)


def river(p):
    x, z = p[:, 0], p[:, 2]
    rx = x - (120 + 60 * torch.sin(z * 0.006))
    w = 9 + 4 * fbm(P2(z * 0.01, z * 0, 3), 2)
    return torch.maximum(torch.maximum(rx.abs() - w, (p[:, 1] - 2.0).abs() - 0.3), z + 150) * 0.8


def water_mat():
    def pat(pl, pw, n):
        foam = sstep(0.55, 0.7, fbm(pw * vec(0.2, 0, 0.06), 3))
        return {"albedo": torch.lerp(srgb("#1a2a2a"), srgb("#e8f0f0"), (0.8 * foam)[:, None]), "rough": 0.05 + 0.5 * foam}
    return Mat("#1a2a2a", rough=0.05, clearcoat=1.0, pattern=pat)


# ─── the village ────────────────────────────────────────────────────────────
#  halls: (x, z offset from the spur's centre, half width, half depth, wall height, roof height, rotation)
HALLS = [(0, 0, 20, 10, 13, 11, 0.25), (-30, 14, 12, 9, 10, 9, -0.35), (28, -18, 14, 9, 11, 9.5, 0.6), (8, 28, 10, 8, 8, 7.5, 0.1),
         (-16, -28, 12, 8, 10, 8, 1.0), (42, 10, 8, 7, 9, 7, -0.2), (-46, -12, 9, 7, 8, 6.5, 0.5)]
TOWERS = [(17, 6, 4.5, 34), (-38, -4, 3.8, 25), (34, -30, 3.2, 20)]          # x, z, radius, height
BASES = {}


def _rot(qx, qz, a):
    c, s = math.cos(a), math.sin(a)
    return c * qx + s * qz, -s * qx + c * qz


def walls(p):
    x, y, z = p[:, 0] - SPUR[0], p[:, 1], p[:, 2] - SPUR[1]
    d = torch.full_like(x, 1e9)
    for (hx, hz, w, dp, h, rh, a) in HALLS:
        g = BASES[(hx, hz)]
        qx, qz = _rot(x - hx, z - hz, a)
        qy = y - g
        body = torch.stack([qx.abs() - w, (qy - h * 0.5 + 3).abs() - h * 0.5 - 3, qz.abs() - dp], 1)
        b = torch.linalg.norm(body.clamp_min(0), dim=1) + body.max(1).values.clamp_max(0)
        # the gable ends: the walls go up to the roof's slope
        gable = torch.maximum(torch.maximum(qx.abs() - w, qz.abs() - dp), (qy - h) + rh * qz.abs() / dp - rh)
        d = torch.minimum(d, torch.minimum(b, torch.maximum(gable * 0.7, h - 0.5 - qy)))
    for (tx, tz, r, h) in TOWERS:
        g = BASES[(tx, tz)]
        qy = y - g
        rr = torch.sqrt((x - tx) ** 2 + (z - tz) ** 2) - r * (1 - 0.06 * (qy / h).clamp(0, 1))
        d = torch.minimum(d, torch.maximum(rr, torch.maximum(qy - h, -qy - 6)))
    # a terrace in front, carried on arcades over the spur's edge
    tx, tz = _rot(x - 2, z - 34, 0.2)
    ty = y - BASES["terrace"]
    slab = torch.maximum(torch.maximum(tx.abs() - 26, tz.abs() - 7), ty.abs() - 0.8)
    k = tx / 6.0
    ax = (k - torch.floor(k) - 0.5) * 6.0
    arch = torch.maximum(torch.sqrt(ax * ax + (ty + 6).clamp_min(0) ** 2) - 2.2, ty + 0.8)
    under = torch.maximum(torch.maximum(tx.abs() - 26, tz.abs() - 6), torch.maximum(ty, -ty - 22))
    arcade = torch.maximum(under, -arch)
    rail = torch.maximum(torch.maximum(tx.abs() - 26, (tz - 6.7).abs() - 0.3), torch.maximum(ty - 2.0, -ty))
    d = torch.minimum(d, torch.minimum(torch.minimum(slab, arcade), rail))
    return d * 0.9


def roofs(p):
    x, y, z = p[:, 0] - SPUR[0], p[:, 1], p[:, 2] - SPUR[1]
    d = torch.full_like(x, 1e9)
    for (hx, hz, w, dp, h, rh, a) in HALLS:
        g = BASES[(hx, hz)]
        qx, qz = _rot(x - hx, z - hz, a)
        qy = y - g - h
        dd, ww = dp + 1.2, w + 1.0
        slope = rh / dd
        plane = (qy - rh + slope * qz.abs()) / math.sqrt(1 + slope * slope)
        r = torch.maximum(torch.maximum(plane.abs() - 0.35, -qy - 0.2), torch.maximum(qx.abs() - ww, qz.abs() - dd))
        d = torch.minimum(d, r)
    for (tx, tz, rad, h) in TOWERS:
        g = BASES[(tx, tz)]
        qy = y - g - h
        rr = torch.sqrt((x - tx) ** 2 + (z - tz) ** 2)
        hc = rad * 3.2
        cone = (rr - (rad + 0.8) * (1 - qy / hc)) * (hc / math.sqrt(hc * hc + rad * rad))
        d = torch.minimum(d, torch.maximum(cone, torch.maximum(-qy, qy - hc)))
    return d * 0.9


def wall_mat():
    def pat(pl, pw, n):
        x, y, z = pw[:, 0], pw[:, 1], pw[:, 2]
        a = srgb("#e4dccc") * (0.8 + 0.25 * fbm(pw * 0.4, 3))[:, None]
        a = torch.lerp(a, srgb("#8a8a6a"), (0.3 * sstep(0.6, 0.8, fbm(pw * 0.15 + 3, 3)))[:, None])     # moss, weathering
        # windows: tall openings in rows on the walls
        u = x + z
        cu = (u / 3.2) - torch.floor(u / 3.2)
        rel = y - BASES[(0, 0)]
        cy = (rel / 4.0) - torch.floor(rel / 4.0)
        vert = (n[:, 1].abs() < 0.3).float() * (rel > 1.5).float()
        win = ((cu - 0.5).abs() < 0.16).float() * (cy > 0.25).float() * (cy < 0.8).float() * vert
        lit = (hash3(torch.floor(torch.stack([u / 3.2, rel / 4.0, x * 0], 1))) > 0.45).float()
        a = a * (1 - 0.75 * win)[:, None]
        return {"albedo": a, "emissive": srgb("#ffb050") * (win * lit * 2.5)[:, None]}
    return Mat("#e4dccc", rough=0.75, pattern=pat)


def roof_mat():
    def pat(pl, pw, n):
        rows = 0.5 + 0.5 * torch.sin(pw[:, 1] * 9.0)
        a = torch.lerp(srgb("#2e3640"), srgb("#4a5260"), (0.4 * rows + 0.6 * fbm(pw * 0.5, 2))[:, None])
        a = torch.lerp(a, srgb("#5a6a3a"), (0.35 * sstep(0.6, 0.75, fbm(pw * 0.2, 3)))[:, None])      # moss
        return {"albedo": a}
    return Mat("#3a4250", rough=0.45, metal=0.1, pattern=pat)


# ─── water and air ──────────────────────────────────────────────────────────
FALLS = [(-250.0, 7.0), (-110.0, 4.5), (215.0, 8.0), (330.0, 5.0), (-330.0, 3.5)]      # x of the lip, half width
CASCADE = (95.0, -455.0)                  # beside the village: the stream falls in tiers from the spur to the river


def fall_density(fx, w):
    zc = float(cliff_z(T1(fx))[0])
    top = float(cliff_top(T1(fx), T1(zc))[0])

    def dens(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        u = (top - y).clamp_min(0)
        out = 14 + 0.035 * u                                  # it leaves the lip and falls clear of the face
        ww = w * (1 + u / 250)
        across = (x - fx) / ww
        along = (z - zc - out) / (3 + u * 0.03)
        streak = fbm(torch.stack([x * (1.5 / w), y * 0.02, z * 0.1], 1), 4)
        sheet = torch.exp(-across ** 6 - along ** 2) * (0.35 + 1.3 * streak) * (1 + u / 150)
        return sheet * (y < top + 1).float() * 0.4
    zz = zc + 14 + 0.035 * top
    return dens, (fx - w * 5 - 10, -5, zc - 10), (fx + w * 5 + 10, top + 2, zz + 20)


def spray(p):
    """Spray at the foot of the falls, a cloud of mist billowing out over the valley."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    d = torch.zeros_like(x)
    bases = [(fx, w, float(cliff_z(T1(fx))[0]) + 40) for (fx, w) in FALLS] + [(CASCADE[0] + 4, 3.0, CASCADE[1] + 15)]
    for (fx, w, zc) in bases:
        r2 = ((x - fx) / (w * 10 + 25)) ** 2 + ((z - zc) / 60) ** 2 + ((y - 15) / 70) ** 2
        d = d + torch.exp(-r2 * 1.5)
    return d * ((fbm(p * 0.02 + 3, 4) - 0.25) * 1.6).clamp_min(0) * 0.03


def valley_mist(p):
    """Mist lying low in the gorge, thicker towards the cliffs, glowing in the sun."""
    y, z = p[:, 1], p[:, 2]
    n = fbm(p * vec(0.004, 0.012, 0.004), 5)
    band = torch.exp(-((y - 60) / 45) ** 2) * sstep(-250, -500, z)
    return band * ((n - 0.42) * 2.5).clamp_min(0) * 0.004


def cascade(p):
    x, y, z = p[:, 0] - CASCADE[0], p[:, 1], p[:, 2] - CASCADE[1]
    d = torch.zeros_like(x)
    for (ox, w) in ((-8.0, 2.2), (0.0, 3.0), (9.0, 1.8), (16.0, 1.4)):
        # tiers: the water slides down a step then falls, leaning out at each step
        step = (y / 18.0) - torch.floor(y / 18.0)
        lean = 6 * (1 - step) + 2
        across = (x - ox - 1.5 * torch.sin(y * 0.05 + ox)) / w
        along = (z - lean - (60 - y) * 0.15) / 3.5
        streak = fbm(torch.stack([x * 0.6, y * 0.03, z * 0.3 + ox], 1), 3)
        d = d + torch.exp(-across ** 4 - along ** 2) * (0.4 + 1.2 * streak)
    return d * (y < 70).float() * (y > 0).float() * 0.5


def scene():
    sc = Scene()
    sc.add(land, land_mat())
    sc.add(cliff, cliff_mat())
    sc.add(spur_rock, cliff_mat())
    sc.add(river, water_mat())
    sc.add(walls, wall_mat())
    sc.add(roofs, roof_mat())
    return sc


def _bases():
    for (hx, hz, *_r) in HALLS:
        BASES[(hx, hz)] = land_at(SPUR[0] + hx, SPUR[1] + hz) - 1.0
    for (tx, tz, *_r) in TOWERS:
        BASES[(tx, tz)] = land_at(SPUR[0] + tx, SPUR[1] + tz) - 1.0
    BASES["terrace"] = land_at(SPUR[0] + 2, SPUR[1] + 26) + 1.0


_bases()
NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    el, az = math.radians(11), math.radians(78)
    sun = (math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el))
    atmo = Atmosphere(sun, sun_power=24.0, mie=18e-6, mie_g=0.8, ground="#4a4a2a")
    cam = Camera((-20.0, 175.0, -60.0), (25, 95, -440), fov=56)
    vols = [Volume(valley_mist, (-700, 0, -760), (700, 180, -200), albedo="#f4ece0", g=0.6, steps=48, light_steps=3, light_dist=150),
            Volume(spray, (-450, -5, -720), (450, 160, -360), albedo="#ffffff", g=0.5, steps=48, light_steps=2, light_dist=60),
            Volume(cascade, (CASCADE[0] - 25, 0, CASCADE[1] - 5), (CASCADE[0] + 35, 72, CASCADE[1] + 30), albedo="#f4faff", g=0.4,
                   steps=48, light_steps=2, light_dist=15, ambient=1.2)]
    for (fx, w) in FALLS:
        f, lo, hi = fall_density(fx, w)
        vols.append(Volume(f, lo, hi, albedo="#f4faff", g=0.4, steps=64, light_steps=2, light_dist=20, ambient=1.2))
    img = render_view(scene(), W, H, cam, None, sun_dir=sun, atmo=atmo, aerial=(0.0004, 0.002, 0.0), sun_soft=0.03,
                      exposure=1.35, steps=420, tmax=6000, ao_scale=2.5, bloom=0.7, dof=None, vignette=0.3,
                      grain=0.01, grade=("#2a3a4a", "#ffe0b0"), ss=ss, volumes=[] if NOVOL else vols)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "legendes.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
