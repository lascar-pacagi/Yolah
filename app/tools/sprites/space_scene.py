"""
space_scene.py — the Espace background in 3D (scenery.py): the solar system
seen from a little above its plane, the Sun on the left, the planets fanned
out around the board (sizes and distances shrunk so that every one shows).

  the Sun       granulation, sunspots and faculae, limb darkening (white-gold
                at the centre, orange at the rim), a corona of streamers and
                prominences arching from the limb (an emissive volume)
  the planets   lit by the Sun as a point light, each with its own surface:
                Mercury cratered, Venus under yellow clouds, the Earth (oceans
                with the Sun's glint, continents, ice, clouds, the blue rim of
                its air, cities on the night side) and the Moon, Mars (dark
                plains, a polar cap, a canyon), Jupiter (belts and zones bent by
                turbulence, the great red storm), Saturn and its rings (ringlets,
                the Cassini division, the planet's shadow on them), Uranus pale
                cyan with its thin upright rings, Neptune deep blue with a dark
                storm and bright clouds
  the belt      asteroids strewn along an arc between Mars and Jupiter
  the sky       the Milky Way, its dust lanes, stars of every colour

    cd app/tools/sprites && python3 space_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RZ, vec, norm, srgb, sd_sphere, fbm, hash3
from scenery import Camera, render_view, Volume

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
CAM = (0.0, 0.0, 0.0)
LOOK = (0.0, 0.0, -1.0)
TAN, ASP = math.tan(math.radians(30)), 16 / 9


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def ridged(p, o=4):
    return 1 - (2 * fbm(p, o) - 1).abs()


def place(sx, sy, depth):
    """The world point seen at screen (sx, sy) ∈ [-1, 1]² at that depth."""
    return (sx * TAN * ASP * depth, sy * TAN * depth, -depth)


SUN_R = 112.0
SUN = place(-0.98, 0.12, 480)
#  name: (screen x, screen y, depth, radius)   — sizes and distances shrunk so that every planet shows
PLANETS = {"mercury": (-0.5, 0.58, 640, 11), "venus": (-0.6, -0.4, 620, 22), "earth": (-0.3, -0.7, 650, 25),
           "mars": (0.05, -0.83, 700, 16), "jupiter": (0.58, -0.48, 760, 76), "saturn": (0.74, 0.3, 980, 62),
           "uranus": (0.3, 0.72, 1400, 40), "neptune": (0.9, 0.8, 1650, 42)}


def pos(name):
    sx, sy, dp, _ = PLANETS[name]
    return place(sx, sy, dp)


def light_terms(pw, n):
    """N·L towards the Sun and N·V towards the eye, for the materials' rims and night sides."""
    L = norm(vec(*SUN) - pw)
    V = norm(vec(*CAM) - pw)
    return (n * L).sum(-1), (n * V).sum(-1).clamp(0, 1)


# ─── the sky ────────────────────────────────────────────────────────────────
def space_sky(d):
    y = d[:, 1].clamp(-1, 1)
    el = torch.asin(y)
    az = torch.atan2(d[:, 0], -d[:, 2])
    c = srgb("#020309").expand(d.shape[0], 3).clone()
    gn = norm(vec(0.3, 0.8, 0.52))
    k = d @ gn
    band = torch.exp(-(k / 0.2) ** 2)
    lanes = ((fbm(d * 7 + 3, 5) - 0.5) * 4).clamp(0, 1) * torch.exp(-(k / 0.06) ** 2)
    glow = band * (0.4 + 0.6 * fbm(d * 4, 4)) * (1 - 0.75 * lanes)
    c = c + torch.lerp(srgb("#3a3050"), srgb("#8a7a6a"), fbm(d * 3 + 7, 3)[:, None]) * (glow * 0.2)[:, None]
    # faint nebulae: reddish hydrogen, a teal wisp
    neb = ((fbm(d * 2.5 + 11, 5) - 0.55) * 3).clamp(0, 1) * band
    c = c + srgb("#a02a40") * (neb * 0.08)[:, None] + srgb("#1a6a7a") * (((fbm(d * 3 + 5, 4) - 0.6) * 3).clamp(0, 1) * band * 0.05)[:, None]
    for (res, thr, gain) in ((900, 0.984, 0.3), (420, 0.992, 1.0), (160, 0.996, 4.0)):
        cell = torch.floor(torch.stack([az * res / math.pi, el * res / math.pi, az * 0 + res], 1))
        h = hash3(cell)
        fx = (az * res / math.pi) - cell[:, 0] - hash3(cell + 1.7)
        fy = (el * res / math.pi) - cell[:, 1] - hash3(cell + 4.1)
        spot = torch.exp(-(fx * fx + fy * fy) / (0.02 if res > 300 else 0.006))
        on = (h > thr).float() * (1 + 3 * band * (res > 500))
        temp = hash3(cell + 9.3)
        col = torch.where((temp < 0.2)[:, None], srgb("#ffb27a"), torch.where((temp > 0.8)[:, None], srgb("#a8c4ff"), srgb("#fff4e8")))
        c = c + col * (on * spot * gain * (0.4 + (h - thr) / (1 - thr)))[:, None]
    return c


# ─── the Sun ────────────────────────────────────────────────────────────────
def sun_mat():
    def pat(pl, pw, n):
        q = pl / SUN_R
        V = norm(vec(*CAM) - pw)
        mu = (n * V).sum(-1).clamp(0, 1)
        # granulation: cells of rising gas, dark lanes between them
        g = ridged(q * 40, 3)
        gran = 0.75 + 0.35 * g ** 2
        # supergranulation and faculae (bright patches near the limb), sunspots (umbra, penumbra)
        fac = sstep(0.6, 0.75, fbm(q * 6 + 2, 3)) * (1 - mu) * 0.6
        sp = fbm(q * 5 + 9, 4)
        umbra = sstep(0.74, 0.78, sp) * (q[:, 1].abs() < 0.5).float()
        pen = sstep(0.69, 0.74, sp) * (q[:, 1].abs() < 0.5).float()
        limb = 0.35 + 0.65 * mu ** 0.55
        col = torch.lerp(srgb("#ff6a10"), srgb("#fff2c8"), (mu ** 0.6)[:, None])
        k = (gran + fac) * limb * (1 - 0.55 * pen) * (1 - 0.8 * umbra)
        return {"albedo": torch.zeros_like(pw), "emissive": col * (2.4 * k)[:, None]}
    return Mat("#000000", rough=1.0, pattern=pat)


def corona_density(p):
    p = p - vec(*SUN)
    r = torch.sqrt((p * p).sum(-1)) / SUN_R
    u = p / (r[:, None] * SUN_R + 1e-3)
    streak = 0.4 + 1.2 * fbm(u * 3.5 + 2, 4) ** 3
    glow = torch.exp(-(r - 1).clamp_min(0) / 0.05) * 0.8 + torch.exp(-(r - 1).clamp_min(0) / 0.3) * streak * 0.25
    # prominences: loops of glowing gas standing on the limb
    prom = torch.zeros_like(r)
    for (a, b, s) in ((0.3, 2.4, 0.18), (2.2, -0.9, 0.12), (4.1, 0.6, 0.15), (5.2, -2.0, 0.1)):
        c = vec(math.cos(a) * math.cos(b * 0.3), math.sin(b * 0.3), math.sin(a) * math.cos(b * 0.3)) * SUN_R
        nrm = norm(vec(-math.sin(a), 0.3, math.cos(a)))
        q = p - c
        hgt = (q * nrm).sum(-1)
        rad = torch.sqrt(((q - nrm[None] * hgt[:, None]) ** 2).sum(-1).clamp_min(1e-6))
        tube = torch.sqrt((rad - s * SUN_R) ** 2 + hgt ** 2)
        prom = prom + torch.exp(-(tube / (0.018 * SUN_R)) ** 2) * (0.6 + 0.8 * fbm(p * 0.15 + a, 3))
    return ((glow + prom * 3.0) * (r > 0.99).float()) * 0.004


def corona_emission(p, s):
    p = p - vec(*SUN)
    r = torch.sqrt((p * p).sum(-1)) / SUN_R
    hot = torch.lerp(srgb("#ff5a1a")[None].expand(p.shape[0], 3), srgb("#ffe0b0")[None].expand(p.shape[0], 3),
                     sstep(1.0, 1.6, r)[:, None])
    return hot * 8.0


# ─── the planets ────────────────────────────────────────────────────────────
def rocky(c1, c2, R, craters=1.0):
    def f(q):
        d = sd_sphere(R)(q)
        cr = 1 - (2 * fbm(q * (2.5 / R), 3) - 1).abs()
        return d - craters * (0.025 * R * cr ** 4 - 0.012 * R * fbm(q * (8 / R), 2))
    return f


def mercury_mat(R):
    def pat(pl, pw, n):
        q = pl / R
        a = torch.lerp(srgb("#5a5652"), srgb("#a8a29a"), fbm(q * 3, 5)[:, None])
        rays = sstep(0.7, 0.8, fbm(q * 12 + 4, 3))
        return {"albedo": torch.lerp(a, srgb("#d8d2c8"), (0.5 * rays)[:, None])}
    return Mat("#8a8680", rough=0.95, pattern=pat)


def venus_mat(R):
    def pat(pl, pw, n):
        q = pl / R
        w = q + 0.35 * (torch.stack([fbm(q * 2, 3), fbm(q * 2 + 5, 3), fbm(q * 2 + 9, 3)], 1) - 0.5)
        bands = 0.5 + 0.5 * torch.sin(w[:, 1] * 9 + 4 * fbm(w * 3, 3))
        a = torch.lerp(srgb("#c8a868"), srgb("#f4e6c0"), (0.5 * bands + 0.5 * fbm(w * 5, 3))[:, None])
        nl, nv = light_terms(pw, n)
        rim = (1 - nv) ** 3 * sstep(-0.3, 0.3, nl)
        return {"albedo": a, "emissive": srgb("#ffe8b0") * (rim * 0.25)[:, None]}
    return Mat("#e8d8a8", rough=0.9, pattern=pat)


def earth_mat(R):
    def pat(pl, pw, n):
        q = pl / R
        cont = fbm(q * 1.6 + 3, 6) + 0.08 * (fbm(q * 8, 3) - 0.5)
        land = sstep(0.52, 0.54, cont)
        lat = q[:, 1].abs()
        desert = sstep(0.55, 0.7, fbm(q * 3 + 7, 3)) * sstep(0.55, 0.15, lat) * sstep(0.08, 0.2, lat)
        ground = torch.lerp(srgb("#2e5a22"), srgb("#4a6a2a"), fbm(q * 9, 3)[:, None])
        ground = torch.lerp(ground, srgb("#c8a868"), desert[:, None])
        ground = torch.lerp(ground, srgb("#6a5a4a"), (0.5 * sstep(0.6, 0.75, ridged(q * 5, 3)))[:, None])
        ocean = torch.lerp(srgb("#06204a"), srgb("#0a3a6a"), sstep(0.5, 0.53, cont)[:, None])
        a = torch.lerp(ocean, ground, land[:, None])
        ice = sstep(0.78, 0.86, lat + 0.06 * fbm(q * 6, 2))
        a = torch.lerp(a, srgb("#f0f4f8"), ice[:, None])
        # clouds: swirls and fronts
        w = q + 0.25 * (torch.stack([fbm(q * 3, 3), fbm(q * 3 + 4, 3), fbm(q * 3 + 8, 3)], 1) - 0.5)
        cl = sstep(0.52, 0.68, fbm(w * 4 + 1, 5)) * (0.7 + 0.3 * sstep(0.2, 0.6, lat))
        a = torch.lerp(a, srgb("#ffffff"), (0.9 * cl)[:, None])
        nl, nv = light_terms(pw, n)
        rim = (1 - nv) ** 2.5 * sstep(-0.25, 0.35, nl)
        city = (hash3(torch.floor(q * 260)) > 0.93).float() * land * (1 - ice) * (1 - cl) * sstep(0.0, -0.2, nl)
        em = srgb("#4a9aff") * (rim * 0.9)[:, None] + srgb("#ffb860") * (city * 0.8)[:, None]
        rough = 0.9 - 0.75 * (1 - land) * (1 - cl) * (1 - ice)            # the oceans glint
        return {"albedo": a, "emissive": em, "rough": rough}
    return Mat("#2a4a8a", rough=0.6, pattern=pat)


def mars_mat(R):
    def pat(pl, pw, n):
        q = pl / R
        a = torch.lerp(srgb("#8a3a1e"), srgb("#d07a48"), fbm(q * 2.5, 5)[:, None])
        dark = sstep(0.52, 0.62, fbm(q * 2 + 3, 4)) * sstep(0.7, 0.3, q[:, 1].abs())
        a = torch.lerp(a, srgb("#4a2a1e"), (0.6 * dark)[:, None])
        canyon = torch.exp(-((q[:, 1] + 0.15 + 0.05 * fbm(q * 4, 2)) / 0.025) ** 2) * sstep(-0.2, 0.4, q[:, 0])
        a = a * (1 - 0.5 * canyon)[:, None]
        cap = sstep(0.82, 0.9, q[:, 1] + 0.05 * fbm(q * 8, 2))
        a = torch.lerp(a, srgb("#f4f0ec"), cap[:, None])
        nl, nv = light_terms(pw, n)
        rim = (1 - nv) ** 3 * sstep(-0.2, 0.3, nl)
        return {"albedo": a, "emissive": srgb("#e8a080") * (rim * 0.12)[:, None]}
    return Mat("#c05a30", rough=0.95, pattern=pat)


def giant_mat(R, cols, nb, storm=None, rim_c="#ffe8c8", turb=0.16):
    """Gas giants: latitude bands, bent and torn by turbulence; an optional storm (lat, lon, size, colour)."""
    cs = [srgb(c) for c in cols]

    def pat(pl, pw, n):
        q = pl / R
        lat = q[:, 1]
        warp = turb * (fbm(q * 3, 5) - 0.5) + 0.4 * turb * (fbm(q * 14 + 3, 4) - 0.5)
        b = (lat + warp) * nb
        k = (torch.sin(b) * 0.5 + 0.5)
        k2 = (torch.sin(b * 2.7 + 1.3) * 0.5 + 0.5)
        a = torch.lerp(cs[0], cs[1], k[:, None])
        a = torch.lerp(a, cs[2], (0.45 * k2)[:, None]) * (0.85 + 0.25 * fbm(q * 18, 3))[:, None]
        # eddies along the belts' edges
        ed = sstep(0.7, 0.85, fbm(torch.stack([q[:, 0] * 10, lat * 30, q[:, 2] * 10], 1), 3))
        a = torch.lerp(a, cs[3], (0.5 * ed)[:, None])
        a = torch.lerp(a, cs[3] * 0.9, sstep(0.75, 0.95, lat.abs())[:, None] * 0.6)
        if storm is not None:
            slat, slon, ssz, scol = storm
            lon = torch.atan2(q[:, 2], q[:, 0])
            sx, sy = (lon - slon) / (ssz * 1.8), (lat - slat) / ssz
            sr = torch.sqrt(sx * sx + sy * sy)
            sw = 0.5 + 0.5 * torch.sin(torch.atan2(sy, sx) * 2 + sr * 7)
            a = torch.lerp(a, srgb(scol) * (0.75 + 0.35 * sw)[:, None], (torch.exp(-sr ** 2 * 1.6))[:, None])
        nl, nv = light_terms(pw, n)
        rim = (1 - nv) ** 3 * sstep(-0.2, 0.3, nl)
        return {"albedo": a, "emissive": srgb(rim_c) * (rim * 0.15)[:, None]}
    return Mat(cols[1], rough=0.95, pattern=pat)


def rings(R, r0, r1, thick):
    def f(q):
        r = torch.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2) / R
        slab = q[:, 1].abs() - thick
        ann = torch.maximum(r0 - r, r - r1) * R
        div = -((r - (r0 + 0.72 * (r1 - r0))).abs() - 0.025) * R           # the Cassini division
        return torch.maximum(torch.maximum(slab, ann), div) * 0.9
    return f


def ring_mat(R, r0, r1, c1="#6a5a4a", c2="#e8d8bc"):
    def pat(pl, pw, n):
        r = torch.sqrt(pl[:, 0] ** 2 + pl[:, 2] ** 2) / R
        u = (r - r0) / (r1 - r0)
        b = fbm(torch.stack([r * 60, r * 0, r * 0], 1), 5) * 1.2 - 0.1
        b2 = torch.sin(r * 400) * 0.5 + 0.5
        a = torch.lerp(srgb(c1), srgb(c2), (0.9 * b + 0.15 * b2 - 0.15).clamp(0, 1)[:, None])
        a = a * (0.35 + 0.65 * sstep(0.0, 0.25, u))[:, None] * (0.5 + 0.5 * sstep(1.0, 0.85, u))[:, None]
        return {"albedo": a}
    return Mat(c2, rough=0.9, pattern=pat)


def belt(p):
    """Asteroids in a band around the Sun between Mars and Jupiter (one per cell, a safe bound)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = torch.sqrt(x * x + z * z)
    band = torch.maximum((r - 510).abs() - 45, y.abs() - 10)
    cell = 7.0
    ix, iy, iz = torch.floor(x / cell), torch.floor(y / cell), torch.floor(z / cell)
    idv = torch.stack([ix, iy, iz], 1)
    c = (idv + 0.5 + (torch.stack([hash3(idv), hash3(idv + 1.1), hash3(idv + 2.3)], 1) - 0.5) * 0.3) * cell
    rr = cell * (0.06 + 0.14 * hash3(idv + 3.3) ** 2)
    keep = hash3(idv + 4.4) < 0.12 * torch.exp(-((r - 510) / 30) ** 2) + 0.015
    q = p - c
    rock = (torch.sqrt((q * q).sum(-1)) - rr) - 0.35 * rr * fbm(q / rr[:, None] * 1.3 + idv, 2)
    edge = cell / 2 - (p - (idv + 0.5) * cell).abs().max(1).values
    d = torch.where(keep, torch.minimum(rock * 0.7, edge + cell * 0.2), edge + cell * 0.2)
    return torch.maximum(d, band)


def scene():
    sc = Scene()
    sc.add(sd_sphere(SUN_R), sun_mat(), T(*SUN))
    R = PLANETS["mercury"][3]
    sc.add(rocky("#5a5652", "#a8a29a", R), mercury_mat(R), T(*pos("mercury")))
    R = PLANETS["venus"][3]
    sc.add(sd_sphere(R), venus_mat(R), T(*pos("venus")))
    R = PLANETS["earth"][3]
    sc.add(sd_sphere(R), earth_mat(R), T(*pos("earth")) * RZ(0.41))
    ex, ey, ez = pos("earth")
    sc.add(rocky("#6a6660", "#b8b2a8", R * 0.27, 0.6), mercury_mat(R * 0.27), T(ex + 70, ey + 22, ez + 10))
    R = PLANETS["mars"][3]
    sc.add(sd_sphere(R), mars_mat(R), T(*pos("mars")) * RZ(0.44))
    R = PLANETS["jupiter"][3]
    sc.add(sd_sphere(R), giant_mat(R, ["#9a6a46", "#ecdcc0", "#c89a6a", "#6a4a34"], 26, (-0.36, 1.9, 0.12, "#b8502e")),
           T(*pos("jupiter")) * RZ(0.05))
    R = PLANETS["saturn"][3]
    S = T(*pos("saturn")) * RZ(0.35) * RX(0.42)
    sc.add(sd_sphere(R), giant_mat(R, ["#c8a878", "#ecdcb4", "#d8bc8c", "#a8885a"], 18, turb=0.05), S)
    sc.add(rings(R, 1.25, 2.35, 0.15), ring_mat(R, 1.25, 2.35), S)
    R = PLANETS["uranus"][3]
    U = T(*pos("uranus")) * RX(1.25) * RZ(0.3)
    sc.add(sd_sphere(R), giant_mat(R, ["#8ad0d8", "#b8ecf0", "#9adce4", "#7ab8c8"], 10, turb=0.04, rim_c="#c8f8ff"), U)
    sc.add(rings(R, 1.6, 2.0, 0.05), ring_mat(R, 1.6, 2.0, "#3a4a50", "#8a9aa0"), U)
    R = PLANETS["neptune"][3]
    sc.add(sd_sphere(R), giant_mat(R, ["#1a3a9a", "#3a6ad0", "#2a50b8", "#e8f0ff"], 14, (-0.3, 1.2, 0.12, "#0a1a5a"), "#8ab0ff", 0.08),
           T(*pos("neptune")))
    if False:
        sc.add(belt, Mat("#6a625a", rough=0.95, pattern=lambda pl, pw, n: {"albedo": torch.lerp(srgb("#3a3430"), srgb("#8a8278"), fbm(pw * 0.8, 3)[:, None])}))
    return sc


NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    cam = Camera(CAM, LOOK, fov=60)
    sx, sy, sz = SUN
    k = SUN_R * 2.2
    vols = [Volume(corona_density, (sx - k, sy - k, sz - k), (sx + k, sy + k, sz + k),
                   albedo="#000000", g=0.0, emission=corona_emission, steps=96, light_steps=0, ambient=0.0)]
    img = render_view(scene(), W, H, cam, space_sky, sun_dir=None, lights=[(SUN, "#fff4e4", 1e7)],
                      ambient=("#06070c", "#030305"), exposure=1.1, steps=260, tmax=4000, ao_scale=0.5, bloom=1.0,
                      vignette=0.25, grain=0.008, grade=("#1a2a4a", "#fff0dc"), ss=ss, volumes=[] if NOVOL else vols, vol_scale=0.5)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "espace.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
