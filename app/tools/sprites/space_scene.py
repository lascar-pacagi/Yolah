"""
space_scene.py — the Espace background in 3D (scenery.py): the night shore of
another world. From a rocky coast, a giant ocean planet rises over the sea;
a dark moon passes in front of it; the system's star burns small and white in
a green-cyan halo.

  the planet   a world of oceans, continents, ice and swirling clouds, lit
               from the left by the star, its air a blue rim at the limb;
               below the sea's horizon (analytic, in the sky)
  the moon     cratered grey rock, its left side lit, the right side faintly
               lit by the planet's blue light
  the star     a white core, a green-cyan glow, faint diffraction spikes
  the sky      the Milky Way, its dust lanes, stars of every colour
  the sea      swell and chop, the sky and the planet mirrored, foam at the
               wave crests and white surf around the rocks
  the shore    layered dark rock: a plateau on the right, slabs and
               outcrops standing in the surf, a heavy mass in the foreground;
               wet and glossy near the water, lichen on the tops, pools
               lights        the star (directional, cold white), the planet's
               blue light from the right (no shadow), a night ambient

    cd app/tools/sprites && python3 space_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, vec, norm, srgb, fbm, hash3, vnoise
from scenery import Camera, render_view, Volume

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
CAM_POS = (0.0, 6.5, 0.0)
CAM_LOOK = (0.0, 5.6, -60.0)
FOV = 56.0
_cam = Camera(CAM_POS, CAM_LOOK, fov=FOV)


def screen_dir(sx, sy):
    """The direction seen at screen (sx, sy) ∈ [-1, 1]² (16:9)."""
    d = _cam.f + sx * _cam.tan * 16 / 9 * _cam.r + sy * _cam.tan * _cam.u
    return norm(d)


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def ridged(p, o=4):
    return 1 - (2 * fbm(p, o) - 1).abs()


STAR = screen_dir(-0.66, 0.66)
PLANET = screen_dir(0.58, -0.02)          # the planet's centre (below the horizon: the sea hides its lower half)
PLANET_R = 0.62                           # angular radius (radians)
MOON = screen_dir(-0.16, 0.42)
MOON_R = 0.072
SEA_Y = 0.0
# the light on the planet and the moon: the star, as seen from them (from the
# viewer's left, a little behind): their faces lit, the terminators on the right
SKY_LIGHT = norm(vec(-0.85, 0.3, 0.3))


# ─── the sky: stars, the Milky Way, the star, the planet and the moon ───────
def _ray_sphere(d, c, r):
    """For unit rays d from the camera, a sphere of angular radius r around
    the unit direction c (at distance 1): hit mask, surface normal."""
    b = d @ c
    s = math.sin(r)
    disc = b * b - (1 - s * s)
    hit = (disc > 0) & (b > 0)
    t = b - torch.sqrt(disc.clamp_min(0))
    p = d * t[:, None]
    n = (p - c[None]) / s
    return hit, n, b, disc


def _rot(n, a, b):
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    x, z = ca * x + sa * z, -sa * x + ca * z
    y, z = cb * y + sb * z, -sb * y + cb * z
    return torch.stack([x, y, z], 1)


def planet_color(n, view):
    """The giant planet's radiance at surface normal n (world), for the view direction."""
    L = SKY_LIGHT
    ndl = n @ L
    mu = (-(n * view).sum(-1)).clamp(0, 1)
    q = _rot(n, 0.7, 0.35)                                  # the globe's own frame (its axis tilted)
    lat = q[:, 1].abs()
    cont = fbm(q * 3.2 + 3, 6) + 0.1 * (fbm(q * 14, 3) - 0.5)
    land = sstep(0.53, 0.56, cont)
    shelf = sstep(0.47, 0.53, cont) * (1 - land)
    cont = cont + 0.05 * (fbm(q * 40, 4) - 0.5) + 0.025 * (fbm(q * 120, 3) - 0.5)   # ragged coasts
    land = sstep(0.53, 0.545, cont)
    ground = torch.lerp(srgb("#24401c"), srgb("#4a5a2a"), fbm(q * 12, 5)[:, None])
    ground = torch.lerp(ground, srgb("#8a6a44"), (0.7 * sstep(0.6, 0.75, fbm(q * 5 + 5, 4)))[:, None])   # deserts
    ground = torch.lerp(ground, srgb("#b8b4ac"), (0.6 * sstep(0.62, 0.8, ridged(q * 10, 5)))[:, None])   # ranges
    ground = ground * (0.8 + 0.4 * fbm(q * 60, 3))[:, None]
    ocean = torch.lerp(srgb("#04152e"), srgb("#0c3a5a"), shelf[:, None]) * (0.85 + 0.3 * fbm(q * 20, 3))[:, None]
    a = torch.lerp(ocean, ground, land[:, None])
    ice = sstep(0.72, 0.82, lat + 0.08 * fbm(q * 5, 3))
    a = torch.lerp(a, srgb("#e8eef4"), ice[:, None])
    # clouds: cyclones, fronts and streets, sheared along the latitudes
    w = q + 0.22 * (torch.stack([fbm(q * 4, 4), fbm(q * 4 + 4, 4), fbm(q * 4 + 8, 4)], 1) - 0.5)
    w = w + 0.05 * (torch.stack([fbm(w * 18, 3), fbm(w * 18 + 4, 3), fbm(w * 18 + 8, 3)], 1) - 0.5)   # small eddies
    w = torch.stack([w[:, 0], w[:, 1] * 2.5, w[:, 2]], 1)
    cf = fbm(w * 6 + 1, 6) + 0.22 * (fbm(w * 30, 4) - 0.5) + 0.12 * (fbm(w * 90, 3) - 0.5)
    cl = sstep(0.56, 0.8, cf) * (0.45 + 0.55 * fbm(w * 40 + 3, 4))
    swirl = sstep(0.7, 0.9, ridged(w * 12 + 2, 4)) * sstep(0.5, 0.65, fbm(q * 3 + 9, 3))
    cl = (cl * 0.9 + 0.5 * swirl).clamp(0, 1)
    a = torch.lerp(a, srgb("#f4f6f8"), cl[:, None])
    lit = sstep(-0.12, 0.35, ndl) * ndl.clamp_min(0) ** 0.6
    glint = torch.exp(-((n @ norm(L - view.mean(0))) - 1).abs() * 60) * (1 - land) * (1 - cl) * 0.0
    col = a * (lit * 0.55)[:, None] + glint[:, None]
    # the air: blue haze thickening toward the limb, brightest on the lit side
    haze = (1 - mu) ** 2.2
    col = col * (1 - 0.55 * haze)[:, None] + srgb("#4a8ad0") * (haze * (0.05 + 0.45 * sstep(-0.3, 0.4, ndl)))[:, None]
    col = col + srgb("#0a1420") * 0.08                       # planetshine of its moons, the night side
    return col


def moon_color(n):
    ndl = n @ SKY_LIGHT
    q = _rot(n, 1.9, -0.4)
    a = torch.lerp(srgb("#2a2c30"), srgb("#6a6a6a"), fbm(q * 3 + 2, 5)[:, None])
    cr = ridged(q * 6 + 1, 3) ** 6
    a = a * (0.7 + 0.5 * cr)[:, None]
    a = torch.lerp(a, srgb("#1e2024"), (0.6 * sstep(0.55, 0.7, fbm(q * 2.2 + 7, 4)))[:, None])     # dark maria
    lit = sstep(-0.05, 0.3, ndl) * ndl.clamp_min(0) ** 0.8
    pl = (n @ PLANET).clamp_min(0)                           # blue light from the planet
    return a * (lit * 1.4)[:, None] + a * srgb("#4a7aaa") * (pl * 0.35)[:, None] + srgb("#06080c") * 0.2


def sky(d):
    N = d.shape[0]
    y = d[:, 1].clamp(-1, 1)
    el = torch.asin(y)
    az = torch.atan2(d[:, 0], -d[:, 2])
    c = srgb("#03050c").expand(N, 3).clone()
    c = c + srgb("#0a1a2a") * torch.exp(-el.clamp_min(0) / 0.25)[:, None] * 0.8     # the horizon's glow
    gn = norm(vec(-0.55, 0.62, 0.55))
    k = d @ gn
    band = torch.exp(-(k / 0.22) ** 2)
    lanes = ((fbm(d * 7 + 3, 5) - 0.5) * 4).clamp(0, 1) * torch.exp(-(k / 0.07) ** 2)
    glow = band * (0.4 + 0.6 * fbm(d * 4, 4)) * (1 - 0.75 * lanes)
    c = c + torch.lerp(srgb("#24384a"), srgb("#6a7a7a"), fbm(d * 3 + 7, 3)[:, None]) * (glow * 0.22)[:, None]
    neb = ((fbm(d * 2.5 + 11, 5) - 0.55) * 3).clamp(0, 1) * band
    c = c + srgb("#1a6a6a") * (neb * 0.06)[:, None]
    for (res, thr, gain) in ((1100, 0.982, 0.25), (500, 0.991, 0.9), (180, 0.996, 3.5)):
        cell = torch.floor(torch.stack([az * res / math.pi, el * res / math.pi, az * 0 + res], 1))
        h = hash3(cell)
        fx = (az * res / math.pi) - cell[:, 0] - hash3(cell + 1.7)
        fy = (el * res / math.pi) - cell[:, 1] - hash3(cell + 4.1)
        spot = torch.exp(-(fx * fx + fy * fy) / (0.02 if res > 300 else 0.006))
        on = (h > thr).float() * (1 + 2 * band * (res > 600))
        temp = hash3(cell + 9.3)
        col = torch.where((temp < 0.2)[:, None], srgb("#ffc29a"), torch.where((temp > 0.75)[:, None], srgb("#b0ccff"), srgb("#f4f6ff")))
        c = c + col * (on * spot * gain * (0.4 + (h - thr) / (1 - thr)))[:, None]
    c = c * sstep(-0.01, 0.02, el)[:, None] + srgb("#03050a") * (1 - sstep(-0.01, 0.02, el))[:, None]
    # the star: core, halo, green-cyan glow, spikes
    cs = (d @ STAR).clamp(-1, 1)
    ang = torch.acos(cs)
    core = torch.exp(-(ang / 0.006) ** 2) * 60 + torch.exp(-(ang / 0.02) ** 2) * 6
    halo = torch.exp(-ang / 0.045) * 0.8 + torch.exp(-ang / 0.18) * 0.22
    perp = d - STAR[None] * cs[:, None]
    sx_ = (perp @ _cam.r)
    sy_ = (perp @ _cam.u)
    spikes = (torch.exp(-(sx_ / 0.0012) ** 2) + torch.exp(-(sy_ / 0.0012) ** 2)) * torch.exp(-ang / 0.025) * 1.0
    c = c + srgb("#f4fbff") * core[:, None] + srgb("#5affd0") * (halo * 0.55)[:, None] + srgb("#a8fff0") * (halo * 0.25)[:, None] \
        + srgb("#e8fff8") * spikes[:, None]
    # the planet: its disk, and the air's glow beyond the limb
    hit, n, b, disc = _ray_sphere(d, PLANET, PLANET_R)
    pc = planet_color(n, d)
    impact = torch.sqrt((1 - b * b).clamp_min(0)) / math.sin(PLANET_R)        # 1 at the limb
    outside = (impact - 1).clamp_min(0)
    sidelit = sstep(-0.6, 0.6, (norm(d - PLANET[None] * b[:, None]) @ SKY_LIGHT))
    rim = torch.exp(-outside / 0.012) * (0.25 + 1.6 * sidelit) * (b > 0).float() * (~hit).float()
    c = torch.where(hit[:, None], pc + c * 0.0, c) + srgb("#5aa0ff") * (rim * 0.6)[:, None]
    # the moon (in front of the planet)
    mh, mn, _, _ = _ray_sphere(d, MOON, MOON_R)
    c = torch.where(mh[:, None], moon_color(mn), c)
    return c


# ─── the shore ──────────────────────────────────────────────────────────────
# rock masses: (cx, cz, rx, rz, top, edge, rot): a mesa with steep warped edges
MASSES = [
    (34.0, -70.0, 30.0, 52.0, 3.4, 4.0, 0.25),         # the plateau on the right
    (60.0, -150.0, 50.0, 60.0, 4.2, 6.0, 0.1),
    (12.0, -42.0, 7.0, 11.0, 3.0, 2.2, -0.4),           # the slab in the middle of the inlet
    (2.5, -62.0, 5.0, 4.0, 2.2, 1.6, 0.6),
    (-8.0, -16.0, 9.0, 7.0, 4.6, 2.6, 0.3),             # the heavy mass in the foreground, left
    (-17.0, -27.0, 6.0, 4.5, 2.6, 1.8, -0.5),
    (-24.0, -52.0, 3.5, 2.5, 1.6, 1.2, 0.2),            # outcrops in the surf
    (-12.0, -80.0, 4.0, 2.5, 1.2, 1.3, 0.9),
    (-40.0, -110.0, 5.0, 3.0, 1.4, 1.5, -0.3),
    (6.0, -24.0, 4.0, 3.0, 2.0, 1.4, 0.4),
    (20.0, -18.0, 12.0, 6.0, 2.4, 2.0, 0.2),            # the foreground rocks on the right
]


def rock_h(x, z):
    """Height of the rock (−4 = the sea floor) over (x, z)."""
    p2 = torch.stack([x, z * 0 + 1.3, z], 1)
    wx = x + 3.0 * (fbm(p2 * 0.06, 3) - 0.5)
    wz = z + 3.0 * (fbm(p2 * 0.06 + 5, 3) - 0.5)
    h = torch.full_like(x, -4.0)
    for (cx, cz, rx, rz, top, edge, rot) in MASSES:
        ca, sa = math.cos(rot), math.sin(rot)
        u = ((wx - cx) * ca - (wz - cz) * sa) / rx
        v = ((wx - cx) * sa + (wz - cz) * ca) / rz
        r = torch.sqrt(u * u + v * v)
        dist = (r - 1) * min(rx, rz)                          # roughly metres outside the outline
        m = top * (1 - 0.25 * r.clamp(0, 1) ** 2) * sstep(edge, -edge * 0.3, dist)
        h = torch.maximum(h, -4.0 + (m + 4.0) * sstep(edge * 1.2, 0, dist))
    p3 = torch.stack([x, h, z], 1)
    land = sstep(-1.5, 0.0, h)
    # bedding: ledges and sharp risers (the layers tilted a little)
    bed = (h + 0.04 * x - 0.03 * z + 0.3 * (fbm(p3 * 0.1, 2) - 0.5)) / 1.2
    fr = bed - torch.floor(bed)
    h = h + land * 1.2 * (sstep(0.8, 0.96, fr) - fr)
    # joints: the blocks split by near-vertical fractures; narrow cracks
    jx = (torch.sin(x * 0.22 + 2.5 * fbm(p3 * 0.08, 2)) ).abs()
    jz = (torch.sin(z * 0.19 + x * 0.07 + 2.5 * fbm(p3 * 0.08 + 3, 2))).abs()
    joint = torch.exp(-(torch.minimum(jx, jz) / 0.06) ** 2)
    crack = ridged(p3 * 0.5 + 2, 3) ** 14
    h = h - land * (0.35 * joint + 0.4 * crack) + 0.15 * (fbm(p3 * 0.5, 3) - 0.5) + 0.06 * (fbm(p3 * 3.0, 2) - 0.5)
    return h


def rock_sdf(p):
    h = rock_h(p[:, 0], p[:, 2])
    return (p[:, 1] - h) * 0.42 - 0.12 * (fbm(p * 0.9, 3) - 0.5) - 0.04 * (fbm(p * 4.0, 2) - 0.5)


def rock_mat():
    def pat(pl, pw, n):
        y = pw[:, 1]
        bed = fbm(pw * torch.tensor([0.4, 3.0, 0.4], device=pw.device), 3)
        a = torch.lerp(srgb("#24221f"), srgb("#4a4640"), bed[:, None])
        a = a * (0.75 + 0.45 * fbm(pw * 3.0, 4))[:, None]
        a = torch.lerp(a, srgb("#3e3a30"), (0.4 * sstep(0.55, 0.7, fbm(pw * 0.8 + 3, 3)))[:, None])
        top = sstep(0.6, 0.9, n[:, 1])
        lich = top * sstep(0.5, 0.65, fbm(pw * 1.3 + 7, 4)) * sstep(1.0, 2.0, y)
        a = torch.lerp(a, srgb("#5a5a3e"), (0.6 * lich)[:, None])
        wet = sstep(1.4, 0.3, y)
        a = a * (1 - 0.45 * wet)[:, None]
        algae = sstep(0.8, -0.2, y) * sstep(0.45, 0.6, fbm(pw * 2.0, 3))
        a = torch.lerp(a, srgb("#1e2a1a"), (0.6 * algae)[:, None])
        rough = 0.85 - 0.6 * wet
        return {"albedo": a, "rough": rough}
    return Mat("#3a3632", rough=0.85, pattern=pat)


# ─── the sea ────────────────────────────────────────────────────────────────
def wave_h(x, z):
    p = torch.stack([x, z * 0, z], 1)
    w = 0.32 * torch.sin(0.11 * x + 0.21 * z + 1.3 * fbm(p * 0.02, 2)) \
        + 0.18 * torch.sin(-0.17 * x + 0.29 * z + 2.0) \
        + 0.25 * (fbm(torch.stack([x * 0.12, z * 0, z * 0.18], 1), 4) - 0.5) \
        + 0.06 * (fbm(p * 0.9, 3) - 0.5)
    return w


def sea_sdf(p):
    return (p[:, 1] - SEA_Y - wave_h(p[:, 0], p[:, 2])) * 0.7


def sea_mat():
    def pat(pl, pw, n):
        x, z = pw[:, 0], pw[:, 2]
        h = rock_h(x, z)
        near = sstep(-2.6, -0.4, h)                                       # shallow by the rocks: surf
        edge = sstep(-1.4, -0.3, h)                                       # breaking against the rock
        wv = wave_h(x, z)
        lines = ridged(torch.stack([x * 0.35, z * 0 + 1.0, z * 0.7], 1) + 0.6 * fbm(pw * 0.2, 2)[:, None], 4) ** 5
        lace = near * sstep(0.35, 0.7, lines + 0.3 * fbm(pw * 2.0, 3))
        surf = edge * sstep(0.35, 0.6, fbm(pw * 1.2 + 2, 4))
        crest = sstep(0.3, 0.45, wv + 0.15 * fbm(pw * 0.6, 3)) * sstep(0.6, 0.78, fbm(pw * 0.3 + 4, 3))
        foam = (0.9 * surf + 0.7 * lace + 0.7 * crest).clamp(0, 1)
        a = torch.lerp(srgb("#020a12"), srgb("#08252e"), near[:, None])
        a = torch.lerp(a, srgb("#b4c0c8"), foam[:, None])
        rough = 0.16 + 0.75 * foam
        return {"albedo": a, "rough": rough}
    return Mat("#04101a", rough=0.16, pattern=pat)


def scene():
    sc = Scene()
    sc.add(rock_sdf, rock_mat())
    sc.add(sea_sdf, sea_mat())
    return sc


# ─── volumes: spray over the surf, mist on the sea ──────────────────────────
def spray_density(p):
    h = rock_h(p[:, 0], p[:, 2])
    near = sstep(-3.4, -0.8, h) * (1 - sstep(-0.5, 0.5, h))
    hgt = torch.exp(-(p[:, 1] - SEA_Y).clamp_min(0) / 1.2)
    return near * hgt * sstep(0.45, 0.7, fbm(p * 0.35 + 1, 4)) * 0.12


def mist_density(p):
    d = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
    return torch.exp(-(p[:, 1] / 6.0)) * sstep(150, 600, d) * (0.6 + 0.4 * fbm(p * 0.01, 3)) * 0.0025


NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    cam = Camera(CAM_POS, CAM_LOOK, fov=FOV)
    sd = STAR.tolist()
    far_planet = (PLANET * 1e5).tolist()
    vols = [Volume(spray_density, (-60, -1, -160), (40, 6, -8), albedo="#dfe6ec", g=0.4, steps=40, light_steps=0),
            Volume(mist_density, (-1500, -1, -3000), (1500, 30, -100), albedo="#9ab0c8", g=0.3, steps=24, light_steps=0)]
    img = render_view(scene(), W, H, cam, sky, sun_dir=sd, sun_col=srgb("#d8f4ff") * 0.8, sun_soft=0.04,
                      lights=[(far_planet, "#5a7aa8", -1e9)],
                      ambient=("#2a3a55", "#0c1018"), exposure=1.35, steps=300, tmax=3000, ao_scale=0.6, bloom=1.0,
                      vignette=0.3, grain=0.01, grade=("#0e1a2a", "#f0fff8"), ss=ss,
                      volumes=[] if NOVOL else vols, vol_scale=0.5)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "espace.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
