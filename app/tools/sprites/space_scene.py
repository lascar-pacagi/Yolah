"""
space_scene.py — the Espace background in 3D (scenery.py): deep space seen
from the little planet of the board (the planet itself is drawn by the app,
assets/backgrounds/planete.png, under the board).

  the giant     a gas giant lit from the side by a distant star: belts and
                zones bent by turbulence, a great storm, the night side dark;
                its rings (bands, a dark division) casting their shadow
  the moons     two small cratered moons, one passing in front of the rings
  the nebula    an emission nebula far away: glowing hydrogen (red), oxygen
                (teal), lanes of dark dust in front, lit at the edges
  the sky       a dense star field with colours and sizes, the band of the
                galaxy, a small spiral galaxy, a few bright stars with glare

    cd app/tools/sprites && python3 space_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, norm, srgb, sd_sphere, sd_cylinder, fbm, hash3
from scenery import Camera, render_view, Volume

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
SUN = (0.88, 0.42, -0.2)                  # towards the distant star (right, a little behind)
GIANT = (-170.0, 70.0, -460.0)
GR = 70.0


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def space_sky(d):
    y = d[:, 1].clamp(-1, 1)
    el = torch.asin(y)
    az = torch.atan2(d[:, 0], -d[:, 2])
    c = srgb("#020309").expand(d.shape[0], 3).clone()
    # the band of the galaxy: soft glow, dark dust lanes, a haze of faint stars
    gn = norm(vec(0.25, 0.85, 0.46))
    k = d @ gn
    band = torch.exp(-(k / 0.22) ** 2)
    lanes = ((fbm(d * 7 + 3, 5) - 0.5) * 4).clamp(0, 1) * torch.exp(-(k / 0.06) ** 2)
    glow = band * (0.4 + 0.6 * fbm(d * 4, 4)) * (1 - 0.75 * lanes)
    c = c + torch.lerp(srgb("#3a3050"), srgb("#8a7a6a"), fbm(d * 3 + 7, 3)[:, None]) * (glow * 0.22)[:, None]
    # stars: three layers of cells (faint, medium, bright), colours from red giants to blue
    for (res, thr, gain) in ((900, 0.985, 0.35), (420, 0.992, 1.2), (160, 0.996, 5.0)):
        cell = torch.floor(torch.stack([az * res / math.pi, el * res / math.pi, az * 0 + res], 1))
        h = hash3(cell)
        # the star sits at a random point of its cell: a small gaussian
        fx = (az * res / math.pi) - cell[:, 0] - hash3(cell + 1.7)
        fy = (el * res / math.pi) - cell[:, 1] - hash3(cell + 4.1)
        spot = torch.exp(-(fx * fx + fy * fy) / (0.02 if res > 300 else 0.006))
        on = (h > thr).float() * (1 + 3 * band * (res > 500))
        temp = hash3(cell + 9.3)
        col = torch.where((temp < 0.2)[:, None], srgb("#ffb27a"), torch.where((temp > 0.8)[:, None], srgb("#a8c4ff"), srgb("#fff4e8")))
        c = c + col * (on * spot * gain * (0.4 + (h - thr) / (1 - thr)))[:, None]
    # a small spiral galaxy, far away
    gd = norm(vec(0.42, 0.42, -0.8))
    u = norm(torch.linalg.cross(gd, vec(0, 1, 0)))
    v = torch.linalg.cross(u, gd)
    px, py = (d @ u) / 0.045, (d @ v) / 0.022
    r = torch.sqrt(px * px + py * py)
    th = torch.atan2(py, px)
    arms = (0.5 + 0.5 * torch.cos(2 * th - 4.0 * torch.log(r + 0.05))) ** 3
    gal = torch.exp(-r * 2.2) * (0.35 + 0.65 * arms) + torch.exp(-(r / 0.12) ** 2) * 1.5
    c = c + srgb("#e8d8c0") * (gal * 0.5 * (d @ gd > 0.9).float())[:, None]
    return c


def giant_mat():
    def pat(pl, pw, n):
        q = pl / GR
        lat = q[:, 1]
        warp = 0.12 * (fbm(q * 3, 5) - 0.5) + 0.05 * (fbm(q * 14 + 3, 4) - 0.5)
        b = lat + warp
        bands = torch.sin(b * 22) * 0.5 + 0.5
        fine = torch.sin(b * 70 + 3 * fbm(q * 6, 3)) * 0.5 + 0.5
        base = torch.lerp(srgb("#a87a52"), srgb("#e6d6ba"), bands[:, None]) * (0.85 + 0.25 * fbm(q * 20, 3))[:, None]
        base = torch.lerp(base, srgb("#8a5a3a"), (0.25 * fine)[:, None])
        base = torch.lerp(base, srgb("#c8b8a8"), sstep(0.75, 0.95, lat.abs())[:, None])        # pale polar regions
        # the great storm: an oval with spiral bands
        sx, sy = (q[:, 0] - 0.35) / 0.18, (lat + 0.32) / 0.08
        sr = torch.sqrt(sx * sx + sy * sy)
        storm = torch.exp(-sr ** 2 * 1.5) * (q[:, 2] > 0).float()
        swirl = 0.5 + 0.5 * torch.sin(torch.atan2(sy, sx) * 2 + sr * 6)
        base = torch.lerp(base, srgb("#c0603a") * (0.8 + 0.3 * swirl)[:, None], storm[:, None])
        return {"albedo": base}
    return Mat("#d8b890", rough=0.95, pattern=pat)


def rings(p):
    """Thin flat rings in the giant's equatorial plane: radii 1.35..2.3 R, a dark division."""
    r = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) / GR
    slab = p[:, 1].abs() - 0.15
    ann = torch.maximum(1.35 - r, r - 2.3) * GR
    div = -((r - 1.95).abs() - 0.035) * GR            # the division: empty
    return torch.maximum(torch.maximum(slab, ann), div) * 0.9


def ring_mat():
    def pat(pl, pw, n):
        r = torch.sqrt(pl[:, 0] ** 2 + pl[:, 2] ** 2) / GR
        b = torch.sin(r * 140) * 0.5 + 0.5
        b2 = fbm(torch.stack([r * 60, r * 0, r * 0], 1), 3)
        a = torch.lerp(srgb("#6a5a4a"), srgb("#e0d0b8"), (0.4 * b + 0.8 * b2 - 0.2).clamp(0, 1)[:, None])
        a = a * sstep(1.35, 1.5, r)[:, None] * (0.6 + 0.4 * sstep(2.3, 2.0, r))[:, None]
        return {"albedo": a}
    return Mat("#c8b8a0", rough=0.9, pattern=pat)


def moon_mat(c="#9a948c"):
    def pat(pl, pw, n):
        a = srgb(c) * (0.6 + 0.5 * fbm(pl * 0.4, 4))[:, None]
        return {"albedo": a}
    return Mat(c, rough=0.95, pattern=pat)


def moon(R):
    def f(q):
        d = sd_sphere(R)(q)
        cr = 1 - (2 * fbm(q * (2.5 / R), 3) - 1).abs()
        return d - 0.03 * R * cr ** 4 + 0.02 * R * fbm(q * (6 / R), 2)
    return f


def scene():
    sc = Scene()
    G = T(*GIANT) * RZ(0.3) * RX(0.5)
    sc.add(sd_sphere(GR), giant_mat(), G)
    sc.add(rings, ring_mat(), G)
    sc.add(moon(9.0), moon_mat(), T(GIANT[0] + 170, GIANT[1] - 40, GIANT[2] + 120))
    sc.add(moon(4.0), moon_mat("#b8a890"), T(GIANT[0] + 60, GIANT[1] + 95, GIANT[2] + 60))
    return sc


# the emission nebula, far behind the giant on the right
NEB = (900.0, 260.0, -3800.0)


def ridged(p, o=4):
    return 1 - (2 * fbm(p, o) - 1).abs()


def neb_shape(p):
    q = (p - vec(*NEB)) / vec(2300, 1200, 1100)
    w = torch.stack([fbm(p * 0.0005 + 1, 4), fbm(p * 0.0005 + 4, 4), fbm(p * 0.0005 + 8, 4)], 1) - 0.5
    r = torch.sqrt(((q + 1.2 * w) ** 2).sum(-1))
    return torch.exp(-r * r * 2.5)


def neb_glow_density(p):
    shape = neb_shape(p)
    fil = ridged(p * 0.0012 + 2, 5) ** 3 + 0.6 * ridged(p * 0.0035 + 5, 3) ** 4
    cloud = fbm(p * 0.0008 + 7, 4)
    return (shape * (0.15 * cloud + 1.2 * fil) * shape).clamp_min(0) * 0.0016


def neb_glow(p, s):
    a = fbm(p * 0.0007 + 5, 4)
    k = ((a - 0.32) * 2.6).clamp(0, 1)[:, None]
    col = torch.lerp(srgb("#20b8c8"), srgb("#ff2a50"), k)
    col = torch.lerp(col, srgb("#9a50ff"), ((fbm(p * 0.0012 + 3, 3) - 0.55) * 4).clamp(0, 1)[:, None] * 0.6)
    hot = ((fbm(p * 0.002 + 9, 3) - 0.6) * 5).clamp(0, 1)[:, None] * srgb("#fff0e0") * 0.8
    return (col * 1.4 + hot) * 1.6


def neb_dust(p):
    shape = neb_shape(p - vec(150, 80, -400))
    lanes = ridged(p * 0.0014 + 4, 5) ** 6
    wisp = fbm(p * 0.004 + 2, 3)
    return (shape * lanes * (0.6 + wisp) - 0.05).clamp_min(0) * 0.004


def render(W=2560, H=1440, ss=1):
    cam = Camera((0, 0, 0), (60, 60, -1000), fov=58)
    vols = [Volume(neb_glow_density, (NEB[0] - 2700, NEB[1] - 1700, NEB[2] - 1500), (NEB[0] + 2700, NEB[1] + 1700, NEB[2] + 1500),
                   albedo="#000000", g=0.3, emission=neb_glow, steps=96, light_steps=0, ambient=0.0),
            Volume(neb_dust, (NEB[0] - 2600, NEB[1] - 1600, NEB[2] - 1100), (NEB[0] + 2900, NEB[1] + 1800, NEB[2] + 1700),
                   albedo="#3a2a24", g=0.6, steps=40, light_steps=3, light_dist=400, ambient=0.0)]
    img = render_view(scene(), W, H, cam, space_sky, sun_dir=SUN, sun_col="#fff6ea", sun_soft=0.002,
                      ambient=("#05060a", "#020203"), exposure=1.2, steps=200, tmax=6000, ao_scale=0.0001, bloom=0.8,
                      vignette=0.25, grain=0.01, grade=("#2a3a6a", "#fff0e0"), ss=ss, volumes=vols, vol_scale=0.5)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "espace.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
