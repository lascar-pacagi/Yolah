"""
legend_scene.py — the Légendes background in 3D (scenery.py): a land of
legend at dusk, seen from a rocky height (an original world, no film's).

  the land      a great valley between ranges of jagged mountains (ridged
                noise, snow on the high crests), mist lying in the valleys,
                a river catching the last light
  the citadel   a white city climbing a spur of rock on the right: tiers of
                walls, towers with pointed roofs, a tall keep, warm windows
  the volcano   far on the left, its crater glowing, a plume of ash lit red
                from below, lava running down a flank
  the sky       the sun just setting (physical sky: the glow on the horizon,
                the Earth's shadow rising opposite), volumetric clouds caught
                in the last light, the first stars

    cd app/tools/sprites && python3 legend_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, norm, srgb, sd_sphere, sd_box, sd_cylinder, sd_cone, fbm, hash3
from scenery import Camera, render_view, Atmosphere, Volume

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
D = vec(0).device


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def ridged(p, o=5):
    return 1 - (2 * fbm(p, o) - 1).abs()


# ─── the land ───────────────────────────────────────────────────────────────
def land_h(x, z):
    q = torch.stack([x * 0.0009, x * 0 + 2, z * 0.0009], 1)
    mount = ridged(q, 7) ** 2.6 * 1100 + 160 * ridged(q * 3.1 + 7, 4) ** 2 + 40 * fbm(q * 9, 3)
    # the valley: low along a wandering line in front, the ranges rising away from it
    vx = x - (300 * torch.sin(z * 0.0012) - 200)
    valley = sstep(500, 2400, vx.abs()) * 0.9 + 0.1
    far = sstep(-500, -2200, z)
    h = mount * valley * (0.25 + 0.75 * far) - 60
    # the height the eye stands on: a rocky ledge in the foreground
    ledge = sstep(-60, -10, z)
    h = h * (1 - ledge) + (40 + 6 * fbm(torch.stack([x * 0.05, x * 0, z * 0.05], 1), 4)) * ledge
    return h


def land_at(x, z):
    return float(land_h(torch.tensor([float(x)], device=D), torch.tensor([float(z)], device=D))[0])


def land(p):
    return (p[:, 1] - land_h(p[:, 0], p[:, 2])) * 0.4


def land_mat():
    def pat(pl, pw, n):
        y = pw[:, 1]
        steep = (1 - n[:, 1]).clamp(0, 1)
        rock = torch.lerp(srgb("#3a3430"), srgb("#6a625a"), fbm(pw * 0.01, 4)[:, None]) * (0.7 + 0.4 * ridged(pw * 0.05, 3))[:, None]
        grass = torch.lerp(srgb("#2e3a1e"), srgb("#5a5a2e"), fbm(pw * 0.006 + 3, 4)[:, None])
        forest = srgb("#1e2a18") * (0.7 + 0.4 * fbm(pw * 0.05, 3))[:, None]
        low = sstep(350, 120, y)[:, None]
        veg = torch.lerp(grass, forest, ((fbm(pw * 0.004 + 6, 3) - 0.45) * 4).clamp(0, 1)[:, None])
        a = torch.lerp(rock, veg, low * (1 - sstep(0.35, 0.6, steep))[:, None])
        snow = sstep(520, 640, y + 80 * fbm(pw * 0.004, 3)) * sstep(0.75, 0.45, steep)
        a = torch.lerp(a, srgb("#e8ecf2") * (0.85 + 0.15 * fbm(pw * 0.03, 2))[:, None], snow[:, None])
        return {"albedo": a}
    return Mat("#4a4440", rough=0.9, pattern=pat)


def river(p):
    x, z = p[:, 0], p[:, 2]
    vx = x - (300 * torch.sin(z * 0.0012) - 200)
    w = 18 + 10 * fbm(torch.stack([z * 0.003, z * 0, z * 0], 1), 2)
    return torch.maximum(torch.maximum(vx.abs() - w, p[:, 1] + 58), -p[:, 1] - 60) * 0.8


# ─── the citadel ────────────────────────────────────────────────────────────
CIT = (560.0, -2700.0)


def stone_white():
    def pat(pl, pw, n):
        a = srgb("#d8d4cc") * (0.75 + 0.3 * fbm(pw * 0.2, 3))[:, None]
        course = ((pw[:, 1] * 0.5 - torch.floor(pw[:, 1] * 0.5)) < 0.05).float()[:, None]
        return {"albedo": a * (1 - 0.15 * course)}
    return Mat("#d8d4cc", rough=0.7, pattern=pat)


def windows():
    def pat(pl, pw, n):
        ang = torch.atan2(pl[:, 0], pl[:, 2])
        col = ((ang * 4.0 / math.pi * 3 - torch.floor(ang * 4.0 / math.pi * 3)) < 0.22).float()
        row = ((pw[:, 1] / 9.0 - torch.floor(pw[:, 1] / 9.0)) < 0.3).float()
        lit = (hash3(torch.floor(torch.stack([ang * 6, pw[:, 1] / 9.0, ang * 0], 1))) > 0.45).float()
        em = srgb("#ffb860") * (col * row * lit * 4.0)[:, None]
        return {"albedo": srgb("#cfcac0").expand(pw.shape[0], 3) * (1 - 0.5 * (col * row)[:, None]), "emissive": em}
    return Mat("#cfcac0", rough=0.7, pattern=pat)


def citadel(sc):
    cx, cz = CIT
    base = land_at(cx, cz)
    st, win = stone_white(), windows()
    roof = Mat("#2a3440", metal=0.3, rough=0.5)
    X = T(cx, base, cz)
    # the spur of rock it stands on
    sc.add(lambda q: sd_cone(170, 230, 110)(q) - 45 * ridged(q * 0.008, 5) - 20 * fbm(q * 0.03, 3), land_mat(), X * T(0, 110, 0), blend=60)
    # seven tiers of walls, each a ring a little higher and narrower
    for k in range(7):
        r = 150 - k * 18
        y = 260 + k * 24
        sc.add(sd_cylinder(13, r, 2), win, X * T(0, y, 0))
        sc.add(lambda q, r=r: torch.maximum(sd_cylinder(3, r + 3, 1)(q), -sd_cylinder(4, r - 2)(q)), st, X * T(0, y + 14, 0))
        # towers on the ring
        for i in range(6 + k):
            a = i * math.tau / (6 + k) + k * 0.4
            tx, tz = math.cos(a) * r, math.sin(a) * r
            sc.add(sd_cylinder(18, 6), win, X * T(tx, y + 6, tz))
            sc.add(sd_cone(7, 7.5, 0.3), roof, X * T(tx, y + 31, tz))
    # the keep: a tall white tower and its spire
    sc.add(sd_cylinder(70, 16, 2), win, X * T(0, 460, 0))
    sc.add(sd_cone(28, 18, 0.5), roof, X * T(0, 558, 0))
    sc.add(sd_sphere(3.5), Mat("#ffe0a0", emissive="#ffd890", emit=12.0), X * T(0, 590, 0))


# ─── the volcano and the air ────────────────────────────────────────────────
VOL = (-3200.0, -8200.0)


def volcano(p):
    x, z = p[:, 0] - VOL[0], p[:, 2] - VOL[1]
    r = torch.sqrt(x * x + z * z)
    h = 2500 - r * 0.62 + 80 * ridged(p * 0.002, 4) - 260 * torch.exp(-(r / 180) ** 2)
    return (p[:, 1] - h.clamp(max=2300)) * 0.35


def volcano_mat():
    def pat(pl, pw, n):
        x, z = pw[:, 0] - VOL[0], pw[:, 2] - VOL[1]
        r = torch.sqrt(x * x + z * z)
        a = srgb("#2a2420") * (0.6 + 0.6 * fbm(pw * 0.004, 4))[:, None]
        az = torch.atan2(x, z)
        lava = torch.exp(-((az - 0.35 - 0.1 * torch.sin(r * 0.004)) / 0.04) ** 2) * sstep(2200, 300, r)
        crater = torch.exp(-((r - 190) / 60) ** 2) * (pw[:, 1] > 2150).float()
        em = srgb("#ff4a10") * ((lava * 3 + crater * 8) * (0.6 + 0.8 * fbm(pw * 0.01, 2)))[:, None]
        return {"albedo": a, "emissive": em}
    return Mat("#2a2420", rough=0.85, pattern=pat)


def ash(p):
    x, y, z = p[:, 0] - VOL[0], p[:, 1] - 2300, p[:, 2] - VOL[1]
    H = 1800.0
    u = (y / H).clamp(0, 1.2)
    rx = x - 1200 * u * u
    r = torch.sqrt(rx * rx + z * z)
    rad = 120 + 500 * u
    shape = (1 - r / rad) * sstep(1.2, 0.9, u) * (y > -40).float()
    n = fbm(p * 0.0025 + vec(0, -0.2, 0), 5)
    return (shape + (n - 0.5) * 1.2).clamp_min(0) * 0.012


def ash_glow(p, s):
    y = p[:, 1] - 2300
    return srgb("#ff4a18") * (torch.exp(-y.clamp_min(0) / 350) * 1.8)[:, None]


def clouds(p):
    y = p[:, 1]
    q = p * vec(0.00035, 0.0009, 0.00035)
    layer = sstep(2600, 2900, y) * sstep(4200, 3500, y)
    n = fbm(q + vec(4, 0, 1), 6)
    return ((n - 0.52) * 5).clamp_min(0) * layer * 0.004


def valley_mist(p):
    g = land_h(p[:, 0], p[:, 2])
    n = fbm(p * vec(0.002, 0.01, 0.002), 4)
    return torch.exp(-(p[:, 1] - g).clamp_min(0) / 25.0) * sstep(20, -50, g) * ((n - 0.35) * 2).clamp_min(0) * 0.003


def scene():
    sc = Scene()
    sc.add(land, land_mat())
    sc.add(river, Mat("#0a1218", rough=0.05, clearcoat=1.0))
    sc.add(volcano, volcano_mat())
    citadel(sc)
    return sc


NOVOL = os.environ.get("NOVOL") is not None


def render(W=2560, H=1440, ss=1):
    el, az = math.radians(1.8), math.radians(-62)
    sun = (math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el))
    atmo = Atmosphere(sun, sun_power=26.0, mie=40e-6, mie_g=0.8, ground="#3a3a2a")
    cam = Camera((0.0, land_at(0, 0) + 90, 0.0), (120, 120, -2000), fov=58)
    vols = [Volume(ash, (VOL[0] - 900, 2200, VOL[1] - 900), (VOL[0] + 2600, 4400, VOL[1] + 900), albedo="#6a5e58", g=0.45,
                   emission=ash_glow, steps=56, light_steps=4, light_dist=600),
            Volume(clouds, (-30000, 2600, -40000), (30000, 4200, 2000), albedo="#ffffff", g=0.7, steps=40, light_steps=5, light_dist=1200),
            Volume(valley_mist, (-3000, -80, -6000), (3000, 200, -200), albedo="#c8ccd8", g=0.5, steps=40, light_steps=2, light_dist=80)]
    lights = [((VOL[0], 2450, VOL[1]), "#ff5a1a", 2600.0), ((CIT[0], land_at(*CIT) + 560, CIT[1]), "#ffc070", 260.0)]
    img = render_view(scene(), W, H, cam, None, sun_dir=sun, atmo=atmo, aerial=(0.00012, 0.0008, 0.0), sun_soft=0.03,
                      lights=lights, exposure=1.6, steps=420, tmax=14000, ao_scale=3.0, bloom=0.8, dof=None, vignette=0.32,
                      grain=0.012, grade=("#3a3a6a", "#ffd8b0"), ss=ss, volumes=[] if NOVOL else vols)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "legendes.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
