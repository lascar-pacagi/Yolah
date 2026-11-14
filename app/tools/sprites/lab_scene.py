"""
lab_scene.py — the Laboratoire's background in 3D (scenery.py): the reactor
hall of a research reactor seen from the edge of its pool.

  the hall      raw concrete, pilasters every five metres, a ceiling of steel
                trusses, rows of high-bay lamps
  the pool      a deep well of white tiles; under the water, the core's grid of
                fuel assemblies glowing with Cherenkov blue, the light
                rippling on the surface and thrown on the walls
  the crane     the yellow bridge of the polar crane over the pool, its hook
  the catwalk   a steel grating bridge across the pool, yellow railings
  the walls     bundles of pipes (grey, red, green), cable trays, valves,
                yellow drums, a hatch, hazard paint around the pool
  the back      the control room's long window, warm screens behind it

    cd app/tools/sprites && python3 lab_scene.py [W H]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_box, sd_capsule, sd_cylinder, sd_torus, fbm
from scenery import Camera, render_view, Volume

def fract(x):
    """x - floor(x): in [0, 1) also for negative x (torch.frac keeps the sign)."""
    return x - torch.floor(x)


ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


def concrete(c="#7a7c7e", k=1.2, stains=0.35):
    base = srgb(c)

    def pat(pl, pw, n):
        a = base * (0.78 + 0.3 * fbm(pw * k * 4, 4))[:, None]
        streak = fbm(pw * vec(k * 6, k * 0.6, k * 6), 3)                      # water streaks running down
        a = a * (1 - stains * ((streak > 0.6).float() * (streak - 0.6) * 2.5)[:, None])
        tie = ((fract(pw[:, 0] * 0.8) < 0.012) | (fract(pw[:, 1] * 0.8) < 0.012)).float()   # formwork joints
        return {"albedo": a * (1 - 0.25 * tie)[:, None], "rough": 0.85 + 0.1 * fbm(pw * 9, 2)}
    return Mat(c, rough=0.9, pattern=pat)


def painted_floor():
    def pat(pl, pw, n):
        x, z = pw[:, 0], pw[:, 2]
        a = srgb("#5a6066") * (0.75 + 0.3 * fbm(pw * 3, 4))[:, None]
        wear = (fbm(pw * vec(1.5, 1, 6), 3) > 0.62).float()
        a = a * (1 - 0.25 * wear)[:, None]
        # hazard stripes around the pool (x in ±7.2, z in −27.2..−8.8), outside the well
        inx, inz = (x.abs() < 7.4), ((z > -27.4) & (z < -8.6))
        band = (inx & inz & ~((x.abs() < 6.6) & (z > -26.6) & (z < -9.4))).float()
        stripe = (fract((x + z) * 0.9) < 0.5).float()
        a = a * (1 - band)[:, None] + band[:, None] * (srgb("#d8a818") * stripe[:, None] + srgb("#141414") * (1 - stripe)[:, None])
        # a painted walkway line
        line = ((x.abs() - 9.0).abs() < 0.08).float()
        a = a * (1 - line)[:, None] + srgb("#c8b030") * line[:, None]
        return {"albedo": a, "rough": 0.35 + 0.4 * fbm(pw * 4, 2)}
    return Mat("#5a6066", rough=0.5, clearcoat=0.3, pattern=pat)


def tiles():
    def pat(pl, pw, n):
        u = pw[:, 0] * 4 + pw[:, 2] * 4
        v = pw[:, 1] * 4
        grout = ((fract(u) < 0.04) | (fract(v) < 0.04)).float()
        deep = ((-pw[:, 1] - 0.5) / 7).clamp(0, 1)[:, None]
        a = srgb("#e8eef2") * (1 - 0.35 * grout)[:, None]
        # the pool's light: the walls glow blue, more at depth
        em = srgb("#1a7aff") * (0.05 + 0.9 * deep)
        return {"albedo": a * (1 - 0.5 * deep), "emissive": em}
    return Mat("#e8eef2", rough=0.2, clearcoat=0.8, pattern=pat)


def water():
    def pat(pl, pw, n):
        x, z = pw[:, 0], pw[:, 2]
        # the core seen through the water: a grid of fuel assemblies, the brightest at the centre
        cx, cz = x, z + 18.0
        core = torch.exp(-((cx / 2.6) ** 2 + (cz / 2.6) ** 2))
        cell = ((fract(cx * 1.6) < 0.82) & (fract(cz * 1.6) < 0.82)).float()
        racks = ((cx.abs() < 3.4) & (cz.abs() < 3.4)).float()
        ripple = 0.75 + 0.5 * fbm(pw * vec(2.2, 1, 2.2) + vec(0, 0, 0), 4)
        caust = (torch.sin(x * 7 + 3 * fbm(pw * 1.5, 2) * 6) * torch.sin(z * 6 + 3 * fbm(pw * 1.3 + 4, 2) * 6)).abs() ** 6
        rack_glow = cell * racks * (0.35 + 0.65 * fbm(torch.stack([torch.floor(cx * 1.6), torch.floor(cz * 1.6), cx * 0], 1) * 0.7, 1))
        glow = srgb("#2a9cff") * (0.08 + 7.0 * core * (0.25 + 0.75 * rack_glow) + 0.6 * caust * (0.2 + core))[:, None] * ripple[:, None]
        far = srgb("#041a4a") * 0.5
        return {"albedo": srgb("#06142a").expand(x.shape[0], 3), "emissive": glow + far}
    return Mat("#06142a", rough=0.03, clearcoat=1.0, pattern=pat)


def steel(c="#8a9098", rough=0.35):
    return Mat(c, metal=0.9, rough=rough, pattern=lambda pl, pw, n: {"albedo": srgb(c) * (0.75 + 0.35 * fbm(pw * 6, 3))[:, None]})


def yellow_paint(c="#e0a81a"):
    return Mat(c, rough=0.45, clearcoat=0.4, pattern=lambda pl, pw, n: {
        "albedo": srgb(c) * (0.8 + 0.25 * fbm(pw * 5, 3))[:, None] * (1 - 0.4 * (fbm(pw * 14, 2) > 0.72).float())[:, None]})


def repeat_x(f, period, lo, hi):
    """f repeated along x every `period` between lo and hi (domain repetition)."""
    def g(p):
        q = p.clone()
        c = torch.round((p[:, 0] - lo) / period).clamp(0, round((hi - lo) / period))
        q[:, 0] = p[:, 0] - (lo + c * period)
        return f(q)
    return g


def repeat_z(f, period, lo, hi):
    def g(p):
        q = p.clone()
        c = torch.round((p[:, 2] - lo) / period).clamp(0, round((hi - lo) / period))
        q[:, 2] = p[:, 2] - (lo + c * period)
        return f(q)
    return g


def scene():
    sc = Scene()
    conc = concrete()
    # floor, the pool cut into it
    sc.add(sd_box(30, 1.0, 40), painted_floor(), T(0, -1.0, -20))
    sc.add(sd_box(6.6, 9.5, 8.6), tiles(), T(0, -8.5, -18), op="sub")
    sc.add(sd_box(6.6, 0.05, 8.6), water(), T(0, -0.7, -18))
    sc.add(sd_box(6.6, 0.4, 8.6), tiles(), T(0, -8.0, -18))          # the bottom (unseen, closes the well)
    # walls and pilasters
    for sgn in (-1, 1):
        sc.add(sd_box(1.0, 14, 40), conc, T(sgn * 15, 12, -20))
        sc.add(repeat_z(sd_box(0.7, 14, 0.6), 5.0, -38, 2), concrete("#6e7072"), T(sgn * 14.0, 12, 0))
    sc.add(sd_box(30, 14, 1.0), conc, T(0, 12, -41))
    # the control room's long window in the back wall, warm screens behind
    def screens():
        def pat(pl, pw, n):
            x = pw[:, 0]
            cols = (fract(x * 0.55) < 0.8).float()
            lines = (fract(pw[:, 1] * 6) < 0.6).float()
            em = (srgb("#ffb860") * 1.6 * cols[:, None] * (0.5 + 0.5 * lines[:, None]) + srgb("#40ffa0") * 0.6 * (1 - cols[:, None])) * (0.7 + 0.6 * fbm(pw * 2, 2))[:, None]
            return {"albedo": srgb("#101418").expand(x.shape[0], 3), "emissive": em}
        return Mat("#101418", rough=0.05, clearcoat=1.0, pattern=pat)
    sc.add(sd_box(9, 1.3, 0.3), screens(), T(0, 7.2, -39.9))
    sc.add(sd_box(9.4, 0.12, 0.6), steel("#3a3e44"), T(0, 5.85, -39.7))
    sc.add(repeat_x(sd_box(0.08, 1.3, 0.35), 1.8, -9, 9), steel("#2a2e34"), T(0, 7.2, -39.6))
    # the ceiling trusses
    st = steel("#5a6068", 0.5)
    sc.add(repeat_z(sd_box(15, 0.35, 0.25), 5.0, -38, 2), st, T(0, 25.5, 0))
    sc.add(repeat_z(sd_box(15, 0.08, 0.08), 5.0, -38, 2), st, T(0, 23.6, 0))
    sc.add(repeat_z(repeat_x(sd_capsule((0, 0, 0), (1.25, 1.9, 0), 0.06), 2.5, -14, 12), 5.0, -38, 2), st, T(0, 23.6, 0))
    sc.add(sd_box(30, 0.5, 40), concrete("#3a3c40"), T(0, 26.5, -20))
    # high-bay lamps
    lamp = Mat("#fff6e8", emissive="#fff2dc", emit=1.0)
    lamp_m = Mat("#fff6e8", rough=0.2, emissive="#fff2dc", emit=14.0)
    sc.add(repeat_z(repeat_x(sd_cylinder(0.12, 0.55, 0.05), 7.0, -10.5, 10.5), 5.0, -36, -1), lamp_m, T(0, 22.8, 0))
    sc.add(repeat_z(repeat_x(sd_cylinder(0.35, 0.6, 0.08), 7.0, -10.5, 10.5), 5.0, -36, -1), steel("#2a2e34"), T(0, 23.2, 0))
    # the polar crane: two yellow box girders across the hall, the trolley, the hook
    yp = yellow_paint()
    for z in (-16.5, -19.5):
        sc.add(sd_box(14.5, 0.9, 0.45, 0.05), yp, T(0, 17.5, z))
        sc.add(repeat_x(sd_box(0.06, 0.7, 0.47), 1.2, -14, 14), Mat("#b88a14", rough=0.5), T(0, 17.5, z))
    for sgn in (-1, 1):
        sc.add(sd_box(0.6, 0.8, 2.6), yp, T(sgn * 13.8, 17.6, -18))
        sc.add(sd_box(0.25, 0.25, 40), steel("#4a4e54"), T(sgn * 13.8, 16.6, -20))           # the rails
    sc.add(sd_box(2.2, 0.7, 2.2, 0.05), yp, T(2.0, 18.8, -18))
    sc.add(sd_box(1.2, 0.6, 1.2, 0.05), Mat("#2a2a2e", metal=0.6, rough=0.4), T(2.0, 18.0, -18))
    for dx in (-0.25, 0.25):
        sc.add(sd_capsule((2.0 + dx, 17.6, -18), (2.0 + dx, 9.2, -18), 0.035), steel("#c8ccd0", 0.25))
    sc.add(sd_box(0.55, 0.6, 0.35, 0.08), yp, T(2.0, 8.8, -18))
    sc.add(sd_torus(0.32, 0.09), steel("#d8dce0", 0.25), T(2.0, 7.9, -18) * RX(math.pi / 2))
    # the catwalk across the pool: grating, side beams, yellow railings
    grating = Mat("#6a7078", metal=0.8, rough=0.45, pattern=lambda pl, pw, n: {
        "albedo": srgb("#6a7078") * (0.5 + 0.6 * ((fract(pw[:, 0] * 12) < 0.3) | (fract(pw[:, 2] * 4) < 0.15)).float())[:, None]})
    zc = -12.0
    sc.add(sd_box(9.0, 0.06, 0.9), grating, T(0, 1.1, zc))
    for sgn in (-1, 1):
        sc.add(sd_box(9.0, 0.18, 0.06), steel("#3e444c"), T(0, 0.98, zc + sgn * 0.92))
        sc.add(sd_capsule((-9, 2.2, zc + sgn * 0.9), (9, 2.2, zc + sgn * 0.9), 0.04), yp)
        sc.add(sd_capsule((-9, 1.65, zc + sgn * 0.9), (9, 1.65, zc + sgn * 0.9), 0.025), yp)
        sc.add(repeat_x(sd_capsule((0, 1.1, 0), (0, 2.2, 0), 0.035), 1.5, -9, 9), yp, T(0, 0, zc + sgn * 0.9))
    for sgn in (-1, 1):
        sc.add(sd_box(0.25, 0.55, 0.25), steel("#3e444c"), T(sgn * 7.6, 0.5, zc))
    # the refuelling machine: a yellow gantry on rails over the pool, its mast down into the water
    for sgn in (-1, 1):
        sc.add(sd_box(0.3, 0.12, 9.2), steel("#4a4e54"), T(sgn * 7.0, 0.12, -18))
        sc.add(sd_box(0.5, 2.2, 0.6, 0.05), yp, T(sgn * 6.8, 2.3, -20.5))
    sc.add(sd_box(7.4, 0.55, 0.9, 0.06), yp, T(0, 4.5, -20.5))
    sc.add(sd_box(1.6, 0.9, 1.4, 0.06), yp, T(-1.2, 5.6, -20.5))
    sc.add(sd_box(1.5, 0.75, 1.3, 0.04), Mat("#1a2028", rough=0.05, clearcoat=1.0, emissive="#ffd8a0", emit=0.25), T(-1.2, 5.65, -19.9))   # the operator's cabin
    sc.add(sd_box(0.35, 3.0, 0.35, 0.03), steel("#c8ccd0", 0.25), T(1.5, 1.5, -20.5))
    sc.add(sd_box(0.5, 0.25, 0.5, 0.03), yp, T(1.5, 3.9, -20.5))
    # railings around the pool
    for sgn in (-1, 1):
        sc.add(sd_capsule((sgn * 7.0, 1.1, -27), (sgn * 7.0, 1.1, -13), 0.04), yp)
        sc.add(repeat_z(sd_capsule((0, 0, 0), (0, 1.1, 0), 0.03), 1.6, -27, -13), yp, T(sgn * 7.0, 0, 0))
    # the pipe bundles along the walls, cable trays, valves
    cols = [("#8a9098", 0.25), ("#a83a2e", 0.18), ("#3a7a4a", 0.16), ("#8a9098", 0.32), ("#c8ccd0", 0.12)]
    for sgn in (-1, 1):
        for i, (c, r) in enumerate(cols):
            y = 5.0 + i * 0.75
            x = sgn * (13.3 - 0.1 * i)
            sc.add(sd_capsule((x, y, 1.0), (x, y, -40), r), Mat(c, metal=0.5 if c.startswith("#8") or c.startswith("#c") else 0.1, rough=0.35, clearcoat=0.3))
            sc.add(repeat_z(sd_cylinder(0.06, r * 1.25, 0.02), 3.0, -39, 0), steel("#3a3e44"), T(x, y, 0) * RX(math.pi / 2))   # flanges
        sc.add(sd_box(0.5, 0.06, 20), steel("#5a5e64"), T(sgn * 13.1, 9.4, -20))
        sc.add(repeat_z(sd_torus(0.3, 0.04), 7.0, -35, -5), Mat("#c82a22", rough=0.4), T(sgn * 12.8, 5.75, 0) * RZ(math.pi / 2))
    # drums and a cask by the walls
    drum = Mat("#d8a01a", rough=0.5, clearcoat=0.3, pattern=lambda pl, pw, n: {
        "albedo": srgb("#d8a01a") * (0.75 + 0.3 * fbm(pw * 8, 2))[:, None] * (1 - 0.6 * (fract(pw[:, 1] * 3.2) < 0.06).float())[:, None]})
    for x, z in ((-11.0, -6.0), (-11.0, -7.4), (-10.0, -6.7), (11.2, -30.0), (10.2, -30.6), (11.2, -31.4)):
        sc.add(sd_cylinder(0.45, 0.32, 0.03), drum, T(x, 0.45, z))
    sc.add(sd_cylinder(1.4, 1.0, 0.1), steel("#9aa0a8", 0.3), T(10.5, 1.4, -14))
    sc.add(repeat_x(sd_torus(1.02, 0.05), 1.0, 0, 0), steel("#7a8088", 0.3), T(10.5, 2.0, -14))
    # radiation warning signs on the pilasters: the trefoil on yellow
    def trefoil(pl, pw, n):
        u, v = pl[:, 2], pl[:, 1]
        r = torch.sqrt(u * u + v * v) / 0.9
        ang = torch.atan2(v, u) + math.pi / 2
        blade = (fract(ang / (2 * math.pi / 3)) < 0.5).float() * (r > 0.22).float() * (r < 0.62).float()
        dot = (r < 0.15).float()
        k = torch.maximum(blade, dot)[:, None]
        a = srgb("#e8c020") * (1 - k) + srgb("#141010") * k
        a = a * (0.75 + 0.3 * fbm(pw * 4, 2))[:, None]
        return {"albedo": a}
    for sgn in (-1, 1):
        for z in (-8.0, -23.0, -33.0):
            sc.add(sd_box(0.03, 0.9, 0.9, 0.05), Mat("#e8c020", rough=0.5, pattern=trefoil), T(sgn * 13.27, 3.6, z))
    return sc


LAMPS = [(x, 22.8, z) for x in (-7.0, 0.0, 7.0) for z in (-6.0, -11.0, -16.0, -21.0, -26.0, -31.0, -36.0)]


def hall_haze(p):
    """Dust hanging in the hall: thin, a little thicker low down."""
    n = fbm(p * 0.15, 3)
    return (0.0012 + 0.002 * torch.exp(-p[:, 1].clamp_min(0) / 6.0)) * (0.5 + n)


def lamp_beams(p, s):
    """The light of the high-bay lamps through the dust: soft cones under each lamp."""
    e = torch.zeros(p.shape[0], device=p.device)
    for (lx, ly, lz) in LAMPS:
        dy = (ly - p[:, 1]).clamp_min(0.01)
        r = torch.sqrt((p[:, 0] - lx) ** 2 + (p[:, 2] - lz) ** 2)
        cone = torch.exp(-(r / (0.4 + 0.3 * dy)) ** 4) * (p[:, 1] < ly).float()
        e = e + cone / (1 + dy * 0.12) ** 2
    return srgb("#fff0dc") * (e * 12.0)[:, None]


def cherenkov(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    inside = torch.exp(-((x / 6.0) ** 8 + ((z + 18) / 8.0) ** 8))
    return inside * torch.exp(-(y + 0.7).clamp_min(0) / 2.0) * 0.25 * (0.6 + 0.6 * fbm(p * 0.6, 3))


def cherenkov_glow(p, s):
    return srgb("#2a8aff") * 2.2


def render(W=2560, H=1440, ss=1):
    cam = Camera((-3.0, 9.5, 4.0), (0.8, -1.0, -20.0), fov=60)
    sky = lambda d: srgb("#05080e").expand(d.shape[0], 3) * 1.0
    lights = [((x, 21.5, z), "#fff0dc", 7.0) for x in (-7.0, 7.0) for z in (-6.0, -16.0, -26.0, -36.0)]
    lights += [((0, 0.5, -18), "#2a9aff", 9.0), ((0, 0.5, -12), "#2a9aff", 6.0), ((0, 0.5, -24), "#2a9aff", 6.0)]
    vols = [Volume(hall_haze, (-14.5, -0.5, -40.5), (14.5, 26.0, 6.0), albedo="#c8ccd0", g=0.3, emission=lamp_beams, steps=48, light_steps=0, ambient=0.6),
            Volume(cherenkov, (-6.8, -0.8, -27), (6.8, 8.0, -9), albedo="#4aa0ff", g=0.3, emission=cherenkov_glow, steps=40, light_steps=0, ambient=0.2)]
    img = render_view(scene(), W, H, cam, sky, sun_dir=None, lights=lights, ambient=("#1a2230", "#0a0e14"),
                      fog=(0.02, "#123048", 0.07, 0.0), exposure=0.95, steps=200, tmax=90, ao_scale=0.25, bloom=1.0,
                      dof=(16.0, 0.4), vignette=0.4, grain=0.012, grade=("#3a6a9a", "#ffe8c8"), ss=ss, volumes=vols)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "laboratoire.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
