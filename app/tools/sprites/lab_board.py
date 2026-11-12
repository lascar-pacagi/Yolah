"""
lab_board.py — the "Laboratoire" board (assets/themes/14_laboratoire.json): a
research lab in a nuclear plant, the reactor pool glowing blue at the back;
the pieces are bodies rebuilt layer by layer (anatomy.py) and a square left
disintegrates into glowing particles.

  assets/backgrounds/laboratoire.jpg  the reactor hall: concrete, the pool and its
                                      blue Cherenkov light, a catwalk, pipes, hazard
                                      stripes, control screens
  assets/art/laboratoire_clair.*      light square: white lab floor tile
  assets/art/laboratoire_sombre.*     dark square: dark steel grating, a faint blue light under it
  assets/art/laboratoire_trou.*       hole: a containment well, deep blue glow, particles rising
  assets/art/laboratoire_cadre.*      the frame: steel, yellow and black hazard stripes, rivets

Still squares; the life is in the scenery (glowing motes drifting up).

    cd app/tools/sprites && python3 lab_board.py
"""
import json
import math
import os
import torch
from render import DEV, srgb, gauss_blur
from space import grid, noise2, save_rgb, to_rgba, write_art, bevel, ASSETS, TAU


def smoothstep(e0, e1, v):
    t = ((v - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def C(h):
    return srgb(h)[:, None, None]


def over(img, col, a):
    return img * (1 - a)[None] + col * a[None]


def rect(X, y, x0, y0, x1, y1):
    return ((X > x0) & (X < x1) & (y > y0) & (y < y1)).float()


def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    cx = ar / 2
    # the hall: concrete walls, panels, darker at the top
    conc = 0.75 + 0.35 * noise2(X * 6, y * 6, 1.0, 4)
    panels = ((torch.frac(X * 3) < 0.006) | (torch.frac(y * 3) < 0.006)).float()
    img = C("#2a3038") * conc[None] * (1 - 0.4 * panels)[None] * (0.4 + 0.6 * smoothstep(0.0, 0.6, y))[None]
    # the pool: a trapezoid in perspective, water lit blue from below
    py0, py1 = 0.55, 0.82
    half = 0.28 + (y - py0) * 0.9
    pool = ((y > py0) & (y < py1) & ((X - cx).abs() < half)).float()
    depth = smoothstep(py0, py1, y)
    core = torch.exp(-(((X - cx) / 0.18) ** 2 + ((y - 0.7) / 0.06) ** 2))
    ripple = 0.85 + 0.15 * noise2(X * 30, y * 80, 2.0, 2)
    water = C("#0a3a8a") * (0.6 + 0.6 * depth)[None] + C("#5ad8ff") * (core * 2.5 * ripple)[None]
    grid_ = ((torch.frac((X - cx) / half.clamp_min(0.01) * 4) < 0.04) | (torch.frac(y * 30) < 0.06)).float() * 0.25   # the fuel racks under water
    water = water * (1 + grid_)[None]
    img = over(img, water, pool)
    # the blue light thrown on the walls and the air
    glow_ = torch.exp(-(((X - cx) / 0.6) ** 2 + ((y - 0.62) / 0.3) ** 2))
    img = img + C("#2a8aff") * (glow_ * 0.35)[None]
    # the pool's rim and a railing
    rim = ((y > py0 - 0.012) & (y < py0) & ((X - cx).abs() < 0.3)).float()
    img = over(img, C("#c8ccd0"), rim)
    for k in range(9):
        px = cx - 0.28 + k * 0.07
        img = over(img, C("#e8c22a"), rect(X, y, px - 0.002, py0 - 0.06, px + 0.002, py0 - 0.012))
    img = over(img, C("#e8c22a"), rect(X, y, cx - 0.29, py0 - 0.062, cx + 0.29, py0 - 0.056))
    # a catwalk crossing high up, with its lights
    cw = rect(X, y, 0.0, 0.2, ar, 0.225)
    img = over(img, C("#3a4048") * (1 - 0.3 * (torch.frac(X * 40) < 0.1).float())[None], cw)
    for k in range(7):
        lx = 0.15 + k * 0.27
        d = torch.sqrt((X - lx) ** 2 + ((y - 0.232) * 2) ** 2)
        img = img + C("#e8f4ff") * (torch.exp(-(d / 0.006) ** 2) * 3 + torch.exp(-(d / 0.05) ** 2) * 0.25)[None]
    # pipes along the walls
    for i, (py, r) in enumerate(((0.32, 0.012), (0.36, 0.008), (0.4, 0.015))):
        pipe = ((y - py).abs() < r).float() * ((X < cx - 0.45) | (X > cx + 0.45)).float()
        sh = 1 - ((y - py) / r).clamp(-1, 1) ** 2
        img = over(img, C("#6a7480") * (0.5 + 0.7 * sh)[None], pipe)
    # control screens on both sides
    for sx in (0.18, 0.42, ar - 0.42, ar - 0.18):
        scr = rect(X, y, sx - 0.09, 0.44, sx + 0.09, 0.53)
        lines = (torch.frac(y * 400) < 0.4).float() * 0.2
        wave = torch.exp(-((y - 0.485 - 0.02 * torch.sin((X - sx) * 60 + sx * 7)) / 0.003) ** 2)
        img = over(img, C("#0c2a1e") * (1 + lines)[None] + C("#5affa0") * (wave * 2)[None], scr)
        img = over(img, C("#1a1e24"), rect(X, y, sx - 0.1, 0.43, sx + 0.1, 0.44) + rect(X, y, sx - 0.1, 0.53, sx + 0.1, 0.54))
    # the floor: hazard stripes along the pool, a polished floor reflecting the blue
    floor = (y > py1).float()
    img = over(img, C("#2a2e34") * (0.8 + 0.3 * noise2(X * 10, y * 30, 5.0, 3))[None] + C("#2a8aff") * (0.15 * (1 - depth))[None], floor)
    stripe = ((y > py1) & (y < py1 + 0.025)).float()
    hz = (torch.frac((X + y) * 18) < 0.5).float()
    img = over(img, C("#e8c22a") * hz[None] + C("#1a1a1a") * (1 - hz)[None], stripe)
    # the trefoil sign on the back wall
    tx, ty, tr = cx, 0.33, 0.07
    d = torch.sqrt((X - tx) ** 2 + (y - ty) ** 2)
    a = torch.atan2(y - ty, X - tx)
    blade = ((torch.cos(3 * (a + math.pi / 2)) > 0.5) & (d > tr * 0.25) & (d < tr)).float()
    sign = (d < tr * 1.15).float()
    img = over(img, C("#e8c22a"), sign * 0.9)
    img = over(img, C("#1a1a1a"), torch.maximum(blade, (d < tr * 0.17).float()) * sign)
    v = (x - 0.5) ** 2 + (y - 0.5) ** 2
    img = img * (1 - 0.5 * v)[None]
    save_rgb(img, os.path.join(ASSETS, "backgrounds", "laboratoire.jpg"))


def lab_tile(S=128):
    x, y = grid(S, S)
    c = C("#d8dde2") * (0.92 + 0.08 * noise2(x * 5, y * 5, 1.0, 3))[None]
    seam = torch.exp(-((x - 0.5) / 0.006) ** 2) + torch.exp(-((y - 0.5) / 0.006) ** 2)    # four tiles per square
    c = c * (1 - 0.18 * seam.clamp(0, 1))[None]
    c = c * (1 + bevel(x, y, 0.04) * 0.4)
    return to_rgba(c, torch.ones_like(x))


def grating(S=128):
    x, y = grid(S, S)
    bars = ((torch.frac(x * 8) < 0.22) | (torch.frac(y * 8) < 0.22)).float()
    below = C("#0a1e3a") + C("#2a8aff") * (0.25 * torch.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.2)))[None]
    steel = C("#3a4048") * (0.8 + 0.3 * noise2(x * 30, y * 30, 2.0, 2))[None]
    c = below * (1 - bars)[None] + steel * bars[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.6)
    return to_rgba(c, torch.ones_like(x))


def well(S=128, frames=12, opening=False):
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v)
    n_frames = 9 if opening else frames
    for f in range(n_frames):
        ph = f / frames
        k = 1.0 if not opening else 1 - (1 - (f + 1) / n_frames) ** 2
        R = 0.74 * k
        inside = (r < R).float()
        c = C("#02081a") + C("#3ac8ff") * (torch.exp(-(r / (R * 0.55 + 1e-3)) ** 2) * 1.6)[None]
        for i in range(7):   # glowing motes rising from the depth
            a = i * 2.4
            rr = 0.15 + 0.08 * (i % 3)
            yy = ((ph + i / 7) % 1)
            mx, my = math.cos(a) * rr * (1 - yy), math.sin(a) * rr * (1 - yy) - yy * 0.3
            d = torch.sqrt((u - mx) ** 2 + (v - my) ** 2)
            c = c + C("#bff0ff") * (torch.exp(-(d / 0.035) ** 2) * 1.5 * math.sin(math.pi * yy))[None]
        rim = torch.exp(-((r - R) / 0.05) ** 2)
        c = over(c, C("#8a949e"), rim * (r >= R - 0.04).float())
        hz = (torch.frac(torch.atan2(v, u) / TAU * 16) < 0.5).float()
        band = ((r > R) & (r < R + 0.1)).float()
        c = over(c, C("#e8c22a") * hz[None] + C("#1a1a1a") * (1 - hz)[None], band)
        alpha = torch.maximum(inside, torch.maximum(band, rim * (r >= R).float())).clamp(0, 1)
        if opening:
            fl = 1 - k
            c = c + C("#e8f8ff") * (torch.exp(-r * 2) * fl * 3)[None]
            alpha = (alpha + fl * torch.exp(-r * 1.5)).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96):
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    steel = C("#5a626c") * (0.8 + 0.3 * noise2(x * 3, y * 40, 1.0, 3))[None]
    hz = (torch.frac((x + y) * 6) < 0.5).float()
    stripe = ((e > 0.09) & (e < 0.24)).float()
    c = over(steel, C("#e8c22a") * hz[None] + C("#1a1a1a") * (1 - hz)[None], stripe)
    along = torch.where((x - 0.5).abs() > (y - 0.5).abs(), y, x)
    across = torch.where((x - 0.5).abs() > (y - 0.5).abs(), torch.minimum(x, 1 - x), torch.minimum(y, 1 - y))
    rivet = (torch.sqrt(((torch.frac(along * 6) - 0.5) / 6) ** 2 + (across - 0.045) ** 2) < 0.018).float()
    c = over(c, C("#b8c0c8"), rivet)
    c = c * (1 + bevel(x, y, 0.05) * 0.6)
    return [to_rgba(c, band_)]


def theme():
    t = {
        "id": "laboratoire", "name": "Laboratoire", "pattern": "flat", "pattern strength": 0.2, "seed": 29,
        "light": "#d8dde2", "dark": "#2a3038",
        "frame top": "#5a626c", "frame bottom": "#3a4048", "frame line": "#e8c22a",
        "coordinates": "#e8f4ff", "hole": "#02081a", "hole rim": "#3ac8ff",
        "background top": "#1a1e24", "background bottom": "#0a1e3a",
        "last move": "#3ac8ff55", "selection": "#3ac8ff8c", "legal move": "#0a3a6aaa", "hint": "#a6ff8cdc",
        "accent": "#5ad8ff",
        "tile light": "laboratoire_clair", "tile dark": "laboratoire_sombre", "hole art": "laboratoire_trou", "frame art": "laboratoire_cadre",
        "background image": "laboratoire.jpg", "motes": True, "piece set": "laboratoire", "break fx": "atoms",
    }
    with open(os.path.join(ASSETS, "themes", "14_laboratoire.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "tiles" in what:
        write_art("laboratoire_clair", "Laboratoire : carrelage", "tile", [("idle", [lab_tile()])], 128)
        write_art("laboratoire_sombre", "Laboratoire : caillebotis", "tile", [("idle", [grating()])], 128)
    if "hole" in what:
        write_art("laboratoire_trou", "Laboratoire : puits de confinement", "hole", [("idle", well()), ("open", well(opening=True))], 128,
                  fps={"idle": 8, "open": 20}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("laboratoire_cadre", "Laboratoire : cadre d'acier", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
