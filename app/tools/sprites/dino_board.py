"""
dino_board.py — the "Dinosaures" board (assets/themes/13_dinosaures.json): a
prehistoric valley at the foot of a volcano; a move splits the ground and the
lava comes up through the square left.

  assets/backgrounds/dinosaures.jpg  the valley: an orange sky of ash, the volcano
                                     erupting, its glow on the clouds, cliffs,
                                     giant ferns and cycads in silhouette, a
                                     river of lava far away, pterosaurs
  assets/art/dinosaures_clair.*      light square: moss and small ferns on earth
  assets/art/dinosaures_sombre.*     dark square: cracked black basalt
  assets/art/dinosaures_trou.*       hole: a pit of lava, its crust breaking and
                                     glowing (idle, slow); the ground splitting (open)
  assets/art/dinosaures_cadre.*      the frame: rough stone, fossil bones set in it

Still squares; the life is in the scenery (embers rising from the volcano).

    cd app/tools/sprites && python3 dino_board.py
"""
import json
import math
import os
import torch
from render import DEV, srgb, gauss_blur
from space import grid, noise2, save_rgb, to_rgba, write_art, bevel, ASSETS, TAU
from ice_board import voronoi_edges


def smoothstep(e0, e1, v):
    t = ((v - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def C(h):
    return srgb(h)[:, None, None]


def over(img, col, a):
    return img * (1 - a)[None] + col * a[None]


def fern(X, y, bx, by, h, lean, seed):
    """A giant fern frond in silhouette: a curved stem, leaflets along it."""
    m = torch.zeros_like(X)
    n = 24
    for i in range(n):
        u = i / (n - 1)
        px = bx + lean * h * u * u
        py = by - h * u
        w = h * 0.22 * math.sin(math.pi * min(1.0, u * 1.1)) * (1 - 0.6 * u)
        if w < 1e-3:   # no leaflet at the very base or tip (a zero length would cover everything)
            continue
        for sgn in (-1, 1):
            ex, ey = px + sgn * w, py - w * 0.35
            # a leaflet: a thin ellipse from the stem to (ex, ey)
            cx, cy = (px + ex) / 2, (py + ey) / 2
            dx, dy = ex - px, ey - py
            L = math.hypot(dx, dy) + 1e-6
            ax, ay = dx / L, dy / L
            qa = (X - cx) * ax + (y - cy) * ay
            qb = -(X - cx) * ay + (y - cy) * ax
            m = torch.maximum(m, ((qa / (L * 0.5)) ** 2 + (qb / (h * 0.012 + 0.002)) ** 2 < 1).float())
        m = torch.maximum(m, (torch.sqrt((X - px) ** 2 + (y - py) ** 2) < h * 0.008 + 0.001).float())
    return m


def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    # the sky: ash, orange near the horizon, dark at the top
    k = smoothstep(0.0, 0.65, y)
    img = C("#1e1418") * (1 - k)[None] + C("#d8682a") * k[None]
    # the volcano, its glow, its plume
    vx, vtop = ar * 0.62, 0.3
    vol = (y > vtop + (X - vx).abs() * 0.75 + 0.03 * noise2(X * 8, y * 2, 3.0, 3)).float() * (y < 0.75).float()
    glow_ = torch.exp(-(((X - vx) / 0.15) ** 2 + ((y - vtop) / 0.08) ** 2))
    rise = (vtop - y).clamp_min(0)
    centre = vx + rise * 0.9 + 0.05 * torch.sin(rise * 12)
    width = 0.05 + rise * 1.1
    billow = noise2((X - centre) * 4, y * 5 - 0.0, 4.0, 5)
    plume = ((billow - 0.3) * 2.5).clamp(0, 1) * torch.exp(-((X - centre) / width) ** 2) * smoothstep(vtop + 0.01, vtop - 0.05, y)
    lit = torch.exp(-rise / 0.12)                                   # lit from below by the crater
    img = over(img, (C("#2e2224") * (0.6 + 0.6 * billow)[None] + C("#ff7a2a") * (lit * 0.9)[None]), plume * 0.95)
    img = img + C("#ff7a2a") * (glow_ * 0.8)[None]
    vcol = C("#2a1c1c") * (0.7 + 0.4 * noise2(X * 20, y * 20, 6.0, 3))[None]
    img = over(img, vcol, vol)
    lava_streak = vol * (torch.exp(-((X - vx - 0.03 * torch.sin(y * 30)) / 0.006) ** 2)) * smoothstep(vtop, vtop + 0.25, y) * smoothstep(0.62, 0.4, y)
    img = img + C("#ff6a1a") * (lava_streak * 2.5)[None]
    img = img + C("#ffb040") * (torch.exp(-(((X - vx) / 0.03) ** 2 + ((y - vtop) / 0.015) ** 2)) * 3)[None]
    # cliffs on both sides, darker
    for i, (cx, w, top) in enumerate(((0.15, 0.45, 0.42), (ar - 0.1, 0.4, 0.47))):
        prof = top + 0.08 * noise2(y * 3, torch.zeros_like(y) + i, 7.0 + i, 3)
        m = ((X - cx).abs() < w * (1 - ((y - prof).clamp_min(0) * 0) ) ).float() * (y > prof + ((X - cx).abs() / w) ** 2 * 0.25).float()
        img = over(img, C("#3a2420") * (0.6 + 0.5 * noise2(X * 12, y * 30, 8.0 + i, 3))[None], m)
    # the valley floor, with a river of lava far away
    ground = 0.72 + 0.02 * noise2(X * 3, torch.zeros_like(X), 9.0, 3)
    g = (y > ground).float()
    img = over(img, C("#2a2418") * (0.6 + 0.5 * noise2(X * 20, y * 40, 10.0, 3))[None], g)
    ry_ = 0.745 + 0.02 * torch.sin(X * 3.1 + 1) + 0.008 * torch.sin(X * 11)
    river = torch.exp(-((y - ry_) / (0.0025 + 0.002 * noise2(X * 4, torch.zeros_like(X), 12.0, 2))) ** 2) * g * \
        smoothstep(0.35, 0.55, noise2(X * 1.3, torch.zeros_like(X), 13.0, 2))          # in places only, like a lava flow
    img = img + C("#ff6a1a") * (river * 2.5)[None]
    # pterosaurs far away: small dark V shapes
    for (px, py, sz) in ((0.4, 0.2, 0.012), (0.52, 0.16, 0.009), (1.2, 0.22, 0.01)):
        dx = (X - px) / sz
        dy = (y - py) / sz
        wing = ((dy - 0.35 * dx.abs() ** 1.4).abs() < 0.12).float() * (dx.abs() < 1.6).float()
        img = img * (1 - wing * 0.8)[None]
    # giant ferns and cycads in front, black against the light
    sil = torch.zeros_like(x)
    for i, (bx, by, h, lean) in enumerate(((0.05, 1.02, 0.55, 0.4), (0.25, 1.05, 0.45, -0.3), (0.42, 1.05, 0.3, 0.5),
                                           (ar - 0.05, 1.02, 0.6, -0.45), (ar - 0.28, 1.05, 0.42, 0.3), (ar - 0.45, 1.05, 0.28, -0.5))):
        sil = torch.maximum(sil, fern(X, y, bx, by, h, lean, i))
    img = over(img, C("#140e0c"), sil)
    v = (x - 0.5) ** 2 + (y - 0.5) ** 2
    img = img * (1 - 0.5 * v)[None]
    save_rgb(img, os.path.join(ASSETS, "backgrounds", "dinosaures.jpg"))


def moss(S=128):
    x, y = grid(S, S)
    n = noise2(x * 6, y * 6, 1.0, 4)
    c = C("#3a4824") * (0.7 + 0.4 * n)[None]
    tufts = smoothstep(0.55, 0.75, noise2(x * 18, y * 18, 2.0, 3))
    c = c * (1 + 0.3 * tufts)[None] + C("#7a9040") * (0.12 * tufts)[None]
    earth = smoothstep(0.62, 0.4, noise2(x * 4, y * 4, 3.0, 3)) * 0.0 + (n < 0.38).float() * 0.6
    c = over(c, C("#4a3a26") * (0.8 + 0.3 * noise2(x * 30, y * 30, 4.0, 2))[None], earth)
    c = c * (1 + bevel(x, y, 0.05) * 0.4)
    return to_rgba(c * 0.95, torch.ones_like(x))


def basalt(S=128):
    x, y = grid(S, S)
    n = noise2(x * 8, y * 8, 5.0, 4)
    c = C("#2a2624") * (0.7 + 0.5 * n)[None]
    e = voronoi_edges(x, y, 6, 13)
    c = c * (1 - 0.6 * torch.exp(-(e / 0.012) ** 2))[None]
    c = c * (1 + bevel(x, y, 0.06) * 0.6)
    return to_rgba(c, torch.ones_like(x))


def lava_pit(S=128, frames=10, opening=False):
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v)
    th = torch.atan2(v, u)
    R0 = 0.76 + 0.07 * noise2(torch.cos(th) * 3 + 2, torch.sin(th) * 3, 6.0, 3)
    n_frames = 9 if opening else frames
    for f in range(n_frames):
        ph = f / frames
        k = 1.0 if not opening else 1 - (1 - (f + 1) / n_frames) ** 2
        R = R0 * k
        inside = (r < R).float()
        # the crust: dark plates over the lava, slowly breathing
        e = voronoi_edges(x + 0.02 * math.sin(TAU * ph), y + 0.02 * math.cos(TAU * ph), 9, 31)
        heat = torch.exp(-(e / (0.02 + 0.01 * math.sin(TAU * ph))) ** 2)
        lava = C("#ff7a1a") * (1.2 + 0.6 * noise2(u * 3, v * 3, 7.0 + ph, 2))[None]
        crust = C("#2a1a14") * (0.6 + 0.4 * noise2(u * 10, v * 10, 8.0, 2))[None]
        c = crust * (1 - heat)[None] + lava * heat[None]
        c = c + C("#ffb040") * (torch.exp(-(r / (R * 0.4)) ** 2) * 0.4 * (0.8 + 0.2 * math.sin(TAU * ph)))[None]
        rim = torch.exp(-((r - R) / 0.06) ** 2)
        c = over(c, C("#3a2a22"), rim * (r >= R - 0.05).float())
        alpha = (inside + rim * (r >= R).float()).clamp(0, 1)
        if opening:
            fl = (1 - k)
            c = c + C("#ffd080") * (torch.exp(-r * 2) * fl * 3)[None]
            alpha = (alpha + fl * torch.exp(-r * 1.5)).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96):
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    c = C("#5a4a3a") * (0.6 + 0.6 * noise2(x * 14, y * 14, 2.0, 4))[None]
    c = c * (1 - 0.4 * torch.exp(-(voronoi_edges(x, y, 10, 5) / 0.01) ** 2))[None]
    along = torch.where((x - 0.5).abs() > (y - 0.5).abs(), y, x)
    across = torch.where((x - 0.5).abs() > (y - 0.5).abs(), torch.minimum(x, 1 - x), torch.minimum(y, 1 - y))
    # fossil vertebrae along the band
    vert = (torch.frac(along * 7) - 0.5).abs()
    bone = ((vert < 0.18).float() * ((across - 0.165).abs() < 0.05).float()) + ((vert < 0.06).float() * ((across - 0.165).abs() < 0.09).float())
    c = over(c, C("#e8dcc0") * (0.85 + 0.2 * noise2(x * 40, y * 40, 3.0, 2))[None], bone.clamp(0, 1) * 0.9)
    c = c * (1 + bevel(x, y, 0.05) * 0.6)
    return [to_rgba(c, band_)]


def theme():
    t = {
        "id": "dinosaures", "name": "Dinosaures", "pattern": "flat", "pattern strength": 0.2, "seed": 23,
        "light": "#3a4824", "dark": "#2a2624",
        "frame top": "#5a4a3a", "frame bottom": "#3a2a20", "frame line": "#e8dcc0",
        "coordinates": "#f2e2c0", "hole": "#2a1a14", "hole rim": "#ff7a1a",
        "background top": "#1e1418", "background bottom": "#2a2418",
        "last move": "#ffb04066", "selection": "#ffb0408c", "legal move": "#1a1410aa", "hint": "#a6ff8cdc",
        "accent": "#ff9a3a",
        "tile light": "dinosaures_clair", "tile dark": "dinosaures_sombre", "hole art": "dinosaures_trou", "frame art": "dinosaures_cadre",
        "background image": "dinosaures.jpg", "embers": True, "piece set": "dinosaures", "break fx": "lava",
    }
    with open(os.path.join(ASSETS, "themes", "13_dinosaures.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "tiles" in what:
        write_art("dinosaures_clair", "Dinosaures : mousse", "tile", [("idle", [moss()])], 128)
        write_art("dinosaures_sombre", "Dinosaures : basalte", "tile", [("idle", [basalt()])], 128)
    if "hole" in what:
        write_art("dinosaures_trou", "Dinosaures : lave", "hole", [("idle", lava_pit()), ("open", lava_pit(opening=True))], 128,
                  fps={"idle": 5, "open": 20}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("dinosaures_cadre", "Dinosaures : cadre fossile", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
