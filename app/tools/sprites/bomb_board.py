"""
bomb_board.py — the "Bombes" board (assets/themes/15_bombes.json): a firing
range of the bomb squad at dusk; every move blows the square left up, the
stronger the piece, the bigger the explosion (break fx "blast" in the app).

  assets/backgrounds/bombes.jpg  the range: a dusk sky, red mesas, a bunker and
                                 a watch tower, fence posts, warning signs,
                                 plumes of smoke far away, dust in the air
  assets/art/bombes_clair.*      light square: dry cracked earth
  assets/art/bombes_sombre.*     dark square: a steel plate, rivets at its corners
  assets/art/bombes_trou.*       hole: a scorched crater, embers breathing (idle);
                                 the crater blown open (open)
  assets/art/bombes_cadre.*      the frame: sandbags on a yellow and black band

Still squares; the life is in the scenery (embers drifting) and in the blasts.

    cd app/tools/sprites && python3 bomb_board.py
"""
import json
import math
import os
import torch
from render import srgb
from space import grid, noise2, save_rgb, to_rgba, write_art, bevel, ASSETS, TAU
from ice_board import voronoi_edges
from dino_board import smoothstep, C, over


def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    # the dusk sky: deep violet at the top, orange at the horizon
    k = smoothstep(0.05, 0.7, y)
    img = C("#1a1428") * (1 - k)[None] + C("#e07a3a") * k[None]
    img = img + C("#ffb060") * (torch.exp(-(((X - ar * 0.3) / 0.5) ** 2 + ((y - 0.66) / 0.08) ** 2)) * 0.6)[None]   # the sun just set
    # thin clouds lit from below
    cl = smoothstep(0.55, 0.75, noise2(X * 2, y * 9, 1.0, 5)) * smoothstep(0.1, 0.3, y) * smoothstep(0.6, 0.4, y)
    img = over(img, C("#ff9a6a") * (0.6 + 0.3 * noise2(X * 6, y * 20, 2.0, 2))[None], cl * 0.5)
    # mesas far away: flat tops, steep sides
    for i, (cx, w, top) in enumerate(((0.35, 0.32, 0.5), (1.15, 0.45, 0.47), (1.62, 0.22, 0.53))):
        side = ((X - cx).abs() / w).clamp_min(0)
        prof = top + 0.25 * smoothstep(0.75, 1.0, side) + 0.006 * noise2(X * 30, torch.zeros_like(X) + i, 3.0 + i, 3)
        m = (y > prof).float() * (side < 1.05).float()
        img = over(img, C("#6a2e24") * (0.65 + 0.4 * noise2(X * 15, y * 40, 4.0 + i, 3))[None] * (0.8 + 0.4 * (1 - side.clamp(0, 1)))[None], m)
    # the plain, ochre, darker near us
    ground = 0.66 + 0.01 * noise2(X * 4, torch.zeros_like(X), 5.0, 3)
    g = (y > ground).float()
    gc = C("#7a5a3a") * (1 - 0.45 * smoothstep(0.66, 1.0, y))[None] * (0.7 + 0.4 * noise2(X * 30, y * 60, 6.0, 3))[None]
    img = over(img, gc, g)
    # craters in the plain, flattened by the perspective
    for (cx, cy, r) in ((0.25, 0.8, 0.07), (0.7, 0.74, 0.04), (1.35, 0.86, 0.09), (1.6, 0.75, 0.035)):
        d = torch.sqrt(((X - cx) / r) ** 2 + ((y - cy) / (r * 0.3)) ** 2)
        img = img * (1 - 0.55 * smoothstep(1.0, 0.6, d))[None] + C("#a07850") * (0.25 * torch.exp(-((d - 1.0) / 0.12) ** 2))[None]
    # a concrete bunker and a watch tower in silhouette
    bunker = ((X - 0.95).abs() < 0.12).float() * (y > 0.62 + 0.04 * ((X - 0.95) / 0.12) ** 2).float() * (y < 0.69).float()
    slit = ((X - 0.95).abs() < 0.05).float() * ((y - 0.655).abs() < 0.004).float()
    img = over(img, C("#3a3434"), bunker)
    img = over(img, C("#ffcf7a"), slit * bunker)
    tw = (((X - 1.42).abs() < 0.004) | ((X - 1.46).abs() < 0.004)).float() * (y > 0.45).float() * (y < 0.67).float()
    cab = ((X - 1.44).abs() < 0.03).float() * ((y - 0.44).abs() < 0.02).float()
    img = over(img, C("#1e1a1e"), torch.maximum(tw, cab))
    img = over(img, C("#ffcf7a"), ((X - 1.44).abs() < 0.012).float() * ((y - 0.442).abs() < 0.006).float())
    # fence posts and a wire along the front, warning signs
    for i in range(26):
        px = 0.05 + i * 0.07
        post = ((X - px).abs() < 0.003).float() * (y > 0.8 - 0.003 * math.sin(i)).float() * (y < 0.9).float()
        img = over(img, C("#1a1414"), post)
    for wy in (0.815, 0.845):
        img = over(img, C("#1a1414"), ((y - wy - 0.004 * torch.sin(X * 40)).abs() < 0.0012).float() * (y > 0.79).float())
    for (sx, sy) in ((0.33, 0.83), (1.05, 0.835), (1.55, 0.83)):
        tri = (y < sy + 0.025).float() * (y > sy - 0.025 + (X - sx).abs() * 1.7).float()
        img = over(img, C("#f2c230"), tri)
        img = over(img, C("#1a1414"), ((X - sx).abs() < 0.0025).float() * (y > sy - 0.008).float() * (y < sy + 0.012).float())
        img = over(img, C("#1a1414"), (((X - sx) ** 2 + (y - sy - 0.018) ** 2) < 0.0025 ** 2).float())
    # plumes of smoke rising far away
    for (pxk, w0, seed) in ((0.6, 0.02, 7.0), (1.3, 0.03, 8.0)):
        rise = (0.68 - y).clamp_min(0)
        centre = pxk + rise * 0.4 + 0.03 * torch.sin(rise * 10)
        width = w0 + rise * 0.35
        bl = noise2((X - centre) * 5, y * 6, seed, 4)
        pl = ((bl - 0.35) * 2.5).clamp(0, 1) * torch.exp(-((X - centre) / width) ** 2) * (y < 0.68).float() * smoothstep(0.1, 0.4, y)
        img = over(img, C("#4a3a3a") * (0.7 + 0.5 * bl)[None] + C("#ff9a5a") * (0.2 * smoothstep(0.4, 0.68, y))[None], pl * 0.8)
    # dust in the air, warm
    img = img + C("#ffb070") * (0.08 * smoothstep(0.4, 0.75, y) * noise2(X * 3, y * 3, 9.0, 3))[None]
    v = (x - 0.5) ** 2 + (y - 0.5) ** 2
    img = img * (1 - 0.55 * v)[None]
    save_rgb(img, os.path.join(ASSETS, "backgrounds", "bombes.jpg"))


def earth(S=128):
    x, y = grid(S, S)
    n = noise2(x * 6, y * 6, 1.0, 4)
    c = C("#b08a5a") * (0.78 + 0.32 * n)[None]
    e = voronoi_edges(x, y, 5, 41)
    c = c * (1 - 0.5 * torch.exp(-(e / 0.008) ** 2))[None]                    # dry cracks
    c = c * (1 + 0.12 * noise2(x * 40, y * 40, 2.0, 2))[None]                  # grit
    c = c * (1 + bevel(x, y, 0.05) * 0.4)
    return to_rgba(c, torch.ones_like(x))


def steel(S=128):
    x, y = grid(S, S)
    n = noise2(x * 3, y * 10, 3.0, 4)
    c = C("#3e4238") * (0.8 + 0.25 * n)[None]
    c = c * (1 + 0.1 * noise2(x * 60, y * 4, 4.0, 2))[None]                    # brushed
    rust = smoothstep(0.6, 0.75, noise2(x * 5, y * 5, 5.0, 3))
    c = over(c, C("#5a4630") * (0.8 + 0.3 * noise2(x * 30, y * 30, 6.0, 2))[None], rust * 0.25)
    for (rx, ry) in ((0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)):  # rivets
        d = torch.sqrt((x - rx) ** 2 + (y - ry) ** 2)
        c = over(c, C("#7a7e72") * (1.2 - 4 * ((x - rx) + (y - ry)))[None], (d < 0.035).float())
        c = c * (1 - 0.4 * torch.exp(-((d - 0.04) / 0.01) ** 2))[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.7)
    return to_rgba(c, torch.ones_like(x))


def crater(S=128, frames=10, opening=False):
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v)
    th = torch.atan2(v, u)
    R0 = 0.8 + 0.08 * noise2(torch.cos(th) * 3 + 2, torch.sin(th) * 3, 6.0, 3)
    n_frames = 9 if opening else frames
    for f in range(n_frames):
        ph = f / frames
        k = 1.0 if not opening else 1 - (1 - (f + 1) / n_frames) ** 2
        R = R0 * k
        inside = (r < R).float()
        depth = (r / R).clamp(0, 1)
        c = C("#1a1410") * (0.5 + 0.7 * depth ** 2)[None] * (0.8 + 0.3 * noise2(u * 8, v * 8, 7.0, 3))[None]
        streak = (torch.sin(th * 13 + 3 * noise2(u * 2, v * 2, 8.0, 2)) > 0.6).float() * smoothstep(0.3, 0.9, depth)
        c = c * (1 - 0.4 * streak)[None]                                             # scorch streaks
        emb = smoothstep(0.7, 0.85, noise2(u * 6, v * 6, 9.0, 2)) * smoothstep(0.7, 0.2, depth)
        c = c + C("#ff7a2a") * (emb * (0.6 + 0.4 * math.sin(TAU * ph + 1.0)) * 1.2)[None]   # embers breathing
        rim = torch.exp(-((r - R) / 0.08) ** 2)
        c = over(c, C("#8a6a48") * (0.8 + 0.4 * noise2(u * 12, v * 12, 10.0, 2))[None], rim * (r >= R - 0.04).float())
        alpha = (inside + rim * (r >= R).float()).clamp(0, 1)
        if opening:
            fl = 1 - k
            c = c + C("#ffd080") * (torch.exp(-r * 2) * fl * 3)[None]
            alpha = (alpha + fl * torch.exp(-r * 1.5)).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96):
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    along = torch.where((x - 0.5).abs() > (y - 0.5).abs(), y, x)
    across = torch.where((x - 0.5).abs() > (y - 0.5).abs(), torch.minimum(x, 1 - x), torch.minimum(y, 1 - y))
    # sandbags: rounded sacks in a row, dark gaps between them, lit on top, a seam across each
    cell = torch.frac(along * 3)
    sack = ((cell - 0.5) / 0.46) ** 2 + ((across - 0.12) / 0.11) ** 2
    bag = (sack < 1).float()
    shade_ = (1.25 - 0.7 * sack.clamp(0, 1)) * (1.1 - 2.0 * (across - 0.08).clamp(-0.1, 0.2))
    c = C("#c8a870") * (0.8 + 0.3 * noise2(x * 40, y * 40, 2.0, 3))[None] * shade_[None]
    c = c * (1 - 0.45 * torch.exp(-((cell - 0.5) / 0.025) ** 2) * bag)[None]
    # the yellow and black band along the inner edge
    inner = ((across - 0.28) .abs() < 0.05).float()
    stripe = (torch.frac((along + across) * 6) < 0.5).float()
    band = C("#f2c230") * stripe[None] + C("#1a1414") * (1 - stripe)[None]
    out = C("#1e1812") * (0.7 + 0.3 * noise2(x * 20, y * 20, 3.0, 2))[None]
    c = out * (1 - bag)[None] + c * bag[None]
    c = over(c, band, inner)
    c = c * (1 + bevel(x, y, 0.05) * 0.5)
    return [to_rgba(c, band_)]


def theme():
    t = {
        "id": "bombes", "name": "Bombes", "pattern": "flat", "pattern strength": 0.2, "seed": 31,
        "light": "#b08a5a", "dark": "#3e4238",
        "frame top": "#a88a5a", "frame bottom": "#5a4a34", "frame line": "#f2c230",
        "coordinates": "#f6e6c0", "hole": "#1a1410", "hole rim": "#ff7a2a",
        "background top": "#1a1428", "background bottom": "#3a2a1e",
        "last move": "#ffc23066", "selection": "#ffc2308c", "legal move": "#1a1410aa", "hint": "#a6ff8cdc",
        "accent": "#ffb030",
        "tile light": "bombes_clair", "tile dark": "bombes_sombre", "hole art": "bombes_trou", "frame art": "bombes_cadre",
        "background image": "bombes.jpg", "embers": True, "piece set": "bombes", "break fx": "blast",
    }
    with open(os.path.join(ASSETS, "themes", "15_bombes.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "tiles" in what:
        write_art("bombes_clair", "Bombes : terre sèche", "tile", [("idle", [earth()])], 128)
        write_art("bombes_sombre", "Bombes : plaque d'acier", "tile", [("idle", [steel()])], 128)
    if "hole" in what:
        write_art("bombes_trou", "Bombes : cratère", "hole", [("idle", crater()), ("open", crater(opening=True))], 128,
                  fps={"idle": 5, "open": 20}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("bombes_cadre", "Bombes : sacs de sable", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
