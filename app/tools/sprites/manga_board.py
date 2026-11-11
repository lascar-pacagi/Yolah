"""
manga_board.py — the "Manga" board (assets/themes/9_manga.json), in flat
anime colours with ink lines:

  assets/backgrounds/manga.jpg   a city at dusk: a big sun, cel-shaded clouds,
                                 rooftops and a tower in silhouette, speed lines
  assets/art/manga_clair.*       light square: tatami with its black border
  assets/art/manga_sombre.*      dark square: lacquered dark wood
  assets/art/manga_trou.*        hole: an ink impact (idle, open: the splash)
  assets/art/manga_cadre.*       the frame: red lacquer, gold corners

    cd app/tools/sprites && python3 manga_board.py
"""
import json
import math
import os
import torch
from render import DEV, srgb, vec, gauss_blur
from space import grid, noise2, save_rgb, to_rgba, write_art, ASSETS, TAU

INK = srgb("#1b1420")


def band(v, edges, cols):
    """Flat colour bands: v in [0, 1] → the colour of the band it falls in."""
    out = cols[0][:, None, None] * torch.ones_like(v)[None]
    for e, c in zip(edges, cols[1:]):
        out = torch.where((v > e)[None], c[:, None, None] * torch.ones_like(v)[None], out)
    return out


def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    sky = band(y, [0.18, 0.32, 0.45, 0.56], [srgb("#2a2a6a"), srgb("#5a3a8a"), srgb("#c85a8a"), srgb("#ff8a6a"), srgb("#ffc07a")])
    # a big sun, low, half behind the city
    sx, sy = 0.62 * ar, 0.6
    d = torch.sqrt((X - sx) ** 2 + (y - sy) ** 2)
    sun = (d < 0.22).float()
    img = sky * (1 - sun)[None] + srgb("#fff0c0")[:, None, None] * sun[None]
    img = img + srgb("#ffd08a")[:, None, None] * (torch.exp(-(d / 0.35) ** 2) * 0.3)[None]
    # cel-shaded clouds: flat shapes with a lit top band, inked
    for i, (cx, cy, w) in enumerate(((0.3, 0.22, 0.35), (1.25, 0.15, 0.4), (0.9, 0.3, 0.25), (1.6, 0.35, 0.3))):
        n = noise2(X * 5 + i * 3, y * 5, 2.0 + i, 3)
        shape = (((X - cx) / w) ** 2 + ((y - cy) / (w * 0.22)) ** 2 + (0.5 - n) * 0.9) < 1
        lit = shape & ((((X - cx) / w) ** 2 + ((y - cy + w * 0.06) / (w * 0.2)) ** 2 + (0.5 - n) * 0.9) < 1)
        img = torch.where(shape[None], srgb("#e88aa8")[:, None, None] * torch.ones_like(x)[None], img)
        img = torch.where(lit[None], srgb("#ffd2c0")[:, None, None] * torch.ones_like(x)[None], img)
        edge = shape.float() - torch.nn.functional.max_pool2d((~shape).float()[None, None] * -1 + 1, 3, 1, 1)[0, 0]
    # the city: rooftops in two layers, a tower with a spire, lit windows
    def skyline(base_h, scale, seed):
        cells = torch.floor(X * scale)
        hgt = base_h - 0.12 * torch.frac(torch.sin(cells * 12.9898 + seed) * 43758.5)
        return hgt
    far = skyline(0.62, 18, 1.0)
    img = torch.where((y > far)[None], srgb("#5a3a6a")[:, None, None] * torch.ones_like(x)[None], img)
    near = skyline(0.72, 11, 7.0)
    tower = ((X - 1.1).abs() < 0.018).float() * (y > 0.3).float() + ((X - 1.1).abs() < 0.006).float() * (y > 0.22).float()
    city = (y > near).float()
    city = torch.maximum(city, tower)
    img = torch.where((city > 0)[None], srgb("#24183a")[:, None, None] * torch.ones_like(x)[None], img)
    win = city * (torch.frac(X * 160) < 0.35).float() * (torch.frac(y * 120) < 0.4).float() * (noise2(torch.floor(X * 160), torch.floor(y * 120), 4.0, 1) > 0.62).float()
    img = img + srgb("#ffd27a")[:, None, None] * (win * 0.9)[None]
    # speed lines from the sun, faint
    th = torch.atan2(y - sy, X - sx)
    lines = ((torch.sin(th * 90) > 0.985).float() * (d > 0.25).float() * (y < 0.5).float()) * 0.05
    img = img + srgb("#ffffff")[:, None, None] * lines[None]
    save_rgb(img * 1.05, os.path.join(ASSETS, "backgrounds", "manga.jpg"))


def tatami(S=128):
    x, y = grid(S, S)
    weave = 0.5 + 0.5 * torch.sin(y * S * 1.2)
    c = srgb("#c8c088")[:, None, None] * (0.85 + 0.12 * weave + 0.06 * noise2(x * 8, y * 30, 1.0, 2))[None]
    border = ((x < 0.09) | (x > 0.91)).float()
    c = c * (1 - border)[None] + srgb("#20242e")[:, None, None] * border[None]
    stitch = border * (torch.frac(y * 10) < 0.15).float()
    c = c + srgb("#c8a85a")[:, None, None] * (stitch * 0.6)[None]
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    c = torch.where((e < 0.015)[None], INK[:, None, None] * torch.ones_like(x)[None], c)
    return to_rgba(c * 0.95, torch.ones_like(x))


def lacquer(S=128):
    x, y = grid(S, S)
    planks = torch.floor(y * 3)
    grain = noise2(x * 3 + planks * 7, y * 40, 3.0, 3)
    c = srgb("#3a1e24")[:, None, None] * (0.8 + 0.4 * grain)[None]
    gap = (torch.frac(y * 3) < 0.04).float()
    c = c * (1 - 0.6 * gap)[None]
    shine = torch.exp(-((x + y - 0.55) / 0.12) ** 2) * 0.12          # a soft gloss, not a stripe
    c = c + srgb("#ffd0c0")[:, None, None] * shine[None]
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    c = torch.where((e < 0.015)[None], INK[:, None, None] * torch.ones_like(x)[None], c)
    return to_rgba(c, torch.ones_like(x))


def ink_hole(S=128, frames=10, opening=False):
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v)
    th = torch.atan2(v, u)
    blob = 0.62 + 0.12 * noise2(torch.cos(th) * 2 + 3, torch.sin(th) * 2, 4.0, 3) + 0.08 * (torch.sin(th * 9) > 0.6).float()
    n_frames = 8 if opening else frames
    for f in range(n_frames):
        ph = f / frames
        k = 1.0 if not opening else 1 - (1 - (f + 1) / n_frames) ** 3
        R = blob * k
        inside = (r < R).float()
        c = INK[:, None, None] * torch.ones_like(x)[None]
        # a white highlight crescent inside the ink, wobbling
        hl = ((r < R * 0.75) & (r > R * 0.55) & (th > -2.6 + 0.1 * math.sin(TAU * ph)) & (th < -1.6)).float()
        c = c + srgb("#ffffff")[:, None, None] * (hl * 0.85)[None]
        # drops around, and a ring of speed lines pulsing
        drops = torch.zeros_like(x)
        for j in range(7):
            a = j * 0.9 + 0.3
            dr = R.mean() * (1.25 + 0.15 * (j % 3)) * k
            drops = torch.maximum(drops, (((u - dr * math.cos(a)) ** 2 + (v - dr * math.sin(a)) ** 2) < (0.035 + 0.02 * (j % 2)) ** 2).float())
        rays = ((torch.sin(th * 24) > 0.85).float() * ((r > R + 0.04) & (r < R + 0.12 + 0.05 * math.sin(TAU * ph))).float())
        alpha = torch.maximum(torch.maximum(inside, drops), rays * 0.85)
        if opening:   # the splash: a white starburst behind, fading
            fl = math.exp(-f * 0.5)
            star = (r < (0.9 + 0.15 * (torch.sin(th * 7) > 0).float()) * (0.6 + 0.4 * k)).float() * fl
            c = c * (1 - star * (1 - inside))[None] + srgb("#ffffff")[:, None, None] * (star * (1 - inside))[None]
            alpha = torch.maximum(alpha, star)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96, frames=1):
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    c = srgb("#b8242a")[:, None, None] * (0.9 + 0.1 * noise2(x * 10, y * 10, 1.0, 2))[None]
    c = torch.where(((e > 0.12) & (e < 0.15))[None], srgb("#1b1420")[:, None, None] * torch.ones_like(x)[None], c)
    shine = (((e - 0.05).abs() < 0.015)).float()
    c = c + srgb("#ffb0a0")[:, None, None] * shine[None] * 0.5
    for (cx, cy) in ((0.165, 0.165), (0.835, 0.165), (0.835, 0.835), (0.165, 0.835)):
        d = torch.max((x - cx).abs(), (y - cy).abs())
        cap = (d < 0.14).float()
        c = c * (1 - cap)[None] + srgb("#e8b44a")[:, None, None] * cap[None] * (0.85 + 0.3 * ((x - cx) < (y - cy)).float())[None]
        c = torch.where(((d - 0.14).abs() < 0.012)[None], INK[:, None, None] * torch.ones_like(x)[None], c)
    c = torch.where(((e < 0.012) | ((e - 1 / 3 + 0.012).abs() < 0.012))[None], INK[:, None, None] * torch.ones_like(x)[None], c)
    return [to_rgba(c, band_)]


def theme():
    t = {
        "id": "manga", "name": "Manga", "pattern": "flat", "pattern strength": 0.2, "seed": 5,
        "light": "#c8c088", "dark": "#3a1e24",
        "frame top": "#b8242a", "frame bottom": "#7a161a", "frame line": "#e8b44a",
        "coordinates": "#ffe8d0", "hole": "#1b1420", "hole rim": "#000000",
        "background top": "#2a2a6a", "background bottom": "#ff8a6a",
        "last move": "#ffe08a66", "selection": "#ff8ab48c", "legal move": "#1b1420aa", "hint": "#5aff8adc",
        "accent": "#ff6a8a",
        "tile light": "manga_clair", "tile dark": "manga_sombre", "hole art": "manga_trou", "frame art": "manga_cadre",
        "background image": "manga.jpg", "petals": True, "piece set": "manga", "break fx": "ink",
    }
    with open(os.path.join(ASSETS, "themes", "9_manga.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "tiles" in what:
        write_art("manga_clair", "Manga : tatami", "tile", [("idle", [tatami()])], 128)
        write_art("manga_sombre", "Manga : laque", "tile", [("idle", [lacquer()])], 128)
    if "hole" in what:
        write_art("manga_trou", "Manga : impact d'encre", "hole", [("idle", ink_hole()), ("open", ink_hole(opening=True))], 128,
                  fps={"idle": 8, "open": 24}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("manga_cadre", "Manga : cadre laqué", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
