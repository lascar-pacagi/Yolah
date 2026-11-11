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
from PIL import Image
from render import DEV, srgb, vec, gauss_blur
from space import grid, noise2, save_rgb, to_rgba, write_art, ASSETS, TAU

INK = srgb("#1b1420")


def band(v, edges, cols):
    """Flat colour bands: v in [0, 1] → the colour of the band it falls in."""
    out = cols[0][:, None, None] * torch.ones_like(v)[None]
    for e, c in zip(edges, cols[1:]):
        out = torch.where((v > e)[None], c[:, None, None] * torch.ones_like(v)[None], out)
    return out


def ink_edges(mask, w=2):
    """The outline of a mask (H, W) bool: a ring of w pixels."""
    m = mask.float()[None, None]
    grown = torch.nn.functional.max_pool2d(m, 2 * w + 1, 1, w)[0, 0]
    return (grown > 0) & ~mask


def island(X, y, cx, cy, w, seed):
    """A floating island: a grassy top (ellipse) and a rocky underside tapering down."""
    n = noise2(X * 12 + seed, y * 12, seed, 3)
    top = (((X - cx) / w) ** 2 + ((y - cy) / (w * 0.18)) ** 2) < 1
    depth = ((y - cy) / (w * 1.1)).clamp(0, 1)
    under = (y > cy) & ((X - cx).abs() < w * (1 - depth ** 0.8) * (0.9 + 0.2 * n)) & (y < cy + w * 1.1)
    return top, under


def background(W=2560, H=1440):
    """A sky sanctuary: a gold and blue sky with a huge ringed planet, rays of
    light, floating islands with waterfalls, a sea of clouds below; flat
    colours and ink lines."""
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    sky = band(y, [0.12, 0.25, 0.38, 0.5], [srgb("#1a2a6a"), srgb("#2a4aa8"), srgb("#4a8ad8"), srgb("#8ac8f0"), srgb("#ffe8b0")])
    img = sky
    # the giant planet with its rings, upper left, behind everything
    px, py, pr = 0.42, 0.2, 0.2
    d = torch.sqrt((X - px) ** 2 + (y - py) ** 2)
    planet = d < pr
    lit = ((X - px) * -0.7 + (y - py) * -0.7) / pr
    pc = band((lit * 0.5 + 0.5).clamp(0, 1), [0.35, 0.65], [srgb("#c86a8a"), srgb("#f2a07a"), srgb("#ffd8a0")])
    bands_ = (torch.sin((y - py) / pr * 9 + noise2(X * 6, y * 6, 3.0, 2) * 2) > 0.5).float() * 0.12
    img = torch.where(planet[None], pc * (1 - bands_)[None], img)
    ry_ = (y - py) - (X - px) * 0.25
    ring = ((((X - px) / (pr * 2.1)) ** 2 + (ry_ / (pr * 0.32)) ** 2) < 1) & ((((X - px) / (pr * 1.5)) ** 2 + (ry_ / (pr * 0.2)) ** 2) > 1)
    ring = ring & ~(planet & (ry_ < 0))
    img = torch.where(ring[None], srgb("#fff0d0")[:, None, None] * torch.ones_like(x)[None], img)
    img = torch.where(ink_edges(planet | ring, 2)[None], INK[:, None, None] * torch.ones_like(x)[None], img)
    # a small moon
    dm = torch.sqrt((X - 1.55) ** 2 + (y - 0.12) ** 2)
    img = torch.where((dm < 0.04)[None], srgb("#f8f4e8")[:, None, None] * torch.ones_like(x)[None], img)
    # rays of light from the top right
    th = torch.atan2(y + 0.2, X - 1.6)
    rays = (torch.sin(th * 40) > 0.6).float() * (y < 0.55).float() * 0.08
    img = img + srgb("#ffffff")[:, None, None] * rays[None]
    # floating islands with waterfalls: far (small, pale) then near
    for (cx, cy, w, seed, far) in ((0.08, 0.36, 0.12, 1.0, True), (1.18, 0.3, 0.09, 2.0, True), (1.62, 0.42, 0.16, 3.0, False),
                                   (0.95, 0.5, 0.07, 4.0, True), (-0.02, 0.62, 0.2, 5.0, False)):
        top, under = island(X, y, cx, cy, w, seed)
        rock = srgb("#8a7aa8" if far else "#6a5a8a")[:, None, None] * torch.ones_like(x)[None]
        rock = rock * (1 - 0.25 * ((X - cx) > 0).float())[None]
        img = torch.where(under[None], rock, img)
        img = torch.where(top[None], srgb("#7ad86a" if far else "#5ac84a")[:, None, None] * torch.ones_like(x)[None], img)
        fall = ((X - cx - w * 0.3).abs() < w * 0.06) & (y > cy) & (y < cy + w * 2.5)
        img = torch.where(fall[None], srgb("#e8f8ff")[:, None, None] * (0.85 + 0.15 * (torch.frac(y * 60) > 0.5).float())[None], img)
        img = torch.where(ink_edges(top | under | fall, 2)[None], INK[:, None, None] * torch.ones_like(x)[None], img)
    # the sea of clouds: lobes in flat bands
    for i, (cy, col, hi) in enumerate(((0.62, "#f2e8f8", "#ffffff"), (0.72, "#d8d0f0", "#f2ecff"), (0.84, "#b8b0e0", "#d8d0f0"))):
        edge = cy - 0.035 * (torch.sin(X * 9 + i * 2) * 0.5 + 0.5) - 0.02 * noise2(X * 6, torch.zeros_like(X) + i, 7.0 + i, 2)
        cl = y > edge
        img = torch.where(cl[None], srgb(col)[:, None, None] * torch.ones_like(x)[None], img)
        light = cl & (y < edge + 0.02)
        img = torch.where(light[None], srgb(hi)[:, None, None] * torch.ones_like(x)[None], img)
        img = torch.where(ink_edges(cl, 2)[None], srgb("#5a4a8a")[:, None, None] * torch.ones_like(x)[None], img)
    save_rgb(img * 1.05, os.path.join(ASSETS, "backgrounds", "manga.jpg"))


def sanctuary(S=2048):
    """The board's platform: a round slab of white marble with a gold rim,
    seen from above at an angle, a rocky underside, two waterfalls; ink lines."""
    x, y = grid(S, S)
    u = (x - 0.5) * 2
    top_y, ry = 0.2, 0.09
    slab = ((u / 0.98) ** 2 + ((y - top_y) / ry) ** 2) < 1
    side = (u.abs() < 0.98) & (y > top_y) & (y < top_y + 0.07) & ~slab
    side = side | (((u / 0.98) ** 2 + ((y - top_y - 0.07) / ry) ** 2) < 1) & (y > top_y)
    depth = ((y - top_y - 0.07) / 0.75).clamp(0, 1)
    rock = (y > top_y + 0.05) & (u.abs() < 0.9 * (1 - depth ** 0.7) * (0.92 + 0.12 * noise2(y * 10, u * 3, 2.0, 3)))
    img = torch.zeros(3, S, S, device=DEV)
    alpha = torch.zeros(S, S, device=DEV)
    rockc = band(((u + 1) / 2).clamp(0, 1), [0.45], [srgb("#8a7ab8"), srgb("#5a4a88")])
    img = torch.where(rock[None], rockc, img)
    sidec = band(((u + 1) / 2).clamp(0, 1), [0.5], [srgb("#f2ecff"), srgb("#c8c0e8")])
    img = torch.where(side[None], sidec, img)
    goldband = side & (y < top_y + 0.035)
    img = torch.where(goldband[None], srgb("#ffc83a")[:, None, None] * torch.ones_like(x)[None], img)
    img = torch.where(slab[None], srgb("#fbf8ff")[:, None, None] * torch.ones_like(x)[None], img)
    for sgn in (-1, 1):
        fall = ((u - 0.55 * sgn).abs() < 0.04) & (y > top_y + 0.06) & (y < 0.98)
        img = torch.where(fall[None], srgb("#e8f8ff")[:, None, None] * (0.85 + 0.15 * (torch.frac(y * 80) > 0.5).float())[None], img)
        rock = rock | fall
    shape = slab | side | rock
    edges = ink_edges(shape, 4) | ink_edges(slab, 3) & shape
    img = torch.where(edges[None], INK[:, None, None] * torch.ones_like(x)[None], img)
    alpha = (shape | edges).float()
    fade = 1 - ((y - 0.85) / 0.15).clamp(0, 1)
    Image.fromarray(to_rgba(img, alpha * fade)).save(os.path.join(ASSETS, "backgrounds", "manga_sanctuaire.png"), optimize=True)


def tatami(S=128):
    """Light square of the sanctuary: white marble, a thin gold inlay, inked edge."""
    x, y = grid(S, S)
    vein = (noise2(x * 3, y * 3, 2.0, 4) - 0.5).abs() < 0.02
    c = srgb("#f6f2ff")[:, None, None] * torch.ones_like(x)[None]
    c = torch.where(vein[None], srgb("#d8d0ec")[:, None, None] * torch.ones_like(x)[None], c)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    c = torch.where(((e - 0.09).abs() < 0.008)[None], srgb("#e8b83a")[:, None, None] * torch.ones_like(x)[None], c)
    c = torch.where((e < 0.015)[None], INK[:, None, None] * torch.ones_like(x)[None], c)
    return to_rgba(c, torch.ones_like(x))


def lacquer(S=128):
    """Dark square of the sanctuary: lapis lazuli with specks of gold."""
    x, y = grid(S, S)
    n = noise2(x * 5, y * 5, 3.0, 4)
    c = band(n, [0.45, 0.62], [srgb("#1e2a78"), srgb("#2a3a98"), srgb("#3a4ab0")])
    speck = noise2(x * 40, y * 40, 9.0, 1) > 0.86
    c = torch.where(speck[None], srgb("#ffd24a")[:, None, None] * torch.ones_like(x)[None], c)
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
    c = srgb("#f2eefc")[:, None, None] * (0.92 + 0.08 * noise2(x * 10, y * 10, 1.0, 2))[None]
    c = torch.where(((e > 0.12) & (e < 0.15))[None], srgb("#1b1420")[:, None, None] * torch.ones_like(x)[None], c)
    shine = (((e - 0.05).abs() < 0.015)).float()
    c = torch.where((shine > 0)[None], srgb("#ffc83a")[:, None, None] * torch.ones_like(x)[None], c)
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
        "light": "#f6f2ff", "dark": "#2a3a98",
        "frame top": "#f2eefc", "frame bottom": "#c8c0e8", "frame line": "#e8b44a",
        "coordinates": "#3a2a6a", "hole": "#1b1420", "hole rim": "#000000",
        "background top": "#2a2a6a", "background bottom": "#ff8a6a",
        "last move": "#ffe08a66", "selection": "#ff8ab48c", "legal move": "#1b1420aa", "hint": "#5aff8adc",
        "accent": "#ffc83a",
        "tile light": "manga_clair", "tile dark": "manga_sombre", "hole art": "manga_trou", "frame art": "manga_cadre",
        "background image": "manga.jpg", "planet image": "manga_sanctuaire.png", "planet size": 1.7, "planet top": 0.72,
        "petals": True, "piece set": "manga", "break fx": "ink",
    }
    with open(os.path.join(ASSETS, "themes", "9_manga.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
        sanctuary()
    if "tiles" in what:
        write_art("manga_clair", "Manga : marbre", "tile", [("idle", [tatami()])], 128)
        write_art("manga_sombre", "Manga : lapis", "tile", [("idle", [lacquer()])], 128)
    if "hole" in what:
        write_art("manga_trou", "Manga : impact d'encre", "hole", [("idle", ink_hole()), ("open", ink_hole(opening=True))], 128,
                  fps={"idle": 8, "open": 24}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("manga_cadre", "Manga : cadre laqué", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
