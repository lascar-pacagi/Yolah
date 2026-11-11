"""
battle_board.py — the "Généraux" board (assets/themes/10_generaux.json): a war
table at dawn over a battlefield, after the sun that pierced the fog on a
famous December morning:

  assets/backgrounds/generaux.jpg   the battlefield: the sun through the fog,
                                    hills, the smoke of the guns, regiments and
                                    flags far away
  assets/art/generaux_clair.*       light square: a staff map on parchment
  assets/art/generaux_sombre.*      dark square: walnut
  assets/art/generaux_trou.*        hole: a smoking cannonball crater (idle, open)
  assets/art/generaux_cadre.*       the frame: gilded wood, laurel leaves

    cd app/tools/sprites && python3 battle_board.py
"""
import json
import math
import os
import torch
from render import DEV, srgb, vec, gauss_blur
from space import grid, noise2, save_rgb, to_rgba, write_art, bevel, ASSETS, TAU


def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    k = (y / 0.6).clamp(0, 1)
    sky = srgb("#4a5a7a")[:, None, None] * (1 - k)[None] + srgb("#f2c08a")[:, None, None] * k[None]
    sx, sy = 1.15, 0.47
    d = torch.sqrt((X - sx) ** 2 + (y - sy) ** 2)
    sky = sky + srgb("#fff0c8")[:, None, None] * (torch.exp(-(d / 0.05) ** 2) * 3 + torch.exp(-(d / 0.3) ** 2) * 0.8)[None]
    th = torch.atan2(y - sy, X - sx)
    rays = (0.5 + 0.5 * torch.sin(th * 26 + noise2(th * 3, d * 2, 2.0, 2) * 3)) * torch.exp(-d * 2.5) * (y < sy).float()
    sky = sky + srgb("#ffe0b0")[:, None, None] * (rays * 0.35)[None]
    img = sky
    # hills, far to near, with fog between
    for i, (h0, amp, sc_, col) in enumerate(((0.5, 0.04, 3.0, "#6a6a7a"), (0.58, 0.05, 4.0, "#4a5040"), (0.7, 0.06, 2.5, "#2e3a24"),
                                              (0.84, 0.05, 2.0, "#1e2618"))):
        hgt = h0 - amp * noise2(X * sc_, torch.zeros_like(X) + i, 5.0 + i, 3)
        m = (y > hgt).float()
        c = srgb(col)[:, None, None] * (0.85 + 0.3 * noise2(X * 20, y * 20, i, 3))[None]
        img = img * (1 - m)[None] + c * m[None]
        fog = torch.exp(-((y - h0 - 0.03) / 0.04) ** 2) * (0.6 - 0.12 * i)
        img = img + srgb("#e8dcc8")[:, None, None] * fog[None] * 0.5
        if i == 1:      # regiments: rows of tiny figures, flags
            for j, (rx, ry) in enumerate(((0.05, 0.585), (1.1, 0.6), (1.22, 0.59))):
                rows = ((X - rx).abs() < 0.08).float() * ((y - ry).abs() < 0.008).float() * (torch.frac(X * 300) < 0.5).float()
                img = img * (1 - rows)[None] + srgb("#1a1a24" if j % 2 else "#3a2020")[:, None, None] * rows[None]
                pole = ((X - rx - 0.07).abs() < 0.0015).float() * (y > ry - 0.05).float() * (y < ry).float()
                flag = ((X - rx - 0.085).abs() < 0.015).float() * ((y - ry + 0.04).abs() < 0.009).float()
                img = img * (1 - pole)[None]
                fc = torch.where((X - rx - 0.07) < 0.01, srgb("#1e3a8a")[:, None, None], torch.where((X - rx - 0.07) < 0.02, srgb("#f2f0ea")[:, None, None], srgb("#c8242a")[:, None, None]))
                img = img * (1 - flag)[None] + fc * flag[None]
    # smoke of the guns, along the middle hills
    smoke = ((noise2(X * 4, y * 8, 9.0, 5) - 0.45) * 2.5).clamp(0, 1) * torch.exp(-((y - 0.64) / 0.06) ** 2)
    img = img * (1 - 0.5 * smoke)[None] + srgb("#d8d0c4")[:, None, None] * (smoke * 0.7)[None]
    v = ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    img = img * (1 - 0.55 * v)[None]
    save_rgb(img, os.path.join(ASSETS, "backgrounds", "generaux.jpg"))


def parchment(S=128):
    x, y = grid(S, S)
    n = noise2(x * 4, y * 4, 1.0, 5)
    c = srgb("#e2cc9a")[:, None, None] * (0.85 + 0.25 * n)[None]
    # contour lines of hills, a river, a dotted road
    h = noise2(x * 2.2, y * 2.2, 7.0, 3)
    contour = (torch.frac(h * 9) < 0.06).float()
    c = c * (1 - 0.1 * contour)[None]
    c = c * (1 + 0.15 * contour)[None] * (1 - 0.12 * contour)[None]
    stain = torch.exp(-(((x - 0.75) / 0.12) ** 2 + ((y - 0.25) / 0.1) ** 2)) * 0.2
    c = c * (1 - stain)[None]
    c = c * (1 + bevel(x, y) * 0.4)
    return to_rgba(c, torch.ones_like(x))


def walnut(S=128):
    x, y = grid(S, S)
    grain = noise2(x * 2.5, y * 28, 3.0, 4)
    rings = 0.5 + 0.5 * torch.sin((y * 20 + grain * 6) * 1.0)
    c = srgb("#4a2e1c")[:, None, None] * (0.7 + 0.35 * rings + 0.2 * grain)[None]
    c = c * (1 + bevel(x, y) * 0.5)
    return to_rgba(c, torch.ones_like(x))


def crater(S=128, frames=12, opening=False):
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v)
    th = torch.atan2(v, u)
    R0 = 0.72 + 0.08 * noise2(torch.cos(th) * 2 + 4, torch.sin(th) * 2, 6.0, 3)
    n_frames = 8 if opening else frames
    for f in range(n_frames):
        ph = f / frames
        k = 1.0 if not opening else 1 - (1 - (f + 1) / n_frames) ** 3
        R = R0 * k
        depth = (r / R).clamp(0, 1)
        inside = (r < R).float()
        c = srgb("#2a2018")[:, None, None] * (0.4 + 0.6 * depth ** 2)[None] * (0.8 + 0.4 * noise2(x * 12, y * 12, 3.0, 3))[None]
        rim = torch.exp(-((r - R) / 0.08) ** 2)
        c = c + srgb("#6a5a44")[:, None, None] * (rim * 0.8)[None]
        ember = (noise2(u * 6 + ph, v * 6, 8.0 + ph * 2, 2) > 0.7).float() * (depth < 0.4).float() * (0.5 + 0.5 * math.sin(TAU * ph * 2))
        c = c + srgb("#ff6a20")[:, None, None] * (ember * 0.8)[None]
        # smoke wisps rising over the crater
        smoke = ((noise2(u * 3, v * 3 + ph * 3, 4.0, 3) - 0.45) * 2).clamp(0, 1) * (r < R * 0.9).float() * 0.5
        c = c * (1 - smoke)[None] + srgb("#b8b0a4")[:, None, None] * smoke[None]
        alpha = (inside + rim * (r >= R).float() * 0.9).clamp(0, 1)
        if opening:
            fl = math.exp(-f * 0.6)
            c = c + srgb("#fff0b0")[:, None, None] * (fl * torch.exp(-r * 2) * 3)[None]
            alpha = (alpha + fl * torch.exp(-r * 1.5)).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96):
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    along = torch.where((x - 0.5).abs() > (y - 0.5).abs(), y, x)        # the position along the edge
    leaf = (0.5 + 0.5 * torch.sin(along * 60)) * torch.exp(-((e - 0.17) / 0.05) ** 2)
    c = srgb("#b88a3a")[:, None, None] * (0.6 + 0.6 * leaf + 0.2 * noise2(x * 30, y * 30, 2.0, 2))[None]
    c = c * (1 - 0.5 * torch.exp(-((e - 0.06) / 0.012) ** 2) - 0.5 * torch.exp(-((e - 0.29) / 0.012) ** 2))[None]
    for (cx, cy) in ((0.165, 0.165), (0.835, 0.165), (0.835, 0.835), (0.165, 0.835)):
        d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        ros = (d < 0.1).float() * (0.6 + 0.4 * (0.5 + 0.5 * torch.cos(torch.atan2(y - cy, x - cx) * 8)))
        c = c * (1 - (d < 0.1).float())[None] + srgb("#e8c060")[:, None, None] * ros[None]
    c = c * (1 + bevel(x, y, 0.04) * 0.7)
    return [to_rgba(c * 1.1, band_)]


def theme():
    t = {
        "id": "generaux", "name": "Généraux", "pattern": "flat", "pattern strength": 0.2, "seed": 13,
        "light": "#e2cc9a", "dark": "#4a2e1c",
        "frame top": "#b88a3a", "frame bottom": "#6a4a1a", "frame line": "#e8c060",
        "coordinates": "#f2e2c0", "hole": "#2a2018", "hole rim": "#6a5a44",
        "background top": "#4a5a7a", "background bottom": "#1e2618",
        "last move": "#ffe08a66", "selection": "#ffd27a8c", "legal move": "#2a2018aa", "hint": "#a6ff8cdc",
        "accent": "#e8c060",
        "tile light": "generaux_clair", "tile dark": "generaux_sombre", "hole art": "generaux_trou", "frame art": "generaux_cadre",
        "background image": "generaux.jpg", "smoke": True, "piece set": "generaux", "break fx": "cannon",
    }
    with open(os.path.join(ASSETS, "themes", "10_generaux.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "tiles" in what:
        write_art("generaux_clair", "Généraux : carte", "tile", [("idle", [parchment()])], 128)
        write_art("generaux_sombre", "Généraux : noyer", "tile", [("idle", [walnut()])], 128)
    if "hole" in what:
        write_art("generaux_trou", "Généraux : cratère", "hole", [("idle", crater()), ("open", crater(opening=True))], 128,
                  fps={"idle": 8, "open": 24}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("generaux_cadre", "Généraux : cadre doré", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
