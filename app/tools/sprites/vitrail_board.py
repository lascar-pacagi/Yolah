"""
vitrail_board.py — the "Vitraux" board (assets/themes/18_vitraux.json): a
cathedral window laid flat, its squares panes of leaded glass.

  assets/art/vitraux_clair.*   light square: pale amber "grisaille" glass, a
                               lattice of diamond quarries in lead, a small
                               painted quatrefoil in the middle
  assets/art/vitraux_sombre.*  dark square: deep blue glass in irregular pieces,
                               a few ruby and gold ones, a lead net
  assets/art/vitraux_trou.*    hole: a broken pane, dark, jagged shards still
                               in the lead round its edge, glints running over
                               them (idle); the pane bursting into coloured
                               shards (open)
  assets/art/vitraux_cadre.*   the frame: carved grey stone, a row of trefoil
                               arches, a gilt fillet along the inner edge
  assets/backgrounds/vitraux.jpg   (vitrail_scene.py: the nave at dusk)

    cd app/tools/sprites && python3 vitrail_board.py [tiles hole frame theme]
"""
import json
import math
import os
import random
import torch
from space import grid, noise2, to_rgba, write_art, bevel, ASSETS, TAU
from dino_board import smoothstep, C, over

LEAD = "#2a2a30"


def _voronoi(x, y, n, seed):
    """Irregular glass pieces: the index of the nearest of n random points and the
    gap to the second nearest (small near a lead line)."""
    rnd = random.Random(seed)
    pts = [(rnd.random(), rnd.random()) for _ in range(n)]
    pts += [(px + dx, py + dy) for (px, py) in pts for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]   # tiles seamlessly
    d = torch.stack([(x - px) ** 2 + (y - py) ** 2 for (px, py) in pts])
    two, idx = torch.topk(d, 2, dim=0, largest=False)
    return idx[0] % n, two[1].sqrt() - two[0].sqrt()


def light_tile(S=128):
    """Grisaille: pale amber and cream quarries (diamonds) in a lead lattice, the
    glass streaked and seedy, a small painted quatrefoil at the centre."""
    x, y = grid(S, S)
    u, v = (x + y) * 2, (x - y) * 2                                              # the diamond lattice (2 × 2 per square)
    fu, fv = torch.remainder(u, 1.0), torch.remainder(v, 1.0)                     # (frac of a negative is negative)
    lead = (torch.minimum(torch.minimum(fu, 1 - fu), torch.minimum(fv, 1 - fv)) < 0.045).float()
    cell = torch.floor(u) * 7 + torch.floor(v) * 3
    tone = 0.9 + 0.12 * torch.frac(torch.sin(cell * 12.9898) * 43758.5)
    c = C("#f2e4c0") * tone[None] * (0.92 + 0.12 * noise2(x * 14, y * 40, 1.0, 3))[None]
    c = c * (1 - 0.15 * (noise2(x * 60, y * 60, 3.0, 2) > 0.8).float())[None]  # seeds (bubbles) in the glass
    # the quatrefoil painted in brown grisaille at the centre
    cx, cy = x - 0.5, y - 0.5
    r = torch.sqrt(cx * cx + cy * cy)
    th = torch.atan2(cy, cx)
    lobe = 0.12 + 0.05 * torch.cos(4 * th)
    paint = (torch.abs(r - lobe) < 0.012).float() + (r < 0.03).float()
    c = c * (1 - 0.55 * paint.clamp(0, 1))[None]
    c = c * (1 - lead)[None] + C(LEAD) * lead[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.35)
    return to_rgba(c, torch.ones_like(x))


def dark_tile(S=128):
    """Deep blue glass cut in irregular pieces, a few ruby and gold, the lead between."""
    x, y = grid(S, S)
    idx, gap = _voronoi(x, y, 14, 21)
    rnd = random.Random(4)
    cols = []
    for i in range(14):
        k = rnd.random()
        cols.append("#b81a28" if k < 0.08 else ("#e0a830" if k < 0.13 else rnd.choice(["#1a3a9a", "#2448b0", "#14307e", "#2a52c0"])))
    c = torch.zeros_like(C("#000000") * x[None])
    for i, h in enumerate(cols):
        c = c + C(h) * (idx == i).float()[None]
    c = c * (0.85 + 0.25 * noise2(x * 20, y * 20, 2.0, 3))[None]
    lead = (gap < 0.025).float()
    edge = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    lead = torch.maximum(lead, (edge < 0.02).float())
    c = c * (1 - lead)[None] + C(LEAD) * lead[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.45)
    return to_rgba(c, torch.ones_like(x))


def broken(S=128, frames=1, opening=False):
    """The broken pane: dark, jagged shards of glass left in the lead round the edge,
    still (idle, one frame); the pane bursting into coloured shards (open)."""
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    th = torch.atan2(v, u)
    r = torch.maximum(u.abs(), v.abs())
    # a jagged inner edge: the shards still held
    saw = torch.abs(torch.remainder(th * 9 / TAU + 0.3 * torch.sin(th * 2), 1.0) - 0.5) * 2              # sharp points, as broken glass
    jag = 0.5 + 0.32 * saw * (0.7 + 0.3 * noise2(torch.cos(th) * 3, torch.sin(th) * 3, 5.0, 2))
    rnd = random.Random(8)
    shards = [(rnd.uniform(0, TAU), rnd.uniform(0.6, 1.2), rnd.choice(["#2448b0", "#b81a28", "#e0a830", "#2a8a4a", "#f2e4c0"])) for _ in range(18)]
    n_frames = 9 if opening else frames
    for f in range(n_frames):
        k = 1.0 if not opening else (f + 1) / n_frames
        R = jag * k + (1 - k) * 0.0
        hole = (r < R).float()
        c = C("#0a0a12") * (0.6 + 0.4 * (r / R.clamp_min(1e-3)).clamp(0, 1) ** 2)[None]
        rim = ((r >= R) & (r < 0.97)).float()                                          # the shards left in place
        sh_col = C("#2448b0") * 0.65 * (0.8 + 0.4 * noise2(torch.cos(th) * 5, torch.sin(th) * 5, 6.0, 1))[None]
        c = c * hole[None] + (sh_col * (0.7 + 0.5 * noise2(u * 6, v * 6, 4.0, 2))[None]) * rim[None]
        crack = (torch.abs(torch.remainder(th * 9 / TAU + 0.3 * torch.sin(th * 2), 1.0) - 0.5) > 0.47).float() * rim   # the cracks between the shards
        lead = ((r - R).abs() < 0.02).float() + (r > 0.97).float() + crack
        c = c * (1 - lead.clamp(0, 1))[None] + C(LEAD) * lead.clamp(0, 1)[None]
        alpha = (hole + rim + lead).clamp(0, 1)
        if opening:                                                                     # the bursting shards
            fl = 1 - k
            for (a, sp, hc) in shards:
                d = k * sp * 1.1
                sx, sy = d * math.cos(a), d * math.sin(a)
                tri = (torch.abs(u - sx) + torch.abs(v - sy) * 1.6) < 0.09 * (1 - 0.3 * k)
                c = torch.where(tri[None], C(hc) * 1.3, c)
                alpha = torch.maximum(alpha, tri.float())
            c = c + C("#fff4e0") * (torch.exp(-r * 2.5) * fl * 2.5)[None]
            alpha = (alpha + fl * torch.exp(-r * 1.8)).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96):
    """Carved grey stone, a row of trefoil arches cut along it, a gilt fillet inside."""
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    along = torch.where((x - 0.5).abs() > (y - 0.5).abs(), y, x)
    across = torch.where((x - 0.5).abs() > (y - 0.5).abs(), torch.minimum(x, 1 - x), torch.minimum(y, 1 - y))
    c = C("#8a8478") * (0.8 + 0.3 * noise2(x * 16, y * 16, 2.0, 4))[None] * (1.1 - 0.5 * across)[None]
    # the trefoil arches: a row of pointed niches, three per side
    cell = torch.frac(along * 3)
    cu, cv = (cell - 0.5) * 3 * 0.33, (across - 0.16) / 0.33
    arch = (torch.abs(cu) < 0.11 * (1 - (cv.clamp(0, 1)) ** 2) + 0.02) & (cv > -0.4) & (cv < 0.6)
    c = c * (1 - 0.45 * arch.float())[None]
    fillet = torch.exp(-((across - 0.31) / 0.012) ** 2)
    c = c * (1 - 0.6 * smoothstep(0.28, 0.33, across))[None] + C("#e0b048") * (fillet * 1.3)[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.5)
    return [to_rgba(c, band_)]


def theme():
    t = {
        "id": "vitraux", "name": "Vitraux", "pattern": "flat", "pattern strength": 0.2, "seed": 47,
        "light": "#f2e4c0", "dark": "#1a3a9a",
        "frame top": "#8a8478", "frame bottom": "#5a564e", "frame line": "#e0b048",
        "coordinates": "#f2e4c0", "hole": "#0a0a12", "hole rim": "#2448b0",
        "background top": "#0a0c18", "background bottom": "#1a1020",
        "last move": "#e0b04866", "selection": "#e0b0488c", "legal move": "#0a0a12aa", "hint": "#8affb0dc",
        "accent": "#e0b048",
        "tile light": "vitraux_clair", "tile dark": "vitraux_sombre", "hole art": "vitraux_trou",
        "frame art": "vitraux_cadre", "background image": "vitraux.jpg", "motes": True,
        "piece set": "vitraux", "break fx": "glass",
    }
    with open(os.path.join(ASSETS, "themes", "18_vitraux.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["tiles", "hole", "frame", "theme"]
    if "tiles" in what:
        write_art("vitraux_clair", "Vitraux : grisaille en losanges", "tile", [("idle", [light_tile()])], 128)
        write_art("vitraux_sombre", "Vitraux : verre bleu de Chartres", "tile", [("idle", [dark_tile()])], 128)
    if "hole" in what:
        write_art("vitraux_trou", "Vitraux : vitre brisée", "hole", [("idle", broken()), ("open", broken(opening=True))], 128,
                  fps={"idle": 6, "open": 20}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("vitraux_cadre", "Vitraux : pierre sculptée", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
