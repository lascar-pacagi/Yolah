"""
cpu_board.py — the "Informatique" board (assets/themes/16_informatique.json):
the history of computing, from the punched card to the circuit board.

  assets/art/informatique_clair.*   light square: a punched card, cream, its
                                    printed columns and a few punched holes
  assets/art/informatique_sombre.*  dark square: a green circuit board, copper
                                    traces at right angles and 45°, gold pads,
                                    white silkscreen marks
  assets/art/informatique_trou.*    hole: a burnt-out square, charred, embers
                                    breathing along its edge, a wisp of smoke
                                    (idle); sparks and a flash as it burns (open)
  assets/art/informatique_cadre.*   the frame: brushed aluminium, screws, a
                                    thin blue light along its inner edge
  assets/backgrounds/informatique.jpg   (cpu_scene.py: the museum hall)

    cd app/tools/sprites && python3 cpu_board.py [tiles hole frame theme]
"""
import json
import math
import os
import random
import torch
from space import grid, noise2, to_rgba, write_art, bevel, ASSETS, TAU
from dino_board import smoothstep, C, over


def card(S=128):
    """A punched card: cream paper, faint printed digits in columns (thin
    rows of marks), a few rectangular holes punched through."""
    x, y = grid(S, S)
    c = C("#efe4c4") * (0.94 + 0.08 * noise2(x * 30, y * 30, 1.0, 3))[None]
    # printed digit marks: 10 rows × 12 columns of tiny dashes
    col = torch.frac(x * 12)
    row = torch.frac(y * 10)
    mark = ((col > 0.35) & (col < 0.65) & (row > 0.42) & (row < 0.58)).float()
    c = c * (1 - 0.22 * mark)[None]
    # the punched holes (dark: the board shows through), a fixed pattern
    rnd = random.Random(5)
    holes = torch.zeros_like(x)
    for cc in range(12):
        for rr in rnd.sample(range(10), 2 if cc % 3 else 1):
            hx, hy = (cc + 0.5) / 12, (rr + 0.5) / 10
            holes = torch.maximum(holes, (((x - hx).abs() < 0.022) & ((y - hy).abs() < 0.04)).float())
    c = c * (1 - holes)[None] + C("#2a2218") * holes[None]
    # a coloured stripe along the top edge, as on many cards
    c = over(c, C("#c8a8d8") * 0.9, ((y < 0.06) & (y > 0.02)).float() * 0.5)
    c = c * (1 + bevel(x, y, 0.05) * 0.35)
    return to_rgba(c, torch.ones_like(x))


def board_pcb(S=128):
    """A circuit board: green solder mask, copper traces (horizontal, vertical
    and at 45°), gold pads and vias, white silkscreen marks."""
    x, y = grid(S, S)
    c = C("#0f3a24") * (0.85 + 0.2 * noise2(x * 20, y * 20, 2.0, 3))[None]
    rnd = random.Random(11)
    tr = torch.zeros_like(x)
    w = 0.012
    for _ in range(7):                                    # straight traces
        if rnd.random() < 0.5:
            yy = rnd.choice([0.15, 0.3, 0.45, 0.6, 0.75, 0.88])
            x0, x1 = sorted((rnd.random(), rnd.random()))
            tr = torch.maximum(tr, (((y - yy).abs() < w) & (x > x0) & (x < x1)).float())
        else:
            xx = rnd.choice([0.12, 0.27, 0.42, 0.58, 0.73, 0.9])
            y0, y1 = sorted((rnd.random(), rnd.random()))
            tr = torch.maximum(tr, (((x - xx).abs() < w) & (y > y0) & (y < y1)).float())
    for _ in range(3):                                    # 45° traces
        ox, oy, L = rnd.random() * 0.6 + 0.2, rnd.random() * 0.6 + 0.2, 0.25
        u = (x - ox) + (y - oy)
        v = (x - ox) - (y - oy)
        tr = torch.maximum(tr, ((v.abs() < w * 1.4) & (u.abs() < L)).float())
    c = over(c, C("#c8843a") * (0.9 + 0.2 * noise2(x * 60, y * 60, 3.0, 2))[None], tr * 0.85)
    # pads and vias
    for _ in range(10):
        px, py = rnd.random() * 0.84 + 0.08, rnd.random() * 0.84 + 0.08
        d = torch.sqrt((x - px) ** 2 + (y - py) ** 2)
        c = over(c, C("#e8c060") * (1.1 - 6 * d)[None], (d < 0.028).float())
        c = over(c, C("#0a1a10"), (d < 0.01).float())
    # an IC footprint: a row of rectangular pads
    for i in range(5):
        px = 0.3 + i * 0.1
        c = over(c, C("#e8c060"), (((x - px).abs() < 0.02) & ((y - 0.06).abs() < 0.03)).float())
    # silkscreen: a white outline box and small marks
    box = ((((x - 0.62).abs() < 0.16) & ((y - 0.5).abs() < 0.1)) & ~(((x - 0.62).abs() < 0.15) & ((y - 0.5).abs() < 0.09))).float()
    c = over(c, C("#e8ece8"), box * 0.8)
    c = c * (1 + bevel(x, y, 0.05) * 0.5)
    return to_rgba(c, torch.ones_like(x))


def burnt(S=128, frames=10, opening=False):
    """The burnt-out square: charred, cracked, embers breathing along a jagged
    edge, a wisp of smoke rising (idle); opening: sparks fly out and a flash
    fades as the square burns."""
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.maximum(u.abs(), v.abs())                                       # square-ish
    th = torch.atan2(v, u)
    edge = 0.86 + 0.06 * noise2(torch.cos(th) * 3 + 4, torch.sin(th) * 3, 6.0, 3)
    n_frames = 9 if opening else frames
    rnd = random.Random(3)
    sparks = [(rnd.uniform(0, TAU), rnd.uniform(0.5, 1.0)) for _ in range(14)]
    for f in range(n_frames):
        ph = f / frames
        k = 1.0 if not opening else 1 - (1 - (f + 1) / n_frames) ** 2
        R = edge * k
        inside = (r < R).float()
        depth = (r / R).clamp(0, 1)
        c = C("#120c08") * (0.4 + 0.8 * depth ** 2)[None] * (0.7 + 0.4 * noise2(u * 9, v * 9, 7.0, 3))[None]
        cracks = torch.exp(-((noise2(u * 5, v * 5, 8.0, 3) - 0.5) / 0.02) ** 2) * smoothstep(0.2, 0.8, depth)
        c = c + C("#ff6a1a") * (cracks * (0.5 + 0.5 * math.sin(TAU * ph)) * 0.6)[None]
        rimz = torch.exp(-((r - R) / 0.06) ** 2)
        c = c + C("#ff8a3a") * (rimz * (0.7 + 0.3 * math.sin(TAU * ph + 1.3)) * 1.1)[None]
        alpha = (inside + rimz * (r >= R).float()).clamp(0, 1)
        if not opening:   # the wisp of smoke: a soft column drifting up (whole cycles per loop)
            sy = torch.remainder(v + 2 * ph, 2.0) - 1
            wisp = torch.exp(-((u - 0.15 * torch.sin(v * 3 + TAU * ph)) / 0.18) ** 2) * smoothstep(1.0, -0.6, v) * (0.6 + 0.4 * noise2(u * 4, sy * 4, 9.0, 2))
            c = c + C("#8a8a8a") * (wisp * 0.25 * inside)[None]
        else:
            fl = 1 - k
            c = c + C("#ffe0a0") * (torch.exp(-r * 2.2) * fl * 3)[None]
            for (a, sp) in sparks:
                d = (f + 1) / n_frames * sp * 1.1
                sx, sy = d * math.cos(a), d * math.sin(a)
                dd = torch.sqrt((u - sx) ** 2 + (v - sy) ** 2)
                s_ = torch.exp(-(dd / 0.04) ** 2) * fl
                c = c + C("#ffd27a") * (s_ * 3)[None]
                alpha = torch.maximum(alpha, s_.clamp(0, 1))
            alpha = (alpha + fl * torch.exp(-r * 1.5)).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96):
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    along = torch.where((x - 0.5).abs() > (y - 0.5).abs(), y, x)
    across = torch.where((x - 0.5).abs() > (y - 0.5).abs(), torch.minimum(x, 1 - x), torch.minimum(y, 1 - y))
    brushed = 0.85 + 0.15 * noise2(along * 200, across * 4, 2.0, 2)
    c = C("#b8bcc4") * brushed[None] * (1.15 - 0.6 * across)[None]
    # screws every third of the side
    cell = torch.frac(along * 3)
    d = torch.sqrt(((cell - 0.5) / 3) ** 2 + (across - 0.14) ** 2)
    screw = (d < 0.035).float()
    c = over(c, C("#8a8e96") * (1.2 - 8 * d)[None], screw)
    c = over(c, C("#4a4e56"), screw * (((cell - 0.5).abs() * 0.33 < 0.004) | ((across - 0.14).abs() < 0.004)).float())
    # the blue light along the inner edge
    led = torch.exp(-((across - 0.31) / 0.012) ** 2)
    c = c * (1 - 0.6 * smoothstep(0.27, 0.33, across))[None] + C("#4ab8ff") * (led * 1.4)[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.5)
    return [to_rgba(c, band_)]


def theme():
    t = {
        "id": "informatique", "name": "Informatique", "pattern": "flat", "pattern strength": 0.2, "seed": 41,
        "light": "#efe4c4", "dark": "#0f3a24",
        "frame top": "#b8bcc4", "frame bottom": "#6a6e76", "frame line": "#4ab8ff",
        "coordinates": "#e8eef6", "hole": "#120c08", "hole rim": "#ff8a3a",
        "background top": "#0a0c12", "background bottom": "#1a1410",
        "last move": "#4ab8ff66", "selection": "#4ab8ff8c", "legal move": "#0a0c12aa", "hint": "#a6ff8cdc",
        "accent": "#4ab8ff",
        "tile light": "informatique_clair", "tile dark": "informatique_sombre", "hole art": "informatique_trou",
        "frame art": "informatique_cadre", "background image": "informatique.jpg", "motes": True,
        "piece set": "informatique",
    }
    with open(os.path.join(ASSETS, "themes", "16_informatique.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["tiles", "hole", "frame", "theme"]
    if "tiles" in what:
        write_art("informatique_clair", "Informatique : carte perforée", "tile", [("idle", [card()])], 128)
        write_art("informatique_sombre", "Informatique : circuit imprimé", "tile", [("idle", [board_pcb()])], 128)
    if "hole" in what:
        write_art("informatique_trou", "Informatique : composant grillé", "hole", [("idle", burnt()), ("open", burnt(opening=True))], 128,
                  fps={"idle": 5, "open": 20}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("informatique_cadre", "Informatique : aluminium brossé", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
