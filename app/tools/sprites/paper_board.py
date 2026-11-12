"""
paper_board.py — the "Origami" board (assets/themes/12_origami.json): a scene
of cut paper, layers laid on one another, each casting a soft shadow on the
one behind; the board is a sheet of washi and indigo paper, and a move tears
the square it leaves.

  assets/backgrounds/origami.jpg   cut-paper landscape: a cream sky, a vermilion
                                   sun, layered mountains from coral to deep
                                   indigo, paper clouds, a band of waves in front
  assets/art/origami_clair.*       light square: cream washi with its fibres
  assets/art/origami_sombre.*      dark square: indigo paper printed with seigaiha waves
  assets/art/origami_trou.*        hole: the paper torn open, a white fibrous edge (idle);
                                   the tear opening (open)
  assets/art/origami_cadre.*       the frame: vermilion paper, gold foil lines, folded corners

Still squares (no pattern running along the diagonals). The life is in the
scenery: paper cranes flying (drawn by the application).

    cd app/tools/sprites && python3 paper_board.py
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


def fibres(x, y, seed, k=1.0):
    """Washi paper: long fibres in every direction and a fine grain (around 1)."""
    f1 = noise2(x * 3 * k, y * 45 * k, seed, 2)
    f2 = noise2(x * 45 * k, y * 3 * k, seed + 1, 2)
    f3 = noise2((x + y) * 30 * k, (x - y) * 3 * k, seed + 2, 2)
    grain = noise2(x * 120 * k, y * 120 * k, seed + 3, 1)
    return 0.92 + 0.06 * (f1 + f2 + f3 - 1.5) * 1.3 + 0.04 * grain


def layer(img, mask, col, shadow_dx, shadow_dy, blur, strength=0.45):
    """Lays a cut sheet: its shadow on what is behind, then the sheet."""
    H, W = mask.shape
    dy, dx = int(shadow_dy * H), int(shadow_dx * W)
    sh = torch.zeros_like(mask)                                     # shifted down-right, nothing wrapping around
    sh[dy:, dx:] = mask[:H - dy, :W - dx]
    sh = gauss_blur(sh[None], blur)[0]
    img = img * (1 - strength * sh * (1 - mask))[None]
    edge = mask - gauss_blur(mask[None], 1.5)[0]                    # a faint light on the cut edge
    return over(img, col * (1 + 0.25 * edge.clamp_min(0))[None], mask)


def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    paper = fibres(X, y, 1.0, 1.5)
    k = smoothstep(0.0, 0.75, y)
    img = (C("#f7ecd8") * (1 - k)[None] + C("#f2c9a0") * k[None]) * paper[None]
    # the sun: a vermilion disc with a paler ring
    sx, sy, sr = ar * 0.72, 0.3, 0.17
    d = torch.sqrt((X - sx) ** 2 + (y - sy) ** 2)
    img = layer(img, (d < sr + 0.03).float(), C("#f4a070") * paper[None], 0.004, 0.008, 10, 0.3)
    img = layer(img, (d < sr).float(), C("#e0503a") * paper[None], 0.004, 0.008, 8, 0.35)
    # paper clouds: unions of discs, flat bottoms
    for i, (cx, cy, w) in enumerate(((0.35, 0.2, 0.2), (1.25, 0.16, 0.16), (1.55, 0.36, 0.14), (0.15, 0.42, 0.12))):
        m = torch.zeros_like(x)
        for j in range(5):
            bx = cx + (j - 2) * w * 0.32
            br = w * (0.28 + 0.18 * math.sin(j * 2.1 + i) ** 2)
            m = torch.maximum(m, (torch.sqrt((X - bx) ** 2 + ((y - cy + br * 0.4) * 1.2) ** 2) < br).float())
        m = m * (y < cy + w * 0.12).float()
        img = layer(img, m, C("#fffaf2") * paper[None], 0.003, 0.01, 10, 0.3)
    # mountains: far to near, each a cut sheet whose shadow falls on the one behind
    for i, (base_y, amp, freq, col) in enumerate(((0.55, 0.18, 1.6, "#eab8a0"), (0.62, 0.17, 2.1, "#d48a88"), (0.7, 0.15, 2.6, "#7a7aa8"),
                                                   (0.78, 0.13, 3.2, "#4a5a8e"), (0.87, 0.1, 3.8, "#2c3e72"))):
        n = noise2(X * freq, torch.zeros_like(X) + i * 3, 50.0 + i, 3)
        peaks = (torch.sin(X * freq * 2.2 + i) * 0.5 + 0.5) * 0.4 + n * 0.8
        top = base_y - amp * peaks
        m = (y > top).float()
        img = layer(img, m, C(col) * fibres(X, y, 60.0 + i)[None], 0.0035, 0.012, 12, 0.4)
    # a band of waves in front: seigaiha cut in deep indigo
    wy = 0.93
    u = X * 14
    row = torch.floor(u)
    fu = u - row - 0.5
    arc = torch.sqrt(fu * fu + ((y - wy) * 14) ** 2)
    rings = (torch.frac(arc * 3) < 0.5).float() * (arc < 1).float()
    m = (y > wy - 0.5 / 14 * torch.sqrt((0.25 - fu * fu).clamp_min(0)) * 2).float()
    img = layer(img, m, C("#1e2a52") * (0.85 + 0.3 * rings)[None], 0.0, 0.01, 10, 0.5)
    v = (x - 0.5) ** 2 + (y - 0.5) ** 2
    img = img * (1 - 0.35 * v)[None]
    save_rgb(img * 0.92, os.path.join(ASSETS, "backgrounds", "origami.jpg"))


def washi(S=128):
    x, y = grid(S, S)
    c = C("#f1e6d2") * fibres(x, y, 3.0)[None]
    g = torch.Generator(device="cpu").manual_seed(4)
    for i in range(6):     # a few flecks of darker fibre
        fx, fy, a = float(torch.rand(1, generator=g)), float(torch.rand(1, generator=g)), float(torch.rand(1, generator=g)) * math.pi
        du, dv = x - fx, y - fy
        along = du * math.cos(a) + dv * math.sin(a)
        across = -du * math.sin(a) + dv * math.cos(a)
        c = c * (1 - 0.25 * (torch.exp(-(across / 0.004) ** 2) * (along.abs() < 0.04).float()))[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.45)
    return to_rgba(c, torch.ones_like(x))


def indigo(S=128):
    """Indigo chiyogami: rows of seigaiha waves, a thin gold line on each."""
    x, y = grid(S, S)
    c = C("#2c3e72") * fibres(x, y, 7.0)[None]
    n = 3                                                           # waves across a square
    u, v = x * n, y * n
    best = torch.full_like(x, 9.0)
    for oy, ox in ((0, 0), (0.5, 0.5), (1, 0), (1, 1), (0, 1), (0.5, -0.5), (0.5, 1.5)):
        cy_ = torch.floor(v - oy) + oy + 1.0
        cx_ = torch.floor(u - ox) + ox + 0.5
        d = torch.sqrt((u - cx_) ** 2 + (v - cy_) ** 2)
        best = torch.where((d < 0.75) & (v < cy_), torch.minimum(best, d), best)
    rings = torch.frac(best * 4.5)
    line_ = torch.exp(-((rings - 0.5) / 0.08) ** 2) * (best < 0.75).float()
    c = c * (1 + 0.35 * line_)[None] + C("#c8a860") * (torch.exp(-((best - 0.72) / 0.015) ** 2) * 0.5)[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.6)
    return to_rgba(c, torch.ones_like(x))


def tear(S=128, frames=1, opening=False):
    """The paper torn open: a dark void, a ragged white fibrous edge, curled flaps."""
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v)
    th = torch.atan2(v, u)
    rag = 0.12 * noise2(torch.cos(th) * 3, torch.sin(th) * 3, 3.0, 3) + 0.05 * noise2(torch.cos(th) * 14, torch.sin(th) * 14, 4.0, 2)
    n_frames = 9 if opening else frames
    for f in range(n_frames):
        k = 1.0 if not opening else 1 - (1 - (f + 1) / n_frames) ** 2
        R = (0.72 + rag) * k
        inside = (r < R).float()
        depth = (r / R.clamp_min(1e-3)).clamp(0, 1)
        # inside: the dark sheet below, in the shadow of the torn edge (the light comes from the top left)
        c = C("#2a1e18") * (0.55 + 0.45 * smoothstep(-0.6, 0.8, u + v))[None] * (0.7 + 0.3 * depth)[None]
        c = c * (1 - 0.55 * torch.exp(-((R - r) / 0.12) ** 2) * (u + v < 0.3).float())[None]
        # the torn edge: the white core of the paper, frayed, wider on the side the tear pulled
        wid = 0.035 + 0.035 * (0.5 + 0.5 * torch.cos(th - 0.8))
        fray = noise2(th * 40, r * 8, 5.0, 2)
        fib = smoothstep(wid + 0.02, wid * 0.4, (r - R).abs() * (1.6 - fray))
        rim = fib * (r >= R - 0.03).float()
        c = over(c, C("#fbf6ec"), (rim * 1.2).clamp(0, 1))
        alpha = (inside + rim).clamp(0, 1)
        if opening:   # the tear runs open: a flash of white fibres at the edge
            c = over(c, C("#fbf6ec"), (torch.exp(-((r - R) / 0.06) ** 2) * (1 - k) * 1.5).clamp(0, 1))
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96):
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    c = C("#c8382e") * fibres(x, y, 9.0)[None]
    gold = torch.exp(-((e - 0.07) / 0.008) ** 2) + torch.exp(-((e - 0.26) / 0.008) ** 2)
    c = over(c, C("#e0b860"), gold.clamp(0, 1) * 0.9)
    # folded corners: a triangle of the paper's back on each corner
    for (cx, cy) in ((0, 0), (1, 0), (1, 1), (0, 1)):
        d = (x - cx).abs() + (y - cy).abs()
        tri = (d < 0.3).float()
        c = over(c, C("#f1e6d2") * (0.85 + 0.15 * (d / 0.3))[None], tri)
        c = c * (1 - 0.35 * torch.exp(-((d - 0.3) / 0.012) ** 2))[None]
    c = c * (1 + bevel(x, y, 0.05) * 0.5)
    return [to_rgba(c, band_)]


def theme():
    t = {
        "id": "origami", "name": "Origami", "pattern": "flat", "pattern strength": 0.2, "seed": 19,
        "light": "#f1e6d2", "dark": "#2c3e72",
        "frame top": "#c8382e", "frame bottom": "#8a2420", "frame line": "#e0b860",
        "coordinates": "#f6efe2", "hole": "#140e0a", "hole rim": "#fbf6ec",
        "background top": "#f7ecd8", "background bottom": "#2c3e72",
        "last move": "#e0503a55", "selection": "#e0503a8c", "legal move": "#c8382ebb", "hint": "#3a9a5adc",
        "accent": "#e0503a",
        "tile light": "origami_clair", "tile dark": "origami_sombre", "hole art": "origami_trou", "frame art": "origami_cadre",
        "background image": "origami.jpg", "cranes": True, "piece set": "origami", "break fx": "paper",
    }
    with open(os.path.join(ASSETS, "themes", "12_origami.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "tiles" in what:
        write_art("origami_clair", "Origami : washi", "tile", [("idle", [washi()])], 128)
        write_art("origami_sombre", "Origami : papier indigo", "tile", [("idle", [indigo()])], 128)
    if "hole" in what:
        write_art("origami_trou", "Origami : papier déchiré", "hole", [("idle", tear()), ("open", tear(opening=True))], 128,
                  fps={"idle": 1, "open": 20}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("origami_cadre", "Origami : cadre vermillon", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
