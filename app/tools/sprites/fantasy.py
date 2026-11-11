"""
fantasy.py — the pictures and drawings of the "Légendes" theme
(assets/themes/8_legendes.json), the board of the orc hordes against the
alliance of elves and dwarves (set "legendes"):

  assets/backgrounds/legendes.jpg      misty mountains at dusk: a smoking volcano
                                       on the left, a white citadel and a forest on
                                       the right
  assets/backgrounds/legendes_roc.png  the rock spur the board stands on
  assets/art/legendes_clair.*          light square: limestone, moss in the joints
  assets/art/legendes_sombre.*         dark square: slate with a rune that glows
  assets/art/legendes_trou.*           hole: a chasm with lava far below (idle, open)
  assets/art/legendes_cadre.*          the frame (9-grid): carved stone, gold, braziers

    cd app/tools/sprites && python3 fantasy.py
"""
import json
import math
import os
import torch
from PIL import Image
from render import DEV, F, srgb, vnoise, fbm, aces, to_srgb, gauss_blur, vec
from space import grid, noise2, save_rgb, to_rgba, write_art, bevel, stars, ASSETS, TAU


def ridge(x, seed, scale, amp, octaves=5):
    """A mountain profile: the height of a ridge line over x."""
    n = torch.zeros_like(x)
    a, f = amp, scale
    for o in range(octaves):
        v = vnoise(torch.stack([x * f, torch.full_like(x, seed + o * 3.7), torch.full_like(x, 0.5)], -1))
        n = n + a * (1 - (v * 2 - 1).abs())          # sharp crests
        a *= 0.5
        f *= 2.1
    return n


# ─── the background ─────────────────────────────────────────────────────────
def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    # the sky: deep blue at the top, burning orange at the horizon, a red glow on the left (the volcano)
    top, mid, hor = srgb("#141a33"), srgb("#5a2d4a"), srgb("#f08a3c")
    k = (y / 0.62).clamp(0, 1)
    sky = top[:, None, None] * (1 - k) ** 2 + mid[:, None, None] * 2 * k * (1 - k) + hor[:, None, None] * k ** 2
    sky = sky + srgb("#ff3a1a")[:, None, None] * torch.exp(-((X - 0.08) ** 2 / 0.08 + (y - 0.55) ** 2 / 0.05)) * 0.8
    clouds = (noise2(X * 2.5, y * 9, 1.0, 5) - 0.45).clamp_min(0) * 1.6 * (y < 0.55).float()
    sky = sky * (1 - 0.35 * clouds) + srgb("#ffb070")[:, None, None] * clouds * (y * 1.4).clamp(0, 1) * 0.6
    sky = sky + stars(x, y, 400, 5, 0.5, W, 0.6) * (1 - k)[None] ** 3
    img = sky
    # mountain layers, far to near: lighter and bluer far away (aerial perspective)
    layers = [(0.50, 0.10, 1.5, "#6a4a6a", 0.0), (0.58, 0.12, 2.2, "#3e2e48", 0.15), (0.70, 0.12, 3.0, "#241c2c", 0.3),
              (0.82, 0.1, 4.0, "#120e16", 0.5)]
    for i, (base_h, amp, sc, col, mist) in enumerate(layers):
        hgt = base_h - ridge(X, 10 + i * 7, sc, amp)
        if i == 1:   # the volcano on the left
            v = torch.exp(-((X - 0.08) / 0.18) ** 2) * 0.22
            hgt = torch.minimum(hgt, base_h - v + 0.03 * ((X - 0.08).abs() < 0.03).float())
        m = (y > hgt).float()
        c = srgb(col)[:, None, None] * (0.8 + 0.4 * noise2(X * 12, y * 12, i, 3))
        img = img * (1 - m) + c * m
        # mist at the foot of each layer
        fog = torch.exp(-((y - base_h - 0.04) / 0.05) ** 2) * mist
        img = img + srgb("#c88a7a")[:, None, None] * fog * 0.35
    # lava glow on the volcano's crater, smoke rising
    crater = torch.exp(-(((X - 0.08) / 0.02) ** 2 + ((y - 0.37) / 0.012) ** 2))
    img = img + srgb("#ff6a1a")[:, None, None] * crater * 3
    smoke = ((noise2(X * 6 + y * 3, y * 4, 3.3, 5) - 0.4).clamp_min(0) * 2.5) * torch.exp(-((X - 0.08 - (0.37 - y) * 0.4) / (0.03 + (0.37 - y) * 0.5)) ** 2) * (y < 0.37).float()
    img = img * (1 - 0.6 * smoke.clamp(0, 1)) + srgb("#4a3030")[:, None, None] * smoke.clamp(0, 1) * 0.5
    # a white citadel on a far peak on the right: towers, a warm light in the windows
    cx = 1.15
    for (dx, w, h) in ((0.0, 0.012, 0.13), (-0.025, 0.009, 0.09), (0.025, 0.009, 0.1), (-0.045, 0.007, 0.06), (0.045, 0.007, 0.07)):
        tower = ((X - cx - dx).abs() < w).float() * (y > 0.47 - h).float() * (y < 0.5).float()
        roof = ((X - cx - dx).abs() < w * (1 - (0.47 - h - y) / 0.03)).float() * (y > 0.44 - h).float() * (y <= 0.47 - h).float()
        tm = (tower + roof).clamp(0, 1)
        img = img * (1 - tm) + srgb("#d8cfc8")[:, None, None] * tm * (0.6 + 0.4 * (X > cx + dx).float())
        win = tower * (((y * 300).floor() % 6) == 0).float() * ((X - cx - dx).abs() < w * 0.3).float()
        img = img + srgb("#ffc070")[:, None, None] * win * 1.5
    # a forest on the near hills, on the right
    fh = 0.83 - 0.05 * noise2(X * 30, torch.zeros_like(X), 2.0, 2) - 0.04 * (X - 1.2).clamp_min(0)
    trees = (y > fh).float() * (X > 1.05).float()
    img = img * (1 - trees) + srgb("#0c140e")[:, None, None] * trees
    # vignette
    v = ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    img = img * (1 - 0.6 * v)[None]
    save_rgb(img, os.path.join(ASSETS, "backgrounds", "legendes.jpg"))


# ─── the rock spur under the board ──────────────────────────────────────────
def rock(S=1536):
    """A rock spur: a jagged silhouette, a relief (height map lit from the
    left, the normal from its slopes), cracks, irregular strata, grass and
    moss on the top, mist at the bottom."""
    x, y = grid(S, S)
    u = (x - 0.5) * 2
    top = 0.18 + 0.012 * noise2(x * 25, torch.zeros_like(x), 1.0, 3)
    depth = ((y - top) / 0.82).clamp(0, 1)
    jag = 0.08 * (noise2(y * 9, torch.full_like(y, 2.0), 4.0, 4) - 0.5) + 0.05 * (noise2(y * 30, torch.full_like(y, 5.0), 5.0, 3) - 0.5)
    half = 0.96 * (1 - depth ** 1.4) + jag * (0.3 + depth)
    inside = (y > top).float() * (u.abs() < half).float()
    # relief: big boulders, strata, fine grain
    hgt = torch.zeros_like(x)
    amp, fx, fy = 0.6, 6.0, 4.0
    for o in range(4):     # ridged noise: sharp facets, taller than wide (cliff faces)
        n = noise2(x * fx, y * fy, 2.0 + o * 1.3, 1)
        hgt = hgt + amp * (1 - (2 * n - 1).abs())
        amp *= 0.5; fx *= 2.1; fy *= 2.3
    hgt = hgt + 0.6 * (1 - (u / half.clamp_min(0.05)).abs().clamp(0, 1) ** 2)      # rounded across the cliff
    gx = torch.zeros_like(hgt); gy = torch.zeros_like(hgt)
    gx[:, 1:-1] = (hgt[:, 2:] - hgt[:, :-2]) * S / 2 * 0.12
    gy[1:-1, :] = (hgt[2:, :] - hgt[:-2, :]) * S / 2 * 0.12
    nrm = torch.nn.functional.normalize(torch.stack([-gx, -gy, torch.ones_like(gx)], 0), dim=0)
    L = torch.nn.functional.normalize(vec(-0.65, -0.45, 0.6), dim=0)
    dif = (nrm * L[:, None, None]).sum(0).clamp_min(0)
    rim = ((nrm * vec(0.7, -0.2, 0.3)[:, None, None]).sum(0)).clamp_min(0) ** 3          # the volcano's red glow from the right
    crack = ((noise2(x * 18, y * 10, 7.0, 4) - 0.5).abs() < 0.006) & (noise2(x * 5, y * 5, 1.5, 2) > 0.5)
    tone = noise2(x * 3, y * 3, 8.0, 3)
    c = srgb("#5f5a54")[:, None, None] * (0.75 + 0.5 * tone)[None]
    c = c * (0.25 + 1.2 * dif)[None] + srgb("#ff6a3a")[:, None, None] * (rim * 0.1)[None]
    c = c * (1 - 0.45 * crack.float())[None]
    # grass on the top, moss hanging down
    grass = (y < top + 0.025 + 0.02 * noise2(x * 50, y * 0, 5.0, 2)).float()
    moss = (noise2(x * 12, y * 22, 6.0, 4) > 0.6).float() * (y < top + 0.18).float()
    g = srgb("#3d5e2a")[:, None, None] * (0.6 + 0.6 * noise2(x * 60, y * 60, 9.0, 2))[None] * (0.4 + dif)[None]
    c = c * (1 - torch.maximum(grass, moss * 0.7))[None] + g * torch.maximum(grass, moss * 0.7)[None]
    edge = ((half - u.abs()) * S * 0.5).clamp(0, 1)
    alpha = inside * edge * ((y - top) * S).clamp(0, 1)
    mist = ((y - 0.62) / 0.3).clamp(0, 1)
    c = c * (1 - mist)[None] + srgb("#5a3a50")[:, None, None] * mist[None] * 0.6
    alpha = alpha * (1 - mist ** 1.5)
    Image.fromarray(to_rgba(c * 1.3, alpha)).save(os.path.join(ASSETS, "backgrounds", "legendes_roc.png"), optimize=True)


# ─── squares ────────────────────────────────────────────────────────────────
def flagstone(x, y, seed):
    """Irregular flagstones: 2 × 2 stones per square, joints, wear."""
    jx = (x * 2 + 0.08 * noise2(y * 4, x * 4, seed, 2))
    jy = (y * 2 + 0.08 * noise2(x * 4, y * 4, seed + 1, 2))
    fx, fy = jx - torch.floor(jx), jy - torch.floor(jy)
    e = torch.minimum(torch.minimum(fx, 1 - fx), torch.minimum(fy, 1 - fy))
    joint = torch.exp(-(e / 0.035) ** 2)
    stone_id = torch.floor(jx) * 3 + torch.floor(jy) * 7
    tint = 0.85 + 0.25 * torch.frac(torch.sin(stone_id * 12.9898) * 43758.5)
    return joint, tint, e


def tile_light(S=128):
    x, y = grid(S, S)
    joint, tint, e = flagstone(x, y, 1.0)
    n = noise2(x * 10, y * 10, 2.0, 5)
    c = srgb("#c9b48e")[:, None, None] * (tint * (0.8 + 0.35 * n))[None]
    moss = joint * (noise2(x * 16, y * 16, 3.0, 3) > 0.45).float()
    c = c * (1 - 0.75 * joint)[None] + srgb("#4f6a2e")[:, None, None] * (moss * 0.6)[None]
    c = c * (1 + 0.25 * (e > 0.04).float() * ((x + y) < 1).float() * 0)[None]
    c = c * (1 + bevel(x, y) * 0.6)
    return to_rgba(c, torch.ones_like(x))


def rune_mask(x, y):
    """A rune made of strokes (original, no real alphabet): a stave and three
    branches, small, in the middle of one of the four flagstones."""
    u, v = (x - 0.25) * 5, (y - 0.27) * 5
    def seg(ax, ay, bx, by, w=0.045):
        px, py = u - ax, v - ay
        dx, dy = bx - ax, by - ay
        h = ((px * dx + py * dy) / (dx * dx + dy * dy)).clamp(0, 1)
        return torch.exp(-((px - h * dx) ** 2 + (py - h * dy) ** 2) / (w * w))
    m = seg(0, -0.5, 0, 0.5) + seg(0, -0.2, 0.3, -0.45) + seg(0, 0.05, -0.3, -0.2) + seg(0, 0.3, 0.3, 0.05)
    return m.clamp(0, 1)


def tile_dark(S=128, frames=12):
    out = []
    x, y = grid(S, S)
    joint, tint, e = flagstone(x, y, 7.0)
    n = noise2(x * 10, y * 10, 9.0, 5)
    rune = rune_mask(x, y)
    for f in range(frames):
        ph = f / frames
        c = srgb("#3a3a44")[:, None, None] * (tint * (0.75 + 0.4 * n))[None]
        c = c * (1 - 0.7 * joint)[None]
        glow_k = 0.15 + 0.25 * (0.5 + 0.5 * math.sin(TAU * ph))
        engraved = rune * 0.45
        c = c * (1 - engraved)[None] + srgb("#ff9a3a")[:, None, None] * (rune * glow_k * 0.5)[None]
        c = c + srgb("#ff7a2a")[:, None, None] * gauss_blur((rune * glow_k)[None], 2.0) * 0.25
        c = c * (1 + bevel(x, y) * 0.6)
        out.append(to_rgba(c, torch.ones_like(x)))
    return out


def chasm(S=128, frames=12, opening=False):
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    th = torch.atan2(v, u)
    r = torch.sqrt(u * u + v * v)
    jag = 0.72 + 0.08 * noise2(torch.cos(th) * 3 + 5, torch.sin(th) * 3, 2.0, 3)
    n_frames = 10 if opening else frames
    for f in range(n_frames):
        ph = f / frames
        k = 1.0
        if opening:
            k = 1 - (1 - min(1.0, (f + 1) / n_frames)) ** 3
        R = jag * k
        inside = (r < R).float()
        depth = (r / R.clamp_min(1e-3)).clamp(0, 1)
        lava = (noise2(u * 3 + ph * 2, v * 3, 4.0 + ph, 4) * 1.2 - 0.2).clamp(0, 1)
        glow_k = 0.7 + 0.3 * math.sin(TAU * ph)
        c = srgb("#0a0606")[:, None, None] * torch.ones_like(x)[None]
        c = c + srgb("#ff4a10")[:, None, None] * ((1 - depth) ** 2 * lava * glow_k * 2.2)[None]
        c = c + srgb("#ffc040")[:, None, None] * ((1 - depth) ** 6 * lava * glow_k * 1.5)[None]
        # the broken lip: rock edges lit by the glow from below
        lip = torch.exp(-((r - R) / 0.06) ** 2)
        rockc = srgb("#4a3a32")[:, None, None] * (0.6 + 0.5 * noise2(x * 20, y * 20, 6.0, 3))[None]
        rockc = rockc + srgb("#ff6a20")[:, None, None] * (lip * 0.5 * glow_k)[None]
        alpha = (inside + lip * (r > R).float()).clamp(0, 1)
        c = c * inside[None] + rockc * (lip * (r >= R).float())[None]
        if opening:   # dust when the ground gives way
            dust = math.exp(-f * 0.6) * torch.exp(-r * 1.2)
            c = c + srgb("#c8b8a0")[:, None, None] * dust[None] * 1.2
            alpha = (alpha + dust * 0.9).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96, frames=8):
    out = []
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band = (e < 1 / 3).float()
    n = noise2(x * 20, y * 20, 8.0, 4)
    for f in range(frames):
        c = srgb("#6a6058")[:, None, None] * (0.65 + 0.5 * n)[None]
        # carving: two grooves and a gold inlay between them
        c = c * (1 - 0.5 * torch.exp(-((e - 0.07) / 0.012) ** 2) - 0.5 * torch.exp(-((e - 0.27) / 0.012) ** 2))[None]
        c = c + srgb("#e0b04a")[:, None, None] * (torch.exp(-((e - 0.17) / 0.01) ** 2) * 0.9)[None]
        # braziers in the corners: a stone bowl and flames that flicker
        for i, (cx, cy) in enumerate(((0.16, 0.16), (0.84, 0.16), (0.84, 0.84), (0.16, 0.84))):
            d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            c = c * (1 - 0.5 * (d < 0.1).float())[None]
            fl = 0.7 + 0.3 * math.sin(TAU * (f / frames * 2 + i * 0.3)) + 0.2 * math.sin(TAU * (f / frames * 5 + i))
            flame = torch.exp(-(d / (0.07 * fl)) ** 2)
            c = c + srgb("#ffb040")[:, None, None] * (flame * 3.0 * fl)[None] + srgb("#ff5010")[:, None, None] * (torch.exp(-(d / 0.14) ** 2) * fl)[None]
        c = c * (1 + bevel(x, y, 0.04) * 0.6)
        out.append(to_rgba(c, band))
    return out


def theme():
    t = {
        "id": "legendes", "name": "Légendes", "pattern": "stone", "pattern strength": 0.4, "seed": 11,
        "light": "#c9b48e", "dark": "#3a3a44",
        "frame top": "#6a6058", "frame bottom": "#3a342e", "frame line": "#e0b04a",
        "coordinates": "#f2dca8", "hole": "#0a0606", "hole rim": "#4a3a32",
        "background top": "#141a33", "background bottom": "#2a1416",
        "last move": "#ffc04a55", "selection": "#ffd27a8c", "legal move": "#ffb04aaa", "hint": "#a6ff8cdc",
        "accent": "#f2b45a",
        "tile light": "legendes_clair", "tile dark": "legendes_sombre", "hole art": "legendes_trou", "frame art": "legendes_cadre",
        "background image": "legendes.jpg", "planet image": "legendes_roc.png", "planet size": 1.9, "planet top": 0.62,
        "embers": True, "piece set": "legendes",
    }
    with open(os.path.join(ASSETS, "themes", "8_legendes.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "rock", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "rock" in what:
        rock()
    if "tiles" in what:
        write_art("legendes_clair", "Légendes : dalle claire", "tile", [("idle", [tile_light()])], 128)
        write_art("legendes_sombre", "Légendes : dalle runique", "tile", [("idle", tile_dark())], 128, fps={"idle": 6})
    if "hole" in what:
        write_art("legendes_trou", "Légendes : gouffre", "hole", [("idle", chasm()), ("open", chasm(opening=True))], 128,
                  fps={"idle": 8, "open": 24}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("legendes_cadre", "Légendes : cadre", "frame", [("idle", frame_art())], 96, fps={"idle": 8})
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
