"""
ice_board.py — the "Banquise" board (assets/themes/11_banquise.json): an ice
floe in the arctic night, under an aurora. Every move breaks the ice behind the
piece: the board is a floe that gives way.

  assets/backgrounds/banquise.jpg   the polar night: stars, an aurora in green
                                    and violet curtains, icebergs on the
                                    horizon, the black sea with floating floes
                                    and the aurora's reflection
  assets/art/banquise_clair.*       light square: packed snow, a few glints
  assets/art/banquise_sombre.*      dark square: blue glacier ice, cracks and bubbles inside
  assets/art/banquise_trou.*        hole: black water, floating ice chips (idle);
                                    the ice cracking and sinking (open)
  assets/art/banquise_cadre.*       the frame: blocks of ice, frost, snowflakes at the corners

The squares are still (no wave running across the board: the user found that
disturbing on the space board); the life is in the scenery (aurora, snow: drawn
by the application).

    cd app/tools/sprites && python3 ice_board.py
"""
import json
import math
import os
import torch
from render import DEV, srgb, gauss_blur
from space import grid, noise2, save_rgb, to_rgba, write_art, bevel, stars, ASSETS, TAU


def smoothstep(e0, e1, v):
    t = ((v - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def C(h):
    return srgb(h)[:, None, None]


def over(img, col, a):
    return img * (1 - a)[None] + col * a[None]


def aurora(X, y, seed, y0, amp, height, c_low, c_high, strength, x0=-1.0, x1=3.0):
    """A ribbon of aurora: a folded lower edge (a thin bright seam), fine rays
    rising from it and fading up, brighter in patches along its length."""
    nx = noise2(X * 1.1, torch.zeros_like(X) + seed, seed, 3)
    edge = y0 + amp * (nx - 0.5) * 2 + 0.025 * torch.sin(X * 7 + seed + 3 * nx)
    dy = edge - y                                                   # > 0 above the edge
    seam = torch.exp(-(dy / 0.006) ** 2) * 0.8
    up = torch.exp(-dy.clamp_min(0) / height) * smoothstep(-0.004, 0.004, dy)
    rays = (0.25 + 1.1 * noise2(X * 160, y * 0.6, seed + 3, 2) ** 3)
    patch = smoothstep(0.35, 0.7, noise2(X * 0.9, torch.zeros_like(X) + seed * 3, seed + 9, 3))
    span = smoothstep(x0, x0 + 0.3, X) * smoothstep(x1, x1 - 0.3, X)
    k = ((up * rays + seam) * patch * span * strength).clamp(0, 2)
    mixc = (dy.clamp_min(0) / (height * 1.8)).clamp(0, 1)
    col = C(c_low) * (1 - mixc)[None] + C(c_high) * mixc[None]
    return col * k[None], k


AURORAS = ((1.0, 0.3, 0.05, 0.11, "#3dffa0", "#9a5aff", 1.3, -0.2, 1.4), (4.0, 0.2, 0.04, 0.09, "#5cffc8", "#d06aff", 0.9, 0.8, 2.0),
           (7.0, 0.38, 0.03, 0.07, "#2ae890", "#3a8aff", 0.7, 1.1, 1.9))


def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    hz = 0.6                                                       # the horizon
    # the night sky: deep blue at the top, teal glow near the horizon
    k = (y / hz).clamp(0, 1)
    sky = C("#02040c") * (1 - k)[None] + C("#0c2a3a") * (k ** 2)[None]
    sky = sky + stars(x, y, 900, 11, 1.0, W, 0.8) * (1 - k)[None]
    # the aurora: three curtains
    glow_sum = torch.zeros_like(x)
    for seed, y0, amp, hgt, cl, ch, st, x0, x1 in AURORAS:
        col, kk = aurora(X, y, seed, y0, amp, hgt, cl, ch, st, x0, x1)
        sky = sky + col * 1.4
        glow_sum = glow_sum + kk
    sky = sky + C("#1a6a5a") * gauss_blur(glow_sum[None], 40)[0][None] * 0.12          # the air lit by it
    img = sky
    # icebergs on the horizon, lit from above by the aurora
    for i, (cx, w, h) in enumerate(((0.25, 0.22, 0.14), (0.62, 0.12, 0.08), (1.35, 0.3, 0.17), (1.62, 0.1, 0.07))):
        q = ((X - cx) / w).clamp(-1, 1)
        qq = (q + 1) * 3.5                                          # 7 facets: straight slopes between random peaks
        i0 = torch.floor(qq)
        fr = qq - i0
        hk = lambda k: 0.45 + 0.55 * torch.frac(torch.sin(k * 12.9898 + i * 78.233) * 43758.5453)
        env = lambda k: (1 - ((k / 3.5 - 1).abs()) ** 1.6).clamp_min(0)
        h0, h1 = hk(i0) * env(i0), hk(i0 + 1) * env(i0 + 1)
        prof = h * (h0 + (h1 - h0) * fr) * (q.abs() < 1).float()
        m = ((y < hz) & (y > hz - prof)).float()
        shade_ = ((hz - y) / prof.clamp_min(1e-3)).clamp(0, 1)
        lit = (h1 > h0).float()                                     # the slopes facing the light (left)
        c = C("#2e5a78") * (0.55 + 0.75 * lit + 0.35 * shade_)[None] + C("#8affd0") * (0.18 * shade_ ** 3)[None]
        img = over(img, c, m)
    # the sea: black-blue, the aurora reflected and broken by the swell
    sea = (y >= hz).float()
    ry = (2 * hz - y).clamp(0, 1)                                  # the mirrored row
    swell = noise2(X * 3, y * 60, 9.0, 3)
    refl = torch.zeros(3, *x.shape, device=DEV)
    for seed, y0, amp, hgt, cl, ch, st, x0, x1 in AURORAS[:2]:
        col, _ = aurora(X + (swell - 0.5) * 0.04, ry, seed, y0, amp, hgt, cl, ch, st, x0, x1)
        refl = refl + col
    depth = ((y - hz) / (1 - hz)).clamp(0, 1)
    streak = smoothstep(0.45, 0.75, noise2(X * 2, y * 90, 12.0, 2))          # the swell breaks the reflection in streaks
    water = C("#03101a") * (1 - 0.5 * depth)[None] + refl * (0.55 * streak * (1 - depth) ** 2)[None]
    img = img * (1 - sea)[None] + water * sea[None]
    # floating floes, flattened by the perspective, bigger nearer
    for i in range(14):
        g = torch.Generator(device="cpu").manual_seed(100 + i)
        fx, fy = float(torch.rand(1, generator=g)) * ar, hz + 0.02 + float(torch.rand(1, generator=g)) ** 1.5 * 0.35
        sz = 0.02 + 0.12 * (fy - hz) / 0.35
        d = torch.sqrt(((X - fx) / sz) ** 2 + ((y - fy) / (sz * 0.25)) ** 2)
        ang = torch.atan2((y - fy) / 0.25, X - fx)
        d = d + 0.45 * (noise2(torch.cos(ang) * 1.5 + i, torch.sin(ang) * 1.5, 20.0 + i, 3) - 0.5) + 0.06 * torch.sin(ang * 9 + i)
        top = (d < 1).float()
        side = ((d >= 1) & (d < 1.0) | ((torch.sqrt(((X - fx) / sz) ** 2 + ((y - fy - sz * 0.06) / (sz * 0.25)) ** 2) < 1) & (d >= 1))).float()
        img = over(img, C("#1e4a64"), side)
        snow = C("#c8dcea") * (0.75 + 0.3 * noise2(X * 40, y * 40, 3.0, 2))[None] + C("#5affc0") * 0.05
        img = over(img, snow, top)
    # the near ice shelf under the board: a long white edge across the bottom
    edge = 0.9 + 0.025 * noise2(X * 4, torch.zeros_like(X), 31.0, 3)
    shelf = (y > edge).float()
    cliff = ((y > edge - 0.0) & (y < edge + 0.035)).float()
    sh = C("#d6e6f2") * (0.8 + 0.25 * noise2(X * 30, y * 30, 32.0, 3))[None]
    img = over(img, sh, shelf)
    img = over(img, C("#4a8ab0") * (0.6 + 0.4 * noise2(X * 60, y * 5, 33.0, 2))[None], cliff * (y > edge + 0.008).float())
    # a cold mist on the water
    mist = ((noise2(X * 3, y * 10, 40.0, 4) - 0.4) * 2).clamp(0, 1) * torch.exp(-((y - hz - 0.05) / 0.06) ** 2)
    img = img + C("#5a9ab0") * (mist * 0.25)[None]
    v = (x - 0.5) ** 2 + (y - 0.5) ** 2
    img = img * (1 - 0.6 * v)[None]
    save_rgb(img, os.path.join(ASSETS, "backgrounds", "banquise.jpg"))


def snow_tile(S=128):
    """Packed snow: soft drifts, a fine grain, a few glints."""
    x, y = grid(S, S)
    drift = noise2(x * 3, y * 3, 1.0, 4)
    grain = noise2(x * 40, y * 40, 2.0, 2)
    c = C("#e4eef6") * (0.86 + 0.14 * drift + 0.06 * grain)[None]
    c = c * (1 - 0.08 * smoothstep(0.55, 0.75, noise2(x * 5, y * 2, 3.0, 3)))[None] + C("#a8c8e8") * 0.03   # wind ripples, blue shade
    g = torch.Generator(device="cpu").manual_seed(5)
    for i in range(9):
        gx, gy = float(torch.rand(1, generator=g)), float(torch.rand(1, generator=g))
        d = torch.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        c = c + C("#ffffff") * (torch.exp(-(d / 0.012) ** 2) * 0.9)[None]
    c = c * (1 + bevel(x, y) * 0.5)
    return to_rgba(c * 0.95, torch.ones_like(x))


def voronoi_edges(x, y, n, seed):
    """Distance to the nearest edge of a Voronoi diagram of n random points (cracks)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = torch.rand(n, 2, generator=g).to(DEV)
    d = torch.stack([torch.sqrt((x - px) ** 2 + (y - py) ** 2) for px, py in pts.tolist()], 0)
    s, _ = d.sort(0)
    return s[1] - s[0]


def ice_tile(S=128):
    """Blue glacier ice: depth, cracks and bubbles inside, a glossy surface."""
    x, y = grid(S, S)
    deep = noise2(x * 2.5, y * 2.5, 5.0, 4)
    c = C("#1c5a86") * (0.7 + 0.5 * deep)[None] + C("#3ab8e0") * (0.25 * smoothstep(0.55, 0.8, deep))[None]
    e = voronoi_edges(x, y, 7, 3)
    crack = torch.exp(-(e / 0.008) ** 2)
    c = c + C("#d8f4ff") * (crack * 0.55)[None]
    e2 = voronoi_edges(x * 1.0 + 0.3, y, 14, 9)
    c = c + C("#9ad8f4") * (torch.exp(-(e2 / 0.005) ** 2) * 0.18)[None]
    g = torch.Generator(device="cpu").manual_seed(8)
    for i in range(12):
        bx, by, br = float(torch.rand(1, generator=g)), float(torch.rand(1, generator=g)), 0.006 + 0.012 * float(torch.rand(1, generator=g))
        d = torch.sqrt((x - bx) ** 2 + (y - by) ** 2)
        c = c + C("#e8f8ff") * (torch.exp(-((d - br) / 0.003) ** 2) * 0.35)[None]
    sheen = torch.exp(-(((x - 0.3) / 0.3) ** 2 + ((y - 0.3) / 0.3) ** 2)) * 0.1   # a soft round highlight (no diagonal: it would line up across the board)
    c = c + C("#ffffff") * sheen[None]
    c = c * (1 + bevel(x, y) * 0.7)
    return to_rgba(c, torch.ones_like(x))


def water_hole(S=128, frames=12, opening=False):
    """Black arctic water in a broken ring of ice; chips of ice bob on it.
    opening: cracks run across the plate, then it sinks and the water closes over it."""
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v)
    th = torch.atan2(v, u)
    R0 = 0.8 + 0.06 * noise2(torch.cos(th) * 3 + 4, torch.sin(th) * 3, 6.0, 3) + 0.04 * torch.sin(th * 11)
    n_frames = 10 if opening else frames
    plate = torch.tensor(ice_tile(S), device=DEV).float().permute(2, 0, 1)[:3] / 255.0
    plate = plate ** 2.2
    for f in range(n_frames):
        ph = f / frames
        if opening:
            k = (f + 1) / n_frames
            R = R0 * smoothstep(0.25, 0.9, torch.tensor(k)).item()
        else:
            R = R0
        inside = (r < R).float()
        swell = noise2(u * 2 + 0.3 * math.sin(TAU * ph), v * 2 + 0.3 * math.cos(TAU * ph), 4.0, 3)
        c = C("#020a12") * (0.6 + 0.6 * swell)[None] + C("#2a8aa8") * (torch.exp(-((r - R * 0.85) / 0.08) ** 2) * 0.25)[None]
        # chips of ice floating near the edge, bobbing
        for i in range(5):
            a = i * 1.3 + 0.4
            rr = R * (0.55 + 0.08 * i % 3)
            cx = math.cos(a) * rr + 0.02 * math.sin(TAU * ph + i)
            cy = math.sin(a) * rr + 0.02 * math.cos(TAU * ph + i * 2)
            dd = torch.maximum((u - cx).abs() * (1.2 + 0.3 * i % 2) + (v - cy).abs() * 0.7, (v - cy).abs() * 1.6) - 0.09
            chip = (dd < 0).float() * inside
            c = over(c, C("#c8e4f2") * (0.8 + 0.3 * (v - cy < -0.02).float())[None], chip)
        rim = torch.exp(-((r - R) / 0.05) ** 2)
        c = c + C("#e8f6ff") * (rim * 0.9)[None]
        alpha = (inside + rim * (r >= R).float()).clamp(0, 1)
        if opening:
            # the plate still over the water, cracked, sinking (darker, smaller) as k grows
            kk = (f + 1) / n_frames
            cr = torch.exp(-(voronoi_edges(x, y, 6, 21) / 0.01) ** 2) * min(1.0, kk * 3)
            pa = (1 - smoothstep(0.35, 0.85, torch.tensor(kk)).item())
            pc = plate * (1 - 0.6 * kk) + C("#ffffff") * cr[None] * 0.8
            a_plate = torch.full_like(x, pa) * (1 - cr * 0.6 * (kk > 0.3))
            c = c * (1 - a_plate)[None] + pc * a_plate[None]
            alpha = torch.maximum(alpha, a_plate)
            splash = torch.exp(-((r - (0.4 + kk)) / 0.06) ** 2) * (kk > 0.4) * (1 - kk) * 2
            c = c + C("#ffffff") * splash[None]
            alpha = (alpha + splash).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96):
    """Blocks of ice with frost, a snowflake on each corner."""
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    band_ = (e < 1 / 3).float()
    along = torch.where((x - 0.5).abs() > (y - 0.5).abs(), y, x)
    blocks = (torch.frac(along * 6) < 0.04).float()
    frost = noise2(x * 18, y * 18, 2.0, 3)
    c = C("#6aa8d0") * (0.75 + 0.45 * frost)[None] + C("#e8f6ff") * (0.25 * smoothstep(0.6, 0.8, frost))[None]
    c = c * (1 - 0.35 * blocks)[None]
    c = c * (1 - 0.5 * torch.exp(-((e - 0.04) / 0.01) ** 2) - 0.4 * torch.exp(-((e - 0.3) / 0.01) ** 2))[None]
    for (cx, cy) in ((0.165, 0.165), (0.835, 0.165), (0.835, 0.835), (0.165, 0.835)):
        du, dv = x - cx, y - cy
        d = torch.sqrt(du * du + dv * dv)
        a = torch.atan2(dv, du)
        arm = torch.exp(-((torch.remainder(a + math.pi / 6, math.pi / 3) - math.pi / 6) * d / 0.006) ** 2) * (d < 0.11).float()
        c = over(c, C("#ffffff"), arm.clamp(0, 1) * 0.95)
    c = c * (1 + bevel(x, y, 0.05) * 0.6)
    return [to_rgba(c * 1.05, band_)]


def theme():
    t = {
        "id": "banquise", "name": "Banquise", "pattern": "flat", "pattern strength": 0.2, "seed": 17,
        "light": "#e4eef6", "dark": "#1c5a86",
        "frame top": "#6aa8d0", "frame bottom": "#2a5a7a", "frame line": "#e8f6ff",
        "coordinates": "#dff2ff", "hole": "#020a12", "hole rim": "#e8f6ff",
        "background top": "#02040c", "background bottom": "#03101a",
        "last move": "#8affd066", "selection": "#8affd08c", "legal move": "#0a3a5aaa", "hint": "#a6ff8cdc",
        "accent": "#7affd0",
        "tile light": "banquise_clair", "tile dark": "banquise_sombre", "hole art": "banquise_trou", "frame art": "banquise_cadre",
        "background image": "banquise.jpg", "snow": True, "aurora": True, "piece set": "banquise", "break fx": "ice",
    }
    with open(os.path.join(ASSETS, "themes", "11_banquise.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "tiles" in what:
        write_art("banquise_clair", "Banquise : neige tassée", "tile", [("idle", [snow_tile()])], 128)
        write_art("banquise_sombre", "Banquise : glace bleue", "tile", [("idle", [ice_tile()])], 128)
    if "hole" in what:
        write_art("banquise_trou", "Banquise : eau noire", "hole", [("idle", water_hole()), ("open", water_hole(opening=True))], 128,
                  fps={"idle": 6, "open": 20}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("banquise_cadre", "Banquise : cadre de glace", "frame", [("idle", frame_art())], 96)
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
