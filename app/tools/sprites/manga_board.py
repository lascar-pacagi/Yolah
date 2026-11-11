"""
manga_board.py — the "Manga" board (assets/themes/9_manga.json), in flat
anime colours with ink lines:

  assets/backgrounds/manga.jpg   a sky sanctuary: a ringed gas giant, cumulus,
                                 floating islands with a shrine and waterfalls
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


def smoothstep(e0, e1, v):
    t = ((v - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def C(h):
    return srgb(h)[:, None, None]


def over(img, col, a):
    return img * (1 - a)[None] + col * a[None]


def cloud_field(X, y, H, base, height, count, rmin, rmax, seed, fill=True):
    """Cumulus: a density of soft puffs (towers rising from base, a flat sea
    below it), broken up by noise. Returns (alpha, light) where light is high on
    the edges facing the sun (upper right)."""
    import random
    g = random.Random(seed)
    dens = torch.zeros_like(X)
    for _ in range(count):
        bx = g.uniform(-0.15, 1.95)
        r = g.uniform(rmin, rmax)
        by = base - g.random() ** 1.7 * height + r * 0.4
        dens += torch.exp(-((X - bx) ** 2 + ((y - by) * 1.25) ** 2) / (r * r))
    if fill:
        dens += smoothstep(base - 0.01, base + 0.05, y) * 1.2
    dd = dens + (noise2(X * 9, y * 9, seed, 5) - 0.5) * 0.45 + (noise2(X * 30, y * 30, seed + 50, 3) - 0.5) * 0.18
    a = smoothstep(0.45, 0.5, dd)
    k = max(2, int(0.014 * H))
    dl = torch.roll(dd, shifts=(k, -k), dims=(0, 1))        # the density towards the sun
    light = ((dd - dl) * 3.5 + 0.25).clamp(0, 1)
    light = light * (0.75 + 0.25 * (1 - smoothstep(base - height, base + 0.06, y)))
    return a, light


def paint_clouds(img, X, y, H, base, height, count, rmin, rmax, seed, haze, hazecol):
    a, light = cloud_field(X, y, H, base, height, count, rmin, rmax, seed)
    shadow, lit, rim = C("#7078c0"), C("#ffffff"), C("#ffd29a")
    col = shadow * (1 - light)[None] + lit * light[None]
    col = col + rim * (smoothstep(0.75, 1.0, light) * 0.35)[None]
    col = col * (1 - haze) + hazecol * haze
    return over(img, col, a)


def island(X, y, H, cx, cy, w, seed, shrine=False, falls=(0.3,), trees=5, grassy=True):
    """A floating island painted on its own layer: grassy top, an earthen lip,
    a sculpted rocky underside with strata, hanging vines, round trees, an
    optional shrine, waterfalls falling into mist. Returns (premultiplied
    colour, alpha)."""
    import random
    g = random.Random(int(seed * 100))
    q, s = (X - cx) / w, (y - cy) / w
    pe = 1.5 / (H * w)                                    # a pixel in island units
    col = torch.zeros(3, *X.shape, device=DEV)
    acc = torch.zeros_like(X)

    def put(c, a):
        nonlocal col, acc
        col = col * (1 - a)[None] + c * a[None]
        acc = acc * (1 - a) + a

    # the rocky underside
    span = (1 - q * q).clamp(0, 1)
    depth = 1.3 * span ** 0.7 * (0.7 + 0.6 * noise2(q * 2.5, torch.zeros_like(q), seed + 1, 3)) \
        + 0.35 * (noise2(q * 12, torch.zeros_like(q), seed + 2, 2) - 0.5) * span
    edge_q = 1 + 0.06 * (noise2(s * 6, q * 0 + seed, seed + 3, 2) - 0.5)
    under = smoothstep(pe, -pe, s.neg()) * smoothstep(depth + pe, depth - pe, s) * smoothstep(edge_q + pe, edge_q - pe, q.abs())
    lit = (0.45 + 0.45 * q).clamp(0, 1)
    strata = 0.5 + 0.5 * torch.sin(s * 30 + noise2(q * 3, s * 3, seed + 4, 3) * 5)
    crack = smoothstep(0.66, 0.72, noise2(q * 6, s * 10, seed + 5, 3)) * 0.4
    tip = (s / depth.clamp_min(0.05)).clamp(0, 1)
    rock = C("#3e3060") * (1 - lit)[None] + C("#c8a084") * lit[None]
    rock = rock * (0.7 + 0.3 * strata)[None] * (1 - 0.35 * crack)[None] * (1 - 0.45 * tip)[None]
    rock = rock + C("#9ad0ff") * (0.18 * (1 - lit) * tip)[None]          # sky light bouncing under
    put(rock, under)
    # the earthen lip and the grassy top
    top_e = q * q + (s / 0.2) ** 2
    jag = 1 + 0.2 * (noise2(q * 5, s * 5, seed + 6, 3) - 0.5)
    lip = smoothstep(jag + 0.02, jag - 0.02, q * q + ((s - 0.05) / 0.2) ** 2)
    if grassy:
        put(C("#5a3a2a") * (0.7 + 0.5 * lit)[None], lip)
    top = smoothstep(jag + 0.02, jag - 0.02, top_e)
    gl = (0.55 + 0.35 * q - 0.6 * s / 0.2).clamp(0, 1)
    grass = C("#2f7a3e") * (1 - gl)[None] + C("#b4e878") * gl[None]
    grass = grass * (0.85 + 0.3 * noise2(q * 14, s * 14, seed + 7, 3))[None]
    if grassy:
        put(grass, top)
    # vines hanging from the lip
    vine_len = 0.1 + 0.35 * noise2(q * 9, torch.zeros_like(q), seed + 8, 2)
    vines = smoothstep(0.1, 0.0, (torch.frac(q * 7 + 3 * noise2(q * 2, s * 1.5, seed + 9, 2)) - 0.5).abs() * 2 - 0.9) * (noise2(q * 4, q * 0, seed + 13, 2) > 0.5).float() \
        * (s > 0.05).float() * smoothstep(vine_len, vine_len - 0.05, s) * smoothstep(1.0, 0.85, q.abs())
    if grassy:
        put(C("#2a5a30") * (0.8 + 0.4 * lit)[None], vines * 0.9)
    # trees: clusters of round crowns
    for _ in range(trees):
        tq = g.uniform(-0.75, 0.75)
        ts = -0.12 * (1 - tq * tq) ** 0.5 * g.uniform(-0.6, 0.9)
        tr = g.uniform(0.09, 0.16)
        trunk = smoothstep(0.018, 0.01, (q - tq).abs()) * (s < ts + 0.02).float() * (s > ts - tr * 1.2).float()
        put(C("#4a2e22"), trunk)
        crown = torch.zeros_like(q)
        shade = torch.zeros_like(q)
        for j in range(4):
            oq, os_, rr = g.uniform(-0.6, 0.6) * tr, -tr * 1.3 - g.uniform(0, 0.8) * tr, tr * g.uniform(0.6, 1.0)
            dd = torch.sqrt((q - tq - oq) ** 2 + (s - ts - os_) ** 2) / rr
            m = smoothstep(1 + pe / rr, 1 - pe / rr, dd)
            ll = (0.5 + 0.5 * ((q - tq - oq) - (s - ts - os_)) / rr * 0.7).clamp(0, 1)
            shade = torch.where(m > crown, ll, shade)
            crown = torch.maximum(crown, m)
        leaf = noise2(q * 40, s * 40, seed + 10, 2)
        tc = C("#1f5a34") * (1 - shade)[None] + C("#8fd86a") * shade[None]
        put(tc * (0.85 + 0.3 * leaf)[None], crown)
    if shrine:   # a white and gold pavilion, on the left of the top
        sq, ss = -0.3, -0.04
        base_ = ((q - sq).abs() < 0.22).float() * ((s - ss).abs() < 0.025).float()
        put(C("#f4f0ea") * (0.8 + 0.3 * lit)[None], base_)
        for pq in (-0.17, -0.06, 0.06, 0.17):
            pil = smoothstep(0.016, 0.01, (q - sq - pq).abs()) * (s < ss).float() * (s > ss - 0.2).float()
            put(C("#fbf8f2") * (0.75 + 0.35 * (pq + 0.2) / 0.4), pil)
        rq = (q - sq) / 0.3
        roof_bot = ss - 0.2 + 0.04 * rq * rq
        roof_top = ss - 0.3 + 0.1 * rq * rq
        roof = smoothstep(1.02, 0.98, rq.abs()) * (s < roof_bot).float() * (s > roof_top).float()
        put(C("#c8323a") * (0.7 + 0.4 * lit)[None], roof)
        put(C("#ffcf4a"), roof * smoothstep(0.012, 0.0, (s - roof_bot).abs()))
        put(C("#ffcf4a"), smoothstep(0.02, 0.012, ((q - sq) ** 2 + (s - roof_top + 0.03) ** 2).sqrt()))
    # waterfalls from the lip, into mist
    for fq in falls:
        x0 = fq + 0.04 * torch.sin(s * 2)
        s0 = 0.18 * (1 - fq * fq) ** 0.5
        wdt = 0.035 * (1 + (s - s0).clamp(0) * 0.25)
        length = 3.2 + g.uniform(0, 1.5)
        body = smoothstep(wdt, wdt * 0.6, (q - x0).abs()) * (s > s0).float() * smoothstep(s0 + length, s0 + length * 0.6, s)
        streak = noise2((q - x0) * 60, s * 3, seed + 11, 3)
        wc = C("#d8f2ff") * (0.75 + 0.45 * streak)[None]
        put(wc, body * (0.55 + 0.45 * streak))
        mist_s = s0 + length * 0.75
        mist = torch.exp(-((q - fq) / 0.35) ** 2 - ((s - mist_s) / 0.5) ** 2) * (0.4 + 0.4 * noise2(q * 6, s * 6, seed + 12, 3))
        put(C("#f4fbff"), mist.clamp(0, 0.7))
    return col, acc


def planet(img, X, y, px, py, pr, sky):
    """A gas giant with rings, lit from the upper right, a blue atmosphere at its rim."""
    a_ = -0.32
    dx, dy = X - px, y - py
    u = dx * math.cos(a_) + dy * math.sin(a_)
    v = -dx * math.sin(a_) + dy * math.cos(a_)
    re = torch.sqrt(u * u + (v / 0.2) ** 2) / pr
    ringd = (0.6 + 0.4 * torch.sin(re * 38 + 2 * noise2(re * 3, re * 0, 3.0, 2))) * smoothstep(1.32, 1.4, re) * smoothstep(2.35, 2.25, re)
    ringd = ringd * (1 - 0.85 * smoothstep(0.05, 0.0, (re - 1.9).abs()))
    ringc = C("#f2dcb8") * (0.8 + 0.3 * smoothstep(1.4, 2.3, re))[None]
    pe = 1.5 / (X.shape[0] * pr)
    d = torch.sqrt(dx * dx + dy * dy) / pr
    inside = smoothstep(1 + pe, 1 - pe, d)
    nz = torch.sqrt((1 - d * d).clamp(0))
    nx, ny = dx / pr, dy / pr
    L = (0.6, -0.45, 0.66)
    diff = (nx * L[0] + ny * L[1] + nz * L[2]).clamp(0)
    lat = v / pr + 0.06 * noise2(u / pr * 6, v / pr * 20, 5.0, 4)
    b1 = 0.5 + 0.5 * torch.sin(lat * 13)
    b2 = 0.5 + 0.5 * torch.sin(lat * 5 + 1.3)
    pc = C("#f4d6a0") * (1 - b1)[None] + C("#d8846a") * b1[None]
    pc = pc * (1 - 0.4 * b2)[None] + C("#a85a7a") * (0.4 * b2)[None]
    shadow_ring = smoothstep(0.02, 0.0, (v / pr - 0.12 - 0.15 * nx).abs() - 0.03) * 0.5    # the rings' shadow across the disc
    pc = pc * (0.06 + 1.15 * diff ** 0.85 * (1 - shadow_ring))[None]
    pc = pc + C("#8fd0ff") * ((1 - nz) ** 3 * (0.3 + diff) * 0.8)[None]
    pc = pc * 0.8 + sky * 0.2
    ringc = ringc * 0.85 + sky * 0.15
    behind = (v < 0).float()
    img = over(img, ringc, ringd * behind * 0.9)
    img = over(img, pc, inside)
    img = over(img, ringc, ringd * (1 - behind) * 0.9 * (1 - inside * (diff < 0.02).float() * 0.5))
    glow = torch.exp(-((d - 1) / 0.08) ** 2) * (d > 1).float() * 0.5
    return img + C("#9ad8ff") * glow[None]


def background(W=2560, H=1440):
    """A sky sanctuary painted like an anime background: a deep sky turning
    gold at the horizon, a ringed gas giant, the sun's rays, towering cumulus,
    floating islands (trees, a shrine, waterfalls into mist) and a sea of
    clouds below."""
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    sky = C("#101a52") * torch.ones_like(x)[None]
    sky = over(sky, C("#2a62c8"), smoothstep(0.0, 0.3, y))
    sky = over(sky, C("#7cc4f2"), smoothstep(0.2, 0.48, y))
    sky = over(sky, C("#ffe2b4"), smoothstep(0.42, 0.62, y))
    sx, sy = 1.52, 0.2
    ds = torch.sqrt((X - sx) ** 2 + (y - sy) ** 2)
    img = sky + C("#fff2d0") * (torch.exp(-(ds / 0.035) ** 2) * 5 + torch.exp(-(ds / 0.2) ** 2) * 0.7 + torch.exp(-ds / 0.7) * 0.25)[None]
    cirrus = smoothstep(0.55, 0.8, noise2(X * 2.5, y * 22, 4.0, 5)) * smoothstep(0.45, 0.1, y)
    img = over(img, C("#fff6ee"), cirrus * 0.3)
    img = planet(img, X, y, 0.62, 0.2, 0.19, sky)
    dm = torch.sqrt((X - 0.2) ** 2 + (y - 0.08) ** 2) / 0.025
    img = over(img, C("#e8eef8") * (0.7 + 0.3 * noise2(X * 80, y * 80, 2.0, 3))[None], smoothstep(1.05, 0.95, dm))
    hz = C("#f6e8dc")
    img = paint_clouds(img, X, y, H, 0.56, 0.16, 70, 0.025, 0.07, 11, 0.5, hz)
    for (cx, cy, w, sd) in ((0.78, 0.42, 0.05, 1.0), (1.42, 0.47, 0.06, 2.0), (0.3, 0.5, 0.04, 3.0)):
        c, a = island(X, y, H, cx, cy, w, sd, falls=(0.25,), trees=3)
        c = c * 0.55 + hz * a[None] * 0.45
        img = img * (1 - a)[None] + c
    img = paint_clouds(img, X, y, H, 0.68, 0.1, 80, 0.035, 0.09, 12, 0.25, hz)
    for (cx, cy, w, sd, sh, fl) in ((0.06, 0.36, 0.2, 4.0, True, (0.35, -0.55)), (1.16, 0.3, 0.085, 5.0, False, (0.1,)),
                                    (1.8, 0.48, 0.14, 6.0, False, (-0.3,))):
        c, a = island(X, y, H, cx, cy, w, sd, shrine=sh, falls=fl, trees=6)
        c = c * 0.9 + hz * a[None] * 0.1
        img = img * (1 - a)[None] + c
    img = paint_clouds(img, X, y, H, 0.82, 0.08, 70, 0.05, 0.12, 13, 0.08, hz)
    img = paint_clouds(img, X, y, H, 0.94, 0.06, 50, 0.07, 0.15, 14, 0.0, hz)
    # rays of the sun, over everything
    th = torch.atan2(y - sy, X - sx)
    rays = smoothstep(0.5, 0.75, noise2(torch.cos(th) * 6, torch.sin(th) * 6, 21.0, 3)) * torch.exp(-ds * 1.1)
    img = img + C("#fff0c8") * (rays * 0.3)[None]
    # motes of light
    import random
    g = random.Random(7)
    for _ in range(90):
        mx, my, mr = g.uniform(0, ar), g.uniform(0, 1), g.uniform(0.0015, 0.004)
        img = img + C("#fff4c8") * (torch.exp(-((X - mx) ** 2 + (y - my) ** 2) / (mr * mr)) * 1.5)[None]
    bright = (img - 1.0).clamp(0)
    img = img + gauss_blur(bright, 18) * 0.6
    v = ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    img = img * (1 - 0.35 * v)[None]
    save_rgb(img, os.path.join(ASSETS, "backgrounds", "manga.jpg"))


def sanctuary(S=2048):
    """The board's platform: a round slab of veined white marble with a gold
    rim seen from above at an angle, a sculpted rocky underside, two waterfalls."""
    x, y = grid(S, S)
    X = x
    c, a = island(X, y, S, 0.5, 0.2, 0.49, 30.0, falls=(-0.55, 0.55), trees=0, grassy=False)
    q, s = (X - 0.5) / 0.49, (y - 0.2) / 0.49
    pe = 2.0 / (S * 0.49)
    slab_e = q * q + (s / 0.2) ** 2
    rim = smoothstep(1 + pe, 1 - pe, q * q + ((s - 0.06) / 0.2) ** 2) * (s > -0.0).float()
    lit = (0.5 + 0.4 * q).clamp(0, 1)
    rimc = C("#b07a1a") * (1 - lit)[None] + C("#ffe08a") * lit[None]
    rimc = rimc * (0.85 + 0.3 * smoothstep(0.03, 0.0, (s - 0.03).abs()))[None]
    c = c * (1 - rim)[None] + rimc * rim[None]
    a = a * (1 - rim) + rim
    slab = smoothstep(1 + pe, 1 - pe, slab_e)
    vein = smoothstep(0.03, 0.0, (noise2(q * 3, s * 9, 31.0, 5) - 0.5).abs()) * 0.25
    mc = C("#fbf8ff") * (0.85 + 0.15 * (0.5 - s / 0.4))[None] * (1 - vein)[None] + C("#b8b0d8") * vein[None]
    c = c * (1 - slab)[None] + mc * slab[None]
    a = a * (1 - slab) + slab
    fade = 1 - ((y - 0.85) / 0.15).clamp(0, 1)
    rgb = c / a.clamp_min(1e-4)[None]
    Image.fromarray(to_rgba(rgb, a * fade)).save(os.path.join(ASSETS, "backgrounds", "manga_sanctuaire.png"), optimize=True)


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
