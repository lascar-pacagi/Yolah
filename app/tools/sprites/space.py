"""
space.py — the pictures and drawings of the space theme (assets/themes/7_espace.json):

  assets/backgrounds/espace.jpg    deep space: nebulae, dust, stars, a far galaxy
  assets/backgrounds/planete.png   the small planet under the board (lit, craters,
                                   lights of bases on the night side, atmosphere)
  assets/art/espace_clair.*        light square: a titanium deck plate
  assets/art/espace_sombre.*       dark square: a navy plate, circuits that pulse
  assets/art/espace_trou.*         hole: a vortex into the void (idle, open)
  assets/art/espace_cadre.*        the board's frame (9-grid), blinking lights

Everything is computed pixel by pixel on the GPU (PyTorch), in linear light,
then tone mapped (render.aces) — the same look as the figurines.

    cd app/tools/sprites && python3 space.py
"""
import json
import math
import os
import numpy as np
import torch
from PIL import Image
from render import DEV, F, srgb, vnoise, fbm, aces, to_srgb, gauss_blur, vec

HERE = os.path.dirname(__file__)
ASSETS = os.path.join(HERE, "..", "..", "assets")
TAU = 2 * math.pi


def grid(w, h):
    y, x = torch.meshgrid(torch.arange(h, device=DEV, dtype=F), torch.arange(w, device=DEV, dtype=F), indexing="ij")
    return (x + 0.5) / w, (y + 0.5) / h


def noise2(x, y, z=0.0, octaves=5):
    p = torch.stack([x, y, torch.full_like(x, z)], -1)
    return fbm(p.reshape(-1, 3), octaves).reshape(x.shape)


def save_rgb(img, path, quality=92):
    """img (3, H, W) linear → sRGB file."""
    a = (to_srgb(aces(img)).clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(a).save(path, quality=quality)


def to_rgba(img, alpha):
    """(3, H, W) linear, (H, W) alpha → uint8 (H, W, 4)."""
    c = to_srgb(aces(img)).clamp(0, 1)
    out = torch.cat([c, alpha[None].clamp(0, 1)], 0).permute(1, 2, 0)
    return (out * 255 + 0.5).byte().cpu().numpy()


def stars(x, y, n, seed, size, w, bright=1.0):
    """n stars (a soft disc each) as a (H, W) map; sizes in pixels."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sx = torch.rand(n, generator=g)
    sy = torch.rand(n, generator=g)
    mag = torch.rand(n, generator=g) ** 6
    col = torch.rand(n, generator=g)
    out = torch.zeros(3, *x.shape, device=DEV)
    H = x.shape[0]
    for i in range(n):
        cx, cy = float(sx[i]), float(sy[i])
        r = size * (0.5 + 3.0 * float(mag[i]))
        x0, x1 = max(0, int(cx * w - r * 4)), min(w, int(cx * w + r * 4) + 1)
        y0, y1 = max(0, int(cy * H - r * 4)), min(H, int(cy * H + r * 4) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        dx = x[y0:y1, x0:x1] * w - cx * w
        dy = y[y0:y1, x0:x1] * H - cy * H
        d2 = dx * dx + dy * dy
        v = torch.exp(-d2 / (2 * r * r)) * (0.4 + 3.0 * float(mag[i])) * bright
        if float(mag[i]) > 0.5:   # diffraction spikes on the brightest
            v = v + 0.5 * bright * (torch.exp(-dx.abs() / (r * 0.6)) * torch.exp(-dy.abs() / (r * 12)) +
                                    torch.exp(-dy.abs() / (r * 0.6)) * torch.exp(-dx.abs() / (r * 12)))
        c = srgb("#ffd6a0") if float(col[i]) < 0.18 else srgb("#a9c6ff") if float(col[i]) < 0.4 else srgb("#ffffff")
        out[:, y0:y1, x0:x1] += c[:, None, None] * v
    return out


# ─── the background ─────────────────────────────────────────────────────────
def background(W=2560, H=1440):
    x, y = grid(W, H)
    ar = W / H
    X = x * ar
    img = torch.zeros(3, H, W, device=DEV)
    img += (srgb("#03040b") * (1 - y[None].permute(1, 2, 0)) + srgb("#0b0920") * y[None].permute(1, 2, 0)).permute(2, 0, 1)
    # nebulae: domain-warped noise in two colours, a darker dust lane
    wx = noise2(X * 1.3, y * 1.3, 3.1, 4)
    wy = noise2(X * 1.3 + 5.2, y * 1.3 + 1.7, 7.7, 4)
    n1 = noise2(X * 1.8 + 2.2 * wx, y * 1.8 + 2.2 * wy, 0.5, 6)
    n2 = noise2(X * 2.6 - 1.5 * wy, y * 2.6 + 1.5 * wx, 9.3, 6)
    band = torch.exp(-((y - 0.35 - 0.18 * torch.sin(X * 2.2)) / 0.26) ** 2)
    neb1 = ((n1 - 0.45).clamp_min(0) * 2.4) ** 2 * band
    neb2 = ((n2 - 0.5).clamp_min(0) * 2.6) ** 2 * (0.4 + 0.6 * band)
    dust = ((noise2(X * 4, y * 4, 4.4, 5) - 0.5).clamp_min(0) * 3).clamp(0, 1) * band
    img += srgb("#7a2a9a")[:, None, None] * neb1 * 0.9
    img += srgb("#1c7fa6")[:, None, None] * neb2 * 0.7
    img += srgb("#ff7a6a")[:, None, None] * (neb1 * neb2) * 0.8
    img *= (1 - 0.55 * dust)
    # a far spiral galaxy
    gx, gy = 0.62 * ar, 0.18
    dx, dy = (X - gx) / 0.09, (y - gy) / 0.04
    r = torch.sqrt(dx * dx + dy * dy) + 1e-4
    th = torch.atan2(dy, dx)
    arms = (0.5 + 0.5 * torch.cos(2 * th - 5.0 * torch.log(r))).clamp(0, 1) ** 2
    gal = torch.exp(-r * 1.4) * (0.35 + arms) + 2.5 * torch.exp(-r * r * 30)
    img += srgb("#ffe2c0")[:, None, None] * gal * 0.25
    # stars: many small, few bright
    img += stars(x, y, 2600, 1, 0.55, W, 0.9)
    img += stars(x, y, 90, 2, 1.1, W, 1.6)
    save_rgb(img, os.path.join(ASSETS, "backgrounds", "espace.jpg"))


# ─── the planet ─────────────────────────────────────────────────────────────
def crater_field(p, n, seed):
    """Craters on the unit sphere: (N, 3) points → height."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1).to(DEV)
    rad = (0.02 + 0.12 * torch.rand(n, generator=g) ** 3).to(DEV)
    h = torch.zeros(p.shape[0], device=DEV)
    for i in range(n):
        d = torch.acos((p @ c[i]).clamp(-1, 1)) / rad[i]
        bowl = (d * d - 1) * (d < 1).float() * 0.6
        rim = torch.exp(-((d - 1) / 0.25) ** 2) * 0.35
        h += (bowl + rim) * rad[i]
    return h


def planet(S=2048):
    x, y = grid(S, S)
    R = 0.40                                  # the disc's radius (the rest: the atmosphere's glow)
    u, v = (x - 0.5) / R, (y - 0.5) / R
    rr = u * u + v * v
    inside = rr < 1
    z = torch.sqrt((1 - rr).clamp_min(0))
    nrm = torch.stack([u, -v, z], -1)        # y up
    # the surface turns a little: the sphere's point under each pixel
    tilt = 0.35
    pu = nrm.reshape(-1, 3)
    py = pu[:, 1] * math.cos(tilt) - pu[:, 2] * math.sin(tilt)
    pz = pu[:, 1] * math.sin(tilt) + pu[:, 2] * math.cos(tilt)
    ps = torch.stack([pu[:, 0], py, pz], -1)
    h = crater_field(ps, 160, 3) + 0.05 * fbm(ps * 6, 5)
    lat = ps[:, 1]
    tone = fbm(ps * 2.5 + 3, 4)
    col_a, col_b, col_c = srgb("#8fa6b8"), srgb("#5b6a86"), srgb("#c9b7a0")
    alb = col_a * (1 - tone[:, None]) + col_b * tone[:, None]
    alb = alb * (1 - 0.35 * (lat.abs() > 0.8).float()[:, None]) + col_c * 0.35 * (fbm(ps * 9, 3) > 0.62).float()[:, None]
    alb = alb + srgb("#e8f2ff") * ((lat.abs() - 0.82).clamp_min(0) * 5).clamp(0, 1)[:, None]      # polar ice
    # normal from the height (finite differences on the sphere, approximated in screen space)
    hh = h.reshape(S, S)
    gx = torch.zeros_like(hh); gy = torch.zeros_like(hh)
    gx[:, 1:-1] = (hh[:, 2:] - hh[:, :-2]) * S * R * 0.5
    gy[1:-1, :] = (hh[2:, :] - hh[:-2, :]) * S * R * 0.5
    bump = torch.nn.functional.normalize(nrm - torch.stack([gx, -gy, torch.zeros_like(gx)], -1) * 0.9, dim=-1).reshape(-1, 3)
    L = torch.nn.functional.normalize(vec(-0.65, 0.55, 0.55), dim=0)
    ndl = (bump @ L)
    day = ndl.clamp_min(0)
    terminator = (ndl * 4).clamp(-1, 1)
    col = alb * (day[:, None] * 2.6 + 0.03)
    # lights of bases on the night side
    lights = ((fbm(ps * 40, 2) > 0.74).float() * (fbm(ps * 3 + 9, 3) > 0.55).float()) * (ndl < -0.05).float()
    col += srgb("#ffc070") * (lights * 2.0)[:, None]
    # atmosphere: a rim on the lit side, and a glow outside the disc
    rim = (1 - z.reshape(-1)) ** 3
    col += srgb("#5fc8ff") * (rim * (0.35 + 0.65 * (terminator * 0.5 + 0.5)))[:, None] * 1.8
    col = col.reshape(S, S, 3).permute(2, 0, 1) * inside[None]
    dist = torch.sqrt(rr)
    glow = torch.exp(-((dist - 1) / 0.035).clamp_min(0) ** 1.2) * (dist > 1).float()
    light_side = (0.5 + 0.5 * ((-u * 0.65 - v * 0.55) / dist.clamp_min(1e-3)))
    col += srgb("#4fb6ff")[:, None, None] * (glow * (0.25 + 0.75 * light_side) * 1.6)
    alpha = inside.float() + glow * (0.25 + 0.75 * light_side) * 0.9 * (~inside).float()
    # anti-aliased edge
    edge = ((1 - dist) * S * R).clamp(0, 1)
    alpha = torch.maximum(alpha * (dist > 1).float(), edge * inside.float()) + 0 * alpha
    alpha = torch.where(inside, edge, glow * (0.25 + 0.75 * light_side) * 0.9)
    Image.fromarray(to_rgba(col, alpha)).save(os.path.join(ASSETS, "backgrounds", "planete.png"), optimize=True)


# ─── squares, hole, frame (drawings of assets/art) ──────────────────────────
def write_art(aid, name, kind, frames_by_anim, size, fps=None, loops=None):
    rows = len(frames_by_anim)
    cols = max(len(f) for _, f in frames_by_anim)
    sheet = np.zeros((rows * size, cols * size, 4), np.uint8)
    anims = []
    for r, (an, frames) in enumerate(frames_by_anim):
        for c, f in enumerate(frames):
            sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = f
        anims.append({"name": an, "fps": (fps or {}).get(an, 8), "loop": (loops or {}).get(an, True), "frames": len(frames)})
    Image.fromarray(sheet).save(os.path.join(ASSETS, "art", aid + ".png"), optimize=True)
    with open(os.path.join(ASSETS, "art", aid + ".json"), "w") as f:
        json.dump({"name": name, "kind": kind, "size": size, "smooth": True, "on stone": False, "animations": anims}, f,
                  indent=2, ensure_ascii=False)


def bevel(x, y, w=0.06):
    """Light on the top-left edges, shade on the bottom-right ones."""
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    k = (1 - e / w).clamp(0, 1)
    top_left = (x < y).float() * 0 + ((x + y) < 1).float()
    return k * (top_left * 0.6 - (1 - top_left) * 0.5)


def plate_light(S=128):
    x, y = grid(S, S)
    brushed = noise2(x * 3, y * 60, 1.0, 3)
    n = noise2(x * 8, y * 8, 2.0, 4)
    base = srgb("#8d95a4")
    c = base[:, None, None] * (0.82 + 0.18 * brushed + 0.1 * (n - 0.5))
    # an engraved hexagon in the middle
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    hexd = torch.maximum(u.abs() * 0.866 + v.abs() * 0.5, v.abs()) - 0.55
    groove = torch.exp(-(hexd / 0.025) ** 2)
    c = c * (1 - 0.35 * groove) + srgb("#ffffff")[:, None, None] * 0.12 * torch.exp(-((hexd - 0.03) / 0.02) ** 2)
    # screws in the corners
    for (sx, sy) in ((0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)):
        d = torch.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        c = c * (1 - 0.6 * (d < 0.035).float()) + srgb("#e6ebf2")[:, None, None] * ((d < 0.03).float() * (0.6 + 0.4 * ((x - sx) < (y - sy)).float()))
    b = bevel(x, y)
    c = c * (1 + b)
    return to_rgba(c * 0.9, torch.ones_like(x))


def plate_dark(S=128, frames=12):
    """A navy plate with a fine circuit texture (many thin traces, like a
    printed board seen from afar) whose light runs slowly along them."""
    out = []
    x, y = grid(S, S)
    n = noise2(x * 8, y * 8, 5.0, 4)
    base = srgb("#121829")
    cells = 9                              # a grid of cells, each with a trace or not
    cx, cy = x * cells, y * cells
    ix, iy = torch.floor(cx), torch.floor(cy)
    fx, fy = cx - ix, cy - iy
    h = vnoise(torch.stack([ix * 1.7 + 0.3, iy * 1.3 + 0.7, torch.full_like(ix, 2.0)], -1).reshape(-1, 3)).reshape(x.shape)
    hor = (h < 0.35).float() * ((fy - 0.5).abs() < 0.06).float()
    ver = ((h > 0.55) & (h < 0.85)).float() * ((fx - 0.5).abs() < 0.06).float()
    via = (h > 0.92).float() * (((fx - 0.5) ** 2 + (fy - 0.5) ** 2) < 0.04).float()
    traces = torch.maximum(torch.maximum(hor, ver), via)
    edge = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))
    traces = traces * (edge > 0.07).float()
    for f in range(frames):
        ph = f / frames
        c = base[:, None, None] * (0.85 + 0.3 * n)
        pulse = 0.25 + 0.75 * (0.5 + 0.5 * torch.sin(TAU * ((x + y) * 1.0 - ph)))
        glowmap = traces * pulse
        halo = gauss_blur(glowmap[None], 1.5)[0]
        c = c + srgb("#2cb8e6")[:, None, None] * (glowmap * 0.55 + halo * 0.35)
        c = c * (1 + bevel(x, y) * 0.8)
        out.append(to_rgba(c, torch.ones_like(x)))
    return out


def vortex(S=128, frames=16, opening=False):
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v) + 1e-4
    th = torch.atan2(v, u)
    n_frames = 10 if opening else frames
    for f in range(n_frames):
        ph = f / frames
        k = 1.0
        if opening:
            k = min(1.0, (f + 1) / n_frames)
            k = 1 - (1 - k) ** 3
        R = 0.78 * k
        swirl = th + 3.5 / (r + 0.25) - ph * TAU
        arms = (0.5 + 0.5 * torch.cos(3 * swirl)) ** 3
        deep = torch.exp(-r * 2.5)
        c = srgb("#020309")[:, None, None] * torch.ones_like(x)
        c = c + srgb("#2a3cff")[:, None, None] * arms * (r / R).clamp(0, 1) * 0.6
        c = c + srgb("#b04cff")[:, None, None] * arms * deep * 0.4
        # stars falling in
        st = (noise2(swirl * 2, r * 6, 3.0, 2) > 0.78).float() * (r < R).float()
        c = c + srgb("#ffffff")[:, None, None] * st * 0.8
        # the energy rim
        rim = torch.exp(-((r - R) / 0.05) ** 2)
        c = c + srgb("#6fe8ff")[:, None, None] * rim * (1.5 + (4.0 * (1 - k) if opening else 0.4 * math.sin(TAU * ph * 2)))
        inside = (r < R).float()
        # the deck plate around the hole, cut (a dark beveled lip)
        lip = torch.exp(-((r - R - 0.06) / 0.04) ** 2) * (r > R).float()
        alpha = (inside + rim * 0.9 + lip).clamp(0, 1)
        c = c * inside + (srgb("#6fe8ff")[:, None, None] * rim * 2.0 + srgb("#0b0f1a")[:, None, None] * lip) * (1 - inside)
        if opening:   # a flash at the start
            fl = math.exp(-f * 0.8)
            c = c + srgb("#c9f6ff")[:, None, None] * fl * torch.exp(-r * 1.5) * 3
            alpha = (alpha + fl * torch.exp(-r * 2)).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def blackhole(S=160, frames=24, opening=False):
    """A black hole seen almost edge-on, after the classic ray-traced pictures:
    the shadow, a thin photon ring, the accretion disk crossing in front of
    the shadow, the far side of the disk bent by the lensing into a halo over
    the top and the bottom, the approaching side brighter (Doppler), the gas
    turning; behind it, the dark space of the hole with stars."""
    out = []
    x, y = grid(S, S)
    u, v = (x - 0.5) * 2, (y - 0.5) * 2
    r = torch.sqrt(u * u + v * v) + 1e-5
    n_frames = 12 if opening else frames
    starsmap = stars(x, y, 90, 31, 0.32, S, 0.55).sum(0).clamp(0, 1)
    for f in range(n_frames):
        ph = f / frames
        k = 1.0
        if opening:
            k = 1 - (1 - min(1.0, (f + 1) / n_frames)) ** 3
        Rs = 0.27 * k                                  # the shadow's radius
        # the hole: deep space, stars, a soft dark lip at the square's edge
        space_c = srgb("#04050c")[:, None, None] * torch.ones_like(x)[None] + srgb("#cfd8ff")[:, None, None] * (starsmap * 0.6)[None]
        hole = (r < 0.93).float()
        # the disk: a thin ellipse (inclination), radii 1.25 .. 3.4 shadow radii
        flat_ = 0.1
        rd = torch.sqrt(u * u + (v / flat_) ** 2) + 1e-5
        phi = torch.atan2(v / flat_, u)
        turn = phi / TAU * 10.0 - ph * 10.0 / (0.25 + rd)            # inner gas turns faster
        swirl = 0.6 * noise2(rd * 14.0, turn, 21.0, 3) + 0.4 * (0.5 + 0.5 * torch.sin(rd * 60 + 3 * noise2(rd * 4, turn * 0.5, 2.0, 2)))
        ring = ((rd > Rs * 1.25) & (rd < Rs * 3.4)).float()
        heat = (1 - ((rd - Rs * 1.25) / (Rs * 2.15)).clamp(0, 1)) ** 1.5
        doppler = (1.0 + 0.95 * (-u / rd)).clamp_min(0.1)
        disk = ring * heat * (0.55 + 0.75 * swirl) * doppler
        front = (v > -0.002).float() + (r > Rs).float() * (v <= 0).float()      # the back half hides behind the shadow
        disk = disk * front.clamp(0, 1)
        # the lensed far side: arcs over and under the shadow
        halo = torch.exp(-((r - Rs * 1.3) / (Rs * 0.15)) ** 2) * (0.25 + 0.75 * (v.abs() / r) ** 1.5)
        halo = halo * (0.55 + 0.6 * noise2(r * 20, torch.atan2(v, u) * 3 - ph * 6, 5.0, 2)) * (1.0 + 0.6 * (-u / r)).clamp_min(0.2)
        photon = torch.exp(-((r - Rs * 1.06) / (Rs * 0.035)) ** 2)
        light = disk * 2.2 + halo * 1.8 + photon * 2.5
        hot = srgb("#fff4dc")[:, None, None] * ((light - 0.8).clamp(0, 2) ** 1.3)[None] * 1.2
        warm = srgb("#ff8a2a")[:, None, None] * (light.clamp(0, 1.2))[None] * 0.8 + srgb("#c8301a")[:, None, None] * (light.clamp(0, 0.5))[None] * 0.5
        c = space_c * hole[None] + hot + warm
        c = c * (r > Rs).float()[None] + (srgb("#000000")[:, None, None] * (r <= Rs).float()[None])
        # the disk passes in front of the shadow
        c = c + (srgb("#ffd8a0")[:, None, None] * (disk * (r <= Rs).float() * 2.0)[None])
        lip = torch.exp(-((r - 0.93) / 0.035) ** 2)
        alpha = (hole + lip * 0.8 + light.clamp(0, 1) * (r >= 0.93).float()).clamp(0, 1)
        c = c * (1 - 0.6 * lip)[None]
        if opening:   # it is born from a flash
            fl = math.exp(-f * 0.5)
            c = c + srgb("#e8f0ff")[:, None, None] * (fl * torch.exp(-r * 2.5) * 4)[None]
            alpha = (alpha * k + fl * torch.exp(-r * 2)).clamp(0, 1)
        out.append(to_rgba(c, alpha))
    return out


def frame_art(S=96, frames=8):
    """A 9-grid frame: corners keep their shape, the edges stretch."""
    out = []
    x, y = grid(S, S)
    e = torch.minimum(torch.minimum(x, y), torch.minimum(1 - x, 1 - y))     # distance to the border
    third = 1 / 3
    band = (e < third).float()
    brushed = noise2(x * 40, y * 40, 7.0, 2)
    base = srgb("#3a4254")
    for f in range(frames):
        c = base[:, None, None] * (0.75 + 0.35 * brushed)
        # ridges along the frame
        ridge = torch.exp(-((e - 0.08) / 0.02) ** 2) * 0.5 - torch.exp(-((e - 0.27) / 0.012) ** 2) * 0.5
        c = c * (1 + ridge)
        # an inner cyan line (the board's edge)
        c = c + srgb("#38d6ff")[:, None, None] * torch.exp(-((e - 0.31) / 0.008) ** 2) * 1.2
        # corner lights, blinking in turn
        for i, (cx, cy) in enumerate(((0.16, 0.16), (0.84, 0.16), (0.84, 0.84), (0.16, 0.84))):
            on = 1.0 if (f // 2) % 4 == i else 0.25
            d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            col = srgb("#ff5a4a") if i % 2 else srgb("#4affb0")
            c = c + col[:, None, None] * (torch.exp(-(d / 0.035) ** 2) * 2.5 * on + torch.exp(-(d / 0.08) ** 2) * 0.6 * on)
        c = c * (1 + bevel(x, y, 0.04) * 0.6)
        alpha = band
        out.append(to_rgba(c, alpha))
    return out


def theme():
    t = {
        "id": "espace", "name": "Espace", "pattern": "flat", "pattern strength": 0.2, "seed": 7,
        "light": "#8d95a4", "dark": "#121829",
        "frame top": "#3a4254", "frame bottom": "#1c212c", "frame line": "#38d6ff",
        "coordinates": "#bfe9ff", "hole": "#020309", "hole rim": "#2a3cff",
        "background top": "#05060f", "background bottom": "#0b0920",
        "last move": "#6fe8ff55", "selection": "#6fe8ff8c", "legal move": "#38d6ffaa", "hint": "#a6ff8cdc",
        "accent": "#6fe3ff",
        "tile light": "espace_clair", "tile dark": "espace_sombre", "hole art": "espace_trou", "frame art": "espace_cadre",
        "background image": "espace.jpg", "planet image": "planete.png", "planet size": 3.6, "planet top": 0.38,
        "stars": True,
    }
    with open(os.path.join(ASSETS, "themes", "7_espace.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["background", "planet", "tiles", "hole", "frame", "theme"]
    if "background" in what:
        background()
    if "planet" in what:
        planet()
    if "tiles" in what:
        write_art("espace_clair", "Espace : dalle claire", "tile", [("idle", [plate_light()])], 128)
        write_art("espace_sombre", "Espace : dalle sombre", "tile", [("idle", plate_dark())], 128, fps={"idle": 6})
    if "hole" in what:
        write_art("espace_trou", "Espace : trou noir", "hole", [("idle", blackhole()), ("open", blackhole(opening=True))], 160,
                  fps={"idle": 12, "open": 24}, loops={"idle": True, "open": False})
    if "frame" in what:
        write_art("espace_cadre", "Espace : cadre", "frame", [("idle", frame_art())], 96, fps={"idle": 4})
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
