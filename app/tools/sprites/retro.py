"""
retro.py — the home computers and game consoles of the museum's first room
(cpu_scene.py), at their real size (metres), their forms and colours as they
were — no maker's name, logo or trademark stripe, as for every figurine:

  computers  the Apple II with its two disk drives and green-text monitor,
             the C64 "breadbox" with its disk drive and blue-screen monitor,
             the ZX Spectrum (small, black, rubber keys), the Atari 520 ST
             (grey wedge, slanted function keys, external drive, the green
             desktop on its colour monitor), the Amiga 500 (its drive in the
             side, the blue and white desktop), the compact Macintosh (the
             black and white screen, its keyboard and mouse), the IBM PC
             (the big box, two drive bays, green text)
  consoles   the Atari 2600 (wood-grain front, ribbed black top, joystick),
             the NES (two greys, front-loading flap, controller), the Mega
             Drive (black, the round top), the PlayStation (grey, the round
             lid, controller with handles), the Game Boy (upright, green screen),
             TV sets showing pixel pictures

Every function adds its machine to a Scene at the transform X (the machine's
foot at X's origin, its front towards X's +z).
"""
import math
import torch
from render import Mat, T, RX, RY, RZ, srgb, sd_box, sd_cylinder, sd_sphere, sd_capsule, sd_torus, fbm, vec
from computers import sd_key_grid, plastic
from heroic import chain, bezier


def emissive_screen(pic, k=1.5):
    """A screen: pic(u, v) → RGB (both 0..1 over the glass, u right, v down)."""
    def pat(pl, pw, n):
        return {"albedo": torch.zeros_like(pl), "emissive": pic(pl) * k}
    return Mat("#000000", rough=0.08, clearcoat=1.0, pattern=pat)


def uv(pl, w, h):
    """Local x, z of a box face of half sizes w (x), h (z) → u, v in 0..1."""
    return (pl[:, 0] / w * 0.5 + 0.5).clamp(0, 1), (pl[:, 2] / h * 0.5 + 0.5).clamp(0, 1)


def col(h, m):
    return srgb(h) * m.float()[:, None]


# ─── screen pictures ────────────────────────────────────────────────────────
def text_lines(u, v, rows=12, cols=30, fill=0.6, seed=1.0, nrows=None):
    r = torch.floor(v * rows)
    c = torch.floor(u * cols)
    on = (torch.sin(c * 12.9898 + r * 78.233 + seed).abs() * 43758.5 % 1 < fill)
    inrow = (v * rows % 1 > 0.2) & (v * rows % 1 < 0.8) & (u * cols % 1 > 0.1) & (u * cols % 1 < 0.9)
    length = (torch.sin(r * 3.7 + seed) * 0.5 + 0.5) * cols
    lines = r < (nrows if nrows is not None else rows)
    return on & inrow & (c < length) & lines


def pic_green_text(w, h, seed=1.0):
    def f(pl):
        u, v = uv(pl, w, h)
        t = text_lines(u * 1.1 - 0.05, v * 1.1 - 0.05, 16, 38, 0.7, seed, 11)
        return col("#3aff6a", t) + srgb("#04140a") * 1.0
    return f


def pic_c64(w, h):
    def f(pl):
        u, v = uv(pl, w, h)
        inner = (u > 0.1) & (u < 0.9) & (v > 0.12) & (v < 0.88)
        t = text_lines((u - 0.1) / 0.8, (v - 0.12) / 0.76, 18, 40, 0.75, 3.0, 5) & inner
        return col("#6c5eb5", ~inner) + col("#352879", inner & ~t) + col("#6c5eb5", t)
    return f


def pic_st_desktop(w, h):
    def f(pl):
        u, v = uv(pl, w, h)
        bar = v < 0.07
        win = (u > 0.35) & (u < 0.9) & (v > 0.2) & (v < 0.75)
        title = win & (v < 0.26)
        icons = ((u > 0.05) & (u < 0.13) & (((v > 0.2) & (v < 0.3)) | ((v > 0.4) & (v < 0.5))))
        g = ~(bar | win | icons)
        return col("#2a9a3a", g) + col("#f2f2f2", bar | (win & ~title) | icons) + col("#222222", title)
    return f


def pic_amiga(w, h):
    def f(pl):
        u, v = uv(pl, w, h)
        bar = v < 0.08
        win = (u > 0.08) & (u < 0.6) & (v > 0.18) & (v < 0.6)
        border = win & ((u < 0.1) | (u > 0.58) | (v < 0.24) | (v > 0.58))
        drawer = (u > 0.2) & (u < 0.28) & (v > 0.32) & (v < 0.42)
        return col("#2a5ab0", ~(bar | win)) + col("#f2f2f2", bar | border) + col("#2a5ab0", win & ~border & ~drawer) + col("#f2a030", drawer)
    return f


def pic_mac(w, h):
    def f(pl):
        u, v = uv(pl, w, h)
        bar = v < 0.07
        dots = ((torch.floor(u * 160) + torch.floor(v * 120)) % 2 == 0)       # the grey dither of the desktop
        win = (u > 0.12) & (u < 0.75) & (v > 0.18) & (v < 0.7)
        title = win & (v < 0.24) & ((torch.floor(v * 120) % 2) == 0)
        icon = (u > 0.82) & (u < 0.92) & (v > 0.14) & (v < 0.24)
        white = bar | (win & ~title) | (~win & ~bar & dots) | icon
        return col("#e8ecec", white) + srgb("#080808") * (~white).float()[:, None]
    return f


def pic_game(w, h, seed=0.0):
    """A pixel picture of a game: sky, ground, blocks, a little figure (no
    existing game: shapes only)."""
    def f(pl):
        u, v = uv(pl, w, h)
        px = torch.floor(u * 32)
        py = torch.floor(v * 24)
        sky = col("#5a9aff", py < 18)
        ground = col("#a0602a", py >= 20) + col("#40c040", (py >= 18) & (py < 20))
        blocks = ((py == 11) & (px > 10 + seed) & (px < 15 + seed)) | ((py == 14) & (px > 20) & (px < 25))
        hero = (px >= 6) & (px <= 7) & (py >= 16) & (py <= 17)
        sun = ((px - 26) ** 2 + (py - 4) ** 2) < 5
        c = sky + ground
        c = torch.where(blocks[:, None], srgb("#e0a040").expand_as(c), c)
        c = torch.where(hero[:, None], srgb("#e03030").expand_as(c), c)
        c = torch.where(sun[:, None], srgb("#ffe060").expand_as(c), c)
        return c
    return f


def pic_gameboy(w, h):
    def f(pl):
        u, v = uv(pl, w, h)
        px, py = torch.floor(u * 20), torch.floor(v * 18)
        shades = [srgb("#9bbc0f"), srgb("#8bac0f"), srgb("#306230"), srgb("#0f380f")]
        k = ((py > 13).long() * 2 + (((px + py) % 7) == 0).long()).clamp(0, 3)
        out = shades[0].expand(u.shape[0], 3).clone()
        for i in range(1, 4):
            out = torch.where((k == i)[:, None], shades[i].expand_as(out), out)
        return out
    return f


# ─── pieces ─────────────────────────────────────────────────────────────────
def monitor(sc, X, w, h, d, case, pic, k=1.4, round_=0.02):
    """A CRT monitor: a box, a deeper back, a bezel, the glowing screen."""
    sc.add(sd_box(w, h, d, round_), case, X * T(0, h, 0))
    sc.add(sd_box(w * 0.7, h * 0.75, d * 0.8, round_), case, X * T(0, h * 0.95, -d * 1.4))
    face = X * T(0, h * 1.02, d + 0.002) * RX(math.pi / 2)
    sc.add(sd_box(w * 0.86, 0.004, h * 0.8, 0.01), Mat("#1a1a1a", rough=0.4), face)
    sw, sh = w * 0.76, h * 0.68
    sc.add(sd_box(sw, 0.006, sh, 0.012), emissive_screen(pic(sw, sh), k), face * T(0, 0.004, 0))


def keyboard_wedge(sc, X, w, d, h_back, h_front, case, keys, nx=14, nz=5, pitch=0.019, inset=(0.0, 0.0)):
    """A computer in its keyboard: a wedge (sloped top), a block of keys on it.
    pitch: 19 mm between keys, as on every real keyboard (the first versions had
    22-26 mm and too few keys: the keyboards looked oversized)."""
    slope = math.atan2(h_back - h_front, 2 * d)
    sc.add(sd_box(w, (h_back + h_front) / 4, d, 0.006), case, X * T(0, (h_back + h_front) / 4, 0))
    top = X * T(inset[0], (h_back + h_front) / 2 + 0.004, inset[1]) * RX(slope)
    sc.add(sd_key_grid(nx, nz, pitch, pitch, pitch * 0.42, 0.006, pitch * 0.42, 0.002), keys, top)
    return top


def tv(sc, X, w, h, pic, k=1.4):
    """A TV set of the 1980s: wood-grain cabinet, the screen, two knobs."""
    wood = Mat("#5a3a22", rough=0.45, clearcoat=0.4,
               pattern=lambda pl, pw, n: {"albedo": srgb("#5a3a22") * (0.9 + 0.1 * torch.sin(pl[:, 1] * 300 + 8 * fbm(pl * 12, 2)))[:, None]})
    sc.add(sd_box(w, h, w * 0.8, 0.02), wood, X * T(0, h, 0))
    face = X * T(-w * 0.12, h, w * 0.8 + 0.002) * RX(math.pi / 2)
    sw, sh = w * 0.7, h * 0.72
    sc.add(sd_box(sw + 0.012, 0.004, sh + 0.012, 0.02), Mat("#1a1a1a", rough=0.4), face)
    sc.add(sd_box(sw, 0.006, sh, 0.03), emissive_screen(pic(sw, sh), k), face * T(0, 0.004, 0))
    for i in range(2):
        sc.add(sd_cylinder(0.006, 0.014, 0.003), Mat("#c8c8c8", metal=0.6, rough=0.3), X * T(w * 0.83, h * (1.35 - 0.5 * i), w * 0.8 + 0.006) * RX(math.pi / 2))


# ─── the computers ──────────────────────────────────────────────────────────
BEIGE = plastic("#d8ceb4")
BEIGE_DARK = plastic("#b8ad92")


# ─── the Apple II (replica: 380 × 450 mm, 110 mm high at the lid) ───────────
A2_CASE, A2_KEY = "#e6dabf", "#6c6354"


def _apple_badge(text, plate="#7a7468", txt="#ece8e0"):
    """A brushed plate with the rainbow apple and a name in the Apple Garamond style."""
    from replica import rainbow_apple, font
    def draw(d, W, H, img):
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H // 8, fill=plate)
        rainbow_apple(d, H * 0.55, H * 0.52, H * 0.8)
        d.text((H * 1.0, H * 0.02), text, font=font("Comfortaa-Bold.ttf", int(H * 0.72)), fill=txt)
    return draw


A2_ROWS = [
    (0, 1.0, [(1, A2_KEY, c) for c in "1234567890"] + [(1, A2_KEY, ":"), (1, A2_KEY, "-"), (1, A2_KEY, "RESET")]),
    (1, 0.5, [(1, A2_KEY, "ESC")] + [(1, A2_KEY, c) for c in "QWERTYUIOP"] + [(1, A2_KEY, "@"), (1, A2_KEY, "REPT"), (1.5, A2_KEY, "RETURN")]),
    (2, 0.25, [(1.25, A2_KEY, "CTRL")] + [(1, A2_KEY, c) for c in "ASDFGHJKL"] + [(1, A2_KEY, ";"), (1, A2_KEY, "←"), (1, A2_KEY, "→")]),
    (3, 0.0, [(1.5, A2_KEY, "SHIFT")] + [(1, A2_KEY, c) for c in "ZXCVBNM,./"] + [(1.75, A2_KEY, "SHIFT")]),
    (4, 0.0, [(1, "#e8e4da", "POWER|#555555"), (2.5, None), (8.5, A2_KEY, "")]),
]


def _disk2(sc, X, n):
    """A Disk II drive: the beige box, its black front with the slot, the door, the
    IN USE lamp, disk II and DRIVE n."""
    from replica import decal, font, block
    sc.add(block(0.1, 0.045, 0.12, 0.006), plastic(A2_CASE, 0.5), X)
    def front(d, W, H, img):
        d.rectangle([0, 0, W, H], fill="#1c1c1c")
        d.rectangle([W * 0.04, H * 0.38, W * 0.96, H * 0.44], fill="#050505")              # the slot
        d.rectangle([W * 0.36, H * 0.12, W * 0.6, H * 0.9], fill="#101010", outline="#2e2e2e", width=3)   # the door
        f = font("Lato-Bold.ttf", int(H * 0.08))
        d.text((W * 0.05, H * 0.62), "IN USE", font=f, fill="#d8d8d8")
        d.polygon([(W * 0.17, H * 0.625), (W * 0.17, H * 0.675), (W * 0.195, H * 0.65)], fill="#d8d8d8")
        d.ellipse([W * 0.22, H * 0.6, W * 0.26, H * 0.68], fill="#5a0a0a")
        d.text((W * 0.05, H * 0.74), "disk", font=font("Lato-Black.ttf", int(H * 0.16)), fill="#e8e8e8")
        d.text((W * 0.2, H * 0.72), "II", font=font("DejaVuSansMono.ttf", int(H * 0.18)), fill="#e8e8e8")
        d.rounded_rectangle([W * 0.72, H * 0.58, W * 0.95, H * 0.9], radius=8, fill="#0a0a0a")
        d.text((W * 0.75, H * 0.68), f"DRIVE {n}", font=f, fill="#d8d8d8")
    sc.add(sd_box(0.092, 0.04, 0.0008, 0.0006), decal(0.092, 0.04, front, base="#1c1c1c", px=1024, axes=(0, 1), flip_v=True), X * T(0, 0.045, 0.1205))


def apple2(sc, X):
    """The Apple II (1977) as in the photographs: the cream case, the brown keyboard
    in its well on the sloping front, the apple ][ plate on the lid, the two Disk II
    drives on top and the Monitor II on them (green text)."""
    from replica import keyboard, decal, block
    case = plastic(A2_CASE, 0.5)
    # the case: the lid flat at 110 mm over the back, the front sloping down to 60 mm
    sc.add(block(0.19, 0.055, 0.225, 0.008, planes=[(0, 1, 0.42, 0.11 + 0.42 * 0.06)], soft=0.006), case, X)
    # the joint line between the lid and the base, and the vents along the sides
    for sx in (-1, 1):
        for i in range(14):
            sc.add(sd_box(0.0012, 0.022, 0.0016, 0.0008), Mat("#a89e86"), X * T(sx * 0.1905, 0.07, -0.2 + i * 0.011))
    # the apple ][ plate on the lid
    sc.add(sd_box(0.06, 0.0006, 0.013, 0.0005), decal(0.06, 0.013, _apple_badge("apple ][", "#8a8478"), base="#8a8478", px=1024), X * T(0, 0.1112, 0.03))
    # the keyboard in its well on the slope
    a = math.atan(0.42)
    zc = 0.16
    ys = lambda z: 0.11 - 0.42 * (z - 0.06)
    sc.add(sd_box(0.165, 0.004, 0.07, 0.004), Mat("#d8ccb0", rough=0.5), X * T(0, ys(zc) - 0.002, zc) * RX(a))      # the well's rim
    sdf, mat, _ = keyboard(A2_ROWS, cap=0.0025, height=0.009, gap=0.0016, round_=0.002, px_per_unit=48, legend_col="#f0ece4")
    sc.add(sdf, mat, X * T(0.005, ys(zc) - 0.001, zc) * RX(a))
    # the two Disk II drives on the lid, the monitor on them
    _disk2(sc, X * T(-0.1, 0.11, -0.13), 2)
    _disk2(sc, X * T(0.1, 0.11, -0.13), 1)
    M = X * T(0, 0.2, -0.175)
    sc.add(block(0.2, 0.15, 0.16, 0.012), plastic("#e2d6bb", 0.5), M)
    face = M * T(-0.025, 0.155, 0.161)
    sc.add(sd_box(0.15, 0.12, 0.004, 0.012), plastic("#d4c8ac", 0.5), face)                                        # the bezel
    sc.add(sd_box(0.125, 0.098, 0.006, 0.02), emissive_screen(lambda pl: pic_green_text(0.125, 0.098, 2.0)(torch.stack([pl[:, 0], pl[:, 2], -pl[:, 1]], -1)), 1.3),
           face * T(0, 0, 0.002))
    sc.add(sd_box(0.012, 0.012, 0.0008, 0.002), decal(0.012, 0.012, lambda d, W, H, img: __import__("replica").rainbow_apple(d, W / 2, H / 2, H * 0.8),
           base="#e2d6bb", px=256, axes=(0, 1), flip_v=True), M * T(0.17, 0.05, 0.1605))
    for i in range(2):
        sc.add(sd_box(0.011, 0.011, 0.002, 0.002), plastic("#d8ccb0"), M * T(0.17, 0.22 - i * 0.1, 0.161))


# ─── the Commodore 64 (replica: 404 × 216 mm, 32 mm at the front, 75 mm at the back) ─
C64_CASE = "#bcb5a5"
C64_KEY, C64_FKEY = "#3b322c", "#c9b48c"


def _c64_body():
    """The "breadbox": a rounded block whose top slopes from the back strip
    (75 mm high) down to the front (45 mm), the front edge one big round nose."""
    k = 0.18                                                       # the slope of the keyboard area
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        # rounded box 404 × 75 × 216 mm, its foot at y = 0
        q = torch.stack([x.abs() - 0.202 + 0.02, (y - 0.0375).abs() - 0.0375 + 0.02, z.abs() - 0.108 + 0.02], dim=-1)
        box = torch.linalg.norm(q.clamp_min(0), dim=-1) + q.max(dim=-1).values.clamp_max(0) - 0.02
        top = (y - (0.075 - k * (z + 0.06).clamp_min(0))) / math.sqrt(1 + k * k)
        h = (0.5 + 0.5 * (top - box) / 0.012).clamp(0, 1)          # smooth max: the nose rounds into the slope
        return top * h + box * (1 - h) + 0.012 * h * (1 - h)
    return f


C64_ROWS = [
    (0, 0.0, [(1, C64_KEY, "←"), (1, C64_KEY, "1"), (1, C64_KEY, "2"), (1, C64_KEY, "3"), (1, C64_KEY, "4"), (1, C64_KEY, "5"),
              (1, C64_KEY, "6"), (1, C64_KEY, "7"), (1, C64_KEY, "8"), (1, C64_KEY, "9"), (1, C64_KEY, "0"), (1, C64_KEY, "+"),
              (1, C64_KEY, "−"), (1, C64_KEY, "£"), (1, C64_KEY, "CLR"), (1, C64_KEY, "INST")]),
    (1, -0.5, [(1.5, C64_KEY, "CTRL")] + [(1, C64_KEY, c) for c in "QWERTYUIOP@*"] + [(1, C64_KEY, "↑"), (1.5, C64_KEY, "RESTORE")]),
    (2, -0.75, [(1, C64_KEY, "RUN"), (1, C64_KEY, "LOCK")] + [(1, C64_KEY, c) for c in "ASDFGHJKL:;="] + [(2, C64_KEY, "RETURN")]),
    (3, -1.0, [(1, C64_KEY, "C="), (1.5, C64_KEY, "SHIFT")] + [(1, C64_KEY, c) for c in "ZXCVBNM,./"] + [(1.5, C64_KEY, "SHIFT"),
              (1, C64_KEY, "CRSR"), (1, C64_KEY, "CRSR")]),
    (4, 2.0, [(9, C64_KEY, "")]),
    # the four function keys, in their own well on the right (one row apart)
    (0, 17.4, [(1.8, C64_FKEY, "f1|#5a4a30")]), (1, 17.4, [(1.8, C64_FKEY, "f3|#5a4a30")]),
    (2, 17.4, [(1.8, C64_FKEY, "f5|#5a4a30")]), (3, 17.4, [(1.8, C64_FKEY, "f7|#5a4a30")]),
]


def _c64_strip(d, W, H, img):
    """The back strip seen from above: 5 grooves across, the badge on the left
    (commodore, the rainbow, 64), POWER and its lamp on the right."""
    from replica import font
    for i in range(5):
        y = int(H * (0.18 + i * 0.15))
        d.rectangle([0, y, W, y + max(1, H // 40)], fill="#8a8476")
    # the badge: a dark rounded plate over the grooves — C=, commodore, the rainbow, 64,
    # laid out from the measured widths of the words
    by0, by1 = int(H * 0.22), int(H * 0.8)
    bh = by1 - by0
    f = font("Lato-Bold.ttf", int(bh * 0.6))
    f64 = font("Lato-Black.ttf", int(bh * 0.8))
    wn, w64 = d.textlength("commodore", font=f), d.textlength("64", font=f64)
    bx0 = int(W * 0.04)
    x = bx0 + bh * 0.35
    cx, cy, r = x + bh * 0.22, (by0 + by1) / 2, bh * 0.24
    x_name = x + bh * 0.6
    x_rb = x_name + wn + bh * 0.3
    x_64 = x_rb + bh * 1.3
    bx1 = int(x_64 + w64 + bh * 0.4)
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=bh // 2, fill="#4a4640", outline="#9a968c", width=max(1, H // 50))
    d.arc([cx - r, cy - r, cx + r, cy + r], 40, 320, fill="#d8d6d0", width=max(1, int(r * 0.45)))
    d.text((x_name, by0 + bh * 0.1), "commodore", font=f, fill="#d8d6d0")
    for i, c in enumerate(["#d8282a", "#f07a1a", "#f2d21a", "#3aa040", "#2a5ab8"]):     # the rainbow
        y0 = by0 + bh * (0.2 + i * 0.12)
        d.rectangle([x_rb, y0, x_rb + bh * 1.1, y0 + bh * 0.12], fill=c)
    d.text((x_64, by0 - bh * 0.02), "64", font=f64, fill="#d8d6d0")
    # POWER and its lamp
    px0, px1 = int(W * 0.86), int(W * 0.95)
    by0, by1 = int(H * 0.25), int(H * 0.78)
    d.rounded_rectangle([px0, by0 + (by1 - by0) * 0.15, px1, by1 - (by1 - by0) * 0.15], radius=(by1 - by0) // 3, fill="#4a4640")
    d.text((px0 + (px1 - px0) * 0.1, by0 + (by1 - by0) * 0.25), "POWER", font=font("Lato-Bold.ttf", int((by1 - by0) * 0.35)), fill="#d8d6d0")


def c64(sc, X):
    """The Commodore 64 "breadbox" (1982) as in the photographs: the warm grey case,
    the brown keys with their white legends, the four beige function keys, the
    grooved back strip with its badge and the red power lamp; beside it the 1541
    disk drive, behind it the colour monitor with the blue screen of BASIC."""
    from replica import keyboard, decal, U
    case = plastic(C64_CASE, 0.5)
    sc.add(_c64_body(), case, X)
    k = 0.18
    a = math.atan(k)
    # the groove along the front (the joint of the two shells)
    sc.add(sd_box(0.2035, 0.0012, 0.1085, 0.02), Mat("#8a8476", rough=0.6), X * T(0, 0.016, 0))
    # the back strip, flat, with its grooves and badge
    sc.add(sd_box(0.19, 0.0006, 0.022, 0.0005), decal(0.19, 0.022, _c64_strip, base=C64_CASE, px=2048), X * T(0, 0.0752, -0.081))
    # the power lamp
    sc.add(sd_sphere(0.0035), Mat("#e01a10", emissive="#ff2a1a", emit=1.5, rough=0.2, clearcoat=1.0), X * T(0.181, 0.0755, -0.081))
    # the keyboard on the slope: the recessed panel's border, the dark wells, the keys
    slope = lambda z: 0.075 - k * (z + 0.06)
    zc = 0.025                                                     # the keyboard's centre (front-back)
    KB = X * T(0, slope(zc) - 0.002, zc) * RX(a)
    sdf, mat, (hw, hd) = keyboard(C64_ROWS, cap=0.0028, height=0.0095, gap=0.0018, round_=0.0022, px_per_unit=56)
    sc.add(sdf, mat, KB * T(0, 0.0, 0))
    # the dark wells under the keys (the main block, the function keys)
    xs0 = -1.0; xs1 = 19.2                                          # the layout's extent in key units (see C64_ROWS)
    off = (xs0 + xs1) / 2
    sc.add(sd_box(8.45 * U, 0.003, 2.5 * U, 0.002), Mat("#1a1614", rough=0.7), KB * T((7.4 - off) * U, -0.002, 0.0))
    sc.add(sd_box(1.05 * U, 0.003, 2.05 * U, 0.002), Mat("#1a1614", rough=0.7), KB * T((18.3 - off) * U, -0.002, -0.5 * U))
    # the ports on the right side: control ports 1 and 2, the switch, the power socket
    def ports(d, W, H, img):
        from replica import font
        d.rectangle([0, 0, W, H], fill="#2a2826")
        f = font("Lato-Bold.ttf", int(H * 0.12))
        for i, lab in enumerate(("CONTROL\nPORT 1", "CONTROL\nPORT 2")):
            x0 = W * (0.03 + i * 0.32)
            d.rectangle([x0, H * 0.35, x0 + W * 0.28, H * 0.95], fill="#b8b8b4")
            d.rounded_rectangle([x0 + W * 0.07, H * 0.5, x0 + W * 0.21, H * 0.85], radius=int(H * 0.08), fill="#3a3a3a")
            d.multiline_text((x0 + W * 0.06, H * 0.05), lab, font=f, fill="#e8e8e8", align="center", spacing=0)
        d.rectangle([W * 0.7, H * 0.45, W * 0.76, H * 0.95], fill="#111111")
        d.text((W * 0.69, H * 0.12), "ON", font=f, fill="#e8e8e8")
        d.ellipse([W * 0.84, H * 0.55, W * 0.92, H * 0.85], fill="#111111")
        d.text((W * 0.8, H * 0.05), "POWER", font=f, fill="#e8e8e8")
    sc.add(sd_box(0.0006, 0.013, 0.06, 0.001), decal(0.06, 0.013, ports, base="#2a2826", px=1024, axes=(2, 1), flip_v=True, flip_u=True),
           X * T(0.2025, 0.03, 0.02))
    # the 1541 disk drive beside it and the monitor behind
    drive = plastic("#c8bfae", 0.5)
    sc.add(sd_box(0.1, 0.038, 0.18, 0.01), drive, X * T(0.32, 0.038, -0.05))
    sc.add(sd_box(0.07, 0.008, 0.002, 0.002), Mat("#3a302a", rough=0.5), X * T(0.31, 0.05, 0.131))
    sc.add(sd_box(0.012, 0.004, 0.002, 0.001), Mat("#3a302a"), X * T(0.31, 0.05, 0.133))
    sc.add(sd_sphere(0.003), Mat("#20c020", emissive="#3aff3a", emit=1.0), X * T(0.25, 0.03, 0.131))
    monitor(sc, X * T(0, 0.0, -0.34), 0.18, 0.15, 0.15, plastic("#c4baa8"), pic_c64)


# ─── the Sinclair ZX Spectrum (replica: 233 × 144 × 30 mm, the rubber keys) ─
ZX_KEY = "#56656e"
Z_ = ZX_KEY
ZX_ROWS = [
    (0, 0.0, [(1, Z_, c) for c in "1234567890"]),
    (1, 0.5, [(1, Z_, c) for c in "QWERTYUIOP"]),
    (2, 0.75, [(1, Z_, c) for c in "ASDFGHJKL"] + [(1.25, Z_, "ENTER")]),
    (3, 0.0, [(1.5, Z_, "CAPS")] + [(1, Z_, c) for c in "ZXCVBNM"] + [(1, Z_, "SYM|#e04030"), (1.5, Z_, "SPACE")]),
]


def _zx_top(d, W, H, img):
    """The back strip: the big sinclair in relief (shown as its lighter outline), ZX Spectrum in white."""
    from replica import font
    f = font("Lato-Black.ttf", int(H * 0.62))
    d.text((W * 0.04, H * 0.02), "sinclair", font=f, fill="#1e1c1c", stroke_width=max(2, int(H * 0.02)), stroke_fill="#3e3a3a")
    d.text((W * 0.05, H * 0.66), "ZX Spectrum", font=font("Lato-Black.ttf", int(H * 0.2)), fill="#f2f2f2")


def _zx_panel(d, W, H, img):
    """The key panel's print: the colour names above the top row, and the rainbow
    running down to the bottom right corner."""
    from replica import font
    f = font("Lato-Bold.ttf", int(H * 0.045))
    names = [("BLUE", "#3a8ae8"), ("RED", "#e04030"), ("MAGENTA", "#d040c0"), ("GREEN", "#40c040"), ("CYAN", "#40c8e0"),
             ("YELLOW", "#f0d020"), ("WHITE", "#f0f0f0"), ("", ""), ("GRAPHICS", "#f0f0f0"), ("DELETE", "#f0f0f0")]
    for i, (n, c) in enumerate(names):
        if n:
            d.text((W * (0.08 + i * 0.087), H * 0.04), n, font=f, fill=c)
    for j, c in enumerate(["#e03028", "#f0c820", "#30b040", "#2a8ae0"]):          # the rainbow
        o = j * W * 0.022
        d.polygon([(W * 0.93 + o, H * 0.05), (W * 0.95 + o, H * 0.05), (W * 0.83 + o, H * 1.0), (W * 0.81 + o, H * 1.0)], fill=c)


def zx(sc, X):
    """The Sinclair ZX Spectrum (1982) as in the photographs: the small black case,
    the raised back strip with sinclair in relief and ZX Spectrum, the grey-blue
    rubber keys, the colour names and the rainbow; the TV behind it."""
    from replica import keyboard, decal, block
    black = Mat("#1e1c1c", rough=0.45, clearcoat=0.3)
    sc.add(block(0.1165, 0.012, 0.072, 0.004, planes=[(0, 1, 0.25, 0.024 + 0.25 * 0.06)], soft=0.003), black, X)
    sc.add(block(0.1165, 0.016, 0.022, 0.004), black, X * T(0, 0, -0.05))                                     # the back strip
    sc.add(sd_box(0.11, 0.0005, 0.02, 0.0004), decal(0.11, 0.02, _zx_top, base="#1e1c1c", px=1024), X * T(0, 0.0321, -0.05))
    sc.add(sd_box(0.112, 0.0005, 0.046, 0.0004), decal(0.112, 0.046, _zx_panel, base="#1e1c1c", px=1024), X * T(0, 0.0242, 0.022))
    sdf, mat, _ = keyboard(ZX_ROWS, cap=0.0015, height=0.005, gap=0.004, round_=0.0025, px_per_unit=48, legend_col="#f2f2f2")
    sc.add(sdf, mat, X * T(0.0, 0.0243, 0.024))
    tv(sc, X * T(0.02, 0.0, -0.3), 0.17, 0.12, lambda w, h: (lambda pl: col("#e8e8e8", torch.ones_like(pl[:, 0], dtype=torch.bool)) - col("#e8e8e8", text_lines(*uv(pl, w, h), 14, 32, 0.6, 7.0, 3)) * 0.9))


# ─── the Atari 520 ST (replica: 470 × 240 mm, the French AZERTY keyboard) ───
ST_CASE, ST_KEY, ST_FKEY = "#d9dad6", "#e9e5d8", "#b9b8b1"


def _st_body():
    """The base: a slab 470 × 240 mm, 30 mm at the front rising to 45 mm under
    the back block; its edges rounded small (the ST is crisp)."""
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        q = torch.stack([x.abs() - 0.235 + 0.006, (y - 0.03).abs() - 0.03 + 0.006, z.abs() - 0.12 + 0.006], dim=-1)
        box = torch.linalg.norm(q.clamp_min(0), dim=-1) + q.max(dim=-1).values.clamp_max(0) - 0.006
        top = (y - (0.03 + 0.06 * (0.12 - z) / 0.24 * 0.25)) / 1.0       # a gentle slope: 30 mm front, 45 mm back
        return torch.maximum(box, top)
    return f


def _st_block():
    """The back block (the cover over the power supply and the drive): 470 × 62 mm,
    its top sloping from 68 mm at the back to 58 mm, its front face slanted back."""
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        zc = -0.089
        q = torch.stack([x.abs() - 0.235 + 0.005, (y - 0.034).abs() - 0.034 + 0.005, (z - zc).abs() - 0.031 + 0.005], dim=-1)
        box = torch.linalg.norm(q.clamp_min(0), dim=-1) + q.max(dim=-1).values.clamp_max(0) - 0.005
        top = y - (0.068 - (z + 0.12) * 0.16)
        face = ((z - (-0.058)) + (y - 0.035) * 0.55) / math.sqrt(1 + 0.3)  # the front face leans back
        return torch.maximum(torch.maximum(box, top), face)
    return f


def _st_top(d, W, H, img):
    """The block's top: the grooves of the vents at 60°, over its left two thirds."""
    for i in range(-40, 160):
        x0 = i * W * 0.012
        d.line([(x0, H * 0.12), (x0 + H * 0.5, H * 0.88)], fill="#b4b5b0", width=max(2, int(W * 0.004)))
    d.rectangle([0, 0, W, H * 0.1], fill=ST_CASE); d.rectangle([0, H * 0.9, W, H], fill=ST_CASE)


K_, F_ = ST_KEY, ST_FKEY
ST_ROWS = [
    # the ten function keys, slanted parallelograms, in a row before the block
    (-1.3, 0.5, [(1.5, F_, "", 0.8, 0.55)] * 10),
    (0, 0, [(1, K_, "Esc|#555555")] + [(1, K_, c + "|#555555") for c in "1234567890)-#"] + [(2, K_, "Bksp|#555555")]
          + [None, (1.5, K_, "Help|#555555"), (1.5, K_, "Undo|#555555"), (0.5, None)] + [(1, K_, c + "|#555555") for c in "()/*"]),
    (1, 0, [(1.5, K_, "Tab|#555555")] + [(1, K_, c + "|#555555") for c in "AZERTYUIOP^$"] + [(1, None), (1.5, K_, "Del|#555555"),
           (0.5, None), (1, K_, "Ins|#555555"), (1, K_, "↑|#555555"), (1, K_, "Clr|#555555"), (0.5, None)] + [(1, K_, c + "|#555555") for c in "789-"]),
    (2, 0, [(1.75, K_, "Control|#555555")] + [(1, K_, c + "|#555555") for c in "QSDFGHJKLMù"] + [(2.25, K_, "Return|#555555"), (1, K_, "`|#555555"),
           (0.5, None), (1, K_, "←|#555555"), (1, K_, "↓|#555555"), (1, K_, "→|#555555"), (0.5, None)] + [(1, K_, c + "|#555555") for c in "456+"]),
    (3, 0, [(1.25, K_, "Shift|#555555"), (1, K_, "<|#555555")] + [(1, K_, c + "|#555555") for c in "WXCVBN,;:="] + [(2.75, K_, "Shift|#555555")]
           + [(5.0, None)] + [(1, K_, c + "|#555555") for c in "123"] + [(1, K_, "Enter|#555555", 2.0)]),
    (4, 1.0, [(1.5, K_, "Alt|#555555"), (9, K_, ""), (1.5, K_, "Caps|#555555")] + [(6.0, None), (2, K_, "0|#555555"), (1, K_, ".|#555555")]),
]
ST_ROWS = [(z, x0, [k if k is not None else (0.5, None) for k in ks]) for (z, x0, ks) in ST_ROWS]


def atari_st(sc, X):
    """The Atari 520 ST (1985): the light grey slab with its cream keys (the French
    AZERTY), the ten slanted grey function keys, the back block with its
    diagonal vents and the blue ATARI 520ST; beside it the SF354 drive and the mouse,
    behind it the SC1224 colour monitor with the green GEM desktop."""
    from replica import keyboard, decal
    case = plastic(ST_CASE, 0.5)
    sc.add(_st_body(), case, X)
    sc.add(_st_block(), case, X)
    tilt = math.atan(0.16)
    sc.add(sd_box(0.15, 0.0006, 0.026, 0.0005), decal(0.15, 0.026, _st_top, base=ST_CASE, px=2048),
           X * T(-0.07, 0.068 - 0.031 * 0.16 + 0.0002, -0.089) * RX(tilt))
    # ATARI 520ST on the slanted front face of the block, on the right
    lean = math.atan(0.55)
    sc.add(sd_box(0.07, 0.007, 0.0006, 0.0004), decal(0.07, 0.007, lambda d, W, H, img: _st_logo(d, W, H), base=ST_CASE, px=1024, axes=(0, 1), flip_v=True),
           X * T(0.15, 0.054, -0.0685 + 0.0008) * RX(-lean))
    # the keyboard on the slab (key tops 10 mm above it)
    sdf, mat, (hw, hd) = keyboard(ST_ROWS, cap=0.002, height=0.009, gap=0.0016, round_=0.0018, px_per_unit=48)
    ytop = lambda z: 0.03 + 0.06 * (0.12 - z) / 0.24 * 0.25
    zc = 0.022
    sc.add(sdf, mat, X * T(0.0, ytop(zc), zc) * RX(math.atan(0.0625)))
    sc.add(sd_box(0.012, 0.0005, 0.0015, 0.0005), Mat("#2a2a2a"), X * T(-0.205, ytop(0.1) + 0.0004, 0.1))      # the power light's window
    # the SF354 disk drive, the mouse, the monitor
    g = plastic(ST_CASE, 0.5)
    sc.add(sd_box(0.1, 0.03, 0.13, 0.006), g, X * T(0.37, 0.03, -0.02))
    sc.add(sd_box(0.05, 0.004, 0.002, 0.001), Mat("#2a2a2a"), X * T(0.36, 0.035, 0.111))
    sc.add(sd_box(0.03, 0.014, 0.05, 0.01), g, X * T(0.29, 0.014, 0.17))
    sc.add(sd_box(0.013, 0.002, 0.012, 0.002), Mat("#4a4a4c", rough=0.5), X * T(0.276, 0.028, 0.135))
    sc.add(sd_box(0.013, 0.002, 0.012, 0.002), Mat("#4a4a4c", rough=0.5), X * T(0.304, 0.028, 0.135))
    monitor(sc, X * T(0, 0.0, -0.36), 0.19, 0.16, 0.16, plastic("#d6d6d2"), pic_st_desktop)


def _st_logo(d, W, H):
    """The right part of the block's top: the Fuji and ATARI 520ST, in blue and grey."""
    from replica import font
    blue = "#1d3f9a"
    fh = H * 0.6
    fx, fy = W * 0.12, H * 0.2
    d.rectangle([fx - fh * 0.06, fy, fx + fh * 0.06, fy + fh], fill=blue)
    for sgn in (-1, 1):
        d.polygon([(fx + sgn * fh * 0.12, fy), (fx + sgn * fh * 0.22, fy), (fx + sgn * fh * 0.5, fy + fh), (fx + sgn * fh * 0.34, fy + fh)], fill=blue)
    d.text((fx + fh * 0.65, fy - fh * 0.12), "ATARI", font=font("Lato-Black.ttf", int(fh * 1.05)), fill=blue)
    d.text((W * 0.62, fy - fh * 0.12), "520ST", font=font("Lato-Black.ttf", int(fh * 1.05)), fill="#34363a")


# ─── the Amiga 500 (replica: 472 × 330 × 70 mm) ─────────────────────────────
AM_CASE, AM_KEY, AM_FKEY = "#ddd8c8", "#ebe8de", "#a8a497"
A_, B_ = AM_KEY, AM_FKEY
_d = lambda t: (1, B_, t + "|#333333")
_l = lambda t: (1, A_, t + "|#333333")
AM_ROWS = [
    (0, 0.0, [(1.25, B_, "Esc|#333333"), (0.5, None)] + [(1.25, B_, f"F{i}|#333333") for i in range(1, 6)] + [(0.5, None)]
          + [(1.25, B_, f"F{i}|#333333") for i in range(6, 11)]),
    (1.25, 0, [(1.5, B_, "`|#333333")] + [_l(c) for c in "1234567890-=\\"] + [_d("←"), (0.5, None),
           (1.5, B_, "Del|#333333"), (1.5, B_, "Help|#333333"), (0.5, None)] + [_d(c) for c in "()/*"]),
    (2.25, 0, [(2, B_, "Tab|#333333")] + [_l(c) for c in "QWERTYUIOP[]"] + [(1.5, B_, "↵|#333333", 2.0), (3.5, None)]
           + [_l(c) for c in "789"] + [_d("-")]),
    (3.25, 0, [(1.25, B_, "Ctrl|#333333"), (1, B_, "Caps|#333333")] + [_l(c) for c in "ASDFGHJKL;'"] + [(2.25, None),
           (1, B_, "↑|#333333"), (2.0, None)] + [_l(c) for c in "456"] + [_d("+")]),
    (4.25, 0, [(2.25, B_, "⇧|#333333")] + [_l(c) for c in "ZXCVBNM,./"] + [(2.75, B_, "⇧|#333333"), (0.5, None),
           _d("←"), _d("↓"), _d("→"), (0.5, None)] + [_l(c) for c in "123"] + [(1, B_, "Ent|#333333", 2.0)]),
    (5.25, 1.0, [(1.25, B_, "Alt|#333333"), (1.25, B_, "A|#333333"), (9, A_, ""), (1.25, B_, "A|#333333"), (1.25, B_, "Alt|#333333"),
           (6.0, None), (2, A_, "0|#333333"), (1, A_, ".|#333333")]),
]


def _amiga_top(d, W, H, img):
    """The top behind the keys: the vents (long grooves) over the back, the raised
    strip with AMIGA embossed on the right, the Commodore badge, POWER and DRIVE."""
    from replica import font
    d.rectangle([0, 0, W, H], fill=AM_CASE)
    for i in range(9):                                                                       # the grooves of the vents
        y = H * (0.06 + i * 0.045)
        d.rectangle([W * 0.02, y, W * 0.98, y + H * 0.018], fill="#bcb7a8")
    d.rectangle([0, H * 0.48, W, H * 0.5], fill="#c8c3b4")                                   # the step of the raised strip
    f = ImageFont_truetype(int(H * 0.2))
    d.text((W * 0.66, H * 0.56), "AMIGA", font=f, fill="#d0cbbc", stroke_width=2, stroke_fill="#c2bdae")
    bx = W * 0.86                                                                            # the C= badge
    d.rectangle([bx, H * 0.82, bx + H * 0.14, H * 0.96], fill="#f0ede4", outline="#9a968a", width=2)
    cx, cy, r = bx + H * 0.06, H * 0.89, H * 0.04
    d.arc([cx - r, cy - r, cx + r, cy + r], 40, 320, fill="#2a50b0", width=max(2, int(r * 0.45)))
    d.polygon([(cx + r * 0.4, cy - r * 0.5), (cx + r * 1.2, cy - r * 0.5), (cx + r * 0.9, cy - r * 0.05), (cx + r * 0.4, cy - r * 0.05)], fill="#2a50b0")
    d.polygon([(cx + r * 0.4, cy + r * 0.05), (cx + r * 0.9, cy + r * 0.05), (cx + r * 1.2, cy + r * 0.5), (cx + r * 0.4, cy + r * 0.5)], fill="#c8202a")


def ImageFont_truetype(size):
    from PIL import ImageFont
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", size)


def amiga(sc, X):
    """The Commodore Amiga 500 (1987) as in the photographs: the cream wedge, the
    grooved vents over the back and its comb of fins, the raised strip with AMIGA
    embossed and the Commodore badge, POWER and DRIVE lights; the keys in their dark
    well (the grey function keys in two groups, the inverted-T cursor keys, the
    numeric pad); the 1084S monitor with Workbench, the external drive, the mouse."""
    from replica import keyboard, decal, block
    case = plastic(AM_CASE, 0.5)
    k = 0.03 / 0.33
    sc.add(block(0.236, 0.035, 0.165, 0.006, planes=[(0, 1, k, 0.07 - k * 0.165)], soft=0.006), case, X)
    for i in range(60):                                                                       # the comb of fins along the back
        sc.add(sd_box(0.0016, 0.004, 0.006, 0.0006), case, X * T(-0.232 + i * 0.0079, 0.068, -0.16))
    sl = math.atan(k)
    ys = lambda z: 0.07 - k * (z + 0.165)
    sc.add(sd_box(0.232, 0.0005, 0.05, 0.0004), decal(0.232, 0.05, _amiga_top, base=AM_CASE, px=2048), X * T(0, ys(-0.1) + 0.0003, -0.1) * RX(sl))
    for i, c in enumerate(("#ff3030", "#4a4a4a")):                                           # POWER, DRIVE
        sc.add(sd_box(0.008, 0.001, 0.0016, 0.0006), Mat(c, emissive=c if i == 0 else None, emit=1.2 if i == 0 else 0.0), X * T(0.2, ys(-0.035) + 0.0006, -0.035 + i * 0.007) * RX(sl))
    # the dark well under the keys, then the keys
    zc = 0.055
    sc.add(sd_box(0.218, 0.002, 0.06, 0.002), Mat("#3a3832", rough=0.7), X * T(0.0, ys(zc) - 0.001, zc) * RX(sl))
    sdf, mat, _ = keyboard(AM_ROWS, cap=0.002, height=0.009, gap=0.0016, round_=0.0018, px_per_unit=48)
    sc.add(sdf, mat, X * T(0.0, ys(zc), zc) * RX(sl))
    sc.add(sd_box(0.0006, 0.004, 0.05, 0.001), Mat("#1a1a1a"), X * T(0.2365, 0.035, -0.04))          # the drive slot in the side
    # the 1084S monitor, the external drive, the mouse
    monitor(sc, X * T(0, 0.0, -0.42), 0.19, 0.16, 0.16, plastic("#d8d4c6"), pic_amiga)
    sc.add(block(0.08, 0.05, 0.12, 0.006), case, X * T(-0.34, 0, -0.3))
    sc.add(sd_box(0.05, 0.002, 0.002, 0.001), Mat("#1a1a1a"), X * T(-0.34, 0.04, -0.179))
    sc.add(block(0.03, 0.014, 0.05, 0.008), case, X * T(0.3, 0, 0.1))
    for sx in (-1, 1):
        sc.add(sd_box(0.012, 0.0015, 0.012, 0.0015), plastic("#c8c3b4"), X * T(0.3 + sx * 0.0135, 0.028, 0.064))


# ─── the Macintosh 128K (replica: 246 × 345 × 276 mm; keyboard 330 × 145) ──
MAC_CASE, MAC_KEY = "#e9e0cc", "#d6cfbe"
M_ = MAC_KEY
MAC_ROWS = [
    (0, 0, [(1, M_, "`|#555555")] + [(1, M_, c + "|#555555") for c in "1234567890-="] + [(1.5, M_, "Bksp|#555555")]),
    (1, 0, [(1.5, M_, "Tab|#555555")] + [(1, M_, c + "|#555555") for c in "QWERTYUIOP[]\\"]),
    (2, 0, [(1.75, M_, "Caps|#555555")] + [(1, M_, c + "|#555555") for c in "ASDFGHJKL;'"] + [(1.75, M_, "Return|#555555")]),
    (3, 0, [(2.25, M_, "Shift|#555555")] + [(1, M_, c + "|#555555") for c in "ZXCVBNM,./"] + [(2.25, M_, "Shift|#555555")]),
    (4, 1.0, [(1, M_, "Opt|#555555"), (1.25, M_, "⌘|#555555"), (7, M_, ""), (1.5, M_, "Enter|#555555"), (1, M_, "Opt|#555555")]),
]


def mac(sc, X):
    """The Macintosh 128K (1984) as in the photographs: the upright beige case, the
    screen deep in its bevelled frame, the floppy slot and the rainbow apple on the
    chin; the separate keyboard and the one-button mouse on its cord."""
    from replica import keyboard, decal, block, rainbow_apple
    case = plastic(MAC_CASE, 0.5)
    # the case: slightly narrower at the back (the sides taper), the foot set in
    sc.add(block(0.123, 0.1725, 0.138, 0.01, planes=[(1, 0, 0.08, 0.123 + 0.08 * 0.0), (-1, 0, 0.08, 0.123)]), case, X * T(0, 0.004, -0.04))
    sc.add(block(0.118, 0.004, 0.13, 0.002), plastic("#b8b0a0"), X * T(0, 0, -0.04))                              # the foot
    F = X * T(0, 0.004, -0.04 + 0.138)                                                                             # the front face (z = 0)
    # the screen in its bevelled recess: a frame sloping inwards, the dark glass
    # the recess: a pocket carved in the front, its walls sloping in (a box cut by four planes)
    sc.add(block(0.096, 0.08, 0.03, 0.003, planes=[(0, 1, -0.9, 0.06), (0, -1, -0.9, 0.06), (1, 0, -0.9, 0.075), (-1, 0, -0.9, 0.075)], y0=False),
           Mat("#cfc6b2", rough=0.55), F * T(0, 0.235, 0.0), op="sub")
    sc.add(sd_box(0.078, 0.06, 0.004, 0.012), emissive_screen(lambda pl: pic_mac(0.078, 0.06)(torch.stack([pl[:, 0], pl[:, 2], -pl[:, 1]], -1)), 1.2),
           F * T(0, 0.235, -0.024))
    # the chin: the floppy slot in its recess, the apple
    sc.add(sd_box(0.045, 0.006, 0.003, 0.002), Mat("#d8d0bc", rough=0.5), F * T(0.035, 0.088, -0.001))
    sc.add(sd_box(0.036, 0.0018, 0.004, 0.001), Mat("#1a1a1a"), F * T(0.03, 0.088, 0.0))
    sc.add(sd_box(0.01, 0.01, 0.0008, 0.002), decal(0.01, 0.01, lambda d, W, H, img: rainbow_apple(d, W / 2, H / 2, H * 0.8),
           base="#ddd4c0", px=256, axes=(0, 1), flip_v=True), F * T(-0.085, 0.05, 0.0005))
    # the keyboard: a cream tray with a rounded rim, the keys
    KX = X * T(0, 0, 0.27)
    sc.add(block(0.165, 0.014, 0.0725, 0.006, planes=[(0, 1, 0.12, 0.024)], soft=0.004), case, KX)
    sdf, mat, _ = keyboard(MAC_ROWS, cap=0.0022, height=0.008, gap=0.0016, round_=0.002, px_per_unit=48)
    sc.add(sdf, mat, KX * T(0.0, 0.019, 0.002) * RX(math.atan(0.12)))
    # the mouse and its cord
    MO = X * T(0.25, 0, 0.27)
    sc.add(block(0.03, 0.018, 0.055, 0.012), case, MO)
    sc.add(sd_box(0.022, 0.002, 0.014, 0.002), plastic("#a8a090", 0.45), MO * T(0, 0.036, -0.032))
    chain(sc, bezier((0.25, 0.01, 0.215), (0.25, 0.01, 0.12), (0.2, 0.01, 0.05), (0.14, 0.05, -0.1), 16), [0.0016] * 16, Mat("#c8c0ae"), X)


# ─── the IBM PC 5150 (replica: unit 500 × 410 × 142 mm, 5151 display, Model F) ─
IBM_CASE, IBM_KEY, IBM_MOD = "#d9d7d1", "#e6e4de", "#a9a8a3"
I_, J_ = IBM_KEY, IBM_MOD
IBM_ROWS = [(r, 0, [(1, J_, f"F{2 * r + 1}|#333333"), (1, J_, f"F{2 * r + 2}|#333333")]) for r in range(5)] + [
    (0, 2.5, [(1, J_, "Esc|#333333")] + [(1, I_, c + "|#333333") for c in "1234567890-="] + [(1.5, J_, "←|#333333"), (1.0, None),
              (2, J_, "Num|#333333"), (2, J_, "Scroll|#333333")]),
    (1, 2.5, [(1.5, J_, "⇥|#333333")] + [(1, I_, c + "|#333333") for c in "QWERTYUIOP[]"] + [(1.5, J_, "↵|#333333", 2.0), (0.5, None)]
              + [(1, I_, c + "|#333333") for c in "789"] + [(1, J_, "−|#333333")]),
    (2, 2.5, [(1.5, J_, "Ctrl|#333333")] + [(1, I_, c + "|#333333") for c in "ASDFGHJKL;'`"] + [(2.0, None)]
              + [(1, I_, c + "|#333333") for c in "456"] + [(1, J_, "+|#333333", 3.0)]),
    (3, 2.5, [(1.5, J_, "⇧|#333333"), (1, I_, "\\|#333333")] + [(1, I_, c + "|#333333") for c in "ZXCVBNM,./"] + [(1.5, J_, "⇧|#333333"),
              (1, J_, "PrtSc|#333333"), (0.5, None)] + [(1, I_, c + "|#333333") for c in "123"]),
    (4, 2.5, [(1.5, J_, "Alt|#333333"), (9, I_, ""), (1.5, J_, "Caps|#333333"), (3.0, None), (2, I_, "0 Ins|#333333"), (1, I_, ". Del|#333333")]),
]


def _ibm_badge(sub):
    """A small square badge: the striped IBM and a line of small print below."""
    from replica import ibm_logo, font
    def draw(d, W, H, img):
        d.rectangle([0, 0, W - 1, H - 1], fill="#e8e6e0", outline="#9a9890", width=max(1, W // 60))
        ibm_logo(d, W * 0.14, H * 0.12, H * 0.32, "#2a2a2e", "#e8e6e0")
        d.multiline_text((W * 0.14, H * 0.55), sub, font=font("Lato-Bold.ttf", int(H * 0.11)), fill="#4a4a50", spacing=1)
    return draw


def _ibm_screen(d, W, H, img):
    """The 5151's picture: green on black, the striped IBM in big letters."""
    from replica import ibm_logo
    d.rectangle([0, 0, W, H], fill="#06120a")
    from replica import SERIF_BOLD
    from PIL import ImageFont
    h = H * 0.3                                                                              # centred: measured first (it was cut on the right)
    w = d.textlength("IBM", font=ImageFont.truetype(SERIF_BOLD, int(h * 1.25)))
    ibm_logo(d, (W - w) / 2, H * 0.32, h, "#3aff8a", "#06120a")
    d.rectangle([W * 0.06, H * 0.86, W * 0.08, H * 0.9], fill="#3aff8a")                    # the cursor


def ibm_pc(sc, X):
    """The IBM PC 5150 (1981) as in the photographs: the wide grey system unit, its
    IBM badge and vent slots on the left, the black bay of the two floppy drives on
    the right; the 5151 display on top (the bevelled screen, the badge and two knobs
    on the right, the striped IBM in green); the Model F keyboard on its legs (the
    function keys in two columns on the left, the numeric pad on the right)."""
    from replica import keyboard, decal, block
    case = plastic(IBM_CASE, 0.55)
    sc.add(block(0.25, 0.071, 0.205, 0.006), case, X * T(0, 0, -0.12))
    F = X * T(0, 0, -0.12 + 0.205)                                                            # its front (z = 0)
    sc.add(block(0.115, 0.042, 0.02, 0.003, y0=False), Mat("#1a1a1a", rough=0.5), F * T(0.095, 0.074, -0.012), op="sub")   # the bay
    sc.add(sd_box(0.113, 0.04, 0.002, 0.002), Mat("#161616", rough=0.5), F * T(0.095, 0.074, -0.03))
    for i in range(2):                                                                       # the two drives: slot, door lever
        x = 0.043 + i * 0.104
        sc.add(sd_box(0.04, 0.0015, 0.003, 0.001), Mat("#050505"), F * T(x, 0.084, -0.027))
        sc.add(sd_box(0.008, 0.016, 0.004, 0.002), Mat("#262626", rough=0.4), F * T(x, 0.074, -0.026))
    sc.add(sd_sphere(0.002), Mat("#e01a10", emissive="#ff2a1a", emit=1.2), F * T(0.13, 0.055, -0.027))
    sc.add(sd_box(0.018, 0.018, 0.0008, 0.001), decal(0.018, 0.018, _ibm_badge("Personal\nComputer"), base="#e8e6e0", px=256, axes=(0, 1), flip_v=True),
           F * T(-0.215, 0.11, 0.0008))
    for i in range(18):                                                                      # the vent slots
        sc.add(sd_box(0.0012, 0.017, 0.002, 0.0008), Mat("#5a5a58"), F * T(-0.215 + i * 0.0062, 0.04, 0.0))
    # the 5151 display on the unit
    M = X * T(-0.02, 0.142, -0.13)
    sc.add(block(0.19, 0.14, 0.17, 0.02), case, M)
    MF = M * T(0, 0, 0.17)
    sc.add(block(0.15, 0.115, 0.025, 0.006, planes=[(0, 1, -1.0, 0.085), (0, -1, -1.0, 0.085), (1, 0, -1.0, 0.12), (-1, 0, -1.0, 0.12)], y0=False),
           Mat("#c8c6c0", rough=0.6), MF * T(-0.03, 0.15, 0.0), op="sub")
    sc.add(sd_box(0.118, 0.088, 0.003, 0.014), decal(0.118, 0.088, _ibm_screen, base="#06120a", px=768, axes=(0, 1), flip_v=True, emissive_k=1.2, rough=0.1, clearcoat=1.0),
           MF * T(-0.03, 0.15, -0.024))
    sc.add(sd_box(0.014, 0.02, 0.0008, 0.001), decal(0.014, 0.02, _ibm_badge("Personal\nComputer\nDisplay"), base="#e8e6e0", px=256, axes=(0, 1), flip_v=True),
           MF * T(0.15, 0.22, 0.0008))
    for i in range(2):
        sc.add(sd_cylinder(0.005, 0.009, 0.002), Mat("#1a1a1a", rough=0.4), MF * T(0.15, 0.12 - i * 0.04, 0.004) * RX(math.pi / 2))
    # the Model F keyboard: a tray with a raised back, the badge, the legs
    K = X * T(0, 0, 0.23)
    sc.add(block(0.25, 0.018, 0.1, 0.008, planes=[(0, 1, 0.3, 0.032)], soft=0.006), case, K)
    sdf, mat, _ = keyboard(IBM_ROWS, cap=0.0028, height=0.011, gap=0.0016, round_=0.002, px_per_unit=48)
    sc.add(sdf, mat, K * T(0.0, 0.026, 0.008) * RX(math.atan(0.3)))
    sc.add(sd_box(0.018, 0.0006, 0.008, 0.0005), decal(0.018, 0.008, lambda d, W, H, img: __import__("replica").ibm_logo(d, W * 0.2, H * 0.15, H * 0.6, "#e8e6e0", "#2a2a2e") if d.rectangle([0, 0, W, H], fill="#2a2a2e") is None else 0,
           base="#2a2a2e", px=256), K * T(-0.215, 0.034 + 0.3 * 0.09 * 0, -0.088) * RX(math.atan(0.3)))
    for sx in (-1, 1):
        sc.add(sd_cylinder(0.006, 0.006, 0.002), case, K * T(sx * 0.255, 0.018, -0.08) * RZ(math.pi / 2))


# ─── the Thomson MO5 (replica: 340 × 220 mm, the black case, the rubber keys) ─
MO5_CASE, MO5_KEY = "#2c2c2e", "#77767a"
G_ = MO5_KEY
MO5_ROWS = [
    (0, 0, [(1, G_, "STOP")] + [(1, G_, c) for c in "1234567890"] + [(1, G_, "-"), (1, G_, "+"), (1, G_, "ACC"), (1, G_, "↑")]),
    (1, 0, [(1, G_, "CNT"), (0.5, None)] + [(1, G_, c) for c in "AZERTYUIOP"] + [(1, G_, "/"), (1, G_, "*"), (1, G_, "←"), (1, G_, "→")]),
    (2, 0, [(1, G_, "RAZ"), (1, G_, "↵")] + [(1, G_, c) for c in "QSDFGHJKLM"] + [(1.6, G_, "ENTREE"), (0.4, None), (1, G_, "↓")]),
    (3, 0, [(1.75, "#c4dc6c", ""), (0.75, None)] + [(1, G_, c) for c in "WXCVBN,.@"] + [(1.75, "#1a1a1c", "BASIC"), (1, G_, "INS"), (1, G_, "EFF")]),
    (4, 4.9, [(3.9, G_, "")]),
]


def _mo5_top(d, W, H, img):
    """The MO5's top in front of the ribs: the big outlined MO5, THOMSON and its mark."""
    from replica import font
    f = font("Lato-Black.ttf", int(H * 0.62))
    d.text((W * 0.04, H * 0.08), "MO5", font=f, fill=MO5_CASE, stroke_width=max(2, int(H * 0.025)), stroke_fill="#c8d890")
    d.text((W * 0.6, H * 0.18), "THOMSON", font=font("Lato-Black.ttf", int(H * 0.2)), fill="#e8e8e8")
    cx, cy, r = W * 0.6 + d.textlength("THOMSON", font=font("Lato-Black.ttf", int(H * 0.2))) + H * 0.16, H * 0.3, H * 0.1
    d.regular_polygon((cx, cy, r), 6, fill="#e8e8e8")
    d.regular_polygon((cx, cy, r * 0.55), 6, fill=MO5_CASE)
    d.text((W * 0.44, H * 0.2), "UC 50.001", font=font("Lato-Bold.ttf", int(H * 0.15)), fill="#d8d8d8")


def mo5(sc, X):
    """The Thomson MO5 (1984), the French schools' computer, as in the photographs:
    the black wedge, five ribs across its back on the right, the cartridge window and
    the INITIAL PROG. button on the left, the outlined MO5 and THOMSON, the grey
    rubber keys (the lime shift, the black BASIC); the light pen on its cord, the
    cassette recorder, the TV."""
    from replica import keyboard, decal, block
    black = Mat(MO5_CASE, rough=0.45, clearcoat=0.3)
    # the wedge: 25 mm at the front (chamfered), 60 mm at the back
    k = 0.035 / 0.16
    sc.add(block(0.17, 0.03, 0.11, 0.004, planes=[(0, 1, k, 0.06 - 0.05 * k), (0, 1, 1.2, 0.03 + 1.2 * 0.1)]), black, X)
    # the back strip (z < -0.05): five ribs over the right two thirds, the left notched too
    for i in range(5):
        z = -0.105 + i * 0.012
        sc.add(sd_box(0.11, 0.003, 0.004, 0.0015), black, X * T(0.055, 0.061, z))
        sc.add(sd_box(0.02, 0.003, 0.004, 0.0015), black, X * T(-0.155, 0.061, z))
    sc.add(sd_box(0.05, 0.002, 0.024, 0.002), Mat("#0c0c0e", rough=0.25, clearcoat=0.8), X * T(-0.075, 0.0605, -0.078))     # the cartridge window
    sc.add(sd_box(0.006, 0.002, 0.003, 0.001), plastic("#e8e8e4"), X * T(-0.125, 0.061, -0.06))                               # INITIAL PROG.
    # the marks on the slope, and the keys
    sl = math.atan(k)
    ys = lambda z: 0.06 - k * (z + 0.05)
    sc.add(sd_box(0.165, 0.0005, 0.03, 0.0004), decal(0.165, 0.03, _mo5_top, base=MO5_CASE, px=2048, rough=0.45),
           X * T(0, ys(-0.02) + 0.0003, -0.02) * RX(sl))
    zc = 0.03
    sdf, mat, _ = keyboard(MO5_ROWS, cap=0.0012, height=0.007, gap=0.0024, round_=0.0018, px_per_unit=48, legend_col="#f0f0f0")
    sc.add(sdf, mat, X * T(0.0, ys(zc), zc) * RX(sl))
    # the light pen lying in front, its cord curling to the back
    sc.add(sd_capsule((0.12, 0.008, 0.16), (0.2, 0.008, 0.12), 0.006, 0.004), plastic("#2a2a2c"), X)
    chain(sc, bezier((0.2, 0.008, 0.12), (0.27, 0.01, 0.06), (0.24, 0.02, -0.05), (0.17, 0.03, -0.1), 14), [0.0025] * 14, Mat("#1a1a1a"), X)
    # the cassette recorder (MO5's own, black) and the TV
    R = X * T(0.3, 0, -0.02)
    sc.add(block(0.06, 0.024, 0.085, 0.006), black, R)
    sc.add(sd_box(0.042, 0.003, 0.03, 0.003), Mat("#5a5a5a", rough=0.2, clearcoat=0.8), R * T(0, 0.049, -0.02))
    for i in range(4):
        sc.add(sd_box(0.007, 0.004, 0.009, 0.002), plastic("#77767a"), R * T(-0.027 + i * 0.018, 0.05, 0.06))
    tv(sc, X * T(0, 0.0, -0.36), 0.19, 0.14, lambda w, h: pic_game(w, h, 9.0))


# ─── the consoles ───────────────────────────────────────────────────────────
def atari2600(sc, X):
    """The Atari 2600 (the four-switch woody, 1980) as in the photographs: the low black
    body, the wood front leaning back, the ribbed black top, the raised panel at the
    back with its four switches, the cartridge slot and VIDEO COMPUTER SYSTEM in
    orange lines, ATARI and the Fuji on the wood; the CX40 joystick; the TV behind."""
    from replica import font, block
    body = Mat("#161412", rough=0.4)
    # the body: the front leans back (the wood), the top rises a little to the back
    sc.add(block(0.168, 0.03, 0.11, 0.006, planes=[(0, 0.35, 1, 0.11 + 0.35 * 0.0)], soft=0.004), body, X)
    wood = Mat("#8a5a2a", rough=0.4, clearcoat=0.5,
               pattern=lambda pl, pw, n: {"albedo": srgb("#8a5a2a") * (0.82 + 0.14 * torch.sin(pl[:, 1] * 900 + 10 * fbm(pl * vec(6, 60, 60), 3))
                                                                      + 0.06 * fbm(pl * vec(3, 200, 200), 2))[:, None]})   # the grain runs along the panel
    lean = math.atan(0.35)
    W_ = X * T(0, 0.03, 0.11 - 0.35 * 0.03 + 0.001) * RX(-lean)
    sc.add(sd_box(0.165, 0.024, 0.002, 0.002), wood, W_)
    def logo(d, Wd, Hd, img):
        d.rectangle([0, 0, Wd, Hd], fill="#000000")
        u = Hd * 0.3                                                                          # the Fuji
        cx = Wd * 0.5
        d.rectangle([cx - u * 0.12, Hd * 0.05, cx + u * 0.12, Hd * 0.6], fill="#f0ece4")
        for sgn in (-1, 1):
            d.polygon([(cx + sgn * u * 0.25, Hd * 0.05), (cx + sgn * u * 0.42, Hd * 0.05), (cx + sgn * u * 0.95, Hd * 0.6), (cx + sgn * u * 0.65, Hd * 0.6)], fill="#f0ece4")
        d.text((Wd * 0.12, Hd * 0.64), "ATARI", font=font("Lato-Black.ttf", int(Hd * 0.3)), fill="#f0ece4")
    from replica import decal
    sc.add(sd_box(0.016, 0.016, 0.0006, 0.0004), decal(0.016, 0.016, logo, base="#8a5a2a", px=256, axes=(0, 1), flip_v=True, rough=0.4, clearcoat=0.5),
           W_ * T(0.14, -0.004, 0.0022))
    # the ribs of the top
    for i in range(16):
        sc.add(sd_box(0.165, 0.0018, 0.0022, 0.001), body, X * T(0, 0.0605, 0.085 - i * 0.0085))
    # the raised panel at the back: switches, slot, the orange lines
    P = X * T(0, 0.06, -0.085)
    sc.add(block(0.15, 0.012, 0.028, 0.003), body, P)
    def panel(d, Wd, Hd, img):
        d.rectangle([0, 0, Wd, Hd], fill="#161412")
        d.rectangle([Wd * 0.02, Hd * 0.1, Wd * 0.98, Hd * 0.9], outline="#d0701a", width=max(2, Hd // 20))
        d.rectangle([Wd * 0.36, Hd * 0.25, Wd * 0.64, Hd * 0.75], outline="#d0701a", width=max(2, Hd // 25))
        d.text((Wd * 0.38, Hd * 0.12), "VIDEO COMPUTER SYSTEM", font=font("Lato-Bold.ttf", int(Hd * 0.1)), fill="#e8e0d0")
        f = font("Lato-Bold.ttf", int(Hd * 0.08))
        for x, lab in ((0.07, "ON  OFF"), (0.2, "COLOR  B•W"), (0.71, "GAME SELECT"), (0.85, "GAME RESET")):
            d.text((Wd * x, Hd * 0.8), lab, font=f, fill="#d0701a")
    sc.add(sd_box(0.148, 0.0005, 0.026, 0.0004), decal(0.148, 0.026, panel, base="#161412", px=1536), P * T(0, 0.0241, 0))
    sc.add(sd_box(0.035, 0.004, 0.007, 0.001), Mat("#050505"), P * T(0, 0.022, 0.0))                       # the cartridge slot
    for x in (-0.12, -0.08, 0.08, 0.12):                                                                    # the four switches
        sc.add(sd_box(0.004, 0.006, 0.003, 0.0015), Mat("#141414", rough=0.3), P * T(x, 0.028, 0.006))
        sc.add(sd_cylinder(0.0015, 0.004, 0.001), Mat("#d8d8d8", metal=0.7, rough=0.25), P * T(x, 0.035, 0.006))
    # the joystick: black base, the red button, the orange ring
    J = X * T(0.26, 0, 0.08)
    sc.add(block(0.04, 0.015, 0.04, 0.008), body, J)
    sc.add(sd_torus(0.016, 0.0012), Mat("#d0701a", rough=0.4), J * T(0, 0.0305, 0))
    sc.add(sd_capsule((0, 0.02, 0), (0, 0.075, 0), 0.007), body, J)
    sc.add(sd_cylinder(0.004, 0.008, 0.002), Mat("#c82020", rough=0.3, clearcoat=0.8), J * T(-0.025, 0.032, -0.025))
    tv(sc, X * T(0, 0.0, -0.35), 0.2, 0.15, lambda w, h: pic_game(w, h, 0.0))


def _text_decal(draw, w, h, base, axes=(0, 2), flip_v=False, px=1024, rough=0.4, clearcoat=0.3):
    from replica import decal
    return decal(w, h, draw, base=base, px=px, axes=axes, flip_v=flip_v, rough=rough, clearcoat=clearcoat)


# ─── the NES (replica: 256 × 203 × 85 mm) ───────────────────────────────────
NES_LIGHT, NES_DARK, NES_RED = "#cfcfcd", "#8d8d8f", "#c8202a"


def nes(sc, X):
    """The Nintendo Entertainment System (1985) as in the photographs: the light grey
    top over the dark grey base, the front flap with Nintendo ENTERTAINMENT SYSTEM in
    red, the black band and the ribs on the right, POWER and RESET, the two ports;
    the controller (black face, cross, red B and A)."""
    from replica import block, font
    light, dark = plastic(NES_LIGHT, 0.5), plastic(NES_DARK, 0.5)
    sc.add(block(0.128, 0.022, 0.1, 0.004), dark, X)                                    # the base
    sc.add(block(0.128, 0.022, 0.1, 0.004), light, X * T(0, 0.044, 0))                  # the top
    # the black band across the top and the right front (over the ports), the ribs behind it
    sc.add(sd_box(0.02, 0.0446, 0.1005, 0.003), plastic("#2a2a2c", 0.45), X * T(0.068, 0.044, 0))
    for i in range(22):                                                                 # the ribs on the top, right of the door
        sc.add(sd_box(0.016, 0.0008, 0.0012, 0.0006), Mat("#a8a8a6", rough=0.5), X * T(0.031, 0.0888, -0.092 + i * 0.0058))
    # the front flap (the cartridge door) with the name, and POWER / RESET under it
    def flap(d, W, H, img):
        d.rectangle([0, 0, W, H], fill=NES_LIGHT)
        d.text((W * 0.05, H * 0.2), "Nintendo", font=font("Lato-Black.ttf", int(H * 0.32)), fill=NES_RED)
        d.text((W * 0.05, H * 0.6), "ENTERTAINMENT SYSTEM", font=font("Lato-Bold.ttf", int(H * 0.18)), fill=NES_RED)
    sc.add(sd_box(0.07, 0.016, 0.0008, 0.0008), _text_decal(flap, 0.07, 0.016, NES_LIGHT, axes=(0, 1), flip_v=True), X * T(-0.04, 0.062, 0.1005))
    sc.add(sd_box(0.075, 0.0008, 0.06, 0.002), Mat("#c4c4c2", rough=0.5), X * T(-0.04, 0.0888, 0.035))                  # the door's lid on top
    for i, lab in enumerate(("POWER", "RESET")):
        def btn(d, W, H, img, lab=lab):
            d.rectangle([0, 0, W, H], fill="#9a9a9c")
            d.text((W * 0.12, H * 0.55), lab, font=font("Lato-Bold.ttf", int(H * 0.32)), fill=NES_RED)
        sc.add(sd_box(0.012, 0.006, 0.003, 0.0015), _text_decal(btn, 0.012, 0.006, "#9a9a9c", axes=(0, 1), flip_v=True), X * T(-0.075 + i * 0.03, 0.022, 0.101))
    sc.add(sd_box(0.0016, 0.0016, 0.001, 0.0005), Mat("#2a0a0a"), X * T(-0.095, 0.022, 0.1015))
    for i in range(2):                                                                  # the ports on the black band
        sc.add(sd_box(0.006, 0.009, 0.002, 0.002), Mat("#101010"), X * T(0.058 + i * 0.02, 0.022, 0.101))
    # the controller: light grey, its black face, the cross, select / start, red B A
    C = X * T(-0.02, 0, 0.19)
    sc.add(block(0.058, 0.007, 0.026, 0.004), light, C)
    def face(d, W, H, img):
        d.rectangle([0, 0, W, H], fill=NES_LIGHT)
        d.rectangle([W * 0.04, H * 0.12, W * 0.96, H * 0.92], fill="#1e1e20")
        for j in range(3):
            d.rectangle([W * 0.36, H * (0.2 + j * 0.12), W * 0.62, H * (0.28 + j * 0.12)], fill="#8a8a8c")
        d.rectangle([W * 0.36, H * 0.56, W * 0.62, H * 0.78], fill="#d8d8d8")
        d.text((W * 0.37, H * 0.44), "SELECT  START", font=font("Lato-Black.ttf", int(H * 0.07)), fill=NES_RED)
        d.text((W * 0.69, H * 0.22), "Nintendo", font=font("Lato-Black.ttf", int(H * 0.1)), fill=NES_RED)
        for j in range(2):
            d.rectangle([W * (0.66 + j * 0.14), H * 0.52, W * (0.78 + j * 0.14), H * 0.8], fill="#d8d8d8")
    sc.add(sd_box(0.056, 0.0005, 0.024, 0.0004), _text_decal(face, 0.056, 0.024, NES_LIGHT), C * T(0, 0.0141, 0))
    sc.add(sd_box(0.008, 0.002, 0.0026, 0.001), Mat("#141414"), C * T(-0.036, 0.0158, 0.004))
    sc.add(sd_box(0.0026, 0.002, 0.008, 0.001), Mat("#141414"), C * T(-0.036, 0.0158, 0.004))
    for j in range(2):
        sc.add(sd_box(0.0042, 0.0012, 0.0016, 0.0016), Mat("#202022"), C * T(-0.009 + j * 0.0105, 0.0152, 0.004))
        sc.add(sd_cylinder(0.0022, 0.0048, 0.001), Mat(NES_RED, rough=0.25, clearcoat=0.9), C * T(0.0235 + j * 0.0165, 0.016, 0.0055))
    chain(sc, bezier((-0.02, 0.008, 0.164), (-0.02, 0.01, 0.13), (0.06, 0.01, 0.13), (0.068, 0.022, 0.102), 14), [0.0018] * 14, Mat("#1a1a1a"), X)
    tv(sc, X * T(0, 0.0, -0.33), 0.2, 0.15, lambda w, h: pic_game(w, h, 5.0))


# ─── the Mega Drive (replica: 280 × 212 × 60 mm, the first model) ───────────
def megadrive(sc, X):
    """The Sega Mega Drive (1988) as in the photographs: the glossy black slab, the
    big round raised disc with the cartridge slot and the gold 16-BIT plate, the
    vents at the back left, volume, power and reset at the front left, MEGA DRIVE
    SEGA on the front right; the three-button pad on its cord."""
    from replica import block, font
    black = Mat("#161618", rough=0.22, clearcoat=0.9)
    sc.add(block(0.14, 0.024, 0.106, 0.01, planes=[(0, 1, 0.6, 0.048 + 0.6 * 0.075)], soft=0.01), black, X)
    disc = X * T(0.03, 0.048, -0.01)
    sc.add(sd_cylinder(0.006, 0.084, 0.004), black, disc)
    sc.add(sd_cylinder(0.0065, 0.08, 0.003), Mat("#202024", rough=0.2, clearcoat=0.9), disc)
    sc.add(sd_box(0.04, 0.008, 0.004, 0.001), Mat("#050505"), disc * T(0, 0.0, -0.045))                           # the slot
    def plate(d, W, H, img):
        d.rectangle([0, 0, W, H], fill="#1c1a1a")
        d.text((W * 0.1, H * 0.22), "16-BIT", font=font("Lato-Black.ttf", int(H * 0.27)), fill="#d8b050")
        d.rectangle([0, H * 0.72, W, H], fill="#f0f0f0")
        d.text((W * 0.12, H * 0.76), "POWER • ON", font=font("Lato-Bold.ttf", int(H * 0.15)), fill="#222222")
    sc.add(sd_box(0.03, 0.0006, 0.026, 0.0005), _text_decal(plate, 0.03, 0.026, "#1c1a1a"), disc * T(0, 0.0072, 0.03))
    for i in range(16):                                                                                         # the vents
        sc.add(sd_box(0.03, 0.002, 0.0012, 0.0006), Mat("#050505"), X * T(-0.095, 0.0478, -0.095 + i * 0.0045))
    # the front-left controls: volume slider, power switch, reset
    sc.add(sd_box(0.006, 0.0015, 0.016, 0.001), Mat("#050505"), X * T(-0.112, 0.0478, 0.05))
    sc.add(sd_box(0.003, 0.004, 0.005, 0.001), Mat("#202022"), X * T(-0.112, 0.05, 0.055))
    sc.add(sd_box(0.012, 0.0015, 0.006, 0.001), Mat("#050505"), X * T(-0.083, 0.0478, 0.042))
    sc.add(sd_box(0.004, 0.003, 0.004, 0.001), Mat("#c82020"), X * T(-0.079, 0.049, 0.042))
    sc.add(sd_box(0.009, 0.0022, 0.006, 0.002), plastic("#e8e8ec", 0.4), X * T(-0.083, 0.0485, 0.062))
    def name(d, W, H, img):
        d.rectangle([0, 0, W, H], fill="#161618")
        d.text((W * 0.02, H * 0.15), "MEGA DRIVE", font=font("Lato-Bold.ttf", int(H * 0.55)), fill="#e8e8e8")
        d.text((W * 0.66, H * 0.0), "SEGA", font=font("Lato-Black.ttf", int(H * 0.8)), fill="#e8e8e8")
    sc.add(sd_box(0.04, 0.0005, 0.0055, 0.0004), _text_decal(name, 0.04, 0.0055, "#161618"), X * T(0.085, 0.0483, 0.066))
    # the pad
    P = X * T(0.05, 0, 0.2)
    sc.add(block(0.055, 0.009, 0.03, 0.014), black, P)
    for sx in (-1, 1):
        sc.add(sd_capsule((sx * 0.035, 0.009, 0.0), (sx * 0.045, 0.009, 0.022), 0.016), black, P)
    sc.add(sd_cylinder(0.002, 0.012, 0.001), Mat("#0c0c0c", rough=0.4), P * T(-0.032, 0.018, 0.0))
    sc.add(sd_box(0.009, 0.0025, 0.003, 0.001), Mat("#202022"), P * T(-0.032, 0.02, 0.0))
    sc.add(sd_box(0.003, 0.0025, 0.009, 0.001), Mat("#202022"), P * T(-0.032, 0.02, 0.0))
    for j in range(3):
        sc.add(sd_cylinder(0.003, 0.0045, 0.001), Mat("#202022", rough=0.3, clearcoat=0.8), P * T(0.018 + j * 0.012, 0.018, 0.008 - j * 0.006))
    sc.add(sd_box(0.004, 0.0015, 0.002, 0.0015), Mat("#2a6ad8"), P * T(0.015, 0.0185, -0.016))
    chain(sc, bezier((0.05, 0.01, 0.17), (0.04, 0.01, 0.13), (-0.02, 0.01, 0.13), (-0.03, 0.02, 0.105), 14), [0.0018] * 14, Mat("#0e0e0e"), X)


# ─── the PlayStation (replica: 270 × 188 × 60 mm, SCPH-1000 era) ─────────────
def playstation(sc, X):
    """The Sony PlayStation (1994) as in the photographs: the grey slab, the round lid
    in the middle with SONY and the PlayStation mark, the raised blocks left and
    right, RESET, POWER and OPEN, the ports 1 and 2 in front, the ribs on the right
    side; the controller with its handles and its four symbols."""
    from replica import block, font
    grey = plastic("#b4b4b8", 0.5)
    sc.add(block(0.135, 0.022, 0.094, 0.006), grey, X)
    for sx in (-1, 1):
        sc.add(block(0.042, 0.007, 0.094, 0.004), grey, X * T(sx * 0.093, 0.044 - 0.007, 0))
    sc.add(block(0.051, 0.002, 0.094, 0.003), grey, X * T(0, 0.042, 0))
    def lid(d, W, H, img):
        d.ellipse([0, 0, W - 1, H - 1], fill="#b9b9bd", outline="#8a8a8e", width=max(2, W // 200))
        d.text((W * 0.38, H * 0.14), "SONY", font=font("DejaVuSerif-Bold.ttf" if False else "Lato-Black.ttf", int(H * 0.07)), fill="#3a3a40")
        cx, cy, u = W * 0.5, H * 0.7, H * 0.06                                                       # the PS mark
        d.polygon([(cx - u * 0.9, cy + u * 0.4), (cx + u * 0.4, cy - u * 0.2), (cx + u * 1.4, cy + u * 0.2), (cx, cy + u * 0.9)], fill="#f2c230")
        d.polygon([(cx - u * 0.6, cy + u * 0.7), (cx + u * 0.2, cy + u * 0.35), (cx + u * 1.2, cy + u * 0.6), (cx + u * 0.3, cy + u * 1.1)], fill="#2a5ad0")
        d.polygon([(cx - u * 0.9, cy + u * 0.4), (cx + u * 0.2, cy - u * 0.1), (cx + u * 0.2, cy + u * 0.35), (cx - u * 0.6, cy + u * 0.7)], fill="#20a050")
        d.rectangle([cx - u * 0.15, cy - u * 1.4, cx + u * 0.25, cy + u * 0.5], fill="#d82a2a")
        d.ellipse([cx, cy - u * 1.4, cx + u * 0.9, cy - u * 0.3], fill="#d82a2a")
        d.text((W * 0.38, H * 0.8), "PlayStation", font=font("Lato-Bold.ttf", int(H * 0.045)), fill="#3a3a40")
    sc.add(sd_cylinder(0.0035, 0.07, 0.003), _text_decal(lid, 0.07, 0.07, "#b9b9bd", px=1024), X * T(0, 0.0475, -0.005))
    # RESET, POWER, OPEN
    sc.add(sd_cylinder(0.002, 0.008, 0.0015), grey, X * T(-0.095, 0.0455, 0.02))
    for x, lab in ((-0.095, "⏻"), (0.095, "⏏")):
        sc.add(sd_cylinder(0.0025, 0.016, 0.002), plastic("#a6a6aa"), X * T(x, 0.0455, 0.06))
    # the ports in front, the ribs on the right side
    for i in range(2):
        sc.add(sd_box(0.016, 0.006, 0.002, 0.001), Mat("#7a7a7e"), X * T(-0.03 + i * 0.06, 0.02, 0.0945))
    for i in range(24):
        sc.add(sd_box(0.0012, 0.016, 0.0016, 0.0006), Mat("#9a9a9e"), X * T(0.1355, 0.022, -0.085 + i * 0.0073))
    # the controller
    C = X * T(0.02, 0, 0.19)
    sc.add(block(0.045, 0.011, 0.022, 0.009), grey, C)
    for sx in (-1, 1):
        sc.add(sd_capsule((sx * 0.04, 0.011, 0.0), (sx * 0.052, 0.009, 0.042), 0.013), grey, C)
    for j, (c, a) in enumerate((("#30b090", 0.0), ("#e05050", 1.5708), ("#5a8ae0", 3.1416), ("#d070c0", 4.7124))):
        sc.add(sd_cylinder(0.0035, 0.0045, 0.0015), Mat("#3a3a3e", rough=0.3, clearcoat=0.8), C * T(0.04 + 0.01 * math.sin(a), 0.0235, -0.01 * math.cos(a)))
        sc.add(sd_cylinder(0.0005, 0.002, 0.0004), Mat(c, emissive=c, emit=0.4), C * T(0.04 + 0.01 * math.sin(a), 0.0272, -0.01 * math.cos(a)))
    for (dx, dz) in ((0, -0.009), (0, 0.009), (-0.009, 0), (0.009, 0)):
        sc.add(sd_box(0.003, 0.0025, 0.003, 0.0012), Mat("#3a3a3e"), C * T(-0.04 + dx, 0.0225, dz))
    chain(sc, bezier((0.02, 0.012, 0.168), (0.02, 0.01, 0.13), (-0.03, 0.01, 0.13), (-0.03, 0.02, 0.095), 14), [0.002] * 14, Mat("#5a5a5e"), X)


def _gb_face(d, W, H, img):
    """The Game Boy's front, as in the photographs: the grey frame round the screen
    (its bottom-right corner rounded) with the magenta and blue lines and DOT MATRIX
    WITH STEREO SOUND, the battery lamp; Nintendo GAME BOY below; B, A, SELECT,
    START in blue; the six speaker slots; OFF·ON along the top."""
    from replica import font
    blue, mag = "#2a2a8c", "#a8205a"
    d.rectangle([0, 0, W, H], fill="#c9c8c2")
    d.line([(0, H * 0.07), (W, H * 0.07)], fill="#b4b3ad", width=max(2, H // 300))            # the top ridge
    d.text((W * 0.1, H * 0.025), "◁OFF•ON▷", font=font("DejaVuSansMono.ttf", int(H * 0.02)), fill="#a9a8a2")
    x0, y0, x1, y1 = W * 0.08, H * 0.1, W * 0.86, H * 0.47
    d.rounded_rectangle([x0, y0, x1, y1 - W * 0.1], radius=int(W * 0.03), fill="#86868f")
    d.rounded_rectangle([x0, y0 + W * 0.05, x1 - W * 0.1, y1], radius=int(W * 0.03), fill="#86868f")
    d.pieslice([x1 - W * 0.22, y1 - W * 0.22, x1, y1], 0, 90, fill="#86868f")              # the round corner
    for k, c in enumerate((mag, blue)):                                                       # the two lines each side of the text
        yy = H * (0.122 + k * 0.008)
        d.line([(W * 0.12, yy), (W * 0.32, yy)], fill=c, width=max(2, H // 400))
        d.line([(W * 0.73, yy + H * 0.008), (W * 0.83, yy + H * 0.008)], fill=c, width=max(2, H // 400))
    d.text((W * 0.34, H * 0.117), "DOT MATRIX WITH STEREO SOUND", font=font("Lato-Regular.ttf", int(H * 0.016)), fill="#e8e8f0")
    d.ellipse([W * 0.14, H * 0.235, W * 0.16, H * 0.245], fill="#1a1a1a")
    d.text((W * 0.1, H * 0.265), "BATTERY", font=font("Lato-Regular.ttf", int(H * 0.014)), fill="#e8e8f0")
    d.text((W * 0.08, H * 0.495), "Nintendo", font=font("Lato-Black.ttf", int(H * 0.03)), fill=blue)
    d.text((W * 0.36, H * 0.484), "GAME BOY", font=font("Lato-Bold.ttf", int(H * 0.045)), fill=blue)
    d.text((W * 0.63, H * 0.73), "B", font=font("Lato-Black.ttf", int(H * 0.022)), fill=blue)
    d.text((W * 0.84, H * 0.705), "A", font=font("Lato-Black.ttf", int(H * 0.022)), fill=blue)
    d.text((W * 0.27, H * 0.845), "SELECT", font=font("Lato-Black.ttf", int(H * 0.018)), fill=blue)
    d.text((W * 0.45, H * 0.84), "START", font=font("Lato-Black.ttf", int(H * 0.018)), fill=blue)
    for i in range(6):                                                                        # the speaker slots, slanted
        x = W * (0.56 + i * 0.065)
        d.line([(x, H * 0.955 - i * H * 0.012), (x + W * 0.04, H * 0.885 - i * H * 0.012)], fill="#4a4a48", width=max(3, int(W * 0.016)))


def gameboy(sc, X):
    """The Nintendo Game Boy (1989) as in the photographs: the light grey slab (90 ×
    148 × 32 mm) with its big round bottom-right corner, the screen in its grey frame,
    the black cross, the magenta B and A, select and start, the speaker slots;
    standing on a small stand."""
    from replica import decal, block
    grey = plastic("#c9c8c2")
    B = X * T(0, 0.077, -0.005) * RX(-0.2)
    # the slab: its bottom-right corner cut and rounded (a soft chamfer)
    sc.add(block(0.045, 0.074, 0.016, 0.005, planes=[(1, -1, 0, 0.088)], soft=0.022, y0=False), grey, B)
    face = B * T(0, 0, 0.0162)
    sc.add(block(0.044, 0.073, 0.0006, 0.0004, planes=[(1, -1, 0, 0.0875)], soft=0.022, y0=False), decal(0.044, 0.073, _gb_face, base="#c9c8c2", px=1536, axes=(0, 1), flip_v=True), face)
    sc.add(sd_box(0.021, 0.019, 0.0008, 0.001), decal(0.021, 0.019, lambda d, W, H, img: _gb_screen(d, W, H), base="#9bbc0f", px=256, axes=(0, 1), flip_v=True,
           emissive_k=0.45), face * T(-0.003, 0.03, 0.0006))
    sc.add(sd_box(0.0095, 0.003, 0.0025, 0.0012), Mat("#141414", rough=0.35), face * T(-0.026, -0.022, 0.002))
    sc.add(sd_box(0.003, 0.0095, 0.0025, 0.0012), Mat("#141414", rough=0.35), face * T(-0.026, -0.022, 0.002))
    for i in range(2):
        sc.add(sd_cylinder(0.0025, 0.0048, 0.0015), Mat("#a8205a", rough=0.25, clearcoat=0.9), face * T(0.012 + i * 0.017, -0.028 + i * 0.006, 0.002) * RX(math.pi / 2))
        sc.add(sd_capsule((-0.011 + i * 0.015 - 0.004, -0.0485 + i * 0.0005, 0.001), (-0.011 + i * 0.015 + 0.004, -0.046 + i * 0.0005, 0.001), 0.0018), Mat("#8f8f95"), face)
    sc.add(block(0.05, 0.006, 0.03, 0.004), plastic("#2a2a2c"), X)                                                 # the stand


def _gb_screen(d, W, H):
    shades = ["#9bbc0f", "#8bac0f", "#306230", "#0f380f"]
    d.rectangle([0, 0, W, H], fill=shades[0])
    for y in range(18):
        for x in range(20):
            k = 2 if y > 13 else (1 if (x + y) % 7 == 0 else 0)
            if k:
                d.rectangle([x * W / 20, y * H / 18, (x + 1) * W / 20, (y + 1) * H / 18], fill=shades[k + (1 if y > 15 else 0)])
