"""
replica.py — tools for exact replicas of real machines (retro.py, the museum of
cpu_scene.py): keyboards laid out key by key, and printed marks (brand names,
logos, key legends) as textures on a part.

  keyboard(rows, ...)   a whole keyboard in ONE primitive (fast to render
                        however many keys): its layout is drawn once into a 2D
                        distance field (every key a rounded rectangle, its real
                        width in key units), then extruded to the keycaps'
                        height; with it, a texture of the key tops (each key's
                        colour, its legend printed on it)
  decal(...)            a material whose albedo is an image (text drawn with a
                        font, or shapes) over a box face, the rest a plain colour

Units: metres; one key unit = 19 mm (the pitch of every real keyboard).
A row is (z, x0, [key, ...]) in key units, z growing towards the user;
a key is (width, colour, legend[, depth[, shear]]) — width in units (1 = a
letter key, 6.25 = a typical space bar), colour a hex string, legend "" for
none, depth in rows (2: a key two rows tall, as an Enter), shear: the key
slanted into a parallelogram (its back edge moved right by shear units: the
Atari ST's function keys). A width-only key leaves a gap (None colour).
"""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from render import Mat, srgb, DEV, F

U = 0.019                    # a key unit (m)
FONTS = "/home/elucterio/Yolah/app/assets/fonts/"


def _lin(img):
    """PIL RGB → linear float tensor (1, 3, H, W) on the device (made once, at build time)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return torch.tensor(a, device=DEV, dtype=F).permute(2, 0, 1)[None].contiguous()


def _sample(tex, u, v):
    """Bilinear lookup of a (1, C, H, W) texture at u, v in 0..1 → (N, C)."""
    g = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1)[None, :, None, :]
    return torch.nn.functional.grid_sample(tex, g, align_corners=False, padding_mode="border")[0, :, :, 0].T


def keyboard(rows, cap=0.0075, height=0.008, gap=0.0016, round_=0.0025, px_per_unit=48, legend_col="#e8e4d8",
             font="Lato-Bold.ttf", legend_size=0.32, dish=True):
    """rows → (sdf, mat, (w, d)): the keycaps of a whole keyboard as one SDF in its
    local frame (x right, y up from the key bases, z towards the user; the layout
    centred), the material printing each key's colour and legend, and the
    layout's half sizes (m). cap: the top's inset (keys taper), height: keycap
    height, gap: between neighbouring keys."""
    keys = []                                    # (x0, z0, x1, z1, colour, legend) in units
    for (z, x0, ks) in rows:
        x = x0
        for k in ks:
            k = tuple(k) if isinstance(k, tuple) else (k, None)
            w, c, lab, dep, sh = k + ("", 1.0, 0.0)[len(k) - 2:] if len(k) >= 2 else (k[0], None, "", 1.0, 0.0)
            if c is not None:
                keys.append((x, z, x + w, z + dep, c, lab or "", sh))
            x += w
    xs0 = min(k[0] for k in keys); xs1 = max(k[2] for k in keys)
    zs0 = min(k[1] for k in keys); zs1 = max(k[3] for k in keys)
    cx, cz = (xs0 + xs1) / 2, (zs0 + zs1) / 2
    hw, hd = (xs1 - xs0) / 2 * U, (zs1 - zs0) / 2 * U
    pad = 0.5                                     # units of margin round the layout
    W = int((xs1 - xs0 + 2 * pad) * px_per_unit); H = int((zs1 - zs0 + 2 * pad) * px_per_unit)
    gx = (np.arange(W) + 0.5) / px_per_unit + xs0 - pad      # cell centres, units
    gz = (np.arange(H) + 0.5) / px_per_unit + zs0 - pad
    X, Z = np.meshgrid(gx, gz)
    d2 = np.full((H, W), 1e3, np.float32)
    g2, r2 = gap / 2 / U, round_ / U
    for (a, b, c_, d_, _, _, sh) in keys:          # rounded rectangle (sheared: a parallelogram), in units
        Xs = X + sh * (Z - (b + d_) / 2) if sh else X
        qx = np.abs(Xs - (a + c_) / 2) - ((c_ - a) / 2 - g2 - r2)
        qz = np.abs(Z - (b + d_) / 2) - ((d_ - b) / 2 - g2 - r2)
        dd = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - r2
        d2 = np.minimum(d2, dd)
    sdf_tex = torch.tensor(d2 * U, device=DEV, dtype=F)[None, None]
    # the key tops: colours and legends
    img = Image.new("RGB", (W, H), (20, 20, 20))
    dr = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(FONTS + font, max(6, int(px_per_unit * legend_size)))
    small = ImageFont.truetype(FONTS + font, max(5, int(px_per_unit * legend_size * 0.7)))
    lc = legend_col
    for (a, b, c_, d_, col, lab, sh) in keys:
        X0 = (a - xs0 + pad) * px_per_unit; Z0 = (b - zs0 + pad) * px_per_unit
        X1 = (c_ - xs0 + pad) * px_per_unit; Z1 = (d_ - zs0 + pad) * px_per_unit
        S = sh * px_per_unit * (Z1 - Z0) / px_per_unit / 2
        dr.polygon([(X0 - S, Z0), (X1 - S, Z0), (X1 + S, Z1), (X0 + S, Z1)], fill=col)
        if lab:
            f_ = fnt if len(lab) <= 2 else small
            lcol = lab_col = lc
            if "|" in lab:                         # "legend|colour": a legend of another colour
                lab, lab_col = lab.split("|")
            dr.text((X0 + px_per_unit * 0.18, Z0 + px_per_unit * 0.12), lab, font=f_, fill=lab_col)
    col_tex = _lin(img)
    ext_w, ext_d = (xs1 - xs0 + 2 * pad) * U, (zs1 - zs0 + 2 * pad) * U
    ox, oz = (xs0 - pad - cx) * U, (zs0 - pad - cz) * U        # the texture's corner in the local frame

    def to_uv(p):
        return (p[:, 0] - ox) / ext_w, (p[:, 2] - oz) / ext_d

    def sdf(p):
        u, v = to_uv(p)
        d = _sample(sdf_tex, u.clamp(0, 1), v.clamp(0, 1))[:, 0]
        # beyond the texture: the distance grows with the distance to it
        out = torch.maximum((u - 0.5).abs() - 0.5, (v - 0.5).abs() - 0.5).clamp_min(0) * max(ext_w, ext_d)
        y = p[:, 1] - height / 2
        # the taper: the key's outline shrinks by `cap` from base to top
        d = d + cap * (y / height + 0.5).clamp(0, 1)
        if dish:   # a faint hollow in each top: the light catches the keys' edges
            pass
        wx = torch.stack([d + out, y.abs() - height / 2], dim=-1)
        return torch.linalg.norm(wx.clamp_min(0), dim=-1) + wx.max(dim=-1).values.clamp_max(0)

    def pat(pl, pw, n):
        u, v = to_uv(pl)
        c = _sample(col_tex, u.clamp(0, 1), v.clamp(0, 1))
        side = (n[:, 1] < 0.7).float()[:, None]           # the keys' sides darker than their tops
        return {"albedo": c * (1 - 0.35 * side)}
    return sdf, Mat("#3a3632", rough=0.45, clearcoat=0.25, pattern=pat), (hw, hd)


def decal(w, h, draw_fn, base="#cccccc", px=1024, rough=0.45, clearcoat=0.25, metal=0.0, axes=(0, 2), flip_v=False,
          emissive_k=0.0, flip_u=False):
    """A material printing an image over a rectangle of half sizes w, h of the
    local plane given by axes (0, 2: the top face xz; 0, 1: a front face xy),
    centred on the local origin; outside it, the base colour. draw_fn(draw, W, H)
    draws on a PIL image (filled with base) — text, logos, stripes."""
    H = int(px * h / w) if w >= h else px
    W = px if w >= h else int(px * w / h)
    img = Image.new("RGB", (W, H), base)
    draw_fn(ImageDraw.Draw(img), W, H, img)
    tex = _lin(img)
    b = srgb(base)
    a0, a1 = axes

    def pat(pl, pw, n):
        u = pl[:, a0] / (2 * w) + 0.5
        v = pl[:, a1] / (2 * h) + 0.5
        if flip_v:
            v = 1 - v
        if flip_u:
            u = 1 - u
        inside = ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).float()[:, None]
        c = _sample(tex, u.clamp(0, 1), v.clamp(0, 1)) * inside + b * (1 - inside)
        out = {"albedo": c}
        if emissive_k > 0:
            out["emissive"] = c * inside * emissive_k
        return out
    return Mat(base, rough=rough, clearcoat=clearcoat, metal=metal, pattern=pat)


def font(name, size):
    return ImageFont.truetype(FONTS + name, size)


def rainbow_apple(d, cx, cy, h):
    """The six-stripe apple (1977-1998) drawn on a PIL image, centred at (cx, cy),
    h tall: two lobes, a bite on the right, the leaf above, the stripes green,
    yellow, orange, red, purple, blue from top to bottom."""
    from PIL import Image as _I, ImageDraw as _D
    S = int(h * 1.2) + 4
    m = _I.new("L", (S, S), 0)
    md = _D.Draw(m)
    r = h * 0.29
    by = S * 0.58
    md.ellipse([S / 2 - r * 1.75, by - r * 1.3, S / 2 + r * 0.2, by + r * 1.45], fill=255)       # the left lobe
    md.ellipse([S / 2 - r * 0.2, by - r * 1.3, S / 2 + r * 1.75, by + r * 1.45], fill=255)       # the right lobe
    md.ellipse([S / 2 - r * 0.5, by - r * 1.55, S / 2 + r * 0.5, by - r * 0.9], fill=0)           # the dip at the top
    md.ellipse([S / 2 + r * 1.15, by - r * 0.7, S / 2 + r * 2.2, by + r * 0.45], fill=0)          # the bite
    md.ellipse([S / 2 - r * 0.15, by - r * 2.6, S / 2 + r * 0.75, by - r * 1.6], fill=255)        # the leaf
    stripes = ["#61bb46", "#fdb827", "#f5821f", "#e03a3e", "#963d97", "#009ddc"]
    col = _I.new("RGB", (S, S))
    cd = _D.Draw(col)
    top, bot = by - r * 2.6, by + r * 1.45
    for i, c in enumerate(stripes):
        y0 = top + (bot - top) * (i / 6) * (1.0 if i else 0.0) if i else 0
        cd.rectangle([0, top + (bot - top) * i / 6 if i else 0, S, top + (bot - top) * (i + 1) / 6], fill=c)
    cd.rectangle([0, top + (bot - top) * 5 / 6, S, S], fill=stripes[5])
    img = d._image
    img.paste(col, (int(cx - S / 2), int(cy - S / 2)), m)


def block(hx, hy, hz, r=0.004, planes=(), soft=0.0, y0=True):
    """A case: a box of half sizes hx, hy, hz with rounded edges (r), its foot at
    y = 0 (y0) or centred, cut by half-spaces: planes = [(nx, ny, nz, c), ...]
    keeps n·p <= c (n normalised here): slopes, chamfers. soft > 0 rounds the
    cut edges (smooth max)."""
    import math as _m
    ps = []
    for (a, b, c, k) in planes:
        L = _m.sqrt(a * a + b * b + c * c)
        ps.append((a / L, b / L, c / L, k / L))
    oy = hy if y0 else 0.0

    def f(p):
        x, y, z = p[:, 0], p[:, 1] - oy, p[:, 2]
        q = torch.stack([x.abs() - hx + r, y.abs() - hy + r, z.abs() - hz + r], dim=-1)
        d = torch.linalg.norm(q.clamp_min(0), dim=-1) + q.max(dim=-1).values.clamp_max(0) - r
        for (a, b, c, k) in ps:
            e = a * p[:, 0] + b * p[:, 1] + c * p[:, 2] - k
            if soft > 0:
                h = (0.5 + 0.5 * (e - d) / soft).clamp(0, 1)
                d = e * h + d * (1 - h) + soft * h * (1 - h)
            else:
                d = torch.maximum(d, e)
        return d
    return f


SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def ibm_logo(d, x, y, h, fill, bg, bars=8):
    """The striped IBM logo: heavy slab-serif letters cut by horizontal bars of the
    background (8 bars), x, y its top-left corner, h its height."""
    f = ImageFont.truetype(SERIF_BOLD, int(h * 1.25))
    d.text((x, y - h * 0.22), "IBM", font=f, fill=fill)
    w = d.textlength("IBM", font=f)
    for i in range(bars - 1):
        yy = y + h * (i + 1) / bars - h * 0.035
        d.rectangle([x - 2, yy, x + w + 2, yy + h * 0.06], fill=bg)
    return w
