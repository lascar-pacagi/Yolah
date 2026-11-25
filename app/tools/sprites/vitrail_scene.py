"""
vitrail_scene.py — the Vitraux's background in 3D (scenery.py): inside a great
gothic cathedral, seen from high above the choir down the length of the nave
to the west rose (inspired by the great French cathedrals, invented here).

  the elevation   three storeys on each side: the great arcade (round pillars
                  with foliage capitals, pointed arches opening on the dark
                  aisles), the gallery (paired arches on colonnettes behind a
                  balustrade), the clerestory (two lancets and an oculus of
                  stained glass per bay, glowing)
  the vault       quadripartite rib vaults: the webs as two crossing pointed
                  barrels, transverse and diagonal ribs, a gilt boss at each
                  crossing; vaulting shafts running down to the capitals
  the floor       the choir's black and white marble (squares and a central
                  star), the carved oak stalls along it, the altar on its steps
                  with candlesticks; the nave paved in stone, rows of chairs on
                  either side of the central aisle, chandeliers hanging low
  the west end    the great rose high on the west wall, the organ case under it
  the light       the windows glowing in colour, warm chandeliers and candles,
                  a faint haze catching it all
  the camera      high over the choir, looking down the nave; the lens shifted
                  so the rose and the far nave fall in the app's free strip
                  between the board and the side panel (as cpu_scene.py)

Every repeated element (pillars, arches, windows, ribs, chairs, balusters...)
is ONE SDF using domain repetition along the bays: the detail costs little.

    cd app/tools/sprites && python3 vitrail_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_box, sd_cylinder, sd_sphere, sd_capsule, sd_torus, sd_cone, fbm
from scenery import Camera, render_view, Volume
from vitraux import stained, rose_glass
from models import glow

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
W = 6.0                  # half the nave's width (pillar centres at x = ±W)
L = 6.0                  # a bay's length
NB = 8                   # bays: pillars at z = 0, -6, ..., -48
ZW = -NB * L             # the west wall
ZE = 14.0                # the east end (behind the camera)
NE = 2                   # bays continue east of z = 0 up to the pillar at z = NE·L (past the camera)
YS = 33.0                # the vault's springing line


# ─── little SDF helpers (all torch, (N,) tensors) ───────────────────────────
def _arch(u, y, w, ys, a, y0):
    """A pointed arch opening in the plane (u across, y up): width 2w, springing at
    ys, its two arcs centred a off the axis (apex at ys + sqrt((w+a)² − a²)),
    bottom at y0. Negative inside the opening."""
    au = u.abs()
    low = au - w
    high = torch.sqrt((au + a) ** 2 + (y - ys).clamp_min(0) ** 2) - (w + a)
    d = torch.where(y < ys, low, high)
    return torch.maximum(d, y0 - y)


def _cell(z, lo, hi, step=L):
    """Domain repetition along z: the index of the nearest repeat (clamped to lo..hi) and the local offset."""
    k = torch.clamp(torch.round(z / step), lo, hi)
    return k, z - k * step


def _box2(a, b, ha, hb):
    qa, qb = a.abs() - ha, b.abs() - hb
    return torch.sqrt(qa.clamp_min(0) ** 2 + qb.clamp_min(0) ** 2) + torch.maximum(qa, qb).clamp_max(0)


# ─── materials ──────────────────────────────────────────────────────────────
def stone(c="#b4a894", k=0.75, s=0.9):
    def pat(pl, pw, n):
        # ashlar: courses of blocks, each its own tone, fine mortar joints
        y, h = pw[:, 1], pw[:, 0] + pw[:, 2]
        row = torch.floor(y / 0.36)
        jy = (torch.remainder(y, 0.36) < 0.015).float()
        jx = (torch.remainder(h + 0.38 * torch.remainder(row, 2), 0.76) < 0.015).float()
        blk = torch.floor((h + 0.38 * torch.remainder(row, 2)) / 0.76) * 3.7 + row * 1.9
        tone = 0.92 + 0.1 * torch.frac(torch.sin(blk) * 4375.5)
        grain = (0.86 + 0.22 * fbm(pw * vec(1.5, 1.5, 1.5), 3)) * (0.9 + 0.18 * fbm(pw * vec(7.0, 7.0, 7.0), 2))
        soot = 1 - 0.3 * ((y - 8.0) / 30.0).clamp(0, 1) * (0.6 + 0.6 * fbm(pw * vec(0.4, 0.15, 0.4), 2))   # darker up in the vault
        c_ = srgb(c) * (tone * grain * soot)[:, None] * (1 - 0.2 * torch.maximum(jx, jy) * s)[:, None]
        return {"albedo": c_}
    return Mat(c, rough=k, pattern=pat)


OAK = Mat("#5a3a20", rough=0.45, clearcoat=0.4,
          pattern=lambda pl, pw, n: {"albedo": srgb("#5a3a20") * ((0.9 + 0.1 * torch.sin(pw[:, 1] * 22 + 9 * fbm(pw * vec(1.5, 6, 1.5), 3)))
                                                          * (0.85 + 0.25 * fbm(pw * vec(0.8, 0.8, 0.8), 2)))[:, None]})
GILT = Mat("#d8b050", metal=0.95, rough=0.25)


# ─── the structure ──────────────────────────────────────────────────────────
def walls():
    """Each side wall (x from W to W+1), pierced by the three storeys' openings."""
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        kb, zb = _cell(z + L / 2, -NB + 1, NE)                                    # bay centres (between pillars)
        slab = torch.maximum((x - (W + 0.5)).abs() - 0.5, (z - (ZW + ZE) / 2).abs() - (ZE - ZW) / 2)
        slab = torch.maximum(slab, y - YS - 2.0)
        arcade = _arch(zb, y, 2.35, 9.6, 1.0, -1.0)                               # the great arcade
        gallery = torch.minimum(_arch(zb - 1.2, y, 0.95, 18.2, 0.35, 16.6), _arch(zb + 1.2, y, 0.95, 18.2, 0.35, 16.6))
        clere = torch.minimum(_arch(zb - 1.1, y, 0.8, 29.5, 0.5, 23.6), _arch(zb + 1.1, y, 0.8, 29.5, 0.5, 23.6))
        ocul = torch.sqrt(zb ** 2 + (y - 31.9) ** 2) - 0.75
        hole = torch.minimum(torch.minimum(arcade, gallery), torch.minimum(clere, ocul))
        hole = torch.maximum(hole, (z - ZW).abs() - 1e9)                         # (all bays)
        inbay = (z < NE * L) & (z > ZW)
        hole = torch.where(inbay, hole, torch.full_like(hole, 10.0))
        d = torch.maximum(slab, -hole)
        # string courses: mouldings along the wall at the storeys
        for yy in (15.6, 23.3):
            d = torch.minimum(d, _box2(x - W + 0.05, y - yy, 0.18, 0.12) + 0.0 * z)
        return d
    return f


def pillars():
    """The round pillars with their capitals and bases, and the vaulting shafts rising
    from the capitals to the springing of the vault."""
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        k, zl = _cell(z, -NB, NE)
        r = torch.sqrt((x - W) ** 2 + zl ** 2)
        shaft = torch.maximum(r - 0.75, (y - 4.6).abs() - 4.6)
        base = torch.maximum(_box2(x - W, zl, 1.0, 1.0), y - 0.6)
        capr = 0.75 + 0.55 * ((y - 8.9) / 0.9).clamp(0, 1)                        # the capital flares to an abacus
        cap = torch.maximum(r - capr, (y - 9.35).abs() - 0.45)
        abacus = torch.maximum(_box2(x - W, zl, 1.4, 1.4), (y - 9.85).abs() - 0.12)
        vs = torch.maximum(torch.sqrt((x - (W - 0.25)) ** 2 + zl ** 2) - 0.22, (y - (YS + 9.9) / 2).abs() - (YS - 9.9) / 2)
        ring = torch.maximum(torch.sqrt((x - (W - 0.25)) ** 2 + zl ** 2) - 0.32, torch.minimum((y - 16.4).abs(), (y - 23.4).abs()) - 0.1)
        return torch.minimum(torch.minimum(torch.minimum(shaft, base), torch.minimum(cap, abacus)), torch.minimum(vs, ring))
    return f


def colonnettes():
    """The gallery's colonnettes (between and beside its paired arches) and the
    clerestory's mullions, set in the wall's openings."""
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        kb, zb = _cell(z + L / 2, -NB + 1, NE)
        d = torch.full_like(y, 10.0)
        for c in (-2.25, 0.0, 2.25):
            col = torch.maximum(torch.sqrt((x - (W + 0.5)) ** 2 + (zb - c) ** 2) - 0.13, (y - 17.6).abs() - 1.1)
            d = torch.minimum(d, col)
        mull = torch.maximum(_box2(x - (W + 0.5), zb, 0.18, 0.14), (y - 27.2).abs() - 3.8)
        return torch.minimum(d, mull)
    return f


def balustrade():
    """The gallery's balustrade: a rail on small balusters along each side."""
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        inn = torch.maximum(z - NE * L, ZW - z)
        rail = torch.maximum(_box2(x - (W - 0.15), y - 17.5, 0.12, 0.08), inn)
        kb, zb = _cell(z, -int(-ZW / 0.35), 0, 0.35)
        bal = torch.maximum(torch.maximum(torch.sqrt((x - (W - 0.15)) ** 2 + zb ** 2) - 0.05, (y - 16.95).abs() - 0.5), inn)
        return torch.minimum(rail, bal)
    return f


def vault():
    """The rib vault: the webs (two crossing pointed barrels: their cavities' union is
    the air), the transverse ribs over the pillars and the diagonal ribs along
    the groins, a boss at each crossing."""
    a1 = 2.0
    R1 = W + a1
    hgt = math.sqrt(R1 ** 2 - a1 ** 2)
    a2 = ((hgt ** 2) - (L / 2) ** 2) / (2 * (L / 2))
    R2 = L / 2 + a2

    def cav(x, y, z):
        k, zb = _cell(z + L / 2, -NB + 1, NE)
        c1 = R1 - torch.sqrt((x.abs() + a1) ** 2 + (y - YS).clamp_min(0) ** 2)
        c2 = R2 - torch.sqrt((zb.abs() + a2) ** 2 + (y - YS).clamp_min(0) ** 2)
        return c1, c2, zb

    def webs(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        c1, c2, zb = cav(x, y, z)
        air = torch.maximum(c1, c2)
        return torch.maximum(torch.maximum(air, YS - 0.5 - y), torch.maximum(x.abs() - W - 1.0, torch.maximum(z - ZE, ZW - z)))

    def ribs(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        c1, c2, zb = cav(x, y, z)
        up = (y > YS - 0.3)
        k, zl = _cell(z, -NB, NE)
        trans = torch.sqrt(c1 ** 2 + zl ** 2) - 0.28                              # over the pillars: the transverse ribs
        diag = torch.sqrt(c1 ** 2 + c2 ** 2) * 0.7 - 0.22                          # along the groins
        boss = torch.sqrt(x ** 2 + (y - (YS + hgt)) ** 2 + zb ** 2) - 0.45
        d = torch.minimum(torch.minimum(trans, diag), boss)
        return torch.where(up, d, torch.full_like(d, 5.0))
    return webs, ribs, hgt


def clerestory_glass():
    """The glass in the clerestory's lancets and oculi (a thin pane in the opening)."""
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        kb, zb = _cell(z + L / 2, -NB + 1, NE)
        clere = torch.minimum(_arch(zb - 1.1, y, 0.8, 29.5, 0.5, 23.6), _arch(zb + 1.1, y, 0.8, 29.5, 0.5, 23.6))
        ocul = torch.sqrt(zb ** 2 + (y - 31.9) ** 2) - 0.75
        pane = torch.maximum(torch.minimum(clere, ocul), (x - (W + 0.85)).abs() - 0.03)
        return torch.maximum(pane, torch.maximum(z - NE * L, ZW - z))
    return f


def aisle():
    """The aisles behind the arcade: their low vault (dark), the outer wall and its
    windows glowing dimly."""
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        ceil = torch.maximum(13.0 - y, W + 0.9 - x)
        # the outer wall, its windows cut 0.5 m deep into it (the glass sits at the back of the opening)
        kb, zb = _cell(z + L / 2, -NB + 1, NE)
        win = torch.maximum(_arch(zb, y, 1.0, 8.5, 0.4, 3.0), torch.maximum(z - NE * L, ZW - z))
        outer = torch.maximum(13.5 - x, -torch.maximum(win, x - 14.0))
        return torch.minimum(ceil, outer)
    return f


def aisle_glass():
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        kb, zb = _cell(z + L / 2, -NB + 1, NE)
        win = _arch(zb, y, 1.0, 8.5, 0.4, 3.0)
        return torch.maximum(torch.maximum(win, (x - 13.95).abs() - 0.03), torch.maximum(z - NE * L, ZW - z))
    return f


# ─── the scene ──────────────────────────────────────────────────────────────
def pews():
    """Rows of oak pews on either side of the central aisle: seat, back, and the
    high end panels with a rounded top facing the aisle."""
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        ax = x.abs()
        kz, zl = _cell(z - 2.0, -26, 0, 1.15)
        inx = torch.maximum(1.5 - ax, ax - 5.4)
        seat = torch.maximum(torch.maximum(inx, (y - 0.46).abs() - 0.04), (zl - 0.1).abs() - 0.22)
        back = torch.maximum(torch.maximum(inx, (y - 0.78).abs() - 0.34), (zl + 0.16).abs() - 0.04)
        kneel = torch.maximum(torch.maximum(inx, (y - 0.12).abs() - 0.04), (zl - 0.42).abs() - 0.06)
        end = torch.maximum(torch.maximum((ax - 1.5).abs() - 0.05, torch.maximum(y - 1.0 - 0.18 * torch.cos((zl + 0.05) * 3.0), -y)), (zl + 0.02).abs() - 0.32)
        d = torch.minimum(torch.minimum(seat, back), torch.minimum(kneel, end))
        return torch.maximum(d, torch.maximum(z - 3.0, (ZW + 17.0) - z))
    return f


def flag_floor():
    """Worn stone flags of uneven sizes, cracked, polished where people walk, a
    little damp: the light pools on them."""
    def pat(pl, pw, n):
        x, z = pw[:, 0], pw[:, 2]
        row = torch.floor(z / 1.1)
        off = torch.frac(torch.sin(row * 7.13) * 437.5) * 1.3
        w = 0.9 + 0.7 * torch.frac(torch.sin(torch.floor((x + off) / 1.3) * 3.1 + row * 1.7) * 4375.5)
        gx = torch.remainder(x + off, 1.3)
        gz = torch.remainder(z, 1.1)
        joint = ((gx < 0.03) | (gz < 0.03)).float()
        tile = torch.floor((x + off) / 1.3) * 3.1 + row * 7.7
        tone = 0.7 + 0.35 * torch.frac(torch.sin(tile) * 4375.5)
        crack = 0.5 * (torch.abs(fbm(pw * vec(1.8, 1.8, 1.8), 3) - 0.5) < 0.005).float()
        wear = torch.exp(-(x / 1.3) ** 2)                                            # the aisle polished by feet
        c = srgb("#8a8274") * (tone * (0.8 + 0.3 * fbm(pw * vec(1.1, 1.1, 1.1), 4)))[:, None] * (1 - 0.55 * torch.maximum(joint, crack))[:, None]
        rough = 0.55 - 0.35 * wear + 0.3 * joint
        # the choir (beyond the last pews): black and white marble set diagonally, veined, polished
        u, v = (x + z) * 0.7071 / 0.9, (x - z) * 0.7071 / 0.9
        chk = torch.remainder(torch.floor(u) + torch.floor(v), 2)
        vein = 0.8 + 0.25 * torch.sin(pw[:, 0] * 3.0 + 9.0 * fbm(pw * vec(0.9, 0.9, 0.9), 4))
        mc = torch.where(chk[:, None] > 0.5, srgb("#e8e2d6") * vein[:, None], srgb("#16130f") * (0.8 + 0.4 * vein)[:, None])
        mj = ((torch.frac(u) < 0.02) | (torch.frac(v) < 0.02)).float()
        border = ((z - (ZW + 16.6)).abs() < 0.25).float()                             # a gilt-brass strip at the choir's edge
        mc = torch.where(border[:, None] > 0.5, srgb("#b08a40"), mc * (1 - 0.4 * mj)[:, None])
        choir = (z < ZW + 16.85).float()[:, None]
        c = c * (1 - choir) + mc * choir
        rough = torch.where(choir[:, 0] > 0.5, torch.full_like(rough, 0.12), rough)
        return {"albedo": c, "rough": rough}
    return Mat("#8a8274", rough=0.4, clearcoat=0.5, pattern=pat)


def _sector(a, b, n):
    """Angular repetition in a plane (a, b): the coordinates rotated into the nearest of n sectors."""
    ang = torch.atan2(b, a)
    k = torch.round(ang / (math.tau / n))
    t = ang - k * (math.tau / n)
    r = torch.sqrt(a ** 2 + b ** 2)
    return r * torch.cos(t), r * torch.sin(t)


def reredos(sc, Z):
    """The high altar and its carved stone reredos, in the church's own stone: a
    screen 5 m high across the choir, a row of pointed niches each with a statue,
    a taller central niche under a pointed gable for the main statue, mouldings
    and a cresting of small pinnacles; in front, the altar on three marble steps
    (a white cloth, an altar cross, six candles), two tall floor candlesticks,
    gilt rails; above it all a gilt gloria (a sunburst of rays round a glowing
    heart) hanging in front of the west lancets, so the screen stands against
    coloured light."""
    marble = Mat("#d8d0c2", rough=0.2, clearcoat=0.6)
    linen = Mat("#ece6da", rough=0.8, sss=0.3)
    for i, (hw, hh) in enumerate(((5.4, 0.15), (4.8, 0.3), (4.2, 0.45))):
        sc.add(sd_box(hw, hh, 2.4 - 0.4 * i, 0.02), marble, T(0, hh, Z + 2.5))
    # the altar: a stone block under a white cloth falling a little over the front and sides
    sc.add(sd_box(2.0, 0.5, 0.6, 0.04), stone("#a89c88", 0.4, 0.0), T(0, 1.4, Z + 2.2))
    sc.add(sd_box(2.1, 0.025, 0.66, 0.01), linen, T(0, 1.925, Z + 2.2))
    sc.add(sd_box(2.1, 0.16, 0.012, 0.005), linen, T(0, 1.77, Z + 2.86))
    sc.add(sd_box(2.1, 0.012, 0.014, 0.0), GILT, T(0, 1.62, Z + 2.875))                     # the cloth's gold border
    sc.add(sd_cone(0.06, 0.16, 0.1, 0.01), GILT, T(0, 2.01, Z + 2.0))                         # the altar cross
    sc.add(sd_box(0.03, 0.42, 0.03, 0.005), GILT, T(0, 2.48, Z + 2.0))
    sc.add(sd_box(0.2, 0.03, 0.03, 0.005), GILT, T(0, 2.65, Z + 2.0))

    def screen(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        zf = z - Z
        side = torch.maximum(torch.maximum(x.abs() - 5.0, (y - 3.55).abs() - 2.65), zf.abs() - 0.3)
        centre = torch.maximum(_arch(x, y, 1.15, 6.6, 0.55, 0.9), zf.abs() - 0.32)    # pointed gable over the centre
        d = torch.minimum(side, centre)
        # the niches, 0.28 deep: small ones along the screen, a tall one in the middle
        k, xl = _cell(x, -4, 4, 1.05)
        small = _arch(xl, y, 0.36, 4.5, 0.14, 2.4)
        small = torch.where(x.abs() > 1.4, small, torch.full_like(small, 10.0))
        big = _arch(x, y, 0.62, 5.9, 0.25, 2.4)
        d = torch.maximum(d, -torch.maximum(torch.minimum(small, big), 0.02 - zf))
        # mouldings: a base course, a string course under the niches, the cornice
        for (yy, hh, dd) in ((1.05, 0.15, 0.4), (2.3, 0.07, 0.36), (6.25, 0.1, 0.38)):
            band = torch.maximum(torch.maximum(x.abs() - 5.1, (y - yy).abs() - hh), zf.abs() - dd)
            if yy > 6:
                band = torch.maximum(band, 1.2 - x.abs())
            d = torch.minimum(d, band)
        # the cresting: a small pinnacle over each niche
        r = 0.09 * (1 - (y - 6.35) / 0.65).clamp(0, 1)
        pin = torch.maximum(torch.sqrt(xl ** 2 + zf ** 2) - r, (y - 6.68).abs() - 0.33) * 0.8
        pin = torch.maximum(pin, torch.maximum(1.4 - x.abs(), x.abs() - 4.8))
        return torch.minimum(d, pin)
    sc.add(screen, stone("#8e846f", 0.6, 0.15))

    def smin(a, b, k):
        h = (0.5 + 0.5 * (b - a) / k).clamp(0, 1)
        return b + (a - b) * h - k * h * (1 - h)

    def ellipsoid(x, y, z, rx, ry, rz):
        return (torch.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2) - 1) * min(rx, ry, rz)

    def figure(u, y, z, crowned):
        """A gothic saint 1.75 high, base at y = 0, facing +z (one unit = 1 m): a long
        robe falling in vertical folds and flaring at the hem, a mantle over the
        shoulders, the hands joined in prayer before the chest, a veil over the
        head and shoulders; `crowned` (a tensor of 0/1) adds a crown of points."""
        plinth = torch.maximum(_box2(u, z, 0.19, 0.13), (y - 0.05).abs() - 0.05)
        th = torch.atan2(u, z)
        t = ((y - 0.1) / 1.2).clamp(0, 1)                                           # 0 at the hem, 1 at the chest
        rr = 0.205 - 0.06 * t + 0.02 * torch.exp(-((y - 1.15) / 0.12) ** 2)
        folds = 0.014 * torch.sin(11 * th + 3 * y) * (1 - 0.8 * t)
        robe = torch.sqrt(u ** 2 + (z / 0.75) ** 2) * 0.8 - (rr + folds) * 0.8
        robe = torch.maximum(robe, torch.maximum(0.1 - y, y - 1.32))
        chest = ellipsoid(u, y - 1.22, z + 0.01, 0.19, 0.13, 0.11)
        d = smin(robe, chest, 0.06)
        d = torch.minimum(d, plinth)
        # the arms: upper arms down the sides, forearms meeting at the joined hands
        au = u.abs()
        def seg(px, py, pz, ax, ay, az, bx, by, bz, r):
            bax, bay, baz = bx - ax, by - ay, bz - az
            h = (((px - ax) * bax + (py - ay) * bay + (pz - az) * baz) / (bax * bax + bay * bay + baz * baz)).clamp(0, 1)
            return torch.sqrt((px - ax - bax * h) ** 2 + (py - ay - bay * h) ** 2 + (pz - az - baz * h) ** 2) - r
        upper = seg(au, y, z, 0.16, 1.27, 0.0, 0.15, 1.02, 0.05, 0.05)
        fore = seg(au, y, z, 0.15, 1.02, 0.05, 0.025, 1.14, 0.13, 0.042)
        hands = ellipsoid(u, y - 1.2, z - 0.14, 0.03, 0.08, 0.035)
        d = smin(d, smin(torch.minimum(upper, fore), hands, 0.02), 0.03)
        # the head, the neck, the veil falling behind onto the shoulders
        neck = seg(u, y, z, 0.0, 1.3, 0.0, 0.0, 1.42, 0.01, 0.045)
        head = ellipsoid(u, y - 1.51, z - 0.015, 0.078, 0.1, 0.088)
        nose = ellipsoid(u, y - 1.5, z - 0.095, 0.014, 0.03, 0.016)
        d = smin(d, smin(neck, torch.minimum(head, nose), 0.02), 0.02)
        veil = torch.abs(ellipsoid(u, y - 1.53, z + 0.005, 0.105, 0.125, 0.11)) - 0.012
        veil = torch.maximum(veil, torch.maximum(z - 0.03, 1.47 - y + 0.0 * u))
        drape = torch.maximum(ellipsoid(u, y - 1.33, z + 0.05, 0.2, 0.2, 0.09), torch.maximum(z - 0.0, y - 1.5))
        d = torch.minimum(d, torch.minimum(veil, smin(drape, chest, 0.03)))
        # the crown (the central statue): a band with eight points
        cu, cv = _sector(u, z + 0.01, 8)
        band = torch.maximum(torch.abs(torch.sqrt(u ** 2 + (z + 0.01) ** 2) - 0.095) - 0.012, (y - 1.66).abs() - 0.025)
        point = torch.maximum(torch.sqrt((cu - 0.095) ** 2 + cv ** 2) - 0.018 * (1 - (y - 1.68) / 0.07).clamp(0, 1), (y - 1.715).abs() - 0.035)
        crown = torch.minimum(band, point)
        crown = torch.where(crowned > 0.5, crown, torch.full_like(crown, 10.0))
        return torch.minimum(d, crown)

    def statues(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        k, xl = _cell(x, -4, 4, 1.05)
        sm = x.abs() > 1.4
        u = torch.where(sm, xl, x)
        sc_ = torch.where(sm, torch.ones_like(x), torch.full_like(x, 1.45))      # the central statue is larger
        zl = z - (Z + 0.1)
        return figure(u / sc_, (y - 2.4) / sc_, zl / sc_, (~sm).float()) * sc_
    sc.add(statues, stone("#b0a690", 0.55, 0.0))
    # the gloria: gilt rays round a glowing heart, hung in front of the central lancet
    G = (0.0, 16.6, Z - 3.0)
    def rays(p):
        a, b, c = p[:, 0] - G[0], p[:, 1] - G[1], p[:, 2] - G[2]
        u, v = _sector(a, b, 28)
        ang = torch.atan2(b, a)
        long_ = (torch.remainder(torch.round(ang / (math.tau / 28)), 2) < 0.5).float()
        rl = 1.6 + 1.2 * long_
        ray = torch.maximum(torch.maximum((u - (0.9 + rl) / 2).abs() - (rl - 0.9) / 2, v.abs() - 0.07 * (1.6 - u / 3.2).clamp_min(0.2)), c.abs() - 0.04)
        ring = torch.maximum((torch.sqrt(a ** 2 + b ** 2) - 0.95).abs() - 0.08, c.abs() - 0.08)
        return torch.minimum(ray, ring)
    sc.add(rays, Mat("#f0c060", metal=0.9, rough=0.2, emissive="#ffc860", emit=0.5))
    sc.add(sd_cylinder(0.05, 0.85), glow("#fff0c0", 6.0), T(*G) * RX(math.pi / 2))
    for x in (-1.6, -1.0, -0.5, 0.5, 1.0, 1.6):                                                     # the altar's candles
        sc.add(sd_cylinder(0.26, 0.035, 0.01), Mat("#f2ead8", rough=0.4, sss=0.5), T(x, 2.22, Z + 2.2))
        sc.add(sd_sphere(0.045), glow("#ffd890", 9.0), T(x, 2.53, Z + 2.2))
    for sx in (-1, 1):
        # the tall floor candlesticks on the steps
        C = T(sx * 3.4, 0.9, Z + 3.6)
        sc.add(sd_cone(0.3, 0.35, 0.12, 0.02), GILT, C * T(0, 0.3, 0))
        sc.add(sd_cylinder(1.1, 0.06), GILT, C * T(0, 1.6, 0))
        for yy in (0.9, 1.6, 2.2):
            sc.add(sd_sphere(0.12), GILT, C * T(0, yy, 0))
        sc.add(sd_cone(0.12, 0.08, 0.2, 0.01), GILT, C * T(0, 2.8, 0))
        sc.add(sd_cylinder(0.3, 0.06, 0.01), Mat("#f2ead8", rough=0.4, sss=0.5), C * T(0, 3.2, 0))
        sc.add(sd_sphere(0.07), glow("#ffd890", 10.0), C * T(0, 3.6, 0))
        # the gilt rails
        R = T(sx * 4.6, 0, Z + 5.4)
        sc.add(sd_box(1.2, 0.04, 0.04, 0.01), GILT, R * T(0, 1.0, 0))
        for k in range(7):
            sc.add(sd_box(0.025, 0.5, 0.025, 0.005), GILT, R * T(-1.1 + k * 0.37, 0.5, 0))


RY_ = 24.5          # the rose's centre height
RR = 5.4            # its radius
LANCETS = ((-3.3, 1.05, 11.6), (0.0, 1.2, 12.6), (3.3, 1.05, 11.6))   # the west lancets: centre x, half width, springing


def west_wall():
    """The west wall, 1.2 m thick: the rose cut out high up, three great lancets
    under it, and between them the gallery of kings: a row of pointed niches
    with a crowned statue in each, standing on a projecting ledge."""
    def holes(x, y):
        rose = torch.sqrt(x ** 2 + (y - RY_) ** 2) - RR
        lan = torch.full_like(x, 10.0)
        for (xc, w, ys) in LANCETS:
            lan = torch.minimum(lan, _arch(x - xc, y, w, ys, 0.45 * w, 2.6))
        return torch.minimum(rose, lan)

    def wall(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        d = torch.maximum(z - ZW, ZW - 1.2 - z)
        d = torch.maximum(d, -holes(x, y))
        # the gallery of kings: niches 0.45 deep
        k, xl = _cell(x, -5, 5, 1.05)
        niche = torch.maximum(torch.maximum(_arch(xl, y, 0.4, 17.6, 0.15, 15.0), ZW - 0.45 - z), x.abs() - 5.6)
        d = torch.maximum(d, -niche)
        ledge = torch.maximum(_box2(x, y - 14.85, 5.8, 0.15), (z - ZW - 0.1).abs() - 0.3)
        cornice = torch.maximum(_box2(x, y - 18.9, 5.8, 0.12), (z - ZW - 0.05).abs() - 0.25)
        return torch.minimum(torch.minimum(d, ledge), cornice)

    def kings(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        k, xl = _cell(x, -5, 5, 1.05)
        zl = z - (ZW - 0.2)
        body = torch.sqrt(xl ** 2 + zl ** 2) - (0.24 - 0.06 * ((y - 15.0) / 1.6).clamp(0, 1))
        body = torch.maximum(body, (y - 15.8).abs() - 0.8)
        head = torch.sqrt(xl ** 2 + (y - 16.82) ** 2 + zl ** 2) - 0.14
        crown = torch.maximum(torch.sqrt(xl ** 2 + zl ** 2) - 0.13, (y - 17.0).abs() - 0.07)
        return torch.minimum(torch.minimum(body, head), crown)

    def lancet_glass(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        lan = torch.full_like(x, 10.0)
        for (xc, w, ys) in LANCETS:
            lan = torch.minimum(lan, _arch(x - xc, y, w, ys, 0.45 * w, 2.6))
        return torch.maximum(lan, (z - (ZW - 0.6)).abs() - 0.03)

    def mullions(p):
        # each lancet split in two lights by a mullion, with a small oculus in its head; iron saddle bars across
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        d = torch.full_like(x, 10.0)
        for (xc, w, ys) in LANCETS:
            m = torch.maximum(torch.maximum((x - xc).abs() - 0.07, (y - (2.6 + ys) / 2).abs() - (ys - 2.6) / 2), (z - (ZW - 0.6)).abs() - 0.08)
            oc = torch.maximum((torch.sqrt((x - xc) ** 2 + (y - (ys + 1.1)) ** 2) - 0.42).abs() - 0.05, (z - (ZW - 0.6)).abs() - 0.08)
            kb, yb = _cell(y, 4, 15, 0.9)
            bar = torch.maximum(torch.maximum(yb.abs() - 0.025, (x - xc).abs() - w), torch.maximum((z - (ZW - 0.55)).abs() - 0.025, y - ys))
            d = torch.minimum(d, torch.minimum(torch.minimum(m, oc), bar))
        return d
    return wall, kings, lancet_glass, mullions


def chandeliers(n=4, yc=11.5, xc=3.2):
    """Gothic crown chandeliers hanging in pairs (x = ±xc) in the first n bays, off
    the nave's axis so they never hide the rose or the retable:
    two gilt hoops with candles round them, chains to a ring, a long chain up to
    the vault. Returns (metal, wax, flames) SDFs (domain repetition along z)."""
    def local(p):
        k, zl = _cell(p[:, 2] + L / 2, -n + 1, 0)
        return p[:, 0].abs() - xc, p[:, 1] - yc, zl

    def metal(p):
        x, y, z = local(p)
        r = torch.sqrt(x ** 2 + z ** 2)
        hoop = torch.sqrt((r - 1.4) ** 2 + y ** 2) - 0.07
        hoop2 = torch.sqrt((r - 0.9) ** 2 + (y + 0.7) ** 2) - 0.05
        u, v = _sector(x, z, 8)                                                         # 8 chains from the hoop to the top ring
        t = ((y - 0.0) / 2.6).clamp(0, 1)
        chain = torch.sqrt((u - 1.4 * (1 - t)) ** 2 + v ** 2 + (y - 2.6 * t) ** 2) - 0.025
        drop = torch.maximum(r - 0.03, -y + 2.6)
        drop = torch.maximum(drop, y - (YS + 8.0 - yc))
        finial = torch.sqrt(x ** 2 + (y + 1.2) ** 2 + z ** 2) - 0.18
        u2, v2 = _sector(x, z, 16)                                                      # the candle cups
        cups = torch.sqrt((u2 - 1.4) ** 2 + v2 ** 2 + (y - 0.1) ** 2) - 0.09
        return torch.minimum(torch.minimum(torch.minimum(hoop, hoop2), torch.minimum(chain, drop)), torch.minimum(finial, cups))

    def wax(p):
        x, y, z = local(p)
        u, v = _sector(x, z, 16)
        return torch.maximum(torch.sqrt((u - 1.4) ** 2 + v ** 2) - 0.035, (y - 0.32).abs() - 0.2)

    def flames(p):
        x, y, z = local(p)
        u, v = _sector(x, z, 16)
        return torch.sqrt((u - 1.4) ** 2 + v ** 2 + ((y - 0.6) * 0.6) ** 2) - 0.045
    return metal, wax, flames


VELVET = Mat("#6a0e14", rough=0.85, sss=0.2,
             pattern=lambda pl, pw, n: {"albedo": srgb("#6a0e14") * (0.7 + 0.3 * torch.sin(pw[:, 2] * 38 + 2 * torch.sin(pw[:, 1] * 3)) ** 2)[:, None]})


def confessional(sc, zc, xw=-13.45):
    """A carved oak confessional against the left aisle's wall, facing the nave:
    the priest's booth in the middle (a panelled door with a grille, a gable, a
    pinnacle and a cross), a penitent's stall on each side behind a red velvet
    curtain, a kneeler, colonnettes with small pinnacles, a moulded cornice."""
    oak = Mat("#3e2614", rough=0.5, clearcoat=0.3, pattern=OAK.pattern)
    dark = Mat("#140c06", rough=0.7)
    brass = Mat("#8a6a30", metal=0.8, rough=0.4)
    xb = xw + 0.62                                                                         # the booths' centre (depth 1.2)
    xf = xw + 1.24                                                                         # their front
    sc.add(sd_box(0.68, 0.1, 1.85, 0.02), oak, T(xb, 0.1, zc))                                # the plinth
    sc.add(sd_box(0.62, 1.55, 0.6, 0.02), oak, T(xb, 1.75, zc))                               # the priest's booth
    sc.add(sd_box(0.04, 1.15, 0.45, 0.01), oak, T(xf + 0.02, 1.4, zc))                        # its door
    for yy in (0.9, 1.75):
        sc.add(sd_box(0.02, 0.32, 0.32, 0.01), dark, T(xf + 0.05, yy, zc))                    # sunk panels
        sc.add(sd_box(0.03, 0.28, 0.28, 0.01), oak, T(xf + 0.06, yy, zc))
    sc.add(sd_box(0.02, 0.2, 0.25, 0.01), dark, T(xf + 0.065, 2.35, zc))                      # the grille
    for k in range(5):
        sc.add(sd_box(0.015, 0.2, 0.012, 0.0), brass, T(xf + 0.08, 2.35, zc - 0.2 + k * 0.1))
    for sz in (-1, 1):
        z = zc + sz * 1.22
        sc.add(sd_box(0.6, 0.05, 0.6, 0.01), oak, T(xb, 2.62, z))                             # the stall's roof
        sc.add(sd_box(0.6, 1.3, 0.04, 0.01), oak, T(xb, 1.3, z + sz * 0.58))                  # its outer side
        sc.add(sd_box(0.03, 1.05, 0.5, 0.01), VELVET, T(xf - 0.1, 1.25, z))                   # the curtain
        sc.add(sd_box(0.03, 0.03, 0.55, 0.01), brass, T(xf - 0.08, 2.35, z))                  # its rod
        sc.add(sd_box(0.18, 0.06, 0.45, 0.01), oak, T(xf + 0.05, 0.28, z))                    # the kneeler
        for zz in (z - 0.58, z + 0.58):                                                       # colonnettes at the front corners
            sc.add(sd_cylinder(1.25, 0.06, 0.01), oak, T(xf - 0.02, 1.45, zz))
            sc.add(sd_cone(0.28, 0.07, 0.0), oak, T(xf - 0.02, 2.95, zz))                     # small pinnacles
    sc.add(sd_box(0.72, 0.08, 1.9, 0.02), oak, T(xb + 0.05, 2.72, zc))                        # the cornice
    sc.add(sd_box(0.6, 0.55, 0.12, 0.02), oak, T(xb, 3.3, zc))                                # the gable's base
    for sz in (-1, 1):                                                                        # the gable's two slopes
        sc.add(sd_box(0.08, 0.06, 0.62, 0.01), oak, T(xf - 0.05, 3.62, zc + sz * 0.3) * RX(sz * 0.7))
    sc.add(sd_cone(0.5, 0.12, 0.0), oak, T(xf - 0.05, 4.4, zc))                               # the pinnacle
    sc.add(sd_box(0.03, 0.22, 0.03, 0.0), GILT, T(xf - 0.05, 5.05, zc))                       # the cross
    sc.add(sd_box(0.03, 0.03, 0.13, 0.0), GILT, T(xf - 0.05, 5.1, zc))


def votive_rack(sc, zc, x=-12.6):
    """A wrought-iron stand of votive candles: three tiers of small lights in red glass."""
    iron = Mat("#1a1612", metal=0.7, rough=0.5)
    for dx, y in ((0.25, 0.85), (0.0, 1.0), (-0.25, 1.15)):
        sc.add(sd_box(0.12, 0.02, 0.75, 0.005), iron, T(x + dx, y, zc))
    for zz in (zc - 0.7, zc + 0.7):
        sc.add(sd_box(0.04, 0.58, 0.04, 0.0), iron, T(x, 0.58, zz))

    def cups(p, top=False):
        x_, y_, z_ = p[:, 0] - x, p[:, 1], p[:, 2] - zc
        tier = torch.clamp(torch.round(-(x_ - 0.25) / 0.25), 0, 2)
        xl = x_ - (0.25 - 0.25 * tier)
        yl = y_ - (0.85 + 0.15 * tier) - (0.09 if top else 0.045)
        k, zl = _cell(z_, -6, 6, 0.11)
        if top:
            return torch.sqrt(xl ** 2 + (yl * 0.6) ** 2 + zl ** 2) - 0.016
        return torch.maximum(torch.sqrt(xl ** 2 + zl ** 2) - 0.035, yl.abs() - 0.025)
    sc.add(lambda p: cups(p), Mat("#c82020", rough=0.3, emissive="#ff6020", emit=0.5))      # red glass cups
    sc.add(lambda p: cups(p, True), glow("#ffc060", 10.0))


def candelabrum(sc, X):
    """A wrought candelabrum fixed to a pillar: five candles on gilt arms."""
    sc.add(sd_cylinder(0.04, 0.12, 0.01), GILT, X)
    for k in range(5):
        x = (k - 2) * 0.17
        y = 0.18 - 0.06 * abs(k - 2)
        sc.add(sd_capsule((0, 0, 0), (x, y, 0.06), 0.015), GILT, X)
        sc.add(sd_cylinder(0.1, 0.025, 0.005), Mat("#f2ead8", rough=0.4, sss=0.5), X * T(x, y + 0.1, 0.06))
        sc.add(sd_sphere(0.035), glow("#ffc870", 9.0), X * T(x, y + 0.24, 0.06))


def scene():
    sc = Scene()
    sc.add(lambda p: p[:, 1], flag_floor())
    st = stone("#6a6052", 0.8)
    sc.add(walls(), st)
    sc.add(pillars(), stone("#74695a", 0.75, 0.5))
    sc.add(colonnettes(), stone("#74695a", 0.75, 0.2))
    sc.add(balustrade(), stone("#74695a", 0.75, 0.0))
    webs, ribs, hgt = vault()
    sc.add(webs, stone("#4e463c", 0.9, 0.4))
    sc.add(ribs, stone("#5e5548", 0.8, 0.0))
    sc.add(aisle(), stone("#3a342c", 0.9))
    sc.add(lambda p: ZE - p[:, 2], stone("#4a4238"))                                           # the east end (behind)
    wall, kings, lancet_glass, mullions = west_wall()
    sc.add(wall, stone("#5a5246"))
    sc.add(kings, stone("#8a806e", 0.6, 0.0))
    sc.add(mullions, stone("#74695a", 0.7, 0.0))
    pal = ["#1a3a9a", "#2448b0", "#b81a28", "#e0a830", "#14307e", "#2a8a4a", "#6a2a9a"]
    sc.add(clerestory_glass(), stained(pal, cell=0.28, emit=2.6, seed=1.0))
    sc.add(aisle_glass(), stained(pal, cell=0.3, emit=2.2, seed=7.0))
    sc.add(pews(), OAK)
    reredos(sc, ZW + 11.0)
    confessional(sc, -11.0)
    votive_rack(sc, -14.4)
    for i in range(NB):                                                                       # candelabra on the pillars' inner faces
        z = -i * L
        if z > -50:
            for sx in (-1, 1):
                candelabrum(sc, T(sx * (W - 0.8), 2.6, z) * RY(-sx * math.pi / 2))
    lg = stained(pal, cell=0.3, emit=2.8, seed=3.0)
    lg.shadow = False
    sc.add(lancet_glass, lg)
    metal, wax_, flames = chandeliers()
    sc.add(metal, Mat("#c8a050", metal=0.9, rough=0.3))
    sc.add(wax_, Mat("#f2ead8", rough=0.4, sss=0.5))
    sc.add(flames, glow("#ffc870", 10.0))
    # the great rose: its glass in the opening (no shadow: the light comes through), its stone tracery:
    # an outer moulded ring, 12 spokes, a ring of 12 small oculi between them, the hub
    Wr = T(0, RY_, ZW - 0.6) * RX(math.pi / 2)
    rose = rose_glass("idle", 0.0, 0.0, emit=3.0, R=RR, shade=False)
    rose.shadow = False
    sc.add(sd_cylinder(0.04, RR), rose, Wr)
    tr = stone("#74695a", 0.7, 0.0)
    sc.add(sd_torus(RR + 0.05, 0.34), tr, Wr)
    sc.add(sd_torus(RR + 0.55, 0.16), tr, T(0, 0, 0.6) * Wr)
    sc.add(sd_torus(1.1, 0.18), tr, Wr)
    sc.add(sd_torus(0.55, 0.1), tr, Wr)
    for k in range(12):
        sc.add(sd_box(1.55, 0.12, 0.12, 0.03), tr, Wr * RY(k * math.tau / 12) * T(2.6, 0, 0))
        sc.add(sd_torus(0.62, 0.08), tr, Wr * RY((k + 0.5) * math.tau / 12) * T(4.45, 0, 0))
        sc.add(sd_torus(0.32, 0.06), tr, Wr * RY((k + 0.5) * math.tau / 12) * T(2.75, 0, 0))
    return sc


def beam_field(p):
    """The beams of light falling from the rose towards the nave floor (0..1):
    soft slanted shafts from points of the rose, along the light's direction."""
    o = torch.stack([p[:, 0], p[:, 1] - 22.0, p[:, 2] - ZW], dim=-1)
    dirv = vec(0.0, -0.72, 0.694)
    tpar = (o * dirv).sum(-1)
    perp = o - tpar[:, None] * dirv
    beam = torch.zeros_like(tpar)
    for (bx, by, bw) in ((0.0, 0.0, 1.5), (-2.4, 1.2, 0.55), (2.6, -0.8, 0.65), (-1.0, -2.6, 0.5), (1.4, 2.4, 0.5), (3.4, 2.0, 0.4), (-3.3, -1.5, 0.45)):
        off = perp - vec(bx, by * 0.694, by * 0.72)
        r = torch.sqrt((off ** 2).sum(-1))
        beam = beam + torch.exp(-(r / bw) ** 2) * (tpar > 0).float() * torch.exp(-tpar / 45.0)
    return beam


def beams(p):
    """The haze (a faint density everywhere) and the beams (denser)."""
    n = fbm(p * vec(0.12, 0.2, 0.12), 3)
    return (0.5 + 0.8 * n) * 0.0012 * torch.exp(-p[:, 1] / 20.0) + 0.02 * beam_field(p)


def beams_glow(p, sig):
    """The beams' own light: warm, streaked by the haze."""
    n = fbm(p * vec(0.3, 0.6, 0.3), 2)
    return srgb("#ffd8a0")[None] * (beam_field(p) * (0.6 + 0.8 * n) * 22.0)[:, None]


def render(W_=2560, H_=1440, ss=1, shift=0.36):
    cam = Camera((0.0, 1.8, 6.0), (0.0, 7.0, ZW), fov=56, shift=(shift, 0.0))
    if os.environ.get("VIT_CAM") == "far":                                                   # a close look at the west end (checks)
        cam = Camera((0.0, 6.0, ZW + 30.0), (0.0, 13.0, ZW), fov=50)
    if os.environ.get("VIT_CAM") == "statues":                                               # a close look at the reredos' statues (checks)
        cam = Camera((1.5, 4.0, ZW + 11.0 + 4.5), (1.5, 3.6, ZW + 11.0), fov=50)
    sky = lambda d: srgb("#020306").expand(d.shape[0], 3) * 1.0
    lights = []
    cols = ["#5a7aff", "#ff5a6a", "#ffc870"]
    for i in range(-NE, NB):
        zm = -L / 2 - i * L
        for sx in (-1, 1):
            lights.append(((sx * 11.0, 6.0, zm), cols[(i + (sx > 0)) % 3], 4.0))       # the aisles' windows
            if -50 < -i * L <= 0:
                lights.append(((sx * (W - 1.1), 2.9, -i * L), "#ffb860", 2.2))          # the candelabra
    lights += [((-12.0, 1.5, -14.4), "#ffb050", 2.5),                                     # the votive candles
               ((-9.0, 4.0, -9.0), "#ffd8b0", 2.0),                                       # a soft light on the confessional
               ((0, 3.5, ZW + 13.5), "#ffd080", 10.0), ((0, 20.0, ZW + 4.0), "#ffe0b0", 26.0),
               ((0, 8.0, ZW + 3.0), "#ffd8a0", 14.0)]                                       # the lancets' light on the retable's back and the kings
    for i in range(4):                                                                       # the chandeliers
        for sx in (-1, 1):
            lights.append(((sx * 3.2, 11.0, -L / 2 - i * L), "#ffc070", 3.5))
    for z in (ZW + 18.0, ZW + 24.0, ZW + 30.0):                                           # the pools of light where the beams land
        lights.append(((0.0, 1.2, z), "#ffe2b0", 5.0))
    vols = [Volume(beams, (-W - 1, 0, ZW), (W + 1, YS + 9, ZE), albedo="#ffe0b8", g=0.6, emission=beams_glow, steps=64, light_steps=0, ambient=1.0)]
    img = render_view(scene(), W_, H_, cam, sky, sun_dir=None, lights=lights, ambient=("#141420", "#0a0806"),
                      exposure=0.95, steps=300, tmax=120, ao_scale=0.6, bloom=1.1, dof=None, vignette=0.45,
                      grain=0.012, grade=("#1e2238", "#ffd8a8"), ss=ss, volumes=vols, refl_occ=1.0)
    return img


if __name__ == "__main__":
    Wd, Hd = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "vitraux.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(Wd, Hd, ss)).save(out, quality=94)
    print("saved", out)
