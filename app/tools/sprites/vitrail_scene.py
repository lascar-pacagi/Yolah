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
        tone = 0.85 + 0.2 * torch.frac(torch.sin(blk) * 4375.5)
        c_ = srgb(c) * (tone * (0.82 + 0.3 * fbm(pw * vec(1.5, 1.5, 1.5), 3)))[:, None] * (1 - 0.35 * torch.maximum(jx, jy) * s)[:, None]
        return {"albedo": c_}
    return Mat(c, rough=k, pattern=pat)


OAK = Mat("#5a3a20", rough=0.45, clearcoat=0.4,
          pattern=lambda pl, pw, n: {"albedo": srgb("#5a3a20") * (0.75 + 0.35 * torch.sin(pw[:, 1] * 30 + 5 * fbm(pw * vec(2, 8, 2), 2)))[:, None]})
GILT = Mat("#d8b050", metal=0.95, rough=0.25)


# ─── the structure ──────────────────────────────────────────────────────────
def walls():
    """Each side wall (x from W to W+1), pierced by the three storeys' openings."""
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        kb, zb = _cell(z + L / 2, -NB + 1, 0)                                    # bay centres (between pillars)
        slab = torch.maximum((x - (W + 0.5)).abs() - 0.5, (z - (ZW + ZE) / 2).abs() - (ZE - ZW) / 2)
        slab = torch.maximum(slab, y - YS - 2.0)
        arcade = _arch(zb, y, 2.35, 9.6, 1.0, -1.0)                               # the great arcade
        gallery = torch.minimum(_arch(zb - 1.2, y, 0.95, 18.2, 0.35, 16.6), _arch(zb + 1.2, y, 0.95, 18.2, 0.35, 16.6))
        clere = torch.minimum(_arch(zb - 1.1, y, 0.8, 29.5, 0.5, 23.6), _arch(zb + 1.1, y, 0.8, 29.5, 0.5, 23.6))
        ocul = torch.sqrt(zb ** 2 + (y - 31.9) ** 2) - 0.75
        hole = torch.minimum(torch.minimum(arcade, gallery), torch.minimum(clere, ocul))
        hole = torch.maximum(hole, (z - ZW).abs() - 1e9)                         # (all bays)
        inbay = (z < 0.0) & (z > ZW)
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
        k, zl = _cell(z, -NB, 0)
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
        kb, zb = _cell(z + L / 2, -NB + 1, 0)
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
        inn = torch.maximum(z - 0.0, ZW - z)
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
        k, zb = _cell(z + L / 2, -NB + 1, 0)
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
        k, zl = _cell(z, -NB, 0)
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
        kb, zb = _cell(z + L / 2, -NB + 1, 0)
        clere = torch.minimum(_arch(zb - 1.1, y, 0.8, 29.5, 0.5, 23.6), _arch(zb + 1.1, y, 0.8, 29.5, 0.5, 23.6))
        ocul = torch.sqrt(zb ** 2 + (y - 31.9) ** 2) - 0.75
        pane = torch.maximum(torch.minimum(clere, ocul), (x - (W + 0.85)).abs() - 0.03)
        return torch.maximum(pane, torch.maximum(z - 0.0, ZW - z))
    return f


def aisle():
    """The aisles behind the arcade: their low vault (dark), the outer wall and its
    windows glowing dimly."""
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        ceil = torch.maximum(13.0 - y, W + 0.9 - x)
        outer = 13.5 - x
        return torch.minimum(ceil, outer)
    return f


def aisle_glass():
    def f(p):
        x, y, z = p[:, 0].abs(), p[:, 1], p[:, 2]
        kb, zb = _cell(z + L / 2, -NB + 1, 0)
        win = _arch(zb, y, 1.0, 8.5, 0.4, 3.0)
        return torch.maximum(torch.maximum(win, (x - 13.45).abs() - 0.03), torch.maximum(z, ZW - z))
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
        crack = (torch.abs(fbm(pw * vec(1.8, 1.8, 1.8), 3) - 0.5) < 0.012).float()
        wear = torch.exp(-(x / 1.3) ** 2)                                            # the aisle polished by feet
        c = srgb("#8a8274") * (tone * (0.8 + 0.3 * fbm(pw * vec(1.1, 1.1, 1.1), 4)))[:, None] * (1 - 0.55 * torch.maximum(joint, crack))[:, None]
        return {"albedo": c, "rough": 0.55 - 0.35 * wear + 0.3 * joint}
    return Mat("#8a8274", rough=0.4, clearcoat=0.5, pattern=pat)


def reredos(sc, Z):
    """The gilt gothic altarpiece: a tall central tower with a spire and a statue in
    its niche, two lower towers with statues, pinnacles, crockets of gold; the altar
    before it on its steps, gilt rails on either side."""
    dark = Mat("#1a120c", rough=0.6)
    marble = Mat("#e8e0d0", rough=0.35, sss=0.2)
    for i, (hw, hh) in enumerate(((4.2, 0.15), (3.6, 0.3), (3.0, 0.45))):
        sc.add(sd_box(hw, hh, 2.0 - 0.4 * i, 0.02), stone("#8a8070", 0.5, 0.3), T(0, hh, Z + 2.5))
    sc.add(sd_box(1.6, 0.5, 0.6, 0.04), stone("#a89c88", 0.4, 0.0), T(0, 1.4, Z + 2.2))
    sc.add(sd_box(1.65, 0.05, 0.65, 0.01), GILT, T(0, 1.92, Z + 2.2))
    sc.add(sd_box(1.2, 0.35, 0.05, 0.02), GILT, T(0, 1.4, Z + 2.82))                                   # the altar's gilt front
    for (x, w, h, sp) in ((0.0, 0.85, 4.6, 2.2), (-1.85, 0.6, 3.2, 1.4), (1.85, 0.6, 3.2, 1.4)):
        sc.add(sd_box(w, h / 2, 0.35, 0.02), GILT, T(x, 1.9 + h / 2, Z))
        sc.add(sd_box(w * 0.62, h * 0.32, 0.1, 0.02), dark, T(x, 1.9 + h * 0.45, Z + 0.3))            # the niche
        sc.add(sd_capsule((x, 1.9 + h * 0.2, Z + 0.32), (x, 1.9 + h * 0.62, Z + 0.32), w * 0.22), marble, T(0, 0, 0))   # the statue
        sc.add(sd_sphere(w * 0.16), marble, T(x, 1.9 + h * 0.7, Z + 0.32))
        sc.add(sd_cone(sp / 2, w * 0.75, 0.02), GILT, T(x, 1.9 + h + sp / 2, Z))                       # the spire
        for sx in (-1, 1):
            sc.add(sd_box(0.07, h / 2 + 0.1, 0.07, 0.01), GILT, T(x + sx * w, 1.9 + h / 2, Z + 0.36))   # buttresses
            sc.add(sd_cone(0.35, 0.09, 0.0), GILT, T(x + sx * w, 2.35 + h, Z + 0.36))                   # pinnacles
    sc.add(sd_box(0.05, 0.45, 0.05, 0.01), GILT, T(0, 1.9 + 4.6 + 2.2 + 0.5, Z))                       # the cross on top
    sc.add(sd_box(0.28, 0.05, 0.05, 0.01), GILT, T(0, 1.9 + 4.6 + 2.2 + 0.7, Z))
    for x in (-1.3, -0.8, 0.8, 1.3):                                                                 # the altar's candles
        sc.add(sd_cylinder(0.22, 0.035, 0.01), Mat("#f2ead8", rough=0.4, sss=0.5), T(x, 2.18, Z + 2.2))
        sc.add(sd_sphere(0.045), glow("#ffd890", 9.0), T(x, 2.45, Z + 2.2))
    for sx in (-1, 1):                                                                               # the gilt rails
        R = T(sx * 4.6, 0, Z + 4.8)
        sc.add(sd_box(1.2, 0.04, 0.04, 0.01), GILT, R * T(0, 1.0, 0))
        for k in range(7):
            sc.add(sd_box(0.025, 0.5, 0.025, 0.005), GILT, R * T(-1.1 + k * 0.37, 0.5, 0))


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
    sc.add(lambda p: torch.maximum(p[:, 2] - ZW, 5.0 - torch.sqrt(p[:, 0] ** 2 + (p[:, 1] - 22.0) ** 2)), stone("#5a5246"))   # the west wall, the rose cut out
    pal = ["#1a3a9a", "#2448b0", "#b81a28", "#e0a830", "#14307e", "#2a8a4a", "#6a2a9a"]
    sc.add(clerestory_glass(), stained(pal, cell=0.28, emit=2.6, seed=1.0))
    sc.add(aisle_glass(), stained(pal, cell=0.3, emit=2.2, seed=7.0))
    sc.add(pews(), OAK)
    reredos(sc, ZW + 11.0)
    for i in range(NB):                                                                       # candelabra on the pillars' inner faces
        z = -i * L
        if z > -50:
            for sx in (-1, 1):
                candelabrum(sc, T(sx * (W - 0.8), 2.6, z) * RY(-sx * math.pi / 2))
    # the great rose: its glass in the opening (no shadow: the light comes through), its stone tracery
    Wr = T(0, 22.0, ZW - 0.2) * RX(math.pi / 2)
    rose = rose_glass("idle", 0.0, 0.0, emit=3.0, R=5.0)
    rose.shadow = False
    sc.add(sd_cylinder(0.04, 5.0, 0.0), rose, Wr)
    sc.add(sd_torus(5.05, 0.32), stone("#74695a", 0.7, 0.0), Wr)
    sc.add(sd_torus(1.1, 0.18), stone("#74695a", 0.7, 0.0), Wr)
    for k in range(12):
        sc.add(sd_box(1.9, 0.13, 0.12, 0.03), stone("#74695a", 0.7, 0.0), Wr * RY(k * math.tau / 12) * T(3.0, 0, 0))
    # under the rose: a blind arcade of lancets on the west wall
    for k in range(7):
        x = (k - 3) * 1.6
        sc.add(sd_box(0.12, 3.0, 0.15, 0.02), stone("#74695a", 0.7, 0.0), T(x - 0.8, 13.5, ZW + 0.15))
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
    sky = lambda d: srgb("#020306").expand(d.shape[0], 3) * 1.0
    lights = []
    cols = ["#5a7aff", "#ff5a6a", "#ffc870"]
    for i in range(NB):
        zm = -L / 2 - i * L
        for sx in (-1, 1):
            lights.append(((sx * 11.0, 6.0, zm), cols[(i + (sx > 0)) % 3], 4.0))       # the aisles' windows
            if -i * L > -50:
                lights.append(((sx * (W - 1.1), 2.9, -i * L), "#ffb860", 2.2))          # the candelabra
    lights += [((0, 3.5, ZW + 13.5), "#ffd080", 10.0), ((0, 20.0, ZW + 4.0), "#ffe0b0", 26.0)]
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
