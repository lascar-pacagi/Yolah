"""
vitraux.py — the "Vitraux" set: the figures of a cathedral's stained glass, made
of coloured glass set in lead and lit from within, from the weakest AI to the
strongest (the user's choice, concept A "la cathédrale"):

  random        Bougie           a church candle in its holder, its flame
  beginner      Colombe          the dove of the Holy Spirit in a glory of rays
  monte_carlo   Lys              the fleur-de-lis / the lily of the windows
  mcts          Cloche           the bell, swinging
  basic_minmax  Gargouille       the gargoyle on its cornice, rain dripping from its jaws
  minmax        Moine copiste    the monk writing at his lectern by a candle
  features      Chevalier        the knight, shield and sword
  mcts_nnue     Licorne          the unicorn rearing
  nnue193       Ange             the angel, wings open
  resnet        Lion ailé        the winged lion (Saint Mark)
  dev           Archange         Saint Michael, the flaming sword, the dragon at his feet
  alphazero     Rosace           the rose window, alive, turning slowly

Every surface is glass cut into pieces (a 3D Voronoi of cells) joined by dark
lead came; each piece takes a colour of the figure's palette, darkens towards
its lead (grisaille) and glows (emissive), as a window does with the sun behind it. Animations: idle — the
light breathes and the figure stirs (flame, wings, bell, quill...); move — a
hop; blocked — a shake, the light dims to a dull red; win — a wave of light
runs through the glass. Every periodic motion is a whole number of cycles over
the loop (t = 1 is t = 0).

    cd app/tools/sprites && python3 vitraux.py preview [name ...]   (contact sheets /tmp/vit_<name>.png)
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm, vec, DEV, F
from models import base, BASE_TOP, TAU, glow
from heroic import chain, bezier, membrane_wing, HEAD

LEAD = "#2a2a30"


# ─── the glass ──────────────────────────────────────────────────────────────
def _hash3(c):
    """A pseudo-random number in 0..1 per integer cell (N, 3)."""
    return torch.frac(torch.sin(c[:, 0] * 127.1 + c[:, 1] * 311.7 + c[:, 2] * 74.7) * 43758.5453)


_OFFS = None


def _offsets():
    global _OFFS
    if _OFFS is None:
        _OFFS = torch.tensor([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], device=DEV, dtype=F)
    return _OFFS


def stained(palette, cell=0.07, lead=0.05, emit=0.9, seed=0.0, wave=None, dim=None, shade=False):
    """Stained glass: the local space cut into cells (3D Voronoi, size `cell`), each
    a piece of glass of one colour of `palette` (hex list), the borders between
    cells dark lead (width `lead`, a share of a cell). The glass glows: emissive
    = its colour × emit. wave (0..1 or None): a band of brighter light sweeping up
    the figure (win); dim (0..1 or None): the light falls to a dull red (blocked).
    shade: each piece darkens towards its lead, as the painted grisaille of real
    windows does (the pieces read as separate panes, the figure gains depth)."""
    pal = torch.stack([srgb(c) for c in palette])                              # (K, 3), made once
    K = len(palette)
    lead_c = srgb(LEAD)

    def pat(pl, pw, n):
        q = pl / cell + seed
        base_c = torch.floor(q)
        f = q - base_c
        offs = _offsets()
        cells = base_c[:, None, :] + offs[None]                                 # (N, 27, 3)
        jit = torch.stack([_hash3(cells.reshape(-1, 3) + o).reshape(cells.shape[:2]) for o in (0.0, 17.3, 41.9)], -1)
        pts = offs[None] + 0.15 + 0.7 * jit                                     # the cells' points, relative to base_c
        d = ((pts - f[:, None, :]) ** 2).sum(-1)                                 # (N, 27)
        d2, idx = torch.topk(d, 2, dim=1, largest=False)
        r1, r2 = d2[:, 0].sqrt(), d2[:, 1].sqrt()
        border = (r2 - r1) < lead                                                # near the bisector: lead
        cid = cells[torch.arange(cells.shape[0], device=DEV), idx[:, 0]]
        col = pal[(_hash3(cid + 5.0) * K).long().clamp(0, K - 1)]
        tone = 0.85 + 0.3 * _hash3(cid + 9.0)[:, None]                         # each piece its own shade
        col = col * tone
        k = torch.full_like(r1, emit)
        if wave is not None:                                                     # the wave of light (win)
            k = k * (1 + 2.5 * torch.exp(-((pl[:, 1] - (wave * 1.2 - 0.1)) / 0.08) ** 2))
        em = col * k[:, None]
        if shade:                                                                # grisaille: darker near the lead
            em = em * (0.45 + 0.55 * ((r2 - r1) / 0.45).clamp(0, 1) ** 0.6)[:, None]
        if dim is not None:
            em = em * (1 - dim) + srgb("#801810") * dim * 0.8
        g = (~border).float()[:, None]
        return {"albedo": col * 0.35 * g + lead_c * (1 - g), "emissive": em * g, "rough": 0.08 + 0.5 * (1 - g[:, 0])}
    return Mat(palette[0], rough=0.08, clearcoat=1.0, pattern=pat)


def light(anim, t, k0=0.9):
    """The glow of the glass over the loop: it breathes (idle, 2 cycles), rises
    (win), sinks (blocked)."""
    if anim == "idle":
        return k0 * (0.85 + 0.15 * math.sin(TAU * 2 * t))
    if anim == "win":
        return k0 * 1.3
    if anim == "blocked":
        return k0 * 0.5
    return k0


def glass(palette, anim, t, cell=0.07, seed=0.0, k0=0.9):
    """The figure's glass for this frame (the light of the animation): thin lead,
    each piece shaded towards its lead (grisaille)."""
    return stained(palette, cell=cell, emit=light(anim, t, k0), seed=seed, lead=0.035, shade=True,
                   wave=(t % 1) if anim == "win" else None,
                   dim=(0.5 + 0.5 * math.sin(TAU * 4 * t)) * 0.6 if anim == "blocked" else None)


LEAD_MAT = Mat(LEAD, metal=0.6, rough=0.4)


def pane(color, anim, t, k0=0.5):
    """ONE piece of glass of a single colour (no lead, no Voronoi cells): a face,
    on which the features are painted. Lit like glass() (the same breathing,
    wave of light on a win, dull red when blocked)."""
    c = srgb(color)
    k = light(anim, t, k0)
    wave = (t % 1) if anim == "win" else None
    dim = (0.5 + 0.5 * math.sin(TAU * 4 * t)) * 0.6 if anim == "blocked" else None

    def pat(pl, pw, n):
        kk = torch.full((pl.shape[0], 1), k, device=DEV, dtype=F)
        if wave is not None:
            kk = kk * (1 + 2.5 * torch.exp(-((pl[:, 1:2] - (wave * 1.2 - 0.1)) / 0.08) ** 2))
        em = c[None, :] * kk
        if dim is not None:
            em = em * (1 - dim) + srgb("#801810") * dim * 0.8
        return {"albedo": (c * 0.35)[None, :].expand(pl.shape[0], 3), "emissive": em, "rough": torch.full((pl.shape[0],), 0.08, device=DEV, dtype=F)}
    return Mat(color, rough=0.08, clearcoat=1.0, pattern=pat)


def stand(anim, t, k=1.0):
    """The pose: a hop (move), a shake (blocked), a small leap (win)."""
    lift, tilt, turn = 0.0, 0.0, 0.0
    if anim == "move":
        lift = 0.05 * abs(math.sin(TAU * t))
    elif anim == "blocked":
        tilt = 0.05 * math.sin(TAU * 4 * t)
        turn = 0.04 * math.sin(TAU * 3 * t)
    elif anim == "win":
        lift = 0.03 * abs(math.sin(TAU * 2 * t))
    return RY(HEAD + turn) * T(0, BASE_TOP + lift * k, 0) * RZ(tilt)


# ─── 1 · Bougie ─────────────────────────────────────────────────────────────
def candle(anim, t, side):
    """The paschal candle on its gilt stand: three feet, a knop, a drip pan; the tall
    candle of ivory glass with wax running down from its rim, red and blue bands,
    the red cross set with five gold grains of incense; the flame of amber glass,
    blue at its root, over a real glow. idle: the flame sways and breathes (one
    sway, three breaths per loop); win: it flares high; blocked: it gutters low."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    # the stand: three curled feet, the stem with its knop, the drip pan
    for k in range(3):
        a = TAU * k / 3 + 0.3
        F_ = S * RY(a)
        chain(sc, [(0, 0.06, 0.0), (0, 0.03, 0.08), (0, 0.012, 0.13), (0, 0.03, 0.15)], [0.016, 0.013, 0.011, 0.009], GILT, F_, blend=0.01)
        sc.add(sd_sphere(0.014), GILT, F_ * T(0, 0.03, 0.155))
    sc.add(sd_cone(0.04, 0.045, 0.022, 0.006), GILT, S * T(0, 0.07, 0))
    sc.add(sd_cylinder(0.04, 0.016, 0.004), GILT, S * T(0, 0.14, 0))
    sc.add(sd_sphere(0.032), GILT, S * T(0, 0.19, 0))                                          # the knop
    sc.add(sd_cylinder(0.025, 0.014, 0.004), GILT, S * T(0, 0.235, 0))
    sc.add(sd_cylinder(0.008, 0.11, 0.004), GILT, S * T(0, 0.265, 0))                          # the drip pan
    sc.add(sd_torus(0.11, 0.008), GILT, S * T(0, 0.275, 0))
    # the candle
    ivory = glass(["#f4ead2", "#efe0b8", "#f8f0de", "#e8d4a8"], anim, t, cell=0.035, k0=0.35)
    sc.add(sd_cylinder(0.2, 0.062, 0.01), ivory, S * T(0, 0.47, 0))
    sc.add(sd_cylinder(0.012, 0.05, 0.004), glass(["#fff6e0", "#f8ecd0"], anim, t, cell=0.02, seed=1.0, k0=0.5), S * T(0, 0.668, 0))   # the pool of wax
    for k, (a, L) in enumerate(((0.3, 0.07), (1.4, 0.12), (2.5, 0.05), (3.6, 0.09), (4.9, 0.06))):   # wax running down
        x, z = 0.066 * math.sin(a), 0.066 * math.cos(a)
        sc.add(sd_capsule((x, 0.672, z), (x, 0.665 - L, z), 0.011, 0.009), ivory, S, blend=0.006)
        sc.add(sd_sphere(0.013), ivory, S * T(x * 1.02, 0.665 - L, z * 1.02))
    red = glass(RED_P, anim, t, cell=0.025, seed=7.0, k0=0.7)
    blue = glass(BLUE_P, anim, t, cell=0.025, seed=8.0, k0=0.7)
    gold = glass(GOLD_P, anim, t, cell=0.015, seed=9.0, k0=1.0)
    sc.add(sd_cylinder(0.014, 0.0645, 0.004), red, S * T(0, 0.31, 0))                           # the painted bands
    sc.add(sd_cylinder(0.008, 0.0645, 0.003), blue, S * T(0, 0.335, 0))
    sc.add(sd_cylinder(0.008, 0.0645, 0.003), blue, S * T(0, 0.6, 0))
    sc.add(sd_box(0.011, 0.075, 0.004, 0.003), red, S * T(0, 0.47, 0.062))                     # the cross
    sc.add(sd_box(0.045, 0.011, 0.004, 0.003), red, S * T(0, 0.49, 0.062))
    for (x, y) in ((0, 0.49), (0, 0.545), (0, 0.415), (-0.033, 0.49), (0.033, 0.49)):            # the five grains of incense
        sc.add(sd_sphere(0.008), gold, S * T(x, y, 0.066))
    # the wick and the flame
    sc.add(sd_capsule((0, 0.67, 0), (0, 0.7, 0), 0.004), LEAD_MAT, S)
    sway = 0.06 * math.sin(TAU * t) if anim == "idle" else (0.02 * math.sin(TAU * 4 * t) if anim == "blocked" else 0.0)
    hgt = {"win": 1.6, "blocked": 0.55}.get(anim, 1.0) * (1 + (0.08 * math.sin(TAU * 3 * t) if anim == "idle" else 0))
    Fl = S * T(0, 0.69, 0) * RZ(sway)
    flame = glass(["#ffcf4a", "#ffb030", "#ffe28a", "#ff9a2a"], anim, t, cell=0.022, seed=3.0, k0=2.2)
    sc.add(sd_ellipsoid(0.03, 0.075 * hgt, 0.03), flame, Fl * T(0, 0.06 * hgt, 0), blend=0.01)
    sc.add(sd_cone(0.03 * hgt, 0.022, 0.001), flame, Fl * T(0, 0.14 * hgt, 0), blend=0.02)      # its pointed tip
    sc.add(sd_ellipsoid(0.017, 0.022, 0.017), glass(["#4a7af0", "#6a9aff"], anim, t, cell=0.015, seed=4.0, k0=1.5), Fl * T(0, 0.012, 0))   # the blue root
    sc.add(sd_sphere(0.028), glow("#fff2c0", 9.0 if anim == "win" else (3.0 if anim == "blocked" else 6.0)), Fl * T(0, 0.06 * hgt, 0))
    return sc


# ─── 2 · Colombe ────────────────────────────────────────────────────────────
def dove(anim, t, side):
    """The dove of the Holy Spirit as the windows show it: descending head first,
    facing us, wings spread, white and pale blue glass, an olive twig in its gold
    beak, in a glory of golden rays held up by a gilt rod; idle: the wings beat
    slowly (one beat per loop); win: three fast beats, the glory blazes; blocked:
    the wings hang."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    C = S * T(0, 0.5, 0)
    sc.add(sd_capsule((0, 0.0, -0.1), (0, 0.38, -0.1), 0.012), GILT, S)                        # the rod
    sc.add(sd_cylinder(0.02, 0.06, 0.01), GILT, S * T(0, 0.02, -0.1))
    # the glory: rays alternately long and short, straight and flamed, behind the dove
    rays = glass(GOLD_P + ["#fff0b0"], anim, t, cell=0.03, seed=9.0, k0=1.3 if anim == "win" else 0.8)
    for k in range(16):
        a = TAU * (k + 0.5) / 16
        L = 0.3 if k % 2 == 0 else 0.21
        sc.add(sd_cone(L / 2, 0.022, 0.002), rays, C * T(0, 0, -0.1) * RZ(a - math.pi / 2) * T(0, 0.07 + L / 2, 0))
    sc.add(sd_cylinder(0.008, 0.075, 0.004), rays, C * T(0, 0, -0.1) * RX(math.pi / 2))
    body = glass(["#f4f6fa", "#e8eef8", "#d4e0f2", "#c4d4ee"], anim, t, cell=0.035, k0=0.16)
    D = C * RX(0.3)                                                                           # head down, coming towards us
    sc.add(sd_ellipsoid(0.075, 0.07, 0.13), body, D, blend=0.03)                              # the body
    sc.add(sd_ellipsoid(0.06, 0.06, 0.065), body, D * T(0, 0.02, 0.12), blend=0.04)          # the breast
    Hd = D * T(0, 0.05, 0.18) * RX(-0.35)
    sc.add(sd_sphere(0.052), body, Hd, blend=0.03)                                            # the head
    beak = glass(["#f0b040", "#e09a30"], anim, t, cell=0.02, k0=0.9)
    sc.add(sd_cone(0.035, 0.016, 0.002), beak, Hd * T(0, -0.008, 0.045) * RX(math.pi / 2))
    for sx in (-1, 1):
        sc.add(sd_sphere(0.009), LEAD_MAT, Hd * T(sx * 0.03, 0.015, 0.032))                  # the eyes
    # the tail: a fan of five feathers behind, rising
    tail = glass(["#e8eef8", "#c8d8ee", "#f4f6fa"], anim, t, cell=0.03, seed=3.0, k0=0.16)
    for k in range(5):
        a = (k - 2) * 0.22
        sc.add(sd_ellipsoid(0.022, 0.006, 0.07), tail, D * T(0, 0.01, -0.13) * RY(a) * RX(-0.35) * T(0, 0, -0.06))
    # the olive twig in the beak
    olive = glass(["#6a9a3a", "#88b848", "#4a7a2a"], anim, t, cell=0.025, seed=5.0, k0=0.7)
    sc.add(sd_capsule((-0.02, -0.01, 0.07), (0.1, -0.05, 0.08), 0.005), LEAD_MAT, Hd)
    for k in range(4):
        x = 0.0 + k * 0.028
        sc.add(sd_ellipsoid(0.022, 0.007, 0.011), olive, Hd * T(x, -0.02 - k * 0.009 + (0.012 if k % 2 else -0.012), 0.075) * RZ(0.5 if k % 2 else -0.5))
    # the wings, spread in the plane facing us
    if anim == "blocked":
        a = -0.25 + 0.04 * math.sin(TAU * 6 * t)
    else:
        beat = 3 if anim == "win" else 1
        a = 0.35 + 0.3 * math.sin(TAU * beat * t)
    wing = glass(["#ffffff", "#e8eef8", "#c8d8f0", "#a8c0e4"], anim, t, cell=0.03, seed=2.0, k0=0.18)
    cover = glass(["#f4f6fa", "#dce6f4"], anim, t, cell=0.025, seed=8.0, k0=0.16)
    for sx in (-1, 1):
        feather_wing(sc, C * T(sx * 0.05, 0.03, 0.02) * RZ(sx * a), sx, wing, cover, length=0.27, n=8, chord=0.1)
    return sc


# ─── 3 · Lys ────────────────────────────────────────────────────────────────
def sd_petal(w, h, th):
    """A pointed petal in the xy plane: a vesica (two arcs meeting in points at
    y = ±h), width 2w; thickness 2·th along z, thinning towards the edges."""
    R = (w * w + h * h) / (2 * w)

    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        d1 = torch.sqrt((x - (R - w)) ** 2 + y ** 2) - R
        d2 = torch.sqrt((x + (R - w)) ** 2 + y ** 2) - R
        return torch.maximum(torch.maximum(d1, d2), z.abs() - th * (1 - 0.6 * (x.abs() / w).clamp(0, 1)))
    return f


def lily(anim, t, side):
    """The fleur-de-lis of the windows (florencée): the tall middle petal with its
    ridge, the two outer petals curling out and down into a scroll, two stamens
    between them, the blue band set with gold studs, the three short petals
    below, all on a gilt stem and foot; idle: the outer petals breathe (one cycle
    per loop); win: they open wide and shine; blocked: they close."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    gold = glass(["#f2c040", "#e8a828", "#ffd860", "#d87a20", "#c86a18"], anim, t, cell=0.035, k0=0.6)
    rim = glass(["#c86a18", "#b85a10", "#d87a20"], anim, t, cell=0.025, seed=2.0, k0=0.6)
    blue = glass(BLUE_P, anim, t, cell=0.03, seed=4.0, k0=0.8)
    open_ = {"idle": 0.08 * math.sin(TAU * t), "win": 0.25 * abs(math.sin(TAU * t)), "blocked": -0.1}.get(anim, 0.0)
    # the foot and the stem
    sc.add(sd_cylinder(0.02, 0.1, 0.008), GILT, S * T(0, 0.02, 0))
    sc.add(sd_cone(0.04, 0.07, 0.025, 0.005), GILT, S * T(0, 0.08, 0))
    sc.add(sd_torus(0.03, 0.008), GILT, S * T(0, 0.125, 0))
    # the middle petal: rim, glass, the ridge
    sc.add(sd_petal(0.1, 0.27, 0.03), rim, S * T(0, 0.58, -0.004))
    sc.add(sd_petal(0.088, 0.255, 0.04), gold, S * T(0, 0.585, 0))
    sc.add(sd_capsule((0, 0.4, 0.03), (0, 0.78, 0.012), 0.006, 0.003), rim, S)
    # the outer petals: a curve out and down, ending in a scroll
    for sx in (-1, 1):
        P = S * T(sx * 0.07, 0.4, 0) * RZ(-sx * open_)
        pts = bezier((0, 0, 0), (sx * 0.06, 0.16, 0), (sx * 0.24, 0.2, 0), (sx * 0.25, 0.06, 0), 7)
        chain(sc, pts, [0.05, 0.055, 0.052, 0.045, 0.037, 0.03, 0.026], gold, P, blend=0.03)
        chain(sc, [(sx * 0.25, 0.06, 0), (sx * 0.23, 0.0, 0), (sx * 0.19, 0.01, 0), (sx * 0.19, 0.05, 0)], [0.026, 0.024, 0.02, 0.016], rim, P, blend=0.01)
        sc.add(sd_capsule((sx * 0.03, 0.1, 0.02), (sx * 0.2, 0.17, 0.035), 0.005, 0.003), rim, P)    # the vein
        # a stamen between the middle and the outer petal
        sc.add(sd_capsule((sx * 0.06, 0.47, 0), (sx * 0.12, 0.64, 0), 0.007, 0.005), rim, S)
        sc.add(sd_sphere(0.022), glass(["#ffd860", "#f2c040"], anim, t, cell=0.015, seed=5.0, k0=1.0), S * T(sx * 0.125, 0.66, 0))
    # the band, with its studs
    sc.add(sd_box(0.15, 0.032, 0.05, 0.016), blue, S * T(0, 0.34, 0))
    for y in (0.307, 0.373):
        sc.add(sd_box(0.15, 0.004, 0.052, 0.003), LEAD_MAT, S * T(0, y, 0))
    for x in (-0.09, 0.0, 0.09):
        sc.add(sd_sphere(0.014), glass(GOLD_P, anim, t, cell=0.012, seed=6.0, k0=1.0), S * T(x, 0.34, 0.052))
    # the three short petals below the band
    sc.add(sd_petal(0.03, 0.06, 0.025), gold, S * T(0, 0.25, 0))
    for sx in (-1, 1):
        chain(sc, bezier((sx * 0.04, 0.3, 0), (sx * 0.1, 0.27, 0), (sx * 0.13, 0.2, 0), (sx * 0.09, 0.18, 0), 5),
              [0.028, 0.026, 0.022, 0.018, 0.014], gold, S, blend=0.015)
    sc.add(sd_cylinder(0.05, 0.022, 0.006), GILT, S * T(0, 0.18, 0))                            # the stem up to the band
    return sc


# palettes shared by the figures
GOLD_P = ["#f2c040", "#e8a828", "#ffd860", "#d87a20"]
BLUE_P = ["#2a4ab8", "#1a3a98", "#3a64d0", "#203a8a"]
RED_P = ["#c81e2a", "#a81820", "#e03040", "#901418"]
GREEN_P = ["#2a8a4a", "#3aa058", "#1a6a38"]
PURPLE_P = ["#6a2a9a", "#8a3ab8", "#4a1a7a"]
WHITE_P = ["#f4f6fa", "#e4ecf8", "#d8e0ee"]
SKIN_P = ["#f0c8a0", "#e8b890", "#f4d4b0"]
STONE_P = ["#5a6a7a", "#4a5a5a", "#6a5a7a", "#3a4a58"]
GILT = Mat("#d8a840", metal=0.95, rough=0.25)
STONE_MAT = Mat("#5e5850", rough=0.6, pattern=lambda pl, pw, n: {"albedo": srgb("#5e5850") * (0.8 + 0.3 * fbm(pl * 20, 2))[:, None]})


def wings(sc, X, mat, span=0.32, spread=0.6, droop=0.0, feathers=4):
    """A pair of wings: each a fan of long feather blades (ellipsoids) from the shoulder,
    raised by `spread` (radians), the tips drooping by `droop`."""
    for sx in (-1, 1):
        W = X * T(sx * 0.05, 0, -0.04) * RZ(sx * spread)
        for k in range(feathers):
            L = span * (1.0 - 0.13 * k)
            a = sx * (0.12 + 0.28 * k) + sx * droop * 0.0
            sc.add(sd_ellipsoid(L / 2, 0.018, 0.045), mat, W * RZ(-sx * 0.28 * k) * T(sx * L / 2, 0.0, -0.01 * k) * RY(sx * 0.15), blend=0.01)


def feather_wing(sc, X, sx, mat, cover, length=0.3, n=8, chord=0.12, root=-1.45, tip=-0.2):
    """One bird's wing in the frame X: its plane the local xy plane, the arm along
    local x·sx from the shoulder at the origin. The leading edge; n flight feathers
    hanging from it, pointing from the angle `root` (radians from the arm, at the
    body) to `tip` (at the end), longer towards the tip; a row of short covert
    feathers of `cover` over their roots."""
    sc.add(sd_capsule((0, 0, 0), (sx * length, 0, 0), 0.02, 0.009), cover, X, blend=0.01)
    for k in range(n):
        u = (k + 0.5) / n
        th = root + (tip - root) * u
        L = chord * (1 + 1.1 * u)
        phi = math.atan2(math.sin(th), sx * math.cos(th))
        sc.add(sd_ellipsoid(L / 2, 0.022, 0.007), mat, X * T(sx * u * length, 0, -0.003 * k) * RZ(phi) * T(L / 2, 0, 0))
    for k in range(n - 1):
        u = (k + 0.8) / n
        th = root + (tip - root) * u
        phi = math.atan2(math.sin(th), sx * math.cos(th))
        sc.add(sd_ellipsoid(chord * 0.28, 0.026, 0.008), cover, X * T(sx * u * length * 0.95, 0, 0.01) * RZ(phi) * T(chord * 0.2, 0, 0))


# ─── 4 · Cloche ─────────────────────────────────────────────────────────────
def sd_bell(H, r_top, r_lip):
    """A bell, its lip at y = 0, its crown at y = H: a solid of revolution, the
    profile narrow at the shoulder, flaring to the lip."""
    def f(p):
        y = p[:, 1]
        u = (y / H).clamp(0, 1)
        r = r_top + (r_lip - r_top) * (1 - u) ** 2.2 + 0.012 * torch.exp(-((u - 0.06) / 0.05) ** 2)   # the flare, the lip's bead
        d = (torch.hypot(p[:, 0], p[:, 2]) - r) * 0.8
        return torch.maximum(d, torch.maximum(-y, y - H))
    return f


def bell(anim, t, side):
    """The church bell in its stone belfry: two pillars with capitals under a
    pointed arch and a small cross; the bell of amber glass hung from a wooden
    yoke with iron straps, an inscription in lead between two gold bands, a gold
    cross on its waist, the crown's loops, the clapper; idle: it swings (one full
    swing per loop); win: it rings hard (two swings); blocked: it trembles."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    # the belfry: plinth, pillars, capitals, the pointed arch, the cross
    sc.add(sd_box(0.39, 0.025, 0.07, 0.008), STONE_MAT, S * T(0, 0.025, 0))
    for sx in (-1, 1):
        sc.add(sd_box(0.035, 0.29, 0.04, 0.008), STONE_MAT, S * T(sx * 0.34, 0.33, 0))
        sc.add(sd_box(0.05, 0.02, 0.055, 0.006), STONE_MAT, S * T(sx * 0.34, 0.63, 0))
        chain(sc, bezier((sx * 0.34, 0.65, 0), (sx * 0.34, 0.8, 0), (sx * 0.16, 0.92, 0), (0, 0.97, 0), 6),
              [0.035, 0.034, 0.033, 0.032, 0.031, 0.03], STONE_MAT, S, blend=0.01)
    sc.add(sd_box(0.012, 0.05, 0.012, 0.003), STONE_MAT, S * T(0, 1.04, 0))
    sc.add(sd_box(0.035, 0.012, 0.012, 0.003), STONE_MAT, S * T(0, 1.05, 0))
    sw = {"idle": 0.25 * math.sin(TAU * t), "win": 0.3 * math.sin(TAU * 2 * t), "blocked": 0.04 * math.sin(TAU * 6 * t)}.get(anim, 0.0)
    P = S * T(0, 0.64, 0) * RZ(sw)                                                          # the axle, on the capitals
    sc.add(sd_cylinder(0.32, 0.012), LEAD_MAT, S * T(0, 0.64, 0) * RZ(math.pi / 2))
    wood = glass(["#6a4020", "#5a3418", "#7a4a28"], anim, t, cell=0.03, seed=5.0, k0=0.3)
    sc.add(sd_box(0.13, 0.03, 0.04, 0.01), wood, P)                                          # the yoke
    for sx in (-1, 1):
        sc.add(sd_box(0.008, 0.05, 0.042, 0.002), LEAD_MAT, P * T(sx * 0.06, -0.02, 0))     # its iron straps
    # the bell
    amber = glass(["#e8a030", "#d88a20", "#f0b848", "#c87818"], anim, t, cell=0.04, k0=0.6)
    gold = glass(GOLD_P, anim, t, cell=0.02, seed=3.0, k0=0.9)
    B = P * T(0, -0.42, 0)
    sc.add(sd_bell(0.33, 0.1, 0.2), amber, B)
    sc.add(sd_sphere(0.1), amber, B * T(0, 0.31, 0), blend=0.03)                               # the shoulder's dome
    for k in range(3):                                                                         # the crown's loops
        sc.add(sd_torus(0.022, 0.007), LEAD_MAT, B * T(0, 0.41, 0) * RY(k * math.pi / 3) * RX(math.pi / 2))
    for y in (0.22, 0.27):                                                                     # the gold bands
        r = 0.1 + 0.1 * (1 - y / 0.33) ** 2.2
        sc.add(sd_torus(r + 0.002, 0.006), gold, B * T(0, y, 0))
    for k in range(14):                                                                        # the inscription between them
        a = TAU * k / 14
        r = 0.1 + 0.1 * (1 - 0.245 / 0.33) ** 2.2 + 0.003
        h = 0.012 if k % 3 else 0.008
        sc.add(sd_box(0.004 + 0.002 * (k % 2), h, 0.004, 0.001), LEAD_MAT, B * RY(a) * T(0, 0.245, r))
    sc.add(sd_torus(0.205, 0.01), gold, B * T(0, 0.025, 0))                                    # the lip
    rz = 0.1 + 0.1 * (1 - 0.13 / 0.33) ** 2.2 + 0.006
    sc.add(sd_box(0.01, 0.05, 0.005, 0.003), gold, B * T(0, 0.12, rz) * RX(-0.35))             # the cross on the waist
    sc.add(sd_box(0.032, 0.01, 0.005, 0.003), gold, B * T(0, 0.135, rz - 0.005) * RX(-0.35))
    cl = -sw * 0.6
    sc.add(sd_capsule((0, 0.25, 0), (0, -0.02, 0), 0.01), LEAD_MAT, B * RZ(cl))              # the clapper
    sc.add(sd_sphere(0.035), LEAD_MAT, B * RZ(cl) * T(0, -0.03, 0))
    return sc


# ─── 5 · Gargouille ─────────────────────────────────────────────────────────
def gargoyle(anim, t, side):
    """The gargoyle crouching on its stone cornice, leaning out over the edge, grey
    and violet glass: forepaws clawing the edge, a ridge of spines down its back,
    bat wings half folded (violet membranes on stone-grey bones), curved horns,
    pointed ears, a brow ridge over eyes alight, the jaws wide open with fangs, a
    tail ending in a barb hanging over the side; a gargoyle is a spout, so rain
    drips from its jaws. idle: it turns its head, its eyes smoulder, drops fall
    (three at a time, nine per loop); win: its wings open wide, eyes blazing, a stream pours;
    blocked: it hunches, the eyes die down."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    stone = glass(STONE_P, anim, t, cell=0.035, k0=0.35)
    dark = glass(["#3a4a58", "#4a4a5a", "#2a3a48"], anim, t, cell=0.03, seed=4.0, k0=0.3)
    # the cornice: a block with a moulded top
    sc.add(sd_box(0.17, 0.06, 0.2, 0.01), STONE_MAT, S * T(0, 0.06, -0.04))
    sc.add(sd_box(0.19, 0.014, 0.215, 0.006), STONE_MAT, S * T(0, 0.125, -0.04))
    sc.add(sd_cylinder(0.17, 0.016), STONE_MAT, S * T(0, 0.1, 0.165) * RZ(math.pi / 2))
    hunch = 0.15 if anim == "blocked" else 0.0
    B = S * T(0, 0.26 - 0.02 * hunch, -0.01) * RX(0.4 + hunch)                                # the body, leaning forward
    sc.add(sd_ellipsoid(0.1, 0.095, 0.14), stone, B, blend=0.04)
    sc.add(sd_ellipsoid(0.085, 0.08, 0.07), stone, B * T(0, 0.01, 0.1), blend=0.04)          # the chest
    for k in range(6):                                                                         # the spines down its back
        sc.add(sd_cone(0.022 - 0.002 * k, 0.014, 0.001), dark, B * T(0, 0.09 - 0.004 * k, 0.07 - 0.04 * k) * RX(-0.5))
    for sx in (-1, 1):
        sc.add(sd_ellipsoid(0.055, 0.075, 0.075), stone, S * T(sx * 0.09, 0.2, -0.08), blend=0.03)   # the haunches
        chain(sc, [(sx * 0.1, 0.18, -0.05), (sx * 0.11, 0.15, 0.0)], [0.03, 0.025], stone, S, blend=0.02)
        sc.add(sd_ellipsoid(0.03, 0.015, 0.04), dark, S * T(sx * 0.11, 0.145, 0.02), blend=0.01)     # the hind feet
        # the forelegs, the paws gripping the edge, claws over it
        chain(sc, [(sx * 0.07, 0.27, 0.08), (sx * 0.085, 0.2, 0.13), (sx * 0.08, 0.15, 0.155)], [0.028, 0.024, 0.02], stone, S, blend=0.02)
        for c in (-1, 0, 1):
            sc.add(sd_cone(0.014, 0.006, 0.001), dark, S * T(sx * 0.08 + c * 0.014, 0.125, 0.18) * RX(-0.2))
    # the head: skull, brow, snout, open jaw, fangs, horns, ears, eyes
    turn = {"idle": 0.3 * math.sin(TAU * t), "blocked": 0.06 * math.sin(TAU * 5 * t)}.get(anim, 0.0)
    gape = {"win": 0.6, "blocked": 0.15}.get(anim, 0.4)
    sc.add(sd_capsule((0, 0.3, 0.08), (0, 0.33, 0.17), 0.045), stone, S, blend=0.03)             # the neck
    H = S * T(0, 0.34, 0.2) * RY(turn) * RX(0.15 + hunch)
    sc.add(sd_ellipsoid(0.065, 0.055, 0.065), stone, H, blend=0.02)
    sc.add(sd_ellipsoid(0.06, 0.018, 0.03), dark, H * T(0, 0.03, 0.04), blend=0.01)             # the brow ridge
    sc.add(sd_ellipsoid(0.045, 0.028, 0.065), stone, H * T(0, -0.005, 0.07), blend=0.02)        # the snout
    for sx in (-1, 1):
        sc.add(sd_sphere(0.008), dark, H * T(sx * 0.015, 0.01, 0.13))                         # the nostrils
    J = H * T(0, -0.03, 0.02) * RX(gape)
    sc.add(sd_ellipsoid(0.04, 0.014, 0.07), stone, J * T(0, -0.01, 0.05), blend=0.01)           # the lower jaw
    sc.add(sd_ellipsoid(0.035, 0.012, 0.055), glass(["#801820", "#601018"], anim, t, cell=0.02, k0=0.4), H * T(0, -0.035, 0.06), blend=0.0)   # the throat
    fang = glass(["#f4ead2", "#ffffff"], anim, t, cell=0.015, seed=6.0, k0=0.5)
    for x in (-0.03, -0.012, 0.012, 0.03):
        L = 0.02 if abs(x) > 0.02 else 0.012
        sc.add(sd_cone(L / 2, 0.006, 0.001), fang, H * T(x, -0.03 - L / 2, 0.11 - abs(x) * 0.6) * RX(math.pi))   # upper fangs
        sc.add(sd_cone(L / 2.5, 0.005, 0.001), fang, J * T(x * 0.9, 0.0 + L / 2.5, 0.09 - abs(x) * 0.6))         # lower fangs
    ek = {"win": 12.0, "blocked": 1.2}.get(anim, 5.0 + 3.0 * math.sin(TAU * 2 * t))
    for sx in (-1, 1):
        sc.add(sd_sphere(0.014), glow("#ff3a1a", ek), H * T(sx * 0.03, 0.017, 0.06))
        chain(sc, bezier((sx * 0.035, 0.04, 0.0), (sx * 0.07, 0.1, -0.02), (sx * 0.1, 0.1, -0.08), (sx * 0.09, 0.06, -0.12), 5),
              [0.016, 0.013, 0.01, 0.007, 0.003], dark, H, blend=0.008)                         # the horns, curving back
        sc.add(sd_cone(0.025, 0.014, 0.001), stone, H * T(sx * 0.065, 0.02, -0.01) * RZ(-sx * 1.2))   # the ears
    # rain dripping from the jaws (a gargoyle is a spout)
    water = glass(["#8ab8e8", "#b8d8f8", "#6a98d8"], anim, t, cell=0.015, seed=7.0, k0=0.7)
    mouth = (0.0, 0.29, 0.31)
    if anim == "win":
        sc.add(sd_capsule(mouth, (0.0, 0.14, 0.36), 0.007, 0.01), water, S)
    elif anim == "idle":
        for k in range(3):
            ph = (3 * t + k / 3) % 1
            y = mouth[1] - 0.17 * ph * ph
            if y > 0.13:
                sc.add(sd_ellipsoid(0.008, 0.013, 0.008), water, S * T(0, y, mouth[2] + 0.04 * ph))
    # the wings: bat wings, half folded (wide open on a win)
    open_ = {"win": 1.0, "blocked": 0.0}.get(anim, 0.35 + 0.05 * math.sin(TAU * t))
    bone = glass(STONE_P, anim, t, cell=0.03, seed=8.0, k0=0.3)
    membrane = glass(PURPLE_P + ["#3a2a5a"], anim, t, cell=0.035, seed=3.0, k0=0.45)
    for sx in (-1, 1):
        w = S * T(sx * 0.07, 0.32, -0.04) * RY(0.5 * sx if sx > 0 else math.pi - 0.5) * RZ(0.2 + 0.3 * open_) * RX(-0.3 + 0.2 * open_)
        membrane_wing(sc, w, 0.18 + 0.1 * open_, -0.5, 0.9 + 0.5 * open_, 3, bone, membrane, thick=0.005, depth=0.25)
    # the tail, over the side, a barb at its end
    tail = bezier((0, 0.18, -0.16), (0.05, 0.14, -0.27), (0.18, 0.12, -0.2), (0.2, 0.0, -0.12), 6)
    chain(sc, tail, [0.026, 0.022, 0.018, 0.014, 0.011, 0.009], stone, S, blend=0.015)
    sc.add(sd_cone(0.025, 0.02, 0.001), dark, S * T(*tail[-1]) * T(0, -0.02, 0) * RX(math.pi))
    return sc


# ─── 6 · Moine copiste ──────────────────────────────────────────────────────
def monk(anim, t, side):
    """The monk copying a book at his lectern: a brown habit with folds, the darker
    scapular down its front, the white cord with its knots, the hood fallen on his
    shoulders, the tonsure, his face bent over the page; the oak lectern, the open
    book (lines of writing, a red and blue initial), the inkpot, a candle. idle: he
    writes (one slow stroke across the page per loop), the candle flickers; win:
    he lifts his quill and his head, the page shines; blocked: he slumps, head bowed."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    M = S * T(-0.04, 0, -0.06) * RY(0.3)                                                      # turned a little towards his lectern
    habit = glass(["#7a4a28", "#6a3a20", "#8a5a30", "#704220"], anim, t, cell=0.04, k0=0.45)
    dark = glass(["#4a2a14", "#5a3018", "#3a2010"], anim, t, cell=0.035, seed=3.0, k0=0.4)
    skin = glass(SKIN_P, anim, t, cell=0.03, seed=2.0, k0=0.5)
    cord = glass(["#f4ead2", "#e8dcc0"], anim, t, cell=0.02, seed=9.0, k0=0.5)
    # the habit, its folds, the scapular, the cord
    sc.add(sd_cone(0.22, 0.165, 0.09, 0.02), habit, M * T(0, 0.22, 0))
    robe_folds(sc, M, habit, 4, 0.38, 0.04, 0.105, 0.16, spread=0.55)
    sc.add(sd_box(0.04, 0.17, 0.006, 0.004), dark, M * T(0, 0.25, 0.128) * RX(-0.16))
    sc.add(sd_torus(0.1, 0.009), cord, M * T(0, 0.4, 0))
    chain(sc, [(0.05, 0.4, 0.09), (0.06, 0.3, 0.122), (0.065, 0.2, 0.138)], [0.006, 0.006, 0.006], cord, M, blend=0.003)
    for y in (0.33, 0.27, 0.21):
        sc.add(sd_sphere(0.011), cord, M * T(0.06 + 0.005 * (0.33 - y) / 0.06, y, 0.125 + 0.13 * (0.33 - y)))
    sc.add(sd_ellipsoid(0.11, 0.09, 0.09), habit, M * T(0, 0.47, 0), blend=0.04)                # shoulders
    sc.add(sd_torus(0.058, 0.026), dark, M * T(0, 0.555, -0.005))                               # the hood, fallen
    sc.add(sd_ellipsoid(0.075, 0.07, 0.045), dark, M * T(0, 0.53, -0.075) * RX(0.3), blend=0.02)
    # the head, bent over the page: face, tonsure
    bow = {"win": -0.25, "blocked": 0.75}.get(anim, 0.22)
    H = M * T(0, 0.58, 0.0) * RX(bow) * T(0, 0.055, 0.008)
    face(sc, H, anim, t, r=0.058, closed=anim != "win")
    hair = glass(["#5a3a20", "#4a2a18", "#6a4428"], anim, t, cell=0.02, k0=0.35)
    sc.add(sd_torus(0.05, 0.016), hair, H * T(0, 0.022, -0.006) * RX(-0.15))                   # the ring of the tonsure
    # the lectern in front of him: post, feet, the sloping board
    wood = glass(["#5a3418", "#6a4020", "#4a2a14"], anim, t, cell=0.035, seed=4.0, k0=0.28)
    sc.add(sd_box(0.02, 0.19, 0.02, 0.006), wood, M * T(0, 0.2, 0.24))
    sc.add(sd_box(0.09, 0.015, 0.02, 0.006), wood, M * T(0, 0.02, 0.24))
    sc.add(sd_box(0.02, 0.015, 0.09, 0.006), wood, M * T(0, 0.02, 0.24))
    ca, sa = math.cos(-0.55), math.sin(-0.55)
    O = (0.0, 0.42, 0.22)

    def bpt(x, y, z):                                                                          # board coordinates → the monk's
        return (O[0] + x, O[1] + y * ca - z * sa, O[2] + y * sa + z * ca)
    Bd = M * T(*O) * RX(-0.55)
    sc.add(sd_box(0.12, 0.008, 0.085, 0.004), wood, Bd)
    sc.add(sd_box(0.12, 0.016, 0.006, 0.003), wood, Bd * T(0, 0.01, -0.085))                    # the ledge
    # the book: red cover, white pages, lines of lead, the initial
    sc.add(sd_box(0.1, 0.005, 0.072, 0.003), glass(RED_P, anim, t, cell=0.03, seed=6.0, k0=0.55), Bd * T(0, 0.012, 0))
    page = glass(["#f8f4e8", "#f0eadb", "#fbf8f0"], anim, t, cell=0.03, seed=5.0, k0=0.6 if anim == "win" else 0.4)
    for sx in (-1, 1):
        sc.add(sd_box(0.045, 0.004, 0.066, 0.002), page, Bd * T(sx * 0.048, 0.019, 0) * RZ(sx * 0.08))
        for k in range(6):
            if sx > 0 and k < 2 and anim == "idle":
                continue                                                                       # the lines he has not written yet
            sc.add(sd_box(0.034, 0.0015, 0.0025, 0.0), LEAD_MAT, Bd * T(sx * 0.05, 0.023 + 0.0035 * sx, 0.048 - k * 0.019))
    sc.add(sd_box(0.012, 0.002, 0.012, 0.001), glass(RED_P, anim, t, cell=0.015, k0=0.9), Bd * T(-0.075, 0.023, 0.047))
    sc.add(sd_box(0.007, 0.0025, 0.007, 0.001), glass(BLUE_P, anim, t, cell=0.015, k0=0.9), Bd * T(-0.075, 0.024, 0.047))
    if anim == "win":
        sc.add(sd_box(0.09, 0.002, 0.06, 0.0), glow("#fff4d0", 1.5), Bd * T(0, 0.03, 0))
    sc.add(sd_cylinder(0.016, 0.016, 0.004), glass(["#1a2a6a", "#22307a"], anim, t, cell=0.02, k0=0.5), Bd * T(0.105, 0.025, -0.06))   # the inkpot
    # the candle on a bracket at the lectern's far corner
    sc.add(sd_cylinder(0.035, 0.012, 0.003), glass(["#f4ead2", "#efe0b8"], anim, t, cell=0.02, seed=7.0, k0=0.35), M * T(0.15, 0.5, 0.3))
    fl = {"blocked": 0.4}.get(anim, 1.0) * (1 + 0.15 * math.sin(TAU * 3 * t))
    sc.add(sd_ellipsoid(0.01, 0.022 * fl, 0.01), glass(["#ffcf4a", "#ffe28a"], anim, t, cell=0.015, k0=2.0), M * T(0.15, 0.55 + 0.01 * fl, 0.3))
    sc.add(sd_sphere(0.012), glow("#fff2c0", 5.0 * fl), M * T(0.15, 0.545, 0.3))
    sc.add(sd_box(0.035, 0.006, 0.016, 0.003), wood, M * T(0.135, 0.462, 0.3))
    # the arms: the right hand writes, the left holds the page
    if anim == "win":
        hand = (0.13, 0.66, 0.14)
    elif anim == "blocked":
        hand = (0.09, 0.36, 0.12)
    else:
        hand = bpt(0.045 + 0.022 * math.sin(TAU * t), 0.035, 0.012 - 0.008 * math.cos(TAU * t))
    chain(sc, [(0.1, 0.5, 0.0), (0.14, 0.4, 0.07), hand], [0.032, 0.036, 0.03], habit, M, blend=0.02)
    sc.add(sd_sphere(0.022), skin, M * T(*hand))
    hx, hy, hz = hand
    sc.add(sd_capsule((hx, hy, hz), (hx + 0.03, hy + 0.11, hz - 0.03), 0.003), LEAD_MAT, M)    # the quill: shaft and vane
    sc.add(sd_capsule((hx + 0.008, hy + 0.035, hz - 0.008), (hx + 0.03, hy + 0.11, hz - 0.03), 0.012, 0.004),
           glass(WHITE_P, anim, t, cell=0.02, seed=8.0, k0=0.6), M)
    left = bpt(-0.06, 0.03, -0.03) if anim != "blocked" else (-0.09, 0.36, 0.12)
    chain(sc, [(-0.1, 0.5, 0.0), (-0.13, 0.4, 0.07), left], [0.032, 0.036, 0.03], habit, M, blend=0.02)
    sc.add(sd_sphere(0.022), skin, M * T(*left))
    return sc


# ─── 7 · Chevalier ──────────────────────────────────────────────────────────
def sd_heater(w, h, th):
    """A heater shield in the xy plane: the flat top at y = h/3, width 2w, the sides
    two arcs meeting at a point h below the top; thickness 2·th along z."""
    yt = h / 3
    R = (w * w + h * h) / (2 * w)

    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        d1 = torch.sqrt((x - (R - w)) ** 2 + (y - yt) ** 2) - R
        d2 = torch.sqrt((x + (R - w)) ** 2 + (y - yt) ** 2) - R
        return torch.maximum(torch.maximum(torch.maximum(d1, d2), y - yt), z.abs() - th)
    return f


def lis(sc, X, mat, k=1.0):
    """A small fleur-de-lis in the xy plane of X (k: its size, 1 = 0.05 high)."""
    sc.add(sd_ellipsoid(0.009 * k, 0.022 * k, 0.004), mat, X * T(0, 0.008 * k, 0))
    for sx in (-1, 1):
        sc.add(sd_ellipsoid(0.007 * k, 0.017 * k, 0.004), mat, X * T(sx * 0.013 * k, 0.0, 0) * RZ(-sx * 0.7))
    sc.add(sd_box(0.016 * k, 0.003 * k, 0.004, 0.0), mat, X * T(0, -0.008 * k, 0))
    sc.add(sd_ellipsoid(0.005 * k, 0.008 * k, 0.004), mat, X * T(0, -0.017 * k, 0))


def knight(anim, t, side):
    """The knight of the windows: mail of silver-blue glass (legs with knee
    pieces, arms), the red surcoat with folds and a white cross, a gold belt, a
    blue cloak behind, the great helm with its eye slit, breathing holes and a gold
    crest, the heater shield of blue glass with three gold lilies and a gold rim,
    the long sword; idle: he shifts his sword (one cycle per loop), the cloak
    stirs; win: he raises the sword high; blocked: the sword droops, the shield sinks."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    mail = glass(["#a8b8d0", "#8898b8", "#c0cce0", "#7888a8"], anim, t, cell=0.03, k0=0.4)
    plate = glass(["#d0dcef", "#b8c8e0", "#e4ecf8"], anim, t, cell=0.03, seed=9.0, k0=0.45)
    red = glass(RED_P, anim, t, cell=0.04, seed=2.0, k0=0.6)
    fold = glass(["#901418", "#a81820", "#801010"], anim, t, cell=0.035, seed=10.0, k0=0.55)
    white = glass(WHITE_P, anim, t, cell=0.025, seed=11.0, k0=0.6)
    gold = glass(GOLD_P, anim, t, cell=0.02, seed=12.0, k0=0.9)
    blue = glass(BLUE_P, anim, t, cell=0.035, seed=3.0, k0=0.65)
    # the legs: mail, knee pieces, feet
    for sx in (-1, 1):
        sc.add(sd_capsule((sx * 0.05, 0.03, 0), (sx * 0.05, 0.27, 0), 0.034, 0.038), mail, S, blend=0.01)
        sc.add(sd_sphere(0.026), plate, S * T(sx * 0.05, 0.14, 0.03))
        sc.add(sd_ellipsoid(0.03, 0.022, 0.05), plate, S * T(sx * 0.05, 0.022, 0.02), blend=0.01)
    # the cloak behind, the surcoat, its folds and cross, the belt
    sway = 0.04 * math.sin(TAU * t) if anim == "idle" else 0.0
    sc.add(sd_ellipsoid(0.15, 0.26, 0.03), blue, S * T(0, 0.33, -0.1) * RX(-0.12 - sway))
    sc.add(sd_cone(0.2, 0.15, 0.1, 0.02), red, S * T(0, 0.35, 0))
    robe_folds(sc, S, fold, 4, 0.3, 0.17, 0.12, 0.145, spread=0.55, rad=(0.012, 0.022))
    sc.add(sd_box(0.016, 0.09, 0.005, 0.003), white, S * T(0, 0.44, 0.115) * RX(-0.12))       # the cross
    sc.add(sd_box(0.055, 0.016, 0.005, 0.003), white, S * T(0, 0.47, 0.111) * RX(-0.12))
    sc.add(sd_torus(0.122, 0.011), gold, S * T(0, 0.32, 0) * RX(0.12))                        # the belt
    sc.add(sd_box(0.016, 0.014, 0.006, 0.003), gold, S * T(0, 0.307, 0.128))
    sc.add(sd_ellipsoid(0.125, 0.075, 0.09), mail, S * T(0, 0.52, 0), blend=0.03)              # the shoulders
    for sx in (-1, 1):
        sc.add(sd_ellipsoid(0.05, 0.03, 0.05), plate, S * T(sx * 0.11, 0.565, 0) * RZ(-sx * 0.4), blend=0.005)
    # the great helm: a flat-topped drum, the eye slit, holes, the gold crest
    Hm = S * T(0, 0.66, 0)
    sc.add(sd_cylinder(0.075, 0.068, 0.02), plate, Hm)
    sc.add(sd_box(0.004, 0.07, 0.004, 0.002), gold, Hm * T(0, -0.005, 0.068))                # the cross-shaped brace
    for k in range(9):                                                                          # the eye slit: a thin line round the front
        a = -0.8 + 0.2 * k
        sc.add(sd_box(0.007, 0.0035, 0.004, 0.001), LEAD_MAT, Hm * RY(a) * T(0, 0.015, 0.067))
    for k in range(5):
        a = -0.6 + 0.3 * k
        if abs(a) > 0.1:
            sc.add(sd_sphere(0.005), LEAD_MAT, Hm * T(0.066 * math.sin(a), -0.035, 0.066 * math.cos(a)))   # breathing holes
    sc.add(sd_torus(0.066, 0.008), gold, Hm * T(0, 0.074, 0))                                  # a gold circlet
    for k in range(5):                                                                          # the crest: a gold crown of points
        a = TAU * k / 5
        sc.add(sd_cone(0.02, 0.012, 0.001), gold, Hm * T(0.05 * math.sin(a), 0.095, 0.05 * math.cos(a)))
    # the shield on the left arm: a heater, blue, three gold lilies, a gold rim
    drop = 0.08 if anim == "blocked" else 0.0
    chain(sc, [(-0.12, 0.55, 0), (-0.16, 0.45, 0.04), (-0.13, 0.4 - drop, 0.1)], [0.03, 0.028, 0.025], mail, S, blend=0.015)
    Sh = S * T(-0.15, 0.4 - drop, 0.13) * RY(-0.4)
    sc.add(sd_heater(0.09, 0.15, 0.008), gold, Sh)
    sc.add(sd_heater(0.08, 0.135, 0.011), blue, Sh * T(0, -0.006, 0))
    for (x, y) in ((-0.035, 0.03), (0.035, 0.03), (0, -0.035)):
        lis(sc, Sh * T(x, y - 0.03, 0.012), gold, 0.9)
    # the sword in the right hand
    if anim == "win":
        hy, a = 0.62, 0.0
    elif anim == "blocked":
        hy, a = 0.38, -2.6
    else:
        hy, a = 0.42, -0.35 + 0.1 * math.sin(TAU * t)
    hand = (0.17, hy, 0.06)
    chain(sc, [(0.12, 0.55, 0), (0.17, 0.47, 0.03), hand], [0.03, 0.028, 0.025], mail, S, blend=0.015)
    sc.add(sd_sphere(0.024), plate, S * T(*hand))
    Sw = S * T(*hand) * RZ(a)
    sc.add(sd_box(0.012, 0.23, 0.004, 0.002), glass(["#e8f0ff", "#c8d8f0", "#ffffff"], anim, t, cell=0.025, k0=0.7), Sw * T(0, 0.27, 0))
    sc.add(sd_cone(0.02, 0.012, 0.001), glass(["#e8f0ff", "#ffffff"], anim, t, cell=0.02, k0=0.7), Sw * T(0, 0.52, 0))
    sc.add(sd_box(0.003, 0.17, 0.0045, 0.0), LEAD_MAT, Sw * T(0, 0.24, 0))                   # the fuller
    sc.add(sd_box(0.055, 0.01, 0.01, 0.003), GILT, Sw * T(0, 0.035, 0))
    sc.add(sd_capsule((0, 0.03, 0), (0, -0.03, 0), 0.008), LEAD_MAT, Sw)
    sc.add(sd_sphere(0.016), GILT, Sw * T(0, -0.04, 0))
    return sc


# ─── 8 · Licorne ────────────────────────────────────────────────────────────
def unicorn(anim, t, side):
    """The unicorn rearing, white and lilac glass: hindquarters, barrel and chest,
    hind legs planted, forelegs folded, gold hooves, the arched neck with a gold
    collar, the head with muzzle, nostrils, eyes, ears and a goat's beard, the
    twisted golden horn, a mane, forelock and long tail of rainbow locks; idle: it
    tosses its head, the tail sways (one cycle each per loop); win: it rears high
    and paws the air (two strokes per loop); blocked: it drops low, head down."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t) * RY(1.0)                                                               # seen in three-quarter profile
    coat = glass(["#f4f0fa", "#e4dcf4", "#d0c4ec", "#f8f4fc"], anim, t, cell=0.04, k0=0.3)
    gold = glass(GOLD_P, anim, t, cell=0.02, seed=6.0, k0=0.9)
    rainbow = glass(["#e83a4a", "#f0a030", "#f0e040", "#40c060", "#3a7ae0", "#9a4ad0"], anim, t, cell=0.025, seed=4.0, k0=0.8)
    rear = {"win": 0.75, "blocked": 0.35}.get(anim, 0.55)
    O = (0.0, 0.17, -0.12)
    cr, sr = math.cos(rear), math.sin(rear)

    def rp(x, y, z):                                                                            # the body's frame → the stand's
        return (O[0] + x, O[1] + y * cr + z * sr, O[2] - y * sr + z * cr)
    R = S * T(*O) * RX(-rear)
    # the body
    sc.add(sd_ellipsoid(0.085, 0.095, 0.1), coat, R * T(0, 0.0, 0.0), blend=0.04)               # hindquarters
    sc.add(sd_ellipsoid(0.08, 0.085, 0.15), coat, R * T(0, 0.01, 0.13), blend=0.05)             # the barrel
    sc.add(sd_ellipsoid(0.075, 0.095, 0.08), coat, R * T(0, 0.025, 0.25), blend=0.05)           # the chest
    # the hind legs, planted: thigh, hock, cannon, gold hoof
    for sx in (-1, 1):
        hip = rp(sx * 0.06, -0.01, 0.0)
        hock = (sx * 0.065, 0.1, -0.2)
        hoof = (sx * 0.06, 0.02, -0.12)
        chain(sc, [hip, (sx * 0.065, hip[1] - 0.05, hip[2] - 0.07), hock, hoof], [0.045, 0.03, 0.02, 0.018], coat, S, blend=0.02)
        sc.add(sd_cylinder(0.014, 0.022, 0.004), gold, S * T(hoof[0], 0.014, hoof[2]))
    # the forelegs, folded (pawing on a win)
    paw = 0.35 * math.sin(TAU * 2 * t) if anim == "win" else 0.0
    for sx in (-1, 1):
        ph = paw * sx
        sh = (sx * 0.05, -0.04, 0.27)
        knee = (sx * 0.055, -0.13 + 0.03 * ph, 0.33 + 0.04 * ph)
        foot = (sx * 0.055, -0.1 + 0.06 * ph, 0.42 + 0.02 * ph)
        chain(sc, [sh, knee, foot], [0.03, 0.02, 0.017], coat, R, blend=0.02)
        sc.add(sd_cylinder(0.013, 0.02, 0.004), gold, R * T(*foot) * RX(1.2))
    # the neck, arched, and its collar
    nod = {"idle": 0.18 * math.sin(TAU * t), "blocked": 0.5}.get(anim, 0.0)
    chain(sc, [(0, 0.05, 0.28), (0, 0.13, 0.33), (0, 0.19, 0.35)], [0.055, 0.047, 0.042], coat, R, blend=0.04)
    sc.add(sd_torus(0.047, 0.009), gold, R * T(0, 0.12, 0.33) * RX(-0.5))
    Hd = R * T(0, 0.21, 0.36) * RX(rear + 0.55 + nod)                                         # the head, looking ahead and down
    sc.add(sd_ellipsoid(0.046, 0.05, 0.06), coat, Hd, blend=0.02)
    sc.add(sd_ellipsoid(0.034, 0.036, 0.065), coat, Hd * T(0, -0.012, 0.065), blend=0.03)          # the muzzle
    for sx in (-1, 1):
        sc.add(sd_sphere(0.007), LEAD_MAT, Hd * T(sx * 0.016, -0.005, 0.115))                  # nostrils
        sc.add(sd_ellipsoid(0.008, 0.01, 0.006), LEAD_MAT, Hd * T(sx * 0.036, 0.015, 0.02))    # eyes
        sc.add(sd_cone(0.018, 0.012, 0.002), coat, Hd * T(sx * 0.025, 0.05, -0.02) * RX(-0.4) * RZ(-sx * 0.3))   # ears
    sc.add(sd_box(0.015, 0.003, 0.004, 0.0), LEAD_MAT, Hd * T(0, -0.03, 0.095))                # the mouth
    chain(sc, [(0, -0.04, 0.06), (0, -0.065, 0.055), (0, -0.08, 0.045)], [0.01, 0.008, 0.004], rainbow, Hd, blend=0.01)   # the beard
    # the horn: a gold cone, its twist marked by lead rings
    Hn = Hd * T(0, 0.05, 0.035) * RX(-1.05)
    sc.add(sd_cone(0.065, 0.014, 0.002), gold, Hn * T(0, 0.065, 0))
    for k in range(5):
        sc.add(sd_torus(0.013 - 0.0022 * k, 0.0018), LEAD_MAT, Hn * T(0, 0.012 + 0.024 * k, 0) * RX(0.35))
    # the mane down the neck's crest, the forelock, the tail: rainbow locks
    for k in range(8):
        u = k / 7
        y, z = 0.05 + 0.15 * u, 0.24 + 0.08 * u                                                # along the back of the neck
        sx = 1 if k % 2 else -1
        w = 0.015 * math.sin(TAU * t) if anim == "idle" else 0.0
        sc.add(sd_capsule((0, y, z), (sx * 0.03, y - 0.05 + w, z - 0.07), 0.017, 0.006), rainbow, R, blend=0.01)
    sc.add(sd_capsule((0, 0.04, 0.0), (0.0, 0.02, 0.05), 0.014, 0.005), rainbow, Hd, blend=0.01)    # the forelock
    swing = 0.06 * math.sin(TAU * t) if anim == "idle" else 0.0
    root = rp(0, 0.05, -0.09)
    for k in range(3):
        dx = (k - 1) * 0.025
        pts = bezier(root, (root[0] + dx, root[1] + 0.03, root[2] - 0.1), (dx + swing, 0.18, -0.34), (dx * 2 + swing * 1.5, 0.03, -0.33 - 0.02 * k), 6)
        chain(sc, pts, [0.018, 0.017, 0.016, 0.014, 0.011, 0.006], rainbow, S, blend=0.01)
    return sc


# ─── 9 · Ange ───────────────────────────────────────────────────────────────
def robe_folds(sc, S, mat, n, y0, y1, r0, r1, spread=0.45, rad=(0.016, 0.028)):
    """Folds down the front of a cone robe (radius r0 at height y0, r1 at y1): n
    ridges fanning out from the waist to the hem, on the robe's surface."""
    for k in range(n):
        phi = (k - (n - 1) / 2) * spread
        pts = []
        for j in range(4):
            u = j / 3
            y = y0 + (y1 - y0) * u
            r = (r0 + (r1 - r0) * u) * 0.97
            ph = phi * (0.6 + 0.4 * u)
            pts.append((r * math.sin(ph), y, r * math.cos(ph)))
        chain(sc, pts, [rad[0] + (rad[1] - rad[0]) * j / 3 for j in range(4)], mat, S, blend=0.02)


def face(sc, H, anim, t, r=0.062, closed=False):
    """The head at H (radius r, looking along +z) as a window shows a face: ONE
    piece of pale glass (no lead across it), the features painted on it in dark
    grisaille and lying on its surface: almond eyes of white glass with a dark
    iris (closed: a lid line), brows almost level, a little higher at the nose
    (a calm look: brows sloping down to the nose read as anger), the line of the
    nose, the mouth."""
    pale = pane(SKIN_P[0], anim, t, 0.5)
    sc.add(sd_sphere(r), pale, H)

    def on(x, y, lift=0.0):                                             # a frame on the sphere at (x, y), z outwards
        z = math.sqrt(max(r * r - x * x - y * y, 1e-6))
        k = (r + lift) / r
        return H * T(x * k, y * k, z * k) * RY(math.atan2(x, z)) * RX(-math.atan2(y, math.hypot(x, z)))
    paint = LEAD_MAT
    white = pane(WHITE_P[0], anim, t, 0.7)
    for sx in (-1, 1):
        ex, ey = sx * 0.37 * r, 0.12 * r
        if closed:
            sc.add(sd_ellipsoid(0.2 * r, 0.035 * r, 0.04 * r), paint, on(ex, ey - 0.02 * r) * RZ(sx * 0.12))
        else:
            sc.add(sd_ellipsoid(0.21 * r, 0.12 * r, 0.04 * r), white, on(ex, ey))
            sc.add(sd_sphere(0.075 * r), paint, on(ex, ey, 0.025 * r))
        sc.add(sd_ellipsoid(0.24 * r, 0.035 * r, 0.04 * r), paint, on(sx * 0.38 * r, 0.4 * r) * RZ(-sx * 0.06))   # the brow, almost level, a little higher at the nose
    sc.add(sd_ellipsoid(0.035 * r, 0.22 * r, 0.04 * r), paint, on(0.04 * r, -0.08 * r))   # the nose
    sc.add(sd_ellipsoid(0.16 * r, 0.035 * r, 0.04 * r), paint, on(0.0, -0.42 * r))       # the mouth


def curls(sc, H, mat, r=0.062, n=9, rows=3, low=-0.2):
    """Hair of curls round the head at H: rows of small balls over the crown,
    the sides and the back, the face left clear."""
    for row in range(rows):
        el = 1.25 - row * (1.25 - low) / max(rows - 1, 1)                 # elevation, from the crown down
        m = max(3, int(n * math.cos(el) + 2))
        for k in range(m):
            az = math.pi * 0.5 + math.pi * k / max(m - 1, 1)                  # the sides and the back (az 0 = +z, the face)
            x, y, z = math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)
            sc.add(sd_sphere(0.024), mat, H * T(x * r, y * r + 0.004, z * r - 0.004), blend=0.008)
    sc.add(sd_ellipsoid(r * 1.02, r * 0.5, r * 0.9), mat, H * T(0, r * 0.6, -0.015), blend=0.01)    # the crown


def angel(anim, t, side):
    """The angel standing in a long robe of pale blue glass with folds, gold at
    its hem, collar and belt, a red stole, hands joined in prayer, a face with
    closed eyes and golden curls, the gold halo disc, great wings of white and
    gold feathers rising behind; idle: the wings breathe (one cycle per loop);
    win: they open wide and the halo blazes; blocked: the wings fold, the head bows."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    robe = glass(["#a8c8f0", "#88b0e8", "#c8dcf8", "#98bcec"], anim, t, cell=0.04, k0=0.4)
    fold = glass(["#6a98d8", "#5a88c8", "#7aa8e0"], anim, t, cell=0.035, seed=6.0, k0=0.4)
    skin = glass(SKIN_P, anim, t, cell=0.03, seed=2.0, k0=0.5)
    gold = glass(GOLD_P, anim, t, cell=0.025, seed=5.0, k0=0.7)
    stole = glass(RED_P, anim, t, cell=0.03, seed=7.0, k0=0.6)
    sc.add(sd_cone(0.26, 0.17, 0.075, 0.02), robe, S * T(0, 0.27, 0))                           # the robe
    robe_folds(sc, S, fold, 5, 0.44, 0.04, 0.09, 0.16)
    sc.add(sd_torus(0.163, 0.014), gold, S * T(0, 0.035, 0))                                     # the hem
    sc.add(sd_torus(0.094, 0.012), gold, S * T(0, 0.44, 0))                                      # the belt
    for sx in (-1, 1):                                                                           # the stole, down the front
        chain(sc, [(sx * 0.03, 0.6, 0.05), (sx * 0.035, 0.45, 0.092), (sx * 0.042, 0.24, 0.128)], [0.011, 0.011, 0.012], stole, S, blend=0.005)
    sc.add(sd_ellipsoid(0.1, 0.075, 0.08), robe, S * T(0, 0.55, 0), blend=0.04)                  # the chest
    sc.add(sd_torus(0.045, 0.011), gold, S * T(0, 0.615, 0.005))                                 # the collar
    # the sleeves, wide, and the hands joined in prayer before the chest
    for sx in (-1, 1):
        chain(sc, [(sx * 0.1, 0.57, 0.0), (sx * 0.11, 0.47, 0.06), (sx * 0.03, 0.51, 0.12)], [0.032, 0.036, 0.042], robe, S, blend=0.02)
        sc.add(sd_ellipsoid(0.012, 0.04, 0.022), skin, S * T(sx * 0.011, 0.56, 0.14) * RX(-0.25), blend=0.004)
    bow = 0.3 if anim == "blocked" else (-0.15 if anim == "win" else 0.0)
    H = S * T(0, 0.635, 0.0) * RX(bow) * T(0, 0.055, 0.005)
    sc.add(sd_cylinder(0.03, 0.026), skin, S * T(0, 0.635, 0.0))                                # the neck
    face(sc, H, anim, t, closed=True)                                                            # the head
    curls(sc, H, glass(GOLD_P, anim, t, cell=0.025, seed=8.0, k0=0.6))
    # the halo: a gold disc behind the head, a ring of light on its rim
    hk = 9.0 if anim == "win" else (1.5 if anim == "blocked" else 3.5)
    sc.add(sd_cylinder(0.004, 0.11, 0.002), gold, H * T(0, 0.02, -0.065) * RX(math.pi / 2))
    sc.add(sd_torus(0.11, 0.008), glow("#ffd860", hk), H * T(0, 0.02, -0.065) * RX(math.pi / 2))
    # the wings: rising behind the shoulders, feathers of white and gold
    spread = {"idle": 1.05 + 0.1 * math.sin(TAU * t), "win": 0.6 + 0.08 * math.sin(TAU * 2 * t), "blocked": 1.4}.get(anim, 1.05)
    wing = glass(["#ffffff", "#f4ecd0", "#f0d890", "#e8f0fc", "#ffe8a0"], anim, t, cell=0.03, seed=3.0, k0=0.45)
    cover = glass(["#f2c040", "#ffd860", "#fbf3d8"], anim, t, cell=0.025, seed=4.0, k0=0.55)
    for sx in (-1, 1):
        feather_wing(sc, S * T(sx * 0.06, 0.57, -0.08) * RY(-sx * 0.35) * RZ(sx * spread), sx, wing, cover,
                     length=0.34, n=9, chord=0.12, root=-2.4, tip=-0.45)
    return sc


# ─── 10 · Lion ailé ─────────────────────────────────────────────────────────
def winged_lion(anim, t, side):
    """The winged lion of Saint Mark, sitting upright as in heraldry: golden glass,
    a mane of flame-shaped locks in orange and red round the face, a muzzle, ears,
    eyes of lead, great wings of gold and white feathers, a tail ending in a tuft,
    the halo, one forepaw on the open book. idle: the wings breathe and the tail
    swings (one cycle each per loop); win: it roars, head up, wings wide;
    blocked: the wings fold low."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    fur = glass(GOLD_P, anim, t, cell=0.04, k0=0.55)
    dark_fur = glass(["#c87818", "#b86810", "#d88a20"], anim, t, cell=0.03, seed=5.0, k0=0.5)
    mane = glass(["#e87a20", "#d85a18", "#f09030", "#c84a10", "#b83a10"], anim, t, cell=0.03, seed=2.0, k0=0.6)
    B = S * T(0, 0.0, -0.02)
    # the body: haunches, hind paws, the upright chest, the forelegs and paws
    sc.add(sd_ellipsoid(0.13, 0.12, 0.15), fur, B * T(0, 0.13, -0.06), blend=0.05)
    for sx in (-1, 1):
        sc.add(sd_ellipsoid(0.045, 0.03, 0.07), dark_fur, B * T(sx * 0.11, 0.03, 0.05), blend=0.02)
    sc.add(sd_ellipsoid(0.095, 0.16, 0.095), fur, B * T(0, 0.31, 0.05) * RX(0.25), blend=0.05)
    for sx in (-1, 1):
        sc.add(sd_capsule((sx * 0.055, 0.3, 0.1), (sx * 0.06, 0.05 + (0.04 if sx > 0 else 0.0), 0.14 + (0.06 if sx > 0 else 0.0)), 0.03, 0.026), fur, B, blend=0.02)
        sc.add(sd_ellipsoid(0.035, 0.022, 0.045), dark_fur, B * T(sx * 0.06, 0.03 + (0.045 if sx > 0 else 0.0), 0.16 + (0.06 if sx > 0 else 0.0)), blend=0.015)
    # the open book under the right forepaw: red cover, two white pages
    Bk = B * T(0.05, 0.02, 0.24) * RX(0.15)
    sc.add(sd_box(0.1, 0.008, 0.07, 0.004), glass(RED_P, anim, t, cell=0.03, seed=6.0, k0=0.6), Bk)
    for sz in (-1, 1):
        sc.add(sd_box(0.045, 0.005, 0.064, 0.002), glass(WHITE_P, anim, t, cell=0.03, seed=7.0, k0=0.5), Bk * T(sz * 0.048, 0.012, 0) * RZ(-sz * 0.1))
    # the head: skull, muzzle, nose, eyes, ears; raised to roar on a win
    roar = 0.35 if anim == "win" else 0.0
    Hd = B * T(0, 0.52, 0.1) * RX(-roar)
    sc.add(sd_ellipsoid(0.07, 0.068, 0.072), fur, Hd, blend=0.02)
    sc.add(sd_ellipsoid(0.045, 0.034, 0.042), fur, Hd * T(0, -0.025, 0.06), blend=0.02)                  # the muzzle
    sc.add(sd_ellipsoid(0.02, 0.012, 0.012), LEAD_MAT, Hd * T(0, -0.005, 0.1))                          # the nose
    sc.add(sd_box(0.025, 0.004, 0.01, 0.002), LEAD_MAT, Hd * T(0, -0.05 - 0.02 * (roar > 0), 0.085))   # the mouth line
    for sx in (-1, 1):
        sc.add(sd_ellipsoid(0.014, 0.009, 0.006), glow("#ffe080", 4.0), Hd * T(sx * 0.03, 0.022, 0.065))  # the eyes, alight
        sc.add(sd_ellipsoid(0.018, 0.005, 0.006), LEAD_MAT, Hd * T(sx * 0.03, 0.035, 0.066) * RZ(sx * 0.3))   # the brows
        sc.add(sd_sphere(0.022), dark_fur, Hd * T(sx * 0.058, 0.058, -0.01), blend=0.01)               # the ears
    # the mane: two rings of flame-shaped locks round the face, and locks falling on the chest
    for ring, (n, r0, r1, rad, z) in enumerate(((14, 0.05, 0.16, 0.034, -0.02), (11, 0.04, 0.12, 0.03, 0.02))):
        for k in range(n):
            a = TAU * (k + 0.5 * ring) / n
            if ring == 1 and abs(math.sin(a)) < 0.3 and math.cos(a) < 0:   # keep the face clear below
                continue
            c, s_ = math.cos(a), math.sin(a)
            sc.add(sd_capsule((r0 * c, r0 * s_, z), (r1 * c, r1 * s_ + 0.01, z - 0.04), rad, 0.006), mane, Hd, blend=0.012)
    for k in range(5):
        x = -0.06 + 0.03 * k
        sc.add(sd_capsule((x, -0.06, 0.02), (x * 1.2, -0.17 - 0.02 * (k % 2), 0.03), 0.026, 0.006), mane, Hd, blend=0.012)
    sc.add(sd_torus(0.15, 0.012), glow("#ffd860", 8.0 if anim == "win" else 3.5), Hd * T(0, 0.05, -0.09) * RX(math.pi / 2 - 0.15))   # the halo
    # the tail: an S curve from the haunches, a tuft at its end
    swing = 0.35 * math.sin(TAU * t) if anim == "idle" else (0.6 if anim == "win" else 0.0)
    tip = (0.2 + 0.05 * math.sin(swing), 0.3 + 0.04 * math.cos(swing), -0.24)
    chain(sc, bezier((0.06, 0.1, -0.18), (0.22, 0.05, -0.24), (0.12 + 0.08 * math.sin(swing), 0.22, -0.3), tip, 6),
          [0.02, 0.018, 0.016, 0.014, 0.012, 0.011], fur, B, blend=0.015)
    sc.add(sd_ellipsoid(0.03, 0.045, 0.03), mane, B * T(*tip) * T(0, 0.03, 0), blend=0.01)
    # the wings: long feathers of gold and white glass, rising behind the shoulders
    spread = {"idle": 0.65 + 0.12 * math.sin(TAU * t), "win": 1.05, "blocked": 0.3}.get(anim, 0.65)
    wing = glass(["#ffd860", "#f2c040", "#f4f6fa", "#e8a828", "#fbf3d8"], anim, t, cell=0.035, seed=4.0, k0=0.6)
    wings(sc, B * T(0, 0.42, -0.04), wing, span=0.36, spread=spread, feathers=6)
    return sc


# ─── 11 · Archange ──────────────────────────────────────────────────────────
def archangel(anim, t, side):
    """The archangel Michael: a red robe with folds, the gold breastplate with its
    plates leaded, shoulder pieces and a skirt of gold and red straps, a blue cloak
    behind, auburn curls and the halo disc, great crimson and gold wings, the
    round shield with its cross on the left arm, the flaming sword raised in the
    right hand, the green dragon coiled at his feet. idle: the flames lick up the
    blade (two waves per loop), the dragon's eye smoulders; win: the sword sweeps up
    blazing, the wings open; blocked: the sword falls, the flames die."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    robe = glass(RED_P, anim, t, cell=0.04, k0=0.55)
    fold = glass(["#901418", "#a81820", "#801010"], anim, t, cell=0.035, seed=9.0, k0=0.5)
    armour = glass(GOLD_P, anim, t, cell=0.035, seed=2.0, k0=0.6)
    skin = glass(SKIN_P, anim, t, cell=0.03, seed=3.0, k0=0.5)
    cloak = glass(BLUE_P, anim, t, cell=0.045, seed=10.0, k0=0.6)
    # the dragon coiled at his feet, its head rising at the front
    scale_ = glass(["#2a8a4a", "#3aa058", "#1a6a38", "#5ab848"], anim, t, cell=0.03, seed=11.0, k0=0.5)
    coil = [(0.21 * math.sin(a), 0.035, 0.21 * math.cos(a)) for a in [math.radians(d) for d in (-200, -150, -100, -50, 0, 40)]]
    coil.append((0.2, 0.08, 0.2))
    chain(sc, coil, [0.012, 0.022, 0.03, 0.035, 0.035, 0.032, 0.03], scale_, S, blend=0.015)
    Dh = S * T(0.2, 0.1, 0.22) * RY(0.6) * RX(-0.3)
    sc.add(sd_ellipsoid(0.035, 0.03, 0.055), scale_, Dh, blend=0.015)                          # its head
    sc.add(sd_ellipsoid(0.025, 0.012, 0.04), scale_, Dh * T(0, -0.03, 0.03) * RX(0.35), blend=0.006)   # its open jaw
    for sx in (-1, 1):
        sc.add(sd_cone(0.02, 0.008, 0.001), scale_, Dh * T(sx * 0.02, 0.03, -0.02) * RX(-0.7))  # its horns
        sc.add(sd_sphere(0.007), glow("#ff3a1a", 1.5 if anim == "win" else 4.0 + 2.0 * math.sin(TAU * 2 * t)), Dh * T(sx * 0.022, 0.012, 0.03))
    # the robe, the cloak behind
    sc.add(sd_cone(0.22, 0.165, 0.09, 0.02), robe, S * T(0, 0.23, 0))
    robe_folds(sc, S, fold, 4, 0.36, 0.05, 0.1, 0.155, spread=0.5)
    sc.add(sd_torus(0.158, 0.012), armour, S * T(0, 0.04, 0))
    sc.add(sd_ellipsoid(0.15, 0.3, 0.035), cloak, S * T(0, 0.3, -0.11) * RX(-0.1))
    # the skirt of straps, the breastplate, the shoulder pieces
    for k in range(10):
        a = TAU * k / 10
        m = armour if k % 2 == 0 else robe
        sc.add(sd_box(0.022, 0.05, 0.006, 0.004), m, S * RY(a) * T(0, 0.4, 0.105) * RX(0.18))
    sc.add(sd_ellipsoid(0.11, 0.1, 0.085), armour, S * T(0, 0.52, 0), blend=0.03)
    sc.add(sd_box(0.004, 0.08, 0.004, 0.002), LEAD_MAT, S * T(0, 0.52, 0.086))                 # the ridge of the breastplate
    for y in (0.47, 0.44):
        sc.add(sd_torus(0.1 - (0.47 - y), 0.006), LEAD_MAT, S * T(0, y, 0) * RX(0.0))         # its lower plates
    sc.add(sd_torus(0.1, 0.01), armour, S * T(0, 0.455, 0))
    for sx in (-1, 1):
        sc.add(sd_ellipsoid(0.055, 0.035, 0.055), armour, S * T(sx * 0.1, 0.58, 0) * RZ(-sx * 0.35), blend=0.01)
    # the head: face, curls, halo
    up = 1.0 if anim == "win" else 0.0
    H = S * T(0, 0.67, 0.0) * RX(-0.12 * up)
    sc.add(sd_cylinder(0.03, 0.026), skin, S * T(0, 0.62, 0))
    face(sc, H, anim, t, r=0.06)
    curls(sc, H, glass(["#c87818", "#a86010", "#e09030"], anim, t, cell=0.025, seed=12.0, k0=0.5), r=0.06)
    hk = 10.0 if anim == "win" else (1.5 if anim == "blocked" else 4.0)
    sc.add(sd_cylinder(0.004, 0.105, 0.002), armour, H * T(0, 0.02, -0.065) * RX(math.pi / 2))
    sc.add(sd_torus(0.105, 0.008), glow("#ffd860", hk), H * T(0, 0.02, -0.065) * RX(math.pi / 2))
    # the shield on the left arm: white with a red cross, a gold rim
    chain(sc, [(-0.12, 0.57, 0), (-0.15, 0.47, 0.05), (-0.1, 0.42, 0.11)], [0.03, 0.028, 0.025], armour, S, blend=0.015)
    Sh = S * T(-0.13, 0.42, 0.13) * RY(-0.35) * RX(math.pi / 2)
    sc.add(sd_cylinder(0.008, 0.085, 0.004), glass(WHITE_P, anim, t, cell=0.03, seed=13.0, k0=0.45), Sh)
    sc.add(sd_torus(0.085, 0.009), armour, Sh)
    sc.add(sd_box(0.014, 0.012, 0.065, 0.003), glass(RED_P, anim, t, cell=0.02, seed=14.0, k0=0.8), Sh * T(0, 0.006, 0))
    sc.add(sd_box(0.065, 0.012, 0.014, 0.003), glass(RED_P, anim, t, cell=0.02, seed=14.0, k0=0.8), Sh * T(0, 0.006, 0))
    # the sword in the right hand, raised; its blade in flames
    if anim == "win":
        hy, sw = 0.78, -0.1 - 0.35 * math.sin(math.pi * t)
    elif anim == "blocked":
        hy, sw = 0.5, -2.7
    else:
        hy, sw = 0.68, -0.3 + 0.04 * math.sin(TAU * t)
    chain(sc, [(0.12, 0.57, 0), (0.17, 0.6, 0.04), (0.16, hy, 0.07)], [0.03, 0.028, 0.025], armour, S, blend=0.015)
    sc.add(sd_sphere(0.022), skin, S * T(0.16, hy, 0.08))
    Sw = S * T(0.16, hy, 0.08) * RZ(sw)
    sc.add(sd_box(0.012, 0.2, 0.004, 0.002), glass(["#fff0c0", "#ffe8a0", "#ffffff"], anim, t, cell=0.025, k0=1.0), Sw * T(0, 0.23, 0))
    sc.add(sd_cone(0.02, 0.012, 0.001), glass(["#fff0c0", "#ffffff"], anim, t, cell=0.02, k0=1.0), Sw * T(0, 0.45, 0))
    sc.add(sd_box(0.055, 0.01, 0.01, 0.003), GILT, Sw * T(0, 0.03, 0))
    sc.add(sd_sphere(0.016), GILT, Sw * T(0, -0.025, 0))
    if anim != "blocked":
        flame = glass(["#ff8a1a", "#ffb030", "#ff5a10", "#ffd060"], anim, t, cell=0.022, seed=6.0, k0=1.6)
        for k in range(7):                                                                     # tongues of fire up both edges of the blade
            ph = (t * 2 + k * 0.37) % 1
            sx = 1 if k % 2 else -1
            y = 0.08 + 0.055 * k + 0.04 * ph
            s_ = 0.05 * (1 - 0.07 * k) * math.sin(math.pi * ph) * (1.3 if anim == "win" else 1.0)
            if s_ > 0.004:
                sc.add(sd_ellipsoid(s_ * 0.7, s_ * 1.8, s_ * 0.5), flame, Sw * T(sx * (0.018 + s_ * 0.4), y, 0) * RZ(-sx * 0.35))
    # the wings, crimson and gold
    spread = {"idle": 1.0 + 0.08 * math.sin(TAU * t), "win": 0.55, "blocked": 1.4}.get(anim, 1.0)
    wing = glass(["#c81e2a", "#e8a828", "#a81820", "#f2c040", "#e03040"], anim, t, cell=0.03, seed=4.0, k0=0.6)
    cover = glass(["#a81820", "#c81e2a", "#901418"], anim, t, cell=0.025, seed=15.0, k0=0.6)
    for sx in (-1, 1):
        feather_wing(sc, S * T(sx * 0.06, 0.57, -0.09) * RY(-sx * 0.35) * RZ(sx * spread), sx, wing, cover,
                     length=0.36, n=9, chord=0.13, root=-2.4, tip=-0.45)
    return sc


# ─── 12 · Rosace ────────────────────────────────────────────────────────────
def rose_glass(anim, t, ang, emit=None, R=0.3, shade=True):
    """The rose window's glass: a designed pattern in polar coordinates (rings and
    12 sectors, petals, a centre), lead between the pieces, each piece darker
    near its lead (grisaille; shade=False: flat, as in the cathedral background),
    turned by `ang`."""
    pal = torch.stack([srgb(c) for c in ("#2a4ab8", "#c81e2a", "#f2c040", "#2a8a4a", "#6a2a9a", "#3a64d0", "#e8a828", "#a81820")])
    lead_c = srgb(LEAD)
    k0 = light(anim, t, 0.7) if emit is None else emit                                    # (the cathedral's rose: its own glow, its own radius)

    def pat(pl, pw, n):
        x, z = pl[:, 0], pl[:, 2]
        r = torch.sqrt(x * x + z * z) / R
        a = torch.remainder(torch.atan2(z, x) + ang, TAU)
        sec = torch.floor(a / (TAU / 12))
        fa = torch.remainder(a, TAU / 12) / (TAU / 12)                                          # 0..1 across a sector
        rings = torch.tensor([0.0, 0.18, 0.32, 0.55, 0.78, 1.0], device=DEV)
        ring = torch.bucketize(r.contiguous(), rings[1:-1])
        rr = r - rings[ring]
        lead = ((fa < 0.04) | (fa > 0.96)) & (r < 0.78)                                          # the spokes run to the outer ring only
        lead = lead | ((r * 1000).long() % 1 == 1)
        for v in (0.18, 0.32, 0.55, 0.78, 0.98):
            lead = lead | ((r - v).abs() < 0.012)
        # petals in the 3rd ring: a pointed arch per sector
        petal = (ring == 3) & ((fa - 0.5).abs() * 2 < (1 - (r - 0.55) / 0.23).clamp(0, 1))
        lead = lead | ((ring == 3) & (((fa - 0.5).abs() * 2 - (1 - (r - 0.55) / 0.23).clamp(0, 1)).abs() < 0.05))
        # the colours of a real rose: deep blue ground, red and gold accents, a gold heart
        even = (sec.long() % 2) == 0
        BLU, RED, GLD, GRN, PUR, BL2 = 0, 1, 2, 3, 4, 5
        idx = torch.full_like(ring, BLU)
        idx = torch.where(ring == 0, torch.full_like(ring, GLD), idx)
        idx = torch.where(ring == 1, torch.full_like(ring, RED), idx)
        idx = torch.where(ring == 2, torch.where(even, torch.full_like(ring, BLU), torch.full_like(ring, BL2)), idx)
        idx = torch.where((ring == 3) & petal, torch.full_like(ring, RED), idx)
        idx = torch.where((ring == 4), torch.full_like(ring, BL2), idx)
        # the outer ring: twelve round oculi (red, gold-rimmed), one per sector
        ca = (sec + 0.5) * (TAU / 12)
        cx_, cz_ = 0.885 * torch.cos(ca), 0.885 * torch.sin(ca)
        ax, az = r * torch.cos(a), r * torch.sin(a)
        dc = torch.sqrt((ax - cx_) ** 2 + (az - cz_) ** 2)
        oc = (ring == 4) & (dc < 0.085)
        idx = torch.where(oc, torch.where(dc < 0.05, torch.full_like(ring, RED), torch.full_like(ring, GLD)), idx)
        lead = lead | ((ring == 4) & (((dc - 0.085).abs() < 0.012) | ((dc - 0.05).abs() < 0.01)))
        col = pal[idx] * (0.85 + 0.25 * torch.frac(torch.sin((sec % 2) * 12.9 + ring * 78.2) * 437.5))[:, None]   # tone by sector parity: the pattern repeats every 2 sectors (TAU/6), so the idle turn of TAU/6 joins up
        g = (~lead).float()[:, None]
        # grisaille: each piece darkens towards its lead (the distance to the nearest lead line)
        dl = torch.minimum(fa, 1 - fa) * (TAU / 12) * r
        dl = torch.where(r < 0.78, dl, torch.full_like(r, 1.0))
        for v in (0.18, 0.32, 0.55, 0.78, 0.98):
            dl = torch.minimum(dl, (r - v).abs())
        dl = torch.where(ring == 3, torch.minimum(dl, ((fa - 0.5).abs() * 2 - (1 - (r - 0.55) / 0.23).clamp(0, 1)).abs() * 0.2), dl)
        dl = torch.where(ring == 4, torch.minimum(dl, torch.minimum((dc - 0.085).abs(), (dc - 0.05).abs())), dl)
        em = col * k0 * ((0.45 + 0.55 * (dl / 0.06).clamp(0, 1) ** 0.6)[:, None] if shade else 1.0)
        if anim == "win":
            em = em * (1 + 2.0 * torch.exp(-((r - (t % 1) * 1.1) / 0.12) ** 2))[:, None]
        if anim == "blocked":
            em = em * 0.5
        return {"albedo": col * 0.35 * g + lead_c * (1 - g), "emissive": em * g, "rough": 0.08 + 0.5 * (1 - g[:, 0])}
    return Mat("#2a4ab8", rough=0.08, clearcoat=1.0, pattern=pat)


def rose_window(anim, t, side):
    """The rose window, alive: a great wheel of glass in its stone frame, standing
    upright, turning slowly (one sixth of a turn per loop: its pattern, alternating
    sector by sector, comes back after two of its 12 sectors), its tracery of lead
    and stone, its glass shaded towards the lead (grisaille); the gothic frame with
    pinnacles, crockets and a gilt cross. win: it spins a whole turn and a ring of
    light runs out from its heart."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    stone = STONE_MAT
    ang = {"idle": TAU / 6 * t, "win": TAU * t, "blocked": 0.03 * math.sin(TAU * 6 * t)}.get(anim, 0.0)
    W = S * T(0, 0.43, 0) * RX(math.pi / 2)                                                     # the wheel's plane faces +z
    sc.add(sd_cylinder(0.018, 0.3, 0.005), rose_glass(anim, t, ang), W)
    sc.add(sd_torus(0.31, 0.03), stone, W)                                                       # the stone rim
    sc.add(sd_torus(0.06, 0.012), stone, W)
    for k in range(12):                                                                          # the tracery: 12 stone spokes, turning
        a = k * TAU / 12 - ang
        sc.add(sd_box(0.1, 0.022, 0.005, 0.003), stone, W * RY(a) * T(0.13, 0, 0))
    # the gothic frame round it: two pillars with pinnacles, a pointed arch with
    # crockets along it, a finial cross at its apex
    for sx in (-1, 1):
        sc.add(sd_box(0.03, 0.27, 0.04, 0.01), stone, S * T(sx * 0.36, 0.42, 0))
        sc.add(sd_box(0.042, 0.015, 0.05, 0.005), stone, S * T(sx * 0.36, 0.7, 0))
        sc.add(sd_cone(0.07, 0.035, 0.002), stone, S * T(sx * 0.36, 0.78, 0))                  # the pinnacle
        sc.add(sd_sphere(0.012), GILT, S * T(sx * 0.36, 0.86, 0))
        arch = bezier((sx * 0.33, 0.69, 0), (sx * 0.33, 0.8, 0), (sx * 0.17, 0.9, 0), (0, 0.95, 0), 6)
        chain(sc, arch, [0.03] * 6, stone, S, blend=0.01)
        for k in (1, 2, 3, 4):                                                                  # crockets
            x, y, _ = arch[k]
            sc.add(sd_cone(0.022, 0.014, 0.002), stone, S * T(x * 1.06, y + 0.035, 0) * RZ(-sx * 0.7))
    sc.add(sd_box(0.012, 0.045, 0.012, 0.003), GILT, S * T(0, 1.02, 0))                       # the finial cross
    sc.add(sd_box(0.032, 0.012, 0.012, 0.003), GILT, S * T(0, 1.03, 0))
    # its foot: a gothic pedestal
    sc.add(sd_box(0.12, 0.06, 0.07, 0.01), stone, S * T(0, 0.06, 0))
    sc.add(sd_cone(0.05, 0.08, 0.04, 0.01), stone, S * T(0, 0.15, 0))
    return sc


SCALE = {"random": 1.3, "beginner": 1.25, "monte_carlo": 1.25, "mcts": 1.25, "basic_minmax": 1.45, "minmax": 1.2,
         "features": 1.2, "mcts_nnue": 1.55, "nnue193": 1.2, "resnet": 1.2, "dev": 1.2, "alphazero": 1.2}
CAST = {"random": ("Bougie", candle), "beginner": ("Colombe", dove), "monte_carlo": ("Lys", lily),
        "mcts": ("Cloche", bell), "basic_minmax": ("Gargouille", gargoyle), "minmax": ("Moine copiste", monk),
        "features": ("Chevalier", knight), "mcts_nnue": ("Licorne", unicorn), "nnue193": ("Ange", angel),
        "resnet": ("Lion ailé", winged_lion), "dev": ("Archange", archangel), "alphazero": ("Rosace", rose_window)}
VITRAUX = {"vit_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"vit_" + k: n for k, (n, _) in CAST.items()}


if __name__ == "__main__":
    import sys
    from PIL import Image
    from render import render
    from models import scaled, SCALE as MSCALE
    names = sys.argv[2:] if len(sys.argv) > 2 else list(VITRAUX)
    for nm in names:
        MSCALE[nm] = SCALE[nm[4:]]
        f = scaled(nm, VITRAUX[nm])
        tiles = []
        for anim in ("idle", "blocked", "win"):
            for k in range(4):
                tiles.append(render(f(anim, k / 4, 0 if k % 2 == 0 else 1), size=192))
        sheet = Image.new("RGBA", (192 * 4, 192 * 3), (40, 44, 52, 255))
        for i, im in enumerate(tiles):
            sheet.alpha_composite(Image.fromarray(im), ((i % 4) * 192, (i // 4) * 192))
        sheet.save(f"/tmp/{nm}.png")
        print(nm, flush=True)
