"""
vitraux.py — the "Vitraux" set: the figures of a cathedral's stained glass, made
of coloured glass set in lead and lit from within, from the weakest AI to the
strongest (the user's choice, concept A "la cathédrale"):

  random        Bougie           a church candle in its holder, its flame
  beginner      Colombe          the dove, wings spread
  monte_carlo   Lys              the fleur-de-lis / the lily of the windows
  mcts          Cloche           the bell, swinging
  basic_minmax  Gargouille       the gargoyle crouching, its eyes alight
  minmax        Moine copiste    the monk at his desk, quill in hand
  features      Chevalier        the knight, shield and sword
  mcts_nnue     Licorne          the unicorn rearing
  nnue193       Ange             the angel, wings open
  resnet        Lion ailé        the winged lion (Saint Mark)
  dev           Archange         the archangel with the flaming sword
  alphazero     Rosace           the rose window, alive, turning slowly

Every surface is glass cut into pieces (a 3D Voronoi of cells) joined by dark
lead came; each piece takes a colour of the figure's palette and glows
(emissive), as a window does with the sun behind it. Animations: idle — the
light breathes and the figure stirs (flame, wings, bell, quill...); move — a
hop; blocked — a shake, the light dims to a dull red; win — a wave of light
runs through the glass. Every periodic motion is a whole number of cycles over
the loop (t = 1 is t = 0).

    cd app/tools/sprites && python3 vitraux.py preview [name ...]   (contact sheets in the scratchpad)
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm, vec, DEV, F
from models import base, BASE_TOP, TAU, glow
from heroic import chain, bezier, HEAD

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


def stained(palette, cell=0.07, lead=0.05, emit=0.9, seed=0.0, wave=None, dim=None):
    """Stained glass: the local space cut into cells (3D Voronoi, size `cell`), each
    a piece of glass of one colour of `palette` (hex list), the borders between
    cells dark lead (width `lead`, a share of a cell). The glass glows: emissive
    = its colour × emit. wave (0..1 or None): a band of brighter light sweeping up
    the figure (win); dim (0..1 or None): the light falls to a dull red (blocked)."""
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
    """The figure's glass for this frame (the light of the animation)."""
    return stained(palette, cell=cell, emit=light(anim, t, k0), seed=seed,
                   wave=(t % 1) if anim == "win" else None,
                   dim=(0.5 + 0.5 * math.sin(TAU * 4 * t)) * 0.6 if anim == "blocked" else None)


LEAD_MAT = Mat(LEAD, metal=0.6, rough=0.4)


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
    """A tall church candle of ivory glass in a gilt holder, its flame of amber glass
    over a real glow; idle: the flame sways and breathes; win: it flares high."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    gold = Mat("#d8a840", metal=0.95, rough=0.25)
    sc.add(sd_cylinder(0.025, 0.2, 0.012), gold, S * T(0, 0.025, 0))                          # the dish
    sc.add(sd_torus(0.2, 0.014), gold, S * T(0, 0.05, 0))
    sc.add(sd_cylinder(0.03, 0.06, 0.01), gold, S * T(0, 0.08, 0))                             # the cup
    ivory = glass(["#f4ead2", "#efe0b8", "#f8f0de", "#e8d4a8"], anim, t, cell=0.06, k0=0.35)
    sc.add(sd_cylinder(0.25, 0.075, 0.015), ivory, S * T(0, 0.36, 0))                          # the candle
    red = glass(["#c81e2a", "#a81820", "#e03040"], anim, t, cell=0.04, seed=7.0, k0=0.7)
    blue = glass(["#2a4ab8", "#1a3a98", "#3a64d0"], anim, t, cell=0.04, seed=8.0, k0=0.7)
    sc.add(sd_cylinder(0.022, 0.079, 0.006), red, S * T(0, 0.2, 0))                             # the painted bands of a paschal candle
    sc.add(sd_cylinder(0.012, 0.079, 0.004), blue, S * T(0, 0.25, 0))
    sc.add(sd_box(0.012, 0.06, 0.004, 0.003), red, S * T(0, 0.43, 0.077))                       # the cross on its front
    sc.add(sd_box(0.04, 0.012, 0.004, 0.003), red, S * T(0, 0.45, 0.077))
    sc.add(sd_capsule((0, 0.6, 0), (0, 0.63, 0), 0.006), LEAD_MAT, S)                           # the wick
    sway = 0.05 * math.sin(TAU * t) if anim == "idle" else (0.02 * math.sin(TAU * 4 * t) if anim == "blocked" else 0.0)
    hgt = {"win": 1.6, "blocked": 0.6}.get(anim, 1.0) * (1 + (0.08 * math.sin(TAU * 3 * t) if anim == "idle" else 0))
    flame = glass(["#ffcf4a", "#ffb030", "#ffe28a", "#ff9a2a"], anim, t, cell=0.035, seed=3.0, k0=2.2)
    sc.add(sd_ellipsoid(0.045, 0.1 * hgt, 0.045), flame, S * T(0, 0.7 + 0.04 * (hgt - 1), 0) * RZ(sway))
    sc.add(sd_sphere(0.03), glow("#fff2c0", 9.0 if anim == "win" else 6.0), S * T(0, 0.66, 0))
    return sc


# ─── 2 · Colombe ────────────────────────────────────────────────────────────
def dove(anim, t, side):
    """The dove of the windows, white and pale blue glass, wings raised, an olive
    twig in its beak; idle: the wings beat slowly; win: they beat fast."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    body = glass(["#f4f6fa", "#dfe8f4", "#c8d8ee", "#b8cce8"], anim, t, cell=0.06, k0=0.14)
    C = S * T(0, 0.42, 0)
    # the dove alights on a slender column of blue glass with a gilt capital
    col = glass(["#2a4ab8", "#3a64d0", "#1a3a98"], anim, t, cell=0.05, seed=6.0, k0=0.6)
    sc.add(sd_cylinder(0.15, 0.035, 0.01), col, S * T(0, 0.15, 0))
    sc.add(sd_cylinder(0.02, 0.06, 0.008), Mat("#d8a840", metal=0.95, rough=0.25), S * T(0, 0.31, 0))
    sc.add(sd_ellipsoid(0.11, 0.1, 0.2), body, C, blend=0.04)                                 # the body
    sc.add(sd_sphere(0.075), body, C * T(0, 0.08, 0.17), blend=0.04)                             # the head
    sc.add(sd_ellipsoid(0.09, 0.025, 0.12), body, C * T(0, 0.01, -0.24) * RX(-0.3), blend=0.02)   # the tail
    beak = glass(["#e8a040", "#d88a30"], anim, t, cell=0.03, k0=0.8)
    sc.add(sd_cone(0.03, 0.02, 0.003), beak, C * T(0, 0.07, 0.25) * RX(math.pi / 2))
    sc.add(sd_sphere(0.012), LEAD_MAT, C * T(0.05, 0.1, 0.2))
    sc.add(sd_sphere(0.012), LEAD_MAT, C * T(-0.05, 0.1, 0.2))
    beat = {"idle": 1, "win": 3, "blocked": 0}.get(anim, 1)
    a = 0.55 + 0.35 * math.sin(TAU * beat * t)
    if anim == "blocked":
        a = 0.3 + 0.05 * math.sin(TAU * 6 * t)
    wing = glass(["#f4f8ff", "#dce6f6", "#b4c8ea", "#8aa8dc"], anim, t, cell=0.055, seed=2.0, k0=0.16)
    for sx in (-1, 1):
        W = C * T(sx * 0.08, 0.06, 0.02) * RZ(sx * a)
        sc.add(sd_ellipsoid(0.26, 0.02, 0.12), wing, W * T(sx * 0.24, 0, -0.02) * RY(sx * 0.25))
    olive = glass(["#6a9a3a", "#88b848", "#4a7a2a"], anim, t, cell=0.03, seed=5.0, k0=0.7)
    sc.add(sd_capsule((0.0, 0.05, 0.28), (0.08, 0.02, 0.33), 0.006), LEAD_MAT, C)
    for k in range(3):
        sc.add(sd_ellipsoid(0.028, 0.008, 0.014), olive, C * T(0.02 + k * 0.025, 0.04 - k * 0.01, 0.3 + k * 0.012) * RY(0.6))
    return sc


# ─── 3 · Lys ────────────────────────────────────────────────────────────────
def lily(anim, t, side):
    """The fleur-de-lis of the windows in gold and blue glass: the tall middle petal,
    the two outer petals curling out, the band across; idle: the petals breathe;
    win: they open wide and shine."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    gold = glass(["#f2c040", "#e8a828", "#ffd860", "#d87a20", "#c86a18"], anim, t, cell=0.055, k0=0.6)
    blue = glass(["#2a4ab8", "#1a3a98", "#3a64d0", "#203a8a"], anim, t, cell=0.06, seed=4.0, k0=0.8)
    open_ = {"idle": 0.08 * math.sin(TAU * t), "win": 0.25 * abs(math.sin(TAU * t)), "blocked": -0.05}.get(anim, 0.0)
    C = S * T(0, 0.0, 0)
    # the middle petal: a tall almond with a pointed top
    sc.add(sd_ellipsoid(0.085, 0.24, 0.05), gold, C * T(0, 0.52, 0), blend=0.02)
    sc.add(sd_cone(0.1, 0.07, 0.004), gold, C * T(0, 0.8, 0), blend=0.03)
    # the outer petals: curved, curling outwards and down
    for sx in (-1, 1):
        P = C * T(sx * 0.1, 0.4, 0) * RZ(-sx * (0.55 + open_))
        pts = [(0, 0, 0), (sx * 0.05, 0.12, 0), (sx * 0.14, 0.18, 0), (sx * 0.22, 0.14, 0), (sx * 0.24, 0.06, 0)]
        chain(sc, pts, [0.05, 0.055, 0.05, 0.04, 0.025], gold, P, blend=0.03)
    # the band across the stem, in blue, and the foot
    sc.add(sd_box(0.17, 0.035, 0.055, 0.02), blue, C * T(0, 0.33, 0))
    sc.add(sd_cylinder(0.11, 0.045, 0.02), gold, C * T(0, 0.18, 0))
    for sx in (-1, 1):
        sc.add(sd_ellipsoid(0.08, 0.04, 0.04), gold, C * T(sx * 0.07, 0.1, 0) * RZ(sx * 0.6))
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


def wings(sc, X, mat, span=0.32, spread=0.6, droop=0.0, feathers=4):
    """A pair of wings: each a fan of long feather blades (ellipsoids) from the shoulder,
    raised by `spread` (radians), the tips drooping by `droop`."""
    for sx in (-1, 1):
        W = X * T(sx * 0.05, 0, -0.04) * RZ(sx * spread)
        for k in range(feathers):
            L = span * (1.0 - 0.13 * k)
            a = sx * (0.12 + 0.28 * k) + sx * droop * 0.0
            sc.add(sd_ellipsoid(L / 2, 0.018, 0.045), mat, W * RZ(-sx * 0.28 * k) * T(sx * L / 2, 0.0, -0.01 * k) * RY(sx * 0.15), blend=0.01)


# ─── 4 · Cloche ─────────────────────────────────────────────────────────────
def bell(anim, t, side):
    """The church bell of amber glass under a small gothic arch of lead, its blue
    band and clapper; idle: it swings (one full swing per loop); win: it rings hard."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    for sx in (-1, 1):                                                                       # the arch: two posts, a pointed top
        sc.add(sd_box(0.025, 0.33, 0.03, 0.008), LEAD_MAT, S * T(sx * 0.27, 0.33, 0))
        sc.add(sd_capsule((sx * 0.27, 0.66, 0), (0, 0.84, 0), 0.025), LEAD_MAT, S)
    sc.add(sd_box(0.3, 0.02, 0.03, 0.006), LEAD_MAT, S * T(0, 0.66, 0))
    sw = {"idle": 0.35 * math.sin(TAU * t), "win": 0.6 * math.sin(TAU * 2 * t), "blocked": 0.04 * math.sin(TAU * 6 * t)}.get(anim, 0.0)
    P = S * T(0, 0.64, 0) * RZ(sw)                                                          # the pivot under the beam
    amber = glass(["#e8a030", "#d88a20", "#f0b848", "#c87818"], anim, t, cell=0.055, k0=0.6)
    sc.add(sd_cone(0.17, 0.2, 0.09, 0.02), amber, P * T(0, -0.21, 0))                       # the bell
    sc.add(sd_sphere(0.09), amber, P * T(0, -0.05, 0), blend=0.03)                           # its crown
    sc.add(sd_torus(0.2, 0.025), amber, P * T(0, -0.37, 0))                                  # the lip
    sc.add(sd_torus(0.15, 0.018), glass(BLUE_P, anim, t, cell=0.04, seed=2.0, k0=0.7), P * T(0, -0.2, 0))
    cl = -sw * 0.6
    sc.add(sd_capsule((0, -0.1, 0), (0, -0.32, 0), 0.01), LEAD_MAT, P * RZ(cl))              # the clapper
    sc.add(sd_sphere(0.035), LEAD_MAT, P * RZ(cl) * T(0, -0.34, 0))
    return sc


# ─── 5 · Gargouille ─────────────────────────────────────────────────────────
def gargoyle(anim, t, side):
    """The gargoyle crouching on its ledge, grey-violet glass, horns, folded bat
    wings, its jaws open; idle: its eyes smoulder and it turns its head; win: its
    wings half open, eyes blazing."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    stone = glass(STONE_P, anim, t, cell=0.05, k0=0.35)
    sc.add(sd_box(0.3, 0.05, 0.16, 0.02), LEAD_MAT, S * T(0, 0.05, 0))                        # the ledge
    B = S * T(0, 0.22, 0)
    sc.add(sd_ellipsoid(0.14, 0.13, 0.17), stone, B, blend=0.04)                               # the crouched body
    for sx in (-1, 1):
        sc.add(sd_ellipsoid(0.06, 0.08, 0.07), stone, B * T(sx * 0.12, -0.07, 0.08), blend=0.03)   # haunches
        sc.add(sd_capsule((sx * 0.08, 0.0, 0.12), (sx * 0.09, -0.12, 0.18), 0.03), stone, B, blend=0.02)  # forelegs
    turn = {"idle": 0.25 * math.sin(TAU * t), "blocked": 0.08 * math.sin(TAU * 5 * t)}.get(anim, 0.0)
    H = B * T(0, 0.17, 0.12) * RY(turn)
    sc.add(sd_ellipsoid(0.09, 0.08, 0.1), stone, H, blend=0.03)                                # the head
    sc.add(sd_ellipsoid(0.06, 0.035, 0.08), stone, H * T(0, -0.03, 0.08), blend=0.02)          # the snout
    sc.add(sd_ellipsoid(0.055, 0.02, 0.07), stone, H * T(0, -0.085, 0.07), blend=0.01)         # the open lower jaw
    for sx in (-1, 1):
        sc.add(sd_cone(0.05, 0.02, 0.003), stone, H * T(sx * 0.06, 0.09, -0.02) * RZ(-sx * 0.5))   # horns
        ek = {"win": 12.0, "blocked": 1.5}.get(anim, 5.0 + 3.0 * math.sin(TAU * 2 * t))
        sc.add(sd_sphere(0.017), glow("#ff3a1a", ek), H * T(sx * 0.04, 0.025, 0.09))
    open_ = 0.9 if anim == "win" else 0.25
    bat = glass(PURPLE_P + ["#3a2a5a"], anim, t, cell=0.05, seed=3.0, k0=0.45)
    wings(sc, B * T(0, 0.08, -0.05), bat, span=0.26, spread=open_, feathers=3)
    return sc


# ─── 6 · Moine copiste ──────────────────────────────────────────────────────
def monk(anim, t, side):
    """The monk in his brown habit at his lectern, the open book of white glass with
    a red initial, his quill moving; idle: he writes (two strokes per loop); win:
    he raises the book to the light."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    habit = glass(["#7a4a28", "#6a3a20", "#8a5a30", "#5a3018"], anim, t, cell=0.06, k0=0.45)
    skin = glass(SKIN_P, anim, t, cell=0.04, seed=2.0, k0=0.5)
    M = S * T(-0.08, 0, 0)
    sc.add(sd_cone(0.22, 0.17, 0.1, 0.02), habit, M * T(0, 0.22, 0))                          # the habit, falling to the ground
    sc.add(sd_ellipsoid(0.12, 0.1, 0.1), habit, M * T(0, 0.45, 0), blend=0.04)                 # shoulders
    sc.add(sd_sphere(0.075), skin, M * T(0, 0.58, 0.01))                                       # the head (tonsure)
    sc.add(sd_torus(0.07, 0.022), habit, M * T(0, 0.53, -0.01) * RX(0.3))                      # the hood fallen back
    sc.add(sd_torus(0.06, 0.018), glass(["#5a3a20", "#4a2a18"], anim, t, cell=0.03, k0=0.3), M * T(0, 0.62, 0))   # the ring of hair
    # the lectern and the book
    L = S * T(0.17, 0, 0.06)
    sc.add(sd_box(0.025, 0.18, 0.025, 0.008), LEAD_MAT, L * T(0, 0.18, 0))
    raise_ = 0.12 if anim == "win" else 0.0
    Bk = L * T(-0.02, 0.4 + raise_, 0) * RZ(0.5 + (0.3 if anim == "win" else 0.0))
    sc.add(sd_box(0.1, 0.012, 0.075, 0.006), LEAD_MAT, Bk)
    page = glass(["#f8f4e8", "#f0eadb", "#fbf8f0"], anim, t, cell=0.04, seed=4.0, k0=0.45)
    for sz in (-1, 1):
        sc.add(sd_box(0.045, 0.006, 0.07, 0.003), page, Bk * T(0, 0.015, sz * 0.0) * T(sz * 0.048, 0, 0) * RZ(-sz * 0.12))
    sc.add(sd_box(0.012, 0.004, 0.014, 0.002), glass(RED_P, anim, t, cell=0.02, k0=0.8), Bk * T(-0.07, 0.024, -0.03))   # the initial
    # the writing arm and the quill
    w = 0.05 * math.sin(TAU * 2 * t) if anim == "idle" else 0.0
    hand = M * T(0.15 + w, 0.44 + raise_ * 0.8, 0.07)
    sc.add(sd_capsule((0, 0.45, 0.02), (0.15 + w, 0.44 + raise_ * 0.8, 0.07), 0.035), habit, M, blend=0.02)
    sc.add(sd_sphere(0.025), skin, hand)
    sc.add(sd_ellipsoid(0.008, 0.07, 0.018), glass(WHITE_P, anim, t, cell=0.02, k0=0.6), hand * T(0.01, 0.06, 0) * RZ(-0.4))
    return sc


# ─── 7 · Chevalier ──────────────────────────────────────────────────────────
def knight(anim, t, side):
    """The knight of the windows: mail of silver-blue glass, the red surcoat with a
    white cross, the great helm, the blue shield with gold lilies, the sword; idle:
    he shifts his sword; win: he raises it high."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    mail = glass(["#a8b8d0", "#8898b8", "#c0cce0", "#7888a8"], anim, t, cell=0.045, k0=0.4)
    red = glass(RED_P, anim, t, cell=0.06, seed=2.0, k0=0.6)
    for sx in (-1, 1):
        sc.add(sd_capsule((sx * 0.05, 0.03, 0), (sx * 0.05, 0.26, 0), 0.035), mail, S)        # legs
    sc.add(sd_cone(0.17, 0.13, 0.11, 0.02), red, S * T(0, 0.36, 0))                           # the surcoat
    sc.add(sd_box(0.015, 0.1, 0.004, 0.003), glass(WHITE_P, anim, t, cell=0.03, k0=0.6), S * T(0, 0.38, 0.125))   # its cross
    sc.add(sd_box(0.06, 0.015, 0.004, 0.003), glass(WHITE_P, anim, t, cell=0.03, k0=0.6), S * T(0, 0.42, 0.122))
    sc.add(sd_ellipsoid(0.13, 0.07, 0.09), mail, S * T(0, 0.53, 0), blend=0.03)                # shoulders
    sc.add(sd_cylinder(0.07, 0.065, 0.02), mail, S * T(0, 0.66, 0))                            # the great helm
    sc.add(sd_box(0.05, 0.006, 0.004, 0.002), LEAD_MAT, S * T(0, 0.67, 0.066))                 # its eye slit
    # the shield on the left arm
    Sh = S * T(-0.17, 0.42, 0.06) * RY(-0.4)
    sc.add(sd_box(0.08, 0.1, 0.012, 0.03), glass(BLUE_P, anim, t, cell=0.04, seed=3.0, k0=0.65), Sh)
    sc.add(sd_cone(0.05, 0.08, 0.005), glass(BLUE_P, anim, t, cell=0.04, seed=3.0, k0=0.65), Sh * T(0, -0.13, 0) * RX(math.pi), blend=0.01)
    for (x, y) in ((-0.035, 0.04), (0.035, 0.04), (0, -0.04)):
        sc.add(sd_sphere(0.017), glass(GOLD_P, anim, t, cell=0.02, k0=0.9), Sh * T(x, y, 0.015))
    # the sword in the right hand
    up = {"win": 1.0}.get(anim, 0.0)
    a = 0.3 + 0.1 * math.sin(TAU * t) if anim == "idle" else (-2.4 * up if anim == "win" else 0.3)
    hand = S * T(0.17, 0.42 + 0.2 * up, 0.04)
    sc.add(sd_capsule((0.1, 0.52, 0), (0.17, 0.42 + 0.2 * up, 0.04), 0.03), mail, S)
    Sw = hand * RZ(a)
    sc.add(sd_box(0.012, 0.2, 0.004, 0.002), glass(["#e8f0ff", "#c8d8f0", "#ffffff"], anim, t, cell=0.03, k0=0.7), Sw * T(0, 0.23, 0))
    sc.add(sd_box(0.05, 0.01, 0.01, 0.003), GILT, Sw * T(0, 0.03, 0))
    sc.add(sd_sphere(0.015), GILT, Sw * T(0, -0.02, 0))
    return sc


# ─── 8 · Licorne ────────────────────────────────────────────────────────────
def unicorn(anim, t, side):
    """The unicorn rearing, white and lilac glass, its golden horn, a mane and tail
    of rainbow glass; idle: it tosses its head; win: it rears high."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    coat = glass(["#f4f0fa", "#e4dcf4", "#d0c4ec", "#f8f4fc"], anim, t, cell=0.06, k0=0.3)
    rear = {"win": 0.45 + 0.1 * math.sin(TAU * 2 * t), "blocked": 0.2}.get(anim, 0.25)
    R = S * T(0, 0.08, -0.1) * RX(-rear)                                                        # it rears on its hind legs
    for sx in (-1, 1):
        sc.add(sd_capsule((sx * 0.06, -0.06, 0), (sx * 0.06, 0.12, 0.02), 0.035), coat, R, blend=0.02)       # hind legs
    sc.add(sd_ellipsoid(0.1, 0.11, 0.22), coat, R * T(0, 0.2, 0.15), blend=0.05)                # the body
    for sx in (-1, 1):                                                                          # forelegs, curled
        sc.add(sd_capsule((sx * 0.06, 0.22, 0.3), (sx * 0.06, 0.12, 0.4), 0.03), coat, R, blend=0.02)
        sc.add(sd_capsule((sx * 0.06, 0.12, 0.4), (sx * 0.06, 0.18, 0.47), 0.026), coat, R, blend=0.02)
    nod = 0.15 * math.sin(TAU * t) if anim == "idle" else 0.0
    N = R * T(0, 0.3, 0.33) * RX(-0.6 + nod)
    sc.add(sd_capsule((0, 0, 0), (0, 0.16, 0.03), 0.06), coat, N, blend=0.04)                  # the neck
    Hd = N * T(0, 0.18, 0.06) * RX(1.2)
    sc.add(sd_ellipsoid(0.055, 0.05, 0.11), coat, Hd, blend=0.03)                              # the head
    sc.add(sd_cone(0.08, 0.016, 0.002), glass(GOLD_P, anim, t, cell=0.025, k0=1.0), Hd * T(0, 0.08, -0.03) * RX(-1.0))   # the horn
    rainbow = glass(["#e83a4a", "#f0a030", "#f0e040", "#40c060", "#3a7ae0", "#9a4ad0"], anim, t, cell=0.03, seed=4.0, k0=0.8)
    chain(sc, [(0, 0.05, -0.05), (0, 0.12, -0.06), (0, 0.19, -0.03)], [0.035, 0.035, 0.025], rainbow, N, blend=0.02)     # the mane
    chain(sc, [(0, 0.2, -0.05), (0, 0.15, -0.16), (0, 0.06, -0.2)], [0.035, 0.03, 0.02], rainbow, R, blend=0.02)          # the tail
    sc.add(sd_sphere(0.01), LEAD_MAT, Hd * T(0.04, 0.02, 0.03))
    sc.add(sd_sphere(0.01), LEAD_MAT, Hd * T(-0.04, 0.02, 0.03))
    return sc


# ─── 9 · Ange ───────────────────────────────────────────────────────────────
def angel(anim, t, side):
    """The angel standing in a robe of pale blue glass, hands joined, great wings
    of white and gold feathers, the golden halo; idle: the wings breathe; win:
    they open wide and the halo blazes."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    robe = glass(["#a8c8f0", "#88b0e8", "#c8dcf8", "#6a98d8"], anim, t, cell=0.06, k0=0.4)
    skin = glass(SKIN_P, anim, t, cell=0.04, seed=2.0, k0=0.5)
    sc.add(sd_cone(0.27, 0.16, 0.08, 0.02), robe, S * T(0, 0.27, 0))
    sc.add(sd_ellipsoid(0.1, 0.07, 0.08), robe, S * T(0, 0.55, 0), blend=0.04)
    sc.add(sd_sphere(0.07), skin, S * T(0, 0.67, 0.0))
    sc.add(sd_ellipsoid(0.075, 0.05, 0.07), glass(GOLD_P, anim, t, cell=0.03, seed=5.0, k0=0.6), S * T(0, 0.71, -0.02), blend=0.02)   # golden hair
    sc.add(sd_capsule((0, 0.5, 0.06), (0, 0.55, 0.11), 0.03), skin, S)                         # the joined hands
    sc.add(sd_torus(0.07, 0.012), glow("#ffd860", 9.0 if anim == "win" else 4.0), S * T(0, 0.79, -0.03) * RX(0.3))   # the halo
    spread = {"idle": 0.55 + 0.1 * math.sin(TAU * t), "win": 0.9, "blocked": 0.4}.get(anim, 0.55)
    wing = glass(["#ffffff", "#f4ecd0", "#f0d890", "#e8f0fc"], anim, t, cell=0.05, seed=3.0, k0=0.45)
    wings(sc, S * T(0, 0.58, -0.06), wing, span=0.36, spread=spread, feathers=4)
    return sc


# ─── 10 · Lion ailé ─────────────────────────────────────────────────────────
def winged_lion(anim, t, side):
    """The winged lion of Saint Mark sitting, golden glass, a mane of orange and red,
    wings raised, the halo, the open book under its paw; idle: the wings move and
    the tail swings; win: it roars, wings wide."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    fur = glass(GOLD_P, anim, t, cell=0.055, k0=0.55)
    mane = glass(["#e87a20", "#d85a18", "#f09030", "#c84a10"], anim, t, cell=0.045, seed=2.0, k0=0.6)
    B = S * T(0, 0.0, -0.02)
    sc.add(sd_ellipsoid(0.12, 0.13, 0.15), fur, B * T(0, 0.15, -0.06), blend=0.05)             # haunches
    sc.add(sd_ellipsoid(0.1, 0.17, 0.1), fur, B * T(0, 0.33, 0.06) * RX(0.3), blend=0.05)       # the chest, upright
    for sx in (-1, 1):
        sc.add(sd_capsule((sx * 0.06, 0.28, 0.12), (sx * 0.06, 0.04, 0.16), 0.032), fur, B, blend=0.02)   # forelegs
    sc.add(sd_box(0.08, 0.01, 0.06, 0.004), glass(["#f8f4e8", "#f0eadb"], anim, t, cell=0.03, k0=0.5), B * T(0.0, 0.025, 0.24))   # the book
    roar = 0.3 if anim == "win" else 0.0
    Hd = B * T(0, 0.52, 0.12) * RX(-roar)
    sc.add(sd_sphere(0.13), mane, Hd * T(0, 0, -0.03), blend=0.02)                            # the mane
    sc.add(sd_ellipsoid(0.075, 0.075, 0.08), fur, Hd * T(0, 0, 0.06), blend=0.03)               # the face
    sc.add(sd_ellipsoid(0.035, 0.025, 0.03), glass(["#c87818", "#a86010"], anim, t, cell=0.02, k0=0.5), Hd * T(0, -0.02, 0.13))
    for sx in (-1, 1):
        sc.add(sd_sphere(0.012), LEAD_MAT, Hd * T(sx * 0.03, 0.02, 0.12))
    sc.add(sd_torus(0.12, 0.012), glow("#ffd860", 8.0 if anim == "win" else 3.5), Hd * T(0, 0.08, -0.08) * RX(0.5))
    swing = 0.3 * math.sin(TAU * t) if anim == "idle" else 0.0
    chain(sc, [(0, 0.1, -0.18), (0.12 * math.cos(swing), 0.12, -0.28), (0.18, 0.25, -0.3)], [0.02, 0.016, 0.014], fur, B, blend=0.02)
    sc.add(sd_sphere(0.03), mane, B * T(0.18, 0.27, -0.3))
    spread = {"idle": 0.6 + 0.12 * math.sin(TAU * t), "win": 1.0, "blocked": 0.4}.get(anim, 0.6)
    wings(sc, B * T(0, 0.42, -0.02), fur, span=0.3, spread=spread, feathers=4)
    return sc


# ─── 11 · Archange ──────────────────────────────────────────────────────────
def archangel(anim, t, side):
    """The archangel Michael in armour of gold glass over a red robe, great crimson
    and gold wings, the halo, raising the flaming sword; idle: the flames lick up
    the blade; win: the sword sweeps up blazing."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    robe = glass(RED_P, anim, t, cell=0.06, k0=0.55)
    armour = glass(GOLD_P, anim, t, cell=0.045, seed=2.0, k0=0.6)
    skin = glass(SKIN_P, anim, t, cell=0.04, seed=3.0, k0=0.5)
    sc.add(sd_cone(0.25, 0.17, 0.09, 0.02), robe, S * T(0, 0.25, 0))
    sc.add(sd_ellipsoid(0.12, 0.11, 0.09), armour, S * T(0, 0.5, 0), blend=0.04)               # the breastplate
    sc.add(sd_sphere(0.07), skin, S * T(0, 0.66, 0))
    sc.add(sd_ellipsoid(0.075, 0.05, 0.075), glass(["#c87818", "#a86010", "#e09030"], anim, t, cell=0.03, k0=0.5), S * T(0, 0.7, -0.02), blend=0.02)
    sc.add(sd_torus(0.075, 0.012), glow("#ffd860", 10.0 if anim == "win" else 5.0), S * T(0, 0.79, -0.04) * RX(0.3))
    spread = {"idle": 0.7 + 0.1 * math.sin(TAU * t), "win": 1.0, "blocked": 0.45}.get(anim, 0.7)
    wing = glass(["#c81e2a", "#e8a828", "#a81820", "#f2c040"], anim, t, cell=0.05, seed=4.0, k0=0.6)
    wings(sc, S * T(0, 0.56, -0.06), wing, span=0.4, spread=spread, feathers=5)
    # the sword raised in the right hand, its blade in flames
    up = 1.0 if anim == "win" else 0.0
    hand = S * T(0.16, 0.6 + 0.1 * up, 0.06)
    sc.add(sd_capsule((0.09, 0.53, 0), (0.16, 0.6 + 0.1 * up, 0.06), 0.03), armour, S)
    Sw = hand * RZ(-0.25 + (-0.4 * math.sin(math.pi * t) if anim == "win" else 0.0))
    sc.add(sd_box(0.012, 0.2, 0.004, 0.002), glass(["#fff0c0", "#ffe080"], anim, t, cell=0.03, k0=1.0), Sw * T(0, 0.23, 0))
    sc.add(sd_box(0.05, 0.01, 0.01, 0.003), GILT, Sw * T(0, 0.03, 0))
    flame = glass(["#ff8a1a", "#ffb030", "#ff5a10", "#ffd060"], anim, t, cell=0.03, seed=6.0, k0=1.8)
    for k in range(4):                                                                         # the flames along the blade
        ph = (t * 2 + k * 0.25) % 1 if anim != "blocked" else 0.5
        y = 0.08 + 0.09 * k + 0.05 * ph
        s_ = 0.04 * (1 - 0.15 * k) * (0.6 + 0.6 * math.sin(math.pi * ph))
        sc.add(sd_ellipsoid(s_ * 0.8, s_ * 1.6, s_ * 0.5), flame, Sw * T(0.02 * math.sin(TAU * (t + k * 0.3)), y, 0.01))
    sc.add(sd_sphere(0.06), glow("#ff9a30", 5.0 if anim != "blocked" else 1.5), Sw * T(0, 0.3, 0))
    return sc


# ─── 12 · Rosace ────────────────────────────────────────────────────────────
def rose_glass(anim, t, ang, emit=None, R=0.3):
    """The rose window's glass: a designed pattern in polar coordinates (rings and
    12 sectors, petals, a centre), lead between the pieces, turned by `ang`."""
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
        em = col * k0
        if anim == "win":
            em = em * (1 + 2.0 * torch.exp(-((r - (t % 1) * 1.1) / 0.12) ** 2))[:, None]
        if anim == "blocked":
            em = em * 0.5
        return {"albedo": col * 0.35 * g + lead_c * (1 - g), "emissive": em * g, "rough": 0.08 + 0.5 * (1 - g[:, 0])}
    return Mat("#2a4ab8", rough=0.08, clearcoat=1.0, pattern=pat)


def rose_window(anim, t, side):
    """The rose window, alive: a great wheel of glass in its stone frame, standing
    upright, turning slowly (one sixth of a turn per loop: its pattern, alternating
    sector by sector, comes back after two of its 12 sectors), its tracery of lead and stone; win: it spins a whole turn and a
    ring of light runs out from its heart."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    stone = Mat("#7a7266", rough=0.6, pattern=lambda pl, pw, n: {"albedo": srgb("#7a7266") * (0.8 + 0.3 * fbm(pl * 20, 2))[:, None]})
    ang = {"idle": TAU / 6 * t, "win": TAU * t, "blocked": 0.03 * math.sin(TAU * 6 * t)}.get(anim, 0.0)
    W = S * T(0, 0.43, 0) * RX(math.pi / 2)                                                     # the wheel's plane faces +z
    sc.add(sd_cylinder(0.018, 0.3, 0.005), rose_glass(anim, t, ang), W)
    sc.add(sd_torus(0.31, 0.03), stone, W)                                                       # the stone rim
    sc.add(sd_torus(0.06, 0.012), stone, W)
    for k in range(12):                                                                          # the tracery: 12 stone spokes, turning
        a = k * TAU / 12 - ang
        sc.add(sd_box(0.1, 0.022, 0.005, 0.003), stone, W * RY(a) * T(0.13, 0, 0))
    # the gothic frame round it: two pillars and a pointed arch of stone
    for sx in (-1, 1):
        sc.add(sd_box(0.03, 0.27, 0.04, 0.01), stone, S * T(sx * 0.36, 0.42, 0))
        sc.add(sd_capsule((sx * 0.36, 0.69, 0), (0, 0.86, 0), 0.03), stone, S)
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
        sheet.save(f"/tmp/claude-1000/-home-elucterio-Yolah/f1710e3a-943c-4701-b5af-ebd93af4fff8/scratchpad/{nm}.png")
        print(nm, flush=True)
