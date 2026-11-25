"""
computers.py — the "Informatique" set: one calculating machine per AI, from
the oldest and weakest to the most powerful, in the order of history (forms
of historical machines, drawn here, no maker's name or logo):

  random        Boulier              a wooden abacus, red and black beads
  beginner      Mécanisme d'Anticythère  a bronze case, its dial and gears
  monte_carlo   Pascaline            a brass box, spoked wheels, digit windows
  mcts          Métier Jacquard      a loom with its chain of punched cards
  basic_minmax  Machine à différences a brass engine: columns of figure wheels, a crank
  minmax        Machine à rotors     a wooden case: keyboard, lamp board, three rotors
  features      Calculateur à tubes  black panels, glowing vacuum tubes, patch cables
  mcts_nnue     Ordinateur à bandes  a mainframe cabinet, two tape reels, lights
  nnue193       Micro-ordinateur     a beige 8-bit machine, its CRT and blinking cursor
  resnet        Supercalculateur     the C-shaped tower with its padded bench
  dev           Centre de calcul     an aisle of server racks, blinking LEDs
  alphazero     Ordinateur quantique the golden chandelier of a dilution refrigerator

Animations: idle — the machine works (beads slide, gears and wheels turn, cards
advance, lamps blink, reels spin, LEDs flicker); move — a hop; blocked — it
jams and shakes (smoke or sparks for the electric ones); win — its result shows (beads
clack, lamps flash, "WIN" on the screen, light cascades). Every periodic
motion is a whole number of turns or cycles over the loop (t = 1 is t = 0).

    cd app/tools/sprites && python3 computers.py preview [name ...]   (contact sheets in /tmp)
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, fbm, vec
from models import base, BASE_TOP, TAU, glow
from heroic import chain, bezier, HEAD


# ─── materials ──────────────────────────────────────────────────────────────
def wood(c="#6a3e22", k=90, dark=0.18):
    """Polished wood: fine grain (streaks along local x, wavy), a varnish highlight."""
    def pat(pl, pw, n):
        g = 0.5 + 0.5 * torch.sin((pl[:, 1] + pl[:, 2]) * k + 7 * fbm(pl * vec(2, 25, 25), 3))
        return {"albedo": srgb(c) * (1 - dark * g)[:, None] * (0.92 + 0.16 * fbm(pl * 30, 2))[:, None]}
    return Mat(c, rough=0.35, clearcoat=0.6, pattern=pat)


BRASS = Mat("#c9a14a", metal=0.95, rough=0.28)
BRASS_DULL = Mat("#a88640", metal=0.9, rough=0.45)
STEEL = Mat("#9aa0a8", metal=0.9, rough=0.3)
DARK_STEEL = Mat("#3a3e44", metal=0.7, rough=0.4)
BLACK = Mat("#16181c", rough=0.45)
GOLD = Mat("#e8b84a", metal=1.0, rough=0.18)
COPPER = Mat("#c8743a", metal=0.95, rough=0.3)


def bronze():
    """Ancient bronze: green-blue patina in the hollows, brown metal on the edges."""
    def pat(pl, pw, n):
        v = fbm(pl * 18, 3)
        k = ((v - 0.35) * 3).clamp(0, 1)[:, None]
        return {"albedo": srgb("#7a5a32") * (1 - k) + srgb("#4f8a76") * k, "rough": (0.4 + 0.4 * k[:, 0])}
    return Mat("#6a5a3a", metal=0.6, rough=0.5, pattern=pat)


def plastic(c, rough=0.5):
    return Mat(c, rough=rough, clearcoat=0.2)


def stand(anim, t, k=1.0):
    """The machine's pose: a hop when it moves, a shake when it jams, a small
    jump of joy when it wins."""
    lift, tilt, turn = 0.0, 0.0, 0.0
    if anim == "move":
        lift = 0.05 * abs(math.sin(TAU * t))
    elif anim == "blocked":
        tilt = 0.06 * math.sin(TAU * 4 * t)
        turn = 0.05 * math.sin(TAU * 3 * t)
    elif anim == "win":
        lift = 0.025 * abs(math.sin(TAU * 2 * t))
    return RY(HEAD + turn) * T(0, BASE_TOP + lift * k, 0) * RZ(tilt)


def sd_gear(R, h, teeth, depth=None, hole=0.0):
    """A spur gear lying in the local xz plane (axis y): outer radius R, half
    thickness h, `teeth` teeth of height depth, an optional axle hole."""
    depth = depth if depth is not None else R * 0.18

    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = torch.sqrt(x * x + z * z) + 1e-7
        a = torch.atan2(z, x)
        tooth = (torch.cos(a * teeth) * 2.2).clamp(-1, 1) * 0.5 + 0.5      # flat-topped teeth
        d = r - (R - depth + depth * tooth)
        d = torch.maximum(d * 0.6, y.abs() - h)
        if hole > 0:
            d = torch.maximum(d, hole - r)
        return d
    return f


def sd_spokes(R, h, n, w):
    """n flat spokes in the xz plane from the centre to radius R."""
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = torch.sqrt(x * x + z * z)
        a = torch.atan2(z, x)
        sector = TAU / n
        a2 = torch.remainder(a + sector / 2, sector) - sector / 2
        lat = (r * torch.sin(a2)).abs() - w
        rad = r - R
        return torch.maximum(torch.maximum(lat, rad), y.abs() - h)
    return f


def wheel(sc, xf, R, h, mat, spokes=4, rim=None):
    """A spoked wheel (axis y): rim, hub, spokes."""
    rim = rim or R * 0.12
    sc.add(sd_torus(R - rim, rim), mat, xf)
    sc.add(sd_cylinder(h * 1.6, R * 0.16, h * 0.3), mat, xf)
    sc.add(sd_spokes(R - rim, h, spokes, R * 0.05), mat, xf)


# ─── 1 · Boulier ────────────────────────────────────────────────────────────
def abacus(anim, t, side):
    """A wooden abacus standing up: two posts on feet, a frame, five rods with
    beads, red and black. idle: a few beads slide to and fro (counting);
    blocked: the beads rattle; win: every bead clacks to one side and back."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    fr = wood("#7a4a26")
    W, H = 0.36, 0.27                       # half width, half height of the frame
    c = S * T(0, 0.35, 0)
    for x in (-W, W):
        sc.add(sd_box(0.03, H + 0.03, 0.036, 0.008), fr, c * T(x, 0, 0))                  # the posts
        sc.add(sd_box(0.03, 0.012, 0.09, 0.006), fr, S * T(x, 0.012, 0))                  # the feet
        sc.add(sd_box(0.018, 0.04, 0.018, 0.004), fr, S * T(x, 0.06, 0))
    for y in (-H, H):
        sc.add(sd_box(W, 0.024, 0.034, 0.008), fr, c * T(0, y, 0))
    sc.add(sd_box(W, 0.008, 0.024, 0.003), fr, c * T(0, H * 0.45, 0))                     # the beam (heaven / earth)
    red = Mat("#b8282a", rough=0.3, clearcoat=0.8)
    blk = Mat("#22201e", rough=0.3, clearcoat=0.8)
    rows = 5
    for i in range(rows):
        y = -H + (i + 0.5) * (2 * H) / rows - 0.005
        sc.add(sd_capsule((-W, 0, 0), (W, 0, 0), 0.0045), BRASS, c * T(0, y, 0))
        for k in range(5):
            # the bead's rest position, then its motion
            x0 = -W + 0.06 + k * 0.062 if k < 3 else W - 0.06 - (4 - k) * 0.062
            ph = (i * 0.37 + k * 0.11) % 1
            if anim == "idle":
                dx = 0.1 * max(0.0, math.sin(TAU * (t - ph))) ** 3 * (1 if k == 2 else 0)
            elif anim == "blocked":
                dx = 0.012 * math.sin(TAU * (4 * t + ph))
            elif anim == "win":
                dx = (0.5 - 0.5 * math.cos(TAU * 2 * t)) * (0.16 if k < 3 else -0.0)
            else:
                dx = 0.0
            sc.add(sd_ellipsoid(0.03, 0.026, 0.03), red if (i + k) % 2 == 0 else blk, c * T(x0 + dx, y, 0))
    return sc


# ─── 2 · Mécanisme d'Anticythère ────────────────────────────────────────────
def antikythera(anim, t, side):
    """The ancient Greek geared calculator, as its best-known fragment shows it:
    a great four-spoked bronze wheel with its teeth, smaller wheels meshing
    with it, green with age, mounted on a wooden board with a dial ring and a
    pointer. idle: the train turns slowly; win: it spins, the pointer's ball
    of gold shining; blocked: it jams and shakes."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    board = S * T(0, 0.3, -0.03)
    sc.add(sd_box(0.29, 0.3, 0.025, 0.008), wood("#5a3420"), board)
    sc.add(sd_box(0.05, 0.03, 0.08, 0.01), wood("#4a2a18"), S * T(0, 0.03, -0.03))          # its foot
    br = bronze()
    face = board * T(0, 0, 0.03) * RX(math.pi / 2)                  # the wheels' plane, facing +z
    speed = {"idle": 1, "win": 4}.get(anim, 0)
    jam = 0.05 * math.sin(TAU * 6 * t) if anim == "blocked" else 0.0
    # the great wheel: 32 teeth, a rim, four spokes
    R, n = 0.17, 32
    # whole turns per loop (1 idle, 4 win): the patina (a texture fixed on the wheel) comes back
    # where it was; with 1/4 turn per loop the spokes looked the same but the patina had moved
    a0 = TAU * speed * t + jam
    big = face * T(-0.03, 0.01, 0.03) * RY(a0)
    sc.add(sd_gear(R, 0.012, n, R * 0.09, hole=R - 0.04), br, big)
    sc.add(sd_spokes(R - 0.03, 0.01, 4, 0.016), br, big)
    sc.add(sd_cylinder(0.016, 0.03, 0.006), br, big)
    # two smaller wheels meshing: angular speed in the inverse ratio of the teeth
    # (16 and 8 teeth, divisors of 32: 2 and 4 whole turns per turn of the great wheel, so the loop joins up)
    for (cx, cz, r, k, nsp, extra) in ((0.15, -0.13, 0.075, 16, 4, 0.0), (-0.13, 0.15, 0.06, 8, 3, 0.3)):
        a = -a0 * n / k + extra
        g = face * T(-0.03 + cx, 0.016, 0.03 + cz) * RY(a)
        sc.add(sd_gear(r, 0.01, k, r * 0.18, hole=r * 0.25), br, g)
        sc.add(sd_spokes(r * 0.8, 0.009, nsp, 0.01), br, g)
    # the dial ring and the pointer with its ball (the sun), on the great wheel's axle
    sc.add(sd_torus(0.215, 0.006), BRASS_DULL, face * T(-0.03, 0.004, 0.03))
    pa = a0                                                           # a whole turn per loop (idle)
    sc.add(sd_capsule((0, 0, 0), (0.2 * math.cos(pa), 0, 0.2 * math.sin(pa)), 0.006, 0.004), BRASS, face * T(-0.03, 0.04, 0.03))
    k = 9.0 if anim == "win" else 3.0
    sc.add(sd_sphere(0.016), glow("#ffd060", k), face * T(-0.03 + 0.2 * math.cos(pa), 0.04, 0.03 + 0.2 * math.sin(pa)))
    return sc


# ─── 3 · Pascaline ──────────────────────────────────────────────────────────
def pascaline(anim, t, side):
    """Pascal's adding machine: a long brass box, a row of six spoked wheels on
    its sloping front, a window above each where the digits go by. idle: the
    wheels are dialled one after the other (a carry passes along); win: all
    turn together; blocked: they jerk."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    box = S * T(0, 0.1, 0)
    sc.add(sd_box(0.36, 0.07, 0.16, 0.014), BRASS, box)
    sc.add(sd_box(0.37, 0.012, 0.17, 0.006), BRASS_DULL, box * T(0, -0.062, 0))          # the base moulding
    top = box * T(0, 0.072, 0.04)                                                           # the wheels' plate, on the top
    sc.add(sd_box(0.34, 0.006, 0.09, 0.004), Mat("#3a2a18", metal=0.3, rough=0.35, clearcoat=0.6), top)
    sc.add(sd_box(0.34, 0.03, 0.03, 0.006), BRASS_DULL, box * T(0, 0.095, -0.1))            # the strip of digit windows
    n = 6
    for i in range(n):
        x = -0.275 + i * 0.11
        # wheel i turns when the carry reaches it (idle), all together (win)
        if anim == "idle":
            # one tenth of a turn while the carry passes over wheel i: the wheel has
            # 10 spokes and the drum 10 digits, so a tenth of a turn looks the same
            # (the loop joins up)
            a = (TAU / 10) * min(1.0, max(0.0, t * n - i))
        elif anim == "win":
            a = TAU * 2 * t
        elif anim == "blocked":
            a = 0.08 * math.sin(TAU * 5 * t + i)
        else:
            a = 0.0
        wheel(sc, top * T(x, 0.014, 0.0) * RY(a), 0.05, 0.005, BRASS, spokes=10)
        sc.add(sd_torus(0.055, 0.005), BRASS_DULL, top * T(x, 0.01, 0.0))
        # the digit window: a dark slot, the drum's digit glowing faintly
        sc.add(sd_box(0.026, 0.004, 0.014, 0.002), Mat("#f2ead8", rough=0.4,
               pattern=lambda pl, pw, n, i=i, a=a: {"albedo": srgb("#f2ead8") * (0.4 + 0.6 * (torch.sin(pl[:, 0] * 260 + a * 10 + i) > 0.2).float())[:, None]}),
               box * T(x, 0.126, -0.1))
    # the stylus lying in front
    sc.add(sd_capsule((-0.2, 0, 0), (0.05, 0, 0.02), 0.006, 0.003), BRASS, S * T(0, 0.012, 0.25))
    return sc


# ─── 4 · Métier Jacquard ────────────────────────────────────────────────────
def jacquard(anim, t, side):
    """A Jacquard loom: a wooden portal frame, the Jacquard head on top, its
    endless chain of punched cards looping down the side, the warp threads, the
    woven cloth with its pattern. idle: the chain advances one card per beat,
    the shuttle crosses; win: it runs fast, the pattern glows; blocked: jams."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    fr = wood("#6a4426")
    for x in (-0.24, 0.24):
        for z in (-0.12, 0.12):
            sc.add(sd_box(0.016, 0.24, 0.016, 0.004), fr, S * T(x, 0.24, z))
    sc.add(sd_box(0.26, 0.016, 0.016, 0.004), fr, S * T(0, 0.47, -0.12))
    sc.add(sd_box(0.26, 0.016, 0.016, 0.004), fr, S * T(0, 0.47, 0.12))
    sc.add(sd_box(0.26, 0.014, 0.014, 0.004), fr, S * T(0, 0.12, 0.12))                     # the breast beam
    # the Jacquard head: a box on the top
    head = S * T(0.02, 0.55, 0)
    sc.add(sd_box(0.16, 0.07, 0.11, 0.01), fr, head)
    sc.add(sd_box(0.12, 0.008, 0.1, 0.003), BRASS, head * T(0, 0.075, 0))
    # the warp: threads from the back beam to the cloth
    thread = Mat("#e8dcc0", rough=0.8)
    for i in range(9):
        x = -0.2 + i * 0.05
        sc.add(sd_capsule((x, 0.46, -0.1), (x, 0.16, 0.1), 0.0025), thread, S)
    # the cloth with its woven pattern
    def cloth(pl, pw, n):
        u, v = pl[:, 0] * 40, pl[:, 2] * 40
        motif = ((torch.sin(u) * torch.sin(v)) > 0.2).float()
        return {"albedo": srgb("#c8b28a") * (1 - motif)[:, None] + srgb("#8a2a3a") * motif[:, None]}
    sc.add(sd_box(0.21, 0.004, 0.06, 0.002), Mat("#c8b28a", rough=0.8, pattern=cloth), S * T(0, 0.13, 0.08) * RX(-0.3))
    # the shuttle crossing
    sx = {"idle": 0.18 * math.sin(TAU * t), "win": 0.18 * math.sin(TAU * 3 * t)}.get(anim, 0.0)
    sc.add(sd_ellipsoid(0.04, 0.012, 0.014), wood("#3a2416"), S * T(sx, 0.2, 0.06))
    # the chain of punched cards: along a loop on the right side of the head
    speed = {"idle": 1, "win": 3, "blocked": 0}.get(anim, 0)
    jam = 0.15 * math.sin(TAU * 4 * t) if anim == "blocked" else 0.0
    def card_pat(pl, pw, n):
        a = (torch.sin(pl[:, 0] * 220) > 0.6).float() * (torch.sin(pl[:, 2] * 140 + 1.3 * torch.floor(pl[:, 0] * 35)) > 0.3).float()
        return {"albedo": srgb("#e9dcb6") * (1 - 0.85 * a)[:, None]}
    card = Mat("#e9dcb6", rough=0.7, pattern=card_pat)
    N = 14
    for k in range(N):
        # the path: a stadium loop in the yz plane beside the head, cards advancing
        s = ((k + (speed * t * 4 if anim != "blocked" else jam)) / N) % 1      # whole cards per loop: periodic
        ang = TAU * s
        y = 0.43 + 0.2 * math.cos(ang)
        z = 0.15 + 0.07 * math.sin(ang)
        sc.add(sd_box(0.1, 0.004, 0.03, 0.002), card, S * T(-0.1, y, z) * RX(-ang + math.pi / 2))
    return sc


# ─── 5 · Machine à différences ──────────────────────────────────────────────
def figure_wheels(digit_phase):
    """The brass of a figure wheel with its ten engraved digits (dark marks
    around the rim, turned by digit_phase tenths of a turn)."""
    def pat(pl, pw, n):
        a = torch.atan2(pl[:, 2], pl[:, 0]) * 10 + digit_phase * TAU
        mark = ((torch.sin(a) > 0.55) & (pl[:, 1].abs() < 0.006)).float()
        return {"albedo": srgb("#d2aa52") * (1 - 0.75 * mark)[:, None]}
    return Mat("#d2aa52", metal=0.95, rough=0.25, pattern=pat)


def difference_engine(anim, t, side):
    """Babbage's difference engine: three columns of figure wheels stacked
    between two plates held by pillars, a crank on top. idle: the crank turns
    one turn and the wheels step, the right column carrying into the others;
    win: everything turns fast; blocked: the crank jerks back and forth."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    plate = Mat("#8a8e94", metal=0.85, rough=0.35)
    sc.add(sd_box(0.3, 0.014, 0.15, 0.006), plate, S * T(0, 0.03, 0))
    sc.add(sd_box(0.3, 0.012, 0.15, 0.006), plate, S * T(0, 0.52, 0))
    for x in (-0.28, 0.28):
        for z in (-0.13, 0.13):
            sc.add(sd_cylinder(0.25, 0.012, 0.004), STEEL, S * T(x, 0.275, z))
    crank = {"idle": TAU * t, "win": TAU * 3 * t, "blocked": 0.5 * math.sin(TAU * 4 * t)}.get(anim, 0.0)
    for c, x in enumerate((-0.17, 0.0, 0.17)):
        sc.add(sd_cylinder(0.24, 0.008), STEEL, S * T(x, 0.28, 0))                              # the column's axle
        for k in range(6):
            # the digit shown: each wheel steps by whole tenths; the steps add up to whole turns over the loop
            if anim == "idle":
                step = math.floor((t * 10 * (3 - c) + k * 0.37) % 10) / 10 * (1 if c == 2 or k < 3 else 0)
                step = (t * (3 - c) * (1 if (c == 2 or k < 3) else 0))   # smooth, whole turns per loop
            elif anim == "win":
                step = t * (2 + c)
            else:
                step = 0.0
            sc.add(sd_cylinder(0.014, 0.06, 0.004), figure_wheels(step * 10), S * T(x, 0.1 + k * 0.072, 0) * RY(TAU * step))
    # the crank: a gear on the top plate, an arm, a handle
    top = S * T(0.2, 0.54, 0.05)
    sc.add(sd_gear(0.06, 0.008, 18), BRASS, top * RY(crank))
    arm = top * RY(crank) * T(0, 0.02, 0)
    sc.add(sd_capsule((0, 0, 0), (0.09, 0, 0), 0.008), STEEL, arm)
    sc.add(sd_capsule((0.09, 0, 0), (0.09, 0.06, 0), 0.01), wood("#3a2416"), arm)
    return sc


# ─── 6 · Machine à rotors ───────────────────────────────────────────────────
def rotor_machine(anim, t, side):
    """A cipher machine of the 1930s: an open wooden case, its lid raised
    behind, three ribbed rotors on top, a board of lamps, a keyboard of round
    keys, the plug board on the front. idle: four keys are pressed in turn,
    a lamp lights for each, the right rotor steps; win: the lamps flash in a
    wave; blocked: it shakes, a red lamp blinks."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    case = wood("#4a3020")
    body = S * T(0, 0.1, 0)
    sc.add(sd_box(0.3, 0.085, 0.24, 0.012), case, body)
    sc.add(sd_box(0.3, 0.012, 0.24, 0.01), case, S * T(0, 0.33, -0.31) * RX(-1.35))        # the open lid
    deck = body * T(0, 0.087, 0)
    sc.add(sd_box(0.28, 0.004, 0.22, 0.003), Mat("#1c1c1e", rough=0.5), deck)
    nkeys = (9, 8, 9)
    presses = 4                                       # keys pressed per loop
    k_now = int(t * presses) % presses if anim == "idle" else -1   # % presses: t = 1 is t = 0 again
    lit = (k_now * 7 + 3) % 26 if anim == "idle" else -1
    u = (t * presses) % 1 if anim == "idle" else 0
    down = 0.012 * max(0.0, math.sin(math.pi * min(1.0, u * 2.5)))
    idx = 0
    key_cap = Mat("#e8e4d8", rough=0.35, clearcoat=0.5)
    for r, nk in enumerate(nkeys):
        for i in range(nk):
            x = (i - (nk - 1) / 2) * 0.058
            z = 0.13 + r * 0.045 - 0.045
            # keyboard
            pressed = idx == (k_now * 5 + 2) % 26
            sc.add(sd_cylinder(0.008, 0.019, 0.004), Mat("#2a2a2c", rough=0.4), deck * T(x, 0.012 - (down if pressed else 0), z + 0.03))
            sc.add(sd_cylinder(0.003, 0.016, 0.002), key_cap, deck * T(x, 0.021 - (down if pressed else 0), z + 0.03))
            # lamp board (behind the keys)
            on = 0.0
            if anim == "idle" and idx == lit:
                on = 1.0
            elif anim == "win":
                on = max(0.0, math.sin(TAU * (2 * t - (i / nk) * 0.5 - r * 0.15))) ** 2
            elif anim == "blocked" and idx == 13:
                on = 1.0 if (t * 6) % 1 < 0.5 else 0.0
            lamp = glow("#ffd24a" if anim != "blocked" else "#ff4a3a", 2 + 8 * on) if on > 0.05 else Mat("#5a5640", rough=0.25, clearcoat=1.0)
            sc.add(sd_cylinder(0.004, 0.015, 0.003), lamp, deck * T(x, 0.004, z - 0.13))
            idx += 1
    # three ribbed rotors, half sunk in the deck at the back (26 ribs: a step of 1/26 turn looks the same)
    step = {"idle": TAU / 26 * min(presses, int(t * presses) + min(1.0, ((t * presses) % 1) * 3)),
            "win": TAU * t, "blocked": 0.04 * math.sin(TAU * 5 * t)}.get(anim, 0.0)
    for j, x in enumerate((-0.07, 0.0, 0.07)):
        a = step if j == 2 else 0.0
        sc.add(sd_gear(0.045, 0.012, 26, 0.006), Mat("#bfc4ca", metal=0.9, rough=0.3), deck * T(x, 0.01, -0.19) * RZ(math.pi / 2) * RY(a))
    # the plug board on the front face: sockets and two patch cords
    front = body * T(0, -0.02, 0.242) * RX(math.pi / 2)
    sc.add(sd_box(0.26, 0.003, 0.05, 0.002), Mat("#18181a", rough=0.5,
           pattern=lambda pl, pw, n: {"albedo": srgb("#18181a") + srgb("#888888") * ((torch.sin(pl[:, 0] * 160) > 0.7) & (torch.sin(pl[:, 2] * 120) > 0.5)).float()[:, None] * 0.5}), front)
    for (a_, b_) in ((-0.12, 0.05), (0.02, 0.16)):
        pts = bezier((a_, 0.0, 0.25), (a_, 0.03, 0.32), (b_, 0.03, 0.32), (b_, 0.0, 0.25), 6)
        chain(sc, pts, [0.006] * 6, Mat("#b02a20", rough=0.5), S * T(0, 0.08, 0), blend=0.003)
    return sc


# ─── 7 · Calculateur à tubes ────────────────────────────────────────────────
def tube_computer(anim, t, side):
    """A vacuum-tube computer of the 1940s: two tall black panels at an angle,
    rows of glowing tubes, dials, patch cables hanging between the plug
    panels. idle: the tubes flicker; win: waves of light run over them;
    blocked: a row goes dark and shorts out in sparks."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    panel = Mat("#1a1c20", metal=0.3, rough=0.45)
    tube_glass = Mat("#c8d8e0", rough=0.05, glass=True)
    for pi_, (px, ang) in enumerate(((-0.19, 0.3), (0.19, -0.3))):
        P = S * T(px, 0, -0.04) * RY(ang)
        sc.add(sd_box(0.19, 0.33, 0.05, 0.008), panel, P * T(0, 0.33, 0))
        sc.add(sd_box(0.2, 0.012, 0.06, 0.004), STEEL, P * T(0, 0.665, 0))
        # rows of tubes on a shelf
        for r in range(3):
            y = 0.24 + r * 0.13
            sc.add(sd_box(0.18, 0.006, 0.03, 0.003), STEEL, P * T(0, y - 0.035, 0.06))
            for i in range(6):
                x = -0.15 + i * 0.06
                ph = (i * 0.31 + r * 0.53 + pi_ * 0.17) % 1
                if anim == "idle":
                    k = 2.5 + 1.5 * math.sin(TAU * (t + ph))
                elif anim == "win":
                    k = 3 + 7 * max(0.0, math.sin(TAU * (2 * t - x * 1.5 - r * 0.2))) ** 2
                elif anim == "blocked":
                    k = 0.0 if r == 1 else 2.5 + 1.5 * math.sin(TAU * (3 * t + ph))
                else:
                    k = 3.0
                sc.add(sd_capsule((0, 0, 0), (0, 0.06, 0), 0.016, 0.013), tube_glass, P * T(x, y - 0.03, 0.06))
                if k > 0.1:
                    sc.add(sd_capsule((0, 0.008, 0), (0, 0.045, 0), 0.008), glow("#ff8a2a", k), P * T(x, y - 0.03, 0.06))
                sc.add(sd_cylinder(0.008, 0.017, 0.003), Mat("#2a2a2a", rough=0.5), P * T(x, y - 0.032, 0.06))
        # dials at the bottom
        for i in range(3):
            sc.add(sd_cylinder(0.006, 0.022, 0.003), Mat("#e8e4d8", rough=0.4), P * T(-0.11 + i * 0.11, 0.1, 0.052) * RX(math.pi / 2))
    # patch cables hanging across the corner
    for j, col in enumerate(("#c8302a", "#e8c030", "#2a6ac8")):
        pts = bezier((-0.3 + j * 0.05, 0.6 - j * 0.06, 0.06), (-0.1, 0.3, 0.2), (0.1, 0.3, 0.2), (0.3 - j * 0.05, 0.56 - j * 0.05, 0.06), 7)
        chain(sc, pts, [0.007] * 7, Mat(col, rough=0.5), S, blend=0.003)
    if anim == "blocked":   # the dead row shorts out: sparks thrown from its tubes
        # On each panel, in front of the dark middle row (y = 0.37, tubes at
        # z = 0.06 in the panel's frame): short bright streaks flying out and up,
        # each living 30 % of a cycle; 4 cycles per loop, staggered, so u is
        # periodic in t. (An opaque smoke read as lumps at this size: none.)
        for pi_, (px, ang) in enumerate(((-0.19, 0.3), (0.19, -0.3))):
            P = S * T(px, 0, -0.04) * RY(ang)
            for i in range(3):
                u = (t * 4 + i * 0.33 + pi_ * 0.15) % 1
                if u > 0.3:
                    continue
                k = u / 0.3
                x0 = -0.12 + 0.12 * i
                a = 0.9 + 0.6 * ((i + pi_) % 3) / 2
                d = 0.02 + 0.12 * k
                x1, y1, z1 = x0 + d * math.cos(a) * (1 if i % 2 else -1), 0.37 + d * math.sin(a), 0.08 + 0.1 * k
                x2, y2, z2 = x0 + (x1 - x0) * 0.5, 0.37 + (y1 - 0.37) * 0.5, 0.08 + (z1 - 0.08) * 0.5
                sc.add(sd_capsule((x2, y2, z2), (x1, y1, z1), 0.007 * (1 - k) + 0.002), glow("#ffc65a", 10.0), P)
            sc.add(sd_sphere(0.012), glow("#ffe0a0", 6.0 + 6.0 * (math.sin(TAU * 8 * t) > 0)), P * T(-0.03 + 0.06 * pi_, 0.37, 0.07))   # the arc at the short
    return sc


# ─── 8 · Ordinateur à bandes ────────────────────────────────────────────────
def reel(sc, xf, R, a, tape_k):
    """A tape reel (axis z): the flange with three holes, the tape wound on it."""
    flange = Mat("#d8dde4", metal=0.6, rough=0.3)
    def holes(p):
        x, y = p[:, 0], p[:, 1]
        r = torch.sqrt(x * x + y * y)
        ang = torch.atan2(y, x)
        sec = TAU / 3
        a2 = torch.remainder(ang + sec / 2, sec) - sec / 2
        return torch.sqrt((r * torch.cos(a2) - R * 0.55) ** 2 + (r * torch.sin(a2)) ** 2) - R * 0.22
    def flange_sdf(p):
        d = torch.maximum(torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - R, p[:, 2].abs() - 0.004)
        return torch.maximum(d, -holes(p))
    sc.add(flange_sdf, flange, xf * RZ(a))
    sc.add(sd_cylinder(0.012, R * (0.3 + 0.45 * tape_k), 0.003), Mat("#3a2a1e", rough=0.4), xf * RX(math.pi / 2) * T(0, -0.012, 0))
    sc.add(sd_cylinder(0.014, R * 0.18, 0.003), STEEL, xf * RX(math.pi / 2))


def tape_mainframe(anim, t, side):
    """Two tape drives of a 1960s mainframe, side by side: tall grey cabinets,
    two reels behind a window, a row of lights below. idle: the reels turn,
    the lights blink; win: they spin fast, all lights on; blocked: they rock
    back and forth, a red light blinks."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    cab = Mat("#c8ccd0", rough=0.4, clearcoat=0.3)
    dark = Mat("#2a3038", rough=0.3, clearcoat=0.8)
    turns = {"idle": 1, "win": 4}.get(anim, 0)
    for c, x in enumerate((-0.15, 0.15)):
        C = S * T(x, 0, 0)
        sc.add(sd_box(0.13, 0.32, 0.12, 0.01), cab, C * T(0, 0.32, 0))
        sc.add(sd_box(0.11, 0.17, 0.004, 0.006), dark, C * T(0, 0.43, 0.121))                 # the window
        for j, y in enumerate((0.52, 0.33)):
            # 3 holes: 1/3 turn looks the same; whole thirds per loop
            a = TAU / 3 * (turns * 3 * t) * (1 if j == 0 else -1) + c * 0.7
            if anim == "blocked":
                a = 0.4 * math.sin(TAU * 3 * t) + c
            reel(sc, C * T(0, y, 0.13), 0.075, a, 0.6 if j == 0 else 0.35)
        sc.add(sd_box(0.03, 0.02, 0.006, 0.004), STEEL, C * T(0, 0.425, 0.13))               # the head between the reels
        # the lights below
        for i in range(6):
            ph = (i * 0.29 + c * 0.41) % 1
            on = {"idle": (t * 4 + ph) % 1 < 0.5, "win": True, "blocked": (i == 2 and (t * 6) % 1 < 0.5)}.get(anim, False)
            col = "#ff3a2a" if anim == "blocked" else ("#ffd24a" if i % 2 else "#4aff8a")
            sc.add(sd_sphere(0.012), glow(col, 6) if on else Mat("#4a4a4a", rough=0.3), C * T(-0.075 + i * 0.03, 0.18, 0.122))
        sc.add(sd_box(0.12, 0.006, 0.006, 0.003), STEEL, C * T(0, 0.12, 0.122))
    return sc


# ─── 9 · Micro-ordinateur ───────────────────────────────────────────────────
def sd_key_grid(nx, nz, sx, sz, kx, ky, kz, rr=0.002):
    """A block of keys in one primitive: nx × nz rounded boxes (half sizes kx,
    ky, kz) every sx along x and sz along z, centred (domain repetition)."""
    half = vec(kx - rr, ky - rr, kz - rr)        # made once (no copy to the GPU per call: CUDA graph capture)

    def f(p):
        ix = torch.clamp(torch.round(p[:, 0] / sx + (nx - 1) / 2), 0, nx - 1) - (nx - 1) / 2
        iz = torch.clamp(torch.round(p[:, 2] / sz + (nz - 1) / 2), 0, nz - 1) - (nz - 1) / 2
        q = torch.stack([p[:, 0] - ix * sx, p[:, 1], p[:, 2] - iz * sz], dim=-1)
        d = q.abs() - half
        return torch.linalg.norm(d.clamp_min(0), dim=-1) + d.max(dim=-1).values.clamp_max(0) - rr
    return f


def screen(anim, t):
    """The CRT's picture (a pattern, emissive): the blue screen of the time, lines
    of light-blue text and a blinking cursor; win: rolling bands of colour;
    blocked: red lines flashing."""
    def pat(pl, pw, n):
        u = pl[:, 0] / 0.13 * 0.5 + 0.5           # 0..1 across
        v = 0.5 - pl[:, 2] / 0.1 * 0.5            # 0..1 down
        inside = ((u > 0.08) & (u < 0.92) & (v > 0.1) & (v < 0.9)).float()
        bg = srgb("#2a2290")
        border = srgb("#6a5ee0")
        img = border * (1 - inside)[:, None] + bg * inside[:, None]
        row = torch.floor((v - 0.1) / 0.8 * 7)
        col = torch.floor((u - 0.08) / 0.84 * 18)
        txt = ((torch.sin(col * 7.3 + row * 3.1).abs() > 0.35) & (row < 3) & ((v - 0.1) / 0.8 * 7 % 1 > 0.25)).float() * inside
        if anim == "win":
            band = torch.remainder(v * 6 - t * 6, 6.0)           # whole bands per loop
            hues = [srgb(c) for c in ("#e04040", "#e0a040", "#e0e040", "#40e060", "#40a0e0", "#a040e0")]
            img = sum(h * ((band >= i) & (band < i + 1)).float()[:, None] for i, h in enumerate(hues))
            txt = txt * 0
        elif anim == "blocked":
            on = 1.0 if (t * 4) % 1 < 0.5 else 0.3
            img = img * (1 - inside)[:, None] + srgb("#5a1010") * inside[:, None]
            txt = ((row == 3) & (col < 9)).float() * inside * on
            img = img + srgb("#ff6050") * txt[:, None]
            return {"albedo": img * 0.2, "emissive": img * 1.6}
        cursor = ((row == 4) & (col == 0)).float() * inside * (1.0 if (t * 4) % 1 < 0.5 else 0.0)
        img = img + srgb("#9a90ff") * (txt + cursor).clamp(0, 1)[:, None]
        return {"albedo": img * 0.2, "emissive": img * 1.4}
    return Mat("#2a2290", rough=0.1, clearcoat=1.0, pattern=pat)


def micro8(anim, t, side):
    """An 8-bit home computer of the 1980s: the beige keyboard unit with its
    keys, a CRT monitor on top showing the blue screen and a blinking cursor,
    a cassette recorder beside it. idle: the cursor blinks, a key goes down;
    win: bands of colour roll over the screen; blocked: red error flashes."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    beige = plastic("#d8ccae")
    brown = plastic("#5a4a3a", 0.4)
    kb = S * T(0, 0.045, 0.12)
    sc.add(sd_box(0.3, 0.035, 0.12, 0.012), beige, kb)
    keys = kb * T(-0.02, 0.04, 0.0) * RX(0.12)
    sc.add(sd_key_grid(13, 4, 0.04, 0.042, 0.016, 0.01, 0.017), brown, keys)
    sc.add(sd_box(0.16, 0.01, 0.017, 0.003), brown, keys * T(0, 0, 0.105))                 # the space bar
    sc.add(sd_key_grid(1, 4, 0.04, 0.042, 0.016, 0.01, 0.017), plastic("#c8783a", 0.4), keys * T(0.29, 0, 0))   # function keys
    # the monitor: a deep rounded box, the screen bulging a little
    mon = S * T(0, 0.27, -0.1)
    sc.add(sd_box(0.2, 0.16, 0.14, 0.04), beige, mon)
    sc.add(sd_box(0.14, 0.11, 0.06, 0.03), beige, mon * T(0, 0, -0.16))
    front = mon * T(0, 0.01, 0.142) * RX(math.pi / 2)
    sc.add(sd_box(0.165, 0.006, 0.125, 0.02), Mat("#1a1a1a", rough=0.4), front)
    sc.add(sd_box(0.13, 0.008, 0.1, 0.03), screen(anim, t), front * T(0, 0.006, 0))
    sc.add(sd_sphere(0.008), glow("#ff3a2a", 5), mon * T(0.17, -0.13, 0.142))               # power light
    # the cassette recorder
    cas = S * T(0.38, 0.04, 0.0)
    sc.add(sd_box(0.07, 0.03, 0.1, 0.01), Mat("#2a2a2c", rough=0.4), cas)
    sc.add(sd_box(0.05, 0.006, 0.04, 0.004), Mat("#6a6a6a", rough=0.3, clearcoat=0.8), cas * T(0, 0.03, -0.03))
    return sc


# ─── 10 · Supercalculateur ──────────────────────────────────────────────────
def sd_ring_sector(R1, R2, h, a0, a1, rr=0.01):
    """The part of a thick ring (radii R1 < R2, half height h) between the
    angles a0 and a1 (around y, from +x towards +z)."""
    mid, half = (a0 + a1) / 2, (a1 - a0) / 2
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = torch.sqrt(x * x + z * z)
        a = torch.atan2(z, x) - mid
        a = torch.remainder(a + math.pi, TAU) - math.pi
        d_ang = (a.abs() - half) * r                 # distance to the cut faces (approximately)
        d_rad = torch.maximum(R1 - r, r - R2)
        d = torch.maximum(torch.maximum(d_rad, d_ang), y.abs() - h)
        return d
    return f


def supercomputer(anim, t, side):
    """The supercomputer of the 1970s: a tall cylinder of cabinets open on one
    side like a C, coloured panels, the padded bench all around its foot
    (the power supplies), lights on the top ring. idle: a light runs round the
    top; win: the panels glow in a wave round the C; blocked: they flash red."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    a0, a1 = -3 * math.pi / 2 + 0.8, math.pi / 2 - 0.8           # the solid part round the back: open towards the camera (+z)
    def panels(pl, pw, n):
        a = torch.atan2(pl[:, 2], pl[:, 0])
        k = 16
        seam = (torch.sin(a * k).abs() < 0.12).float()
        hue = 0.5 + 0.5 * torch.sin(a * 1.0)
        col = srgb("#a02a34") * hue[:, None] + srgb("#3a2a6a") * (1 - hue)[:, None]
        col = col * (1 - 0.7 * seam)[:, None]
        em = None
        if anim == "win":
            w = torch.clamp(torch.sin(a * 2 - TAU * 2 * t), 0, 1) ** 3
            em = col * w[:, None] * 2.5
        elif anim == "blocked":
            em = srgb("#ff2a1a") * (0.3 if (t * 4) % 1 < 0.5 else 0.0) * (1 - seam)[:, None] * ((pl[:, 1] > 0.1) & (pl[:, 1] < 0.16)).float()[:, None] * 6
        out = {"albedo": col}
        if em is not None:
            out["emissive"] = em
        return out
    tower = S * T(0, 0.33, 0)
    sc.add(sd_ring_sector(0.13, 0.3, 0.24, a0, a1), Mat("#c82a4a", rough=0.35, clearcoat=0.7, pattern=panels), tower)
    sc.add(sd_ring_sector(0.13, 0.3, 0.012, a0, a1), Mat("#2a2a2e", rough=0.3), tower * T(0, 0.25, 0))
    # the bench: leather cushions round the foot
    def leather(pl, pw, n):
        a = torch.atan2(pl[:, 2], pl[:, 0])
        seam = (torch.sin(a * 16).abs() < 0.08).float()
        return {"albedo": srgb("#b07a3a") * (1 - 0.5 * seam)[:, None] * (0.9 + 0.15 * fbm(pl * 20, 2))[:, None]}
    sc.add(sd_ring_sector(0.3, 0.44, 0.045, a0 - 0.06, a1 + 0.06, 0.02), Mat("#b07a3a", rough=0.6, clearcoat=0.3, pattern=leather), S * T(0, 0.055, 0))
    # lights on the top ring
    n = 12
    for i in range(n):
        a = a0 + (a1 - a0) * (i + 0.5) / n
        on = {"idle": int(t * n) % n == i,   # (% n: at t = 1 the first light again, the loop joins up)
              "win": True, "blocked": False}.get(anim, False)
        sc.add(sd_sphere(0.012), glow("#7affd0", 7) if on else Mat("#2a4a40", rough=0.3, clearcoat=1.0),
               tower * T(0.22 * math.cos(a), 0.265, 0.22 * math.sin(a)))
    return sc


# ─── 11 · Centre de calcul ──────────────────────────────────────────────────
def rack_front(anim, t, seed):
    """The front of a server rack: dark mesh, units, rows of tiny LEDs blinking
    (each at a whole number of blinks per loop)."""
    def pat(pl, pw, n):
        y = pl[:, 1]
        unit = torch.floor((y + 1) / 0.03)
        gap = ((y + 1) / 0.03 % 1 < 0.12).float()
        mesh = 0.8 + 0.2 * (torch.sin(pl[:, 0] * 500) * torch.sin(pl[:, 1] * 500) > 0).float()
        col = srgb("#1c1e22") * mesh[:, None] * (1 - 0.5 * gap)[:, None]
        cell = torch.floor(pl[:, 0] / 0.012)
        h = torch.frac(torch.sin(unit * 12.9898 + cell * 78.233 + seed * 3.1) * 43758.5)
        led = ((torch.frac((pl[:, 0] / 0.012)) < 0.45) & ((y + 1) / 0.03 % 1 > 0.55) & ((y + 1) / 0.03 % 1 < 0.8)).float()
        led = led * (pl[:, 0] > 0.02).float()
        k = torch.floor(h * 4) + 1                    # 1..4 blinks per loop
        if anim == "win":
            on = (torch.sin(TAU * (2 * t - y * 3)) > 0.2).float()
        elif anim == "blocked":
            on = torch.full_like(h, 1.0 if (t * 4) % 1 < 0.5 else 0.0)
        else:
            on = (torch.sin(TAU * (k * t + h)) > 0.3).float()
        ledcol = srgb("#ff3a2a") if anim == "blocked" else (srgb("#3aa0ff") * (h > 0.3).float()[:, None] + srgb("#3aff8a") * (h <= 0.3).float()[:, None])
        em = ledcol * (led * on)[:, None] * 3.0
        return {"albedo": col + em * 0.2, "emissive": em}
    return Mat("#1c1e22", metal=0.4, rough=0.4, pattern=pat)


def datacenter(anim, t, side):
    """An aisle of a computing centre: two rows of black server racks facing
    each other on a raised floor, their fronts full of blinking LEDs, a yellow
    cable tray above the aisle, cold light from the floor. idle: the LEDs
    blink; win: waves of light run up the racks; blocked: they flash red."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    floor = Mat("#9aa0a8", metal=0.3, rough=0.5,
                pattern=lambda pl, pw, n: {"albedo": srgb("#9aa0a8") * (1 - 0.4 * ((torch.remainder(pl[:, 0], 0.1) < 0.006) | (torch.remainder(pl[:, 2], 0.1) < 0.006)).float())[:, None]})
    sc.add(sd_box(0.34, 0.01, 0.3, 0.006), floor, S * T(0, 0.01, -0.02))
    rackm = Mat("#26282c", metal=0.5, rough=0.35)
    k = 0
    # two rows of racks, all facing us (the cold aisle in front of each row); the back row stands higher on its plinth
    for z, y0, h in ((-0.17, 0.05, 0.3), (0.13, 0.0, 0.27)):
        for x in (-0.2, 0.0, 0.2):
            R = S * T(x, y0 + h + 0.02, z)
            sc.add(sd_box(0.095, h, 0.09, 0.006), rackm, R)
            sc.add(sd_box(0.088, h - 0.015, 0.004, 0.002), rack_front(anim, t, k), R * T(0, 0, 0.092) * RX(math.pi / 2) * RX(-math.pi / 2))
            k += 1
        if y0 > 0:
            sc.add(sd_box(0.3, y0 / 2 + 0.01, 0.1, 0.004), Mat("#3a3e44", metal=0.4, rough=0.4), S * T(0, y0 / 2 + 0.01, z))
    # the cable tray above the back row, cables in it
    tray = Mat("#e0b020", metal=0.6, rough=0.4)
    for zz in (-0.21, -0.13):
        sc.add(sd_box(0.31, 0.012, 0.004, 0.002), tray, S * T(0, 0.72, zz))
    for x in (-0.28, -0.14, 0.0, 0.14, 0.28):
        sc.add(sd_box(0.004, 0.003, 0.04, 0.001), tray, S * T(x, 0.71, -0.17))
    sc.add(sd_box(0.3, 0.008, 0.03, 0.006), Mat("#2a5ac8", rough=0.5), S * T(0, 0.72, -0.17))
    # cold light along the floor in front of each row
    for z in (0.25, -0.05):
        sc.add(sd_box(0.3, 0.002, 0.012, 0.001), glow("#7ac8ff", 2.5), S * T(0, 0.022 + (0.05 if z < 0 else 0), z))
    return sc


# ─── 12 · Ordinateur quantique ──────────────────────────────────────────────
def quantum(anim, t, side):
    """The inside of a quantum computer: the golden "chandelier" of the dilution
    refrigerator hanging from its frame, stages of gold plates getting smaller
    downwards, rods between them, coiled cables, the chip at the bottom glowing
    blue. idle: the chip breathes, sparks of light circle it; win: rings of
    light rise along the stages; blocked: the light stutters, red."""
    sc = Scene()
    base(sc, side)
    S = stand(anim, t)
    # the frame: three slender posts and a ring, the chandelier hanging from it
    for a in range(3):
        ang = TAU * a / 3 + math.pi / 2
        sc.add(sd_cylinder(0.42, 0.008, 0.003), DARK_STEEL, S * T(0.4 * math.cos(ang), 0.42, 0.4 * math.sin(ang)))
        sc.add(sd_capsule((0.4 * math.cos(ang), 0.84, 0.4 * math.sin(ang)), (0, 0.84, 0), 0.007), DARK_STEEL, S)
    sc.add(sd_torus(0.4, 0.01), DARK_STEEL, S * T(0, 0.84, 0))
    sc.add(sd_cylinder(0.014, 0.3, 0.006), GOLD, S * T(0, 0.8, 0))
    levels = [(0.64, 0.23), (0.49, 0.165), (0.36, 0.115), (0.25, 0.08)]
    for i, (y, R) in enumerate(levels):
        sc.add(sd_cylinder(0.009, R, 0.005), GOLD, S * T(0, y, 0))
        if i == 0:
            top_y = 0.8
        else:
            top_y = levels[i - 1][0]
        nrod = 6
        for k in range(nrod):
            a = TAU * k / nrod + i * 0.3
            r = R * 0.8
            sc.add(sd_capsule((r * math.cos(a), y, r * math.sin(a)), (r * math.cos(a), top_y, r * math.sin(a)), 0.006), GOLD, S)
        # coiled cables between the stages (silver), a few turns each
        for k in range(3):
            a = TAU * k / 3 + 0.6 + i
            pts = []
            for j in range(9):
                u = j / 8
                yy = top_y + (y - top_y) * u
                ra = R * 0.45 + 0.012 * math.sin(u * TAU * 3)
                pts.append((ra * math.cos(a + 0.4 * u), yy, ra * math.sin(a + 0.4 * u)))
            chain(sc, pts, [0.005] * len(pts), Mat("#c8ccd4", metal=0.9, rough=0.25), S, blend=0.002)
    # the cold finger and the chip
    sc.add(sd_cylinder(0.08, 0.035, 0.01), GOLD, S * T(0, 0.22, 0))
    if anim == "blocked":
        k = 2.0 + 6.0 * (1.0 if (t * 5) % 1 < 0.4 else 0.0)
        chip = glow("#ff4a3a", k)
    else:
        k = {"idle": 5 + 3 * math.sin(TAU * t), "win": 9 + 4 * math.sin(TAU * 2 * t)}.get(anim, 6)
        chip = glow("#5ab8ff", k)
    sc.add(sd_box(0.04, 0.01, 0.04, 0.004), chip, S * T(0, 0.135, 0))
    # sparks circling the chip (qubits): whole turns per loop
    for q in range(5):
        a = TAU * (t * (1 if anim != "win" else 2) + q / 5)
        yq = 0.16 + 0.03 * math.sin(TAU * (t + q / 5))
        sc.add(sd_sphere(0.008), glow("#bfe8ff", 7), S * T(0.09 * math.cos(a), yq, 0.09 * math.sin(a)))
    if anim == "win":       # rings of light climbing the stages
        for r_ in range(2):
            u = (t * 2 + r_ * 0.5) % 1
            y = 0.2 + 0.55 * u
            R = 0.08 + 0.2 * u
            sc.add(sd_torus(R, 0.006), glow("#8ad8ff", 6 * (1 - u) + 1), S * T(0, y, 0))
    return sc


SCALE = {"random": 1.35, "beginner": 1.3, "monte_carlo": 1.3, "mcts": 1.3, "basic_minmax": 1.25, "minmax": 1.25,
         "features": 1.25, "mcts_nnue": 1.25, "nnue193": 1.2, "resnet": 1.2, "dev": 1.2, "alphazero": 1.2}
CAST = {"random": ("Boulier", abacus), "beginner": ("Mécanisme d'Anticythère", antikythera),
        "monte_carlo": ("Pascaline", pascaline), "mcts": ("Métier Jacquard", jacquard),
        "basic_minmax": ("Machine à différences", difference_engine), "minmax": ("Machine à rotors", rotor_machine),
        "features": ("Calculateur à tubes", tube_computer), "mcts_nnue": ("Ordinateur à bandes", tape_mainframe),
        "nnue193": ("Micro-ordinateur", micro8), "resnet": ("Supercalculateur", supercomputer),
        "dev": ("Centre de calcul", datacenter), "alphazero": ("Ordinateur quantique", quantum)}
COMPUTERS = {"cpu_" + k: f for k, (_, f) in CAST.items()}
NAMES = {"cpu_" + k: n for k, (n, _) in CAST.items()}


if __name__ == "__main__":
    import sys
    import numpy as np
    from PIL import Image
    from render import render
    from models import scaled, SCALE as MSCALE
    names = sys.argv[2:] if len(sys.argv) > 2 else list(COMPUTERS)
    for nm in names:
        MSCALE[nm] = SCALE[nm[4:]]
        f = scaled(nm, COMPUTERS[nm])
        tiles = []
        for anim in ("idle", "blocked", "win"):
            for k in range(4):
                tiles.append(render(f(anim, k / 4, 0 if k % 2 == 0 else 1), size=192))
        sheet = Image.new("RGBA", (192 * 4, 192 * 3), (40, 44, 52, 255))
        for i, im in enumerate(tiles):
            sheet.alpha_composite(Image.fromarray(im), ((i % 4) * 192, (i // 4) * 192))
        sheet.save(f"/tmp/claude-1000/-home-elucterio-Yolah/f1710e3a-943c-4701-b5af-ebd93af4fff8/scratchpad/{nm}.png")
        print(nm, flush=True)
