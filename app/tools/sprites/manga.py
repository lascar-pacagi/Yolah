"""
manga.py — the "Manga" set: one original character per AI, from the weakest
to the strongest, and the user drawn in the same style; rendered with the
toon mode of render.py (flat bands, ink lines). Their power shows three ways:
gems on the base (1 to 5), an aura of flames behind them (from the middle of
the ladder on, growing), and hair that stands up more and more.

Built on the humanoid rig of legends.py, with an anime head (anime_head):
big eyes with two highlights, a thick upper lash line, a small nose and
mouth, hair made of spikes (cones) around the skull.
"""
import math
import torch
from render import Scene, Mat, T, RX, RY, RZ, vec, srgb, sd_sphere, sd_ellipsoid, sd_box, sd_capsule, sd_cylinder, \
    sd_torus, sd_cone, sd_plane_cut, length
from models import base, BASE_TOP, TAU, wave, glow, gold, sd_lid, sd_lower_lid
from legends import humanoid, base_pose, held


def flat(c, rough=0.6, metal=0.0):
    return Mat(c, rough=rough, metal=metal)


INK = "#1b1420"


# ─── the head ───────────────────────────────────────────────────────────────
def anime_eye(iris, w=1.0, h=1.0):
    """A flat anime eye: white, a tall iris darker at the top, a pupil, two
    highlights (emissive), in eye-local coordinates (z out of the face)."""
    ic = srgb(iris)

    def pat(pl, pw, n):
        x, y = pl[:, 0] / w, pl[:, 1] / h          # −1..1 across the eye
        r_iris = ((x / 0.62) ** 2 + ((y + 0.05) / 0.8) ** 2)
        a = torch.where((r_iris < 0.36)[:, None], ic * (0.55 + 0.6 * ((y + 0.3) / 0.9).clamp(0, 1))[:, None], srgb("#fbfaf7").expand_as(pl))
        a = torch.where((((x / 0.35) ** 2 + ((y + 0.02) / 0.45) ** 2) < 0.36 * 0.36 * 4)[:, None] & (r_iris < 0.36)[:, None], srgb("#120c10").expand_as(pl), a)
        hl = ((((x + 0.12) / 0.14) ** 2 + ((y - 0.18) / 0.14) ** 2) < 1).float() + 0.8 * ((((x - 0.1) / 0.07) ** 2 + ((y + 0.15) / 0.07) ** 2) < 1).float()
        return {"albedo": a, "emissive": vec(1, 1, 1) * (hl.clamp(0, 1) * 1.6)[:, None]}
    return Mat("#ffffff", rough=0.5, pattern=pat)


def anime_head(sc, x, r, skin_c, hair, iris="#3a6ad0", expr="calm", spikes=None, extra=None, t=0.0, blink=False):
    """expr: calm, smirk, determined, worried, happy, angry, fierce."""
    sk = flat(skin_c)
    sc.add(sd_ellipsoid(r, r * 0.98, r * 0.95), sk, x)
    sc.add(sd_ellipsoid(r * 0.72, r * 0.6, r * 0.75), sk, x * T(0, -r * 0.42, r * 0.2), blend=r * 0.35)   # a pointed chin
    for sgn in (-1, 1):
        sc.add(sd_ellipsoid(r * 0.14, r * 0.22, r * 0.08), sk, x * T(r * 0.96 * sgn, -r * 0.05, 0))      # ears
    # eyes: flat discs on the face, tall
    for sgn in (-1, 1):
        ec = x * T(r * 0.29 * sgn, -r * 0.14, r * 0.9) * RY(0.28 * sgn)
        if blink or expr == "happy":
            sc.add(sd_capsule((-r * 0.17, -r * 0.02, 0), (0, r * 0.06, 0), r * 0.03), flat(INK), ec * T(0, 0, r * 0.06))
            sc.add(sd_capsule((0, r * 0.06, 0), (r * 0.17, -r * 0.02, 0), r * 0.03), flat(INK), ec * T(0, 0, r * 0.06))
            continue
        h = r * (0.27 if expr not in ("angry", "fierce", "determined", "smirk") else 0.2)
        sc.add(sd_ellipsoid(r * 0.19, h, r * 0.06), anime_eye(iris, r * 0.19, h), ec)
        # upper lash line, tilted by the expression
        tilt = {"angry": -0.45, "fierce": -0.55, "determined": -0.3, "worried": 0.35, "smirk": -0.15}.get(expr, -0.05)
        sc.add(sd_capsule((-r * 0.22, 0, 0), (r * 0.22, 0, 0), r * 0.03), flat(INK),
               ec * T(0, h * 0.92, r * 0.03) * RZ(-tilt * sgn))
        # brows
        sc.add(sd_capsule((-r * 0.16, 0, 0), (r * 0.16, 0, 0), r * 0.03), flat(hair["c"]),
               ec * T(0, h + r * 0.2, r * 0.02) * RZ(-(tilt * 1.4) * sgn))
    sc.add(sd_ellipsoid(r * 0.05, r * 0.06, r * 0.05), sk, x * T(0, -r * 0.32, r * 0.95), blend=r * 0.05)   # nose
    mo = x * T(0, -r * 0.55, r * 0.93)
    if expr in ("happy", "fierce", "angry"):
        sc.add(sd_ellipsoid(r * 0.18, r * 0.12, r * 0.05), flat("#7a1a28"), mo)
        if expr in ("fierce", "angry"):
            sc.add(sd_box(r * 0.15, r * 0.025, r * 0.04, r * 0.01), flat("#ffffff"), mo * T(0, r * 0.07, r * 0.01))
    elif expr == "worried":
        sc.add(sd_capsule((-r * 0.1, -r * 0.03, 0), (r * 0.1, -r * 0.03, 0), r * 0.025), flat(INK), mo * T(0, 0, r * 0.02) * RZ(0.15))
    elif expr == "smirk":
        sc.add(sd_capsule((-r * 0.1, 0, 0), (r * 0.12, r * 0.04, 0), r * 0.02), flat(INK), mo * T(0, 0, r * 0.03))
    else:
        sc.add(sd_capsule((-r * 0.09, 0, 0), (r * 0.09, 0, 0), r * 0.02), flat(INK), mo * T(0, 0, r * 0.03))
    # hair: a cap, then spikes around the skull (a list of (theta, phi, length, width))
    hm = flat(hair["c"], 0.95 if hair.get("buzz") else 0.45)
    zig = hair.get("zig", 7.0)

    def cap(p):
        # the hair over the skull, down to a zigzag fringe above the brows at
        # the front, lower at the back
        if hair.get("buzz"):      # very short: a thin matte shell, a high hairline receding at the temples
            ell = sd_ellipsoid(r * 1.025, r * 1.025, r * 1.02)(p)
            front = ((p[:, 2] / r - 0.1) * 2.5).clamp(0, 1)
            edge = r * (0.42 + 0.35 * (p[:, 0] / r) ** 2) * front + (-r * 0.3) * (1 - front)
            return torch.maximum(ell, (edge - p[:, 1]) * 0.7)
        ell = sd_ellipsoid(r * 1.08, r * 1.08, r * 1.07)(p)
        front = ((p[:, 2] / r - 0.1) * 2.5).clamp(0, 1)
        edge = (r * (0.16 + 0.1 * torch.sin(p[:, 0] / r * zig).abs())) * front + (-r * 0.55) * (1 - front)
        return torch.maximum(ell, (edge - p[:, 1]) * 0.7)
    sc.add(cap, hm, x * T(0, r * 0.04, -r * 0.01))
    for (th, ph, ln, wd) in hair["spikes"]:
        # th: angle around the vertical axis (0 = front), ph: elevation; the spike points outwards
        d = (math.sin(th) * math.cos(ph), math.sin(ph), math.cos(th) * math.cos(ph))
        rot = RY(th) * RX(math.pi / 2 - ph)
        sc.add(sd_cone(ln * r * 0.5, wd * r, r * 0.02, r * 0.01), hm,
               x * T(d[0] * r * 0.82, d[1] * r * 0.82 + r * 0.05, d[2] * r * 0.82) * rot * T(0, ln * r * 0.5, 0), blend=r * 0.08)
    if extra:
        extra(sc, x, r)


def spikes_ring(n, ph, ln, wd, th0=-2.6, th1=2.6, jitter=0.0, seed=0):
    out = []
    for k in range(n):
        th = th0 + (th1 - th0) * k / max(1, n - 1)
        out.append((th + jitter * math.sin(k * 7.1 + seed), ph + jitter * 0.5 * math.cos(k * 3.3 + seed), ln, wd))
    return out


# ─── power: gems on the base, an aura ──────────────────────────────────────
def gems(sc, tier):
    for k in range(tier):
        a = math.pi / 2 + (k - (tier - 1) / 2) * 0.22          # in front, on the base's rim
        col = "#ffd24a" if tier >= 5 else "#7fe0ff" if tier >= 3 else "#ff8a5a"
        sc.add(sd_ellipsoid(0.03, 0.022, 0.03), glow(col, 4.0 + tier), T(0.5 * math.cos(a) * 0.98, 0.1, 0.5 * math.sin(a) * 0.98))


def aura(sc, level, col, t, body):
    """level 0..1: flames rising behind the figure (thin glowing cones), flickering."""
    if level <= 0:
        return
    n = int(6 + 10 * level)
    for k in range(n):
        a = -1.3 + 2.6 * k / (n - 1)
        h = (0.25 + 0.45 * level) * (0.75 + 0.35 * math.sin(TAU * (t * 3 + k * 0.37)))
        x = 0.3 * math.sin(a) * (0.9 + 0.3 * level)
        z = -0.12 - 0.12 * math.cos(a)
        sc.add(sd_cone(h * 0.5, 0.035 + 0.03 * level, 0.002, 0.002), glow(col, 3.0 + 6.0 * level),
               body * T(x, BASE_TOP + 0.12 + h * 0.5, z) * RZ(-0.25 * math.sin(a)))


# ─── one character: rig + head + outfit + power ─────────────────────────────
def character(anim, t, side, spec):
    sc = Scene()
    base(sc, side)
    p = base_pose(anim, t)
    if "pose" in spec:
        spec["pose"](p, anim, t)
    m = {"skin": flat(spec.get("skin", "#f6d6c0")), "top": flat(spec["top"]), "legs": flat(spec["legs"]),
         "boots": flat(spec.get("boots", "#2a2a30")), "arms": flat(spec.get("arms", spec["top"]))}
    HR = 0.25
    rig = humanoid(sc, p, m, w=spec.get("w", 0.95), h=spec.get("h", 0.85), head_r=HR)
    if "outfit" in spec:
        spec["outfit"](sc, rig, anim, t)
    expr = {"idle": spec.get("expr", "calm"), "move": "determined", "blocked": "worried", "win": "happy"}[anim]
    if spec.get("tier", 1) >= 4 and anim == "win":
        expr = "fierce"
    head = rig["head"] * RX(-0.2)
    anime_head(sc, head, HR, spec.get("skin", "#f6d6c0"), spec["hair"], spec.get("iris", "#3a6ad0"), expr,
               extra=spec.get("head_extra"), t=t, blink=(anim == "idle" and abs(t - 0.55) < 0.04))
    if "hands" in spec:
        spec["hands"](sc, rig, anim, t)
    gems(sc, spec.get("tier", 1))
    lvl = spec.get("aura", 0.0) * (1.4 if anim == "win" else 1.0)
    aura(sc, min(1.0, lvl), spec.get("aura_col", "#ffd24a"), t, rig["body"])
    return sc


# hair shapes
MESSY = {"c": "#3a2a24", "spikes": spikes_ring(7, 0.5, 0.45, 0.28, jitter=0.4) + spikes_ring(4, 0.15, 0.4, 0.25, -0.6, 0.6, 0.3)}
PONY = {"c": "#c8484a", "spikes": spikes_ring(5, 0.2, 0.3, 0.3, -0.7, 0.7, 0.2) + [(math.pi, 0.1, 1.2, 0.3)]}
CAP_HAIR = {"c": "#f2c84a", "spikes": spikes_ring(5, 0.0, 0.35, 0.28, -1.2, 1.2, 0.25)}
SWORD_HAIR = {"c": "#20283a", "spikes": spikes_ring(9, 0.6, 0.6, 0.27, jitter=0.25) + spikes_ring(3, 0.2, 0.45, 0.25, -0.5, 0.5)}
BRAWL_HAIR = {"c": "#c86a2a", "spikes": spikes_ring(6, 0.9, 0.5, 0.3, -1.5, 1.5, 0.2)}
MAGE_HAIR = {"c": "#6a4ac8", "spikes": spikes_ring(6, 0.1, 0.35, 0.3, -1.0, 1.0, 0.2) + [(2.6, -0.4, 1.0, 0.35), (-2.6, -0.4, 1.0, 0.35)]}
RONIN_HAIR = {"c": "#181820", "spikes": spikes_ring(8, 0.4, 0.55, 0.26, jitter=0.3) + [(math.pi, -0.2, 0.9, 0.25)]}
PSY_HAIR = {"c": "#e8e4f8", "spikes": [(a, -0.25 + 0.3 * math.sin(a * 3), 0.9, 0.3) for a in [1.4 + i * (2 * math.pi - 2.8) / 8 for i in range(9)]]}
HERO_HAIR = {"c": "#d8342a", "spikes": spikes_ring(11, 0.8, 0.9, 0.28, jitter=0.2) + spikes_ring(5, 0.3, 0.7, 0.28, -0.8, 0.8, 0.2)}
BOSS_HAIR = {"c": "#f4f2f6", "spikes": spikes_ring(9, 0.3, 0.8, 0.26, jitter=0.3) + [(math.pi + d, -0.3, 1.4, 0.3) for d in (-0.4, 0, 0.4)]}


def sword(sc, hand, anim, ln=0.32, col="#d8dce2"):
    sw = held(hand, -math.pi if anim == "win" else -1.3)
    sc.add(sd_capsule((0, -0.06, 0), (0, 0.04, 0), 0.014), flat("#3a2a50"), sw)
    sc.add(sd_box(0.045, 0.008, 0.02, 0.004), flat("#d8b040", 0.4, 1.0), sw * T(0, 0.05, 0))
    sc.add(sd_box(0.016, ln, 0.004, 0.003), flat(col, 0.2, 1.0), sw * T(0, 0.06 + ln, 0))


SPECS = {
    "random": dict(name="Tobi", tier=1, top="#2a2e44", legs="#2a2e44", hair=MESSY, iris="#6a4a2a", expr="calm",
                   head_extra=lambda sc, x, r: sc.add(sd_box(r * 0.16, r * 0.07, r * 0.02, r * 0.02), flat("#f2e2c8"),
                                                      x * T(r * 0.5, -r * 0.3, r * 0.8) * RY(0.6) * RZ(0.4)),
                   outfit=lambda sc, rig, a, t: [sc.add(sd_sphere(0.012), flat("#d8b040", 0.4, 1.0), rig["chest"] * T(0, 0.06 - 0.05 * k, 0.015)) for k in range(3)]),
    "beginner": dict(name="Yuna", tier=1, top="#3a5ac8", legs="#2a2a40", hair=PONY, iris="#2aa86a", expr="determined",
                     head_extra=lambda sc, x, r: (sc.add(sd_torus(r * 1.02, r * 0.07), flat("#2a2a40"), x * T(0, r * 0.32, 0) * RX(0.18)),
                                                  sc.add(sd_box(r * 0.2, r * 0.12, r * 0.03, r * 0.02), flat("#c8ccd4", 0.3, 1.0), x * T(0, r * 0.42, r * 0.93) * RX(0.2))),
                     outfit=lambda sc, rig, a, t: sc.add(sd_torus(0.07, 0.03), flat("#e84a3a"), rig["torso"] * T(0, 0.3, 0.01))),
    "monte_carlo": dict(name="Rin", tier=2, top="#e8e4dc", legs="#c83a4a", hair=CAP_HAIR, iris="#c8484a", expr="smirk",
                        head_extra=lambda sc, x, r: (sc.add(sd_plane_cut(sd_ellipsoid(r * 1.1, r * 0.75, r * 1.1), (0, -1, 0), 0.0), flat("#c83a4a"), x * T(0, r * 0.35, 0)),
                                                     sc.add(sd_ellipsoid(r * 0.6, r * 0.06, r * 0.5), flat("#c83a4a"), x * T(0, r * 0.38, r * 0.85) * RX(-0.2))),
                        hands=lambda sc, rig, a, t: [sc.add(sd_box(0.03, 0.045, 0.003, 0.004), flat("#ffffff" if k % 2 else "#f8e8e8"),
                                                            rig["r"] * RX(-1.2) * RZ(-0.5 + 0.25 * k) * T(0, 0.06, 0.005 * k)) for k in range(4)]),
    "mcts": dict(name="Kaito", tier=2, top="#2a6a5a", legs="#e8e4dc", hair=SWORD_HAIR, iris="#3a8ac8", expr="calm", aura=0.0,
                 hands=lambda sc, rig, a, t: sword(sc, rig["r"], a)),
    "basic_minmax": dict(name="Gorō", tier=3, top="#e8e4dc", legs="#3a3a48", hair=BRAWL_HAIR, iris="#8a5a2a", expr="smirk", w=1.15,
                         aura=0.25, aura_col="#ff8a3a",
                         head_extra=lambda sc, x, r: sc.add(sd_torus(r * 1.02, r * 0.08), flat("#e83a3a"), x * T(0, r * 0.3, 0) * RX(0.15)),
                         hands=lambda sc, rig, a, t: [sc.add(sd_sphere(0.06), flat("#c82a2a", 0.35), rig[h]) for h in ("l", "r")]),
    "nnue193": dict(name="Mei", tier=3, top="#4a3a8a", legs="#4a3a8a", hair=MAGE_HAIR, iris="#8a5ae8", expr="calm", aura=0.35, aura_col="#8ab0ff",
                    head_extra=lambda sc, x, r: [sc.add(sd_torus(r * 0.17, r * 0.025), flat("#2a2a30", 0.3, 0.8), x * T(r * 0.36 * s_, -r * 0.08, r * 0.95) * RY(0.35 * s_) * RX(math.pi / 2))
                                                 for s_ in (-1, 1)],
                    outfit=lambda sc, rig, a, t: sc.add(sd_cone(0.12, 0.17, 0.1, 0.02), flat("#4a3a8a"), rig["torso"] * T(0, -0.04, 0)),
                    hands=lambda sc, rig, a, t: sc.add(sd_sphere(0.045), glow("#8ab0ff", 7.0 + 3 * wave(t, 2)), rig["l"] * T(0, -0.02, 0.06))),
    "features": dict(name="Shin", tier=4, top="#2a2a30", legs="#2a2a30", hair=RONIN_HAIR, iris="#c8a03a", expr="smirk", aura=0.45, aura_col="#6affd0",
                     outfit=lambda sc, rig, a, t: (sc.add(sd_cone(0.17, 0.18, 0.12, 0.02), flat("#2a2a30"), rig["torso"] * T(0, -0.03, 0)),
                                                   sc.add(sd_torus(0.08, 0.035), flat("#c83a3a"), rig["torso"] * T(0, 0.3, 0.01))),
                     hands=lambda sc, rig, a, t: sword(sc, rig["r"], a, 0.36, "#e8f4ff")),
    "resnet": dict(name="Aiko", tier=4, top="#f2f0f8", legs="#6a4ac8", hair=PSY_HAIR, iris="#d84ae8", expr="calm", aura=0.65, aura_col="#c86aff",
                   hands=lambda sc, rig, a, t: [sc.add(sd_ellipsoid(0.03, 0.025, 0.03), flat("#6a6070", 0.5),
                                                       rig["body"] * T(0.3 * math.cos(TAU * (t + k / 3)), 0.75 + 0.05 * math.sin(TAU * (2 * t + k / 3)), 0.3 * math.sin(TAU * (t + k / 3))))
                                                for k in range(3)]),
    "dev": dict(name="Ryū", tier=5, top="#1e2a4a", legs="#e8e4dc", hair=HERO_HAIR, iris="#ffb02a", expr="determined", aura=0.85, aura_col="#4ab0ff",
                outfit=lambda sc, rig, a, t: sc.add(sd_torus(0.15, 0.022), flat("#2a3a8a"), rig["torso"] * T(0, 0.07, 0)),
                hands=lambda sc, rig, a, t: [sc.add(sd_sphere(0.05), glow("#8ad8ff", 6.0 + 4 * (a == "win")), rig[h]) for h in ("l", "r")]),
    "alphazero": dict(name="Kuro", tier=5, top="#1a1424", legs="#1a1424", hair=BOSS_HAIR, iris="#ff2a4a", expr="smirk", aura=1.0, aura_col="#a04aff", skin="#e8dcd8",
                      outfit=lambda sc, rig, a, t: (sc.add(sd_cone(0.17, 0.2, 0.13, 0.02), flat("#1a1424"), rig["torso"] * T(0, -0.02, 0)),
                                                    sc.add(sd_plane_cut(sd_ellipsoid(0.17, 0.12, 0.13), (0, 0, 1), 0.02), flat("#3a1a4a"), rig["torso"] * T(0, 0.36, -0.02)),
                                                    [sc.add(sd_cone(0.08, 0.03, 0.004, 0.004), flat("#e8d8a8", 0.4), rig["head"] * T(0.12 * s_, 0.2, 0) * RZ(-0.5 * s_) * T(0, 0.08, 0))
                                                     for s_ in (-1, 1)])),
}
def buzz_extra(sc, x, r):
    """The user's stubble: a flat darker zone on the jaw and the upper lip (inked)."""
    beard = flat("#c99c86")
    sc.add(sd_plane_cut(sd_ellipsoid(r * 0.75, r * 0.55, r * 0.78), (0, 1, 0.35), -r * 0.18), beard, x * T(0, -r * 0.47, r * 0.2), blend=0.0)
    sc.add(sd_ellipsoid(r * 0.2, r * 0.05, r * 0.05), beard, x * T(0, -r * 0.45, r * 0.93))


def tee_outfit(side):
    return lambda sc, rig, a, t: sc.add(sd_torus(0.06, 0.018), flat("#d8d4cc" if side == 0 else "#2a2a30"), rig["torso"] * T(0, 0.29, 0.01))


BUZZ = {"c": "#4a403a", "spikes": [], "zig": 0.0, "buzz": True}


def human_manga(anim, t, side):
    spec = dict(name="Vous", tier=0, top="#1b1c20" if side == 0 else "#ebe9e4", legs="#2a3448", hair=BUZZ, iris="#7fa0b8",
                expr="calm", skin="#f0c8b0", head_extra=buzz_extra, outfit=tee_outfit(side))
    return character(anim, t, side, spec)


# the strength ladder: the roster's order, weakest first (ai_roster.json)
ORDER = ["random", "beginner", "monte_carlo", "mcts", "basic_minmax", "nnue193", "features", "resnet", "dev", "alphazero"]


def make(ai):
    return lambda anim, t, side: character(anim, t, side, SPECS[ai])


MANGA = {"manga_" + ai: make(ai) for ai in ORDER}
MANGA["manga_human"] = human_manga
NAMES = {"manga_" + ai: SPECS[ai]["name"] for ai in ORDER}
NAMES["manga_human"] = "Humain (manga)"
