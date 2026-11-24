"""
pixel_ships.py — the "8 bits" set: one spaceship per AI in the pixel art of
the home computers of the 1980s (Commodore 64 style: multicolour sprites,
pixels twice as wide as high, the machine's 16-colour palette), from the
smallest pod to the mothership.

  random        Capsule          a little pod, one porthole
  beginner      Navette          a small shuttle, stubby wings
  monte_carlo   Chasseur         a fighter, swept wings
  mcts          Éclaireur        a scout, long nose, thin wings
  basic_minmax  Intercepteur     a delta wing
  minmax        Chasseur lourd   twin engines, wing guns
  features      Bombardier       broad wings, bomb bay
  mcts_nnue     Soucoupe         a flying saucer, lights round its rim
  nnue193       Canonnière       side pods, four guns
  resnet        Frégate          a long hull, two engine pods
  dev           Croiseur         a heavy cruiser, turrets
  alphazero     Vaisseau-mère    the mothership: the whole sprite, rows of lights

Each ship is drawn here as text, the left half of a 12 × 21 multicolour
sprite (one character = one double-wide pixel), mirrored for the right half:
  .  transparent   a  hull (the side's colour: dark grey or white)
  d  hull shade    b  outline (black)     c  the ship's accent colour
  e  an engine nozzle (the flame comes out below it)
  g  a gun (the shots leave from it when the ship wins)
  l  a light (blinks)

Animations (all loops join up: whole cycles over the frames):
  idle     8 frames: the flames flicker, the ship hovers up and down a pixel,
           its lights blink
  move     8 frames: full thrust, long flames
  blocked  8 frames: the hull flashes red, the ship shakes, engines off
  win      8 frames: it fires, shots climbing from its guns, flames on

Frames are drawn on a 192 px canvas, each C64 pixel a block of 6 × 6 (hires
pixel 6 px wide, so a multicolour pixel is 12 × 6), no smoothing: the app
shows them as figurines ("smooth": the big blocks survive the filtering).

    cd app/tools/sprites && python3 pixel_ships.py [preview]
"""
import json
import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "..", "assets", "art")

PAL = {   # the C64 palette (Pepto's measurements)
    "black": "#000000", "white": "#ffffff", "red": "#68372b", "cyan": "#70a4b2", "purple": "#6f3d86",
    "green": "#588d43", "blue": "#352879", "yellow": "#b8c76f", "orange": "#6f4f25", "brown": "#433900",
    "lred": "#9a6759", "dgrey": "#444444", "grey": "#6c6c6c", "lgreen": "#9ad284", "lblue": "#6c5eb5", "lgrey": "#959595",
}
SIDES = {"": ("grey", "dgrey"), "_blanc": ("white", "lgrey")}        # hull, hull shade (black side: grey, to stay
                                                                     # visible on the black squares of space)
FLAME = ["white", "yellow", "orange", "red"]

# ─── the ships: left halves, 6 double-wide pixels (the centre on the right) ──
SHIPS = {
    "random": ("Capsule", "cyan", [
        ".....b",
        "....ba",
        "....bc",
        "...bac",
        "...baa",
        "...bad",
        "..bbad",
        "..b.be",
    ]),
    "beginner": ("Navette", "lblue", [
        ".....b",
        "....ba",
        "....ac",
        "....cc",
        "...baa",
        "..bbaa",
        ".bdaaa",
        ".bbdad",
        "...bbe",
    ]),
    "monte_carlo": ("Chasseur", "lred", [
        ".....b",
        ".....a",
        "....ba",
        "....bc",
        "....ac",
        "b...aa",
        "ab.baa",
        "aabdaa",
        "dabaad",
        "b.bbae",
        "....b.",
    ]),
    "mcts": ("Éclaireur", "green", [
        ".....g",
        ".....a",
        ".....a",
        "....ba",
        "....bc",
        "....ac",
        "....aa",
        "...baa",
        "b..baa",
        "ab.daa",
        "dabdaa",
        "b.bbde",
        "....b.",
    ]),
    "basic_minmax": ("Intercepteur", "yellow", [
        ".....b",
        ".....a",
        "....ba",
        "....ac",
        "....cc",
        "...baa",
        "...daa",
        "..bdaa",
        ".bdaaa",
        "bdaaaa",
        "baaaad",
        "bbbdae",
        "....b.",
    ]),
    "minmax": ("Chasseur lourd", "orange", [
        ".....b",
        ".....a",
        "....ba",
        "....ac",
        "g...cc",
        "a..baa",
        "a.bdaa",
        "abdaaa",
        "aaaaaa",
        "daacaa",
        "dabdad",
        "bebbbe",
        ".b..b.",
    ]),
    "features": ("Bombardier", "purple", [
        ".....b",
        "....ba",
        "....ac",
        "....cc",
        "....aa",
        "...baa",
        "..bdaa",
        "bbdaaa",
        "aaaaaa",
        "daaaaa",
        "bdcdcd",
        ".bbbdd",
        "..e.be",
        "..b.b.",
    ]),
    "mcts_nnue": ("Soucoupe", "lgreen", [
        "......",
        "....bb",
        "...bcc",
        "...ccc",
        "..bbbb",
        ".baaaa",
        "baalaa",
        "daaaal",
        ".dddda",
        "..bbbb",
        "...e.e",
    ]),
    "nnue193": ("Canonnière", "cyan", [
        "g....b",
        "a....a",
        "a...ba",
        "a...ac",
        "a...cc",
        "ab.baa",
        "aabdaa",
        "aaaaaa",
        "daaaaa",
        "dalaaa",
        "dabdaa",
        "bbbdad",
        "e..bbe",
        "b...b.",
    ]),
    "resnet": ("Frégate", "lblue", [
        ".....g",
        ".....a",
        ".....a",
        "....ba",
        "....ac",
        "....cc",
        "....aa",
        "....la",
        "....aa",
        "bb..aa",
        "aab.aa",
        "aaabaa",
        "daabaa",
        "dadbaa",
        "bbdbda",
        "e...be",
        "b...b.",
    ]),
    "dev": ("Croiseur", "red", [
        ".....b",
        ".....a",
        "....ba",
        "....ac",
        "....cc",
        "...gaa",
        "...baa",
        "..baaa",
        "..dala",
        "..daaa",
        "bbdaaa",
        "aaaaaa",
        "dalaaa",
        "daaaaa",
        "dddada",
        "bdbbdd",
        "e.e.be",
        "b.b.b.",
    ]),
    "alphazero": ("Vaisseau-mère", "yellow", [
        ".....b",
        "....ba",
        "....ac",
        "...bcc",
        "...aaa",
        "..baaa",
        "..dala",
        ".bdaaa",
        ".daaaa",
        "bdalaa",
        "aaaaaa",
        "dlaala",
        "aaaaaa",
        "daaaaa",
        "dcdcdc",
        "dddddd",
        "bbdddd",
        ".e.e.e",
        ".b.b.b",
    ]),
}

PX = 4                    # hires pixel size in the frame: a multicolour pixel is 2 PX wide, and the
                          # sprites are Y-expanded (as the C64 could): 2 PX high, square pixels of 8 × 8
CANVAS = 192
N_FRAMES = 8


def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def full(rows):
    """Left halves → full 12-pixel rows (mirrored)."""
    return [r + r[::-1] for r in rows]


def draw_frame(rows, accent, side, anim, k):
    hull, shade = SIDES[side]
    img = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    W = len(rows[0])
    H = len(rows)
    # placement: centred, the ship's bottom at 78 % of the frame (space for the shots above)
    x0 = CANVAS // 2 - W * PX          # each multicolour pixel is 2*PX wide
    y0 = int(CANVAS * 0.82) - H * 2 * PX
    dx = dy = 0
    if anim == "idle":
        dy = [0, 0, -1, -1, -1, -1, 0, 0][k] * PX
    elif anim == "blocked":
        dx = [1, -1, 1, -1, 0, 1, -1, 0][k] * PX
    elif anim == "move":
        dy = -PX if k in (2, 3, 4, 5) else 0
    elif anim == "win":
        dy = [0, -1, -1, 0, 0, -1, -1, 0][k] * PX

    def put(cx, cy, col):
        """One multicolour pixel (2 hires wide) at sprite coordinates."""
        X = x0 + dx + cx * 2 * PX
        Y = y0 + dy + cy * 2 * PX
        if 0 <= X < CANVAS - 2 * PX + 1 and 0 <= Y < CANVAS - 2 * PX + 1:
            img[Y:Y + 2 * PX, X:X + 2 * PX] = hexc(PAL[col]) if col in PAL else hexc(col)

    # the shadow on the ground: a dithered ellipse (checker of black pixels), under the ship
    sy = int(CANVAS * 0.91) // PX
    for hx in range(-W + 1, W):
        for hy in range(-1, 2):
            if (hx / (W * 0.8)) ** 2 + (hy / 1.3) ** 2 <= 1 and (hx + hy) % 2 == 0:
                X = CANVAS // 2 + hx * PX
                Y = sy * PX + hy * PX
                img[Y:Y + PX, X:X + PX] = (0, 0, 0, 150)
    flash = anim == "blocked" and k % 2 == 0
    engines, guns = [], []
    for cy, row in enumerate(rows):
        for cx, ch in enumerate(row):
            if ch == ".":
                continue
            col = {"a": hull, "d": shade, "b": "black", "c": accent, "e": "dgrey", "g": "lgrey"}.get(ch)
            if ch == "l":
                on = (k // 2 + cx + cy) % 2 == 0 if anim != "blocked" else False
                col = "white" if (on or anim == "win") else accent
            if flash and ch in "ad":
                col = "lred" if ch == "a" else "red"
            put(cx, cy, col)
            if ch == "e":
                engines.append((cx, cy))
            if ch == "g":
                guns.append((cx, cy))
    # flames under the nozzles
    if anim != "blocked":
        for (cx, cy) in engines:
            n = {"idle": [1, 2, 2, 1, 2, 1, 2, 2][(k + cx) % 8], "move": [3, 4, 4, 3, 4, 4, 3, 4][(k + cx) % 8], "win": 2}.get(anim, 2)
            for j in range(n):
                put(cx, cy + 1 + j, FLAME[min(len(FLAME) - 1, j + (k + cx) % 2)])
    # the shots: from the guns (or the nose), climbing, one volley every 4 frames
    if anim == "win":
        srcs = guns or [(len(rows[0]) // 2 - 1, 0), (len(rows[0]) // 2, 0)]
        for (cx, cy) in srcs:
            for v in range(2):
                h = (k + 4 * v) % 8          # whole cycles over the 8 frames
                yy = cy - 2 - h * 2
                for j in range(2):
                    put(cx, yy + j, "white" if j == 0 else "yellow")
    return img


def build(ai, preview=False):
    name, accent, half = SHIPS[ai]
    rows = full(half)
    anims = [("idle", 8, True), ("move", 16, False), ("blocked", 10, True), ("win", 10, True)]
    for side in ("", "_blanc"):
        sheet = np.zeros((CANVAS * len(anims), CANVAS * N_FRAMES, 4), np.uint8)
        for r, (an, fps, loop) in enumerate(anims):
            for k in range(N_FRAMES):
                sheet[r * CANVAS:(r + 1) * CANVAS, k * CANVAS:(k + 1) * CANVAS] = draw_frame(rows, accent, side, an, k)
        aid = f"fig_pix_{ai}{side}"
        if preview:
            return sheet
        Image.fromarray(sheet).save(os.path.join(ART, aid + ".png"), optimize=True)
        meta = {"name": name + (" (blanc)" if side else ""), "kind": "piece", "size": CANVAS, "smooth": True, "on stone": False,
                "box": 1.4, "anchor": [0.5, 0.7],
                "animations": [{"name": an, "fps": fps, "loop": loop, "frames": N_FRAMES} for an, fps, loop in anims]}
        if side:
            meta["variant of"] = f"fig_pix_{ai}"
        else:
            meta["white"] = f"fig_pix_{ai}_blanc"
        json.dump(meta, open(os.path.join(ART, aid + ".json"), "w"), indent=2, ensure_ascii=False)
    print(ai, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["preview"]:
        out = sys.argv[2]
        tiles = []
        for ai in SHIPS:
            name, accent, half = SHIPS[ai]
            rows = full(half)
            row = [draw_frame(rows, accent, s, an, k) for s, an, k in (("", "idle", 0), ("_blanc", "idle", 2), ("", "win", 3), ("_blanc", "blocked", 0))]
            tiles.append(np.concatenate(row, axis=1))
        sheet = np.concatenate(tiles, axis=0)
        bg = np.zeros_like(sheet)
        bg[..., :3] = (53, 40, 121)
        bg[..., 3] = 255
        a = sheet[..., 3:4] / 255.0
        comp = (sheet[..., :3] * a + bg[..., :3] * (1 - a)).astype(np.uint8)
        Image.fromarray(comp).resize((comp.shape[1] // 2, comp.shape[0] // 2), Image.NEAREST).save(out)
    else:
        for ai in SHIPS:
            build(ai)
