"""
pixel_board.py — the "Vaisseaux 8 bits" board (assets/themes/17_vaisseaux.json),
in the pixel art of the 1980s home computers (C64 palette, chunky pixels).

  assets/art/vaisseaux_clair.*    light square: the C64's blue, a few faint stars
  assets/art/vaisseaux_sombre.*   dark square: black space, dim stars
  assets/art/vaisseaux_trou.*     hole: a rift in space, its rim of pixels
                                  flickering purple and blue (idle); a pixel
                                  explosion, a ring of debris going out (open)
  assets/art/vaisseaux_cadre.*    the frame: the machine's light blue border
  assets/backgrounds/vaisseaux.png  deep space at 320 × 180 scaled × 8 without
                                  smoothing: a starfield, a ringed planet
                                  dithered, a moon, a far nebula, and the
                                  bands of colour of a loading screen over a
                                  skyline of mountains at the bottom

    cd app/tools/sprites && python3 pixel_board.py [tiles hole frame background theme]
"""
import json
import math
import os
import random
import numpy as np
from PIL import Image
from pixel_ships import PAL, hexc

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "..", "assets")
ART = os.path.join(ASSETS, "art")
TAU = 2 * math.pi
B = 8            # pixel block in the 128 px tiles: a 16 × 16 grid


def tile(base, stars, seed):
    g = np.zeros((16, 16, 4), np.uint8)
    g[...] = hexc(PAL[base])
    rnd = random.Random(seed)
    for col, n in stars:
        for _ in range(n):
            g[rnd.randrange(1, 15), rnd.randrange(1, 15)] = hexc(PAL[col])
    # a one-pixel bevel: the C64 had none, but it marks the squares
    return np.kron(g, np.ones((B, B, 1), np.uint8))


def write(aid, name, kind, anims, size):
    rows = len(anims)
    cols = max(len(f) for _, f, _, _ in anims)
    sheet = np.zeros((rows * size, cols * size, 4), np.uint8)
    for r, (an, frames, fps, loop) in enumerate(anims):
        for c, f in enumerate(frames):
            sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = f
    Image.fromarray(sheet).save(os.path.join(ART, aid + ".png"), optimize=True)
    json.dump({"name": name, "kind": kind, "size": size, "smooth": True, "on stone": False,
               "animations": [{"name": an, "fps": fps, "loop": loop, "frames": len(fr)} for an, fr, fps, loop in anims]},
              open(os.path.join(ART, aid + ".json"), "w"), indent=2, ensure_ascii=False)


def rift(frames=8, opening=False):
    out = []
    n = 9 if opening else frames
    rnd = random.Random(4)
    debris = [(rnd.uniform(0, TAU), rnd.uniform(0.6, 1.0), rnd.choice(["yellow", "orange", "white", "lred"])) for _ in range(22)]
    for f in range(n):
        g = np.zeros((16, 16, 4), np.uint8)
        for y in range(16):
            for x in range(16):
                u, v = (x - 7.5) / 7.5, (y - 7.5) / 7.5
                r = max(abs(u), abs(v)) * 0.6 + math.hypot(u, v) * 0.4
                if opening:
                    k = (f + 1) / n
                    R = 1.05 * k
                    if r < R * 0.85:
                        g[y, x] = hexc(PAL["black"])
                    elif r < R:
                        g[y, x] = hexc(PAL[["white", "yellow", "orange", "red", "brown"][min(4, f // 2)]])
                    continue
                if r < 0.86:
                    g[y, x] = hexc(PAL["black"])
                    # a few stars deep in the rift, twinkling
                    if (x * 7 + y * 13) % 23 == 0 and (f + x) % 4 < 2:
                        g[y, x] = hexc(PAL["lgrey"])
                elif r < 1.02:
                    ph = (math.atan2(v, u) / TAU * 8 + f) % 8      # whole turns over the frames
                    g[y, x] = hexc(PAL["lblue" if int(ph) % 2 == 0 else "purple"])
        if opening:   # the debris going out
            for (a, sp, col) in debris:
                d = (f + 1) / n * sp * 9
                x, y = int(round(7.5 + d * math.cos(a))), int(round(7.5 + d * math.sin(a)))
                if 0 <= x < 16 and 0 <= y < 16 and f < n - 1:
                    g[y, x] = hexc(PAL[col])
        out.append(np.kron(g, np.ones((B, B, 1), np.uint8)))
    return out


def frame_art():
    """The light blue border, a lighter line outside and a darker one inside (chunky pixels)."""
    S, b = 96, 8                                   # 12 × 12 blocks
    g = np.zeros((12, 12, 4), np.uint8)
    for y in range(12):
        for x in range(12):
            e = min(x, y, 11 - x, 11 - y)
            if e < 4:
                col = "lblue"
                if e == 0:
                    col = "lgrey" if (x < y) == (x + y < 11) or x + y < 11 else "blue"
                    col = "lgrey" if (y == 0 or x == 0) else "blue"
                if e == 3:
                    col = "blue"
                g[y, x] = hexc(PAL[col])
    return np.kron(g, np.ones((b, b, 1), np.uint8))


def background():
    W, H = 320, 180
    img = np.zeros((H, W, 3), np.uint8)
    rnd = random.Random(8)

    def put(x, y, col):
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = hexc(PAL[col])[:3]
    # stars of three colours
    for _ in range(420):
        put(rnd.randrange(W), rnd.randrange(H - 40), rnd.choice(["dgrey", "dgrey", "grey", "lgrey", "white", "lblue"]))
    # a far nebula: dithered purple and blue blobs
    for y in range(H):
        for x in range(W):
            n = math.sin(x * 0.03 + 1) * math.cos(y * 0.05) + math.sin((x + y) * 0.02)
            d = math.hypot((x - 90) / 70, (y - 60) / 35)
            if d < 1 and n > 0.2 and (x + y) % 2 == 0:
                img[y, x] = hexc(PAL["purple" if d > 0.55 else "blue"])[:3]
    # a ringed planet, dithered: lit from the upper left
    cx, cy, R = 238, 66, 40
    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy < R * R:
                l = (-dx * 0.6 - dy * 0.8) / R            # light
                band = int((dy + 0.15 * dx) / 7) % 3
                base = ["orange", "brown", "orange"][band]
                if l > 0.45:
                    col = "yellow" if (x + y) % 2 == 0 or l > 0.7 else base
                elif l > -0.2:
                    col = base
                elif l > -0.55:
                    col = base if (x + y) % 2 == 0 else "black"
                else:
                    col = "black"
                put(x, y, col)
    # its ring: an ellipse in front (lower half) and behind (hidden by the planet)
    for a in range(720):
        t = a / 720 * TAU
        for w in (0, 1, 2):
            rx, ry = R * 1.75 + w * 2, R * 0.42 + w * 0.8
            x, y = int(cx + rx * math.cos(t)), int(cy + ry * math.sin(t) - (x_ := 0))
            front = math.sin(t) > 0
            inside = (x - cx) ** 2 + (y - cy) ** 2 < R * R
            if front or not inside:
                put(x, y, ["lgrey", "grey", "dgrey"][w])
    # a moon
    for y in range(H):
        for x in range(W):
            dx, dy = x - 160, y - 30
            if dx * dx + dy * dy < 81:
                put(x, y, "lgrey" if dx + dy < 0 else ("grey" if (x + y) % 2 == 0 else "dgrey"))
    # the bands of a loading screen over the horizon
    bands = ["blue", "purple", "lblue", "red", "lred", "orange", "yellow"]
    for i, col in enumerate(bands):
        y0 = H - 40 + i * 3
        img[y0:y0 + 3, :] = hexc(PAL[col])[:3]
    # mountains in silhouette in front of the bands
    for x in range(W):
        h = 18 + 9 * math.sin(x * 0.05) + 6 * math.sin(x * 0.13 + 1) + 3 * math.sin(x * 0.31)
        img[int(H - h):, x] = hexc(PAL["black"])[:3]
    big = np.kron(img, np.ones((8, 8, 1), np.uint8))
    Image.fromarray(big).save(os.path.join(ASSETS, "backgrounds", "vaisseaux.png"), optimize=True)


def theme():
    t = {
        "id": "vaisseaux", "name": "Vaisseaux 8 bits", "pattern": "flat", "pattern strength": 0.0, "seed": 64,
        "light": "#352879", "dark": "#000000",
        "frame top": "#6c5eb5", "frame bottom": "#352879", "frame line": "#959595",
        "coordinates": "#6c5eb5", "hole": "#000000", "hole rim": "#6f3d86",
        "background top": "#000000", "background bottom": "#352879",
        "last move": "#b8c76f66", "selection": "#70a4b28c", "legal move": "#000000aa", "hint": "#9ad284dc",
        "accent": "#70a4b2",
        "tile light": "vaisseaux_clair", "tile dark": "vaisseaux_sombre", "hole art": "vaisseaux_trou",
        "frame art": "vaisseaux_cadre", "background image": "vaisseaux.png", "piece set": "vaisseaux", "break fx": "pixel",
    }
    with open(os.path.join(ASSETS, "themes", "17_vaisseaux.json"), "w") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["tiles", "hole", "frame", "background", "theme"]
    if "tiles" in what:
        write("vaisseaux_clair", "Vaisseaux : espace bleu", "tile", [("idle", [tile("blue", [("lblue", 3), ("grey", 2)], 1)], 1, True)], 128)
        write("vaisseaux_sombre", "Vaisseaux : espace noir", "tile", [("idle", [tile("black", [("dgrey", 4), ("grey", 1)], 2)], 1, True)], 128)
    if "hole" in what:
        write("vaisseaux_trou", "Vaisseaux : faille", "hole", [("idle", rift(), 6, True), ("open", rift(opening=True), 18, False)], 128)
    if "frame" in what:
        write("vaisseaux_cadre", "Vaisseaux : bordure", "frame", [("idle", [frame_art()], 1, True)], 96)
    if "background" in what:
        background()
    if "theme" in what:
        theme()
    print("done:", ", ".join(what))
