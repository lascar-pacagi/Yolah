"""
build.py [MODEL ...] — renders the figurines into assets/art.

For each model: assets/art/fig_<model>.json/.png (black side) and
fig_<model>_blanc.json/.png (white side, "variant of" the black one).
One row per animation (idle, move, blocked, win), one column per frame.
Without arguments: every model of models.MODELS.

    cd app/tools/sprites && python3 build.py sentinel human
    python3 build.py ice_mcts:blanc      (one side only: noir or blanc)
"""
import json
import math
import os
import sys
import time
import numpy as np
from PIL import Image
import render
from render import render as draw
from models import MODELS, BASE_TOP, _add_legends
import legends
import manga
import generals
from models import scaled, SCALE
_add_legends()
import shonen
import epic
for _k, _f in shonen.SHONEN.items():         # the manga set: one character per AI (shonen.py)
    SCALE[_k] = shonen.SCALE[_k[6:]]
    MODELS[_k] = scaled(_k, _f)
for _k, _f in epic.EPIC.items():             # the Légendes set: one character per AI (epic.py)
    SCALE[_k] = epic.SCALE[_k[4:]]
    MODELS[_k] = scaled(_k, _f)
import arctic
import origami
import dinos
import anatomy
import bombs
for _k, _f in bombs.BOMBS.items():           # the Bombes set (bombs.py)
    SCALE[_k] = bombs.SCALE[_k[5:]]
    MODELS[_k] = scaled(_k, _f)
for _k, _f in anatomy.ANATOMY.items():       # the Laboratoire set (anatomy.py)
    SCALE[_k] = anatomy.SCALE[_k[4:]]
    MODELS[_k] = scaled(_k, _f)
for _k, _f in dinos.DINOS.items():           # the Dinosaures set (dinos.py)
    SCALE[_k] = dinos.SCALE[_k[5:]]
    MODELS[_k] = scaled(_k, _f)
for _k, _f in arctic.ARCTIC.items():         # the Banquise set (arctic.py)
    SCALE[_k] = arctic.SCALE[_k[4:]]
    MODELS[_k] = scaled(_k, _f)
for _k, _f in origami.ORIGAMI.items():       # the Origami set (origami.py)
    SCALE[_k] = origami.SCALE[_k[4:]]
    MODELS[_k] = scaled(_k, _f)
for _k, _f in generals.GENERALS.items():   # the Généraux: all the same size (sets.json "grow"), large
    SCALE[_k] = 1.42
    MODELS[_k] = scaled(_k, _f)
TOON = set()                     # the manga set is rendered like painted figures now (shonen.py, fourth version)

ART = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "art")
SIZE = 192
VIEW = 2.0                      # world units across the frame
CENTER = (0.0, 0.62, 0.0)       # world point at the frame's centre
BASE_DIAMETER = 1.12            # world units (models.base)
BASE_ON_SQUARE = 0.8            # the base's diameter in squares
ANIMS = [("idle", 16, 10, True), ("move", 8, 24, False), ("blocked", 12, 12, True), ("win", 16, 12, True)]
NAMES = {"sentinel": "Sentinelle", "human": "Humain", "tin_toy": "Jouet m\u00e9canique", "roller": "Roulix",
         "dicer": "T\u00eate de d\u00e9", "tree_bot": "Arborescent", "retro": "R\u00e9tro 1955", "crt": "Cathodique",
         "geometer": "G\u00e9om\u00e8tre", "android": "Androïde", "core": "Noyau",
         "chrono": "Chrono", "sylve": "Sylve"}


NAMES.update(legends.NAMES)
NAMES.update(shonen.NAMES)
NAMES.update(epic.NAMES)
NAMES.update(generals.NAMES)
NAMES.update(arctic.NAMES)
NAMES.update(origami.NAMES)
NAMES.update(dinos.NAMES)
NAMES.update(anatomy.NAMES)
NAMES.update(bombs.NAMES)
NAMES["human_legendes"] = "Humain (L\u00e9gendes)"


def anchor():
    """Where the centre of the base's top falls in the frame (fractions)."""
    el = math.radians(render.ELEVATION)
    v = (BASE_TOP - CENTER[1]) * math.cos(el)     # orthographic: height on screen
    return 0.5, 0.5 - v / VIEW


def build(model, sides=(0, 1)):
    t0 = time.time()
    box = BASE_ON_SQUARE * VIEW / BASE_DIAMETER
    ax, ay = anchor()
    cols = max(n for _, n, _, _ in ANIMS)
    for side, suffix in ((0, ""), (1, "_blanc")):
        if side not in sides:
            continue
        sheet = np.zeros((SIZE * len(ANIMS), SIZE * cols, 4), np.uint8)
        for r, (anim, n, fps, loop) in enumerate(ANIMS):
            for k in range(n):
                t = k / n if loop else k / (n - 1)
                sheet[r * SIZE:(r + 1) * SIZE, k * SIZE:(k + 1) * SIZE] = draw(MODELS[model](anim, t, side), size=SIZE,
                                                                                 view=VIEW, center=CENTER, toon=model in TOON)
        aid = f"fig_{model}{suffix}"
        Image.fromarray(sheet).save(os.path.join(ART, aid + ".png"), optimize=True)
        j = {"name": NAMES.get(model, model) + (" (blanc)" if side else ""), "kind": "piece", "size": SIZE,
             "smooth": True, "on stone": False, "box": round(box, 4), "anchor": [round(ax, 4), round(ay, 4)],
             "animations": [{"name": a, "fps": fps, "loop": loop, "frames": n} for a, n, fps, loop in ANIMS]}
        if model in generals.PLATES:
            j["plate"] = generals.PLATES[model]
        if model in generals.STARS:
            j["stars"] = generals.STARS[model]
        if side == 0:
            j["white"] = f"fig_{model}_blanc"
        else:
            j["variant of"] = f"fig_{model}"
        with open(os.path.join(ART, aid + ".json"), "w") as f:
            json.dump(j, f, indent=2, ensure_ascii=False)
    print(f"{model}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    for m in sys.argv[1:] or list(MODELS):          # "model:blanc" / "model:noir" = one side only
        m, _, only = m.partition(":")
        build(m, {"": (0, 1), "noir": (0,), "blanc": (1,)}[only])
