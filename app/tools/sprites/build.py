"""
build.py [MODEL ...] — renders the figurines into assets/art.

For each model: assets/art/fig_<model>.json/.png (black side) and
fig_<model>_blanc.json/.png (white side, "variant of" the black one).
One row per animation (idle, move, blocked, win), one column per frame.
Without arguments: every model of models.MODELS.

    cd app/tools/sprites && python3 build.py sentinel human
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
_add_legends()

ART = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "art")
SIZE = 192
VIEW = 2.0                      # world units across the frame
CENTER = (0.0, 0.62, 0.0)       # world point at the frame's centre
BASE_DIAMETER = 1.12            # world units (models.base)
BASE_ON_SQUARE = 0.8            # the base's diameter in squares
ANIMS = [("idle", 16, 10, True), ("move", 8, 24, False), ("blocked", 12, 12, True), ("win", 16, 12, True)]
NAMES = {"sentinel": "Sentinelle", "human": "Humain", "tin_toy": "Jouet m\u00e9canique", "roller": "Roulix",
         "dicer": "T\u00eate de d\u00e9", "tree_bot": "Arborescent", "retro": "R\u00e9tro 1955", "crt": "Cathodique",
         "geometer": "G\u00e9om\u00e8tre", "android": "Androïde", "core": "Noyau"}


NAMES.update(legends.NAMES)
NAMES["human_legendes"] = "Humain (L\u00e9gendes)"


def anchor():
    """Where the centre of the base's top falls in the frame (fractions)."""
    el = math.radians(render.ELEVATION)
    v = (BASE_TOP - CENTER[1]) * math.cos(el)     # orthographic: height on screen
    return 0.5, 0.5 - v / VIEW


def build(model):
    t0 = time.time()
    box = BASE_ON_SQUARE * VIEW / BASE_DIAMETER
    ax, ay = anchor()
    cols = max(n for _, n, _, _ in ANIMS)
    for side, suffix in ((0, ""), (1, "_blanc")):
        sheet = np.zeros((SIZE * len(ANIMS), SIZE * cols, 4), np.uint8)
        for r, (anim, n, fps, loop) in enumerate(ANIMS):
            for k in range(n):
                t = k / n if loop else k / (n - 1)
                sheet[r * SIZE:(r + 1) * SIZE, k * SIZE:(k + 1) * SIZE] = draw(MODELS[model](anim, t, side), size=SIZE,
                                                                                 view=VIEW, center=CENTER)
        aid = f"fig_{model}{suffix}"
        Image.fromarray(sheet).save(os.path.join(ART, aid + ".png"), optimize=True)
        j = {"name": NAMES.get(model, model) + (" (blanc)" if side else ""), "kind": "piece", "size": SIZE,
             "smooth": True, "on stone": False, "box": round(box, 4), "anchor": [round(ax, 4), round(ay, 4)],
             "animations": [{"name": a, "fps": fps, "loop": loop, "frames": n} for a, n, fps, loop in ANIMS]}
        if side == 0:
            j["white"] = f"fig_{model}_blanc"
        else:
            j["variant of"] = f"fig_{model}"
        with open(os.path.join(ART, aid + ".json"), "w") as f:
            json.dump(j, f, indent=2, ensure_ascii=False)
    print(f"{model}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    for m in sys.argv[1:] or list(MODELS):
        build(m)
