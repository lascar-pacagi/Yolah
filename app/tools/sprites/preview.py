"""preview.py MODEL [ANIM] [OUT.png] — a contact sheet: frames of an animation at
full size, and the idle frame at the size of a square (on light and dark squares)."""
import sys
import time
import numpy as np
from PIL import Image
from render import render
from models import MODELS

FRAMES = {"idle": 16, "move": 8, "blocked": 12, "win": 16}


def over(bg, im, x, y):
    bg.alpha_composite(Image.fromarray(im), (x, y))


def main():
    name = sys.argv[1]
    anim = sys.argv[2] if len(sys.argv) > 2 else "idle"
    out = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/{name}_{anim}.png"
    nshow = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    size = int(sys.argv[5]) if len(sys.argv) > 5 else 192
    t0 = time.time()
    frames = [[render(MODELS[name](anim, k / nshow, side), size=size) for k in range(nshow)] for side in (0, 1)]
    dt = time.time() - t0
    W = size * nshow
    sheet = Image.new("RGBA", (W, size * 2 + 260), (40, 44, 52, 255))
    for side in (0, 1):
        for k, f in enumerate(frames[side]):
            over(sheet, f, k * size, side * size)
    # at board size: 96 px squares, the sprite box = 1.5 squares
    q = 96
    y0 = size * 2 + 20
    for i in range(6):
        col = (143, 184, 63, 255) if i % 2 == 0 else (138, 90, 50, 255)
        sheet.paste(Image.new("RGBA", (q, q), col), (40 + i * q, y0 + 80))
    for i, side in enumerate((0, 1, 0, 1, 0, 1)):
        im = Image.fromarray(frames[side][i % nshow]).resize((int(q * 1.5),) * 2, Image.LANCZOS)
        sheet.alpha_composite(im, (40 + i * q + q // 2 - int(q * 0.75), y0 + 80 + q // 2 - int(q * 1.5 * 0.66)))
    sheet.save(out)
    print(f"{out}: {2 * nshow} frames in {dt:.1f} s")


if __name__ == "__main__":
    main()
