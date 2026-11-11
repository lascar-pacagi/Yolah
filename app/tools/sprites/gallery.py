"""gallery.py OUT [ANIM] [T] [SIDE] — every model side by side."""
import sys
from PIL import Image, ImageDraw
from render import render
from models import MODELS, _add_legends
_add_legends()
out = sys.argv[1]
anim = sys.argv[2] if len(sys.argv) > 2 else "idle"
t = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
side = int(sys.argv[4]) if len(sys.argv) > 4 else 0
names = [n for n in MODELS if len(sys.argv) < 6 or n.startswith(tuple(sys.argv[5].split(',')))]
S = 224
cols = 6
rows = (len(names) + cols - 1) // cols
sheet = Image.new("RGBA", (S * cols, S * rows), (46, 52, 62, 255))
d = ImageDraw.Draw(sheet)
for i, n in enumerate(names):
    try:
        im = render(MODELS[n](anim, t, side), size=S)
        sheet.alpha_composite(Image.fromarray(im), ((i % cols) * S, (i // cols) * S))
    except Exception as e:
        print(n, "FAILED", repr(e)[:300])
    d.text(((i % cols) * S + 6, (i // cols) * S + 4), n, fill=(230, 230, 230, 255))
sheet.save(out)
