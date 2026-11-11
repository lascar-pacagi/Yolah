"""closeup.py MODEL ANIM T OUT [view cx cy cz] — one large frame, for details."""
import sys
from PIL import Image
from render import render
from models import MODELS
m, anim, t, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
view = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
c = tuple(float(x) for x in sys.argv[6:9]) if len(sys.argv) > 8 else (0, 0.85, 0)
im = render(MODELS[m](anim, t, 0), size=384, view=view, center=c, ground=False)
bg = Image.new("RGBA", (384, 384), (60, 66, 78, 255)); bg.alpha_composite(Image.fromarray(im)); bg.save(out)
