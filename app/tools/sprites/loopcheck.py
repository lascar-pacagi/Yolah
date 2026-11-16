"""
loopcheck.py — every looping animation must join up: the frame at t=1 is the frame
at t=0 (a motion like `t * 0.6` radians or `TAU * t * 0.7` jumps back at the loop).

    cd app/tools/sprites && python3 loopcheck.py [prefix ...]   (all the used figures by default)

wrap: mean difference t=1 vs t=0 (0 when periodic); last->first: the step between the
last frame and the first, for scale. Run it after changing a figure's animation.
"""
import sys, subprocess, numpy as np
sys.path.insert(0, '.')
import build
tracked = subprocess.run(['git', 'ls-files', build.ART], capture_output=True, text=True).stdout.split()
models = sorted({p.rsplit('/', 1)[1][4:-4] for p in tracked if p.endswith('.png') and p.rsplit('/', 1)[1].startswith('fig_') and not p.endswith('_blanc.png')} & set(build.MODELS))
if len(sys.argv) > 1: models = [m for m in models if any(m.startswith(a) for a in sys.argv[1:])]
SZ = 96
def img(m, anim, t):
    return build.draw(build.MODELS[m](anim, t, 0), size=SZ, view=build.VIEW, center=build.CENTER, toon=m in build.TOON).astype(float)
for m in models:
    anims = build.ANIMS_FOR.get(m, build.ANIMS)
    for anim, n, fps, loop in anims:
        if not loop: continue
        a0, a1, a_last = img(m, anim, 0.0), img(m, anim, 1.0), img(m, anim, (n - 1) / n)
        wrap = np.abs(a1 - a0).mean()
        step = np.abs(a0 - a_last).mean() + 1e-6
        flag = "  <-- NOT PERIODIC" if wrap > 0.05 else ""
        print(f"{m:24s} {anim:8s} wrap {wrap:6.2f}  last->first {step:6.2f}{flag}", flush=True)
