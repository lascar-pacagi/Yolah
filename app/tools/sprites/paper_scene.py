"""
paper_scene.py — the Origami background in 3D (scenery.py): a paper diorama,
lit like a studio photograph.

  the box       a backdrop of cream paper, the sun a disc of red paper on a
                ring of orange, clouds cut from white paper standing forward
  the land      five layers of cut paper hills, pink to mauve to blue to
                navy, each standing a little in front of the last, slightly
                curled, their cut edges showing the white core of the paper
  the sea       a front layer of scalloped waves, two folded paper boats on
                it, paper cranes hanging on threads from above
  the light     a soft key light from the upper left (long soft shadows of
                each layer on the next), a warm fill, paper fibres and grain

    cd app/tools/sprites && python3 paper_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RX, RY, RZ, vec, norm, srgb, sd_sphere, sd_box, sd_capsule, sd_cylinder, fbm, hash3
from scenery import Camera, render_view

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
TH = 0.012          # the paper's thickness (the diorama is ~16 units wide)


def sstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def paper(c, fibre=0.06, core=True):
    def pat(pl, pw, n):
        f = fbm(pw * vec(60, 60, 60), 3)
        streak = fbm(pw * vec(8, 90, 8) + 3, 2)
        a = srgb(c) * (1 - fibre + fibre * 2 * (0.6 * f + 0.4 * streak))[:, None]
        if core:
            # the cut edge: the face turned sideways shows the paper's white core
            side = (1 - n[:, 2].abs()).clamp(0, 1)
            a = torch.lerp(a, srgb("#f4efe6"), (sstep(0.75, 0.95, side) * 0.7)[:, None])
        return {"albedo": a}
    return Mat(c, rough=0.92, sss=0.55, pattern=pat)


def layer(profile, z0, curl=0.05):
    """A sheet standing at depth z0, cut along y = profile(x) (paper below the line)."""
    def f(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        zz = z - z0 - curl * torch.sin(x * 0.35 + z0)
        prof = profile(x)
        # distance to the cut line, roughly (divided by the slope)
        e = 0.02
        slope = (profile(x + e) - prof) / e
        d2 = (y - prof) / torch.sqrt(1 + slope * slope)
        return torch.maximum(torch.maximum(d2, zz.abs() - TH), torch.maximum(x.abs() - 9.5, -y - 6))
    return f


def hills(seed, base, amp, freq):
    def prof(x):
        q = torch.stack([x * freq + seed, x * 0 + seed, x * 0], 1)
        return base + amp * (fbm(q, 4) - 0.5) * 2 + 0.25 * amp * torch.sin(x * freq * 2.3 + seed)
    return prof


def disc(cx, cy, r, z0):
    def f(p):
        d = torch.sqrt((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2) - r
        return torch.maximum(d, (p[:, 2] - z0).abs() - TH)
    return f


def cloud(cx, cy, s, z0):
    lobes = [(-1.0, 0.0, 0.55), (-0.35, 0.35, 0.75), (0.45, 0.25, 0.65), (1.05, 0.0, 0.45)]

    def f(p):
        x, y = (p[:, 0] - cx) / s, (p[:, 1] - cy) / s
        d = torch.full_like(x, 1e9)
        for (lx, ly, lr) in lobes:
            d = torch.minimum(d, torch.sqrt((x - lx) ** 2 + (y - ly) ** 2) - lr)
        d = torch.maximum(d, -(y + 0.25))            # a flat bottom
        return torch.maximum(d * s, (p[:, 2] - z0).abs() - TH)
    return f


def waves(z0):
    """The sea's front edge: a row of scallops (half discs)."""
    def prof(x):
        k = x / 0.7
        u = (k - torch.floor(k)) - 0.5
        return -3.6 + 0.32 * torch.sqrt((0.25 - u * u).clamp_min(0)) * 2
    return layer(prof, z0, curl=0.02)


def boat(cx, cy, z0, s):
    """A folded paper boat: a hull (trapezoid prism) and its sail (triangle)."""
    def f(p):
        x, y, z = (p[:, 0] - cx) / s, (p[:, 1] - cy) / s, (p[:, 2] - z0) / s
        hull = torch.maximum(torch.maximum(y - 0.0, -y - 0.35), x.abs() - (0.9 - (-y) * 0.9))
        hull = torch.maximum(hull, z.abs() - 0.25 + (-y) * 0.3)
        sail = torch.maximum(torch.maximum(-y, y - 0.9 + x.abs() * 1.2), z.abs() - 0.02)
        return torch.minimum(hull, sail) * s * 0.7
    return f


def crane(cx, cy, z0, s, ang):
    """A paper crane hanging on its thread: body, two wings, neck and tail as folded points."""
    def f(p):
        c, si = math.cos(ang), math.sin(ang)
        x0, y, z0_ = (p[:, 0] - cx) / s, (p[:, 1] - cy) / s, (p[:, 2] - z0) / s
        x = c * x0 - si * z0_
        z = si * x0 + c * z0_
        body = torch.maximum(torch.maximum(x.abs() * 0.6 + y.abs() - 0.3, z.abs() - 0.15 + y * 0.2), -y - 0.2)
        wing = torch.maximum(torch.maximum((z.abs() - 0.1) * 0.0 + x.abs() - 0.35 + (z.abs()) * 0.3, (y - 0.05 - z.abs() * 0.5).abs() - 0.015), z.abs() - 1.0)
        neck = torch.maximum(torch.maximum((y - (x - 0.2) * 1.4).abs() - 0.03, z.abs() - 0.03), torch.maximum(-(x - 0.2), x - 0.95))
        tail = torch.maximum(torch.maximum((y - (-x - 0.2) * 1.2).abs() - 0.03, z.abs() - 0.03), torch.maximum(-(-x - 0.2), -x - 0.85))
        head = torch.maximum(torch.maximum((y - 0.88 + (x - 0.95) * 0.6).abs() - 0.025, z.abs() - 0.025), torch.maximum(-(x - 0.92), x - 1.12))
        return torch.minimum(torch.minimum(torch.minimum(body, wing), torch.minimum(neck, tail)), head) * s * 0.6
    return f


def scene():
    sc = Scene()
    # the backdrop and the box's floor
    sc.add(sd_box(12, 8, 0.05), paper("#efe4d0", 0.05, core=False), T(0, 1.5, -6.2))
    sc.add(sd_box(12, 0.05, 4), paper("#d8cdb8", 0.05, core=False), T(0, -6.05, -2.5))
    for sgn in (-1, 1):   # the box's side walls
        sc.add(sd_box(0.05, 8, 4), paper("#c8bca6", 0.05, core=False), T(sgn * 9.6, 1.5, -2.5))
    # the sun: a disc on a ring, standing forward of the backdrop
    sc.add(disc(2.6, 2.4, 1.55, -5.85), paper("#f0a050"))
    sc.add(disc(2.6, 2.4, 1.25, -5.75), paper("#e04a30"))
    # clouds
    for (cx, cy, s_, z0) in ((-4.8, 3.6, 0.85, -5.4), (3.9, 3.2, 0.7, -5.2), (-1.0, 2.6, 0.55, -5.0), (6.4, 4.4, 0.5, -5.5)):
        sc.add(cloud(cx, cy, s_, z0), paper("#fbf8f2"))
    # the hills, back to front
    cols = ["#f0b8a8", "#e08e8c", "#a8789a", "#6a6aa0", "#3a4a82"]
    for k, c in enumerate(cols):
        z0 = -4.6 + k * 0.85
        prof = hills(1.7 * k + 3, 0.9 - k * 0.85, 1.25 - 0.12 * k, 0.22 + 0.07 * k)
        sc.add(layer(prof, z0), paper(c))
    # the sea and its waves, two boats
    sc.add(waves(0.2), paper("#24346a"))
    sc.add(layer(lambda x: -3.25 + 0.06 * torch.sin(x * 2.0), -0.35), paper("#2e4a8a"))
    sc.add(boat(-2.4, -3.2, -0.15, 0.55), paper("#f6f2ea"))
    sc.add(boat(3.2, -3.18, -0.15, 0.4), paper("#f2c050"))
    # cranes hanging on threads
    thread = Mat("#8a8078", rough=0.8)
    for (cx, cy, z0, s_, a) in ((-6.2, 2.0, -1.0, 0.55, 0.5), (5.6, 1.2, -0.6, 0.45, -0.7), (0.8, 3.4, -2.6, 0.38, 1.1)):
        sc.add(crane(cx, cy, z0, s_, a), paper("#e8483a" if a > 0 else "#f6f2ea"))
        sc.add(sd_capsule((cx, cy + 0.35 * s_, z0), (cx, 9.0, z0), 0.006), thread)
    return sc


def render(W=2560, H=1440, ss=1):
    cam = Camera((0.0, -0.2, 14.0), (0.0, -0.2, -3.0), fov=40)
    key = norm(vec(-0.55, 0.65, 0.52))
    sky = lambda d: srgb("#f2ece2").expand(d.shape[0], 3) * 0.9
    img = render_view(scene(), W, H, cam, sky, sun_dir=tuple(key.tolist()), sun_col="#fff2e0", sun_soft=0.12,
                      lights=[((7.0, 3.0, 8.0), "#ffd8b0", 9.0)], ambient=("#c8c4d8", "#8a8070"), exposure=1.05, steps=200,
                      tmax=40, ao_scale=0.08, bloom=0.3, dof=(13.0, 0.25), vignette=0.25, grain=0.01,
                      grade=("#3a4a7a", "#fff0dc"), ss=ss)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "origami.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
