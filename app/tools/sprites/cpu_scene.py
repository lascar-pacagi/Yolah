"""
cpu_scene.py — the Informatique's background in 3D (scenery.py): the long hall
of a computer museum at night, history laid out along its aisle.

  the hall      a polished dark floor of large stone tiles, walls of dark wood
                panels, a ceiling with a track of spotlights; each exhibit
                stands on its plinth in a pool of warm light, a small glowing
                label at its foot
  the exhibits  the machines of the set at full size (computers.py, the same
                models): the abacus and the Pascaline in vitrines, the
                difference engine, the Jacquard loom, the rotor machine, the
                vacuum-tube panels glowing, the tape drives, the C-shaped
                supercomputer, an 8-bit computer on a desk
  the end       a glass wall opening on a working computing centre, its aisle
                of racks blinking blue; in front of it, hanging over the
                aisle as the climax of the visit, the golden chandelier of a
                quantum computer, its chip glowing
  the air       a faint haze catching the spotlights' cones

    cd app/tools/sprites && python3 cpu_scene.py [W H out [ss]]
"""
import math
import os
import sys
import torch
from PIL import Image
from render import Scene, Mat, T, RY, vec, srgb, sd_box, sd_cylinder, fbm
from render import Part
from scenery import Camera, render_view, Volume
import computers
import retro
from models import BASE_PARTS

ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


def place(sc, model, x, z, a, k, y=0.0, anim="idle", t=0.0, side=1):
    """Puts a figurine model (computers.py) into the hall: without its base,
    turned by a around y, scaled by k, its foot at (x, y, z)."""
    m = model(anim, t, side)
    W = T(x, y, z) * RY(a)
    for part in m.parts[BASE_PARTS:]:
        p2 = Part(part.sdf, part.mat, W * part.xf, part.blend, part.op)
        p2.scale = (k, vec(x, y, z))
        sc.parts.append(p2)


def stone_floor():
    def pat(pl, pw, n):
        x, z = pl[:, 0], pl[:, 2]
        gx = torch.remainder(x, 2.0)
        gz = torch.remainder(z, 2.0)
        grout = ((gx < 0.02) | (gz < 0.02)).float()
        tile_id = torch.floor(x / 2.0) * 7.1 + torch.floor(z / 2.0) * 3.3
        tone = 0.75 + 0.25 * torch.frac(torch.sin(tile_id) * 4375.5)
        vein = fbm(pl * vec(0.6, 0.6, 0.6) + vec(3, 0, 1), 4)
        c = srgb("#2a2a2e") * (tone * (0.8 + 0.4 * vein))[:, None] * (1 - 0.6 * grout)[:, None]
        return {"albedo": c, "rough": 0.12 + 0.2 * grout}
    return Mat("#2a2a2e", metal=0.0, rough=0.15, clearcoat=1.0, pattern=pat)


def wood_panels():
    def pat(pl, pw, n):
        u = pl[:, 0] + pl[:, 2]
        panel = (torch.remainder(u, 1.6) < 0.03).float()
        grain = 0.5 + 0.5 * torch.sin(pl[:, 1] * 40 + 6 * fbm(pl * vec(0.5, 3, 0.5), 3))
        return {"albedo": srgb("#3a2216") * (0.75 + 0.3 * grain)[:, None] * (1 - 0.6 * panel)[:, None]}
    return Mat("#3a2216", rough=0.75, clearcoat=0.0, pattern=pat)


def scene():
    sc = Scene()
    sc.add(lambda p: p[:, 1], stone_floor())                                      # the floor (y = 0)
    walls = wood_panels()
    sc.add(lambda p: 10.0 - p[:, 0].abs(), walls)                                 # the side walls (x = ±10)
    sc.add(lambda p: 8.0 - p[:, 1], Mat("#141416", rough=0.8))                     # the ceiling
    # the spotlight tracks along the ceiling
    for x in (-5.0, 5.0):
        sc.add(sd_box(0.06, 0.05, 30.0), Mat("#1a1a1c", metal=0.6, rough=0.4), T(x, 7.7, -20.0))
    plinth = Mat("#24242a", rough=0.35, clearcoat=0.6)
    label = Mat("#f2ead8", emissive="#fff2d8", emit=0.5)
    vitrine = Mat("#d8e8f0", rough=0.02, glass=True)
    # exhibits: (model, side x, z, turn, scale, plinth height, vitrine)
    exhibits = [
        (computers.abacus, -4.6, -15.5, 0.5, 3.2, 0.8, True),
        (computers.difference_engine, -5.4, -21.0, 0.6, 6.0, 0.3, False),
        (computers.jacquard, -5.6, -27.5, 0.6, 6.0, 0.25, False),
        (computers.tube_computer, -5.8, -34.0, 0.7, 7.0, 0.15, False),
        (computers.pascaline, 4.6, -15.5, -0.5, 3.2, 0.8, True),
        (computers.rotor_machine, 5.2, -20.5, -0.6, 3.8, 0.8, True),
        (computers.tape_mainframe, 5.6, -26.5, -0.6, 6.5, 0.15, False),
        (computers.supercomputer, 5.8, -33.5, -0.4, 6.5, 0.15, False),
        (computers.micro8, -3.0, -39.5, 0.3, 3.4, 0.7, False),
    ]
    for (model, x, z, a, k, h, glass) in exhibits:
        sc.add(sd_box(2.0, h / 2, 1.7, 0.03), plinth, T(x, h / 2, z))
        sc.add(sd_box(0.3, 0.06, 0.01, 0.01), label, T(x + (1.3 if x < 0 else -1.3), h * 0.55, z + 1.71))
        place(sc, model, x, z, a, k, y=h)
        if glass:
            sc.add(sd_box(1.6, 1.0, 1.4, 0.02), vitrine, T(x, h + 1.0, z))
    # the end of the hall: a glass wall, the computing centre beyond it
    sc.add(sd_box(10.0, 4.0, 0.05), Mat("#a8c8e0", rough=0.02, glass=True), T(0, 4.0, -44.0))
    sc.add(sd_box(10.0, 0.15, 0.2), Mat("#1a1a1c", metal=0.6, rough=0.4), T(0, 0.15, -44.0))
    for z in (-47.0, -52.5):
        for x in (-6.5, -2.2, 2.2, 6.5):
            place(sc, computers.datacenter, x, z, 0.0, 6.5, anim="idle", t=(x * 0.13 + z * 0.07) % 1)
    sc.add(lambda p: p[:, 2] + 59.0, Mat("#0a1018", rough=0.6))                    # the centre's back wall (z = -59)
    # the climax: the quantum chandelier hanging over the aisle, in front of the glass wall
    place(sc, computers.quantum, 0.0, -37.5, 0.3, 9.0, y=0.6, anim="idle", t=0.25)
    sc.add(sd_cylinder(0.3, 3.8, 0.05), Mat("#1a1a1c", metal=0.5, rough=0.3), T(0, 0.3, -37.5))
    # the first room: home computers on the left, game consoles on the right, on long display tables
    table = Mat("#2a1e16", rough=0.35, clearcoat=0.6)
    # (shown at 1.8 times their size, as exhibition replicas: at real size they were lost in the hall)
    left = [retro.apple2, retro.ibm_pc, retro.c64, retro.zx, retro.mo5, retro.mac]
    right = [retro.atari_st, retro.amiga, retro.atari2600, retro.nes, retro.megadrive, retro.playstation, retro.gameboy]
    H, K = 0.85, 1.8
    for sx, items, z0, step in ((-1, left, -0.8, 1.55), (1, right, -0.8, 1.35)):
        x = sx * 4.0
        L = len(items) * step / 2 + 0.4
        zc = z0 - L + 0.6
        sc.add(sd_box(0.75, H / 2, L, 0.02), table, T(x, H / 2, zc))
        sc.add(sd_box(0.76, 0.01, L + 0.01, 0.005), Mat("#c8a050", metal=0.9, rough=0.3), T(x, H, zc))
        for i, f in enumerate(items):
            z = z0 - i * step
            # turned towards the aisle and a little towards the visitors coming in
            tmp = Scene()
            f(tmp, T(0, 0, 0))
            Wm = T(x, H + 0.012, z) * RY(-sx * 0.95)
            for part in tmp.parts:
                p2 = Part(part.sdf, part.mat, Wm * part.xf, part.blend, part.op)
                p2.scale = (K, vec(x, H + 0.012, z))
                sc.parts.append(p2)
    return sc


def haze(p):
    y = p[:, 1]
    n = fbm(p * vec(0.12, 0.3, 0.12), 3)
    return (0.5 + 0.8 * n) * 0.012 * torch.exp(-y / 9.0)


def render(W=2560, H=1440, ss=1, shift=0.36):
    # the vanishing point a little right of the middle (shift): in the app, the board covers the
    # left two thirds and the side panel the right edge; the hall's end falls in the strip between
    cam = Camera((0.0, 3.0, 3.5), (0.0, 1.6, -30.0), fov=56, shift=(shift, 0.0))
    sky = lambda d: srgb("#05070c").expand(d.shape[0], 3) * 1.0
    lights = []
    for (x, z) in ((-4.6, -15.5), (-5.4, -21.0), (-5.6, -27.5), (-5.8, -34.0), (4.6, -15.5), (5.2, -20.5), (5.6, -26.5), (5.8, -33.5), (-3.0, -39.5)):
        lights.append(((x * 0.75, 7.2, z + 1.8), "#ffd8a8", 4.0))
    lights += [((0, 7.5, -33.0), "#ffe8c0", 9.0), ((0, 1.5, -36.0), "#5ab8ff", 6.0)]
    lights += [((x, 6.5, z), "#4a8aff", 10.0) for z in (-46.0, -51.0) for x in (-4.4, 4.4)]
    lights += [((sx * 3.4, 4.0, z), "#ffe2c0", 2.5) for sx in (-1, 1) for z in (-1.0, -4.0, -7.0, -10.0)]       # the first room's tables       # the first room's tables
    vols = [Volume(haze, (-10, 0, -44), (10, 8, 3), albedo="#d8d0c8", g=0.5, steps=40, light_steps=0, ambient=0.5)]
    img = render_view(scene(), W, H, cam, sky, sun_dir=None, lights=lights, ambient=("#1a1e28", "#0c0a0a"),
                      exposure=1.0, steps=260, tmax=80, ao_scale=0.3, bloom=1.0, dof=(12.0, 0.25), vignette=0.42,
                      grain=0.01, grade=("#2a3a5a", "#ffe2c0"), ss=ss, volumes=vols, refl_occ=1.0)
    return img


if __name__ == "__main__":
    W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2560, 1440)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ASSETS, "backgrounds", "informatique.jpg")
    ss = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    Image.fromarray(render(W, H, ss)).save(out, quality=94)
    print("saved", out)
