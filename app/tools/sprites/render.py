"""
render.py — a small offline renderer for the app's sprites (assets/art).

Characters are built from signed distance functions (spheres, rounded boxes,
capsules, ...), each part with its own transform and material, and rendered
on the GPU (PyTorch) by sphere tracing with an orthographic camera looking
down at the board at ELEVATION degrees, like a miniature on a game board.

Shading, in linear light: a warm key light (top left, soft shadows), a cool
fill, a strong rim light from behind, ambient occlusion, GGX specular, a
studio environment for the reflections of metals, emissive parts with bloom,
a shadow-catcher ground plane (the sprite carries its own contact shadow),
filmic tone mapping. Rendered SS times larger, then averaged down
(anti-aliasing), with straight alpha for SDL.

A model is a function frame(t) → Scene; t in [0, 1) along the animation.
"""
import ctypes
import glob
import math
import os
import torch


def _blocking_sync():
    """The CPU sleeps while it waits for the GPU (cudaDeviceScheduleBlockingSync)
    instead of spinning on a core. Must run before CUDA is initialised."""
    libs = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cuda_runtime", "lib", "libcudart.so*"))
    for name in libs + ["libcudart.so.12", "libcudart.so"]:
        try:
            ctypes.CDLL(name).cudaSetDeviceFlags(0x04)
            return
        except OSError:
            continue


if torch.cuda.is_available() and not torch.cuda.is_initialized():
    _blocking_sync()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
F = torch.float32
GRAPHS = True        # replay Scene.dist from CUDA graphs (see Scene.dist)


# ─── vectors ────────────────────────────────────────────────────────────────
_LAST = None         # the event after the last graph replayed (Scene.dist)
_VEC_MEMO = None     # set while a CUDA graph is warmed up then captured (Scene.dist)


def vec(*a):
    if _VEC_MEMO is not None:          # the same constants in the warm-up run and the capture: no copy while capturing
        v = _VEC_MEMO.get(a)
        if v is None:
            v = _VEC_MEMO[a] = torch.tensor(a, dtype=F, device=DEV)
        return v
    return torch.tensor(a, dtype=F, device=DEV)


def norm(v):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)


def length(p):
    return torch.linalg.norm(p, dim=-1)


def srgb(h):
    """'#rrggbb' → linear rgb tensor."""
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    c = torch.tensor([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], dtype=F, device=DEV)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=F, device=DEV)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=F, device=DEV)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=F, device=DEV)


I3 = None


class X:
    """A rigid transform (rotation R, then translation t): world = R · local + t."""

    def __init__(self, R=None, t=(0, 0, 0)):
        self.R = R if R is not None else torch.eye(3, dtype=F, device=DEV)
        self.t = t if torch.is_tensor(t) else vec(*t)

    def __mul__(self, o):          # self ∘ o
        return X(self.R @ o.R, self.R @ o.t + self.t)

    def to_local(self, p):         # (N, 3) world → local
        return (p - self.t) @ self.R

    def point(self, *q):           # a local point → world
        return self.R @ vec(*q) + self.t


def T(x=0, y=0, z=0):
    return X(t=(x, y, z))


def RX(a): return X(rot_x(a))
def RY(a): return X(rot_y(a))
def RZ(a): return X(rot_z(a))


# ─── distance functions (local coordinates, p: (N, 3)) ──────────────────────
def sd_sphere(r):
    return lambda p: length(p) - r


def sd_ellipsoid(rx, ry, rz):
    r = vec(rx, ry, rz)

    def f(p):
        k0 = length(p / r)
        k1 = length(p / (r * r))
        return k0 * (k0 - 1) / k1.clamp_min(1e-6)
    return f


def sd_box(bx, by, bz, rr=0.0):
    b = vec(bx - rr, by - rr, bz - rr)

    def f(p):
        q = p.abs() - b
        return length(q.clamp_min(0)) + q.max(dim=-1).values.clamp_max(0) - rr
    return f


def sd_capsule(a, b, r, r2=None):
    """Segment a → b, radius r (r2 at b: a cone with round ends)."""
    a, b = vec(*a), vec(*b)
    r2 = r if r2 is None else r2

    def f(p):
        pa, ba = p - a, b - a
        h = ((pa @ ba) / (ba @ ba)).clamp(0, 1)
        return length(pa - h[:, None] * ba) - (r + (r2 - r) * h)
    return f


def sd_cylinder(h, r, rr=0.0):
    """Vertical (y), half height h, radius r, edges rounded by rr."""
    def f(p):
        d = torch.stack([torch.hypot(p[:, 0], p[:, 2]) - r + rr, p[:, 1].abs() - h + rr], dim=-1)
        return length(d.clamp_min(0)) + d.max(dim=-1).values.clamp_max(0) - rr
    return f


def sd_torus(R, r):
    """In the xz plane."""
    def f(p):
        q = torch.stack([torch.hypot(p[:, 0], p[:, 2]) - R, p[:, 1]], dim=-1)
        return length(q) - r
    return f


def sd_cone(h, r1, r2, rr=0.0):
    """Vertical frustum, y from −h (radius r1) to h (radius r2)."""
    def f(p):
        y = p[:, 1].clamp(-h, h)
        r = r1 + (r2 - r1) * (y + h) / (2 * h)
        d = torch.stack([torch.hypot(p[:, 0], p[:, 2]) - r + rr, p[:, 1].abs() - h + rr], dim=-1)
        return (length(d.clamp_min(0)) + d.max(dim=-1).values.clamp_max(0)) * 0.9 - rr
    return f


def sd_plane_cut(f, n, d):
    """f cut by the half space n · p > d (keeps n · p ≤ d)."""
    n = norm(vec(*n))

    def g(p):
        return torch.maximum(f(p), p @ n - d)
    return g


def displace(f, g, k):
    return lambda p: f(p) + k * g(p)


def smin(a, b, k):
    if k <= 0:
        return torch.minimum(a, b)
    h = (0.5 + 0.5 * (b - a) / k).clamp(0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


# ─── noise (for surfaces) ───────────────────────────────────────────────────
def hash3(p):
    p = p * 0.3183099 + 0.1
    p = (p - torch.floor(p)) * 17.0
    h = p[..., 0] * p[..., 1] * p[..., 2] * (p[..., 0] + p[..., 1] + p[..., 2])
    return h - torch.floor(h)


_CORNERS = {}


def vnoise(p):
    i = torch.floor(p)
    f = p - i
    u = f * f * (3 - 2 * f)
    n = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1]) * (u[..., 2] if dz else 1 - u[..., 2])
                c = _CORNERS.get((dx, dy, dz))      # made once: no copy to the GPU per call (graph capture)
                if c is None:
                    c = _CORNERS[(dx, dy, dz)] = vec(dx, dy, dz)
                n = n + w * hash3(i + c)
    return n


_CORNER_LIST = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


def _fbm_fused(p, octaves: int):
    """fbm with the corners as Python constants: torch.compile fuses it into one kernel."""
    s, a, tot = 0, 0.5, 0
    for _ in range(octaves):
        i = torch.floor(p)
        f = p - i
        u = f * f * (3 - 2 * f)
        n = 0
        for dx, dy, dz in _CORNER_LIST:
            w = (u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1]) * (u[..., 2] if dz else 1 - u[..., 2])
            q = (i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz)
            q = [(c * 0.3183099 + 0.1) for c in q]
            q = [(c - torch.floor(c)) * 17.0 for c in q]
            h = q[0] * q[1] * q[2] * (q[0] + q[1] + q[2])
            n = n + w * (h - torch.floor(h))
        s = s + a * n
        tot += a
        p = p * 2.03 + 1.7
        a *= 0.5
    return s / tot


FUSED_NOISE = DEV == "cuda" and os.environ.get("RENDER_NO_COMPILE") is None
_fbm_compiled = torch.compile(_fbm_fused, dynamic=True) if FUSED_NOISE else None


def fbm(p, octaves=4):
    if _fbm_compiled is not None and p.is_cuda and p.shape[0] >= 64:
        return _fbm_compiled(p, octaves)
    s, a, tot = 0, 0.5, 0
    for _ in range(octaves):
        s = s + a * vnoise(p)
        tot += a
        p = p * 2.03 + 1.7
        a *= 0.5
    return s / tot


# ─── materials and scenes ───────────────────────────────────────────────────
class Mat:
    """albedo (sRGB hex), metal 0..1, rough 0..1, emissive (hex) × emit,
    pattern(pl, pw, n) → (albedo, rough, emissive) changes: pl local point,
    pw world point, n world normal, all (M, 3); returns a dict of overrides."""

    def __init__(self, albedo="#888888", metal=0.0, rough=0.5, emissive=None, emit=0.0, pattern=None,
                 sss=0.0, clearcoat=0.0, glass=False):
        self.albedo = srgb(albedo)
        self.metal, self.rough = metal, rough
        self.emissive = srgb(emissive) * emit if emissive else None
        self.pattern = pattern
        self.sss = sss               # skin: light wraps around, reddish
        self.clearcoat = clearcoat   # a sharp extra highlight (paint, eyes)
        self.glass = glass           # transparent: reflections and a tint over what is behind


class Part:
    def __init__(self, sdf, mat, xf=None, blend=0.0, op="add"):
        self.sdf, self.mat, self.xf = sdf, mat, xf or X()
        self.blend, self.op = blend, op
        self.scale = None         # (s, centre): the part scaled by s about centre

    def local(self, p):
        if self.scale is not None:
            s, c = self.scale
            p = c + (p - c) / s
        return self.xf.to_local(p)

    def d(self, p):
        v = self.sdf(self.local(p))
        return v * self.scale[0] if self.scale is not None else v


class Scene:
    def __init__(self):
        self.parts = []

    def add(self, sdf, mat, xf=None, blend=0.0, op="add"):
        self.parts.append(Part(sdf, mat, xf, blend, op))
        return self

    def scale_from(self, first, s, centre=(0, 0, 0)):
        """Scales the parts from index `first` on by s about centre (the figure, not its base)."""
        c = vec(*centre)
        for part in self.parts[first:]:
            part.scale = (s, c)
        return self

    def has_glass(self):
        return any(part.mat.glass for part in self.parts)

    def dist(self, p, with_id=False, glass=False):
        """Distance to the opaque parts (glass=False) or to the glass ones.
        On CUDA, the plain distance is replayed from a CUDA graph (one graph
        per input size bucket): one launch instead of thousands, so the
        CPU stays nearly idle and the GPU does the work."""
        if with_id or not GRAPHS or p.device.type != "cuda" or p.shape[0] < 1024:
            return self._dist(p, with_id, glass)
        m = p.shape[0]
        cap = 1024
        while cap < m:                 # buckets 1024 · 4^k: few captures per frame (each costs one eager pass)
            cap *= 4
        key = (cap, glass)
        g = self.__dict__.setdefault("_graphs", {}).get(key)
        if g is None:
            global _VEC_MEMO
            memo = _VEC_MEMO = self.__dict__.setdefault("_memo", {})
            try:
                buf = torch.zeros(cap, 3, dtype=F, device=DEV)
                if glass not in self.__dict__.setdefault("_warm", set()):    # one warm-up per scene: fills the memo
                    self._warm.add(glass)
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        self._dist(buf, False, glass)
                    torch.cuda.current_stream().wait_stream(s)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    out = self._dist(buf, False, glass)
                g = (gr, buf, out, memo)       # the memo (the scene's) keeps the captured constants alive
            except Exception:
                g = False
            finally:
                _VEC_MEMO = None
            self._graphs[key] = g
        if g is False:
            return self._dist(p, with_id, glass)
        gr, buf, out, _ = g
        global _LAST
        if _LAST is not None:          # at most two graphs queued: the CPU sleeps here (blocking sync)
            _LAST.synchronize()        # instead of spinning in a full launch queue
        buf[:m] = p
        gr.replay()
        r = out[:m].clone()
        _LAST = torch.cuda.Event(blocking=True)
        _LAST.record()
        return r

    def _dist(self, p, with_id=False, glass=False):
        d = None
        best = None
        ids = None
        for i, part in enumerate(self.parts):
            if part.mat.glass != glass:
                continue
            di = part.d(p)
            if part.op == "sub":
                if d is not None:
                    d = smax(d, -di, part.blend)
                    if with_id:          # what is cut away no longer claims the surface
                        best = torch.maximum(best, -di)
                continue
            if part.op == "int":
                if d is not None:
                    d = smax(d, di, part.blend)
                continue
            if d is None:
                d, best = di, di
                ids = torch.zeros_like(di, dtype=torch.long)
            else:
                if with_id:
                    closer = di < best
                    ids = torch.where(closer, torch.full_like(ids, i), ids)
                    best = torch.minimum(best, di)
                d = smin(d, di, part.blend)
        if d is None:
            d = torch.full((p.shape[0],), 1e3, device=p.device)
            ids = torch.zeros_like(d, dtype=torch.long)
        return (d, ids) if with_id else d


# ─── camera and lights ──────────────────────────────────────────────────────
ELEVATION = 32.0     # degrees above the horizon
KEY = norm(vec(-0.42, 1.0, 0.5))
KEY_COL = srgb("#fff1dc") * 3.4
FILL = norm(vec(0.75, 0.25, 0.6))
FILL_COL = srgb("#9fc2ff") * 0.9
RIM = norm(vec(0.35, 0.45, -0.85))
RIM_COL = srgb("#cfe4ff") * 3.2


def env(r):
    """Studio environment seen along r: a bright cool top, a warm horizon,
    a dark floor, two soft boxes."""
    y = r[:, 1:2]
    top = srgb("#dfe9ff") * 1.6
    hor = srgb("#8a7a6e") * 1.0
    flo = srgb("#3a3632") * 0.7
    c = torch.where(y > 0, hor + (top - hor) * y.clamp(0, 1) ** 0.6, hor + (flo - hor) * (-y).clamp(0, 1) ** 0.4)
    for d, k in ((norm(vec(-0.6, 0.6, 0.5)), 6.0), (norm(vec(0.7, 0.4, -0.5)), 4.0)):
        c = c + k * ((r @ d).clamp_min(0) ** 40)[:, None]
    return c


def ambient(n):
    y = n[:, 1:2]
    return srgb("#5a6478") * (0.55 + 0.45 * y) + srgb("#3a2e28") * 0.25 * (1 - y.abs())


def aces_curve(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0, 1)


def aces(x):
    """Filmic curve; bright saturated colours keep their hue (a mix of the
    per-channel curve and the curve applied to the largest channel)."""
    per = aces_curve(x)
    m = x.max(dim=0, keepdim=True).values.clamp_min(1e-6)
    hue = x * (aces_curve(m) / m)
    return 0.4 * per + 0.6 * hue


def to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def gauss_blur(img, sigma):
    """img (C, H, W)."""
    k = int(sigma * 3) * 2 + 1
    x = torch.arange(k, device=DEV, dtype=F) - k // 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    g = g / g.sum()
    c = img.shape[0]
    im = img[:, None]
    im = torch.nn.functional.conv2d(torch.nn.functional.pad(im, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    im = torch.nn.functional.conv2d(torch.nn.functional.pad(im, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))
    return im[:, 0]


def render(scene, size=192, ss=2, view=2.0, center=(0.0, 0.62, 0.0), ground=True, bloom=1.0, shadow_strength=0.55,
           exposure=1.25, steps=160, toon=False, ink="#1b1420", line=1.0):
    """RGBA uint8 (size, size, 4) numpy array. view: world units across the frame;
    center: the world point at the frame's centre.

    toon: the manga look: flat shading in bands (light, half tone, shadow), a
    hard highlight, a rim band, and ink lines (colour `ink`, about `line`
    final pixels wide) on the silhouette, on jumps in depth, on sharp creases
    and where the flat colour changes (use flat materials)."""
    n = size * ss
    el = math.radians(ELEVATION)
    fwd = vec(0, -math.sin(el), -math.cos(el))      # looking down, towards −z
    right = vec(1, 0, 0)
    up = torch.linalg.cross(right, fwd)
    up = -up if up[1] < 0 else up
    js, is_ = torch.meshgrid(torch.arange(n, device=DEV, dtype=F), torch.arange(n, device=DEV, dtype=F), indexing="ij")
    u = ((is_ + 0.5) / n - 0.5) * view
    v = (0.5 - (js + 0.5) / n) * view
    c = vec(*center)
    org = (c + u[..., None] * right + v[..., None] * up - fwd * 6.0).reshape(-1, 3)
    d = fwd.expand_as(org)
    N = org.shape[0]
    t = torch.zeros(N, device=DEV, dtype=F)
    hit = torch.zeros(N, dtype=torch.bool, device=DEV)
    alive = torch.ones(N, dtype=torch.bool, device=DEV)
    eps = 0.6 * view / n
    tmax = 14.0
    for _ in range(steps):
        idx = alive.nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        p = org[idx] + d[idx] * t[idx, None]
        dist = scene.dist(p)
        t[idx] += dist * 0.85
        h = dist < eps
        hit[idx[h]] = True
        out = t[idx] > tmax
        alive[idx[h | out]] = False
    color = torch.zeros(N, 3, device=DEV)
    emis = torch.zeros(N, 3, device=DEV)
    albbuf = torch.zeros(N, 3, device=DEV)
    gbuf = torch.full((N,), -1, dtype=torch.long, device=DEV)     # toon: ink group (the material) of each pixel
    nbuf = torch.zeros(N, 3, device=DEV)
    alpha = hit.float()
    hi = hit.nonzero().squeeze(1)
    if hi.numel():
        p = org[hi] + d[hi] * t[hi, None]
        _, ids = scene.dist(p, with_id=True)
        e = 0.0015
        nrm = torch.zeros_like(p)
        for k in range(3):
            o = torch.zeros(3, device=DEV)
            o[k] = e
            nrm[:, k] = scene.dist(p + o) - scene.dist(p - o)
        nrm = norm(torch.nan_to_num(nrm, nan=0.0) + vec(0, 1e-6, 0))
        V = -d[hi]
        # per-part materials and patterns
        alb = torch.zeros_like(p); met = torch.zeros(p.shape[0], device=DEV); rou = torch.zeros_like(met)
        emi = torch.zeros_like(p); sss = torch.zeros_like(met); cc = torch.zeros_like(met)
        for i, part in enumerate(scene.parts):
            m = ids == i
            if not m.any():
                continue
            mt = part.mat
            a, r, em = mt.albedo.expand(int(m.sum()), 3), torch.full((int(m.sum()),), mt.rough, device=DEV), None
            if mt.emissive is not None:
                em = mt.emissive.expand(int(m.sum()), 3)
            if mt.pattern is not None:
                o = mt.pattern(part.local(p[m]), p[m], nrm[m])
                a = o.get("albedo", a)
                r = o.get("rough", r)
                if "emissive" in o:
                    em = o["emissive"] if em is None else em + o["emissive"]
            alb[m] = a
            rou[m] = r
            met[m] = mt.metal
            sss[m] = mt.sss
            cc[m] = mt.clearcoat
            if em is not None:
                emi[m] = em
        # occlusion
        ao = torch.ones_like(met)
        for k in range(1, 6):
            h = 0.03 * k
            ao = ao - (h - scene.dist(p + nrm * h)).clamp_min(0) * (0.5 ** k) * 9.0
        ao = ao.clamp(0.15, 1)
        # soft shadow from the key light
        sh = torch.ones_like(met)
        ts = torch.full_like(met, 0.02)
        for _ in range(48):
            dd = scene.dist(p + nrm * 0.004 + KEY * ts[:, None])
            sh = torch.minimum(sh, 10.0 * dd / ts)
            ts = ts + dd.clamp(0.01, 0.15)
        sh = sh.clamp(0, 1)
        sh = sh * sh * (3 - 2 * sh)
        F0 = 0.04 * (1 - met[:, None]) + alb * met[:, None]
        diff_col = alb * (1 - met[:, None])
        col = torch.zeros_like(p)
        if toon:
            albbuf[hi] = alb
            nbuf[hi] = nrm
            keys, gid = {}, torch.zeros(len(scene.parts), dtype=torch.long, device=DEV)
            for i, part in enumerate(scene.parts):      # same look = same group: no ink line between them
                mt = part.mat
                k = (tuple(round(float(v), 3) for v in mt.albedo), round(mt.rough, 2), round(mt.metal, 2), mt.pattern is not None and id(mt))
                gid[i] = keys.setdefault(k, len(keys))
            gbuf[hi] = gid[ids]
        if toon:
            ndv_ = (nrm * V).sum(-1).clamp(0, 1)
            lit = (nrm @ KEY).clamp_min(0) * sh
            band = torch.where(lit > 0.45, torch.full_like(lit, 1.0), torch.where(lit > 0.1, torch.full_like(lit, 0.66), torch.full_like(lit, 0.42)))
            rimb = ((1 - ndv_) > 0.7).float() * ((nrm @ RIM) > -0.3).float() * 0.3
            Hk = norm(KEY + V)
            hl = (((nrm * Hk).sum(-1).clamp_min(0)) > (0.9 + 0.09 * (1 - rou))).float() * (1 - rou) * 1.2
            Rt = norm(-V - 2 * ((-V * nrm).sum(-1, keepdim=True)) * nrm)
            metal_band = torch.where(Rt[:, 1] > 0.25, torch.full_like(lit, 1.25), torch.full_like(lit, 0.45))
            base_c = alb * (1 - met[:, None]) * (band + rimb)[:, None] * 1.9 + alb * met[:, None] * metal_band[:, None] * 1.5
            col = base_c + hl[:, None] + emi
            color[hi] = torch.nan_to_num(col, nan=0.0)
            emis[hi] = emi
        else:
            for L, LC, s_ in ((KEY, KEY_COL, sh), (FILL, FILL_COL, None), (RIM, RIM_COL, None)):
                ndl = (nrm @ L)
                wrap = (ndl + sss) / (1 + sss)
                dif = wrap.clamp_min(0)
                if s_ is not None:
                    dif = dif * s_
                Hh = norm(L + V)
                ndh = (nrm * Hh).sum(-1).clamp_min(0)
                a2 = (rou ** 2).clamp_min(0.002) ** 2
                D = a2 / (math.pi * (ndh * ndh * (a2 - 1) + 1) ** 2)
                ndv = (nrm * V).sum(-1).clamp_min(1e-3)
                fres = F0 + (1 - F0) * ((1 - (Hh * V).sum(-1).clamp(0, 1)) ** 5)[:, None]
                spec = (D * ndl.clamp_min(0) / (4 * ndv.clamp_min(0.2)))[:, None] * fres
                if s_ is not None:
                    spec = spec * s_[:, None]
                sss_tint = 1 + sss[:, None] * (srgb("#ff6a4a") - 1) * (1 - ndl.clamp(0, 1))[:, None] * 0.6
                col = col + LC * (diff_col * dif[:, None] * sss_tint / math.pi * 3.2 + spec)
                if s_ is not None:   # clear coat: a tight extra highlight
                    ccD = (ndh ** 900) * 18
                    col = col + LC * (cc * ccD * s_)[:, None] * 0.25
            ndv = (nrm * V).sum(-1).clamp(0, 1)
            R = norm(-V - 2 * (( -V * nrm).sum(-1, keepdim=True)) * nrm)
            fr = F0 + (1 - F0) * ((1 - ndv) ** 5)[:, None]
            envc = env(R)
            rough_k = (1 - rou)[:, None] ** 2
            col = col + envc * fr * (0.25 + 0.75 * rough_k) * ao[:, None] * (0.6 + 0.4 * sh[:, None])
            col = col + diff_col * ambient(nrm) * ao[:, None] * 0.9
            col = col + emi
            color[hi] = torch.nan_to_num(col, nan=0.0)
            emis[hi] = emi
    # glass: reflections and a tint over the opaque result (or over nothing)
    if scene.has_glass():
        tg = torch.zeros(N, device=DEV, dtype=F)
        hg = torch.zeros(N, dtype=torch.bool, device=DEV)
        al = torch.ones(N, dtype=torch.bool, device=DEV)
        for _ in range(steps // 2):
            idx = al.nonzero().squeeze(1)
            if idx.numel() == 0:
                break
            pp = org[idx] + d[idx] * tg[idx, None]
            dist = scene.dist(pp, glass=True)
            tg[idx] += dist * 0.9
            h = dist < eps
            hg[idx[h]] = True
            al[idx[h | (tg[idx] > tmax)]] = False
        front = hg & (~hit | (tg < t))
        gi = front.nonzero().squeeze(1)
        if gi.numel():
            pg = org[gi] + d[gi] * tg[gi, None]
            _, gid = scene.dist(pg, with_id=True, glass=True)
            e = 0.0015
            ng = torch.zeros_like(pg)
            for k in range(3):
                o = torch.zeros(3, device=DEV)
                o[k] = e
                ng[:, k] = scene.dist(pg + o, glass=True) - scene.dist(pg - o, glass=True)
            ng = norm(torch.nan_to_num(ng) + vec(0, 1e-6, 0))
            Vg = -d[gi]
            ndv = (ng * Vg).sum(-1).clamp(0, 1)
            fr = 0.05 + 0.95 * (1 - ndv) ** 4
            Rg = norm(-Vg - 2 * ((-Vg * ng).sum(-1, keepdim=True)) * ng)
            tint = torch.zeros_like(pg)
            for i, part in enumerate(scene.parts):
                m = gid == i
                if m.any() and part.mat.glass:
                    tint[m] = part.mat.albedo
            Hk = norm(KEY + Vg)
            spec = ((ng * Hk).sum(-1).clamp_min(0) ** 300) * 6.0
            refl = env(Rg) * fr[:, None] * 1.1 + KEY_COL * spec[:, None]
            behind = color[gi] * tint * (1 - fr[:, None])
            color[gi] = behind + refl
            emis[gi] = emis[gi] * (1 - fr[:, None])
            a_old = alpha[gi]
            alpha[gi] = torch.maximum(a_old, (0.18 + 0.8 * fr).clamp(0, 1))
            hit = hit | front
    # ink lines (toon): compare each pixel with its right and lower neighbours
    if toon:
        Hm = hit.reshape(n, n)
        Dm = torch.where(Hm, t.reshape(n, n), torch.full((n, n), 99.0, device=DEV))
        Gm = gbuf.reshape(n, n)
        Nm = nbuf.reshape(n, n, 3)
        edge = torch.zeros(n, n, dtype=torch.bool, device=DEV)
        for dy, dx in ((0, 1), (1, 0)):
            a_h, b_h = Hm[:n - dy, :n - dx], Hm[dy:, dx:]
            sil = a_h != b_h
            both = a_h & b_h
            dz = (Dm[:n - dy, :n - dx] - Dm[dy:, dx:]).abs() > 0.035
            crease = (Nm[:n - dy, :n - dx] * Nm[dy:, dx:]).sum(-1) < 0.55
            dcol = Gm[:n - dy, :n - dx] != Gm[dy:, dx:]
            e = sil | (both & (dz | crease | dcol))
            edge[:n - dy, :n - dx] |= e
            edge[dy:, dx:] |= e
        k = max(1, int(round(line * ss)))
        if k > 1:
            edge = torch.nn.functional.max_pool2d(edge[None, None].float(), k * 2 - 1, 1, k - 1)[0, 0] > 0
        glowing = (emis.reshape(n, n, 3).sum(-1) > 0.05)
        glowing = torch.nn.functional.max_pool2d(glowing[None, None].float(), 3, 1, 1)[0, 0] > 0
        edge = edge & ~glowing                 # no ink around what glows (auras, flames, lights)
        ef = edge.reshape(-1)
        color[ef] = srgb(ink)
        emis[ef] = 0
        alpha[ef] = 1.0
        hit = hit | ef
    # ground: the contact shadow (black with alpha)
    if ground:
        mi = (~hit).nonzero().squeeze(1)
        if mi.numel():
            o, dd = org[mi], d[mi]
            tg = -o[:, 1] / dd[:, 1]
            pg = o + dd * tg[:, None]
            near = length(pg[:, [0, 2]]) < 1.6
            sel = mi[near]
            pg = pg[near]
            if sel.numel():
                occ = torch.zeros(sel.shape[0], device=DEV)
                for k in range(1, 7):
                    h = 0.05 * k
                    occ = occ + (h - scene.dist(pg + vec(0, h, 0))).clamp_min(0) * (0.6 ** k) * 6.0
                sh = torch.ones(sel.shape[0], device=DEV)
                ts = torch.full_like(sh, 0.02)
                for _ in range(40):
                    q = scene.dist(pg + KEY * ts[:, None])
                    sh = torch.minimum(sh, 6.0 * q / ts)
                    ts = ts + q.clamp(0.01, 0.2)
                sh = sh.clamp(0, 1)
                a = ((1 - sh) * 0.75 + occ.clamp(0, 1)).clamp(0, 1) * shadow_strength
                alpha[sel] = a
                color[sel] = 0
    color = color.reshape(n, n, 3).permute(2, 0, 1)
    emis = emis.reshape(n, n, 3).permute(2, 0, 1)
    alpha = alpha.reshape(1, n, n)
    # tone map, premultiply, average down
    col = aces(color * exposure)
    col = to_srgb(col)
    pre = torch.cat([col * alpha, alpha], 0)[None]
    pre = torch.nn.functional.avg_pool2d(pre, ss)[0]
    if bloom > 0:
        em = torch.nn.functional.avg_pool2d(emis[None], ss)[0]
        glow = gauss_blur(em, size / 40) * 0.9 + gauss_blur(em, size / 14) * 0.6
        glow = to_srgb(aces(glow * bloom)).clamp(0, 1)
        ga = glow.max(0, keepdim=True).values
        rgb = pre[:3] + glow * (1 - pre[3:4]) + glow * 0.12 * pre[3:4]
        a = (pre[3:4] + ga * (1 - pre[3:4])).clamp(0, 1)
        pre = torch.cat([rgb.clamp(0, 1), a], 0)
    a = pre[3:4]
    rgb = torch.where(a > 1e-4, pre[:3] / a.clamp_min(1e-4), torch.zeros_like(pre[:3])).clamp(0, 1)
    out = torch.cat([rgb, a], 0).permute(1, 2, 0)
    return (out * 255 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
