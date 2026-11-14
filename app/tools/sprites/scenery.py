"""
scenery.py — a perspective renderer for the backgrounds of the boards: the
same signed distance scenes as the figurines (render.Scene, Mat, the sd_*
shapes), seen by a camera with a field of view, lit like a film set:

  sun          a directional light with soft shadows (penumbra by `sun_soft`)
  sky          a function of the direction: the colour of what the rays miss,
               and the light reflected by glossy surfaces
  lights       point lights (position, colour, radius): soft shadows, falloff
  emissive     materials that glow (screens, lamps, lava, the reactor's pool)
  ambient      the sky's light from above, darkened by occlusion
  fog          height fog (density, colour, falloff): depth and atmosphere
  shafts       light shafts: the fog lit where the sun reaches, at a quarter
               of the resolution, blurred and added
  post         bloom, filmic tone curve (ACES), depth of field (the far
               scenery softer, as through a lens focused on the board),
               vignette, a fine film grain, a colour grade

render_view(...) returns an (H, W, 3) uint8 array. Rays are traced in tiles
so that a 2560 × 1440 image fits in memory.
"""
import math
import numpy as np
import torch
from render import DEV, F, vec, norm, length, srgb, aces, to_srgb, gauss_blur, fbm


class Camera:
    def __init__(self, pos, look, fov=50.0, up=(0, 1, 0)):
        self.pos = vec(*pos)
        f = norm(vec(*look) - self.pos)
        r = norm(torch.linalg.cross(f, vec(*up)))
        u = torch.linalg.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.tan = math.tan(math.radians(fov) / 2)

    def rays(self, W, H, js, is_):
        a = W / H
        x = ((is_ + 0.5) / W * 2 - 1) * self.tan * a
        y = (1 - (js + 0.5) / H * 2) * self.tan
        d = norm(self.f + x[:, None] * self.r + y[:, None] * self.u)
        return self.pos.expand_as(d), d


def gradient_sky(zenith, horizon, ground=None, sun_dir=None, sun_col=None, sun_size=0.9995, haze=None):
    """A sky: horizon → zenith, a sun disc with its halo; below the horizon `ground`."""
    z, h = srgb(zenith), srgb(horizon)
    g = srgb(ground) if ground else h * 0.4
    hz = srgb(haze) if haze else h

    def sky(d):
        y = d[:, 1:2]
        up = y.clamp(0, 1) ** 0.55
        c = h * (1 - up) + z * up
        c = torch.where(y < 0, hz * (1 - (-y * 4).clamp(0, 1)) + g * (-y * 4).clamp(0, 1), c)
        if sun_dir is not None:
            sd = norm(vec(*sun_dir))
            cs = (d @ sd).clamp(-1, 1)[:, None]
            disc = (cs > sun_size).float()
            halo = cs.clamp(0, 1) ** 64 * 0.6 + cs.clamp(0, 1) ** 8 * 0.15
            c = c + srgb(sun_col or "#fff2d8") * (disc * 8.0 + halo)
        return c
    return sky


def _march(scene, o, d, tmax, steps, pix, glass=False):
    N = o.shape[0]
    t = torch.full((N,), 0.01, device=DEV, dtype=F)
    hit = torch.zeros(N, dtype=torch.bool, device=DEV)
    alive = torch.ones(N, dtype=torch.bool, device=DEV)
    for _ in range(steps):
        idx = alive.nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        p = o[idx] + d[idx] * t[idx, None]
        dist = scene.dist(p, glass=glass) if glass else scene.dist(p)
        t[idx] += dist * 0.9
        h = dist < pix * t[idx].clamp_min(0.05)
        hit[idx[h]] = True
        alive[idx[h | (t[idx] > tmax)]] = False
    return t, hit


def _soft_shadow(scene, p, nrm, L, maxd, soft, steps=40):
    scene = getattr(scene, "shadow_scene", None) or scene      # thin wires etc. can be left out of the shadows
    sh = torch.ones(p.shape[0], device=DEV)
    ts = torch.full_like(sh, 0.02)
    act = torch.ones_like(sh, dtype=torch.bool)
    for _ in range(steps):
        dd = scene.dist(p + nrm * 0.01 + L * ts[:, None])
        sh = torch.where(act, torch.minimum(sh, dd / (soft * ts)), sh)
        ts = ts + dd.clamp(0.02, 0.6)
        act = act & (ts < maxd) & (sh > 0.001)
        if not act.any():
            break
    sh = sh.clamp(0, 1)
    return sh * sh * (3 - 2 * sh)


def _materials(scene, p, ids, nrm):
    n = p.shape[0]
    alb = torch.zeros_like(p); met = torch.zeros(n, device=DEV); rou = torch.zeros_like(met)
    emi = torch.zeros_like(p); sss = torch.zeros_like(met); cc = torch.zeros_like(met)
    for i, part in enumerate(scene.parts):
        m = ids == i
        if not m.any():
            continue
        mt = part.mat
        k = int(m.sum())
        a, r, em = mt.albedo.expand(k, 3), torch.full((k,), mt.rough, device=DEV), None
        if mt.emissive is not None:
            em = mt.emissive.expand(k, 3)
        if mt.pattern is not None:
            o = mt.pattern(part.local(p[m]), p[m], nrm[m])
            a = o.get("albedo", a)
            r = o.get("rough", r)
            if "emissive" in o:
                em = o["emissive"] if em is None else em + o["emissive"]
        alb[m] = a; rou[m] = r; met[m] = mt.metal; sss[m] = mt.sss; cc[m] = mt.clearcoat
        if em is not None:
            emi[m] = em
    return alb, met, rou, emi, sss, cc


def _shade(scene, o, d, t, hit, sky, sun_dir, sun_col, sun_soft, lights, amb, ao_scale, tmax):
    N = o.shape[0]
    col = sky(d)
    emis = torch.zeros(N, 3, device=DEV)
    hi = hit.nonzero().squeeze(1)
    if hi.numel() == 0:
        return col, emis
    p = o[hi] + d[hi] * t[hi, None]
    _, ids = scene.dist(p, with_id=True)
    e = 0.0008 + 0.0008 * t[hi]
    nrm = torch.zeros_like(p)
    for k in range(3):
        oo = torch.zeros(3, device=DEV)
        oo[k] = 1.0
        nrm[:, k] = scene.dist(p + oo * e[:, None]) - scene.dist(p - oo * e[:, None])
    nrm = norm(torch.nan_to_num(nrm) + vec(0, 1e-6, 0))
    V = -d[hi]
    alb, met, rou, emi, sss, cc = _materials(scene, p, ids, nrm)
    ao = torch.ones_like(met)
    for k in range(1, 6):
        h = ao_scale * k
        ao = ao - (h - scene.dist(p + nrm * h)).clamp_min(0) * (0.5 ** k) * (1.6 / ao_scale)
    ao = ao.clamp(0.1, 1)
    F0 = 0.04 * (1 - met[:, None]) + alb * met[:, None]
    dif_c = alb * (1 - met[:, None])
    out = torch.zeros_like(p)

    def light(L, LC, sh, att):
        ndl = (nrm * L).sum(-1)
        wrap = ((ndl + sss) / (1 + sss)).clamp_min(0)
        Hh = norm(L + V)
        ndh = (nrm * Hh).sum(-1).clamp_min(0)
        a2 = (rou ** 2).clamp_min(0.002) ** 2
        D = a2 / (math.pi * (ndh * ndh * (a2 - 1) + 1) ** 2)
        ndv = (nrm * V).sum(-1).clamp_min(1e-3)
        fres = F0 + (1 - F0) * ((1 - (Hh * V).sum(-1).clamp(0, 1)) ** 5)[:, None]
        spec = (D * ndl.clamp_min(0) / (4 * ndv.clamp_min(0.2)))[:, None] * fres
        k = (sh * att)[:, None]
        return LC * k * (dif_c * wrap[:, None] / math.pi * 3.0 + spec + (cc * ndh ** 600 * 4)[:, None])
    if sun_dir is not None:
        L = norm(vec(*sun_dir))
        sh = _soft_shadow(scene, p, nrm, L, tmax, sun_soft)
        out = out + light(L, srgb(sun_col) * 3.0 if isinstance(sun_col, str) else sun_col, sh, torch.ones_like(sh))
    for lp, lc, lr in lights:
        lv = vec(*lp) - p
        dist = length(lv).clamp_min(1e-3)
        L = lv / dist[:, None]
        att = 1.0 / (1.0 + (dist / lr) ** 2)
        sh = _soft_shadow(scene, p, nrm, L, dist * 0.95, 0.08, steps=24) if lr > 0 else torch.ones_like(dist)
        out = out + light(L, srgb(lc) * 3.0 if isinstance(lc, str) else lc, sh, att)
    R = norm(-V - 2 * ((-V * nrm).sum(-1, keepdim=True)) * nrm)
    ndv = (nrm * V).sum(-1).clamp(0, 1)
    fr = F0 + (1 - F0) * ((1 - ndv) ** 5)[:, None]
    out = out + sky(R) * fr * ((1 - rou)[:, None] ** 2 * 0.8 + 0.1) * ao[:, None]
    up = (nrm[:, 1:2] * 0.5 + 0.5)
    out = out + dif_c * (amb[0] * up + amb[1] * (1 - up)) * ao[:, None]
    out = out + emi
    col[hi] = torch.nan_to_num(out)
    emis[hi] = emi
    return col, emis


def render_view(scene, W, H, cam, sky, sun_dir=None, sun_col=None, sun_soft=0.06, lights=(), ambient=None,
                fog=None, shafts=0.0, exposure=1.0, steps=220, tmax=80.0, ao_scale=0.08, bloom=0.8, dof=None, vignette=0.35,
                grain=0.02, grade=None, tile=1 << 18, ss=1, atmo=None, aerial=None, volumes=(), vol_scale=0.5):
    """fog: (density, colour, height falloff, base height). dof: (focus distance, strength).
    grade: (shadows tint, highlights tint) as hex. Returns uint8 (H, W, 3).
    atmo: an Atmosphere (then sky, sun_col and ambient default to its own);
    aerial: (density, height falloff, base height): the distance hazed into
    the colour of the air (atmo.haze); volumes: Volume list, marched at
    vol_scale of the resolution."""
    if atmo is not None:
        sky = sky or atmo.sky
        sun_col = atmo.sun_col if sun_col is None else sun_col
        ambient = atmo.ambient if ambient is None else ambient
    sun_col = "#fff0d8" if sun_col is None else sun_col
    ambient = ("#40506a", "#1a1a20") if ambient is None else ambient
    Ws, Hs = W * ss, H * ss
    amb = (_col(ambient[0]), _col(ambient[1]))
    hz = _Haze(aerial, atmo) if aerial is not None and atmo is not None else None
    js, is_ = torch.meshgrid(torch.arange(Hs, device=DEV, dtype=F), torch.arange(Ws, device=DEV, dtype=F), indexing="ij")
    js, is_ = js.reshape(-1), is_.reshape(-1)
    M = js.shape[0]
    color = torch.zeros(M, 3, device=DEV)
    emis = torch.zeros(M, 3, device=DEV)
    depth = torch.zeros(M, device=DEV)
    pix = 1.2 * 2 * cam.tan / Hs
    for s0 in range(0, M, tile):
        sl = slice(s0, min(M, s0 + tile))
        o, d = cam.rays(Ws, Hs, js[sl], is_[sl])
        t, hit = _march(scene, o, d, tmax, steps, pix)
        c, em = _shade(scene, o, d, t, hit, sky, sun_dir, sun_col, sun_soft, lights, amb, ao_scale, tmax)
        tt = torch.where(hit, t, torch.full_like(t, tmax))
        if fog is not None:
            dens, fcol, fall, y0 = fog
            # height fog: integral of exp(-fall (y - y0)) along the ray
            oy, dy = o[:, 1] - y0, d[:, 1]
            k = fall * dy
            integ = torch.where(k.abs() > 1e-4, torch.exp(-fall * oy) * (1 - torch.exp(-k * tt)) / k, torch.exp(-fall * oy) * tt)
            a = (1 - torch.exp(-dens * integ.clamp_min(0))).clamp(0, 1)[:, None]
            fc = srgb(fcol) if isinstance(fcol, str) else fcol
            if sun_dir is not None:
                cs = (d @ norm(vec(*sun_dir))).clamp(0, 1)[:, None]
                fc = fc + _col(sun_col) * cs ** 8 * 0.5
            c = c * (1 - a) + fc * a
            em = em * (1 - a)
        if hz is not None:
            a = torch.where(hit, hz(o, d, tt), torch.zeros_like(tt))[:, None]
            c = c * (1 - a) + hz.col(d) * a
            em = em * (1 - a)
        color[sl], emis[sl], depth[sl] = c, em, tt
    color = color.reshape(Hs, Ws, 3).permute(2, 0, 1)
    emis = emis.reshape(Hs, Ws, 3).permute(2, 0, 1)
    depth = depth.reshape(Hs, Ws)
    if volumes:
        vw, vh = int(Ws * vol_scale), int(Hs * vol_scale)
        vj, vi = torch.meshgrid(torch.arange(vh, device=DEV, dtype=F), torch.arange(vw, device=DEV, dtype=F), indexing="ij")
        vj, vi = vj.reshape(-1), vi.reshape(-1)
        dmin = -torch.nn.functional.adaptive_max_pool2d(-depth[None, None], (vh, vw))[0, 0].reshape(-1)
        Tv = torch.zeros(vh * vw, device=DEV)
        Iv = torch.zeros(vh * vw, 3, device=DEV)
        L = norm(vec(*sun_dir)) if sun_dir is not None else None
        for s0 in range(0, vh * vw, tile):
            sl = slice(s0, min(vh * vw, s0 + tile))
            o, d = cam.rays(vw, vh, vj[sl], vi[sl])
            Tv[sl], Iv[sl] = _render_volumes(volumes, o, d, dmin[sl], L, _col(sun_col), amb[0], hz)
        up = lambda x: torch.nn.functional.interpolate(x[None], size=(Hs, Ws), mode="bilinear", align_corners=False)[0]
        Tv = up(Tv.reshape(1, vh, vw))
        Iv = up(Iv.reshape(vh, vw, 3).permute(2, 0, 1))
        color = color * Tv + Iv
        emis = emis * Tv
    if shafts > 0 and sun_dir is not None:
        color = color + _shafts(scene, cam, W, H, sun_dir, sun_col, tmax, depth, Ws, Hs) * shafts
    if ss > 1:
        color = torch.nn.functional.avg_pool2d(color[None], ss)[0]
        emis = torch.nn.functional.avg_pool2d(emis[None], ss)[0]
        depth = torch.nn.functional.avg_pool2d(depth[None, None], ss)[0, 0]
    if dof is not None:
        focus, strength = dof
        coc = ((depth - focus).abs() / depth.clamp_min(1e-3)).clamp(0, 1) * strength
        b1, b2 = gauss_blur(color, max(1.0, H / 400)), gauss_blur(color, max(2.0, H / 160))
        k1 = (coc * 2).clamp(0, 1)[None]
        k2 = ((coc - 0.5) * 2).clamp(0, 1)[None]
        color = color * (1 - k1) + b1 * k1
        color = color * (1 - k2) + b2 * k2
    if bloom > 0:
        bright = (color - 1.2).clamp_min(0) + emis * 0.6
        color = color + (gauss_blur(bright, H / 120) * 0.6 + gauss_blur(bright, H / 40) * 0.5 + gauss_blur(bright, H / 12) * 0.35) * bloom
    col = aces(color * exposure)
    if grade is not None:
        lum = (col * vec(0.2126, 0.7152, 0.0722)[:, None, None]).sum(0, keepdim=True)
        sh_t, hl_t = srgb(grade[0])[:, None, None], srgb(grade[1])[:, None, None]
        col = col * (sh_t * (1 - lum) + hl_t * lum) / ((sh_t + hl_t) / 2).clamp_min(1e-3) * 0.5 + col * 0.5
    if vignette > 0:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=DEV), torch.linspace(-1, 1, W, device=DEV), indexing="ij")
        v = 1 - vignette * ((xx * xx * 0.6 + yy * yy) ** 1.4).clamp(0, 1)
        col = col * v[None]
    col = to_srgb(col.clamp(0, 1))
    if grain > 0:
        col = col + (torch.rand(1, H, W, device=DEV) - 0.5) * grain
    out = col.clamp(0, 1).permute(1, 2, 0)
    return (out * 255 + 0.5).to(torch.uint8).cpu().numpy()


def _shafts(scene, cam, W, H, sun_dir, sun_col, tmax, depth, Ws, Hs, n=40):
    """Light shafts: along each ray (quarter resolution), the share of fog
    samples the sun reaches, blurred and scaled up."""
    w, h = Ws // 4, Hs // 4
    js, is_ = torch.meshgrid(torch.arange(h, device=DEV, dtype=F), torch.arange(w, device=DEV, dtype=F), indexing="ij")
    o, d = cam.rays(w, h, js.reshape(-1), is_.reshape(-1))
    dq = torch.nn.functional.interpolate(depth[None, None], size=(h, w), mode="area")[0, 0].reshape(-1)
    L = norm(vec(*sun_dir))
    acc = torch.zeros(o.shape[0], device=DEV)
    for k in range(n):
        s_ = (k + torch.rand(1, device=DEV)) / n
        tt = (dq.clamp(max=tmax * 0.6) * s_)
        p = o + d * tt[:, None]
        lit = torch.ones_like(acc)
        ts = torch.full_like(acc, 0.05)
        for _ in range(20):
            dd = scene.dist(p + L * ts[:, None])
            lit = torch.minimum(lit, (dd / (0.05 * ts)).clamp(0, 1))
            ts = ts + dd.clamp(0.05, 1.5)
        acc = acc + lit / n
    cs = (d @ L).clamp(0, 1)
    img = (acc * (0.25 + 0.75 * cs ** 4)).reshape(1, h, w)
    img = gauss_blur(img, 2.0)
    img = torch.nn.functional.interpolate(img[None], size=(Hs, Ws), mode="bilinear", align_corners=False)[0]
    return _col(sun_col)[:, None, None] * img * 0.35


# ─── a physical sky, aerial perspective and volumes ─────────────────────────
def _col(c):
    return srgb(c) if isinstance(c, str) else c


def _sphere_exit(o, d, R):
    """Distance along d from o (inside the sphere of radius R about the origin) to its surface."""
    b = (o * d).sum(-1)
    c = (o * o).sum(-1) - R * R
    return -b + torch.sqrt((b * b - c).clamp_min(0))


class Atmosphere:
    """Rayleigh and Mie single scattering in an Earth-sized atmosphere,
    integrated numerically once on a table of directions (azimuth ×
    elevation, finer near the horizon) and then looked up.

    sun_dir     towards the sun (its elevation sets the hour: the colours of
                dusk and dawn come out of the physics)
    sun_power   the sun's intensity (the sky's brightness follows)
    mie         Mie (haze, dust) scattering at sea level: 21e-6 clear,
                60e-6 hazy, 200e-6 dusty
    rayleigh    scale of the air's scattering (1 = Earth)
    tint        a slight colour grade of the whole sky (hex)

    .sky(d)       radiance of the sky in directions d (with the sun disc)
    .sun_col      the sunlight reaching the ground (its colour at that hour)
    .ambient      (sky light from above, light bounced from the ground)
    .haze(d)      the colour of distant air in directions d (aerial perspective)
    """
    RE, RA, HR, HM = 6360e3, 6420e3, 7994.0, 1200.0

    def __init__(self, sun_dir, sun_power=22.0, mie=21e-6, mie_g=0.76, rayleigh=1.0, ground="#5a4a3a",
                 tint=None, res=(384, 128), sun_size=0.99996, samples=32, light_samples=8):
        self.sun = norm(vec(*sun_dir))
        self.power, self.g, self.sun_size = sun_power, mie_g, sun_size
        self.bR = vec(5.8e-6, 13.5e-6, 33.1e-6) * rayleigh
        self.bM = mie
        self.bO = vec(0.650e-6, 1.881e-6, 0.085e-6) * 1.0       # ozone: absorbs orange, keeps twilight blue
        self.ground = srgb(ground)
        self.tint = srgb(tint) / srgb(tint).max() if tint else None
        W, H = res
        v = (torch.arange(H, device=DEV, dtype=F) + 0.5) / H
        el = -0.1 + (math.pi / 2 + 0.1) * v * v
        az = (torch.arange(W, device=DEV, dtype=F) + 0.5) / W * math.tau - math.pi
        E, A = torch.meshgrid(el, az, indexing="ij")
        d = torch.stack([torch.cos(E) * torch.sin(A), torch.sin(E), -torch.cos(E) * torch.cos(A)], -1).reshape(-1, 3)
        self.table = self._scatter(d, samples, light_samples).reshape(H, W, 3).permute(2, 0, 1)[None]
        # sunlight at the ground: the sun's light through the whole air column
        o = vec(0, self.RE + 2.0, 0)[None]
        odR, odM, odO = self._optical(o, self.sun[None], light_samples * 4)
        self.sun_T = torch.exp(-self._tau(odR, odM, odO))[0]
        self.sun_col = self.sun_T * sun_power / 7.0
        if self.tint is not None:
            self.sun_col = self.sun_col * (0.7 + 0.3 * self.tint)
        # sky light from above: the cosine-weighted mean of the upper half of the table
        w = torch.sin(E.clamp_min(0)) * torch.cos(E)
        up = (self.table[0] * w[None]).sum((1, 2)) / w.sum()
        self.ambient = (up * 1.6, up * 0.5 * self.ground / self.ground.max() + self.sun_col * self.ground * 0.15 * max(0.0, float(self.sun[1])))

    def _optical(self, o, d, n):
        """Optical depths (Rayleigh, Mie) from o along d to the top of the air; inf if the earth is in the way."""
        L = _sphere_exit(o, d, self.RA)
        ds = L / n
        odR = torch.zeros_like(L)
        odM = torch.zeros_like(L)
        odO = torch.zeros_like(L)
        for k in range(n):
            p = o + d * ((k + 0.5) * ds)[:, None]
            h = length(p) - self.RE
            odR = odR + torch.exp(-h / self.HR) * ds
            odM = odM + torch.exp(-h / self.HM) * ds
            odO = odO + self._ozone(h) * ds
        # the earth's shadow
        b = (o * d).sum(-1)
        c = (o * o).sum(-1) - self.RE * self.RE
        blocked = (b < 0) & (b * b - c > 0)
        inf = torch.full_like(odR, 1e9)
        return torch.where(blocked, inf, odR), torch.where(blocked, inf, odM), torch.where(blocked, inf, odO)

    @staticmethod
    def _ozone(h):
        return (1 - (h - 25000).abs() / 15000).clamp_min(0)

    def _tau(self, r, m, oz):
        return self.bR * r[:, None] + self.bM * 1.1 * m[:, None] + self.bO * oz[:, None]

    def _scatter(self, d, n, nl):
        N = d.shape[0]
        o = vec(0, self.RE + 2.0, 0).expand(N, 3)
        L = _sphere_exit(o, d, self.RA)
        # rays into the ground stop there
        b = (o * d).sum(-1)
        c = (o * o).sum(-1) - self.RE * self.RE
        disc = b * b - c
        tg = -b - torch.sqrt(disc.clamp_min(0))
        hitg = (disc > 0) & (tg > 0)
        L = torch.where(hitg, tg, L)
        ds = L / n
        mu = (d @ self.sun)
        pR = 3 / (16 * math.pi) * (1 + mu * mu)
        g = self.g
        pM = 3 / (8 * math.pi) * ((1 - g * g) * (1 + mu * mu)) / ((2 + g * g) * (1 + g * g - 2 * g * mu).clamp_min(1e-6) ** 1.5)
        sR = torch.zeros(N, 3, device=DEV)
        sM = torch.zeros(N, 3, device=DEV)
        odR = torch.zeros(N, device=DEV)
        odM = torch.zeros(N, device=DEV)
        odO = torch.zeros(N, device=DEV)
        for k in range(n):
            p = o + d * ((k + 0.5) * ds)[:, None]
            h = length(p) - self.RE
            hr, hm = torch.exp(-h / self.HR) * ds, torch.exp(-h / self.HM) * ds
            odR, odM, odO = odR + hr, odM + hm, odO + self._ozone(h) * ds
            lR, lM, lO = self._optical(p, self.sun.expand(N, 3), nl)
            T = torch.exp(-self._tau(odR + lR, odM + lM, odO + lO))
            sR = sR + T * hr[:, None]
            sM = sM + T * hm[:, None]
        out = self.power * (sR * self.bR * pR[:, None] + sM * self.bM * pM[:, None])
        # the ground seen below the horizon: lit by the sun, through the air
        Tv = torch.exp(-self._tau(odR, odM, odO))
        gl = self.ground * self.power / 7.0 * max(0.05, float(self.sun[1])) * 0.3
        out = torch.where(hitg[:, None], out + Tv * gl, out)
        if self.tint is not None:
            out = out * (0.75 + 0.25 * self.tint)
        return out

    def _lookup(self, d):
        el = torch.asin(d[:, 1].clamp(-1, 1)).clamp(-0.1, math.pi / 2)
        v = torch.sqrt(((el + 0.1) / (math.pi / 2 + 0.1)).clamp(0, 1))
        az = torch.atan2(d[:, 0], -d[:, 2])
        u = az / math.pi
        grid = torch.stack([u, v * 2 - 1], -1)[None, None]
        # wrap the azimuth: pad one column on each side
        t = self.table
        t = torch.cat([t[..., -1:], t, t[..., :1]], -1)
        W = self.table.shape[-1]
        grid = grid.clone()
        grid[..., 0] = grid[..., 0] * W / (W + 2)
        return torch.nn.functional.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T

    def sky(self, d):
        c = self._lookup(d)
        cs = (d @ self.sun).clamp(-1, 1)
        # the sun's disc, limb-darkened, through the air
        r = ((1 - cs) / (1 - self.sun_size)).clamp(0, 1)
        disc = (cs > self.sun_size).float() * (1 - 0.6 * (1 - torch.sqrt((1 - r).clamp_min(0))))
        c = c + self.sun_T * self.power * 6.0 * disc[:, None] * (d[:, 1] > -0.01).float()[:, None]
        return c

    def haze(self, d):
        dd = d.clone()
        dd[:, 1] = d[:, 1].clamp(0.015, 0.25)
        return self._lookup(norm(dd))


class Volume:
    """A participating medium (smoke, cloud, dust, ash): density(p) → σ
    (extinction per unit length) inside the box lo..hi, lit by the sun
    (shadowed by itself), the sky and its own emission.

    albedo      the colour of the particles (single scattering albedo)
    g           the phase function's anisotropy (forward scattering: silver
                linings towards the sun)
    emission    f(p, σ) → (M, 3) radiance: glowing ash, fire
    steps       samples along each ray inside the box; light_steps towards the sun
    light_dist  how far the light march looks towards the sun"""

    def __init__(self, density, lo, hi, albedo="#e0e0e0", g=0.5, emission=None, steps=48, light_steps=6,
                 light_dist=None, ambient=1.0):
        self.density, self.lo, self.hi = density, vec(*lo), vec(*hi)
        self.albedo = _col(albedo)
        self.g, self.emission = g, emission
        self.steps, self.light_steps = steps, light_steps
        self.light_dist = light_dist or float(length(self.hi - self.lo)) * 0.35
        self.amb = ambient

    def span(self, o, d):
        inv = 1.0 / torch.where(d.abs() < 1e-8, torch.full_like(d, 1e-8), d)
        t0 = (self.lo - o) * inv
        t1 = (self.hi - o) * inv
        tn = torch.minimum(t0, t1).max(-1).values.clamp_min(0)
        tf = torch.maximum(t0, t1).min(-1).values
        return tn, tf


def _hg(mu, g):
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * mu).clamp_min(1e-6) ** 1.5)


def _render_volumes(vols, o, d, depth, sun, sun_col, amb_up, haze_fn):
    """Marches the volumes along the rays (o, d) up to depth: returns the
    transmittance (M,) and the in-scattered light (M, 3), far volumes first."""
    M = o.shape[0]
    T = torch.ones(M, device=DEV)
    I = torch.zeros(M, 3, device=DEV)
    order = sorted(vols, key=lambda v: -float(length((v.lo + v.hi) / 2 - o[0])))
    for v in order:
        tn, tf = v.span(o, d)
        tf = torch.minimum(tf, depth)
        act = (tf > tn).nonzero().squeeze(1)
        if act.numel() == 0:
            continue
        oo, dd, a0, a1 = o[act], d[act], tn[act], tf[act]
        n = act.numel()
        dt = (a1 - a0) / v.steps
        tv = torch.ones(n, device=DEV)
        iv = torch.zeros(n, 3, device=DEV)
        mu = (dd @ sun) if sun is not None else None
        ph = (0.75 * _hg(mu, v.g) + 0.25 * _hg(mu, -0.25)) * 4 * math.pi if sun is not None else None
        jit = torch.rand(n, device=DEV)
        for k in range(v.steps):
            t = a0 + (k + jit) * dt
            p = oo + dd * t[:, None]
            live = (tv > 0.01).nonzero().squeeze(1)
            if live.numel() == 0:
                break
            sig = v.density(p[live]).clamp_min(0)
            m = (sig > 1e-5).nonzero().squeeze(1)
            if m.numel() == 0:
                continue
            idx = live[m]
            s = sig[m]
            pp = p[idx]
            L = torch.zeros(idx.numel(), 3, device=DEV)
            if sun is not None:
                od = torch.zeros(idx.numel(), device=DEV)
                ls = v.light_dist / v.light_steps
                for j in range(v.light_steps):
                    od = od + v.density(pp + sun * ((j + 0.5) * ls)).clamp_min(0) * ls
                Tl = torch.exp(-od) * 0.7 + torch.exp(-od * 0.25) * 0.3          # multiple scattering, cheaply
                powder = 1 - torch.exp(-s * 2 * dt[idx])
                L = sun_col * (Tl * ph[idx] * (0.5 + 0.5 * powder))[:, None]
            hfrac = ((pp[:, 1] - v.lo[1]) / (v.hi[1] - v.lo[1]).clamp_min(1e-3)).clamp(0, 1)
            L = L + amb_up * v.amb * (0.45 + 0.55 * hfrac)[:, None]
            L = L * v.albedo
            if v.emission is not None:
                L = L + v.emission(pp, s)
            if haze_fn is not None:
                a = haze_fn(oo[idx], dd[idx], t[idx])[:, None]
                L = L * (1 - a) + haze_fn.col(dd[idx]) * a
            absorb = 1 - torch.exp(-s * dt[idx])
            iv[idx] = iv[idx] + (tv[idx] * absorb)[:, None] * L
            tv[idx] = tv[idx] * (1 - absorb)
        I[act] = I[act] * tv[:, None] + iv
        T[act] = T[act] * tv
    return T, I


class _Haze:
    """Aerial perspective: height-dependent air between the eye and a point."""
    def __init__(self, aerial, atmo):
        self.dens, self.fall, self.y0 = aerial
        self.atmo = atmo

    def __call__(self, o, d, t):
        oy, dy = o[:, 1] - self.y0, d[:, 1]
        k = self.fall * dy
        integ = torch.where(k.abs() > 1e-5, torch.exp(-self.fall * oy) * (1 - torch.exp(-k * t)) / torch.where(k.abs() > 1e-5, k, torch.ones_like(k)),
                            torch.exp(-self.fall * oy) * t)
        return (1 - torch.exp(-self.dens * integ.clamp_min(0))).clamp(0, 1)

    def col(self, d):
        return self.atmo.haze(d)
