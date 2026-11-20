"""
gltf_mesh.py — a skinned, animated glTF 2.0 mesh (one textured primitive, as
Sketchfab exports them) posed at any time of its clips, then turned into an
SDF part for render.py, so that a downloaded model is lit, shadowed and
framed exactly like the figurines built from primitives.

    m = GltfMesh("models/pteranodon/scene.gltf")
    v = m.pose("walking", 1.5)                 # (V, 3) skinned vertices, glTF units
    part_sdf, pattern = m.sdf_part(v, xf)      # see sdf_part

Skinning: linear blend, joint matrix = global(joint) · inverse bind matrix;
the clips' samplers are LINEAR (translations, scales) and spherical linear
(rotations). The texture colour of a point is read once per vertex sample:
sdf_part scatters many points of the surface (area weighted, with their UV
colour) into a grid, the unsigned distance to the nearest point in a narrow
band, a chamfer distance further out, minus a small thickness: wings that
are single double-sided sheets get a body, closed shapes stay closed.
"""
import json
import math
import os
import numpy as np
import torch
from PIL import Image
import render
from render import DEV, F

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(a, b, u):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = math.acos(d)
        q = (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)
    return q / np.linalg.norm(q)


class GltfMesh:
    def __init__(self, path):
        self.dir = os.path.dirname(path)
        g = self.g = json.load(open(path))
        self.bin = open(os.path.join(self.dir, g["buffers"][0]["uri"]), "rb").read()
        prim = g["meshes"][0]["primitives"][0]
        a = prim["attributes"]
        self.pos = self.acc(a["POSITION"]).astype(np.float64)
        self.uv = self.acc(a["TEXCOORD_0"]).astype(np.float64)
        self.joints = self.acc(a["JOINTS_0"]).astype(np.int64)
        w = self.acc(a["WEIGHTS_0"]).astype(np.float64)
        self.weights = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
        self.tri = self.acc(prim["indices"]).astype(np.int64).reshape(-1, 3)
        skin = g["skins"][0]
        self.skin_joints = skin["joints"]
        self.ibm = self.acc(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
        mat = g["materials"][prim.get("material", 0)]
        ext = mat.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness")
        tex = ext["diffuseTexture"]["index"] if ext else mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        img = g["images"][g["textures"][tex]["source"]]["uri"]
        self.tex = np.asarray(Image.open(os.path.join(self.dir, img)).convert("RGB"), np.float32) / 255.0
        self.occ = None                       # the baked ambient occlusion, multiplied into the colour
        if "occlusionTexture" in mat:
            oimg = g["images"][g["textures"][mat["occlusionTexture"]["index"]]["source"]]["uri"]
            o = np.asarray(Image.open(os.path.join(self.dir, oimg)).convert("L"), np.float32) / 255.0
            self.occ = o
        n = len(g["nodes"])
        self.parent = [-1] * n
        for i, nd in enumerate(g["nodes"]):
            for c in nd.get("children", []):
                self.parent[c] = i
        self.order = []                       # parents before children
        seen = set()

        def visit(i):
            if i in seen:
                return
            if self.parent[i] >= 0:
                visit(self.parent[i])
            seen.add(i)
            self.order.append(i)
        for i in range(n):
            visit(i)
        self.clips = {}
        for an in g.get("animations", []):
            ch = []
            for c in an["channels"]:
                s = an["samplers"][c["sampler"]]
                ch.append((c["target"]["node"], c["target"]["path"], self.acc(s["input"]).reshape(-1).astype(np.float64),
                           self.acc(s["output"]).astype(np.float64)))
            self.clips[an["name"]] = (ch, max(c[2][-1] for c in ch))

    def acc(self, i):
        a = self.g["accessors"][i]
        bv = self.g["bufferViews"][a["bufferView"]]
        nc = _NC[a["type"]]
        dt = np.dtype(_CT[a["componentType"]])
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or nc * dt.itemsize
        raw = np.frombuffer(self.bin, np.uint8, count=stride * (a["count"] - 1) + nc * dt.itemsize, offset=off)
        out = np.lib.stride_tricks.as_strided(raw, (a["count"], nc * dt.itemsize), (stride, 1)).copy()
        out = out.view(dt).reshape(a["count"], nc)
        if a.get("normalized"):
            out = out.astype(np.float64) / np.iinfo(dt).max
        return out

    def duration(self, clip):
        return self.clips[clip][1]

    def _local(self, clip, time):
        trs = {}
        for i, nd in enumerate(self.g["nodes"]):
            if "matrix" in nd:
                trs[i] = np.array(nd["matrix"], np.float64).reshape(4, 4).T
            else:
                trs[i] = [np.array(nd.get("translation", [0, 0, 0]), np.float64),
                          np.array(nd.get("rotation", [0, 0, 0, 1]), np.float64),
                          np.array(nd.get("scale", [1, 1, 1]), np.float64)]
        if clip is not None:
            for node, path, ts, vals in self.clips[clip][0]:
                if isinstance(trs[node], np.ndarray):
                    continue
                k = int(np.clip(np.searchsorted(ts, time) - 1, 0, len(ts) - 2)) if len(ts) > 1 else 0
                if len(ts) == 1:
                    v = vals[0]
                else:
                    u = float(np.clip((time - ts[k]) / max(ts[k + 1] - ts[k], 1e-9), 0, 1))
                    v = _slerp(vals[k], vals[k + 1], u) if path == "rotation" else vals[k] + u * (vals[k + 1] - vals[k])
                trs[node][{"translation": 0, "rotation": 1, "scale": 2}[path]] = v
        out = {}
        for i, v in trs.items():
            if isinstance(v, np.ndarray):
                out[i] = v
            else:
                m = np.eye(4)
                m[:3, :3] = _quat_mat(v[1]) * v[2][None, :]
                m[:3, 3] = v[0]
                out[i] = m
        return out

    def globals(self, clip=None, time=0.0, turn=None):
        """Global matrices of every node; turn: {node index: (axis, angle)}, a rotation about a
        WORLD axis through the node's position, carried by its children (a head shake)."""
        loc = self._local(clip, time)
        glob = {}
        for i in self.order:
            p = self.parent[i]
            glob[i] = loc[i] if p < 0 else glob[p] @ loc[i]
            if turn and i in turn:
                axis, ang = turn[i]
                ax = np.asarray(axis, np.float64) / np.linalg.norm(axis)
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                R = np.eye(4)
                R[:3, :3] = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
                c = glob[i][:3, 3].copy()
                Tc, Ti = np.eye(4), np.eye(4)
                Tc[:3, 3], Ti[:3, 3] = c, -c
                glob[i] = Tc @ R @ Ti @ glob[i]
        return glob

    def node(self, name):
        return next(i for i, nd in enumerate(self.g["nodes"]) if nd.get("name", "").startswith(name))

    def pose(self, clip=None, time=0.0, turn=None):
        """(V, 3) skinned vertex positions at `time` seconds of `clip` (None: the bind pose)."""
        glob = self.globals(clip, time, turn)
        J = np.stack([glob[j] @ self.ibm[k] for k, j in enumerate(self.skin_joints)])     # (J, 4, 4)
        M = np.einsum("vk,vkij->vij", self.weights, J[self.joints])                     # (V, 4, 4)
        return np.einsum("vij,vj->vi", M[:, :3, :3], self.pos) + M[:, :3, 3]

    def joint_position(self, name, clip=None, time=0.0):
        return self.globals(clip, time)[self.node(name)][:3, 3].copy()

    def loop_pose(self, clip, start, length, u, fade=0.3, turn=None):
        """A seamless loop cut from a clip: [start, start + length) played for u in [0, 1),
        its last `fade` cross-faded into the poses just before `start` (so u = 1 is u = 0)."""
        p = self.pose(clip, start + u * length, turn)
        if u > 1 - fade:
            w = (u - (1 - fade)) / fade
            w = w * w * (3 - 2 * w)
            p = (1 - w) * p + w * self.pose(clip, start + (u - 1) * length, turn)
        return p

    def surface_samples(self, verts, spacing, seed=0):
        """Points scattered over the posed surface, about `spacing` apart, with their texture colour
        (linear light): (P, 3) and (P, 3) tensors on the device."""
        v = torch.tensor(verts, dtype=F, device=DEV)
        tri = torch.tensor(self.tri, device=DEV)
        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        area = 0.5 * torch.linalg.cross(b - a, c - a).norm(dim=1)
        n = int(area.sum().item() / spacing ** 2 * 1.6) + len(self.tri)
        gen = torch.Generator(device=DEV).manual_seed(seed)
        f = torch.multinomial(area + 1e-12, n, replacement=True, generator=gen)
        r1 = torch.rand(n, device=DEV, generator=gen).sqrt()
        r2 = torch.rand(n, device=DEV, generator=gen)
        w = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
        p = w[:, 0:1] * a[f] + w[:, 1:2] * b[f] + w[:, 2:3] * c[f]
        uv = torch.tensor(self.uv, dtype=F, device=DEV)
        q = w[:, 0:1] * uv[tri[f, 0]] + w[:, 1:2] * uv[tri[f, 1]] + w[:, 2:3] * uv[tri[f, 2]]
        tex = torch.tensor(self.tex, device=DEV)
        H, W = tex.shape[:2]
        x = (q[:, 0] % 1.0 * (W - 1)).round().long()
        y = (q[:, 1] % 1.0 * (H - 1)).round().long()          # glTF: v down from the top of the image
        col = tex[y, x] ** 2.2
        if self.occ is not None:
            occ = torch.tensor(self.occ, device=DEV)
            oy = (q[:, 1] % 1.0 * (occ.shape[0] - 1)).round().long()
            ox = (q[:, 0] % 1.0 * (occ.shape[1] - 1)).round().long()
            col = col * occ[oy, ox][:, None]
        return p, col


def grid_part(points, colors, h, thickness, band=3, pad=0.03):
    """An SDF (function of (N, 3) local points) and a colour lookup for a cloud
    of surface points: the unsigned distance to the nearest point, exact in a
    band of `band` cells, chamfer outside, minus `thickness`; beyond the grid,
    the distance to its box plus the grid's border value."""
    lo = points.min(0).values - pad
    hi = points.max(0).values + pad
    dims = ((hi - lo) / h).ceil().long() + 1
    D, Hh, W = int(dims[2]), int(dims[1]), int(dims[0])        # grid[z, y, x]
    nvox = D * Hh * W
    P = points.shape[0]
    assert P < (1 << 24), P
    EMPTY = (1 << 62)
    key = torch.full((nvox,), EMPTY, dtype=torch.long, device=DEV)   # (distance in 1e-4 cells) << 24 | point index
    cell = ((points - lo) / h).floor().long()
    r = torch.arange(-band, band + 1, device=DEV)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    idx_pt = torch.arange(P, device=DEV)
    for o in offs:
        c = cell + o
        ok = ((c >= 0) & (c < dims)).all(1)
        cc = c[ok]
        centre = lo + (cc.to(F) + 0.5) * h
        d = (centre - points[ok]).norm(dim=1)
        k = ((d / h * 1e4).long() << 24) | idx_pt[ok]
        lin = (cc[:, 2] * Hh + cc[:, 1]) * W + cc[:, 0]
        key.scatter_reduce_(0, lin, k, reduce="amin")
    near = key < EMPTY
    dist = torch.full((nvox,), 1e3, device=DEV)                       # far: filled by the chamfer below
    dist[near] = (key[near] >> 24).to(F) / 1e4 * h
    col = torch.zeros(nvox, 3, device=DEV)
    col[near] = colors[key[near] & ((1 << 24) - 1)]
    # chamfer outward: each pass lets the distance grow by at most one cell step
    g = dist.reshape(1, 1, D, Hh, W)
    for _ in range(int(max(D, Hh, W))):
        m = -torch.nn.functional.max_pool3d(-g, 3, 1, 1) + h * math.sqrt(3)
        g2 = torch.minimum(g, m)
        if torch.equal(g2, g):
            break
        g = g2
    sdf_grid = (g - thickness).contiguous()
    col_grid = col.reshape(D, Hh, W, 3).permute(3, 0, 1, 2)[None].contiguous()
    c0 = lo + 0.5 * h                                   # the first and last cell centres (align_corners)
    span = (dims.to(F) - 1) * h
    mid, half = c0 + span / 2, span / 2

    def to_uvw(p):
        return ((p - mid) / half)[None, :, None, None, :]     # grid_sample: (x, y, z) in [-1, 1]

    def sdf(p):
        inside = torch.nn.functional.grid_sample(sdf_grid, to_uvw(p), align_corners=True, padding_mode="border").reshape(-1)
        q = (p - mid).abs() - half
        outside = q.clamp(min=0).norm(dim=1)
        return inside + outside

    def colour(p):
        return torch.nn.functional.grid_sample(col_grid, to_uvw(p), align_corners=True, padding_mode="border").reshape(3, -1).T

    return sdf, colour
