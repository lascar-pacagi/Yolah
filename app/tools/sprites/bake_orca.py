"""bake_orca.py — the orca's signed distance grid for the Banquise orca (arctic.py).

From ref/orca.stl (a free low-poly killer whale from CGTrader, "Orcinus orca or
killer whale low poly", royalty-free licence; 950 triangles; the head at y≈-0.4,
the flukes at y≈11.7, the back toward +z), smoothed by Loop subdivision, the
signed distance is computed by brute force on the GPU (the nearest triangle; the
sign from the generalised winding number) over a grid and saved in mesh units:

  ref/orca_sdf.npz   sdf (D, H, W) over z, y, x ; lo, hi (the grid's box, x y z)
"""
import os
import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
m = trimesh.load(os.path.join(HERE, "ref", "orca.stl"))
m = m.subdivide_loop(iterations=2)
print("triangles", len(m.faces))
dev = "cuda"
Vt = torch.tensor(m.vertices[m.faces], dtype=torch.float32, device=dev)       # F×3×3
A, B, C = Vt[:, 0], Vt[:, 1], Vt[:, 2]

lo = np.array([-2.7, -0.8, -0.9]); hi = np.array([2.7, 12.1, 3.1])
step = 0.04
nx, ny, nz = [int(np.ceil((hi[k] - lo[k]) / step)) + 1 for k in range(3)]
hi = lo + step * (np.array([nx, ny, nz]) - 1)
print("grid", nx, ny, nz)


def tri_dist2(p):
    """Squared distance from points p (N×3) to every triangle (N×F), Ericson's closest point."""
    p = p[:, None, :]
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = (va + vb + vc)
    v = vb / (denom + 1e-20)
    w = vc / (denom + 1e-20)
    q = A + ab * v[..., None] + ac * w[..., None]                              # inside the face
    # the regions outside: vertices and edges
    def seg(P0, P1):
        e = P1 - P0
        t = (((p - P0) * e).sum(-1) / ((e * e).sum(-1) + 1e-20)).clamp(0, 1)
        return P0 + e * t[..., None]
    qa, qb, qc = seg(A, B), seg(B, C), seg(C, A)
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    dq = ((p - q) ** 2).sum(-1)
    de = torch.minimum(torch.minimum(((p - qa) ** 2).sum(-1), ((p - qb) ** 2).sum(-1)), ((p - qc) ** 2).sum(-1))
    return torch.where(inside, dq, de).min(1).values


def winding(p):
    """The generalised winding number (≈1 inside, 0 outside), from the triangles' solid angles."""
    a, b, c = A[None] - p[:, None], B[None] - p[:, None], C[None] - p[:, None]
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    num = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
    den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
    return (2 * torch.atan2(num, den)).sum(1) / (4 * np.pi)


xs = torch.linspace(lo[0], hi[0], nx, device=dev)
ys = torch.linspace(lo[1], hi[1], ny, device=dev)
zs = torch.linspace(lo[2], hi[2], nz, device=dev)
out = np.zeros((nz, ny, nx), np.float32)
for k in range(nz):
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    p = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full_like(gx.reshape(-1), float(zs[k]))], 1)
    ds = []
    for c in range(0, len(p), 1024):
        pc = p[c:c + 1024]
        d = tri_dist2(pc).sqrt()
        wn = winding(pc)
        ds.append(torch.where(wn.abs() > 0.5, -d, d))
    out[k] = torch.cat(ds).reshape(ny, nx).cpu().numpy()
    if k % 10 == 0:
        print(k, "/", nz, flush=True)
np.savez_compressed(os.path.join(HERE, "ref", "orca_sdf.npz"), sdf=out, lo=lo, hi=hi)
print("saved")
