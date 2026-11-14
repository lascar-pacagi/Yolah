"""portrait.py FIT.npz FRONT.jpg — bakes the player's face for the human figurine.

FIT.npz comes from a 3DDFA_V2 fit of the player's photos (~/opt/3DDFA_V2, the
mb1 model): the neutral 3DMM shape (identity only, no expression) `neutral`
(3×N, BFM units, y up, z toward the viewer), the dense vertices of the fit on
the front photo `front_dense` (3×N, image pixels), the triangles and the 68
keypoints. Writes ref/human_face3d.npz in the head's local units (the pupils at
x = ±0.085, y = 0.032; the face looks along +z):

  depth   (H, W)   z of the face's surface over the grid (NaN off the face)
  albedo  (H, W, 3) linear albedo from the photo, its lighting removed
  x0, y0, step     the grid: x = x0 + j·step, y = y0 - i·step
  lm      (68, 3)  the keypoints in head units
"""
import os
import sys
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

fit = np.load(sys.argv[1])
photo = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.float32) / 255.0
lin = photo ** 2.2
V = fit["neutral"].T.astype(np.float64)                     # N×3
kp = fit["kpt"]
vi = kp[0::3] // 3
L = V[vi]
eyes = [L[36:42].mean(0), L[42:48].mean(0)]
s = 0.17 / np.linalg.norm(eyes[1] - eyes[0])
mid = (eyes[0] + eyes[1]) / 2
P = (V - mid) * s + np.array([0.0, 0.032, 0.235])
lm = (L - mid) * s + np.array([0.0, 0.032, 0.235])

# vertex normals, to keep the surfaces that face the front
tri = fit["tri"]
tri = tri if tri.shape[1] == 3 else tri.T
fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
vn = np.zeros_like(P)
for k in range(3):
    np.add.at(vn, tri[:, k], fn)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
if vn[vi[30], 2] < 0:                                          # orientation of the triangles
    vn = -vn

# colours from the front photo at each vertex's position in it (visible ones)
img = fit["front_dense"].T                                    # N×3, x, y in pixels
H0, W0 = photo.shape[:2]
xi = np.clip(img[:, 0], 0, W0 - 1)
yi = np.clip(img[:, 1], 0, H0 - 1)
def bilinear(im, x, y):
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, im.shape[1] - 1); y1 = np.minimum(y0 + 1, im.shape[0] - 1)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x1] * fx * (1 - fy)
            + im[y1, x0] * (1 - fx) * fy + im[y1, x1] * fx * fy)
col = bilinear(lin, xi, yi)

keep = vn[:, 2] > 0.25
# the photo's lighting out: a 2nd-order spherical-harmonics light fitted on the
# vertex normals (in the photo's frame), per channel, robustly (beard, brows
# and eyes are outliers); albedo = colour / shading
R = fit["front_R"]
Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
nc = vn @ Rn.T
nc[:, 1] *= -1                                                  # image y is down
x, y, z = nc[:, 0], -nc[:, 1], nc[:, 2]
Y = np.stack([np.ones_like(x), x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1], 1)
use = keep.copy()
lum_v = col @ np.array([0.2126, 0.7152, 0.0722])
w = use.astype(np.float64)
for it in range(6):
    coef, *_ = np.linalg.lstsq(Y * w[:, None], lum_v * w, rcond=None)
    pred = Y @ coef
    r = lum_v - pred
    sd = 1.4826 * np.median(np.abs(r[use]))
    w = use * (np.abs(r) < 2.0 * sd)
shade = np.clip(pred, 1e-3, None) / np.median(pred[keep])
ratio = np.clip(lum_v / np.clip(pred, 1e-3, None), 0, 1.25)          # highlights clipped
vcol = col / np.clip(lum_v, 1e-4, None)[:, None] * (ratio * np.median(pred[keep]))[:, None]

step = 0.6 / 600
x0, y0 = -0.3, 0.32
W, H = 600, int((0.32 + 0.42) / step)
gx, gy = np.meshgrid(x0 + step * np.arange(W), y0 - step * np.arange(H))
depth = LinearNDInterpolator(P[keep, :2], P[keep, 2])(gx, gy)
alb = LinearNDInterpolator(P[keep, :2], vcol[keep])(gx, gy)
ok = ~np.isnan(depth)
def blur_masked(a, sig):
    w = gaussian_filter(ok.astype(np.float32), sig)
    a0 = np.where(ok[..., None] if a.ndim == 3 else ok, a, 0)
    if a.ndim == 3:
        return np.stack([gaussian_filter(a0[..., c], sig) for c in range(3)], -1) / (w[..., None] + 1e-6)
    return gaussian_filter(a0, sig) / (w + 1e-6)
base_l = np.median((alb @ np.array([0.2126, 0.7152, 0.0722]))[ok])
# the colour cast of the lights (warm on one side, cool on the other) out:
# a nearly uniform skin chroma, symmetric, under the luminance from the photo
LW = np.array([0.2126, 0.7152, 0.0722])
lum = np.nan_to_num(alb) @ LW
chroma = blur_masked(np.nan_to_num(alb), 14.0)
chroma = chroma / (chroma @ LW + 1e-6)[..., None]
chroma = 0.5 * (chroma + chroma[:, ::-1])
chroma = 0.8 * np.median(chroma[ok], 0) + 0.2 * chroma
# the stubble and brows (dark or grey hair): less saturated than the skin
lb = blur_masked(lum, 6.0)
grey = np.clip((lb - lum) / (lb + 1e-6) * 3, 0, 1)[..., None]
chroma = chroma * (1 - 0.5 * grey) + 0.5 * grey
albedo = np.clip(chroma * lum[..., None], 0, 1)
albedo[~ok] = np.nan
# the depth: lips smoothed (the figurine's lips are modelled and animated), and
# extended past the face's edge by a steep fall (the cranium covers it)
from scipy.ndimage import distance_transform_edt
dfill = np.where(ok, depth, 0.0)
mouth = lm[48:68].mean(0)
e = ((gx - mouth[0]) / 0.085) ** 2 + ((gy - mouth[1]) / 0.04) ** 2
wm = np.clip((1.6 - e) / 0.8, 0, 1) * ok
dsm = blur_masked(dfill, 10.0)
dfill = dfill * (1 - wm) + (dsm - 0.004) * wm
dist, (ii, jj) = distance_transform_edt(~ok, return_indices=True)
depth = np.where(ok, dfill, dfill[ii, jj] - 3.0 * dist * step)
outline = (dist - distance_transform_edt(ok)) * step                # signed distance to the face's outline
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "human_face3d.npz")
np.savez_compressed(out, depth=depth.astype(np.float32), albedo=albedo.astype(np.float32), mask=ok, out=outline.astype(np.float32),
                    x0=x0, y0=y0, step=step, lm=lm.astype(np.float32))
print("saved", out, "base", base_l, "chin", lm[8], "nose", lm[30], "mouth", lm[48], lm[54])
Image.fromarray((np.nan_to_num(albedo) ** (1 / 2.2) * 255).astype(np.uint8)).save(sys.argv[3] if len(sys.argv) > 3 else "/dev/null", "PNG")
