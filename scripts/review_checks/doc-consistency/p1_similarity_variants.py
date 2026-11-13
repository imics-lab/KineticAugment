"""P1 robustness: is the similarity failure an artefact of handedness or of z-scale alone?

Variants (per frame, upper-body landmarks with visibility >= 0.5, pixel coords x*w, y*h, z*w):
  sim        7 params, proper rotation (doc's proposal)
  sim+refl   7 params, reflection allowed (rules out a handedness artefact)
  sim+zs     8 params: similarity followed by a free z-scale (pseudo-depth scale)
  sim xy     2-D similarity on x,y only (orthographic, drop pseudo-depth)
  affine     12 params
Also: sign of det(affine linear part).
"""
import glob
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

VIS = 0.5
DIMS = {"01986": (320, 240), "01987": (1280, 720), "01988": (640, 480),
        "01991": (288, 192), "01992": (288, 192)}
UPPER = list(range(0, 17)) + [23, 24]


def umeyama(src, dst, reflect=False):
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    C = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(src.shape[1])
    if not reflect and np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    var = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(S) @ D) / var
    t = md - s * R @ ms
    return s, R, t


def rms(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


out = {k: [] for k in ["sim", "sim+refl", "sim+zs", "sim xy", "affine"]}
dets = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    stem = Path(f).stem.split("_")[0]
    d = np.load(f)
    W, V = d["landmarks"].astype(float), d["visibility"].astype(float)
    I = np.load(f"data/test_output/{stem}_landmarks.npy").astype(float)[:, :33]
    w, h = DIMS[stem]
    Ipx = I * np.array([w, h, w])
    per = {k: [] for k in out}
    for t in range(len(W)):
        if not np.isfinite(W[t]).all() or np.abs(I[t]).sum() == 0:
            continue
        m = [i for i in UPPER if V[t, i] >= VIS]
        if len(m) < 6:
            continue
        src, tgt = W[t, m], Ipx[t, m]
        sw = np.linalg.norm(Ipx[t, 11, :2] - Ipx[t, 12, :2])
        s, R, tr = umeyama(src, tgt)
        per["sim"].append(rms(s * src @ R.T + tr, tgt) / sw)
        s, R, tr = umeyama(src, tgt, reflect=True)
        per["sim+refl"].append(rms(s * src @ R.T + tr, tgt) / sw)
        s2, R2, t2 = umeyama(src[:, :2], tgt[:, :2])
        # 2-D: drop pseudo-depth entirely (rotation in image plane only)
        # better: orthographic projection of a 3-D rotation; fit with least squares
        def ortho(p):
            Rm = Rotation.from_rotvec(p[:3]).as_matrix()
            return ((p[3] * src @ Rm.T)[:, :2] + p[4:6] - tgt[:, :2]).ravel()
        s0, R0, t0 = umeyama(src, tgt)
        r = least_squares(ortho, np.r_[Rotation.from_matrix(R0).as_rotvec(), s0, t0[:2]])
        per["sim xy"].append(np.sqrt(np.mean(r.fun.reshape(-1, 2) ** 2 * 2)) / sw)

        def simz(p):
            Rm = Rotation.from_rotvec(p[:3]).as_matrix()
            q = p[3] * src @ Rm.T + p[4:7]
            q[:, 2] *= p[7]
            return (q - tgt).ravel()
        r = least_squares(simz, np.r_[Rotation.from_matrix(R0).as_rotvec(), s0, t0, 1.0])
        per["sim+zs"].append(np.sqrt(np.mean(np.sum(r.fun.reshape(-1, 3) ** 2, axis=1))) / sw)
        X = np.hstack([src, np.ones((len(m), 1))])
        A, *_ = np.linalg.lstsq(X, tgt, rcond=None)
        per["affine"].append(rms(X @ A, tgt) / sw)
        dets.append(np.sign(np.linalg.det(A[:3])))
    print(stem, "  ".join(f"{k} {np.median(v)*100:5.1f}%" for k, v in per.items()))
    for k in out:
        out[k].extend(per[k])
print("pooled", "  ".join(f"{k} {np.median(v)*100:5.1f}%" for k, v in out.items()))
print("affine det sign counts:", {int(k): int(v) for k, v in zip(*np.unique(dets, return_counts=True))})
print("(residual = RMS per landmark / image-plane shoulder width in px)")
