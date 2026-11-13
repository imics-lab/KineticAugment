"""Fitted pseudo-depth scale: image z (x image width) relative to x/y, from world->image fit."""
import glob
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
DIMS = {"01986": (320, 240), "01987": (1280, 720), "01988": (640, 480), "01991": (288, 192), "01992": (288, 192)}
UPPER = list(range(0, 17)) + [23, 24]
allz = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    stem = Path(f).stem.split("_")[0]; d = np.load(f)
    W, V = d["landmarks"].astype(float), d["visibility"].astype(float)
    I = np.load(f"data/test_output/{stem}_landmarks.npy").astype(float)[:, :33]
    w, h = DIMS[stem]; Ipx = I * np.array([w, h, w]); zs = []
    for t in range(len(W)):
        if not np.isfinite(W[t]).all() or np.abs(I[t]).sum() == 0: continue
        m = [i for i in UPPER if V[t, i] >= 0.5]
        if len(m) < 6: continue
        src, tgt = W[t, m], Ipx[t, m]
        def r(p):
            q = p[3] * src @ Rotation.from_rotvec(p[:3]).as_matrix().T + p[4:7]; q[:, 2] *= p[7]
            return (q - tgt).ravel()
        s0 = np.linalg.norm(tgt[:, :2] - tgt[:, :2].mean(0)) / np.linalg.norm(src[:, :2] - src[:, :2].mean(0))
        sol = least_squares(r, np.r_[0, 0, 0, s0, tgt.mean(0) - s0 * src.mean(0), 1.0])
        zs.append(sol.x[7])
    allz += zs
    print(f"{stem}: fitted z-scale (image z / metric-consistent z) median {np.median(zs):.2f}  p5 {np.percentile(zs,5):.2f}  p95 {np.percentile(zs,95):.2f}")
print(f"pooled median {np.median(allz):.2f}")
