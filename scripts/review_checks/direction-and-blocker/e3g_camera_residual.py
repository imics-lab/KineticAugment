"""E3g: per-frame camera fit between MediaPipe WORLD landmarks and IMAGE landmarks of the same
Holistic call (re-done with the corrected weak-perspective solver of E3e; E3b part 1 used a buggy
alternating solver). Residual on upper-body landmarks 0-22 with vis>0.5, % image shoulder width."""
import warnings
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")
D = Path(__file__).resolve().parent / "extract"
def wp_res(P, Q):
    def res(x):
        return ((np.exp(x[0]) * P @ R.from_rotvec(x[1:4]).as_matrix()[:2].T + x[4:6]) - Q).ravel()
    best = None
    for rv0 in ([0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi]):
        r = least_squares(res, np.r_[np.log(400.0), rv0, Q.mean(0)])
        if best is None or r.cost < best.cost: best = r
    return np.linalg.norm(best.fun.reshape(-1, 2), axis=1)
def aff_res(P, Q):
    A = np.c_[P, np.ones(len(P))]; M, *_ = np.linalg.lstsq(A, Q, rcond=None)
    return np.linalg.norm(A @ M - Q, axis=1)
wp, af = [], []
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I, Pw, V = d["image"] * np.array([W, H, W]), d["world"].astype(float), d["visibility"]
    for t in range(len(I)):
        ok = np.isfinite(Pw[t]).all(1) & (V[t] > 0.5)
        idx = [i for i in range(23) if ok[i]]
        if len(idx) < 6: continue
        sw = np.linalg.norm(I[t, 11, :2] - I[t, 12, :2])
        wp.append(np.median(wp_res(Pw[t, idx], I[t, idx, :2])) / sw * 100)
        af.append(np.median(aff_res(Pw[t, idx], I[t, idx, :2])) / sw * 100)
wp, af = np.array(wp), np.array(af)
print(f"frames {len(wp)}: per-frame median landmark residual, weak-perspective: median {np.median(wp):.2f} p95 {np.percentile(wp,95):.2f} %SW; "
      f"2x3 affine: median {np.median(af):.2f} p95 {np.percentile(af,95):.2f} %SW")
