"""E3d: verify the weak-perspective fit with a generic nonlinear least-squares solver."""
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
D = Path(__file__).resolve().parent / "extract"
UPPER = list(range(0, 23))
def wp_res(x, P, Q):
    s, rv, t = np.exp(x[0]), x[1:4], x[4:6]
    return ((s * P @ R.from_rotvec(rv).as_matrix()[:2].T + t) - Q).ravel()
out_wp, out_aff, out_wp_aniso = [], [], []
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I = d["image"] * np.array([W, H, W]); Pw = d["world"].astype(np.float64); V = d["visibility"]
    for t in range(len(I)):
        ok = np.isfinite(Pw[t]).all(1) & (V[t] > 0.5)
        idx = [i for i in UPPER if ok[i]]
        if len(idx) < 6: continue
        P, Q = Pw[t, idx], I[t, idx, :2]
        sw = np.linalg.norm(I[t, 11, :2] - I[t, 12, :2])
        best = None
        for rv0 in ([0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi]):
            x0 = np.r_[np.log(400.0), rv0, Q.mean(0)]
            r = least_squares(wp_res, x0, args=(P, Q))
            if best is None or r.cost < best.cost: best = r
        res = np.linalg.norm(best.fun.reshape(-1, 2), axis=1)
        out_wp.append(np.median(res) / sw * 100)
        A = np.c_[P, np.ones(len(idx))]; M, *_ = np.linalg.lstsq(A, Q, rcond=None)
        out_aff.append(np.median(np.linalg.norm(A @ M - Q, axis=1)) / sw * 100)
pr = lambda a: f"median {np.median(a):.2f} p95 {np.percentile(a,95):.2f}"
print("weak perspective (nonlinear LSQ, 4 restarts), xy residual %SW:", pr(out_wp))
print("affine 2x3, xy residual %SW:", pr(out_aff))
