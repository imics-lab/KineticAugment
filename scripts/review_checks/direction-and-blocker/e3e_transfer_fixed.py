"""E3e (supersedes E3b part 2, whose alternating weak-perspective solver was buggy; see E3d):
delta transfer validated on natural motion, with a correctly fitted per-frame camera.
Camera options: weak perspective (s,R,t; nonlinear LSQ) and 2x3 affine. Also reports the
'full replacement' alternative (project reconstructed world landmarks through the camera) at ZERO
augmentation, i.e. what a consumer would get if the converter's output replaced the input."""
import warnings
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE

D = Path(__file__).resolve().parent / "extract"
UPPER = list(range(0, 23))
def wp_fit(P, Q):
    def res(x):
        return ((np.exp(x[0]) * P @ R.from_rotvec(x[1:4]).as_matrix()[:2].T + x[4:6]) - Q).ravel()
    best = None
    for rv0 in ([0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi]):
        r = least_squares(res, np.r_[np.log(400.0), rv0, Q.mean(0)])
        if best is None or r.cost < best.cost: best = r
    x = best.x
    return np.exp(x[0]) * R.from_rotvec(x[1:4]).as_matrix()[:2], x[4:6]     # linear (2,3), t
def aff_fit(P, Q):
    A = np.c_[P, np.ones(len(P))]; M, *_ = np.linalg.lstsq(A, Q, rcond=None)
    return M[:3].T, M[3]

sk = SMPLXSkeleton(model_path="models")
cams, data = {}, {}
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I, Pw, V = d["image"] * np.array([W, H, W]), d["world"].astype(np.float64), d["visibility"]
    data[f.stem] = (I, Pw, V)
    for t in range(len(I)):
        ok = np.isfinite(Pw[t]).all(1) & (V[t] > 0.5)
        idx = [i for i in UPPER if ok[i]]
        if len(idx) < 6: continue
        cams[(f.stem, t)] = {"wp": wp_fit(Pw[t, idx], I[t, idx, :2]), "aff": aff_fit(Pw[t, idx], I[t, idx, :2])}

SEGS = [(13, 15), (14, 16), (11, 13), (12, 14)]
print("delta transfer on natural motion (segment vectors), errors in % image shoulder width")
for cam in ("wp", "aff"):
    for k in (1, 3, 5, 10):
        err, base, rel = [], [], []
        for (c, t), cm in cams.items():
            I, Pw, V = data[c]
            if t + k >= len(I) or not np.isfinite(Pw[t+k]).all(): continue
            Lin, _ = cm[cam]
            sw = np.linalg.norm(I[t, 11, :2] - I[t, 12, :2])
            for a, b in SEGS:
                if not ((V[t, [a, b]] > 0.5).all() and (V[t+k, [a, b]] > 0.5).all()): continue
                vi_t, vi_k = I[t, b, :2] - I[t, a, :2], I[t+k, b, :2] - I[t+k, a, :2]
                dw = (Pw[t+k, b] - Pw[t+k, a]) - (Pw[t, b] - Pw[t, a])
                m = np.linalg.norm(vi_k - vi_t)
                if m / sw * 100 < 2: continue
                e = np.linalg.norm(vi_t + Lin @ dw - vi_k)
                err.append(e / sw * 100); base.append(m / sw * 100); rel.append(e / m)
        err, base, rel = map(np.array, (err, base, rel))
        print(f"  [{cam:3s}] k={k:2d} n={len(err):4d}: true displacement median {np.median(base):5.1f}%SW | "
              f"transfer error median {np.median(err):4.1f} p95 {np.percentile(err,95):5.1f}%SW | "
              f"relative median {np.median(rel):.2f} p95 {np.percentile(rel,95):.2f}")

# Full replacement at zero augmentation: camera(reconstruct_body(fit_body(world))) vs original image
full_wp, full_aff = [], []
MP = [m for _, m in ARM_CORRESPONDENCE]
for (c, t), cm in cams.items():
    I, Pw, V = data[c]
    ok = (V[t] > 0.5) & np.isfinite(Pw[t]).all(1)
    try:
        out = reconstruct_body(fit_body(Pw[t], ok, sk), sk)
    except ValueError:
        continue
    sw = np.linalg.norm(I[t, 11, :2] - I[t, 12, :2])
    for name, store in (("wp", full_wp), ("aff", full_aff)):
        Lin, tr = cm[name]
        for m in MP:
            if ok[m]:
                store.append(np.linalg.norm(Lin @ out[m] + tr - I[t, m, :2]) / sw * 100)
print("\nzero-augmentation image error, FULL REPLACEMENT (project reconstructed world): "
      f"wp median {np.median(full_wp):.2f} p95 {np.percentile(full_wp,95):.2f}%SW; "
      f"affine median {np.median(full_aff):.2f} p95 {np.percentile(full_aff,95):.2f}%SW")
print("zero-augmentation image error, DELTA TRANSFER: 0 by construction (delta = recon(theta') - recon(theta) = 0)")
