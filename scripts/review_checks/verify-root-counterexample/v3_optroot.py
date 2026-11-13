"""Is the doc's 'root orientation only' row (5.13%) an oracle? Optimise the root.

For every frame: root rotvec (3 dof) optimised with arm IK re-solved under it.
  objective D: sum of squared direct residuals with optimal translation
  objective A: sum of squared residuals after free similarity (the aligned metric)
Pooled median of per-landmark error (% of clip median shoulder width), as in the harness.
"""
import glob
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

CORR = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]
SK = SMPLXSkeleton(model_path="models")
REST = SK.rest_joints
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def umeyama(src, tgt, scale=True):
    ms, mt = src.mean(0), tgt.mean(0)
    a, b = src - ms, tgt - mt
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean() if scale else 1.0
    return s * src @ Rm.T + (mt - s * Rm @ ms)


def sl_root(lm):
    a = REST[16] - REST[17]; a /= np.linalg.norm(a)
    b = lm[11] - lm[12]; b /= np.linalg.norm(b)
    ax = np.cross(a, b); s = np.linalg.norm(ax)
    return ax / s * np.arctan2(s, a @ b) if s > 1e-12 else np.zeros(3)


def residuals(rv, masked, pres, tgt, mode):
    rots = solve_swing_ik(masked, SK, SMPLX_ARM_BONES, global_orient=rv)
    J = forward_kinematics(rots, SK)
    src = np.array([J[s] for s, _ in pres])
    if mode == "D":
        out = src + (tgt - src).mean(0)
    else:
        out = umeyama(src, tgt)
    return np.linalg.norm(out - tgt, axis=1)


res = {"sl_D": [], "opt_D": [], "sl_A": [], "opt_A": []}
nfr = 0
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path)
    P = d["landmarks"].astype(np.float64); V = d["visibility"]
    OK = np.isfinite(P).all(-1) & (V >= 0.5)
    both = OK[:, 11] & OK[:, 12]
    W = float(np.median(np.linalg.norm(P[both, 11] - P[both, 12], axis=-1)))
    for i in range(0, len(P), STEP):
        lm, ok = P[i], OK[i]
        pres = [(s, m) for s, m in CORR if ok[m]]
        if len(pres) < 3 or not (ok[11] and ok[12]):
            continue
        nfr += 1
        masked = np.where(ok[:, None], lm, np.nan)
        tgt = np.array([lm[m] for _, m in pres])
        x0 = sl_root(lm)
        for mode in ("D", "A"):
            e0 = residuals(x0, masked, pres, tgt, mode)
            f = lambda x: float((residuals(x, masked, pres, tgt, mode) ** 2).sum())
            best = minimize(f, x0, method="Powell", options=dict(xtol=1e-6, ftol=1e-12, maxfev=3000))
            e1 = residuals(best.x, masked, pres, tgt, mode)
            if (e1 ** 2).sum() > (e0 ** 2).sum():
                e1 = e0
            res["sl_" + mode] += list(e0 / W); res["opt_" + mode] += list(e1 / W)
print(f"frames used: {nfr} (every {STEP})")
for k, v in res.items():
    print(f"  {k:6s} pooled median {np.median(v)*100:5.2f}%   mean {np.mean(v)*100:5.2f}%  n={len(v)}")
