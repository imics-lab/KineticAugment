"""After a shoulder-line root: translation-only vs an extra rigid post-fit.
Real-data direct error, and whether rigid breaks exact directions on a differently shaped body."""
import glob
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

CORR = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]
SK = SMPLXSkeleton(model_path="models")
REST = SK.rest_joints


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


def fit(lm, ok, sk=SK):
    rots = solve_swing_ik(np.where(ok[:, None], lm, np.nan), sk, SMPLX_ARM_BONES, global_orient=sl_root(lm))
    J = forward_kinematics(rots, sk)
    pres = [(s, m) for s, m in CORR if ok[m]]
    src = np.array([J[s] for s, _ in pres]); tgt = np.array([lm[m] for _, m in pres])
    return src + (tgt - src).mean(0), umeyama(src, tgt, False), tgt, pres


tr_all, rg_all, per = [], [], {}
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); P = d["landmarks"].astype(float); V = d["visibility"]
    OK = np.isfinite(P).all(-1) & (V >= 0.5)
    both = OK[:, 11] & OK[:, 12]
    W = float(np.median(np.linalg.norm(P[both, 11] - P[both, 12], axis=-1)))
    t_c, r_c = [], []
    for lm, ok in zip(P, OK):
        if not (ok[11] and ok[12]):
            continue
        a, b, tgt, _ = fit(lm, ok)
        t_c += list(np.linalg.norm(a - tgt, axis=1) / W); r_c += list(np.linalg.norm(b - tgt, axis=1) / W)
    per[Path(path).stem[:5]] = (np.median(t_c) * 100, np.median(r_c) * 100)
    tr_all += t_c; rg_all += r_c
print("pooled direct median: translation-only %.2f%%  rigid post-fit %.2f%%" % (np.median(tr_all) * 100, np.median(rg_all) * 100))
print("per clip (transl, rigid):", {k: tuple(round(x, 2) for x in v) for k, v in per.items()})

# synthetic: shaped truth, neutral fit -> are bone + shoulder-line directions exact?
shaped = SMPLXSkeleton(model_path="models", betas=np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0, 0, 0, 0, 0]))
rng = np.random.default_rng(8)
truth = np.zeros((55, 3))
for j, *_ in SMPLX_ARM_BONES:
    truth[j] = rng.normal(0, 0.3, 3)
truth[0] = R.from_euler("y", 25, degrees=True).as_rotvec()
J = forward_kinematics(truth, shaped)
lm = np.full((33, 3), np.nan)
for s, m in CORR:
    lm[m] = J[s]
ok = np.isfinite(lm).all(1)
a, b, tgt, pres = fit(lm, ok)
idx = {m: i for i, (_, m) in enumerate(pres)}
for name, out in (("translation", a), ("rigid", b)):
    worst = 0
    for p, c in [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]:
        u = out[idx[c]] - out[idx[p]]; v = tgt[idx[c]] - tgt[idx[p]]
        worst = max(worst, np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
    print(f"shaped synthetic, {name:11s}: worst segment direction error {worst:.2e} deg, worst pos {np.linalg.norm(out-tgt,axis=1).max()*1e3:.2f} mm")
