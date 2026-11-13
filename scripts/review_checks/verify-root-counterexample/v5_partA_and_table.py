"""Reproduce Part A's 70 mm with the script's own RNG stream, report mean too;
try to reproduce the doc's retracted root table (10/20/45 deg -> 9.87/81, 16.55/119, 47.04/307 mm);
evaluate the old strict-xfail case (seed 4, 30 deg about y) with the shipped converter."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE as CORR, fit_body, reconstruct_body

SK = SMPLXSkeleton(model_path="models")


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


def errs(truth):
    J = forward_kinematics(truth, SK)
    lm = np.full((33, 3), np.nan)
    for s, m in CORR:
        lm[m] = J[s]
    ok = np.isfinite(lm).all(1)
    tgt = np.array([lm[m] for _, m in CORR])
    out = reconstruct_body(fit_body(lm, ok, SK), SK)
    conv = np.linalg.norm(np.array([out[m] for _, m in CORR]) - tgt, axis=1)
    fj = forward_kinematics(solve_swing_ik(np.where(ok[:, None], lm, np.nan), SK, SMPLX_ARM_BONES), SK)
    src = np.array([fj[s] for s, _ in CORR])
    raw = np.linalg.norm(src - tgt, axis=1)
    sim = np.linalg.norm(umeyama(src, tgt) - tgt, axis=1)
    return conv, raw, sim


for deg in (10, 20, 30, 45):
    rng = np.random.default_rng(42)          # identical stream to measure_round_trip.run_synthetic
    C, Rw, S = [], [], []
    for _ in range(200):
        truth = np.zeros((55, 3))
        axis = rng.normal(size=3)
        truth[0] = axis / np.linalg.norm(axis) * np.deg2rad(deg)
        for joint, _, _, _ in SMPLX_ARM_BONES:
            truth[joint] = rng.normal(0, 0.4, 3)
        c, r, s = errs(truth)
        C.append(c); Rw.append(r); S.append(s)
    C, Rw, S = map(np.array, (C, Rw, S))
    f = lambda a: f"mean {a.mean()*1e3:6.2f} max {a.max()*1e3:6.1f} mean-of-trial-max {a.max(1).mean()*1e3:6.2f}"
    print(f"{deg:2d} deg seed42 | converter {f(C)} | raw {f(Rw)} | sim {f(S)}")
print("doc table: 10 -> 9.87/81 ; 20 -> 16.55/119 ; 45 -> 47.04/307 (mean/max mm)")

rng = np.random.default_rng(4)
truth = np.zeros((55, 3))
for joint, _, _, _ in SMPLX_ARM_BONES:
    truth[joint] = rng.normal(0, 0.3, 3)
truth[0] = R.from_euler("y", 30, degrees=True).as_rotvec()
c, r, s = errs(truth)
print(f"old strict-xfail case (seed 4, y 30): converter max {c.max()*1e3:.2f} mean {c.mean()*1e3:.2f} mm")
