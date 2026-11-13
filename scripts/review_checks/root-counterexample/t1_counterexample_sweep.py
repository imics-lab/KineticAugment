"""Stress the reviewer's root counterexample: axis, angle, pose, alignment type, converter path."""
import numpy as np
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE

sk = SMPLXSkeleton(model_path="models")
CORR = ARM_CORRESPONDENCE

def lms_from(truth):
    J = forward_kinematics(truth, sk)
    L = np.full((33, 3), np.nan)
    for j, m in CORR: L[m] = J[j]
    return L

def errs(truth):
    L = lms_from(truth)
    fit = solve_swing_ik(L, sk, SMPLX_ARM_BONES)
    F = forward_kinematics(fit, sk)
    src = np.array([F[j] for j, _ in CORR]); tgt = np.array([L[m] for _, m in CORR])
    raw = np.linalg.norm(src - tgt, axis=1)
    s, R, t = similarity_transform(src, tgt); sim = np.linalg.norm(apply_similarity(src, s, R, t) - tgt, axis=1)
    s2, R2, t2 = similarity_transform(src, tgt, allow_scale=False); rig = np.linalg.norm(apply_similarity(src, 1.0, R2, t2) - tgt, axis=1)
    tr = np.linalg.norm(src + (tgt - src).mean(0) - tgt, axis=1)  # translation only
    v = np.isfinite(L).all(1)
    conv = reconstruct_body(fit_body(L, v, sk), sk)
    cv = np.array([np.linalg.norm(conv[m] - L[m]) for _, m in CORR])
    return dict(raw=raw, sim=sim, rig=rig, tr=tr, conv=cv, scale=s)

print("== Reviewer's exact case: T-pose truth, root rotvec [0,0,0.7] ==")
t = np.zeros((55, 3)); t[0] = [0, 0, 0.7]
e = errs(t)
for k in ("raw", "sim", "rig", "tr", "conv"):
    print(f"  {k:5s} max={e[k].max()*1000:8.2f} mm mean={e[k].mean()*1000:8.2f} mm")
print(f"  similarity scale fitted = {e['scale']:.4f}")

print("\n== Rest shoulder geometry ==")
rest = sk.rest_joints
print("  L shoulder(16)", rest[16], " R shoulder(17)", rest[17], " pelvis(0)", rest[0])
sv = rest[16] - rest[17]; print("  rest L-R shoulder vector", sv, " angle from +x axis (deg)", np.degrees(np.arccos(sv[0] / np.linalg.norm(sv))))
print("  rest shoulder width (m)", np.linalg.norm(sv))
chain = []; j = 16
while j >= 0: chain.append(int(j)); j = sk.parents[j]
print("  ancestors of L shoulder (16):", chain)

print("\n== Axis x angle sweep, T-pose arms (mm, converter=rigid folded into global_orient) ==")
print(f"  {'axis':4s}{'deg':>5s}{'raw max':>10s}{'sim mean':>10s}{'sim max':>9s}{'conv mean':>10s}{'conv max':>9s}")
for ax_name, ax in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
    for deg in (10, 20, 30, 40.1, 45):
        t = np.zeros((55, 3)); t[0] = np.array(ax) * np.radians(deg)
        e = errs(t)
        print(f"  {ax_name:4s}{deg:5.1f}{e['raw'].max()*1e3:10.2f}{e['sim'].mean()*1e3:10.2f}{e['sim'].max()*1e3:9.2f}{e['conv'].mean()*1e3:10.2f}{e['conv'].max()*1e3:9.2f}")

print("\n== Random arm poses (N(0,0.4) per arm joint, as harness), random root axis, 200 trials ==")
print("  per-trial mean over 6 joints; then mean over trials / worst single joint over all trials")
for deg in (10, 20, 30, 45):
    rng = np.random.default_rng(42)
    simm, simx, cm, cx = [], 0, [], 0
    for _ in range(200):
        tr_ = np.zeros((55, 3)); a = rng.normal(size=3); tr_[0] = a / np.linalg.norm(a) * np.radians(deg)
        for jj, *_ in SMPLX_ARM_BONES: tr_[jj] = rng.normal(0, 0.4, 3)
        e = errs(tr_)
        simm.append(e['sim'].mean()); simx = max(simx, e['sim'].max()); cm.append(e['conv'].mean()); cx = max(cx, e['conv'].max())
    print(f"  {deg:3d} deg: similarity mean {np.mean(simm)*1e3:7.2f} mm worst {simx*1e3:7.2f} | rigid-converter mean {np.mean(cm)*1e3:7.2f} mm worst {cx*1e3:7.2f}")

print("\n== Per-axis on random arm poses (rigid converter), yaw (y) vs roll (z) vs pitch (x) ==")
for ax_name, ax in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
    for deg in (10, 30):
        rng = np.random.default_rng(7); cm = []; cx = 0
        for _ in range(200):
            tr_ = np.zeros((55, 3)); tr_[0] = np.array(ax) * np.radians(deg)
            for jj, *_ in SMPLX_ARM_BONES: tr_[jj] = rng.normal(0, 0.4, 3)
            e = errs(tr_); cm.append(e['conv'].mean()); cx = max(cx, e['conv'].max())
        print(f"  axis {ax_name} {deg:2d} deg: mean {np.mean(cm)*1e3:7.2f} mm, worst {cx*1e3:7.2f} mm")
