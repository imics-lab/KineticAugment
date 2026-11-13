"""Step 7 contract claims about converter.py, tested independently.

C1  SMPL-X forward on returned params reproduces reconstruct_body within 1e-5 m
C2  rigid frame transform folds exactly into global_orient/transl (root pivot compensated)
C3  reconstruct takes only (fit, skeleton) -- and does the fit carry everything?
C4  NaN where not modelled
C5  <3 valid correspondences raises
"""
import dataclasses
import glob
import inspect

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.alignment import similarity_transform
from kinetic_augment.body_model.converter import (ARM_CORRESPONDENCE, BodyFit, fit_body,
                                                  reconstruct_body)
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

torch.set_grad_enabled(False)
rng = np.random.default_rng(7)
ARM_MP = [m for _, m in ARM_CORRESPONDENCE]


def smplx_joints(skel, fit, betas=None):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
    b = fit.betas if betas is None else betas
    return skel.model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient),
                      transl=t(fit.transl), betas=t(b), return_verts=False).joints[0].numpy().astype(np.float64)


def fwd_vs_recon(skel, fit, betas=None):
    j = smplx_joints(skel, fit, betas)
    r = reconstruct_body(fit, skel)
    return max(np.linalg.norm(r[m] - j[s]) for s, m in ARM_CORRESPONDENCE)


def synth(skel, root_rotvec=np.zeros(3), R=np.eye(3), t=np.zeros(3), sig=0.4):
    rots = np.zeros((55, 3))
    rots[0] = root_rotvec
    for jnt, *_ in SMPLX_ARM_BONES:
        rots[jnt] = rng.normal(0, sig, 3)
    joints = forward_kinematics(rots, skel) @ R.T + t
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lm[m] = joints[s]
    return lm, np.isfinite(lm).all(1)


skel0 = SMPLXSkeleton(model_path="models")
betas_b = rng.normal(0, 2.0, 10)
skelb = SMPLXSkeleton(model_path="models", betas=betas_b)

print("=== C1: SMPL-X forward(fit params) vs reconstruct_body(fit, skeleton) ===")
cases = {}
# a) the existing test's regime: betas 0, unrotated, translated
cases["betas=0, root 0, translated (test regime)"] = max(
    fwd_vs_recon(skel0, fit_body(*synth(skel0, t=rng.normal(0, 1, 3)), skel0)) for _ in range(50))
# b) arbitrary rigid rotation of the whole landmark set -> nontrivial global_orient
go_angles = []
worst = 0.0
for _ in range(50):
    R = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
    f = fit_body(*synth(skel0, R=R, t=rng.normal(0, 1, 3)), skel0)
    go_angles.append(np.degrees(np.linalg.norm(f.global_orient)))
    worst = max(worst, fwd_vs_recon(skel0, f))
cases["betas=0, landmark frame randomly rotated (global_orient up to %.0f deg)" % max(go_angles)] = worst
# c) synthetic root rotated (xfail regime) -- forward should still agree with reconstruct
cases["betas=0, synthetic root rotated 0.7 rad"] = max(
    fwd_vs_recon(skel0, fit_body(*synth(skel0, root_rotvec=Rotation.random(random_state=k).as_rotvec() * 0 + rng.normal(0, 1, 3) / np.sqrt(3) * 0.7), skel0))
    for k in range(50))
# d) real clips, every frame
real_worst, real_n, real_go = 0.0, 0, []
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path)
    pose = d["landmarks"].astype(np.float64)
    ok_all = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
    for fr, ok in zip(pose, ok_all):
        try:
            f = fit_body(fr, ok, skel0)
        except ValueError:
            continue
        real_n += 1
        real_go.append(np.degrees(np.linalg.norm(f.global_orient)))
        real_worst = max(real_worst, fwd_vs_recon(skel0, f))
cases[f"betas=0, REAL clips ({real_n} frames; |global_orient| median {np.median(real_go):.0f} deg)"] = real_worst
# e) nonzero-betas skeleton, using BodyFit as returned
worst_b_asreturned = max(fwd_vs_recon(skelb, fit_body(*synth(skelb, t=rng.normal(0, 1, 3)), skelb)) for _ in range(20))
cases["SHAPED skeleton (|betas|=%.1f), BodyFit used as returned" % np.linalg.norm(betas_b)] = worst_b_asreturned
worst_b_manual = max(fwd_vs_recon(skelb, fit_body(*synth(skelb, R=Rotation.random(random_state=k).as_matrix(), t=rng.normal(0, 1, 3)), skelb), betas=betas_b) for k in range(20))
cases["SHAPED skeleton, betas passed to SMPL-X MANUALLY, rotated frame"] = worst_b_manual
for k, v in cases.items():
    print(f"  {'PASS' if v < 1e-5 else 'FAIL'}  {v:.3e} m   {k}")
fb = fit_body(*synth(skelb), skelb)
print(f"  BodyFit.betas from shaped skeleton = {fb.betas.tolist()}  (skeleton betas norm {np.linalg.norm(betas_b):.2f})")

print("\n=== C2: rigid transform folds exactly into global_orient/transl ===")
worst_fold, worst_fold55, worst_naive = 0.0, 0.0, 0.0
for k in range(50):
    sk = skel0 if k % 2 == 0 else skelb
    R = Rotation.random(random_state=1000 + k).as_matrix()
    lm, ok = synth(sk, root_rotvec=rng.normal(0, 0.5, 3), R=R, t=rng.normal(0, 1, 3))
    # independent recomputation of what fit_body is supposed to encode
    rots = solve_swing_ik(np.where(ok[:, None], lm, np.nan), sk, SMPLX_ARM_BONES)
    posed = forward_kinematics(rots, sk)
    src = np.array([posed[s] for s, m in ARM_CORRESPONDENCE if ok[m]])
    tgt = np.array([lm[m] for s, m in ARM_CORRESPONDENCE if ok[m]])
    _, Rr, tr = similarity_transform(src, tgt, allow_scale=False)
    expected_all = posed @ Rr.T + tr                                 # 55 joints
    f = fit_body(lm, ok, sk)
    rr = np.zeros((55, 3)); rr[0] = f.global_orient; rr[1:22] = f.body_pose.reshape(21, 3)
    got_all = forward_kinematics(rr, sk) + f.transl
    worst_fold55 = max(worst_fold55, np.abs(got_all[:22] - expected_all[:22]).max())
    out = reconstruct_body(f, sk)
    worst_fold = max(worst_fold, max(np.linalg.norm(out[m] - expected_all[s]) for s, m in ARM_CORRESPONDENCE))
    # naive fold (no pivot compensation): transl = translation only
    naive = forward_kinematics(rr, sk) + tr
    worst_naive = max(worst_naive, np.abs(naive[:22] - expected_all[:22]).max())
print(f"  reconstruct_body vs independently computed R*FK+t (arm joints): max {worst_fold:.3e} m")
print(f"  FK(global_orient, body_pose)+transl vs R*FK+t over body joints 0..21: max {worst_fold55:.3e} m")
print(f"  control: same fold WITHOUT root-pivot compensation: max {worst_naive:.3e} m  "
      f"(|J0| = {np.linalg.norm(skel0.rest_joints[0]):.3f} m)")

print("\n=== C3: does reconstruction depend only on the parameters? ===")
print("  signature:", list(inspect.signature(reconstruct_body).parameters))
lm, ok = synth(skel0, t=np.array([0.1, 0.2, 0.3]))
f = fit_body(lm, ok, skel0)
f_beta = dataclasses.replace(f, betas=rng.normal(0, 3, 10))
d_betas = np.nanmax(np.abs(reconstruct_body(f_beta, skel0) - reconstruct_body(f, skel0)))
print(f"  changing fit.betas (random, sigma 3) changes reconstruct_body output by: {d_betas:.3e} m  "
      f"-> betas field is IGNORED by reconstruct_body")
d_skel = np.nanmax(np.linalg.norm(reconstruct_body(f, skelb) - reconstruct_body(f, skel0), axis=1))
print(f"  same fit, shaped vs default skeleton: outputs differ by up to {d_skel*1000:.1f} mm "
      f"-> shape lives in the skeleton argument, not in the params")
print("  BodyFit fields:", [x.name for x in dataclasses.fields(BodyFit)], "(no gender / model id / flat_hand_mean)")

print("\n=== C4: NaN where not modelled ===")
out = reconstruct_body(f, skel0)
nan_rows = [i for i in range(33) if np.isnan(out[i]).all()]
print(f"  NaN rows: {len(nan_rows)} of 33; non-NaN rows: {sorted(set(range(33)) - set(nan_rows))}")
lm2, ok2 = synth(skel0, t=np.array([0.1, 0.2, 0.3]))
ok2[15] = False  # left wrist unobserved
out2 = reconstruct_body(fit_body(lm2, ok2, skel0), skel0)
print(f"  left wrist INVALID in input -> reconstructed L wrist = {np.round(out2[15], 3).tolist()} "
      f"(not NaN; true was {np.round(lm2[15], 3).tolist()}, error {np.linalg.norm(out2[15]-lm2[15])*1000:.0f} mm)")
ok3 = ok2.copy(); ok3[13] = False; ok3[15] = False
out3 = reconstruct_body(fit_body(lm2, ok3, skel0), skel0)
print(f"  left elbow+wrist INVALID -> reconstructed L elbow/wrist present: {np.isfinite(out3[[13, 15]]).all()}; "
      f"BodyFit records which landmarks were observed: {'validity' in [x.name for x in dataclasses.fields(BodyFit)]}")

print("\n=== C5: <3 valid correspondences raises ===")
lm, ok = synth(skel0)
for n in range(0, 7):
    v = np.zeros(33, bool); v[ARM_MP[:n]] = True
    try:
        fit_body(lm, v, skel0); res = "ok"
    except ValueError:
        res = "raises ValueError"
    print(f"  {n} valid correspondences: {res}")
v = np.ones(33, bool)
lmn = lm.copy(); lmn[[11, 13, 15]] = np.nan
try:
    fit_body(lmn, v, skel0); print("  validity=True but 3 NaN landmarks: ok")
except ValueError:
    print("  validity=True but 3 NaN landmarks (3 finite left): raises? no -> see above")
# collinear: only the left arm valid and the arm perfectly straight
rots = np.zeros((55, 3)); rots[16] = [0, 0, -1.0]
J = forward_kinematics(rots, skel0)
lm = np.full((33, 3), np.nan)
for s, m in ARM_CORRESPONDENCE:
    lm[m] = J[s]
v = np.zeros(33, bool); v[[11, 13, 15]] = True
try:
    fb = fit_body(lm, v, skel0)
    out = reconstruct_body(fb, skel0)
    e = np.linalg.norm(out[[11, 13, 15]] - lm[[11, 13, 15]], axis=1)
    print(f"  3 valid but COLLINEAR (straight left arm only): does NOT raise; observed error max {e.max():.2e} m; "
          f"R shoulder (unobserved) placed {np.linalg.norm(out[12]-J[17])*1000:.0f} mm from truth")
except ValueError as ex:
    print("  collinear raises:", ex)
# perturbed straight arm: rotate the truth about the arm axis and show the fit cannot tell
axis = (J[20] - J[16]) / np.linalg.norm(J[20] - J[16])
diffs = []
for ang in (0.0, 0.5, 1.0, 2.0):
    Rax = Rotation.from_rotvec(axis * ang).as_matrix()
    Jr = (J - J[16]) @ Rax.T + J[16]
    lmr = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lmr[m] = Jr[s]
    o = reconstruct_body(fit_body(lmr, v, skel0), skel0)
    diffs.append((ang, np.linalg.norm(o[12] - Jr[17]) * 1000))
print("  straight-arm truth rotated about the arm axis -> R shoulder error (mm):",
      [(a, round(d, 1)) for a, d in diffs], "(frame undetermined by collinear points)")

# valid-but-unsolvable correspondence: wrist valid, elbow invalid
errs = []
for _ in range(50):
    lm, ok = synth(skel0, t=rng.normal(0, 1, 3))
    ok[13] = False
    o = reconstruct_body(fit_body(lm, ok, skel0), skel0)
    errs.append(max(np.linalg.norm(o[m] - lm[m]) for m in (11, 12, 14, 16)))
print(f"  L elbow invalid but L wrist valid: error on the FULLY observed joints (shoulders, R arm): "
      f"median {np.median(errs)*1000:.1f} mm, max {np.max(errs)*1000:.1f} mm (exact when all valid)")
