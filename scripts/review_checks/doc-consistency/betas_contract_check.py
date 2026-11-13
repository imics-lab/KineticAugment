"""Does BodyFit really carry everything needed to reconstruct? (doc 819-831: 'Reconstruction takes
only the parameters', 'The output is pure SMPL-X', 'Verified against the real model ... 1e-5 m').

C1  reconstruct_body ignores fit.betas: change betas, output unchanged?
C2  fit_body on a non-neutral skeleton returns betas=0, so SMPL-X forward on the returned
    parameters disagrees with reconstruct_body(fit, shaped_skeleton).
C3  Real clips: SMPL-X forward on fit_body params vs reconstruct_body (neutral skeleton).
"""
import glob

import numpy as np
import torch

from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES

sk = SMPLXSkeleton(model_path="models")
betas = np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0, 0, 0, 0, 0])
shaped = SMPLXSkeleton(model_path="models", betas=betas)


def smplx_joints(fit, model):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
    with torch.no_grad():
        return model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient),
                     transl=t(fit.transl), betas=t(fit.betas), return_verts=False).joints[0].numpy()


rng = np.random.default_rng(0)
truth = np.zeros((55, 3))
for j, *_ in SMPLX_ARM_BONES:
    truth[j] = rng.normal(0, 0.3, 3)
truth[0] = [0.1, 0.4, -0.05]
J = forward_kinematics(truth, shaped) + np.array([0.2, -0.3, 1.1])
L = np.full((33, 3), np.nan)
for s, m in ARM_CORRESPONDENCE:
    L[m] = J[s]
ok = np.isfinite(L).all(1)

fit = fit_body(L, ok, shaped)
print("C2 fit_body(shaped skeleton) returned betas:", fit.betas)
ours = reconstruct_body(fit, shaped)
theirs = smplx_joints(fit, sk.model)
gap = max(np.linalg.norm(ours[m] - theirs[s]) for s, m in ARM_CORRESPONDENCE)
print(f"C2 SMPL-X forward on returned params vs reconstruct_body(fit, shaped): max gap {gap*1000:.1f} mm")
fit_b = type(fit)(fit.body_pose, fit.global_orient, fit.transl, betas)
theirs_b = smplx_joints(fit_b, sk.model)
gap_b = max(np.linalg.norm(ours[m] - theirs_b[s]) for s, m in ARM_CORRESPONDENCE)
print(f"C2 ... if BodyFit had carried the true betas: max gap {gap_b*1000:.4f} mm")

fit0 = fit_body(L, ok, sk)
out0 = reconstruct_body(fit0, sk)
fit0b = type(fit0)(fit0.body_pose, fit0.global_orient, fit0.transl, rng.normal(0, 2, 10))
out0b = reconstruct_body(fit0b, sk)
print(f"C1 reconstruct_body with betas changed to random: max output change "
      f"{np.nanmax(np.abs(out0b - out0)):.3e} m (SMPL-X forward would move joints by "
      f"{max(np.linalg.norm(smplx_joints(fit0b, sk.model)[s] - smplx_joints(fit0, sk.model)[s]) for s, _ in ARM_CORRESPONDENCE)*1000:.1f} mm)")

worst = 0.0
n = 0
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f)
    P = d["landmarks"].astype(float)
    okk = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    for frame, o in zip(P, okk):
        if sum(o[m] for _, m in ARM_CORRESPONDENCE) < 3:
            continue
        fr = fit_body(frame, o, sk)
        a = reconstruct_body(fr, sk)
        b = smplx_joints(fr, sk.model)
        worst = max(worst, max(np.linalg.norm(a[m] - b[s]) for s, m in ARM_CORRESPONDENCE))
        n += 1
print(f"C3 real clips ({n} frames): SMPL-X forward vs reconstruct_body worst gap {worst:.2e} m (doc: within 1e-5 m)")
