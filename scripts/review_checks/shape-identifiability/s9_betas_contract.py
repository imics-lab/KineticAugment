"""New-issue check: if Step 10 hands fit_body a shaped skeleton, are the returned
BodyFit parameters still 'genuine SMPL-X parameters' for the real forward pass?"""
import os
import numpy as np, torch
from common import SNAP
os.chdir(SNAP)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics

betas = np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0, 0, 0, 0, 0])
shaped = SMPLXSkeleton(model_path="models", betas=betas)
print("SMPLXSkeleton stores betas attribute:", hasattr(shaped, "betas"))
rng = np.random.default_rng(3)
rots = np.zeros((55, 3))
for j, *_ in SMPLX_ARM_BONES: rots[j] = rng.normal(0, 0.3, 3)
posed = forward_kinematics(rots, shaped) + np.array([-0.4, 0.2, 1.5])
lm = np.full((33, 3), np.nan)
for s, m in ARM_CORRESPONDENCE: lm[m] = posed[s]
ok = np.isfinite(lm).all(1)
fit = fit_body(lm, ok, shaped)
ours = reconstruct_body(fit, shaped)
t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
with torch.no_grad():
    J = shaped.model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient), transl=t(fit.transl),
                     betas=t(fit.betas), return_verts=False).joints[0].numpy()
    J2 = shaped.model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient), transl=t(fit.transl),
                      betas=t(betas), return_verts=False).joints[0].numpy()
print("fit.betas returned:", fit.betas)
e_ret = max(np.linalg.norm(ours[m] - J[s]) for s, m in ARM_CORRESPONDENCE)
e_true = max(np.linalg.norm(ours[m] - J2[s]) for s, m in ARM_CORRESPONDENCE)
print(f"reconstruct_body vs real SMPL-X forward(using fit.betas):   max {e_ret*1000:.2f} mm")
print(f"reconstruct_body vs real SMPL-X forward(using skeleton's betas): max {e_true*1000:.4f} mm")
