"""Is BodyFit self-describing when the skeleton carries a non-default shape?"""
import numpy as np, torch
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.kinematics import forward_kinematics
b = np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0, 0, 0, 0, 0])
shaped = SMPLXSkeleton(model_path="models", betas=b)
rng = np.random.default_rng(1)
rots = np.zeros((55, 3)); rots[0] = [0.2, -0.3, 0.1]
for j in (16, 17, 18, 19): rots[j] = rng.normal(0, 0.4, 3)
J = forward_kinematics(rots, shaped)
lm = np.full((33, 3), np.nan)
for s, m in ARM_CORRESPONDENCE: lm[m] = J[s]
fit = fit_body(lm, np.isfinite(lm).all(1), shaped)
rec = reconstruct_body(fit, shaped)
print("BodyFit.betas returned with a shaped skeleton:", fit.betas)
print("round trip through shaped skeleton, worst arm error (m): %.2e" % max(np.linalg.norm(rec[m]-lm[m]) for _, m in ARM_CORRESPONDENCE))
t = lambda a: torch.tensor(a, dtype=torch.float32)[None]
with torch.no_grad():
    jx = shaped.model(global_orient=t(fit.global_orient), body_pose=t(fit.body_pose), transl=t(fit.transl), betas=t(fit.betas), return_verts=False).joints[0].numpy()
    jx2 = shaped.model(global_orient=t(fit.global_orient), body_pose=t(fit.body_pose), transl=t(fit.transl), betas=t(b), return_verts=False).joints[0].numpy()
print("real SMPL-X forward on the serialized BodyFit vs reconstruct_body, worst (m): %.3e" % max(np.linalg.norm(jx[s]-rec[m]) for s, m in ARM_CORRESPONDENCE))
print("same but with the skeleton's true betas substituted, worst (m): %.3e" % max(np.linalg.norm(jx2[s]-rec[m]) for s, m in ARM_CORRESPONDENCE))
