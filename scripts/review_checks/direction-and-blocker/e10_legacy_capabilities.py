"""E10: are the six retired legacy tests a 'rejected design' or the adopted contract?
Checks the capabilities they encode against the new converter:
  (a) model-free fitting (SimpleFitter)       -> fit_body with SyntheticSkeleton
  (b) reconstruct from parameters alone       -> reconstruct_body(fit, skeleton) signature
  (c) genuine-parameter output shapes         -> body_pose (63,), global_orient (3,)
Also: does reconstruct_body honour BodyFit.betas (needed by any shape augmentation)?"""
import inspect, warnings
import numpy as np, torch
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SyntheticSkeleton, SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, BodyFit, ARM_CORRESPONDENCE

L = np.load("data/test_output/01986_world.npz")
P, V = L["landmarks"][10].astype(float), L["visibility"][10] > 0.5
try:
    f = fit_body(P, V, SyntheticSkeleton()); print("(a) model-free fit OK", f.body_pose.shape)
except Exception as e:
    print(f"(a) model-free fit with SyntheticSkeleton (num_joints={SyntheticSkeleton().num_joints}): {type(e).__name__}: {e}")

print("(b) reconstruct_body params:", list(inspect.signature(reconstruct_body).parameters))
sk = SMPLXSkeleton(model_path="models")
f = fit_body(P, V, sk)
print("(c) shapes:", f.body_pose.shape, f.global_orient.shape, f.transl.shape, f.betas.shape)
betas = np.array([2.0, -1.5, 1.0, 0, 0, 0, 0, 0, 0, 0])
g = BodyFit(f.body_pose, f.global_orient, f.transl, betas)
ours0, ours1 = reconstruct_body(f, sk), reconstruct_body(g, sk)
t32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
with torch.no_grad():
    J = sk.model(body_pose=t32(g.body_pose), global_orient=t32(g.global_orient), transl=t32(g.transl),
                 betas=t32(betas), return_verts=False).joints[0].numpy()
d_ours = max(np.linalg.norm(ours1[m] - ours0[m]) for _, m in ARM_CORRESPONDENCE)
d_real = max(np.linalg.norm(ours1[m] - J[s]) for s, m in ARM_CORRESPONDENCE)
print(f"betas changed -> reconstruct_body output moved by {d_ours:.2e} m; "
      f"real SMPL-X forward with those betas differs from reconstruct_body by up to {d_real*1000:.1f} mm")
