"""T5: (1) explain the float64 FK residual (smplx batch_rodrigues adds 1e-8 before the norm);
(2) size of the betas-dropping contract gap for realistic shapes."""
import types

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES

torch.set_grad_enabled(False)
rng = np.random.default_rng(99)

# (1) float64 model, FK with and without emulating smplx's rodrigues epsilon
m64 = smplx.create(model_path="models", model_type="smplx", gender="neutral", use_pca=False,
                   flat_hand_mean=True, batch_size=1, dtype=torch.float64)


def emulate_eps(r):
    ang = np.linalg.norm(r + 1e-8, axis=1, keepdims=True)
    return r / ang * ang  # axis r/ang, angle ang -> rotvec = r/ang*ang... keep explicit below


worst_plain = worst_eps = 0.0
for _ in range(20):
    betas = rng.normal(0, 2, 10)
    v = m64.v_template + torch.einsum("l,mkl->mk", torch.tensor(betas), m64.shapedirs[:, :, :10])
    sk = types.SimpleNamespace(rest_joints=(m64.J_regressor @ v)[:55].numpy(),
                               parents=m64.parents[:55].numpy().astype(int), num_joints=55)
    rots = rng.normal(0, 0.6, (55, 3)); rots[0] = rng.normal(0, 1.2, 3)
    transl = rng.normal(0, 1, 3)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)[None]
    ref = m64(betas=T(betas), global_orient=T(rots[0]), body_pose=T(rots[1:22].reshape(-1)),
              jaw_pose=T(rots[22]), leye_pose=T(rots[23]), reye_pose=T(rots[24]),
              left_hand_pose=T(rots[25:40].reshape(-1)), right_hand_pose=T(rots[40:55].reshape(-1)),
              transl=T(transl), expression=torch.zeros(1, 10, dtype=torch.float64),
              return_verts=False).joints[0, :55].numpy()
    worst_plain = max(worst_plain, np.abs(forward_kinematics(rots, sk) + transl - ref).max())
    # smplx: angle = |r + 1e-8|, axis = r / angle  ->  equivalent rotvec = axis * angle'... build matrices directly
    ang = np.linalg.norm(rots + 1e-8, axis=1, keepdims=True)
    axis = rots / ang
    K = np.zeros((55, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -axis[:, 2], axis[:, 1], axis[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -axis[:, 0], -axis[:, 1], axis[:, 0]
    s, c = np.sin(ang)[:, :, None], np.cos(ang)[:, :, None]
    Rm = np.eye(3)[None] + s * K + (1 - c) * K @ K
    rots_eps = Rotation.from_matrix(Rm).as_rotvec()
    worst_eps = max(worst_eps, np.abs(forward_kinematics(rots_eps, sk) + transl - ref).max())
print("(1) float64 SMPL-X vs FK, 20 random full poses, nonzero betas/root/transl:")
print(f"    plain FK max abs err {worst_plain:.2e} m; FK with smplx's rodrigues +1e-8 emulated: {worst_eps:.2e} m")

# (2) betas dropped by fit_body: SMPL-X forward(fit) vs reconstruct_body(fit, shaped skeleton)
print("(2) shaped skeleton -> fit_body -> SMPL-X forward with fit.betas (as returned):")
for label, betas in (("test_converter.py TestBodyShape betas", np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0, 0, 0, 0, 0])),
                     ("betas ~ N(0,1)", rng.normal(0, 1, 10)),
                     ("betas ~ N(0,2)", rng.normal(0, 2, 10))):
    sk = SMPLXSkeleton(model_path="models", betas=betas)
    worst_ret = worst_true = 0.0
    for _ in range(30):
        rots = np.zeros((55, 3))
        for j, *_ in SMPLX_ARM_BONES:
            rots[j] = rng.normal(0, 0.4, 3)
        J = forward_kinematics(rots, sk) + rng.normal(0, 1, 3)
        lm = np.full((33, 3), np.nan)
        for s_, m_ in ARM_CORRESPONDENCE:
            lm[m_] = J[s_]
        f = fit_body(lm, np.isfinite(lm).all(1), sk)
        ours = reconstruct_body(f, sk)
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
        for b, key in ((f.betas, "ret"), (betas, "true")):
            jj = sk.model(body_pose=t(f.body_pose), global_orient=t(f.global_orient), transl=t(f.transl),
                          betas=t(b), return_verts=False).joints[0].numpy()
            e = max(np.linalg.norm(ours[mp] - jj[sx]) for sx, mp in ARM_CORRESPONDENCE)
            if key == "ret":
                worst_ret = max(worst_ret, e)
            else:
                worst_true = max(worst_true, e)
    print(f"    {label:40s} |betas|={np.linalg.norm(betas):.2f}: with fit.betas (zeros) max {worst_ret*1000:6.2f} mm; "
          f"with the skeleton's betas {worst_true:.1e} m")
