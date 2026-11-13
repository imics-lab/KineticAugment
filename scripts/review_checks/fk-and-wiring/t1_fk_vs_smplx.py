"""A001: FK (skeleton.py + kinematics.py) vs real SMPL-X forward.

Random full 55-joint poses, NONZERO betas, NONZERO global_orient, NONZERO transl.
Only SMPLX_NEUTRAL.npz exists locally, so gender coverage is neutral only.
Also: float64 control to show residual is float32 round-off; and a shape-blind
control (skeleton built with betas=0 against model run with betas=b) to show the
skeleton genuinely takes shape into account.
"""
import types
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics

torch.set_grad_enabled(False)
rng = np.random.default_rng(12345)


def run_model(model, rots, betas, transl, dtype):
    t = lambda a: torch.tensor(np.asarray(a), dtype=dtype)[None]
    out = model(
        betas=t(betas),
        global_orient=t(rots[0]),
        body_pose=t(rots[1:22].reshape(-1)),
        jaw_pose=t(rots[22]),
        leye_pose=t(rots[23]),
        reye_pose=t(rots[24]),
        left_hand_pose=t(rots[25:40].reshape(-1)),
        right_hand_pose=t(rots[40:55].reshape(-1)),
        transl=t(transl),
        expression=torch.zeros(1, 10, dtype=dtype),
        return_verts=False,
    )
    return out.joints[0].numpy().astype(np.float64)


def random_pose(sig_root, sig_body, sig_hand):
    rots = np.zeros((55, 3))
    rots[0] = rng.normal(0, sig_root, 3)
    rots[1:22] = rng.normal(0, sig_body, (21, 3))
    rots[22:25] = rng.normal(0, 0.2, (3, 3))
    rots[25:55] = rng.normal(0, sig_hand, (30, 3))
    return rots


print("=== A001 FK vs real SMPL-X forward (neutral; male/female weights not present) ===")
worst_all = 0.0
worst_body = 0.0
worst_shapeblind = 0.0
n_trials = 0
for b_i in range(6):
    betas = np.zeros(10) if b_i == 0 else rng.normal(0, 2.0, 10)
    skel = SMPLXSkeleton(model_path="models", betas=betas)
    skel0 = SMPLXSkeleton(model_path="models") if b_i > 0 else None
    for k in range(10):
        rots = random_pose(sig_root=1.2, sig_body=0.6, sig_hand=0.4)
        transl = rng.normal(0, 1.0, 3)
        ref = run_model(skel.model, rots, betas, transl, torch.float32)
        assert ref.shape == (127, 3), ref.shape
        fk = forward_kinematics(rots, skel) + transl
        e = np.linalg.norm(fk - ref[:55], axis=1)
        worst_all = max(worst_all, e.max())
        worst_body = max(worst_body, e[:22].max())
        if skel0 is not None:
            fk0 = forward_kinematics(rots, skel0) + transl
            worst_shapeblind = max(worst_shapeblind, np.linalg.norm(fk0 - ref[:55], axis=1).max())
        n_trials += 1
    print(f"  betas set {b_i}: |betas|={np.linalg.norm(betas):.2f}  running max err (55 joints) = {worst_all:.3e} m")

print(f"\n  trials={n_trials}  (6 beta vectors x 10 poses; root sigma 1.2 rad, transl sigma 1 m)")
print(f"  MAX |FK - SMPLX.joints[:55]| over all 55 joints : {worst_all:.3e} m")
print(f"  MAX over body joints 0..21                       : {worst_body:.3e} m")
print(f"  shape-blind control (skeleton betas=0, model betas=b): max err {worst_shapeblind:.3e} m")

# float64 control: build a float64 model and a skeleton whose rest joints come
# from the float64 template, to show the float32 residual is pure round-off.
m64 = smplx.create(model_path="models", model_type="smplx", gender="neutral", use_pca=False,
                   flat_hand_mean=True, batch_size=1, dtype=torch.float64)
worst64 = 0.0
for k in range(10):
    betas = rng.normal(0, 2.0, 10)
    v = m64.v_template + torch.einsum("l,mkl->mk", torch.tensor(betas), m64.shapedirs[:, :, :10])
    rest = (m64.J_regressor @ v)[:55].numpy()
    sk = types.SimpleNamespace(rest_joints=rest, parents=m64.parents[:55].numpy().astype(int), num_joints=55)
    rots = random_pose(1.2, 0.6, 0.4)
    transl = rng.normal(0, 1.0, 3)
    ref = run_model(m64, rots, betas, transl, torch.float64)
    worst64 = max(worst64, np.linalg.norm(forward_kinematics(rots, sk) + transl - ref[:55], axis=1).max())
print(f"  float64 control (same maths, float64 model): max err {worst64:.3e} m")

# Large-angle stress: rotations near pi
worst_big = 0.0
betas_big = rng.normal(0, 2.0, 10)
skel = SMPLXSkeleton(model_path="models", betas=betas_big)
for k in range(10):
    rots = rng.normal(0, 1.0, (55, 3))
    rots *= (rng.uniform(2.5, 3.1, (55, 1)) / np.linalg.norm(rots, axis=1, keepdims=True))
    transl = rng.normal(0, 1.0, 3)
    ref = run_model(skel.model, rots, betas_big, transl, torch.float32)
    worst_big = max(worst_big, np.linalg.norm(forward_kinematics(rots, skel) + transl - ref[:55], axis=1).max())
print(f"  large-angle stress (|rotvec| in [2.5,3.1] rad every joint): max err {worst_big:.3e} m")

# Does the skeleton use the shaped template? Compare rest joints vs model rest output.
betas = rng.normal(0, 2.0, 10)
skel = SMPLXSkeleton(model_path="models", betas=betas)
rest_ref = run_model(skel.model, np.zeros((55, 3)), betas, np.zeros(3), torch.float32)[:55]
skel0 = SMPLXSkeleton(model_path="models")
print(f"  rest joints (betas={np.round(betas[:3],2)}...) vs SMPL-X zero-pose output: "
      f"max {np.abs(skel.rest_joints - rest_ref).max():.3e} m; "
      f"vs betas=0 skeleton differ by up to {np.linalg.norm(skel.rest_joints - skel0.rest_joints, axis=1).max()*1000:.1f} mm")
print(f"  parents[:22] = {skel.parents[:22].tolist()}")
