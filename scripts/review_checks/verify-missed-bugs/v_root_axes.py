"""A002 vs N1: root error about a generic axis costs position error; a pitch about the shoulder axis (like C=diag(1,-1,-1)) costs none."""
import numpy as np
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES
sk = SMPLXSkeleton(model_path="models"); rng = np.random.default_rng(3)
axis_sh = sk.rest_joints[16] - sk.rest_joints[17]; axis_sh /= np.linalg.norm(axis_sh)
for label, rv in (("0.7 rad about z (review counterexample axis)", [0, 0, 0.7]),
                  ("180 deg about x (MediaPipe y-down convention C)", [np.pi, 0, 0]),
                  ("180 deg about exact rest shoulder axis", np.pi * axis_sh)):
    errs = []
    for _ in range(50):
        rot = np.zeros((55, 3)); rot[0] = rv
        for j, *_ in SMPLX_ARM_BONES: rot[j] = rng.normal(0, 0.4, 3)
        J = forward_kinematics(rot, sk); L = np.full((33, 3), np.nan)
        for s, m in ARM_CORRESPONDENCE: L[m] = J[s]
        out = reconstruct_body(fit_body(L, np.isfinite(L).all(1), sk), sk)
        errs.append(max(np.linalg.norm(out[m] - L[m]) for _, m in ARM_CORRESPONDENCE))
    print(f"{label:<50} converter worst arm error over 50 poses: median {1000*np.median(errs):.2f} mm, max {1000*np.max(errs):.2f} mm")
