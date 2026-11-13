import numpy as np
from scipy.spatial.transform import Rotation as Rot
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
sk = SMPLXSkeleton(model_path="models")
def lm(rots, R, t):
    j = forward_kinematics(rots, sk) @ R.T + t; L = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE: L[m] = j[s]
    return L
for bend in (0, 1, 30, 90):
    rots = np.zeros((55, 3)); rots[16] = [0, 0, -1.2]; rots[18] = [0, np.deg2rad(bend), 0]
    j = forward_kinematics(rots, sk); p = j[16]
    for name, ax in (("arm axis", (j[20]-j[16])), ("random axis", np.array([0.3, 0.8, -0.5]))):
        ax = ax / np.linalg.norm(ax); R0 = Rot.from_rotvec(ax * 1.0).as_matrix()
        L = lm(rots, R0, p - R0 @ p)
        vv = np.zeros(33, bool); vv[[11, 13, 15]] = True
        out = reconstruct_body(fit_body(L, vv, sk), sk)
        print(f"bend {bend:2d} deg, 1 rad about {name:11s}: L-arm max err {max(np.linalg.norm(out[m]-L[m]) for m in (11,13,15))*1e3:.2g} mm, R sh err {np.linalg.norm(out[12]-L[12])*1e3:.0f} mm")
