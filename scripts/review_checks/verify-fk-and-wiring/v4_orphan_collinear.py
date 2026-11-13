import numpy as np, warnings
from scipy.spatial.transform import Rotation as Rot
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
sk = SMPLXSkeleton(model_path="models"); ARM = [m for _, m in ARM_CORRESPONDENCE]
rng = np.random.default_rng(2024)
def lm(rots, R=np.eye(3), t=np.zeros(3)):
    j = forward_kinematics(rots, sk) @ R.T + t; L = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE: L[m] = j[s]
    return L, np.isfinite(L).all(1)
def pose():
    r = np.zeros((55, 3))
    for j, *_ in SMPLX_ARM_BONES: r[j] = rng.normal(0, .4, 3)
    return r
def orphan(valid, clean, trials=300):
    e = []
    for _ in range(trials):
        L, v = lm(pose(), np.eye(3), rng.normal(0, .3, 3)); vv = np.zeros(33, bool); vv[valid] = True
        out = reconstruct_body(fit_body(L, vv, sk), sk)
        e.append(np.linalg.norm(out[clean] - L[clean], axis=1).max())
    return np.median(e) * 1e3, np.max(e) * 1e3
print("orphan L wrist (L elbow invalid); error on clean joints 11,12,14,16: med %.1f max %.1f mm" % orphan([11, 12, 14, 15, 16], [11, 12, 14, 16]))
print("orphan L elbow (L shoulder invalid); clean joints 12,14,16:        med %.1f max %.1f mm" % orphan([12, 13, 14, 15, 16], [12, 14, 16]))
print("L sh + R sh + L wrist only; clean joints 11,12:                   med %.1f max %.1f mm" % orphan([11, 12, 15], [11, 12]))
print("control: all valid:                                               med %.2g max %.2g mm" % orphan(ARM, ARM))
# collinear triple with a hanging straight left arm
for down in (-1.2,):
    rots = np.zeros((55, 3)); rots[16] = [0, 0, down]
    j = forward_kinematics(rots, sk); ax = j[20] - j[16]; ax /= np.linalg.norm(ax)
    p = j[16]
    d = np.linalg.norm(np.cross(j[17] - p, ax))
    for ang in (0.5, 1.0, 2.0):
        R0 = Rot.from_rotvec(ax * ang).as_matrix(); L, _ = lm(rots, R0, p - R0 @ p)
        vv = np.zeros(33, bool); vv[[11, 13, 15]] = True
        out = reconstruct_body(fit_body(L, vv, sk), sk)
        print(f"collinear hanging arm, R sh dist to axis {d*1e3:.0f} mm, rot {ang}: R sh err {np.linalg.norm(out[12]-L[12])*1e3:.0f} mm (2d sin(a/2)={2*d*np.sin(ang/2)*1e3:.0f})")
# near-collinear: 1 degree elbow bend
rots = np.zeros((55, 3)); rots[16] = [0, 0, -1.2]; rots[18] = [0, np.deg2rad(1), 0]
j = forward_kinematics(rots, sk); ax = j[20] - j[16]; ax /= np.linalg.norm(ax); p = j[16]
R0 = Rot.from_rotvec(ax * 1.0).as_matrix(); L, _ = lm(rots, R0, p - R0 @ p)
vv = np.zeros(33, bool); vv[[11, 13, 15]] = True
out = reconstruct_body(fit_body(L, vv, sk), sk)
print(f"1-deg bent arm, rot 1.0 about arm: R sh err {np.linalg.norm(out[12]-L[12])*1e3:.1f} mm")
