"""The strict-xfail rotated-root test: its own error today, and whether a shoulder-only root fixes it."""
import numpy as np
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik, _swing_rotation
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE as CORR, BodyFit

sk = SMPLXSkeleton(model_path="models"); rest = sk.rest_joints

def random_arm_pose(seed):  # copied from tests/test_converter.py
    rng = np.random.default_rng(seed); rots = np.zeros((55, 3))
    for joint, _, _, _ in SMPLX_ARM_BONES: rots[joint] = rng.normal(0, 0.3, 3)
    return rots

def landmarks_from(rotations, offset=np.zeros(3)):
    J = forward_kinematics(rotations, sk) + offset
    L = np.full((33, 3), np.nan)
    for s, m in CORR: L[m] = J[s]
    return L, np.isfinite(L).all(1)

def fit_body_shoulder_root(L, validity):
    """Prototype Step 8 variant: root from the shoulder line only (rotation about it is a gauge)."""
    a = rest[16] - rest[17]; a /= np.linalg.norm(a)
    b = L[11] - L[12]; b /= np.linalg.norm(b)
    R = _swing_rotation(a, b)
    rv = Rotation.from_matrix(R).as_rotvec()
    masked = np.where(validity[:, None], L, np.nan)
    rots = solve_swing_ik(masked, sk, SMPLX_ARM_BONES, global_orient=rv)
    J = forward_kinematics(rots, sk)
    pairs = [(s, m) for s, m in CORR if validity[m]]
    t = np.mean([L[m] - J[s] for s, m in pairs], axis=0)
    return BodyFit(body_pose=rots[1:22].reshape(63).copy(), global_orient=rots[0], transl=t)

truth = random_arm_pose(4); truth[0] = Rotation.from_euler("y", 30, degrees=True).as_rotvec()
L, v = landmarks_from(truth)
out = reconstruct_body(fit_body(L, v, sk), sk)
e = np.array([np.linalg.norm(out[m] - L[m]) for _, m in CORR])
print(f"xfail test case (seed 4, 30 deg about y) with current converter: max {e.max()*1e3:.2f} mm, mean {e.mean()*1e3:.2f} mm; per landmark mm: {np.round(e*1e3,2)}")
out2 = reconstruct_body(fit_body_shoulder_root(L, v), sk)
e2 = np.array([np.linalg.norm(out2[m] - L[m]) for _, m in CORR])
print(f"same case with shoulder-only root prototype: max {e2.max():.3e} m  -> {'PASSES' if e2.max() < 1e-6 else 'FAILS'} the test's atol=1e-6")
print("nose landmark (index 0) present in the test fixture?", np.isfinite(L[0]).all())

worst = 0; rng = np.random.default_rng(42)
for _ in range(200):
    t = np.zeros((55, 3)); a = rng.normal(size=3); t[0] = a / np.linalg.norm(a) * np.radians(rng.uniform(0, 180))
    for j, *_ in SMPLX_ARM_BONES: t[j] = rng.normal(0, 0.4, 3)
    L, v = landmarks_from(t, offset=rng.normal(0, 1, 3))
    o = reconstruct_body(fit_body_shoulder_root(L, v), sk)
    worst = max(worst, max(np.linalg.norm(o[m] - L[m]) for _, m in CORR))
print(f"shoulder-only root prototype, 200 random poses, random root axis and angle 0-180 deg, random offset: worst {worst:.3e} m")

# Missing-landmark case: one shoulder invalid -> the shoulder-line root is undefined
L, v = landmarks_from(truth); v = v.copy(); v[12] = False
try:
    o = reconstruct_body(fit_body(L, v, sk), sk)
    print("current converter with R shoulder invalid: runs; max err on valid landmarks (mm):",
          round(max(np.linalg.norm(o[m] - L[m]) for _, m in CORR if v[m]) * 1e3, 2))
except Exception as ex:
    print("current converter with R shoulder invalid raised", type(ex).__name__, ex)
