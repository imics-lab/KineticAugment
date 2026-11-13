"""
Minimal synthetic demonstration of the coordinate-convention bug.

One physical pose (arms hanging, forearms forward) is written in two frames:
  SMPL-X frame   : x = subject left, y = UP,   z = forward (toward camera)
  MediaPipe world: x = subject left, y = DOWN, z = away from camera   (= C @ p, C = diag(1,-1,-1))
fit_body should return the SAME body_pose for both (only global_orient/transl may differ).
"""
import numpy as np
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
C = np.diag([1.0, -1.0, -1.0])
rv = lambda e: Rotation.from_euler("xyz", e, degrees=True).as_rotvec()

truth = np.zeros((55, 3))
truth[16] = rv([0, 0, -70])     # left upper arm down
truth[17] = rv([0, 0, 70])      # right upper arm down
truth[18] = rv([0, -60, 0])     # left forearm forward
truth[19] = rv([0, 60, 0])      # right forearm forward
J = forward_kinematics(truth, sk) + np.array([0.0, -0.3, 0.0])
lm_smplx = np.full((33, 3), np.nan)
for s, m in ARM_CORRESPONDENCE:
    lm_smplx[m] = J[s]
v = np.isfinite(lm_smplx).all(1)
lm_mp = lm_smplx @ C.T          # the same physical pose in MediaPipe's world convention

def body_local(fit):
    r = np.zeros((55, 3)); r[1:22] = fit.body_pose.reshape(21, 3)
    return forward_kinematics(r, sk)

for label, lm in (("SMPL-X-frame input", lm_smplx), ("MediaPipe-frame input", lm_mp)):
    f = fit_body(lm, v, sk)
    out = reconstruct_body(f, sk)
    err = max(np.linalg.norm(out[m] - lm[m]) for _, m in ARM_CORRESPONDENCE)
    B = body_local(f)
    print(f"{label}:")
    print(f"   direct round-trip max error         {1000*err:.2e} mm")
    print(f"   |global_orient|                     {np.degrees(np.linalg.norm(f.global_orient)):.1f} deg")
    print(f"   body-local L elbow - L shoulder (y) {B[18,1]-B[16,1]:+.3f} m  (truth {J[18,1]-J[16,1]:+.3f}: below shoulder)")
    print(f"   body-local L wrist  - L shoulder (z){B[20,2]-B[16,2]:+.3f} m  (truth {J[20,2]-J[16,2]:+.3f}: in front)")
    print(f"   L shoulder rotvec {np.round(f.body_pose.reshape(21,3)[15],3)}  (truth {np.round(truth[16],3)})")
