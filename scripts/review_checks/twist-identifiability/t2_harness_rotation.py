"""A003 step 1: replicate measure_round_trip.run_synthetic (seed 42, 200 trials, N(0,0.4) per
rotvec component, root 0) EXACTLY, and additionally measure what the harness does not: the
rotation error of the recovered parameters vs the generating ones, split into swing/twist.
Also checks the swing_ik SMPL-X test (seed 7, sigma 0.25)."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

sk = SMPLXSkeleton(model_path="models")
rest = sk.rest_joints


def twist_about(rot: R, axis):
    """signed twist angle (deg) of rotation rot about unit axis (swing-twist decomposition)."""
    q = rot.as_quat()  # x y z w
    p = np.dot(q[:3], axis)
    return np.degrees(2 * np.arctan2(p, q[3]))


def synthetic_landmarks(rotations):
    joints = forward_kinematics(rotations, sk)
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lm[m] = joints[s]
    return lm, np.isfinite(lm).all(axis=1)


rng = np.random.default_rng(42)
pos_worst = 0.0
rot_err = {j: [] for j, *_ in SMPLX_ARM_BONES}
tw_err = {j: [] for j, *_ in SMPLX_ARM_BONES}
for _ in range(200):
    truth = np.zeros((55, 3))
    for joint, _, _, _ in SMPLX_ARM_BONES:
        truth[joint] = rng.normal(0, 0.4, 3)
    lm, ok = synthetic_landmarks(truth)
    fit = fit_body(lm, ok, sk)
    out = reconstruct_body(fit, sk)
    for _, m in ARM_CORRESPONDENCE:
        pos_worst = max(pos_worst, np.linalg.norm(out[m] - lm[m]))
    got = np.zeros((55, 3)); got[0] = fit.global_orient; got[1:22] = fit.body_pose.reshape(21, 3)
    for j, c, *_ in SMPLX_ARM_BONES:
        e = R.from_rotvec(truth[j]).inv() * R.from_rotvec(got[j])
        rot_err[j].append(np.degrees(e.magnitude()))
        ax = rest[c] - rest[j]; ax /= np.linalg.norm(ax)
        tw_err[j].append(abs(twist_about(R.from_rotvec(truth[j]), ax)))

print("measure_round_trip.run_synthetic replica (seed 42, 200 poses, root 0 deg)")
print(f"  worst joint POSITION error (what the harness reports): {pos_worst:.3e} m")
names = {16: "L shoulder", 18: "L elbow", 17: "R shoulder", 19: "R elbow"}
for j in rot_err:
    a = np.array(rot_err[j]); t = np.array(tw_err[j])
    print(f"  {names[j]:<11} ROTATION error deg: median {np.median(a):6.2f}  p95 {np.percentile(a,95):6.2f}  "
          f"max {a.max():6.2f}  | frac < 1e-6 deg: {np.mean(a<1e-6):.2f} | |truth local twist| median {np.median(t):5.1f}")
allr = np.concatenate([rot_err[j] for j in rot_err])
print(f"  ALL arm joints: rotation error median {np.median(allr):.2f} deg, max {allr.max():.2f} deg; "
      f"poses with every arm joint < 1e-6 deg: "
      f"{np.mean(np.all(np.stack([np.array(rot_err[j]) for j in rot_err]) < 1e-6, axis=0)):.2f}")

# swing_ik SMPL-X unit test fixture (seed 7, sigma 0.25)
rng = np.random.default_rng(7)
truth = np.zeros((55, 3))
for joint, _, _, _ in SMPLX_ARM_BONES:
    truth[joint] = rng.normal(0, 0.25, 3)
J = forward_kinematics(truth, sk)
mp = np.zeros((33, 3))
for pj, cj, a, b in SMPLX_ARM_BONES:
    mp[a] = J[pj]; mp[b] = J[cj]
sol = solve_swing_ik(mp, sk, SMPLX_ARM_BONES)
print("\ntests/test_swing_ik.py::test_recovers_arm_rotations_from_their_own_landmarks fixture (seed 7):")
for j in names:
    print(f"  {names[j]:<11} rotation error {np.degrees((R.from_rotvec(truth[j]).inv()*R.from_rotvec(sol[j])).magnitude()):6.2f} deg")
print(f"  max child-joint position error {max(np.linalg.norm(forward_kinematics(sol, sk)[c]-J[c]) for _,c,_,_ in SMPLX_ARM_BONES):.2e} m")
