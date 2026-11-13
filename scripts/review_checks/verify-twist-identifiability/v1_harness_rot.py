# Independent replica of measure_round_trip.run_synthetic that also scores rotations.
import numpy as np
from scipy.spatial.transform import Rotation as R
import importlib.util, sys
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

sk = SMPLXSkeleton(model_path="models")
rest = sk.rest_joints

def geo(a, b):
    return np.degrees((R.from_rotvec(a).inv() * R.from_rotvec(b)).magnitude())

def twist_angle(rv, axis):
    # swing-twist: q = swing*twist, twist about axis (in local/rest frame)
    q = R.from_rotvec(rv).as_quat()  # x,y,z,w
    v = q[:3]; w = q[3]
    p = np.dot(v, axis) * axis
    t = np.array([*p, w]); n = np.linalg.norm(t)
    if n < 1e-12: return 180.0
    t /= n
    return np.degrees(2*np.arccos(min(1, abs(t[3]))))

for root_deg in (0, 30):
    rng = np.random.default_rng(42)
    pos_worst = 0.0
    errs = {j: [] for j, *_ in SMPLX_ARM_BONES}
    tw = {j: [] for j, *_ in SMPLX_ARM_BONES}
    allok = 0
    for _ in range(200):
        truth = np.zeros((55, 3))
        if root_deg:
            ax = rng.normal(size=3); truth[0] = ax/np.linalg.norm(ax)*np.deg2rad(root_deg)
        for j, *_ in SMPLX_ARM_BONES:
            truth[j] = rng.normal(0, 0.4, 3)
        J = forward_kinematics(truth, sk)
        lm = np.full((33, 3), np.nan)
        for s, m in ARM_CORRESPONDENCE: lm[m] = J[s]
        fit = fit_body(lm, np.isfinite(lm).all(1), sk)
        out = reconstruct_body(fit, sk)
        for s, m in ARM_CORRESPONDENCE:
            pos_worst = max(pos_worst, np.linalg.norm(out[m]-lm[m]))
        bp = fit.body_pose.reshape(21, 3)
        ok = True
        for j, c, *_ in SMPLX_ARM_BONES:
            e = geo(truth[j], bp[j-1]); errs[j].append(e); ok &= e < 1e-6
            axis = rest[c]-rest[j]; axis /= np.linalg.norm(axis)
            tw[j].append(twist_angle(truth[j], axis))
        allok += ok
    print(f"root_deg={root_deg}: worst pos err {pos_worst:.3e} m; poses with all 4 arm rots <1e-6deg: {allok}/200")
    names = {16:'L shoulder',18:'L elbow',17:'R shoulder',19:'R elbow'}
    allv = []
    for j in errs:
        e = np.array(errs[j]); allv += list(e)
        t = np.array(tw[j])
        print(f"  {names[j]:<11} rot err med/p95/max {np.median(e):6.2f}/{np.percentile(e,95):6.2f}/{e.max():6.2f} deg"
              f" | truth twist med {np.median(t):6.2f} | max|err-twist| {np.max(np.abs(e-t)):.2e}")
    print(f"  all arm joints median {np.median(allv):.2f} deg")

# seed-7 fixture from tests/test_swing_ik.py
rng = np.random.default_rng(7)
truth = np.zeros((55, 3))
for j, *_ in SMPLX_ARM_BONES: truth[j] = rng.normal(0, 0.25, 3)
J = forward_kinematics(truth, sk)
mp = np.zeros((33, 3))
for s, m in ARM_CORRESPONDENCE: mp[m] = J[s]
rots = solve_swing_ik(mp, sk, SMPLX_ARM_BONES)
Jr = forward_kinematics(rots, sk)
print("seed7 fixture: max child pos err", max(np.linalg.norm(Jr[c]-J[c]) for _, c, *_ in SMPLX_ARM_BONES))
for j, *_ in SMPLX_ARM_BONES:
    print(f"  joint {j}: rot err {geo(truth[j], rots[j]):.2f} deg")
