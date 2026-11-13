"""E2b: Would Step 8 (a correct torso root) make zero-noise clamping a no-op?
Fits arms relative to a torso frame from shoulders+nose (the Step 8 design), then applies the repo's
clamp_joint_angles exactly as JointAnglePerturbation(respect_limits=True) does, and measures
(a) elbow bend before/after, (b) wrist displacement. Also checks which local axis carries SMPL-X
elbow flexion at rest (to test the joint-limit axis convention)."""
import warnings
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.joint_mapping import clamp_joint_angles, SMPLX_JOINT_LIMITS

sk = SMPLXSkeleton(model_path="models")
with torch.no_grad():
    J = sk.model(return_verts=False).joints[0].numpy()
rest_nose = J[55].astype(np.float64)          # SMPL-X extra landmark 55 = nose
rest = sk.rest_joints

# (0) Which local axis is elbow flexion in SMPL-X? Bend the left forearm toward the body front (+z).
def wrist_after(rv):
    rots = np.zeros((55, 3)); rots[18] = rv
    return forward_kinematics(rots, sk)[20]
for ax, name in [([1,0,0],'x'), ([0,1,0],'y'), ([0,0,1],'z')]:
    w = wrist_after(np.array(ax) * np.radians(-60))
    d = (w - rest[18]); d /= np.linalg.norm(d)
    print(f"left elbow -60deg about local {name}: forearm dir = {np.round(d,3)} (front is +z)")
print("repo limits left_elbow:", SMPLX_JOINT_LIMITS["left_elbow"], "right_elbow:", SMPLX_JOINT_LIMITS["right_elbow"])

def kabsch(A, B):
    a, b = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T

names = {16: "left_shoulder", 17: "right_shoulder", 18: "left_elbow", 19: "right_elbow"}
bend_before, bend_after, wdisp, violated = [], [], [], 0
nfr = 0
for c in ["01986", "01987", "01988", "01991", "01992"]:
    d = np.load(f"data/test_output/{c}_world.npz")
    L, V = d["landmarks"].astype(np.float64), d["visibility"]
    for t in range(len(L)):
        ok = (V[t] > 0.5) & np.isfinite(L[t]).all(1)
        if not (ok[0] and ok[11] and ok[12] and ok[13] and ok[14]):
            continue
        Rt = kabsch(np.stack([rest[16], rest[17], rest_nose]), np.stack([L[t,11], L[t,12], L[t,0]]))
        masked = np.where(ok[:, None], L[t], np.nan)
        rots = solve_swing_ik(masked, sk, SMPLX_ARM_BONES, global_orient=R.from_matrix(Rt).as_rotvec())
        nfr += 1
        cl = rots.copy()
        any_v = False
        for j, nm in names.items():
            cl[j] = clamp_joint_angles(nm, rots[j])
            any_v |= not np.allclose(cl[j], rots[j])
        violated += any_v
        P0, P1 = forward_kinematics(rots, sk), forward_kinematics(cl, sk)
        sw = np.linalg.norm(L[t,11] - L[t,12])
        for e, w, s in [(18, 20, 16), (19, 21, 17)]:
            if not ok[{18:15,19:16}[e]]:
                continue
            def bend(P):
                u, v = P[e]-P[s], P[w]-P[e]
                return np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v), -1, 1)))
            bend_before.append(bend(P0)); bend_after.append(bend(P1))
            wdisp.append(np.linalg.norm(P1[w]-P0[w]) / sw * 100)
bb, ba, wd = map(np.array, (bend_before, bend_after, wdisp))
print(f"frames (torso-rooted fit): {nfr}; frames where clamp alters an arm joint: {violated}/{nfr}")
print(f"elbow bend (0=straight) before clamp: median {np.median(bb):.1f} deg  [p5 {np.percentile(bb,5):.1f}, p95 {np.percentile(bb,95):.1f}]")
print(f"elbow bend after  clamp: median {np.median(ba):.1f} deg  [p5 {np.percentile(ba,5):.1f}, p95 {np.percentile(ba,95):.1f}]")
print(f"wrist displacement from zero-noise clamp: median {np.median(wd):.1f}%SW p95 {np.percentile(wd,95):.1f}%SW max {wd.max():.1f}%SW")
