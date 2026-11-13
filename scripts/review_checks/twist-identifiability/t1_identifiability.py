"""
A003: which arm twists are identifiable from shoulder/elbow/wrist joint positions?
All position checks use the REAL SMPL-X forward pass (smplx package, neutral, flat_hand_mean=True),
float64 throughout so differences are not float32 noise.
"""
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R

torch.set_default_dtype(torch.float64)
model = smplx.create("models", model_type="smplx", gender="neutral", use_pca=False,
                     flat_hand_mean=True, batch_size=1, dtype=torch.float64)
with torch.no_grad():
    rest = (model.J_regressor @ model.v_template)[:55].numpy()

LS, LE, LW = 16, 18, 20
u_up = rest[LE] - rest[LS]; u_up /= np.linalg.norm(u_up)      # rest humerus axis (shoulder-local)
u_fa = rest[LW] - rest[LE]; u_fa /= np.linalg.norm(u_fa)      # rest forearm axis (elbow-local)

def forward(pose55):
    """pose55: (55,3) local axis-angle. Returns joints(55,3), verts(10475,3) from real SMPL-X."""
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float64)).reshape(1, -1)
    with torch.no_grad():
        out = model(global_orient=t(pose55[0]), body_pose=t(pose55[1:22]),
                    left_hand_pose=t(pose55[25:40]), right_hand_pose=t(pose55[40:55]),
                    jaw_pose=t(pose55[22]), leye_pose=t(pose55[23]), reye_pose=t(pose55[24]),
                    return_verts=True)
    return out.joints[0, :55].numpy(), out.vertices[0].numpy()

def geo_deg(r1, r2):
    return np.degrees((R.from_rotvec(r1).inv() * R.from_rotvec(r2)).magnitude())

# --- realistic signing-like base pose, built anatomically (hinge elbow) -------------
# left shoulder: arm hangs down-forward; elbow flexed 90 deg about the SMPL-X hinge axis (-y for left;
# SMPLify angle prior penalises +y on the left elbow).
base = np.zeros((55, 3))
base[LS] = (R.from_euler("z", -70, degrees=True) * R.from_euler("y", -20, degrees=True)).as_rotvec()
base[LE] = np.array([0.0, -np.radians(90), 0.0])
J0, V0 = forward(base)

arm_vert = None
print("=" * 78)
print("Base pose: left shoulder swing ~70deg down / 20deg fwd, left elbow flexed 90deg (hinge)")
fa = J0[LW] - J0[LE]; ua = J0[LE] - J0[LS]
print(f"elbow included angle {np.degrees(np.arccos(np.dot(fa,ua)/np.linalg.norm(fa)/np.linalg.norm(ua))):.1f} deg between bone vectors")

def report(tag, pose):
    J, V = forward(pose)
    dj_arm = max(np.linalg.norm(J[k] - J0[k]) for k in (LS, LE, LW))
    dj_all = np.max(np.linalg.norm(J - J0, axis=1))
    dv = np.linalg.norm(V - V0, axis=1)
    print(f"{tag:<44} rotdiff S {geo_deg(base[LS], pose[LS]):6.1f}  E {geo_deg(base[LE], pose[LE]):6.1f} deg | "
          f"max|dJ| sh/el/wr {dj_arm:.1e} m, all55 {dj_all:.1e} m | verts max {dv.max()*1000:6.1f} mm mean(moved>1mm) "
          f"{dv[dv>1e-3].mean()*1000 if (dv>1e-3).any() else 0:5.1f} mm, n>1mm {int((dv>1e-3).sum())}")
    return J, V

print("\n(A) TERMINAL twist: elbow local rotation post-multiplied by twist about rest forearm axis")
for th in (30, 60, 90):
    p = base.copy()
    p[LE] = (R.from_rotvec(base[LE]) * R.from_rotvec(np.radians(th) * u_fa)).as_rotvec()
    report(f"  forearm twist {th:3d} deg", p)

print("\n(B) NON-terminal twist: shoulder twisted about humerus, elbow (3-DOF) compensates")
for th in (30, 60, 90):
    p = base.copy()
    T = R.from_rotvec(np.radians(th) * u_up)
    p[LS] = (R.from_rotvec(base[LS]) * T).as_rotvec()
    p[LE] = (T.inv() * R.from_rotvec(base[LE])).as_rotvec()
    report(f"  humerus twist {th:3d} deg + elbow comp.", p)

print("\n(C) shoulder twist WITHOUT compensation (what a hinge elbow would force): wrist moves")
for th in (30, 60, 90):
    p = base.copy()
    p[LS] = (R.from_rotvec(base[LS]) * R.from_rotvec(np.radians(th) * u_up)).as_rotvec()
    J, _ = forward(p)
    print(f"  humerus twist {th:3d} deg: elbow moves {np.linalg.norm(J[LE]-J0[LE])*1000:7.3f} mm, "
          f"wrist moves {np.linalg.norm(J[LW]-J0[LW])*1000:6.1f} mm")

# --- swing IK output vs truth for (A)/(B): what the solver returns --------------------
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics
sk = SMPLXSkeleton(model_path="models")
p = base.copy(); T = R.from_rotvec(np.radians(60) * u_up)
p[LS] = (R.from_rotvec(base[LS]) * T).as_rotvec(); p[LE] = (T.inv() * R.from_rotvec(base[LE])).as_rotvec()
p[LE] = (R.from_rotvec(p[LE]) * R.from_rotvec(np.radians(45) * u_fa)).as_rotvec()
Jt = forward_kinematics(p, sk)
mp = np.full((33, 3), np.nan)
for (dj, cj, a, b) in SMPLX_ARM_BONES:
    mp[a] = Jt[dj]; mp[b] = Jt[cj]
sol = solve_swing_ik(mp, sk, SMPLX_ARM_BONES)
Js = forward_kinematics(sol, sk)
print("\n(D) truth = base + 60deg humerus twist(compensated) + 45deg forearm twist; swing IK solution:")
print(f"  shoulder rot error {geo_deg(p[LS], sol[LS]):.2f} deg, elbow rot error {geo_deg(p[LE], sol[LE]):.2f} deg, "
      f"max arm-joint position error {max(np.linalg.norm(Js[k]-Jt[k]) for k in (LS,LE,LW)):.1e} m")
print(f"  IK solution vs anatomical base: shoulder {geo_deg(base[LS], sol[LS]):.2f} deg, elbow {geo_deg(base[LE], sol[LE]):.2f} deg")
