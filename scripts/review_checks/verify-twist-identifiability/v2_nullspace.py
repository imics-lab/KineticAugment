# Null-space check on the REAL SMPL-X forward pass (float64).
import numpy as np, torch, smplx
from scipy.spatial.transform import Rotation as R
torch.set_default_dtype(torch.float64)
m = smplx.create("models", model_type="smplx", gender="neutral", use_pca=False,
                 flat_hand_mean=True, batch_size=1, dtype=torch.float64)
with torch.no_grad():
    out0 = m()
J0 = out0.joints[0, :55].numpy()
rest = J0.copy()
def fwd(full):  # full (55,3) local rotvecs
    t = lambda a: torch.tensor(a[None], dtype=torch.float64)
    with torch.no_grad():
        o = m(global_orient=t(full[0]), body_pose=t(full[1:22].reshape(-1)),
              jaw_pose=t(full[22]), leye_pose=t(full[23]), reye_pose=t(full[24]),
              left_hand_pose=t(full[25:40].reshape(-1)), right_hand_pose=t(full[40:55].reshape(-1)))
    return o.joints[0, :55].numpy(), o.vertices[0].numpy()
LS, LE, LW = 16, 18, 20
u = rest[LE]-rest[LS]; u /= np.linalg.norm(u)
f = rest[LW]-rest[LE]; f /= np.linalg.norm(f)
print("rest angle between upper-arm and forearm bones: %.2f deg" % np.degrees(np.arccos(u@f)))
base = np.zeros((55, 3))
base[LS] = R.from_euler('xyz', [10, 20, -40], degrees=True).as_rotvec()
base[LE] = np.array([0, -np.pi/2, 0])     # left elbow 90 deg about local -y
base[LW] = R.from_euler('xyz', [5, -10, 15], degrees=True).as_rotvec()
Jb, Vb = fwd(base)
ue = Jb[LE]-Jb[LS]; fe = Jb[LW]-Jb[LE]
print("base: angle between bones %.1f deg" % np.degrees(np.arccos(ue@fe/np.linalg.norm(ue)/np.linalg.norm(fe))))
def rep(tag, p):
    J, V = fwd(p)
    d = np.linalg.norm(J-Jb, axis=1)
    arm = max(d[LS], d[LE], d[LW])
    print(f"  {tag:<44} arm3 {arm:.1e} m | all55 {d.max():.1e} m | max vert {1000*np.linalg.norm(V-Vb,axis=1).max():6.1f} mm")
for th in (30, 60, 90):
    print(f"theta={th}")
    T = R.from_rotvec(np.radians(th)*f)
    # (A) forearm twist only (terminal segment in the 6-landmark set)
    p = base.copy(); p[LE] = (R.from_rotvec(base[LE])*T).as_rotvec(); rep("A  forearm twist", p)
    # (A') + wrist compensation
    p[LW] = (T.inv()*R.from_rotvec(base[LW])).as_rotvec(); rep("A' forearm twist + wrist comp", p)
    # (B) humerus twist with elbow compensation (3-DOF elbow)
    Tu = R.from_rotvec(np.radians(th)*u)
    p = base.copy(); p[LS] = (R.from_rotvec(base[LS])*Tu).as_rotvec()
    p[LE] = (Tu.inv()*R.from_rotvec(base[LE])).as_rotvec()
    rep("B  humerus twist + elbow comp", p)
    e_sh = np.degrees((R.from_rotvec(p[LS]).inv()*R.from_rotvec(base[LS])).magnitude())
    e_el = np.degrees((R.from_rotvec(p[LE]).inv()*R.from_rotvec(base[LE])).magnitude())
    print(f"     B rotation diffs: shoulder {e_sh:.1f} deg, elbow {e_el:.1f} deg")
    # (C) humerus twist uncompensated
    p = base.copy(); p[LS] = (R.from_rotvec(base[LS])*Tu).as_rotvec(); rep("C  humerus twist, no comp", p)
