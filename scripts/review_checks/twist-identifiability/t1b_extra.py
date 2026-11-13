"""Supplements to t1: (A') forearm twist compensated by the 3-DOF wrist -> all 55 joints identical;
(B') under humerus twist, the ELBOW's local *swing* also changes (so 'swing is exact' holds only for
bone world directions, not for the elbow's local parameters)."""
import numpy as np, torch, smplx
from scipy.spatial.transform import Rotation as R
torch.set_default_dtype(torch.float64)
model = smplx.create("models", model_type="smplx", gender="neutral", use_pca=False,
                     flat_hand_mean=True, batch_size=1, dtype=torch.float64)
with torch.no_grad():
    rest = (model.J_regressor @ model.v_template)[:55].numpy()
LS, LE, LW = 16, 18, 20
u_up = rest[LE]-rest[LS]; u_up/=np.linalg.norm(u_up)
u_fa = rest[LW]-rest[LE]; u_fa/=np.linalg.norm(u_fa)
print(f"rest included angle humerus vs forearm: {np.degrees(np.arccos(u_up@u_fa)):.1f} deg")
def fwd(p):
    t=lambda a: torch.tensor(np.asarray(a,dtype=np.float64)).reshape(1,-1)
    with torch.no_grad():
        o=model(global_orient=t(p[0]),body_pose=t(p[1:22]),left_hand_pose=t(p[25:40]),
                right_hand_pose=t(p[40:55]),return_verts=True)
    return o.joints[0,:55].numpy(), o.vertices[0].numpy()
base=np.zeros((55,3))
base[LS]=(R.from_euler("z",-70,degrees=True)*R.from_euler("y",-20,degrees=True)).as_rotvec()
base[LE]=np.array([0,-np.radians(90),0])
J0,V0=fwd(base)
print("(A') forearm twist + compensating wrist (joint 20 local rot pre-multiplied by inverse twist)")
for th in (30,60,90):
    p=base.copy(); T=R.from_rotvec(np.radians(th)*u_fa)
    p[LE]=(R.from_rotvec(base[LE])*T).as_rotvec(); p[LW]=(T.inv()*R.from_rotvec(base[LW])).as_rotvec()
    J,V=fwd(p); dv=np.linalg.norm(V-V0,axis=1)
    print(f"  {th:3d} deg: max|dJ| all 55 joints {np.max(np.linalg.norm(J-J0,axis=1)):.1e} m; verts max {dv.max()*1000:.1f} mm")
def swing_twist(rotvec, axis):
    q=R.from_rotvec(rotvec).as_quat()  # x,y,z,w
    v=q[:3]; w=q[3]; p=np.dot(v,axis)*axis
    tw=np.array([*p,w]); tw/=np.linalg.norm(tw); twist=R.from_quat(tw)
    swing=R.from_rotvec(rotvec)*twist.inv()
    ang=lambda r: np.degrees(r.magnitude())
    s=np.sign(np.dot(tw[:3],axis)) or 1
    return ang(swing), s*ang(twist)
print("(B') elbow local rotation under compensated humerus twist: swing/twist about rest forearm axis")
for th in (0,30,60,90):
    T=R.from_rotvec(np.radians(th)*u_up); e=(T.inv()*R.from_rotvec(base[LE])).as_rotvec()
    sw,tw=swing_twist(e,u_fa)
    print(f"  humerus twist {th:3d}: elbow rotvec {np.round(e,3)}  swing {sw:6.1f} deg  twist {tw:6.1f} deg")
