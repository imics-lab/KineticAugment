import glob, numpy as np
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.joint_mapping import clamp_joint_angles, SMPLX_JOINT_LIMITS
from kinetic_augment.body_model.converter import fit_body
sk = SMPLXSkeleton(model_path="models")
J0 = forward_kinematics(np.zeros((55, 3)), sk)
print("rest L wrist", np.round(J0[20], 3))
for name, rv in [("x", [np.pi/2, 0, 0]), ("+y", [0, np.pi/2, 0]), ("-y", [0, -np.pi/2, 0]), ("z", [0, 0, np.pi/2]), ("-z", [0, 0, -np.pi/2])]:
    p = np.zeros((55, 3)); p[18] = rv
    J = forward_kinematics(p, sk); d = J[20]-J0[20]
    print(f"L elbow 90deg about local {name:<2}: wrist moves {1000*np.linalg.norm(d):5.0f} mm, dir {np.round(d/np.linalg.norm(d),2)}")
for name, rv in [("x", [np.pi/2, 0, 0]), ("y", [0, np.pi/2, 0]), ("z", [0, 0, np.pi/2])]:
    p = np.zeros((55, 3)); p[16] = rv
    J = forward_kinematics(p, sk); d = J[18]-J0[18]
    print(f"L shoulder 90deg about local {name}: elbow moves {1000*np.linalg.norm(d):5.0f} mm")
c = clamp_joint_angles("left_elbow", np.array([0, -np.pi/2, 0]))
p = np.zeros((55, 3)); p[18] = [0, -np.pi/2, 0]; Ja = forward_kinematics(p, sk)
p[18] = c; Jb = forward_kinematics(p, sk)
print("clamp(left_elbow,[0,-1.571,0]) ->", np.round(c, 3), "wrist moves %.0f mm" % (1000*np.linalg.norm(Ja[20]-Jb[20])))
# zero-noise clamp on real data, using fits after a fixed MP->SMPL-X flip (anatomically oriented root)
FLIP = R.from_euler('x', 180, degrees=True).as_matrix()
names = {16: "left_shoulder", 17: "right_shoulder", 18: "left_elbow", 19: "right_elbow"}
moves = {"L": [], "R": []}
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f)
    for lm, vis in zip(d["landmarks"].astype(float), d["visibility"]):
        ok = np.isfinite(lm).all(1) & (vis >= 0.5)
        try: fit = fit_body(lm @ FLIP.T, ok, sk)
        except ValueError: continue
        rots = np.zeros((55, 3)); rots[1:22] = fit.body_pose.reshape(21, 3)
        cl = rots.copy()
        for j, n in names.items(): cl[j] = clamp_joint_angles(n, rots[j])
        Ja, Jb = forward_kinematics(rots, sk), forward_kinematics(cl, sk)
        if ok[15] and ok[13]: moves["L"].append(np.linalg.norm(Ja[20]-Jb[20]))
        if ok[16] and ok[14]: moves["R"].append(np.linalg.norm(Ja[21]-Jb[21]))
for s, m in moves.items():
    m = np.array(m)*1000
    print(f"{s}: zero-noise clamp (shoulders+elbows) moves wrist median {np.median(m):.0f} mm, >50mm in {np.mean(m>50)*100:.0f}% (n={len(m)})")
