"""How much head/neck motion leaks into a shoulders+nose torso estimate (true torso fixed).

Truth: real SMPL-X forward pass; nose = joints[55] (skinned vertex landmark).
Estimator: rotation carrying the rest (shoulder line, nose) frame to the posed one.
Also: arm reconstruction invariance between nose root and hip root on real data.
"""
import glob
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

SK = SMPLXSkeleton(model_path="models")
M = SK.model.double()


def joints(body_pose):
    with torch.no_grad():
        return M(body_pose=torch.tensor(body_pose)[None], return_verts=False).joints[0].numpy()


def frame(l, r, apex):
    x = l - r; x /= np.linalg.norm(x)
    y = apex - 0.5 * (l + r); y -= (y @ x) * x; y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], 1)


J0 = joints(np.zeros(63))
F0 = frame(J0[16], J0[17], J0[55])
print("rest nose elevation above shoulder midline, angle forward of torso up-axis: %.1f deg" % np.degrees(
    np.arctan2((J0[55] - 0.5 * (J0[16] + J0[17]))[2], (J0[55] - 0.5 * (J0[16] + J0[17]))[1])))
for name, joint, axis in [("neck(12) pitch", 12, "x"), ("head(15) pitch", 15, "x"),
                          ("head(15) yaw", 15, "y"), ("head(15) roll", 15, "z")]:
    for deg in (15, 20, 30, 40, -20):
        bp = np.zeros(63)
        bp[(joint - 1) * 3:(joint - 1) * 3 + 3] = R.from_euler(axis, deg, degrees=True).as_rotvec()
        J = joints(bp)
        Rest = frame(J[16], J[17], J[55]) @ F0.T   # true torso = identity
        e = R.from_matrix(Rest)
        rv = e.as_rotvec()
        print(f"  {name:15s} {deg:4d} deg -> false torso rotation {np.degrees(e.magnitude()):6.2f} deg, "
              f"axis {np.round(rv / (np.linalg.norm(rv) + 1e-15), 3)}; shoulders moved {np.abs(J[16:18]-J0[16:18]).max():.1e} m")

# real data: nose-root vs hip-root arm reconstructions identical after translation?
REST = SK.rest_joints
REST_NOSE = J0[55]
CORR = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]
worst, nosevis = 0.0, []
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); P = d["landmarks"].astype(float); V = d["visibility"]
    OK = np.isfinite(P).all(-1) & (V >= 0.5)
    nosevis.append(np.nanmin(np.where(np.isfinite(P[:, 0]).all(-1), V[:, 0], np.nan)))
    for lm, ok in zip(P, OK):
        if not (ok[11] and ok[12] and ok[0]):
            continue
        outs = []
        for apex_o, apex_r in ((lm[0], REST_NOSE), (0.5 * (lm[23] + lm[24]), 0.5 * (REST[1] + REST[2]))):
            Rr = frame(lm[11], lm[12], apex_o) @ frame(REST[16], REST[17], apex_r).T
            rots = solve_swing_ik(np.where(ok[:, None], lm, np.nan), SK, SMPLX_ARM_BONES,
                                  global_orient=R.from_matrix(Rr).as_rotvec())
            Jf = forward_kinematics(rots, SK)
            pres = [(s, m) for s, m in CORR if ok[m]]
            src = np.array([Jf[s] for s, _ in pres]); tgt = np.array([lm[m] for _, m in pres])
            outs.append(src + (tgt - src).mean(0))
        worst = max(worst, np.abs(outs[0] - outs[1]).max())
print("max |nose-root recon - hip-root recon| over real frames: %.2e m" % worst)
print("min nose visibility per clip:", np.round(nosevis, 4))
