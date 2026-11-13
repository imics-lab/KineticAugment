"""E9: does the Step 8 design (root from shoulders+nose) repair local-angle semantics?
Same as E6/E7 but the root is set from a Kabsch fit of (L shoulder, R shoulder, nose) between the
SMPL-X rest skeleton and the observation (the design the red-phase Step 8 tests describe), then
arms are solved by swing IK relative to that root. Delta-transfer (R2) vs landmark-native (R3)."""
import warnings
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics

sk = SMPLXSkeleton(model_path="models")
with torch.no_grad():
    rest_nose = sk.model(return_verts=False).joints[0].numpy()[55].astype(np.float64)
rest = sk.rest_joints
def kabsch(A, B):
    a, b = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T
def bend(S, E, Wr):
    u, v = E - S, Wr - E
    return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
SIGN = {18: -1.0, 19: 1.0}; DEG = 10.0
up, front, ysign, db2, agree, ratio = [], [], [], [], [], []
for c in ["01986", "01987", "01988", "01991", "01992"]:
    d = np.load(f"data/test_output/{c}_world.npz")
    L, V = d["landmarks"].astype(np.float64), d["visibility"]
    for t in range(len(L)):
        ok = (V[t] > 0.5) & np.isfinite(L[t]).all(1)
        if not (ok[0] and ok[11] and ok[12]):
            continue
        Rt = kabsch(np.stack([rest[16], rest[17], rest_nose]), np.stack([L[t, 11], L[t, 12], L[t, 0]]))
        up.append((Rt @ [0, 1, 0])[1]); front.append((Rt @ [0, 0, 1])[2])
        rots = solve_swing_ik(np.where(ok[:, None], L[t], np.nan), sk, SMPLX_ARM_BONES,
                              global_orient=R.from_matrix(Rt).as_rotvec())
        P0 = forward_kinematics(rots, sk)
        for j, (s_mp, e_mp, w_mp, s_j, w_j) in ((18, (11, 13, 15, 16, 20)), (19, (12, 14, 16, 17, 21))):
            if not ok[w_mp] or not ok[e_mp]:
                continue
            S, E, Wr = L[t, s_mp], L[t, e_mp], L[t, w_mp]
            b0 = bend(S, E, Wr)
            if b0 < 15 or b0 > 160:
                continue
            ysign.append(np.sign(rots[j, 1]) == np.sign(SIGN[j]))
            r = rots.copy(); r[j, 1] += SIGN[j] * np.radians(DEG)
            P1 = forward_kinematics(r, sk)
            r2 = Wr + (P1[w_j] - P0[w_j])
            n = np.cross(E - S, Wr - E); n /= np.linalg.norm(n)
            r3 = E + R.from_rotvec(n * np.radians(DEG)).apply(Wr - E)
            if bend(S, E, r3) < b0:
                r3 = E + R.from_rotvec(-n * np.radians(DEG)).apply(Wr - E)
            d2, d3 = r2 - Wr, r3 - Wr
            db2.append(bend(S, E, r2) - b0)
            agree.append(np.degrees(np.arccos(np.clip(d2 @ d3 / np.linalg.norm(d2) / np.linalg.norm(d3), -1, 1))))
            ratio.append(np.linalg.norm(d2) / np.linalg.norm(d3))
up, front, ysign, db2, agree, ratio = map(np.array, (up, front, ysign, db2, agree, ratio))
q = lambda x: f"median {np.median(x):.2f}, p5 {np.percentile(x,5):.2f}, p95 {np.percentile(x,95):.2f}"
print(f"frames: {len(up)}; torso-root body UP y-comp (upright => -1): {q(up)}; FRONT z-comp (facing camera => -1): {q(front)}")
print(f"arms: {len(db2)}; elbow local-y has the anatomical flexion sign: {ysign.mean()*100:.1f}%")
print(f"achieved bend change for +{DEG:.0f} deg local-y flex (R2): {q(db2)}")
print(f"R2 vs R3 wrist-motion direction disagreement (deg): {q(agree)}; magnitude ratio {q(ratio)}")
