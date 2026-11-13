import glob, warnings
import numpy as np
from scipy.spatial.transform import Rotation as R
warnings.filterwarnings("ignore")
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE, BodyFit
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics
sk = SMPLXSkeleton(model_path="models")
res_full, res_missing = [], []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); L = d["landmarks"].astype(float); V = d["visibility"]
    for t in range(L.shape[0]):
        lm = L[t]; ok = np.isfinite(lm).all(1) & (V[t] >= 0.5)
        try: fc = fit_body(lm, ok, sk)
        except ValueError: continue
        rc = reconstruct_body(fc, sk)
        axis = rc[11] - rc[12]; axis /= np.linalg.norm(axis)
        G2 = R.from_rotvec(axis * np.pi).as_matrix() @ R.from_rotvec(fc.global_orient).as_matrix()
        def resolve(G):
            rots = solve_swing_ik(np.where(ok[:, None], lm, np.nan), sk, SMPLX_ARM_BONES, global_orient=R.from_matrix(G).as_rotvec())
            posed = forward_kinematics(rots, sk)
            return reconstruct_body(BodyFit(rots[1:22].reshape(63), rots[0], rc[11] - posed[16]), sk)
        rc = resolve(R.from_rotvec(fc.global_orient).as_matrix())
        r2 = resolve(G2)
        allok = all(ok[m] for _, m in ARM_CORRESPONDENCE)
        err = max(np.linalg.norm(r2[m] - rc[m]) for _, m in ARM_CORRESPONDENCE if ok[m])
        (res_full if allok else res_missing).append(err)
print("frames with all 6 arm landmarks valid: n=%d, worst change after 180deg flip %.2e m" % (len(res_full), max(res_full)))
print("frames with some arm landmark invalid: n=%d, worst change %.2e m, median %.2e" % (len(res_missing), max(res_missing), np.median(res_missing)))
