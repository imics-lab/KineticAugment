"""N3 / R-missing: valid landmarks downstream of an invalid non-leaf landmark."""
import glob, numpy as np
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES
sk = SMPLXSkeleton(model_path="models")
rng = np.random.default_rng(2024)
cases = {"L wrist(15) invalid [leaf control]": 15, "L elbow(13) invalid, wrist valid": 13, "L shoulder(11) invalid": 11, "R elbow(14) invalid": 14}
res = {k: [] for k in cases}; fabricated = []
for trial in range(50):
    rot = np.zeros((55, 3))
    for j, *_ in SMPLX_ARM_BONES: rot[j] = rng.normal(0, 0.4, 3)
    J = forward_kinematics(rot, sk) + np.array([0.1, -0.3, 0.5])
    L = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE: L[m] = J[s]
    for name, bad in cases.items():
        v = np.isfinite(L).all(1); v[bad] = False
        out = reconstruct_body(fit_body(L, v, sk), sk)
        res[name].append(max(np.linalg.norm(out[m] - L[m]) for _, m in ARM_CORRESPONDENCE if v[m]))
        if bad == 15: fabricated.append(np.linalg.norm(out[15] - L[15]))
for k, v in res.items():
    v = np.array(v) * 1000; print(f"{k:<38} worst-valid-landmark error over 50 poses: median {np.median(v):.2f} mm, max {v.max():.2f} mm")
print(f"wrist invalid -> reconstruct_body still returns finite wrist; median {1000*np.median(fabricated):.1f} mm from truth")
# real-data occurrence
n = cnt = 0
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); ok = np.isfinite(d["landmarks"]).all(-1) & (d["visibility"] >= 0.5)
    for o in ok:
        n += 2
        cnt += int((not o[13] and o[15]) or (not o[11] and (o[13] or o[15]))) + int((not o[14] and o[16]) or (not o[12] and (o[14] or o[16])))
print(f"real data: orphan configurations in {cnt}/{n} arm-frames")
