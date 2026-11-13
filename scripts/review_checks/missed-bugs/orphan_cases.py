"""
fit_body: a landmark whose bone could not be solved (because its PARENT landmark is invalid) is
still used as a rigid-fit correspondence, posed at the rest direction. Its invalid neighbour
therefore DOES influence the fit, contradicting 'Invalid landmarks never influence the fit'.
"""
import glob
import numpy as np
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES
sk = SMPLXSkeleton(model_path="models")
SW = np.linalg.norm(sk.rest_joints[16] - sk.rest_joints[17])

def synth(seed):
    rng = np.random.default_rng(seed)
    r = np.zeros((55, 3))
    for j, *_ in SMPLX_ARM_BONES:
        r[j] = rng.normal(0, 0.3, 3)
    J = forward_kinematics(r, sk) + np.array([0.1, 0.2, 0.3])
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lm[m] = J[s]
    return lm, np.isfinite(lm).all(1)

cases = {
    "L elbow invalid (wrist valid)":      ([13], [11, 12, 14, 15, 16]),
    "L shoulder invalid (elbow+wrist valid)": ([11], [12, 13, 14, 15, 16]),
    "L wrist invalid (control: leaf)":    ([15], [11, 12, 13, 14, 16]),
}
for label, (drop, scored) in cases.items():
    worst = []
    for seed in range(50):
        lm, v = synth(seed); v = v.copy(); v[drop] = False
        out = reconstruct_body(fit_body(lm, v, sk), sk)
        worst.append(max(np.linalg.norm(out[m] - lm[m]) for m in scored))
    worst = np.array(worst) * 1000
    print(f"  {label:<40} max error on the VALID landmarks over 50 exact synthetic poses: "
          f"median {np.median(worst):7.2f} mm, max {worst.max():7.2f} mm ({100*worst.max()/1000/SW:.0f}% SMPL-X shoulder width)")

print("\nUnobserved in-model landmark is fabricated, not NaN (and with no observed/defaulted flag):")
lm, v = synth(3); v = v.copy(); v[15] = False
out = reconstruct_body(fit_body(lm, v, sk), sk)
print(f"  wrist 15 invalid -> reconstruct_body()[15] = {np.round(out[15],3)} (finite: {np.isfinite(out[15]).all()}), "
      f"{1000*np.linalg.norm(out[15]-lm[15]):.0f} mm from the true wrist")

print("\nHow often does each orphan case occur in the real clips (vis >= thr)?")
for thr in (0.5, 0.9):
    cnt = {"elbow invalid & wrist valid": 0, "shoulder invalid & elbow valid": 0, "wrist invalid": 0}; n = 0
    for p in sorted(glob.glob("data/test_output/*_world.npz")):
        d = np.load(p); vis = d["visibility"]; ok = np.isfinite(d["landmarks"]).all(-1) & (vis >= thr)
        for sh, el, wr in ((11, 13, 15), (12, 14, 16)):
            cnt["elbow invalid & wrist valid"] += int(np.sum(~ok[:, el] & ok[:, wr]))
            cnt["shoulder invalid & elbow valid"] += int(np.sum(~ok[:, sh] & ok[:, el]))
            cnt["wrist invalid"] += int(np.sum(~ok[:, wr]))
            n += len(ok)
    print(f"  threshold {thr}: " + ", ".join(f"{k}: {v}/{n} arm-frames" for k, v in cnt.items()))
