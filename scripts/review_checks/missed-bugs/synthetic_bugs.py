"""Synthetic demonstrations of converter / IK / alignment edge-case bugs."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.alignment import similarity_transform
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, _swing_rotation, solve_swing_ik

sk = SMPLXSkeleton(model_path="models")
ARM = [mp for _, mp in ARM_CORRESPONDENCE]


def lm_from(skel, rots, offset=(0, 0, 0)):
    J = forward_kinematics(rots, skel) + np.asarray(offset)
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lm[m] = J[s]
    return lm, np.isfinite(lm).all(1)


def arm_pose(seed, sd=0.3):
    rng = np.random.default_rng(seed)
    r = np.zeros((55, 3))
    for j, *_ in SMPLX_ARM_BONES:
        r[j] = rng.normal(0, sd, 3)
    return r


def err(out, lm, idx):
    return {i: round(1000 * float(np.linalg.norm(out[i] - lm[i])), 2) for i in idx}


print("=" * 72)
print("B. Valid wrist whose ELBOW is invalid: wrist still used as a Kabsch correspondence")
print("=" * 72)
worst_b = []
for seed in range(20):
    lm, val = lm_from(sk, arm_pose(seed), offset=(0.1, 0.2, 0.3))
    v2 = val.copy(); v2[13] = False                      # left elbow invalid, left wrist valid
    out = reconstruct_body(fit_body(lm, v2, sk), sk)
    v3 = v2.copy(); v3[15] = False                       # also drop the wrist (the fix)
    out3 = reconstruct_body(fit_body(lm, v3, sk), sk)
    e_valid = max(np.linalg.norm(out[i] - lm[i]) for i in (11, 12, 14, 16, 15))
    e_fixed = max(np.linalg.norm(out3[i] - lm[i]) for i in (11, 12, 14, 16))
    worst_b.append((e_valid, e_fixed))
    if seed == 0:
        print("  seed 0, per-landmark error mm, elbow invalid (wrist kept):", err(out, lm, (11, 12, 14, 16, 15)))
        print("  seed 0, per-landmark error mm, elbow+wrist invalid       :", err(out3, lm, (11, 12, 14, 16)))
wb = np.array(worst_b)
print(f"  20 seeds: max error over the VALID landmarks {1000*wb[:,0].min():.1f}..{1000*wb[:,0].max():.1f} mm "
      f"(median {1000*np.median(wb[:,0]):.1f}); dropping the orphaned wrist too: max {1000*wb[:,1].max():.2e} mm")

print()
print("=" * 72)
print("C. A valid landmark exactly at the origin: IK drops it, Kabsch still uses it")
print("=" * 72)
truth = arm_pose(3)
J = forward_kinematics(truth, sk)
lm, val = lm_from(sk, truth, offset=-J[16])                # left shoulder lands exactly on (0,0,0)
print("  left shoulder landmark:", lm[11], " validity:", bool(val[11]))
out = reconstruct_body(fit_body(lm, val, sk), sk)
print("  error mm (should be ~0, the input is exact SMPL-X):", err(out, lm, ARM))
lm2, val2 = lm_from(sk, truth, offset=-J[16] + 1e-9)       # 1 nm away from the origin
out2 = reconstruct_body(fit_body(lm2, val2, sk), sk)
print("  same pose shifted by 1e-9 m                     :", err(out2, lm2, ARM))

print()
print("=" * 72)
print("D. Three collinear correspondences: Kabsch rotation about the line is arbitrary")
print("=" * 72)
truth = np.zeros((55, 3))
truth[16] = [0.0, 0.0, -0.9]                              # left arm straight, angled down
lm, val = lm_from(sk, truth, offset=(0.0, 0.3, 0.0))
v = val.copy(); v[[12, 14, 16]] = False                   # right arm and right shoulder unobserved
rng = np.random.default_rng(0)
gos = []
for k in range(6):
    noisy = lm + rng.normal(0, 1e-4, lm.shape)            # 0.1 mm jitter
    f = fit_body(noisy, v, sk)
    gos.append(Rotation.from_rotvec(f.global_orient))
    o = reconstruct_body(f, sk)
    if k == 0:
        print("  observed-landmark error mm:", err(o, lm, (11, 13, 15)))
spread = max((a.inv() * b).magnitude() for a in gos for b in gos)
print(f"  6 fits with 0.1 mm noise: max pairwise global_orient disagreement {np.degrees(spread):.1f} deg")
print("  (fit_body accepts this frame: 3 correspondences, no rank / conditioning check)")

print()
print("=" * 72)
print("E. betas: BodyFit.betas is never read by reconstruct_body, and a shaped skeleton is not recorded")
print("=" * 72)
lm, val = lm_from(sk, arm_pose(5), offset=(0.0, 0.0, 0.0))
fit = fit_body(lm, val, sk)
a = reconstruct_body(fit, sk)
fit.betas = np.array([3.0, -2.0, 0, 0, 0, 0, 0, 0, 0, 0])
b = reconstruct_body(fit, sk)
t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)[None]
with torch.no_grad():
    jm = sk.model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient), transl=t(fit.transl),
                  betas=t(fit.betas), return_verts=False).joints[0].numpy()
print(f"  set fit.betas=[3,-2,0..]: reconstruct_body changes by {np.nanmax(np.abs(a-b)):.1e} m; "
      f"real SMPL-X with the same params differs from reconstruct_body by "
      f"{1000*max(np.linalg.norm(jm[s]-b[m]) for s,m in ARM_CORRESPONDENCE):.1f} mm")
betas = np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0, 0, 0, 0, 0])
shaped = SMPLXSkeleton(model_path="models", betas=betas)
lm, val = lm_from(shaped, arm_pose(6))
fit = fit_body(lm, val, shaped)
ours = reconstruct_body(fit, shaped)
with torch.no_grad():
    jm = shaped.model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient), transl=t(fit.transl),
                      betas=t(fit.betas), return_verts=False).joints[0].numpy()
    jm_true = shaped.model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient), transl=t(fit.transl),
                           betas=t(betas), return_verts=False).joints[0].numpy()
print(f"  shaped skeleton: fit.betas = {fit.betas[:3]}...; reconstruct_body vs SMPL-X(fit params): "
      f"{1000*max(np.linalg.norm(jm[s]-ours[m]) for s,m in ARM_CORRESPONDENCE):.1f} mm; "
      f"vs SMPL-X(fit params but true betas): {1000*max(np.linalg.norm(jm_true[s]-ours[m]) for s,m in ARM_CORRESPONDENCE):.2e} mm")

print()
print("=" * 72)
print("H. Swing near 180 degrees: numeric accuracy and parameter continuity")
print("=" * 72)
rng = np.random.default_rng(1)
worst = 0.0
for _ in range(20000):
    a = rng.normal(size=3); a /= np.linalg.norm(a)
    eps = 10 ** rng.uniform(-14, -2)
    b = -a + eps * rng.normal(size=3); b /= np.linalg.norm(b)
    R = _swing_rotation(a, b)
    worst = max(worst, np.linalg.norm(R @ a - b))
print(f"  20000 near-antiparallel pairs: max |R a - b| = {worst:.2e}")
# continuity: a forearm sweeping through the antipode of the rest direction
a = np.array([1.0, 0, 0])
prev = None; jumps = []
for th in np.linspace(-1e-3, 1e-3, 11):
    b = np.array([-np.cos(th), np.sin(th) * 0.0 + 1e-6, np.sin(th)]); b /= np.linalg.norm(b)
    rv = Rotation.from_matrix(_swing_rotation(a, b)).as_rotvec()
    if prev is not None:
        jumps.append(np.linalg.norm(rv - prev))
    prev = rv
print(f"  direction moving 0.2 mrad in total through the antipode: max step in rotvec = {max(jumps):.2f} rad")
