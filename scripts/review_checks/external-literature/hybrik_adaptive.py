"""Prior-art test: HybrIK (Li et al., CVPR 2021) distinguishes 'naive' IK (bone direction =
observed child - observed parent; error accumulates down the chain) from 'adaptive' IK
(direction = observed child - *reconstructed* parent; per-joint error does not accumulate).
The snapshot's solve_swing_ik is the naive form. This measures, on the stored real clips and
with the snapshot's own fit_body/reconstruct_body, what the adaptive form changes.
Same metric as scripts/measure_round_trip.py (direct, % per-clip median shoulder width)."""
import glob, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE, BodyFit
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.alignment import similarity_transform
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik

skel = SMPLXSkeleton(model_path="models")
NAMES = [11, 12, 13, 14, 15, 16]
root = skel.rest_joints[0]

def rigid_fit(rotations, frame, usable):
    posed = forward_kinematics(rotations, skel)
    pairs = [(s, m) for s, m in ARM_CORRESPONDENCE if usable[m]]
    src = np.array([posed[s] for s, _ in pairs]); tgt = np.array([frame[m] for _, m in pairs])
    _, R, t = similarity_transform(src, tgt, allow_scale=False)
    return R, t

def to_fit(rotations, R, t):
    rr = R @ Rotation.from_rotvec(rotations[0]).as_matrix()
    return BodyFit(body_pose=rotations[1:22].reshape(63).copy(),
                   global_orient=Rotation.from_matrix(rr).as_rotvec(), transl=R @ root + t - root)

def fit_adaptive(frame, ok, iters=3):
    usable = ok & np.isfinite(frame).all(axis=1)
    if sum(usable[m] for _, m in ARM_CORRESPONDENCE) < 3: raise ValueError
    masked = np.where(usable[:, None], frame, np.nan)
    rot = solve_swing_ik(masked, skel, SMPLX_ARM_BONES)          # naive init
    R, t = rigid_fit(rot, frame, usable)
    for _ in range(iters):
        local = (masked - t) @ R                                  # landmarks into skeleton frame (R^T(x-t))
        # pass 1: shoulders replaced by reconstructed shoulders
        lm = local.copy(); posed = forward_kinematics(rot, skel)
        for j, _, mp_p, _ in SMPLX_ARM_BONES:
            if j in (16, 17) and usable[mp_p]: lm[mp_p] = posed[j]
        rot = solve_swing_ik(lm, skel, SMPLX_ARM_BONES)
        # pass 2: elbows replaced by reconstructed elbows
        posed = forward_kinematics(rot, skel)
        for j, _, mp_p, _ in SMPLX_ARM_BONES:
            if j in (18, 19) and usable[mp_p]: lm[mp_p] = posed[j]
        rot = solve_swing_ik(lm, skel, SMPLX_ARM_BONES)
        R, t = rigid_fit(rot, frame, usable)
    return to_fit(rot, R, t)

res = {"naive": {m: [] for m in NAMES}, "adaptive": {m: [] for m in NAMES}}
mm = {"naive": [], "adaptive": []}
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); pose = d["landmarks"].astype(float)
    ok_all = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
    both = ok_all[:, 11] & ok_all[:, 12]
    width = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
    for frame, ok in zip(pose, ok_all):
        for name, fn in (("naive", lambda f, o: fit_body(f, o, skel)), ("adaptive", fit_adaptive)):
            try: out = reconstruct_body(fn(frame, ok), skel)
            except ValueError: continue
            for m in NAMES:
                if ok[m]:
                    e = np.linalg.norm(out[m] - frame[m]); res[name][m].append(e / width); mm[name].append(e)
for name in res:
    allv = np.concatenate([res[name][m] for m in NAMES])
    print(f"{name:9s} pooled direct median {np.median(allv)*100:.2f}%  p95 {np.percentile(allv,95)*100:.2f}%  "
          f"median {np.median(mm[name])*1000:.1f} mm  | per joint median %: " +
          " ".join(f"{m}:{np.median(res[name][m])*100:.2f}" for m in NAMES))
