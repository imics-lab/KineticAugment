"""(a) Is the current converter's SMPL-X body upside down in the MediaPipe world frame?
   (b) shoulder-only vs optimised root on the SAME frames; shoulder-root + rigid post-fit."""
import glob, sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, sys.argv[1])
from t2_real_root import (sk, rest, fit_with_root, root_shoulder_only, root_hip, opt_root, CORR, NAMES, VIS)
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik
from kinetic_augment.body_model.converter import fit_body
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity

pel_y, head_y, sh_y, sh_rot_cur, sh_rot_hip = [], [], [], [], []
e_sh, e_opt, e_shrig = [], [], []
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); P = d["landmarks"].astype(np.float64); V = d["visibility"]
    okall = np.isfinite(P).all(-1) & (V >= VIS)
    both = okall[:, 11] & okall[:, 12]
    width = float(np.median(np.linalg.norm(P[both, 11] - P[both, 12], axis=-1)))
    for f, (L, ok) in enumerate(zip(P, okall)):
        if not (ok[11] and ok[12] and ok[0]) or sum(ok[m] for m in NAMES) < 3: continue
        fit = fit_body(L, ok, sk)
        rots = np.zeros((55, 3)); rots[0] = fit.global_orient; rots[1:22] = fit.body_pose.reshape(21, 3)
        J = forward_kinematics(rots, sk) + fit.transl
        pel_y.append(J[0, 1]); head_y.append(J[15, 1]); sh_y.append((J[16, 1] + J[17, 1]) / 2)
        sh_rot_cur += [np.degrees(np.linalg.norm(rots[16])), np.degrees(np.linalg.norm(rots[17]))]
        _, rh = fit_with_root(L, ok, root_hip(L))
        sh_rot_hip += [np.degrees(np.linalg.norm(rh[16])), np.degrees(np.linalg.norm(rh[17]))]
        # shoulder-root then rigid post-fit (what Step 8 would give if the converter's rigid fit is kept)
        masked = np.where(ok[:, None], L, np.nan)
        rr = solve_swing_ik(masked, sk, SMPLX_ARM_BONES, global_orient=Rotation.from_matrix(root_shoulder_only(L)).as_rotvec())
        JJ = forward_kinematics(rr, sk)
        pairs = [(s, m) for s, m in CORR if ok[m]]
        src = np.array([JJ[s] for s, _ in pairs]); tgt = np.array([L[m] for _, m in pairs])
        _, R, t = similarity_transform(src, tgt, allow_scale=False)
        out = apply_similarity(src, 1.0, R, t)
        e_shrig += list(np.linalg.norm(out - tgt, axis=1) / width)
        if f % 4 == 0:
            o, _ = fit_with_root(L, ok, root_shoulder_only(L))
            e_sh += [np.linalg.norm(o[m] - L[m]) / width for m in o]
            Ro = opt_root(L, ok, root_hip(L)); o, _ = fit_with_root(L, ok, Ro)
            e_opt += [np.linalg.norm(o[m] - L[m]) / width for m in o]

print("(a) Current converter output, MediaPipe world frame (y DOWN; observed nose y ~ -0.6, hip centre y = 0, shoulders y ~ -0.45):")
print(f"    fitted SMPL-X pelvis y: median {np.median(pel_y):+.3f} m; head(15) y: median {np.median(head_y):+.3f}; shoulder-mid y: median {np.median(sh_y):+.3f}")
print(f"    shoulder local rotation magnitude: current converter median {np.median(sh_rot_cur):.1f} deg (p5 {np.percentile(sh_rot_cur,5):.1f}, p95 {np.percentile(sh_rot_cur,95):.1f});"
      f" with shoulders+hip root median {np.median(sh_rot_hip):.1f} deg (p5 {np.percentile(sh_rot_hip,5):.1f}, p95 {np.percentile(sh_rot_hip,95):.1f})")
print("(b) Same subsample (every 4th frame):")
print(f"    shoulder-only root + translation: median {np.median(e_sh)*100:.2f}%  p95 {np.percentile(e_sh,95)*100:.2f}%  n={len(e_sh)}")
print(f"    3-dof optimised root + translation: median {np.median(e_opt)*100:.2f}%  p95 {np.percentile(e_opt,95)*100:.2f}%")
print(f"    all frames: shoulder-only root + IK + RIGID post-fit: median {np.median(e_shrig)*100:.2f}%  p95 {np.percentile(e_shrig,95)*100:.2f}%  n={len(e_shrig)}")
