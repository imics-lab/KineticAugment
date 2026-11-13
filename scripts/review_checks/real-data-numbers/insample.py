"""How much does the converter's per-frame rigid fit (now serialised as global_orient/transl) absorb?

 - translation only (rotation fixed at the IK frame, i.e. the identity)  -> per-frame 3 dof
 - per-clip rotation (one R per clip, LS over all frames) + per-frame translation
 - per-frame rigid (= converter 'direct')
 - per-frame similarity (= 'aligned')
 - leave-one-out on the alignment: IK uses all observed landmarks, but the rigid transform is
   fitted on the other observed joints and the held-out joint is scored.
Also: shoulder-anchor share of the direct error, and SMPL-X rest shoulder width vs observed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

from common import *  # noqa

sk = SMPLXSkeleton(model_path="models")
clips = load_clips()
rest = sk.rest_joints
print(f"SMPL-X rest shoulder width (joints 16-17): {np.linalg.norm(rest[16]-rest[17]):.4f} m; "
      f"observed per-clip median widths: {[round(w,4) for *_, w in clips]}")

E = {k: [] for k in ["transl_only", "clip_rot", "rigid", "sim", "loo_rigid", "rigid_same_joints_as_loo"]}
for name, pose, oks, width in clips:
    frames = []
    for frame, ok in zip(pose, oks):
        present = [m for m in MP if ok[m]]
        if len(present) < 3:
            continue
        pts, _ = ik_points(frame, ok, sk)
        src = np.array([pts[m] for m in present]); tgt = np.array([frame[m] for m in present])
        frames.append((src, tgt, present))
        E["transl_only"] += list(np.linalg.norm(src - src.mean(0) + tgt.mean(0) - tgt, axis=1) / width)
        for mode in ("rigid", "sim"):
            al, _ = align(src, tgt, mode)
            E[mode] += list(np.linalg.norm(al - tgt, axis=1) / width)
        if len(present) >= 4:
            for i in range(len(present)):
                keep = [k for k in range(len(present)) if k != i]
                _, R, t = similarity_transform(src[keep], tgt[keep], allow_scale=False)
                E["loo_rigid"].append(np.linalg.norm(R @ src[i] + t - tgt[i]) / width)
            al, _ = align(src, tgt, "rigid")
            E["rigid_same_joints_as_loo"] += list(np.linalg.norm(al - tgt, axis=1) / width)
    # one rotation per clip: Procrustes on centred stacks
    A = np.concatenate([s - s.mean(0) for s, _, _ in frames]); B = np.concatenate([t - t.mean(0) for _, t, _ in frames])
    _, R, _ = similarity_transform(A, B, allow_scale=False)
    for s, t, _ in frames:
        E["clip_rot"] += list(np.linalg.norm((s - s.mean(0)) @ R.T + t.mean(0) - t, axis=1) / width)
for k, v in E.items():
    print(f"{k:<26} median {pct(v):6.3f}%   p95 {pct(v,95):6.3f}%   n={len(v)}")
