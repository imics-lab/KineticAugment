"""A005: can the scored output be rebuilt from saved parameters alone?

1. Fit every real frame, save ONLY BodyFit fields to an npz (no landmarks, no transforms).
2. In this process, reload the npz and rebuild joints with the REAL SMPL-X forward pass (torch,
   betas from the file) -- not the project's FK -- then score against the landmarks.
3. Check whether reconstruct_body honours BodyFit.betas (it should if betas are part of the state).
4. Synthetic: truth root = 180 deg about x (the MediaPipe y-down / z-away convention) -- does the
   position metric notice that the fitted global_orient is ~180 deg wrong?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from common import *  # noqa
from kinetic_augment.body_model.converter import BodyFit

W = Path(__file__).parent
sk = SMPLXSkeleton(model_path="models")
clips = load_clips()

params, meta = [], []
for ci, (name, pose, oks, width) in enumerate(clips):
    for fi, (frame, ok) in enumerate(zip(pose, oks)):
        try:
            f = fit_body(frame, ok, sk)
        except ValueError:
            continue
        params.append(np.concatenate([f.global_orient, f.body_pose, f.transl, f.betas]))
        meta.append((ci, fi))
np.savez(W / "fits.npz", params=np.array(params), meta=np.array(meta))
del params

d = np.load(W / "fits.npz")
P = torch.tensor(d["params"], dtype=torch.float32)
with torch.no_grad():
    J = []
    for i in range(0, len(P), 64):
        b = P[i:i + 64]
        model_b = sk.model
        out = [model_b(global_orient=b[k:k+1, 0:3], body_pose=b[k:k+1, 3:66], transl=b[k:k+1, 66:69],
                       betas=b[k:k+1, 69:79], return_verts=False).joints[0].numpy() for k in range(len(b))]
        J += out
errs_smplx, errs_ours, maxdiff = [], [], 0.0
for (ci, fi), j in zip(d["meta"], J):
    name, pose, oks, width = clips[ci]
    frame, ok = pose[fi], oks[fi]
    p = d["params"][len(errs_smplx) and 0]  # placeholder, not used
    for m in MP:
        if ok[m]:
            errs_smplx.append(np.linalg.norm(j[SX[m]] - frame[m]) / width)
    f = fit_body(frame, ok, sk)
    rec = reconstruct_body(f, sk)
    for m in MP:
        maxdiff = max(maxdiff, float(np.linalg.norm(rec[m] - j[SX[m]])))
        if ok[m]:
            errs_ours.append(np.linalg.norm(rec[m] - frame[m]) / width)
print(f"frames serialised: {len(d['meta'])}, values per frame: {d['params'].shape[1]}")
print(f"direct error rebuilt by REAL SMPL-X forward from the saved npz: median {pct(errs_smplx):.3f}%  p95 {pct(errs_smplx,95):.3f}%")
print(f"direct error via reconstruct_body (project FK):               median {pct(errs_ours):.3f}%")
print(f"max |real SMPL-X joint - reconstruct_body| over all frames/arm joints: {maxdiff:.2e} m")

# 3. does reconstruct_body honour betas?
name, pose, oks, width = clips[0]
f = fit_body(pose[20], oks[20], sk)
f2 = BodyFit(f.body_pose, f.global_orient, f.transl, betas=np.r_[2.0, np.zeros(9)])
r1, r2 = reconstruct_body(f, sk), reconstruct_body(f2, sk)
with torch.no_grad():
    t = lambda a: torch.tensor(a, dtype=torch.float32)[None]
    jb = sk.model(global_orient=t(f2.global_orient), body_pose=t(f2.body_pose), transl=t(f2.transl),
                  betas=t(f2.betas), return_verts=False).joints[0].numpy()
print(f"\nreconstruct_body change when betas[0]: 0 -> 2  : {max(np.linalg.norm(r2[m]-r1[m]) for m in MP):.2e} m")
print(f"real SMPL-X forward change for the same params : {max(np.linalg.norm(jb[SX[m]]-r1[m]) for m in MP):.3e} m")

# 4. synthetic 180-degree x flip
rng = np.random.default_rng(0)
worst_pos, rootdiff = 0.0, []
for trial in range(50):
    truth = np.zeros((55, 3))
    truth[0] = np.array([np.pi, 0, 0])
    for jn in (16, 17, 18, 19):
        truth[jn] = rng.normal(0, 0.4, 3)
    jt = forward_kinematics(truth, sk)
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE:
        lm[m] = jt[s]
    v = np.isfinite(lm).all(1)
    fit = fit_body(lm, v, sk)
    out = reconstruct_body(fit, sk)
    worst_pos = max(worst_pos, max(np.linalg.norm(out[m] - lm[m]) for m in MP))
    Rt = Rotation.from_rotvec(truth[0]); Rf = Rotation.from_rotvec(fit.global_orient)
    rootdiff.append(np.degrees((Rf * Rt.inv()).magnitude()))
print(f"\nsynthetic, truth root = 180 deg about x (MediaPipe axis convention), 50 trials:")
print(f"  worst arm-joint position error: {worst_pos*1000:.2f} mm")
print(f"  fitted global_orient vs truth: median {np.median(rootdiff):.1f} deg (min {min(rootdiff):.1f})")
# same with 30 deg about y for reference (the doc's xfail case)
