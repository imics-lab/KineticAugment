"""Follow-ups: exact-collinear Kabsch, real-data swing angles / temporal jumps, head-above-shoulder,
exact zeros in real data, FK vs SMPL-X with nonzero betas."""
import glob
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

sk = SMPLXSkeleton(model_path="models")
C = np.diag([1.0, -1.0, -1.0])
t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)[None]

print("D'. EXACTLY collinear single arm (shoulder, elbow, wrist on one line), other arm unobserved")
for d in ([0.3, -0.9, 0.3], [0.0, -1.0, 0.0], [0.5, 0.2, 0.8], [1.0, 0.0, 0.0]):
    d = np.asarray(d) / np.linalg.norm(d)
    lm = np.full((33, 3), np.nan)
    p = np.array([0.18, -0.45, 0.05])
    lm[11], lm[13], lm[15] = p, p + 0.27 * d, p + 0.52 * d
    v = np.isfinite(lm).all(1)
    f = fit_body(lm, v, sk)
    o = reconstruct_body(f, sk)
    e = max(np.linalg.norm(o[i] - lm[i]) for i in (11, 13, 15))
    print(f"  dir {np.round(d,2)}: |global_orient| = {np.degrees(np.linalg.norm(f.global_orient)):7.2f} deg, "
          f"observed max err {1000*e:.1f} mm")

print("\nSwing angles and temporal parameter jumps on real data (current fit vs C-converted fit)")
names = {16: "L shoulder", 17: "R shoulder", 18: "L elbow", 19: "R elbow"}
for tag, M in (("current", np.eye(3)), ("with C", C)):
    angles = {j: [] for j in names}
    jumps = {j: [] for j in names}
    head_above = []
    for path in sorted(glob.glob("data/test_output/*_world.npz")):
        d = np.load(path)
        prev = None
        for lm, vis in zip(d["landmarks"].astype(np.float64), d["visibility"]):
            if not np.isfinite(lm).all():
                prev = None; continue
            try:
                f = fit_body(lm @ M.T, vis >= 0.5, sk)
            except ValueError:
                prev = None; continue
            bp = f.body_pose.reshape(21, 3)
            for j in names:
                angles[j].append(np.degrees(np.linalg.norm(bp[j - 1])))
                if prev is not None:
                    jumps[j].append(np.linalg.norm(bp[j - 1] - prev[j - 1]))
            prev = bp
            with torch.no_grad():
                J = sk.model(body_pose=t(f.body_pose), global_orient=t(f.global_orient), transl=t(f.transl),
                             betas=t(f.betas), return_verts=False).joints[0].numpy() @ M  # back to MP frame
            up = (lm[11] + lm[12]) / 2 - (lm[23] + lm[24]) / 2
            head_above.append(np.dot(J[15] - (lm[11] + lm[12]) / 2, up) > 0)
    print(f"  [{tag}] SMPL-X head joint above the observed shoulder midpoint: {100*np.mean(head_above):.1f}% of frames")
    for j, n in names.items():
        a, jm = np.asarray(angles[j]), np.asarray(jumps[j])
        print(f"  [{tag}] {n:<11} swing angle median {np.median(a):6.1f}  max {a.max():6.1f} deg;"
              f"  frame-to-frame |d rotvec| p99 {np.percentile(jm,99):.2f}  max {jm.max():.2f} rad")

print("\nExact zeros in real world landmarks (arm landmarks 11-16):")
tot = zeros = 0
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    L = np.load(path)["landmarks"][:, 11:17]
    tot += L.shape[0] * L.shape[1]
    zeros += int(np.sum(np.all(L == 0, axis=-1)))
print(f"  {zeros} of {tot} arm landmark rows are exactly (0,0,0)")

print("\nFK on a SHAPED SMPLXSkeleton vs real SMPL-X with the same betas (not covered by the suite):")
betas = np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0.2, 0, 0, 0, 0])
shaped = SMPLXSkeleton(model_path="models", betas=betas)
rng = np.random.default_rng(0)
go, bp = rng.normal(0, 0.3, 3), rng.normal(0, 0.3, 63)
rots = np.zeros((55, 3)); rots[0] = go; rots[1:22] = bp.reshape(21, 3)
with torch.no_grad():
    J = shaped.model(global_orient=t(go), body_pose=t(bp), betas=t(betas), return_verts=False).joints[0, :55].numpy()
print(f"  max |FK - SMPL-X| over 22 body joints: {np.abs(forward_kinematics(rots, shaped)[:22] - J[:22]).max():.2e} m")
