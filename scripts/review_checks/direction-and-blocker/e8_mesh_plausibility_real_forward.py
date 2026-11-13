"""E8: confirm E7 with the REAL SMPL-X forward pass (not the repo's FK): run smplx on each
fit_body() result and compare unconstrained body parts with MediaPipe landmarks the fit ignores.
  - head joint (15) vs MediaPipe nose (0): is the head above or below the shoulders?
  - pelvis (0) / hips (1,2) vs MediaPipe hip landmarks (23,24) (world origin = hip centre)
MediaPipe world is y-down, so 'above the shoulders' means smaller y."""
import warnings
import numpy as np, torch
warnings.simplefilter("ignore")
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE

sk = SMPLXSkeleton(model_path="models")
t32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
nose_above, head_above, head_err, hip_err, arm_check = [], [], [], [], []
for c in ["01986", "01987", "01988", "01991", "01992"]:
    d = np.load(f"data/test_output/{c}_world.npz")
    L, V = d["landmarks"].astype(np.float64), d["visibility"]
    for t in range(len(L)):
        ok = (V[t] > 0.5) & np.isfinite(L[t]).all(1)
        try:
            f = fit_body(L[t], ok, sk)
        except ValueError:
            continue
        with torch.no_grad():
            J = sk.model(body_pose=t32(f.body_pose), global_orient=t32(f.global_orient),
                         transl=t32(f.transl), betas=t32(f.betas), return_verts=False).joints[0].numpy()
        sw = np.linalg.norm(L[t, 11] - L[t, 12])
        sh_mid_mp = (L[t, 11] + L[t, 12]) / 2; sh_mid_sx = (J[16] + J[17]) / 2
        nose_above.append(L[t, 0, 1] < sh_mid_mp[1])
        head_above.append(J[15, 1] < sh_mid_sx[1])
        head_err.append(np.linalg.norm(J[15] - L[t, 0]) / sw * 100)
        hip_mid_mp = (L[t, 23] + L[t, 24]) / 2
        hip_err.append(np.linalg.norm((J[1] + J[2]) / 2 - hip_mid_mp) / sw * 100)
        rb = reconstruct_body(f, sk)
        arm_check.append(max(np.linalg.norm(rb[m] - J[s]) for s, m in ARM_CORRESPONDENCE))
na, ha, he, hp, ac = map(np.array, (nose_above, head_above, head_err, hip_err, arm_check))
print(f"frames: {len(na)}; real-forward vs reconstruct_body on arm joints: max {ac.max():.2e} m")
print(f"MediaPipe nose above shoulders (y-down frame): {na.mean()*100:.1f}% of frames")
print(f"SMPL-X head joint above its shoulders, same frame:  {ha.mean()*100:.1f}% of frames")
print(f"|SMPL-X head - MP nose|: median {np.median(he):.0f}%SW  (a correct torso would be ~ <50%SW)")
print(f"|SMPL-X hip centre - MP hip centre|: median {np.median(hp):.0f}%SW")
