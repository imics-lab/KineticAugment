"""Check the coordinate convention the delivered converter actually produces on real clips.
MediaPipe world: y down, z away from camera (verified from data). SMPL-X: y up, z = body forward.
Question: after fit_body, where is the SMPL-X head relative to the shoulders, in the landmark frame?"""
import glob
import numpy as np
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

sk = SMPLXSkeleton(model_path="models")
go_deg, head_above, n = [], 0, 0
nose_err = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); L = d["landmarks"].astype(np.float64); V = d["visibility"]
    for fr, vis in zip(L, V):
        ok = np.isfinite(fr).all(1) & (vis >= 0.5)
        try:
            fit = fit_body(fr, ok, sk)
        except ValueError:
            continue
        rot = np.zeros((55, 3)); rot[0] = fit.global_orient; rot[1:22] = fit.body_pose.reshape(21, 3)
        J = forward_kinematics(rot, sk) + fit.transl
        n += 1
        go_deg.append(np.degrees(np.linalg.norm(fit.global_orient)))
        # landmark frame: y DOWN. head above shoulders <=> head_y < mean shoulder_y
        sh_y = 0.5 * (J[16, 1] + J[17, 1])
        head_above += int(J[15, 1] < sh_y)
        # pelvis vs MP hip-centre origin (hips at y~0, shoulders at y~-0.45)
        nose_err.append(np.linalg.norm(J[15] - fr[0]))
print(f"frames fitted: {n}")
print(f"global_orient angle (deg): median {np.median(go_deg):.1f}  p5 {np.percentile(go_deg,5):.1f}  p95 {np.percentile(go_deg,95):.1f}")
print(f"SMPL-X head joint ABOVE shoulders in landmark frame (y-down): {head_above}/{n} frames")
print(f"distance SMPL-X head joint -> MediaPipe nose: median {np.median(nose_err):.3f} m")
