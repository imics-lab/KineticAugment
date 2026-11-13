"""Is the converter's SMPL-X torso oriented consistently with the observed (unscored) torso landmarks?

MediaPipe world landmarks: origin at hip centre; check axis directions from the data itself
(nose vs shoulders vs hips). Then compare where the converter's fitted SMPL-X puts its
head/neck/pelvis joints relative to the observed nose/hips.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from common import *  # noqa

sk = SMPLXSkeleton(model_path="models")
clips = load_clips()
print("MP world axis check (median over frames, metres):")
for name, pose, ok, width in clips:
    nose = pose[:, 0]; sh = (pose[:, 11] + pose[:, 12]) / 2; hip = (pose[:, 23] + pose[:, 24]) / 2
    print(f"  {name}: nose-y {np.median(nose[:,1]):+.3f}  shoulder-mid-y {np.median(sh[:,1]):+.3f}  hip-mid-y {np.median(hip[:,1]):+.3f}"
          f" | nose-z {np.median(nose[:,2]):+.3f} shoulder-z {np.median(sh[:,2]):+.3f}"
          f" | Lsh-x {np.median(pose[:,11,0]):+.3f} Rsh-x {np.median(pose[:,12,0]):+.3f}")

# converter output: where are SMPL-X head (15), neck (12), pelvis (0) relative to shoulder midpoint?
up_dot, head_rel, ang_from_I, ang_from_flip = [], [], [], []
model = sk.model
flip = np.diag([1.0, -1.0, -1.0])
for name, pose, oks, width in clips:
    for frame, ok in zip(pose, oks):
        if sum(ok[m] for m in MP) < 3:
            continue
        fit = fit_body(frame, ok, sk)
        rots = np.zeros((55, 3)); rots[0] = fit.global_orient; rots[1:22] = fit.body_pose.reshape(21, 3)
        J = forward_kinematics(rots, sk) + fit.transl
        shmid = (J[16] + J[17]) / 2
        # SMPL-X "up" of torso in MP frame: neck - pelvis
        up_smplx = J[12] - J[0]; up_smplx /= np.linalg.norm(up_smplx)
        up_obs = (frame[11] + frame[12]) / 2 - (frame[23] + frame[24]) / 2; up_obs /= np.linalg.norm(up_obs)
        up_dot.append(float(up_smplx @ up_obs))
        head_rel.append(float((J[15] - shmid)[1]))
        Rg = Rotation.from_rotvec(fit.global_orient).as_matrix()
        ang_from_I.append(np.degrees(Rotation.from_matrix(Rg).magnitude()))
        ang_from_flip.append(np.degrees(Rotation.from_matrix(Rg @ flip.T).magnitude()))
up_dot = np.array(up_dot)
print(f"\ncos(SMPL-X pelvis->neck , observed hip-mid->shoulder-mid): median {np.median(up_dot):+.3f}  min {up_dot.min():+.3f} max {up_dot.max():+.3f}")
print(f"SMPL-X head joint y minus shoulder-mid y (MP frame, +y=down): median {np.median(head_rel):+.3f} m")
print(f"|global_orient| deg: median {np.median(ang_from_I):.1f};  angle from the y/z flip diag(1,-1,-1): median {np.median(ang_from_flip):.1f}")

# Real SMPL-X forward on one frame's params: does the head end up on the opposite side of the shoulders from the nose?
name, pose, oks, width = clips[0]
fr, ok = pose[10], oks[10]
fit = fit_body(fr, ok, sk)
with torch.no_grad():
    o = model(body_pose=torch.tensor(fit.body_pose[None], dtype=torch.float32),
              global_orient=torch.tensor(fit.global_orient[None], dtype=torch.float32),
              transl=torch.tensor(fit.transl[None], dtype=torch.float32),
              betas=torch.zeros(1, 10))
J = o.joints[0].numpy().astype(float)
rec = reconstruct_body(fit, sk)
print(f"\nclip {name} frame 10, real SMPL-X forward:")
print(f"  max |SMPL-X forward - reconstruct_body| over arm joints: {max(np.linalg.norm(J[SX[m]] - rec[m]) for m in MP):.2e} m")
print(f"  observed nose y {fr[0,1]:+.3f}, observed shoulder-mid y {(fr[11,1]+fr[12,1])/2:+.3f}, observed hip-mid y {(fr[23,1]+fr[24,1])/2:+.3f}")
print(f"  SMPL-X head y {J[15,1]:+.3f}, SMPL-X shoulder-mid y {(J[16,1]+J[17,1])/2:+.3f}, SMPL-X pelvis y {J[0,1]:+.3f}")
print(f"  SMPL-X wrist y L {J[20,1]:+.3f} R {J[21,1]:+.3f}; observed wrist y L {fr[15,1]:+.3f} R {fr[16,1]:+.3f}")
# Elbow flexion/shoulder angles in the SMPL-X canonical frame: are arms raised overhead relative to the torso?
rots = np.zeros((55, 3)); rots[1:22] = fit.body_pose.reshape(21, 3)
Jc = forward_kinematics(rots, sk)
print(f"  canonical frame (global_orient=0): head y {Jc[15,1]:+.3f}, shoulder y {Jc[16,1]:+.3f}, elbow y {Jc[18,1]:+.3f}, wrist y {Jc[20,1]:+.3f}  (+y=up in SMPL-X)")
