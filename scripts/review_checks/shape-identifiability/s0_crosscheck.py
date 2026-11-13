"""Sanity: the exact-linear model in common.py == the repo's SMPLXSkeleton and the
real smplx forward pass; posed bone lengths are pose-invariant; FD Jacobian matches."""
import os
import numpy as np, torch, smplx
from common import SNAP, OBS, NB, joints, lengths, length_jac
os.chdir(SNAP)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

rng = np.random.default_rng(0)
worst = 0
for _ in range(5):
    b = rng.normal(size=NB) * 2
    sk = SMPLXSkeleton(model_path="models", betas=b)
    worst = max(worst, np.abs(sk.rest_joints - joints(b)).max())
print(f"max |SMPLXSkeleton.rest_joints - common.joints| over 5 random betas: {worst:.2e} m")

# FD Jacobian via the repo skeleton (float32 inside smplx -> use 0.05 step)
b0 = np.zeros(NB); h = 0.05
def L_of(b):
    R = SMPLXSkeleton(model_path="models", betas=b).rest_joints
    return np.array([np.linalg.norm(R[c] - R[p]) for p, c, _ in OBS])
Jfd = np.stack([(L_of(b0 + h*e) - L_of(b0 - h*e)) / (2*h) for e in np.eye(NB)], 1)
Jan = length_jac(b0, with_scale=False)
print(f"max |J_fd - J_analytic| = {np.abs(Jfd-Jan).max()*1000:.4f} mm/sigma (|J| max {np.abs(Jan).max()*1000:.2f})")

# posed lengths through the real smplx forward are pose-invariant
model = smplx.create("models", model_type="smplx", gender="neutral", use_pca=False,
                     flat_hand_mean=True, num_betas=10)
b = rng.normal(size=NB)
dev = 0
for _ in range(20):
    bp = torch.tensor(rng.normal(0, 0.5, (1, 63)), dtype=torch.float32)
    go = torch.tensor(rng.normal(0, 0.5, (1, 3)), dtype=torch.float32)
    with torch.no_grad():
        J = model(betas=torch.tensor(b[None], dtype=torch.float32), body_pose=bp, global_orient=go).joints[0, :55].numpy()
    Lp = np.array([np.linalg.norm(J[c] - J[p]) for p, c, _ in OBS])
    d = np.abs(Lp - lengths(b)); dev = max(dev, d[:4].max()); devsw = max(globals().get('devsw', 0), d[4])
    globals()['devsw'] = devsw
print(f"max |posed - rest| over 20 random full-body poses: 4 arm lengths {dev*1000:.5f} mm; shoulder width {devsw*1000:.1f} mm")
print("  (shoulder width uses two different branches: it is invariant only because collars/spine are rigid chains;"
      " check separately below)")
# shoulder width under collar motion only
for ang in (0.0, 0.2, 0.4):
    bp = torch.zeros(1, 63); bp[0, (13-1)*3 + 2] = ang   # left collar about z
    with torch.no_grad():
        J = model(betas=torch.zeros(1, 10), body_pose=bp).joints[0, :55].numpy()
    print(f"  left-collar z-rot {ang:.1f} rad: shoulder width = {np.linalg.norm(J[16]-J[17])*1000:.2f} mm")
