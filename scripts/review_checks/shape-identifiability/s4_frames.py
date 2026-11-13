"""Q4: do many frames add shape information? Full real-SMPL-X-forward Jacobians.

For T frames with shared shape (10 betas [+ scale]) and per-frame free pose, the
number of shape directions the observations constrain is
    rank([J_shape | blockdiag(J_pose_t)]) - sum_t rank(J_pose_t),
computed via the Schur complement  M = sum_t J_s,t^T (I - P_pose,t) J_s,t.
"""
import os
import numpy as np, torch, smplx
from common import SNAP
torch.set_default_dtype(torch.float64)
model = smplx.create(f"{SNAP}/models", model_type="smplx", gender="neutral", use_pca=False,
                     flat_hand_mean=True, num_betas=10, batch_size=1, dtype=torch.float64)

ARM = [16, 17, 18, 19]
TORSO = [3, 6, 9, 13, 14]   # spine1-3, collars

def make_fn(free_joints, observed, rest_pose):
    def fn(shape, pose):
        betas, scale = shape[:10], shape[10]
        go = pose[0:3]; tr = pose[3:6]
        bp = rest_pose.clone()
        for i, j in enumerate(free_joints):
            bp[j - 1] = pose[6 + 3 * i: 9 + 3 * i]
        out = model(betas=betas[None], global_orient=go[None], body_pose=bp.reshape(1, 63))
        J = out.joints[0, :55]
        return (scale * J[observed] + tr).reshape(-1)  # scale about origin, then transl
    return fn

def analyse(T, free_joints, observed, with_scale, rng):
    nshape = 11 if with_scale else 10
    M = np.zeros((nshape, nshape)); pose_rank = 0; Jall = []
    for t in range(T):
        rest = torch.zeros(21, 3)
        for j in free_joints:
            rest[j - 1] = torch.tensor(rng.normal(0, 0.5 if j in ARM else 0.15, 3))
        beta0 = torch.tensor(rng.normal(0, 1, 10))
        shape = torch.cat([beta0 * 0 + torch.tensor(BASE_BETA), torch.ones(1)])
        pose = torch.cat([torch.tensor(rng.normal(0, 0.3, 3)), torch.zeros(3),
                          rest[[j - 1 for j in free_joints]].reshape(-1)])
        fn = make_fn(free_joints, observed, rest)
        Js, Jp = torch.autograd.functional.jacobian(fn, (shape, pose))
        Js = Js.numpy()[:, :nshape]; Jp = Jp.numpy()
        U, s, _ = np.linalg.svd(Jp, full_matrices=False)
        r = int((s > s[0] * 1e-9).sum()); pose_rank += r
        U = U[:, :r]
        R = Js - U @ (U.T @ Js)        # shape effect not explainable by this frame's pose
        M += R.T @ R
    ev = np.sort(np.linalg.eigvalsh(M))[::-1].clip(0)
    sv = np.sqrt(ev)
    ident = int((sv > sv[0] * 1e-6).sum())
    return ident, sv / np.sqrt(T) * 1000   # mm per unit beta, per-frame-normalised

BASE_BETA = np.zeros(10)
cases = [
    ("converter: 6 arm joints, pose=root+transl+16..19", ARM, [16, 17, 18, 19, 20, 21]),
    ("6 arm joints, pose also frees spine1-3+collars", ARM + TORSO, [16, 17, 18, 19, 20, 21]),
    ("6 arm joints + hips(1,2), converter pose", ARM, [1, 2, 16, 17, 18, 19, 20, 21]),
    ("6 arm joints + hips, spine+collars free", ARM + TORSO, [1, 2, 16, 17, 18, 19, 20, 21]),
]
for with_scale in (False, True):
    print(f"\n######## shape unknowns = 10 betas{' + scale' if with_scale else ''} ########")
    for name, free, obs in cases:
        for T in (1, 2, 5, 20):
            rng = np.random.default_rng(100 + T)
            ident, sv = analyse(T, free, obs, with_scale, rng)
            print(f"{name:<52} T={T:<3} identifiable shape dims={ident:<3} "
                  f"sv/sqrt(T) (mm per unit)={np.array2string(sv, precision=3, suppress_small=True, max_line_width=200)}")
