"""Independent checks: real SMPL-X serialization, betas side channel, torso orientation,
synthetic 180-deg root blindness, error vs landmark count, Step-4 float32 path."""
import glob
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES

MP = [11, 12, 13, 14, 15, 16]
sk = SMPLXSkeleton(model_path="models")
model = sk.model


def smplx_joints(fit, betas=None):
    b = np.zeros(10) if betas is None else betas
    with torch.no_grad():
        out = model(body_pose=torch.tensor(fit.body_pose[None], dtype=torch.float32),
                    global_orient=torch.tensor(fit.global_orient[None], dtype=torch.float32),
                    transl=torch.tensor(fit.transl[None], dtype=torch.float32),
                    betas=torch.tensor(b[None], dtype=torch.float32))
    return out.joints[0, :55].numpy().astype(np.float64)


def full_fk(fit, skel):
    rots = np.zeros((55, 3)); rots[0] = fit.global_orient; rots[1:22] = fit.body_pose.reshape(21, 3)
    return forward_kinematics(rots, skel) + fit.transl


# ---- 1. real-data serialization through real SMPL-X; torso orientation ----
errs, maxdiff, n_by_count = [], 0.0, {}
flip = np.diag([1.0, -1.0, -1.0])
go_from_id, go_from_flip, cos_torso, head_below, nose_side = [], [], [], [], []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f)
    pose = d["landmarks"].astype(np.float64)
    ok_all = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
    both = ok_all[:, 11] & ok_all[:, 12]
    W = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
    for fr, ok in zip(pose, ok_all):
        try:
            fit = fit_body(fr, ok, sk)
        except ValueError:
            continue
        # serialize to plain numbers and back
        blob = np.concatenate([fit.body_pose, fit.global_orient, fit.transl, fit.betas]).astype(np.float64)
        assert blob.shape == (79,)
        J = smplx_joints(fit)
        rb = reconstruct_body(fit, sk)
        pres = [m for m in MP if ok[m]]
        for s_, m in ARM_CORRESPONDENCE:
            maxdiff = max(maxdiff, np.linalg.norm(J[s_] - rb[m]))
            if ok[m]:
                errs.append(np.linalg.norm(J[s_] - fr[m]) / W)
        n = len(pres)
        n_by_count.setdefault(n, []).extend([np.linalg.norm(rb[m] - fr[m]) / W for m in pres])
        R = Rot.from_rotvec(fit.global_orient).as_matrix()
        go_from_id.append(np.degrees(Rot.from_matrix(R).magnitude()))
        go_from_flip.append(np.degrees(Rot.from_matrix(flip.T @ R).magnitude()))
        P = full_fk(fit, sk)
        hipmid = (fr[23] + fr[24]) / 2; shmid = (fr[11] + fr[12]) / 2
        v_fit = P[12] - P[0]; v_obs = shmid - hipmid
        cos_torso.append(v_fit @ v_obs / np.linalg.norm(v_fit) / np.linalg.norm(v_obs))
        # head relative to shoulder midpoint along observed "up" (hip->shoulder)
        up = v_obs / np.linalg.norm(v_obs)
        fit_shmid = (P[16] + P[17]) / 2
        head_below.append(-(P[15] - fit_shmid) @ up)
        nose_side.append((fr[0] - shmid) @ up)
p = lambda v, q=50: np.percentile(v, q)
print(f"[serialize] real SMPL-X forward on 79 saved numbers: direct median {100*p(errs):.3f}%  p95 {100*p(errs,95):.3f}%  n={len(errs)}  max|smplx-reconstruct_body| {maxdiff:.2e} m")
print("[count] direct median by #valid arm landmarks:", {k: (round(100 * p(v), 2), len(v)) for k, v in sorted(n_by_count.items())})
print(f"[torso] |global_orient| from identity median {p(go_from_id):.1f} deg; from x-flip median {p(go_from_flip):.1f} (min {np.min(go_from_flip):.1f})")
print(f"[torso] cos(fit pelvis->neck, obs hipmid->shouldermid) median {p(cos_torso):.3f} max {np.max(cos_torso):.3f}")
print(f"[torso] fitted head distance BELOW fitted shoulder-mid along observed up: median {p(head_below):.3f} m; observed nose ABOVE shoulder-mid median {p(nose_side):.3f} m")

# ---- 2. synthetic 180-deg root about x: arm metric blindness ----
rng = np.random.default_rng(3)
worst, rooterr = 0.0, []
for t in range(50):
    truth = np.zeros((55, 3)); truth[0] = [np.pi, 0, 0]
    for j, *_ in SMPLX_ARM_BONES:
        truth[j] = rng.normal(0, 0.4, 3)
    P = forward_kinematics(truth, sk)
    lm = np.full((33, 3), np.nan)
    for s_, m in ARM_CORRESPONDENCE:
        lm[m] = P[s_]
    ok = np.isfinite(lm).all(1)
    fit = fit_body(lm, ok, sk)
    rb = reconstruct_body(fit, sk)
    worst = max(worst, max(np.linalg.norm(rb[m] - lm[m]) for _, m in ARM_CORRESPONDENCE))
    rooterr.append(np.degrees((Rot.from_rotvec(fit.global_orient).inv() * Rot.from_rotvec(truth[0])).magnitude()))
print(f"[synthetic 180x] worst arm error {1000*worst:.2f} mm; fitted root error median {np.median(rooterr):.1f} deg")
r = sk.rest_joints
print(f"[rest] L shoulder {r[16]}, R shoulder {r[17]} (y,z asymmetry {abs(r[16,1]-r[17,1])*1000:.1f} mm, {abs(r[16,2]-r[17,2])*1000:.1f} mm)")

# ---- 3. betas side channel ----
betas = np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0, 0, 0, 0, 0])
shaped = SMPLXSkeleton(model_path="models", betas=betas)
truth = np.zeros((55, 3))
for j, *_ in SMPLX_ARM_BONES:
    truth[j] = rng.normal(0, 0.3, 3)
P = forward_kinematics(truth, shaped)
lm = np.full((33, 3), np.nan)
for s_, m in ARM_CORRESPONDENCE:
    lm[m] = P[s_]
ok = np.isfinite(lm).all(1)
fit = fit_body(lm, ok, shaped)
rb = reconstruct_body(fit, shaped)
J0 = smplx_joints(fit); Jb = smplx_joints(fit, betas)
print(f"[betas] fit.betas = {fit.betas}")
print(f"[betas] max |SMPL-X(betas=fit.betas) - reconstruct_body(shaped)| {max(np.linalg.norm(J0[s]-rb[m]) for s,m in ARM_CORRESPONDENCE):.3e} m; with true betas {max(np.linalg.norm(Jb[s]-rb[m]) for s,m in ARM_CORRESPONDENCE):.3e} m")
fit2 = fit_body(lm, ok, shaped); fit2.betas = np.array([2.0] + [0] * 9, dtype=float)
print(f"[betas] reconstruct_body change when betas[0] 0->2: {np.nanmax(np.abs(reconstruct_body(fit2, shaped) - rb)):.2e} m")

# ---- 4. Step-4 direction error on a float32 path (cached image-space landmarks, real SMPL-X forward) ----
angs64, angs32 = [], []
for f in sorted(glob.glob("data/test_output/*_landmarks.npy")):
    arr = np.load(f).astype(np.float64)
    pose = arr[:, :33]
    for fr in pose:
        if not np.all(np.isfinite(fr[11:17])) or np.any(np.all(fr[11:17] == 0, axis=1)):
            continue
        rots = solve_swing_ik(fr, sk, SMPLX_ARM_BONES)
        P = forward_kinematics(rots, sk)
        with torch.no_grad():
            out = model(body_pose=torch.tensor(rots[1:22].reshape(1, 63), dtype=torch.float32),
                        global_orient=torch.tensor(rots[0][None], dtype=torch.float32))
        J = out.joints[0, :55].numpy().astype(np.float64)
        for a, b, ma, mb in [(16, 18, 11, 13), (18, 20, 13, 15), (17, 19, 12, 14), (19, 21, 14, 16)]:
            o = fr[mb] - fr[ma]
            for PP, store in [(P, angs64), (J, angs32)]:
                v = PP[b] - PP[a]
                store.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(v, o)), v @ o)))
print(f"[step4] cached landmarks n={len(angs64)}: float64 FK median {np.median(angs64):.2e} max {np.max(angs64):.2e} deg; real SMPL-X (float32) median {np.median(angs32):.4f} max {np.max(angs32):.4f} deg")
