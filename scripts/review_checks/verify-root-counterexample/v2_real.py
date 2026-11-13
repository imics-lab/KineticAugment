"""Independent real-data checks for the root-orientation cluster.

Re-implements: shoulder-line root, nose-apex root, hip-apex root, converter output
geometry (upside-down?), torso-relative wrist sign, realism of root angle, hip
origin / visibility, per-clip aligned medians.
"""
import glob
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik
from kinetic_augment.body_model.converter import fit_body, reconstruct_body

CORR = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]
MPS = [m for _, m in CORR]
SK = SMPLXSkeleton(model_path="models")
REST = SK.rest_joints
with torch.no_grad():
    REST_NOSE = SK.model(return_verts=False).joints[0, 55].numpy().astype(np.float64)
REST_HIPC = 0.5 * (REST[1] + REST[2])
print("rest nose", REST_NOSE, " rest hip-centre", REST_HIPC, " pelvis", REST[0])


def umeyama(src, tgt, scale=True):
    ms, mt = src.mean(0), tgt.mean(0)
    a, b = src - ms, tgt - mt
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean() if scale else 1.0
    return s * src @ Rm.T + (mt - s * Rm @ ms)


def frame(l, r, apex):
    x = l - r; x /= np.linalg.norm(x)
    y = apex - 0.5 * (l + r); y = y - (y @ x) * x
    n = np.linalg.norm(y)
    if n < 1e-9:
        return None
    y /= n
    return np.stack([x, y, np.cross(x, y)], 1)


def torso_root(lm, apex_obs, apex_rest):
    Fo = frame(lm[11], lm[12], apex_obs)
    Fr = frame(REST[16], REST[17], apex_rest)
    return Fo @ Fr.T


def shoulder_line_root(lm):
    a = REST[16] - REST[17]; a /= np.linalg.norm(a)
    b = lm[11] - lm[12]; b /= np.linalg.norm(b)
    ax = np.cross(a, b); s = np.linalg.norm(ax); c = a @ b
    return R.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix() if s > 1e-12 else np.eye(3)


def fit_with_root(lm, ok, Rroot):
    masked = np.where(ok[:, None], lm, np.nan)
    rots = solve_swing_ik(masked, SK, SMPLX_ARM_BONES, global_orient=R.from_matrix(Rroot).as_rotvec())
    J = forward_kinematics(rots, SK)
    pres = [(s, m) for s, m in CORR if ok[m]]
    src = np.array([J[s] for s, _ in pres]); tgt = np.array([lm[m] for _, m in pres])
    t = (tgt - src).mean(0)
    return rots, J + t, pres


files = sorted(glob.glob("data/test_output/*_world.npz"))
res = {k: [] for k in ["base_d", "base_a", "sl_d", "sl_a", "nose_d", "hip_d"]}
perclip = {}
stats = dict(sl_angle=[], go_from_I=[], go_from_flip=[], head_below=[], nose_vs_hip=[],
             minrot_vs_nose=[], hipmid=[], wsign_conv=[], wsign_nose=[], wsign_hip=[],
             wsign_conv_cam=[], wsign_hip_cam=[], wsign_nose_cam=[], nose_pitch=[],
             head_below_nose=[], head_below_hip=[], hipvis=[], cos_conv=[])
FLIP = R.from_euler("x", 180, degrees=True).as_matrix()
for path in files:
    clip = Path(path).stem.replace("_world", "")
    d = np.load(path)
    P = d["landmarks"].astype(np.float64); V = d["visibility"]
    OK = np.isfinite(P).all(-1) & (V >= 0.5)
    both = OK[:, 11] & OK[:, 12]
    W = float(np.median(np.linalg.norm(P[both, 11] - P[both, 12], axis=-1)))
    pc = {k: [] for k in ["base_a", "sl_a", "base_d", "sl_d"]}
    pitches = []
    for lm, ok, vis in zip(P, OK, V):
        stats["hipmid"].append(np.linalg.norm(0.5 * (lm[23] + lm[24])))
        stats["hipvis"].append(bool(vis[23] >= 0.5 and vis[24] >= 0.5))
        try:
            fit = fit_body(lm, ok, SK)
        except ValueError:
            continue
        out = reconstruct_body(fit, SK)
        pres = [m for m in MPS if ok[m]]
        tgt = np.array([lm[m] for m in pres]); src = np.array([out[m] for m in pres])
        e_d = np.linalg.norm(src - tgt, axis=1) / W
        e_a = np.linalg.norm(umeyama(src, tgt) - tgt, axis=1) / W
        res["base_d"] += list(e_d); res["base_a"] += list(e_a); pc["base_d"] += list(e_d); pc["base_a"] += list(e_a)
        # converter geometry
        Rg = R.from_rotvec(fit.global_orient).as_matrix()
        stats["go_from_I"].append(np.degrees(R.from_matrix(Rg).magnitude()))
        stats["go_from_flip"].append(np.degrees(R.from_matrix(FLIP.T @ Rg).magnitude()))
        rots = np.zeros((55, 3)); rots[0] = fit.global_orient; rots[1:22] = fit.body_pose.reshape(21, 3)
        Jc = forward_kinematics(rots, SK) + fit.transl
        shm = 0.5 * (lm[11] + lm[12])
        up_cam = np.array([0, -1.0, 0])  # MediaPipe y points down
        stats["head_below"].append((Jc[15] - shm) @ up_cam < 0)
        v1 = Jc[15] - 0.5 * (Jc[16] + Jc[17]); v2 = lm[0] - shm
        stats["cos_conv"].append(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2))
        # shoulder-line angle (realism of the gate-relevant root error)
        a = REST[16] - REST[17]; b = lm[11] - lm[12]
        stats["sl_angle"].append(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))
        # shoulder-line root
        Rsl = shoulder_line_root(lm)
        _, Jsl, prs = fit_with_root(lm, ok, Rsl)
        src2 = np.array([Jsl[s] for s, _ in prs]); tgt2 = np.array([lm[m] for _, m in prs])
        ed = np.linalg.norm(src2 - tgt2, axis=1) / W
        ea = np.linalg.norm(umeyama(src2, tgt2) - tgt2, axis=1) / W
        res["sl_d"] += list(ed); res["sl_a"] += list(ea); pc["sl_d"] += list(ed); pc["sl_a"] += list(ea)
        # nose and hip roots
        hipc = 0.5 * (lm[23] + lm[24])
        Rn = torso_root(lm, lm[0], REST_NOSE)
        Rh = torso_root(lm, hipc, REST_HIPC)
        rn, Jn, prs_n = fit_with_root(lm, ok, Rn)
        rh, Jh, prs_h = fit_with_root(lm, ok, Rh)
        res["nose_d"] += list(np.linalg.norm(np.array([Jn[s] - lm[m] for s, m in prs_n]), axis=1) / W)
        res["hip_d"] += list(np.linalg.norm(np.array([Jh[s] - lm[m] for s, m in prs_h]), axis=1) / W)
        stats["nose_vs_hip"].append(np.degrees(R.from_matrix(Rn.T @ Rh).magnitude()))
        stats["minrot_vs_nose"].append(np.degrees(R.from_matrix(Rsl.T @ Rn).magnitude()))
        stats["head_below_nose"].append((Jn[15] - shm) @ up_cam < 0)
        stats["head_below_hip"].append((Jh[15] - shm) @ up_cam < 0)
        # nose pitch relative to hip-based torso frame (about the shoulder axis)
        Fh = frame(lm[11], lm[12], hipc)
        vn = Fh.T @ (lm[0] - shm)
        pitches.append(np.degrees(np.arctan2(vn[2], -vn[1])))
        # torso-relative wrist sign: body frame = FK with root zeroed
        if ok[15]:
            obs_up = shm - hipc; obs_up /= np.linalg.norm(obs_up)
            h_obs = (lm[15] - lm[11]) @ obs_up
            h_cam = (lm[15] - lm[11]) @ up_cam
            for key, rr in (("conv", rots), ("nose", rn), ("hip", rh)):
                r0 = rr.copy(); r0[0] = 0
                Jb = forward_kinematics(r0, SK)
                h_fit = (Jb[20] - Jb[16])[1]
                stats["wsign_" + key].append((h_fit, h_obs))
                stats["wsign_" + key + "_cam"].append((h_fit, h_cam))
    stats["nose_pitch"].append((clip, np.std(pitches), np.ptp(pitches)))
    perclip[clip] = {k: np.median(v) * 100 for k, v in pc.items()}

pct = lambda v, q=50: np.percentile(v, q) * 100
print("\n=== pooled medians, % shoulder width ===")
for k, v in res.items():
    print(f"  {k:8s} median {pct(v):5.2f}%   n={len(v)}")
print("  reduction aligned (sl vs base): %.1f%%" % (100 * (1 - pct(res['sl_a']) / pct(res['base_a']))))
print("\n=== per clip medians (%) ===")
for c, v in perclip.items():
    print("  ", c, {k: round(x, 2) for k, x in v.items()})
q = lambda v: f"median {np.median(v):.2f} p95 {np.percentile(v, 95):.2f} max {np.max(v):.2f}"
print("\n=== realism: observed vs rest shoulder-line angle (deg):", q(stats["sl_angle"]),
      " frac>=20:", np.mean(np.array(stats["sl_angle"]) >= 20), " frac>=30:", np.mean(np.array(stats["sl_angle"]) >= 30))
print("=== converter global_orient: from identity", q(stats["go_from_I"]), "| from Rx(180)", q(stats["go_from_flip"]))
print("=== head (joint15) below shoulder midpoint in camera vertical: conv %.3f nose-root %.3f hip-root %.3f" % (
    np.mean(stats["head_below"]), np.mean(stats["head_below_nose"]), np.mean(stats["head_below_hip"])))
print("=== cos(fitted shoulder->head, observed shoulder->nose) conv median %.3f" % np.median(stats["cos_conv"]))
print("=== nose-root vs hip-root angle (deg):", q(stats["nose_vs_hip"]))
print("=== shoulder-line minimal root vs nose-root (deg):", q(stats["minrot_vs_nose"]), "min %.2f" % np.min(stats["minrot_vs_nose"]))
print("=== nose pitch about shoulder axis wrt hip torso frame (std, range) per clip:",
      [(c, round(s, 1), round(r, 1)) for c, s, r in stats["nose_pitch"]])
hm = np.array(stats["hipmid"])
print("=== |hip midpoint| (mm): median %.2f max %.2f, n=%d; both hips vis>=0.5: %.3f" % (
    np.median(hm) * 1e3, hm.max() * 1e3, len(hm), np.mean(stats["hipvis"])))
for key in ("conv", "nose", "hip"):
    for suf in ("", "_cam"):
        a = np.array(stats["wsign_" + key + suf])
        agree = np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1]))
        print(f"=== L wrist height sign agreement ({key}{suf or ' vs MP torso axis'}): {agree:.3f}  corr {np.corrcoef(a.T)[0,1]:+.3f}  n={len(a)}")
