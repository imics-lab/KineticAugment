"""Independent: (a) a shoulders+nose Kabsch root (my own implementation) -> upright? elbow sign?
(b) +10deg 'flex' added JAP-style to the local-y rotvec component, current fit vs step-8-style fit
(c) zero-noise repo joint-limit clamp on arm joints, both fits.
(d) arm-landmark invariance: does the step-8 root change reconstructed arm landmarks vs current fit?"""
import glob, warnings
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
warnings.filterwarnings("ignore")
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE, BodyFit
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.joint_mapping import clamp_joint_angles

sk = SMPLXSkeleton(model_path="models")
with torch.no_grad():
    J0 = sk.model(return_verts=False).joints[0].numpy().astype(float)
assert np.allclose(J0[:55], sk.rest_joints, atol=1e-5)
rest_nose = J0[55]
print("rest nose (SMPL-X idx 55):", rest_nose.round(3), " rest head:", sk.rest_joints[15].round(3))

def kabsch(src, dst):
    a = src - src.mean(0); b = dst - dst.mean(0)
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

def fit_step8(lm, ok):
    Rroot = kabsch(np.array([sk.rest_joints[16], sk.rest_joints[17], rest_nose]), np.array([lm[11], lm[12], lm[0]]))
    masked = np.where(ok[:, None], lm, np.nan)
    rots = solve_swing_ik(masked, sk, SMPLX_ARM_BONES, global_orient=R.from_matrix(Rroot).as_rotvec())
    posed = forward_kinematics(rots, sk)
    idx = [(s, m) for s, m in ARM_CORRESPONDENCE if ok[m]]
    t = np.mean([lm[m] - posed[s] for s, m in idx], axis=0)
    return BodyFit(body_pose=rots[1:22].reshape(63).copy(), global_orient=rots[0].copy(), transl=t)

def bend(arm):  # arm = (sh, el, wr) ; 0 = straight
    u = arm[1] - arm[0]; f = arm[2] - arm[1]
    return np.degrees(np.arccos(np.clip(u.dot(f) / np.linalg.norm(u) / np.linalg.norm(f), -1, 1)))

SIDES = [("L", 17, (11, 13, 15), -1.0), ("R", 18, (12, 14, 16), +1.0)]  # body_pose idx, mp idx, flexion sign on local y
out = {k: [] for k in ["cur_flex", "s8_flex", "cur_sign", "s8_sign", "cur_clamp_disp", "s8_clamp_disp", "s8_bend_before", "s8_bend_after", "cur_up", "s8_up", "cur_front", "s8_front", "armdiff_s8_vs_cur", "s8_vs_input"]}
nfr = 0
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); L = d["landmarks"].astype(float); V = d["visibility"]
    for t in range(L.shape[0]):
        lm = L[t]; ok = np.isfinite(lm).all(1) & (V[t] >= 0.5)
        try:
            fc = fit_body(lm, ok, sk)
        except ValueError:
            continue
        if not ok[0]:
            continue
        nfr += 1
        f8 = fit_step8(lm, ok)
        sw = np.linalg.norm(lm[11] - lm[12])
        rc, r8 = reconstruct_body(fc, sk), reconstruct_body(f8, sk)
        diffs = [np.linalg.norm(rc[m] - r8[m]) / sw for _, m in ARM_CORRESPONDENCE if ok[m]]
        out["armdiff_s8_vs_cur"].append(max(diffs))
        out["s8_vs_input"].append(np.median([np.linalg.norm(r8[m] - lm[m]) / sw for _, m in ARM_CORRESPONDENCE if ok[m]]))
        left = lm[11] - lm[12]; left /= np.linalg.norm(left)
        up = lm[0] - 0.5 * (lm[11] + lm[12]); up -= up.dot(left) * left; up /= np.linalg.norm(up)
        front = np.cross(left, up)
        for tag, fit in (("cur", fc), ("s8", f8)):
            G = R.from_rotvec(fit.global_orient).as_matrix()
            out[tag + "_up"].append((G @ [0, 1, 0]).dot(up)); out[tag + "_front"].append((G @ [0, 0, 1]).dot(front))
        for side, bi, (s, e, w), sgn in SIDES:
            if not (ok[s] and ok[e] and ok[w]):
                continue
            for tag, fit in (("cur", fc), ("s8", f8)):
                bp = fit.body_pose.reshape(21, 3)
                out[tag + "_sign"].append(np.sign(bp[bi, 1]) == sgn)
                rec = reconstruct_body(fit, sk)
                b0 = bend(rec[[s, e, w]])
                bp2 = bp.copy(); bp2[bi, 1] += sgn * np.radians(10)
                rec2 = reconstruct_body(BodyFit(bp2.reshape(63), fit.global_orient, fit.transl), sk)
                out[tag + "_flex"].append(bend(rec2[[s, e, w]]) - b0)
                # zero-noise clamp on shoulder + elbow of this side (as JAP respect_limits does)
                bp3 = bp.copy()
                for name, j in ((side == "L" and "left_shoulder" or "right_shoulder", bi - 2), (side == "L" and "left_elbow" or "right_elbow", bi)):
                    bp3[j] = clamp_joint_angles(name, bp3[j])
                rec3 = reconstruct_body(BodyFit(bp3.reshape(63), fit.global_orient, fit.transl), sk)
                out[tag + "_clamp_disp"].append(np.linalg.norm(rec3[w] - rec[w]) / sw * 100)
                if tag == "s8":
                    out["s8_bend_before"].append(b0); out["s8_bend_after"].append(bend(rec3[[s, e, w]]))
a = {k: np.array(v, dtype=float) for k, v in out.items()}
p = lambda x: "median %.2f p5 %.2f p95 %.2f" % tuple(np.percentile(x, [50, 5, 95]))
print("frames (nose valid):", nfr, " arms:", len(a["cur_flex"]))
print("cos(fitted up, data up):    current", p(a["cur_up"]), "| step8", p(a["s8_up"]))
print("cos(fitted front, data front): current", p(a["cur_front"]), "| step8", p(a["s8_front"]))
print("elbow local-y on anatomical flexion side: current %.1f%% | step8 %.1f%%" % (100 * a["cur_sign"].mean(), 100 * a["s8_sign"].mean()))
print("+10deg local-y flex -> bend change deg: current", p(a["cur_flex"]), "| step8", p(a["s8_flex"]))
print("zero-noise clamp wrist displacement %SW: current", p(a["cur_clamp_disp"]), "| step8", p(a["s8_clamp_disp"]))
print("step8 bend before clamp", p(a["s8_bend_before"]), "after", p(a["s8_bend_after"]))
print("frac arms where step8 clamp moves wrist >1%%SW: %.3f" % (a["s8_clamp_disp"] > 1).mean())
print("max over frames of max-arm |recon(step8) - recon(current)| %%SW: %.3f (median %.3f)" % (a["armdiff_s8_vs_cur"].max() * 100, np.median(a["armdiff_s8_vs_cur"]) * 100))
print("step8-style direct error vs input, per-frame median over arm landmarks %SW:", p(a["s8_vs_input"] * 100))
