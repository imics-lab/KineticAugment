"""Independent re-check of D-BLOCK-2 numbers using the tester's single-pass Holistic extraction (data only).
(a) per-frame weak-perspective camera world->image (my own init + LS refinement), upper-body landmarks.
(b) world vs image agreement of 3-frame 2-D segment-angle changes (with and without the >5deg filter).
(c) full replacement: project fit_body/reconstruct_body arms through the per-frame camera vs image arms;
    also project ORIGINAL world arms through the same camera (camera-only residual) to separate the parts.
(d) delta transfer error on natural motion t->t+3 (relative to true image displacement), arm landmarks."""
import glob, warnings
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
warnings.filterwarnings("ignore")
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
D = Path("scripts/review_checks/direction-and-blocker/extract")

def wp_fit(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Qc.T @ Pc, full_matrices=False)  # 2x3
    R2 = U @ Vt
    R3 = np.vstack([R2, np.cross(R2[0], R2[1])])
    if np.linalg.det(R3) < 0: R3[2] *= -1
    s0 = np.sum(Qc * (Pc @ R2.T)) / np.sum((Pc @ R2.T) ** 2)
    def res(x):
        return ((np.exp(x[0]) * P @ R.from_rotvec(x[1:4]).as_matrix()[:2].T + x[4:6]) - Q).ravel()
    best = None
    for init in (R.from_matrix(R3).as_rotvec(), np.zeros(3), [np.pi, 0, 0]):
        x0 = np.r_[np.log(abs(s0) + 1e-9), init, Q.mean(0) - abs(s0) * P.mean(0) @ R.from_rotvec(init).as_matrix()[:2].T]
        r = least_squares(res, x0)
        if best is None or r.cost < best.cost: best = r
    x = best.x
    return lambda X: np.exp(x[0]) * X @ R.from_rotvec(x[1:4]).as_matrix()[:2].T + x[4:6]

cam_res, fr_err, cam_only_arm, ang_w, ang_i, rel, gap0, gap1 = [], [], [], [], [], [], [], []
gap2 = []
SEG = [(11, 13), (13, 15), (12, 14), (14, 16)]
ARMS = [11, 12, 13, 14, 15, 16]
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I = d["image"][..., :2] * np.array([W, H]); Pw = d["world"].astype(float); V = d["visibility"]
    LH = d["left_hand"][..., :2] * np.array([W, H]); RH = d["right_hand"][..., :2] * np.array([W, H])
    T = len(I); cams = [None] * T; recon = [None] * T
    for t in range(T):
        ok = np.isfinite(Pw[t]).all(1) & (V[t] > 0.5)
        idx = [i for i in range(23) if ok[i]]
        if len(idx) < 6: continue
        sw = np.linalg.norm(I[t, 11] - I[t, 12])
        cam = wp_fit(Pw[t, idx], I[t, idx]); cams[t] = cam
        cam_res.append(np.median(np.linalg.norm(cam(Pw[t, idx]) - I[t, idx], axis=1)) / sw * 100)
        try:
            fit = fit_body(Pw[t], ok, sk)
        except ValueError:
            continue
        rc = reconstruct_body(fit, sk); recon[t] = rc
        arms = [m for m in ARMS if ok[m]]
        fr_err.append(np.median(np.linalg.norm(cam(rc[arms]) - I[t, arms], axis=1)) / sw * 100)
        cam_only_arm.append(np.median(np.linalg.norm(cam(Pw[t, arms]) - I[t, arms], axis=1)) / sw * 100)
        # hand detachment: pose wrist (15 left / 16 right) vs hand-model wrist (index 0). MediaPipe 'left_hand' is subject's left.
        for pw, hand in ((15, LH), (16, RH)):
            if ok[pw] and np.isfinite(hand[t, 0]).all():
                gap0.append(np.linalg.norm(I[t, pw] - hand[t, 0]) / sw * 100)
                gap1.append(np.linalg.norm(cam(rc[pw:pw + 1])[0] - hand[t, 0]) / sw * 100)
                gap2.append(np.linalg.norm(cam(Pw[t, pw:pw + 1])[0] - hand[t, 0]) / sw * 100)
    for t in range(T - 3):
        if cams[t] is None or cams[t + 3] is None: continue
        ok0 = np.isfinite(Pw[t]).all(1) & (V[t] > 0.5); ok1 = np.isfinite(Pw[t + 3]).all(1) & (V[t + 3] > 0.5)
        sw = np.linalg.norm(I[t, 11] - I[t, 12])
        for a, b in SEG:
            if not (ok0[a] and ok0[b] and ok1[a] and ok1[b]): continue
            ai = lambda tt: np.degrees(np.arctan2(*(I[tt, b] - I[tt, a])[::-1]))
            aw = lambda tt: np.degrees(np.arctan2(*(Pw[tt, b, :2] - Pw[tt, a, :2])[::-1]))
            wrap = lambda x: (x + 180) % 360 - 180
            ang_i.append(wrap(ai(t + 3) - ai(t))); ang_w.append(wrap(aw(t + 3) - aw(t)))
        for m in ARMS:
            if not (ok0[m] and ok1[m]): continue
            true_d = I[t + 3, m] - I[t, m]
            pred_d = cams[t](Pw[t + 3, m:m + 1])[0] - cams[t](Pw[t, m:m + 1])[0]
            if np.linalg.norm(true_d) / sw > 0.02:
                rel.append(np.linalg.norm(pred_d - true_d) / np.linalg.norm(true_d))
p = lambda x: "median %.2f p5 %.2f p95 %.2f (n=%d)" % (*np.percentile(x, [50, 5, 95]), len(x))
print("(a) weak-perspective per-frame residual, upper body 0-22, %SW:", p(cam_res))
print("(c) FULL REPLACEMENT arm error in image %SW:", p(fr_err))
print("    camera-only (project original world arms) %SW:", p(cam_only_arm))
print("    pose-wrist vs hand-wrist gap %SW: original", p(gap0), "| full replacement", p(gap1))
ang_i, ang_w = np.array(ang_i), np.array(ang_w)
for thr in (0, 5):
    m = np.abs(ang_i) > thr
    print("(b) 3-frame 2-D segment angle change, |image change|>%d: n=%d corr %.3f sign agree %.1f%% |w-i|/|i| median %.2f" % (
        thr, m.sum(), np.corrcoef(ang_i[m], ang_w[m])[0, 1], 100 * (np.sign(ang_i[m]) == np.sign(ang_w[m])).mean(),
        np.median(np.abs(ang_w[m] - ang_i[m]) / np.abs(ang_i[m]))))
print("    pose-wrist vs hand-wrist gap, projecting ORIGINAL world wrist (camera only):", p(gap2))
print("(d) delta transfer of natural 3-frame displacement (true >2%SW), relative error:", p(rel))
