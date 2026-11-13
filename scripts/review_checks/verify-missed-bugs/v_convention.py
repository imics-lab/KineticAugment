"""Independent check of N1/A002: is the fitted body upside-down on real MediaPipe world data?"""
import glob, numpy as np, torch
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
C = np.diag([1.0, -1.0, -1.0])
VIS = 0.5
def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))

# ---- 1. data convention, measured from the data itself
ups, fwds, hand, lr = [], [], [], []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); P = d["landmarks"].astype(float)
    for fr in P:
        if not np.isfinite(fr).all(): continue
        sm = (fr[11] + fr[12]) / 2; hm = (fr[23] + fr[24]) / 2
        up = (sm - hm) / np.linalg.norm(sm - hm)
        l2r = fr[11] - fr[12]; l2r /= np.linalg.norm(l2r)
        nose = fr[0] - sm; fwd = nose - np.dot(nose, l2r) * l2r - np.dot(nose, up) * up; fwd /= np.linalg.norm(fwd)
        ups.append(up); fwds.append(fwd); lr.append(l2r); hand.append(np.dot(np.cross(l2r, up), fwd))
ups, fwds, lr = map(np.array, (ups, fwds, lr))
print(f"frames {len(ups)}: mean up dir {ups.mean(0).round(3)}, mean fwd {fwds.mean(0).round(3)}, mean L-R dir {lr.mean(0).round(3)}, handedness sign mean {np.mean(hand):.3f}")
print("SMPL-X rest: L shoulder(16)-R shoulder(17) =", (sk.rest_joints[16]-sk.rest_joints[17]).round(3), " head(15)-neck(12) =", (sk.rest_joints[15]-sk.rest_joints[12]).round(3))

# ---- 2. fit as the converter does (raw MP frame) vs with C applied first
def stats(transform):
    up_err, el_above, wr_front, head_above, head_nose, direct = [], [], [], [], [], []
    tr_el, tr_wr = [], []
    for f in sorted(glob.glob("data/test_output/*_world.npz")):
        d = np.load(f); P = d["landmarks"].astype(float); Vv = d["visibility"]
        for fr, vis in zip(P, Vv):
            ok = np.isfinite(fr).all(1) & (vis >= VIS)
            if not np.isfinite(fr).all(): continue
            X = fr @ transform.T
            try: fit = fit_body(X, ok, sk)
            except ValueError: continue
            out = reconstruct_body(fit, sk)
            Rg = R.from_rotvec(fit.global_orient).as_matrix()
            true_up = (X[11] + X[12]) / 2 - (X[23] + X[24]) / 2
            up_err.append(ang(Rg @ [0, 1, 0], true_up))
            # body-local (root-frame) arm geometry of the FIT vs of the TRUTH (truth = C-frame of data)
            Ct = C @ transform.T  # maps current frame back to SMPL-X-like frame
            for s, e, w in ((11, 13, 15), (12, 14, 16)):
                if ok[s] and ok[e]:
                    el_above.append((Rg.T @ (out[e] - out[s]))[1] > 0)
                    tr_el.append((C @ (fr[e] - fr[s]))[1] > 0)
                if ok[s] and ok[w]:
                    wr_front.append((Rg.T @ (out[w] - out[s]))[2] > 0)
                    tr_wr.append((C @ (fr[w] - fr[s]))[2] > 0)
            t = lambda a: torch.tensor(a, dtype=torch.float32)[None]
            with torch.no_grad():
                J = sk.model(body_pose=t(fit.body_pose), global_orient=t(fit.global_orient), transl=t(fit.transl),
                             betas=t(fit.betas), return_verts=False).joints[0].numpy().astype(float)
            sm = (X[11] + X[12]) / 2
            head_above.append(np.dot(J[15] - sm, true_up) > 0)
            head_nose.append(np.linalg.norm(J[15] - X[0]))
            for m in (11, 12, 13, 14, 15, 16):
                if ok[m]: direct.append(np.linalg.norm(out[m] - X[m]))
    return dict(n=len(up_err), up_err_med=np.median(up_err), elbow_above_pct=100*np.mean(el_above),
                truth_elbow_above_pct=100*np.mean(tr_el), wrist_front_pct=100*np.mean(wr_front),
                truth_wrist_front_pct=100*np.mean(tr_wr), head_above_shoulders_pct=100*np.mean(head_above),
                head_to_nose_med_mm=1000*np.median(head_nose), direct_med_mm=1000*np.median(direct))
for name, T in (("as-is (MediaPipe frame)", np.eye(3)), ("C applied first", C)):
    print(name, {k: (round(float(v), 2) if not isinstance(v, int) else v) for k, v in stats(T).items()})
