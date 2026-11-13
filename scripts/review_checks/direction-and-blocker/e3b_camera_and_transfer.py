"""E3b: Is 'no camera model exists to project world-space fits back' a real blocker?
(1) Fit a per-frame camera world -> image from the 33 correspondences MediaPipe returns in the
    same call; report residuals.
(2) Test DELTA transfer: a world-space displacement mapped through the per-frame camera's linear
    part and added to the ORIGINAL image landmarks. Validated against nature's own 'augmentation':
    the subject's real motion between frames t and t+k. Predict image forearm/upper-arm vectors at t+k
    from image at t + camera_t(world_{t+k} - world_t); compare with the observed image at t+k.
All image coords converted to isotropic pixels: X=x*W, Y=y*H, Z=z*W."""
import glob, numpy as np
from pathlib import Path
from kinetic_augment.body_model.alignment import similarity_transform

D = Path(__file__).resolve().parent / "extract"
UPPER = list(range(0, 23))          # face points, shoulders, elbows, wrists, hand points
def iso(img, W, H):
    return img * np.array([W, H, W], dtype=np.float64)

def weak_persp(Pw, Q2):
    """xy ~ s*R[:2] Pw + t. Closed-form init via affine then nearest scaled-orthonormal rows."""
    A = np.c_[Pw, np.ones(len(Pw))]
    M, *_ = np.linalg.lstsq(A, Q2, rcond=None)      # (4,2)
    L = M[:3].T                                       # (2,3)
    U, S, Vt = np.linalg.svd(L, full_matrices=False)
    s = S.mean(); Rr = U @ Vt                         # 2x3 with orthonormal rows
    t = (Q2 - s * Pw @ Rr.T).mean(0)
    for _ in range(50):                               # alternate refine
        Qc = Q2 - t; Pc = Pw
        U, S, Vt = np.linalg.svd(Qc.T @ Pc, full_matrices=False)
        Rr = U @ Vt; s = np.trace(Rr @ Pc.T @ Qc) / np.sum(Pc**2)
        t = (Q2 - s * Pw @ Rr.T).mean(0)
    return s, Rr, t

res3d, res2d, resaff = [], [], []
cams = {}
for f in sorted(D.glob("*.npz")):
    d = np.load(f); W, H = int(d["width"]), int(d["height"])
    I, Pw, V = iso(d["image"], W, H), d["world"].astype(np.float64), d["visibility"]
    for t in range(len(I)):
        ok = np.isfinite(Pw[t]).all(1) & np.isfinite(I[t]).all(1) & (V[t] > 0.5)
        idx = [i for i in UPPER if ok[i]]
        if len(idx) < 6: continue
        sw = np.linalg.norm(I[t, 11, :2] - I[t, 12, :2])
        s, Rm, tr = similarity_transform(Pw[t, idx], I[t, idx], allow_scale=True)
        pred = s * Pw[t, idx] @ Rm.T + tr
        res3d.append(np.median(np.linalg.norm(pred[:, :2] - I[t, idx, :2], axis=1)) / sw * 100)
        s2, R2, t2 = weak_persp(Pw[t, idx], I[t, idx, :2])
        p2 = s2 * Pw[t, idx] @ R2.T + t2
        res2d.append(np.median(np.linalg.norm(p2 - I[t, idx, :2], axis=1)) / sw * 100)
        A = np.c_[Pw[t, idx], np.ones(len(idx))]; M, *_ = np.linalg.lstsq(A, I[t, idx, :2], rcond=None)
        resaff.append(np.median(np.linalg.norm(A @ M - I[t, idx, :2], axis=1)) / sw * 100)
        cams[(f.stem, t)] = (s2, R2, t2, s, Rm, tr)
pr = lambda a: f"median {np.median(a):.2f}  p95 {np.percentile(a,95):.2f}"
print(f"(1) per-frame camera fit, residual on upper-body landmarks, xy, % image shoulder width; frames={len(res2d)}")
print("    3D similarity (Umeyama on X,Y,Z), xy residual:", pr(res3d))
print("    weak-perspective (s,R,t; 6 dof+scale), xy residual:", pr(res2d))
print("    general 2x3 affine (8 dof), xy residual:", pr(resaff))

# (2) delta transfer validated on natural motion
print("\n(2) delta transfer of world displacement to image (natural motion t -> t+k), errors % image SW")
SEGS = {"L forearm": (13, 15), "R forearm": (14, 16), "L upper arm": (11, 13), "R upper arm": (12, 14)}
for k in (1, 3, 5, 10):
    err, base, rel = [], [], []
    for f in sorted(D.glob("*.npz")):
        d = np.load(f); W, H = int(d["width"]), int(d["height"])
        I, Pw, V = iso(d["image"], W, H), d["world"].astype(np.float64), d["visibility"]
        for t in range(len(I) - k):
            if (f.stem, t) not in cams: continue
            s2, R2, t2, *_ = cams[(f.stem, t)]
            sw = np.linalg.norm(I[t, 11, :2] - I[t, 12, :2])
            for a, b in SEGS.values():
                if not (V[t, [a, b]] > 0.5).all() or not (V[t+k, [a, b]] > 0.5).all(): continue
                vi_t, vi_k = I[t, b, :2] - I[t, a, :2], I[t+k, b, :2] - I[t+k, a, :2]
                dw = (Pw[t+k, b] - Pw[t+k, a]) - (Pw[t, b] - Pw[t, a])
                pred = vi_t + s2 * R2 @ dw
                obs_motion = np.linalg.norm(vi_k - vi_t)
                if obs_motion / sw * 100 < 2: continue   # need a real displacement to transfer
                err.append(np.linalg.norm(pred - vi_k) / sw * 100)
                base.append(obs_motion / sw * 100)
                rel.append(np.linalg.norm(pred - vi_k) / obs_motion)
    err, base, rel = map(np.array, (err, base, rel))
    print(f"  k={k:2d}: n={len(err):4d}  true image displacement median {np.median(base):5.1f}%SW | "
          f"transfer error median {np.median(err):4.1f}%SW p95 {np.percentile(err,95):5.1f} | "
          f"relative error median {np.median(rel):.2f} p95 {np.percentile(rel,95):.2f}")
