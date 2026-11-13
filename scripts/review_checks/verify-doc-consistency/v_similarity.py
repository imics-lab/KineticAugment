"""Can a 7-number similarity act as the 'camera' between metric world landmarks and the cached
image-normalised landmarks (doc L470-479)? Independent re-implementation."""
import glob
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

DIMS = {"01986": (320, 240), "01987": (1280, 720), "01988": (640, 480), "01991": (288, 192), "01992": (288, 192)}
IDX = np.arange(0, 25)  # upper-body pose landmarks (face points, shoulders, arms, hands, hips)


def umeyama(src, tgt, scale=True):
    ms, mt = src.mean(0), tgt.mean(0)
    a, b = src - ms, tgt - mt
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    Rm = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum(1).mean() if scale else 1.0
    return s, Rm, mt - s * Rm @ ms


def sim_z(src, tgt):
    """similarity + independent z gain on the target side (8 params)."""
    s0, R0, t0 = umeyama(src, tgt)
    x0 = np.r_[np.log(s0), R.from_matrix(R0).as_rotvec(), t0, 0.0]
    def res(x):
        s, rv, t, kz = np.exp(x[0]), x[1:4], x[4:7], np.exp(x[7])
        p = s * src @ R.from_rotvec(rv).as_matrix().T + t
        p[:, 2] *= kz
        return (p - tgt).ravel()
    r = least_squares(res, x0)
    return r.fun.reshape(-1, 3), np.exp(r.x[7])


rows = {k: [] for k in ["sim_norm", "sim_px", "sim_px_xy_only_resid", "simz_px", "ortho6_px", "affine_px"]}
zgain = []
per_clip = {}
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    clip = path.split("/")[-1][:5]
    W, H = DIMS[clip]
    d = np.load(path)
    Pw, V = d["landmarks"].astype(np.float64), d["visibility"]
    Pi = np.load(f"data/test_output/{clip}_landmarks.npy")[:, :33].astype(np.float64)
    clip_rows = []
    for f in range(len(Pw)):
        sel = IDX[(V[f, IDX] >= 0.5) & np.isfinite(Pw[f, IDX]).all(1) & np.any(Pi[f, IDX] != 0, axis=1)]
        if len(sel) < 6:
            continue
        src = Pw[f, sel]
        tn = Pi[f, sel]
        tp = tn * np.array([W, H, W])
        sw_n = np.linalg.norm(Pi[f, 11, :2] - Pi[f, 12, :2])                 # normalised xy width
        sw_p = np.linalg.norm((Pi[f, 11, :2] - Pi[f, 12, :2]) * [W, H])       # pixel xy width
        def rms(e):
            return np.sqrt((e ** 2).sum(1).mean())
        s, Rm, t = umeyama(src, tn)
        rows["sim_norm"].append(rms(s * src @ Rm.T + t - tn) / sw_n)
        s, Rm, t = umeyama(src, tp)
        e = s * src @ Rm.T + t - tp
        rows["sim_px"].append(rms(e) / sw_p)
        rows["sim_px_xy_only_resid"].append(np.sqrt((e[:, :2] ** 2).sum(1).mean()) / sw_p)
        clip_rows.append(rms(e) / sw_p)
        e8, kz = sim_z(src, tp)
        rows["simz_px"].append(rms(e8) / sw_p)
        zgain.append(kz)
        # orthographic / weak-perspective: xy only, s*R[:2] X + t (6 params)
        def ores(x):
            p = np.exp(x[0]) * src @ R.from_rotvec(x[1:4]).as_matrix()[:2].T + x[4:6]
            return (p - tp[:, :2]).ravel()
        s0, R0, t0 = umeyama(src, tp)
        o = least_squares(ores, np.r_[np.log(s0), R.from_matrix(R0).as_rotvec(), t0[:2]])
        rows["ortho6_px"].append(np.sqrt((o.fun.reshape(-1, 2) ** 2).sum(1).mean()) / sw_p)
        A = np.c_[src, np.ones(len(src))]
        M, *_ = np.linalg.lstsq(A, tp, rcond=None)
        rows["affine_px"].append(rms(A @ M - tp) / sw_p)
    per_clip[clip] = np.median(clip_rows)

for k, v in rows.items():
    v = np.array(v) * 100
    print(f"{k:22s} median {np.median(v):6.1f}%  p95 {np.percentile(v,95):6.1f}%  (n={len(v)})")
print("per-clip similarity (px) median %:", {k: round(v * 100, 1) for k, v in per_clip.items()})
print(f"fitted z gain kz (pred_z*kz = image z): median {np.median(zgain):.2f}, p5 {np.percentile(zgain,5):.2f}, p95 {np.percentile(zgain,95):.2f}")
