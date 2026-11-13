"""Verifier's independent implementation of naive vs adaptive (HybrIK-style) arm IK,
plus a least-squares optimum (lower bound for any 4-joint arm fit with template bone lengths).
Only the SMPL-X rest skeleton comes from the project; Kabsch, chain IK and metric are re-written.
Also cross-checks that my 'naive' reproduces the project's fit_body/reconstruct_body numbers."""
import glob, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body

sk = SMPLXSkeleton(model_path="models"); rest = sk.rest_joints
ARMS = [(16, 18, 20, 11, 13, 15), (17, 19, 21, 12, 14, 16)]  # sh, el, wr joints ; mp sh, el, wr
MP = [11, 12, 13, 14, 15, 16]

def kabsch(src, tgt):
    cs, ct = src.mean(0), tgt.mean(0)
    U, S, Vt = np.linalg.svd((src - cs).T @ (tgt - ct))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, ct - R @ cs

def swing(a, b):
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    ax = np.cross(a, b); s = np.linalg.norm(ax); c = a @ b
    if s < 1e-12: return np.eye(3)
    return Rot.from_rotvec(ax/s*np.arctan2(s, c)).as_matrix()

def chain(lm_local, ok, adaptive):
    """Return posed (in skeleton frame) positions dict for sh/el/wr of both arms."""
    P = {}
    for sh, el, wr, ms, me, mw in ARMS:
        s = rest[sh].copy(); P[sh] = s
        u_rest, f_rest = rest[el]-rest[sh], rest[wr]-rest[el]
        W1 = np.eye(3)
        if ok[ms] and ok[me]:
            base = s if adaptive else lm_local[ms]
            W1 = swing(u_rest, lm_local[me] - base)
        e = s + W1 @ u_rest; P[el] = e
        W2 = W1
        if ok[ms] and ok[me] and ok[mw]:
            base = e if adaptive else lm_local[me]
            W2 = W1 @ swing(f_rest, W1.T @ (lm_local[mw] - base))
        P[wr] = e + W2 @ f_rest
    return P

PAIRS = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]
def fit(frame, ok, adaptive, iters):
    use = [(j, m) for j, m in PAIRS if ok[m]]
    if len(use) < 3: return None
    # naive first (in model frame; naive directions are rotation-equivariant so frame doesn't matter for local swings
    # only if global R known -> do as project: IK with identity root, then rigid fit)
    R, t = np.eye(3), np.zeros(3)
    P = chain(frame, ok, False)
    for k in range(1 + (iters if adaptive else 0)):
        if k > 0:
            loc = (frame - t) @ R   # R^T (x - t)
            P = chain(loc, ok, True)
        src = np.array([P[j] for j, _ in use]); tgt = np.array([frame[m] for _, m in use])
        R, t = kabsch(src, tgt)
    out = {m: R @ P[j] + t for j, m in PAIRS}
    return out

def lsq(frame, ok, init):
    use = [(j, m) for j, m in PAIRS if ok[m]]
    if len(use) < 3: return None
    def pos(x):
        R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:6]; P = {}
        for i, (sh, el, wr, *_ ) in enumerate(ARMS):
            W1 = Rot.from_rotvec(x[6+6*i:9+6*i]).as_matrix(); W2 = W1 @ Rot.from_rotvec(x[9+6*i:12+6*i]).as_matrix()
            P[sh] = rest[sh]; P[el] = rest[sh] + W1 @ (rest[el]-rest[sh]); P[wr] = P[el] + W2 @ (rest[wr]-rest[el])
        return {m: R @ P[j] + t for j, m in PAIRS}
    def res(x):
        p = pos(x); return np.concatenate([p[m] - frame[m] for _, m in use])
    best = None
    for x0 in init:
        r = least_squares(res, x0, method="lm" if 3*len(use) >= len(x0) else "trf")
        if best is None or r.cost < best.cost: best = r
    return pos(best.x)

res = {k: {m: [] for m in MP} for k in ("proj_naive", "my_naive", "my_adapt1", "my_adapt3", "lsq")}
mm = {k: [] for k in res}
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); pose = d["landmarks"].astype(float)
    okA = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
    both = okA[:, 11] & okA[:, 12]
    width = float(np.median(np.linalg.norm(pose[both, 11]-pose[both, 12], axis=-1)))
    for frame, ok in zip(pose, okA):
        outs = {}
        try: rp = reconstruct_body(fit_body(frame, ok, sk), sk); outs["proj_naive"] = {m: rp[m] for m in MP}
        except ValueError: continue
        outs["my_naive"] = fit(frame, ok, False, 0)
        outs["my_adapt1"] = fit(frame, ok, True, 1)
        outs["my_adapt3"] = fit(frame, ok, True, 3)
        # LSQ init: zero + a few; parametrise global rot as identity-start, local swings free
        x0 = np.zeros(18); x0[3:6] = frame[[11, 12]].mean(0) - (rest[16]+rest[17])/2
        outs["lsq"] = lsq(frame, ok, [x0, x0 + np.r_[0, np.pi, 0, np.zeros(15)]])
        for k, o in outs.items():
            for m in MP:
                if ok[m]:
                    e = np.linalg.norm(o[m]-frame[m]); res[k][m].append(e/width); mm[k].append(e)
for k in res:
    allv = np.concatenate([res[k][m] for m in MP])
    print(f"{k:11s} n={len(allv)} pooled median {np.median(allv)*100:.2f}%  p95 {np.percentile(allv,95)*100:.2f}%  median {np.median(mm[k])*1000:.1f} mm | "
          + " ".join(f"{m}:{np.median(res[k][m])*100:.2f}" for m in MP))
