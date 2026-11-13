"""Are the ~1.5 rad frame-to-frame elbow jumps caused by wrist-visibility dropouts (bone left at zero)?"""
import glob
import numpy as np
from kinetic_augment.body_model.converter import fit_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
rows = []
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path)
    prev = None
    for lm, vis in zip(d["landmarks"].astype(np.float64), d["visibility"]):
        ok = np.isfinite(lm).all(1) & (vis >= 0.5)
        try:
            f = fit_body(lm, ok, sk)
        except ValueError:
            prev = None; continue
        bp = f.body_pose.reshape(21, 3)
        cur = (bp, ok, lm)
        if prev is not None:
            for j, w in ((18, 15), (19, 16)):
                both_obs = prev[1][w] and ok[w]
                jump = np.linalg.norm(bp[j-1] - prev[0][j-1])
                rows.append((both_obs, jump, bool(prev[1][w]) != bool(ok[w])))
        prev = cur
r = np.array(rows, dtype=float)
obs, jump, toggle = r[:,0] > 0, r[:,1], r[:,2] > 0
print(f"elbow frame pairs: {len(r)}; wrist observed in both frames: {obs.sum()}, wrist validity toggles: {toggle.sum()}")
print(f"  |d rotvec| when wrist observed in both: p99 {np.percentile(jump[obs],99):.2f} max {jump[obs].max():.2f} rad")
print(f"  |d rotvec| on validity toggles        : median {np.median(jump[toggle]):.2f} max {jump[toggle].max():.2f} rad")
print(f"  jumps > 1.0 rad: {np.sum(jump>1.0)} total, of which on toggles: {np.sum((jump>1.0)&toggle)}")
