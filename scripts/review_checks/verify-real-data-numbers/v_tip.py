import glob, numpy as np
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
B = [(11, 13), (13, 15), (12, 14), (14, 16)]
tip = {b: [] for b in B}
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); pose = d["landmarks"].astype(np.float64)
    okA = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
    both = okA[:, 11] & okA[:, 12]
    W = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
    for fr, ok in zip(pose, okA):
        try: out = reconstruct_body(fit_body(fr, ok, sk), sk)
        except ValueError: continue
        for a, b in B:
            if ok[a] and ok[b]:
                v, o = out[b] - out[a], fr[b] - fr[a]
                th = np.arctan2(np.linalg.norm(np.cross(v, o)), v @ o)
                tip[(a, b)].append(2 * np.linalg.norm(o) * np.sin(th / 2) / W)
print("tip displacement implied by direction error alone (median % width):",
      {f"{a}-{b}": round(100 * np.median(v), 2) for (a, b), v in tip.items()},
      "pooled", round(100 * np.median(np.concatenate(list(tip.values()))), 2))
