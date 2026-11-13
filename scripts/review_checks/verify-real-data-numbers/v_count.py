import glob, numpy as np
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
by = {}
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); pose = d["landmarks"].astype(np.float64)
    okA = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
    both = okA[:, 11] & okA[:, 12]
    W = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
    for fr, ok in zip(pose, okA):
        try: out = reconstruct_body(fit_body(fr, ok, sk), sk)
        except ValueError: continue
        n = sum(ok[m] for m in [11,12,13,14,15,16])
        by.setdefault(n, []).append([np.linalg.norm(out[m]-fr[m])/W for m in [11,12,13,14]])
for n, v in sorted(by.items()):
    print(f"n_valid={n}: frames {len(v)}  median over shoulders+elbows only {100*np.median(np.array(v)):.2f}%")
