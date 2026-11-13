import glob, numpy as np
from kinetic_augment.body_model.converter import fit_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models"); mx = {16: 0, 17: 0, 18: 0, 19: 0}
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); P = d["landmarks"].astype(float); OK = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    for fr, ok in zip(P, OK):
        try: bp = fit_body(fr, ok, sk).body_pose.reshape(21, 3)
        except ValueError: continue
        for j in mx: mx[j] = max(mx[j], np.degrees(np.linalg.norm(bp[j - 1])))
print("max swing angle per driven joint (deg):", {k: round(v, 1) for k, v in mx.items()})
