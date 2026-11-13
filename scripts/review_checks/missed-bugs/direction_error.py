"""Bone-direction error of the Step 7 converter's DIRECT output on real clips (doc: 'exact by construction')."""
import glob
import numpy as np
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
BONES = [(11, 13, "L upper"), (13, 15, "L fore"), (12, 14, "R upper"), (14, 16, "R fore")]
ang = {n: [] for *_, n in BONES}; go = []
for p in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(p)
    for lm, vis in zip(d["landmarks"].astype(float), d["visibility"]):
        ok = np.isfinite(lm).all(1) & (vis >= 0.5)
        try:
            f = fit_body(lm, ok, sk)
        except ValueError:
            continue
        out = reconstruct_body(f, sk); go.append(np.degrees(np.linalg.norm(f.global_orient)))
        for a, b, n in BONES:
            if ok[a] and ok[b]:
                u, w = out[b] - out[a], lm[b] - lm[a]
                ang[n].append(np.degrees(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))))
for n, v in ang.items():
    print(f"  {n:<8} direct direction error: median {np.median(v):.2f} deg, p95 {np.percentile(v,95):.2f}, max {np.max(v):.2f}")
print(f"  |global_orient| (= Kabsch rotation applied on top of the IK): median {np.median(go):.2f} deg, max {np.max(go):.2f}")
