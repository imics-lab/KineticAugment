"""Per-clip similarity-aligned medians: shipped converter vs prototype shoulder-line root (doc: 01988 5.06 -> 5.11)."""
import glob, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
import conftest  # noqa
import step8_proto as P8
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE as CORR
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity
sk = SMPLXSkeleton(model_path="models")
tot = {"ship": [], "proto": []}
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); Pm = d["landmarks"].astype(np.float64); V = d["visibility"]
    okall = np.isfinite(Pm).all(-1) & (V >= 0.5)
    both = okall[:, 11] & okall[:, 12]
    width = float(np.median(np.linalg.norm(Pm[both, 11] - Pm[both, 12], axis=-1)))
    row = {}
    for k, f in (("ship", lambda L, ok: fit_body(L, ok, sk)), ("proto", lambda L, ok: P8.fit_body(L, ok, sk, apex="nose"))):
        e = []
        for L, ok in zip(Pm, okall):
            try: out = reconstruct_body(f(L, ok), sk)
            except ValueError: continue
            ms = [m for _, m in CORR if ok[m]]
            src = np.array([out[m] for m in ms]); tgt = np.array([L[m] for m in ms])
            s, R, t = similarity_transform(src, tgt)
            e += list(np.linalg.norm(apply_similarity(src, s, R, t) - tgt, axis=1) / width)
        row[k] = np.median(e) * 100; tot[k] += e
    print(f"{Path(path).stem:14s} aligned median: shipped {row['ship']:.2f}%  shoulder-line root {row['proto']:.2f}%")
print(f"pooled: shipped {np.median(tot['ship'])*100:.2f}%  root {np.median(tot['proto'])*100:.2f}%  reduction {(1-np.median(tot['proto'])/np.median(tot['ship']))*100:.1f}%")
