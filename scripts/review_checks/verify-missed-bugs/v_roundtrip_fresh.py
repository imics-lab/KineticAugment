"""Own implementation of the direct/aligned headline (same definitions as measure_round_trip.run_real), on stored vs fresh-per-clip data."""
import sys, numpy as np
from pathlib import Path
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
E = Path(sys.argv[1]); ARM = [11, 12, 13, 14, 15, 16]
BONES = [(11, 13), (13, 15), (12, 14), (14, 16)]
def run(paths):
    D, A, per = [], [], {m: [] for m in ARM}; Lerr = {b: [] for b in BONES}; bl = {}
    for p in paths:
        d = np.load(p); P = d["landmarks"].astype(float); ok_all = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
        both = ok_all[:, 11] & ok_all[:, 12]
        w = np.median(np.linalg.norm(P[both, 11] - P[both, 12], axis=-1))
        stem = Path(p).stem.split("_")[-2] if "world" in p.name else Path(p).stem.split("_")[-1]
        bl[stem] = (w, {b: np.median(np.linalg.norm(P[ok_all[:, b[0]] & ok_all[:, b[1]], b[1]] - P[ok_all[:, b[0]] & ok_all[:, b[1]], b[0]], axis=-1)) for b in BONES})
        for fr, ok in zip(P, ok_all):
            try: out = reconstruct_body(fit_body(fr, ok, sk), sk)
            except ValueError: continue
            pres = [m for m in ARM if ok[m]]
            for m in pres:
                e = np.linalg.norm(out[m] - fr[m]) / w; D.append(e); per[m].append(e)
            src = np.array([out[m] for m in pres]); tgt = np.array([fr[m] for m in pres])
            A += list(np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt)) - tgt, axis=1) / w)
            for a, b in BONES:
                if ok[a] and ok[b]:
                    Lerr[(a, b)].append(abs(np.linalg.norm(out[b] - out[a]) - np.linalg.norm(fr[b] - fr[a])) / w)
    d, a = 100 * np.median(D), 100 * np.median(A)
    print(f"  direct {d:.2f}%  aligned {a:.2f}%  gap {d - a:.2f} pts | per-joint direct med: " + " ".join(f"{m}:{100*np.median(per[m]):.2f}" for m in ARM)
          + " | bone len err med: " + " ".join(f"{a}-{b}:{100*np.median(v):.2f}" for (a, b), v in Lerr.items()))
    for k, (w, b) in bl.items():
        print(f"    {k}: shoulder width {w:.4f} m, median bone lengths " + " ".join(f"{a}-{c}:{v:.4f}" for (a, c), v in b.items()))
stems = ["01986", "01987", "01988", "01991", "01992"]
print("STORED (data/test_output, shared tracker):"); run([Path(f"data/test_output/{s}_world.npz") for s in stems])
print("FRESH (one tracker per clip):"); run([E / f"fresh_{s}.npz" for s in stems])
