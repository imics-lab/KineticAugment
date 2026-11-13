"""Q5: does a sequence-level shape fit generalise within a clip? Fit betas on the
first half of each clip's frames (medians of the 5 lengths), score the converter's
direct error on the second half (and vice versa). Contiguous halves, because the
lengths are strongly autocorrelated (lag-1 ACF ~0.9)."""
import glob, os
import numpy as np
from scipy.optimize import least_squares
from common import SNAP, NB, lengths, length_jac
os.chdir(SNAP)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
MP = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]; ARM = [11, 12, 13, 14, 15, 16]

def meds(P, ok):
    return np.array([np.median(np.linalg.norm(P[ok[:, a] & ok[:, b], a] - P[ok[:, a] & ok[:, b], b], axis=1)) for a, b in MP])
def map_fit(L, sig):
    w = 1000.0 / sig
    return least_squares(lambda b: np.concatenate([(lengths(b) - L) * w, b]), np.zeros(NB),
                         jac=lambda b: np.vstack([length_jac(b, with_scale=False) * w, np.eye(NB)])).x
def err(P, ok, sk, width):
    e = []
    for f, o in zip(P, ok):
        try: out = reconstruct_body(fit_body(f, o, sk), sk)
        except ValueError: continue
        e += [np.linalg.norm(out[m] - f[m]) / width for m in ARM if o[m]]
    return e

configs = ["betas=0", "MAP 10mm", "MAP 3mm", "MAP 1mm"]
pool_in = {c: [] for c in configs}; pool_out = {c: [] for c in configs}
base = SMPLXSkeleton(model_path="models")
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    clip = os.path.basename(path)[:5]; d = np.load(path)
    P = d["landmarks"].astype(float); ok = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    n = len(P); halves = [(slice(0, n // 2), slice(n // 2, n)), (slice(n // 2, n), slice(0, n // 2))]
    width = meds(P, ok)[4]
    for tr, te in halves:
        L = meds(P[tr], ok[tr])
        for c in configs:
            b = np.zeros(NB) if c == "betas=0" else map_fit(L, float(c.split()[1][:-2]))
            sk = base if c == "betas=0" else SMPLXSkeleton(model_path="models", betas=b)
            pool_in[c] += err(P[tr], ok[tr], sk, width); pool_out[c] += err(P[te], ok[te], sk, width)
    print(f"clip {clip}: half-medians differ by up to {np.abs(meds(P[:n//2], ok[:n//2]) - meds(P[n//2:], ok[n//2:])).max()*1000:.1f} mm "
          f"(per segment: {np.round((meds(P[:n//2], ok[:n//2]) - meds(P[n//2:], ok[n//2:]))*1000,1)})")
b_in = np.median(pool_in["betas=0"]) * 100; b_out = np.median(pool_out["betas=0"]) * 100
print("\nPooled direct median error, % shoulder width (fit half -> other half)")
for c in configs:
    i = np.median(pool_in[c]) * 100; o = np.median(pool_out[c]) * 100
    print(f"  {c:<9} in-sample {i:5.2f}% ({100*(b_in-i)/b_in:+5.1f}%)   held-out {o:5.2f}% ({100*(b_out-o)/b_out:+5.1f}% vs betas=0)")
