"""Q5: what does a per-clip, sequence-level shape fit buy on the real round trip?
Runs the repo's converter (fit_body/reconstruct_body) per clip with:
  baseline betas=0 | MAP betas (sigma=10,3,1 mm) | box-bounded unregularised LS |
  per-bone length oracle (skeleton edited directly, NOT reachable by betas).
"""
import copy, glob, os
import numpy as np
from scipy.optimize import least_squares
from common import SNAP, NB, lengths, length_jac, body_stats
os.chdir(SNAP)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body

MP = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
ARM = [11, 12, 13, 14, 15, 16]
fits = np.load("scripts/review_checks/shape-identifiability/map_fits.npy", allow_pickle=True).item()

def load(path):
    d = np.load(path); P = d["landmarks"].astype(np.float64)
    ok = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    return P, ok

def clip_medians(P, ok):
    out = []
    for a, b in MP:
        m = ok[:, a] & ok[:, b]
        out.append(np.median(np.linalg.norm(P[m, a] - P[m, b], axis=-1)))
    return np.array(out)

def oracle_skeleton(base, L):
    sk = copy.copy(base); R = base.rest_joints.copy()
    mid = (R[16] + R[17]) / 2; u = (R[16] - R[17]) / np.linalg.norm(R[16] - R[17])
    newL, newR = mid + u * L[4] / 2, mid - u * L[4] / 2
    for s, e, w, new in ((16, 18, 20, newL), (17, 19, 21, newR)):
        ua = (R[e] - R[s]) / np.linalg.norm(R[e] - R[s]); fa = (R[w] - R[e]) / np.linalg.norm(R[w] - R[e])
        i = 0 if s == 16 else 2
        R[s] = new; R[e] = new + ua * L[i]; R[w] = R[e] + fa * L[i + 1]
    sk.rest_joints = R
    return sk

def box_ls(Lobs):
    return least_squares(lambda b: (lengths(b) - Lobs) * 1000, np.zeros(NB),
                         jac=lambda b: length_jac(b, with_scale=False) * 1000, bounds=(-3, 3)).x

def run(P, ok, sk, width):
    errs = []
    for frame, o in zip(P, ok):
        try:
            out = reconstruct_body(fit_body(frame, o, sk), sk)
        except ValueError:
            continue
        for m in ARM:
            if o[m]:
                errs.append(np.linalg.norm(out[m] - frame[m]) / width)
    return np.array(errs)

base = SMPLXSkeleton(model_path="models")
pooled = {}
print(f"{'clip':<6}{'config':<26}{'|beta|':>7}{'height':>8}{'vol L':>7}{'direct median %SW':>19}")
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    clip = os.path.basename(path)[:5]; P, ok = load(path)
    Lobs = clip_medians(P, ok); width = Lobs[4]
    configs = {"baseline betas=0": np.zeros(NB),
               "MAP sigma=10mm": fits[f"{clip}_10.0"],
               "MAP sigma=3mm": fits[f"{clip}_3.0"],
               "MAP sigma=1mm": fits[f"{clip}_1.0"],
               "unreg LS box|b|<=3": box_ls(Lobs)}
    for name, b in configs.items():
        sk = SMPLXSkeleton(model_path="models", betas=b)
        e = run(P, ok, sk, width); pooled.setdefault(name, []).append(e)
        st = body_stats(b)
        print(f"{clip:<6}{name:<26}{np.linalg.norm(b):7.2f}{st['height_m']*100:8.1f}{st['volume_L']:7.1f}{np.median(e)*100:18.2f}%")
    e = run(P, ok, oracle_skeleton(base, Lobs), width); pooled.setdefault("per-bone length ORACLE", []).append(e)
    print(f"{clip:<6}{'per-bone length ORACLE':<26}{'n/a':>7}{'n/a':>8}{'n/a':>7}{np.median(e)*100:18.2f}%")

print("\nPOOLED direct median (% shoulder width), all clips:")
b0 = np.median(np.concatenate(pooled["baseline betas=0"])) * 100
for name, es in pooled.items():
    v = np.median(np.concatenate(es)) * 100
    print(f"  {name:<26} {v:6.2f}%   reduction vs baseline {100*(b0-v)/b0:5.1f}%")
