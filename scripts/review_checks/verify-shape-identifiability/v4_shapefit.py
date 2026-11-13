"""Independent re-run of the 'what does a per-clip shape fit buy' numbers.
MAP betas fitted with the REAL smplx torch forward (autograd Jacobian), not a
hand-rolled linear model; scored with the repo's converter. Splits: full-clip
in-sample, contiguous halves (both directions), and contiguous thirds (fit on 2, test on 1)."""
import os, sys, types, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import least_squares
from vcommon import lengths, length_jacobian, load_clips, MP_SEG, SNAP
os.chdir(SNAP)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body

ARM = [11, 12, 13, 14, 15, 16]
clips = load_clips()

def meds(P, ok):
    return np.array([np.median(np.linalg.norm(P[ok[:, a] & ok[:, b], a] - P[ok[:, a] & ok[:, b], b], axis=1)) for a, b, _ in MP_SEG])

def map_fit(L, sig_mm):
    w = 1e3 / sig_mm
    r = least_squares(lambda b: np.concatenate([(lengths(b) - L) * w, b]), np.zeros(10),
                      jac=lambda b: np.vstack([length_jacobian(b) * w, np.eye(10)]))
    return r.x

def errs(P, ok, sk, width):
    e = []
    for f, o in zip(P, ok):
        try:
            out = reconstruct_body(fit_body(f, o, sk), sk)
        except ValueError:
            continue
        e += [np.linalg.norm(out[m] - f[m]) / width for m in ARM if o[m]]
    return e

def oracle(base, L):
    R = base.rest_joints.copy()
    mid = (R[16] + R[17]) / 2
    for sh, sgn in ((16, 1), (17, -1)):
        u = (R[16] - R[17]) / np.linalg.norm(R[16] - R[17])
        R[sh] = mid + sgn * u * L[4] / 2
    for (p, c), Lk in zip(((16, 18), (18, 20), (17, 19), (19, 21)), L[:4]):
        d = base.rest_joints[c] - base.rest_joints[p]
        R[c] = R[p] + d / np.linalg.norm(d) * Lk
    sk = types.SimpleNamespace(rest_joints=R, parents=base.parents, names=base.names, num_joints=base.num_joints)
    return sk

base = SMPLXSkeleton(model_path="models")
SIGS = [10.0, 3.0, 1.0]
cfgs = ["betas=0"] + [f"MAP{s:g}" for s in SIGS] + ["oracle"]
full = {c: [] for c in cfgs}
split = {k: {c: ([], []) for c in cfgs} for k in ("halves", "thirds")}
bnorm = {}
for clip, (P, V) in clips.items():
    ok = np.isfinite(P).all(-1) & (V >= 0.5)
    width = meds(P, ok)[4]; n = len(P)
    # full clip in-sample
    L = meds(P, ok)
    for c in cfgs:
        if c == "betas=0": sk = base
        elif c == "oracle": sk = oracle(base, L)
        else:
            b = map_fit(L, float(c[3:])); bnorm.setdefault(c, []).append(np.linalg.norm(b))
            sk = SMPLXSkeleton(model_path="models", betas=b)
        full[c] += errs(P, ok, sk, width)
    for kind, K in (("halves", 2), ("thirds", 3)):
        edges = np.linspace(0, n, K + 1).astype(int)
        for k in range(K):
            te = np.zeros(n, bool); te[edges[k]:edges[k + 1]] = True; tr = ~te
            L = meds(P[tr], ok[tr])
            for c in cfgs:
                if c == "betas=0": sk = base
                elif c == "oracle": sk = oracle(base, L)
                else: sk = SMPLXSkeleton(model_path="models", betas=map_fit(L, float(c[3:])))
                split[kind][c][0].extend(errs(P[tr], ok[tr], sk, width))
                split[kind][c][1].extend(errs(P[te], ok[te], sk, width))
    print("done", clip, flush=True)

b0 = np.median(full["betas=0"]) * 100
print(f"\nFULL-CLIP in-sample pooled direct median (% SW); baseline {b0:.2f}%")
for c in cfgs:
    v = np.median(full[c]) * 100
    extra = f"  |beta| per clip {np.round(bnorm[c], 2)}" if c in bnorm else ""
    print(f"  {c:<8} {v:5.2f}%  reduction {100*(b0-v)/b0:5.1f}%{extra}")
for kind in ("halves", "thirds"):
    bi = np.median(split[kind]["betas=0"][0]) * 100; bo = np.median(split[kind]["betas=0"][1]) * 100
    print(f"\n{kind}: baseline in {bi:.2f}% out {bo:.2f}%")
    for c in cfgs:
        i = np.median(split[kind][c][0]) * 100; o = np.median(split[kind][c][1]) * 100
        print(f"  {c:<8} in-sample {i:5.2f}% ({100*(bi-i)/bi:+5.1f}%)  held-out {o:5.2f}% ({100*(bo-o)/bo:+5.1f}%)")
