"""Real-data prevalence of (a) defaulted-but-unmarked modelled landmarks,
(b) orphan correspondences, (c) one-arm-only frames; harness threshold >= 0.5."""
import glob, warnings, numpy as np
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
warnings.filterwarnings("ignore")
sk = SMPLXSkeleton(model_path="models"); ARM = [m for _, m in ARM_CORRESPONDENCE]
tot = fitted = raised = 0; defaulted = 0; defaulted_rows = 0; orphan = 0; one_arm = 0
derr = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    z = np.load(f); Ls, Vs = z["landmarks"].astype(np.float64), z["visibility"]
    for L, vis in zip(Ls, Vs):
        tot += 1
        ok = np.isfinite(L).all(1) & (vis >= 0.5)
        orphan += int((not ok[13] and ok[15]) or (not ok[14] and ok[16]) or (not ok[11] and ok[13]) or (not ok[12] and ok[14]))
        one_arm += int(ok[ARM].sum() >= 3 and (not ok[[12, 14, 16]].any() or not ok[[11, 13, 15]].any()))
        try:
            out = reconstruct_body(fit_body(L, ok, sk), sk)
        except ValueError:
            raised += 1; continue
        fitted += 1
        inv = [m for m in ARM if not ok[m]]
        fin = [m for m in inv if np.isfinite(out[m]).all()]
        defaulted += int(len(fin) > 0); defaulted_rows += len(fin)
        for m in fin:
            if np.isfinite(L[m]).all():
                sw = np.linalg.norm(L[11] - L[12]); derr.append(np.linalg.norm(out[m] - L[m]) / sw)
print(f"frames {tot}: fitted {fitted}, raised {raised}")
print(f"frames whose output contains a finite value for an INVALID modelled landmark: {defaulted} ({100*defaulted/fitted:.1f}% of fitted), rows {defaulted_rows}")
if derr: print(f"  those defaulted values vs the (low-visibility) raw landmark: median {100*np.median(derr):.1f}% shoulder width, p95 {100*np.percentile(derr,95):.1f}%")
print(f"orphan configurations: {orphan};  one-arm-only frames: {one_arm}")
