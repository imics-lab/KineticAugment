"""For frames where a modelled landmark (wrist) was INVALID, what does reconstruct_body emit for it?"""
import glob
import numpy as np
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
n_inv = n_finite = 0; errs = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); P = d["landmarks"].astype(float)
    ok = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    for frame, o in zip(P, ok):
        if sum(o[m] for _, m in ARM_CORRESPONDENCE) < 3: continue
        out = reconstruct_body(fit_body(frame, o, sk), sk)
        sw = np.linalg.norm(frame[11] - frame[12])
        for m in (15, 16):
            if not o[m]:
                n_inv += 1
                if np.isfinite(out[m]).all():
                    n_finite += 1
                    errs.append(np.linalg.norm(out[m] - frame[m]) / sw)  # vs MediaPipe's own low-vis guess
print(f"invalid wrist instances: {n_inv}; reconstruct_body emitted a finite position for {n_finite}")
print(f"distance of that emitted (straight-arm default) wrist from MediaPipe's low-visibility estimate: "
      f"median {np.median(errs)*100:.1f}% shoulder width, p95 {np.percentile(errs,95)*100:.1f}%")
