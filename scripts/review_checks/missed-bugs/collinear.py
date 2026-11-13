"""
fit_body accepts any 3 valid correspondences (MIN_CORRESPONDENCES = 3) and has no rank check.
One arm only (shoulder, elbow, wrist) that is nearly straight is a rank-1 point set: the rotation
about the arm line is unconstrained, so global_orient (the whole torso) is decided by noise.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")

d = np.array([0.52, 0.21, 0.83]); d /= np.linalg.norm(d)
p = np.array([0.18, -0.45, 0.05])
base = np.full((33, 3), np.nan)
base[11], base[13], base[15] = p, p + 0.27 * d, p + 0.52 * d
v = np.isfinite(base).all(1)
rng = np.random.default_rng(0)
for sigma in (0.0, 1e-6, 1e-4, 1e-3):
    gos, errs = [], []
    for k in range(8):
        lm = base.copy()
        if sigma:
            lm[[11, 13, 15]] += rng.normal(0, sigma, (3, 3))
        f = fit_body(lm, v, sk)
        gos.append(Rotation.from_rotvec(f.global_orient))
        o = reconstruct_body(f, sk)
        errs.append(max(np.linalg.norm(o[i] - lm[i]) for i in (11, 13, 15)))
    spread = max((a.inv() * b).magnitude() for a in gos for b in gos)
    print(f"  noise {sigma*1000:7.3f} mm: global_orient spread across 8 draws {np.degrees(spread):6.1f} deg; "
          f"|global_orient| {np.degrees(gos[0].magnitude()):6.1f} deg; observed-landmark max err {1000*max(errs):.1f} mm")
# where does the unobserved right shoulder (i.e. the torso) end up across draws?
rs = []
for k in range(8):
    lm = base.copy(); lm[[11, 13, 15]] += rng.normal(0, 1e-4, (3, 3))
    rs.append(reconstruct_body(fit_body(lm, v, sk), sk)[12])
rs = np.array(rs)
print(f"  reconstructed (unobserved) R shoulder across 8 draws at 0.1 mm noise: max pairwise distance "
      f"{1000*max(np.linalg.norm(a-b) for a in rs for b in rs):.0f} mm")
