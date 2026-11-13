"""Supplement to s1: 10-beta-only Jacobian (scale fixed, as converter.py) at random betas,
plus the scale column expressed per 1% so units are comparable."""
import numpy as np
from common import NB, length_jac, OBS
rng = np.random.default_rng(0)
pts = [("betas=0", np.zeros(NB))] + [(f"N(0,I) #{i}", rng.normal(size=NB)) for i in range(5)] + \
      [(f"N(0,4I) #{i}", 2 * rng.normal(size=NB)) for i in range(3)]
for tag, b in pts:
    for lab, M in (("10b (5 obs)", length_jac(b, with_scale=False)),
                   ("10b (4 arm)", length_jac(b, obs=OBS[:4], with_scale=False)),
                   ("10b+scale/1% (5 obs)", length_jac(b) * np.r_[np.ones(NB), 0.01])):
        s = np.linalg.svd(M, compute_uv=False)
        print(f"{tag:<12} {lab:<22} rank={int((s > s[0]*1e-6).sum())} nullity={M.shape[1]-int((s > s[0]*1e-6).sum())} "
              f"sv(mm/unit)={np.round(s*1000,3)} cond={s[0]/s[-1]:.0f}")
