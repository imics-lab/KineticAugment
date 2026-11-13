"""Which fields does reconstruct_body actually read? (attribute-access recorder)"""
import numpy as np
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE, fit_body, reconstruct_body
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton

class Rec:
    def __init__(self, obj, log, tag): object.__setattr__(self, "_o", obj); object.__setattr__(self, "_l", log); object.__setattr__(self, "_t", tag)
    def __getattr__(self, n): self._l.add(f"{self._t}.{n}"); return getattr(self._o, n)

sk = SMPLXSkeleton(model_path="models")
rots = np.zeros((55, 3)); rots[16] = [0.2, -0.5, 0.1]; rots[18] = [0, 0.4, 0.3]
J = forward_kinematics(rots, sk)
lm = np.full((33, 3), np.nan)
for s, m in ARM_CORRESPONDENCE: lm[m] = J[s]
f = fit_body(lm, np.isfinite(lm).all(1), sk)
log = set()
reconstruct_body(Rec(f, log, "fit"), Rec(sk, log, "skeleton"))
print("reconstruct_body reads:", sorted(log))
log2 = set()
fit_body(lm, np.isfinite(lm).all(1), Rec(sk, log2, "skeleton"))
print("fit_body reads from skeleton:", sorted(log2))
# 3 valid, one shoulder missing (Step 8 red test expects a raise)
v = np.zeros(33, bool); v[[12, 14, 16]] = True
try:
    fit_body(lm, v, sk); print("R shoulder+elbow+wrist only (no L shoulder): fits, no raise (Step 8 test will require a raise)")
except ValueError as e:
    print("raises:", e)
