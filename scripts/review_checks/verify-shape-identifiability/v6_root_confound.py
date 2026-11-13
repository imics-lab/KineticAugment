"""Is 'root pose' a real confounder of a shape fit, as the smplx audit says and the
tester calls over-broad? Truth = TEMPLATE body (betas=0) with a rotated root.
Fit betas by minimising the repo converter's position residual (the 'small
optimisation' Step 10 contemplates). If betas move away from 0 and reduce the
error, root misestimation is being absorbed as shape.
Also: posterior sd of height with the real-smplx Jacobian."""
import os, types, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from vcommon import *
os.chdir(SNAP)
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
m = model(10)
VT = m.v_template.detach().numpy().astype(float); SDIR = m.shapedirs.detach().numpy().astype(float)
JR = m.J_regressor.detach().numpy().astype(float); PAR = m.parents.numpy().astype(int)[:55]
def skel(b):
    R = (JR @ (VT + SDIR @ b))[:55]
    return types.SimpleNamespace(rest_joints=R, parents=PAR, names=[str(i) for i in range(55)], num_joints=55)
rng = np.random.default_rng(5)
frames = []
for t in range(12):
    bp = np.zeros((21, 3))
    for j in (15, 16, 17, 18): bp[j] = rng.normal(0, 0.5, 3)
    go = Rotation.from_euler("xyz", rng.normal(0, 1, 3) / np.sqrt(3) * np.radians(30)).as_rotvec()
    with torch.no_grad():
        J = forward(np.zeros(10), bp.reshape(63), go, np.zeros(3)).joints[0, :55].numpy()
    lm = np.full((33, 3), np.nan)
    for s, mm in ARM_CORRESPONDENCE: lm[mm] = J[s]
    frames.append(lm)
def err(b):
    sk = skel(b); e = []
    for lm in frames:
        ok = np.isfinite(lm).all(1)
        rec = reconstruct_body(fit_body(lm, ok, sk), sk)
        e += list(np.linalg.norm(rec[ok] - lm[ok], axis=1))
    return np.array(e)
e0 = err(np.zeros(10))
print(f"truth betas=0, root rotated ~30deg: converter error at betas=0: mean {e0.mean()*1e3:.2f} mm, max {e0.max()*1e3:.2f} mm")
for lam in (1e-6, 1e-4):
    r = minimize(lambda b: np.mean(err(b)**2) + lam * b @ b, np.zeros(10), method="Powell", options=dict(maxiter=3000, xtol=1e-3, ftol=1e-10))
    e1 = err(r.x)
    print(f" position-residual shape fit (lam={lam:g}): |beta|={np.linalg.norm(r.x):.2f}  mean err {e1.mean()*1e3:.2f} mm (from {e0.mean()*1e3:.2f}), height {height(r.x)*100:.1f} cm vs truth {height(np.zeros(10))*100:.1f} cm")
    print("   beta =", np.round(r.x, 2))

# posterior sd of height, real-smplx Jacobian
b0 = np.zeros(10); Jl = length_jacobian(b0)
h = 1e-3; g = np.array([(height(b0 + h*e) - height(b0 - h*e)) / (2*h) for e in np.eye(10)])
print(f"\nprior sd height {np.linalg.norm(g)*100:.2f} cm")
for sL in (1, 3, 10, 20):
    P = np.linalg.inv(Jl.T @ Jl / (sL/1e3)**2 + np.eye(10))
    print(f" sigma_L={sL:2d} mm: posterior sd height {np.sqrt(g @ P @ g)*100:.2f} cm")
