import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, smplx
from scipy.optimize import minimize
from scipy.stats import chi2
from vcommon import *
os.chdir(SNAP)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
np.set_printoptions(precision=3, suppress=True, linewidth=160)
rng = np.random.default_rng(11)

def lm_from(beta, bp, go=np.zeros(3), tr=np.zeros(3)):
    with torch.no_grad():
        J = forward(beta, bp, go, tr).joints[0, :55].numpy()
    lm = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE: lm[m] = J[s]
    return lm, J

print("=== (a) BodyFit.betas contract with a shaped skeleton ===")
for beta in (np.array([-2.0, 1.0, 0, 0.5, 0, 0, 0, 0, 0, 1.0]), np.r_[1.0, np.zeros(9)]):
    sk = SMPLXSkeleton(model_path="models", betas=beta)
    bp = np.zeros((21, 3))
    for j in (15, 16, 17, 18): bp[j] = rng.normal(0, 0.4, 3)
    lm, _ = lm_from(beta, bp.reshape(63), tr=np.array([0.1, -0.3, 2.0]))
    ok = np.isfinite(lm).all(1)
    fit = fit_body(lm, ok, sk)
    rec = reconstruct_body(fit, sk)
    _, Jret = lm_from(fit.betas, fit.body_pose, fit.global_orient, fit.transl)
    _, Jtru = lm_from(beta, fit.body_pose, fit.global_orient, fit.transl)
    e_ret = max(np.linalg.norm(rec[m] - Jret[s]) for s, m in ARM_CORRESPONDENCE) * 1e3
    e_tru = max(np.linalg.norm(rec[m] - Jtru[s]) for s, m in ARM_CORRESPONDENCE) * 1e3
    e_in = np.nanmax(np.linalg.norm(rec - lm, axis=1)) * 1e3
    print(f" betas={beta[:4]}..: fit.betas nonzero? {np.any(fit.betas != 0)}; rec vs input {e_in:.4f} mm; "
          f"rec vs smplx(fit.betas) {e_ret:.2f} mm; rec vs smplx(true betas) {e_tru:.4f} mm")

print("\n=== (b) SMPLXSkeleton with 16 betas ===")
try:
    SMPLXSkeleton(model_path="models", betas=np.zeros(16)); print(" accepted")
except Exception as ex:
    print(" raised", type(ex).__name__, str(ex)[:100])

print("\n=== (c) fore/upper ratio reach within prior radius (real smplx, float64) ===")
def ratio(b, side):
    L = lengths(b)
    return {"L": L[1]/L[0], "R": L[3]/L[2], "pooled": (L[1]+L[3])/(L[0]+L[2])}[side]
r95 = np.sqrt(chi2.ppf(0.95, 10)); print(f" chi2(10) 95% radius {r95:.3f}")
for nb, rad in ((10, r95), (10, 6.0), (10, 10.0)):
    out = []
    for side in ("L", "R", "pooled"):
        ext = []
        for sgn in (1, -1):
            best = None
            for s in range(3):
                x0 = np.random.default_rng(s).normal(0, 0.1, nb)
                r = minimize(lambda b: -sgn * ratio(b, side), x0, method="SLSQP",
                             constraints=[dict(type="ineq", fun=lambda b: rad**2 - b @ b)])
                v = ratio(r.x, side)
                best = v if best is None else (max(best, v) if sgn > 0 else min(best, v))
            ext.append(best)
        out.append(f"{side} {min(ext):.3f}-{max(ext):.3f}")
    print(f" nb={nb} |b|<={rad:.2f}: " + "  ".join(out))
print(f" template: L {ratio(np.zeros(10),'L'):.3f} R {ratio(np.zeros(10),'R'):.3f} pooled {ratio(np.zeros(10),'pooled'):.3f}")

print("\n=== (d) scale vs betas, posterior with real-smplx Jacobian ===")
b0 = np.zeros(10); J = length_jacobian(b0); L0 = lengths(b0)
for ssd in (0.01, 0.03, 0.05):
    for sL in (3, 10):
        s = sL / 1e3
        A = np.hstack([J, L0[:, None] * ssd])
        Pp = np.linalg.inv(A.T @ A / s**2 + np.eye(11))
        print(f" scale prior sd {ssd*100:.0f}% sigma_L {sL:2d}mm: post/prior sd {np.sqrt(Pp[10,10]):.2f}, corr(b0,scale) {Pp[0,10]/np.sqrt(Pp[0,0]*Pp[10,10]):+.2f}")

print("\n=== (e) two bodies, same 5 lengths, through the converter (pairs from v2, real smplx) ===")
pairs = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.', 'pairs.npy')) if False else np.load("scripts/review_checks/verify-shape-identifiability/pairs.npy")
hi, lo = pairs[0]
print(" heights", height(hi)*100, height(lo)*100, "  max |dL| mm", np.abs(lengths(hi)-lengths(lo)).max()*1e3)
skA, skB, skD = (SMPLXSkeleton(model_path="models", betas=b) for b in (hi, lo, np.r_[1.0, np.zeros(9)]))
res = {"A": [], "B": [], "D(+1sd b0)": []}
for t in range(60):
    bp = np.zeros((21, 3))
    for j in (15, 16, 17, 18): bp[j] = rng.normal(0, 0.5, 3)
    lm, _ = lm_from(hi, bp.reshape(63), tr=rng.normal(0, 0.3, 3))
    ok = np.isfinite(lm).all(1)
    for k, sk in (("A", skA), ("B", skB), ("D(+1sd b0)", skD)):
        rec = reconstruct_body(fit_body(lm, ok, sk), sk)
        res[k] += list(np.linalg.norm(rec[ok] - lm[ok], axis=1) * 1e3)
for k, v in res.items():
    print(f"  truth=A, fit skeleton {k:<11}: median {np.median(v):.3f} mm  max {np.max(v):.3f} mm")
