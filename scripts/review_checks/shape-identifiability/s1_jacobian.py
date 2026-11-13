"""Q1/Q2: rank, singular values and conditioning of observed-length Jacobian."""
import numpy as np
from common import OBS, lengths, length_jac, J0, A, NB

np.set_printoptions(precision=4, suppress=True, linewidth=140)
names = [n for *_, n in OBS]
print("Rest lengths (betas=0), m:", dict(zip(names, lengths(np.zeros(NB)).round(4))))

def report(tag, M):
    s = np.linalg.svd(M, compute_uv=False)
    tol = s[0] * 1e-6
    print(f"{tag:<44} shape={M.shape} rank(tol 1e-6)={int((s>tol).sum())} "
          f"sv={s} cond={s[0]/s[-1]:.3g}")
    return s

rng = np.random.default_rng(0)
points = [("betas=0", np.zeros(NB))] + [(f"betas~N(0,I) #{i}", rng.normal(size=NB)) for i in range(5)] \
         + [(f"betas~N(0,2^2) #{i}", 2*rng.normal(size=NB)) for i in range(3)]
print("\n--- Jacobian of the 5 observed lengths (m per unit beta; scale col = m per unit scale) ---")
for tag, b in points:
    report(tag + " [10 betas + scale]", length_jac(b))
print()
for tag, b in points[:1]:
    report(tag + " [10 betas only]", length_jac(b, with_scale=False))
    report(tag + " [4 arm lengths, 10 betas]", length_jac(b, obs=OBS[:4], with_scale=False))
    report(tag + " [4 arm lengths, 10 betas+scale]", length_jac(b, obs=OBS[:4]))

# Relative (log-length) Jacobian: unit-free, per 1 sigma of beta, in %
Jb = length_jac(np.zeros(NB), with_scale=False)
L0 = lengths(np.zeros(NB))
print("\n--- d(length)/d(beta_k) at betas=0, in mm per 1 sigma ---")
print("            " + " ".join(f"b{k:<5d}" for k in range(NB)))
for n, row in zip(names, Jb * 1000):
    print(f"{n:<14}" + " ".join(f"{v:6.2f}" for v in row))

# L/R symmetry
print("\n--- Left/right duplication ---")
def cos(a, b): return a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print(f"cos(L upper, R upper) = {cos(Jb[0], Jb[2]):.6f}   |L-R|/|L| = {np.linalg.norm(Jb[0]-Jb[2])/np.linalg.norm(Jb[0]):.4f}")
print(f"cos(L fore , R fore ) = {cos(Jb[1], Jb[3]):.6f}   |L-R|/|L| = {np.linalg.norm(Jb[1]-Jb[3])/np.linalg.norm(Jb[1]):.4f}")
T = np.array([[.5,0,.5,0,0],[0,.5,0,.5,0],[0,0,0,0,1],[1,0,-1,0,0],[0,1,0,-1,0]])
Jt = T @ Jb
tn = ["upper mean", "fore mean", "shoulder W", "upper L-R", "fore L-R"]
print("max |d/dbeta| per 1 sigma (mm) of symmetric/antisymmetric combos:")
for n, row in zip(tn, Jt*1000):
    print(f"  {n:<12} norm={np.linalg.norm(row):7.3f} mm/sigma   row={row}")
s = np.linalg.svd(Jb, compute_uv=False)
print("SVD (10 betas, 5 lengths), mm per unit-norm beta:", s*1000)
U, s, Vt = np.linalg.svd(Jb)
print("left singular vectors (columns, in [Lup,Lfo,Rup,Rfo,SW]):")
print(U)

# Using more of the shape space (up to 300 betas) - does antisymmetric become reachable?
for nb in (10, 16, 50, 100, 300):
    Jb_n = length_jac(np.zeros(nb), with_scale=False)
    s = np.linalg.svd(Jb_n, compute_uv=False)
    Jt_n = T @ Jb_n
    print(f"nb={nb:<4} sv(mm/sigma)={np.round(s*1000,3)} | antisym norms upper={np.linalg.norm(Jt_n[3])*1000:.3f} fore={np.linalg.norm(Jt_n[4])*1000:.3f} mm/sigma-ball")

# template asymmetry itself
print("\nTemplate L-R asymmetry at betas=0 (mm): upper", (L0[0]-L0[2])*1000, " fore", (L0[1]-L0[3])*1000)

# Scale redundancy (length space)
Js = length_jac(np.zeros(NB))
x, res, rk, _ = np.linalg.lstsq(Js[:, :NB], Js[:, NB], rcond=None)
print("\n--- Scale vs betas in LENGTH space ---")
print("scale column L0 =", L0, " best beta combo:", x, " residual norm:", np.linalg.norm(Js[:, :NB] @ x - Js[:, NB]))
print("cos(d/d beta0, d/d scale) =", cos(Js[:, 0], Js[:, NB]))

# Scale vs betas in FULL JOINT space (all 55 joints, root-relative)
D = (J0 - J0[0]).reshape(-1)
Af = (A - A[0:1]).reshape(-1, NB)
x, *_ = np.linalg.lstsq(Af, D, rcond=None)
r = np.linalg.norm(Af @ x - D) / np.linalg.norm(D)
print("\n--- Scale vs betas in FULL 55-joint space (root-relative) ---")
print(f"best beta approx of 1% uniform scale: beta={x*0.01}  relative residual={r:.4f}")
print(f"cos(dJ/dbeta0, uniform-scale direction) = {cos(Af[:,0], D):.4f}")
for k in range(3):
    print(f"cos(dJ/dbeta{k}, scale) = {cos(Af[:,k], D):.4f}")
