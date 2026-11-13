"""Q2/Q5: what the 5 lengths actually teach about shape, under the SMPL-X prior
beta ~ N(0, I) and Gaussian length noise sigma_L (linear-Gaussian, exact here
because lengths are near-linear in beta over the prior's range).

Also: is global scale separable from betas? Give scale a prior sd and see how
much its posterior shrinks and how it correlates with beta0.
"""
import numpy as np
from common import OBS, NB, lengths, length_jac, body_stats

np.set_printoptions(precision=3, suppress=True, linewidth=150)
b0 = np.zeros(NB)
J = length_jac(b0, with_scale=False)                       # (5,10) m / sigma
Js = length_jac(b0, with_scale=True)[:, NB:]               # (5,1) m / unit scale

# gradients of unobserved anatomy (finite differences; body_stats is piecewise smooth)
keys = ["height_m", "L_leg_m", "hip_width_m", "pelvis_to_neck_m", "volume_L", "chest_depth_m"]
def grad(f, h=1e-3):
    return np.array([(f(b0 + h*e) - f(b0 - h*e)) / (2*h) for e in np.eye(NB)])
G = {k: grad(lambda b, k=k: body_stats(b)[k]) for k in keys}
unit = {"height_m": (100, "cm"), "L_leg_m": (100, "cm"), "hip_width_m": (100, "cm"),
        "pelvis_to_neck_m": (100, "cm"), "volume_L": (1, "L"), "chest_depth_m": (100, "cm")}

print("Prior sd of unobserved quantities under beta~N(0,I) (linearised):")
for k in keys:
    m, u = unit[k]; print(f"   {k:<18} {np.linalg.norm(G[k])*m:6.2f} {u}")

print("\n=== Betas only (scale fixed = 1, as converter.py does) ===")
for sL in (1, 3, 5, 10, 20):
    s = sL / 1000
    P = np.linalg.inv(J.T @ J / s**2 + np.eye(NB))
    ev = np.linalg.eigvalsh(P)                     # posterior/prior variance ratios along eigendirs
    learned = int((ev < 0.5).sum())
    sv = np.linalg.svd(J, compute_uv=False)
    dof = float((sv**2 / (sv**2 + s**2)).sum())    # effective number of parameters
    row = "  ".join(f"{k.split('_')[0]}:{np.linalg.norm(G[k])*unit[k][0]:.1f}->{np.sqrt(G[k] @ P @ G[k])*unit[k][0]:.1f}"
                    for k in keys)
    print(f" sigma_L={sL:>2} mm: eff.dof={dof:4.2f}  dims with var halved={learned}  "
          f"post/prior var eig={np.sort(ev)[:5]}")
    print(f"     prior->posterior sd: {row}")

print("\n=== Betas + free scale with prior sd s_sd (both metric => scale should be 1) ===")
for s_sd in (0.01, 0.03, 0.05, 0.10):
    for sL in (3, 10):
        s = sL / 1000
        A = np.hstack([J, Js * s_sd])               # scale in units of its prior sd
        P = np.linalg.inv(A.T @ A / s**2 + np.eye(NB + 1))
        sd = np.sqrt(np.diag(P))
        corr = P[0, NB] / np.sqrt(P[0, 0] * P[NB, NB])
        print(f" scale prior sd={s_sd*100:4.1f}%  sigma_L={sL:>2}mm: scale posterior sd={sd[NB]*s_sd*100:5.2f}% "
              f"(shrink {sd[NB]:.2f}x prior), beta0 post sd={sd[0]:.2f}, corr(beta0,scale)={corr:+.3f}")

# Exactness of the redundancy: unit-norm direction in beta space that mimics +1% scale on the 5 lengths
x, *_ = np.linalg.lstsq(J, Js[:, 0] * 0.01, rcond=None)
print(f"\nmin-norm beta reproducing +1% scale on all 5 lengths exactly: |beta|={np.linalg.norm(x):.3f}, beta={x}")
print(f"  residual {np.linalg.norm(J@x - Js[:,0]*0.01)*1000:.2e} mm; "
      f"its side effect on height {G['height_m']@x*100:+.2f} cm (pure 1% scale would be {1.721:.2f} cm)")
x0 = Js[:, 0] @ J[:, 0] / (J[:, 0] @ J[:, 0]) * 0.01
r0 = np.linalg.norm(J[:, 0] * x0 - Js[:, 0] * 0.01) / np.linalg.norm(Js[:, 0] * 0.01)
print(f"  beta0 alone: {x0:.3f} sigma reproduces 1% scale with relative residual {r0*100:.1f}% on the 5 lengths")
