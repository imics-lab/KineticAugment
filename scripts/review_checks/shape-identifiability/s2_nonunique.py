"""Q3: two clearly different bodies with (near-)identical observed lengths."""
import numpy as np
from scipy.optimize import minimize
from common import OBS, NB, lengths, length_jac, body_stats

np.set_printoptions(precision=3, suppress=True, linewidth=140)
names = [n for *_, n in OBS]
BOUND = 3.0
RADIUS = 3.06  # ||beta|| <= chi2(10) MEDIAN radius: each body is more typical than half the prior

def find(target_L, objective, sign, x0):
    cons = {"type": "eq", "fun": lambda b: (lengths(b) - target_L) * 1000.0,
            "jac": lambda b: length_jac(b, with_scale=False) * 1000.0}
    ball = {"type": "ineq", "fun": lambda b: RADIUS**2 - b @ b, "jac": lambda b: -2 * b}
    res = minimize(lambda b: -sign * objective(b), x0, method="SLSQP",
                   constraints=[cons, ball], bounds=[(-BOUND, BOUND)] * NB,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x, res

def show(tag, b, ref_L):
    st = body_stats(b)
    L = lengths(b)
    print(f"  {tag:<10} |beta|={np.linalg.norm(b):5.2f}  beta={b}")
    print(f"  {'':<10} obs lengths (mm)={np.round(L*1000,2)}  max |dL| vs ref = {np.abs(L-ref_L).max()*1000:.4f} mm")
    print(f"  {'':<10} height={st['height_m']*100:6.1f} cm  L leg={st['L_leg_m']*100:5.1f} cm  hip W={st['hip_width_m']*100:5.1f} cm  "
          f"pelvis->neck={st['pelvis_to_neck_m']*100:5.1f} cm  volume={st['volume_L']:6.1f} L  depth={st['chest_depth_m']*100:5.1f} cm")
    return st

targets = {"betas=0 (template)": np.zeros(NB)}
rng = np.random.default_rng(3)
r = rng.normal(size=NB); targets["random N(0,I) body (|b|=1.5)"] = r / np.linalg.norm(r) * 1.5

for tname, b_ref in targets.items():
    L_ref = lengths(b_ref)
    print(f"\n=== Observed lengths fixed to those of: {tname}  L={np.round(L_ref*1000,2)} mm ===")
    show("reference", b_ref, L_ref)
    for qname, f in [("height", lambda b: body_stats(b)["height_m"]),
                     ("leg", lambda b: body_stats(b)["L_leg_m"]),
                     ("volume", lambda b: body_stats(b)["volume_L"])]:
        hi, r1 = find(L_ref, f, +1, b_ref.copy())
        lo, r2 = find(L_ref, f, -1, b_ref.copy())
        print(f" -- extremise {qname} (SLSQP ok: {r1.success}/{r2.success})")
        s_hi = show(f"{qname} MAX", hi, L_ref)
        s_lo = show(f"{qname} MIN", lo, L_ref)
        print(f"     => difference: height {100*(s_hi['height_m']-s_lo['height_m']):+.1f} cm, "
              f"L leg {100*(s_hi['L_leg_m']-s_lo['L_leg_m']):+.1f} cm, volume {s_hi['volume_L']-s_lo['volume_L']:+.1f} L, "
              f"|beta_hi - beta_lo| = {np.linalg.norm(hi-lo):.2f} sigma")
        np.save(f"pair_{qname}_{'tmpl' if 'template' in tname else 'rand'}.npy", np.stack([hi, lo]))

# Prior plausibility: chi^2(10) 95% quantile = 18.31 -> |beta| < 4.28
print("\nAll solutions constrained to ||beta|| <= 3.06 (chi2(10) median radius).")
print("chi2(10) 95% radius = 4.28; 99% radius = 4.80 (|beta| below these is typical under the SMPL-X prior)")
