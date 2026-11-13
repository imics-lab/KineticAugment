"""Step 8 adds the nose (rigid to the head under V1's neck/head=0 policy). If the
shoulder-shoulder-nose triangle were treated as rigid, how much shape info would it add?"""
import numpy as np
from smplx.vertex_ids import vertex_ids
from common import NB, OBS, A_full, SD, VT, joints, length_jac
NOSE = vertex_ids["smplx"]["nose"]
def nose(b): return VT[NOSE] + SD[NOSE, :, :NB] @ b
def jac_nose(b, j):
    d = nose(b) - joints(b)[j]; u = d / np.linalg.norm(d)
    return u @ (SD[NOSE, :, :NB] - A_full[j, :, :NB])
b0 = np.zeros(NB)
J5 = length_jac(b0, with_scale=False)
J7 = np.vstack([J5, jac_nose(b0, 16), jac_nose(b0, 17)])
for tag, M in (("5 lengths (V1 converter)", J5), ("5 lengths + nose-Lsh + nose-Rsh", J7)):
    s = np.linalg.svd(M, compute_uv=False)
    print(f"{tag:<34} rank={np.linalg.matrix_rank(M, tol=s[0]*1e-6)} sv(mm/sigma)={np.round(s*1000,3)}")
print(f"nose->L shoulder rest {np.linalg.norm(nose(b0)-joints(b0)[16])*1000:.1f} mm; "
      f"cos(L,R nose rows)={J7[5]@J7[6]/np.linalg.norm(J7[5])/np.linalg.norm(J7[6]):.4f}")
for sL in (3, 10):
    s = sL / 1000
    for tag, M in (("5", J5), ("7", J7)):
        sv = np.linalg.svd(M, compute_uv=False); print(f"  sigma_L={sL}mm obs={tag}: eff.dof={(sv**2/(sv**2+s**2)).sum():.2f}")
