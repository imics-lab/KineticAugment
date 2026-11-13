"""float64: FK using smplx's exact (slightly non-orthogonal) rodrigues matrices."""
import numpy as np, smplx, torch
from smplx.lbs import batch_rodrigues
torch.set_grad_enabled(False)
rng = np.random.default_rng(99)
m = smplx.create(model_path="models", model_type="smplx", gender="neutral", use_pca=False,
                 flat_hand_mean=True, batch_size=1, dtype=torch.float64)
par = m.parents[:55].numpy().astype(int)
def fk_mats(R, rest, transl):
    off = rest.copy(); off[1:] -= rest[par[1:]]
    W = np.zeros((55, 3, 3)); P = np.zeros((55, 3))
    for j in range(55):
        p = par[j]
        if p < 0: W[j], P[j] = R[j], off[j]
        else: W[j] = W[p] @ R[j]; P[j] = W[p] @ off[j] + P[p]
    return P + transl
from scipy.spatial.transform import Rotation
w_scipy = w_rod = 0.0
for _ in range(20):
    betas = rng.normal(0, 2, 10)
    v = m.v_template + torch.einsum("l,mkl->mk", torch.tensor(betas), m.shapedirs[:, :, :10])
    rest = (m.J_regressor @ v)[:55].numpy()
    rots = rng.normal(0, 0.6, (55, 3)); rots[0] = rng.normal(0, 1.2, 3); tr = rng.normal(0, 1, 3)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)[None]
    ref = m(betas=T(betas), global_orient=T(rots[0]), body_pose=T(rots[1:22].reshape(-1)), jaw_pose=T(rots[22]),
            leye_pose=T(rots[23]), reye_pose=T(rots[24]), left_hand_pose=T(rots[25:40].reshape(-1)),
            right_hand_pose=T(rots[40:55].reshape(-1)), transl=T(tr), expression=torch.zeros(1, 10, dtype=torch.float64),
            return_verts=False).joints[0, :55].numpy()
    w_scipy = max(w_scipy, np.abs(fk_mats(Rotation.from_rotvec(rots).as_matrix(), rest, tr) - ref).max())
    w_rod = max(w_rod, np.abs(fk_mats(batch_rodrigues(torch.tensor(rots)).numpy(), rest, tr) - ref).max())
print(f"float64: FK with scipy rotations max {w_scipy:.2e} m; FK with smplx.lbs.batch_rodrigues matrices max {w_rod:.2e} m")
