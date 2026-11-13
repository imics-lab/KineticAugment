"""Shared linear SMPL-X shape model (exact: joints are linear in betas)."""
import numpy as np

SNAP = "."
NB = 10

_m = np.load(f"{SNAP}/models/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
VT = _m["v_template"].astype(np.float64)              # (V,3)
SD = _m["shapedirs"][:, :, :300].astype(np.float64)    # (V,3,300) shape part
JR = _m["J_regressor"].astype(np.float64)              # (55,V)
F = _m["f"].astype(np.int64)
PARENTS = _m["kintree_table"][0].astype(np.int64)
J0 = JR @ VT                                           # (55,3)
A_full = np.einsum("jv,vck->jck", JR, SD)              # (55,3,300) dJ/dbeta
A = A_full[:, :, :NB]

# Observables used by converter.fit_body (rigid frame on 6 arm joints):
# SMPL-X 16/17 shoulders, 18/19 elbows, 20/21 wrists.
OBS = [  # (parent, child, name)
    (16, 18, "L upper arm"),
    (18, 20, "L forearm"),
    (17, 19, "R upper arm"),
    (19, 21, "R forearm"),
    (16, 17, "shoulder width"),
]


def joints(beta, nb=None):
    beta = np.asarray(beta, float)
    k = len(beta)
    return J0 + np.einsum("jck,k->jc", A_full[:, :, :k], beta)


def verts(beta):
    beta = np.asarray(beta, float)
    return VT + np.einsum("vck,k->vc", SD[:, :, : len(beta)], beta)


def lengths(beta, scale=1.0, obs=OBS):
    J = joints(beta)
    return scale * np.array([np.linalg.norm(J[c] - J[p]) for p, c, _ in obs])


def length_jac(beta, scale=1.0, obs=OBS, with_scale=True):
    """Exact Jacobian of observed lengths wrt (betas[, scale])."""
    beta = np.asarray(beta, float)
    k = len(beta)
    J = joints(beta)
    rows = []
    for p, c, _ in obs:
        d = J[c] - J[p]
        u = d / np.linalg.norm(d)
        rows.append(scale * u @ (A_full[c, :, :k] - A_full[p, :, :k]))
    Jb = np.array(rows)
    if with_scale:
        return np.hstack([Jb, lengths(beta, 1.0, obs)[:, None]])
    return Jb


def mesh_volume(V):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def body_stats(beta):
    """Unobserved anatomy, T-pose (rest), metres / litres."""
    V = verts(beta)
    J = joints(beta)
    leg = lambda h, k, a: np.linalg.norm(J[k] - J[h]) + np.linalg.norm(J[a] - J[k])
    return {
        "height_m": V[:, 1].max() - V[:, 1].min(),
        "L_leg_m": leg(1, 4, 7),
        "R_leg_m": leg(2, 5, 8),
        "hip_width_m": np.linalg.norm(J[1] - J[2]),
        "pelvis_to_neck_m": np.linalg.norm(J[12] - J[0]),
        "volume_L": mesh_volume(V) * 1000,
        "chest_depth_m": V[:, 2].max() - V[:, 2].min(),
    }
