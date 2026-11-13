"""Shared helpers for the twist-identifiability cluster (A003). float64 SMPL-X forward."""
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as R

torch.set_default_dtype(torch.float64)
MODEL = smplx.create("models", model_type="smplx", gender="neutral", use_pca=False,
                     flat_hand_mean=True, batch_size=1, dtype=torch.float64)
with torch.no_grad():
    REST = (MODEL.J_regressor @ MODEL.v_template)[:55].numpy()

LS, RS, LE, RE, LW, RW = 16, 17, 18, 19, 20, 21
# (shoulder, elbow, wrist, anatomical flexion axis in elbow-local frame).
# Left elbow flexes about -y, right about +y (SMPLify angle prior penalises +y left / -y right;
# verified geometrically in t5: -y on the left elbow carries the forearm to body-forward +z).
ARMS = {"L": (LS, LE, LW, np.array([0.0, -1.0, 0.0])),
        "R": (RS, RE, RW, np.array([0.0, 1.0, 0.0]))}


def unit(v):
    return v / np.linalg.norm(v)


def rest_dirs(side):
    s, e, w, _ = ARMS[side]
    return unit(REST[e] - REST[s]), unit(REST[w] - REST[e])


def forward(pose55, transl=None):
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float64)).reshape(1, -1)
    with torch.no_grad():
        out = MODEL(global_orient=t(pose55[0]), body_pose=t(pose55[1:22]),
                    left_hand_pose=t(pose55[25:40]), right_hand_pose=t(pose55[40:55]),
                    jaw_pose=t(pose55[22]), leye_pose=t(pose55[23]), reye_pose=t(pose55[24]),
                    transl=t(np.zeros(3) if transl is None else transl), return_verts=True)
    return out.joints[0, :55].numpy(), out.vertices[0].numpy()


def geo_deg(a, b):
    """geodesic angle between two rotations given as rotvecs or Rotation objects."""
    ra = a if isinstance(a, R) else R.from_rotvec(a)
    rb = b if isinstance(b, R) else R.from_rotvec(b)
    return np.degrees((ra.inv() * rb).magnitude())


def twist_deg(rot, axis):
    """signed twist (deg) of `rot` about unit `axis` from the swing-twist decomposition."""
    q = rot.as_quat()
    t = np.degrees(2 * np.arctan2(np.dot(q[:3], axis), q[3]))
    return (t + 180.0) % 360.0 - 180.0


def swing(d_from, d_to):
    ax = np.cross(d_from, d_to)
    s = np.linalg.norm(ax); c = np.dot(d_from, d_to)
    if s < 1e-12:
        return R.identity()
    return R.from_rotvec(ax / s * np.arctan2(s, c))


def frame_rot(a1, a2, b1, b2):
    """rotation carrying the orthonormal frame built from (a1,a2) onto the one from (b1,b2).
    Exact when angle(a1,a2) == angle(b1,b2)."""
    def fr(x, y):
        e1 = unit(x); e3 = unit(np.cross(x, y)); e2 = np.cross(e3, e1)
        return np.stack([e1, e2, e3], axis=1)
    return R.from_matrix(fr(b1, b2) @ fr(a1, a2).T)


def hinge_ik(side, u_local, v_local):
    """Hinge-constrained arm IK in the shoulder's PARENT frame.
    u_local, v_local: observed humerus / forearm unit directions expressed in the shoulder's parent
    frame. Elbow restricted to R_axis(phi) about the anatomical flexion axis (SMPL-X y).
    Returns list of (R_shoulder, phi_deg) solutions (0, 1 or 2 branches); phi>0 = flexion."""
    h, f = rest_dirs(side)
    ax = ARMS[side][3]
    cos_obs = np.clip(np.dot(u_local, v_local), -1, 1)
    # h . R_ax(phi) f = A + B cos phi + C sin phi   (Rodrigues about unit ax)
    fpar = np.dot(f, ax) * ax; fperp = f - fpar
    A = np.dot(h, fpar); B = np.dot(h, fperp); C = np.dot(h, np.cross(ax, f))
    rho = np.hypot(B, C)
    k = (cos_obs - A) / rho
    if abs(k) > 1:
        return []
    base = np.arctan2(C, B); d = np.arccos(k)
    sols = []
    for phi in (base + d, base - d):
        g = R.from_rotvec(ax * phi).apply(f)
        Rs = frame_rot(h, g, u_local, v_local)
        sols.append((Rs, np.degrees((phi + np.pi) % (2 * np.pi) - np.pi)))
    return sols
