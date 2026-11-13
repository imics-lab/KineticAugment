"""Independent closed-form oracle decomposition (verify-real-data-numbers).

Model of the converter's reconstructed arm landmarks (spine/collars held at zero):
  shoulders = p0 + R (rest_shoulder - p0)          [R = root rotation about rest pelvis]
  elbow     = shoulder + L_up   * unit(obs elbow - obs shoulder)
  wrist     = elbow    + L_fore * unit(obs wrist - obs elbow)
IK places each solved bone exactly along its observed world direction for any root R.
Validated against fit_body's IK+FK (root 0) and against solve_swing_ik with a given root.
"""
import glob, itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE

MP = [11, 12, 13, 14, 15, 16]
SM = {11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21}
CHAIN = [(11, 13), (13, 15), (12, 14), (14, 16)]


def fitT(src, tgt, scale, rot=True):
    ms, mt = src.mean(0), tgt.mean(0)
    a, b = src - ms, tgt - mt
    if rot:
        U, S, Vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(Vt.T @ U.T)); D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T; sv = (S * np.diag(D)).sum()
    else:
        R = np.eye(3); sv = np.trace(b.T @ a)
    s = sv / (a ** 2).sum() if scale else 1.0
    return s * (src - ms) @ R.T + mt


def swing(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    ax = np.cross(u, v); sa = np.linalg.norm(ax); ca = u @ v
    if sa < 1e-12:
        return np.eye(3)
    return Rot.from_rotvec(ax / sa * np.arctan2(sa, ca)).as_matrix()


sk = SMPLXSkeleton(model_path="models")
rest = sk.rest_joints
p0 = rest[0]
REST_LEN = {(11, 13): np.linalg.norm(rest[18] - rest[16]), (13, 15): np.linalg.norm(rest[20] - rest[18]),
            (12, 14): np.linalg.norm(rest[19] - rest[17]), (14, 16): np.linalg.norm(rest[21] - rest[19])}
REST_W = np.linalg.norm(rest[16] - rest[17])


def recon(fr, ok, R, lens, width):
    sL = p0 + R @ (rest[16] - p0); sR = p0 + R @ (rest[17] - p0)
    if width is not None:
        mid = (sL + sR) / 2; k = width / REST_W
        sL = mid + k * (sL - mid); sR = mid + k * (sR - mid)
    pos = {11: sL, 12: sR}
    for a, b in CHAIN:
        if ok[a] and ok[b]:
            d = fr[b] - fr[a]; pos[b] = pos[a] + lens[(a, b)] * d / np.linalg.norm(d)
    return pos


def frame_err(fr, ok, R, lens, width, metric, W):
    pres = [m for m in MP if ok[m]]
    pos = recon(fr, ok, R, lens, width)
    src = np.array([pos[m] for m in pres]); tgt = np.array([fr[m] for m in pres])
    if metric == "sim":
        al = fitT(src, tgt, True)
    elif metric == "rigid":
        al = fitT(src, tgt, False)
    else:  # translation only
        al = fitT(src, tgt, False, rot=False)
    return np.linalg.norm(al - tgt, axis=1) / W


# ---------------- load ---------------
clips = []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f)
    pose = d["landmarks"].astype(np.float64)
    ok_all = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
    both = ok_all[:, 11] & ok_all[:, 12]
    W = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
    clen = {}
    for a, b in CHAIN:
        m = ok_all[:, a] & ok_all[:, b]
        clen[(a, b)] = float(np.median(np.linalg.norm(pose[m, b] - pose[m, a], axis=-1)))
    frames = []
    for fr, ok in zip(pose, ok_all):
        if sum(ok[m] for m in MP) < 3:
            continue
        frames.append((fr, ok))
    clips.append(dict(name=Path(f).stem[:5], W=W, clen=clen, frames=frames))

def analytic_root(fr, ok):
    return swing(rest[16] - rest[17], fr[11] - fr[12])


def opt_root(fr, ok, lens, width, metric, W):
    best = None
    for init in [np.zeros(3), Rot.from_matrix(analytic_root(fr, ok)).as_rotvec()]:
        fun = lambda rv: (frame_err(fr, ok, Rot.from_rotvec(rv).as_matrix(), lens, width, metric, W) ** 2).sum()
        r = minimize(fun, init, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return Rot.from_rotvec(best.x).as_matrix()


