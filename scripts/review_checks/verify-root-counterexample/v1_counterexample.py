"""Independent re-check of the root counterexample.

Truth comes from the real smplx forward pass (not the repo FK). Alignment uses my
own Umeyama implementation. Fit uses the repo's solve_swing_ik (the thing under test)
and the shipped converter fit_body/reconstruct_body.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik
from kinetic_augment.body_model.converter import fit_body, reconstruct_body

CORR = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]
SK = SMPLXSkeleton(model_path="models")


def smplx_joints(root, body_pose63):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)[None]
    m = SK.model.double()
    with torch.no_grad():
        out = m(global_orient=t(root), body_pose=t(body_pose63),
                betas=torch.zeros(1, 10, dtype=torch.float64), return_verts=False)
    return out.joints[0].numpy()


def umeyama(src, tgt, scale=True):
    ms, mt = src.mean(0), tgt.mean(0)
    a, b = src - ms, tgt - mt
    C = b.T @ a / len(src)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean() if scale else 1.0
    return s * src @ Rm.T + (mt - s * Rm @ ms)


def case(root_rv, arm=None, label=""):
    bp = np.zeros(63)
    if arm is not None:
        bp[:] = arm
    J = smplx_joints(root_rv, bp)
    lm = np.full((33, 3), np.nan)
    for s, m in CORR:
        lm[m] = J[s]
    valid = np.isfinite(lm).all(1)
    # zero-root IK (review-time harness)
    fitted = solve_swing_ik(np.where(valid[:, None], lm, 0.0) + 0.0, SK, SMPLX_ARM_BONES)
    fj = forward_kinematics(fitted, SK)
    src = np.array([fj[s] for s, _ in CORR]); tgt = np.array([lm[m] for _, m in CORR])
    raw = np.linalg.norm(src - tgt, axis=1)
    sim = np.linalg.norm(umeyama(src, tgt, True) - tgt, axis=1)
    rig = np.linalg.norm(umeyama(src, tgt, False) - tgt, axis=1)
    # shipped converter
    out = reconstruct_body(fit_body(lm, valid, SK), SK)
    conv = np.linalg.norm(np.array([out[m] for _, m in CORR]) - tgt, axis=1)
    # direction errors of 4 arm segments + shoulder line, on zero-root fit
    def ang(u, v):
        return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    segs = [(16, 18, 11, 13), (18, 20, 13, 15), (17, 19, 12, 14), (19, 21, 14, 16)]
    dir_err = [ang(fj[c] - fj[p], lm[mc] - lm[mp]) for p, c, mp, mc in segs]
    sh_err = ang(fj[16] - fj[17], lm[11] - lm[12])
    lens_t = [np.linalg.norm(J[c] - J[p]) for p, c, _, _ in segs] + [np.linalg.norm(J[16] - J[17])]
    lens_f = [np.linalg.norm(fj[c] - fj[p]) for p, c, _, _ in segs] + [np.linalg.norm(fj[16] - fj[17])]
    print(f"{label:34s} raw max/mean {raw.max()*1e3:7.2f}/{raw.mean()*1e3:7.2f} mm | "
          f"sim {sim.max()*1e3:6.2f}/{sim.mean()*1e3:6.2f} | rigid {rig.max()*1e3:6.2f}/{rig.mean()*1e3:6.2f} | "
          f"conv {conv.max()*1e3:6.2f}/{conv.mean()*1e3:6.2f} | dir max {max(dir_err):.2e} deg | "
          f"shoulder-line {sh_err:6.2f} deg | len diff {np.max(np.abs(np.array(lens_t)-lens_f)):.1e}")
    return raw, sim, rig, conv


print("SMPL-X forward truth vs repo FK at rest (max diff over 55 joints):",
      np.abs(smplx_joints(np.zeros(3), np.zeros(63))[:55] - SK.rest_joints).max())
print("\n== the reviewer's exact case: root rotvec [0,0,0.7], T-pose arms ==")
case([0, 0, 0.7], label="z 0.7rad T-pose")
print("\n== axis sensitivity (T-pose) ==")
for ax in "xyz":
    for deg in (10, 20, 30, 40, 45):
        rv = R.from_euler(ax, deg, degrees=True).as_rotvec()
        case(rv, label=f"{ax} {deg}deg T-pose")
print("\n== random arm poses N(0,0.4), random axis, per angle: pooled stats over 200 trials ==")
rng = np.random.default_rng(123)
for deg in (10, 20, 30, 45):
    allr, alls, allc = [], [], []
    for _ in range(200):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        bp = np.zeros(63)
        for j in (16, 17, 18, 19):
            bp[(j - 1) * 3:(j - 1) * 3 + 3] = rng.normal(0, 0.4, 3)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            r, s, g, c = case(ax * np.deg2rad(deg), bp)
        allr.append(r); alls.append(s); allc.append(c)
    allr, alls, allc = map(np.concatenate, (allr, alls, allc))
    print(f"{deg:2d} deg: raw mean/max {allr.mean()*1e3:6.2f}/{allr.max()*1e3:6.1f} | "
          f"sim mean/max {alls.mean()*1e3:6.2f}/{alls.max()*1e3:6.1f} | "
          f"converter mean/max {allc.mean()*1e3:6.2f}/{allc.max()*1e3:6.1f} mm")
