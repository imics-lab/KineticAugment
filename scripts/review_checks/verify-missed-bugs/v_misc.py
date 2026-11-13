"""A001, N5 (betas), N6 (origin), N8 (collinear), N7 (elbow jumps/fabrication), N9 (antipode)."""
import glob, numpy as np, torch
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, _swing_rotation
sk = SMPLXSkeleton(model_path="models")
t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
def smplx_joints(model, body_pose, go, transl, betas):
    with torch.no_grad():
        return model(body_pose=t(body_pose), global_orient=t(go), transl=t(transl), betas=t(betas), return_verts=False).joints[0].numpy().astype(float)
rng = np.random.default_rng(11)
# ---- A001: FK vs real SMPL-X, full random body pose, root, transl, shape
worst = 0
for i in range(20):
    b = rng.normal(0, 1.5, 10); shaped = SMPLXSkeleton(model_path="models", betas=b)
    rot = np.zeros((55, 3)); rot[:22] = rng.normal(0, 0.5, (22, 3)); tr = rng.normal(0, 1, 3)
    ours = forward_kinematics(rot, shaped) + tr
    J = smplx_joints(sk.model, rot[1:22].reshape(63), rot[0], tr, b)[:55]
    worst = max(worst, np.abs(ours - J).max())
print(f"A001: FK(shaped skeleton) vs SMPL-X forward, 20 random full-body poses w/ root+transl+betas: max |diff| {worst:.2e} m")
# converter output vs SMPL-X forward on real frames
worst = 0; n = 0
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); P = d["landmarks"].astype(float); OK = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    for fr, ok in list(zip(P, OK))[::7]:
        fit = fit_body(fr, ok, sk); out = reconstruct_body(fit, sk)
        J = smplx_joints(sk.model, fit.body_pose, fit.global_orient, fit.transl, fit.betas)
        worst = max(worst, max(np.abs(out[m] - J[s]).max() for s, m in ARM_CORRESPONDENCE)); n += 1
print(f"A001: reconstruct_body vs SMPL-X forward on fit params, {n} real frames: max |diff| {worst:.2e} m")
# ---- N5 betas
b = np.array([2.0, -1.5, 0.5, 0, 0, 0, 0, 0, 0, 0]); shaped = SMPLXSkeleton(model_path="models", betas=b)
rot = np.zeros((55, 3))
for j, *_ in SMPLX_ARM_BONES: rot[j] = rng.normal(0, 0.4, 3)
Jt = forward_kinematics(rot, shaped); L = np.full((33, 3), np.nan)
for s, m in ARM_CORRESPONDENCE: L[m] = Jt[s]
v = np.isfinite(L).all(1); fit = fit_body(L, v, shaped); out = reconstruct_body(fit, shaped)
J0 = smplx_joints(sk.model, fit.body_pose, fit.global_orient, fit.transl, fit.betas)
Jb = smplx_joints(sk.model, fit.body_pose, fit.global_orient, fit.transl, b)
print(f"N5: fit.betas returned = {fit.betas[:3]}...; SMPL-X(returned params) vs reconstruct_body: max {1000*max(np.linalg.norm(out[m]-J0[s]) for s,m in ARM_CORRESPONDENCE):.2f} mm; with true betas: {1000*max(np.linalg.norm(out[m]-Jb[s]) for s,m in ARM_CORRESPONDENCE):.4f} mm")
fit.betas = np.array([3.0, -2, 0, 0, 0, 0, 0, 0, 0, 0]); out2 = reconstruct_body(fit, shaped)
print(f"N5: changing fit.betas changes reconstruct_body by {np.nanmax(np.abs(out2-out)):.1e} m")
# ---- N6 origin
rot = np.zeros((55, 3))
for j, *_ in SMPLX_ARM_BONES: rot[j] = rng.normal(0, 0.4, 3)
J = forward_kinematics(rot, sk)
for eps in (0.0, 1e-9):
    L = np.full((33, 3), np.nan)
    for s, m in ARM_CORRESPONDENCE: L[m] = J[s] - J[16] + eps
    out = reconstruct_body(fit_body(L, np.isfinite(L).all(1), sk), sk)
    print(f"N6: left shoulder at offset {eps}: per-landmark error mm", np.round([1000*np.linalg.norm(out[m]-L[m]) for _, m in ARM_CORRESPONDENCE], 2))
# ---- N8 collinear: only left arm valid, exactly straight
for dvec in ([0.3, -0.9, 0.2], [1, 0, 0], [0, -1, 0]):
    dvec = np.array(dvec) / np.linalg.norm(dvec)
    L = np.full((33, 3), np.nan); L[11] = [0.15, -0.45, 0.0]; L[13] = L[11] + 0.27 * dvec; L[15] = L[13] + 0.25 * dvec
    v = np.isfinite(L).all(1)
    gos = []
    for noise in (0.0, 1e-6, 1e-4):
        Ln = L.copy(); Ln[[11, 13, 15]] += np.random.default_rng(1).normal(0, noise, (3, 3)) if noise else 0
        fit = fit_body(Ln, v, sk); out = reconstruct_body(fit, sk)
        gos.append(f"noise {noise:g}: |go| {np.degrees(np.linalg.norm(fit.global_orient)):.1f} deg, err {1000*max(np.linalg.norm(out[m]-Ln[m]) for m in (11,13,15)):.2f} mm")
    print("N8 dir", dvec.round(2), " | ".join(gos))
# ---- N7 elbow jumps on wrist-validity toggles; fabricated wrists
tog, both = [], []
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); P = d["landmarks"].astype(float); OK = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    fits = [fit_body(fr, ok, sk) for fr, ok in zip(P, OK)]
    for i in range(1, len(fits)):
        for elbow_bp, wrist in ((17, 15), (18, 16)):  # body_pose joint index = joint-1 (18->17, 19->18)
            a = R.from_rotvec(fits[i-1].body_pose.reshape(21, 3)[elbow_bp]); c = R.from_rotvec(fits[i].body_pose.reshape(21, 3)[elbow_bp])
            jump = (a.inv() * c).magnitude()
            if OK[i-1, wrist] != OK[i, wrist]: tog.append(jump)
            elif OK[i-1, wrist] and OK[i, wrist]: both.append(jump)
tog, both = np.array(tog), np.array(both)
print(f"N7: elbow rotation jump on wrist-validity toggle: n={len(tog)}, median {np.median(tog):.2f} rad, max {tog.max():.2f}; both-observed: n={len(both)}, median {np.median(both):.3f}, p99 {np.percentile(both,99):.2f}, max {both.max():.2f}; jumps>1rad: toggles {np.sum(tog>1)}, non-toggle {np.sum(both>1)}")
# ---- N9 antipode
a = np.array([1.0, 0, 0]); prev = None; maxjump = 0
for th in np.linspace(np.pi - 1e-3, np.pi + 1e-3, 11):
    b = np.array([np.cos(th), np.sin(th) * 0.0 + np.sin(th), 1e-4]); b /= np.linalg.norm(b)
    rv = R.from_matrix(_swing_rotation(a, b)).as_rotvec()
    if prev is not None: maxjump = max(maxjump, np.linalg.norm(rv - prev))
    prev = rv
print(f"N9: swing rotvec max step across antipode sweep (0.2 mrad steps): {maxjump:.2f} rad")
