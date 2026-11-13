"""Real-clip root analysis: what the current converter's root is, how much the nose
moves in the shoulder frame, and what shoulder / shoulder+nose / shoulder+hip roots do."""
import glob
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik, _swing_rotation
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE as CORR

sk = SMPLXSkeleton(model_path="models")
rest = sk.rest_joints
with torch.no_grad():
    rest127 = sk.model(return_verts=False).joints[0].numpy().astype(np.float64)
REST_NOSE = rest127[55]
print("rest nose (SMPL-X 127-joint #55):", REST_NOSE, " rest head(15):", rest[15], " rest[0]-vs-127[0] diff", np.abs(rest127[0]-rest[0]).max())
VIS = 0.5
NAMES = {11: "Lsh", 12: "Rsh", 13: "Lel", 14: "Rel", 15: "Lwr", 16: "Rwr"}

def frame_from(lsh, rsh, third):
    """Right-handed frame: x along L-R shoulder line, y = third-point direction orthogonalised."""
    x = lsh - rsh; x = x / np.linalg.norm(x)
    y = third - (lsh + rsh) / 2; y = y - (y @ x) * x; y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)

F_REST_NOSE = frame_from(rest[16], rest[17], REST_NOSE)
F_REST_HIP = frame_from(rest[16], rest[17], rest[0])
FLIP = Rotation.from_euler("x", 180, degrees=True).as_matrix()

def root_shoulder_only(L):
    """Gauge: start from the MP<-SMPL-X convention flip, then minimal rotation aligning the shoulder line."""
    a = FLIP @ (rest[16] - rest[17]); a /= np.linalg.norm(a)
    b = L[11] - L[12]; b /= np.linalg.norm(b)
    return _swing_rotation(a, b) @ FLIP

def root_nose(L): return frame_from(L[11], L[12], L[0]) @ F_REST_NOSE.T
def root_hip(L):  return frame_from(L[11], L[12], np.zeros(3)) @ F_REST_HIP.T   # MP world origin = hip centre

def fit_with_root(L, ok, R, post="translation"):
    masked = np.where(ok[:, None], L, np.nan)
    rot = solve_swing_ik(masked, sk, SMPLX_ARM_BONES, global_orient=Rotation.from_matrix(R).as_rotvec())
    J = forward_kinematics(rot, sk)
    pairs = [(s, m) for s, m in CORR if ok[m]]
    src = np.array([J[s] for s, _ in pairs]); tgt = np.array([L[m] for _, m in pairs])
    if post == "translation":
        out = src + (tgt - src).mean(0)
    return {m: o for (_, m), o in zip(pairs, out)}, rot

def opt_root(L, ok, R0):
    """Oracle-ish: choose root (3 dof) to minimise 6-landmark error, IK re-solved, translation closed form."""
    def cost(rv):
        o, _ = fit_with_root(L, ok, Rotation.from_rotvec(rv).as_matrix())
        return sum(np.sum((o[m] - L[m]) ** 2) for m in o)
    r = minimize(cost, Rotation.from_matrix(R0).as_rotvec(), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-12, maxiter=4000))
    return Rotation.from_rotvec(r.x).as_matrix()

def angle(Ra, Rb): return np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude())

res = {k: [] for k in ("conv", "sh", "nose", "hip", "opt")}
per_clip = {}
conv_go_from_I, conv_go_from_flip, head_dir_dot = [], [], []
nose_theta_all, hip_theta_all, nose_vs_hip, jit_nose, jit_hip, sh_vs_opt = [], [], [], [], [], []
pitch_shoulder_rot_shift = []
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); P = d["landmarks"].astype(np.float64); V = d["visibility"]
    okall = np.isfinite(P).all(-1) & (V >= VIS)
    clip = Path(path).stem.replace("_world", "")
    both = okall[:, 11] & okall[:, 12]
    width = float(np.median(np.linalg.norm(P[both, 11] - P[both, 12], axis=-1)))
    cr = {k: [] for k in res}; th_n, th_h = [], []; prevRn = prevRh = None
    for f, (L, ok) in enumerate(zip(P, okall)):
        if not (ok[11] and ok[12] and ok[0]) or sum(ok[m] for m in NAMES) < 3:
            prevRn = prevRh = None; continue
        # current converter
        fit = fit_body(L, ok, sk); out = reconstruct_body(fit, sk)
        Rg = Rotation.from_rotvec(fit.global_orient).as_matrix()
        conv_go_from_I.append(angle(np.eye(3), Rg)); conv_go_from_flip.append(angle(FLIP, Rg))
        # where does the fitted SMPL-X body's head point, relative to the observed nose direction?
        rots = np.zeros((55, 3)); rots[0] = fit.global_orient; rots[1:22] = fit.body_pose.reshape(21, 3)
        J = forward_kinematics(rots, sk) + fit.transl
        up_fit = J[15] - (J[16] + J[17]) / 2; up_obs = L[0] - (L[11] + L[12]) / 2
        head_dir_dot.append(up_fit @ up_obs / np.linalg.norm(up_fit) / np.linalg.norm(up_obs))
        for m in NAMES:
            if ok[m]: cr["conv"].append(np.linalg.norm(out[m] - L[m]) / width)
        # estimators
        Rs, Rn, Rh = root_shoulder_only(L), root_nose(L), root_hip(L)
        for key, R in (("sh", Rs), ("nose", Rn), ("hip", Rh)):
            o, _ = fit_with_root(L, ok, R)
            for m in o: cr[key].append(np.linalg.norm(o[m] - L[m]) / width)
        if f % 4 == 0:   # oracle on a subsample (slow)
            Ro = opt_root(L, ok, Rh); o, _ = fit_with_root(L, ok, Ro)
            for m in o: cr["opt"].append(np.linalg.norm(o[m] - L[m]) / width)
        # nose / hip direction in the shoulder frame: angle about the shoulder axis
        x = L[11] - L[12]; x /= np.linalg.norm(x)
        def theta(v):
            v = v - (v @ x) * x
            # reference axes in plane perpendicular to x: e1 = projected camera-up (-y), e2 = x cross e1
            e1 = np.array([0, -1.0, 0]); e1 = e1 - (e1 @ x) * x; e1 /= np.linalg.norm(e1); e2 = np.cross(x, e1)
            return np.degrees(np.arctan2(v @ e2, v @ e1))
        mid = (L[11] + L[12]) / 2
        th_n.append(theta(L[0] - mid)); th_h.append(theta(mid - 0.0))
        nose_vs_hip.append(angle(Rn, Rh))
        if prevRn is not None: jit_nose.append(angle(prevRn, Rn)); jit_hip.append(angle(prevRh, Rh))
        prevRn, prevRh = Rn, Rh
    th_n, th_h = np.array(th_n), np.array(th_h)
    per_clip[clip] = dict(n=len(th_n), width=width,
        nose_std=th_n.std(), nose_rng=np.ptp(th_n), nose_p=np.percentile(th_n - np.median(th_n), [5, 95]),
        hip_std=th_h.std(), hip_rng=np.ptp(th_h),
        rel_std=(th_n - th_h).std(), rel_rng=np.ptp(th_n - th_h),
        **{k: np.median(v) * 100 for k, v in cr.items()})
    for k in res: res[k] += cr[k]

print("\n== Current converter (zero-root IK + rigid fit) on real frames ==")
print(f"  |global_orient| vs identity: median {np.median(conv_go_from_I):.1f} deg; vs Rx(180) convention flip: median {np.median(conv_go_from_flip):.1f} deg, min {np.min(conv_go_from_flip):.1f}")
hd = np.array(head_dir_dot)
print(f"  cos(fitted SMPL-X shoulder->head, observed shoulder->nose): median {np.median(hd):.3f}; frames with cos<0 (fitted head points AWAY from real head): {np.mean(hd<0)*100:.1f}%")

print("\n== Nose and hip-centre direction about the shoulder axis (deg) ==")
print(f"  {'clip':6s}{'n':>4s}{'nose std':>9s}{'nose rng':>9s}{'nose p5..p95':>16s}{'hip std':>8s}{'hip rng':>8s}{'nose-hip std':>13s}{'nose-hip rng':>13s}")
for c, r in per_clip.items():
    print(f"  {c:6s}{r['n']:4d}{r['nose_std']:9.2f}{r['nose_rng']:9.2f}   {r['nose_p'][0]:6.2f}..{r['nose_p'][1]:5.2f}{r['hip_std']:8.2f}{r['hip_rng']:8.2f}{r['rel_std']:13.2f}{r['rel_rng']:13.2f}")
print(f"  angle between nose-based and hip-based root, all frames: median {np.median(nose_vs_hip):.2f} deg, p95 {np.percentile(nose_vs_hip,95):.2f}, max {np.max(nose_vs_hip):.2f}")
print(f"  frame-to-frame root change: nose-based median {np.median(jit_nose):.3f} p95 {np.percentile(jit_nose,95):.3f} max {np.max(jit_nose):.2f} deg; hip-based median {np.median(jit_hip):.3f} p95 {np.percentile(jit_hip,95):.3f} max {np.max(jit_hip):.2f} deg")

print("\n== Six-landmark direct error, median % shoulder width (pooled, then per clip) ==")
lab = dict(conv="current converter (zero-root IK + rigid)", sh="shoulder-only root + IK + translation",
           nose="shoulders+nose root + IK + translation", hip="shoulders+hip-centre root + IK + translation",
           opt="3-dof optimised root + IK + translation (every 4th frame)")
for k in res:
    print(f"  {lab[k]:58s} median {np.median(res[k])*100:5.2f}%  p95 {np.percentile(res[k],95)*100:6.2f}%  n={len(res[k])}")
print(f"  {'clip':6s}" + "".join(f"{k:>8s}" for k in res))
for c, r in per_clip.items():
    print(f"  {c:6s}" + "".join(f"{r[k]:8.2f}" for k in res))
d = np.abs(np.array(res['sh']) - np.array(res['nose'])).max(); d2 = np.abs(np.array(res['sh']) - np.array(res['hip'])).max()
print(f"  max |sh - nose| per-landmark difference: {d*100:.2e} %;  max |sh - hip|: {d2*100:.2e} %   (gauge check)")
