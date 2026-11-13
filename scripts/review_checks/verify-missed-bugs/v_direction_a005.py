"""R-direction and A005: bone-direction error of the DIRECT output; rigid re-alignment of direct output; scale of aligned fit."""
import glob, numpy as np
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity
from kinetic_augment.body_model.converter import fit_body, reconstruct_body
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models")
BONES = {"L upper": (11, 13), "L fore": (13, 15), "R upper": (12, 14), "R fore": (14, 16)}
ang = {k: [] for k in BONES}; ang_ik = {k: [] for k in BONES}; go = []; rigid_gain = []; scales = []
def angle(a, b): return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
for f in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(f); P = d["landmarks"].astype(float); OK = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    for fr, ok in zip(P, OK):
        try: fit = fit_body(fr, ok, sk)
        except ValueError: continue
        out = reconstruct_body(fit, sk)
        # IK-only (before Kabsch) posed directions
        rots = solve_swing_ik(np.where(ok[:, None], fr, np.nan), sk, SMPLX_ARM_BONES); J = forward_kinematics(rots, sk)
        smap = {11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21}
        for k, (a, b) in BONES.items():
            if ok[a] and ok[b]:
                ang[k].append(angle(out[b] - out[a], fr[b] - fr[a]))
                ang_ik[k].append(angle(J[smap[b]] - J[smap[a]], fr[b] - fr[a]))
        go.append(np.degrees(np.linalg.norm(fit.global_orient)))
        pres = [m for m in range(11, 17) if ok[m]]
        src = np.array([out[m] for m in pres]); tgt = np.array([fr[m] for m in pres])
        s, Rr, t = similarity_transform(src, tgt, allow_scale=False)
        rigid_gain.append(np.sum((src - tgt) ** 2) - np.sum((apply_similarity(src, s, Rr, t) - tgt) ** 2))
        scales.append(similarity_transform(src, tgt)[0])
for k in BONES:
    v = np.array(ang[k]); w = np.array(ang_ik[k])
    print(f"{k:<8} direct-output direction error: median {np.median(v):.2f} deg, p95 {np.percentile(v,95):.2f}, max {v.max():.2f} | before Kabsch: max {w.max():.2e} deg")
print(f"|global_orient| median {np.median(go):.2f} deg, p95 {np.percentile(go,95):.2f}")
print(f"rigid re-alignment of direct output: max SSE reduction {max(rigid_gain):.2e} m^2 (min {min(rigid_gain):.2e})")
print(f"free-similarity scale on direct output: median {np.median(scales):.4f}, p5 {np.percentile(scales,5):.4f}, p95 {np.percentile(scales,95):.4f}")
