"""Step 8 prototype on the real clips: error (direct + similarity-aligned) and torso plausibility,
for the shipped converter and for prototype roots (nose apex / hip-centre apex / shoulder-only)."""
import glob, sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0, sys.argv[1])
import conftest  # noqa: F401  (adds rest_nose to SMPLXSkeleton)
import step8_proto as P8
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE as CORR
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity

sk = SMPLXSkeleton(model_path="models")
VIS = 0.5
MODEL = {
    "shipped converter (zero-root IK + rigid)": lambda L, ok: fit_body(L, ok, sk),
    "proto: nose apex + IK + translation": lambda L, ok: P8.fit_body(L, ok, sk, apex="nose"),
    "proto: hip-centre apex + IK + translation": lambda L, ok: P8.fit_body(L, ok, sk, apex="hip"),
    "proto: shoulder-only (no apex) + IK + transl": lambda L, ok: P8.fit_body(L, ok, sk, apex=None),
    "proto: nose apex + IK + rigid post-fit": lambda L, ok: P8.fit_body(L, ok, sk, apex="nose", post="rigid"),
}
res = {k: dict(direct=[], aligned=[], head_cos=[], torso_cos=[], up_ok=[], roots=[], clip_direct={}) for k in MODEL}
for path in sorted(glob.glob("data/test_output/*_world.npz")):
    d = np.load(path); Pm = d["landmarks"].astype(np.float64); V = d["visibility"]
    okall = np.isfinite(Pm).all(-1) & (V >= VIS)
    clip = Path(path).stem.replace("_world", "")
    both = okall[:, 11] & okall[:, 12]
    width = float(np.median(np.linalg.norm(Pm[both, 11] - Pm[both, 12], axis=-1)))
    for k in MODEL: res[k]["clip_direct"][clip] = []; res[k]["roots"].append(None)
    for L, ok in zip(Pm, okall):
        for k, f in MODEL.items():
            r = res[k]
            try:
                fit = f(L, ok)
            except ValueError:
                r["roots"].append(None); continue
            out = reconstruct_body(fit, sk)
            ms = [m for _, m in CORR if ok[m]]
            e = [np.linalg.norm(out[m] - L[m]) / width for m in ms]
            r["direct"] += e; r["clip_direct"][clip] += e
            src = np.array([out[m] for m in ms]); tgt = np.array([L[m] for m in ms])
            s, R, t = similarity_transform(src, tgt)
            r["aligned"] += list(np.linalg.norm(apply_similarity(src, s, R, t) - tgt, axis=1) / width)
            rots = np.zeros((55, 3)); rots[0] = fit.global_orient; rots[1:22] = fit.body_pose.reshape(21, 3)
            J = forward_kinematics(rots, sk) + fit.transl
            sh = (J[16] + J[17]) / 2
            if ok[0]:
                a = J[15] - sh; b = L[0] - (L[11] + L[12]) / 2
                r["head_cos"].append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            a = J[12] - J[0]; b = (L[11] + L[12]) / 2 - 0.0     # fitted pelvis->neck vs MP hip-centre->shoulder-mid
            r["torso_cos"].append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            r["up_ok"].append(J[15, 1] < sh[1])     # MP y is DOWN: head above shoulders means smaller y
            r["roots"].append(Rotation.from_rotvec(fit.global_orient))

print(f"{'method':46s}{'direct med':>11s}{'p95':>7s}{'aligned med':>12s}{'n':>6s} | head-up% torso-cos med  head-vs-nose cos med | root jitter med/p95/max deg")
for k, r in res.items():
    rr = r["roots"]; jit = [np.degrees((a.inv() * b).magnitude()) for a, b in zip(rr[:-1], rr[1:]) if a is not None and b is not None]
    print(f"{k:46s}{np.median(r['direct'])*100:10.2f}%{np.percentile(r['direct'],95)*100:6.2f}%{np.median(r['aligned'])*100:11.2f}%{len(r['direct']):6d} |"
          f" {np.mean(r['up_ok'])*100:6.1f}%  {np.median(r['torso_cos']):+.3f}        {np.median(r['head_cos']):+.3f}            |"
          f" {np.median(jit):.2f}/{np.percentile(jit,95):.2f}/{np.max(jit):.2f}")
print("\nper-clip direct median (% shoulder width)")
print(f"{'clip':>8s}" + "".join(f"{i:>9d}" for i in range(len(MODEL))))
for c in res[next(iter(res))]["clip_direct"]:
    print(f"{c:>8s}" + "".join(f"{np.median(res[k]['clip_direct'][c])*100:9.2f}" for k in MODEL))
print("  columns:", {i: k for i, k in enumerate(MODEL)})
a = np.array(res["proto: nose apex + IK + translation"]["direct"]); b = np.array(res["proto: hip-centre apex + IK + translation"]["direct"])
c = np.array(res["proto: shoulder-only (no apex) + IK + transl"]["direct"])
print(f"\ngauge check: max |nose - hip| per-landmark {np.abs(a-b).max():.2e}, max |nose - shoulder-only| {np.abs(a-c).max():.2e} (fraction of width)")
rn = res["proto: nose apex + IK + translation"]["roots"]; rh = res["proto: hip-centre apex + IK + translation"]["roots"]
dd = [np.degrees((x.inv() * y).magnitude()) for x, y in zip(rn, rh) if x is not None and y is not None]
print(f"nose-root vs hip-root torso orientation: median {np.median(dd):.2f}, p95 {np.percentile(dd,95):.2f}, max {np.max(dd):.2f} deg")
