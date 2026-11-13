"""Reproduce headline numbers and examine frames / scale / directions / lengths."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.spatial.transform import Rotation

from common import *  # noqa

sk = SMPLXSkeleton(model_path="models")
clips = load_clips()

res = {k: [] for k in ["old_sim", "direct", "direct_then_rigid", "direct_then_sim", "direct_then_scaleonly", "ikframe_none"]}
perclip = {k: {} for k in ["old_sim", "direct"]}
scales = []
dir_err = {k: {n: [] for *_, n in BONES} for k in ["ikframe", "direct", "aligned_sim"]}
len_err = {k: {n: [] for *_, n in BONES} for k in ["direct", "aligned_sim"]}
dir_pos_equiv = {n: [] for *_, n in BONES}  # position error a direction error alone would cause at bone tip
root_angle_from_flip = []
rigid_angle = []
skipped = 0
for name, pose, oks, width in clips:
    pc = {"old_sim": [], "direct": []}
    for frame, ok in zip(pose, oks):
        present = [m for m in MP if ok[m]]
        if len(present) < 3:
            skipped += 1
            continue
        tgt = np.array([frame[m] for m in present])
        # old harness: IK (root=0) -> FK -> per-frame similarity
        pts, rots = ik_points(frame, ok, sk)
        src = np.array([pts[m] for m in present])
        al, (s, R, t) = align(src, tgt, "sim")
        scales.append(s)
        e = np.linalg.norm(al - tgt, axis=1) / width
        res["old_sim"] += list(e); pc["old_sim"] += list(e)
        res["ikframe_none"] += list(np.linalg.norm(src - tgt, axis=1) / width)
        # new converter direct
        fit = fit_body(frame, ok, sk)
        out = reconstruct_body(fit, sk)
        dsrc = np.array([out[m] for m in present])
        e = np.linalg.norm(dsrc - tgt, axis=1) / width
        res["direct"] += list(e); pc["direct"] += list(e)
        for mode, key in (("rigid", "direct_then_rigid"), ("sim", "direct_then_sim")):
            a2, _ = align(dsrc, tgt, mode)
            res[key] += list(np.linalg.norm(a2 - tgt, axis=1) / width)
        # scale only about centroid (no extra rotation), LS-optimal scale
        c_s, c_t = dsrc.mean(0), tgt.mean(0)
        A, B = dsrc - c_s, tgt - c_t
        k = float((A * B).sum() / (A * A).sum())
        a3 = c_t + k * A
        res["direct_then_scaleonly"] += list(np.linalg.norm(a3 - tgt, axis=1) / width)
        # rigid rotation of the converter relative to the IK frame
        _, Rr, _ = similarity_transform(src, tgt, allow_scale=False)
        rigid_angle.append(np.degrees(Rotation.from_matrix(Rr).magnitude()))
        # directions and lengths
        al_map = {m: al[i] for i, m in enumerate(present)}
        for a, b, n in BONES:
            if ok[a] and ok[b]:
                obs = frame[b] - frame[a]
                dir_err["ikframe"][n].append(angle_deg(pts[b] - pts[a], obs))
                dir_err["direct"][n].append(angle_deg(out[b] - out[a], obs))
                dir_err["aligned_sim"][n].append(angle_deg(al_map[b] - al_map[a], obs))
                Lm = np.linalg.norm(out[b] - out[a])
                len_err["direct"][n].append(abs(Lm - np.linalg.norm(obs)) / width)
                len_err["aligned_sim"][n].append(abs(np.linalg.norm(al_map[b] - al_map[a]) - np.linalg.norm(obs)) / width)
                th = np.radians(dir_err["aligned_sim"][n][-1])
                dir_pos_equiv[n].append(np.linalg.norm(obs) * 2 * np.sin(th / 2) / width)
    for k in pc:
        perclip[k][name] = pct(pc[k])

print(f"skipped frames: {skipped}")
for k, v in res.items():
    print(f"{k:<24} pooled median {pct(v):6.3f}%   p95 {pct(v,95):6.3f}%   n={len(v)}")
print("per-clip medians:")
for k in perclip:
    print("  ", k, {c: round(v, 2) for c, v in perclip[k].items()})
sc = np.array(scales)
print(f"fitted similarity scale: median {np.median(sc):.4f}  p5 {np.percentile(sc,5):.4f}  p95 {np.percentile(sc,95):.4f}  mean {sc.mean():.4f}")
print(f"converter rigid rotation magnitude (deg, vs IK frame): median {np.median(rigid_angle):.2f}  p5 {np.percentile(rigid_angle,5):.2f} p95 {np.percentile(rigid_angle,95):.2f}")
print("\nDirection error per bone (median deg / p95 deg):")
for k in dir_err:
    print(f"  {k:<12}", "  ".join(f"{n}: {np.median(v):.3f}/{np.percentile(v,95):.2f}" for n, v in dir_err[k].items()))
allk = {k: np.concatenate([v for v in dir_err[k].values()]) for k in dir_err}
print("  pooled median:", {k: round(float(np.median(v)), 3) for k, v in allk.items()})
print("\nLength error per bone (median % shoulder width):")
for k in len_err:
    print(f"  {k:<12}", "  ".join(f"{n}: {pct(v):.2f}" for n, v in len_err[k].items()),
          f"  POOLED {pct(np.concatenate(list(len_err[k].values()))):.2f}")
print("\nPosition displacement at bone tip implied by aligned direction error alone (median % width):")
print("  ", "  ".join(f"{n}: {pct(v):.2f}" for n, v in dir_pos_equiv.items()),
      f"  POOLED {pct(np.concatenate(list(dir_pos_equiv.values()))):.2f}")
