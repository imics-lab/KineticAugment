"""Attempt to reconstruct the doc's Step 5/6 oracle decomposition (no script for it exists in the repo).

Doc table (aligned = per-frame similarity, pooled median % shoulder width):
  baseline 5.94 | root only 5.13 (13.7%) | bone length only 3.86 (35.0%) | both 2.64 (55.5%)

Variants tried:
  root oracle   : per-frame 3-dof root rotvec chosen to minimise the scored residual, IK re-solved
                  relative to that root (solve_swing_ik(global_orient=...)); objective = SSE after
                  alignment. Inits: identity (+ a few perturbations) -- also a flip init for reference.
  length oracle : 'frame' = this frame's observed arm-bone lengths; 'clip' = per-clip median observed
                  lengths; optionally also oracle shoulder width ('+sw').
Each configuration is scored under both metrics: 'sim' (similarity, the 5.94 lineage) and 'rigid'
(the converter's own direct metric, 6.55 lineage).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from common import *  # noqa

base = SMPLXSkeleton(model_path="models")
clips = load_clips()
ROOT_INITS = [np.zeros(3)] + [np.array(v) * np.radians(15) for v in
                               ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]


def score(frame, ok, skel, root, mode, present, width):
    pts, _ = ik_points(frame, ok, skel, root)
    src = np.array([pts[m] for m in present])
    tgt = np.array([frame[m] for m in present])
    al, (s, R, t) = align(src, tgt, mode)
    return np.linalg.norm(al - tgt, axis=1) / width, s


def best_root(frame, ok, skel, mode, present, width, inits):
    def f(r):
        e, _ = score(frame, ok, skel, r, mode, present, width)
        return float((e ** 2).sum())
    best = None
    for r0 in inits:
        r = minimize(f, r0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def run(mode, root_oracle, length_oracle, shoulder_oracle=False, inits=ROOT_INITS):
    errs, per_clip, scales, roots = [], {}, [], []
    for name, pose, oks, width in clips:
        # per-clip median observed lengths
        clip_len = {}
        for (a, b), child in BONE_CHILD.items():
            v = oks[:, a] & oks[:, b]
            clip_len[child] = float(np.median(np.linalg.norm(pose[v, b] - pose[v, a], axis=-1)))
        both = oks[:, 11] & oks[:, 12]
        clip_sw = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
        pc = []
        for frame, ok in zip(pose, oks):
            present = [m for m in MP if ok[m]]
            if len(present) < 3:
                continue
            skel = EditableSkeleton(base)
            if length_oracle == "frame":
                skel.set_lengths(obs_lengths(frame, ok))
            elif length_oracle == "clip":
                skel.set_lengths(clip_len)
            if shoulder_oracle == "frame" and ok[11] and ok[12]:
                skel.set_shoulder_width(np.linalg.norm(frame[11] - frame[12]))
            elif shoulder_oracle == "clip":
                skel.set_shoulder_width(clip_sw)
            root = best_root(frame, ok, skel, mode, present, width, inits) if root_oracle else None
            e, s = score(frame, ok, skel, root, mode, present, width)
            errs += list(e); pc += list(e); scales.append(s)
            if root is not None:
                roots.append(np.degrees(np.linalg.norm(root)))
        per_clip[name] = pct(pc)
    return pct(errs), per_clip, np.array(scales), np.array(roots)


if __name__ == "__main__":
    configs = []
    for mode in ("sim", "rigid"):
        configs += [
            (mode, False, None, False),
            (mode, True, None, False),
            (mode, False, "frame", False),
            (mode, False, "clip", False),
            (mode, True, "frame", False),
            (mode, True, "clip", False),
            (mode, False, "frame", "frame"),
            (mode, True, "frame", "frame"),
        ]
    results = {}
    for cfg in configs:
        t0 = time.time()
        med, pc, sc, roots = run(*cfg)
        results[cfg] = med
        mode, ro, lo, so = cfg
        tag = f"{mode:<5} root={'oracle' if ro else '0':<6} len={str(lo):<5} shoulderW={str(so):<5}"
        extra = f" | scale med {np.median(sc):.4f}" if mode == "sim" else ""
        if len(roots):
            extra += f" | |root| med {np.median(roots):.2f} deg p95 {np.percentile(roots,95):.2f}"
        print(f"{tag}  pooled {med:6.3f}%  per-clip {[round(v,2) for v in pc.values()]}{extra}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(Path(__file__).parent / "oracle_results.npy", {str(k): v for k, v in results.items()}, allow_pickle=True)
