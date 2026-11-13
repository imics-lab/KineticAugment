"""Oracle decomposition of the real-data round-trip error, re-derived from scratch.

No script for the doc's Step 5/6 decomposition (5.94 / 5.13 / 3.86 / 2.64) exists in the repo,
so this rebuilds it with a closed-form arm model that is verified against the project's
solve_swing_ik + forward_kinematics path first.

Closed form (exact because swing IK reproduces every observed bone direction, 0.000 deg):
    with spine and collars at zero and root rotation R about the pelvis rest position p0,
        S_k = p0 + R (rest[shoulder_k] - p0)
        E_k = S_k + L_upper_k * u_k          (u_k = observed upper-arm unit direction)
        W_k = E_k + L_fore_k  * f_k          (f_k = observed forearm unit direction)
    then a per-frame alignment ('sim' = Umeyama with scale, 'rigid' = no scale) to the observed
    landmarks, and per-landmark Euclidean error / clip median shoulder width.

Factors (each an "oracle", i.e. fitted/peeked on the scored data):
    R : root.  none | sh (minimal rotation taking rest shoulder line to observed) |
               opt (3-dof rotvec minimising SSE of the scored residual, multi-start)
    L : arm lengths.  rest | clip (per-clip median observed length) | frame (this frame's)
    W : shoulder width. rest | clip | frame (symmetric about the shoulder midpoint)
"""
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from common import *  # noqa

OUT = Path(__file__).parent
sk = SMPLXSkeleton(model_path="models")
rest = np.array(sk.rest_joints)
P0 = rest[0]
SH = {11: 16, 12: 17}
REST_LEN = {(11, 13): np.linalg.norm(rest[18] - rest[16]), (13, 15): np.linalg.norm(rest[20] - rest[18]),
            (12, 14): np.linalg.norm(rest[19] - rest[17]), (14, 16): np.linalg.norm(rest[21] - rest[19])}
REST_SW = float(np.linalg.norm(rest[16] - rest[17]))


def umeyama(src, tgt, scale):
    ms, mt = src.mean(0), tgt.mean(0)
    A, B = src - ms, tgt - mt
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    C = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        C[2, 2] = -1
        S = S.copy(); S[2] = -S[2]
    Rm = U @ C @ Vt
    s = S.sum() / ((A ** 2).sum() / len(src)) if scale else 1.0
    return s, Rm, mt - s * Rm @ ms


def model_points(frame, ok, Rroot, lengths, shoulders):
    """Closed-form arm joints. shoulders: dict mp->rest position (possibly width-edited)."""
    pts = {}
    for sm in (11, 12):
        pts[sm] = P0 + Rroot @ (shoulders[sm] - P0)
    for (a, b) in ((11, 13), (12, 14), (13, 15), (14, 16)):
        if ok[a] and ok[b]:
            d = frame[b] - frame[a]
            pts[b] = pts[a] + lengths[(a, b)] * d / np.linalg.norm(d)
        else:
            pts[b] = None
    return pts


def residual(frame, ok, present, Rroot, lengths, shoulders, mode, width):
    pts = model_points(frame, ok, Rroot, lengths, shoulders)
    src = np.array([pts[m] for m in present])
    tgt = np.array([frame[m] for m in present])
    if mode == "none":
        al = src - src.mean(0) + tgt.mean(0)  # translation only
        s = 1.0
    else:
        s, Rm, t = umeyama(src, tgt, mode == "sim")
        al = s * src @ Rm.T + t
    return np.linalg.norm(al - tgt, axis=1) / width, s


def shoulder_swing(frame, shoulders):
    d_rest = shoulders[11] - shoulders[12]
    d_obs = frame[11] - frame[12]
    a, b = d_rest / np.linalg.norm(d_rest), d_obs / np.linalg.norm(d_obs)
    ax = np.cross(a, b)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(ax / s * np.arctan2(s, a @ b)).as_matrix()


rng = np.random.default_rng(0)
INITS = [np.zeros(3)] + [Rotation.random(random_state=k).as_rotvec() for k in range(12)] + \
        [np.array([np.pi, 0, 0]), np.array([0, np.pi, 0]), np.array([0, 0, np.pi])]


def opt_root(frame, ok, present, lengths, shoulders, mode, width):
    f = lambda r: float((residual(frame, ok, present, Rotation.from_rotvec(r).as_matrix(),
                                  lengths, shoulders, mode, width)[0] ** 2).sum())
    best = None
    for r0 in INITS:
        r = minimize(f, r0, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-14, "maxiter": 4000, "maxfev": 8000})
        if best is None or r.fun < best.fun - 1e-15:
            best = r
    return Rotation.from_rotvec(best.x).as_matrix(), best


def verify_closed_form(clips, n=60):
    """Closed form vs the project's solve_swing_ik(global_orient) + forward_kinematics."""
    worst = 0.0
    for k in range(n):
        name, pose, oks, width = clips[k % len(clips)]
        fi = rng.integers(len(pose))
        frame, ok = pose[fi], oks[fi]
        r = Rotation.random(random_state=k).as_rotvec() * rng.uniform(0, 1)
        pts_ref, _ = ik_points(frame, ok, sk, root=r)
        pts = model_points(frame, ok, Rotation.from_rotvec(r).as_matrix(), REST_LEN,
                           {11: rest[16], 12: rest[17]})
        for m in MP:
            if pts[m] is not None and ok[m]:
                worst = max(worst, float(np.linalg.norm(pts[m] - pts_ref[m])))
    return worst


def main():
    clips = load_clips()
    print(f"closed form vs solve_swing_ik+FK, 60 random (frame, root) pairs: worst {verify_closed_form(clips):.2e} m")

    root_opts = ["none", "sh", "opt"]
    len_opts = ["rest", "clip", "frame"]
    sw_opts = ["rest", "clip", "frame"]
    modes = ["sim", "rigid"]
    configs = [(m, r, l, w) for m in modes for r in root_opts for l in len_opts for w in sw_opts]

    # per-landmark residuals stored per config, aligned across configs (same frame/landmark order)
    results = {c: [] for c in configs}
    clip_of = []
    scales = {c: [] for c in configs}
    root_deg = {c: [] for c in configs}
    t0 = time.time()
    for ci, (name, pose, oks, width) in enumerate(clips):
        clip_len = {}
        for (a, b) in REST_LEN:
            v = oks[:, a] & oks[:, b]
            clip_len[(a, b)] = float(np.median(np.linalg.norm(pose[v, b] - pose[v, a], axis=-1)))
        clip_sw = width
        for frame, ok in zip(pose, oks):
            present = [m for m in MP if ok[m]]
            if len(present) < 3:
                continue
            clip_of += [ci] * len(present)
            frame_len = {k: (np.linalg.norm(frame[k[1]] - frame[k[0]]) if ok[k[0]] and ok[k[1]]
                             else REST_LEN[k]) for k in REST_LEN}
            for c in configs:
                mode, ro, lo, wo = c
                lengths = {"rest": REST_LEN, "clip": clip_len, "frame": frame_len}[lo]
                sw = {"rest": REST_SW, "clip": clip_sw,
                      "frame": float(np.linalg.norm(frame[11] - frame[12]))}[wo]
                mid = (rest[16] + rest[17]) / 2
                half = (rest[16] - rest[17]) / 2
                half = half / np.linalg.norm(half) * sw / 2
                shoulders = {11: mid + half, 12: mid - half}
                if ro == "none":
                    Rr = np.eye(3)
                elif ro == "sh":
                    Rr = shoulder_swing(frame, shoulders)
                else:
                    Rr, _ = opt_root(frame, ok, present, lengths, shoulders, mode, width)
                e, s = residual(frame, ok, present, Rr, lengths, shoulders, mode, width)
                results[c] += list(e)
                scales[c].append(s)
                root_deg[c].append(np.degrees(Rotation.from_matrix(Rr).magnitude()))
        print(f"  clip {name} done ({time.time() - t0:.0f}s)", flush=True)

    clip_of = np.array(clip_of)
    names = [c[0] for c in clips]
    np.savez(OUT / "decomp_results.npz", clip_of=clip_of,
             **{"|".join(c): np.array(v) for c, v in results.items()},
             **{"scale|" + "|".join(c): np.array(v) for c, v in scales.items()},
             **{"root|" + "|".join(c): np.array(v) for c, v in root_deg.items()})

    print("\nconfig (mode root len sw)          pooled med   red.%   per-clip medians          scale med  |root| med")
    for c in configs:
        v = np.array(results[c])
        base = np.median(results[(c[0], "none", "rest", "rest")])
        pc = [np.median(v[clip_of == k]) * 100 for k in range(len(clips))]
        print(f"  {c[0]:<5} {c[1]:<4} {c[2]:<5} {c[3]:<5}   {np.median(v)*100:6.3f}%  {100*(1-np.median(v)/base):6.1f}   "
              f"{' '.join(f'{x:5.2f}' for x in pc)}   {np.median(scales[c]):.4f}   {np.median(root_deg[c]):6.2f}")


if __name__ == "__main__":
    main()
