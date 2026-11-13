"""Independent re-check of headline real-data numbers (verify-real-data-numbers).

Written from scratch; does not import the tester's scripts. Uses own Kabsch/Umeyama
implementation for the evaluation-time fits so a bug in alignment.py would show up.
"""
import glob
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import solve_swing_ik, SMPLX_ARM_BONES
from kinetic_augment.body_model.kinematics import forward_kinematics

MP = [11, 12, 13, 14, 15, 16]
BONES = [(11, 13), (13, 15), (12, 14), (14, 16)]
BNAME = ["L up", "L fore", "R up", "R fore"]


def my_fit(src, tgt, scale=True, rot=True):
    """least squares s,R,t: s R src + t ~ tgt. own implementation."""
    ms, mt = src.mean(0), tgt.mean(0)
    a, b = src - ms, tgt - mt
    if rot:
        H = a.T @ b
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        sv = (S * np.diag(D)).sum()
    else:
        R = np.eye(3)
        sv = np.trace(b.T @ a)
    s = sv / (a ** 2).sum() if scale else 1.0
    t = mt - s * R @ ms
    return s, R, t


def ang(u, v):
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    sk = SMPLXSkeleton(model_path="models")
    files = sorted(glob.glob("data/test_output/*_world.npz"))
    E = {k: [] for k in ["direct", "sim", "rigid_on_direct", "transl_only", "loo", "ikframe_sim"]}
    perjoint = {m: [] for m in MP}
    clipE = {}
    scales, rotmag, dir_direct, dir_ik, len_al, len_dir = [], [], {b: [] for b in BNAME}, [], {b: [] for b in BNAME}, {b: [] for b in BNAME}
    cov = {m: 0 for m in MP}
    nframes, skipped = 0, 0
    for f in files:
        d = np.load(f)
        pose = d["landmarks"].astype(np.float64)
        ok_all = np.isfinite(pose).all(-1) & (d["visibility"] >= 0.5)
        both = ok_all[:, 11] & ok_all[:, 12]
        W = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
        name = Path(f).stem[:5]
        cE = []
        for fr, ok in zip(pose, ok_all):
            nframes += 1
            for m in MP:
                cov[m] += int(ok[m])
            try:
                fit = fit_body(fr, ok, sk)
            except ValueError:
                skipped += 1
                continue
            out = reconstruct_body(fit, sk)
            pres = [m for m in MP if ok[m]]
            src = np.array([out[m] for m in pres]); tgt = np.array([fr[m] for m in pres])
            e = np.linalg.norm(src - tgt, axis=1) / W
            E["direct"] += list(e); cE += list(e)
            for m, ee in zip(pres, e):
                perjoint[m].append(ee)
            s, R, t = my_fit(src, tgt, scale=True)
            al = s * src @ R.T + t
            E["sim"] += list(np.linalg.norm(al - tgt, axis=1) / W)
            scales.append(s)
            s2, R2, t2 = my_fit(src, tgt, scale=False)
            E["rigid_on_direct"] += list(np.linalg.norm(src @ R2.T + t2 - tgt, axis=1) / W)
            # IK frame output (root=0, no rigid) -> translation-only fit & LOO rigid
            masked = np.where(ok[:, None], fr, np.nan)
            rots = solve_swing_ik(masked, sk, SMPLX_ARM_BONES)
            posed = forward_kinematics(rots, sk)
            ik = np.array([posed[s_] for s_, m in ARM_CORRESPONDENCE if ok[m]])
            # translation-only
            _, _, tt = my_fit(ik, tgt, scale=False, rot=False)
            E["transl_only"] += list(np.linalg.norm(ik + tt - tgt, axis=1) / W)
            # sim on IK frame (old harness lineage)
            s3, R3, t3 = my_fit(ik, tgt, scale=True)
            E["ikframe_sim"] += list(np.linalg.norm(s3 * ik @ R3.T + t3 - tgt, axis=1) / W)
            # LOO rigid: fit on all-but-one, score the held-out one
            if len(pres) >= 4:
                for i in range(len(pres)):
                    keep = [j for j in range(len(pres)) if j != i]
                    _, Rl, tl = my_fit(ik[keep], tgt[keep], scale=False)
                    E["loo"].append(np.linalg.norm(Rl @ ik[i] + tl - tgt[i]) / W)
            # rotation magnitude of the converter's rigid (out vs IK frame)
            _, Rc, _ = my_fit(ik, src, scale=False)
            rotmag.append(np.degrees(Rot.from_matrix(Rc).magnitude()))
            for (a, b), bn in zip(BONES, BNAME):
                if ok[a] and ok[b]:
                    dir_direct[bn].append(ang(out[b] - out[a], fr[b] - fr[a]))
                    ia = [m for _, m in ARM_CORRESPONDENCE].index(a); ib = [m for _, m in ARM_CORRESPONDENCE].index(b)
                    pa = posed[ARM_CORRESPONDENCE[ia][0]]; pb = posed[ARM_CORRESPONDENCE[ib][0]]
                    dir_ik.append(ang(pb - pa, fr[b] - fr[a]))
                    lo = np.linalg.norm(fr[b] - fr[a])
                    len_dir[bn].append(abs(np.linalg.norm(out[b] - out[a]) - lo) / W)
                    ali = s * (out[b] - out[a])
                    len_al[bn].append(abs(np.linalg.norm(ali) - lo) / W)
        clipE[name] = np.median(cE) * 100
    p = lambda v, q=50: np.percentile(v, q) * 100
    print(f"frames {nframes} skipped {skipped}")
    for k, v in E.items():
        print(f"{k:18s} median {p(v):7.3f}%  p95 {p(v,95):7.3f}%  n={len(v)}")
    print("per-joint direct med/p95/cov:")
    for m in MP:
        print(f"  {m}: {p(perjoint[m]):.2f} / {p(perjoint[m],95):.2f}  cov {100*cov[m]/nframes:.0f}%")
    print("per-clip direct:", {k: round(v, 2) for k, v in clipE.items()})
    print(f"sim scale median {np.median(scales):.4f} p5 {np.percentile(scales,5):.4f} p95 {np.percentile(scales,95):.4f}")
    print(f"converter rigid rotation vs IK frame: median {np.median(rotmag):.2f} p95 {np.percentile(rotmag,95):.2f} deg")
    print("direction err direct (deg):", {b: (round(np.median(v), 3), round(np.percentile(v, 95), 2)) for b, v in dir_direct.items()},
          "pooled", round(np.median(np.concatenate(list(dir_direct.values()))), 3))
    print(f"direction err IK frame: median {np.median(dir_ik):.2e} max {np.max(dir_ik):.2e} n={len(dir_ik)}")
    print("length err aligned:", {b: round(p(v), 2) for b, v in len_al.items()}, "pooled", round(p(np.concatenate(list(len_al.values()))), 2))
    print("length err direct:", {b: round(p(v), 2) for b, v in len_dir.items()}, "pooled", round(p(np.concatenate(list(len_dir.values()))), 2))


if __name__ == "__main__":
    main()
