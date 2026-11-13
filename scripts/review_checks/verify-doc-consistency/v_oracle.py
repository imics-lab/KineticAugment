"""Independent oracle decomposition (doc L651-663): what do an oracle root and oracle per-clip
arm lengths explain, in the aligned (per-frame similarity) metric of the old harness?"""
import glob, sys, time
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares

from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity

CORR = [(16, 11), (17, 12), (18, 13), (19, 14), (20, 15), (21, 16)]
full = SMPLXSkeleton(model_path="models")
base = SimpleNamespace(rest_joints=full.rest_joints[:22].copy(), parents=full.parents[:22].copy(), num_joints=22)
BONES = [(16, 18, 11, 13), (18, 20, 13, 15), (17, 19, 12, 14), (19, 21, 14, 16)]


def skel_with_lengths(lengths, shoulder_w=None):
    rj = base.rest_joints.copy()
    if shoulder_w is not None:  # scale collar->shoulder offsets symmetrically about the spine3 midline
        cur = np.linalg.norm(rj[16] - rj[17])
        mid = (rj[16] + rj[17]) / 2
        for s in (16, 17):
            rj[s] = mid + (rj[s] - mid) * shoulder_w / cur
    # rebuild arm chain with new lengths along rest directions
    orig = base.rest_joints
    for (p, c, _, _), L in zip(BONES, lengths):
        u = orig[c] - orig[p]
        u /= np.linalg.norm(u)
        rj[c] = rj[p] + u * L
    return SimpleNamespace(rest_joints=rj, parents=base.parents, num_joints=22)


def residuals(frame, ok, skel, root):
    lm = np.where(ok[:, None], frame, np.nan)
    rot = solve_swing_ik(lm, skel, SMPLX_ARM_BONES, global_orient=root)
    J = forward_kinematics(rot, skel)
    pres = [(s, m) for s, m in CORR if ok[m]]
    src = np.array([J[s] for s, _ in pres]); tgt = np.array([frame[m] for _, m in pres])
    al = apply_similarity(src, *similarity_transform(src, tgt))
    return al - tgt, [m for _, m in pres]


def best_root(frame, ok, skel):
    best = None
    for x0 in (np.zeros(3), np.array([0, 0.3, 0]), np.array([0, -0.3, 0]), np.array([0.3, 0, 0])):
        r = least_squares(lambda x: residuals(frame, ok, skel, x)[0].ravel(), x0, x_scale=0.1)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def main():
    t0 = time.time()
    results = {k: [] for k in ["baseline", "root", "length", "both", "length+sw", "both+sw"]}
    per_clip = {}
    for path in sorted(glob.glob("data/test_output/*_world.npz")):
        clip = path.split("/")[-1][:5]
        d = np.load(path)
        P = d["landmarks"].astype(np.float64); V = d["visibility"]
        okA = np.isfinite(P).all(-1) & (V >= 0.5)
        both = okA[:, 11] & okA[:, 12]
        width = float(np.median(np.linalg.norm(P[both, 11] - P[both, 12], axis=-1)))
        lens = []
        for p, c, mp, mc in BONES:
            m = okA[:, mp] & okA[:, mc]
            lens.append(float(np.median(np.linalg.norm(P[m, mc] - P[m, mp], axis=-1))))
        sk_len = skel_with_lengths(lens)
        sk_len_sw = skel_with_lengths(lens, shoulder_w=width)
        clip_res = {k: [] for k in results}
        for f in range(len(P)):
            ok = okA[f]
            if not (ok[11] and ok[12]) or sum(ok[m] for _, m in CORR) < 3:
                continue
            for key, skel, opt in [("baseline", base, False), ("root", base, True), ("length", sk_len, False),
                                   ("both", sk_len, True), ("length+sw", sk_len_sw, False), ("both+sw", sk_len_sw, True)]:
                root = best_root(P[f], ok, skel) if opt else np.zeros(3)
                e, _ = residuals(P[f], ok, skel, root)
                vals = list(np.linalg.norm(e, axis=1) / width)
                results[key] += vals; clip_res[key] += vals
        per_clip[clip] = {k: np.median(v) * 100 for k, v in clip_res.items()}
        print(clip, {k: round(v, 2) for k, v in per_clip[clip].items()}, f"{time.time()-t0:.0f}s", flush=True)
    b = np.median(results["baseline"])
    for k, v in results.items():
        m = np.median(v)
        print(f"{k:10s} pooled median {m*100:.2f}%  reduction {(1-m/b)*100:.1f}%  n={len(v)}")


main()
