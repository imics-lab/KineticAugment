"""Try to reproduce the doc's retracted-section table (10/20/45 deg: mean 9.87/16.55/47.04 mm, max 81/119/307 mm)."""
import numpy as np
from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.kinematics import forward_kinematics
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES, solve_swing_ik
from kinetic_augment.body_model.converter import fit_body, reconstruct_body, ARM_CORRESPONDENCE as CORR
sk = SMPLXSkeleton(model_path="models")
DOC = {10: (9.87, 81), 20: (16.55, 119), 45: (47.04, 307)}
def metrics(truth):
    J = forward_kinematics(truth, sk); L = np.full((33, 3), np.nan)
    for s, m in CORR: L[m] = J[s]
    F = forward_kinematics(solve_swing_ik(L, sk, SMPLX_ARM_BONES), sk)
    src = np.array([F[s] for s, _ in CORR]); tgt = np.array([L[m] for _, m in CORR])
    raw = np.linalg.norm(src - tgt, axis=1) * 1e3
    sim = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt)) - tgt, axis=1) * 1e3
    rig = np.linalg.norm(apply_similarity(src, *similarity_transform(src, tgt, allow_scale=False)) - tgt, axis=1) * 1e3
    out = reconstruct_body(fit_body(L, np.isfinite(L).all(1), sk), sk)
    conv = np.array([np.linalg.norm(out[m] - L[m]) for _, m in CORR]) * 1e3
    return dict(raw=raw, sim=sim, rig=rig, conv=conv)
best = []
for sigma in (0.0, 0.3, 0.4):
    for trials in (1, 20, 50, 100, 200):
        for seed in range(0, 60 if trials == 1 else 3):
            rng = np.random.default_rng(seed)
            rows = {}
            for deg in (0, 10, 20, 45):
                acc = {k: [] for k in ("raw", "sim", "rig", "conv")}
                for _ in range(trials):
                    t = np.zeros((55, 3))
                    if deg:
                        a = rng.normal(size=3); t[0] = a / np.linalg.norm(a) * np.radians(deg)
                    if sigma:
                        for j, *_ in SMPLX_ARM_BONES: t[j] = rng.normal(0, sigma, 3)
                    for k, v in metrics(t).items(): acc[k].append(v)
                rows[deg] = {k: np.concatenate(v) for k, v in acc.items()}
            for mk in ("raw", "sim", "rig", "conv"):
                for xk in ("raw", "sim", "rig", "conv"):
                    err = sum(abs(rows[d][mk].mean() - DOC[d][0]) / DOC[d][0] + abs(rows[d][xk].max() - DOC[d][1]) / DOC[d][1] for d in DOC)
                    best.append((err, sigma, trials, seed, mk, xk, [(round(rows[d][mk].mean(), 2), round(rows[d][xk].max(), 1)) for d in DOC]))
best.sort(key=lambda r: r[0])
for b in best[:8]: print(f"rel.err {b[0]:.3f} sigma {b[1]} trials {b[2]} seed {b[3]} mean:{b[4]} max:{b[5]} -> {b[6]}")
print("doc:", [DOC[d] for d in DOC])
