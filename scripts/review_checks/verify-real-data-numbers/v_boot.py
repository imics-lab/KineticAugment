"""Bootstrap CIs, order-dependence, scale under oracles. Imports own closed form (v_cf)."""
import sys, itertools
sys.path.insert(0, "scripts/review_checks/verify-real-data-numbers")
import numpy as np
import v_cf as C

def cfg_err(root, arm, width, metric, want_scale=False):
    """returns list of (clip_idx, frame_idx, err array) and scales"""
    rows, scales = [], []
    for ci, c in enumerate(C.clips):
        for fi, (fr, ok) in enumerate(c["frames"]):
            if arm == "rest": lens = C.REST_LEN
            elif arm == "clip": lens = c["clen"]
            else: lens = {k: np.linalg.norm(fr[k[1]] - fr[k[0]]) if ok[k[0]] and ok[k[1]] else 0 for k in C.CHAIN}
            w = None if width == "rest" else (c["W"] if width == "clip" else np.linalg.norm(fr[11] - fr[12]))
            R = np.eye(3) if root == "none" else C.analytic_root(fr, ok)
            e = C.frame_err(fr, ok, R, lens, w, metric, c["W"])
            rows.append((ci, fi, e))
            if want_scale:
                pres = [m for m in C.MP if ok[m]]
                pos = C.recon(fr, ok, R, lens, w)
                src = np.array([pos[m] for m in pres]); tgt = np.array([fr[m] for m in pres])
                a = src - src.mean(0); b = tgt - tgt.mean(0)
                U, S, Vt = np.linalg.svd(a.T @ b); d = np.sign(np.linalg.det(Vt.T @ U.T))
                scales.append((S * np.array([1, 1, d])).sum() / (a ** 2).sum())
    return rows, scales

med = lambda rows: np.median(np.concatenate([r[2] for r in rows])) * 100

for metric in ["sim", "rigid"]:
    print(f"===== metric {metric}")
    R = {}
    for root in ["none", "analytic"]:
        for arm in ["rest", "clip"]:
            for width in ["rest", "clip"]:
                R[(root, arm, width)] = cfg_err(root, arm, width, metric)[0]
    base = med(R[("none", "rest", "rest")])
    red = {k: 100 * (1 - med(v) / base) for k, v in R.items()}
    for k, v in red.items():
        print(f"   {k}: median {med(R[k]):.3f}%  reduction {v:.1f}%")
    # root increment across all 3-factor orderings
    incs = []
    for order in itertools.permutations(["root", "arm", "width"]):
        state = {"root": "none", "arm": "rest", "width": "rest"}
        for fac in order:
            before = red[(state["root"], state["arm"], state["width"])]
            state[fac] = "analytic" if fac == "root" else "clip"
            after = red[(state["root"], state["arm"], state["width"])]
            if fac == "root": incs.append(round(after - before, 1))
    print(f"   root increment over 6 orderings (root/arm/width): {sorted(incs)}")
    inter = red[("analytic", "clip", "rest")] - red[("analytic", "rest", "rest")] - red[("none", "clip", "rest")]
    print(f"   2-factor (root x arm, width rest) interaction: {inter:.2f} pts; Shapley root {(red[('analytic','rest','rest')] + red[('analytic','clip','rest')] - red[('none','clip','rest')])/2:.1f}  arm {(red[('none','clip','rest')] + red[('analytic','clip','rest')] - red[('analytic','rest','rest')])/2:.1f}")
    # bootstraps
    rng = np.random.default_rng(0)
    keys = [("none", "rest", "rest"), ("analytic", "rest", "rest"), ("none", "clip", "rest"), ("analytic", "clip", "rest")]
    for mode in ["frame", "clip"]:
        stats = []
        for _ in range(1000):
            if mode == "clip":
                pick = rng.integers(0, 5, 5)
                sel = lambda rows: np.concatenate([r[2] for ci in pick for r in rows if r[0] == ci])
            else:
                n = len(R[keys[0]]); idx = rng.integers(0, n, n)
                sel = lambda rows, idx=idx: np.concatenate([rows[i][2] for i in idx])
            m = [np.median(sel(R[k])) for k in keys]
            rr = [1 - x / m[0] for x in m]
            stats.append((rr[1] * 100, rr[2] * 100, (rr[3] - rr[1] - rr[2]) * 100))
        stats = np.array(stats)
        lo, hi = np.percentile(stats, 2.5, 0), np.percentile(stats, 97.5, 0)
        print(f"   {mode}-bootstrap 95% CI: root [{lo[0]:.1f},{hi[0]:.1f}]  arm [{lo[1]:.1f},{hi[1]:.1f}]  interaction [{lo[2]:.1f},{hi[2]:.1f}]")

# unexplained breakdown: alternative order (per-frame arms first, then width)
print("===== 'unexplained' path, sim metric, analytic root")
for arm, width in [("clip", "rest"), ("clip", "clip"), ("frame", "clip"), ("frame", "frame"), ("frame", "rest")]:
    for metric in ["sim", "rigid"]:
        rows, _ = cfg_err("analytic", arm, width, metric)
        base = med(cfg_err("none", "rest", "rest", metric)[0])
        print(f"   arm={arm:5s} width={width:5s} {metric:5s}: {med(rows):.3f}%  remaining {100*med(rows)/base:.1f}% of baseline")

print("===== similarity scale under configurations")
for root, arm, width in [("none", "rest", "rest"), ("none", "clip", "rest"), ("none", "rest", "clip"), ("none", "clip", "clip"),
                         ("analytic", "clip", "clip"), ("none", "frame", "frame"), ("analytic", "frame", "frame")]:
    _, sc = cfg_err(root, arm, width, "sim", want_scale=True)
    print(f"   root={root:8s} arm={arm:5s} width={width:5s}: median {np.median(sc):.4f}  p5 {np.percentile(sc,5):.4f}  p95 {np.percentile(sc,95):.4f}")
print(f"   REST_W {C.REST_W:.4f}  clip widths {[round(c['W'],4) for c in C.clips]}")
print(f"   rest arm lengths {[round(v,4) for v in C.REST_LEN.values()]}")
