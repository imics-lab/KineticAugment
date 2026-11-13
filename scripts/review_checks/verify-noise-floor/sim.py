#!/usr/bin/env python3
"""Verifier simulation: what do Method A / B read when the TRUE per-axis noise sigma is known?
Truth = repo smoothed arm trajectories, further Gaussian-smoothed (sigma=3 frames).
Noise = iid N(0, s^2) per axis. Optional One Euro(0.1,40,1) filter (validated vs MediaPipe to 1.2e-7 m).
Estimators re-implemented to match scripts/measure_noise_floor.py exactly (est_A/est_B from verify.py).
Reference value: sqrt(3)*s (the quantity B targets for xyz under iid; also RMS 3-D noise magnitude).
"""
import math, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import est_A, est_B, OE, load, CLIPS, ARM

rng = np.random.default_rng(0)

def oe_track(x, ok):
    out = np.full_like(x, np.nan); fs = None
    for t in range(len(x)):
        if not ok[t]:
            fs = None; continue
        if fs is None:
            fs = [OE(0.1, 40.0, 1.0) for _ in range(3)]
        for k in range(3):
            out[t, k] = fs[k].apply((t + 1) * 33333, x[t, k])
    return out

def truths(stationary=False):
    for c in CLIPS:
        L, ok = load("repo", c)
        for j in ARM:
            o = ok[:, j]
            if o.mean() < 0.5:
                continue
            x = L[:, j].copy()
            idx = np.arange(len(x))
            for k in range(3):
                x[~o, k] = np.interp(idx[~o], idx[o], x[o, k])
            tr = gaussian_filter1d(x, 3.0, axis=0, mode="nearest")
            if stationary:
                tr = np.repeat(tr[:1], len(tr), axis=0)
            yield c, j, tr, o

def run(s_mm, filt, stationary=False, step=1, reps=3):
    s = s_mm / 1000.0; ref = math.sqrt(3) * s
    A, B, A0, B0 = [], [], [], []
    for _ in range(reps):
        for c, j, tr, o in truths(stationary):
            p = tr + rng.normal(0, s, tr.shape)
            if filt:
                p = oe_track(p, o)
            pp, oo, tt = p[::step], o[::step], tr[::step]
            A.append(est_A(pp, oo, [0, 1, 2]) / ref); B.append(est_B(pp, oo, [0, 1, 2]) / ref)
            A0.append(est_A(tt, oo, [0, 1, 2]) / ref); B0.append(est_B(tt, oo, [0, 1, 2]) / ref)
    f = lambda v: np.nanmedian(v)
    return f(A), f(B), f(A0), f(B0)

print("pooled median over 28 arm (clip,landmark) entries x 3 reps; values are estimator / (sqrt(3)*s)")
print("A0/B0 = estimator applied to the noise-free truth (motion leakage), same normalisation")
for stat in (True, False):
    for filt in (False, True):
        for s_mm in (0.5, 1.0, 2.0, 4.0):
            for step in (1, 2, 3):
                if stat and step > 1:
                    continue
                a, b, a0, b0 = run(s_mm, filt, stat, step)
                print(f"{'stationary' if stat else 'real-motion':<11} filt={str(filt):<5} s={s_mm:>3} mm/axis step={step}: "
                      f"A/ref={a:.3f} B/ref={b:.3f} | truth-only A0/ref={a0:.3f} B0/ref={b0:.3f}")
