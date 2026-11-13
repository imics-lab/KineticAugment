"""Identify the temporal filter MediaPipe applies to pose_world_landmarks.

Hypothesis: default (smooth_landmarks=True) == OneEuro(nosmooth) with
min_cutoff=0.1, beta=40, derivate_cutoff=1.0, value scaling disabled, at the
Python solution API's simulated 30 fps timestamps (solution_base.py:312-314).
If the per-frame raw predictions are identical in both runs (same ROI chain),
re-filtering the nosmooth output should reproduce the default output exactly.
"""
import math
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parent / "extract"
CLIPS = ["01986", "01987", "01988", "01991", "01992"]
DT = 33333e-6  # simulated timestamp increment, seconds


class LowPass:
    def __init__(self):
        self.init = False; self.raw = None; self.stored = None
    def apply(self, v, a):
        r = v if not self.init else a * v + (1 - a) * self.stored
        self.init = True; self.raw = v; self.stored = r
        return r


class OneEuro:
    def __init__(self, min_cutoff, beta, dcutoff, freq=30.0):
        self.mc, self.beta, self.dc, self.freq = min_cutoff, beta, dcutoff, freq
        self.x, self.dx = LowPass(), LowPass(); self.last_t = 0
    def alpha(self, cutoff):
        te = 1.0 / self.freq; tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    def apply(self, t_ns, v):
        if self.last_t != 0 and t_ns != 0:
            self.freq = 1.0 / ((t_ns - self.last_t) * 1e-9)
        self.last_t = t_ns
        dv = (v - self.x.raw) * self.freq if self.x.init else 0.0
        edv = self.dx.apply(dv, self.alpha(self.dc))
        cutoff = self.mc + self.beta * abs(edv)
        return self.x.apply(v, self.alpha(cutoff))


def one_euro_track(track, params):
    """track: (T,33,3); reset filters on missing frames (as MediaPipe does)."""
    T = track.shape[0]
    out = np.full_like(track, np.nan)
    filters = None
    for t in range(T):
        f = track[t]
        if not np.isfinite(f).all():
            filters = None
            continue
        if filters is None:
            filters = [[OneEuro(*params) for _ in range(3)] for _ in range(33)]
        t_ns = int(round((t + 1) * 33333 * 1000))  # timestamps are 33333us*k
        for j in range(33):
            for k in range(3):
                out[t, j, k] = filters[j][k].apply(t_ns, float(f[j, k]))
    return out


def main():
    cands = [(0.1, 40.0, 1.0), (0.05, 80.0, 1.0), (0.1, 10.0, 1.0), (0.01, 40.0, 1.0)]
    for params in cands:
        errs = []
        first_eq = []
        for c in CLIPS:
            raw = np.load(W / "nosmooth" / f"{c}_world.npz")["landmarks"].astype(np.float64)
            sm = np.load(W / "default" / f"{c}_world.npz")["landmarks"].astype(np.float64)
            pred = one_euro_track(raw, params)
            ok = np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(sm).all(axis=(1, 2))
            errs.append(np.abs(pred[ok] - sm[ok]).ravel())
            first_eq.append(np.abs(raw[0] - sm[0]).max())
        e = np.concatenate(errs)
        print(f"OneEuro{params}: |refiltered(nosmooth) - default|  max={e.max():.2e} m  "
              f"median={np.median(e):.2e} m   frame0 raw==smoothed max diff={max(first_eq):.1e}")


if __name__ == "__main__":
    main()
