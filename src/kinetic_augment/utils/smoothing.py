"""
Temporal smoothing of landmark tracks, applied before fitting.

Every filter takes an array of shape (T, ..., D): T frames, any number of points, D coordinates
per point. A point is valid in a frame when all D coordinates are finite. Each point is smoothed
separately over each run of consecutive valid frames, so a filter never reaches across a gap
(a dropped hand, a lost face) and missing frames stay NaN. Runs shorter than `MIN_RUN` frames
are left as they are.

Filters:
  * `gaussian_smooth` -- centred Gaussian, zero lag; it needs future frames, so offline only.
  * `one_euro_smooth` -- the One Euro filter (Casiez et al., CHI 2012): causal, with a cutoff
    that rises with speed, so slow motion is smoothed hard and fast motion lags less. This is
    the kind of filter a real-time tracker uses.
"""

from typing import Callable, Dict, NamedTuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

MIN_RUN = 3


def valid_runs(valid: np.ndarray):
    """(start, stop) of each run of True in a 1-D boolean array."""
    padded = np.concatenate([[False], np.asarray(valid, bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _per_run(x: np.ndarray, smooth_run: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply `smooth_run` ((L, D) -> (L, D)) to every point's runs of valid frames."""
    x = np.asarray(x, dtype=np.float64)
    T, D = x.shape[0], x.shape[-1]
    flat = x.reshape(T, -1, D)
    out = flat.copy()
    valid = np.isfinite(flat).all(axis=-1)
    for p in range(flat.shape[1]):
        for start, stop in valid_runs(valid[:, p]):
            if stop - start >= MIN_RUN:
                out[start:stop, p] = smooth_run(flat[start:stop, p])
    return out.reshape(x.shape)


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Centred Gaussian over time, `sigma` in frames; ends are padded by repetition."""
    return _per_run(x, lambda run: gaussian_filter1d(run, sigma, axis=0, mode="nearest"))


def one_euro_smooth(
    x: np.ndarray, fps: float, min_cutoff: float = 1.0, beta: float = 5.0, d_cutoff: float = 1.0
) -> np.ndarray:
    """
    One Euro filter, restarted at each run. `min_cutoff` and `d_cutoff` are in Hz, `beta` in
    1 / (units per second): the cutoff is min_cutoff + beta * speed, with speed the norm of the
    point's smoothed velocity, so all D coordinates of a point share one cutoff.
    """
    te = 1.0 / fps

    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    a_d = alpha(d_cutoff)

    def run(points):
        out = np.empty_like(points)
        out[0], dx_hat = points[0], np.zeros(points.shape[1])
        for t in range(1, len(points)):
            dx = (points[t] - out[t - 1]) / te
            dx_hat = a_d * dx + (1 - a_d) * dx_hat
            a = alpha(min_cutoff + beta * np.linalg.norm(dx_hat))
            out[t] = a * points[t] + (1 - a) * out[t - 1]
        return out

    return _per_run(x, run)


class Smoothing(NamedTuple):
    label: str
    description: str
    apply: Callable[[np.ndarray, float], np.ndarray]  # (x, fps) -> smoothed x


SMOOTHING: Dict[str, Smoothing] = {
    "off": Smoothing("off", "MediaPipe landmarks as detected", lambda x, fps: np.array(x)),
    "gauss1": Smoothing(
        "Gaussian σ=1",
        "centred Gaussian, σ = 1 frame (light, no lag)",
        lambda x, fps: gaussian_smooth(x, 1.0),
    ),
    "gauss2": Smoothing(
        "Gaussian σ=2",
        "centred Gaussian, σ = 2 frames (strong, no lag)",
        lambda x, fps: gaussian_smooth(x, 2.0),
    ),
    "oneeuro": Smoothing(
        "One Euro",
        "causal One Euro filter (min cutoff 1 Hz, β 5), as a live tracker would",
        lambda x, fps: one_euro_smooth(x, fps),
    ),
}
