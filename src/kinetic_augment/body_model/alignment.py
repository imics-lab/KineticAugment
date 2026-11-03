"""
Similarity alignment between coordinate spaces (Umeyama, 1991).

SMPL-X joints are metric and pelvis-anchored; MediaPipe landmarks are
image-normalised with a pseudo-depth z. Comparing the two requires first
removing the scale, rotation and translation between them. Those three are
camera and convention artefacts, not information the conversion is supposed to
preserve, so measuring round-trip error without removing them would report
nuisance rather than loss.
"""

from typing import Optional, Tuple

import numpy as np


def similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool = True,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Least-squares similarity carrying `source` onto `target`.

    Solves for the scale, rotation and translation minimising
    ``|| s * R @ x + t - y ||^2``, following Umeyama's closed-form method. The
    rotation is constrained to be proper, so a mirrored target is fitted with
    the nearest true rotation rather than a reflection.

    Args:
        source: (N, 3) points to be transformed.
        target: (N, 3) points to match. Must be the same length as `source`.
        allow_scale: when False, scale is fixed at 1.0 and only rotation and
            translation are fitted. Use this whenever both spaces are already
            metric: a free scale absorbs systematic correspondence error into a
            parameter that looks like a fit, hiding it from the residual.
        weights: (N,) non-negative per-point weights; None weights all points equally. Points
            with zero weight do not influence the fit.

    Returns:
        (scale, rotation (3, 3), translation (3,)).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape != target.shape:
        raise ValueError(f"shape mismatch: {source.shape} vs {target.shape}")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"expected (N, 3) arrays, got {source.shape}")
    if source.shape[0] < 3:
        raise ValueError(f"need at least 3 points, got {source.shape[0]}")

    if weights is None:
        w = np.full(source.shape[0], 1.0 / source.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != (source.shape[0],) or (w < 0).any() or not np.isfinite(w).all():
            raise ValueError("weights must be (N,), finite and non-negative")
        if (w > 0).sum() < 3:
            raise ValueError("need at least 3 positively weighted points")
        w = w / w.sum()

    src_mean = w @ source
    tgt_mean = w @ target
    src_centred = source - src_mean
    tgt_centred = target - tgt_mean

    covariance = (tgt_centred * w[:, None]).T @ src_centred
    u, singular_values, vt = np.linalg.svd(covariance)

    # Flip the least-significant axis if the raw solution is a reflection.
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
        singular_values = singular_values.copy()
        singular_values[2] = -singular_values[2]

    rotation = u @ correction @ vt

    if allow_scale:
        source_variance = float(w @ (src_centred ** 2).sum(axis=1))
        scale = 1.0 if source_variance == 0 else singular_values.sum() / source_variance
    else:
        scale = 1.0
    translation = tgt_mean - scale * rotation @ src_mean

    return float(scale), rotation, translation


def apply_similarity(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply a similarity returned by `similarity_transform`."""
    return scale * (np.asarray(points, dtype=np.float64) @ rotation.T) + translation
