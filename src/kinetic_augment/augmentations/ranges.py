"""Seeded range-sampling helpers for per-clip augmentations."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

_DISTRIBUTIONS = ("uniform", "log_uniform")


def clip_rng(seed: int, clip_id: str, key: str) -> np.random.Generator:
    """Generator that depends only on (seed, clip_id, key)."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), zlib.crc32(clip_id.encode()), zlib.crc32(key.encode())])
    )


@dataclass(frozen=True)
class RangeSpec:
    """A closed range with a sampling distribution."""

    lo: float
    hi: float
    distribution: str = "uniform"

    def draw(self, rng: np.random.Generator) -> float:
        if self.distribution == "log_uniform":
            return float(np.exp(rng.uniform(np.log(self.lo), np.log(self.hi))))
        return float(rng.uniform(self.lo, self.hi))


def parse_range(value: Any, default_distribution: str = "uniform") -> RangeSpec:
    """Parse ``[lo, hi]`` or ``{"range": [lo, hi], "distribution": ...}``."""
    distribution = default_distribution
    bounds = value
    if isinstance(value, dict):
        if "range" not in value:
            raise ValueError(f"Range dict needs a 'range' key, got {value!r}")
        bounds = value["range"]
        distribution = value.get("distribution", default_distribution)
    if distribution not in _DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution {distribution!r}")
    if (
        not isinstance(bounds, (list, tuple))
        or len(bounds) != 2
        or not all(isinstance(b, (int, float, np.number)) for b in bounds)
    ):
        raise ValueError(f"Range must be [lo, hi], got {value!r}")
    lo, hi = float(bounds[0]), float(bounds[1])
    if lo > hi:
        raise ValueError(f"Range lo > hi: {value!r}")
    if distribution == "log_uniform" and lo <= 0:
        raise ValueError(f"log_uniform needs positive bounds, got {value!r}")
    return RangeSpec(lo, hi, distribution)


def check_truncation(sigma: Any, limit: Any, label: str) -> None:
    """
    Raise unless `sigma >= 0` and, when `sigma > 0`, `limit > 0`.

    A zero (or negative) limit with a positive sigma would make `truncated_normal` reject every
    sample forever, so the augmentations call this when they are built (at profile load).
    """
    sigma, limit = float(sigma), float(limit)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"{label}: sigma must be finite and >= 0, got {sigma!r}")
    if sigma > 0 and not limit > 0:
        raise ValueError(f"{label}: truncation must be > 0 when sigma > 0, got {limit!r}")


def truncated_normal(
    rng: np.random.Generator, sigma: float, limit: float, size: Optional[Union[int, tuple]] = None
) -> np.ndarray:
    """
    Normal(0, sigma) by rejection resampling until ``|x| <= limit``.

    Raises:
        ValueError: `sigma < 0`, or `sigma > 0` with `limit <= 0` (which would never finish).
    """
    check_truncation(sigma, limit, "truncated_normal")
    shape = () if size is None else size
    if sigma == 0:
        return np.zeros(shape)
    out = rng.normal(0.0, sigma, size=shape)
    bad = np.abs(out) > limit
    while np.any(bad):
        out = np.where(bad, rng.normal(0.0, sigma, size=shape), out)
        bad = np.abs(out) > limit
    return np.asarray(out)
