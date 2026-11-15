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
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"Range bounds must be finite, got {value!r}")
    if lo > hi:
        raise ValueError(f"Range lo > hi: {value!r}")
    if distribution == "log_uniform" and lo <= 0:
        raise ValueError(f"log_uniform needs positive bounds, got {value!r}")
    return RangeSpec(lo, hi, distribution)


def _number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    return float(value)


def check_unit(value: Any, label: str) -> float:
    """`value` as a float, raising unless it is a finite number in [0, 1] (a probability / rate)."""
    v = _number(value, label)
    if not (np.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError(f"{label} must be in [0, 1], got {value!r}")
    return v


def check_non_negative(value: Any, label: str) -> float:
    """`value` as a float, raising unless it is finite and >= 0 (a limit, sigma or angle)."""
    v = _number(value, label)
    if not (np.isfinite(v) and v >= 0.0):
        raise ValueError(f"{label} must be finite and >= 0, got {value!r}")
    return v


def check_positive(value: Any, label: str) -> float:
    """`value` as a float, raising unless it is finite and > 0."""
    v = _number(value, label)
    if not (np.isfinite(v) and v > 0.0):
        raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    return v


def check_range(
    value: Any,
    label: str,
    distribution: str = "uniform",
    low: Optional[float] = None,
    high: Optional[float] = None,
    low_open: bool = False,
) -> RangeSpec:
    """
    Parse `value` as a range (see `parse_range`) and require its bounds to lie in `[low, high]`.

    `low_open` makes the lower bound exclusive (so `low=0, low_open=True` means "> 0"). Either
    limit may be None for "no limit". The error names `label`.
    """
    try:
        spec = parse_range(value, distribution)
    except ValueError as err:
        raise ValueError(f"{label}: {err}") from None
    if low is not None and (spec.lo <= low if low_open else spec.lo < low):
        sign = ">" if low_open else ">="
        raise ValueError(f"{label} lower bound must be {sign} {low}, got {value!r}")
    if high is not None and spec.hi > high:
        raise ValueError(f"{label} upper bound must be <= {high}, got {value!r}")
    return spec


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


MAX_COUPLED_TRIES = 100


def resample_coupled(draw, in_bounds, repair, max_tries: int = MAX_COUPLED_TRIES):
    """
    Truncate a coupled draw without clamping it.

    Calls `draw()` up to `max_tries` times and returns the first result for which
    `in_bounds(x)` holds, so the accepted vector keeps its coupling and no mass sits on a bound.
    If every try fails, `repair(x)` fixes the last draw (it should resample only the out-of-bound
    components from their truncated marginals). The loop is bounded.
    """
    x = draw()
    for _ in range(max_tries - 1):
        if in_bounds(x):
            return x
        x = draw()
    return x if in_bounds(x) else repair(x)


def truncated_ball(
    rng: np.random.Generator, sigma: float, limit: float, dim: int = 3
) -> np.ndarray:
    """
    `dim`-vector of iid Normal(0, sigma) components truncated to the ball ``|x| < limit``.

    Exact and loop-free: the norm is ``sigma * chi(dim)`` truncated by inverse-CDF, the direction
    uniform on the sphere.
    """
    from scipy.stats import chi

    check_truncation(sigma, limit, "truncated_ball")
    if sigma == 0:
        return np.zeros(dim)
    u = rng.uniform(0.0, chi.cdf(limit / sigma, dim))
    radius = min(sigma * float(chi.ppf(u, dim)), np.nextafter(limit, 0.0))
    direction = rng.normal(size=dim)
    return direction / np.linalg.norm(direction) * radius
