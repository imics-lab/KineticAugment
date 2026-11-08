"""
Week-4 temporal augmentations: time warping, frame dropping and hand dropout.

Time warping builds fractional source times and resamples the clip with `resample_clip`.
`times_from_rates` turns per-interval output durations into those times: `durations[i]` is how
many output frames source interval `i` lasts. The output grid is 0..round(total) with the last
point pinned to `total`, so the first and last source frames are always kept and the grid steps
are exactly one output frame (only the final step can differ, shorter or longer).

Frame dropping keeps a subset of frames (nearest to a fps grid, or random with a gap limit in
real time from `clip.times_s`), optionally filling the dropped ones by holding or interpolating.
Hand dropout works on the outputs: runs of NaN hand rows, drawn only on detected frames.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import dataclasses

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import tier1
from .base import AugmentationRegistry, ClipAugmentation
from .clip import ChainContext, ClipData, resample_clip, select_frames
from .ranges import parse_range

__all__ = [
    "times_from_rates",
    "varied_durations",
    "segment_durations",
    "TimeWarp",
    "FrameDrop",
    "HandDropout",
]

WARP_MODES = ("none", "varied", "segment")
DROP_MODES = ("rate", "random")
FILL_MODES = ("none", "hold", "interpolate")


def _mode_probs(weights: Any, allowed: Sequence[str], label: str) -> np.ndarray:
    """Normalised weights in `allowed` order; unknown keys, negatives or all-zero raise."""
    if not isinstance(weights, dict):
        raise ValueError(f"{label} must be a dict of mode weights, got {weights!r}")
    unknown = [k for k in weights if k not in allowed]
    if unknown:
        raise ValueError(f"Unknown {label} keys {unknown!r}; expected {list(allowed)!r}")
    values = np.array([float(weights.get(k, 0.0)) for k in allowed])
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError(f"{label} must be non-negative with a positive sum, got {weights!r}")
    return values / values.sum()


def times_from_rates(durations: np.ndarray) -> np.ndarray:
    """
    Source times at the output frames `0..round(total)` for per-interval output `durations`.

    `durations[i]` is the output length of source interval `i`; the cumulative sum gives each
    source frame's output time. At least 2 frames; the endpoints (source 0 and T-1) are kept.
    """
    durations = np.asarray(durations, dtype=np.float64)
    if durations.ndim != 1 or len(durations) < 1 or np.any(durations <= 0):
        raise ValueError("durations must be a non-empty positive 1-D array")
    cumulative = np.concatenate([[0.0], np.cumsum(durations)])
    total = float(cumulative[-1])
    count = max(int(round(total)), 1)
    out = np.arange(count + 1, dtype=np.float64)
    out[-1] = total
    return np.interp(out, cumulative, np.arange(len(cumulative), dtype=np.float64))


def varied_durations(
    count: int, knot_rates: Sequence[float], protect_end: float, total_factor: float
) -> np.ndarray:
    """
    Per-interval output durations of a smoothly varying playback speed, shape (count-1,).

    `knot_rates` (speeds) are interpolated with a monotone cubic over the first
    `1 - protect_end` of the clip and held constant over the rest; durations are `1 / rate`,
    normalised so they sum to `total_factor * (count - 1)`.
    """
    n = count - 1
    rates = np.asarray(knot_rates, dtype=np.float64)
    if n < 1:
        raise ValueError("need at least 2 frames")
    if len(rates) < 2 or np.any(rates <= 0):
        raise ValueError("need at least 2 positive knot rates")
    if not 0.0 <= protect_end < 1.0:
        raise ValueError(f"protect_end must be in [0, 1), got {protect_end}")
    split = n * (1.0 - protect_end)
    centres = np.arange(n) + 0.5
    if split <= 0:
        rate = np.full(n, rates[-1])
    else:
        curve = PchipInterpolator(np.linspace(0.0, split, len(rates)), rates)
        rate = np.where(centres <= split, curve(np.minimum(centres, split)), rates[-1])
    durations = 1.0 / rate
    return durations * (total_factor * n / durations.sum())


def segment_durations(
    count: int,
    start_frac: float,
    width_frac: float,
    factor: float,
    uniform: float,
    blend_frames: int = 3,
) -> np.ndarray:
    """
    Per-interval output durations, shape (count-1,): `uniform` everywhere and
    `uniform * factor` inside the window `[start_frac, start_frac + width_frac]` of the clip,
    with a linear blend over the first and last `blend_frames` intervals inside the window.
    Intervals whose centre lies outside the window keep exactly `uniform`.
    """
    n = count - 1
    if n < 1:
        raise ValueError("need at least 2 frames")
    lo, hi = start_frac * n, (start_frac + width_frac) * n
    centres = np.arange(n) + 0.5
    blend = max(float(blend_frames), 1e-9)
    # ramp inward: 0 at the window edges, 1 after `blend_frames` intervals inside (or lower at
    # the centre of a window narrower than twice the blend); exactly 0 outside the window
    weight = np.clip(np.minimum(centres - lo, hi - centres) / blend, 0.0, 1.0)
    return uniform * (1.0 + (factor - 1.0) * weight)


@AugmentationRegistry.register("temporal.time_warp")
class TimeWarp(ClipAugmentation):
    """Retime the clip: uniform duration change, a varying speed curve, or one slowed segment."""

    key = "time_warp"
    regions = ("temporal",)
    DEFAULT_RANGES = {
        "duration": [0.6, 1.65],
        "mode_weights": {"none": 1, "varied": 0, "segment": 0},
        "num_knots": 4,
        "knot_rate": [0.7, 1.43],
        "protect_end": 0.2,
        "segment_width": [0.2, 0.6],
        "segment_factor": [0.6, 1.65],
    }
    DISTRIBUTIONS = {
        "duration": "log_uniform",
        "knot_rate": "log_uniform",
        "segment_factor": "log_uniform",
    }

    def __init__(self, ranges: Optional[Dict[str, Any]] = None, probability: float = 1.0):
        super().__init__(ranges=ranges, probability=probability)
        self._probs = _mode_probs(self.ranges["mode_weights"], WARP_MODES, "mode_weights")
        knots = self.ranges["num_knots"]
        if int(knots) != knots or knots < 2:
            raise ValueError(f"num_knots must be an integer >= 2, got {knots!r}")
        if not 0.0 <= float(self.ranges["protect_end"]) < 1.0:
            raise ValueError("protect_end must be in [0, 1)")
        for k in ("knot_rate", "segment_width", "segment_factor"):
            parse_range(self.ranges[k], self.DISTRIBUTIONS.get(k, "uniform"))

    def _spec(self, name: str):
        return parse_range(self.ranges[name], self.DISTRIBUTIONS.get(name, "uniform"))

    def sample(self, rng: np.random.Generator, clip: Any, ctx: ChainContext) -> Dict[str, Any]:
        if ctx is None or ctx.tier1_draw is None:
            raise ValueError("time warping needs ChainContext.tier1_draw (the uniform duration)")
        mode = WARP_MODES[int(rng.choice(len(WARP_MODES), p=self._probs))]
        knots = self._spec("knot_rate")
        width = float(self._spec("segment_width").draw(rng))
        width = min(max(width, 0.0), 1.0)
        return {
            "mode": mode,
            "duration": float(ctx.tier1_draw.duration),
            "knot_rates": [float(knots.draw(rng)) for _ in range(int(self.ranges["num_knots"]))],
            "start_frac": float(rng.uniform(0.0, 1.0 - width)),
            "width_frac": width,
            "factor": float(self._spec("segment_factor").draw(rng)),
        }

    def times(self, count: int, draw: Dict[str, Any]) -> np.ndarray:
        """Fractional source times for a clip of `count` frames under `draw`."""
        mode, duration = draw["mode"], float(draw["duration"])
        if count < 2 or mode == "none":
            return tier1.duration_times(count, duration)
        if mode == "varied":
            durations = varied_durations(
                count,
                draw["knot_rates"],
                float(self.ranges["protect_end"]),
                duration,
            )
        elif mode == "segment":
            durations = segment_durations(
                count,
                float(draw["start_frac"]),
                float(draw["width_frac"]),
                float(draw["factor"]),
                duration,
            )
        else:
            raise ValueError(f"Unknown time warp mode {mode!r}")
        return times_from_rates(durations)

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        times = self.times(target.frames, draw)
        draw["output_frames"] = int(len(times))
        if len(times) == target.frames and np.array_equal(times, np.arange(target.frames)):
            return target
        return resample_clip(target, times)

    def identity_draw(self) -> Dict[str, Any]:
        return {
            "mode": "none",
            "duration": 1.0,
            "knot_rates": [1.0] * int(self.ranges["num_knots"]),
            "start_frac": 0.0,
            "width_frac": 0.0,
            "factor": 1.0,
        }


@AugmentationRegistry.register("temporal.frame_drop")
class FrameDrop(ClipAugmentation):
    """Drop frames, by a lower frame-rate grid or at random without opening long gaps."""

    key = "frame_drop"
    regions = ("temporal",)
    DEFAULT_RANGES = {
        "mode_weights": {"rate": 0, "random": 1},
        "target_fps": [12, 25],
        "min_fps": 8,
        "p": [0.0, 0.2],
        "max_gap_ms": 125,
        "fill": "none",
    }

    def __init__(self, ranges: Optional[Dict[str, Any]] = None, probability: float = 1.0):
        super().__init__(ranges=ranges, probability=probability)
        self._probs = _mode_probs(self.ranges["mode_weights"], DROP_MODES, "mode_weights")
        if self.ranges["fill"] not in FILL_MODES:
            raise ValueError(f"fill must be one of {FILL_MODES}, got {self.ranges['fill']!r}")
        if float(self.ranges["min_fps"]) <= 0 or float(self.ranges["max_gap_ms"]) <= 0:
            raise ValueError("min_fps and max_gap_ms must be positive")
        parse_range(self.ranges["target_fps"])
        parse_range(self.ranges["p"])

    def sample(self, rng: np.random.Generator, clip: Any, ctx: Any) -> Dict[str, Any]:
        mode = DROP_MODES[int(rng.choice(len(DROP_MODES), p=self._probs))]
        target = float(parse_range(self.ranges["target_fps"]).draw(rng))
        p = float(parse_range(self.ranges["p"]).draw(rng))
        uniforms = rng.random(clip.frames)
        times = np.asarray(clip.times_s, dtype=np.float64)
        if mode == "rate":
            fps = min(max(target, float(self.ranges["min_fps"])), float(clip.fps))
            keep = self._rate_keep(times, fps)
        else:
            keep = self._random_keep(times, p, uniforms, float(self.ranges["max_gap_ms"]) / 1000)
        return {
            "mode": mode,
            "target_fps": target,
            "p": p,
            "keep": [int(i) for i in keep],
            "fill": self.ranges["fill"],
        }

    @staticmethod
    def _rate_keep(times: np.ndarray, fps: float) -> np.ndarray:
        count = len(times)
        if count < 3:
            return np.arange(count)
        grid = np.arange(0.0, times[-1] - times[0] + 1e-9, 1.0 / fps) + times[0]
        index = np.searchsorted(times, grid)
        index = np.clip(index, 1, count - 1)
        nearer_left = (grid - times[index - 1]) <= (times[index] - grid)
        index = np.where(nearer_left, index - 1, index)
        return np.unique(np.concatenate([[0], index, [count - 1]]))

    @staticmethod
    def _random_keep(times: np.ndarray, p: float, uniforms: np.ndarray, max_gap: float):
        count = len(times)
        keep = [0]
        for t in range(1, count - 1):
            if uniforms[t] < p and times[t + 1] - times[keep[-1]] <= max_gap + 1e-9:
                continue
            keep.append(t)
        if count > 1:
            keep.append(count - 1)
        return np.array(keep)

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: Any) -> ClipData:
        keep = draw.get("keep")
        count = target.frames
        if keep is None or len(keep) == count:
            return target
        keep = np.asarray(keep, dtype=int)
        draw["kept_frames"] = int(len(keep))
        fill = draw.get("fill", "none")
        if fill == "none":
            return select_frames(target, keep)
        if fill == "hold":
            previous = keep[np.searchsorted(keep, np.arange(count), side="right") - 1]
            return select_frames(target, previous).replace(times_s=target.times_s.copy())
        if fill == "interpolate":
            kept = select_frames(target, keep)
            fractional = np.interp(
                np.arange(count, dtype=np.float64), keep, np.arange(len(keep), dtype=np.float64)
            )
            return resample_clip(kept, fractional)
        raise ValueError(f"Unknown fill {fill!r}")

    def identity_draw(self) -> Dict[str, Any]:
        return {"mode": "rate", "target_fps": 0.0, "p": 0.0, "keep": None, "fill": "none"}


@AugmentationRegistry.register("temporal.hand_dropout")
class HandDropout(ClipAugmentation):
    """Runs of missing hand detections: NaN hand rows and `hand_valid` false."""

    key = "hand_dropout"
    stage = "outputs"
    regions = ("temporal", "fingers")
    DEFAULT_RANGES = {"start_rate": [0.0, 0.04], "run_length_mean": 1.9, "max_run_frames": 10}

    def __init__(self, ranges: Optional[Dict[str, Any]] = None, probability: float = 1.0):
        super().__init__(ranges=ranges, probability=probability)
        parse_range(self.ranges["start_rate"])
        if float(self.ranges["run_length_mean"]) < 1.0:
            raise ValueError("run_length_mean must be >= 1")
        if int(self.ranges["max_run_frames"]) < 1:
            raise ValueError("max_run_frames must be >= 1")

    def sample(self, rng: np.random.Generator, clip: Any, ctx: Any) -> Dict[str, Any]:
        valid = np.asarray(clip.hand_valid, dtype=bool)
        count = valid.shape[0]
        rate = float(parse_range(self.ranges["start_rate"]).draw(rng))
        starts = rng.random((count, 2))
        lengths = rng.geometric(1.0 / float(self.ranges["run_length_mean"]), size=(count, 2))
        cap = int(self.ranges["max_run_frames"])
        runs: List[List[int]] = []
        for hand in range(2):
            t = 0
            while t < count:
                if valid[t, hand] and starts[t, hand] < rate:
                    length = min(int(lengths[t, hand]), cap)
                    end = t + 1
                    while end < min(t + length, count) and valid[end, hand]:
                        end += 1
                    runs.append([hand, t, end - t])
                    t = end
                else:
                    t += 1
        return {"start_rate": rate, "runs": runs}

    def apply(self, target: Any, draw: Dict[str, Any], ctx: Any) -> Any:
        runs = draw.get("runs") or []
        if not runs:
            return target
        hands_world = target.hands_world.copy()
        hands_image = None if target.hands_image is None else target.hands_image.copy()
        hand_valid = target.hand_valid.copy()
        for hand, start, length in runs:
            sl = slice(int(start), int(start) + int(length))
            hands_world[sl, int(hand)] = np.nan
            if hands_image is not None:
                hands_image[sl, int(hand)] = np.nan
            hand_valid[sl, int(hand)] = False
        draw["masked_hand_frames"] = int(sum(r[2] for r in runs))
        return dataclasses.replace(
            target, hands_world=hands_world, hands_image=hands_image, hand_valid=hand_valid
        )

    def identity_draw(self) -> Dict[str, Any]:
        return {"start_rate": 0.0, "runs": []}
