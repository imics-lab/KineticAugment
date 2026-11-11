"""
Extreme draws for the Augmentation Lab: pin a sampled draw to the edge of its range.

The Lab's "extreme" switch shows the most extreme version of a setting. `extreme_draw` takes a
draw the augmentation just sampled and moves its values to the edge of the range they came from
(`edge` is "high" = the positive / upper edge, "low" = the negative / lower edge). Keys the rule
does not name are copied. `make_hook` wraps it as the chain's `draw_hook(name, draw, aug)`.

Rules per entry are in `_RULES`; the signs are high = +, low = -. Shoulder and wrist angles are
norms with an axis, so "low" keeps the angle and flips the axis; the elbow's angle is signed.

Frame drop's `keep` list depends on the clip's timeline, which the chain hook does not see. It is
rebuilt on a uniform timeline of `keep[-1] + 1` frames (a kept list always ends on the last frame)
at the clip's fps, or 30 fps without a clip; pass `clip` to use its real `times_s` instead. Random
mode's per-frame uniforms are not stored in a draw, so they come from a fixed seed.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..augmentations.body import _MIRROR
from ..augmentations.face import BLENDSHAPE_GROUPS
from ..augmentations.profile import Profile, tier1_ranges
from ..augmentations.ranges import parse_range
from ..augmentations.tier1 import Tier1Ranges
from ..augmentations.timing import FrameDrop
from ..evaluation.datasets.face_features import MEDIAPIPE_BLENDSHAPE_NAMES

_DEFAULT_FPS = 30.0
_UNIFORMS_SEED = 0


def _sign(edge: str) -> float:
    if edge not in ("low", "high"):
        raise ValueError(f"edge must be 'low' or 'high', got {edge!r}")
    return 1.0 if edge == "high" else -1.0


def _bound(spec, edge: str) -> float:
    return spec.hi if edge == "high" else spec.lo


def _joint_noise(draw, ranges, tier1, edge, **_):
    s = _sign(edge)
    limit, limit_e = float(ranges["truncate_deg"]), float(ranges["elbow_truncate_deg"])
    mode = ranges["side_mode"]
    out = copy.deepcopy(draw)
    by_key = {(j["side"], j["joint"]): j for j in out["joints"]}
    for j in out["joints"]:
        if j["joint"] == "elbow":
            j["angle_deg"] = s * limit_e
    for joint in ("shoulder", "wrist"):
        left = by_key.get(("left", joint))
        right = by_key.get(("right", joint))
        if left is not None:
            left["angle_deg"] = limit
            left["axis"] = tuple(s * float(a) for a in left["axis"])
        if right is not None:
            right["angle_deg"] = limit
            if left is not None and mode == "mirror":
                right["axis"] = tuple(float(a) for a in np.asarray(left["axis"]) * _MIRROR)
            elif left is not None and mode == "synchronous":
                right["axis"] = tuple(left["axis"])
            else:
                right["axis"] = tuple(s * float(a) for a in right["axis"])
    out["drift"] = None
    return out


def _limb_length(draw, ranges, tier1, edge, **_):
    spec = parse_range(ranges["factor"])
    out = dict(draw)
    out["factors"] = {bone: _bound(spec, edge) for bone in draw["factors"]}
    return out


def _movement_size(draw, ranges, tier1, edge, **_):
    spec = parse_range(ranges["factor"], "log_uniform")
    return {**draw, "factor": _bound(spec, edge)}


def _location_jitter(draw, ranges, tier1, edge, **_):
    s = _sign(edge)
    direction = np.asarray(draw["offset_mm"], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
    offset = s * float(ranges["truncate_mm"]) * direction
    return {**draw, "offset_mm": [float(v) for v in offset]}


def _finger_noise(draw, ranges, tier1, edge, **_):
    value = _sign(edge) * float(ranges["truncate_deg"])
    return {
        **draw,
        "mcp_deg": [value] * len(draw["mcp_deg"]),
        "pip_deg": [value] * len(draw["pip_deg"]),
    }


def _face(draw, ranges, tier1, edge, clip=None, **_):
    s = _sign(edge)
    names = tuple(clip.blendshape_names) if clip is not None else MEDIAPIPE_BLENDSHAPE_NAMES
    offsets = list(draw["offsets"])
    for group, member in BLENDSHAPE_GROUPS.items():
        g = ranges[group]
        if not bool(g["enabled"]):
            continue
        for i, name in enumerate(names[1:]):
            if member(name) and i < len(offsets):
                offsets[i] = s * float(g["truncate"])
    return {**draw, "offsets": offsets, "noise": None}


def _view(draw, ranges, tier1, edge, **_):
    s = _sign(edge)
    return {**draw, "yaw_deg": s * tier1.yaw_deg, "pitch_deg": s * tier1.pitch_deg}


def _scale(draw, ranges, tier1, edge, **_):
    return {**draw, "factor": float(tier1.scale[1] if edge == "high" else tier1.scale[0])}


def _time_warp(draw, ranges, tier1, edge, **_):
    if draw.get("mode") != "none":  # varied / segment have no single edge
        return dict(draw)
    return {**draw, "duration": float(tier1.duration[1] if edge == "high" else tier1.duration[0])}


def _frame_drop(draw, ranges, tier1, edge, clip=None, **_):
    out = dict(draw)
    keep = draw.get("keep")
    if keep is not None and len(keep) > 0:
        count = int(keep[-1]) + 1
    elif clip is not None:
        count = int(clip.frames)
    else:
        return out
    fps = float(clip.fps) if clip is not None else _DEFAULT_FPS
    if clip is not None and len(clip.times_s) == count:
        times = np.asarray(clip.times_s, dtype=np.float64)
    else:
        times = np.arange(count, dtype=np.float64) / fps
    max_gap = float(ranges["max_gap_ms"]) / 1000
    if draw["mode"] == "rate":
        spec = parse_range(ranges["target_fps"])
        out["target_fps"] = target = _bound(spec, edge)  # low = fewer frames kept
        grid_fps = min(max(target, float(ranges["min_fps"])), fps)
        new_keep = FrameDrop._close_gaps(times, FrameDrop._rate_keep(times, grid_fps), max_gap)
    else:
        spec = parse_range(ranges["p"])
        out["p"] = p = spec.lo if edge == "high" else spec.hi  # low = more dropped
        uniforms = np.random.default_rng(_UNIFORMS_SEED).random(count)
        new_keep = FrameDrop._random_keep(times, p, uniforms, max_gap)
    out["keep"] = [int(i) for i in new_keep]
    return out


_RULES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "body.joint_coupled_noise": _joint_noise,
    "body.limb_length_scaling": _limb_length,
    "body.movement_size": _movement_size,
    "body.location_jitter": _location_jitter,
    "fingers.joint_coupled_noise": _finger_noise,
    "face.expression_jitter": _face,
    "scene.view": _view,
    "scene.scale": _scale,
    "temporal.time_warp": _time_warp,
    "temporal.frame_drop": _frame_drop,
}

EXTREME_ENTRIES = frozenset(_RULES)


def extreme_draw(
    name: str,
    draw: Dict[str, Any],
    ranges: Dict[str, Any],
    tier1: Tier1Ranges,
    edge: str,
    clip: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    A new draw for entry `name` with its sampled values pinned to the `edge` of their range.

    Args:
        draw: the augmentation's sampled draw (not modified).
        ranges: the augmentation's resolved ranges (`aug.ranges`).
        tier1: the profile's Tier 1 limits, for the scene entries and the time-warp duration.
        edge: "low" or "high".
        clip: optional; frame drop uses its timeline and face jitter its blendshape names.

    Raises:
        ValueError: `edge` is not "low" or "high", or `name` has no extreme rule.
    """
    _sign(edge)
    if name not in _RULES:
        raise ValueError(f"{name!r} has no extreme draw; known: {sorted(_RULES)}")
    return _RULES[name](draw, ranges, tier1, edge, clip=clip)


def make_hook(
    edge: str, profile: Profile, clip: Optional[Any] = None
) -> Callable[[str, Dict[str, Any], Any], Dict[str, Any]]:
    """The chain's `draw_hook` that pins `EXTREME_ENTRIES` to `edge` and passes the rest through."""
    _sign(edge)
    tier1 = tier1_ranges(profile)

    def hook(name: str, draw: Dict[str, Any], aug: Any) -> Dict[str, Any]:
        if name not in EXTREME_ENTRIES:
            return draw
        return extreme_draw(name, draw, aug.ranges, tier1, edge, clip=clip)

    return hook
