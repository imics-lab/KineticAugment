"""
Week-4 finger augmentations on the SMPL-X hand joints.

`FingerCoupledNoise` draws one flexion offset per finger MCP and PIP for the clip (constant over
time). Index, middle, ring and pinky are coupled to their neighbour in that order
(`x_f = c * x_(f-1) + sqrt(1 - c^2) * z_f`); the thumb is independent. The DIP follows the PIP at
`dip_ratio`. `HandshapeJitter` draws a small rotation on every finger joint, larger on fingers
that are not selected (see `handshape.selected_fingers`), and redraws it while it would change the
handshape class (`handshape.features_changed`); after `max_redraws` redraws it gives up and
returns the identity draw with `draw["fallback"] = "no_valid_redraw"`.

Both apply their offsets to every frame with finite hand rotations, on each hand that is valid in
at least one frame, in the convention of `handshape.apply_finger_offsets` (the same offsets move
both hands as mirror images). Draws are plain lists, so they are JSON-serialisable.
"""

from typing import Any, Dict, Optional

import numpy as np

from .base import AugmentationRegistry, ClipAugmentation
from .clip import ChainContext, ClipData
from .handshape import (
    FINGER_SLOTS,
    FINGERS,
    HAND_START,
    apply_finger_offsets,
    feature_signature,
    features_changed,
    handshape_features,
    selected_fingers,
)
from .ranges import truncated_normal

SIDES = ("left", "right")
_NEIGHBOUR_CHAIN = (1, 2, 3, 4)  # index -> middle -> ring -> pinky in FINGERS order


def _check_non_negative(ranges: Dict[str, Any], *keys: str) -> None:
    for k in keys:
        if float(ranges[k]) < 0:
            raise ValueError(f"{k} must be >= 0, got {ranges[k]!r}")


def _valid_hands(clip: ClipData):
    """The sides that are valid in at least one frame."""
    return [s for i, s in enumerate(SIDES) if bool(np.asarray(clip.hand_valid)[:, i].any())]


def _zeros() -> np.ndarray:
    return np.zeros((15, 3))


@AugmentationRegistry.register("fingers.joint_coupled_noise")
class FingerCoupledNoise(ClipAugmentation):
    """Coupled flexion noise on every finger's MCP and PIP, DIP following the PIP."""

    key = "joint_coupled_noise"
    regions = ("fingers",)
    DEFAULT_RANGES = {
        "sigma_deg": 3.75,
        "truncate_deg": 7.5,
        "finger_coupling": 0.5,
        "dip_ratio": 0.667,
    }

    def __init__(self, ranges: Optional[Dict[str, Any]] = None, probability: float = 1.0):
        super().__init__(ranges=ranges, probability=probability)
        _check_non_negative(self.ranges, "sigma_deg", "truncate_deg", "dip_ratio")
        if not 0.0 <= float(self.ranges["finger_coupling"]) <= 1.0:
            raise ValueError("finger_coupling must be in [0, 1]")

    def _chain(self, rng: np.random.Generator) -> np.ndarray:
        """One joint level's offsets for the five fingers (FINGERS order), before truncation."""
        sigma, c = float(self.ranges["sigma_deg"]), float(self.ranges["finger_coupling"])
        z = rng.normal(0.0, sigma, size=5)
        x = z.copy()
        for prev, f in zip((None,) + _NEIGHBOUR_CHAIN[:-1], _NEIGHBOUR_CHAIN):
            if prev is not None:
                x[f] = c * x[prev] + np.sqrt(1.0 - c**2) * z[f]
        return x

    def sample(self, rng: np.random.Generator, clip: ClipData, ctx: Any) -> Dict[str, Any]:
        limit = float(self.ranges["truncate_deg"])
        mcp = np.clip(self._chain(rng), -limit, limit)
        pip = np.clip(self._chain(rng), -limit, limit)
        return {
            "mcp_deg": [float(v) for v in mcp],
            "pip_deg": [float(v) for v in pip],
            "dip_ratio": float(self.ranges["dip_ratio"]),
        }

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        offsets = _zeros()
        ratio = float(draw.get("dip_ratio", self.ranges["dip_ratio"]))
        for f, name in enumerate(FINGERS):
            s0, s1, s2 = FINGER_SLOTS[name]
            mcp, pip = float(draw["mcp_deg"][f]), float(draw["pip_deg"][f])
            offsets[s0, 0], offsets[s1, 0], offsets[s2, 0] = mcp, pip, ratio * pip
        rotations = np.array(target.rotations, dtype=np.float64, copy=True)
        for side in _valid_hands(target):
            rotations = apply_finger_offsets(rotations, target.rest_joints, side, offsets)
        return target.replace(rotations=rotations)

    def identity_draw(self) -> Dict[str, Any]:
        return {
            "mcp_deg": [0.0] * 5,
            "pip_deg": [0.0] * 5,
            "dip_ratio": float(self.ranges["dip_ratio"]),
        }


@AugmentationRegistry.register("fingers.handshape_jitter")
class HandshapeJitter(ClipAugmentation):
    """Per-joint finger jitter that leaves the handshape class (finger states, spread) alone."""

    key = "handshape_jitter"
    regions = ("fingers",)
    DEFAULT_RANGES = {
        "nonselected_sigma_deg": 7.5,
        "selected_sigma_deg": 3.75,
        "selected_flexion_sigma_deg": 2.0,
        "truncate_sigmas": 2.0,
        "max_redraws": 5,
        "feature_margin_deg": 5.0,
    }

    def __init__(self, ranges: Optional[Dict[str, Any]] = None, probability: float = 1.0):
        super().__init__(ranges=ranges, probability=probability)
        _check_non_negative(
            self.ranges,
            "nonselected_sigma_deg",
            "selected_sigma_deg",
            "selected_flexion_sigma_deg",
            "truncate_sigmas",
            "max_redraws",
            "feature_margin_deg",
        )

    def _draw_hand(self, rng: np.random.Generator, selected) -> np.ndarray:
        """(15, 3) offsets in degrees. The thumb always gets the non-selected sigma."""
        r = self.ranges
        trunc = float(r["truncate_sigmas"])
        offsets = _zeros()
        for f, name in enumerate(FINGERS):
            if selected[f] and name != "thumb":
                sigmas = (
                    float(r["selected_flexion_sigma_deg"]),
                    float(r["selected_sigma_deg"]),
                    float(r["selected_sigma_deg"]),
                )
            else:
                sigmas = (float(r["nonselected_sigma_deg"]),) * 3
            for slot in FINGER_SLOTS[name]:
                for axis, sigma in enumerate(sigmas):
                    offsets[slot, axis] = float(truncated_normal(rng, sigma, trunc * sigma))
        return offsets

    def _hand_state(self, clip: ClipData, side: str, ctx: Any):
        """(frame mask, features) of one hand over its valid, finite frames, or None."""
        start = HAND_START[side]
        i = SIDES.index(side)
        finite = np.isfinite(clip.rotations[:, start : start + 15]).all(axis=(1, 2))
        frames = np.asarray(clip.hand_valid)[:, i].astype(bool) & finite
        if not frames.any():
            return None
        feats = handshape_features(clip.rotations[frames], clip.rest_joints, ctx.parents, side)
        return frames, feats

    def sample(self, rng: np.random.Generator, clip: ClipData, ctx: Any) -> Dict[str, Any]:
        margin = float(self.ranges["feature_margin_deg"])
        state = {}
        for side in _valid_hands(clip):
            hand = self._hand_state(clip, side, ctx)
            if hand is not None:
                state[side] = hand
        draw = self.identity_draw()
        if not state:
            return draw
        selected = {s: selected_fingers(f) for s, (_, f) in state.items()}
        symmetric = len(state) == 2 and feature_signature(state["left"][1]) == feature_signature(
            state["right"][1]
        )
        draw["symmetric"] = bool(symmetric)
        draw["selected"] = {s: list(v) for s, v in selected.items()}
        for attempt in range(int(self.ranges["max_redraws"]) + 1):
            if symmetric:
                shared = self._draw_hand(rng, selected["left"])
                offsets = {s: shared for s in state}
            else:
                offsets = {s: self._draw_hand(rng, selected[s]) for s in state}
            ok = True
            for side, (frames, before) in state.items():
                moved = apply_finger_offsets(
                    clip.rotations[frames], clip.rest_joints, side, offsets[side]
                )
                after = handshape_features(moved, clip.rest_joints, ctx.parents, side)
                if features_changed(before, after, margin):
                    ok = False
                    break
            if ok:
                for side in state:
                    draw["offsets_deg"][side] = offsets[side].tolist()
                draw["redraws"] = attempt
                return draw
        draw["fallback"] = "no_valid_redraw"
        draw["redraws"] = int(self.ranges["max_redraws"])
        return draw

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        rotations = np.array(target.rotations, dtype=np.float64, copy=True)
        for side in _valid_hands(target):
            offsets = np.asarray(draw["offsets_deg"][side], dtype=np.float64)
            rotations = apply_finger_offsets(rotations, target.rest_joints, side, offsets)
        return target.replace(rotations=rotations)

    def identity_draw(self) -> Dict[str, Any]:
        zeros = _zeros().tolist()
        return {
            "offsets_deg": {"left": zeros, "right": [list(r) for r in zeros]},
            "symmetric": True,
        }
