"""
Per-augmentation measures for the week-4 trial: how far one run moved the outputs.

`measure(base, aug, aligned)` compares two `ClipOutputs`. The frame-by-frame keys
(`wrist_moved_mm_max`, `bone_length_change_mm_max`) need the same frames in both, so they appear
only when `aligned` is true (equal frame counts and no temporal augmentation). The other keys
describe the augmented outputs on their own (or as a ratio of means) and always appear:

    jitter_ratio       mean |second difference| of the pose wrists, aug / base
    face_rigidity_mm   how much more the face-to-head distances vary over the clip than in base
                       (per point: the largest departure from its median distance, aug minus
                       base, floored at 0); the face rides on the head, so this should be ~0
    hand_face_cm_min   closest distance from any valid hand point to any face point
    in_frame_pct       share of image points (arms, valid hands, face) inside [0, 1]^2; present
                       only when the outputs carry image arrays
"""

from typing import Dict

import numpy as np

from .outputs import ClipOutputs

ALIGNED_ONLY = ("wrist_moved_mm_max", "bone_length_change_mm_max")
WRISTS = (15, 16)
BONES = ((11, 13), (13, 15), (12, 14), (14, 16))


def _nanmax(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else 0.0


def _wrist_moved_mm(base: ClipOutputs, aug: ClipOutputs) -> float:
    pose = np.linalg.norm(
        aug.pose_world[:, list(WRISTS)] - base.pose_world[:, list(WRISTS)], axis=-1
    )
    hands = np.linalg.norm(aug.hands_world[:, :, 0] - base.hands_world[:, :, 0], axis=-1)
    return 1000.0 * max(_nanmax(pose), _nanmax(hands))


def _bone_change_mm(base: ClipOutputs, aug: ClipOutputs) -> float:
    worst = 0.0
    for a, b in BONES:
        length_base = np.linalg.norm(base.pose_world[:, a] - base.pose_world[:, b], axis=-1)
        length_aug = np.linalg.norm(aug.pose_world[:, a] - aug.pose_world[:, b], axis=-1)
        worst = max(worst, _nanmax(np.abs(length_aug - length_base)))
    return 1000.0 * worst


def _wrist_jitter(outputs: ClipOutputs) -> float:
    wrists = outputs.pose_world[:, list(WRISTS)]
    if len(wrists) < 3:
        return 0.0
    second = np.abs(wrists[2:] - 2.0 * wrists[1:-1] + wrists[:-2])
    second = second[np.isfinite(second)]
    return float(second.mean()) if len(second) else 0.0


def _face_spread_mm(outputs: ClipOutputs) -> float:
    use = outputs.face_valid & outputs.body_ok
    if not use.any():
        return 0.0
    distance = np.linalg.norm(outputs.face_world[use] - outputs.head_world[use][:, None], axis=-1)
    spread = np.abs(distance - np.nanmedian(distance, axis=0))
    return 1000.0 * _nanmax(spread)


def _hand_face_cm_min(outputs: ClipOutputs) -> float:
    best = np.inf
    for t in np.flatnonzero(outputs.face_valid & outputs.body_ok):
        face = outputs.face_world[t]
        for side in range(2):
            if not outputs.hand_valid[t, side]:
                continue
            hand = outputs.hands_world[t, side]
            gap = np.linalg.norm(hand[:, None] - face[None], axis=-1)
            if np.isfinite(gap).any():
                best = min(best, float(np.nanmin(gap)))
    return float("nan") if not np.isfinite(best) else 100.0 * best


def _in_frame_pct(outputs: ClipOutputs) -> float:
    arms = outputs.pose_image[:, 11:17].reshape(-1, 2)
    hands = outputs.hands_image[outputs.hand_valid].reshape(-1, 2)
    face = outputs.face_image[outputs.face_valid].reshape(-1, 2)
    points = np.concatenate([arms, hands, face])
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return float("nan")
    inside = ((points >= 0.0) & (points <= 1.0)).all(axis=1)
    return 100.0 * float(inside.mean())


def measure(base: ClipOutputs, aug: ClipOutputs, aligned: bool) -> Dict[str, float]:
    """Measures of `aug` against `base`; see the module docstring for the keys."""
    out: Dict[str, float] = {}
    if aligned:
        out["wrist_moved_mm_max"] = _wrist_moved_mm(base, aug)
        out["bone_length_change_mm_max"] = _bone_change_mm(base, aug)
    jitter_base, jitter_aug = _wrist_jitter(base), _wrist_jitter(aug)
    if jitter_base > 1e-12:
        out["jitter_ratio"] = jitter_aug / jitter_base
    else:
        out["jitter_ratio"] = 1.0 if jitter_aug <= 1e-12 else float("inf")
    out["face_rigidity_mm"] = max(0.0, _face_spread_mm(aug) - _face_spread_mm(base))
    out["hand_face_cm_min"] = _hand_face_cm_min(aug)
    if aug.pose_image is not None and aug.hands_image is not None and aug.face_image is not None:
        out["in_frame_pct"] = _in_frame_pct(aug)
    return out
