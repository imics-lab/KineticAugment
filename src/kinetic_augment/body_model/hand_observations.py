"""
MediaPipe hand observations for the V1.2 hand round trip.

A hand clip holds, for every decoded video frame and each anatomical side
(LEFT = the signer's left, RIGHT = their right), MediaPipe's 21 hand world
landmarks -- metres, camera-aligned axes (x right, y down, z away from the
camera), origin near the knuckles -- and its 21 image landmarks, with the pose
landmarks of the same frame. Missing data is NaN, never zero.

A detection's side comes from geometry: the pose wrist (15 = left, 16 = right)
nearest the detection's own image wrist. Detection order carries no side, and
the legacy detector's handedness label assumes mirrored (selfie) input -- on
unmirrored video it names the other hand -- so the label is kept and compared
(`label_agrees`) but never used to assign.
"""

import json
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

HAND_CLIP_SCHEMA = 1
NUM_POSE_LANDMARKS = 33
LEFT, RIGHT = 0, 1
SIDES = ("left", "right")
POSE_WRISTS = (15, 16)  # the pose's left and right wrists
MAX_WRIST_GAP = 0.10    # fraction of image height
SIDE_MARGIN = 0.03      # fraction of image height
# What the legacy detector's label means on unmirrored video: the opposite hand.
LABEL_SIDE_UNMIRRORED = {"Left": RIGHT, "Right": LEFT}

_ARRAYS = ("world", "image", "present", "handedness_score", "label_agrees", "unassigned",
           "pose_world", "pose_image", "pose_visibility", "timestamp_ms")


@dataclass
class HandClip:
    """One clip's hand observations; T = decoded frames, sides indexed LEFT / RIGHT."""

    world: np.ndarray             # (T, 2, 21, 3) float32 metres; NaN where that side has no hand
    image: np.ndarray             # (T, 2, 21, 3) float32: x / width, y / height, relative z
    present: np.ndarray           # (T, 2) bool: a detection was assigned to that side
    handedness_score: np.ndarray  # (T, 2) float32 detector handedness score; NaN where absent
    label_agrees: np.ndarray      # (T, 2) bool: the label, read as unmirrored, named that side
    unassigned: np.ndarray        # (T,) int8: detections given no side
    pose_world: np.ndarray        # (T, 33, 3) float32 metres, hip-centred; NaN where no pose
    pose_image: np.ndarray        # (T, 33, 3) float32 normalised image coordinates; NaN, no pose
    pose_visibility: np.ndarray   # (T, 33) float32; 0 where no pose
    timestamp_ms: np.ndarray      # (T,) float64, from the video decoder
    width: int
    height: int
    fps: float
    meta: Dict = field(default_factory=dict)  # detector version and settings, source hash


def assign_sides(hand_wrists, pose_wrists, max_gap: float, margin: float) -> np.ndarray:
    """
    Anatomical side of each detected hand, from where its wrist lies in the image.

    Pose wrists take their nearest detections, nearest pair first. A detection stays
    unassigned (-1) when it is farther than `max_gap` from that pose wrist, when it
    is not at least `margin` nearer to it than to the other wrist (hands too close to
    tell apart), or when that side is already taken -- a second detection of one hand.

    Args:
        hand_wrists: (D, 2) image positions of the detections' wrists.
        pose_wrists: (2, 2) image positions of pose wrists 15 (left) and 16 (right);
            NaN where the pose is missing.
        max_gap, margin: distances in the same isotropic units as the positions.

    Returns:
        (D,) int: LEFT, RIGHT or -1.
    """
    hand_wrists = np.asarray(hand_wrists, dtype=np.float64).reshape(-1, 2)
    pose_wrists = np.asarray(pose_wrists, dtype=np.float64).reshape(2, 2)
    distance = np.linalg.norm(hand_wrists[:, None, :] - pose_wrists[None, :, :], axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)

    sides = np.full(len(hand_wrists), -1)
    for flat in np.argsort(distance, axis=None, kind="stable"):
        detection, side = np.unravel_index(flat, distance.shape)
        if sides[detection] != -1 or side in sides or distance[detection, side] > max_gap:
            continue
        if distance[detection, 1 - side] - distance[detection, side] < margin:
            continue
        sides[detection] = side
    return sides


def save_hand_clip(path, clip: HandClip) -> None:
    """Write a clip to .npz, stamping the schema version into its metadata."""
    meta = dict(clip.meta, schema=HAND_CLIP_SCHEMA)
    np.savez_compressed(
        path,
        **{name: getattr(clip, name) for name in _ARRAYS},
        width=np.array(clip.width),
        height=np.array(clip.height),
        fps=np.array(clip.fps),
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )


def load_hand_clip(path) -> HandClip:
    """
    Read a clip written by `save_hand_clip`.

    Raises:
        ValueError: the file's schema is not HAND_CLIP_SCHEMA.
    """
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != HAND_CLIP_SCHEMA:
            raise ValueError(f"{path}: hand clip schema {meta.get('schema')}, "
                             f"expected {HAND_CLIP_SCHEMA}")
        return HandClip(
            **{name: data[name] for name in _ARRAYS},
            width=int(data["width"]), height=int(data["height"]), fps=float(data["fps"]),
            meta=meta,
        )
