"""
Shared clips for the week-4 augmentation tests.

`synthetic_clip` builds a small `ClipData` on the real SMPL-X skeleton with a known motion, and
`real_clip` runs the round trip on the pilot clip `data/test_output/01991` (slow; cached).
Both need the SMPL-X assets; use them under `tests.hand_assets.requires_smplx`.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from kinetic_augment.augmentations.clip import ClipData, SceneTransform, clip_from_result
from kinetic_augment.augmentations.tier1 import WeakCamera
from tests.hand_assets import MODEL_DIR

REPO = Path(__file__).resolve().parents[1]
REAL_CLIP = REPO / "data" / "test_output" / "01991"

# MediaPipe FaceLandmarker's 52 blendshape names, in its output order.
# fmt: off
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)
# fmt: on

# The MCP joint (first joint) of each finger, as offsets into a 15-joint SMPL-X hand block.
MCP_SLOTS = (0, 3, 6, 9, 12)
LEFT_HAND_START, RIGHT_HAND_START = 25, 40

_cache = {}


def get_skeleton():
    """The neutral SMPL-X skeleton (300 betas), loaded once per test session."""
    if "skeleton" not in _cache:
        from kinetic_augment.body_model.skeleton import SMPLXSkeleton

        _cache["skeleton"] = SMPLXSkeleton(MODEL_DIR)
    return _cache["skeleton"]


def synthetic_clip(frames: int = 30, seed: int = 0) -> ClipData:
    """
    A synthetic clip on the zero-shape SMPL-X skeleton at 30 fps.

    Motion: every rotation zero except a 0-40 degree sinusoidal elbow bend on joints 18/19
    (about the local y axis, forearms swinging forward) and a constant 20 degree MCP flexion
    about the local z axis on the first joint of every finger of both hands (slots 0, 3, 6,
    9, 12 of joints 25-39 and 40-54; for the thumb that slot is its CMC). The left side turns
    by the negative angle and the right by the positive one, so both sides bend the same way.
    `face_local` is 478 uniform random points within +-8 cm per frame, blendshapes are uniform
    in [0, 1], every validity mask is true, and the weak camera has a random per-frame offset.
    """
    rng = np.random.default_rng(seed)
    skeleton = get_skeleton()
    t = np.arange(frames)
    bend = np.radians(20.0 * (1.0 - np.cos(2.0 * np.pi * t / max(frames, 1))))  # 0..40 deg
    rotations = np.zeros((frames, 55, 3))
    rotations[:, 18, 1] = -bend
    rotations[:, 19, 1] = bend
    for slot in MCP_SLOTS:
        rotations[:, LEFT_HAND_START + slot, 2] = -np.radians(20.0)
        rotations[:, RIGHT_HAND_START + slot, 2] = np.radians(20.0)
    camera = WeakCamera((0.6, 0.6), 0.5 + 0.02 * rng.standard_normal((frames, 2)), 0.0)
    return ClipData(
        clip_id="synthetic",
        rotations=rotations,
        transl=np.zeros((frames, 3)),
        betas=np.zeros(300),
        rest_joints=skeleton.rest_joints.copy(),
        face_local=rng.uniform(-0.08, 0.08, (frames, 478, 3)),
        face_valid=np.ones(frames, dtype=bool),
        blendshapes=rng.uniform(0.0, 1.0, (frames, 52)),
        blendshape_names=BLENDSHAPE_NAMES,
        body_ok=np.ones(frames, dtype=bool),
        hand_valid=np.ones((frames, 2), dtype=bool),
        times_s=t / 30.0,
        fps=30.0,
        camera=camera,
        scene=SceneTransform(),
    )


def signing_span(hands) -> np.ndarray:
    """First to last frame in which pose 13, 14, 15 and 16 all have visibility >= 0.5."""
    pose = np.asarray(hands.pose_world)
    ok = np.isfinite(pose).all(axis=-1) & (np.asarray(hands.pose_visibility) >= 0.5)
    both = np.flatnonzero(ok[:, [13, 14, 15, 16]].all(axis=1))
    if not len(both):
        pytest.skip("pilot clip has no frame with both arms visible")
    return np.arange(both[0], both[-1] + 1)


def real_clip() -> Tuple[ClipData, object, object]:
    """
    The pilot clip 01991's signing span through `run_clip`: (ClipData, HandClip, FaceClip).

    Slow (it fits every frame), so the result is cached for the test session. Skips when the
    pilot npz files or the face assets are absent.
    """
    if "real" in _cache:
        return _cache["real"]
    hands_path, face_path = Path(f"{REAL_CLIP}_hands.npz"), Path(f"{REAL_CLIP}_face.npz")
    if not (hands_path.exists() and face_path.exists()):
        pytest.skip("pilot hand/face clips absent")
    from tests.face_assets import HAVE_FACE_ASSETS

    if not HAVE_FACE_ASSETS:
        pytest.skip("FLAME / embedding / SMPL-X assets absent")
    from kinetic_augment.body_model.face_observations import load_face_clip
    from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip
    from kinetic_augment.body_model.hand_observations import load_hand_clip

    hands, face = load_hand_clip(hands_path), load_face_clip(face_path)
    models = load_face_models(MODEL_DIR)
    result = run_clip(hands, face, models, frames=signing_span(hands))
    clip = clip_from_result(REAL_CLIP.name, result, hands, face, models.skeleton)
    _cache["real"] = (clip, hands, face)
    return _cache["real"]
