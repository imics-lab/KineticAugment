"""
Integration tests for the hand extractor: real MediaPipe on the pilot clips.
Skipped when mediapipe or the videos are absent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("torch")  # the extractor imports hand_model, which needs torch

REPO = Path(__file__).resolve().parents[1]
VIDEOS = REPO / "data" / "sign_videos"
sys.path.insert(0, str(REPO / "scripts"))

from extract_hand_landmarks import extract_all  # noqa: E402

from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402

needs_videos = pytest.mark.skipif(
    not (VIDEOS / "01991.mp4").exists() or not (VIDEOS / "01992.mp4").exists(),
    reason="pilot videos absent",
)


@needs_videos
def test_a_clip_does_not_depend_on_the_clip_before_it(tmp_path):
    """
    01991 extracted after 01992 equals 01991 extracted alone.

    The order matters. A `Hands` tracker kept from 01992 changes 01991's hand landmarks, but
    one kept from 01991 leaves 01992's unchanged, so only this order catches a `Hands`
    detector shared between clips. A shared `Holistic` changes `pose_world` in either order.
    """
    extract_all([VIDEOS / "01992.mp4", VIDEOS / "01991.mp4"], tmp_path / "after")
    extract_all([VIDEOS / "01991.mp4"], tmp_path / "alone")
    after = load_hand_clip(tmp_path / "after" / "01991_hands.npz")
    alone = load_hand_clip(tmp_path / "alone" / "01991_hands.npz")
    for name in ("world", "image", "present", "handedness_score", "pose_world"):
        np.testing.assert_array_equal(getattr(after, name), getattr(alone, name))


@needs_videos
def test_hands_are_metric_sided_and_missing_as_nan(tmp_path):
    extract_all([VIDEOS / "01992.mp4"], tmp_path)
    clip = load_hand_clip(tmp_path / "01992_hands.npz")
    frames = len(clip.present)
    assert clip.world.shape == (frames, 2, 21, 3)
    assert clip.image.shape == (frames, 2, 21, 3)
    assert clip.present.any(axis=0).all()                 # both hands appear in this clip
    assert np.isnan(clip.world[~clip.present]).all()      # an absent side is NaN, never zeros
    assert np.isfinite(clip.world[clip.present]).all()
    seen = clip.world[clip.present]
    span = np.linalg.norm(seen[:, 9] - seen[:, 0], axis=-1)
    assert 0.05 < np.median(span) < 0.13                  # wrist to middle MCP, in metres
    assert clip.label_agrees[clip.present].mean() > 0.9   # the label, read as unmirrored
    assert clip.meta["mediapipe"] and clip.meta["source"] == "01992.mp4"
