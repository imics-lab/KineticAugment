"""
Tests for scripts/extract_world_landmarks.py.

MediaPipe Holistic in video mode is stateful: it tracks a region of interest and
smooths landmarks across frames. One instance shared across clips carries that
state from the end of one clip into the start of the next, so a clip's stored
landmarks depended on which clip happened to be processed before it -- by a
median 33 mm on one clip's arm landmarks. Per-clip shape fitting reads those
landmarks, so the data must be a function of the clip alone.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mediapipe")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_world_landmarks.py"
# The two smallest sample clips; the second is measurably affected by the first.
CLIPS = [ROOT / "data" / "sign_videos" / "01991.mp4", ROOT / "data" / "sign_videos" / "01992.mp4"]


def load_extractor():
    spec = importlib.util.spec_from_file_location("extract_world_landmarks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not all(clip.exists() for clip in CLIPS), reason="sample videos not present")
def test_a_clip_does_not_inherit_tracker_state_from_the_previous_clip(tmp_path):
    extractor = load_extractor()

    extractor.extract_all(CLIPS, tmp_path / "after_another")
    extractor.extract_all(CLIPS[1:], tmp_path / "alone")

    after_another = np.load(tmp_path / "after_another" / "01992_world.npz")
    alone = np.load(tmp_path / "alone" / "01992_world.npz")
    np.testing.assert_array_equal(after_another["landmarks"], alone["landmarks"])
    np.testing.assert_array_equal(after_another["visibility"], alone["visibility"])
