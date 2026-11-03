"""
Integration tests for the face extractor: real FaceLandmarker on a pilot clip.
Skipped when mediapipe, the model or the video is absent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mediapipe")

from tests.face_assets import HAVE_FACE_LANDMARKER, MEDIAPIPE_DIR  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VIDEO = REPO / "data" / "sign_videos" / "01991.mp4"
HANDS = REPO / "data" / "test_output" / "01991_hands.npz"
sys.path.insert(0, str(REPO / "scripts"))

pytestmark = pytest.mark.skipif(not (HAVE_FACE_LANDMARKER and VIDEO.exists()),
                                reason="face landmarker model or pilot video absent")


@pytest.fixture(scope="module")
def face():
    from extract_face_landmarks import extract_face
    return extract_face(VIDEO, MEDIAPIPE_DIR / "face_landmarker.task")


def test_a_face_is_found_on_most_frames(face):
    assert face.landmarks.shape[1:] == (478, 3)
    assert face.valid.mean() > 0.9
    assert "jawOpen" in face.blendshape_names and len(face.blendshape_names) == 52


def test_frames_line_up_with_the_hand_clip(face):
    if not HANDS.exists():
        pytest.skip("pilot hand clip absent")
    from kinetic_augment.body_model.hand_observations import load_hand_clip
    hands = load_hand_clip(HANDS)
    np.testing.assert_array_equal(face.timestamp_ms, hands.timestamp_ms)


def test_head_matrix_puts_the_face_in_front_of_the_camera(face):
    """OpenGL camera looks down -Z, so translation z is negative; this also pins the layout."""
    tz = face.head_matrix[face.valid, 2, 3]
    assert np.median(tz) < 0
    bottom_row = face.head_matrix[face.valid, 3]
    np.testing.assert_allclose(bottom_row, np.broadcast_to([0, 0, 0, 1], bottom_row.shape),
                               atol=1e-5)
