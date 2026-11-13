"""Tests for FaceClip: pixel conversion and file round trip."""

import numpy as np
import pytest

from kinetic_augment.body_model.face_observations import (
    FaceClip, load_face_clip, save_face_clip,
)


def clip(frames=2):
    landmarks = np.zeros((frames, 478, 3), dtype=np.float32)
    landmarks[:, 0] = [0.5, 0.25, -0.1]
    return FaceClip(
        landmarks=landmarks,
        blendshapes=np.zeros((frames, 52), dtype=np.float32),
        head_matrix=np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)),
        valid=np.ones(frames, dtype=bool),
        timestamp_ms=np.arange(frames, dtype=np.float64) * 40.0,
        width=640, height=480, fps=25.0,
        blendshape_names=tuple(f"b{i}" for i in range(52)),
    )


def test_pixels_scale_z_by_width_like_x():
    np.testing.assert_allclose(clip().pixels()[0, 0], [320.0, 120.0, -64.0])


def test_save_load_round_trip(tmp_path):
    original = clip()
    save_face_clip(tmp_path / "c_face.npz", original)
    loaded = load_face_clip(tmp_path / "c_face.npz")
    np.testing.assert_array_equal(loaded.landmarks, original.landmarks)
    np.testing.assert_array_equal(loaded.valid, original.valid)
    assert loaded.blendshape_names == original.blendshape_names
    assert (loaded.width, loaded.height, loaded.fps) == (640, 480, 25.0)


def test_wrong_schema_is_refused(tmp_path):
    save_face_clip(tmp_path / "c_face.npz", clip())
    data = dict(np.load(tmp_path / "c_face.npz"))
    data["meta"] = np.array('{"schema": 99}')
    np.savez(tmp_path / "bad_face.npz", **data)
    with pytest.raises(ValueError, match="schema"):
        load_face_clip(tmp_path / "bad_face.npz")
