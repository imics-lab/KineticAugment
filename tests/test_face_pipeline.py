"""Tests for the clip pipeline: hand-near-face and a real pilot clip end to end."""

from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.body_model.face_pipeline import hand_near_face
from tests.face_assets import MODEL_DIR, requires_face_assets

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data" / "test_output" / "01991"


def face_box(lo=100.0, hi=200.0):
    face = np.full((478, 3), np.nan)
    face[:4] = [[lo, lo, 0], [hi, lo, 0], [hi, hi, 0], [lo, hi, 0]]
    return face


def test_hand_near_face_detects_overlap_and_containment():
    square = lambda c, r: np.array([[c - r, c - r], [c + r, c - r], [c + r, c + r], [c - r, c + r]])
    face = face_box()
    assert hand_near_face(face, [square(150.0, 10.0)], pad=5.0)  # hand inside the face box
    assert hand_near_face(face, [square(150.0, 500.0)], pad=5.0)  # hand covers the whole face
    assert hand_near_face(face, [square(208.0, 4.0)], pad=5.0)  # within the padding
    assert not hand_near_face(face, [square(400.0, 10.0)], pad=5.0)
    assert not hand_near_face(face, [], pad=5.0)


@requires_face_assets
def test_run_clip_on_pilot_frames(tmp_path):
    hands_path, face_path = Path(f"{CLIP}_hands.npz"), Path(f"{CLIP}_face.npz")
    if not (hands_path.exists() and face_path.exists()):
        pytest.skip("pilot hand/face clips absent")
    from kinetic_augment.body_model.face_observations import load_face_clip
    from kinetic_augment.body_model.face_pipeline import (
        load_face_models,
        run_clip,
        save_face_features,
    )
    from kinetic_augment.body_model.face_stream import transform_points
    from kinetic_augment.body_model.hand_observations import load_hand_clip

    hands, face = load_hand_clip(hands_path), load_face_clip(face_path)
    models = load_face_models(MODEL_DIR)
    result = run_clip(hands, face, models, frames=np.arange(0, 20))
    unattached = run_clip(hands, face, models, frames=np.arange(0, 20), attach_face_stream=False)
    assert len(result.head_fits) == 20 and result.rotations.shape == (20, 55, 3)
    np.testing.assert_array_equal(result.rotations, unattached.rotations)
    np.testing.assert_array_equal(result.transl, unattached.transl)
    for attached_geometry, original_geometry in zip(result.geometry, unattached.geometry):
        if attached_geometry is None or original_geometry is None:
            assert attached_geometry is original_geometry
        else:
            np.testing.assert_allclose(
                attached_geometry.distance, original_geometry.distance, equal_nan=True
            )
            np.testing.assert_array_equal(attached_geometry.region, original_geometry.region)
    assert result.face_attachment_valid.any()
    for i in np.flatnonzero(result.face_attachment_valid):
        world = transform_points(result.face_stream_local[i], result.head_transform[i])
        pixels = transform_points(world, np.linalg.inv(result.face_pixel_to_world[i]))
        np.testing.assert_allclose(pixels, result.face_pixels[i], atol=1e-6, rtol=0.0)
    assert result.face_pixels.dtype == np.float64
    valid = [f for f in result.head_fits if f.valid]
    assert len(valid) >= 15
    assert np.median([f.rms_fraction for f in valid]) < 0.15
    save_face_features(tmp_path / "x_face_features.npz", result, face)
    data = np.load(tmp_path / "x_face_features.npz")
    assert data["distance"].shape == (20, 2, 7)
    assert data["blendshapes"].shape == (20, 52)
    for key in ("image_distance", "image_overlap", "image_region", "image_subregion", "image_near"):
        assert data[key].shape == (20, 2, 7), key
    for key in ("head_scale_px_per_m", "head_camera_rotation_deg"):
        assert data[key].shape == (20,), key
    on_valid = np.array([f.valid and (f.rigid_weights > 0).sum() >= 3 for f in result.head_fits])
    assert on_valid.sum() >= 15
    rotation = data["head_camera_rotation_deg"]
    assert np.isfinite(rotation[on_valid]).all()
    assert np.isfinite(data["head_scale_px_per_m"][on_valid]).all()
    absent = np.isnan(data["image_distance"])
    assert (data["image_region"][absent] == -1).all() and not data["image_overlap"][absent].any()
    print("median head_camera_rotation_deg:", np.median(rotation[on_valid]))
