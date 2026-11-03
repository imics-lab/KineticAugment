"""Versioned face feature archive and stream preflight validation."""

from types import SimpleNamespace

import numpy as np
import pytest

from kinetic_augment.body_model.face_features import (
    FACE_FEATURE_SCHEMA,
    load_face_features,
    save_face_features,
    validate_face_inputs,
)
from kinetic_augment.body_model.face_pipeline import (
    _face_visibility,
    _mask_geometry,
    geometry_valid_mask,
    hand_near_face,
)
from kinetic_augment.body_model.hand_face_geometry import HandFaceGeometry


def streams(frames=2):
    times = np.arange(frames, dtype=np.float64) * 40.0
    hands = SimpleNamespace(
        width=640,
        height=480,
        timestamp_ms=times,
        world=np.zeros((frames, 2, 21, 3)),
        image=np.zeros((frames, 2, 21, 3)),
        present=np.zeros((frames, 2), bool),
        pose_world=np.zeros((frames, 33, 3)),
        pose_image=np.zeros((frames, 33, 3)),
        pose_visibility=np.ones((frames, 33)),
        meta={},
    )
    face = SimpleNamespace(
        width=640,
        height=480,
        timestamp_ms=times,
        landmarks=np.zeros((frames, 478, 3)),
        blendshapes=np.full((frames, 52), 0.75),
        head_matrix=np.zeros((frames, 4, 4)),
        valid=np.ones(frames, bool),
        meta={},
    )
    return hands, face


def test_preflight_rejects_dimensions_timestamps_and_source_mismatch():
    hands, face = streams()
    face.width += 1
    with pytest.raises(ValueError, match="dimensions"):
        validate_face_inputs(hands, face)
    face.width -= 1
    hands.timestamp_ms[1] = hands.timestamp_ms[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_face_inputs(hands, face)
    hands.timestamp_ms[1] = 40.0
    hands.meta["source_sha256"] = "a"
    face.meta["source_sha256"] = "b"
    with pytest.raises(ValueError, match="source hashes"):
        validate_face_inputs(hands, face)


def test_preflight_marks_missing_hashes_allowed():
    hands, face = streams()
    validate_face_inputs(hands, face)


def test_geometry_mask_rejects_missing_fit_unposed_hand_and_unreached_correction():
    both_posed = np.array([True, True])
    none_corrected = np.zeros(2, bool)
    reached = np.ones(2, bool)
    assert not geometry_valid_mask(False, True, both_posed, none_corrected, reached).any()
    assert not geometry_valid_mask(True, False, both_posed, none_corrected, reached).any()
    assert geometry_valid_mask(True, True, [True, False], none_corrected, reached).tolist() == [
        True,
        False,
    ]
    assert geometry_valid_mask(True, True, both_posed, [False, True], [True, False]).tolist() == [
        True,
        False,
    ]
    geometry = HandFaceGeometry(
        np.ones((2, 7)), np.zeros((2, 7), np.int8), np.zeros((2, 7), np.int8), np.ones((2, 7), bool)
    )
    _mask_geometry(geometry, [True, False])
    assert np.isfinite(geometry.distance[0]).all()
    assert np.isnan(geometry.distance[1]).all()
    assert (geometry.region[1] == -1).all()
    assert not geometry.contact[1].any()


def test_hand_near_face_detects_polygon_edges_crossing_box():
    face = np.full((478, 3), np.nan)
    face[:4] = [[100, 100, 0], [200, 100, 0], [200, 200, 0], [100, 200, 0]]
    diamond = np.array([[150, 70], [230, 150], [150, 230], [70, 150]])
    assert hand_near_face(face, [diamond], pad=0)


def test_face_point_visibility_excludes_hand_occlusion():
    face = np.full((478, 3), np.nan)
    face[33] = [0, 0, 0]
    face[263] = [100, 0, 0]
    face[0] = [50, 50, 0]
    face[1] = [100, 100, 0]
    square = np.array([[40, 40], [60, 40], [60, 60], [40, 60]])
    visible = _face_visibility(face, [square])
    assert not visible[0]
    assert visible[1]


def test_archive_round_trip_preserves_legacy_fields_and_validates_schema(tmp_path):
    n = 2
    fits = [
        SimpleNamespace(
            neck_pose=np.zeros(3),
            head_pose=np.zeros(3),
            jaw_pose=np.zeros(3),
            valid=True,
            reliable=True,
            jaw_reliable=True,
            rms_fraction=0.01,
            points_used=12,
            neck_mode="rest",
            ear_residual_m=np.nan,
        )
        for _ in range(n)
    ]
    result = SimpleNamespace(
        head_fits=fits,
        geometry=[None] * n,
        image_geometry=[None] * n,
        frames=np.arange(n),
        head_scale_px_per_m=np.ones(n),
        head_camera_rotation_deg=np.zeros(n),
        wrist_target=np.full((n, 2, 3), np.nan),
        wrist_corrected=np.zeros((n, 2), bool),
        depth_ratio=np.full((n, 2), np.nan),
        wrist_residual_m=np.full((n, 2), np.nan),
        wrist_shift_m=np.full((n, 2), np.nan),
        wrist_reached=np.zeros((n, 2), bool),
        hand_near_face=np.zeros(n, bool),
        body_ok=np.zeros(n, bool),
        betas=np.zeros(300),
        face_landmarks=np.full((n, 478, 3), 0.25),
        face_pixels=np.zeros((n, 478, 3)),
        face_stream_local=np.full((n, 478, 3), np.nan),
        head_transform=np.full((n, 4, 4), np.nan),
        face_pixel_to_world=np.full((n, 4, 4), np.nan),
        face_observed=np.ones(n, bool),
        face_attachment_valid=np.zeros(n, bool),
        face_point_valid=np.ones((n, 478), bool),
        face_point_visible=np.ones((n, 478), bool),
        head_world_rotation=np.tile(np.eye(3), (n, 1, 1)),
        head_relative_rotation=np.tile(np.eye(3), (n, 1, 1)),
        pose_image=np.zeros((n, 33, 3)),
        pose_world=np.zeros((n, 33, 3)),
        pose_visibility=np.ones((n, 33)),
        hand_image=np.zeros((n, 2, 21, 3)),
        hand_world=np.zeros((n, 2, 21, 3)),
        hand_present=np.zeros((n, 2), bool),
        geometry_valid=np.zeros((n, 2), bool),
        contact_valid=np.zeros((n, 2, 7), bool),
        timestamp_ms=np.arange(n) * 40.0,
        depth_corrected=False,
        fov_deg=63.0,
        hands_meta={},
    )
    hands, face = streams(n)
    face.blendshape_names = ("smile",)
    face.fps = 25.0
    face.meta = {"source_sha256": "known"}
    hands.meta = {"source_sha256": "known"}
    path = tmp_path / "features.npz"
    save_face_features(path, result, face, hands.meta)
    loaded = load_face_features(path)
    assert loaded["distance"].shape == (n, 2, 7)
    assert loaded["contact"].shape == (n, 2, 7)
    assert not loaded["contact_valid"].any()
    assert loaded["meta"]["schema"] == FACE_FEATURE_SCHEMA
    assert loaded["meta"]["provenance_complete"]
    assert loaded["meta"]["blendshape_names"] == ["smile"]
    np.testing.assert_array_equal(loaded["face_landmarks"], np.full((n, 478, 3), 0.25))
    np.testing.assert_array_equal(loaded["blendshapes"], np.full((n, 52), 0.75))

    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    broken = tmp_path / "bad.npz"
    for missing_key in ("blendshapes", "fps", "neck_pose", "distance", "image_distance"):
        incomplete = dict(arrays)
        del incomplete[missing_key]
        np.savez(broken, **incomplete)
        with pytest.raises(ValueError, match=missing_key):
            load_face_features(broken)

    for malformed_key, malformed_value in (
        ("blendshapes", np.zeros((n, 51))),
        ("fps", np.array([25.0])),
        ("distance", np.zeros((n, 2, 6))),
    ):
        malformed = dict(arrays)
        malformed[malformed_key] = malformed_value
        np.savez(broken, **malformed)
        with pytest.raises(ValueError, match=malformed_key):
            load_face_features(broken)

    non_object_meta = dict(arrays)
    non_object_meta["meta"] = np.array('[{"schema":1}]')
    np.savez(broken, **non_object_meta)
    with pytest.raises(ValueError, match="JSON object"):
        load_face_features(broken)

    arrays["meta"] = np.array('{"schema":99}')
    np.savez(broken, **arrays)
    with pytest.raises(ValueError, match="schema"):
        load_face_features(broken)
    arrays["meta"] = np.array('{"schema":1}')
    arrays["face_landmarks"] = arrays["face_landmarks"][:, :-1]
    np.savez(broken, **arrays)
    with pytest.raises(ValueError, match="face_landmarks"):
        load_face_features(broken)
