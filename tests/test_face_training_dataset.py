"""Contract tests for the masked face-feature training adapter."""

import json

import numpy as np
import pytest

from kinetic_augment.evaluation.datasets.face_features import (
    FACE_SUBSET_IDS,
    FaceTrainingSample,
    WLASLFaceDataset,
    collate_face_samples,
    sample_from_face_features,
)


def feature_fixture(times=(0.0, 40.0, 80.0)):
    n = len(times)
    pose = np.zeros((n, 33, 3), dtype=float)
    pose[:, 11, :2] = [0.25, 0.5]
    pose[:, 12, :2] = [0.75, 0.5]
    pose[:, 13:17, :2] = np.arange(4)[None, :, None] * 0.01 + [0.3, 0.6]
    hands = np.ones((n, 2, 21, 3), dtype=float) * 0.5
    face = np.zeros((n, 478, 3), dtype=float)
    face[:, 33] = [0.25, 0.5, 0]
    face[:, 263] = [0.75, 0.5, 0]
    for point in FACE_SUBSET_IDS:
        face[:, point, :2] = [0.5 + point * 1e-4, 0.5]
    face[:, 33, :2] = [0.25, 0.5]
    face[:, 263, :2] = [0.75, 0.5]
    face_local = face.copy()
    features = {
        "frames": np.arange(n) * 2,
        "timestamp_ms": np.asarray(times),
        "width": np.array(640),
        "height": np.array(480),
        "pose_image": pose,
        "pose_visibility": np.ones((n, 33)),
        "hand_image": hands,
        "hand_present": np.ones((n, 2), bool),
        "face_point_valid": np.ones((n, 478), bool),
        "face_point_visible": np.ones((n, 478), bool),
        "face_pixels": face * [640, 480, 640],
        "face_stream_local": face_local,
        "face_attachment_valid": np.ones(n, bool),
        "head_relative_rotation": np.tile(np.eye(3), (n, 1, 1)),
        "reliable": np.ones(n, bool),
        "blendshapes": np.tile(np.arange(52), (n, 1)) / 52,
        "meta": {"schema": 1, "blendshape_names": ["b{}".format(i) for i in range(52)]},
    }
    return features


def test_stream_masks_are_independent_finite_and_use_fixed_face_subset():
    features = feature_fixture()
    features["pose_visibility"][:] = 0
    features["hand_present"][:] = False
    features["face_pixels"][1, 61] = np.nan
    sample = sample_from_face_features(features)
    assert sample.streams["body_hands"].shape == (3, 96)
    assert not sample.masks["body_hands"].any()
    assert sample.masks["face"][0].all()
    assert not sample.masks["face"][1, 0]
    assert np.isfinite(sample.streams["face"]).all()
    np.testing.assert_array_equal(sample.streams["face"][1, :2], [0, 0])
    assert sample.metadata["face_subset_ids"] == list(FACE_SUBSET_IDS)
    assert sample.metadata["head_rotation_order"].startswith("first column")


def test_unavailable_normalization_anchors_mask_only_their_stream():
    features = feature_fixture()
    features["face_point_visible"][:] = False
    sample = sample_from_face_features(features)
    assert not sample.masks["face"].any()
    assert sample.masks["body_hands"].any()
    assert sample.masks["head"].all()
    assert not sample.masks["blendshapes"].any()


def test_head_local_face_requires_attachment_and_blendshape_occlusion_masks_whole_stream():
    features = feature_fixture()
    features["face_attachment_valid"][1] = False
    features["face_point_visible"][2, 61] = False
    sample = sample_from_face_features(features, face_space="head_local")
    assert sample.streams["face"].shape == (3, 78)
    assert sample.masks["face"][0].all()
    assert not sample.masks["face"][1].any()
    assert sample.masks["blendshapes"][1].all()
    assert not sample.masks["blendshapes"][2].any()
    assert sample.masks["blendshapes"][0].all()


def test_bad_schema_and_shape_are_rejected():
    features = feature_fixture()
    features["meta"]["schema"] = 9
    with pytest.raises(ValueError, match="schema"):
        sample_from_face_features(features)
    features = feature_fixture()
    features["hand_image"] = features["hand_image"][:, :, :-1]
    with pytest.raises(ValueError, match="hand_image"):
        sample_from_face_features(features)


def test_select_and_resample_keep_all_streams_on_one_grid_and_guard_gaps():
    features = feature_fixture((0.0, 40.0, 200.0))
    features["pose_image"][:, 13, 0] = [0.1, 0.3, 0.9]
    sample = sample_from_face_features(features)
    selected = sample.select_frames([0, 2])
    np.testing.assert_array_equal(selected.timestamp_ms, [0, 200])
    assert all(len(v) == 2 for v in selected.streams.values())
    resampled = sample.resample([20, 80, 240], max_gap_ms=100)
    assert all(len(v) == 3 for v in resampled.streams.values())
    assert resampled.interpolated.tolist() == [True, False, False]
    assert resampled.masks["face"][0].all()
    assert not resampled.masks["face"][1].any()  # missing span across 40..200 is not filled
    assert not resampled.masks["face"][2].any()  # no extrapolation
    assert resampled.source_frames[0].tolist() == [0, 2]
    chained = resampled.resample([20], max_gap_ms=100)
    assert chained.interpolated.tolist() == [True]
    np.testing.assert_array_equal(chained.source_frames, resampled.source_frames[:1])
    with pytest.raises(ValueError, match="strictly increasing"):
        sample.resample([40, 40])
    with pytest.raises(ValueError, match="strictly increasing"):
        sample.select_frames([2, 1])


def test_head_rotation_resamples_with_so3_slerp_at_180_degrees():
    times = np.array([0.0, 40.0])
    rotations = np.zeros((2, 3, 3))
    rotations[0] = np.eye(3)
    rotations[1] = np.diag([-1.0, -1.0, 1.0])
    sample = FaceTrainingSample(
        streams={"head": np.stack([np.r_[r[:, 0], r[:, 1]] for r in rotations]).astype(np.float32)},
        masks={"head": np.ones((2, 6), bool)},
        timestamp_ms=times,
        source_frames=np.array([[7, 7], [8, 8]]),
        source_time_ms=np.column_stack([times, times]),
        interpolated=np.zeros(2, bool),
    )
    halfway = sample.resample([20]).streams["head"][0]
    first_column, second_column = halfway[:3], halfway[3:]
    np.testing.assert_allclose(first_column, [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(second_column, [-1, 0, 0], atol=1e-6)


def test_collate_pads_masks_and_enforces_channel_layout():
    first = sample_from_face_features(feature_fixture((0, 40)))
    second = sample_from_face_features(feature_fixture((0, 40, 80)))
    x, lengths, labels = collate_face_samples([(first, 2), (second, 1)])
    assert lengths.tolist() == [2, 3]
    assert labels.tolist() == [2, 1]
    assert x["streams"]["face"].shape == (2, 3, 52)
    assert not x["masks"]["face"][0, 2].any()
    mismatch = sample_from_face_features(feature_fixture(), include_blendshapes=False)
    with pytest.raises(ValueError, match="share stream layout"):
        collate_face_samples([(first, 0), (mismatch, 0)])


def test_missing_name_no_face_clip_batches_with_named_clip_without_valid_unnamed_scores():
    from kinetic_augment.evaluation.datasets.face_features import MEDIAPIPE_BLENDSHAPE_NAMES

    ordinary_features = feature_fixture((0, 40))
    ordinary_features["meta"]["blendshape_names"] = list(MEDIAPIPE_BLENDSHAPE_NAMES)
    no_face_features = feature_fixture((0, 40, 80))
    no_face_features["meta"]["blendshape_names"] = []
    no_face_features["face_point_valid"][:] = False
    no_face_features["face_point_visible"][:] = False
    no_face_features["blendshapes"][:] = np.nan

    ordinary = sample_from_face_features(ordinary_features)
    no_face = sample_from_face_features(no_face_features)
    inputs, lengths, labels = collate_face_samples([(ordinary, 0), (no_face, 1)])

    assert no_face.metadata["blendshape_names"] == list(MEDIAPIPE_BLENDSHAPE_NAMES)
    assert no_face.masks["blendshapes"].sum() == 0
    assert inputs["streams"]["blendshapes"].shape == (2, 3, 52)
    assert not inputs["masks"]["blendshapes"][1].any()
    assert lengths.tolist() == [2, 3]
    assert labels.tolist() == [0, 1]
    assert inputs["masks"]["blendshapes"][0].any()


def test_blendshapes_disabled_collation_ignores_absent_blendshape_metadata():
    from kinetic_augment.evaluation.datasets.face_features import MEDIAPIPE_BLENDSHAPE_NAMES

    first_features, second_features = feature_fixture(), feature_fixture()
    first_features["meta"]["blendshape_names"] = list(MEDIAPIPE_BLENDSHAPE_NAMES)
    second_features["meta"]["blendshape_names"] = []
    second_features["face_point_valid"][:] = False
    second_features["face_point_visible"][:] = False
    second_features["blendshapes"][:] = np.nan
    first = sample_from_face_features(first_features, include_blendshapes=False)
    second = sample_from_face_features(second_features, include_blendshapes=False)
    second.metadata["blendshape_names"] = ["different", "unused"]

    inputs, _, _ = collate_face_samples([(first, 0), (second, 1)])
    assert tuple(inputs["streams"]) == ("body_hands", "face", "head")


def test_body_only_collation_accepts_unnamed_face_clip():
    no_face_features = feature_fixture()
    no_face_features["meta"]["blendshape_names"] = []
    no_face_features["face_point_valid"][:] = False
    no_face_features["face_point_visible"][:] = False
    no_face_features["blendshapes"][:] = np.nan
    samples = [
        sample_from_face_features(feature_fixture()),
        sample_from_face_features(no_face_features),
    ]
    body_only = [
        FaceTrainingSample(
            streams={"body_hands": sample.streams["body_hands"]},
            masks={"body_hands": sample.masks["body_hands"]},
            timestamp_ms=sample.timestamp_ms,
            source_frames=sample.source_frames,
            source_time_ms=sample.source_time_ms,
            interpolated=sample.interpolated,
            metadata=sample.metadata,
        )
        for sample in samples
    ]

    inputs, _, _ = collate_face_samples([(body_only[0], 0), (body_only[1], 1)])
    assert tuple(inputs["streams"]) == ("body_hands",)


def test_collate_rejects_incompatible_valid_blendshape_channel_order():
    first_features, second_features = feature_fixture(), feature_fixture()
    second_features["meta"]["blendshape_names"] = list(
        reversed(second_features["meta"]["blendshape_names"])
    )
    first = sample_from_face_features(first_features)
    second = sample_from_face_features(second_features)

    with pytest.raises(ValueError, match="blendshape channel order"):
        collate_face_samples([(first, 0), (second, 1)])


def test_wlasl_uses_annotation_order_and_requested_split_without_leakage(tmp_path, monkeypatch):
    (tmp_path / "features").mkdir()
    annotation = [
        {
            "gloss": "FIRST",
            "instances": [
                {"video_id": "tr1", "split": "train"},
                {"video_id": "te1", "split": "test"},
                {"video_id": "va1", "split": "val"},
            ],
        },
        {"gloss": "SECOND", "instances": [{"video_id": "tr2", "split": "train"}]},
    ]
    (tmp_path / "WLASL_v0.3.json").write_text(json.dumps(annotation), encoding="utf-8")
    for video_id in ("tr1", "tr2", "te1", "va1"):
        (tmp_path / "features" / (video_id + "_face_features.npz")).touch()
    monkeypatch.setattr(
        "kinetic_augment.evaluation.datasets.face_features.load_face_features",
        lambda path: feature_fixture(),
    )
    train = WLASLFaceDataset(tmp_path, "features", subset="wlasl100", split="train")
    assert train.get_class_names()[:2] == ["FIRST", "SECOND"]
    assert [(train[i][0].metadata["video_id"], train[i][1]) for i in range(len(train))] == [
        ("tr1", 0),
        ("tr2", 1),
    ]
    assert all(train[i][0].metadata["source_split"] == "train" for i in range(len(train)))
    test = WLASLFaceDataset(tmp_path, "features", subset="wlasl100", split="test")
    assert test[0][0].metadata["video_id"] == "te1"
    wrong_video = feature_fixture()
    wrong_video["meta"]["video_id"] = "another-video"
    monkeypatch.setattr(
        "kinetic_augment.evaluation.datasets.face_features.load_face_features",
        lambda path: wrong_video,
    )
    with pytest.raises(ValueError, match="does not match annotation"):
        train[0]
    with pytest.raises(ValueError, match="No usable"):
        WLASLFaceDataset(tmp_path, "missing", subset="wlasl100", split="train")
