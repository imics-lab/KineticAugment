"""Tests for the face expression jitter (groups, expression map, draws, fallbacks)."""

import numpy as np
import pytest

from kinetic_augment.augmentations.face import (
    BLENDSHAPE_GROUPS,
    ExpressionJitter,
    fit_expression_map,
)
from kinetic_augment.augmentations.clip import ClipData
from kinetic_augment.body_model.face_correspondence import RIGID
from tests.clip_fixtures import synthetic_clip
from tests.hand_assets import requires_smplx

NAMES = (
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
)  # fmt: skip


class _Clip:
    """Minimal ClipData stand-in with the face fields."""

    def __init__(self, face_local, face_valid, blendshapes, fps=30.0):
        self.face_local = face_local
        self.face_valid = face_valid
        self.blendshapes = blendshapes
        self.blendshape_names = NAMES
        self.fps = fps

    def replace(self, **changes):
        out = _Clip(self.face_local, self.face_valid, self.blendshapes, self.fps)
        for k, v in changes.items():
            setattr(out, k, v)
        return out


def _clip(frames=60, seed=0, valid=None):
    rng = np.random.default_rng(seed)
    face = rng.normal(size=(frames, 478, 3))
    scores = rng.uniform(0.2, 0.8, size=(frames, 52))
    ok = np.ones(frames, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    face[~ok] = np.nan
    scores[~ok] = np.nan
    return _Clip(face, ok, scores)


def _same(a, b):
    return np.array_equal(a, b, equal_nan=True)


def test_groups_partition_names():
    for name in NAMES[1:]:
        hits = [g for g, member in BLENDSHAPE_GROUPS.items() if member(name)]
        assert len(hits) == 1, (name, hits)
    assert not any(member("_neutral") for member in BLENDSHAPE_GROUPS.values())
    assert BLENDSHAPE_GROUPS["brows"]("browInnerUp")


def test_identity_exact():
    aug = ExpressionJitter()
    clip = _clip()
    out = aug.apply(clip, aug.identity_draw(), None)
    assert _same(out.face_local, clip.face_local)
    assert _same(out.blendshapes, clip.blendshapes)


def test_mouth_off_by_default():
    aug = ExpressionJitter()
    clip = _clip()
    draw = aug.sample(np.random.default_rng(1), clip, None)
    out = aug.apply(clip, draw, None)
    mouth = [i for i, n in enumerate(NAMES) if BLENDSHAPE_GROUPS["mouth_jaw"](n)]
    assert _same(out.blendshapes[:, mouth], clip.blendshapes[:, mouth])
    assert not _same(out.blendshapes, clip.blendshapes)


def test_scores_stay_in_unit_interval():
    aug = ExpressionJitter(
        ranges={"mouth_jaw": {"enabled": True}, "frame_noise": 0.2}, probability=1.0
    )
    clip = _clip(frames=40)
    clip.blendshapes[:5] = 0.0
    clip.blendshapes[5:10] = 1.0
    for seed in range(200):
        draw = aug.sample(np.random.default_rng(seed), clip, None)
        out = aug.apply(clip, draw, None)
        assert out.blendshapes.min() >= 0.0 and out.blendshapes.max() <= 1.0


def test_rigid_points_fixed():
    aug = ExpressionJitter(ranges={"mouth_jaw": {"enabled": True}})
    clip = _clip()
    draw = aug.sample(np.random.default_rng(3), clip, None)
    out = aug.apply(clip, draw, None)
    idx = list(RIGID)
    assert _same(out.face_local[:, idx], clip.face_local[:, idx])
    assert not _same(out.face_local, clip.face_local)


def test_map_recovers_synthetic_basis():
    rng = np.random.default_rng(0)
    w_true = rng.normal(size=(51, 478 * 3)) * 0.1
    median = rng.normal(size=(478, 3))
    scores = np.zeros((400, 52))
    scores[:, 1:] = rng.uniform(0, 1, size=(400, 51))
    face = median + ((scores[:, 1:] - scores[:, 1:].mean(0)) @ w_true).reshape(-1, 478, 3)
    valid = np.ones(400, dtype=bool)
    w, _ = fit_expression_map(face[:300], scores[:300], valid[:300])
    held = (scores[300:, 1:] - scores[:300, 1:].mean(0)) @ w_true
    pred = (scores[300:, 1:] - scores[:300, 1:].mean(0)) @ w
    keep = np.ones(478 * 3, dtype=bool)
    for p in RIGID:
        keep[3 * p : 3 * p + 3] = False
    rel = np.linalg.norm(pred[:, keep] - held[:, keep]) / np.linalg.norm(held[:, keep])
    assert rel < 0.05
    assert not w[:, ~keep].any()


def test_no_face_fallback():
    aug = ExpressionJitter()
    clip = _clip(frames=40, valid=[False] * 40)
    draw = aug.sample(np.random.default_rng(0), clip, None)
    out = aug.apply(clip, draw, None)
    assert draw["fallback"] == "too_few_face_frames"
    assert _same(out.face_local, clip.face_local)
    assert _same(out.blendshapes, clip.blendshapes)


def test_too_few_frames_changes_scores_only():
    aug = ExpressionJitter()
    clip = _clip(frames=40, valid=[True] * 10 + [False] * 30)
    draw = aug.sample(np.random.default_rng(0), clip, None)
    out = aug.apply(clip, draw, None)
    assert draw["fallback"] == "too_few_face_frames"
    assert _same(out.face_local, clip.face_local)
    assert not _same(out.blendshapes, clip.blendshapes)
    assert np.isnan(out.blendshapes[10:]).all()


def test_invalid_frames_untouched():
    aug = ExpressionJitter()
    clip = _clip(frames=60, valid=[True] * 40 + [False] * 20)
    draw = aug.sample(np.random.default_rng(0), clip, None)
    out = aug.apply(clip, draw, None)
    assert "fallback" not in draw
    assert np.isnan(out.face_local[40:]).all()
    assert not _same(out.face_local[:40], clip.face_local[:40])


def test_frame_noise_varies_over_time_and_validates_fps():
    aug = ExpressionJitter(ranges={"frame_noise": 0.05})
    clip = _clip(frames=60)
    draw = aug.sample(np.random.default_rng(0), clip, None)
    noise = np.asarray(draw["noise"])
    assert noise.shape == (60, 51)
    assert noise.std(axis=0).max() > 0
    mouth = [i for i, n in enumerate(NAMES[1:]) if BLENDSHAPE_GROUPS["mouth_jaw"](n)]
    assert not noise[:, mouth].any()


@requires_smplx
def test_frame_noise_fps_guard_real_clip():
    aug = ExpressionJitter(ranges={"frame_noise": 0.05})
    clip = synthetic_clip(frames=40).replace(fps=4.0)
    with pytest.raises(ValueError, match="fps"):
        aug.sample(np.random.default_rng(0), clip, None)


@requires_smplx
def test_real_clip_round_trip():
    aug = ExpressionJitter(ranges={"mouth_jaw": {"enabled": True}})
    clip = synthetic_clip(frames=40)
    before = (clip.face_local.copy(), clip.blendshapes.copy(), clip.face_valid.copy())
    draw = aug.sample(np.random.default_rng(5), clip, None)
    out = aug.apply(clip, draw, None)
    assert isinstance(out, ClipData)
    assert _same(clip.face_local, before[0]) and _same(clip.blendshapes, before[1])
    assert _same(clip.face_valid, before[2])
    assert out.face_local.shape == clip.face_local.shape
    assert out.blendshapes.shape == clip.blendshapes.shape
    idx = list(RIGID)
    assert _same(out.face_local[:, idx], clip.face_local[:, idx])
    assert not _same(out.face_local, clip.face_local)
    same = aug.apply(clip, aug.identity_draw(), None)
    assert isinstance(same, ClipData)
    assert _same(same.face_local, clip.face_local) and _same(same.blendshapes, clip.blendshapes)


def test_noise_frame_count_mismatch_raises():
    aug = ExpressionJitter(ranges={"frame_noise": 0.05})
    clip = _clip(frames=60)
    draw = aug.sample(np.random.default_rng(0), clip, None)
    with pytest.raises(ValueError, match="frames"):
        aug.apply(_clip(frames=50), draw, None)


def test_unknown_group_keys_raise():
    with pytest.raises(ValueError):
        ExpressionJitter(ranges={"nose": {"enabled": True}})
    with pytest.raises(ValueError):
        ExpressionJitter(ranges={"brows": {"scale": 1.0}})
