"""Tests for movement-size scaling and location jitter (`augmentations.body`)."""

import json

import numpy as np
import pytest

from kinetic_augment.augmentations.base import AugmentationRegistry
from kinetic_augment.augmentations.body import (
    LocationJitter,
    MovementSizeScaling,
    wrist_paths_torso,
    wrist_world,
)
from kinetic_augment.augmentations.clip import ChainContext
from kinetic_augment.augmentations.ranges import clip_rng
from tests.clip_fixtures import get_skeleton, synthetic_clip
from tests.hand_assets import requires_smplx

pytestmark = requires_smplx


def make_ctx():
    skeleton = get_skeleton()
    return ChainContext(skeleton=skeleton, parents=skeleton.parents, pivot=np.zeros(3))


def bent_clip(frames=12, bend_deg=40.0):
    """
    Clip with every elbow bent at least `bend_deg` (bend_deg + 0..40 degrees over the clip), so
    the exact-value tests have reachable targets while the wrists still move.
    """
    clip = synthetic_clip(frames=frames)
    rotations = clip.rotations.copy()
    bend = np.radians(bend_deg + 40.0 * np.sin(np.linspace(0.0, np.pi, frames)))
    rotations[:, 18, 1] = -bend
    rotations[:, 19, 1] = bend
    return clip.replace(rotations=rotations)


def paths(clip):
    return wrist_paths_torso(clip, make_ctx().parents)[0]


def rms_extent(p):
    ok = np.isfinite(p).all(axis=(1, 2))
    p = p[ok]
    return np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))  # (2,)


def test_registered():
    assert AugmentationRegistry.get("body.movement_size") is MovementSizeScaling
    assert AugmentationRegistry.get("body.location_jitter") is LocationJitter


def test_movement_identity_is_exact():
    clip = synthetic_clip(frames=12)
    aug = MovementSizeScaling()
    for draw in (aug.identity_draw(), {"factor": 1.0}):
        out = aug.apply(clip, draw, make_ctx())
        assert np.array_equal(out.rotations, clip.rotations, equal_nan=True)


def test_location_identity_is_exact():
    clip = synthetic_clip(frames=12)
    aug = LocationJitter()
    out = aug.apply(clip, aug.identity_draw(), make_ctx())
    assert np.array_equal(out.rotations, clip.rotations, equal_nan=True)


def test_movement_scales_path_extent():
    clip = bent_clip()
    draw = {"factor": 1.2}
    out = MovementSizeScaling().apply(clip, draw, make_ctx())
    ratio = rms_extent(paths(out)) / rms_extent(paths(clip))
    assert np.allclose(ratio, 1.2, atol=0.01)
    assert draw["residual_m"] < 1e-3
    assert set(draw["start_end_displacement_mm"]) == {"start", "end"}
    json.dumps(draw)


def test_location_offset_constant():
    clip = bent_clip()
    draw = {"offset_mm": [20.0, -15.0, 10.0]}
    out = LocationJitter().apply(clip, draw, make_ctx())
    delta = paths(out) - paths(clip)  # (T, 2, 3) in the torso frame
    assert np.abs(delta - np.array([0.020, -0.015, 0.010])).max() < 5e-4
    json.dumps(draw)


def test_sampled_offset_within_truncation():
    clip = synthetic_clip(frames=2)
    aug = LocationJitter()
    for seed in range(300):
        draw = aug.sample(clip_rng(seed, "clip", aug.key), clip, make_ctx())
        assert len(draw["offset_mm"]) == 3
        assert np.abs(draw["offset_mm"]).max() <= 25.0
    json.dumps(draw)


def test_movement_factor_in_range():
    clip = synthetic_clip(frames=2)
    aug = MovementSizeScaling()
    for seed in range(300):
        draw = aug.sample(clip_rng(seed, "clip", aug.key), clip, make_ctx())
        assert 0.8 <= draw["factor"] <= 1.2


def test_hand_distance_kept_by_location():
    clip = bent_clip()
    out = LocationJitter().apply(clip, {"offset_mm": [20.0, 15.0, -10.0]}, make_ctx())

    def gap(c):
        w = wrist_world(c, make_ctx().parents)
        return np.linalg.norm(w[:, 0] - w[:, 1], axis=1)

    assert np.abs(gap(out) - gap(clip)).max() < 5e-4


def test_unfitted_frames_untouched():
    clip = bent_clip(frames=8)
    rotations = clip.rotations.copy()
    rotations[3] = np.nan
    body_ok = clip.body_ok.copy()
    body_ok[3] = False
    clip = clip.replace(rotations=rotations, body_ok=body_ok)
    for aug, draw in (
        (MovementSizeScaling(), {"factor": 1.1}),
        (LocationJitter(), {"offset_mm": [10.0, 10.0, 10.0]}),
    ):
        out = aug.apply(clip, draw, make_ctx())
        assert np.isnan(out.rotations[3]).all()
        assert np.isfinite(np.delete(out.rotations, 3, axis=0)).all()
        assert np.array_equal(out.transl, clip.transl)


def test_location_jitter_validates_sigma_and_truncation():
    for bad in ({"sigma_mm": -1.0}, {"truncate_mm": 0.0}, {"sigma_mm": 5.0, "truncate_mm": -1}):
        with pytest.raises(ValueError):
            LocationJitter(bad)
    LocationJitter({"sigma_mm": 0.0, "truncate_mm": 0.0})  # zero strength stays allowed
