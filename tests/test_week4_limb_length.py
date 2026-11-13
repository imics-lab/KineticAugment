"""Tests for limb length scaling with wrist retargeting (`augmentations.body`)."""

import json

import numpy as np
import pytest

from kinetic_augment.augmentations.base import AugmentationRegistry
from kinetic_augment.augmentations.body import LimbLengthScaling, scale_bone
from kinetic_augment.augmentations.clip import ChainContext
from kinetic_augment.augmentations.outputs import build_outputs
from kinetic_augment.augmentations.ranges import clip_rng
from tests.clip_fixtures import get_skeleton, synthetic_clip
from tests.hand_assets import requires_smplx

pytestmark = requires_smplx

ARMS = {"left": (16, 18, 20), "right": (17, 19, 21)}


def make_ctx():
    skeleton = get_skeleton()
    return ChainContext(skeleton=skeleton, parents=skeleton.parents, pivot=np.zeros(3))


def length(rest, a, b):
    return float(np.linalg.norm(rest[a] - rest[b]))


def wrists(clip):
    pose = build_outputs(clip, get_skeleton()).pose_world
    return pose[:, 15:17]


def bent_clip(frames=12, bend_deg=40.0):
    """Synthetic clip with every elbow bent `bend_deg`, so shortened arms can still reach."""
    clip = synthetic_clip(frames=frames)
    rotations = clip.rotations.copy()
    rotations[:, 18, 1] = -np.radians(bend_deg)
    rotations[:, 19, 1] = np.radians(bend_deg)
    return clip.replace(rotations=rotations)


def straight_clip(frames=4):
    clip = synthetic_clip(frames=frames)
    rotations = clip.rotations.copy()
    rotations[:, 18:20] = 0.0
    return clip.replace(rotations=rotations)


def draw_of(factors):
    return {"factors": dict(factors)}


def test_registered():
    assert AugmentationRegistry.get("body.limb_length_scaling") is LimbLengthScaling


def test_identity_draw_is_exact():
    clip = synthetic_clip(frames=12)
    aug = LimbLengthScaling()
    for draw in (aug.identity_draw(), draw_of({"upper_arm": 1.0, "forearm": 1.0})):
        out = aug.apply(clip, draw, make_ctx())
        assert np.array_equal(out.rotations, clip.rotations, equal_nan=True)
        assert np.array_equal(out.rest_joints, clip.rest_joints)


def test_scale_bone_moves_descendants():
    skeleton = get_skeleton()
    rest = skeleton.rest_joints
    out = scale_bone(rest, skeleton.parents, 20, 1.1)
    delta = 0.1 * (rest[20] - rest[18])
    assert np.allclose(out[20], rest[20] + delta, atol=1e-12)
    assert np.allclose(out[25], rest[25] + delta, atol=1e-12)  # a finger joint follows
    assert np.array_equal(out[18], rest[18])
    assert np.array_equal(out[21], rest[21])  # the other arm is untouched


def test_bone_lengths_scaled():
    clip = synthetic_clip(frames=6)
    out = LimbLengthScaling().apply(clip, draw_of({"forearm": 1.07}), make_ctx())
    for _, elbow, wrist in ARMS.values():
        assert length(out.rest_joints, elbow, wrist) == pytest.approx(
            1.07 * length(clip.rest_joints, elbow, wrist), abs=1e-9
        )
    for shoulder, elbow, _ in ARMS.values():
        assert length(out.rest_joints, shoulder, elbow) == pytest.approx(
            length(clip.rest_joints, shoulder, elbow), abs=1e-9
        )


def test_hand_scales_fingers_about_wrist():
    clip = synthetic_clip(frames=2)
    out = LimbLengthScaling().apply(clip, draw_of({"hand": 1.05}), make_ctx())
    rest, new = clip.rest_joints, out.rest_joints
    assert np.allclose(new[20], rest[20])
    assert np.allclose(new[25:40] - new[20], 1.05 * (rest[25:40] - rest[20]), atol=1e-12)
    assert np.allclose(new[40:55] - new[21], 1.05 * (rest[40:55] - rest[21]), atol=1e-12)


def test_retarget_keeps_wrists():
    clip = synthetic_clip(frames=30)
    aug = LimbLengthScaling()
    draw = draw_of({"upper_arm": 1.07, "forearm": 1.07})
    out = aug.apply(clip, draw, make_ctx())
    assert out.rest_joints is not clip.rest_joints
    assert np.abs(wrists(out) - wrists(clip)).max() < 1e-3
    assert draw["residual_m"] < 1e-3


def test_retarget_keeps_wrists_when_shortening():
    # Shortening is only reachable with a bent elbow; every frame here is bent 40 degrees.
    clip = bent_clip()
    out = LimbLengthScaling().apply(clip, draw_of({"upper_arm": 0.95, "forearm": 0.95}), make_ctx())
    assert np.abs(wrists(out) - wrists(clip)).max() < 1e-3


def test_no_retarget_moves_wrists():
    clip = synthetic_clip(frames=30)
    aug = LimbLengthScaling(ranges={"retarget": False})
    out = aug.apply(clip, draw_of({"upper_arm": 1.07, "forearm": 1.07}), make_ctx())
    assert np.array_equal(out.rotations, clip.rotations, equal_nan=True)
    assert np.abs(wrists(out) - wrists(clip)).max() > 5e-3


def test_nan_frames_are_skipped():
    clip = synthetic_clip(frames=6)
    rotations = clip.rotations.copy()
    rotations[2] = np.nan
    body_ok = clip.body_ok.copy()
    body_ok[2] = False
    clip = clip.replace(rotations=rotations, body_ok=body_ok)
    out = LimbLengthScaling().apply(clip, draw_of({"forearm": 1.07}), make_ctx())
    assert np.isnan(out.rotations[2]).all()
    assert np.isfinite(np.delete(out.rotations, 2, axis=0)).all()


def test_unreachable_records_residual():
    clip = straight_clip()
    draw = draw_of({"upper_arm": 0.93, "forearm": 0.93})
    out = LimbLengthScaling().apply(clip, draw, make_ctx())
    assert np.isfinite(out.rotations).all()
    assert draw["residual_m"] > 0
    json.dumps(draw)


def test_draws_in_range():
    clip = synthetic_clip(frames=2)
    aug = LimbLengthScaling(ranges={"bones": ["upper_arm", "forearm", "hand"]})
    for seed in range(500):
        draw = aug.sample(clip_rng(seed, "clip", aug.key), clip, make_ctx())
        assert set(draw["factors"]) == {"upper_arm", "forearm", "hand"}
        for f in draw["factors"].values():
            assert isinstance(f, float) and 0.93 <= f <= 1.07
    json.dumps(draw)


def test_unknown_bone_raises():
    clip = synthetic_clip(frames=2)
    aug = LimbLengthScaling(ranges={"bones": ["thigh"]})
    with pytest.raises(ValueError):
        aug.sample(clip_rng(0, "clip", aug.key), clip, make_ctx())
