"""Tests for handshape features, coupled finger noise and handshape jitter (week-4 fingers)."""

import json

import numpy as np
import pytest
from scipy.optimize import brentq

from kinetic_augment.augmentations.base import AugmentationRegistry
from kinetic_augment.augmentations.clip import ChainContext
from kinetic_augment.augmentations.fingers import FingerCoupledNoise, HandshapeJitter
from kinetic_augment.augmentations.handshape import (
    EXTENDED_MAX_DEG,
    FINGER_SLOTS,
    FINGERS,
    FLEXED_MIN_DEG,
    HAND_START,
    HandshapeFeatures,
    apply_finger_offsets,
    features_changed,
    flexion_axes,
    handshape_features,
    selected_fingers,
)
from kinetic_augment.augmentations.ranges import clip_rng
from tests.clip_fixtures import get_skeleton, synthetic_clip
from tests.hand_assets import requires_smplx

pytestmark = requires_smplx


def make_ctx():
    skeleton = get_skeleton()
    return ChainContext(skeleton=skeleton, parents=skeleton.parents, pivot=np.zeros(3))


def sample(aug, clip, seed):
    return aug.sample(clip_rng(seed, "clip", aug.key), clip, make_ctx())


def features(clip_or_rot, side):
    skeleton = get_skeleton()
    rot = getattr(clip_or_rot, "rotations", clip_or_rot)
    return handshape_features(rot, skeleton.rest_joints, skeleton.parents, side)


def zero_clip(frames=4):
    clip = synthetic_clip(frames=frames)
    return clip.replace(rotations=np.zeros_like(clip.rotations))


def fake(bend, spread=10.0, thumb=10.0):
    bend = np.broadcast_to(np.asarray(bend, dtype=float).reshape(1, -1), (1, 5)).copy()
    state = np.zeros((1, 5), dtype=int)
    return HandshapeFeatures(bend, state, np.array([spread]), np.array([thumb]))


def test_registered():
    assert AugmentationRegistry.get("fingers.joint_coupled_noise") is FingerCoupledNoise
    assert AugmentationRegistry.get("fingers.handshape_jitter") is HandshapeJitter


def test_features_flat_hand_extended():
    for side in ("left", "right"):
        f = features(zero_clip(), side)
        assert f.bend_deg.shape == (4, 5)
        assert np.all(f.state == 0), f.bend_deg[0]


def test_features_fist_flexed():
    skeleton = get_skeleton()
    for side in ("left", "right"):
        rot = np.zeros((2, 55, 3))
        axes = flexion_axes(skeleton.rest_joints, side)
        for name in FINGERS[1:]:
            for slot in FINGER_SLOTS[name][:2]:
                rot[:, HAND_START[side] + slot] = np.radians(90.0) * axes[slot]
        f = features(rot, side)
        assert np.all(f.state[:, 1:] == 2), f.bend_deg[0]
        assert selected_fingers(f) == (True, False, False, False, False)


def test_flexion_is_the_same_sense_on_both_hands_and_mirrored():
    skeleton = get_skeleton()
    left = flexion_axes(skeleton.rest_joints, "left")
    right = flexion_axes(skeleton.rest_joints, "right")
    assert np.allclose(np.linalg.norm(left, axis=1), 1.0)
    # The hands are mirror images: the right axes are the left axes with (x, -y, -z).
    assert np.all(np.sum(left * right * np.array([1, -1, -1]), axis=1) > 0.98)
    base = features(zero_clip(), "left").bend_deg[0]
    for side in ("left", "right"):
        offsets = np.zeros((15, 3))
        offsets[0, 0] = 20.0  # +20 deg flexion of the index MCP
        rot = apply_finger_offsets(np.zeros((1, 55, 3)), skeleton.rest_joints, side, offsets)
        bend = features(rot, side).bend_deg[0]
        assert bend[1] - base[1] > 15.0, (side, bend, base)


def test_same_offsets_move_both_hands_as_mirror_images():
    skeleton = get_skeleton()
    offsets = np.random.default_rng(3).uniform(-15, 15, (15, 3))
    out = np.zeros((1, 55, 3))
    for side in ("left", "right"):
        out = apply_finger_offsets(out, skeleton.rest_joints, side, offsets)
    left, right = features(out, "left"), features(out, "right")
    rest = features(np.zeros((1, 55, 3)), "left"), features(np.zeros((1, 55, 3)), "right")
    # The rest hands differ by a few degrees, so compare the change from each hand's rest.
    assert np.allclose(
        left.bend_deg - rest[0].bend_deg, right.bend_deg - rest[1].bend_deg, atol=2.5
    )
    assert abs(left.spread_deg[0] - right.spread_deg[0]) < 1.5


def test_features_changed_margin():
    assert not features_changed(fake([0, 0, 0, 0, FLEXED_MIN_DEG - 2]), fake([0, 0, 0, 0, 114]))
    assert features_changed(fake([0, 0, 0, 0, FLEXED_MIN_DEG - 20]), fake([0, 0, 0, 0, 118]))
    assert features_changed(fake([0, 0, 0, 0, 10]), fake([0, 0, 0, 0, EXTENDED_MAX_DEG + 8]))
    assert not features_changed(fake([10] * 5), fake([10] * 5))
    # Spread and thumb classes, with their own thresholds.
    assert features_changed(fake([10] * 5, spread=10), fake([10] * 5, spread=40))
    assert features_changed(fake([10] * 5, thumb=20), fake([10] * 5, thumb=60))
    assert not features_changed(fake([10] * 5, spread=10), fake([10] * 5, spread=29))


def test_selected_fingers_defaults_to_index():
    f = fake([150] * 5)
    f.state[:] = 2
    assert selected_fingers(f) == (False, True, False, False, False)
    f.state[:, 3] = 1
    assert selected_fingers(f) == (False, False, False, True, False)


def test_finger_noise_identity_exact():
    clip = synthetic_clip(frames=8)
    clip.rotations[2] = np.nan
    aug = FingerCoupledNoise()
    out = aug.apply(clip, aug.identity_draw(), make_ctx())
    assert np.array_equal(out.rotations, clip.rotations, equal_nan=True)


def test_finger_noise_dip_follows_pip():
    skeleton = get_skeleton()
    clip = zero_clip(frames=3)
    aug = FingerCoupledNoise()
    out = aug.apply(clip, sample(aug, clip, 4), make_ctx())
    for side in ("left", "right"):
        axes = flexion_axes(skeleton.rest_joints, side)
        for name in FINGERS:
            s0, s1, s2 = FINGER_SLOTS[name]
            pip = out.rotations[0, HAND_START[side] + s1]
            dip = out.rotations[0, HAND_START[side] + s2]
            assert np.linalg.norm(dip) == pytest.approx(0.667 * np.linalg.norm(pip), abs=1e-9)
            assert np.dot(dip, axes[s2]) == pytest.approx(0.667 * np.dot(pip, axes[s1]), abs=1e-6)


def test_finger_noise_same_offsets_on_both_hands():
    clip = zero_clip(frames=2)
    aug = FingerCoupledNoise()
    out = aug.apply(clip, sample(aug, clip, 9), make_ctx())
    for slot in range(15):
        left = np.linalg.norm(out.rotations[0, 25 + slot])
        right = np.linalg.norm(out.rotations[0, 40 + slot])
        assert left == pytest.approx(right, abs=1e-12)


def test_finger_noise_limits():
    clip = synthetic_clip(frames=2)
    aug = FingerCoupledNoise()
    for seed in range(500):
        draw = sample(aug, clip, seed)
        values = np.concatenate([draw["mcp_deg"], draw["pip_deg"]])
        assert np.all(np.abs(values) <= 7.5 + 1e-9)
        assert np.all(np.abs(np.asarray(draw["pip_deg"]) * draw["dip_ratio"]) <= 7.5)
    json.dumps(draw)


def test_finger_noise_neighbour_coupling():
    clip = synthetic_clip(frames=2)
    aug = FingerCoupledNoise(ranges={"finger_coupling": 0.9, "truncate_deg": 100.0})
    draws = [sample(aug, clip, s) for s in range(400)]
    mcp = np.array([d["mcp_deg"] for d in draws])
    assert np.corrcoef(mcp[:, 1], mcp[:, 2])[0, 1] > 0.8  # index-middle
    assert abs(np.corrcoef(mcp[:, 0], mcp[:, 1])[0, 1]) < 0.2  # thumb is independent


def test_finger_noise_skips_nan_frames_and_missing_hand():
    clip = synthetic_clip(frames=4)
    clip.rotations[1] = np.nan
    hand_valid = clip.hand_valid.copy()
    hand_valid[:, 1] = False
    clip = clip.replace(hand_valid=hand_valid)
    aug = FingerCoupledNoise()
    out = aug.apply(clip, sample(aug, clip, 2), make_ctx())
    assert np.isnan(out.rotations[1]).all()
    assert np.array_equal(out.rotations[:, 40:55], clip.rotations[:, 40:55], equal_nan=True)
    assert not np.array_equal(out.rotations[0, 25:40], clip.rotations[0, 25:40])


def test_handshape_jitter_identity_exact():
    clip = synthetic_clip(frames=6)
    aug = HandshapeJitter()
    out = aug.apply(clip, aug.identity_draw(), make_ctx())
    assert np.array_equal(out.rotations, clip.rotations, equal_nan=True)


def test_handshape_jitter_preserves_features():
    clip = synthetic_clip(frames=10)
    aug = HandshapeJitter()
    used = 0
    for seed in range(50):
        draw = sample(aug, clip, seed)
        json.dumps(draw)
        if draw.get("fallback"):
            assert draw["fallback"] == "no_valid_redraw"
            continue
        used += 1
        out = aug.apply(clip, dict(draw), make_ctx())
        for side in ("left", "right"):
            assert not features_changed(features(clip, side), features(out, side))
    assert used >= 25


def test_handshape_jitter_symmetric_when_hands_match():
    clip = synthetic_clip(frames=6)
    draw = sample(HandshapeJitter(), clip, 1)
    assert draw["symmetric"] is True
    assert draw["offsets_deg"]["left"] == draw["offsets_deg"]["right"]


def test_handshape_jitter_independent_when_hands_differ():
    clip = synthetic_clip(frames=6)
    rot = clip.rotations.copy()
    rot[:, 40 + 3, 2] = np.radians(100.0)  # right middle MCP strongly flexed
    clip = clip.replace(rotations=rot)
    draw = sample(HandshapeJitter(), clip, 1)
    assert draw["symmetric"] is False


def test_handshape_jitter_fallback():
    skeleton = get_skeleton()
    base = zero_clip(frames=3)

    def posed(angle):
        rot = base.rotations.copy()
        for side, start in (("left", 25), ("right", 40)):
            axes = flexion_axes(skeleton.rest_joints, side)
            for slot in (0, 3, 6, 9):
                rot[:, start + slot] = np.radians(angle) * axes[slot]
        return rot

    angle = brentq(lambda a: features(posed(a), "left").bend_deg[0, 1] - FLEXED_MIN_DEG, 20, 170)
    clip = base.replace(rotations=posed(angle))
    aug = HandshapeJitter(
        ranges={
            "nonselected_sigma_deg": 300.0,
            "selected_sigma_deg": 300.0,
            "feature_margin_deg": 0.0,
        }
    )
    draw = sample(aug, clip, 0)
    assert draw["fallback"] == "no_valid_redraw"
    out = aug.apply(clip, draw, make_ctx())
    assert np.array_equal(out.rotations, clip.rotations, equal_nan=True)


def test_missing_hand_skipped():
    clip = synthetic_clip(frames=6)
    hand_valid = clip.hand_valid.copy()
    hand_valid[:, 1] = False
    clip = clip.replace(hand_valid=hand_valid)
    aug = HandshapeJitter()
    draw = sample(aug, clip, 5)
    assert not draw.get("fallback")
    out = aug.apply(clip, draw, make_ctx())
    assert np.array_equal(out.rotations[:, 40:55], clip.rotations[:, 40:55])
    assert not np.array_equal(out.rotations[:, 25:40], clip.rotations[:, 25:40])


def test_unknown_range_key_rejected():
    with pytest.raises(ValueError):
        FingerCoupledNoise(ranges={"nope": 1.0})
    with pytest.raises(ValueError):
        HandshapeJitter(ranges={"nope": 1.0})


def test_handshape_jitter_rejects_zero_truncation():
    with pytest.raises(ValueError, match="truncate_sigmas"):
        HandshapeJitter({"truncate_sigmas": 0.0})
    HandshapeJitter(
        {
            "truncate_sigmas": 0.0,
            "nonselected_sigma_deg": 0.0,
            "selected_sigma_deg": 0.0,
            "selected_flexion_sigma_deg": 0.0,
        }
    )
