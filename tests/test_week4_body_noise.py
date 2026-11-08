"""Tests for coupled arm-joint noise and the Tier 1 arm-joint wrapper (`augmentations.body`)."""

import dataclasses
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.augmentations.arm_ik import world_pose
from kinetic_augment.augmentations.base import AugmentationRegistry
from kinetic_augment.augmentations.body import ArmCoupledNoise, Tier1ArmJoints
from kinetic_augment.augmentations.clip import ChainContext
from kinetic_augment.augmentations.ranges import clip_rng
from kinetic_augment.augmentations.tier1 import ArmJointDraw, Tier1Draw
from tests.clip_fixtures import get_skeleton, synthetic_clip
from tests.hand_assets import requires_smplx

pytestmark = requires_smplx


def make_ctx(tier1_draw=None):
    skeleton = get_skeleton()
    return ChainContext(
        skeleton=skeleton, parents=skeleton.parents, pivot=np.zeros(3), tier1_draw=tier1_draw
    )


def sample(aug, clip, seed, ctx=None):
    return aug.sample(clip_rng(seed, "clip", aug.key), clip, ctx or make_ctx())


def test_registered():
    assert AugmentationRegistry.get("body.joint_coupled_noise") is ArmCoupledNoise
    assert AugmentationRegistry.get("body.tier1_arm_joints") is Tier1ArmJoints


def test_identity_draw_is_exact():
    clip = synthetic_clip(frames=12)
    for aug in (ArmCoupledNoise(), Tier1ArmJoints()):
        out = aug.apply(clip, aug.identity_draw(), make_ctx(_tier1_empty()))
        assert np.array_equal(out.rotations, clip.rotations, equal_nan=True)


def _tier1_empty():
    return Tier1Draw(False, 0.0, 0.0, 1.0, (), 1.0, (0.5, 0.5))


def test_draws_respect_limits():
    clip = synthetic_clip(frames=4)
    aug = ArmCoupledNoise()
    for seed in range(500):
        draw = sample(aug, clip, seed)
        assert len(draw["joints"]) == 6
        for j in draw["joints"]:
            limit = 9.5 if j["joint"] == "elbow" else 5.0
            assert abs(j["angle_deg"]) <= limit + 1e-9


def test_draw_is_json_serialisable():
    clip = synthetic_clip(frames=30)
    draw = sample(ArmCoupledNoise(ranges={"drift_deg": 1.0}), clip, 3)
    json.dumps(draw, default=lambda o: o.tolist())


def _vectors(draw, side, joint):
    for j in draw["joints"]:
        if j["side"] == side and j["joint"] == joint:
            return j["angle_deg"] * np.asarray(j["axis"])
    raise AssertionError


def test_mirror_coupling_correlates_sides():
    clip = synthetic_clip(frames=2)
    aug = ArmCoupledNoise()
    left, right = [], []
    for seed in range(2000):
        draw = sample(aug, clip, seed)
        left.append(_vectors(draw, "left", "shoulder"))
        right.append(_vectors(draw, "right", "shoulder"))
    left, right = np.array(left), np.array(right)
    assert np.corrcoef(left[:, 0], right[:, 0])[0, 1] == pytest.approx(0.7, abs=0.05)
    assert np.corrcoef(left[:, 1], -right[:, 1])[0, 1] == pytest.approx(0.7, abs=0.05)


def test_side_modes():
    clip = synthetic_clip(frames=2)
    draw = sample(ArmCoupledNoise(ranges={"side_mode": "synchronous"}), clip, 1)
    assert np.allclose(_vectors(draw, "left", "wrist"), _vectors(draw, "right", "wrist"))
    draw = sample(ArmCoupledNoise(ranges={"side_mode": "independent"}), clip, 1)
    assert not np.allclose(_vectors(draw, "left", "wrist"), _vectors(draw, "right", "wrist"))
    with pytest.raises(ValueError):
        sample(ArmCoupledNoise(ranges={"side_mode": "sideways"}), clip, 1)


def test_constant_over_clip():
    clip = synthetic_clip(frames=20)
    aug = ArmCoupledNoise()
    draw = sample(aug, clip, 7)
    out = aug.apply(clip, draw, make_ctx())
    new = Rotation.from_rotvec(out.rotations[:, 16])
    old = Rotation.from_rotvec(clip.rotations[:, 16])
    delta = (new * old.inv()).as_rotvec()
    assert np.linalg.norm(delta[0]) > 0
    assert np.allclose(delta, delta[0], atol=1e-9)


def test_only_arm_joints_change_and_bones_keep_length():
    clip = synthetic_clip(frames=10)
    aug = ArmCoupledNoise()
    out = aug.apply(clip, sample(aug, clip, 11), make_ctx())
    unchanged = [j for j in range(55) if j not in range(16, 22)]
    assert np.array_equal(out.rotations[:, unchanged], clip.rotations[:, unchanged])
    assert not np.array_equal(out.rotations[:, 16:22], clip.rotations[:, 16:22])
    skeleton = get_skeleton()
    parents = np.asarray(skeleton.parents, dtype=int)
    for t in (0, 5):
        _, a = world_pose(clip.rotations[t], clip.rest_joints, parents)
        _, b = world_pose(out.rotations[t], clip.rest_joints, parents)
        la = np.linalg.norm(a[1:] - a[parents[1:]], axis=1)
        lb = np.linalg.norm(b[1:] - b[parents[1:]], axis=1)
        assert np.allclose(la, lb, atol=1e-9)


def test_nan_frames_untouched():
    clip = synthetic_clip(frames=10)
    rotations = clip.rotations.copy()
    rotations[5] = np.nan
    body_ok = clip.body_ok.copy()
    body_ok[5] = False
    clip = clip.replace(rotations=rotations, body_ok=body_ok)
    aug = ArmCoupledNoise()
    out = aug.apply(clip, sample(aug, clip, 2), make_ctx())
    assert np.isnan(out.rotations[5]).all()
    assert not np.array_equal(out.rotations[4], clip.rotations[4])
    assert not np.array_equal(out.rotations[6], clip.rotations[6])


def test_drift_is_smooth():
    clip = synthetic_clip(frames=90)
    aug = ArmCoupledNoise(ranges={"drift_deg": 1.0})
    draw = sample(aug, clip, 4)
    drift = np.asarray(draw["drift"])
    assert drift.shape == (90, 6)
    assert np.abs(np.diff(drift, axis=0)).max() < 0.5
    assert 0.2 < drift.std() < 3.0
    out = aug.apply(clip, draw, make_ctx())
    assert not np.array_equal(out.rotations[0, 16], out.rotations[40, 16])
    assert sample(ArmCoupledNoise(), clip, 4)["drift"] is None


def test_tier1_wrapper_uses_ctx_draw():
    clip = synthetic_clip(frames=6)
    draw = Tier1Draw(
        False, 0, 0, 1.0, (ArmJointDraw("left", "shoulder", 4.0, (0.0, 0.0, 1.0)),), 1.0, (0.5, 0.5)
    )
    ctx = make_ctx(draw)
    aug = Tier1ArmJoints()
    d = aug.sample(clip_rng(0, "clip", aug.key), clip, ctx)
    assert d["joints"] == [dataclasses.asdict(draw.arm_joints[0])] or len(d["joints"]) == 1
    out = aug.apply(clip, d, ctx)
    delta = (
        Rotation.from_rotvec(out.rotations[:, 16])
        * Rotation.from_rotvec(clip.rotations[:, 16]).inv()
    )
    assert np.allclose(np.degrees(delta.magnitude()), 4.0)
    assert np.array_equal(out.rotations[:, 17:], clip.rotations[:, 17:])


def test_tier1_wrapper_requires_draw():
    clip = synthetic_clip(frames=3)
    with pytest.raises(ValueError, match="tier1_draw"):
        Tier1ArmJoints().sample(clip_rng(0, "clip", "x"), clip, make_ctx(None))


@pytest.mark.parametrize(
    "bad",
    [
        {"chain_coupling": 1.5},
        {"chain_coupling": -0.1},
        {"side_coupling": 1.2},
        {"side_coupling": -0.5},
        {"sigma_deg": -1.0},
        {"elbow_truncate_deg": -1.0},
        {"drift_deg": -0.5},
        {"side_mode": "sideways"},
    ],
)
def test_arm_noise_validates_ranges(bad):
    with pytest.raises(ValueError):
        ArmCoupledNoise(bad)


def test_arm_noise_coupling_bounds_allowed():
    for c in (0.0, 1.0):
        aug = ArmCoupledNoise({"chain_coupling": c, "side_coupling": c})
        draw = aug.sample(np.random.default_rng(0), None, None)
        assert all(np.isfinite(j["angle_deg"]) for j in draw["joints"])
