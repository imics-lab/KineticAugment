"""Extreme draws: pin an augmentation's sampled values to its range edges (lab, task 3)."""

from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.augmentations.base import AugmentationRegistry
from kinetic_augment.augmentations.clip import ChainContext
from kinetic_augment.augmentations.profile import load_profile, resolved_ranges, tier1_ranges
from kinetic_augment.augmentations.tier1 import sample_tier1
from kinetic_augment.lab.extreme import EXTREME_ENTRIES, extreme_draw, make_hook
from tests.clip_fixtures import get_skeleton, synthetic_clip
from tests.hand_assets import requires_smplx

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
pytestmark = requires_smplx


def _setup(name, profile_name="slr_week4_default.yaml", frames=40, **range_overrides):
    profile = load_profile(CONFIGS / profile_name)
    ranges = {**resolved_ranges(profile, name), **range_overrides}
    aug = AugmentationRegistry.get(name)(ranges=ranges)
    tier1 = tier1_ranges(profile)
    clip = synthetic_clip(frames=frames)
    skeleton = get_skeleton()
    ctx = ChainContext(
        skeleton=skeleton,
        parents=np.asarray(skeleton.parents),
        pivot=np.zeros(3),
        tier1_draw=sample_tier1(np.random.default_rng(3), tier1),
    )
    draw = aug.sample(np.random.default_rng(0), clip, ctx)
    return profile, aug, tier1, clip, draw


def _pin(name, edge, **kw):
    profile, aug, tier1, clip, draw = _setup(name, **kw)
    return draw, extreme_draw(name, draw, aug.ranges, tier1, edge), aug, tier1, clip


def test_entries_listed():
    assert "body.joint_coupled_noise" in EXTREME_ENTRIES
    assert "temporal.frame_drop" in EXTREME_ENTRIES
    assert "scene.mirror" not in EXTREME_ENTRIES


def test_joint_noise_high_pins_to_truncation():
    _, out, *_ = _pin("body.joint_coupled_noise", "high")
    angles = {(j["side"], j["joint"]): j["angle_deg"] for j in out["joints"]}
    assert angles[("left", "elbow")] == 9.5 and angles[("left", "shoulder")] == 5.0
    assert angles[("right", "elbow")] == 9.5 and angles[("right", "wrist")] == 5.0
    assert out["drift"] is None


def test_joint_noise_low_is_negative_of_high():
    _, high, *_ = _pin("body.joint_coupled_noise", "high")
    _, low, *_ = _pin("body.joint_coupled_noise", "low")
    for h, lo in zip(high["joints"], low["joints"]):
        if h["joint"] == "elbow":
            assert lo["angle_deg"] == -h["angle_deg"]
        else:
            assert lo["angle_deg"] == h["angle_deg"]
            assert np.allclose(lo["axis"], -np.asarray(h["axis"]))
            assert np.isclose(np.linalg.norm(h["axis"]), 1.0)


def test_joint_noise_right_axis_mirrors_left():
    _, out, *_ = _pin("body.joint_coupled_noise", "high")
    by = {(j["side"], j["joint"]): np.asarray(j["axis"]) for j in out["joints"]}
    for joint in ("shoulder", "wrist"):
        assert np.allclose(by[("right", joint)], by[("left", joint)] * [1, -1, -1])


def test_joint_noise_synchronous_right_equals_left():
    _, out, *_ = _pin("body.joint_coupled_noise", "low", side_mode="synchronous")
    by = {(j["side"], j["joint"]): np.asarray(j["axis"]) for j in out["joints"]}
    assert np.allclose(by[("right", "shoulder")], by[("left", "shoulder")])


def test_joint_noise_does_not_mutate_input():
    draw, out, *_ = _pin("body.joint_coupled_noise", "high")
    assert out is not draw and draw["joints"][0] is not out["joints"][0]
    assert draw["drift"] is None or out["drift"] is None


@pytest.mark.parametrize("edge,pick", [("low", 0), ("high", 1)])
def test_limb_length_pins_to_range(edge, pick):
    _, out, aug, *_ = _pin("body.limb_length_scaling", edge)
    expected = aug.ranges["factor"][pick]
    assert set(out["factors"]) == {"upper_arm", "forearm"}
    assert all(v == expected for v in out["factors"].values())


@pytest.mark.parametrize("edge,pick", [("low", 0), ("high", 1)])
def test_movement_size_pins_to_range(edge, pick):
    _, out, aug, *_ = _pin("body.movement_size", edge)
    assert out["factor"] == aug.ranges["factor"][pick]


def test_location_jitter_pins_length_and_sign():
    draw, high, aug, *_ = _pin("body.location_jitter", "high")
    _, low, *_ = _pin("body.location_jitter", "low")
    h, lo, d = (np.asarray(x["offset_mm"]) for x in (high, low, draw))
    assert np.isclose(np.linalg.norm(h), aug.ranges["truncate_mm"])
    assert np.allclose(lo, -h)
    assert np.allclose(h / np.linalg.norm(h), d / np.linalg.norm(d))


def test_location_jitter_zero_direction_goes_plus_x():
    profile, aug, tier1, clip, _ = _setup("body.location_jitter")
    out = extreme_draw("body.location_jitter", {"offset_mm": [0, 0, 0]}, aug.ranges, tier1, "high")
    assert out["offset_mm"] == [aug.ranges["truncate_mm"], 0.0, 0.0]


def test_finger_noise_pins_to_truncation():
    _, high, aug, *_ = _pin("fingers.joint_coupled_noise", "high")
    _, low, *_ = _pin("fingers.joint_coupled_noise", "low")
    limit = aug.ranges["truncate_deg"]
    assert high["mcp_deg"] == [limit] * 5 and high["pip_deg"] == [limit] * 5
    assert low["mcp_deg"] == [-limit] * 5 and low["pip_deg"] == [-limit] * 5
    assert high["dip_ratio"] == aug.ranges["dip_ratio"]


def test_face_pins_enabled_groups_only():
    _, high, aug, *_ = _pin("face.expression_jitter", "high")
    _, low, *_ = _pin("face.expression_jitter", "low")
    assert high["noise"] is None
    names = synthetic_clip(frames=4).blendshape_names[1:]
    for i, n in enumerate(names):
        if n.startswith("brow") or n.startswith("eye"):
            assert high["offsets"][i] == 0.125 and low["offsets"][i] == -0.125
        elif n.startswith("cheekSquint") or n.startswith("noseSneer"):
            assert high["offsets"][i] == 0.075 and low["offsets"][i] == -0.075
        else:  # mouth_jaw is disabled: stays zero
            assert high["offsets"][i] == 0.0 and low["offsets"][i] == 0.0


def test_face_drops_frame_noise():
    profile, aug, tier1, clip, draw = _setup("face.expression_jitter", frame_noise=0.05)
    assert draw["noise"] is not None
    out = extreme_draw("face.expression_jitter", draw, aug.ranges, tier1, "high")
    assert out["noise"] is None


def test_view_pins_to_tier1_limits():
    _, high, aug, tier1, _ = _pin("scene.view", "high")
    _, low, *_ = _pin("scene.view", "low")
    assert (high["yaw_deg"], high["pitch_deg"]) == (tier1.yaw_deg, tier1.pitch_deg)
    assert (low["yaw_deg"], low["pitch_deg"]) == (-tier1.yaw_deg, -tier1.pitch_deg)


def test_scale_pins_to_tier1_range():
    _, high, aug, tier1, _ = _pin("scene.scale", "high")
    _, low, *_ = _pin("scene.scale", "low")
    assert high["factor"] == tier1.scale[1] and low["factor"] == tier1.scale[0]


def test_time_warp_uniform_pins_duration():
    _, high, aug, tier1, _ = _pin("temporal.time_warp", "high")
    _, low, *_ = _pin("temporal.time_warp", "low")
    assert high["mode"] == "none"
    assert high["duration"] == tier1.duration[1] and low["duration"] == tier1.duration[0]


def test_time_warp_other_modes_unchanged():
    profile, aug, tier1, clip, draw = _setup("temporal.time_warp")
    for mode in ("varied", "segment"):
        out = extreme_draw("temporal.time_warp", {**draw, "mode": mode}, aug.ranges, tier1, "high")
        assert out == {**draw, "mode": mode}


def test_frame_drop_random_mode_pins_p_and_recomputes_keep():
    _, low, aug, tier1, clip = _pin("temporal.frame_drop", "low")
    _, high, *_ = _pin("temporal.frame_drop", "high")
    assert low["mode"] == "random" and low["p"] == aug.ranges["p"][1]
    assert high["p"] == aug.ranges["p"][0]
    assert high["keep"] == list(range(clip.frames))  # p = 0 drops nothing
    assert len(low["keep"]) < clip.frames
    assert low["keep"][0] == 0 and low["keep"][-1] == clip.frames - 1


def test_frame_drop_rate_mode_pins_fps_and_recomputes_keep():
    weights = {"rate": 1, "random": 0}
    _, low, aug, tier1, clip = _pin("temporal.frame_drop", "low", mode_weights=weights)
    _, high, *_ = _pin("temporal.frame_drop", "high", mode_weights=weights)
    assert low["mode"] == "rate"
    assert low["target_fps"] == 12 and high["target_fps"] == 25
    assert len(low["keep"]) < len(high["keep"]) <= clip.frames


def test_frame_drop_uses_clip_when_given():
    profile, aug, tier1, clip, draw = _setup("temporal.frame_drop")
    a = extreme_draw("temporal.frame_drop", draw, aug.ranges, tier1, "low", clip=clip)
    b = extreme_draw("temporal.frame_drop", draw, aug.ranges, tier1, "low")
    assert a["keep"] == b["keep"]  # uniform 30 fps clip: the fallback timeline is exact


def test_hook_ignores_other_entries():
    profile, aug, tier1, clip, draw = _setup("scene.view")
    hook = make_hook("high", profile)
    other = {"anything": [1, 2]}
    assert hook("scene.mirror", other, aug) is other
    assert hook("body.tier1_arm_joints", other, aug) is other


def test_hook_applies_extreme_draw():
    profile, aug, tier1, clip, draw = _setup("scene.view")
    out = make_hook("low", profile)("scene.view", draw, aug)
    assert out["yaw_deg"] == -tier1.yaw_deg


def test_every_entry_pins_without_error():
    profile = load_profile(CONFIGS / "slr_week4_default.yaml")
    hook = make_hook("high", profile)
    for name in sorted(EXTREME_ENTRIES):
        _, aug, _, _, draw = _setup(name)
        assert isinstance(hook(name, draw, aug), dict)
