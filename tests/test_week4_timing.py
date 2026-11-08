"""Week-4 temporal augmentations: time warping, frame dropping, hand dropout."""

import numpy as np
import pytest

from kinetic_augment.augmentations import tier1
from kinetic_augment.augmentations.base import AugmentationRegistry
from kinetic_augment.augmentations.clip import ChainContext
from kinetic_augment.augmentations.outputs import build_outputs
from kinetic_augment.augmentations.timing import (
    FrameDrop,
    HandDropout,
    TimeWarp,
    segment_durations,
    times_from_rates,
    varied_durations,
)
from tests.hand_assets import requires_smplx


def make_ctx(duration=1.0):
    from tests.clip_fixtures import get_skeleton

    skeleton = get_skeleton()
    draw = tier1.Tier1Draw(False, 0.0, 0.0, duration, (), 1.0, (0.5, 0.5))
    return ChainContext(
        skeleton=skeleton, parents=skeleton.parents, pivot=np.zeros(3), tier1_draw=draw
    )


def test_times_from_rates_uniform():
    times = times_from_rates(np.full(20, 1.5))
    assert len(times) == 31
    np.testing.assert_allclose(times, np.linspace(0, 20, 31), atol=1e-12)


def test_times_from_rates_minimum_two_frames():
    times = times_from_rates(np.array([0.2]))
    assert len(times) == 2 and times[0] == 0.0 and times[-1] == 1.0


@requires_smplx
def test_uniform_matches_tier1():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=30)
    aug = TimeWarp()
    draw = aug.sample(np.random.default_rng(0), clip, make_ctx(1.25))
    assert draw["mode"] == "none"
    np.testing.assert_array_equal(aug.times(30, draw), tier1.duration_times(30, 1.25))
    assert aug.apply(clip, draw, make_ctx(1.25)).frames == len(tier1.duration_times(30, 1.25))


@requires_smplx
def test_identity_draw_is_exact():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=12)
    for aug in (TimeWarp(), FrameDrop()):
        out = aug.apply(clip, aug.identity_draw(), make_ctx())
        assert out.frames == clip.frames
        np.testing.assert_array_equal(out.rotations, clip.rotations)
        np.testing.assert_array_equal(out.times_s, clip.times_s)


def test_varied_monotonic_and_total():
    count, protect = 40, 0.2
    for seed in range(200):
        rng = np.random.default_rng(seed)
        duration = float(np.exp(rng.uniform(np.log(0.6), np.log(1.65))))
        rates = np.exp(rng.uniform(np.log(0.7), np.log(1.43), size=4))
        durations = varied_durations(count, rates, protect, duration)
        assert durations.shape == (count - 1,) and np.all(durations > 0)
        np.testing.assert_allclose(durations.sum(), duration * (count - 1), atol=1e-9)
        times = times_from_rates(durations)
        assert np.all(np.diff(times) > 0)
        assert len(times) == int(round((count - 1) * duration)) + 1
        assert times[0] == 0.0 and times[-1] == count - 1
        # the protected tail has a constant playback rate (the final step may be clipped)
        split = (count - 1) * (1 - protect)
        tail = durations[np.arange(count - 1) + 0.5 > split]
        assert len(tail) > 2
        np.testing.assert_allclose(tail, tail[0], atol=1e-9)
        steps = np.diff(times)
        start = int(np.ceil(durations[: len(durations) - len(tail)].sum())) + 2
        np.testing.assert_allclose(steps[start:-1], 1.0 / tail[0], atol=1e-9)


def test_segment_only_changes_window():
    count, uniform, factor = 60, 1.3, 0.6
    durations = segment_durations(count, 0.3, 0.3, factor, uniform, blend_frames=3)
    centres = np.arange(count - 1) + 0.5
    lo, hi = 0.3 * (count - 1), 0.6 * (count - 1)
    outside = (centres < lo - 3) | (centres > hi + 3)
    np.testing.assert_allclose(durations[outside], uniform, atol=1e-12)
    inside = (centres > lo) & (centres < hi)
    np.testing.assert_allclose(durations[inside], uniform * factor, atol=1e-12)
    times = times_from_rates(durations)
    steps = np.diff(times)
    first = int(np.floor(lo - 3 - 1) / uniform) - 1
    np.testing.assert_allclose(steps[: max(first, 1)], 1.0 / uniform, atol=1e-9)


@requires_smplx
@pytest.mark.parametrize("frames", [2, 3, 4])
def test_short_clip(frames):
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=frames)
    ctx = make_ctx(0.6)
    for mode in ("none", "varied", "segment"):
        aug = TimeWarp({"mode_weights": {"none": 0, "varied": 0, "segment": 0, mode: 1}})
        for seed in range(10):
            draw = aug.sample(np.random.default_rng(seed), clip, ctx)
            assert draw["mode"] == mode
            assert aug.apply(clip, draw, ctx).frames >= 2
    for mode in ("rate", "random"):
        for fill in ("none", "hold", "interpolate"):
            aug = FrameDrop({"mode_weights": {"rate": 0, "random": 0, mode: 1}, "fill": fill})
            for seed in range(10):
                draw = aug.sample(np.random.default_rng(seed), clip, ctx)
                assert aug.apply(clip, draw, ctx).frames >= 2


@requires_smplx
def test_time_warp_modes_apply_and_registered():
    from tests.clip_fixtures import synthetic_clip

    assert AugmentationRegistry.get("temporal.time_warp") is TimeWarp
    clip = synthetic_clip(frames=30)
    ctx = make_ctx(1.4)
    for mode in ("varied", "segment"):
        aug = TimeWarp(
            {
                "mode_weights": {
                    "none": 0,
                    "varied": int(mode == "varied"),
                    "segment": int(mode == "segment"),
                }
            }
        )
        draw = aug.sample(np.random.default_rng(3), clip, ctx)
        out = aug.apply(clip, draw, ctx)
        assert out.frames == draw["output_frames"]
        assert out.times_s[0] == clip.times_s[0]
        np.testing.assert_allclose(out.times_s[-1], clip.times_s[-1])


def test_time_warp_validation():
    for bad in (
        {"mode_weights": {"none": 1, "wobble": 1}},
        {"mode_weights": {"none": -1, "varied": 2, "segment": 0}},
        {"mode_weights": {"none": 0, "varied": 0, "segment": 0}},
    ):
        with pytest.raises(ValueError):
            TimeWarp(bad)
    for bad in (
        {"mode_weights": {"rate": 1, "random": 1, "x": 1}},
        {"mode_weights": {"rate": 0, "random": 0}},
        {"fill": "smear"},
    ):
        with pytest.raises(ValueError):
            FrameDrop(bad)


@requires_smplx
def test_time_warp_needs_tier1_draw():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    skeleton = get_skeleton()
    ctx = ChainContext(skeleton=skeleton, parents=skeleton.parents, pivot=np.zeros(3))
    with pytest.raises(ValueError, match="tier1_draw"):
        TimeWarp().sample(np.random.default_rng(0), synthetic_clip(frames=5), ctx)


@requires_smplx
def test_frame_drop_gap_limit():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=60)
    aug = FrameDrop({"p": [0.5, 0.9]})
    dropped_any = False
    for seed in range(500):
        draw = aug.sample(np.random.default_rng(seed), clip, None)
        out = aug.apply(clip, draw, None)
        assert out.times_s[0] == clip.times_s[0] and out.times_s[-1] == clip.times_s[-1]
        assert np.max(np.diff(out.times_s)) <= 0.125 + 1e-9
        dropped_any |= out.frames < clip.frames
    assert dropped_any


@requires_smplx
def test_frame_drop_rate_mode():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=60)
    aug = FrameDrop({"mode_weights": {"rate": 1, "random": 0}, "target_fps": [10, 10]})
    draw = aug.sample(np.random.default_rng(0), clip, None)
    out = aug.apply(clip, draw, None)
    assert 18 <= out.frames <= 22 and draw["keep"][0] == 0 and draw["keep"][-1] == 59


@requires_smplx
def test_frame_drop_fill_modes():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=40)
    frames = {}
    for fill in ("none", "hold", "interpolate"):
        aug = FrameDrop({"p": [0.6, 0.6], "fill": fill})
        draw = aug.sample(np.random.default_rng(2), clip, None)
        out = aug.apply(clip, draw, None)
        frames[fill] = out.frames
        if fill != "none":
            np.testing.assert_allclose(out.times_s, clip.times_s, atol=1e-9)
    assert frames["hold"] == frames["interpolate"] == 40
    assert frames["none"] < 40
    keep = np.array(draw["keep"])
    held = FrameDrop({"p": [0.6, 0.6], "fill": "hold"})
    out = held.apply(clip, draw | {"fill": "hold"} if False else dict(draw, fill="hold"), None)
    dropped = np.setdiff1d(np.arange(40), keep)[0]
    previous = keep[keep < dropped].max()
    np.testing.assert_array_equal(out.rotations[dropped], clip.rotations[previous])


@requires_smplx
def test_hand_dropout_runs():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=60)
    hv = clip.hand_valid.copy()
    hv[:, :] = True
    hv[:, 1] = False
    hv[10:14, 0] = False
    outputs = build_outputs(clip.replace(hand_valid=hv), get_skeleton())
    aug = HandDropout({"start_rate": [0.3, 0.3]})
    total = 0
    for seed in range(100):
        draw = aug.sample(np.random.default_rng(seed), outputs, None)
        out = aug.apply(outputs, draw, None)
        for hand, start, length in draw["runs"]:
            assert hand == 0 and 1 <= length <= 10
            assert outputs.hand_valid[start : start + length, hand].all()
            total += length
        newly = outputs.hand_valid & ~out.hand_valid
        assert newly.sum() == sum(r[2] for r in draw["runs"])
        assert not newly[:, 1].any()
        np.testing.assert_array_equal(out.hands_world[:, 1], outputs.hands_world[:, 1])
        np.testing.assert_array_equal(out.hands_world[10:14, 0], outputs.hands_world[10:14, 0])
        assert np.isnan(out.hands_world[~out.hand_valid & outputs.hand_valid]).all()
        if outputs.hands_image is not None:
            assert np.isnan(out.hands_image[newly]).all()
    assert total > 0
    ident = aug.apply(outputs, aug.identity_draw(), None)
    for name in ("hands_world", "hands_image", "hand_valid"):
        np.testing.assert_array_equal(getattr(ident, name), getattr(outputs, name))


def test_hand_dropout_truncates_runs():
    class Fake:
        hand_valid = np.ones((200, 2), dtype=bool)

    aug = HandDropout({"start_rate": [1.0, 1.0], "run_length_mean": 50, "max_run_frames": 4})
    draw = aug.sample(np.random.default_rng(0), Fake(), None)
    assert draw["runs"] and max(r[2] for r in draw["runs"]) <= 4
