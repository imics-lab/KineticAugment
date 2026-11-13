"""Week-4 profiles and chain: loader checks, shipped ranges, determinism and stream independence."""

import re
from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.augmentations import tier1
from kinetic_augment.augmentations.chain import ORDER, AugmentationChain
from kinetic_augment.augmentations.outputs import build_outputs
from kinetic_augment.augmentations.profile import load_profile, tier1_ranges
from tests.hand_assets import requires_smplx

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NAMES = ("default", "ambitious", "literature")
PATHS = [CONFIGS / f"slr_week4_{n}.yaml" for n in NAMES]


def default_path():
    return CONFIGS / "slr_week4_default.yaml"


def ranges_of(profile, name):
    return profile.entry(name).ranges


def test_shipped_profiles_load():
    for path in PATHS:
        profile = load_profile(path)
        assert profile.schema_version == 1
        assert {e.name for e in profile.entries} >= set(ORDER)


def test_unknown_key_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "seed: 0\nschema_version: 1\nregions:\n  body:\n    joint_coupled_noise:\n"
        "      enabled: true\n      ranges:\n        bogus: 1\n"
    )
    with pytest.raises(ValueError):
        load_profile(bad)


def test_schema_region_and_override_errors(tmp_path):
    for text in (
        "seed: 0\nschema_version: 2\nregions: {}\n",
        "seed: 0\nschema_version: 1\nregions:\n  legs: {}\n",
        "seed: 0\nschema_version: 1\nregions:\n  body:\n    nope: {enabled: true}\n",
    ):
        path = tmp_path / "p.yaml"
        path.write_text(text)
        with pytest.raises(ValueError):
            load_profile(path)
    with pytest.raises(ValueError):
        load_profile(default_path(), enable=["body.nonexistent"])


def test_override_enable():
    assert not load_profile(default_path()).entry("face.expression_jitter").enabled
    profile = load_profile(default_path(), enable=["face.expression_jitter"])
    assert profile.entry("face.expression_jitter").enabled
    off = load_profile(default_path(), disable=["temporal.time_warp"])
    assert not off.entry("temporal.time_warp").enabled


def test_both_arm_noises_rejected():
    with pytest.raises(ValueError):
        load_profile(default_path(), enable=["body.tier1_arm_joints"])


def test_defaulted_keys_are_printed(tmp_path, capsys):
    path = tmp_path / "p.yaml"
    path.write_text(
        "seed: 0\nschema_version: 1\nregions:\n  scene:\n    view:\n      enabled: true\n"
        "      ranges:\n        yaw_deg: 10\n"
    )
    load_profile(path)
    assert "pitch_deg" in capsys.readouterr().out


def test_default_ranges_match_spec_table():
    profile = load_profile(default_path())
    assert ranges_of(profile, "scene.view") == {"yaw_deg": 25.0, "pitch_deg": 6.0}
    assert ranges_of(profile, "temporal.time_warp")["duration"] == [0.6, 1.65]
    assert ranges_of(profile, "scene.scale")["factor"] == [0.75, 1.25]
    assert ranges_of(profile, "scene.uncentre")["limit"] == 0.125
    loc = ranges_of(profile, "body.location_jitter")
    assert (loc["sigma_mm"], loc["truncate_mm"]) == (12.5, 25.0)
    assert ranges_of(profile, "body.limb_length_scaling")["factor"] == [0.93, 1.07]
    assert ranges_of(profile, "body.movement_size")["factor"] == [0.8, 1.2]
    drop = ranges_of(profile, "temporal.frame_drop")
    assert (drop["min_fps"], drop["max_gap_ms"]) == (8, 125)
    assert ranges_of(profile, "temporal.hand_dropout")["max_run_frames"] == 10
    assert ranges_of(profile, "temporal.hand_dropout")["start_rate"] == [0.0, 0.05]
    on = {e.name for e in profile.entries if e.enabled}
    assert on == {
        "body.joint_coupled_noise",
        "fingers.joint_coupled_noise",
        "temporal.time_warp",
        "temporal.frame_drop",
        "scene.mirror",
        "scene.view",
        "scene.scale",
        "scene.uncentre",
    }
    assert ranges_of(profile, "temporal.time_warp")["mode_weights"] == {
        "none": 1,
        "varied": 0,
        "segment": 0,
    }
    assert ranges_of(profile, "temporal.frame_drop")["mode_weights"] == {"rate": 0, "random": 1}


def test_literature_and_ambitious_profiles():
    lit = load_profile(CONFIGS / "slr_week4_literature.yaml")
    assert ranges_of(lit, "scene.view") == {"yaw_deg": 20.0, "pitch_deg": 5.0}
    assert ranges_of(lit, "temporal.time_warp")["duration"] == [0.67, 1.5]
    assert ranges_of(lit, "scene.scale")["factor"] == [0.8, 1.2]
    assert ranges_of(lit, "scene.uncentre")["limit"] == 0.1
    arms = ranges_of(lit, "body.joint_coupled_noise")
    assert (arms["sigma_deg"], arms["truncate_deg"]) == (2.0, 4.0)
    assert (arms["elbow_sigma_deg"], arms["elbow_truncate_deg"]) == (3.75, 7.5)
    drop = ranges_of(lit, "temporal.frame_drop")
    assert (drop["min_fps"], drop["max_gap_ms"]) == (10, 100)
    assert not lit.entry("fingers.joint_coupled_noise").enabled  # no proven range
    amb = load_profile(CONFIGS / "slr_week4_ambitious.yaml")
    off = {e.name for e in amb.entries if not e.enabled}
    assert off == {"body.tier1_arm_joints"}
    assert ranges_of(amb, "temporal.time_warp")["mode_weights"] == {
        "none": 1,
        "varied": 1,
        "segment": 1,
    }
    assert ranges_of(amb, "face.expression_jitter")["mouth_jaw"]["enabled"] is False


@pytest.mark.parametrize("path", PATHS, ids=NAMES)
def test_every_range_has_proven_comment(path):
    """Each value line under a `ranges:` block carries `# proven:` or `# setting:`."""
    lines = path.read_text().splitlines()
    in_ranges, base_indent, checked = False, 0, 0
    for number, line in enumerate(lines, 1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if in_ranges and indent <= base_indent:
            in_ranges = False
        if re.match(r"\s*ranges:\s*(#.*)?$", line):
            in_ranges, base_indent = True, indent
            continue
        if not in_ranges:
            continue
        body = line.split("#")[0].strip()
        if body.endswith(":"):  # a nested block header (mode_weights, brows, ...)
            continue
        checked += 1
        assert re.search(r"#\s*(proven|setting):", line), f"{path.name}:{number} lacks a comment"
    assert checked > 40


def test_tier1_ranges_mapping():
    r = tier1_ranges(load_profile(default_path()))
    assert (r.yaw_deg, r.pitch_deg) == (25.0, 6.0)
    assert r.duration == (0.6, 1.65) and r.scale == (0.75, 1.25)
    assert r.uncentre == 0.125 and r.mirror_p == 0.5
    assert r.arm_joint_deg == 5.0 and r.arm_joint_p == 0.3


class _Skeleton:
    parents = np.array([-1, 0, 1])


def test_tier1_stream_unchanged(monkeypatch):
    from kinetic_augment.augmentations import chain as chain_module

    profile = load_profile(default_path())
    monkeypatch.setattr(chain_module, "tier1_ranges", lambda p: tier1.TIER1)
    chain = AugmentationChain(profile, _Skeleton())
    clip = type("C", (), {"clip_id": "1991"})()
    expected = tier1.sample_tier1(np.random.default_rng([profile.seed, 1991]))
    assert chain.tier1_draw_for(clip) == expected


def all_disabled():
    profile = load_profile(default_path())
    for e in profile.entries:
        e.enabled = False
    return profile


def _equal(a, b):
    return np.array_equal(a, b, equal_nan=True)


@requires_smplx
def test_chain_disabled_is_identity():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=12)
    skeleton = get_skeleton()
    result = AugmentationChain(all_disabled(), skeleton).run(clip)
    expected = build_outputs(clip, skeleton)
    assert result.applied == [] and result.draws == {} and result.fallbacks == {}
    for name in ("pose_world", "hands_world", "face_world", "blendshapes", "hand_valid"):
        assert _equal(getattr(result.outputs, name), getattr(expected, name)), name


@requires_smplx
def test_chain_is_deterministic():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=24)
    chain = AugmentationChain(load_profile(CONFIGS / "slr_week4_ambitious.yaml"), get_skeleton())
    a, b = chain.run(clip), chain.run(clip)
    assert a.draws == b.draws and a.applied == b.applied and a.fallbacks == b.fallbacks
    for name in ("pose_world", "hands_world", "face_world", "blendshapes", "hand_valid"):
        assert _equal(getattr(a.outputs, name), getattr(b.outputs, name)), name
    assert set(a.draws) <= set(ORDER)


@requires_smplx
def test_adding_entry_keeps_other_draws():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=24)
    skeleton = get_skeleton()
    base = AugmentationChain(load_profile(default_path()), skeleton).run(clip)
    more = AugmentationChain(
        load_profile(default_path(), enable=["body.location_jitter"]), skeleton
    ).run(clip)
    assert "body.location_jitter" in more.draws and "body.location_jitter" not in base.draws
    assert base.draws["body.joint_coupled_noise"] == more.draws["body.joint_coupled_noise"]


@requires_smplx
def test_uncentre_without_camera_falls_back():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=12).replace(camera=None)
    result = AugmentationChain(load_profile(default_path()), get_skeleton()).run(clip)
    assert result.fallbacks == {"scene.uncentre": "no_camera"}
    assert "scene.uncentre" not in result.applied


@requires_smplx
def test_probability_zero_skips_every_entry():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    profile = load_profile(CONFIGS / "slr_week4_ambitious.yaml")
    for e in profile.entries:
        e.probability = 0.0
    result = AugmentationChain(profile, get_skeleton()).run(synthetic_clip(frames=12))
    assert result.applied == []
    # gated out is recorded, so a report can tell it from disabled
    assert result.draws == {e.name: {"gated": True} for e in profile.entries if e.enabled}


@requires_smplx
@pytest.mark.parametrize(
    "drop",
    [{}, {"p": [1.0, 1.0]}, {"mode_weights": {"rate": 1, "random": 0}, "target_fps": [8, 8]}],
    ids=["default", "p1", "rate8"],
)
def test_chain_warp_then_drop_on_frame_clock(drop):
    """A 1.65x warp then frame drop: gaps, fps and timestamps all on the frame-index clock."""
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    profile = load_profile(default_path())
    warp, frame_drop = profile.entry("temporal.time_warp"), profile.entry("temporal.frame_drop")
    warp.ranges["duration"] = [1.65, 1.65]
    frame_drop.ranges.update(drop)
    warp.probability = frame_drop.probability = 1.0
    clip = synthetic_clip(frames=60)
    result = AugmentationChain(profile, get_skeleton()).run(clip)
    assert result.draws["temporal.time_warp"]["output_frames"] == 98
    times = result.outputs.times_s
    max_gap_s = frame_drop.ranges["max_gap_ms"] / 1000.0
    gaps = np.diff(times)
    assert gaps.max() <= max_gap_s + 1e-9
    assert (len(times) - 1) / (times[-1] - times[0]) >= frame_drop.ranges["min_fps"]
    steps = gaps * result.clip.fps
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    assert np.all(np.round(steps) >= 1)


@pytest.mark.parametrize(
    "region, key, ranges",
    [
        ("body", "location_jitter", "sigma_mm: 12.5\n        truncate_mm: 0"),
        ("fingers", "handshape_jitter", "truncate_sigmas: 0"),
        ("face", "expression_jitter", "brows: {enabled: true, sigma: 0.06, truncate: 0}"),
        ("body", "joint_coupled_noise", "chain_coupling: 1.5"),
    ],
)
def test_bad_strength_fails_at_load(tmp_path, region, key, ranges):
    """A value that would hang or turn NaN at sample time is refused when the profile loads."""
    path = tmp_path / "p.yaml"
    path.write_text(
        f"seed: 0\nschema_version: 1\nregions:\n  {region}:\n    {key}:\n"
        f"      enabled: true\n      ranges:\n        {ranges}\n"
    )
    with pytest.raises(ValueError):
        load_profile(path)


@pytest.mark.parametrize(
    "text, key",
    [
        ("seed: 0\nschema_version: 1\nregions:\n  scene:\n    view: {enable: true}\n", "enable"),
        ("seed: 0\nschema_version: 1\nregions:\n  scene:\n    view: {range: {}}\n", "range"),
        ("sed: 3\nseed: 0\nschema_version: 1\nregions: {}\n", "sed"),
        ("seed: 0\nschema_version: 1\nprofile: x\nregions: {}\n", "profile"),
        ("schema_version: 1\nregions: {}\n", "seed"),
        ("seed: 0\nschema_version: 1\nregions:\n  scene:\n    view: true\n", "scene.view"),
    ],
    ids=["entry-typo", "entry-range", "top-typo", "top-unknown", "no-seed", "entry-not-mapping"],
)
def test_loader_rejects_unknown_or_missing_keys(tmp_path, text, key):
    path = tmp_path / "p.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=f"'{key}'"):
        load_profile(path)


def test_minimal_profile_loads(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("seed: 7\nschema_version: 1\nregions:\n  scene:\n    view: {enabled: true}\n")
    profile = load_profile(path)
    assert profile.seed == 7 and profile.entry("scene.view").enabled


@pytest.mark.parametrize(
    "entry, ranges",
    [
        ("scene.scale", {"factor": {"range": [0.8, 1.2], "distribution": "log_uniform"}}),
        ("temporal.time_warp", {"duration": {"range": [0.6, 1.65], "distribution": "uniform"}}),
    ],
)
def test_tier1_ranges_rejects_other_distribution(entry, ranges):
    profile = load_profile(default_path())
    profile.entry(entry).ranges.update(ranges)
    with pytest.raises(ValueError, match="distribution"):
        tier1_ranges(profile)


def test_tier1_ranges_accepts_its_own_distribution():
    profile = load_profile(default_path())
    profile.entry("scene.scale").ranges["factor"] = {"range": [0.8, 1.2], "distribution": "uniform"}
    profile.entry("temporal.time_warp").ranges["duration"] = {
        "range": [0.7, 1.4],
        "distribution": "log_uniform",
    }
    r = tier1_ranges(profile)
    assert r.scale == (0.8, 1.2) and r.duration == (0.7, 1.4)


def test_wrong_distribution_fails_at_load(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        "seed: 0\nschema_version: 1\nregions:\n  scene:\n    scale:\n      enabled: true\n"
        "      ranges:\n        factor: {range: [0.8, 1.2], distribution: log_uniform}\n"
    )
    with pytest.raises(ValueError, match="distribution"):
        load_profile(path)


def only(*names):
    profile = all_disabled()
    for name in names:
        profile.entry(name).enabled = True
        profile.entry(name).probability = 1.0
    return profile


@requires_smplx
def test_chain_records_wrist_movement():
    """Spec section 6: every clip-stage entry reports how far it moved the wrists."""
    import json

    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=12)
    names = (
        "body.joint_coupled_noise",
        "fingers.joint_coupled_noise",
        "body.limb_length_scaling",
        "temporal.time_warp",
    )
    profile = only(*names)
    profile.entry("body.limb_length_scaling").ranges["retarget"] = False
    result = AugmentationChain(profile, get_skeleton()).run(clip)
    moved = {n: result.draws[n].get("wrist_moved_mm_max") for n in names}
    assert isinstance(moved["body.joint_coupled_noise"], float)
    assert moved["body.joint_coupled_noise"] > 1.0
    assert moved["body.limb_length_scaling"] > 1.0  # retarget off: the wrists move, and say so
    assert moved["fingers.joint_coupled_noise"] == 0.0  # fingers sit below the wrist
    assert moved["temporal.time_warp"] is None  # frame count changes: not measured
    json.dumps(result.draws)


@requires_smplx
def test_chain_copies_draw_fallbacks():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=12)  # fewer face frames than min_face_frames (30)
    result = AugmentationChain(only("face.expression_jitter"), get_skeleton()).run(clip)
    assert result.fallbacks == {"face.expression_jitter": "too_few_face_frames"}


@requires_smplx
def test_arm_noise_drift_draw_is_json():
    import json

    from tests.clip_fixtures import get_skeleton, synthetic_clip

    profile = only("body.joint_coupled_noise")
    profile.entry("body.joint_coupled_noise").ranges["drift_deg"] = 1.0
    result = AugmentationChain(profile, get_skeleton()).run(synthetic_clip(frames=12))
    drift = result.draws["body.joint_coupled_noise"]["drift"]
    assert isinstance(drift, list) and len(drift) == 12
    json.dumps(result.draws)
