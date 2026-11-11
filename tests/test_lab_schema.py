"""The lab knob schema: knobs.yaml merged with a profile's values (lab, task 4)."""

from pathlib import Path

import pytest

from kinetic_augment.augmentations.base import AugmentationRegistry
from kinetic_augment.augmentations.profile import load_profile, resolved_ranges
from kinetic_augment.lab.extreme import EXTREME_ENTRIES
from kinetic_augment.lab.runs import RUNS, merge
from kinetic_augment.lab.schema import build_schema, load_knobs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
LAB_IDS = [k for k, r in RUNS.items() if r.entry is not None]
NOT_EXTREME = {"temporal.time_warp.varied", "temporal.time_warp.segment"}
FIXED = {"mode_weights"}


def _flat(ranges, prefix=""):
    out = {}
    for key, value in ranges.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def _nest(flat):
    out = {}
    for dotted, value in flat.items():
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _schema(name="slr_week4_default.yaml"):
    return build_schema(load_profile(CONFIGS / name))


def _entries():
    return {e["id"]: e for e in _schema()["entries"]}


def test_every_lab_entry_has_knobs():
    assert sorted(load_knobs()) == sorted(LAB_IDS)
    assert len(LAB_IDS) == 17
    assert sorted(_entries()) == sorted(LAB_IDS)


@pytest.mark.parametrize("lab_id", LAB_IDS)
def test_every_range_key_is_a_knob(lab_id):
    run = RUNS[lab_id]
    expected = {
        k for k in _flat(AugmentationRegistry.get(run.entry).DEFAULT_RANGES) if k not in FIXED
    }
    expected = {k for k in expected if not k.startswith("mode_weights.")}
    keys = {k["key"] for k in _entries()[lab_id]["knobs"]}
    assert keys == expected


@pytest.mark.parametrize("profile_file", ["slr_week4_default.yaml", "slr_week4_ambitious.yaml"])
def test_defaults_within_limits(profile_file):
    for entry in _schema(profile_file)["entries"]:
        for knob in entry["knobs"]:
            value = knob["default"]
            if knob["kind"] in ("scalar", "range"):
                lo, hi = knob["limits"]
                for v in value if knob["kind"] == "range" else [value]:
                    assert lo <= v <= hi, (entry["id"], knob["key"], value, knob["limits"])
                assert knob["step"] and knob["step"] > 0
                if knob["proven"]:
                    assert lo <= knob["proven"][0] <= knob["proven"][1] <= hi
            elif knob["kind"] == "enum":
                values = value if knob["multi"] else [value]
                assert set(values) <= set(knob["options"]), (entry["id"], knob["key"])
            else:
                assert knob["kind"] == "bool" and isinstance(value, bool)


def test_plain_text_present():
    for entry in _schema()["entries"]:
        assert entry["title"].strip() and entry["does"].strip()
        assert isinstance(entry["changes"], list) and isinstance(entry["keeps"], list)
        for knob in entry["knobs"]:
            assert knob["plain"].strip(), (entry["id"], knob["key"])
            assert knob["technical"].strip() and knob["key"] in knob["technical"]
            assert knob["kind"] in ("scalar", "range", "enum", "bool")
            assert knob["plain"] != knob["technical"]
        assert 1 <= sum(1 for k in entry["knobs"] if k["share"]) <= 2


def test_extreme_flag_matches_extreme_entries():
    for lab_id, entry in _entries().items():
        expected = RUNS[lab_id].entry in EXTREME_ENTRIES and lab_id not in NOT_EXTREME
        assert entry["extreme"] is expected, lab_id


def test_schema_shape_and_run_overrides():
    schema = _schema()
    assert schema["profile"] == "slr_week4_default" and isinstance(schema["seed"], int)
    assert schema["profiles"] == ["default", "ambitious", "literature"]
    entries = _entries()
    assert entries["scene.mirror"]["knobs"][0]["default"] == 1.0
    assert entries["scene.mirror"]["entry"] == "scene.mirror"
    assert entries["scene.mirror"]["group"] == "scene"
    assert entries["scene.mirror"]["overrides"] == {"p": 1.0}
    hd = {k["key"]: k for k in entries["temporal.hand_dropout"]["knobs"]}
    assert hd["start_rate"]["default"] == [0.05, 0.05]
    assert hd["start_rate"]["kind"] == "range"


@pytest.mark.parametrize("lab_id", LAB_IDS)
def test_limits_accepted_by_augmentation(lab_id):
    run = RUNS[lab_id]
    cls = AugmentationRegistry.get(run.entry)
    profile = load_profile(CONFIGS / "slr_week4_ambitious.yaml")
    base = merge(resolved_ranges(profile, run.entry), run.overrides)
    flat = _flat(base)
    for knob in _entries()[lab_id]["knobs"]:
        if knob["kind"] == "range":
            lo, hi = knob["limits"]
            variants = [[lo, hi]]
        elif knob["kind"] == "scalar":
            variants = list(knob["limits"])
        elif knob["kind"] == "bool":
            variants = [True, False]
        elif knob["multi"]:
            variants = [[o] for o in knob["options"]]
        else:
            variants = list(knob["options"])
        for value in variants:
            ranges = _nest({**flat, knob["key"]: value})
            for mw in FIXED & set(base):
                ranges[mw] = base[mw]
            try:
                cls(ranges=ranges)
            except Exception as exc:  # pragma: no cover - failure message only
                raise AssertionError(f"{lab_id} {knob['key']}={value!r}: {exc}") from exc
