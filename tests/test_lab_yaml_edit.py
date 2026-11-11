"""The Lab's Save button output: every lab entry, edited in the browser code, still loads as a profile."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kinetic_augment.augmentations.profile import load_profile, resolved_ranges
from kinetic_augment.lab.saving import validate_profile_text
from kinetic_augment.lab.schema import build_schema

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "slr_week4_default.yaml"
NODE = shutil.which("node")


def _dig(ranges, dotted):
    node = ranges
    for part in dotted.split("."):
        node = node[part]
    return node


def _same(a, b):
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (
            isinstance(a, (list, tuple))
            and isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b
    return abs(float(a) - float(b)) < 1e-9


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if NODE is None:
        pytest.skip("node is not on PATH")
    tmp = tmp_path_factory.mktemp("yaml_edit")
    schema_path, out_path = tmp / "schema.json", tmp / "out.json"
    schema_path.write_text(json.dumps(build_schema(load_profile(BASE))))
    script = ROOT / "tests" / "js" / "build_saved.js"
    run = subprocess.run(
        [NODE, str(script), str(schema_path), str(BASE), str(out_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(out_path.read_text())


def test_every_lab_entry_is_saved(built):
    schema = build_schema(load_profile(BASE))
    assert sorted(built) == sorted(e["id"] for e in schema["entries"])


def test_saved_yaml_validates_and_carries_the_edits(built):
    for entry_id, case in built.items():
        profile = validate_profile_text(case["text"])
        assert profile.name == case["name"], entry_id
        entry = profile.entry(case["chain"])
        assert entry.enabled, entry_id
        ranges = resolved_ranges(profile, case["chain"])
        for key, value in case["edits"].items():
            assert _same(_dig(ranges, key), value), (entry_id, key, _dig(ranges, key), value)
        for key, value in case["values"].items():
            assert _same(_dig(ranges, key), value), (entry_id, key)
        assert "# viewer: was" in case["text"], entry_id


def _flat(tree, prefix=""):
    out = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


def test_saved_yaml_writes_only_moved_knobs_never_demo_overrides(built):
    """Rulings R18/R20: mirror p and hand-dropout start_rate are written only if moved; a mode
    weight only when it turns an edited variant on."""
    base = load_profile(BASE)
    schema = {e["id"]: e for e in build_schema(base)["entries"]}
    singles = 0
    for entry_id, case in built.items():
        assert set(case["edits"]) == set(case["values"]), entry_id
        assert not any(k.startswith("mode_weights") for k in case["edits"]), entry_id
        assert not any(k.startswith("mode_weights") for k in case["values"]), entry_id
        single = case.get("single")
        if single is None:
            continue
        singles += 1
        assert set(single["edits"]) == set(single["values"]), entry_id
        profile = validate_profile_text(single["text"])
        ranges = resolved_ranges(profile, case["chain"])
        base_ranges = resolved_ranges(base, case["chain"])
        for key, value in single["values"].items():
            assert _same(_dig(ranges, key), value), (entry_id, key)
        for key in _flat(schema[entry_id]["overrides"]):
            if key.startswith("mode_weights.") and single["weights"]:
                continue  # checked in test_saved_yaml_turns_on_an_edited_variants_weight
            if key not in single["values"]:
                assert _same(_dig(ranges, key), _dig(base_ranges, key)), (entry_id, key)
    assert singles >= 16


def test_saved_yaml_turns_on_an_edited_variants_weight(built):
    """Ruling R20: an edited variant whose mode weight is 0 in the base is saved with it at 1."""
    base = load_profile(BASE)
    schema = {e["id"]: e for e in build_schema(base)["entries"]}
    turned = 0
    for entry_id, case in built.items():
        weights = (schema[entry_id]["overrides"] or {}).get("mode_weights")
        if not weights:
            continue
        mode = next(m for m, w in weights.items() if w == 1)
        base_weights = resolved_ranges(base, case["chain"])["mode_weights"]
        for label in ("single", None):
            sub = case.get(label) if label else case
            if sub is None:
                continue
            profile = validate_profile_text(sub["text"])  # still a valid profile
            saved = resolved_ranges(profile, case["chain"])["mode_weights"]
            if base_weights[mode] == 0:
                turned += 1
                assert sub["weights"] == {case["chain"]: [mode]}, entry_id
                assert saved[mode] == 1, entry_id
                assert f"{mode}: 1" in sub["text"] and "# viewer: was 0" in sub["text"]
            else:
                assert sub["weights"] == {}, entry_id
            for other in base_weights:
                if other != mode:
                    assert saved[other] == base_weights[other], (entry_id, other)
    assert turned >= 6  # varied, segment, rate (whole and single each)


def test_saved_yaml_keeps_every_line_and_comment(built):
    base_lines = BASE.read_text().split("\n")
    for entry_id, case in built.items():
        lines = case["text"].split("\n")
        assert len(lines) == len(base_lines), entry_id
        for old, new in zip(base_lines, lines):
            if "#" in old and not old.lstrip().startswith("#"):
                comment = old[old.index("  #") :] if "  #" in old else ""
                assert comment in new, (entry_id, old, new)
