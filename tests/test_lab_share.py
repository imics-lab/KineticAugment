"""The share build: the pane grid, the size-budget choice, mesh decimation, and the zip."""

import base64
import gzip
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.augmentations.profile import load_profile
from kinetic_augment.lab.pane import schema_defaults
from kinetic_augment.lab.schema import build_schema
from kinetic_augment.lab.share import (
    choose_grid,
    decimate_mesh,
    knob_stops,
    plan_grid,
    range_from_marker,
    stops_table,
)
from tests.hand_assets import MODEL_DIR, requires_smplx

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "src" / "kinetic_augment" / "lab" / "static"
CLIPS = REPO / "demo" / "round_trip" / "clips"


@pytest.fixture(scope="module")
def schema():
    return build_schema(load_profile(REPO / "configs" / "slr_week4_default.yaml"))


def _entry(schema, entry_id):
    return next(e for e in schema["entries"] if e["id"] == entry_id)


def test_plan_grid_includes_defaults_and_proven_edges(schema):
    reqs = plan_grid(schema, 3, 0)
    # a scalar with a proven range: the default pane, and both proven edges as stops
    jitter = "body.location_jitter"
    stops = knob_stops(_entry(schema, jitter)["knobs"][0], 3)
    assert 12.5 in stops and 0 in stops and 10 in stops and 25 in stops
    assert stops == sorted(stops) and len(set(stops)) == len(stops)
    keys = {(r.entry_id, r.mode, r.reroll, json.dumps(r.values)) for r in reqs}
    assert (jitter, "random", 0, "{}") in keys
    assert (jitter, "random", 0, json.dumps({"sigma_mm": 10.0})) in keys
    # a range knob: stops are [lo, hi] pairs, the default among them, sorted by the high end
    scale = knob_stops(_entry(schema, "scene.scale")["knobs"][0], 3)
    assert [0.75, 1.25] in scale and all(isinstance(s, list) for s in scale)
    assert [r[1] for r in scale] == sorted(r[1] for r in scale)
    assert [0.78, 1.2] in scale and [0.8, 1.17] in scale  # proven edges, by the marker rule
    # whole profiles and rerolls
    ids = {r.entry_id for r in reqs}
    assert {"combined.default", "combined.ambitious", "combined.literature"} <= ids
    more = plan_grid(schema, 3, 2)
    rerolled = [r for r in more if r.entry_id == jitter and r.reroll > 0]
    assert sorted(r.reroll for r in rerolled) == [1, 2] and all(not r.values for r in rerolled)
    assert len(more) > len(reqs)


def test_plan_grid_keys_carry_the_build_profile(schema):
    # The manifest's `profile` is the profile its keys carry; the app asks with that name.
    for profile in ("default", "ambitious", "literature"):
        reqs = plan_grid(schema, 3, 0, profile)
        keys = [r.share_key(schema_defaults(schema, r.entry_id)) for r in reqs]
        assert {json.loads(k)["profile"] for k in keys} == {profile}


def test_plan_grid_is_unique_by_share_key(schema):
    reqs = plan_grid(schema, 5, 3)
    keys = [r.share_key(schema_defaults(schema, r.entry_id)) for r in reqs]
    assert len(keys) == len(set(keys))
    assert all(r.clip_id == "" for r in reqs)


def test_plan_grid_extreme_only_where_allowed(schema):
    reqs = plan_grid(schema, 3, 1)
    extreme = [r for r in reqs if r.mode == "extreme"]
    assert extreme
    allowed = {e["id"] for e in schema["entries"] if e["extreme"]}
    assert {r.entry_id for r in extreme} == allowed
    for eid in allowed:
        assert {r.edge for r in extreme if r.entry_id == eid} == {"low", "high"}
    assert not any(r.mode == "extreme" and r.reroll for r in reqs)


def test_plan_grid_has_enum_options_at_defaults(schema):
    reqs = plan_grid(schema, 3, 0)
    sides = {r.values["side_mode"] for r in reqs if "side_mode" in r.values}
    assert sides == {"independent", "synchronous"}  # "mirror" is the default pane


def test_range_stops_match_the_page_rule():
    log = {"default": [0.6, 1.65], "limits": [0.4, 2.5], "step": 0.01}
    assert range_from_marker(log, 2.0) == [0.49, 2.0]  # 0.99 / 2 = 0.495 rounds down, as in JS
    fixed = {"default": [0.0, 0.2], "limits": [0, 0.5], "step": 0.01}
    assert range_from_marker(fixed, 0.4) == [0.0, 0.4]
    single = {"default": [0.05, 0.05], "limits": [0, 0.1], "step": 0.005}
    assert range_from_marker(single, 0.1) == [0.1, 0.1]
    linear = {"default": [12, 25], "limits": [8, 50], "step": 1}
    assert range_from_marker(linear, 30) == [8, 30]  # the low end stops at the limit


def test_stops_table_only_share_scalar_and_range(schema):
    table = stops_table(schema, 5)
    assert set(table["scene.scale"]) == {"factor"}
    for entry in schema["entries"]:
        for knob in entry["knobs"]:
            has = knob["key"] in table.get(entry["id"], {})
            assert has == bool(knob.get("share") and knob["kind"] in ("scalar", "range"))


def test_choose_grid_shrinks_to_budget(schema):
    assert choose_grid(schema, 3, 1_000, 10**12) == (7, 3)
    order = [(7, 3), (5, 3), (5, 2), (3, 2), (3, 0)]
    budgets = (10**10, 3 * 10**8, 10**8, 3 * 10**7)
    picks = [choose_grid(schema, 3, 1_000_000, b) for b in budgets]
    assert [order.index(p) for p in picks] == sorted(order.index(p) for p in picks)
    assert picks[0] != picks[-1]
    assert choose_grid(schema, 3, 10**9, 10**6) == (3, 0)  # nothing fits: the smallest
    n = len(plan_grid(schema, 7, 3))
    assert choose_grid(schema, 1, 100, n * 100 + 10) == (7, 3)
    assert choose_grid(schema, 1, 100, n * 100 + 10, fixed_bytes=10_000) != (7, 3)


def test_share_index_has_no_module_or_fetch():
    html = re.sub(r"<!--.*?-->", "", (STATIC / "index.html").read_text(), flags=re.S)
    assert 'type="module"' not in html and "import " not in html
    for path in STATIC.glob("*.js"):
        text = path.read_text()
        assert 'type="module"' not in text, path.name
        assert not re.search(r"^\s*import\s", text, re.M), path.name
        assert "import(" not in text, path.name
        if path.name == "sources.js":  # only LiveSource may fetch
            assert "fetch(" not in text.split("// ---- share mode")[1].split("Lab.chooseSource")[0]
        else:
            assert "fetch(" not in text, path.name


@requires_smplx
def test_decimate_mesh_target_and_valid_faces():
    from kinetic_augment.body_model.face_pipeline import load_face_models

    model = load_face_models(MODEL_DIR).skeleton.model
    faces = np.asarray(model.faces)
    verts = np.asarray(model.v_template, dtype=np.float64)
    index, small = decimate_mesh(verts, faces, target=3000)
    assert 2400 <= len(index) <= 3600
    assert small.dtype == np.uint16 and small.max() < len(index)
    assert index.min() >= 0 and index.max() < len(verts)
    assert len(np.unique(index)) == len(index)  # no two kept vertices map to one original
    assert (small[:, 0] != small[:, 1]).all() and (small[:, 1] != small[:, 2]).all()
    again_index, again_faces = decimate_mesh(verts, faces, target=3000)
    assert np.array_equal(index, again_index) and np.array_equal(small, again_faces)


@requires_smplx
@pytest.mark.skipif(not (CLIPS / "57949").is_dir(), reason="demo clip 57949 absent")
def test_build_share_small(tmp_path):
    from kinetic_augment.body_model.face_pipeline import load_face_models
    from kinetic_augment.lab.bundle import decode_pane
    from kinetic_augment.lab.clipstore import ClipStore
    from kinetic_augment.lab.share import build_share

    models = load_face_models(MODEL_DIR)
    out = tmp_path / "lab.zip"
    summary = build_share(
        ["57949"],
        ClipStore(CLIPS, models),
        models,
        out,
        budget_mb=500,
        grid=(3, 0),
        entries=["body.location_jitter", "combined.default"],
    )
    assert summary["panes"] >= 6 and summary["unique_blobs"] >= 2 and summary["size_bytes"] > 0
    assert summary["unique_blobs"] == summary["panes"] + 1  # every pane and the base: one file each
    root = "augmentation-lab/"
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert root + "index.html" in names and root + "data/manifest.js" in names
        assert root + "README.txt" in names
        assert not [n for n in names if n.lower().endswith((".npz", ".pkl", ".task"))]
        assert not [n for n in names if n.endswith("index.md")]
        manifest = zf.read(root + "data/manifest.js").decode()
        data = json.loads(manifest[len("window.LAB_DATA = ") : manifest.rindex(";")])
        load = zf.read(root + "data/load.js").decode()
        assert "data/manifest.js" in load and "data/57949/" not in load  # blobs load lazily
        assert load.count("<script") == 1
        sha = data["base"]["57949"]
        assert f"{root}data/57949/{sha}.js" in names
        assert any(n.startswith(root + "data/57949/frames/0000") for n in names)
        assert data["clips"][0]["id"] == "57949" and data["rerolls"] == 0
        assert data["profile"] == "default"
        assert {json.loads(k)["profile"] for k in data["panes"]} == {data["profile"]}
        assert "body.location_jitter" in data["stops"]
        for key, blob_id in data["panes"].items():
            assert json.loads(key)["clip_id"] == "57949"
            assert f"{root}data/57949/{blob_id}.js" in names

        def blob(blob_id):
            text = zf.read(f"{root}data/57949/{blob_id}.js").decode()
            b64 = re.search(r'addBlob\("[0-9a-f]+","([^"]+)"\)', text).group(1)
            return decode_pane(gzip.decompress(base64.b64decode(b64)))

        base_meta, base_arrays = blob(sha)
        meta, arrays = blob(next(iter(data["panes"].values())))
        assert base_meta["scale_i16"] == meta["scale_i16"] < 10000.0  # coarser than the live 0.1 mm
        assert base_arrays["mesh_verts"].shape[1] == arrays["mesh_verts"].shape[1]
        assert base_arrays["mesh_faces"].max() < base_arrays["mesh_verts"].shape[1]
