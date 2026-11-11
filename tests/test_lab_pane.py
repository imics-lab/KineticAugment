"""The Lab pane runner and clip store."""

import pickle
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.augmentations.chain import AugmentationChain
from kinetic_augment.augmentations.profile import load_profile
from kinetic_augment.lab.bundle import decode_pane, encode_pane
from kinetic_augment.lab.clipstore import ClipStore
from kinetic_augment.lab.pane import PaneRequest, pane_profile, prepare_clip, run_pane
from kinetic_augment.lab.runs import RUNS, run_profile
from tests.clip_fixtures import REAL_CLIP, signing_span
from tests.hand_assets import MODEL_DIR, requires_smplx

REPO = Path(__file__).resolve().parents[1]
AMBITIOUS = REPO / "configs" / "slr_week4_ambitious.yaml"


def _req(entry="body.location_jitter", **kw):
    return PaneRequest(**{"clip_id": "c", "entry_id": entry, "values": {}, "mode": "random", **kw})


def test_pane_profile_only_entry_enabled():
    base = load_profile(AMBITIOUS)
    out = pane_profile(base, _req())
    on = [e.name for e in out.entries if e.enabled]
    assert on == ["body.location_jitter"]
    assert out.entry("body.location_jitter").probability == 1.0
    assert out.seed == base.seed
    assert [e.name for e in base.entries if e.enabled] != on  # base untouched


def test_pane_profile_values_are_undotted():
    base = load_profile(AMBITIOUS)
    out = pane_profile(base, _req("face.expression_jitter", values={"brows.sigma": 0.05}))
    assert out.entry("face.expression_jitter").ranges["brows"]["sigma"] == 0.05


def test_pane_profile_reroll_changes_seed():
    base = load_profile(AMBITIOUS)
    assert pane_profile(base, _req(reroll=3)).seed == base.seed + 3


def test_pane_profile_rejects_bad_value():
    base = load_profile(AMBITIOUS)
    req = _req("body.joint_coupled_noise", values={"chain_coupling": 1.2})
    with pytest.raises(ValueError):
        pane_profile(base, req)


def test_request_key_is_canonical():
    a = _req(values={"a": 1, "b": 2})
    b = _req(values={"b": 2, "a": 1})
    assert a.key() == b.key()
    assert a.key() != _req(values={"a": 1, "b": 3}).key()


def test_clipstore_lists_adds_and_serves_frames(tmp_path):
    store = ClipStore(tmp_path, models=None)
    assert store.list() == []
    frames = tmp_path / "src"
    frames.mkdir()
    for i in range(3):
        (frames / f"{i:04d}.jpg").write_bytes(b"x")
    store.add("abc", {"r": 1}, {"h": 2}, {"f": 3}, frames)
    assert store.list() == [{"id": "abc", "frames": 3, "source": "fit"}]
    assert store.frame_path("abc", 2) == tmp_path / "abc" / "frames" / "0002.jpg"
    with open(tmp_path / "abc" / "fit.pkl", "rb") as handle:
        assert pickle.load(handle) == ({"r": 1}, {"h": 2}, {"f": 3})
    with pytest.raises(KeyError):
        store.frame_path("abc", 3)
    with pytest.raises(KeyError):
        store.get("missing")


def test_run_pane_extreme_gated_by_lab_entry_flag():
    """Time-warp varied/segment share a chain entry that has an extreme, but the lab entry not."""
    base = load_profile(AMBITIOUS)
    for entry in ("temporal.time_warp.varied", "temporal.time_warp.segment"):
        with pytest.raises(ValueError, match="no extreme for this one"):
            run_pane(None, _req(entry, mode="extreme"), base, None)


def test_clipstore_concurrent_get_fits_once(tmp_path, monkeypatch):
    import kinetic_augment.lab.clipstore as cs

    (tmp_path / "c").mkdir()
    for name in ("hands.npz", "face.npz"):
        (tmp_path / "c" / name).write_bytes(b"x")
    calls = []

    def slow_refit(self, d):
        calls.append(d)
        time.sleep(0.3)
        return ("r", "h", "f")

    monkeypatch.setattr(cs.ClipStore, "_refit", slow_refit)
    monkeypatch.setattr(cs, "prepare_clip", lambda *a: object())
    store, got = ClipStore(tmp_path, None), []
    threads = [threading.Thread(target=lambda: got.append(store.get("c"))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(got) == 2 and got[0] is got[1]


@requires_smplx
def test_clipstore_refits_from_exported_caches_once(fitted, tmp_path, monkeypatch):
    import kinetic_augment.body_model.face_pipeline as fp

    models = fitted[0]
    (tmp_path / "01991").mkdir()
    shutil.copy(f"{REAL_CLIP}_hands.npz", tmp_path / "01991" / "hands.npz")
    shutil.copy(f"{REAL_CLIP}_face.npz", tmp_path / "01991" / "face.npz")
    first = ClipStore(tmp_path, models).get("01991")
    assert (tmp_path / "01991" / "fit.pkl").exists()
    assert first.clip.frames > 0

    def no_refit(*args, **kwargs):
        raise AssertionError("refit although fit.pkl exists")

    monkeypatch.setattr(fp, "run_clip", no_refit)
    second = ClipStore(tmp_path, models).get("01991")
    assert second.clip.frames == first.clip.frames


@pytest.fixture(scope="module")
def fitted():
    from tests.face_assets import HAVE_FACE_ASSETS

    hands_path, face_path = Path(f"{REAL_CLIP}_hands.npz"), Path(f"{REAL_CLIP}_face.npz")
    if not (HAVE_FACE_ASSETS and hands_path.exists() and face_path.exists()):
        pytest.skip("pilot clip or face assets absent")
    from kinetic_augment.body_model.face_observations import load_face_clip
    from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip
    from kinetic_augment.body_model.hand_observations import load_hand_clip

    hands, face = load_hand_clip(hands_path), load_face_clip(face_path)
    models = load_face_models(MODEL_DIR)
    result = run_clip(hands, face, models, frames=signing_span(hands))
    return models, result, hands, face


@pytest.fixture(scope="module")
def prep(fitted):
    models, result, hands, face = fitted
    return prepare_clip("01991", result, hands, face, models)


@requires_smplx
def test_prepared_clip_has_faces_and_no_images(prep):
    assert prep.base_arrays["mesh_faces"][1] == "uint16"
    assert not any(k.startswith("img_") for k in prep.base_arrays)
    assert prep.n_images == prep.clip.frames


@requires_smplx
def test_clipstore_pickles_the_fit(fitted, tmp_path):
    models, result, hands, face = fitted
    store = ClipStore(tmp_path, models)
    src = tmp_path / "frames_src"
    src.mkdir()
    (src / "0000.jpg").write_bytes(b"x")
    store.add("01991", result, hands, face, src)
    got = store.get("01991")
    assert got.clip.frames == len(result.frames)
    assert store.get("01991") is got  # memory cache


@requires_smplx
def test_run_pane_random_matches_week4_export(prep, fitted):
    models = fitted[0]
    base = load_profile(AMBITIOUS)
    meta, arrays = run_pane(prep, _req(profile="ambitious"), base, models)
    profile = run_profile(RUNS["body.location_jitter"], {"ambitious": base})
    res = AugmentationChain(profile, models.skeleton).run(prep.clip)
    got = arrays["aug_hands"][0]
    want = np.where(res.outputs.hand_valid[:, :, None, None], res.outputs.hands_world, np.nan)
    assert np.allclose(got, want.astype(np.float32), equal_nan=True)
    assert not any(k.startswith("img_") for k in arrays)
    assert meta["n"] == prep.clip.frames and meta["mode"] == "random"
    decode_pane(encode_pane(meta, arrays))  # the payload encodes


@requires_smplx
def test_run_pane_extreme_location_offset_is_truncation(prep, fitted):
    models = fitted[0]
    base = load_profile(AMBITIOUS)
    meta, _ = run_pane(prep, _req(mode="extreme", edge="high"), base, models)
    offset = np.asarray(meta["draws"]["body.location_jitter"]["offset_mm"])
    assert np.isclose(np.linalg.norm(offset), meta["ranges"]["body.location_jitter"]["truncate_mm"])
    assert meta["edge"] == "high"


@requires_smplx
def test_run_pane_extreme_rejects_entries_without_one(prep, fitted):
    models = fitted[0]
    with pytest.raises(ValueError, match="no extreme for this one"):
        run_pane(prep, _req("scene.mirror", mode="extreme"), load_profile(AMBITIOUS), models)


@requires_smplx
def test_run_pane_whole_profile_uses_given_base(prep, fitted):
    models = fitted[0]
    base = load_profile(AMBITIOUS)
    meta, _ = run_pane(prep, _req("combined.ambitious"), base, models)
    assert set(meta["ranges"]) == {e.name for e in base.entries if e.enabled}


@requires_smplx
def test_run_pane_under_two_seconds(prep, fitted):
    models = fitted[0]
    base = load_profile(AMBITIOUS)
    run_pane(prep, _req(), base, models)  # warm up
    started = time.time()
    run_pane(prep, _req(reroll=1), base, models)
    elapsed = time.time() - started
    print(f"run_pane took {elapsed:.2f} s")
    assert elapsed < 3.0, f"run_pane took {elapsed:.2f} s"
