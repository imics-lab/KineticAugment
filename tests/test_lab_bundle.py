"""Pane blob encode/decode and the week-4 helpers moved into kinetic_augment.lab."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.lab import runs
from kinetic_augment.lab.bundle import decode_pane, encode_pane


def _arrays(rng):
    t, v = 7, 11
    mesh = np.cumsum(rng.integers(-20000, 20000, (t, v, 3)), axis=0)
    mesh = np.clip(mesh, -32767, 32767).astype(np.int16)
    mesh[3] = -mesh[2]  # a big frame jump, still within int16
    hands = rng.normal(size=(t, 2, 21, 3)).astype(np.float32)
    hands[2, 0] = np.nan
    return {
        "times": (np.arange(t, dtype=np.float32) * 0.5, "float32"),
        "mesh_verts": (mesh, "int16"),
        "aug_arm": (rng.normal(size=(t, 6, 3)).astype(np.float32), "float32"),
        "aug_hands": (hands, "float32"),
        "aug_face": (np.full((t, 5, 3), np.nan, np.float32), "float32"),
        "mesh_faces": (rng.integers(0, 65535, (13, 3)).astype(np.uint16), "uint16"),
    }


def _split(blob):
    (size,) = struct.unpack("<I", blob[4:8])
    return json.loads(blob[8 : 8 + size].decode("utf-8")), 8 + size


@pytest.mark.parametrize("delta", [True, False])
def test_encode_decode_round_trip(delta):
    arrays = _arrays(np.random.default_rng(0))
    meta, out = decode_pane(encode_pane({"fps": 25, "name": "x"}, arrays, mesh_delta=delta))
    assert meta == {"fps": 25, "name": "x"}
    assert list(out) == list(arrays)
    for name, (a, dtype) in arrays.items():
        assert out[name].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(out[name], a)  # NaN positions equal too


def test_delta_is_stored_as_differences():
    arrays = _arrays(np.random.default_rng(1))
    mesh = arrays["mesh_verts"][0]
    for delta in (True, False):
        header, base = _split(encode_pane({}, arrays, mesh_delta=delta))
        assert header["arrays"]["mesh_verts"]["delta"] is delta
        assert header["arrays"]["mesh_faces"]["delta"] is False
    blob = encode_pane({}, arrays, mesh_delta=True)
    header, base = _split(blob)
    start = base + header["arrays"]["mesh_verts"]["offset"]
    stored = np.frombuffer(blob, np.int16, count=mesh.size, offset=start).reshape(mesh.shape)
    np.testing.assert_array_equal(stored[0], mesh[0])
    np.testing.assert_array_equal(stored[1], mesh[1].astype(np.int32) - mesh[0])


def test_blob_header():
    blob = encode_pane({"a": 1}, _arrays(np.random.default_rng(2)))
    assert blob[:4] == b"KLAB"
    header, base = _split(blob)
    assert base % 4 == 0
    assert header["a"] == 1
    for info in header["arrays"].values():
        assert info["offset"] % 4 == 0
        assert set(info) == {"dtype", "shape", "offset", "delta"}


def test_demo_week4_still_imports():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        import demo_week4
    finally:
        sys.path.remove(scripts)
    assert demo_week4.RUNS is runs.RUNS
    assert demo_week4.run_profile is runs.run_profile
