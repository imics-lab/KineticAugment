"""
Write the Node test fixtures for the Lab frontend: a pane blob from `bundle.encode_pane`, its
decoded values, and a share manifest keyed by `PaneRequest.share_key()`.

Usage: `python tests/js/make_fixture.py [out_dir]` (default `tests/js/fixtures`).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kinetic_augment.lab.bundle import SCALE_I16, decode_pane, encode_pane, jsonable  # noqa: E402
from kinetic_augment.lab.pane import PaneRequest, schema_defaults  # noqa: E402

T, V = 4, 5

JC, FX = "body.joint_coupled_noise", "face.expression_jitter"
SHARE_ENTRIES = [
    {
        "id": FX,
        "knobs": [{"key": "brows.sigma", "default": 1.0}, {"key": "mouth.sigma", "default": 0.5}],
    },
    {
        "id": JC,
        "knobs": [
            {"key": "sigma_deg", "default": 2.5},
            {"key": "elbow_sigma_deg", "default": 4.75},
        ],
    },
]
# (PaneRequest fields, UI requests that must find it)
SHARE_CASES = [
    (
        {"entry_id": FX, "values": {"brows.sigma": 1.0, "mouth.sigma": 0.8}},
        [
            {"entry_id": FX, "values": {"mouth.sigma": 0.8}},
            {"entry_id": FX, "values": {"brows.sigma": 1, "mouth.sigma": 0.8}, "edge": "low"},
        ],
    ),
    (
        {"entry_id": JC, "values": {}, "reroll": 2},
        [
            {"entry_id": JC, "reroll": 2},
            {"entry_id": JC, "reroll": 2, "values": {"sigma_deg": 2.5}, "edge": "low"},
        ],
    ),
    (
        {"entry_id": JC, "values": {"elbow_sigma_deg": 10.0}, "mode": "extreme", "edge": "low"},
        [
            {"entry_id": JC, "mode": "extreme", "edge": "low", "values": {"elbow_sigma_deg": 10}},
            {
                "entry_id": JC,
                "mode": "extreme",
                "edge": "low",
                "values": {"elbow_sigma_deg": 10, "sigma_deg": 2.5},
            },
        ],
    ),
    (
        {"entry_id": JC, "values": {}, "mode": "extreme", "edge": "high"},
        [
            {"entry_id": JC, "mode": "extreme"},
            {"entry_id": JC, "mode": "extreme", "edge": "high", "values": {"sigma_deg": 2.5}},
        ],
    ),
    (
        {"entry_id": "combined.default", "reroll": 1},
        [{"entry_id": "combined.default", "reroll": 1}],
    ),
]
SHARE_MISSES = [
    {"entry_id": JC, "mode": "extreme", "edge": "low"},
    {"entry_id": JC, "values": {"sigma_deg": 3.1}},
    {"entry_id": JC, "reroll": 1},
    {"entry_id": FX, "values": {}},
]


def fixture_arrays():
    """Small pane arrays with NaN points, an int16 delta that wraps and uint16 faces above 32767."""
    times = np.array([0.0, 1.5, 2.0, 3.25], np.float32)
    verts = np.zeros((T, V, 3), np.int16)
    verts[:, :, 0] = np.arange(V)[None, :] * 100 + np.arange(T)[:, None] * 7
    verts[:, :, 1] = -np.arange(V)[None, :] * 3
    verts[0, 0, 2], verts[1, 0, 2] = -30000, 30000  # a 60000 jump: the int16 delta wraps
    verts[2, 0, 2], verts[3, 0, 2] = -32767, 32767
    arm = np.arange(T * 6 * 3, dtype=np.float32).reshape(T, 6, 3) / 10
    arm[1, 2] = np.nan  # one missing elbow
    hands = np.linspace(-1, 1, T * 2 * 21 * 3, dtype=np.float32).reshape(T, 2, 21, 3)
    hands[2, 1] = np.nan  # the right hand missing in frame 2
    face = np.full((T, 3, 3), 0.25, np.float32)
    face[0, 1, 1] = np.nan
    faces = np.array([[0, 1, 2], [2, 3, 4], [65000, 40000, 1]], np.uint16)
    return {
        "times": (times, "float32"),
        "mesh_verts": (verts, "int16"),
        "aug_arm": (arm, "float32"),
        "aug_hands": (hands, "float32"),
        "aug_face": (face, "float32"),
        "mesh_faces": (faces, "uint16"),
    }


def fixture_meta():
    return {
        "clip": "fx",
        "n": T,
        "fps": 29.97,
        "scale_i16": SCALE_I16,
        "body_ok": [1, 1, 0, 1],
        "mirrored": False,
        "metrics": {"wrist_mm": 12.5, "missing": float("nan")},
        "note": "héllo",
    }


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = encode_pane(fixture_meta(), fixture_arrays())
    (out_dir / "pane.bin").write_bytes(blob)
    (out_dir / "pane.bin.gz").write_bytes(gzip.compress(blob, mtime=0))
    meta, arrays = decode_pane(blob)
    expected = {
        "meta": jsonable(meta),
        "arrays": {
            name: {
                "dtype": str(a.dtype),
                "shape": list(a.shape),
                "values": [None if np.isnan(v) else float(v) for v in a.astype(np.float64).ravel()],
            }
            for name, a in arrays.items()
        },
    }
    (out_dir / "pane.expected.json").write_text(json.dumps(expected, indent=1) + "\n")

    # A share manifest as Task 10 writes it: PaneRequest.share_key -> blob id. Each case also
    # lists requests as the UI sends them (values at default or not, any edge in random mode)
    # that the browser must resolve to that case; `misses` must resolve to nothing.
    sha1 = hashlib.sha1(blob).hexdigest()
    schema = {"profile": "slr_week4_default", "seed": 7, "entries": SHARE_ENTRIES}
    cases = []
    for i, (fields, lookups) in enumerate(SHARE_CASES):
        req = PaneRequest("fx", **fields)
        key = req.share_key(schema_defaults(schema, req.entry_id))
        cases.append({"id": sha1 if i == 0 else f"case{i}", "key": key, "lookups": lookups})
    share = {
        "schema": schema,
        "panes": {c["key"]: c["id"] for c in cases},
        "cases": cases,
        "misses": SHARE_MISSES,
        "keys": [PaneRequest("fx", **fields).key() for fields, _ in SHARE_CASES],
        "sha1": sha1,
        "stops": {"body.joint_coupled_noise": {"sigma_deg": [0.0, 1.25, 2.5, 4.0, 5.0, 7.5, 10.0]}},
        "rerolls": 3,
    }
    (out_dir / "share.json").write_text(json.dumps(share, indent=1) + "\n")
    write_range_parity(out_dir)


def write_range_parity(out_dir: Path) -> None:
    """Every range knob of the real schema, its stops, and `range_from_marker` at many markers."""
    from kinetic_augment.augmentations.profile import load_profile
    from kinetic_augment.lab.schema import build_schema
    from kinetic_augment.lab.share import knob_stops, range_from_marker

    schema = build_schema(load_profile(ROOT / "configs" / "slr_week4_default.yaml"))
    knobs = []
    for entry in schema["entries"]:
        for knob in entry["knobs"]:
            if knob["kind"] != "range":
                continue
            lo, hi = knob["limits"]
            markers = [lo + (hi - lo) * i / 40 for i in range(41)] + [knob["default"][1]]
            stops = knob_stops(knob, 7) + knob_stops(knob, 3)
            markers += [s[1] for s in stops] + [s[0] for s in stops]
            knobs.append(
                {
                    "id": entry["id"],
                    "knob": knob,
                    "stops": jsonable(stops),
                    "markers": markers,
                    "expected": [range_from_marker(knob, m) for m in markers],
                }
            )
    (out_dir / "range_parity.json").write_text(json.dumps(jsonable(knobs)) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "fixtures")
