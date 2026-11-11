"""Bundle helpers shared by the round-trip demo and the Augmentation Lab, and the pane blob."""

import json
import math
import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from kinetic_augment.augmentations.chain import AugmentationChain

SCALE_I16 = 10000.0  # mesh vertices stored as int16 in 0.1 mm, as the Tier 1 bundles
ARMS = slice(11, 17)  # MediaPipe pose arms, the demo's ARM_MP order


def jsonable(value):
    """Plain-JSON copy (numpy to lists and numbers, NaN / inf to None)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def source_frames(chain: AugmentationChain, result, count: int) -> np.ndarray:
    """
    The (fractional) source frame each output frame of `result` shows.

    Replays the time-warp times and the frame-drop selection from their recorded draws, in the
    chain's order; every other entry keeps the frames as they are.
    """
    src = np.arange(count, dtype=np.float64)
    draw = result.draws.get("temporal.time_warp")
    if draw and not draw.get("gated"):
        times = chain.augmentations["temporal.time_warp"].times(len(src), draw)
        if not (len(times) == len(src) and np.array_equal(times, np.arange(len(src)))):
            src = np.interp(times, np.arange(len(src)), src)
    draw = result.draws.get("temporal.frame_drop")
    if draw and not draw.get("gated") and draw.get("keep") is not None:
        keep = np.asarray(draw["keep"], dtype=int)
        if len(keep) != len(src):
            if draw.get("fill", "none") == "none":
                src = src[keep]
            elif draw["fill"] == "hold":
                src = src[keep[np.searchsorted(keep, np.arange(len(src)), side="right") - 1]]
    if len(src) != result.clip.frames:
        raise ValueError(f"replayed {len(src)} source times for {result.clip.frames} frames")
    return src


def write_data(path: Path, arrays):
    """{name: (array, dtype)} back to back, 4-byte aligned; returns the manifest and size."""
    manifest, offset = {}, 0
    with open(path, "wb") as fh:
        for name, (a, dtype) in arrays.items():
            a = np.ascontiguousarray(a)
            pad = (-offset) % 4
            fh.write(b"\0" * pad)
            offset += pad
            fh.write(a.tobytes())
            manifest[name] = {"dtype": dtype, "shape": list(a.shape), "offset": offset}
            offset += a.nbytes
    return manifest, offset


def bundle_arrays(outputs, vertices, times, mp_indices):
    """The Tier 1 bundle arrays from week-4 outputs, hands and face masked where not valid."""
    count = len(times)
    hand_mask = outputs.hand_valid[:, :, None, None]
    face_mask = outputs.face_valid[:, None, None]
    size = next((len(v) for v in vertices if v is not None), 0)
    verts = np.zeros((count, size, 3), np.int16)
    ok = np.zeros(count, bool)
    for t, v in enumerate(vertices):
        if v is not None and np.isfinite(v).all():
            verts[t] = np.clip(np.round(v * SCALE_I16), -32767, 32767)
            ok[t] = True

    def f32(points, shape):
        """float32 points, or NaN of `shape` when the clip has no camera."""
        if points is None:
            return np.full(shape, np.nan, np.float32)
        return np.asarray(points, np.float32)

    no_camera = outputs.pose_image is None
    arrays = {
        "times": (np.asarray(times, np.float32), "float32"),
        "mesh_verts": (verts, "int16"),
        "aug_arm": (f32(outputs.pose_world[:, ARMS], None), "float32"),
        "aug_hands": (f32(np.where(hand_mask, outputs.hands_world, np.nan), None), "float32"),
        "aug_face": (
            f32(np.where(face_mask, outputs.face_world, np.nan)[:, mp_indices], None),
            "float32",
        ),
        "img_arm": (
            f32(None if no_camera else outputs.pose_image[:, ARMS], (count, 6, 2)),
            "float32",
        ),
        "img_hands": (
            f32(
                None if no_camera else np.where(hand_mask, outputs.hands_image, np.nan),
                (count, 2, 21, 2),
            ),
            "float32",
        ),
        "img_face": (
            f32(
                None if no_camera else np.where(face_mask, outputs.face_image, np.nan),
                (count, 478, 2),
            )[:, mp_indices],
            "float32",
        ),
    }
    return arrays, ok


PANE_MAGIC = b"KLAB"
DELTA_ARRAY = "mesh_verts"


def encode_pane(
    meta: Dict, arrays: Dict[str, Tuple[np.ndarray, str]], mesh_delta: bool = True
) -> bytes:
    """
    One pane as a single blob: `KLAB`, uint32 LE header length, UTF-8 JSON header, array bytes.

    The header is `meta` plus `arrays: {name: {dtype, shape, offset, delta}}`. It is padded with
    spaces so the array bytes start 4-byte aligned; offsets count from the first array byte and
    are 4-byte aligned. With `mesh_delta`, `mesh_verts` (int16) is stored as frame 0 followed by
    frame-to-frame differences (`delta: true`). Gzip is the caller's business.
    """
    manifest, chunks, offset = {}, [], 0
    for name, (a, dtype) in arrays.items():
        a = np.ascontiguousarray(a, dtype=dtype)
        delta = bool(mesh_delta and name == DELTA_ARRAY)
        if delta:
            if a.dtype != np.int16 or a.ndim < 1:
                raise ValueError(f"{name} must be a non-empty int16 array to be delta-coded")
            wide = a.astype(np.int32)
            diff = np.concatenate([wide[:1], np.diff(wide, axis=0)])
            a = diff.astype(np.int16)  # a jump past int16 wraps, and the cumsum wraps it back
        pad = (-offset) % 4
        chunks.append(b"\0" * pad)
        offset += pad
        chunks.append(a.tobytes())
        manifest[name] = {
            "dtype": dtype,
            "shape": list(a.shape),
            "offset": offset,
            "delta": delta,
        }
        offset += a.nbytes
    header = dict(meta)
    header["arrays"] = manifest
    raw = json.dumps(jsonable(header), separators=(",", ":"), allow_nan=False).encode("utf-8")
    raw += b" " * ((-(len(PANE_MAGIC) + 4 + len(raw))) % 4)
    return PANE_MAGIC + struct.pack("<I", len(raw)) + raw + b"".join(chunks)


def decode_pane(blob: bytes) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Inverse of `encode_pane`: (meta without `arrays`, {name: array}), deltas undone."""
    if blob[:4] != PANE_MAGIC:
        raise ValueError("not a Lab pane blob")
    (size,) = struct.unpack("<I", blob[4:8])
    meta = json.loads(blob[8 : 8 + size].decode("utf-8"))
    manifest = meta.pop("arrays")
    base = 8 + size
    arrays = {}
    for name, info in manifest.items():
        dtype = np.dtype(info["dtype"])
        count = int(np.prod(info["shape"], dtype=np.int64))
        start = base + info["offset"]
        a = np.frombuffer(blob, dtype=dtype, count=count, offset=start).reshape(info["shape"])
        if info.get("delta"):
            a = np.cumsum(a, axis=0, dtype=np.int32).astype(dtype)
        arrays[name] = a.copy()
    return meta, arrays
