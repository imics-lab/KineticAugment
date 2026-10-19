"""Training adapters for versioned face feature archives.

The default face stream is a compact project subset of MediaPipe landmarks. Its order is
explicitly recorded on every sample; it is not a claim of identical CoSign topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from kinetic_augment.body_model.face_features import FACE_FEATURE_SCHEMA, load_face_features

FACE_SUBSET_IDS = (
    61,
    291,
    0,
    17,
    13,
    14,
    78,
    308,
    33,
    133,
    362,
    263,
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    1,
    4,
    168,
    152,
)
BODY_POSE_IDS = (11, 12, 13, 14, 15, 16)
HEAD_ROTATION_ORDER = "first column xyz, then second column xyz (column-major 6D)"
BLENDSHAPE_COUNT = 52
# MediaPipe's documented FaceLandmarker blendshape order. This layout lets a clip with no
# detector output share a batch with observed clips; its score mask remains entirely false.
MEDIAPIPE_BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)


@dataclass
class FaceTrainingSample:
    """Aligned model streams and their observation masks for one source clip."""

    streams: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    timestamp_ms: np.ndarray
    source_frames: np.ndarray
    source_time_ms: np.ndarray
    interpolated: np.ndarray
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.timestamp_ms = np.asarray(self.timestamp_ms, dtype=np.float64)
        self.source_frames = np.asarray(self.source_frames, dtype=np.int64)
        self.source_time_ms = np.asarray(self.source_time_ms, dtype=np.float64)
        self.interpolated = np.asarray(self.interpolated, dtype=bool)
        n = len(self.timestamp_ms)
        if self.timestamp_ms.ndim != 1 or not np.isfinite(self.timestamp_ms).all():
            raise ValueError("timestamp_ms must be a finite one-dimensional array")
        if np.any(np.diff(self.timestamp_ms) <= 0):
            raise ValueError("sample timestamps must be strictly increasing")
        if self.source_frames.shape != (n, 2) or self.source_time_ms.shape != (n, 2):
            raise ValueError("source frame and time mappings must have shape (T, 2)")
        if self.interpolated.shape != (n,):
            raise ValueError("interpolated must have shape (T,)")
        if set(self.streams) != set(self.masks):
            raise ValueError("streams and masks must have identical names")
        for name, stream in list(self.streams.items()):
            value = np.asarray(stream, dtype=np.float32)
            mask = np.asarray(self.masks[name], dtype=bool)
            if value.ndim != 2 or value.shape[0] != n or mask.shape != value.shape:
                raise ValueError("{} stream and mask must have matching (T, D) shapes".format(name))
            self.streams[name] = np.where(mask & np.isfinite(value), value, 0).astype(np.float32)
            self.masks[name] = mask & np.isfinite(value)
        self.metadata = dict(self.metadata)

    def select_frames(self, indices: Sequence[int]) -> "FaceTrainingSample":
        """Select ordered source rows identically across every stream and mask."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.timestamp_ms)):
            raise ValueError("indices must be a one-dimensional selection within the sample")
        if len(indices) and np.any(np.diff(indices) <= 0):
            raise ValueError("selected frame indices must be strictly increasing")
        return FaceTrainingSample(
            streams={k: v[indices].copy() for k, v in self.streams.items()},
            masks={k: v[indices].copy() for k, v in self.masks.items()},
            timestamp_ms=self.timestamp_ms[indices].copy(),
            source_frames=self.source_frames[indices].copy(),
            source_time_ms=self.source_time_ms[indices].copy(),
            interpolated=self.interpolated[indices].copy(),
            metadata=self.metadata,
        )

    def resample(
        self, timestamp_ms: Sequence[float], max_gap_ms: float = 100
    ) -> "FaceTrainingSample":
        """Sample every stream on one strictly increasing time grid without extrapolation."""
        target = np.asarray(timestamp_ms, dtype=np.float64)
        if target.ndim != 1 or not np.isfinite(target).all() or np.any(np.diff(target) <= 0):
            raise ValueError("resampling timestamps must be finite and strictly increasing")
        if not np.isfinite(max_gap_ms) or max_gap_ms < 0:
            raise ValueError("max_gap_ms must be finite and nonnegative")
        source_t = self.timestamp_ms
        n_out = len(target)
        out = {k: np.zeros((n_out, v.shape[1]), np.float32) for k, v in self.streams.items()}
        out_masks = {k: np.zeros_like(v, dtype=bool) for k, v in out.items()}
        frames = np.full((n_out, 2), -1, np.int64)
        source_times = np.full((n_out, 2), np.nan, np.float64)
        interpolated = np.zeros(n_out, bool)

        for row, time in enumerate(target):
            right = int(np.searchsorted(source_t, time, side="left"))
            if right < len(source_t) and source_t[right] == time:
                frames[row] = self.source_frames[right]
                source_times[row] = self.source_time_ms[right]
                interpolated[row] = self.interpolated[right]
                for name in out:
                    out[name][row] = self.streams[name][right]
                    out_masks[name][row] = self.masks[name][right]
                continue
            left = right - 1
            if left < 0 or right >= len(source_t) or source_t[right] - source_t[left] > max_gap_ms:
                continue
            alpha = (time - source_t[left]) / (source_t[right] - source_t[left])
            frames[row] = (self.source_frames[left, 0], self.source_frames[right, 1])
            source_times[row] = (self.source_time_ms[left, 0], self.source_time_ms[right, 1])
            for name in out:
                valid = self.masks[name][left] & self.masks[name][right]
                if name == "head":
                    # Reliable head rows carry six valid coordinates; interpolate the rotation
                    # on SO(3), then encode with the same two-column convention.
                    valid = np.full(out[name].shape[1], bool(valid.all()), dtype=bool)
                    if valid.all():
                        a = _rotation_from_6d(self.streams[name][left])
                        b = _rotation_from_6d(self.streams[name][right])
                        out[name][row] = _rotation_to_6d(_slerp_pair(a, b, alpha))
                        out_masks[name][row] = True
                    continue
                out[name][row, valid] = (1.0 - alpha) * self.streams[name][
                    left, valid
                ] + alpha * self.streams[name][right, valid]
                out_masks[name][row] = valid
            interpolated[row] = True
        return FaceTrainingSample(
            streams=out,
            masks=out_masks,
            timestamp_ms=target,
            source_frames=frames,
            source_time_ms=source_times,
            interpolated=interpolated,
            metadata=self.metadata,
        )


def sample_from_face_features(
    features: Dict, face_space: str = "image", include_blendshapes: bool = True
) -> FaceTrainingSample:
    """Build masked, finite training streams from one loaded schema-1 feature archive."""
    if face_space not in ("image", "head_local"):
        raise ValueError("face_space must be 'image' or 'head_local'")
    required = (
        "frames",
        "timestamp_ms",
        "width",
        "height",
        "pose_image",
        "pose_visibility",
        "hand_image",
        "hand_present",
        "face_point_valid",
        "face_point_visible",
        "face_pixels",
        "face_stream_local",
        "face_attachment_valid",
        "head_relative_rotation",
        "reliable",
        "blendshapes",
        "meta",
    )
    missing = [key for key in required if key not in features]
    if missing:
        raise ValueError(
            "face feature archive missing required fields: {}".format(", ".join(missing))
        )
    meta = features["meta"]
    if not isinstance(meta, dict) or meta.get("schema") != FACE_FEATURE_SCHEMA:
        raise ValueError("face feature metadata has incompatible schema")
    frames = np.asarray(features["frames"])
    timestamps = np.asarray(features["timestamp_ms"], dtype=np.float64)
    n = len(frames)
    if n == 0:
        raise ValueError("face feature archive contains no frames")
    if frames.shape != (n,) or timestamps.shape != (n,) or not np.isfinite(timestamps).all():
        raise ValueError("frames and timestamps must be one-dimensional and aligned")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("feature timestamps must be strictly increasing")
    width, height = float(np.asarray(features["width"]).item()), float(
        np.asarray(features["height"]).item()
    )
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive and finite")
    expected = {
        "pose_image": (n, 33, 3),
        "pose_visibility": (n, 33),
        "hand_image": (n, 2, 21, 3),
        "hand_present": (n, 2),
        "face_point_valid": (n, 478),
        "face_point_visible": (n, 478),
        "face_pixels": (n, 478, 3),
        "face_stream_local": (n, 478, 3),
        "face_attachment_valid": (n,),
        "head_relative_rotation": (n, 3, 3),
        "reliable": (n,),
        "blendshapes": (n, 52),
    }
    for key, shape in expected.items():
        if np.asarray(features[key]).shape != shape:
            raise ValueError(
                "{} must have shape {}; got {}".format(key, shape, np.shape(features[key]))
            )

    streams: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    # Pose and hand coordinates are converted to isotropic pixels before clip normalization.
    pose = np.asarray(features["pose_image"], dtype=np.float64)[:, BODY_POSE_IDS, :2]
    pose *= np.array([width, height])
    visibility = np.asarray(features["pose_visibility"], dtype=np.float64)[:, BODY_POSE_IDS]
    pose_ok = np.isfinite(pose).all(axis=-1) & np.isfinite(visibility) & (visibility >= 0.5)
    shoulders = pose[:, :2]
    shoulder_ok = pose_ok[:, :2].all(axis=1)
    shoulder_width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    shoulder_ok &= np.isfinite(shoulder_width) & (shoulder_width > 1.0)
    shoulder_scale = _valid_median(shoulder_width[shoulder_ok])
    body_center = np.zeros((n, 2), dtype=np.float64)
    body_center[shoulder_ok] = shoulders[shoulder_ok].mean(axis=1)
    body_scale_ok = shoulder_scale is not None
    norm_pose = (pose - body_center[:, None, :]) / (shoulder_scale or 1.0)
    pose_ok &= shoulder_ok[:, None] & body_scale_ok

    hands = np.asarray(features["hand_image"], dtype=np.float64)[..., :2]
    hands *= np.array([width, height])
    hand_present = np.asarray(features["hand_present"], dtype=bool)
    hand_ok = np.isfinite(hands).all(axis=-1) & hand_present[:, :, None]
    hands_norm = (hands - body_center[:, None, None, :]) / (shoulder_scale or 1.0)
    hand_ok &= shoulder_ok[:, None, None] & body_scale_ok
    body_values = np.concatenate([norm_pose.reshape(n, -1), hands_norm.reshape(n, -1)], axis=1)
    body_mask = np.concatenate(
        [
            np.repeat(pose_ok[..., None], 2, axis=-1).reshape(n, -1),
            np.repeat(hand_ok[..., None], 2, axis=-1).reshape(n, -1),
        ],
        axis=1,
    )
    streams["body_hands"], masks["body_hands"] = _sanitize(body_values, body_mask)

    point_valid = np.asarray(features["face_point_valid"], dtype=bool)
    point_visible = np.asarray(features["face_point_visible"], dtype=bool)
    face_ids = np.asarray(FACE_SUBSET_IDS, dtype=np.int64)
    if face_space == "image":
        face_values = np.asarray(features["face_pixels"], dtype=np.float64)[:, face_ids, :2]
        eye_a, eye_b = (
            face_values[:, FACE_SUBSET_IDS.index(33)],
            face_values[:, FACE_SUBSET_IDS.index(263)],
        )
        face_base_mask = point_valid[:, face_ids] & point_visible[:, face_ids]
    else:
        face_values = np.asarray(features["face_stream_local"], dtype=np.float64)[:, face_ids]
        eye_a, eye_b = (
            face_values[:, FACE_SUBSET_IDS.index(33), :],
            face_values[:, FACE_SUBSET_IDS.index(263), :],
        )
        attached = np.asarray(features["face_attachment_valid"], dtype=bool)
        face_base_mask = point_valid[:, face_ids] & point_visible[:, face_ids] & attached[:, None]
    eye_valid = (
        face_base_mask[:, FACE_SUBSET_IDS.index(33)] & face_base_mask[:, FACE_SUBSET_IDS.index(263)]
    )
    eye_dist = np.linalg.norm(eye_a - eye_b, axis=1)
    eye_valid &= np.isfinite(eye_dist) & (eye_dist > 1e-8)
    eye_scale = _valid_median(eye_dist[eye_valid])
    center = np.zeros((n, face_values.shape[-1]), dtype=np.float64)
    center[eye_valid] = (eye_a[eye_valid] + eye_b[eye_valid]) / 2
    face_norm = (face_values - center[:, None, :]) / (eye_scale or 1.0)
    face_mask = face_base_mask & eye_valid[:, None] & (eye_scale is not None)
    streams["face"], masks["face"] = _sanitize(
        face_norm.reshape(n, -1), np.repeat(face_mask, face_values.shape[-1], axis=1)
    )

    rotation = np.asarray(features["head_relative_rotation"], dtype=np.float64)
    reliable = np.asarray(features["reliable"], dtype=bool)
    head_values = np.concatenate([rotation[:, :, 0], rotation[:, :, 1]], axis=1)
    head_mask = np.repeat(reliable[:, None], 6, axis=1) & np.isfinite(head_values)
    streams["head"], masks["head"] = _sanitize(head_values, head_mask)

    blendshape_names = list(meta.get("blendshape_names", []))
    unnamed_blendshapes = not blendshape_names
    if unnamed_blendshapes:
        blendshape_names = list(MEDIAPIPE_BLENDSHAPE_NAMES)
    if len(blendshape_names) != BLENDSHAPE_COUNT:
        raise ValueError("blendshape_names must name all 52 ordered score channels")
    if include_blendshapes:
        bs = np.asarray(features["blendshapes"], dtype=np.float64)
        # Scores are masked as a unit if any selected mouth, eye, or brow point is occluded.
        observed = point_valid[:, face_ids] & point_visible[:, face_ids]
        face_clear = observed.all(axis=1)
        # Without channel names, finite numbers have unknown semantics and must stay invalid.
        bs_mask = np.isfinite(bs) & face_clear[:, None] & (not unnamed_blendshapes)
        streams["blendshapes"], masks["blendshapes"] = _sanitize(bs, bs_mask)

    metadata = {
        "schema": FACE_FEATURE_SCHEMA,
        "face_subset_ids": list(FACE_SUBSET_IDS),
        "face_subset_description": "26-point project subset; not identical CoSign topology",
        "face_space": face_space,
        "face_units": (
            "eye-distance normalized image pixels"
            if face_space == "image"
            else "eye-distance normalized head-local coordinates"
        ),
        "body_units": "isotropic pixels normalized by reliable clip median shoulder width",
        "stream_units": {
            "body_hands": "dimensionless shoulder-width-normalized image x/y",
            "face": (
                "dimensionless eye-distance-normalized image x/y"
                if face_space == "image"
                else "dimensionless eye-distance-normalized head-local x/y/z"
            ),
            "head": "dimensionless rotation matrix first two columns",
            "blendshapes": "unitless ordered MediaPipe scores",
        },
        "head_rotation_order": HEAD_ROTATION_ORDER,
        "blendshape_names": blendshape_names,
        "source": dict(meta),
    }
    return FaceTrainingSample(
        streams=streams,
        masks=masks,
        timestamp_ms=timestamps,
        source_frames=np.column_stack([frames, frames]),
        source_time_ms=np.column_stack([timestamps, timestamps]),
        interpolated=np.zeros(n, dtype=bool),
        metadata=metadata,
    )


class WLASLFaceDataset:
    """WLASL annotations paired with precomputed ``<video_id>_face_features.npz`` files."""

    SUBSETS = ("wlasl100", "wlasl300", "wlasl1000", "wlasl2000")
    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root_path,
        feature_dir,
        annotation_file="WLASL_v0.3.json",
        split="train",
        subset="wlasl100",
        face_space="image",
        include_blendshapes=True,
    ):
        self.root_path = Path(root_path)
        feature_path = Path(feature_dir)
        self.feature_dir = (
            feature_path if feature_path.is_absolute() else self.root_path / feature_path
        )
        self.split, self.subset = split, subset
        self.face_space, self.include_blendshapes = face_space, include_blendshapes
        if subset not in self.SUBSETS:
            raise ValueError("Unknown subset: {}. Choose from {}".format(subset, self.SUBSETS))
        if split not in self.SPLITS:
            raise ValueError("Unknown split: {}. Choose from {}".format(split, self.SPLITS))
        if face_space not in ("image", "head_local"):
            raise ValueError("face_space must be 'image' or 'head_local'")
        with (self.root_path / annotation_file).open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        subset_count = int(subset.removeprefix("wlasl"))
        self.gloss_list: List[str] = []
        self.samples: List[Dict] = []
        for entry in annotations[:subset_count]:
            label = len(self.gloss_list)
            gloss = entry["gloss"]
            self.gloss_list.append(gloss)
            for instance in entry.get("instances", []):
                if instance.get("split", "train") != split:
                    continue
                video_id = str(instance.get("video_id", ""))
                if not video_id:
                    continue
                path = self.feature_dir / "{}_face_features.npz".format(video_id)
                if path.is_file():
                    self.samples.append(
                        {
                            "video_id": video_id,
                            "gloss": gloss,
                            "label": label,
                            "source_split": split,
                            "feature_path": path,
                        }
                    )
        if not self.samples:
            raise ValueError(
                "No usable face feature archives for {} {} under {}".format(
                    split, subset, self.feature_dir
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        features = load_face_features(record["feature_path"])
        archive_video_id = _archive_video_id(features["meta"])
        if archive_video_id is not None and archive_video_id != record["video_id"]:
            raise ValueError(
                "feature archive video id {} does not match annotation {}".format(
                    archive_video_id, record["video_id"]
                )
            )
        sample = sample_from_face_features(features, self.face_space, self.include_blendshapes)
        sample.metadata.update(
            {
                "video_id": record["video_id"],
                "gloss": record["gloss"],
                "source_split": record["source_split"],
                "source_identity": record["video_id"],
            }
        )
        return sample, int(record["label"])

    def get_class_names(self):
        return list(self.gloss_list)


def collate_face_samples(batch):
    """Pad aligned streams into ``(x, lengths, labels)`` torch batches with false masks."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    import torch

    samples, labels = zip(*batch)
    names = tuple(samples[0].streams)
    layout = (
        names,
        tuple(samples[0].streams[name].shape[1] for name in names),
        tuple(samples[0].metadata.get("face_subset_ids", ())),
        (
            tuple(samples[0].metadata.get("blendshape_names", ()))
            if "blendshapes" in names
            else ()
        ),
    )
    check_blendshape_order = "blendshapes" in names
    for sample in samples:
        other = (
            tuple(sample.streams),
            tuple(sample.streams[name].shape[1] for name in sample.streams),
            tuple(sample.metadata.get("face_subset_ids", ())),
            tuple(sample.metadata.get("blendshape_names", ())) if check_blendshape_order else (),
        )
        if other != layout:
            raise ValueError(
                "all samples must share stream layout, subset, and blendshape channel order"
            )
    lengths = torch.as_tensor([len(s.timestamp_ms) for s in samples], dtype=torch.long)
    max_length = int(lengths.max().item())
    streams, masks = {}, {}
    for name in names:
        dim = samples[0].streams[name].shape[1]
        values = np.zeros((len(samples), max_length, dim), dtype=np.float32)
        valid = np.zeros((len(samples), max_length, dim), dtype=bool)
        for row, sample in enumerate(samples):
            length = len(sample.timestamp_ms)
            values[row, :length] = sample.streams[name]
            valid[row, :length] = sample.masks[name]
        streams[name] = torch.as_tensor(values, dtype=torch.float32)
        masks[name] = torch.as_tensor(valid, dtype=torch.bool)
    return {"streams": streams, "masks": masks}, lengths, torch.as_tensor(labels, dtype=torch.long)


def _sanitize(values, mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    return np.where(mask, values, 0).astype(np.float32), mask


def _valid_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else None


def _archive_video_id(meta):
    for source in (
        meta,
        meta.get("face_source_metadata", {}),
        meta.get("hand_source_metadata", {}),
    ):
        for key in ("video_id", "source_video_id"):
            value = source.get(key) if isinstance(source, dict) else None
            if value:
                return str(value)
    return None


def _rotation_from_6d(value):
    first = value[:3]
    second = value[3:]
    first = first / np.linalg.norm(first)
    second = second - first * np.dot(first, second)
    second = second / np.linalg.norm(second)
    third = np.cross(first, second)
    return np.column_stack([first, second, third])


def _rotation_to_6d(rotation):
    return np.concatenate([rotation[:, 0], rotation[:, 1]]).astype(np.float32)


def _slerp_pair(start, end, alpha):
    from scipy.spatial.transform import Rotation, Slerp

    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([start, end])))([alpha]).as_matrix()[0]
