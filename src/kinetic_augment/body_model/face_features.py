"""Versioned serialization and input validation for fitted face feature archives."""

import json
import numpy as np

from .face_observations import NUM_FACE_POINTS

FACE_FEATURE_SCHEMA = 1


def validate_face_inputs(hands, face):
    """Reject mismatched streams before any model fitting is attempted."""
    if hands.width <= 0 or hands.height <= 0 or face.width <= 0 or face.height <= 0:
        raise ValueError("image dimensions must be positive")
    if (hands.width, hands.height) != (face.width, face.height):
        raise ValueError("hand and face clips have different image dimensions")
    hand_t = np.asarray(hands.timestamp_ms, dtype=np.float64)
    face_t = np.asarray(face.timestamp_ms, dtype=np.float64)
    if hand_t.ndim != 1 or face_t.ndim != 1 or len(hand_t) != len(face_t):
        raise ValueError("hand and face clips have different frame counts")
    if not np.isfinite(hand_t).all() or not np.isfinite(face_t).all():
        raise ValueError("timestamps must be finite")
    if (
        np.any(np.diff(hand_t) <= 0)
        or np.any(np.diff(face_t) <= 0)
        or not np.array_equal(hand_t, face_t)
    ):
        raise ValueError("hand and face timestamps must match and be strictly increasing")
    expected = {
        "hands.world": (len(hand_t), 2, 21, 3),
        "hands.image": (len(hand_t), 2, 21, 3),
        "hands.present": (len(hand_t), 2),
        "hands.pose_world": (len(hand_t), 33, 3),
        "hands.pose_image": (len(hand_t), 33, 3),
        "hands.pose_visibility": (len(hand_t), 33),
        "face.landmarks": (len(face_t), NUM_FACE_POINTS, 3),
        "face.blendshapes": (len(face_t), 52),
        "face.head_matrix": (len(face_t), 4, 4),
        "face.valid": (len(face_t),),
    }
    for name, shape in expected.items():
        obj, attr = name.split(".")
        actual = np.asarray(getattr(hands if obj == "hands" else face, attr)).shape
        if actual != shape:
            raise ValueError("{} must have shape {}; got {}".format(name, shape, actual))

    hand_hash = _source_hash(hands.meta)
    face_hash = _source_hash(face.meta)
    if hand_hash and face_hash and hand_hash != face_hash:
        raise ValueError("hand and face clips have different source hashes")


def _source_hash(meta):
    for key in ("source_sha256", "source_hash", "video_sha256"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


def feature_metadata(face, hands, result):
    """Return reproducibility metadata while retaining the detector metadata verbatim."""
    from .face_pipeline import HAND_POINTS, NECK_MODES
    from .face_regions import REGIONS, SUBREGIONS

    face_meta = dict(getattr(face, "meta", {}) or {})
    hand_meta = dict(hands if isinstance(hands, dict) else getattr(hands, "meta", {}) or {})
    source_hash = _source_hash(face_meta) or _source_hash(hand_meta)
    return {
        "schema": FACE_FEATURE_SCHEMA,
        "regions": list(REGIONS),
        "subregions": list(SUBREGIONS),
        "hand_points": list(HAND_POINTS),
        "neck_modes": list(NECK_MODES),
        "blendshape_names": list(face.blendshape_names),
        "face_source_metadata": face_meta,
        "hand_source_metadata": hand_meta,
        "source": face_meta.get("source", hand_meta.get("source", "unknown")),
        "source_sha256": source_hash or "unknown",
        "provenance_complete": bool(
            source_hash and _source_hash(face_meta) and _source_hash(hand_meta)
        ),
        "coordinate_conventions": {
            "face_landmarks": "normalized x/width, y/height, z/width",
            "face_pixels": "x*width, y*height, z*width; camera axes x-right y-down z-away",
            "face_stream_local": "metres in the SMPL-X head joint frame",
            "pose_world": "metres, MediaPipe world axes",
            "pose_image": "normalized MediaPipe image coordinates",
            "hand_world": "metres, camera-aligned axes",
            "hand_image": "normalized MediaPipe image coordinates",
            "transforms": "homogeneous column-vector transforms; world_from_head maps head to world",
        },
        "diagnostics": {
            "image_*": "unmasked 2D proximity diagnostics; not accepted fitted 3D geometry",
            "contact": "distance-threshold diagnostic; consult all-false contact_valid before use",
        },
        "depth_corrected": bool(result.depth_corrected),
        "fov_deg": float(result.fov_deg),
        "selected_frame_indices": np.asarray(result.frames).astype(int).tolist(),
        "extractor_identifier": face_meta.get("mediapipe", "unknown"),
        "face_model_identifier": face_meta.get("model", "unknown"),
        "hand_detector_identifier": hand_meta.get("mediapipe", "unknown"),
        "code_revision": face_meta.get("code_revision", "unknown"),
    }


def save_face_features(path, result, face, hands_meta):
    """Assemble compatible and versioned fields, then write a compressed NPZ archive."""
    from .face_pipeline import HAND_POINTS, NECK_MODES

    fits, n = result.head_fits, len(result.head_fits)
    shape = (n, 2, len(HAND_POINTS))
    distance, region, subregion = (
        np.full(shape, np.nan),
        np.full(shape, -1, np.int8),
        np.full(shape, -1, np.int8),
    )
    contact = np.zeros(shape, bool)
    for i, geometry in enumerate(result.geometry):
        if geometry is not None:
            distance[i], region[i], subregion[i], contact[i] = (
                geometry.distance,
                geometry.region,
                geometry.subregion,
                geometry.contact,
            )
    image = {
        "image_distance": np.full(shape, np.nan),
        "image_overlap": np.zeros(shape, bool),
        "image_region": np.full(shape, -1, np.int8),
        "image_subregion": np.full(shape, -1, np.int8),
        "image_near": np.zeros(shape, bool),
    }
    for i, geometry in enumerate(result.image_geometry):
        if geometry is not None:
            image["image_distance"][i], image["image_overlap"][i] = (
                geometry.distance,
                geometry.overlap,
            )
            image["image_region"][i], image["image_subregion"][i] = (
                geometry.region,
                geometry.subregion,
            )
            image["image_near"][i] = geometry.near
    arrays = dict(
        frames=result.frames,
        neck_pose=np.array([fit.neck_pose for fit in fits]),
        head_pose=np.array([fit.head_pose for fit in fits]),
        jaw_pose=np.array([fit.jaw_pose for fit in fits]),
        valid=np.array([fit.valid for fit in fits]),
        reliable=np.array([fit.reliable for fit in fits]),
        jaw_reliable=np.array([fit.jaw_reliable for fit in fits]),
        rms_fraction=np.array([fit.rms_fraction for fit in fits]),
        points_used=np.array([fit.points_used for fit in fits]),
        neck_mode=np.array([NECK_MODES.index(fit.neck_mode) for fit in fits], np.int8),
        ear_residual_m=np.array([fit.ear_residual_m for fit in fits]),
        distance=distance,
        region=region,
        subregion=subregion,
        contact=contact,
        **image,
        head_scale_px_per_m=result.head_scale_px_per_m,
        head_camera_rotation_deg=result.head_camera_rotation_deg,
        wrist_target=result.wrist_target,
        wrist_corrected=result.wrist_corrected,
        depth_ratio=result.depth_ratio,
        wrist_residual_m=result.wrist_residual_m,
        wrist_shift_m=result.wrist_shift_m,
        wrist_reached=result.wrist_reached,
        hand_near_face=result.hand_near_face,
        body_ok=result.body_ok,
        blendshapes=face.blendshapes[result.frames],
        betas=result.betas,
        face_landmarks=result.face_landmarks,
        face_pixels=result.face_pixels,
        face_stream_local=result.face_stream_local,
        head_transform=result.head_transform,
        face_pixel_to_world=result.face_pixel_to_world,
        face_observed=result.face_observed,
        face_attachment_valid=result.face_attachment_valid,
        face_point_valid=result.face_point_valid,
        face_point_visible=result.face_point_visible,
        head_world_rotation=result.head_world_rotation,
        head_relative_rotation=result.head_relative_rotation,
        pose_image=result.pose_image,
        pose_world=result.pose_world,
        pose_visibility=result.pose_visibility,
        hand_image=result.hand_image,
        hand_world=result.hand_world,
        hand_present=result.hand_present,
        geometry_valid=result.geometry_valid,
        contact_valid=result.contact_valid,
        timestamp_ms=face.timestamp_ms[result.frames],
        width=np.array(face.width),
        height=np.array(face.height),
        fps=np.array(face.fps),
    )
    meta = feature_metadata(face, hands_meta, result)
    np.savez_compressed(path, **arrays, meta=np.array(json.dumps(meta, sort_keys=True)))


def load_face_features(path):
    """Load a schema-1 face archive as plain arrays plus decoded ``meta``."""
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data.files:
            raise ValueError("{}: missing feature metadata".format(path))
        try:
            meta = json.loads(str(data["meta"].item()))
        except (TypeError, ValueError) as exc:
            raise ValueError("{}: invalid feature metadata".format(path)) from exc
        if not isinstance(meta, dict):
            raise ValueError("{}: feature metadata must be a JSON object".format(path))
        if meta.get("schema") != FACE_FEATURE_SCHEMA:
            raise ValueError(
                "{}: face feature schema {}, expected {}".format(
                    path, meta.get("schema"), FACE_FEATURE_SCHEMA
                )
            )
        result = {key: data[key].copy() for key in data.files if key != "meta"}
    _validate_archive_shapes(path, result)
    result["meta"] = meta
    return result


def _validate_archive_shapes(path, data):
    time_shapes = {
        "frames": (),
        "neck_pose": (3,),
        "head_pose": (3,),
        "jaw_pose": (3,),
        "valid": (),
        "reliable": (),
        "jaw_reliable": (),
        "rms_fraction": (),
        "points_used": (),
        "neck_mode": (),
        "ear_residual_m": (),
        "distance": (2, 7),
        "region": (2, 7),
        "subregion": (2, 7),
        "contact": (2, 7),
        "image_distance": (2, 7),
        "image_overlap": (2, 7),
        "image_region": (2, 7),
        "image_subregion": (2, 7),
        "image_near": (2, 7),
        "head_scale_px_per_m": (),
        "head_camera_rotation_deg": (),
        "wrist_target": (2, 3),
        "wrist_corrected": (2,),
        "depth_ratio": (2,),
        "wrist_residual_m": (2,),
        "wrist_shift_m": (2,),
        "wrist_reached": (2,),
        "hand_near_face": (),
        "body_ok": (),
        "blendshapes": (52,),
        "face_landmarks": (478, 3),
        "face_pixels": (478, 3),
        "face_stream_local": (478, 3),
        "head_transform": (4, 4),
        "face_pixel_to_world": (4, 4),
        "face_observed": (),
        "face_attachment_valid": (),
        "face_point_valid": (478,),
        "face_point_visible": (478,),
        "head_world_rotation": (3, 3),
        "head_relative_rotation": (3, 3),
        "pose_image": (33, 3),
        "pose_world": (33, 3),
        "pose_visibility": (33,),
        "hand_image": (2, 21, 3),
        "hand_world": (2, 21, 3),
        "hand_present": (2,),
        "geometry_valid": (2,),
        "contact_valid": (2, 7),
        "timestamp_ms": (),
    }
    static_shapes = {"betas": (300,), "width": (), "height": (), "fps": ()}
    n = len(data.get("frames", ()))
    for key, shape in time_shapes.items():
        if key not in data or data[key].shape != (n,) + shape:
            raise ValueError("{}: {} has invalid or missing shape".format(path, key))
    for key, shape in static_shapes.items():
        if key not in data or data[key].shape != shape:
            raise ValueError("{}: {} has invalid or missing shape".format(path, key))
    if data["width"].item() <= 0 or data["height"].item() <= 0:
        raise ValueError("{}: image dimensions must be positive".format(path))
    if not np.isfinite(data["fps"].item()) or data["fps"].item() <= 0:
        raise ValueError("{}: fps must be positive and finite".format(path))
