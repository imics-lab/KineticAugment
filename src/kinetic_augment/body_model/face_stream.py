"""Reversible attachment of observed MediaPipe face landmarks to a posed head."""

from dataclasses import dataclass

import numpy as np

from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform
from kinetic_augment.body_model.face_correspondence import RIGID
from kinetic_augment.body_model.face_observations import NUM_FACE_POINTS

_ROTATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FaceAttachment:
    """Observed face in head-local coordinates and the transforms that recover it."""

    local: np.ndarray  # (478, 3), observed landmarks in the head joint frame
    world_from_head: np.ndarray  # (4, 4), head-frame to world-frame rigid transform
    pixel_to_world: np.ndarray  # (4, 4), observed-pixel to posed-world similarity
    valid: bool


def _points(value, name, expected_shape=None):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("{} must have shape (N, 3); got {}".format(name, points.shape))
    if expected_shape is not None and points.shape != expected_shape:
        raise ValueError("{} must have shape {}; got {}".format(name, expected_shape, points.shape))
    return points


def _proper_rotation(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3); got {}".format(rotation.shape))
    if not np.isfinite(rotation).all():
        return None
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_TOLERANCE, rtol=0.0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_TOLERANCE, rtol=0.0)):
        raise ValueError("rotation must be a proper orthonormal 3x3 matrix")
    return rotation


def to_head_frame(points, rotation, position):
    """Map world points into a head joint frame; missing point rows stay missing."""
    points = _points(points, "points")
    rotation = _proper_rotation(rotation)
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("position must have shape (3,); got {}".format(position.shape))
    result = np.full(points.shape, np.nan, dtype=np.float64)
    if rotation is None or not np.isfinite(position).all():
        return result
    finite = np.isfinite(points).all(axis=1)
    result[finite] = (points[finite] - position) @ rotation
    return result


def from_head_frame(points, rotation, position):
    """Map head-local points into world coordinates; missing point rows stay missing."""
    points = _points(points, "points")
    rotation = _proper_rotation(rotation)
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("position must have shape (3,); got {}".format(position.shape))
    result = np.full(points.shape, np.nan, dtype=np.float64)
    if rotation is None or not np.isfinite(position).all():
        return result
    finite = np.isfinite(points).all(axis=1)
    result[finite] = points[finite] @ rotation.T + position
    return result


def transform_points(points, matrix):
    """Apply a 4x4 homogeneous affine transform, retaining missing rows as NaN.

    Args:
        points: (N, 3) array; rows with any non-finite coordinate are treated as missing.
        matrix: finite (4, 4) affine transform with final row [0, 0, 0, 1].

    Raises:
        ValueError: if points or matrix has an invalid shape or matrix is not affine.
    """
    points = _points(points, "points")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4); got {}".format(matrix.shape))
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise ValueError("matrix must be an affine homogeneous transform")
    result = np.full(points.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    result[finite] = points[finite] @ matrix[:3, :3].T + matrix[:3, 3]
    return result


def _invalid_attachment():
    return FaceAttachment(
        local=np.full((NUM_FACE_POINTS, 3), np.nan, dtype=np.float64),
        world_from_head=np.full((4, 4), np.nan, dtype=np.float64),
        pixel_to_world=np.full((4, 4), np.nan, dtype=np.float64),
        valid=False,
    )


def attach_face(face_px, rigid_weights, posed_rigid, head_rotation, head_position):
    """Attach all observed face points with one weighted RIGID-point similarity.

    Facial deformation in ``face_px`` is retained: only the rigid anchor rows determine the
    similarity, which is then applied to the original 478 observations. Anchors with missing
    coordinates or zero weight are omitted. Fewer than three usable non-collinear anchors, a
    degenerate fit, or a non-finite head transform returns an invalid all-NaN attachment.

    Shape errors and negative/non-finite weights raise ``ValueError``. ``head_rotation`` must be
    a proper orthonormal rotation; a non-finite rotation or position yields an invalid attachment.
    No unit or coordinate conversion is performed.
    """
    face_px = _points(face_px, "face_px", (NUM_FACE_POINTS, 3))
    posed_rigid = _points(posed_rigid, "posed_rigid", (len(RIGID), 3))
    weights = np.asarray(rigid_weights, dtype=np.float64)
    if weights.shape != (len(RIGID),):
        raise ValueError("rigid_weights must have shape ({}); got {}".format(len(RIGID), weights.shape))
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("rigid_weights must be finite and non-negative")
    rotation = _proper_rotation(head_rotation)
    position = np.asarray(head_position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("head_position must have shape (3,); got {}".format(position.shape))
    if rotation is None or not np.isfinite(position).all():
        return _invalid_attachment()

    source = face_px[list(RIGID)]
    usable = (weights > 0) & np.isfinite(source).all(axis=1) & np.isfinite(posed_rigid).all(axis=1)
    source_fit, target_fit, fit_weights = source[usable], posed_rigid[usable], weights[usable]
    if source_fit.shape[0] < 3:
        return _invalid_attachment()
    # Three anchors only determine a stable 3-D rotation when they span a plane.
    fit_weights = fit_weights / fit_weights.max()
    normalized = fit_weights / fit_weights.sum()
    source_centered = source_fit - normalized @ source_fit
    target_centered = target_fit - normalized @ target_fit
    if (np.linalg.matrix_rank(source_centered * np.sqrt(normalized[:, None])) < 2 or
            np.linalg.matrix_rank(target_centered * np.sqrt(normalized[:, None])) < 2):
        return _invalid_attachment()
    try:
        scale, align_rotation, translation = similarity_transform(
            source_fit, target_fit, allow_scale=True, weights=fit_weights
        )
    except (ValueError, np.linalg.LinAlgError):
        return _invalid_attachment()
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(align_rotation).all() or not np.isfinite(translation).all():
        return _invalid_attachment()

    pixel_to_world = np.eye(4, dtype=np.float64)
    pixel_to_world[:3, :3] = scale * align_rotation
    pixel_to_world[:3, 3] = translation
    world = apply_similarity(face_px, scale, align_rotation, translation)
    local = to_head_frame(world, rotation, position)

    world_from_head = np.eye(4, dtype=np.float64)
    world_from_head[:3, :3] = rotation
    world_from_head[:3, 3] = position
    return FaceAttachment(local, world_from_head, pixel_to_world, True)
