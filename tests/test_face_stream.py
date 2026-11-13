import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.face_correspondence import RIGID
from kinetic_augment.body_model.face_observations import NUM_FACE_POINTS
from kinetic_augment.body_model.face_stream import (
    attach_face,
    from_head_frame,
    to_head_frame,
    transform_points,
)


def scene():
    rng = np.random.default_rng(1729)
    face = rng.normal(size=(NUM_FACE_POINTS, 3))
    # Keep rigid points well spread and non-coplanar for a stable similarity fit.
    face[list(RIGID)] *= 1.7
    alignment_rotation = Rotation.from_rotvec([0.31, -0.22, 0.47]).as_matrix()
    scale = 1.8
    translation = np.array([0.4, -1.3, 2.1])
    posed = scale * (face[list(RIGID)] @ alignment_rotation.T) + translation
    head_rotation = Rotation.from_rotvec([-0.14, 0.51, 0.23]).as_matrix()
    head_position = np.array([-0.8, 0.6, 1.2])
    return face, posed, alignment_rotation, scale, translation, head_rotation, head_position


def test_attachment_round_trips_observations_with_full_similarity_and_head_pose():
    face, posed, align_r, scale, translate, head_r, head_p = scene()
    attachment = attach_face(face, np.ones(len(RIGID)), posed, head_r, head_p)
    assert attachment.valid
    expected_world = scale * (face @ align_r.T) + translate
    np.testing.assert_allclose(attachment.local, to_head_frame(expected_world, head_r, head_p), atol=1e-8)
    np.testing.assert_allclose(from_head_frame(attachment.local, head_r, head_p), expected_world, atol=1e-8)
    np.testing.assert_allclose(transform_points(face, attachment.pixel_to_world), expected_world, atol=1e-8)
    np.testing.assert_allclose(attachment.pixel_to_world[:3, :3], scale * align_r, atol=1e-8)
    np.testing.assert_allclose(attachment.world_from_head[:3, :3], head_r, atol=1e-8)
    np.testing.assert_allclose(attachment.world_from_head[:3, 3], head_p, atol=1e-8)


def test_nonrigid_mouth_and_brow_changes_survive_attachment():
    face, posed, _, _, _, head_r, head_p = scene()
    changed = face.copy()
    changed[[13, 14, 61, 291, 105, 334]] += np.array([0.13, -0.21, 0.17])
    attached = attach_face(changed, np.ones(len(RIGID)), posed, head_r, head_p)
    expected = to_head_frame(
        transform_points(changed, attached.pixel_to_world), head_r, head_p
    )
    np.testing.assert_allclose(attached.local, expected, atol=1e-8)
    assert not np.allclose(attached.local[[13, 14, 61, 291, 105, 334]],
                           attach_face(face, np.ones(len(RIGID)), posed, head_r, head_p).local[
                               [13, 14, 61, 291, 105, 334]])


def test_same_observations_attach_to_arbitrary_rigid_reposing():
    face, posed, _, _, _, _, _ = scene()
    first = attach_face(face, np.ones(len(RIGID)), posed, np.eye(3), np.zeros(3))
    new_rotation = Rotation.from_rotvec([0.2, 0.3, -0.4]).as_matrix()
    new_position = np.array([3.0, -2.0, 1.4])
    reposed = posed @ new_rotation.T + new_position
    second = attach_face(face, np.ones(len(RIGID)), reposed, new_rotation, new_position)
    assert first.valid and second.valid
    np.testing.assert_allclose(first.local, second.local, atol=1e-8)


def test_partial_missing_face_and_anchor_rows_are_preserved_or_excluded():
    face, posed, _, _, _, head_r, head_p = scene()
    face[10] = np.nan
    face[RIGID[0]] = np.nan
    posed[1] = np.nan
    weights = np.ones(len(RIGID))
    weights[2] = 0.0
    attachment = attach_face(face, weights, posed, head_r, head_p)
    assert attachment.valid
    assert np.isnan(attachment.local[10]).all()
    assert np.isnan(attachment.local[RIGID[0]]).all()
    finite = np.ones((3, 3))
    finite[1] = np.nan
    transformed = transform_points(finite, np.eye(4))
    np.testing.assert_array_equal(transformed[[0, 2]], finite[[0, 2]])
    assert np.isnan(transformed[1]).all()


@pytest.mark.parametrize("weights,face_change,posed_change", [
    (np.zeros(len(RIGID)), None, None),
    (np.eye(1, len(RIGID), 0).ravel() + np.eye(1, len(RIGID), 1).ravel(), None, None),
    (None, "collinear", "collinear"),
])
def test_zero_or_degenerate_anchors_return_invalid_nan_attachment(weights, face_change, posed_change):
    face, posed, _, _, _, head_r, head_p = scene()
    if weights is None:
        weights = np.zeros(len(RIGID))
        weights[:3] = 1
        indices = list(RIGID[:3])
        face[indices] = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        posed[:3] = np.array([[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]])
    attachment = attach_face(face, weights, posed, head_r, head_p)
    assert not attachment.valid
    assert np.isnan(attachment.local).all()
    assert np.isnan(attachment.pixel_to_world).all()
    assert np.isnan(attachment.world_from_head).all()


def test_nonfinite_head_transform_returns_invalid_and_bad_rotation_is_rejected():
    face, posed, _, _, _, _, head_p = scene()
    result = attach_face(face, np.ones(len(RIGID)), posed, np.full((3, 3), np.nan), head_p)
    assert not result.valid and np.isnan(result.local).all()
    with pytest.raises(ValueError, match="proper orthonormal"):
        attach_face(face, np.ones(len(RIGID)), posed, 2 * np.eye(3), head_p)


def test_helpers_report_shape_errors():
    with pytest.raises(ValueError, match="points must have shape"):
        transform_points(np.zeros((3, 2)), np.eye(4))
    with pytest.raises(ValueError, match="matrix must have shape"):
        transform_points(np.zeros((3, 3)), np.eye(3))
