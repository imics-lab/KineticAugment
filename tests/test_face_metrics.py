"""Tests for the V1.3 gate metrics (design §7)."""

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.face_metrics import (
    GL_TO_IMAGE, correspondence_floor, gap_components, head_pose_errors_deg, matrix_rotation,
)


def test_matrix_rotation_removes_scale():
    rotation = Rotation.from_rotvec([0.1, 0.4, -0.2]).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = 1.7 * rotation
    np.testing.assert_allclose(matrix_rotation(matrix), rotation, atol=1e-12)


def test_head_pose_errors_vanish_for_a_consistent_pair():
    """
    Image-axis head rotations R_img(t) = F R_gl(t) C for a constant C: the relative rotations
    agree once the OpenGL axes are flipped, whatever C is.
    """
    rng = np.random.default_rng(0)
    gl = Rotation.from_rotvec(0.3 * rng.normal(size=(12, 3))).as_matrix()
    constant = Rotation.from_rotvec([0.5, -1.0, 0.2]).as_matrix()
    fitted = GL_TO_IMAGE @ gl @ constant
    matrices = np.tile(np.eye(4), (12, 1, 1))
    matrices[:, :3, :3] = 3.0 * gl
    errors = head_pose_errors_deg(fitted, matrices, np.ones(12, bool))
    assert np.nanmax(errors) < 1e-6


def test_correspondence_floor_is_zero_for_a_similar_copy():
    rng = np.random.default_rng(1)
    rest = 0.05 * rng.normal(size=(17, 3))
    canonical = np.zeros((468, 3))
    from kinetic_augment.body_model.face_correspondence import RIGID
    canonical[list(RIGID)] = 100.0 * rest @ Rotation.from_rotvec([0, 0.3, 0]).as_matrix().T
    assert correspondence_floor(canonical, rest) < 1e-9


def test_gap_components_separate_depth_from_image_plane():
    closest = np.array([0.1, -0.2, 0.3])
    depth, plane = gap_components(closest + [0.0, 0.0, 0.15], closest)
    assert abs(depth - 0.15) < 1e-12 and plane < 1e-12
    depth, plane = gap_components(closest + [0.03, -0.04, 0.0], closest)
    assert depth < 1e-12 and abs(plane - 0.05) < 1e-12
