"""Tests for the 11-point neck fit (head-placement spike): synthetic recovery of the head joint."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.hand_assets import MODEL_DIR, requires_smplx


@pytest.fixture(scope="module")
def body():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton

    betas = np.zeros(300)
    betas[[0, 2, 7]] = [0.8, -0.6, 1.2]
    return SMPLXSkeleton(model_path=MODEL_DIR).shaped(betas)


def truth_rotations():
    """A y-down world like MediaPipe's, with arms, a swung neck and a turned head."""
    rotations = np.zeros((55, 3))
    rotations[0] = (
        Rotation.from_rotvec([np.pi, 0, 0]) * Rotation.from_rotvec([0.05, 0.3, -0.04])
    ).as_rotvec()
    rotations[16:22] = 0.3 * np.random.default_rng(4).normal(size=(6, 3))
    rotations[12] = [0.15, 0.25, -0.05]
    rotations[15] = [0.1, -0.2, 0.08]
    return rotations


@requires_smplx
def test_neck_fit_recovers_the_head_joint_and_reproduces_the_head_rotation(body):
    from kinetic_augment.body_model.assembly import HEAD, NECK, global_rotations
    from kinetic_augment.body_model.head_placement import fit_neck_to_points
    from kinetic_augment.body_model.kinematics import forward_kinematics

    truth = truth_rotations()
    transl = np.array([0.02, -0.1, 0.3])
    joints = forward_kinematics(truth, body) + transl
    head_world = global_rotations(truth, body.parents)[HEAD]
    offsets = 0.08 * np.random.default_rng(1).normal(size=(11, 3))  # rest offsets from the head
    observed = joints[HEAD] + offsets @ head_world.T
    weights = np.ones(11)
    weights[[2, 7]] = 0.0
    observed[[2, 7]] += 0.5  # zero-weight points may be anywhere

    body_only = truth.copy()
    body_only[[NECK, HEAD]] = 0.0
    fit = fit_neck_to_points(body_only, transl, body, offsets, observed, weights, head_world)
    assert fit is not None and fit.points_used == 9

    placed = body_only.copy()
    placed[NECK], placed[HEAD] = fit.neck, fit.head
    got = forward_kinematics(placed, body) + transl
    assert np.linalg.norm(got[HEAD] - joints[HEAD]) < 1e-3
    np.testing.assert_allclose(fit.head_joint_target, joints[HEAD], atol=1e-9)
    got_world = global_rotations(placed, body.parents)[HEAD]
    assert Rotation.from_matrix(got_world.T @ head_world).magnitude() < 1e-6


@requires_smplx
def test_neck_fit_without_weighted_points_returns_none(body):
    from kinetic_augment.body_model.head_placement import fit_neck_to_points

    fit = fit_neck_to_points(
        np.zeros((55, 3)),
        np.zeros(3),
        body,
        np.zeros((11, 3)),
        np.zeros((11, 3)),
        np.zeros(11),
        np.eye(3),
    )
    assert fit is None
