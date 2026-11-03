"""
Tests for assembly: one SMPL-X pose from the body, hand and head fits.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.hand_assets import MODEL_DIR, requires_smplx

BETAS = np.zeros(300)
BETAS[[0, 1, 40]] = [1.0, -0.5, 1.5]


@pytest.fixture(scope="module")
def skeleton():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton
    return SMPLXSkeleton(model_path=MODEL_DIR)


def test_kabsch_recovers_a_rotation():
    from kinetic_augment.body_model.assembly import kabsch

    rng = np.random.default_rng(0)
    source = rng.normal(size=(6, 3))
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.9]).as_matrix()
    np.testing.assert_allclose(kabsch(source, source @ rotation.T + 5.0), rotation, atol=1e-10)


def test_body_rotations_layout():
    from kinetic_augment.body_model.assembly import body_rotations
    from kinetic_augment.body_model.converter import BodyFit

    fit = BodyFit(body_pose=np.arange(63.0), global_orient=np.array([1.0, 2.0, 3.0]),
                  transl=np.zeros(3))
    rotations = body_rotations(fit)
    assert rotations.shape == (55, 3)
    np.testing.assert_array_equal(rotations[0], [1, 2, 3])
    np.testing.assert_array_equal(rotations[1:22].ravel(), np.arange(63.0))
    assert not rotations[22:].any()


def test_attach_head_writes_neck_head_and_jaw():
    from kinetic_augment.body_model.assembly import attach_head

    head = SimpleNamespace(neck_pose=np.full(3, 0.1), head_pose=np.full(3, 0.2),
                           jaw_pose=np.array([0.3, 0, 0]))
    rotations = attach_head(np.zeros((55, 3)), head)
    np.testing.assert_array_equal(rotations[12], 0.1)
    np.testing.assert_array_equal(rotations[15], 0.2)
    np.testing.assert_array_equal(rotations[22], [0.3, 0, 0])


@requires_smplx
def test_pose_mesh_joints_match_forward_kinematics(skeleton):
    from kinetic_augment.body_model.assembly import pose_mesh
    from kinetic_augment.body_model.kinematics import forward_kinematics

    rng = np.random.default_rng(1)
    rotations = 0.3 * rng.normal(size=(55, 3))
    transl = np.array([0.1, -0.2, 0.3])
    mesh = pose_mesh(skeleton, rotations, BETAS, transl)
    expected = forward_kinematics(rotations, skeleton.shaped(BETAS)) + transl
    np.testing.assert_allclose(mesh.joints, expected, atol=1e-4)
    assert mesh.vertices.shape == (10475, 3)


@requires_smplx
@pytest.mark.parametrize("side", ["left", "right"])
def test_attach_hands_turns_the_palm_onto_the_target(skeleton, side):
    from kinetic_augment.body_model.assembly import (
        HAND_SLICE, PALM, WRIST, attach_hands, global_rotations,
    )
    from kinetic_augment.body_model.hand_converter import HandFit
    from kinetic_augment.body_model.hand_model import HandModel

    other = "right" if side == "left" else "left"
    models = {s: HandModel("smplx", s, model_dir=MODEL_DIR) for s in ("left", "right")}
    import torch

    hand_pose = 0.2 * np.random.default_rng(2).normal(size=45)
    with torch.no_grad():
        rest = models[side].points(np.zeros(3), hand_pose, np.zeros(10))[0].numpy()
    world = Rotation.from_rotvec([0.4, 1.1, -0.3]).as_matrix()
    target = rest @ world.T + np.array([0.3, 0.1, 0.0])
    fits = {
        side: HandFit("smplx", side, np.zeros(3), hand_pose, np.zeros(10), np.zeros(3)),
        other: HandFit("smplx", other, np.full(3, np.nan), np.full(45, np.nan),
                       np.full(10, np.nan), np.full(3, np.nan), status="no_wrist"),
    }
    recon = {side: target, other: np.full((21, 3), np.nan)}
    body = 0.2 * np.random.default_rng(3).normal(size=(55, 3))
    body[22:] = 0.0
    rotations, posed = attach_hands(body, skeleton.parents, fits, recon, models)
    assert posed == {side: True, other: False}
    wrist_world = global_rotations(rotations, skeleton.parents)[WRIST[side]]
    np.testing.assert_allclose(wrist_world, world, atol=1e-6)
    np.testing.assert_allclose(rotations[HAND_SLICE[side]].ravel(), hand_pose)
    assert not rotations[HAND_SLICE[other]].any()
    np.testing.assert_array_equal(rotations[WRIST[other]], body[WRIST[other]])
    assert PALM == (0, 5, 9, 13, 17)
