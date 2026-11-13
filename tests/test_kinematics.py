"""
Tests for the skeleton source and forward kinematics (V1 round-trip, Steps 2-3).

The synthetic-skeleton tests use a straight 3-bone chain of unit-length bones
along +x, so every expected value below is computable by hand. The SMPL-X tests
check that our forward kinematics agrees with the real model; if it does not,
the parameters we fit do not mean what the model thinks they mean.
"""

import numpy as np
import pytest

from kinetic_augment.body_model.skeleton import SyntheticSkeleton, SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics

SMPLX_MODEL_PATH = "models"

try:
    import torch
    import smplx  # noqa: F401 -- imported to detect availability
    _smplx_ok = True
except ImportError:
    _smplx_ok = False

requires_smplx = pytest.mark.skipif(not _smplx_ok, reason="torch/smplx not installed")


def rot_z(deg):
    """Axis-angle vector for a rotation of `deg` degrees about +z."""
    return np.array([0.0, 0.0, np.deg2rad(deg)], dtype=np.float64)


class TestSyntheticSkeleton:
    def test_exposes_consistent_shapes(self):
        s = SyntheticSkeleton()
        assert s.rest_joints.shape == (s.num_joints, 3)
        assert s.parents.shape == (s.num_joints,)
        assert len(s.names) == s.num_joints

    def test_root_has_no_parent(self):
        assert SyntheticSkeleton().parents[0] == -1

    def test_every_non_root_parent_precedes_its_child(self):
        # Forward kinematics walks the array in order, so a parent must already
        # be resolved by the time its child is reached.
        s = SyntheticSkeleton()
        for j in range(1, s.num_joints):
            assert s.parents[j] < j

    def test_chain_is_unit_bones_along_x(self):
        s = SyntheticSkeleton()
        np.testing.assert_allclose(
            s.rest_joints,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        )


class TestForwardKinematics:
    def test_zero_rotation_reproduces_rest_pose(self):
        s = SyntheticSkeleton()
        posed = forward_kinematics(np.zeros((s.num_joints, 3)), s)
        np.testing.assert_allclose(posed, s.rest_joints, atol=1e-12)

    def test_rotating_a_joint_does_not_move_that_joint(self):
        s = SyntheticSkeleton()
        rots = np.zeros((s.num_joints, 3))
        rots[1] = rot_z(90)
        posed = forward_kinematics(rots, s)
        np.testing.assert_allclose(posed[1], [1.0, 0.0, 0.0], atol=1e-12)

    def test_rotating_a_joint_moves_its_child(self):
        # Joint 1 turns 90 deg about z. Its child sits one unit along +x at rest,
        # so it swings to one unit along +y, offset by the parent at x=1.
        s = SyntheticSkeleton()
        rots = np.zeros((s.num_joints, 3))
        rots[1] = rot_z(90)
        posed = forward_kinematics(rots, s)
        np.testing.assert_allclose(posed[2], [1.0, 1.0, 0.0], atol=1e-12)

    def test_rotation_propagates_to_grandchild(self):
        # The same 90 deg at joint 1 must carry joint 3 with it: [3,0,0] -> [1,2,0].
        s = SyntheticSkeleton()
        rots = np.zeros((s.num_joints, 3))
        rots[1] = rot_z(90)
        posed = forward_kinematics(rots, s)
        np.testing.assert_allclose(posed[3], [1.0, 2.0, 0.0], atol=1e-12)

    def test_global_orient_rotates_whole_chain(self):
        s = SyntheticSkeleton()
        rots = np.zeros((s.num_joints, 3))
        rots[0] = rot_z(90)
        posed = forward_kinematics(rots, s)
        np.testing.assert_allclose(posed[3], [0.0, 3.0, 0.0], atol=1e-12)

    def test_rejects_wrong_rotation_count(self):
        s = SyntheticSkeleton()
        with pytest.raises(ValueError):
            forward_kinematics(np.zeros((s.num_joints + 1, 3)), s)


@requires_smplx
class TestSMPLXSkeleton:
    def test_exposes_the_55_model_joints(self):
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        assert s.num_joints == 55
        assert s.rest_joints.shape == (55, 3)

    def test_parents_match_the_model(self):
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        np.testing.assert_array_equal(s.parents[:22],
                                      [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                                       9, 9, 9, 12, 13, 14, 16, 17, 18, 19])

    def test_rest_bone_lengths_are_anatomically_sane(self):
        # Verified against the real model in Step 0 of the V1 scope.
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        upper_arm = np.linalg.norm(s.rest_joints[18] - s.rest_joints[16])
        forearm = np.linalg.norm(s.rest_joints[20] - s.rest_joints[18])
        assert upper_arm == pytest.approx(0.2676, abs=1e-3)
        assert forearm == pytest.approx(0.2531, abs=1e-3)


@requires_smplx
class TestForwardKinematicsMatchesSMPLX:
    """
    The critical validation: our FK must reproduce what SMPL-X itself computes.
    If these disagree, fitted parameters are not valid SMPL-X parameters.
    """

    def _smplx_joints(self, model, global_orient, body_pose):
        with torch.no_grad():
            out = model(
                global_orient=torch.tensor(global_orient, dtype=torch.float32)[None],
                body_pose=torch.tensor(body_pose, dtype=torch.float32)[None],
                return_verts=False,
            )
        return out.joints[0, :55].numpy()

    def test_matches_on_rest_pose(self):
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        expected = self._smplx_joints(s.model, np.zeros(3), np.zeros(63))
        ours = forward_kinematics(np.zeros((55, 3)), s)
        np.testing.assert_allclose(ours, expected, atol=1e-5)

    def test_matches_on_a_random_body_pose(self):
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        rng = np.random.default_rng(0)
        global_orient = rng.normal(0, 0.3, 3)
        body_pose = rng.normal(0, 0.3, 63)

        expected = self._smplx_joints(s.model, global_orient, body_pose)

        rots = np.zeros((55, 3))
        rots[0] = global_orient
        rots[1:22] = body_pose.reshape(21, 3)
        ours = forward_kinematics(rots, s)

        np.testing.assert_allclose(ours[:22], expected[:22], atol=1e-5)


@requires_smplx
class TestShapedSMPLXSkeleton:
    # Coefficients past the conventional first ten exercise the full shape space,
    # which the Step 10 shape fit uses.
    BETAS = np.zeros(300)
    BETAS[:5] = [1.5, -1.0, 0.5, 0.8, -0.3]
    BETAS[[50, 299]] = [2.0, -1.5]

    @staticmethod
    def _model_output(model, betas):
        with torch.no_grad():
            return model(
                betas=torch.tensor(betas, dtype=torch.float32)[None], return_verts=False,
            ).joints[0].numpy()

    def test_rest_joints_match_the_model_for_nonzero_betas(self):
        # The betas path was written in Step 2 and exercised by nothing until the
        # body-shape cases needed it.
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH, betas=self.BETAS)
        expected = self._model_output(s.model, self.BETAS)
        np.testing.assert_allclose(s.rest_joints, expected[:55], atol=1e-5)

    def test_rest_nose_is_the_models_own_nose_landmark(self):
        # SMPL-X appends the nose vertex as output joint 55. Pairing MediaPipe's
        # nose with it, rather than with the head joint, avoids a ~30 degree bias
        # in the direction the two points sit from the shoulders.
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH, betas=self.BETAS)
        expected = self._model_output(s.model, self.BETAS)
        np.testing.assert_allclose(s.rest_nose, expected[55], atol=1e-5)

    def test_reshaping_matches_the_model_and_leaves_the_original_alone(self):
        # Reconstruction re-shapes the skeleton from each fit's own betas, so a
        # shaped copy must be the model's body, and must not leak into the original.
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        template = s.rest_joints.copy()

        shaped = s.shaped(self.BETAS)

        expected = self._model_output(s.model, self.BETAS)
        np.testing.assert_allclose(shaped.rest_joints, expected[:55], atol=1e-5)
        np.testing.assert_allclose(shaped.rest_nose, expected[55], atol=1e-5)
        np.testing.assert_array_equal(s.rest_joints, template)

    def test_ten_betas_mean_the_same_body_in_the_full_shape_space(self):
        # Ten coefficients are the common convention. Missing coefficients are zero.
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        padded = np.zeros(300)
        padded[:10] = self.BETAS[:10]

        short = s.shaped(self.BETAS[:10])

        np.testing.assert_allclose(short.rest_joints,
                                   self._model_output(s.model, padded)[:55], atol=1e-5)

    def test_rejects_more_betas_than_the_shape_space_holds(self):
        s = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)
        with pytest.raises(ValueError):
            s.shaped(np.zeros(301))
