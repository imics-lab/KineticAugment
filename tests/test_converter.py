"""
Tests for the V1 converter contract.

Contract: world-coordinate body landmarks + per-landmark validity
          -> SMPL-X parameters
          -> reconstructed world-coordinate body landmarks.

Reconstruction must not see the original landmarks. Everything needed to
rebuild them has to live in the returned parameters -- otherwise the round trip
is a subtraction and an addition, which is the failure mode of the displacement
encoding this replaces.

The torso cases cover what the earlier suite could not distinguish. A claim that
root orientation could not affect the round trip survived for a while precisely
because no test had a rotated root.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

try:
    import torch
    import smplx  # noqa: F401 -- imported to detect availability
    _smplx_ok = True
except ImportError:
    _smplx_ok = False

pytestmark = pytest.mark.skipif(not _smplx_ok, reason="torch/smplx not installed")

SMPLX_MODEL_PATH = "models"
NOSE_MP = 0
HIPS = [(1, 23), (2, 24)]   # (SMPL-X joint, MediaPipe landmark)
HEAD_JOINT = 15
LEFT_COLLAR = 13
SPINE_JOINTS = (3, 6, 9)
ARM_SEGMENTS_MP = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
# A body other than the template, using coefficients past the conventional ten.
BODY_BETAS = np.zeros(300)
BODY_BETAS[[0, 1, 2, 3, 40]] = [1.5, -1.0, 0.5, 0.8, 2.0]


@pytest.fixture(scope="module")
def skeleton():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton
    return SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)


def landmarks_from(skeleton, rotations, offset=(0.0, 0.0, 0.0), with_nose=False,
                   with_hips=False):
    """
    Pose SMPL-X and lay the corresponded joints out as MediaPipe landmarks.

    The nose is posed as a point rigidly attached to the head joint. On the real
    mesh it is a skinned vertex; the two coincide while neck and head rotations
    are zero, which holds for every case here and is what the fitter assumes.
    """
    from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE
    from kinetic_augment.body_model.kinematics import forward_kinematics

    extended = SimpleNamespace(
        rest_joints=np.vstack([skeleton.rest_joints, skeleton.rest_nose]),
        parents=np.append(skeleton.parents, HEAD_JOINT),
        num_joints=skeleton.num_joints + 1,
    )
    posed = forward_kinematics(np.vstack([rotations, np.zeros((1, 3))]), extended)
    posed = posed + np.asarray(offset)

    landmarks = np.full((33, 3), np.nan)
    for smplx_idx, mp_idx in ARM_CORRESPONDENCE:
        landmarks[mp_idx] = posed[smplx_idx]
    if with_nose:
        landmarks[NOSE_MP] = posed[-1]
    if with_hips:
        for smplx_idx, mp_idx in HIPS:
            landmarks[mp_idx] = posed[smplx_idx]
    return landmarks, np.isfinite(landmarks).all(axis=1)


def random_arm_pose(seed):
    from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES
    rng = np.random.default_rng(seed)
    rots = np.zeros((55, 3))
    for joint, _, _, _ in SMPLX_ARM_BONES:
        rots[joint] = rng.normal(0, 0.3, 3)
    return rots


def rotvec(degrees_xyz):
    return Rotation.from_euler("xyz", degrees_xyz, degrees=True).as_rotvec()


def assert_arms_match(actual, expected, atol):
    from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE
    for _, mp_idx in ARM_CORRESPONDENCE:
        np.testing.assert_allclose(actual[mp_idx], expected[mp_idx], atol=atol)


def round_trip(skeleton, landmarks, validity):
    from kinetic_augment.body_model.converter import fit_body, reconstruct_body
    return reconstruct_body(fit_body(landmarks, validity, skeleton), skeleton)


class TestContract:
    def test_reconstruction_does_not_take_the_original_landmarks(self):
        from kinetic_augment.body_model.converter import reconstruct_body
        params = list(inspect.signature(reconstruct_body).parameters)
        assert params == ["fit", "skeleton"]

    def test_the_v1_converter_is_the_packages_public_api(self):
        # The deprecated classes are what the package advertised; consumers should
        # find the converter that replaced them at the same import path.
        import kinetic_augment.body_model as body_model
        from kinetic_augment.body_model import converter, shape, skeleton

        assert body_model.fit_shape is shape.fit_shape
        assert body_model.fit_body is converter.fit_body
        assert body_model.reconstruct_body is converter.reconstruct_body
        assert body_model.SMPLXSkeleton is skeleton.SMPLXSkeleton
        assert {"fit_shape", "fit_body", "reconstruct_body", "BodyFit",
                "SMPLXSkeleton"} <= set(body_model.__all__)

    def test_fit_returns_genuine_smplx_parameter_shapes(self, skeleton):
        from kinetic_augment.body_model.converter import fit_body
        landmarks, validity = landmarks_from(skeleton, random_arm_pose(0))

        fit = fit_body(landmarks, validity, skeleton)

        assert fit.body_pose.shape == (63,)
        assert fit.global_orient.shape == (3,)
        assert fit.transl.shape == (3,)
        assert fit.betas.shape == (skeleton.model.num_betas,)


class TestRoundTrip:
    def test_exact_with_unrotated_root(self, skeleton):
        landmarks, validity = landmarks_from(
            skeleton, random_arm_pose(1), offset=(0.3, -0.8, 2.1))

        out = round_trip(skeleton, landmarks, validity)

        assert_arms_match(out, landmarks, atol=1e-9)

    def test_exact_with_rotated_root(self, skeleton):
        truth = random_arm_pose(4)
        truth[0] = Rotation.from_euler("y", 30, degrees=True).as_rotvec()
        landmarks, validity = landmarks_from(skeleton, truth)

        out = round_trip(skeleton, landmarks, validity)

        assert_arms_match(out, landmarks, atol=1e-9)

    def test_does_not_reconstruct_what_it_does_not_model(self, skeleton):
        # Face, hands and legs are outside V1. Reconstructing them would mean
        # inventing values; returning NaN states that honestly.
        from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE

        landmarks, validity = landmarks_from(skeleton, random_arm_pose(2), with_nose=True)
        out = round_trip(skeleton, landmarks, validity)

        modelled = {mp for _, mp in ARM_CORRESPONDENCE}
        for i in range(33):
            if i not in modelled:
                assert np.all(np.isnan(out[i])), f"landmark {i} should not be reconstructed"

    @pytest.mark.parametrize("shape", ["template", "fitted"])
    def test_smplx_forward_pass_reproduces_the_reconstruction(self, skeleton, shape):
        # Requirement 2 of the scope: the returned parameters must mean to SMPL-X
        # exactly what they mean to us. A rotated root makes global_orient and its
        # pivot compensation non-trivial; a fitted shape makes betas non-trivial.
        from kinetic_augment.body_model.converter import (
            ARM_CORRESPONDENCE, fit_body, reconstruct_body)

        body = skeleton if shape == "template" else skeleton.shaped(BODY_BETAS)
        truth = random_arm_pose(3)
        truth[0] = rotvec([20, -35, 10])
        landmarks, validity = landmarks_from(
            body, truth, offset=(-0.4, 0.2, 1.5), with_nose=True)
        fit = fit_body(landmarks, validity, body)
        ours = reconstruct_body(fit, skeleton)

        def t(a):
            return torch.tensor(a, dtype=torch.float32)[None]

        with torch.no_grad():
            joints = skeleton.model(
                body_pose=t(fit.body_pose), global_orient=t(fit.global_orient),
                transl=t(fit.transl), betas=t(fit.betas), return_verts=False,
            ).joints[0].numpy()

        for smplx_idx, mp_idx in ARM_CORRESPONDENCE:
            np.testing.assert_allclose(ours[mp_idx], joints[smplx_idx], atol=1e-5)


class TestTorso:
    def test_exact_with_spine_bend(self, skeleton):
        # With the collars at zero, spine rotations carry both shoulders rigidly,
        # so a root rotation plus translation reproduces them exactly.
        truth = random_arm_pose(10)
        for joint, angles in zip(SPINE_JOINTS, ([10, 0, 5], [5, 10, 0], [0, -8, 6])):
            truth[joint] = rotvec(angles)
        landmarks, validity = landmarks_from(skeleton, truth, offset=(0.1, 0.2, 0.3))

        out = round_trip(skeleton, landmarks, validity)

        assert_arms_match(out, landmarks, atol=1e-9)

    def test_nose_recovers_the_true_root_rotation(self, skeleton):
        from kinetic_augment.body_model.converter import fit_body

        truth = random_arm_pose(11)
        truth[0] = rotvec([25, 30, -10])
        landmarks, validity = landmarks_from(skeleton, truth, with_nose=True)

        fit = fit_body(landmarks, validity, skeleton)

        error = Rotation.from_rotvec(fit.global_orient).inv() * Rotation.from_rotvec(truth[0])
        assert error.magnitude() < 1e-9

    def test_torso_pitch_does_not_affect_arm_reconstruction(self, skeleton):
        # Rotation about the shoulder axis moves neither shoulder, and the arms are
        # solved in world directions, so no arm landmark can constrain it. The nose
        # fixes it for anatomical plausibility only; arm reconstruction must be
        # identical with or without the nose.
        from kinetic_augment.body_model.converter import fit_body, reconstruct_body

        truth = random_arm_pose(12)
        truth[0] = rotvec([25, 30, 0])
        landmarks, with_nose = landmarks_from(skeleton, truth, with_nose=True)
        without_nose = with_nose.copy()
        without_nose[NOSE_MP] = False

        fit_a = fit_body(landmarks, with_nose, skeleton)
        fit_b = fit_body(landmarks, without_nose, skeleton)

        # Not vacuous: the two fits genuinely disagree about the root.
        disagreement = (Rotation.from_rotvec(fit_a.global_orient).inv()
                        * Rotation.from_rotvec(fit_b.global_orient)).magnitude()
        assert np.degrees(disagreement) > 5

        assert_arms_match(reconstruct_body(fit_a, skeleton),
                          reconstruct_body(fit_b, skeleton), atol=1e-9)

    def test_hips_recover_the_root_when_the_nose_is_missing(self, skeleton):
        # A hand over the face hides the nose. The hips sit on the torso too, so
        # they still say which way is up.
        from kinetic_augment.body_model.converter import fit_body

        truth = random_arm_pose(15)
        truth[0] = rotvec([20, -40, 15])
        landmarks, validity = landmarks_from(skeleton, truth, offset=(0.4, -0.3, 1.8),
                                             with_hips=True)

        fit = fit_body(landmarks, validity, skeleton)

        error = Rotation.from_rotvec(fit.global_orient).inv() * Rotation.from_rotvec(truth[0])
        assert error.magnitude() < 1e-9

    def test_without_nose_or_hips_a_mediapipe_body_is_not_turned_upside_down(self, skeleton):
        # MediaPipe's world frame is y-down with the subject facing -z, so a frontal
        # SMPL-X body sits in it turned 180 degrees about the shoulder axis -- the
        # one rotation the shoulders cannot reveal. Its origin is the hip centre,
        # which still holds the answer when no landmark beyond the arms is valid.
        from kinetic_augment.body_model.converter import fit_body

        truth = random_arm_pose(16)
        truth[0] = (Rotation.from_euler("x", 180, degrees=True)
                    * Rotation.from_euler("y", 15, degrees=True)).as_rotvec()
        landmarks, validity = landmarks_from(skeleton, truth, with_hips=True)
        landmarks = landmarks - 0.5 * (landmarks[23] + landmarks[24])
        validity[[23, 24]] = False

        fit = fit_body(landmarks, validity, skeleton)

        error = Rotation.from_rotvec(fit.global_orient).inv() * Rotation.from_rotvec(truth[0])
        assert error.magnitude() < 1e-9

    @pytest.mark.xfail(
        strict=True,
        reason="V1 holds the collar joints at zero. A shrug moves one shoulder about its "
               "collar joint, changing the shoulder geometry in a way no root rotation can "
               "reproduce. strict=True so the marker is removed if collars are ever fitted.",
    )
    def test_collar_motion_round_trips_exactly(self, skeleton):
        truth = random_arm_pose(7)
        truth[LEFT_COLLAR] = rotvec([0, 0, 20])
        landmarks, validity = landmarks_from(skeleton, truth)

        out = round_trip(skeleton, landmarks, validity)

        assert_arms_match(out, landmarks, atol=1e-6)

    def test_torso_rotation_falls_back_when_the_apex_is_degenerate(self):
        # An apex lying on the shoulder line says nothing about rotation around
        # that line. The estimator must fall back rather than divide by zero.
        from kinetic_augment.body_model.converter import torso_rotation

        left, right = np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
        rotation = torso_rotation(left, right, left, right,
                                  rest_apex=np.array([3.0, 0.0, 0.0]),
                                  obs_apex=np.array([5.0, 0.0, 0.0]))

        assert np.all(np.isfinite(rotation))
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)


class TestBodyShape:
    def test_different_shape_keeps_bone_directions_exact(self, skeleton):
        # A length mismatch must not be absorbed by bending a bone toward the
        # observed joint. Directions stay exact and the mismatch appears honestly
        # as position error, which is what a shape fit can later act on.
        from kinetic_augment.body_model.skeleton import SMPLXSkeleton

        shaped = SMPLXSkeleton(model_path=SMPLX_MODEL_PATH,
                               betas=np.array([1.5, -1.0, 0.5, 0.8, -0.3, 0, 0, 0, 0, 0]))
        truth = random_arm_pose(8)
        truth[0] = rotvec([0, 25, 0])
        landmarks, validity = landmarks_from(shaped, truth)

        out = round_trip(skeleton, landmarks, validity)

        for parent, child in ARM_SEGMENTS_MP:
            got = out[child] - out[parent]
            want = landmarks[child] - landmarks[parent]
            np.testing.assert_allclose(got / np.linalg.norm(got),
                                       want / np.linalg.norm(want), atol=1e-9)

        # Not vacuous: the shapes really differ, so positions cannot all match.
        worst = max(np.linalg.norm(out[m] - landmarks[m]) for _, m in ARM_SEGMENTS_MP)
        assert worst > 1e-3

    def test_fit_records_the_skeletons_shape(self, skeleton):
        from kinetic_augment.body_model.converter import fit_body

        body = skeleton.shaped(BODY_BETAS)
        landmarks, validity = landmarks_from(body, random_arm_pose(13))

        fit = fit_body(landmarks, validity, body)

        np.testing.assert_array_equal(fit.betas, BODY_BETAS)

    def test_parameters_alone_carry_the_shape(self, skeleton):
        # Fitted on the shaped body, reconstructed against the bare template: the
        # shape can only reach the reconstruction through fit.betas.
        from kinetic_augment.body_model.converter import fit_body, reconstruct_body

        body = skeleton.shaped(BODY_BETAS)
        truth = random_arm_pose(14)
        truth[0] = rotvec([0, 40, 0])
        landmarks, validity = landmarks_from(body, truth, offset=(0.2, -0.1, 1.0))

        out = reconstruct_body(fit_body(landmarks, validity, body), skeleton)

        assert_arms_match(out, landmarks, atol=1e-9)

    def test_reconstruction_refuses_a_shape_its_skeleton_cannot_apply(self, skeleton):
        # A skeleton with no shape space would otherwise drop the betas silently.
        from kinetic_augment.body_model.converter import BodyFit, reconstruct_body

        bare = SimpleNamespace(rest_joints=skeleton.rest_joints, parents=skeleton.parents,
                               num_joints=skeleton.num_joints, rest_nose=skeleton.rest_nose)
        fit = BodyFit(body_pose=np.zeros(63), global_orient=np.zeros(3),
                      transl=np.zeros(3), betas=BODY_BETAS)

        with pytest.raises(ValueError):
            reconstruct_body(fit, bare)


class TestValidity:
    def test_landmarks_it_could_not_place_come_back_missing(self, skeleton):
        # An unobserved joint leaves its bone unsolved, at the rest pose. Returning
        # that position would invent a landmark, which augmented data would inherit.
        landmarks, validity = landmarks_from(skeleton, random_arm_pose(15))
        validity = validity.copy()
        validity[15] = False   # left wrist: the left forearm cannot be solved
        validity[14] = False   # right elbow: nor can anything below the right shoulder

        out = round_trip(skeleton, landmarks, validity)

        for mp_idx in (15, 14, 16):
            assert np.all(np.isnan(out[mp_idx])), f"landmark {mp_idx} was invented"

    def test_an_orphaned_landmark_does_not_disturb_the_rest(self, skeleton):
        # A valid wrist below an invalid elbow cannot be placed. Letting its residual
        # into the translation would shift every landmark that was placed exactly.
        truth = random_arm_pose(16)
        truth[0] = rotvec([0, 30, 10])
        landmarks, validity = landmarks_from(skeleton, truth)
        validity = validity.copy()
        validity[14] = False   # right elbow invalid, right wrist still valid

        out = round_trip(skeleton, landmarks, validity)

        for mp_idx in (11, 12, 13, 15):
            np.testing.assert_allclose(out[mp_idx], landmarks[mp_idx], atol=1e-9)

    def test_invalid_landmarks_are_excluded_from_the_fit(self, skeleton):
        landmarks, validity = landmarks_from(skeleton, random_arm_pose(5))
        corrupted = landmarks.copy()
        corrupted[15] = [99.0, 99.0, 99.0]   # left wrist: wildly wrong...
        validity = validity.copy()
        validity[15] = False                  # ...but flagged invalid

        clean = round_trip(skeleton, landmarks, validity)
        dirty = round_trip(skeleton, corrupted, validity)

        np.testing.assert_allclose(dirty[11], clean[11], atol=1e-9)

    def test_rotated_root_with_a_missing_wrist_reconstructs_the_rest(self, skeleton):
        truth = random_arm_pose(9)
        truth[0] = rotvec([10, 35, -5])
        landmarks, validity = landmarks_from(skeleton, truth)
        validity = validity.copy()
        validity[15] = False   # left wrist unobserved

        out = round_trip(skeleton, landmarks, validity)

        for mp_idx in (11, 12, 13, 14, 16):
            np.testing.assert_allclose(out[mp_idx], landmarks[mp_idx], atol=1e-9)

    @pytest.mark.parametrize("root_deg", [(0, 0, 0), (10, 35, -5)])
    def test_an_unplaced_arm_hangs_at_the_side(self, skeleton, root_deg):
        # Left at the rest pose, an unseen arm would stick straight out (SMPL-X rests in a
        # T-pose). It hangs down the torso instead, tilted slightly outward.
        from kinetic_augment.body_model.converter import HANG_ABDUCTION_DEG, fit_body
        from kinetic_augment.body_model.kinematics import forward_kinematics

        truth = random_arm_pose(21)
        truth[0] = rotvec(root_deg)
        landmarks, validity = landmarks_from(skeleton, truth, with_nose=True)
        validity = validity.copy()
        validity[14] = False   # right elbow: the whole right arm is unplaced
        validity[15] = False   # left wrist: only the left forearm is unplaced

        fit = fit_body(landmarks, validity, skeleton)
        rotations = np.zeros((skeleton.num_joints, 3))
        rotations[0], rotations[1:22] = fit.global_orient, fit.body_pose.reshape(21, 3)
        posed = forward_kinematics(rotations, skeleton)
        root = Rotation.from_rotvec(fit.global_orient).as_matrix()

        a = np.radians(HANG_ABDUCTION_DEG)
        for joint, child, outward in ((17, 19, -1), (19, 21, -1), (18, 20, 1)):
            d = posed[child] - posed[joint]
            local = root.T @ (d / np.linalg.norm(d))   # in the rest torso frame
            np.testing.assert_allclose(local, [outward * np.sin(a), -np.cos(a), 0], atol=1e-9)
        # The observed left upper arm is untouched.
        d_obs = landmarks[13] - landmarks[11]
        d_fit = posed[18] - posed[16]
        np.testing.assert_allclose(
            d_fit / np.linalg.norm(d_fit), d_obs / np.linalg.norm(d_obs), atol=1e-9
        )

    def test_refuses_to_fit_without_both_shoulders(self, skeleton):
        # The shoulders anchor the torso frame. Without both, the root rotation is
        # undefined, and guessing it would be silent garbage.
        from kinetic_augment.body_model.converter import fit_body

        landmarks, validity = landmarks_from(skeleton, random_arm_pose(6))
        validity = validity.copy()
        validity[11] = False

        with pytest.raises(ValueError):
            fit_body(landmarks, validity, skeleton)
