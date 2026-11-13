"""
Tests for per-clip SMPL-X body shape (V1 Step 10).

The converter reproduces bone directions exactly, so a skeleton enters the round
trip only through five arm proportions: shoulder width and four bone lengths.
For each frame the reconstruction error is linear in them, which makes a clip's
summed squared round-trip error an exact quadratic. These tests hold the shape
fit to that error, measured by the converter itself, and to SMPL-X's own shape
distribution as the definition of a plausible body.

Expected radii are chi-square quantiles: sqrt(chi2.ppf(q, df)).
"""

import numpy as np
import pytest

try:
    import torch  # noqa: F401
    import smplx  # noqa: F401
    _smplx_ok = True
except ImportError:
    _smplx_ok = False

pytestmark = pytest.mark.skipif(not _smplx_ok, reason="torch/smplx not installed")

SMPLX_MODEL_PATH = "models"
ARM_LANDMARKS = range(11, 17)
# A random draw from SMPL-X's shape prior: a typical body, |beta| = 16.1 over all
# 300 coefficients, whose arm proportions sit well inside the plausible region.
TYPICAL_BETAS = np.random.default_rng(1).normal(size=300)
# Arm proportions far outside SMPL-X's population, as MediaPipe's are.
UNUSUAL_BETAS = np.zeros(300)
UNUSUAL_BETAS[[0, 1, 2]] = [6.0, -6.0, 5.0]


@pytest.fixture(scope="module")
def skeleton():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton
    return SMPLXSkeleton(model_path=SMPLX_MODEL_PATH)


@pytest.fixture(scope="module")
def two_beta_skeleton():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton
    return SMPLXSkeleton(model_path=SMPLX_MODEL_PATH, num_betas=2)


def clip_from(bodies, seed):
    """One frame per body -- repeat a body for a rigid clip -- in random arm poses."""
    from kinetic_augment.body_model.converter import ARM_CORRESPONDENCE
    from kinetic_augment.body_model.kinematics import forward_kinematics
    from kinetic_augment.body_model.swing_ik import SMPLX_ARM_BONES

    rng = np.random.default_rng(seed)
    landmarks = np.full((len(bodies), 33, 3), np.nan)
    for t, body in enumerate(bodies):
        rotations = np.zeros((55, 3))
        rotations[0] = rng.normal(0, 0.5, 3)
        for joint, _, _, _ in SMPLX_ARM_BONES:
            rotations[joint] = rng.normal(0, 0.4, 3)
        posed = forward_kinematics(rotations, body) + rng.normal(0, 0.5, 3)
        for smplx_idx, mp_idx in ARM_CORRESPONDENCE:
            landmarks[t, mp_idx] = posed[smplx_idx]
    return landmarks, np.isfinite(landmarks).all(axis=-1)


def round_trip_errors(landmarks, validity, skeleton, betas):
    """The converter's own per-landmark round-trip errors over a clip, in metres."""
    from kinetic_augment.body_model.converter import fit_body, reconstruct_body

    body = skeleton.shaped(betas)
    errors = []
    for frame, ok in zip(landmarks, validity):
        if ok[11] and ok[12]:
            out = reconstruct_body(fit_body(frame, ok, body), skeleton)
            errors += [np.linalg.norm(out[m] - frame[m]) for m in ARM_LANDMARKS if ok[m]]
    return np.array(errors)


def round_trip_sse(landmarks, validity, skeleton, betas):
    return float(np.sum(round_trip_errors(landmarks, validity, skeleton, betas) ** 2))


class TestRoundTripQuadratic:
    def test_is_the_converters_own_squared_round_trip_error(self, skeleton):
        # A different body every frame makes the clip non-rigid, as MediaPipe's is.
        from kinetic_augment.body_model.shape import arm_proportions, round_trip_quadratic

        rng = np.random.default_rng(20)
        bodies = [skeleton.shaped(rng.normal(0, 1.0, 10)) for _ in range(6)]
        landmarks, validity = clip_from(bodies, seed=20)
        validity[2, 16] = False   # a missing wrist
        validity[4, 12] = False   # a missing shoulder: that frame cannot be fitted
        evaluator = rng.normal(0, 1.0, 10)

        quadratic = round_trip_quadratic(landmarks, validity)

        assert quadratic.frames == 5
        got = quadratic.error(arm_proportions(skeleton.shaped(evaluator).rest_joints))
        assert got == pytest.approx(
            round_trip_sse(landmarks, validity, skeleton, evaluator), rel=1e-9)


class TestFitShape:
    def test_round_trips_a_typical_body_losslessly(self, skeleton):
        from kinetic_augment.body_model.shape import fit_shape

        landmarks, validity = clip_from([skeleton.shaped(TYPICAL_BETAS)] * 8, seed=21)

        betas = fit_shape(landmarks, validity, skeleton)

        assert round_trip_errors(landmarks, validity, skeleton, betas).max() < 1e-6

    def test_prefers_the_most_typical_body_with_those_proportions(self, skeleton):
        # Five proportions cannot identify 300 coefficients. The prior settles the
        # rest: of all bodies with these proportions, the most typical is returned,
        # not the one that generated the data. A reported body is a convention.
        from kinetic_augment.body_model.shape import fit_shape

        landmarks, validity = clip_from([skeleton.shaped(TYPICAL_BETAS)] * 8, seed=21)

        betas = fit_shape(landmarks, validity, skeleton)

        assert np.linalg.norm(betas) < np.linalg.norm(TYPICAL_BETAS) / 4

    @pytest.mark.parametrize("plausibility, radius", [(0.95, 3.3272357), (0.99, 3.8841051)])
    def test_goes_no_further_than_the_plausible_region(self, skeleton, plausibility, radius):
        # Unusual proportions pull the fit to the boundary, and no further.
        from kinetic_augment.body_model.shape import fit_shape

        landmarks, validity = clip_from([skeleton.shaped(UNUSUAL_BETAS)] * 8, seed=22)

        betas = fit_shape(landmarks, validity, skeleton, plausibility=plausibility)

        assert np.linalg.norm(betas) == pytest.approx(radius, abs=1e-6)

    def test_is_the_lowest_round_trip_error_the_plausible_region_allows(
            self, skeleton, two_beta_skeleton):
        # Brute force over a two-coefficient shape space settles what the best
        # plausible body is, independently of the optimiser. Two observable
        # directions give radius sqrt(chi2.ppf(0.95, 2)). The grid is scored with
        # the quadratic, which the test above pins to the converter's own error.
        from kinetic_augment.body_model.shape import (
            arm_proportions, fit_shape, round_trip_quadratic)

        rng = np.random.default_rng(23)
        bodies = [skeleton.shaped(UNUSUAL_BETAS + rng.normal(0, 0.5, 300)) for _ in range(8)]
        landmarks, validity = clip_from(bodies, seed=23)
        quadratic = round_trip_quadratic(landmarks, validity)
        radius = 2.4477468

        def error(betas):
            return quadratic.error(arm_proportions(two_beta_skeleton.shaped(betas).rest_joints))

        betas = fit_shape(landmarks, validity, two_beta_skeleton)

        grid = [radius * r * np.array([np.cos(a), np.sin(a)])
                for r in np.linspace(0, 1, 50) for a in np.linspace(0, 2 * np.pi, 240)]
        assert error(betas) == pytest.approx(
            round_trip_sse(landmarks, validity, two_beta_skeleton, betas), rel=1e-9)
        assert np.linalg.norm(betas) <= radius + 1e-6
        assert error(betas) <= min(error(g) for g in grid)

    def test_invalid_landmarks_do_not_influence_the_shape(self, skeleton):
        from kinetic_augment.body_model.shape import fit_shape

        rng = np.random.default_rng(24)
        bodies = [skeleton.shaped(UNUSUAL_BETAS + rng.normal(0, 0.5, 300)) for _ in range(6)]
        landmarks, validity = clip_from(bodies, seed=24)
        validity[1, 15] = False          # a wrist flagged invalid...
        validity[3, 11] = False          # ...and a frame that cannot be fitted
        corrupted = landmarks.copy()
        corrupted[1, 15] = [9.0, 9.0, 9.0]
        corrupted[3] = 7.0

        np.testing.assert_array_equal(fit_shape(corrupted, validity, skeleton),
                                      fit_shape(landmarks, validity, skeleton))

    def test_a_clip_with_nothing_to_fit_keeps_the_template(self, skeleton):
        from kinetic_augment.body_model.shape import fit_shape

        landmarks, validity = clip_from([skeleton.shaped(UNUSUAL_BETAS)] * 3, seed=25)
        validity[:, 12] = False

        np.testing.assert_array_equal(fit_shape(landmarks, validity, skeleton), np.zeros(300))


class TestArmProportions:
    def test_reads_shoulder_width_then_left_and_right_arm_bones(self):
        from kinetic_augment.body_model.shape import arm_proportions

        rest = np.zeros((55, 3))
        rest[16], rest[17] = [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]     # shoulders
        rest[18], rest[20] = [0.0, -0.2, 0.0], [0.0, -0.45, 0.0]  # left elbow, wrist
        rest[19], rest[21] = [0.3, -0.3, 0.0], [0.3, -0.3, 0.4]   # right elbow, wrist

        np.testing.assert_allclose(arm_proportions(rest), [0.3, 0.2, 0.25, 0.3, 0.4])
