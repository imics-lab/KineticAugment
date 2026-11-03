"""
Tests for the head converter (design §4): occlusion, head rotation, jaw, neck/head split, fit_head.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.head_converter import (
    HeadRotation, head_rotation, occluded, points_in_polygon,
)

PIXELS_PER_METRE = 4000.0


def synthetic_face(seed=0, noise_px=0.0):
    rng = np.random.default_rng(seed)
    template = 0.04 * rng.normal(size=(17, 3))
    rotation = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    observed = PIXELS_PER_METRE * template @ rotation.T + [320.0, 200.0, 50.0]
    observed += noise_px * rng.normal(size=observed.shape)
    return template, observed, rotation


def angle_deg(a, b):
    return np.degrees(Rotation.from_matrix(a.T @ b).magnitude())


def test_head_rotation_recovers_the_rotation():
    template, observed, rotation = synthetic_face(noise_px=0.5)
    fit = head_rotation(template, observed, np.ones(17), reference_length=250.0)
    assert isinstance(fit, HeadRotation)
    assert angle_deg(fit.rotation, rotation) < 0.5
    assert fit.points_used == 17
    assert fit.rms_fraction < 0.01


def test_outliers_are_dropped_in_the_robust_pass():
    template, observed, rotation = synthetic_face(noise_px=0.5)
    observed[[2, 7]] += 30.0
    fit = head_rotation(template, observed, np.ones(17), reference_length=250.0)
    assert fit.points_used == 15
    assert angle_deg(fit.rotation, rotation) < 0.5


def test_noise_free_points_are_all_kept():
    template, observed, _ = synthetic_face()
    assert head_rotation(template, observed, np.ones(17), 250.0).points_used == 17


def test_too_few_points_give_nan():
    template, observed, _ = synthetic_face()
    weights = np.zeros(17)
    weights[:2] = 1.0
    fit = head_rotation(template, observed, weights, 250.0)
    assert np.isnan(fit.rotation).all() and fit.points_used == 2


def test_occluded_marks_inside_and_near_points():
    square = [np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])]
    points = np.array([[50.0, 50.0], [105.0, 50.0], [120.0, 50.0], [-3.0, -3.0]])
    np.testing.assert_array_equal(occluded(points, square, pad=10.0),
                                  [True, True, False, True])
    assert not occluded(points, [], pad=10.0).any()


def test_points_in_polygon_convex_square():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    points = np.array([[5.0, 5.0], [15.0, 5.0], [5.0, -1.0], [-1.0, 5.0]])
    np.testing.assert_array_equal(points_in_polygon(points, square),
                                  [True, False, False, False])


def test_points_in_polygon_non_convex_u_shape():
    u_shape = np.array([[0.0, 0.0], [9.0, 0.0], [9.0, 9.0], [6.0, 9.0], [6.0, 3.0],
                        [3.0, 3.0], [3.0, 9.0], [0.0, 9.0]])
    points = np.array([[1.5, 6.0],   # left arm
                       [7.5, 6.0],   # right arm
                       [4.5, 6.0],   # notch between the arms
                       [4.5, 1.5],   # base
                       [4.5, 12.0]])  # above everything
    np.testing.assert_array_equal(points_in_polygon(points, u_shape),
                                  [True, True, False, True, False])


def test_points_in_polygon_empty_points():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert points_in_polygon(np.empty((0, 2)), square).shape == (0,)


from tests.face_assets import MODEL_DIR, requires_face_assets  # noqa: E402

BETAS = np.zeros(300)
BETAS[[0, 2, 7]] = [0.8, -0.6, 1.2]


@pytest.fixture(scope="module")
def body():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton
    return SMPLXSkeleton(model_path=MODEL_DIR).shaped(BETAS)


@pytest.fixture(scope="module")
def corr():
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    return load_face_correspondence(MODEL_DIR)


@pytest.fixture(scope="module")
def curve(body, corr):
    from kinetic_augment.body_model.head_converter import jaw_curve
    return jaw_curve(body, corr)


@requires_face_assets
def test_jaw_curve_opens_with_positive_pitch(curve):
    from kinetic_augment.body_model.head_converter import JAW_GRID_DEG

    assert len(curve.pitch_rad) == len(JAW_GRID_DEG)
    assert np.all(np.diff(curve.gap_ratio) >= 0)
    assert curve.gap_ratio[40] > curve.gap_ratio[0] + 0.05  # 10 degrees opens the lips visibly


@requires_face_assets
def test_fit_jaw_pitch_recovers_a_posed_jaw(body, corr, curve):
    from kinetic_augment.body_model.assembly import JAW, pose_mesh
    from kinetic_augment.body_model.head_converter import JAW_AXIS, fit_jaw_pitch, lip_gap_ratio

    rotations = np.zeros((55, 3))
    rotations[JAW] = np.radians(12.3) * JAW_AXIS
    vertices = pose_mesh(body, rotations, body.betas, np.zeros(3)).vertices
    upper, lower, right, left = corr.points(vertices, (13, 14, 33, 263))
    pitch = fit_jaw_pitch(lip_gap_ratio(upper, lower, right, left), curve)
    assert abs(np.degrees(pitch) - 12.3) < 1.0


@requires_face_assets
def test_fit_jaw_pitch_clamps_to_the_grid(curve):
    from kinetic_augment.body_model.head_converter import fit_jaw_pitch

    assert fit_jaw_pitch(-1.0, curve) == 0.0
    assert np.isclose(np.degrees(fit_jaw_pitch(10.0, curve)), 30.0)


SCALE, OFFSET = 3000.0, np.array([300.0, 150.0, 40.0])


def posed_body_rotations():
    """A y-down world like MediaPipe's: the body turned upside down, slightly rotated."""
    rotations = np.zeros((55, 3))
    rotations[0] = (Rotation.from_rotvec([np.pi, 0, 0])
                    * Rotation.from_rotvec([0.05, 0.3, -0.04])).as_rotvec()
    rotations[16:22] = 0.3 * np.random.default_rng(4).normal(size=(6, 3))  # arms
    return rotations


def observe(body, corr, rotations, transl):
    """Pixel landmarks (478, 3; NaN off the embedding) and the world ear midpoint."""
    from kinetic_augment.body_model.assembly import pose_mesh
    from kinetic_augment.body_model.head_converter import EAR_VERTICES

    mesh = pose_mesh(body, rotations, body.betas, transl)
    face_px = np.full((478, 3), np.nan)
    face_px[corr.mp_indices] = SCALE * corr.points(mesh.vertices) + OFFSET
    return face_px, mesh.vertices[list(EAR_VERTICES)].mean(axis=0), mesh


@pytest.fixture(scope="module")
def template(body, corr):
    from kinetic_augment.body_model.head_converter import head_template
    return head_template(body, corr)


@requires_face_assets
@pytest.mark.parametrize("with_ears", [True, False])
def test_fit_head_recovers_head_orientation_and_jaw(body, corr, template, with_ears):
    from kinetic_augment.body_model.assembly import HEAD, attach_head, global_rotations
    from kinetic_augment.body_model.head_converter import fit_head

    truth = posed_body_rotations()
    truth[12] = [0.15, 0.25, -0.05]
    truth[15] = [0.1, -0.2, 0.08]
    truth[22] = [np.radians(9.0), 0, 0]
    transl = np.array([0.02, -0.1, 0.3])
    face_px, ears, _ = observe(body, corr, truth, transl)
    body_only = truth.copy()
    body_only[[12, 15, 22]] = 0.0

    fit = fit_head(face_px, [], body_only, transl, template, ears if with_ears else None)
    assert fit.valid and fit.reliable and fit.jaw_reliable
    assert fit.neck_mode == ("swing" if with_ears else "split")
    got = global_rotations(attach_head(body_only, fit), body.parents)[HEAD]
    want = global_rotations(truth, body.parents)[HEAD]
    assert angle_deg(got, want) < 0.5
    assert abs(np.degrees(fit.jaw_pose[0]) - 9.0) < 1.0
    if with_ears:
        # The ear vertices are partly skinned to the neck and moved by pose correctives, so the
        # rigid-head ear model is approximate: millimetres, not exact.
        assert fit.ear_residual_m < 0.005


@requires_face_assets
def test_neck_swing_leaves_the_arms_alone(body, corr, template):
    from kinetic_augment.body_model.assembly import attach_head
    from kinetic_augment.body_model.head_converter import fit_head
    from kinetic_augment.body_model.kinematics import forward_kinematics

    truth = posed_body_rotations()
    truth[12] = [0.3, 0.0, 0.1]
    face_px, ears, _ = observe(body, corr, truth, np.zeros(3))
    body_only = truth.copy()
    body_only[12] = 0.0
    fit = fit_head(face_px, [], body_only, np.zeros(3), template, ears)
    before = forward_kinematics(body_only, body)[16:22]
    after = forward_kinematics(attach_head(body_only, fit), body)[16:22]
    np.testing.assert_allclose(after, before, atol=1e-12)


@requires_face_assets
def test_no_face_gives_an_invalid_rest_fit(template):
    from kinetic_augment.body_model.head_converter import fit_head

    fit = fit_head(None, [], posed_body_rotations(), np.zeros(3), template)
    assert not fit.valid and not fit.reliable and not fit.jaw_reliable
    assert not fit.neck_pose.any() and not fit.head_pose.any() and not fit.jaw_pose.any()


@requires_face_assets
def test_a_hand_over_the_mouth_keeps_the_jaw_at_rest(body, corr, template):
    from kinetic_augment.body_model.head_converter import fit_head

    truth = posed_body_rotations()
    truth[22] = [np.radians(15.0), 0, 0]
    face_px, ears, _ = observe(body, corr, truth, np.zeros(3))
    mouth = face_px[[13, 14], :2].mean(axis=0)
    square = [mouth + np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])]
    fit = fit_head(face_px, square, truth, np.zeros(3), template, ears)
    assert not fit.jaw_reliable and not fit.jaw_pose.any()


@requires_face_assets
def test_a_hand_over_most_of_the_face_is_unreliable(body, corr, template):
    from kinetic_augment.body_model.head_converter import fit_head

    face_px, ears, _ = observe(body, corr, posed_body_rotations(), np.zeros(3))
    lo, hi = np.nanmin(face_px[:, :2], axis=0), np.nanmax(face_px[:, :2], axis=0)
    cover = [np.array([lo - 50, [hi[0] + 50, lo[1] - 50], hi + 50, [lo[0] - 50, hi[1] + 50]])]
    fit = fit_head(face_px, cover, posed_body_rotations(), np.zeros(3), template, ears)
    assert fit.valid and not fit.reliable
