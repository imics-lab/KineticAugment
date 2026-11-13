"""
Tests for similarity alignment (V1 round-trip, Step 5).

SMPL-X produces metric, pelvis-anchored coordinates; MediaPipe produces
image-normalised ones. Comparing them requires removing the similarity
(scale, rotation, translation) between the two spaces first. Absolute placement
is camera-dependent and is not information the conversion is meant to preserve.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.alignment import similarity_transform, apply_similarity


def sample_points():
    rng = np.random.default_rng(11)
    return rng.normal(0, 1, (8, 3))


class TestSimilarityTransform:
    def test_identity_when_already_aligned(self):
        x = sample_points()
        s, R, t = similarity_transform(x, x)
        assert s == pytest.approx(1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-9)

    def test_recovers_a_known_similarity(self):
        x = sample_points()
        true_s = 3.7
        true_R = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
        true_t = np.array([1.5, -2.0, 0.25])
        y = true_s * (x @ true_R.T) + true_t

        s, R, t = similarity_transform(x, y)

        assert s == pytest.approx(true_s)
        np.testing.assert_allclose(R, true_R, atol=1e-9)
        np.testing.assert_allclose(t, true_t, atol=1e-9)

    def test_applying_the_transform_reproduces_the_target(self):
        x = sample_points()
        true_R = Rotation.from_rotvec([0.1, 0.4, -0.3]).as_matrix()
        y = 2.25 * (x @ true_R.T) + np.array([-1.0, 0.5, 3.0])

        s, R, t = similarity_transform(x, y)

        np.testing.assert_allclose(apply_similarity(x, s, R, t), y, atol=1e-9)

    def test_never_produces_a_reflection(self):
        # A mirrored target must be fitted with a proper rotation, not a flip.
        x = sample_points()
        y = x * np.array([1.0, 1.0, -1.0])

        _, R, _ = similarity_transform(x, y)

        assert np.linalg.det(R) == pytest.approx(1.0)

class TestRigidTransform:
    """
    Both SMPL-X and MediaPipe world landmarks are metric, so the true scale
    between them is 1.0. A free scale parameter silently absorbs correspondence
    error, so the converter's own transform must be rigid.
    """

    def test_scale_is_exactly_one(self):
        x = sample_points()
        y = 4.0 * x + np.array([1.0, 2.0, 3.0])

        s, _, _ = similarity_transform(x, y, allow_scale=False)

        assert s == 1.0

    def test_recovers_a_known_rigid_transform(self):
        x = sample_points()
        true_R = Rotation.from_rotvec([0.2, 0.1, -0.4]).as_matrix()
        true_t = np.array([0.5, -1.5, 2.0])
        y = x @ true_R.T + true_t

        s, R, t = similarity_transform(x, y, allow_scale=False)

        assert s == 1.0
        np.testing.assert_allclose(R, true_R, atol=1e-9)
        np.testing.assert_allclose(t, true_t, atol=1e-9)
        np.testing.assert_allclose(apply_similarity(x, s, R, t), y, atol=1e-9)

    def test_leaves_residual_when_the_target_is_scaled(self):
        # The point of forbidding scale: a scaled target must show up as error
        # rather than being absorbed.
        x = sample_points()
        y = 1.3 * x

        rigid = apply_similarity(x, *similarity_transform(x, y, allow_scale=False))
        free = apply_similarity(x, *similarity_transform(x, y, allow_scale=True))

        assert np.linalg.norm(rigid - y) > 1e-3
        assert np.linalg.norm(free - y) < 1e-9

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            similarity_transform(np.zeros((5, 3)), np.zeros((4, 3)))

    def test_rejects_too_few_points(self):
        # Three non-collinear points are the minimum for a 3-D similarity.
        with pytest.raises(ValueError):
            similarity_transform(np.zeros((2, 3)), np.zeros((2, 3)))


def test_weights_none_matches_uniform_weights():
    from kinetic_augment.body_model.alignment import similarity_transform

    rng = np.random.default_rng(5)
    source, target = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    plain = similarity_transform(source, target)
    weighted = similarity_transform(source, target, weights=np.full(8, 3.0))
    np.testing.assert_allclose(plain[0], weighted[0])
    np.testing.assert_allclose(plain[1], weighted[1], atol=1e-12)
    np.testing.assert_allclose(plain[2], weighted[2], atol=1e-12)


def test_zero_weight_points_are_ignored():
    from kinetic_augment.body_model.alignment import similarity_transform
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(6)
    source = rng.normal(size=(10, 3))
    rotation = Rotation.from_rotvec([0.2, 0.5, -0.1]).as_matrix()
    target = 2.0 * source @ rotation.T + 1.0
    target[:3] += 50.0  # gross outliers
    weights = np.ones(10)
    weights[:3] = 0.0
    scale, fitted, translation = similarity_transform(source, target, weights=weights)
    np.testing.assert_allclose(scale, 2.0)
    np.testing.assert_allclose(fitted, rotation, atol=1e-10)


def test_too_few_weighted_points_raise():
    from kinetic_augment.body_model.alignment import similarity_transform

    with pytest.raises(ValueError, match="positively weighted"):
        similarity_transform(np.eye(3) * [1, 2, 3] + [[0, 0, 0]] * 3,
                             np.eye(3), weights=np.array([1.0, 1.0, 0.0]))
