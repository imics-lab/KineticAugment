"""Focused checks for the real-clip attachment on/off comparison gate."""

import numpy as np

from scripts.validate_face_integration import compare_regression_values


def test_regression_comparison_requires_matching_masks_and_compares_finite_values():
    reference = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    candidate = reference.copy()
    valid = np.array([[True, True], [True, False]])

    passed = compare_regression_values(
        reference, candidate, valid[:, :, None], valid[:, :, None]
    )

    assert passed["pass"] is True
    assert passed["valid_mask_equal"] is True
    assert passed["finite_mask_equal"] is True
    assert passed["compared_finite_count"] == 9
    assert passed["max_abs_delta"] == 0.0


def test_regression_comparison_fails_when_hand_value_or_validity_is_lost():
    reference = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    candidate = reference.copy()
    reference_valid = np.ones((2, 2), dtype=bool)
    candidate_valid = reference_valid.copy()
    candidate_valid[0, 1] = False
    candidate[0, 1, 0] = np.nan

    failed = compare_regression_values(
        reference,
        candidate,
        reference_valid[:, :, None],
        candidate_valid[:, :, None],
    )

    assert failed["pass"] is False
    assert failed["valid_mask_equal"] is False
    assert failed["finite_mask_equal"] is False
    assert failed["compared_finite_count"] == 9


def test_regression_comparison_broadcasts_real_pipeline_array_shapes():
    body_ok = np.array([True, False, True])
    rotations = np.zeros((3, 55, 3, 3))
    rotation_result = compare_regression_values(rotations, rotations.copy(), body_ok, body_ok)
    assert rotation_result["pass"] is True
    assert rotation_result["compared_finite_count"] == 2 * 55 * 3 * 3

    translations = np.zeros((3, 3))
    translation_result = compare_regression_values(
        translations, translations.copy(), body_ok, body_ok
    )
    assert translation_result["pass"] is True
    assert translation_result["compared_finite_count"] == 2 * 3

    geometry_valid = np.array([[True, False], [False, False], [True, True]])
    distances = np.zeros((3, 2, 7))
    distance_result = compare_regression_values(
        distances,
        distances.copy(),
        geometry_valid[:, :, None],
        geometry_valid[:, :, None],
    )
    assert distance_result["pass"] is True
    assert distance_result["compared_finite_count"] == 3 * 7
