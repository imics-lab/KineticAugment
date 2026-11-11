"""
Tests for the superseded image-space converters in kinetic_augment.legacy.

MediaPipeToSMPLX / SMPLXToMediaPipe are pure numpy: they store bone vectors in SMPL-X-shaped
slots and add them back onto the original landmarks. They need neither torch nor the SMPL-X
model files, so (unlike the SMPLXWrapper tests) these tests have no skip conditions.
"""

import warnings

import numpy as np
import pytest

from kinetic_augment.legacy.mp_to_smplx import MediaPipeToSMPLX
from kinetic_augment.legacy.smplx_to_mp import SMPLXToMediaPipe


@pytest.fixture
def landmarks():
    rng = np.random.default_rng(0)
    return (rng.standard_normal((543, 3)) * 0.1).astype(np.float32)


@pytest.fixture
def fitter():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return MediaPipeToSMPLX()


@pytest.fixture
def projector():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return SMPLXToMediaPipe()


class TestMediaPipeToSMPLX:
    def test_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="converter"):
            MediaPipeToSMPLX()

    def test_fit_single_frame(self, fitter, landmarks):
        params = fitter.fit(landmarks)  # (543, 3) is promoted to a batch of 1
        assert params["body_pose"].shape == (1, 63)
        assert params["global_orient"].shape == (1, 3)
        assert params["betas"].shape == (1, 10)
        assert params["left_hand_pose"].shape == (1, 45)
        assert params["right_hand_pose"].shape == (1, 45)

    def test_fit_batch(self, fitter, landmarks):
        batch = np.stack([landmarks, landmarks + 0.01])
        params = fitter.fit(batch)
        assert params["body_pose"].shape == (2, 63)

    def test_absent_hands_encode_to_zero(self, fitter, landmarks):
        landmarks[468:] = 0.0  # LH + RH blocks missing
        params = fitter.fit(landmarks)
        assert not params["left_hand_pose"].any()
        assert not params["right_hand_pose"].any()


class TestSMPLXToMediaPipe:
    def test_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="converter"):
            SMPLXToMediaPipe()

    def test_project_full_frame(self, fitter, projector, landmarks):
        params = fitter.fit(landmarks)
        out = projector.project(params, landmarks[None])
        assert out.shape == (1, 543, 3)

    def test_project_pose_only(self, fitter, projector, landmarks):
        params = fitter.fit(landmarks)
        out = projector.project(params, landmarks[None], return_full_frame=False)
        assert out.shape == (1, 33, 3)

    def test_round_trip_is_identity(self, fitter, projector, landmarks):
        """Encoding then decoding against the same frame reproduces the body landmarks."""
        params = fitter.fit(landmarks)
        out = projector.project(params, landmarks[None])
        np.testing.assert_allclose(out[0, 11:17], landmarks[11:17], atol=1e-5)
