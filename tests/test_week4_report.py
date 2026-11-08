"""Week-4 report measures: identity, a known wrist move, and the unaligned key set."""

import dataclasses

import numpy as np
import pytest

from kinetic_augment.augmentations.outputs import build_outputs
from kinetic_augment.augmentations.report import ALIGNED_ONLY, measure
from tests.clip_fixtures import get_skeleton, synthetic_clip
from tests.hand_assets import requires_smplx


def base_outputs():
    return build_outputs(synthetic_clip(frames=20), get_skeleton())


@requires_smplx
def test_measure_identity_zero():
    base = base_outputs()
    out = measure(base, base, aligned=True)
    assert out["wrist_moved_mm_max"] == 0.0
    assert out["bone_length_change_mm_max"] == 0.0
    assert out["face_rigidity_mm"] == 0.0
    assert out["jitter_ratio"] == pytest.approx(1.0)
    assert np.isfinite(out["hand_face_cm_min"])
    assert 0.0 <= out["in_frame_pct"] <= 100.0


@requires_smplx
def test_measure_detects_wrist_move():
    base = base_outputs()
    hands = base.hands_world.copy()
    hands[:, 0] += np.array([0.01, 0.0, 0.0])  # the left hand, 1 cm
    pose = base.pose_world.copy()
    pose[:, 15] += np.array([0.01, 0.0, 0.0])
    aug = dataclasses.replace(base, hands_world=hands, pose_world=pose)
    out = measure(base, aug, aligned=True)
    assert out["wrist_moved_mm_max"] == pytest.approx(10.0, abs=1e-6)
    assert out["bone_length_change_mm_max"] > 0.0  # the forearm bone 13-15 changed


@requires_smplx
def test_measure_unaligned_skips_frame_keys():
    base = base_outputs()
    out = measure(base, base, aligned=False)
    assert not set(ALIGNED_ONLY) & set(out)
    assert {"jitter_ratio", "face_rigidity_mm", "hand_face_cm_min", "in_frame_pct"} <= set(out)
