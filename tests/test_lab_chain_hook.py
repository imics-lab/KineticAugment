"""The chain's draw hook and profiles built from dicts (Augmentation Lab, task 1)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from kinetic_augment.augmentations.chain import AugmentationChain
from kinetic_augment.augmentations.profile import load_profile, profile_from_dict
from tests.clip_fixtures import get_skeleton, synthetic_clip
from tests.hand_assets import requires_smplx

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_profile_from_dict_matches_load_profile():
    path = CONFIGS / "slr_week4_default.yaml"
    data = yaml.safe_load(path.read_text())
    a, b = load_profile(path), profile_from_dict(data, source=str(path))
    assert a == b


def test_profile_from_dict_rejects_like_load_profile():
    data = yaml.safe_load((CONFIGS / "slr_week4_default.yaml").read_text())
    data.pop("seed")
    with pytest.raises(ValueError, match="seed"):
        profile_from_dict(data)


@requires_smplx
def test_hook_none_is_unchanged():
    clip, skeleton = synthetic_clip(frames=24), get_skeleton()
    profile = load_profile(CONFIGS / "slr_week4_ambitious.yaml")
    a = AugmentationChain(profile, skeleton).run(clip)
    b = AugmentationChain(profile, skeleton, draw_hook=None).run(clip)
    assert a.draws == b.draws
    assert np.array_equal(a.outputs.pose_world, b.outputs.pose_world, equal_nan=True)


@requires_smplx
def test_hook_replaces_draw_before_apply():
    clip, skeleton = synthetic_clip(frames=12), get_skeleton()
    profile = load_profile(CONFIGS / "slr_week4_default.yaml", enable=["body.location_jitter"])

    def hook(name, draw, aug):
        return {"offset_mm": [25.0, 0.0, 0.0]} if name == "body.location_jitter" else draw

    res = AugmentationChain(profile, skeleton, draw_hook=hook).run(clip)
    assert res.draws["body.location_jitter"]["offset_mm"] == [25.0, 0.0, 0.0]
