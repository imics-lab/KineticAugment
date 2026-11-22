"""The mirror blacklist is enforced: a blacklisted gloss is never mirrored."""

import csv
import os

import numpy as np
import pytest

from kinetic_augment.augmentations import chain as chain_module
from kinetic_augment.augmentations import tier1
from kinetic_augment.augmentations.blacklist import (
    CLIP_GLOSSES,
    DEFAULT_BLACKLIST,
    blacklist_category,
    gloss_for_clip,
    load_blacklist,
    resolve_path,
)
from kinetic_augment.augmentations.chain import AugmentationChain
from kinetic_augment.augmentations.clip import ChainContext
from kinetic_augment.augmentations.profile import load_profile
from kinetic_augment.augmentations.scene import Mirror
from tests.clip_fixtures import get_skeleton, synthetic_clip
from tests.hand_assets import requires_smplx

CONFIGS = resolve_path("configs")


class _Clip:
    def __init__(self, gloss):
        self.gloss = gloss


def _ctx(mirror):
    draw = tier1.Tier1Draw(mirror, 0.0, 0.0, 1.0, (), 1.0, (0.5, 0.5))
    return ChainContext(skeleton=None, parents=None, pivot=np.zeros(3), tier1_draw=draw)


def _sample(gloss, mirror, aug=None):
    clip = None if gloss is False else _Clip(gloss)
    return (aug or Mirror()).sample(np.random.default_rng(0), clip, _ctx(mirror))


def test_blacklisted_gloss_never_mirrors():
    for gloss in ("left", "LEFT", "Northwest", "heart attack"):
        for _ in range(50):
            assert _sample(gloss, True)["mirror"] is False
    assert _sample("east", True)["blocked"] == "blacklist:absolute_side"
    assert _sample("heart", True)["blocked"] == "blacklist:caution"


def test_non_blacklisted_unchanged():
    assert _sample("book", True) == {"mirror": True}
    assert _sample("book", False) == {"mirror": False}


def test_missing_gloss_allows_mirroring():
    unknown = {"mirror": True, "blacklist": "gloss_unknown"}
    assert _sample(None, True) == unknown
    assert _sample(False, True) == unknown  # no clip at all
    assert _sample(None, False) == {"mirror": False, "blacklist": "gloss_unknown"}
    assert _sample(None, True, Mirror(ranges={"blacklist": None})) == {"mirror": True}
    assert _sample("not-a-gloss", True) == {"mirror": True}


def test_blacklist_null_disables_and_custom_path(tmp_path):
    assert _sample("left", True, Mirror(ranges={"blacklist": None})) == {"mirror": True}
    custom = tmp_path / "bl.yaml"
    custom.write_text("categories:\n  mine: {glosses: [book]}\n")
    aug = Mirror(ranges={"blacklist": str(custom)})
    assert _sample("book", True, aug)["blocked"] == "blacklist:mine"
    assert _sample("left", True, aug) == {"mirror": True}


def test_malformed_blacklist_file_fails_at_load(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("categories: [unclosed\n")
    with pytest.raises(ValueError, match="scene.mirror.blacklist"):
        Mirror(ranges={"blacklist": str(bad)})
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("other: 1\n")
    with pytest.raises(ValueError, match="scene.mirror.blacklist"):
        Mirror(ranges={"blacklist": str(wrong)})


def test_blacklist_validation():
    with pytest.raises(ValueError, match="scene.mirror.blacklist"):
        Mirror(ranges={"blacklist": 3})
    with pytest.raises(ValueError, match="scene.mirror.blacklist"):
        Mirror(ranges={"blacklist": "configs/does_not_exist.yaml"})


def test_default_path_resolves_from_any_cwd(tmp_path):
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        Mirror()
        assert load_blacklist(DEFAULT_BLACKLIST)["left"] == "absolute_side"
        assert blacklist_category("RIGHT", DEFAULT_BLACKLIST) == "absolute_side"
    finally:
        os.chdir(old)


def test_clip_gloss_map():
    assert gloss_for_clip("01991") == "all"
    assert gloss_for_clip("nope") is None


def test_csv_mirror_safe_agrees_with_yaml():
    with resolve_path(CLIP_GLOSSES).open(newline="") as f:
        for row in csv.DictReader(f):
            category = blacklist_category(row["gloss"], DEFAULT_BLACKLIST)
            assert (row["mirror_safe"] == "no") == (category is not None), row["clip_id"]
            assert row["mirror_rule"] == (category or "")


@pytest.mark.parametrize("name", ["default", "literature", "ambitious"])
def test_profiles_give_blacklist(name):
    profile = load_profile(CONFIGS / f"slr_week4_{name}.yaml")
    entry = next(e for e in profile.entries if e.name == "scene.mirror")
    assert entry.ranges["blacklist"] == DEFAULT_BLACKLIST


def test_clipdata_gloss_defaults_to_none():
    assert synthetic_clip(frames=2).gloss is None


@requires_smplx
def test_chain_fills_gloss_and_blocks_mirror(monkeypatch):
    skeleton = get_skeleton()
    profile = load_profile(CONFIGS / "slr_week4_default.yaml", enable=["scene.mirror"])
    glosses = {"b": "left", "a": "book"}
    monkeypatch.setattr(chain_module, "gloss_for_clip", lambda cid: glosses.get(cid))
    chain = AugmentationChain(profile, skeleton)
    mirrored_when_allowed = 0
    for seed_clip in range(40):
        clip = synthetic_clip(frames=6).replace(clip_id=str(seed_clip))
        free = chain.run(clip)
        glosses[str(seed_clip)] = "left"
        blocked = chain.run(clip)
        assert blocked.draws["scene.mirror"]["mirror"] is False
        assert blocked.draws["scene.mirror"]["blocked"] == "blacklist:absolute_side"
        assert not blocked.clip.scene.mirror
        assert clip.gloss is None  # the input is not mutated
        mirrored_when_allowed += free.draws["scene.mirror"]["mirror"]
    assert mirrored_when_allowed > 0  # the unblocked clips still mirror
