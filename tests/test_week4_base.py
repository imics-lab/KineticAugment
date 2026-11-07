"""Tests for week-4 regions, ClipAugmentation base and range sampling."""

import numpy as np
import pytest

from kinetic_augment.augmentations.base import (
    REGIONS,
    AugmentationRegistry,
    BaseAugmentation,
    ClipAugmentation,
)
from kinetic_augment.augmentations.ranges import clip_rng, parse_range, truncated_normal


class _Dummy(ClipAugmentation):
    key = "dummy"
    DEFAULT_RANGES = {"a": 1.0}

    def sample(self, rng, clip, ctx):
        return {"a": self.ranges["a"]}

    def apply(self, target, draw, ctx):
        return target

    def identity_draw(self):
        return {"a": 0.0}


@pytest.fixture
def cleanup_registry():
    before = set(AugmentationRegistry._registry)
    yield
    for name in set(AugmentationRegistry._registry) - before:
        del AugmentationRegistry._registry[name]


def test_clip_rng_is_deterministic_and_independent():
    a = clip_rng(1, "clip", "k").random(5)
    assert np.array_equal(a, clip_rng(1, "clip", "k").random(5))
    assert not np.array_equal(a, clip_rng(1, "clip", "other").random(5))


def test_parse_range_forms():
    assert parse_range([0.8, 1.2]).distribution == "uniform"
    spec = parse_range({"range": [0.6, 1.65], "distribution": "log_uniform"})
    rng = np.random.default_rng(0)
    assert all(0.6 <= spec.draw(rng) <= 1.65 for _ in range(1000))
    with pytest.raises(ValueError):
        parse_range([2, 1])
    with pytest.raises(ValueError):
        parse_range([0, 1], default_distribution="log_uniform")
    with pytest.raises(ValueError):
        parse_range(3.0)


def test_log_uniform_is_symmetric_in_log():
    spec = parse_range([0.6, 1 / 0.6], default_distribution="log_uniform")
    rng = np.random.default_rng(0)
    x = np.array([spec.draw(rng) for _ in range(20000)])
    assert abs(np.mean(np.log(x))) < 0.01


def test_truncated_normal_respects_limit():
    rng = np.random.default_rng(0)
    x = truncated_normal(rng, 2.5, 5.0, size=10000)
    assert np.max(np.abs(x)) <= 5.0
    assert abs(np.std(x) - 2.20) / 2.20 < 0.10
    assert np.array_equal(truncated_normal(rng, 0.0, 5.0, size=4), np.zeros(4))


def test_registry_refuses_clash(cleanup_registry):
    class Fresh(BaseAugmentation):
        def apply(self, data, **kwargs):
            return data

    with pytest.raises(ValueError):
        AugmentationRegistry.register("JointCoupledNoise")(Fresh)
    AugmentationRegistry.register("clash.once")(Fresh)
    AugmentationRegistry.register("clash.once")(Fresh)


def test_clip_augmentation_rejects_unknown_key():
    with pytest.raises(ValueError, match="b"):
        _Dummy(ranges={"b": 2})
    assert _Dummy(ranges={}).defaulted == ["a"]
    assert _Dummy(ranges={"a": 2.0}).defaulted == []


def test_list_by_region(cleanup_registry):
    @AugmentationRegistry.register("face.dummy")
    class FaceDummy(_Dummy):
        regions = ("face",)

    assert set(("face",)) <= set(REGIONS)
    assert "face.dummy" in AugmentationRegistry.list_by_region("face")
    assert "face.dummy" not in AugmentationRegistry.list_by_region("body")
