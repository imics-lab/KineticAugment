import numpy as np

from kinetic_augment.utils.smoothing import (
    SMOOTHING,
    gaussian_smooth,
    one_euro_smooth,
    valid_runs,
)


def test_valid_runs():
    assert valid_runs(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], bool)) == [(1, 3), (4, 5), (6, 9)]
    assert valid_runs(np.zeros(4, bool)) == []


def test_constant_track_is_unchanged():
    x = np.tile(np.array([0.1, -0.2, 0.3]), (30, 4, 1))
    for name, s in SMOOTHING.items():
        np.testing.assert_allclose(s.apply(x, 30.0), x, atol=1e-12, err_msg=name)


def test_gaps_stay_nan_and_do_not_leak():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2, 3))
    x[10:15, 0] = np.nan  # point 0 drops out
    x[20:40, 1] = 100.0  # point 1 jumps; its first run must not see the jump
    x[19, 1] = np.nan
    for name, s in SMOOTHING.items():
        y = s.apply(x, 30.0)
        assert np.isnan(y[10:15, 0]).all(), name
        assert np.isfinite(np.delete(y, np.s_[10:15], axis=0)[:, 0]).all(), name
        assert np.abs(y[:19, 1]).max() < 10, name
        np.testing.assert_allclose(y[20:, 1], 100.0, err_msg=name)


def test_short_runs_untouched():
    x = np.full((10, 1, 2), np.nan)
    x[4:6, 0] = [[1.0, 2.0], [3.0, 4.0]]
    np.testing.assert_array_equal(gaussian_smooth(x, 2.0), x)


def test_smoothing_reduces_noise():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 2 * np.pi, 90)
    clean = np.stack([np.sin(t), np.cos(t)], axis=-1)[:, None]
    noisy = clean + rng.normal(scale=0.05, size=clean.shape)
    for name in ("gauss1", "gauss2", "oneeuro"):
        y = SMOOTHING[name].apply(noisy, 30.0)
        assert np.abs(y - clean).mean() < np.abs(noisy - clean).mean(), name


def test_one_euro_is_causal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 3, 3))
    y = one_euro_smooth(x, 30.0)
    x2 = x.copy()
    x2[30:] += 5.0
    np.testing.assert_array_equal(one_euro_smooth(x2, 30.0)[:30], y[:30])
