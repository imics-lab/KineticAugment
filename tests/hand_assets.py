"""
Shared helpers for the hand tests: where the licensed model assets live, skip
marks for when they are absent, and a cache so each HandModel loads once.
"""

from pathlib import Path

import pytest

MODEL_DIR = str(Path(__file__).resolve().parents[1] / "models")
HAVE_MANO = all(
    (Path(MODEL_DIR) / "mano" / f"MANO_{side}.npz").exists() for side in ("LEFT", "RIGHT")
)
HAVE_SMPLX = (Path(MODEL_DIR) / "smplx" / "SMPLX_NEUTRAL.npz").exists()

try:
    import smplx  # noqa: F401
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

requires_mano = pytest.mark.skipif(not (HAVE_TORCH and HAVE_MANO), reason="MANO npz assets absent")
requires_smplx = pytest.mark.skipif(
    not (HAVE_TORCH and HAVE_SMPLX), reason="SMPL-X assets absent"
)
BACKENDS = [
    pytest.param("mano", marks=requires_mano),
    pytest.param("smplx", marks=requires_smplx),
]
SIDES = ["left", "right"]

_cache = {}


def training_poses(side, count, seed):
    """
    `count` real hand poses from MANO's training set (`hands_coeffs`), in the
    flat-hand-mean convention: hands_mean + coefficients @ hands_components.
    The SMPL-X hand pose space is the same (identical mean and components).
    """
    import numpy as np

    raw = np.load(Path(MODEL_DIR) / "mano" / f"MANO_{side.upper()}.npz")
    rows = np.random.default_rng(seed).choice(len(raw["hands_coeffs"]), count, replace=False)
    return raw["hands_mean"] + raw["hands_coeffs"][rows] @ raw["hands_components"]


def hand(backend, side):
    """A HandModel per (backend, side), loaded once per test session."""
    from kinetic_augment.body_model.hand_model import HandModel

    if (backend, side) not in _cache:
        _cache[(backend, side)] = HandModel(backend, side, model_dir=MODEL_DIR)
    return _cache[(backend, side)]
