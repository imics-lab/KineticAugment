"""
Apply KineticAugment augmentations to one landmark clip using the current Pipeline API.

Run from the repo root (after `pip install -e .`, or with `PYTHONPATH=src`):

    PYTHONPATH=src python examples/run_augmentation.py

Uses data/test_output/01986_landmarks.npy (a (frames, 543, 3) MediaPipe clip) when it exists,
otherwise random synthetic data of the same shape.
"""

from pathlib import Path

import numpy as np

import kinetic_augment
from kinetic_augment import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP = REPO_ROOT / "data" / "test_output" / "01986_landmarks.npy"
SLR_CONFIG = REPO_ROOT / "configs" / "slr_profile.yaml"


def load_clip() -> np.ndarray:
    if CLIP.exists():
        print(f"Loading {CLIP.relative_to(REPO_ROOT)}")
        return np.load(CLIP).astype(np.float32)
    print("Real clip not found; using synthetic data.")
    return np.random.default_rng(0).random((60, 543, 3)).astype(np.float32)


def report(name: str, original: np.ndarray, augmented: np.ndarray) -> None:
    moved = float(np.nanmean(np.abs(augmented - original)))
    print(f"  {name:<28} shape={augmented.shape}  mean |change|={moved:.5f}")


def main() -> None:
    print(f"kinetic_augment {kinetic_augment.__version__}")
    clip = load_clip()
    print(f"Clip shape: {clip.shape}")

    print("\nFrom a built-in preset:")
    for preset in ("conservative", "moderate"):
        pipeline = Pipeline.from_preset(preset, seed=1)
        report(f"preset '{preset}'", clip, pipeline.process(clip))

    print("\nFrom YAML:")
    pipeline = Pipeline.from_yaml(SLR_CONFIG)
    report(SLR_CONFIG.name, clip, pipeline.process(clip))


if __name__ == "__main__":
    main()
