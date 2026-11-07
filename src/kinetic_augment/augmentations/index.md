# Augmentations index

- `__init__.py` — Exports augmentation interfaces and concrete transformation classes.
- `base.py` — Defines augmentation configuration, base classes (incl. `ClipAugmentation`, region tags), and registry.
- `ranges.py` — Seeded per-clip RNG, range parsing (uniform and log-uniform), truncated-normal sampling.
- `extrinsic.py` — Implements global landmark-space transforms and noise.
- `intrinsic.py` — Implements SMPL-X joint, hand, limb, and orientation perturbations.
- `temporal.py` — Implements time warping, speed variation, and frame dropping.
- `tier1.py` — Tier 1 augmentations (Augmentation_Grounding.md §3): per-clip sampling from the evidence ranges, view/mirror/scale transforms, uniform duration change, and a weak-perspective camera that keeps the hands in frame when uncentring.
- `index.md` — Index of modular augmentation implementations.
