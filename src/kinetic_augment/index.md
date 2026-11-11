# src/kinetic_augment/

| Item | Description |
|---|---|
| `__init__.py` | Public API re-exports (`Pipeline`, datasets, deprecated `AugmentationPipeline`); `__version__` |
| `.gitkeep` | Placeholder file |
| `augmentations/` | Registered extrinsic/intrinsic/temporal transforms and the Tier 1 augmentations |
| `body_model/` | MediaPipe <-> SMPL-X conversion for body, hands and face; SMPL-X wrapper |
| `constraints/` | Constraint engine: joint limits, velocity limits, collision |
| `evaluation/` | Metrics, datasets, classifiers and benchmark runner |
| `pipeline/` | Registry-based `Pipeline`, config, presets, chains, PyTorch datasets |
| `utils/` | 543-landmark layout, left/right maps, smoothing |
| `legacy/` | **Legacy.** Deprecated pipeline, function-style augmentations, canonicalization, old converters |
