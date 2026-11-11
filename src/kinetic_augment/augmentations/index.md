# augmentations/

| File | Description |
|---|---|
| `__init__.py` | Exports the registered augmentation classes |
| `base.py` | Base classes (`BaseAugmentation`, landmark/SMPL-X variants) and `AugmentationRegistry` |
| `extrinsic.py` | Landmark-space transforms: rotation, scaling, flip, trajectory jitter, noise |
| `intrinsic.py` | SMPL-X parameter-space joint and hand perturbations |
| `temporal.py` | Time warping, speed variation, frame dropping |
| `tier1.py` | Tier 1 augmentations (mirror, view, duration, arm joints, scale, uncentre); not in the registry or `Pipeline` |
