# Package index

- `.gitkeep` — Empty placeholder retained in the package directory.
- `__init__.py` — Defines package exports and optional evaluation access.
- `augmentations.py` — Legacy functional implementations of landmark and temporal augmentations.
- `augmentations/` — Class-based extrinsic, intrinsic, and temporal augmentation modules.
- `body_model/` — SMPL-X wrapper, joint mappings, and MediaPipe conversion code.
- `canonicalization.py` — Moves landmark frames to and from a canonical body-centered coordinate frame.
- `constraints/` — Joint, velocity, and self-collision constraint enforcement.
- `evaluation/` — Metrics, datasets, recurrent classifiers, and benchmark tools.
- `lab/` — Augmentation Lab: shared run table, bundle helpers and the pane blob format.
- `pipeline/` — Configurable augmentation pipeline, chaining, presets, and datasets.
- `pipeline_legacy.py` — Deprecated wrapper for the earlier augmentation pipeline API.
- `utils/` — Landmark data-format conversion helpers and left/right index maps.
- `index.md` — Index of the KineticAugment package modules and subpackages.
