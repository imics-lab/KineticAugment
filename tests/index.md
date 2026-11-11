# tests/

Run with `pytest tests/` from the repo root.

| File | Description |
|---|---|
| `__init__.py` | Package marker |
| `test_body_model.py` | Joint mapping, `SMPLXWrapper` (skipped without torch/models), `SMPLXParamsContainer` |
| `test_legacy_converters.py` | Legacy image-space converters (`kinetic_augment.legacy`); pure numpy |
| `test_smoothing.py` | Temporal landmark smoothing filters |
| `test_tier1.py` | Tier 1 augmentations: sampling, transforms, re-timing, frame check |
