# legacy/ (legacy, deprecated)

Superseded code kept for reference; new work must not build on it. See `__init__.py` for replacements.

| File | Description |
|---|---|
| `__init__.py` | Docstring mapping each module to its replacement; no imports |
| `pipeline.py` | Deprecated `AugmentationPipeline` (still exported as `kinetic_augment.AugmentationPipeline`) |
| `augmentations.py` | Function-style augmentations used only by `pipeline.py` |
| `canonicalization.py` | Shoulder-centred canonical space; used only by `pipeline.py` |
| `mp_to_smplx.py` | Old image-space `MediaPipeToSMPLX` (replaced by `body_model.converter.fit_body`) |
| `smplx_to_mp.py` | Old `SMPLXToMediaPipe` (replaced by `body_model.converter.reconstruct_body`) |
| `index.md` | This index |
