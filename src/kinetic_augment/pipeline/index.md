# pipeline/

| File | Description |
|---|---|
| `__init__.py` | Exports `Pipeline`, `AugmentationChain`, config, presets, datasets |
| `pipeline.py` | `Pipeline` (`from_preset`, `from_yaml`, `from_dict`, `process`), landmark and SMPL-X modes |
| `chain.py` | `AugmentationChain`: probability-based composable execution |
| `config.py` | `PipelineConfig` / `ConstraintConfig` dataclasses with YAML serialization |
| `presets.py` | Built-in presets (none, conservative, moderate, aggressive) |
| `dataset.py` | PyTorch Dataset wrappers with on-the-fly augmentation |
| `index.md` | This index |
