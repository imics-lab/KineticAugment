# scripts/legacy/ (legacy)

Print-based scripts for the deprecated stack; not run by pytest. Run from the repo root.

| File | Description |
|---|---|
| `phase1_smoke.py` | Phase 1: joint mapping, legacy MediaPipe-to-SMPL-X conversion, writes `data/test_output/` |
| `phase2_smoke.py` | Phase 2: augmentation classes and constraint system |
| `phase3_smoke.py` | Phase 3: PyBullet collision detection |
| `phase4_smoke.py` | Phase 4: `Pipeline`, presets, datasets |
| `phase5_smoke.py` | Phase 5: evaluation metrics, models, benchmark runner |
| `render_landmarks.py` | Overlay video of original vs legacy SMPL-X round-trip landmarks |
| `render_augmented_from_yaml.py` | Render videos per pipeline in `configs/legacy/augments.yaml` (own copy of the augmentations) |
