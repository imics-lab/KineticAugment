# Augmentation trials index

Output of `scripts/augment_trials.py`. Derived from WLASL footage outside the repo; regenerate rather than commit.

- `<clip>_augmentations.mp4` — Source video beside the SMPL-X figure: round trip and the Tier 1 augmentations (view, mirror, duration, arm joints, scale, uncentre, combined) at the clip's drawn values, with a metrics panel.
- `augment_trials.json` — Seed, clip pool, Tier 1 ranges, and per clip: the draw, the camera fit, the uncentre shifts and the measurements (zero-strength identity, bone lengths, wrist displacement, jitter, hand points in frame).
- `index.md` — This index.
