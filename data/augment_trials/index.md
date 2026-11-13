# Augmentation trials index

Output of `scripts/augment_trials.py` (Tier 1 on the body + hands round trip) and `scripts/augment_face_trials.py` (V1.3: body + hands + observed face). Derived from WLASL footage outside the repo; regenerate rather than commit.

- `<clip>_augmentations.mp4` — Source video beside the SMPL-X figure: round trip and the Tier 1 augmentations (view, mirror, duration, arm joints, scale, uncentre, combined) at the clip's drawn values, with a metrics panel.
- `augment_trials.json` — Seed, clip pool, Tier 1 ranges, and per clip: the draw, the camera fit, the uncentre shifts and the measurements (zero-strength identity, bone lengths, wrist displacement, jitter, hand points in frame).
- `<clip>_face_augmentations.mp4` — Source video beside the SMPL-X figure carrying the observed MediaPipe face: round trip, yaw, mirror, scale, uncentre, arm pose, body shape, head turn and time warp, with a metrics panel and a hand-to-face distance strip.
- `augment_face_trials.json` — Augmentation parameters, whether the depth-corrected pass was used, and per-clip measurements (zero-strength identity, face rigidity, face and wrist displacement, hand-to-face distances).
- `<clip>_week4_augmentations.mp4` — Week-4 trial grid (`scripts/augment_week4_trials.py`): one panel per augmentation (arms, hands, face points over the grey base) plus a combined panel; temporal panels loop at their own length.
- `augment_week4_trials.json` — Per clip and run: draws, applied entries, fallbacks, frame count, and measures (wrist move, bone-length change, jitter ratio, face rigidity, closest hand-to-face, in-frame share).
- `index.md` — This index.
