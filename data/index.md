# Data index

- `augment_trials/` — Default output of `scripts/augment_trials.py` (per-clip augmentation videos and JSON measurements); created on first run.
- `augments.yaml` — Example pipelines covering baseline and configured landmark augmentations.
- `hand_variants/` — Output of `scripts/export_hand_variants.py`: every clip's hands rebuilt by the two hand pipelines (one shape per track, with and without a per-frame scale), for a classifier comparison.
- `round_trip_eval/` — Default output of `scripts/evaluate_round_trip.py` (landmark caches, report, videos); created on first run.
- `wlasl/` — WLASL annotation file (not committed) and the clip → gloss map with mirror safety.
- `augment_trials/` — Sample augmentation videos with the observed face carried through each augmentation, and their measurements (not committed; regenerate with `scripts/augment_face_trials.py`).
- `sign_videos/` — Five source sign-language clips used by the extraction examples.
- `round_trip_eval/` — Evaluation set of 20 unseen clips, not committed (derived from footage outside the repo); holds `*_hands.npz`, `*_world.npz`, the face caches `*_face.npz` and `*_face_features.npz`, and the gate report `face_report.json`.
- `test_output/` — Saved landmark arrays, visualizations, and extraction summary for the sample clips.
- `index.md` — Index of sample data and generated reference outputs in this folder.
