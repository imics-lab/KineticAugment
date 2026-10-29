# Data index

- `augment_trials/` — Default output of `scripts/augment_trials.py` (per-clip augmentation videos and JSON measurements); created on first run.
- `augments.yaml` — Example pipelines covering baseline and configured landmark augmentations.
- `hand_variants/` — Output of `scripts/export_hand_variants.py`: every clip's hands rebuilt by the two hand pipelines (one shape per track, with and without a per-frame scale), for a classifier comparison.
- `round_trip_eval/` — Default output of `scripts/evaluate_round_trip.py` (landmark caches, report, videos); created on first run.
- `sign_videos/` — Five source sign-language clips used by the extraction examples.
- `test_output/` — Saved landmark arrays, visualizations, and extraction summary for the sample clips.
- `index.md` — Index of sample data and generated reference outputs in this folder.
