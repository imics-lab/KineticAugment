# Configuration index

- `har_running_profile.yaml` — HAR profile weighting augmentations for running motions.
- `mirror_blacklist.yaml` — WLASL glosses that must not be mirrored, by category (docs/Mirror_Unsafe_Signs.md).
- `slr_profile.yaml` — SLR profile for signer variation while preserving sign meaning.
- `slr_profile_extended.yaml` — Expanded SLR profile covering signer, environment, and camera variation.
- `slr_week4_default.yaml` — Week-4 default profile: scene augmentations, uniform duration, arm and finger joint coupled noise and frame dropping at the used (about 25% past proven) ranges; the rest present but off.
- `slr_week4_ambitious.yaml` — Week-4 profile with every augmentation on at the used ranges (mouth/jaw and Tier 1 arm joints off).
- `slr_week4_literature.yaml` — Week-4 baseline: the default profile's augmentations at the proven ranges.
- `index.md` — Index of task-specific YAML profiles in this folder.
