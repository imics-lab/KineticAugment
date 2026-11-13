# configs/

| Item | Description |
|---|---|
| `slr_profile.yaml` | Valid SLR config for `Pipeline.from_yaml` (generated from the `moderate` preset, no PoseFlipping) |
| `mirror_blacklist.yaml` | WLASL glosses that must not be mirrored, by category (docs/Mirror_Unsafe_Signs.md) |
| `slr_week4_default.yaml` | Week-4 default profile: scene augmentations, uniform duration, arm and finger joint coupled noise and frame dropping at the used (about 25% past proven) ranges; the rest present but off |
| `slr_week4_ambitious.yaml` | Week-4 profile with every augmentation on at the used ranges (mouth/jaw and Tier 1 arm joints off) |
| `slr_week4_literature.yaml` | Week-4 baseline: the default profile's augmentations at the proven ranges |
| `viewer_saved/` | Settings saved from the Lab (full profiles, one YAML each); see its `index.md` |
| `legacy/` | **Legacy.** Configs in the deprecated `AugmentationPipeline` format |
| `index.md` | This index |
