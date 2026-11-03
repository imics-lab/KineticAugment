# To-do

Open work items. Each links to the evidence that motivates it.

## Apply the recommended augmentation ranges (2026-11-02)

Source: the *Recommended* bullets in [`docs/Augmentation_Grounding.md`](docs/Augmentation_Grounding.md)
§3 and the [tolerance report](<docs/augmentation-evidence/reports/Sign augmentation tolerance evidence.md>).
None of these is in `kinetic_augment.augmentations.tier1` yet.

### Tier 1 (`src/kinetic_augment/augmentations/tier1.py`, `tests/test_tier1.py`)

- [ ] Duration: widen 0.8–1.25× to about 0.67–1.5×, log-uniform.
- [ ] Arm joints: let the elbow go to ±7.5°; keep the shoulder and wrist at ±4°.
- [ ] Mirror: skip clips whose gloss is on the mirror blacklist
      ([`docs/Mirror_Unsafe_Signs.md`](docs/Mirror_Unsafe_Signs.md)), and make p a configurable
      value instead of a fixed 0.5.
- [ ] View: make the yaw range configurable so ±45° can be run as an ablation arm; keep ±20°
      as the default and pitch at ±5°. Add no roll.
- [ ] Re-run `scripts/augment_trials.py` and update §6.2 of `Augmentation_Grounding.md` with the
      new draws.

### Tier 2 (new code)

- [ ] Location shifts in a face- or torso-anchored frame: σ ≈ 10 mm per clip, truncated at about
      20 mm; sideways up to about 40–50 mm, toward the midline only; forehead lowering 10–40 mm,
      downward only, blocked near the eye and for lower-location minimal pairs.
- [ ] Hand-dropout runs that follow MediaPipe's failure pattern; frame drop no lower than about
      10 fps equivalent; tiny noise only.
- [ ] Handshape jitter sized by finger role (larger on non-selected fingers and the thumb,
      smallest on selected-finger flexion), with a geometric flexion check.

### Measurements that set the remaining bounds

- [ ] Natural spread in our reconstructed clips per gloss: location SD, lowering, MANO joint
      angles, palm normals, path length.
- [ ] Round-trip noise floor in the same face-anchored frame.
- [ ] One-at-a-time classifier ablation on a split with unseen test signers, 3–5 seeds: yaw ±20°
      vs ±45°, duration 0.8–1.25× vs 0.67–1.5×, mirror p = 0 vs 0.5 (with per-class deltas),
      elbow ±7.5°, pre-training on augmented clips vs mixing.
