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

## Candidate augmentations for a raw-MediaPipe classifier (2026-11-10)

The classifier reads raw MediaPipe landmarks as a time series, normalised and unnormalised
(README, *Classifier input*). These candidates target that format. None is implemented. Each needs
a measurement before a range is set, and the one-at-a-time ablation above to keep it. Order
revised after an independent second review.

0. [ ] **Baselines first.** Train on original clips, on round-trip clips with no edits, and on the
       literature profile (`slr_week4_literature.yaml`), on a signer-held-out split with 3-5 seeds.
       Without this, no candidate below can be judged. Always keep the original extracted clips in
       the training set.
1. [ ] **MediaPipe residual re-injection (experiment).** The round trip returns rigid SMPL-X/MANO
       geometry; real MediaPipe has non-rigid bones, hand-size drift and noisy z (V1 Findings,
       V1.2 Findings §10). Store the unedited residual (MediaPipe minus round trip) and add it back
       to the augmented output. Caveats:
       - The residual mixes detector error, our fit error and real motion the fit lost. Adding it
         back can restore articulation an edit deliberately changed. Split it into a per-clip
         systematic bias and a time-varying part, and test each.
       - Store it in a local frame (per bone, hand-relative) so it can be rotated, scaled, mirrored
         and time-resampled with the edit.
       - Borrowing another clip's residual is riskier than reusing the clip's own.
       - Rendering SMPL-X and re-running MediaPipe on it is a research arm only: an untextured
         avatar is out of MediaPipe's training distribution, so its errors are unlikely to match
         real video. Validate any noise source against measured real error distributions.
2. [ ] **Measured detector failures.** Beyond hand-dropout runs: hands missing at clip start and
       end, face and low-visibility pose dropouts, frozen landmarks, jumps when tracking recovers,
       temporally correlated jitter and per-frame hand-scale drift. Measure each rate on the local
       WLASL cache first, as `scripts/measure_hand_dropout.py` did. Handedness swaps only if they
       are actually observed. Use the classifier's real missing-value convention.
3. [ ] **Onset and offset jitter.** Trim or pad the idle lead-in and lead-out by a few frames.
       Boundaries must keep the preparatory handshape, the final hold and compound parts, not just
       the movement. Trims of 32 frames cost 1.8-2.1 pp in pose ISLR (Woods & Rana 2023, in the
       tolerance report), so keep this small and idle-only.
4. [ ] **Aspect ratio and resolution.** Crop, pad and resize change raw coordinates differently.
       Generate both coordinate forms from one consistent camera and image transform, drawn from
       the formats found in WLASL and the target data. Hands must stay in frame. A resolution
       change alone does not reproduce resolution-dependent detector degradation.
5. [ ] **Small camera roll (ablation arm, not default).** Currently excluded (Grounding §3): ±4.5°
       hurt pose-based translation, but SPOTER's in-plane rotation (±13°) was part of a +2.27 pp
       WLASL100 gain. Roll is a camera property, not a production change, and raw coordinates do
       not cancel it. Test a small image-plane rotation about the camera centre with aspect
       correction.
6. [ ] **Depth (z) perturbation (low priority).** z is unreliable but still carries orientation and
       approach/contact information, and independent per-stream factors can break cross-stream
       relations. Only small perturbations around each stream's native depth origin, as an
       experiment.
7. [ ] **Weak drop (Grounding §3 item 9).** Already specified; blocked on ASL-LEX gating. This is a
       production change, separate from detector dropout, and needs its own gate.

Not recommended: smooth whole-body sway or head motion (head movement can be grammatical), and
skeleton mixup or cutmix, which belong in the training loop rather than this framework.

### Per-sign metadata and validation (later stage)

- [ ] Download ASL-LEX 2.0 and join it to `data/wlasl/clip_glosses.csv`, matching lexical variants,
      not only English glosses (Grounding §5). It gives handshape, location, movement, contact and
      sign type. It does not give non-manuals, mouthings, frame-aligned importance or geometric
      tolerances; ASL Signbank and Sem-Lex add variants and recordings. Lexical non-manuals need
      expert annotation.
- [ ] Data-driven importance: per gloss, the spread of each body part across signers. Low spread
      means the part matters and should be augmented less.
- [ ] A weaker trained classifier as an alarm, not a gate: compare its predictions on original and
      augmented clips, flag confident predictions of a *different* sign (especially minimal pairs)
      for review, audit flag rates per class, and never filter with test data. A hard gate would
      reject the hard, valid samples augmentation exists to add.
