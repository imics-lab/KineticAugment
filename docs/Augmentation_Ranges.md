# Augmentation Ranges: Proven and Used

The week-4 augmentation profiles (`configs/slr_week4_default.yaml`, `slr_week4_ambitious.yaml`,
`slr_week4_literature.yaml`) set every range about **25% beyond the widest range the evidence
supports**, in the direction of more variation. This page lists the proven range next to the used
one for every augmentation, so a reader can see how far past the evidence each one goes.

## The 25% rule

- "Proven" means the widest range with sign-language or human-perception evidence in
  [Augmentation_Grounding.md](Augmentation_Grounding.md) that was not shown to hurt. **[M]**,
  **[L]** and **[T]** are Grounding's evidence keys.
- The used range is the proven range widened by about 25%. Log-uniform ranges are widened by 25%
  in log space, so the low and high ends stay symmetric.
- Probabilities (the mirror p) are not widened.
- Where the literature has no range, the row says so, and the used value is the designer's
  estimate widened by 25%.
- Excluded items with no range to widen stay excluded: camera roll, shear, temporal reversal,
  cropping.

## Proven against used

| Augmentation | Proven range (source) | Used in week-4 profiles |
|---|---|---|
| View yaw | ±20° (Spoken2Sign **[M7]**) | ±25° |
| View pitch | ±5° ("a few degrees" **[T]**) | ±6° |
| Mirror | p = 0.5 **[M6]**; cost 1.3 pp on WLASL once **[T]** | p = 0.5 (a probability, not widened) |
| Uniform duration | 0.67–1.5×; 0.5× hurt once **[L6.2, M6, M3, T]** | 0.6–1.65×, log-uniform |
| Arm joints: shoulder, wrist | ±4° at p = 0.3 (SPOTER **[M1]**) | ±5° |
| Arm joints: elbow | ±7.5°, no harm in pose-based translation **[T]** | ±9.5° |
| Whole-scene scale | 0.8–1.2 (GISLR **[M10]**) | 0.75–1.25 |
| Uncentre | ±0.1 of the frame (GISLR **[M10]**) | ±0.125, hands still kept in the frame |
| Limb length | no literature; between-signer spread 5.7% in this project's MediaPipe data (Grounding §6.1) | ±7% |
| Movement size | no measured range; 0.85–1.15 is Grounding's linguistic example **[L2.10, L2.11]** | 0.8–1.2, log-uniform |
| Location jitter | σ = 10 mm, truncated at 20 mm (production spread **[T]**) | σ = 12.5 mm, truncated at 25 mm |
| Joint coupled noise, arms | as arm joints above **[M1, T]** | shoulder/wrist σ = 2.5°, truncated at 5°; elbow σ = 4.75°, truncated at 9.5° |
| Joint coupled noise, fingers | no published range | σ = 3.75°, truncated at 7.5° (estimate 3° / 6°, widened) |
| Handshape jitter | no published range; GISLR "finger-tree rotate" gives no angle **[T]** | non-selected and thumb σ = 7.5°; selected σ = 3.75°; selected flexion σ = 2°; all truncated at 2σ |
| Expression jitter | no published range; lexical mouth patterns excluded **[L7.3, L1.5]** | brows and eyes σ = 0.0625, truncated at 0.125; cheeks/nose σ = 0.0375, truncated at 0.075; mouth/jaw off |
| Varied time warp | none: non-uniform warps are excluded **[L6.3, L6.4]** | knot rates 0.7–1.43×, log-uniform |
| Segment time warp | none: excluded as above | window 20–60% of the clip, factor 0.6–1.65× |
| Frame dropping | ≥ 10 fps, where human intelligibility stops improving **[T]**; masked runs ≤ 8 frames **[M4, M10]** | ≥ 8 fps; max gap 125 ms; p = 0–0.2; target 12–25 fps |
| Hand-dropout runs | runs ≤ 8 frames **[M4, M10]**; start rate none, measured 0.039 per detected interior hand-frame (`scripts/measure_hand_dropout.py`) | runs of 1–10 frames; start rate 0–0.05 |

## Notes

- The mouth/jaw blendshape group stays off even in the ambitious profile, because lexical mouth
  patterns can change the sign. It has a range in the YAML so an experiment can switch it on.
- Mirror: the 1.3 pp cost on WLASL was measured once, so the mirror is a documented risk rather
  than a free augmentation. Signs on `configs/mirror_blacklist.yaml` need the per-sign gate.
- Uniform duration: 0.5× hurt once, which is why the proven range stops at 0.67×.
- Limb length: Grounding §6.1 found the between-signer spread (2.9–5.7%) at the noise floor of
  MediaPipe, so there is no literature range and the augmentation is an opt-in experiment arm.
- Hand dropout: the start rate has no literature range. The default upper bound of 0.05 is above
  the 0.039 run starts per detected interior hand-frame measured by
  `scripts/measure_hand_dropout.py`.
- `body.tier1_arm_joints` (SPOTER-style arm joints with one shoulder/wrist range) is off in every
  shipped profile. If enabled it replaces `body.joint_coupled_noise`; the two cannot both be on.
  Its Tier 1 draw uses `shoulder_wrist_deg` for all three joints and ignores `elbow_deg`.

## Profiles

| Profile | What it contains |
|---|---|
| `slr_week4_default.yaml` | Scene augmentations (mirror, view, scale, uncentre), uniform duration, arm joint coupled noise and random frame dropping at the used ranges. Everything else is present but off. |
| `slr_week4_ambitious.yaml` | Everything on at the used ranges, with all three time-warp modes and both frame-drop modes (mouth/jaw and the Tier 1 arm joints stay off). |
| `slr_week4_literature.yaml` | The default profile's augmentations at the proven ranges. Augmentations with no proven range are off. |

A single ablation arm is made by overriding `enabled` on the default profile:
`load_profile(path, enable=["face.expression_jitter"])`.

## Keeping the copies in step

The table lives here, in the README, in the YAML comments and in
`docs/superpowers/specs/2026-11-07-week4-augmentations-design.md` (§5.0). When a range changes,
update all four together.
