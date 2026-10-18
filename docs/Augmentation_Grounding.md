# Meaning-preserving augmentation: what the research supports

**Date:** 2026-10-11. **Branch:** `new-augmentation-attempt`. **Evidence:**
[linguistics notes](augmentation-evidence/linguistics.md) and
[SLR augmentation notes](augmentation-evidence/slr_augmentation.md). Each claim there has its
source URL and a verification label. A bracketed tag such as **[L2.4]** or **[M1]** below points
to a row or section in those notes (L = linguistics, M = machine learning).

**First trial:** `scripts/augment_trials.py`, run on clips 17083 and 57641 (§6).

The question is which augmentations of a signer's 3-D skeleton (SMPL-X body, MANO hands) keep an
isolated sign's meaning, and how strong they can be. Two kinds of evidence answer it:

- **Sign-language linguistics:** what varies between signers and productions without changing
  the sign, and what is contrastive.
- **Sign-recognition practice:** which pose augmentations published systems use, with what
  ranges, and what ablations measured.

## 1. What the linguistics says

**Contrastive:**
- **The five parameters are contrastive:** handshape, location, movement, palm orientation and
  non-manual markers. Each has minimal pairs, for example location APPLE/ONION and
  MOTHER/FATHER, handshape CAR/WHICH, movement PROPOSE/MAYBE, orientation TEMPORARY/TRAIN
  **[L1.3]**.
- **Handshape is perceived categorically; location is not.** Deaf signers show categorical
  perception for contrastive handshape continua, but not for place-of-articulation continua
  **[L5.1–5.3]**. So within-category finger variation is tolerated, and crossing a category is
  not.
- **Timing can be morphology.** Aspectual modulations change repetition, holds, tension and rate
  of movement **[L6.4]**. Telic verbs end with stronger deceleration than atelic ones **[L6.3]**.
- **Some face and mouth components are lexical.** NOT-YET requires the TH mouth morpheme and LATE
  does not **[L7.3]**. Sentence-level markers (brow raise, negative headshake) are not part of a
  citation sign **[L7.1–7.2]**.

**Not contrastive:**
- **Left versus right hand.** "Signing with the left or the right hand is not lexically
  contrastive in ASL" **[L3.1]**.
- **Location within limits.** Forward–backward and sideways shifts "cannot create a different
  lexical item" **[L2.4]**. Forehead signs are often produced lower, and more so at fast rates
  **[L2.1–2.5]**.
  - Lowering has a direction. It acts on the higher member of a pair (BROTHER is lowered,
    SISTER is not), it is graded, and signers avoid the eyes **[L2.5–2.6]**.
- **Signing rate.** Rate changes are realised mainly as changes in sign duration **[L6.2]**.
- **Movement size (register).** Shouted signing enlarges movement and whispered signing shrinks
  it. Lexical entries specify perceptual targets, and the joints are filled in phonetically
  **[L2.10]**.
- **One-handed versions of two-handed signs (weak drop)** are allowed only under conditions. They
  are blocked by alternating movement, continuous contact between the hands, contact between
  analogous hand parts, and signs located on the weak hand **[L4.1–4.4]**.

No source gives numeric tolerances in millimetres or degrees. **Bounds must be set
empirically.**

## 2. What sign-recognition practice says

| Augmentation | Measured effect | Range used | Source |
|---|---|---|---|
| Uniform speed change / resampling | Largest effect found: −15.65 pp on WLASL100 when removed | length between the 25th and 75th percentiles; rate 0.5–1.5 | **[M6, M8]** Laines 2023; GISLR 1st place |
| 3-D view rotation of SMPL-X `global_orient` | +1.46 top-1 WLASL, +1.29 MSASL | ±20° | **[M7]** Spoken2Sign |
| In-plane rotation | +2.27 pp WLASL100 | ±13° | **[M1]** SPOTER |
| Arm-joint rotation down the chain | +2.27 pp WLASL100 | p = 0.3 per joint, ±4° | **[M1]** SPOTER |
| Squeeze / perspective | +1.88 / +1.50 pp | 0–15% / 0–10% | **[M1]** SPOTER |
| Mirror with left/right swap | small: −0.69 pp when removed. Handedness normalisation neutral in fingerspelling. Never reported to hurt | p = 0.5 | **[M6]** Laines; Shi 2023 |
| Global scale / bone length | about none: −0.22 pp when removed | 0.5–1.0 | **[M6]** Laines |
| Joint / frame / clip masking | large as a *pretraining* pretext (SignBERT) | ratio 40%, clips ≤ 8 frames | **[M4]** |
| Axis or channel masking | **hurts** (action recognition, collapse to 3.9%) | — | **[M10]** AS-CAL, STARS |
| Random temporal crop | rejected by ASL Citizen: can reduce a compound to one root sign | — | **[M3]** |
| Temporal reversal | used in action recognition only | — | **[M8]** (excluded for sign) |
| Phonology as auxiliary targets | +8.7 pp (SL-GCN, WLASL) from ASL-LEX handshape and minor location | — | **[M9]** Kezar 2023 |

**Gap.** No published work perturbs MANO hand pose under phonological constraints **[M7]**.
SGNify's symmetry and handshape-invariance priors and SignAvatars' biomechanical limits are the
nearest ready-made constraint sets. A constrained hand augmentation would be new work.

## 3. Recommended catalogue

The tiers below combine both kinds of evidence:

- **Tier 1** is supported by linguistics *and* by measured recognition gains.
- **Tier 2** is supported by linguistics but needs a per-sign gate or an invariant check.
- **Excluded** items are likely to change the sign.

Every augmentation draws its parameters once per clip (V1 Problem Scope §11).

### Tier 1: safe for every sign

1. **3-D view rotation.** Rotate SMPL-X `global_orient` about the vertical axis by ±20°, plus a
   few degrees of pitch. The hand's location relative to the body is unchanged by construction.
   **[M7]**
2. **Mirror with sides swapped.** Swap SMPL-X left/right joints and MANO hands, and mirror the
   face stream too, so the dominant hand becomes the other hand. **[L3.1, M6]**
   - Caveats not tested by anyone: indexing and pointing signs, and the few signs located on a
     lateral body part.
3. **Uniform rate change.** Scale the clip's duration by a single factor. Start at 0.8–1.25×,
   inside the 0.5–1.5 used in practice. Do not crop. **[L6.2, M6, M3]**
4. **Small arm-joint rotations.** Shoulder, elbow and wrist each with p = 0.3, ±4° **[M1]**,
   followed by the location check in §4.
5. **Whole-scene scale (camera distance).** Scale everything (body, hands, face) by one factor
   about the shoulder centre. Start at 0.85–1.15; practice uses 0.9–1.1 (ST-GCN), 0.8–1.2
   (GISLR) and 0.5–1.0 (Laines) **[M5b, M10, M6]**.
   - Every relative position is kept, so no phonological parameter can change.
   - This is not body shape (item 8, dropped). It is the same body seen from nearer or farther
     away.
6. **Uncentring (subject position in the frame).** Shift the whole scene by one offset per clip.
   Practice uses ±10 px of 512 (SAM-SLR), ±0.1 normalised (GISLR) and ±0.2 (ST-GCN)
   **[M5b, M10]**.
   - Relative positions are kept.
   - Constraint: both hands must stay inside the frame for the whole clip. A sign that leaves
     the frame loses information.

**Caveats for scale and uncentring.**
- **The classifier's normalisation can cancel them.** A classifier that centres on the shoulders
  and scales by shoulder width undoes them exactly (OpenHands, ASL Citizen) **[M2, M3]**. That is
  why removing scale cost only 0.22 pp in Laines **[M6]**. They pay off for models that see raw
  frame coordinates, and as protection against normalisation errors.
- **Apply them last, in output space.** Never feed a scaled body back into the SMPL-X fit, which
  would read it as a different person.
- **In a real camera, a subject moved sideways is also seen from a slightly different angle.** A
  pure shift ignores that. Once a camera model exists, both augmentations should become moves of
  the body in camera space (depth for scale, x and y for position) followed by projection.

### Tier 2: per-sign gate or invariant check

5. **Movement-size scaling (whisper ↔ shout).** Scale the hand path about its centroid, for
   example by 0.85–1.15. Keep the start and end location regions. **[L2.10, L2.11]**
6. **Location shifts.**
   - Neutral space: small shifts in any direction.
   - Body-anchored signs: shift forward/back and sideways only **[L2.4]**.
   - Forehead signs: lower only, downward, partially, never to the eyes, and not when a
     lower-location minimal pair exists. Sample the amount from the observed lowering
     distribution **[L2.1–2.6]**.
7. **Within-category handshape jitter in MANO space.** The category comes from the ASL-LEX
   features (selected fingers, flexion, spread, thumb position).
   - The two hands' finger poses stay tied in symmetric signs (SGNify).
   - Bounds come from SignAvatars' biomechanical limits.
   - Reject any sample whose classified handshape features change. **[L5.1, L1.8, M7]**
   - This has no published precedent and is the project's opportunity.
8. ~~**Body shape with location retargeting.**~~ **Dropped (2026-10-18).** The idea was to change
   `betas`, then re-solve the arm IK so each hand keeps its position relative to the body
   landmarks it is anchored to (the perceptual-target view **[L2.10]**).
   - Naive shape change fails this: in the trial it moved the wrists 68–83 mm (§6).
   - There is no recognition-gain evidence for body shape **[M10]**.
   - Real signers barely differ in MediaPipe's proportions, so there is little variation to
     reproduce (§6.1). The `LimbLengthScaling` augmentation was removed with this decision.
9. **Weak drop.** Remove the weak hand only when ASL-LEX Sign Type is symmetrical, contact is
   absent, and the movement is not alternating. Never on asymmetric signs or signs located on
   the weak hand. **[L4.3, L4.5]**
10. **Detector-failure simulation.** Mask joints or short runs of frames (≤ 8), or add very small
    noise. Never mask a whole axis. **[M4, M10]**

### Excluded

- **Crossing a handshape, orientation, movement-shape or location category**: for example moving
  a chin sign up to the forehead, or changing a straight path to an arc **[L1.3, L2.6]**.
- **Non-uniform time warps, adding or removing repetitions, holds or looping.** These imitate
  aspect, and change endpoint deceleration (telicity) **[L6.3, L6.4]**.
- **Temporal reversal and aggressive temporal crops** **[M3, M8]**.
- **Mirroring one hand only, or swapping the hands' roles within an asymmetric sign**
  **[L3.5]**.
- **Changing lexical mouth patterns or facial expressions**: NOT-YET's TH, UGLY's face
  **[L7.3, L1.5]**.
- **Shear and axis masking.** They distort body geometry and hurt in action recognition
  **[M10]**.

## 4. Invariant checks after every augmentation

Because the literature gives no numeric tolerances, each augmented clip should be checked
against the original and rejected if a phonological property changed.

| Invariant | How to measure it in this pipeline |
|---|---|
| Location region of the dominant hand at its key frames | V1.3 hand-to-face distances with ASL-LEX regions; body regions for torso signs |
| Hand contact (±) and contact with the weak hand | V1.3 signed distance and contact flag; distance between the two MANO hands |
| Handshape category | selected fingers, flexion, spread and thumb, classified from the MANO pose |
| Palm orientation class | palm normal relative to the torso frame |
| Repetitions and holds | peaks and plateaus in wrist speed |
| End deceleration | wrist speed profile over the last frames |

## 5. Per-sign gating with ASL-LEX

ASL-LEX 2.0 codes each sign's Sign Type, Major and Minor Location, Contact, Selected Fingers,
Flexion, Spread, Thumb Position, Path Movement, Repeated Movement and Internal Movement
**[L8.2]**. WLASL glosses have been aligned to it; about 48% of WLASL videos carry its
phonological labels **[M9]**.

- **Signs with labels** get Tiers 1 and 2, gated by those fields (see the gating list in
  [linguistics.md §8](augmentation-evidence/linguistics.md)).
- **Signs without labels** get Tier 1 only.

**Licence.** asl-lex.org lists the database as CC BY-NC 4.0, while the OSF project says CC BY 4.0
**[L8.4]**. Assume non-commercial use until this is resolved.

## 6. What this means for the first trial

`scripts/augment_trials.py` ran five augmentations on clips 17083 and 57641:

| Trial setting | Evidence-based setting | Verdict |
|---|---|---|
| Yaw +25° | ±20° (Spoken2Sign) | Slightly beyond the tested range. Use ±20°. |
| Mirror with sides swapped | p = 0.5, sides swapped | Matches. Keep it. |
| Scale ×0.85 (added later) | 0.8–1.2 | In range. Bones exactly −15%; shoulder-normalised pose unchanged (0.00%). |
| Off-centre by (+0.6, +0.2) shoulder widths, about 21 cm (added later) | ±0.1–0.2 of the normalised frame | Shoulder-normalised pose unchanged (0.00%). No camera model yet, so the hands-in-frame check could not be applied. |
| Elbows +15°, shoulders +10° (wrists moved 94–115 mm) | ±4° per joint, p = 0.3 (SPOTER) | About 3–4× too strong, and it moved the hand between locations. Reduce it and add the location check. |
| `betas[0]` +2 (arms 11–15% longer, wrists moved 68–83 mm) | No recognition evidence. Location must be kept. | Dropped (Tier 2, item 8; §6.1). |
| Non-uniform time warp, k = 0.5 | Uniform rate change only | Replace it. Non-uniform warps can imitate aspect and alter end deceleration. |

### 6.1 Why body shape was dropped

A one-off measurement (2026-10-18; scripts and raw numbers not committed) compared how much arm
proportions differ between signers in MediaPipe's world landmarks with what sampling SMPL-X
shapes produces. It used the 5 pilot clips and the 20 random WLASL clips of Week 2
(seed 20261007). Clip 13630 has no visible left forearm, so the forearm rows cover 24 clips.

| Ratio | MediaPipe, 25 clips: mean / CV (robust CV) | SMPL-X β ~ N(0, I), within the plausibility bound: mean / CV |
|---|---|---|
| Upper arm / shoulder width | 0.748 / 5.7% (2.9%) | 0.865 / 7.5% |
| Forearm / shoulder width | 0.720 / 9.7% | 0.801 / 7.4% |
| Forearm / upper arm | 0.963 / 9.6% | 0.926 / 2.4% |
| Left / right forearm | 0.944 / 20.6% | 1.013 / 0.3% |

- **Signer differences are at the noise floor.** Upper arm / shoulder width varies by about
  3–6% between clips. MediaPipe's own frame-to-frame jitter in the same ratio is 3.7%, and the
  body round-trip error is 3.1% (P21).
- **The forearm spread is detector error.** People are left–right symmetric to within a percent
  or two, yet MediaPipe's left/right forearm ratio varies by 20.6%.
- **The fitted shapes are not a usable sampling range.** 24 of 25 `fit_shape` fits sit on the
  plausibility bound. They mostly encode one constant gap between where MediaPipe and SMPL-X put
  the shoulder joint (P17), not differences between signers.
- **SMPL-X's population is the wrong distribution.** Its mean is 15% off from MediaPipe's, its
  spread about 2.5× too wide, and it varies shoulder width by about 10%, against MediaPipe's 3.1%
  (Finding 10).
- **Limits.** 25 clips, some sharing a signer, so the real spread may be somewhat wider. Three
  clips (57641, 63229, 05631) account for most of the gap between the plain and robust CV.

## 7. Suggested next steps

1. Rework the trial to Tier 1 with the evidence ranges: yaw ±20°, mirror, uniform rate
   0.8–1.25×, arm joints ±4° at p = 0.3, scale 0.85–1.15, and uncentring with both hands kept in
   the frame.
2. Implement the location-region and contact invariants (needs the V1.3 face regions) and use
   them to reject augmented clips.
3. Obtain ASL-LEX 2.0 and the WLASL alignment, and decide the licence question.
4. Prototype within-category MANO handshape jitter with a handshape-feature classifier as its
   check (Tier 2, item 7).
5. Measure the effect on a classifier, one augmentation at a time, as SPOTER's ablation did
   **[M1]**. That is the only test of whether each augmentation helps.
