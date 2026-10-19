# V1 Findings: MediaPipe ↔ SMPL-X Round Trip

**Date:** 2026-10-05 · **Covers:** V1 Steps 0–10 · **Status:** V1 complete; result accepted
2026-10-05.

This is the summary of what V1 established, written as source material for the next
presentation. The sections run in storyline order. Full evidence, method and history are in
[`V1_Problem_Scope.md`](V1_Problem_Scope.md); "Finding n" and "Step n" below refer to it. Every
number regenerates from the scripts in [Section 9](#9-reproduce).

**One unit throughout.** Error is the median distance between a landmark and its round-trip
reconstruction, as a percentage of the clip's shoulder width. At a typical shoulder width of
0.33 m, 1% ≈ 3.3 mm.

---

## 1. The answer

**The question.** Can MediaPipe body landmarks be converted to SMPL-X parameters and back without
losing information?

**The answer.** Through a plausible SMPL-X body, every arm bone keeps its exact direction, and the
arm landmarks come back to within **3.52%** of shoulder width (about 11.6 mm). On frames the body
shape was not fitted to, the figure is **3.99%**. This was accepted as V1's result on 2026-10-05.

**What is lost is not conversion error.**

| Error band | What it is |
| --- | --- |
| 0 → 2.36% | **MediaPipe's own non-rigidity.** Its bones change length 2–15% from frame to frame within a clip. No rigid body, SMPL-X or any other, can follow that. 2.36% is the best any constant skeleton achieves: the *rigid-skeleton floor*, V1's gate. |
| 2.36 → 3.52% | **MediaPipe's skeleton is not a human body.** Its forearms are longer than its upper arms, and some clips have left–right differences no human body model produces. A plausible SMPL-X body cannot copy either. |

In that sense the round trip *denoises* rather than loses: what it discards is variation no
physical body exhibits.

### In context: MediaPipe against itself

How far do MediaPipe's own landmarks move when one reasonable setting changes? Same statistic as
the round trip, same clips, no alignment:

| Change to MediaPipe | Its arm landmarks move (median / p95) |
| --- | --- |
| Temporal smoothing off | 2.41% / 5.67% |
| **The V1 round trip, for comparison** | **3.52%** / 7–15% by joint |
| Static-image mode instead of video mode | 9.22% / 20.04% |
| Heavy model instead of full | 25.47% / 88.46% |

The conversion loses about as much as switching off MediaPipe's smoothing. That is a fraction of
how much MediaPipe's own settings disagree with each other. These are disagreements, not errors
against ground truth, which these clips do not have. *(`measure_detector_agreement.py`)*

---

## 2. How the number moved

| Stage | Error | What changed |
| --- | --- | --- |
| Converter found in the repo | 3.7e-9 | Never evaluated SMPL-X. It subtracted the input and added it back, so this is not a conversion |
| First real conversion, image-space landmarks | 50.05%¹ | Measures coordinate distortion, not conversion: image coordinates divide x by width and y by height |
| MediaPipe world landmarks | 5.94%¹ | Metric, hip-centred input |
| Honest metric: direct error | 6.55% | No free scale fitted at evaluation time |
| Root from shoulders and nose | 6.02% | Bone directions now exact (0.00°) |
| Clean data | 5.92% | One MediaPipe instance per clip |
| **Per-clip body shape (V1)** | **3.52%** | One plausible SMPL-X shape per clip; 3.99% held out |
| Rigid-skeleton floor | 2.36% | The best any constant-shape skeleton can achieve; 3.41% held out |

¹ *Aligned* figures. An extra rotation, translation and scale were fitted at evaluation time,
which hides error. Every figure from Step 7 on is the direct error a consumer of the converter
actually gets.

"Held out" means the body shape was fitted on one half of each clip and scored on the other.

---

## 3. Findings

### The starting point

1. **The converter in the repo never used SMPL-X.** It stored raw bone displacements in the slots
   SMPL-X reads as rotations, then added them back, so its near-zero error was arithmetic.
   `betas` and `global_orient` had no effect. That left the project's three SMPL-X-based
   augmentations non-functional. *(Section 3)*
2. **Four setup facts silently corrupt results if missed:**
   - The SMPL-X weights were never in git; they are licensed and must be installed locally.
   - `smplx.create` needs `model_path="models"`.
   - The forward pass returns 127 joints; the kinematic chain is only the first 55.
   - Without `flat_hand_mean=True`, a "zero" pose silently includes up to 0.85 rad of mean hand
     pose.

   *(Findings 1, 2, 6)*

### MediaPipe's data

3. **Image-space landmarks cannot be fitted to a metric body.** MediaPipe's default output divides
   x by image width and y by image height. The five clips have three different aspect ratios, so
   no single correction exists. Its *world* landmarks are metric and hip-centred. Switching to them
   took the error from 50% to 6%. *(Finding 9)*
4. **Only the arms are observed.** In upper-body signing video, knee and ankle visibility is at
   most 0.03; MediaPipe extrapolates those points. V1 therefore fits:
   - the six arm landmarks (shoulders, elbows, wrists), which drive 4 of SMPL-X's 21 body joints
     plus the root;
   - the nose, used only for body orientation.

   *(Finding 5)*
5. **The five clips show at least four people,** so shape is fitted per clip. Shoulder width
   varies only 2.4% across them, against 10% in SMPL-X's human population, a sign that MediaPipe
   imposes its own body prior. *(Finding 10)*
6. **MediaPipe's skeleton is not rigid.** Within a clip, bone lengths vary 2–15% from frame to
   frame. The drift is slow and image-dependent, and none of the tested explanations accounts for
   it:
   - constant joint-centre offsets explain about 4%;
   - it is not specific to depth;
   - temporal smoothing explains only part of it.

   *(Finding 12)*
7. **MediaPipe's arm proportions are not a human's.** Compared against SMPL-X's population:

   | Clip | Forearm / upper arm | σ from population | Upper arm, left − right | σ from population |
   | --- | --- | --- | --- | --- |
   | SMPL-X population | 0.927 ± 0.024 | | −7.3 ± 1.8 mm | |
   | 01986 | 1.068 | +5.9 | −5.5 mm | +1.0 |
   | 01987 | 0.834 | −3.9 | −8.5 mm | −0.7 |
   | 01988 | 1.084 | +6.6 | +23.6 mm | +17.6 |
   | 01991 | 1.046 | +5.0 | −31.1 mm | −13.6 |
   | 01992 | 1.078 | +6.3 | −1.3 mm | +3.4 |

   - Four of five clips have forearms longer than upper arms.
   - Two have left–right differences no SMPL-X body has, because its shape space is almost
     perfectly symmetric.

   *(Finding 14)*
8. **The extraction pipeline leaked state between clips.** One MediaPipe instance was reused for
   every clip, so each clip inherited the previous clip's tracking state, shifting landmarks by up
   to 114 mm. Fixed, with a test: one instance per clip, which makes extraction deterministic. All
   data were re-extracted and the Step 9 analysis re-run; its conclusions hold.
   *(Step 9, re-baseline)*

### The conversion

9. **Bone directions are reproduced exactly.** The method is analytic swing inverse kinematics,
   solved relative to a root taken from the shoulders and nose. Direction error is 0.00°.
   - It recovers joint *positions*, not rotations. Twist about a bone's own axis moves no joint,
     so V1 sets it to zero by policy.
   - The same decomposition underlies HybrIK. *(Steps 4, 8)*
10. **Torso pitch is invisible to the arm landmarks.** Rotation about the shoulder axis moves no
    arm landmark. The nose sets it so the body is upright and facing correctly in all 409 frames;
    it cannot change the error. Without the nose, the hip centre takes over: still upright in
    409/409 frames, with the root a median 7.0° (max 18.4°) from the nose-based one.
    *(Finding 11; Step 8 note)*
11. **The round trip sees the body through five numbers:** shoulder width and four arm bone lengths.
    For a clip, the squared round-trip error is an exact quadratic in those five numbers. It
    matches the converter to 4e-14, which makes fitting the body shape exact and cheap.
    *(Finding 13)*
12. **One plausible body per clip closes most of the gap.** Fitting shape takes the error from
    5.92% to 3.52%, 68% of the way to the floor. Held out, it reaches 3.99%, 77% of the way.
    - It uses all 300 SMPL-X shape coefficients; the conventional 10 reach 3.90%.
    - It minimises the round-trip error itself; matching median proportions instead reaches 3.68%.
    - It removes the hidden scale. Before, an evaluation-time similarity wanted to shrink the
      output by about 2% per frame, read at the time as correspondence error. With the fitted
      shape the scale is 0.9997: the deficit was body shape.

    *(Step 10)*
13. **A shape prior is essential.** Without the plausibility bound, the fit reproduces MediaPipe's
    proportions with a body 3.81 m tall (10 coefficients, |β| = 337). The bound requires the body
    to be no more unusual than 95% of SMPL-X's population. The data always prefer MediaPipe's
    proportions, so this level is a policy fixed in advance, not a tuned value. *(Finding 15)*
14. **The floor is optimistic on unseen frames.** Fitted on half a clip and tested on the other
    half, even the best constant skeleton scores 3.41%, not 2.36%. MediaPipe's proportions drift
    within a clip. *(Finding 16)*
15. **Bending bones to chase positions is not worth it.** HybrIK's *adaptive* IK aims each bone
    from its reconstructed parent rather than its observed one, so positions improve and directions
    do not stay exact. On top of the fitted shape:

    | | Position error | Bone-direction error (median / p95) |
    | --- | --- | --- |
    | V1, exact directions | 3.52% | 0.00° / 0.00° |
    | Adaptive, 1 pass | 3.43% | 1.21° / 4.13° |
    | Adaptive, 3 passes | 3.68% | 1.81° / 5.46° |

    One pass buys 0.09 points for a degree of distortion in every arm joint, and more passes make
    positions worse. The review assessment's larger gain (6.55% → 5.82%) was measured on the
    template body, where most of it is what shape fitting now provides. V1 keeps directions exact,
    because augmentation perturbs joint angles. *(Study S7)*
16. **The converter no longer invents landmarks.** When an elbow or wrist is not observed, its bone
    stays at the rest pose. That position used to be returned as if fitted, on 61 of 409 real
    frames (14.9%), a median 46% of shoulder width from MediaPipe's own estimate. It is now
    reported as missing. A related fix stops a valid wrist below an invalid elbow from shifting the
    other landmarks. Neither changes the headline, which scores only observed landmarks.
    *(Step 10, contract fixes)*

---

## 4. What changed since Presentation 1

Presentation 1 (2026-10-04) came before any of these measurements. Several of its statements
have since been overturned or refined:

| Presentation 1 said | What we know now |
| --- | --- |
| "Lossless" = round-trip error at or below MediaPipe's own jitter (§5) | Jitter (0.18%) measures something different: a constant offset has zero jitter. The gate is the rigid-skeleton floor, 2.36% |
| MediaPipe's `z` is not metric: "the core difficulty". Forearm drifts 30%; forearm/upper arm 1.42 with `z` (§4) | Those figures came from image-space landmarks. World landmarks are metric: bone lengths drift 2–15% within a clip, and depth is not the dominant cause |
| "Left/right symmetry is fine (2.7% mismatch)" (§4) | That was one clip, in image space. In world space two clips differ left to right by 24–31 mm in upper-arm length |
| V1 is body-only: 12 landmarks, 8 joints (§5) | Legs are unobserved, so V1 fits the 6 arm landmarks: 4 joints plus the root, with the nose for orientation |
| "Only 8 of 21 joints are constrained … 24 meaningful numbers" (§8.2) | 4 of 21 are constrained: 8 degrees of freedom per frame once twist is zeroed, plus the root's rotation and translation |
| Worth re-extracting with `pose_world_landmarks`? (§7.1) | Done. Error fell from 50% (distortion) to 6% |
| Analytic swing IK, with shape refinement only if needed (§5) | Both done. Shape was measured to be needed (Step 9) and fitted per clip (Step 10) |
| Rejected: storing a per-landmark residual (§5) | Still rejected. Its smallest form, five numbers per clip, would reach the floor exactly; it remains a candidate for augmentation |
| Hand position relative to the torso is fully determined by the arm rotations (§8.1) | True once the root is fitted from the torso (Step 8). The review assessment found the earlier fit's body upside down |
| World landmarks exist "for the 33 pose landmarks; not face or hands" (§7.1) | Not for the legacy Holistic solution used here. MediaPipe's separate hand models (legacy Hands, the Tasks API) do give hand world landmarks, each in its own hand-centred frame (review assessment). Relevant to V2 |

---

## 5. Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-01 | Define "lossless" against a measured threshold, not bit-exactness | A bit-exact round trip is impossible: the detector is non-rigid, and joint centres differ between the two models |
| 2026-10-01 | V1 is body only; hands and face are V2 | The smallest honest unit |
| 2026-10-01 | Analytic swing IK; shape only if measured to be needed | Deterministic, testable bone by bone, with twist as an explicit policy |
| 2026-10-04 | Fit MediaPipe world landmarks | A metric body needs metric input |
| 2026-10-04 | Build the converter separately; deprecate the legacy classes | Their callers pass image-space data, and no camera model exists yet |
| 2026-10-05 | The gate is the rigid-skeleton floor | It uses the same statistic as the score; jitter does not |
| 2026-10-05 | Shape: per clip, all 300 coefficients, 95% plausibility, minimising round-trip error | Each choice beat its alternative, fitted and held out (Step 10) |
| 2026-10-05 | **Accept 3.52% (3.99% held out)** | The remainder is MediaPipe geometry outside the human shape space |
| 2026-10-05 | Keep bone directions exact; reject adaptive IK | 0.09 points of position error is not worth a degree of joint-angle distortion |
| 2026-10-05 | Report landmarks the fit could not place as missing | Invented positions would pass into augmented data unmarked |

---

## 6. Limits of the V1 result

- **Positions, not rotations.** Twist is unobservable from joint positions and set to zero, so
  palm orientation is not represented.
- **No shrugs.** Collar motion is not representable; the test for it is a strict expected
  failure.
- **Arms only.** Legs are unobserved in this footage, and hands and face are V2.
- **The body shape is a convention.** Five proportions cannot identify 300 coefficients. The
  fitted body is the most typical human closest to MediaPipe's skeleton, not the signer's anatomy.
- **World space only.** Feeding the image-space augmentation pipeline needs a world-to-image
  camera model.
- **Five clips, at least four people.** The figures describe this data set, not MediaPipe in
  general.

---

## 7. Open for the next phase

1. **Does the conversion change the training distribution?** Augmented samples made through SMPL-X
   have rigid bones and human proportions; real MediaPipe data has neither. A classifier trained on
   both may see two distributions. This is *inference, not yet tested*, and testable once the
   converter is wired into augmentation. The five-number per-clip correction is the candidate fix
   for proportions.
2. **Integration blockers.**
   - Verified:
     - a world-to-image camera model is needed;
     - intrinsic augmentation must be temporally coherent, with one perturbation per clip rather
       than per frame (Scope, Section 11).
   - Reported by the review assessment and not yet re-verified:
     - the joint-limit table puts the elbow hinge on the wrong axis;
     - the SMPL-X augmentations are refused by landmark-mode chains;
     - the presets are landmark-only, and the YAML configs fail to load;
     - the augmented dataset mishandles parameter dictionaries;
     - limb-length scaling cannot express per-bone changes with global `betas`.

     It also sketches a camera design: a per-frame weak-perspective fit, with a 6.6% median
     residual.
3. **Twist is a policy with consequences.** Zero twist is invisible to the round trip, but the
   review assessment measured augmented wrists 4–13 mm apart under different twist choices. A
   hidden twist can also move the mesh by up to 86 mm with identical joints.
4. **Should shape be fitted per signer rather than per clip?** That needs held-out evidence across
   clips (V2).
5. **Still open from Presentation 1:**
   - A valid held-out classification benchmark.
   - The SMPL-X-rotations-as-features ablation (§8). The fitter it needed now exists.

---

## 8. Related work

Not yet cited in the V1 scope; found by the review assessment. Licence terms are as it reports
them.

| Work | What it does | Relevance |
| --- | --- | --- |
| **SGNify** (Forte et al., CVPR 2023) | Fits SMPL-X to sign-language *video*, with linguistic priors for the hands | The closest prior art. It starts from video, so it does not replace a landmark round trip. Its hand priors matter for V2. Non-commercial |
| **SignAvatars** (Yu et al., ECCV 2024) | A large 3-D sign-language motion dataset with SMPL-X/MANO fits, including a WLASL-derived subset | If our clips are WLASL videos (their IDs suggest so; not verified), its fits could cross-check our arm pose and shape. The fits are estimates, not ground truth. CC BY-NC-SA, access on request |
| **mmhuman3d** (OpenMMLab) | SMPLify(-X) fitting with a 3-D keypoint loss and a MediaPipe keypoint convention | An off-the-shelf, optimisation-based comparator for V1's analytic fit. Apache-2.0 |
| **HybrIK** (Li et al., CVPR 2021) | Analytic swing–twist IK, with twist estimated from the image | The decomposition V1 uses; its adaptive variant is Finding 15 |

---

## 9. Reproduce

```bash
python scripts/extract_world_landmarks.py     # once: one MediaPipe instance per clip
python scripts/measure_round_trip.py          # headline: 3.52% fitted, 3.99% held out, floor 2.36%
python scripts/study_shape_fit.py             # Step 10 evidence, S0–S6; adaptive IK, S7 (slow)
python scripts/diagnose_residual.py           # what lies below the floor (--unsmoothed adds E6)
python scripts/measure_detector_agreement.py  # MediaPipe against itself (runs MediaPipe; slow)
python -m pytest                              # 103 passed, 2 expected failures
```
