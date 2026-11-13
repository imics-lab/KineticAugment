# KineticAugment V1: empirical test of the review, final assessment

**Date:** 2026-10-05.
**Subject:** the snapshot of the user's working tree. Step 7 is marked COMPLETE in `docs/V1_Problem_Scope.md`. The suite gives **66 passed, 13 failed, 2 xfailed**; all 13 failures are the Step 8 red-phase tests, which need `SMPLXSkeleton.rest_nose`. They are in-progress TDD and are not counted as findings.
**Method:** ten claim clusters. Each was tested by one agent and re-measured independently by a second agent with separately written code. Where the two disagreed, the version with reproducible computation is used below.

---

## 1. Bottom line

**The review improves the V1 direction and is mostly correct.**
- All seven ledger claims (A001–A007) survived independent re-measurement.
- Every number it quotes reproduces. The root counterexample reproduces byte for byte: 1.207e-6°, 319.15 mm, 79.18 mm, 44.70 mm.
- The plan has already absorbed most of it: the Step 6 verdict was retracted, the direct round trip (6.55%) replaced the aligned 5.94% as the headline, the noise floor became a reference, the shape claim was bounded, and V1 is now world → SMPL-X → world.

**The review's own errors are minor:**
- a one-pose number written as a bound;
- a wrong "one capture setup" remark;
- an unofficial citation mirror;
- "four" length observables where the current fit has five;
- root pose listed too broadly as a shape confounder;
- a recommendation for confidence weighting with no valid source of confidence named. The doc then chose MediaPipe `visibility`, which does not track positional noise.

**Its real weakness is what it missed.** It correctly treated the unestimated root as an error magnitude (about 13–19% of the residual). It did not notice that on real data the fitted SMPL-X body is upside down: `global_orient` is about identity in MediaPipe's y-down frame, the head is below the shoulders in 409/409 frames, and every elbow is encoded as hyperextended. The arm-position metric is mathematically blind to this. The review's own Finding 2 asked for a plausibility check, which would have caught it, and that is the one recommendation the plan did not adopt.

**Other misses:**
- the joint-limit table puts the elbow hinge on the wrong axis (latent until augmentation is wired);
- the extractor carries tracker state from one clip into the next;
- the Step 6 "oracle" decomposition is mislabelled and non-additive.

**"Step 7 COMPLETE" is not complete by the doc's own definition of done.**
- DoD items 1–2 and Deliverables 1–2 still name the deprecated legacy classes.
- DoD item 3 demands rotation recovery, which is unattainable.

**The single most important next step:** implement Step 8 (root from shoulders + nose) and gate it on a real-data, MediaPipe-convention plausibility check: head above shoulders, and elbows flexed on the anatomical side. Every downstream use of the parameters inherits the inversion until that is fixed: `body_pose` augmentation, joint limits, the mesh, and the shape fit.

---

## 2. Claim-by-claim

### 2a. The review's claims

Where this table says "true at review time" and the claim concerned the doc, it holds at least for the doc text that survives. The pre-revision doc and the reviewed `measure_round_trip.py` are untracked in git, so earlier wording cannot be checked directly.

| Review claim | Verdict | Key evidence (numbers) | Addressed in current doc/code? |
|---|---|---|---|
| **A001.** The skeleton/FK path matches real SMPL-X joints, within 1.37e-7 m. | **Confirmed.** The wording overstates its precision. | 60 random full poses with shape \|β\| up to about 9, root σ 1.5 rad, transl σ 1.5 m, all 55 joints: max 4.3–4.5e-7 m (float32). Shape-blind control: 130–303 mm, so `J_regressor` does use shape. 1.37e-7 is a typical *single-pose, β=0* value, not a bound. | Holds in code. `tests/test_kinematics.py:141-158` covers only β=0, σ 0.3, no transl, joints[:22]. Only the NEUTRAL model is present, so male/female cannot be checked. |
| **A002 / F1.** Root orientation affects position error, so "cannot affect the gate" is false. | **Confirmed.** | Counterexample reproduces exactly (independent truth from the real smplx forward). Rigid-converter mean is 49.14 mm. Only joints 0, 3, 6, 9, 13 and 14 move a shoulder; the IK joints 16–19 cannot. Real-clip share (§4.4): 12.7–18.6% of the residual depending on estimator and metric; clip-bootstrap CI [2.7, 26.3]%; driven largely by clip 01991 (9.20 → 5.07%). | Retracted at L509-534, and Step 8 is planned. **Not struck:** L503-507 ("must come from bone length") and L595-600 ("direction exact by construction"). |
| **A003.** The 200-pose harness proves positions, not rotations; terminal twist is unidentifiable. | **Confirmed, and understated.** | Replica scoring rotations: median 18.06°, max 75.24°, 0/200 poses recovered. Shoulder error equals the true humerus twist to 5e-12°. Under SMPL-X's 3-DOF elbow, each arm has a **2-D** null space that includes a *compensated humerus twist*: all 55 joints stay within 1.6e-9 m while vertices move up to 86 mm. | Partly. L579-583 is corrected, but L176 (DoD 3), L686 ("Rotations … recovered to machine precision") and L746-749 (5a pass condition) still claim rotation recovery. |
| **A004.** 0.18% is not a calibrated pass/fail noise floor. Smoothing is on, Method B assumes iid noise, and it is the wrong statistic. | **Confirmed.** | Smoothing is on (`holistic.py:80`). The filter is identified exactly as One Euro (0.1, 40, 1.0), to 1.2e-7 m. Same predictions: 0.175% smoothed, 0.743% unsmoothed (4.2×), 1.363% static mode (7.8×). Decimating to 15/10 fps gives 0.434/0.660%. Method B's speed gate alone reads 0.38–0.39× on stationary iid noise. The round trip barely moves (6.42–6.55%). Same-statistic repeatability under benign perturbations is 1.9–3.6% direct. | Principle adopted (Correction box L53-75). Gate wording remains at L47-51, L178, L219-220, L331 ("V1 error budget 0.18% ≈ 0.6 mm"), L573 ("33× the budget"), L944, L952. `measure_noise_floor.py:214` still prints "PROPOSED V1 ERROR BUDGET". |
| **A005.** 5.94% is not the delivered converter's direct round trip. | **Confirmed.** True at review time, now fixed. | Old pipeline reproduces 5.944% exactly. Current converter: 79 numbers per frame → real smplx forward gives **6.551%** (p95 13.73%), matching `reconstruct_body` to 1.70e-7 m on 409 frames. The 0.607-point gap is entirely the free scale (median 0.9726). | Addressed by Step 7. Aligned figures still headline at L571, L573, L599 (6.66% length; direct is 7.86%) and in the Step 6 decomposition. |
| **A006.** Four arm lengths cannot identify 10 betas + scale; a regularised solution is a convention, not anatomy. | **Confirmed.** Identification is even weaker than a 5-vs-11 count suggests. | Length Jacobian (real smplx autograd): 5×10, rank 5, singular values 41.5 / 17.6 / 2.87 / 0.47 / 0.045 mm/σ, condition number 923. Adding scale adds no rank. L/R rows are near-duplicates (cosine 0.998, 0.99994). Effective dof under the N(0,I) prior: 1.3–3.1. Two bodies with identical 5 lengths (to 1.5e-11 mm) differ by 12.3 cm in height and 28 L in volume, and the converter cannot tell them apart (0.25 mm vs a 24.6 mm control). | Adopted in Step 10. Stale contrary text at L209-216 ("betas and a global scale closes that gap") and L721-725 (see §4.10). Step 10 lacks held-out evaluation and a prior-strength policy. |
| **A007.** The public classes still encode/decode raw displacements. | **Confirmed.** | `body_pose` slots are bitwise equal to child − parent; magnitudes equal bone lengths. Through real SMPL-X: 189.6 mm median mean-arm error (57% SW) vs 19.9 mm (6.0%) for `fit_body`. Zero params + `target_mp` reproduce the input to 1.2e-7, so the params carry nothing. | Resolved by decision: deprecated, not rewired. DoD 1-2 (L172-175) and Deliverables 1-2 (L927-928) still require them to work, so they cannot be met. |
| **F2.** Validity ≠ fidelity ≠ plausibility. The scope requires a plausible mesh but excludes assessing it. | **Confirmed, and understated.** | DoD 1 asks for a "plausible mesh"; L951-953 says V1 "does not measure" it. On 409 real frames the delivered fit is upside down and back-facing in 100% of frames, with every elbow hyperextended (§4.1). Identical 55-joint skeletons can have meshes 30–91 mm apart. | **Not addressed.** No plausibility test exists. |
| **F3.** Numbers reproduce; post-alignment direction error is 2.1–2.8°; decompose in a common frame. | **Confirmed.** | Per-bone medians 2.57 / 2.65 / 2.81 / 2.13°, p95 4.55–5.41°, identical for the direct and aligned outputs (the Umeyama rotation does not depend on scale). In a common frame the direction error implies a tip displacement of 3.22% SW, about half the 6.66% length term. The quoted word "negligible" cannot be checked (the reviewed script was untracked and has been rewritten). | Not addressed. L600 still reads "exact by construction". The current harness reports no direction error at all. |
| **F4.** World coords are the right baseline but not ground truth; visibility is not uncertainty; jitter ≠ accuracy; 0.6 mm is a unit conversion. | **Confirmed.** | The proto has only x, y, z, visibility, presence; Holistic clears `presence`. Spearman ρ(visibility, jitter) is −0.15 to +0.18; against tracked-vs-static disagreement it is +0.10 to +0.22. The full and heavy models both jitter about 0.2% but disagree by a median 25.7% SW. Published BlazePose GHUM Full 3-D MAE is 39 mm (yoga, GHUM-fitted pseudo-GT, not like-for-like). 0.175% × 315.9 mm = 0.55 mm. | Partly. The Correction concedes jitter ≠ accuracy, but L331 is unannotated. Step 10 (L918-919) uses visibility as weights, which contradicts this finding. |
| **F5.** The doc needs one contract; it mixes image vs world, 12 vs 6 landmarks, plan vs done, position vs rotation; the 7-number similarity is not a camera. | **Confirmed.** Duplication is wider than stated. | `fit_body` sets 4 body_pose joints in 351/409 frames and never 8. A world → image similarity leaves a 50–62% SW residual; even affine or weak-perspective models leave 6–19%; the image-z gain is 3.03×. Steps 2/3, 4 and 5 each have done-and-plan headings, and Step 7 now has three (L786, L809, L872). | Code and the Step 7 sections adopt world-only. Not propagated to L42, Section 4 (L124-158), DoD, Deliverables, Sections 9–10, the plan texts, or `alignment.py:4-9`. |
| **Rec 1.** I/O contract: validity/confidence, sequence identity, observed-vs-defaulted, param-only reconstruction. | **Confirmed.** Partly adopted. | Param-only reconstruction, the validity mask and the rigid fold are done. **Missing:** an observed/defaulted flag (live on real data, §5), confidence, sequence identity, and betas in the contract (§4.5). | Partly. |
| **Rec 2.** Fix the root; test rotated roots, torso, shape, missing joints and origin points; the nose is a heuristic. | **Confirmed.** | Step 8 red tests cover rotated root, spine bend, shape, a missing wrist, both shoulders and pitch invariance. A ~40-line prototype passes all 14. Nose leak: a 20° neck bend becomes 18.8° of false torso pitch, while a pure 30° head nod leaks 7.5°. Nose root vs hip root: median 8.5°, max 22°. The apex choice cannot change the arm metric (agreement to 5.6e-16 m). | Adopted as Step 8, not yet implemented. Missing: a MediaPipe-convention (y-down) fixture, the origin case, the missing-elbow case. |
| **Rec 3.** Analytic IK as baseline; small regularised fit; scale fixed; no uniqueness claim. | **Confirmed.** | Converter uses `allow_scale=False`. Scale is cheap to mimic with betas (about 0.9σ per 1%), and its posterior barely shrinks (0.98× of prior). | Adopted, but the regularised fit is demoted to an optional item after Step 9. |
| **Rec 4.** Report direct mm and %SW, per-joint stats, a common-frame decomposition, separate Procrustes, root-only / shape-only / both ablations, held-out data. | **Confirmed.** | Held-out evaluation matters: the MAP-betas gain is 18.9% in-sample vs 10.1% held-out (contiguous halves; 17.8 vs 12.3% for thirds). | Mostly adopted. Missing: mm units, held-out evaluation, common-frame directions. |
| **Refs.** HybrIK = analytic swing + extra twist information; SMPLify-X = fitting with priors. | **Confirmed.** Thin. | Paper text verified. Omitted: HybrIK's *adaptive* IK (§4.8). SMPLify-X fits 2-D keypoints, so metric world landmarks need a 3-D keypoint term. | n/a |
| **Checks.** 27 targeted tests pass; synthetic max 2.993e-16 m. | **Confirmed.** | `test_swing_ik.py` (12) + `test_kinematics.py` (15) = 27 passed. | n/a |

### 2b. The doc's Step 7 contract claims (added after the review)

| Doc claim (Step 7) | Verdict | Evidence |
|---|---|---|
| The SMPL-X forward pass on the returned params reproduces `reconstruct_body` within 1e-5 m. | **True only for β = 0.** | 1.7e-7 m on 409 real frames. With a shaped skeleton it misses by 21–186 mm (\|β\| 2–6.7), because `fit.betas` is always zero. That contract test is currently red (Step 8 fixture), so only the signature test protects it. |
| The rigid frame folds exactly into `global_orient`/`transl`. | **Confirmed.** | 6e-15 m against an independent R·FK + t. Without pivot compensation: 84–341 mm. Exact folding does not make the frame *estimate* right: a whole-body rotation gives 17 / 52 / 150 / 260 mm at 10 / 30 / 90 / 180°. |
| Reconstruction takes only (fit, skeleton); no side-channel state. | **Literally true, overstated in spirit.** | `fit.betas` is never read (random betas change the output by 0.0 m); shape travels in the skeleton object. |
| NaN where not modelled; invalid landmarks never influence the fit. | **True as stated, but the principle behind it is broken.** | 27/33 rows are NaN, and junk values in invalid rows changed the fit 0 times in 820–840 comparisons. However, invalid **modelled** landmarks come back as finite rest-pose values with nothing marking them. This happens in **58/409 real frames (14.2%)**, a median 67.5% SW from MediaPipe's own estimate. |
| Fewer than 3 correspondences raises. | **True, but 3 is not sufficient.** | Any single-arm subset leaves the torso frame unobservable: 288–301 mm error on the unobserved shoulder, collinear or not. Orphan wrists are a separate failure (§4.6). Step 8's both-shoulders rule closes the single-arm case only. |
| Synthetic "70 mm worst at 30°". | **Accurate, but worst-case.** | 7.003e-2 m reproduces exactly; the mean over the same 200 trials is 27.8 mm. |

---

## 3. Errors and overreach in the review (verified only)

1. **A001's precision is presented as a bound.** 1.37e-7 m is one random pose with β=0. With shape and translation the float32 envelope is about 4–5e-7 m. The conclusion stands.
2. **The measurement audit says "five clips from one capture setup."** This is wrong. The clips are 320×240 at 25 fps, 1280×720 at 29.97, 640×480 at 30, and two at 288×192 / 29.97, and the doc's Finding 10 identifies at least 4 people.
3. **The smoothing-default citation is `chuoling.github.io`**, an unofficial mirror. The content matches the installed `holistic.py:80` and the upstream graph, so this is a sourcing issue only.
4. **"Four observed arm lengths."** Under the current two-shoulder rigid fit with collars at zero, shoulder width is a fifth observable. This is negligible: four matched the 4-bone table in the plan the review read, and the conclusion (underdetermined) holds either way.
5. **The smplx audit lists root pose next to scale and correspondence offsets as a shape confounder.** Root error leaves bone lengths untouched. It biases betas only in a weakly regularised *position-residual* fit (about 2σ of β drift for a 0.3 mm error reduction), so it is much weaker than the other two.
6. **It recommends a "confidence-weighted robust fit" without naming a valid confidence source.** This is consistent with its own warning that visibility is not uncertainty, but it is under-specified. The doc then adopted `visibility` as the weight (L918-919), which the measurements contradict (ρ ≈ 0).
7. **The proposed ablation (root-only / shape-only / both) omits the joint-centre offset factor**, the doc's leading Step 9 hypothesis. It also does not warn that the decomposition is non-additive under a free-scale metric (§4.4).
8. **A003 is understated.** The null space is not just terminal-segment twist: a compensated, non-terminal humerus twist is invisible to all 55 joints.
9. **The decisive omission:** the review did not detect the 180° torso inversion (§4.1), even though its Finding 2 and its "fixed coordinate convention" wording point straight at it.

**Criticisms of the review that testers raised and verifiers rejected.** These are listed so they are not repeated:
- *"The doc never claimed the public classes used the new path" (A007).* Refuted. Finding 7 (L447) said the six public-class xfails should flip when Step 4 landed, and Step 4 is marked complete.
- *"'Most consequential' overstates root's magnitude."* The review explicitly disclaimed the real-clip magnitude and ranked by consequence: root invalidated the Step 6 verdict and reordered the plan.
- *"Non-identifiability is caused by using only six landmarks."* Rejected. The review scoped its claim to six joint positions, and the ball-joint null space survives even with full hand landmarks unless a prior is added.
- *"Observed A/B ratios contradict iid noise."* Refuted. iid noise plus real motion reproduces the observed ratios (0.73 / 1.10 / 1.64 vs 0.72 / 1.03 / 1.57).
- *"The doc's 32% hip figure is wrong."* Refuted. It reproduces (31.7% / 32.1%) as an in-frame fraction. The hip centre sitting at the world origin is a MediaPipe convention, not an observation.

---

## 4. Errors the review missed (verified only), ranked by severity

"Existed at review time" means the defect was present in what the review audited. "Post-review" means it entered with the Step 6/7 work.

### 4.1 HIGH: the fitted SMPL-X body is upside down and back-facing in 100% of real frames (existed at review time)

**Where:** `src/kinetic_augment/body_model/converter.py:79-96`. IK runs with an identity root, then Kabsch on arm joints only. MediaPipe world coordinates are y-down with the face toward −z; SMPL-X is y-up and z-forward. No axis convention is stated in the doc or tested.

**Evidence** (5 independent reproductions; re-run for this report):
- Fitted up-axis is a median **169.4°** from the data's hip→shoulder up.
- `global_orient` is a median 3.6° from identity and 178.5° from Rx(180°).
- The SMPL-X head joint is above the shoulders in **0%** of frames (MediaPipe nose: 100%) and sits 380 mm from the nose. The pelvis is 300% SW from MediaPipe's hip centre.
- Elbow above shoulder in 100% of arms (truth 0%); wrist in front in 0% (truth 100%).
- Applying C = diag(1, −1, −1) first gives an up-error of 10.7°, head above the shoulders 100%, and direct error 21.65 vs 21.60 mm.

**Why no metric sees it:** a 180° turn about the shoulder axis moves no arm landmark (≤5.6e-16 m about the exact rest shoulder axis).

**Consequences:**
- Every elbow is encoded as about 85° of hyperextension (L local-y +84°, R −86°).
- A "+10° flex" through local angles *straightens* the bend by 9.5–9.8°. With a shoulders+nose root it gives +9.7°.
- `presentation_one.md:182-183` ("hand location relative to torso is fully determined by the arm rotations") is false for the shipped converter: the wrist is on the wrong side of the shoulder in 98.6–100% of frames.

**Fix:** Step 8's nose-apex root absorbs C (head above shoulders 100%, elbows 100% anatomical). However:
- no current or red test uses MediaPipe's convention (roots are ≤40° in the SMPL-X frame);
- the planned degenerate-apex fallback (minimal shoulder-line rotation or identity) lands 150.8–172.1° from the nose root on every real frame, so it would re-invert the body and make `global_orient` jump about 160° on a nose dropout. This is latent: nose visibility is ≥0.9993 in these clips.

**Repro:** `work/verify-missed-bugs/v_convention.py`, `work/verify-direction-and-blocker/v1_orientation.py`, `v3c_flip_fixed.py`.

### 4.2 HIGH, latent in V1: the joint-limit table puts the SMPL-X elbow hinge on the wrong axis (existed)

**Where:** `src/kinetic_augment/body_model/joint_mapping.py:279-289`. It is used by `JointAnglePerturbation(respect_limits=True)` (`intrinsic.py:139-142`), `JointLimitConstraint` and collision resolution.

**Evidence:**
- A 60° rotation about the elbow's local x moves the forearm only 3.7–5.3°, so x is twist. Local y is flexion: −y on the left, +y on the right.
- The table labels x as flexion over (0, 2.62) and restricts y to ±0.17.
- `clamp_joint_angles('left_elbow', [0, -1.571, 0])` returns [0, −0.17, 0], moving the wrist 325 mm.
- Even after a Step-8-style root, a zero-noise clamp moves wrists a median **76% SW** (p95 102%) and cuts the elbow bend from 78° to 23°.
- The shoulder rows look mislabelled in the same way.

**Status:** latent because the converter is not wired into augmentation. It is the first thing to break at integration.

**Repro:** `work/verify-twist-identifiability/v5_limits.py`, `work/verify-direction-and-blocker/v6_elbow_axes.py`, `v2_step8_flex_clamp.py`.

### 4.3 MEDIUM: the extractor reuses one MediaPipe Holistic instance across all clips, so stored landmarks depend on processing order (existed)

**Where:** `scripts/extract_world_landmarks.py:74-83`. Compare `src/kinetic_augment/evaluation/datasets/wlasl.py:242`, which creates one instance per video.

**Evidence:**
- A shared-instance re-run reproduces the stored `.npz` bit for bit. Fresh per-clip runs are deterministic.
- Shared vs fresh, frame-0 max arm deviation: 01987 114 mm, 01988 79, 01991 24, 01992 103. Second-half medians: 01987 34 mm, 01992 25 mm.
- Processing in reverse order changes a *different* set of clips (01986: 168 mm at frame 0), which shows the order dependence directly.
- Both tracker ROI and filter state leak. MediaPipe also stamps every frame at 30 fps (`solution_base.py:312-314`), although 01986 is 25 fps.

**Impact:** small on the headlines; direct 6.55 → 6.45%, aligned 5.94 → 6.05%, and the "free scale hides 0.61 pts" gap becomes 0.40. Finding 10's 01987 lengths shift by up to +4.9%, but its CV conclusions survive (shoulder-width CV 2.1 → 2.4%).

**Repro:** `work/verify-external-literature/vx_extract.py` then `vx_compare.py`.

### 4.4 MEDIUM: the Step 6 decomposition is mislabelled, non-additive and unscripted (post-review)

**Where:** doc L646-663, L772-776, L902.

**Evidence:**
- **No script reproduces it** (grep finds no oracle code).
- It reproduces exactly only as: sim metric + **analytic 2-DOF shoulder-line root** + per-clip median arm lengths, with shoulder width left at rest. That also reproduces 01988's 5.06 → 5.11%.
- The "root oracle" is therefore not an oracle. A true 3-DOF optimum gives **4.84% aligned (−18.6%)** and improves 01988 to 4.43%.
- The shape oracle omits shoulder width; including it gives 38.5% alone and 61.4% with root.
- On the direct/rigid metric: root 12.7%, length 34.5%, both 47.1%, interaction −0.2, leaving **52.9%**, not 44.5%.
- The positive interaction on the sim metric (+6.8 pts) has a frame-bootstrap CI of [3.8, 10.2] but a clip-bootstrap CI of [−7.7, 17.6], so it is not established at clip level.

**What the "unexplained 44.5%" really is:** it is *located*, not explained. Once directions are exact, per-frame segment lengths plus shoulder width reproduce the input exactly, which is a tautology. The split is order-dependent: 19–34 points from frame-to-frame length variation and 6–20 from per-clip shoulder width. It is not a floor either: a per-frame 3-DOF root with per-clip lengths and width reaches 1.97% (sim) / 2.18% (rigid).

**The scale 0.9726 is not evidence of correspondence error** (doc L878-880 claims it is). It goes to 0.995 under per-clip lengths+width, so the cause (shape vs a constant offset) is unidentified.

**The retraction table at L525-530** (9.87/81, 16.55/119, 47.04/307 mm) matches no swept configuration.

**Repro:** `work/verify-real-data-numbers/v_oracle.py`, `v_boot.py`, `v_floor.py`; `work/verify-root-counterexample/v3_optroot.py`; `work/verify-doc-consistency/v_counterexample.py`.

### 4.5 MEDIUM, latent: `BodyFit.betas` is a dead field (post-review)

**Where:** `converter.py:52` (defaults to zeros), `:92-96` (never set), `:99-114` (`reconstruct_body` never reads it). `skeleton.py:48-80` does not store betas at all.

**Evidence:** with a shaped skeleton, the real SMPL-X forward on the returned params misses `reconstruct_body` by 42–112 mm (\|β\| 2–6), against 2e-7 m when the true betas are passed.

**Impact:** this falsifies "reconstruction takes only the parameters" for any shaped body, and breaks as soon as Step 10 fits shape. No test covers it.

**Repro:** `work/verify-fk-and-wiring/v3_contract.py`.

### 4.6 MEDIUM, latent: orphan correspondences corrupt the rigid fit (post-review)

**Where:** `converter.py:72` admits any valid landmark as a Kabsch pair, while `swing_ik.py:127` leaves a bone at rest if either endpoint is invalid.

**Evidence:**
- Left elbow invalid with wrist valid gives an error on *fully observed* landmarks of median 60–86 mm, max 150–385 mm across four independent runs (statistic-dependent). A missing shoulder gives 25–37 / 74–126 mm. A missing leaf (wrist) gives 0.
- Real clips: 0/410 frames affected.
- Related: an exact-origin landmark is dropped by IK (`swing_ik.py:77`) but kept by Kabsch, giving 22–86 mm errors. This is unreachable on real float32 data.

**Repro:** `work/verify-missed-bugs/v_orphan.py`, `work/verify-fk-and-wiring/v4_orphan_collinear.py`, `v4b_single_arm.py`.

### 4.7 MEDIUM: Finding 4's "z is 1.2× x/y, no axis weighting needed" divides a two-axis norm by a one-axis value (existed)

**Where:** `scripts/measure_noise_floor.py:134, 211-213`; doc L339-349.

**Evidence:**
- Isotropic noise would print "0.70×". Per axis on the real clips (Method B): x 0.043%, y 0.033%, z 0.074%, so z/rms(xy) = **1.92** (Method A: 3.27).
- Smaller related issues:
  - Hips are pooled into the floor even though the hip-centred origin pins them. Arm-only, the floor is 0.22%.
  - The speed-gate bias is 2.5×.
  - The frame-rate dependence comes from motion leaking into the estimators, not from temporal correlation.

**Repro:** `work/verify-missed-bugs/v_noise.py`, `work/verify-noise-floor/verify.py`, `ratio.py`.

### 4.8 MEDIUM: HybrIK's adaptive IK was not considered (existed)

**Where:** `swing_ik.py:127-133` uses the naive form (direction = observed child − observed parent).

**Evidence:**
- Adaptive IK (direction = observed child − *reconstructed* parent, alternated with the rigid refit) gives a direct error of **6.55 → 5.82%** after 3 iterations, or 5.64% after 1, with p95 13.73 → 12.30%.
- The template-length least-squares floor is 5.23%.
- No new parameters; the cost is giving up exact bone directions.
- It answers the §9 fallback "analytic IK accumulates error" without full gradient fitting.
- The "SVD root solve not adopted" half of the external-literature tester's claim is wrong: `fit_body` already does Umeyama rigid registration. What it lacks is root-before-swing ordering, which Step 8 adds.

**Repro:** `work/verify-external-literature/vx_adaptive.py`.

### 4.9 MEDIUM: sign-language SMPL-X prior art is not cited (existed)

- **SGNify** (CVPR 2023; non-commercial) fits SMPL-X to sign-language *video*.
- **SignAvatars** (ECCV 2024; CC BY-NC-SA, access by request) ships per-video SMPL-X/MANO annotations, including a WLASL-derived subset.
- **mmhuman3d** (Apache-2.0) has a MediaPipe keypoint convention and SMPLify(-X) registrants with a 3-D keypoint loss.

None of these replaces a landmark-array round trip, but they are ready-made baselines (see §5.4).

### 4.10 MEDIUM: the doc says the forearm/upper-arm ratio is "precisely what betas exists to absorb" (existed)

**Where:** L721-725.

**Evidence:** observed fore/upper ratios are 1.046–1.102 (4/5 clips). Within the 95% prior radius (\|β\| ≤ 4.28) SMPL-X reaches at most 0.990, and matching the observed ratios takes \|β\| ≈ 10. The passage also compares a pooled observed ratio with a left-only template ratio. Finding 10 and Step 9 contradict it, but it was never struck.

**Repro:** `work/verify-shape-identifiability/v5_misc.py` (section c).

### 4.11 LOW: smaller verified items

- **Doc self-contradictions:**
  - DoD 1-3 and Deliverables 1-2 and 4 cannot be met under the Step 7 decision.
  - The "Blocked … raised for a decision" text (L867-868) sits beside "Wiring decision" (L788).
  - "Step 7" still means the shape step at L281, L507, L635, L640, L663, L725 and L941, and L782 says "Steps 7–9" where the plan has Steps 7–10.
  - Suite counts disagree (L800 vs L813-814).
  - L458 says `SMPLXSkeleton` wraps `SMPLXWrapper`; it does not.
  - L990 cites `mp_to_smplx.py:96`, which is now `:104`.
  - `body_model/__init__.py:9-17` advertises the deprecated class with the wrong constructor, and `fit_body`/`reconstruct_body` are not in `__all__`.
- **`presentation_one.md`** (local notes) still says lossless = jitter, treats the noise floor as the target, and describes 12 landmarks / 8 joints and "24 meaningful numbers" (measured: 12 in 351/409 frames). It also has a unit error: "a metre" is 344 px.
- **`scripts/test_phase1.py:343-350`** is a third legacy caller that the Blocked list omits. It already raises TypeError (missing `target_mp`); the error is caught and printed.
- **`fit_body` accepts image-space input silently.**
- **Rigid post-fit vs translation-only after a root estimate:** 5.72% vs 6.02% direct, but the rigid version tilts bone directions by 0.16°. This trade-off is in a test comment only, not in the doc.
- **Hand world landmarks exist** (legacy Hands; Tasks HolisticLandmarker), each in its own hand-centred frame. `extract_world_landmarks.py:20-22` and `presentation_one.md:157` say they do not.
- **Twist policy is not neutral for augmentation:** identical joints with different twist give augmented wrists 4–13 mm apart.
- **Side effect of this audit, not a project defect:** an earlier run downloaded `pose_landmark_heavy.tflite` (27.7 MB, mtime 2026-10-04 23:59) into the user's real venv at `.venv/lib/python3.11/site-packages/mediapipe/modules/pose_landmark/`. It is an on-demand cache file and safe to delete.

---

## 5. Direction

### 5.1 What the revised plan adopted

- Revise before shape: Step 6 retracted, Steps 7–10 introduced.
- World → SMPL-X → world contract with parameter-only reconstruction, a validity mask, the rigid frame folded into `global_orient`/`transl`, and NaN for unmodelled landmarks.
- Root handling as Step 8, with red tests for rotated roots, spine bend, shape, a missing wrist and both shoulders.
- Direct and aligned errors reported side by side, per joint, with p95 and coverage.
- The noise floor treated as a reference (principle only).
- Shape treated as underdetermined: prior, per-clip, bounded claim, scale fixed.
- Positions-not-rotations in the Step 5 result (not propagated).
- Image-space return deferred to V2.

### 5.2 Where the plan diverged, and who is right

| Divergence | Review | Plan | Assessment |
|---|---|---|---|
| Plausibility check | Add a small mesh/pose sanity check or narrow the "plausible mesh" promise. | Neither: DoD 1 still promises it, and nothing measures it. | **Review is right, decisively.** A head-above-shoulders check fails on 100% of current fits (§4.1). |
| Observed vs defaulted | Output must say which values are defaulted. | Not adopted. | **Review is right.** Live in 14.2% of real frames: the median defaulted wrist is 67.5% SW from MediaPipe's estimate, and elbow params jump a median 0.94 rad on wrist-validity toggles. |
| Order of offset vs shape | Analytic init + small regularised fit, evaluated by root / shape / both ablations. | Step 9 (joint-centre offset) strictly before Step 10 (shape); regularised fit optional. | **Both partly right.** The plan's worry, betas absorbing correspondence bias, is supported: the ratios are unreachable by betas, and the scale deficit cannot separate shape from offset. But sequential attribution depends on order (§4.4), and the review's factorial omits the offset factor. Best: a factorial over {root, per-clip shoulder width/offset, arm lengths/betas} on the **direct** metric, with held-out halves and clip-level CIs. Both come after Step 8. |
| Confidence weights | Confidence-weighted fit (no source named). | `visibility` as weights (L918-919). | **The plan's choice is wrong** (ρ ≈ 0; visibility is also low-pass filtered). Use robust losses or a measured per-landmark noise model. |
| Held-out evaluation | Required for shared shape. | Absent from Step 10. | **Review is right.** The gain roughly halves held-out (18.9 → 10.1%). |
| Nose apex | "Heuristic; the head moves independently." | Nose apex for `global_orient`. | **Both are right, about different things.** The apex cannot affect the arm metric, and any apex (nose or hip centre) fixes the inversion. The review's warning applies to plausibility and to what `body_pose` means: a 20° neck bend leaks 18.8° into torso pitch. Keep the nose, document the pitch ambiguity, and make the fallback a convention flip or hip-centre apex, never identity. |
| Direction error in a common frame | Report it. | L600 still says "exact by construction". | **Review is right** (2.1–2.8° in the output). |

### 5.3 The "Blocked: wiring into the public classes" decision

**The outcome is sound.** Keeping the legacy image-space classes deprecated and unwired, and deferring integration, is exactly the review's contract: an image-space return needs its own projection contract.

**The stated reasons are uneven:**

1. *"Every caller passes image-space data."* True, but narrow. The callers are three demo scripts (`render_landmarks.py:124-128`, `test_phase2.py:486`, `test_phase1.py:319-350`), and the third is already broken. No pipeline or augmentation code calls either class.
2. *"Interchange is image space; no camera model exists."* Factually accurate. But a camera is buildable: a per-frame weak-perspective world → image fit has a median residual of 6.60% SW (p95 9.40%). Two caveats, both verified:
   - "Delta transfer" (augment in world space, map the *delta* to the image, add it to the original landmarks) is exact only at zero augmentation. On natural 3-frame motion its median relative error is 0.58, and the sign agreement is 76% over all frames (99% only for changes over 5°).
   - The 8.5% SW error and doubled wrist–hand gap of full replacement come from the world/image camera mapping, not the round trip: projecting the *original* world landmarks gives 9.4% SW and 11.2% SW.
3. *"The six legacy xfails describe a rejected design."* Partly wrong. One or two of them (`project()` from parameters alone) encode the adopted contract, as Finding 7 itself says.

**The real blockers are larger than a camera model:**
- All five `SMPLXAugmentation` classes are refused by landmark-mode chains.
- All four presets are landmark-only.
- All three `configs/*.yaml` fail to load.
- `AugmentedLandmarkDataset` calls `.astype` on parameter dicts.
- The joint-limit axes are wrong (§4.2).
- `LimbLengthScaling` adds `scale_range` samples to betas, and its config's per-bone semantics cannot be expressed with global betas.
- The body is inverted (§4.1).

**Proposal to unblock:**
- **(a) Close V1 formally.** Make `converter.fit_body`/`reconstruct_body` the V1 deliverable, export them, rewrite DoD 1-3 and Deliverables 1-2/4 to name them, and change the Blocked section to "decided".
- **(b) V2 integration design:**
  - single-pass extraction storing world *and* image landmarks from one Holistic instance per clip;
  - fit and augment in SMPL-X parameter space after Step 8 and the joint-limit fix;
  - map world deltas to the image through a per-frame weak-perspective camera fitted to that frame's own world/image pair;
  - leave face and hand landmarks untouched except for a rigid offset following the pose-wrist delta.
- **(c) Measure before committing:** the zero-augmentation identity (must be exact), and agreement for a known synthetic world-space perturbation.
- **(d) Keep a landmark-native kinematic augmentation as a baseline.** Rotating or scaling observed bones directly gives an exact +10.00° flex. SMPL-X's added value is anatomical joint frames and limits, a twist convention, a shape prior, MANO hands for V2, and a mesh for contact checks. That value is real, so "SMPL-X adds only a shape prior and a mesh" overstates the case against it.

### 5.4 Prior art and build-vs-reuse

- **No change for V1.** SGNify and SignAvatars start from video, not landmark arrays, so V1's landmark round trip is not replaced. SignAvatars annotations are themselves fitted estimates, not ground truth.
- **They do change Step 10 and V2 economics.**
  - mmhuman3d's SMPLify(-X) registrant with a 3-D keypoint loss and a MediaPipe mapping is an off-the-shelf comparator for the planned regularised fit.
  - If the cached clips are WLASL videos (inferred from their IDs, not verified; no WLASL JSON is in the repo), SignAvatars' per-video SMPL-X could validate the arm pose and shape.
  - SGNify's sign-specific hand priors are directly relevant to V2 hands.
- **Licensing:** SGNify and SignAvatars are non-commercial, mmhuman3d is Apache-2.0.
- **Repo state:** VPoser weights and `human_body_prior` are not installed, so a pose prior needs new downloads.

---

## 6. Recommended next steps (ordered, each with a falsifiable acceptance check)

1. **Finish Step 8 with a convention-aware root and a plausibility gate.**
   - Accept when all 14 Step 8 tests pass and the collar xfail stays xfail.
   - Add a synthetic fixture: SMPL-X landmarks with a known root, then C = diag(1, −1, −1) applied. It must recover `global_orient` within 1e-6 rad.
   - On all 409 real frames:
     - head joint (15) above the shoulder midpoint along the observed hip→shoulder axis in ≥99% of frames (now 0%);
     - median fitted-up error ≤ 30° (now 169°);
     - ≥99% of solved elbows on the anatomical flexion side (now 0%);
     - direct arm median ≤ 6.1% (a Step-8-like estimator gives 6.02%).
   - Nose-invalid fallback: with the nose masked on real frames, `global_orient` stays within 30° of the nose-based result. The minimal-rotation/identity fallback fails this by about 160°.
2. **Close the contract leaks in `converter.py`.**
   - (a) Carry betas: store them on `SMPLXSkeleton`, copy them into `BodyFit`, and have `reconstruct_body` assert they match. *Check:* a shaped-skeleton test, real SMPL-X forward vs `reconstruct_body` ≤ 1e-5 m (now 42–112 mm).
   - (b) Add an observed/defaulted mask to `BodyFit`; defaulted rows are NaN or flagged. *Check:* 0 of the 58 affected real frames return unflagged finite values.
   - (c) Use only landmarks whose chain was solved as Kabsch pairs. *Check:* elbow-invalid/wrist-valid synthetic case, error on valid landmarks ≤ 1e-9 m, or the fit raises.
   - (d) Treat exact-origin points consistently in IK and Kabsch. *Check:* left shoulder at (0,0,0) is exact.
3. **Fix extraction provenance.** Create one Holistic instance per clip, record the smoothing setting, and re-extract.
   - *Check:* forward-order and reverse-order extraction give bit-identical per-clip files.
   - Re-run the harness and update the headline (expected direct ≈ 6.45%).
4. **Give the doc one contract** (no new science; a falsifiable checklist):
   - DoD and Deliverables name `fit_body`/`reconstruct_body`;
   - L176, L686 and L746-749 say positions;
   - L503-507 and L600 are struck;
   - L331 and L573 are labelled as reference values;
   - Step numbering is fixed;
   - Finding 4 is per-axis;
   - the Blocked section is marked decided;
   - Section 4 says 6 landmarks;
   - L721-725 is struck.
   - *Check:* `grep -n "MediaPipeToSMPLX.fit\|recovered to machine precision\|33× the budget\|exact by construction"` returns only struck or historical lines.
5. **Re-derive the decomposition as a committed script, on the direct metric.**
   - Factors: optimal 3-DOF root; per-clip shoulder width/offset; per-clip arm lengths; adaptive IK as an extra arm.
   - Report clip-bootstrap CIs and held-out (contiguous-half) results.
   - *Check:* the script reproduces its own table from a clean checkout. Report the interaction and its clip-level CI before naming any term "dominant".
6. **Then Steps 9 and 10 as one factorial study.**
   - MAP betas with an explicit prior-strength policy (\|β\| ≤ ~3) vs a per-clip offset model vs both.
   - Robust loss, no visibility weights.
   - Use mmhuman3d SMPLify-X with a 3-D keypoint loss as an external comparator.
   - *Check:* report held-out reduction alongside in-sample; claim only what survives held-out. Current expectation: about 10% for plausible betas.
7. **Before any V2 integration, fix the joint-limit axes** (`joint_mapping.py:279-289`).
   - *Check:* on Step-8 fits, a zero-noise clamp moves wrists ≤ 1% SW in ≥ 95% of arms (now a median of 76% SW).
   - *Check:* a "+10° flex" through `JointAnglePerturbation` changes the elbow bend by +10° ± 1° median.
8. **Prototype V2 delta transfer** (§5.3 b–c).
   - *Check:* the zero-augmentation output equals the input exactly.
   - Report image-displacement error for known synthetic world perturbations before choosing between delta transfer and full replacement.
9. **Noise reference.** Replace the 0.18% "budget" with a same-statistic repeatability band from re-detections (unsmoothed vs smoothed, static, and image perturbations: 1.9–3.6% direct), labelled as a reference, not a gate.
   - *Check:* the script no longer prints "PROPOSED V1 ERROR BUDGET".

---

## 7. Reproduction

> **Note:** the `work/<folder>/` scripts cited throughout this report now live in [`scripts/review_checks/<folder>/`](../scripts/review_checks/index.md); read that index first. Run them from the repo root with `PYTHONPATH=src:scripts/review_checks/<folder>`. The commands below use the original scratchpad layout (`$SNAP` = repo root, `$W` = `scripts/review_checks`).
>
> **Status update (2026-10-05):** after this report was written, Step 8 was completed (suite 81 passed / 2 xfailed). Finding 4.1, the upside-down torso, no longer reproduces on current code: `main-check/upside.py` now reports a 6.4° median torso error, with the head above the shoulders in 173/173 frames. The other findings have not been re-checked against Step 8.

```sh
SCR=/private/tmp/claude-501/-Users-bemnetbeshah-Workspace-cv-research-KineticAugment--claude-worktrees-phase-one-research-review-18e5d2/837b3061-f39d-49fd-acbd-016ba3aa8fee/scratchpad
SNAP=$SCR/snap; W=$SCR/work
PY=/Users/bemnetbeshah/Workspace/cv-research/KineticAugment/.venv/bin/python
run() { (cd $SNAP && PYTHONPATH=$SNAP/src $PY "$@"); }

# Suite state (expect 66 passed / 13 failed [Step 8 red] / 2 xfailed) and headlines
(cd $SNAP && PYTHONPATH=$SNAP/src $PY -m pytest -q tests/)
run scripts/measure_round_trip.py                    # direct 6.55%, aligned 5.94%, synthetic 5.32e-15 m / 70.0 mm
run scripts/measure_noise_floor.py                   # 0.18% (A_xyz 0.175)

# Review claims
run $W/verify-fk-and-wiring/v1_fk.py                 # A001
run $W/verify-root-counterexample/v1_counterexample.py   # A002 counterexample
run $W/verify-root-counterexample/v2_real.py         # root share on real data, nose vs hip
run $W/verify-twist-identifiability/v1_harness_rot.py    # A003 rotation error 18.06 deg median
run $W/verify-twist-identifiability/v2_nullspace.py      # 2-D twist null space
run $W/verify-noise-floor/verify.py                  # A004 smoothing/static/frame-rate
run $W/verify-real-data-numbers/v_headline.py        # A005, F3
run $W/verify-fk-and-wiring/v2_legacy.py             # A007
(cd $W/verify-shape-identifiability && PYTHONPATH=$SNAP/src:. $PY v1_jacobian.py)   # A006 rank/dof
(cd $W/verify-shape-identifiability && PYTHONPATH=$SNAP/src:. $PY v2_nonunique.py)
(cd $W/verify-shape-identifiability && PYTHONPATH=$SNAP/src:. $PY v4_shapefit.py)   # held-out shape gain

# Missed issues
run $W/verify-missed-bugs/v_convention.py            # 4.1 inversion (re-run for this report: up err 169.44 deg, head-above 0%)
run $W/verify-direction-and-blocker/v3c_flip_fixed.py    # arm metric blind to the flip
run $W/verify-twist-identifiability/v5_limits.py     # 4.2 joint-limit axes
run $W/verify-direction-and-blocker/v2_step8_flex_clamp.py
run $W/verify-external-literature/vx_extract.py && run $W/verify-external-literature/vx_compare.py   # 4.3 (runs MediaPipe; minutes)
run $W/verify-real-data-numbers/v_oracle.py; run $W/verify-real-data-numbers/v_boot.py           # 4.4
run $W/verify-root-counterexample/v3_optroot.py
run $W/verify-fk-and-wiring/v3_contract.py           # 4.5 betas, Step 7 contract
run $W/verify-missed-bugs/v_orphan.py                # 4.6
run $W/verify-fk-and-wiring/v5_realdata_validity.py  # defaulted landmarks, 58/409 frames
run $W/verify-missed-bugs/v_noise.py                 # 4.7
run $W/verify-external-literature/vx_adaptive.py     # 4.8 adaptive IK
(cd $W/verify-shape-identifiability && PYTHONPATH=$SNAP/src:. $PY v5_misc.py)       # 4.10 ratio reachability
run $W/verify-external-literature/vx_visibility.py   # visibility vs jitter
run $W/verify-direction-and-blocker/v4_camera_transfer.py; run $W/verify-direction-and-blocker/v4b_camera_only_gap.py   # 5.3
run $W/verify-doc-consistency/v_similarity.py        # 7-number similarity residual
```

**Where the evidence lives.** Every cluster's tester scripts are in `$W/<cluster>/` and its verifier scripts in `$W/verify-<cluster>/`, each with captured `*.out` / `*_out.txt` next to it. A Step 8 prototype that passes all 14 red tests is at `$W/root-counterexample/proto_tests/step8_proto.py`. Treat it as a reference only: its fallback has the ~160° flip hazard described in §4.1.

**Limits on what could be verified:**
- The pre-revision doc and the reviewed harness are untracked in git, so "true at review time" judgments rest on surviving text and the doc's own quotations.
- Only the NEUTRAL SMPL-X model is present.
- Two things rest on inference: whether the clips are WLASL IDs, and the like-for-like value of the 39 mm published MAE.