# V1.3 audit: faces for landmark-based sign recognition

**Date:** 2026-10-11. **Code audited:** `2205f16` on `feat/face-v1.3`.
**Intended outcome:** generate augmented landmark/geometry sequences that improve sign
recognition on real, unseen signers while preserving the sign's meaning.
The user confirmed landmark/geometry recognition as the target, rather than avatar generation.

**Recommendation: continue the face feature stream, but revise the dependency on the phase-A
geometry gate.** The branch contains useful extraction, fitting and diagnostic components.
It does not yet establish that its metric hand-to-face contacts are reliable training features
or that its augmentations preserve sign labels. Preserve observed facial motion now; evaluate
3-D contact and SMPL-X-based augmentation separately. The best representation for this particular
recognizer remains an empirical question, not something a round-trip test can settle.

Reviewed inputs: the supplied design and findings, their repository versions
([design](V1_3_Face_Design.md), [findings](V1_3_Findings.md)), the
[implementation plan](V1_3_Implementation_Plan.md), current face/body/hand code, tests, cached
observations and features, and primary research linked below. The current branch also includes
image proximity and optional depth correction added after the supplied findings. Those additions
are included in this audit. No production fitting code was changed.

Follow-up integration coverage after Tasks 1–5 is recorded in
[V1.3 Face Integration Results](V1_3_Face_Integration_Results.md); this audit remains the historical
review of recognition-data and geometry assumptions.

## 1. What was tested

The [probe script](../scripts/audit_face_training.py) and
[machine-readable evidence](V1_3_Audit_Evidence.json) distinguish cached measurements from fresh
execution. Cache file hashes, implementation hashes and dependency versions are recorded.

| Check | Result and scope |
| --- | --- |
| Full pytest suite with this branch on `PYTHONPATH` | 252 passed, 1 expected failure, 3 failures and 3 setup errors in 85.53 s. All six unsuccessful checks were MediaPipe extraction checks failing to create macOS OpenGL contexts in the sandbox. |
| Those six extraction checks outside the sandbox | 6 passed in 17.83 s. Together the runs account for all 259 cases: 258 passed and 1 expected failure. This was not a single clean full-suite run. |
| Expected failure | Existing collar-motion round trip: V1 fixes collar joints at zero. Not a new face failure. |
| Fresh full fitting from cached observations | 45 distinct frames across 01986, 01991, 09849 and 63665, each with depth correction off and on: 90 frame evaluations. Real SMPL-X/MANO/FLAME assets were available. |
| Cached default features | All 20 evaluation clips / 1,236 frames, plus all 5 pilot clips / 410 frames inspected. |
| Cached depth-corrected features | 5 pilot clips / 410 frames and the available 6 evaluation clips / 338 frames. These six are a subset, not a replacement 20-clip gate. |
| Targeted counterexamples | Polygon overlap, constant head-orientation bias, and hand/head scale and assumed camera sensitivity. |
| Visual inspection | Region render and source frames 14, 17 and 21 of 01986. These are qualitative checks, not independently annotated ground truth. |

An initial plain `.venv/bin/python -m pytest` imported the parent checkout and failed collection
with eight missing-module errors. `.venv` is a symlink to the parent environment. Explicit
`PYTHONPATH=src` selects this branch without changing that shared environment.

The cached jaw correlations reproduce the eight values in the findings, including the failures
on 63665 and 69364. I did not rerun the full 1,236-frame gate, train a classifier, or benchmark
external reconstruction models. Recognition improvement therefore remains unmeasured.

## 2. Findings that affect training

### A. Depth correction is an experimental estimate, not a contact-label generator

In the five pilot caches, only **131/524 corrected hand-frames (25%)** satisfy the existing
3 cm wrist-reach tolerance. Median wrist error is **5.45 cm**, p95 **15.54 cm**. All **nine**
hand-frames flagged as contact occur on corrected wrists that were **not reached**.

The six available evaluation caches are more encouraging about contacts: 29 of 30 contact
hand-frames have reached wrists. But only 85/338 corrected hand-frames meet the reach criterion
overall, and there is no independently labelled contact reference. Reaching an estimated target
does not prove the target is correct.

This behavior also reproduces on current code. On 01986 frames 14–21, correction yields eight
contact hand-frames, zero reached wrists out of 16 corrections, and a median wrist residual of
9.55 cm. The three inspected source frames show the hands visibly separated from the face.
On sampled 63665 frames, the one predicted contact also belongs to an unreached wrist. The
pipeline's docstring already acknowledges this failure class; the audit confirms it persists.

The mechanism matters: [`_depth_pass`](../src/kinetic_augment/body_model/face_pipeline.py)
re-fits directions with fixed bone lengths and a fixed first-pass translation. It cannot move
the shoulder/elbow chain freely to satisfy the requested wrist. It still exports the resulting
mesh's `contact` values. `wrist_reached` is separately exported, so a training reader can easily
mistake these diagnostics for accepted measurements.

There is also a fundamental calibration sensitivity. In a noise-free synthetic scene with the
head at 1 m and wrist at 0.9 m, changing only the inferred hand size by 5% changes recovered
wrist depth by **4.5 cm**. A 5% head-size change moves it **5 cm**. Using 45° instead of 63°
vertical FOV shifts it by **4.79 cm**. These are sensitivity probes, not estimates of actual
detector error, but each exceeds the 2 cm contact threshold. `fit_shape` estimates five body
proportions; it does not independently measure the signer's head size.

**Action:** retain raw and corrected hypotheses with provenance and quality masks. Exclude
unreached corrections from contact features; keep contact unknown when unvalidated. Even
`wrist_reached=True` is insufficient for hard contact supervision, particularly because its
3 cm tolerance already exceeds the contact threshold. Keep correction off for production
augmentation until measured against labelled positive and negative events.

### B. Invalid faces still produce numeric geometry

Across the 20 evaluation caches, **63 of 95 invalid face frames** contain numeric distances and
region labels; 64 unreliable frames contain geometry. A fresh run on five frames from the
entirely undetected-face clip 09849 reproduces this in three frames.

[`run_clip`](../src/kinetic_augment/body_model/face_pipeline.py) assembles the default/rest head
and measures against it even when `HeadFit.valid=False`. The exported validity flag is honest,
but no dedicated geometry-valid flag prevents use of the invented head placement. A missing
hand similarly has `contact=False` alongside NaN distances; a boolean alone conflates unknown
with a negative observation.

**Action:** make the training adapter consume explicit per-stream/per-hand masks. A trustworthy
geometry mask needs body validity, hand availability/quality, reliable head placement, and any
correction-quality checks. Preserve unknown contact separately from observed non-contact. Do
not discard an entire useful hand/body clip simply because its face is missing.

### C. The head gate verifies consistency, with important blind spots

The findings correctly disclose that both head estimates come from the same landmarks.
There is a stronger blind spot: multiplying every estimated orientation on the template side
by a fixed **45°** rotation gives a median gate error of approximately **1.3e-14°**. The
clip-relative computation cancels that constant offset exactly. This is a regression test for
relative motion/conventions, not an absolute alignment or independent accuracy test.

The later `head_camera_rotation_deg` diagnostic also cannot establish the pose-world axis
assumption. The head was already oriented using the face observations that `project_head`
fits back to; a small residual rotation is largely enforced by this construction.

Jaw pitch and `jawOpen` share a similar evidence limitation: Google's blendshape model receives
the facial landmarks, so agreement is not independent jaw ground truth. Preserve the richer lip
trajectory and coefficients; treat the one-axis jaw fit as a geometric approximation.
[MediaPipe's documented model pipeline](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).

**Action:** retain these useful consistency checks, but name them accordingly. For metric
geometry claims, add a small calibrated multi-view/contact dataset or independent reference.
Manually checked 2-D reprojection can test image alignment; it cannot establish absolute depth.

### D. The near/far split has both selection bias and a reproducible overlap bug

The findings already identify selection bias: near-face is only defined when detection succeeds,
so its invalid rate is zero by construction. In addition,
[`hand_near_face`](../src/kinetic_augment/body_model/face_pipeline.py) only checks vertex
containment. A thin hand polygon crossing the entire face rectangle can intersect its edges
while containing no rectangle corner and having no vertex inside. The supplied probe returns
`False` for exactly that overlapping geometry. Its incidence in the real clips is unmeasured.

**Action:** use a complete polygon/rectangle intersection check. Evaluate detection failures in
an independent near/occlusion annotation stratum, or explicitly use an unknown stratum. Do not
interpret overlap as physical contact or automatic proof of occlusion.

### E. The current training path does not consume the new face representation

Phase C's `face_stream.py` and corresponding tests are absent. Raw `_face.npz` archives already
preserve 478 points, but `_face_features.npz` has no `face_stream_local`. The
[WLASL reader](../src/kinetic_augment/evaluation/datasets/wlasl.py) loads `<video_id>.npy`, and
the baseline classifier defaults to the legacy 543×3 input. There is no demonstrated path from
these face feature files to a trained recognizer. Extraction completion is not training integration.

Feature exports also omit timestamps, source hashes, schema version, correction mode/FOV and
code/model hashes; their metadata mostly contains vocabulary names. The observation archive has
better metadata, but it does not carry through. Existing feature caches visibly have different
sets of keys. Timestamp equality in `run_clip` does not prevent pairing different videos with
the same frame schedule.

**Action:** define a versioned adapter and feature contract, preserve source identity/timestamps,
and validate cross-stream source hashes and dimensions. Never blindly reshape 478-point face
data into the legacy layout. Make the next experiment train through that adapter.

### F. Useful region/distance diagnostics should remain approximate

The render is broadly interpretable, but the Eye and Ear regions are generous; the Ear area
includes adjacent anatomy. This is not a validated annotation ontology. Hard nearest-vertex
labels introduce jumps around boundaries, especially far from the face. Preserve continuous
relative location features and use coarse labels only near relevant regions, with uncertainty.

Seven query points per hand can miss knuckle, finger-side and back-of-hand contact. The open
surface and nearest-normal sign rule are not a general inside/outside test, and forcing boundary
distances positive hides some penetrations. These limitations are acceptable for diagnostics;
they are inadequate as the sole label-preservation gate for large augmentations. Evaluate denser
hand surfaces and appropriate collision geometry if contact-preserving augmentation becomes necessary.

## 3. What the research suggests for this goal

| Approach | Evidence and implication |
| --- | --- |
| Separate facial keypoint groups | **CoSign (ICCV 2023)** uses 77 keypoints: 9 body, 21 per hand, 8 mouth and 18 other face points, with group-specific modeling. Its supplemental ablation reduces test WER from 27.8 for body+hands to 22.8 with mouth and 21.9 with additional face signals. This is direct evidence for skeletal facial features, on continuous recognition datasets; it is not a predicted WLASL accuracy gain. [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.pdf), [ablation](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Jiao_CoSign_Exploring_Co-occurrence_ICCV_2023_supplemental.pdf). |
| Multiple keypoint streams | **MSKA**, published in Pattern Recognition in 2025, explicitly separates left hand, right hand, face and whole-body streams. It is a relevant architecture reference if simple fusion helps but underuses facial cues. It does not establish the need for metrically correct facial meshes. [Paper](https://arxiv.org/abs/2405.05672), [authors' implementation](https://github.com/sutwangyan/MSKA). |
| Compact face subset | **TwoStream-SLR** uses 26 selected facial points within its 79-keypoint stream. This supports testing a compact subset rather than assuming all 478 must be classifier inputs. Its final system also uses RGB, so its headline results are not a landmarks-only benchmark. [Paper](https://papers.nips.cc/paper_files/paper/2022/file/6cd3ac24cdb789beeaa9f7145670fcae-Paper-Conference.pdf). |
| Evidence cited in the design | The 2025 facial-regions paper uses RGB crops and 12 DGS glosses. Body+mouth improves top-1 from 80.53 to 88.24 for CSN (**7.71 points**), but from 84.03 to 86.42 for MViT (**2.39 points**). The latter top-1 difference is not statistically significant by the paper's analysis. This motivates testing faces, not promising a universal 7–8 point gain. [Paper](https://arxiv.org/html/2507.20884v1). |
| Better reconstruction if actually needed | **SGNify** adds sign-specific linguistic constraints and self-contact modeling to monocular reconstruction, with mocap/perceptual evaluation. Borrow its joint fitting/validation ideas when developing a contact-sensitive geometry path; changing trackers alone will not certify contact. [Project](https://sgnify.is.tue.mpg.de/index.html), [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.pdf). |
| Explicit expression reconstruction | **SMIRK** reconstructs expressive FLAME geometry and is a reasonable later comparator if measured landmark failures justify it. It adds dependencies, model assets and another estimator; no result inspected here demonstrates improved sign recognition from replacing the present stream with SMIRK. [Authors' code](https://github.com/georgeretsi/smirk). |

MediaPipe face z is relative, scaled like x under weak perspective. Its metric conversion uses a
virtual camera and a canonical model. Hand world landmarks use a hand-centred metric origin.
Combining these estimates is a modeling choice, not a calibrated shared scene supplied by the
detector. This explains why exact synthetic recovery and plausible units do not by themselves
validate real contact. [Face geometry documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md),
[hand output documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md).

## 4. Recommended face representation

Keep an immutable observation archive with all 478 face points, the ordered 52 blendshape scores,
head matrix, timestamps, source/model identifiers and masks. Keeping the archive dense allows
later comparisons without repeating extraction. Do not treat those scores as calibrated
linguistic labels, and do not translate them directly into SMPL-X expression coefficients.

Use a **separate facial temporal branch** beside body/hands. Start with a compact, explicitly
indexed mouth/brow/eye/nose subset and a simple fusion layer. A 26-point anatomical counterpart
to published baselines is one starting condition; compare it against denser lips/brows/eyes.
Their landmark IDs are not automatically interchangeable with MediaPipe IDs.

The proposed inputs are:

| Stream | Coordinates / information | Important constraint |
| --- | --- | --- |
| Body and hands | Existing observed baseline and fitted geometry as separately identified variants | Retain the unaugmented observed baseline; measure whether fitting itself helps or hurts. |
| Facial deformation | Face-centred, scale-normalized facial points; image x/y first, relative z as a separate ablation | Normalize from stable rigid anchors. Preserve expression; do not subtract each clip's mean mouth/brow expression. |
| Facial coefficients | Ordered 52 scores with masks, tested as an addition and as a compact alternative | Landmark-derived, hence correlated with the point stream. Retain only if useful. |
| Head motion | Head orientation relative to torso, plus temporal changes; use a continuous rotation encoding | Removing rigid head pose from local facial coordinates must not erase nods, tilts and shakes. |
| Hand/face relation | Image-space offsets/distances to mouth, nose, chin and forehead, normalized by a stable head scale | Label as 2-D proximity. Retain neck/throat anchors, which the current image surface deliberately excludes. |
| Experimental metric relation | Original/corrected signed distances, reach error and validity | Keep out of the initial production feature set or expose with explicit uncertainty; no hard ground-truth contact claim. |

Phase C can supply head-local coordinates for re-posing, but they remain model-scaled pseudo-3-D
observations. Keep the original image observations and the head pose separately. Image-only face
features need not wait for reliable pose-world translation. Conversely, synthetic viewpoint or
body-shape augmentation involving metric hand-to-face relationships still needs its geometry gate.

At the tensor boundary, replace missing values with a fixed fill value **and supply masks**.
Mask unavailable features before projection/fusion; `0 * NaN` does not remove NaNs. Use masked
losses/statistics. Facial absence must be distinguishable from a neutral expression. Mask temporal
derivatives over missing spans, and report any interpolation rather than passing it off as observed.

## 5. Process and acceptance gates

The existing process has useful strengths: explicit assumptions, real assets, synthetic recovery,
held-out clip inspection, and a findings document that admits unmeasured cases. The missing link
is a task-level criterion connecting geometric fidelity to recognition performance.

Use the already inspected 20 clips as a **development set** from now on. Repeatedly choosing
thresholds and corrections from them makes them unsuitable as the final untouched test. Split
by source/signing identity before producing augmented variants; keep all derivatives of a source
in its split. Where signer identity is unavailable, document that limitation and at least group
by original source/video and near duplicates. Fit normalization and tune augmentation only on
training/validation data. Match the official WLASL benchmark where applicable and add a genuinely
signer-disjoint evaluation rather than assuming the supplied split guarantees it.

Run a staged ablation with the same source clips, splits, model budget and several seeds:

1. Observed body+hands; then the fitted body+hands counterpart. This measures the conversion cost.
2. Add compact facial x/y trajectories, with missingness handling.
3. Add relative facial z, blendshapes and head motion one at a time; then their useful combination.
4. Add 2-D hand-to-face relations. Test mouth-only versus mouth+brow/eye subsets, and compact
   versus dense facial points. A shuffled-face negative control should destroy any real facial benefit.
5. Only after this baseline works, compare label-preserving augmentation against matched
   unaugmented training. Test original versus corrected 3-D relations separately after validation.

Report macro accuracy/F1, top-1/top-5, class/support counts, variability across seeds and paired
uncertainty over independent source clips/signers. Inspect contact/location-confusable signs,
mouth-dependent cases, occlusion, head turns, small faces and complete face dropout separately.
Choose the representation on validation; evaluate the frozen choice once on the untouched test.
Do not claim a best representation until those comparisons exist.

Create a small, purposefully labelled geometry set containing definite contact, near-but-not-
touching, ordinary signing space and uncertain/occluded examples. Label event timing, hand and
target region without inferring the event from the gloss name. Assess contact precision/recall,
region accuracy, reprojection, coverage and temporal stability. Qualified signer review should
assess whether augmented minimal pairs remain distinguishable. Multi-view capture is needed for
strong metric accuracy claims. Thresholds must be chosen for these tasks, not inherited solely
from an unrelated 2 cm proximity convention.

For augmentation, preserve facial deformation and its timing relative to the hands. Apply the
same temporal mapping to every stream. Do not jitter jaw, brows or head trajectories independently,
or assume arbitrary static head offsets preserve a sign. View changes must transform all geometry
consistently; transformed blendshape left/right channels require explicit handling under mirroring.
Rigidly transporting stored facial points does not predict newly exposed or occluded observations.
Reject or model those cases rather than presenting hallucinated visibility as measured data.
Temporal smoothing should be an ablation with onset/peak timing checks; indiscriminate smoothing
can remove the brief motion the recognizer needs.

## 6. Assumption ledger and decision

Audit IDs are local to this report and do not overwrite the existing O/P/F ledger.

| ID | Claim and dependency | Evidence / confidence / status | Next verification and failure signal |
| --- | --- | --- | --- |
| FA01 | Facial features improve this landmark recognizer; depends on FA02/FA03 | Primary papers support plausibility; no local training result. **Likely, open.** | Matched ablations on untouched sources. Failure: no repeatable recognition improvement. |
| FA02 | Extracted facial observations are available and synchronized | Archive checks and six fresh extraction tests. **Confirmed for tested clips, verified.** | Add source-hash/dimension contract. Failure: mismatched source or timing. |
| FA03 | A training adapter actually consumes face values and masks; depends on FA02 | Current WLASL path reads legacy `.npy`; phase C absent. **Risky, broken for the intended new-stream integration.** | End-to-end loader/model test and missing-modality case. Failure: unused features or non-finite predictions. |
| FA04 | Head agreement establishes independent absolute accuracy | Shared input and 45° counterexample. **Risky, broken.** | Independent reference plus absolute convention test. Failure: low agreement error despite wrong orientation. |
| FA05 | Corrected contact is accurate enough for training constraints; depends on FA04/FA06 | Unreached contacts, scale sensitivity, no labelled reference. **Risky, open.** | Contact/non-contact event set and calibration. Failure: contacts on visibly separated hands, poor precision or unstable labels. |
| FA06 | Exported numeric geometry denotes an observed reliable head | 63 invalid cached frames carry geometry; fresh reproduction. **Risky, broken.** | Explicit accepted/unknown masks. Failure: invalid-head distances accepted by training. |
| FA07 | Near/far splitting measures difficult-frame coverage | Detector-conditioned split plus crossing-polygon counterexample. **Risky, broken.** | Independent strata and complete intersection. Failure: occluded detection losses excluded from the near denominator. |
| FA08 | Augmented variants preserve labels and improve generalization; depends on FA01/FA05 where contact-sensitive | No task-level augmentation trial. **Unknown, open.** | Paired real-test ablation and signer review. Failure: gains only on synthetic data or degraded confusable signs. |

Dependency map: recognition benefit `FA01 <- FA02 + FA03`; safe contact-sensitive augmentation
`FA08 <- FA01 + FA05`, with `FA05 <- FA04 + FA06` and evaluation coverage checked by `FA07`.

**Gate decision: revise-plan** for treating phase A as certification of contact-preserving
training data. **Go with risks** for an observation-preserving face stream and recognition
adapter, with FA01 tested through ablations. Fix masking/provenance and the overlap bug before
using geometry-derived labels. Retain the head/jaw/mesh work as a useful auxiliary path.

## 7. Reproduction

From the branch root:

```sh
rtk proxy env PYTHONPATH=src .venv/bin/python -m pytest -q --tb=long
rtk proxy env PYTHONPATH=src .venv/bin/python -m pytest tests/test_extract_face_landmarks.py tests/test_extract_hand_landmarks.py tests/test_extract_world_landmarks.py -q --tb=long
rtk proxy env PYTHONPATH=src .venv/bin/python scripts/audit_face_training.py --output /tmp/face-audit.json --fresh
```

The second command needed macOS graphics access outside the sandbox in this session. The probe
reads local licensed assets and existing observations; it never rewrites the caches. Omitting
`--fresh` runs the synthetic and cache-inspection parts only. The geometry probes reproduce
limitations intentionally; their presence is not a claim that the product is repaired.

Deliverables are this report, its JSON evidence and the probe script. The missing root
`CLAUDE.md` relative symlink was restored according to the project instructions. No model weights,
source videos, fitting implementation, training model, or historical findings were replaced.
