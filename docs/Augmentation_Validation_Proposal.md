# Augmentation validation proposal

Status: proposed design, reviewed by Claude Opus 5.5. No new validators or tests have
been implemented as part of this proposal.

## Outcome and scope

Give each augmentation reproducible evidence about implementation correctness, observation
coverage, classifier-format compatibility, and eventual recognition benefit. These are separate
questions; no single quality score or automatic check certifies sign meaning or usefulness.

The authoritative input requirement is [README: Classifier input](../README.md#classifier-input):
raw MediaPipe time series, normalized image coordinates with relative depth, and absolute image
units. There is no shoulder centering or shoulder-width normalization. Do not substitute the
existing normalized face-feature adapter for this downstream contract.

This proposal supersedes the acceptance guarantees in [05_Validation_Framework.md](05_Validation_Framework.md).
It is a design for the first implementation slice, not a claim that the classifier integration exists.

## Existing evidence

- `tests/test_tier1.py`, `tests/test_week4_scene.py`, `tests/test_week4_clip.py`, and
  `tests/test_week4_profile.py` already cover many identity, composition, sampling, camera,
  mirroring, and timing properties. Extend coverage rather than duplicating these tests.
- Focused baseline command: `.venv/bin/python -m pytest tests/test_tier1.py tests/test_week4_report.py tests/test_week4_scene.py -q`.
  Result during this design pass: **46 passed**, no skips, in 13.16 seconds. This is not a full-suite result.
- Direct probe of `augmentations.report._nanmax(np.array([np.nan, np.nan]))` returned **0.0**.
  An unavailable measurement can therefore look like zero error.
- `report._in_frame_pct` pools arms, hands, and face points. Hundreds of face points can obscure
  one lost hand. Frame coverage and finite-data coverage must be reported separately by part.
- `ClipOutputs` supplies normalized image x/y only; pose rows outside 11–16 are unpopulated.
  It does not carry source image dimensions or generated detector-confidence channels.

## Framework choice

Use the existing **pytest + NumPy** stack. Use parameterized cases and fixed random seeds to
cover transforms, frame shapes, missingness, and boundaries. Test against small independently
calculated examples, not another call to the same implementation. No new dependencies are needed.

Keep fast tests independent of licensed model assets by exercising `SceneTransform.apply_points`,
`WeakCamera.project`, and synthetic `ClipOutputs`. Keep existing SMPL-X/MANO integration tests as
a separate layer; absent assets mean NOT EVALUATED, not a successful integration check.

Use JSON for machine-readable reports and a short Markdown summary. Encode unavailable numeric
values as `null` with a reason and observation count, rather than nonstandard JSON NaN/Infinity.
Preserve existing report consumers through an additive or explicitly versioned result format.

## First implementation slice

### 1. Honest measurements and per-part coverage

Each check should report its name, status, value, units, numerator/denominator where relevant,
reason, and source clip/frame references. Retain augmentation name, actual sampled parameters,
seed, profile identity, and code revision for reproduction.

- All-missing measurements return NOT EVALUATED. Missing camera means frame checks are unavailable.
- A point declared valid but containing unexpected nonfinite values is an engineering failure.
  Documented missing slots are excluded explicitly, never silently counted as valid.
- Report left hand, right hand, face, and arms separately: expected points, finite points,
  in-frame points, and newly out-of-frame points relative to the comparable baseline.
- Include worst-frame counts and longest affected duration, so averages cannot hide a lost
  hand during a crucial part of a sign. Use timestamps where available.
- Out-of-frame coordinates normally produce REVIEW, not an automatic schema failure.
  A promised in-frame constraint failing on previously in-frame points is a separate hard failure.
- Finite/in-frame does not establish visibility. Self-occlusion and detector confidence remain
  NOT EVALUATED until an observation model supplies evidence. Never manufacture confidence.

Negative tests: all values missing; one hand entirely absent with the face intact; valid-mask
and coordinate disagreement; absent camera; an already out-of-frame source; a single-frame
clip; a clip with no comparable baseline points. Verify both the value and its coverage/status.

### 2. Missing mathematical tests

| Operation or boundary | Expected relationship | Deliberately faulty case the check should catch |
| --- | --- | --- |
| Uniform scene scale | Coherent 3-D distances and same-frame projected x/y differences multiply by the sampled factor | Scale applied to body but not hands |
| Scene translation | Pairwise differences remain unchanged; projected shift equals the requested image offset | Pixel values used as normalized offsets |
| Image-unit conversion | x uses width, y uses height; normalized/pixel round trip agrees | Swapped dimensions on a non-square frame |
| Mirror | Reflection about the projected scene pivot, with correct side relabeling | Treating every mirror as x → 1−x or forgetting a side swap |
| View rotation | Preserves 3-D distances and joint angles | Rotating only one stream; do not require 2-D distance preservation |
| Missingness and mutation | Missing observations stay missing; source arrays remain unchanged | Filling an absent hand or modifying the original sample |
| Composition | Final output follows the documented order; coverage is measured after the full chain | Scale then shift pushes a hand outside while reporting only the shift stage |

Use explicit boundary examples plus bounded seeded sweeps. Verify the validator rejects injected
faults; a test that only accepts good examples does not establish detection ability. Reuse existing
identity, mirror-twice, determinism, and composition tests where they already cover the contract.

Thresholds must follow the operation and coordinate space. A universal 5% bone-length rule would
reject intended scaling. Raw frame-to-frame jitter changes under retiming; compare aligned physical
times and expected timing effects rather than treating every increase as a defect. Distance between
hands and face is a review signal; contact is part of many signs and is not inherently a collision.

### 3. Classifier compatibility inventory

Initially report each field as PRODUCED, NOT PRODUCED, or UNKNOWN. Do not reject every current
geometry sample simply because the eventual exporter is incomplete.

- Normalized image x/y: produced for the supported fitted points.
- Relative image z: not produced by the current image-output arrays; coherent world z is not a substitute.
- Complete pose coverage: not produced; only six arm landmarks are populated.
- Absolute x/y: convertible when original width/height are supplied; not self-contained in `ClipOutputs`.
- Confidence/visibility: no generated per-point detector output. Fitting-valid masks have different meaning.
- Exact consumer requirements: still need landmark subsets/order, missing-value encoding, face count,
  absolute-depth convention, and timestamp versus fixed-frame-rate handling.

These unknowns do not prevent the mathematical and reporting work. They do prevent declaring an
export fully compatible with the intended classifier. A sample input file or loader specification
is the cheapest decisive evidence for the remaining details.

## Later stages

After the first slice, run reproducible corpus sweeps with mild, current, and stronger parameters.
Measure original observations, reconstructed identity output, and augmented output separately to
avoid blaming reconstruction error on augmentation. Select random examples across signs/signers
alongside worst cases for human review. Calibrate warning thresholds against real training data;
do not derive universal acceptance bounds from a small pilot or tune on the final test set.

Recognition experiments follow actual export integration: separate source clips and signers before
augmentation; keep every derived sample with its source; compare equal training budgets and paired
seeds; evaluate single augmentations and their combinations on real validation clips. Reserve the
final test set for the selected recipe. Semantic validity needs independent sign-language evidence;
recognition benefit remains NOT MEASURED until these experiments run.

## Assumption ledger and gate

| ID | Claim / dependency | Evidence and confidence | Verification / failure signal |
| --- | --- | --- | --- |
| A001 | Raw scale and position reach the intended classifier | Confirmed requirement: README and user instructions | Inspect actual loader when connecting; unexpected centering would violate this requirement |
| A002 | Existing tools can support the first slice | Confirmed: focused 46-test run, NumPy/pytest installed | Run asset-free tests separately; hidden model downloads or skips reveal dependency problems |
| A003 | Missing-data reporting needs correction | Confirmed: `_nanmax` probe and report source | All-missing negative test must yield unavailable evidence, never zero error |
| A004 | Current outputs already satisfy the full consumer contract | Broken: 2-D image arrays and partial pose output | Inventory fields against loader; missing depth/dimensions must remain explicit |
| A005 | Exact downstream schema is sufficiently specified | Unknown beyond README; depends on A001 | Inspect sample/loader for z, masks, ordering and timing before exporter acceptance tests |
| A006 | A useful realism threshold is known for every augmentation | Unknown | Calibrate on representative real data; excessive false alerts signal bad thresholds |
| A007 | Passing automated geometry checks proves meaning or recognition gain | Unsupported and excluded | Human semantic review and controlled recognition experiments supply separate evidence |

Dependency map: A002 → first-slice tests; A003 → trustworthy metrics; A001 + resolved A004/A005 →
classifier export validation; trustworthy metrics + A006 → corpus alerts; export validation +
independent experiments → recognition evidence. A007 is not used as a premise.

Gate: **go-with-risks for the first slice only**. Exact schema A005 and calibration A006 remain
open. First verify missingness negatives and image-space relationships, then inventory schema,
then expand to assets/corpus/training. The highest risk is reporting geometry correctness as
classifier readiness or recognition benefit.

## Claude review incorporated

Claude Opus 5.5 reviewed the initial proposal in Cloudroom thread `thr_4dexbjyymt` and recommended
revision: prioritize report correctness, reuse existing tests, make the first compatibility check
an inventory, and defer broad corpus/training infrastructure. This document incorporates those
changes. The revised document itself has not had a separate review.

Reference: [pytest parameterization](https://docs.pytest.org/en/stable/how-to/parametrize.html).
