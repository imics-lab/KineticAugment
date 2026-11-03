# 5. What is implemented, and what evidence do we have?

[Guide index](index.md) · [Previous: data and evaluation](04-data-training-evaluation.md) · [Next: learning roadmap](06-learning-roadmap.md)

**Audit date: 2026-10-01. Source snapshot: `45472c6a1f99e24ac5763eeae7acb3be01ff4886`, latest source commit 2026-05-08.** This is a research handoff, not a certification. There is useful infrastructure here: registered transformations, composable chains, four runnable landmark presets, cached sample clips, body-model interfaces, constraint modules, classifiers and evaluation code. The audit also found integration gaps that matter before interpreting anatomical or recognition claims.

Use three questions separately: does code exist, did a bounded check exercise its behavior, and has a scientific experiment established its value? A successful array-shape check answers only part of the second question. Research intent and the professor's immediate assignment remain **unknown**.

## Evidence matrix

**Verified now** means reproduced in the environment below; **code present** and **inference** denote inspected source without execution of that particular behavior. Confidence describes the observation, not effectiveness. File links and line ranges use the source snapshot above.

| Area | Status and concrete evidence | Confidence and limit |
| --- | --- | --- |
| Landmark transformations and presets | **Implemented / verified now:** `none`, `conservative`, `moderate`, `aggressive` processed ten cached frames into finite `(10,543,3)` arrays. `none` was exactly unchanged; other presets changed values. [presets.py](../../src/kinetic_augment/pipeline/presets.py), lines 44–209. | High; executable augmentation smoke, no label or anatomy validation. |
| Actual SMPL-X interface | **Code present / untested here:** `SMPLXWrapper` invokes `smplx.create` and the model's forward function. [smplx_wrapper.py](../../src/kinetic_augment/body_model/smplx_wrapper.py), lines 120–131, 143–220. | High for implementation; no weights to execute a pretrained forward pass. |
| Landmark conversion | **Implemented / verified now:** `MediaPipeToSMPLX.fit` encodes child-minus-parent vectors; projection adds them to reference anchors. Two cached frames round-tripped with max absolute difference `3.725290298461914e-09`. [mp_to_smplx.py](../../src/kinetic_augment/body_model/mp_to_smplx.py), lines 88–143; [smplx_to_mp.py](../../src/kinetic_augment/body_model/smplx_to_mp.py), lines 138–172. | High; algebraic consistency with reference landmarks, not optimized SMPL-X fitting. See [geometry](03-math-and-geometry.md#1-three-numbers-three-different-meanings). |
| Conversion plus intrinsic operations | **Partial integration / verified now:** zero-noise `JointAnglePerturbation` changed projected landmarks by a max `0.5608986616134644`; clamping treats encoded components as angles. Changing `betas`/`global_orient` left projection identical; `LimbLengthScaling` changed betas but not reconstructed points. [intrinsic.py](../../src/kinetic_augment/augmentations/intrinsic.py), lines 98–145, 454–491; projector lines 84–123. | High; raw displacements cannot be assumed to satisfy rotation-vector contracts. Finger ordering also differs between encoder and intrinsic weights (**source-only**), detailed in chapter 3. |
| Anatomical enforcement in landmark Pipeline | **Missing wiring / verified now:** `enforce_constraints=True` with an empty landmark plan returned the sample unchanged and never called a spy constraint engine, with or without a supplied wrapper. [pipeline.py](../../src/kinetic_augment/pipeline/pipeline.py), lines 122–158. | High; the four presets use landmark mode. Their outputs have no implemented anatomical enforcement through this route. |
| Canonicalization / output-space flags | **Partial / verified now:** empty-plan `canonicalize=True, output_space='canonical'` produced bit-identical input. The modern `process` does not call normalization utilities. [config.py](../../src/kinetic_augment/pipeline/config.py), lines 128–130, 178–179; Pipeline lines 98–129. | High; utilities exist separately. Degenerate axes and zero-sentinel ambiguity are **source deductions**, not executed here. |
| Checked-in task YAMLs | **Historical examples / verified failing:** all three profiles fail `Pipeline.from_yaml` registry validation. Removing unknown names in memory still fails on SMPL-X-only operations in default landmark mode; probability zero does not bypass construction. [config.py](../../src/kinetic_augment/pipeline/config.py), lines 119, 247–259; [chain.py](../../src/kinetic_augment/pipeline/chain.py), lines 104–143. | High; migration needs schema and representation decisions, not just renaming. See [entry paths](02-code-map.md#legacy-alternatives-and-misleading-entry-points). |
| Legacy / command entry points | **Partial / verified failing:** legacy constructor raises `AttributeError` for `global_rotation`; example `--help` fails importing `AugmentationPipeline` from the modern package; declared console entry target `kinetic_augment.cli` is absent and import raises `ModuleNotFoundError`. [pipeline_legacy.py](../../src/kinetic_augment/pipeline_legacy.py), lines 28–29, 57–66; [example](../../examples/run_augmentation.py), lines 11–12; [pyproject.toml](../../pyproject.toml), lines 83–84. | High; installed console command itself was not installed/invoked. |
| Joint / temporal constraints | **Implemented, partly verified:** invalid second batch row reported zero joint violations, while enforcement clamped it to about 2.62. [joint_limits.py](../../src/kinetic_augment/constraints/joint_limits.py), lines 86–95, 159–171. **Source deduction:** default soft velocity blending can leave speed above the configured cap. [velocity_limits.py](../../src/kinetic_augment/constraints/velocity_limits.py), lines 216–239. | High; angle representation, units, fps, batch and sequence handling need explicit contracts. |
| Native collision checks | **Implemented / untested:** PyBullet optional groups skipped. **Source/API observations:** fixed `0.1` capsule cylinder height, first-row updates and contact-normal convention mismatch; see [collision explanation](03-math-and-geometry.md#6-collision-proxies-and-heuristic-resolution). | High for static observations; no native pose/contact reproduction or mesh-accuracy study. |
| Training and variable lengths | **Implemented / integration failing now:** ordinary collation of 2- and 3-frame samples raises `stack expects each tensor to be equal size`. A miniature annotation fixture through `BenchmarkRunner.run` fails before its first optimizer step. [runner.py](../../src/kinetic_augment/evaluation/benchmark/runner.py), lines 279–292, 336–354. | High; padding-aware dataset and lengths-aware recurrent models exist but are not wired into this path. [Chapter 4](04-data-training-evaluation.md#3-batch-variable-length-sequences-correctly). |
| Benchmark success reporting | **Partial / verified now:** missing dataset produced `error=None`, `train_history=None`, `test_metrics=None`, yet four quality metrics. Training-load failure is caught separately. Runner lines 266–297. | High; explicitly inspect history and test metrics before saying training ran. |
| Quality metrics | **Implemented, partly verified:** quality branch uses 20 synthetic 30-frame random walks and disables enforcement (runner lines 186–245). Straight elbow/knee fixture scored 0.0 compliance; all-zero missing landmarks scored 1.0. [constraint metric](../../src/kinetic_augment/evaluation/metrics/constraint.py), lines 273–341. | High for these probes; missing-data masks and angle conventions need correction/calibration. Wrong hand offsets and jerk-unit issue are **source-only** findings in [metric definitions](04-data-training-evaluation.md#6-read-metric-definitions-before-their-scores). |
| Replay / seed control | **Partial / verified now:** resetting Pipeline seed did not replay probability-one random scaling; max difference `0.7208622694015503`. Chain controls its own step-selection RNG, while transformations use global NumPy. [chain.py](../../src/kinetic_augment/pipeline/chain.py), lines 82, 219–228; [extrinsic.py](../../src/kinetic_augment/augmentations/extrinsic.py), lines 77–85, 124. | High; model initialization, Torch and worker randomness also need research controls. |
| Semantic / environmental checks | **Planned or missing wiring:** parser ignores top-level semantic/environment directives; no handshape or grammatical-marker validator found in complete runtime inventory. [config.py](../../src/kinetic_augment/pipeline/config.py), lines 167–186; [extended profile](../../configs/slr_profile_extended.yaml), lines 142–152. | High for absent wiring; effect on labels is an open empirical question. |
| Recognition improvement | **Historical goal / unverified:** no real held-out recognition result artifact in tracked inventory. Phase5 report uses explicit mock accuracies 0.65/0.72, [test_phase5.py](../../scripts/test_phase5.py), lines 620–638. | No measured improvement established. Mock numbers are report-format fixtures. |

One additional **verified** call-contract hazard: a landmark operation accepted in SMPL-X mode is silently skipped on dict data; intrinsic operations are skipped on a list of parameter dicts ([chain.py](../../src/kinetic_augment/pipeline/chain.py), lines 145–149, 173–179; [base.py](../../src/kinetic_augment/augmentations/base.py), lines 114–122, 168–187). Count actual applied operations, not just configured steps.

## What history and assets establish

The [development roadmap](../06_Development_Roadmap.md), lines 5–15, 439–455, declares all phases complete and production-ready, and cites `0.5974 RMSE`. These are **historical claims**. The checked-in [summary JSON](../../data/test_output/test_summary.json) has `conversion.simple_fitter=null`, `smplx_available=true` and `round_trip_error=null`; it does not corroborate that RMSE or today's model availability.

History gives a concrete reason to read old fitting explanations carefully:

| Source history | Observed change, not inferred motive |
| --- | --- |
| `1939ca6`, 2026-01-13 | Added conversion files with an optimizer-based approach. |
| `f9c6d7b`, 2026-05-08 | Replaced the prior Adam/L2/regularization/`fit_sequence`/`SimpleFitter` implementation with current bone encoding; commit title mentions discrete changes and video renderer. |
| `3238058`, 2026-05-08; merge `45472c6` | Updated phase scripts, then merged. Some stale callers still reference removed APIs. |

Use `git log -- src/kinetic_augment/body_model/mp_to_smplx.py` and `git show f9c6d7b^:src/kinetic_augment/body_model/mp_to_smplx.py` to inspect this distinction. Historical source is not a tested alternative implementation in this audit.

Full recursive inspection of both the original and isolated checkout, including ignored contents, found five tracked sample videos, five cached finite float32 landmark arrays, five analysis PNGs and historical summary/config artifacts. Array frame counts for IDs `01986/01987/01988/01991/01992` are `83/75/144/49/59`; all have 543 points and 3 coordinates. No `.npz`, `.pkl`, `.pt` or `.pth` weights were present anywhere under either `models/`, and no WLASL annotation JSON was present under either `data/`. The MANO folder's `SMPL-X__FLAME_vertex_ids.npy` is an auxiliary map, not model weights. This says nothing about downloads elsewhere on the user's computer.

Coordinates come from estimator `x/y/z`, not world-landmark meters. Missing landmark blocks are saved as zeros without confidence masks ([wlasl.py](../../src/kinetic_augment/evaluation/datasets/wlasl.py), lines 264–284). Available clips are useful for reading arrays and smoke checks; their existence does not establish dataset coverage, clean splits or signer-disjoint evaluation.

## Commands run and their limits

The audit used CPU-only source imports through `PYTHONPATH=src`; it did **not** test editable installation or packaging. Default host Python was **3.14.6**, without NumPy/Torch/SciPy/pytest/YAML/smplx/PyBullet. An isolated Python **3.11.15** environment installed only public smoke dependencies:

| Dependency | Recorded version |
| --- | --- |
| NumPy / SciPy / PyYAML | 2.4.6 / 1.17.1 / 6.0.3 |
| Torch / smplx / pytest | 2.14.1 / 0.1.28 / 9.1.1 |
| PyBullet / MediaPipe / human-body-prior | Not installed in audit environment |

These versions describe one observation, not a lockfile or compatibility promise. `[dev]` pins MediaPipe `0.10.21`; a dry-run resolver in Python 3.14.4 failed because that release had no `cp314` wheel (reported tags `cp39/cp310/cp311/cp312`). Python 3.11 was successful for the listed smoke packages. `[full]` excludes the pose-prior extra, and no VPoser runtime integration was found. Packaging version is `0.2.0`, runtime `__version__` is `0.4.0` ([metadata](../../pyproject.toml), line 7; [public module](../../src/kinetic_augment/__init__.py), line 38).

The seven original smoke checks used the audit Python, repo cwd, and environment `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Each had a 60-second limit, except Phase5 at 120 seconds; none timed out. The final traceback row below was a separate direct rerun recorded in the command manifest, with only `PYTHONPATH` and `PYTHONDONTWRITEBYTECODE` explicitly set.

| Exact Python command arguments | Exit / observed output | What that supports |
| --- | --- | --- |
| `python -m pytest -p no:cacheprovider` | 1; **23 passed, 2 failed, 10 skipped** | Two imports of removed `SimpleFitter` fail; asset-dependent paths skipped. |
| `python scripts/test_phase2.py` | 0; **23/23** individual checks | Shape/change/scalar smoke; its main does not map printed failures to exit failure. |
| `python scripts/test_phase3.py` | 0; body-part groups **5/5**, engine groups **4/4**; physics groups skipped | No native collision proof. Skips and caught model-load failures can count as successful groups. |
| `python scripts/test_phase4.py` | 0; **36/36** | YAML tests construct temporary modern profiles; legacy check is import/docstring only. |
| `python scripts/test_phase5.py` | 0; **10/10** groups | Synthetic metrics/models and existing 50-random-sample, two-epoch test using the same dataset for training and validation; mock report formatting. |
| `python examples/run_augmentation.py --help` | 1; `ImportError` | Failure before argument parsing. |
| `python -c 'import kinetic_augment.cli'` | 1; `ModuleNotFoundError` | Declared entry module is absent. |
| `python -m pytest -p no:cacheprovider tests/test_body_model.py::TestSimpleFitter --tb=long` | 1; same two failed imports | Narrow rerun for full tracebacks, not an independent clean suite. |

Only the existing Phase1 **conversion function** was isolated and called on one cached frame: it caught a missing-model assertion and returned `smplx_available=False, round_trip_error=None`. The entire Phase1 video/extraction script was not run. Stale constructor/projector signatures were separately reproduced as `TypeError`; model skips can conceal these failures. No pretrained forward pass, full extraction, PyBullet native execution, CUDA/MPS/distributed validation, real WLASL training, licensed downloads or long training occurred.

## Reproduce a small observation safely

Use the [isolated setup](06-learning-roadmap.md#safe-setup-options) and run from the repository root. This self-contained probe reads one cached clip and writes no project files. It reproduces the conversion-contract, constraint-bypass and variable-length checks; caught errors are observations, not a green test suite.

```python
import numpy as np
from torch.utils.data import DataLoader
from kinetic_augment import Pipeline, PipelineConfig
from kinetic_augment.pipeline import AugmentedLandmarkDataset
from kinetic_augment.body_model.mp_to_smplx import MediaPipeToSMPLX
from kinetic_augment.body_model.smplx_to_mp import SMPLXToMediaPipe
from kinetic_augment.augmentations import JointAnglePerturbation

x = np.load("data/test_output/01986_landmarks.npy")[:2]
params = MediaPipeToSMPLX().fit(x)
projector = SMPLXToMediaPipe(object())  # unused by current projector
baseline = projector.project(params, x)
changed = projector.project(
    JointAnglePerturbation({"stddev": 0.0}).apply(params), x
)
print("round trip", np.max(np.abs(baseline - x)))
print("zero-noise intrinsic delta", np.max(np.abs(changed - baseline)))

class Spy:
    def enforce(self, *args, **kwargs):
        raise AssertionError("engine called")
    enforce_sequence = enforce

pipeline = Pipeline(PipelineConfig(plan=[]))
pipeline.constraint_engine = Spy()
print("landmark bypass", np.array_equal(
    pipeline.process(x, enforce_constraints=True), x
))

samples = [np.zeros((t, 543, 3), dtype=np.float32) for t in (2, 3)]
dataset = AugmentedLandmarkDataset(samples, labels=[0, 1], mode="val")
try:
    next(iter(DataLoader(dataset, batch_size=2)))
except RuntimeError as error:
    print("collation error", error)
```

Save the snippet outside the repository and invoke it with `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python <probe-file>`. Expected observations at this snapshot are tiny reconstruction error, a substantial zero-noise delta, `landmark bypass True`, and unequal-size stacking error. Exact floating-point values may differ with dependency versions. The meaning of these observations is more important than matching the final decimal.

For a bounded suite rerun, this portable wrapper uses the environment's interpreter and records failures; choose just one command at a time:

```python
import os, subprocess, sys
command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
env = dict(os.environ, PYTHONPATH="src", PYTHONDONTWRITEBYTECODE="1",
           CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
result = subprocess.run(command, env=env, timeout=60)
print("exit", result.returncode)
```

A `TimeoutExpired` means the selected bound was exceeded, not that the code passed or failed a correctness assertion. To check profiles independently, call `Pipeline.from_yaml(path)` for each of the three files in [configs](../../configs/index.md); catch and print each exception. To verify local asset absence, recursively list `models/` and `data/`, including ignored files, rather than relying only on `git ls-files`.

## Proposed work in dependency order

This list is **proposed work**, not an undocumented assignment. Coordinate priorities with the professor before selecting a substantial implementation project.

| Priority / purpose | Bounded completion criterion |
| --- | --- |
| **A. Establish representation and observation contracts.** Choose true model rotations versus explicit bone vectors; document units, joint/finger layout, missing masks, fps and root/shape behavior. | One documented contract per public path; deterministic fixtures show which fields change projection, finger-slot correspondence and intended articulated propagation. Invalid cross-representation calls fail visibly. |
| **A. Restore one supported onboarding route.** Align a minimal profile, imports and stale tests with the selected contract. | A clean isolated setup runs a named preset/profile and bounded meaningful checks; optional asset skips are explicit; entry points either work or are clearly withdrawn. |
| **B. Make training measurable and replayable.** Wire lengths/padding, expose missing data and preserve histories/checkpoints/configs. | A miniature variable-length fixture completes a bounded train/val step with correct lengths and visible metrics; reruns under controlled RNG agree within a stated tolerance. No silent training omission. |
| **B. Validate specific constraint/metric behavior.** Check batch rows, angle conventions, velocity limits, contact-normal direction, proxy geometry and missing detections. | Hand-computable fixtures report expected violations/units; native collision fixtures test separation direction when dependencies/assets are available. Each metric documents its interpretation and failure cases. |
| **C. Validate the research objective.** Assemble approved data/splits, establish baseline, control compute and repeat seeds; review semantics with domain expertise. | Versioned sample/signer/split manifest; held-out comparisons with uncertainty and equal training budget; qualitative label checks and ablations that actually change the intended mechanism. |

Coordinates, fitting quality, temporal continuity, anatomical propagation, collision approximations and semantics interact: fixing a score alone cannot validate the input representation. A useful next milestone is one traceable path from a documented sample to a reproducible result, with explicit limits, before expanding features.
