# 6. A practical learning roadmap

[Guide index](index.md) · [Previous: project status](05-project-status.md)

You already know the core ML loop: examples enter a model, a loss drives weight updates, and held-out data tests generalization. The unfamiliar part here is what makes a transformed motion clip a valid example. Work outward from one array and one deterministic transformation, then connect geometry, batching and evidence. This learning sequence is a suggestion; the professor's near-term research goal has not been supplied.

**Source snapshot:** `45472c6a1f99e24ac5763eeae7acb3be01ff4886` (2026-05-08). **Audit date:** 2026-10-01. Suggested experiments below are not new measured results. Recorded outcomes and environment limits are in [chapter 5](05-project-status.md).

## Reading sequence and small deliverables

| Session | Read / do | A useful stopping point |
| --- | --- | --- |
| First orientation | [Overview](01-overview.md), then [code map](02-code-map.md). Identify estimator, augmenter and recognizer. | Explain where one `(T,543,3)` clip enters and leaves the modern Pipeline; name the representation hazard. |
| One sample and one transform | Inspect the cached array below; follow chapter 2's fixed scaling trace. Read [status matrix](05-project-status.md#evidence-matrix) before choosing a YAML/example. | Annotate shape, block boundaries and missing values; explain why doubling all coordinates is an algebraic operation rather than proof of anatomical realism. |
| Geometry | Read [chapter 3](03-math-and-geometry.md), with encoder/projector open beside it. | Explain displacement versus rotation, an orthogonal coordinate frame, temporal interpolation and a soft constraint. |
| Data to model | Read [chapter 4](04-data-training-evaluation.md), especially virtual versions, padding, valid lengths and training/evaluation mode. | Draw a batch with two unequal clip lengths; identify when augmentation runs and which frames the model should use. |
| Evidence and research scope | Read [chapter 5](05-project-status.md), then agree on a question with the mentor. | Write a one-page experiment specification: question, data/splits, baseline, transformation contract, metric, compute budget and success criterion. |

The older [architecture](../02_Framework_Architecture.md), [augmentation catalogue](../03_Augmentation_Catalogue.md), [profiles](../04_Task_Specific_Profiles.md), [validation framework](../05_Validation_Framework.md) and [roadmap](../06_Development_Roadmap.md) explain design goals and historical vocabulary. Compare their claims with live source and chapter 5; they are not an alternative executable specification. The [original glossary](../GLOSSARY.md) is a useful broader reference.

## Safe setup options

First inspect an existing approved environment, if one is available: `python --version` and `python -m pip show numpy scipy PyYAML torch smplx pytest`. Avoid changing an environment used for someone else's experiments merely to follow this guide.

For a new smoke environment, Python 3.11 is the version family used in the audit. Create the environment outside the repo, activate it, then return to the repo root:

```sh
python3.11 -m venv ../kinetic-guide-env
source ../kinetic-guide-env/bin/activate
python -m pip install numpy scipy pyyaml torch smplx pytest
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
```

This is a minimal source-import setup, not a package-install check. If `python3.11` is unavailable but `uv` is already installed with Python 3.11 available, `uv venv --python 3.11 ../kinetic-guide-env` is an alternative to the first line. No global dependency installation is needed. Record the resulting versions with `python -m pip freeze` outside the repo; [chapter 5](05-project-status.md#commands-run-and-their-limits) lists the audit versions for comparison. Reusing those exact public package versions may help reproduce the audit, but it does not certify the whole project.

The conventional `python -m pip install -e '.[dev]'` is a separate packaging path. It was not executed in this audit. On the default Python 3.14 host, the pinned MediaPipe 0.10.21 cannot resolve a compatible wheel; adding an interpreter upper bound or modernizing dependencies would be future maintenance work. `[full]` also does not include the pose-prior extra. Model-library installation supplies code, not licensed SMPL-X/MANO weights. Obtain assets only through the project's approved licensing process if a later task requires them; no weights are needed for the first array exercises.

Run commands from the repo root so relative sample paths resolve. In this guide, `python` means the activated environment's interpreter; the default host has `python3` and may not have `python`. Confirm `python -c 'import sys; print(sys.executable)'` before debugging a package import. Use chapter 5's timeout wrapper for bounded checks instead of launching every script at once. Phase1 can extract/write media outputs, so reading cached arrays is the first route.

## First sample to inspect

Use [01986_landmarks.npy](../../data/test_output/01986_landmarks.npy), backed by [01986.mp4](../../data/sign_videos/01986.mp4) and [its analysis image](../../data/test_output/01986_analysis.png). Read the assets; do not regenerate or overwrite them during orientation. The clip's task label and annotation provenance should be confirmed with the mentor rather than inferred from its numeric ID.

```python
import numpy as np
x = np.load("data/test_output/01986_landmarks.npy")
print(x.shape, x.dtype, np.isfinite(x).all())
blocks = {"pose": (0, 33), "face": (33, 501),
          "left_hand": (501, 522), "right_hand": (522, 543)}
for name, (start, end) in blocks.items():
    block = x[:, start:end]
    print(name, block.shape, "all-zero frames",
          int(np.all(block == 0, axis=(1, 2)).sum()))
```

The **recorded** whole-array observation is `(83,543,3)`, `float32`, all finite. The block-zero counts above are an inspection exercise, not reported audit results. Layout comes from [data_formats.py](../../src/kinetic_augment/utils/data_formats.py), lines 8–21: 33 pose + 468 face + 21 left hand + 21 right hand = 543 points, or 1629 flattened coordinates per frame.

Finite coordinates can still be missing, misdetected or unsuitable for a geometry formula. Zero-filled missing blocks need a validity mask; a real point can also happen to lie at the origin. The extractor saves estimator coordinates rather than world meters. Inspecting `x/y/z` ranges cannot establish a physical unit conversion.

## A small glossary tied to this implementation

| Term | Meaning here / next place to look |
| --- | --- |
| Landmark | An estimated point with three coordinates. It is an observation, not necessarily a joint center or a measured physical distance. [Data layout](../../src/kinetic_augment/utils/data_formats.py). |
| Sequence / frame / batch | A sequence is a clip of `T` frames; one frame is `(543,3)`; a padded batch commonly has `(B,T_max,543,3)` plus valid lengths. [Batch walkthrough](04-data-training-evaluation.md#3-batch-variable-length-sequences-correctly). |
| Extrinsic augmentation | Changes global spatial placement, such as rotating/scaling coordinates. [extrinsic.py](../../src/kinetic_augment/augmentations/extrinsic.py). |
| Intrinsic augmentation | Intended to change articulated body parameters, such as local joint rotations; requires the expected parameter semantics. [intrinsic.py](../../src/kinetic_augment/augmentations/intrinsic.py). |
| Temporal augmentation | Changes the sampling or timing of a clip. A changed parameter does not ensure the current implementation changes the output. [temporal.py](../../src/kinetic_augment/augmentations/temporal.py). |
| Axis-angle / bone vector | Axis-angle encodes a rotation with direction and angle in radians; a bone vector is child-minus-parent displacement in coordinate units. Equal shape does not make them interchangeable. [Representation explanation](03-math-and-geometry.md#1-three-numbers-three-different-meanings). |
| SMPL-X / betas | A parametric human body model / shape coefficients for that model. The actual wrapper and the current landmark encoder/projector are distinct paths. [Body-model explanation](01-overview.md#positions-and-body-parameters-describe-different-things). |
| Forward kinematics | Computes joint positions by propagating local rotations along an articulated hierarchy. Current additive bone reconstruction does not provide standard rotation propagation. [Geometry](03-math-and-geometry.md#1-three-numbers-three-different-meanings). |
| Inverse fitting | Estimates model parameters from observations through an objective/optimizer; current `fit` is a direct encoder, while older source had an optimizer. [Fitting versus training](03-math-and-geometry.md#7-inverse-fitting-versus-classifier-training). |
| Canonicalization | Expresses points in a chosen body-centered coordinate frame; modern Pipeline flags do not invoke it. [Coordinate-frame walkthrough](03-math-and-geometry.md#2-canonicalization-choosing-a-body-centered-coordinate-frame). |
| Constraint / proxy | A rule for selected validity properties / a simplified geometry for collision checks. Neither proves all anatomical or semantic validity. [Constraints](03-math-and-geometry.md#5-constraints-approximations-with-assumptions). |
| Padding / valid length / mask | Added frames for equal batch shapes / number of real frames / which entries are usable. Frame lengths do not substitute for per-joint missing-detection masks. [Dataset code](../../src/kinetic_augment/pipeline/dataset.py). |
| Dataset mode / model mode | Dataset train mode controls augmentation; `model.train()` or `model.eval()` controls neural-network behavior. See [chapter 4](04-data-training-evaluation.md#2-dataset-indexing-and-augmentation-placement). |
| Quality metric / recognition metric | A statistic about motion / performance on predicting held-out labels. Proxy quality scores are not demonstrated recognition improvements. [Metric table](04-data-training-evaluation.md#6-read-metric-definitions-before-their-scores). |
| Ablation / baseline / leakage | Remove one mechanism to isolate its contribution / reference experiment / unintended information crossing a split. [Study design](04-data-training-evaluation.md#7-design-a-fair-study-after-the-integration-gaps-are-addressed). |
| Seed / replay | Initial RNG state / reproducing actual samples and results. Resetting only chain selection does not reset all augmentation or training randomness. [Status evidence](05-project-status.md#evidence-matrix). |

## Bounded learning experiments

Each experiment should fit in a short CPU session, use copies or temporary arrays, and record interpreter/dependency versions. Predictions below are based on source and existing audit evidence; run them to understand a mechanism, not to generate a recognition claim.

1. **Deterministic scale.** Follow chapter 2's fixed scale factor 2 on its two-frame artificial input. Expect the nonzero points to double and shape to remain unchanged. Then ask whether scaling normalized estimator coordinates has the intended camera/body interpretation. This isolates implementation from scientific meaning.
2. **Identity baseline.** Process ten cached frames with `Pipeline.from_preset('none')`. Expect exact array equality at this snapshot. Then try `conservative`; record shape, finite values and max delta. A changed result does not establish preserved sign meaning.
3. **Encoding contract.** Use chapter 5's self-contained probe. Predict tiny round-trip error but a nonzero projected change after zero-noise angle perturbation. Explain which clamping/representation assumption caused it. No model weights are required because the current projector does not execute its wrapper.
4. **Unequal lengths.** Create 2- and 3-frame arrays and use default DataLoader collation as in chapter 5. Expect a stacking error. Read `LandmarkSequenceDataset.__getitem__` to sketch padding plus lengths; distinguish padded frames from zero-valued missing joints. No training is needed.
5. **Seed scope.** Make a probability-one `GlobalScaling` plan with range `[0.8,1.2]`. Process the same copied input twice with `pipeline.reset_seed(42)` between calls. The audit found different outputs. Then explicitly reset `np.random.seed(10)` before each call as a separate controlled experiment; predict repeatable NumPy sampling for that path, while noting this says nothing about Torch/worker RNGs.
6. **Metric interpretation.** Build all-zero and straight-limb fixtures from the joint triplets `(11,13,15)`, `(12,14,16)`, `(23,25,27)`, `(24,26,28)`. The audited compliance metric gave 1.0 and 0.0 respectively. Explain missing-data treatment and the difference between the interior angle and flexion convention before interpreting the score as anatomical validity.

Suggested notebook/report columns: input contract, intended property, predicted result, observed result, code path, limitations. Save new outputs outside tracked `data/`. Large dataset experiments and model downloads belong to a separately agreed research task.

## Check your understanding

| Question | Answer / hint |
| --- | --- |
| If `(83,543,3)` becomes `(83,1629)`, what changed? | Only representation shape: 543×3 coordinate features per frame. No temporal aggregation or model training happened. |
| Does a near-zero encoder/projector error prove correct SMPL-X rotations? | No. Subtraction/addition using original anchors can reconstruct displacements accurately without estimating rotations or anatomy. |
| Why can `stddev=0` still change encoded data? | The operation also clamps values as angle components; zero noise does not disable those deterministic changes. |
| Does setting `enforce_constraints=True` validate landmark presets? | No at this snapshot; the modern landmark branch bypasses the constraint engine. |
| Why does a valid `length=2` matter in a padded three-frame clip? | The recurrent model should treat the third frame as padding rather than another observation. A joint mask handles a different problem. |
| Is a higher VelocityMetric score equivalent to faster motion? | No. Its returned score measures relative velocity dispersion; inspect the formula in chapter 4. |
| What is wrong with augmenting before splitting? | Original and transformed versions can cross train/test boundaries, exposing nearly the same observation to both. Split original identities first. |
| If a benchmark result has quality scores and `error=None`, did training run? | Check `train_history` and `test_metrics`; the current runner can suppress loading failure and still produce synthetic quality scores. |

## Where to look when confused

| Symptom / question | First file or check |
| --- | --- |
| `ModuleNotFoundError` for NumPy/Torch or project package | Confirm `sys.executable`, installed packages and repo-root `PYTHONPATH=src`. Dependency presence in another interpreter does not help this one. |
| MediaPipe will not install on current Python | Read [dependency limits](05-project-status.md#commands-run-and-their-limits) and [pyproject.toml](../../pyproject.toml); compare Python ABI with the pinned release before attempting unrelated fixes. |
| YAML says unknown operation / incompatible mode | [config.py](../../src/kinetic_augment/pipeline/config.py) validation, [chain.py](../../src/kinetic_augment/pipeline/chain.py) step construction and registry names; begin with a working preset. |
| Legacy/example fails on import or lowercase augmentation name | [Entry-point caveats](02-code-map.md#legacy-alternatives-and-misleading-entry-points); modern `Pipeline` and legacy `AugmentationPipeline` are different APIs. |
| Array shape is right but motion looks wrong | Read [representation contracts](02-code-map.md#representation-contracts), joint layout, missing blocks, units and intrinsic operation assumptions. |
| Constraints appear to do nothing | Check Pipeline mode and landmark bypass before tuning thresholds; inspect [pipeline.py](../../src/kinetic_augment/pipeline/pipeline.py), lines 122–158. |
| Same seed gives different augmentation | Inspect chain selection versus global NumPy RNG; record all seeds and worker/model randomness. |
| DataLoader raises unequal tensor size | Inspect actual post-augmentation lengths, padding/collation and runner dataset choice; [chapter 4 batching](04-data-training-evaluation.md#3-batch-variable-length-sequences-correctly). |
| Recognition training silently disappears | Inspect dataset root/annotation/cached-file counts, `train_history`, `test_metrics`, and runner's caught loading exceptions. |
| A test skips or references `SimpleFitter` | Separate missing assets from removed API calls. [Status](05-project-status.md#commands-run-and-their-limits) records both; downloading weights alone does not reconcile callers. |
| A metric seems surprisingly good | Check missing-data behavior, aggregation, coordinate units and synthetic versus real input before trusting the score. |

## Questions to resolve with the professor or prior researcher

- What is the immediate research question and deliverable: understanding the system, repairing a route, studying a specific augmentation, or evaluating recognition? Which task/data domain comes first?
- Which representation should be authoritative, and should the May bone encoding be retained, renamed or replaced? What was its intended scope? Repository history records changes but does not establish the rationale.
- Which data, annotations, landmark estimator/version, clip bounds, fps and signer metadata are approved and available? What split and leakage policy should the experiment follow?
- Which transformations are believed to preserve labels in that domain? Who can review handshape, facial grammatical content, intended contact and temporal meaning?
- What baseline/results/checkpoints exist outside the repository, and can their exact configs, seeds and data manifests be recovered? Which roadmap claims have supporting artifacts?
- What compute budget and success criterion are realistic? Are model licenses/dependencies already approved? Which failed integration paths should be repaired before running a study?

A useful initial contribution is a small, reproducible explanation of one path and its assumptions. Choose the eventual implementation or experimental milestone using the [prioritized work table](05-project-status.md#proposed-work-in-dependency-order) and the answers above.
