# Start here: researching KineticAugment

Welcome. This guide is for an undergraduate who can read Python, train a basic neural network, and understands train/validation/test splits. You do not need prior experience with body models or biomechanics. Its goal is to help you explain this repository, follow a sample through its code, and decide what evidence a research experiment would need.

**Source snapshot:** `45472c6a1f99e24ac5763eeae7acb3be01ff4886` (latest source commit: **2026-05-08**). **Audit date:** **2026-10-01**. Source links and line ranges refer to that snapshot. External references explain concepts; they do not prove that this implementation works.

## Chapters and folder index

This table lists every file in this folder, including this index. All six chapters are available; source line references use the snapshot above.

| File | What you will learn | Availability |
| --- | --- | --- |
| [index.md](index.md) | Entry point, learning paths, and evidence conventions. | Available |
| [01-overview.md](01-overview.md) | Research motivation, motion representations, and augmentation limits. | Available |
| [02-code-map.md](02-code-map.md) | Real modules, tensor shapes, entry points, and a source-level sample trace. | Available |
| [03-math-and-geometry.md](03-math-and-geometry.md) | Geometry that changes motion: representations, canonicalization, time and constraints. | Available |
| [04-data-training-evaluation.md](04-data-training-evaluation.md) | From a video to a defensible experiment: datasets, training, metrics and comparisons. | Available |
| [05-project-status.md](05-project-status.md) | Bounded verification, gaps, and unproven claims. | Available |
| [06-learning-roadmap.md](06-learning-roadmap.md) | A learning sequence and possible research contributions. | Available |

## Choose your first route

- **First afternoon:** read chapter 1, follow chapter 2's deterministic scaling example, then inspect the status chapter before running an existing profile.
- **Understand the mathematics:** read chapter 1's representation warning, then chapters 3 and 2 together. Keep the difference between rotation vectors and bone vectors visible.
- **Design an experiment:** read chapters 4 and 5, return to the relevant source path in chapter 2, then use chapter 6 to scope a question and baseline.

## How to interpret evidence

| Label | Meaning |
| --- | --- |
| **Code present** | The cited implementation exists and was read at the snapshot. This is not a runtime test. |
| **Verified now** | A bounded command ran in the stated environment, with an observed result. Its scope matters. |
| **Historical claim** | README, docstring, roadmap, or earlier documentation says this; the guide does not adopt it as a verified result. |
| **Inference** | A conclusion from source or theory, with assumptions stated. |
| **Proposed work** | A possible future investigation or change, not completed work. |
| **Unknown intent** | The repository does not establish why a researcher made a choice. Ask the mentor instead of inventing a rationale. |

Architecture statements in chapters 1 and 2 are **code present** unless labeled otherwise. Their worked example is a source-level trace, not a claimed successful execution. Verification belongs in [project status](05-project-status.md). Passing a synthetic test would still not establish label preservation or recognition gains.

## Quick orientation

KineticAugment transforms already estimated human motion. Think of its pipeline as a recipe: choose transformations, apply them to a clip, optionally correct some constraints, and pass the clip to a recognizer. The recipe is separate from the camera's landmark estimator and the classifier's learned decision rule.

Start at [`src/kinetic_augment/__init__.py`](../../src/kinetic_augment/__init__.py) for public imports, [`pipeline/pipeline.py`](../../src/kinetic_augment/pipeline/pipeline.py) for orchestration, and [`augmentations/`](../../src/kinetic_augment/augmentations/index.md) for transformations. Checked-in YAML files and older examples do not all match that modern path; chapter 2 explains the mismatches.

The original [documentation index](../index.md) remains useful for terminology and historical design goals. Read its strong completeness or plausibility language as motivation until the status chapter supplies evidence.

## Find a term or next action

Use the [implementation glossary](06-learning-roadmap.md#a-small-glossary-tied-to-this-implementation) for unfamiliar vocabulary, the [troubleshooting table](06-learning-roadmap.md#where-to-look-when-confused) for an execution symptom, and the [proposed work priorities](05-project-status.md#proposed-work-in-dependency-order) to scope a contribution. The [recorded checks](05-project-status.md#commands-run-and-their-limits) distinguish observed runtime behavior from historical claims.
