# 09: Plausibility Metrics and Validation on Recorded Motion

**Status: planned (Phase 9 of the [roadmap](./06_Development_Roadmap.md)).** Nothing in this document is implemented yet.

## 1. Why

The [validation framework](./05_Validation_Framework.md) already measures quality, smoothness, diversity, and constraint satisfaction for the body alone. Once the engine knows about the floor and about objects ([07](./07_Object_Aware_Constraints.md)), two things are missing:

- metrics for the body's relation to the floor and to objects, defined so that they can be compared with published numbers;
- evidence that the engine's own judgments are right, measured against motion where the truth was recorded.

## 2. Metrics to add

All of them are aggregates of the violation report ([08](./08_Differentiable_Losses_and_Violation_Report.md)) or of the contact labels, and belong in `evaluation/metrics/`.

| Metric | What it measures |
|---|---|
| Ground penetration | How far the body goes below the floor |
| Floating | How far the body is above the floor when it should be supported |
| Foot skating | How much a foot that is on the floor slides |
| Object penetration depth | How far body points go inside an object, as mean and maximum over a sequence |
| Joint-limit violation rate | Share of frames with at least one joint outside its range (extends `JointLimitComplianceMetric` to sequences with hands) |
| Contact precision and recall | Agreement between the engine's contact labels and recorded contact |

The first three follow the measures used by PhysDiff (Yuan et al., "PhysDiff: Physics-Guided Human Motion Diffusion Model", ICCV 2023), so that its published numbers on HumanML3D can serve as a baseline. Implement them from the definitions in that paper, cite the section in the code, and reproduce one of its reported numbers before trusting the implementation.

## 3. Validation on recorded grasps

GRAB and ARCTIC record whole-body motion together with the objects being handled, registered to SMPL-X and MANO, and GRAB provides contact annotations. They are the reference for three measurements.

**3.1 Recorded motion is valid.** Run the full report on recorded sequences. Violations should be near zero. Where they are not, the cause is either an error in our constraint (a limit that is too tight, a wrong unit or frame) or noise in the recording. Each nonzero channel needs an explanation before the engine is used to judge other motion.

**3.2 Contact baseline.** Compare the engine's contact labels on recorded motion with the dataset's own contact annotations, as precision and recall. The two will not agree perfectly, because they use different thresholds and different meshes. This pair of numbers is the baseline for 3.3.

**3.3 Repair preserves contact.** Perturb recorded sequences with the augmentation operators, repair them with the engine, and measure contact precision and recall again. The target: both stay within 0.05 of the baseline of 3.2. This is the check that the engine's repair gives back the grasps that were really there, and it is the first go/no-go milestone of the KineticSynth project.

## 4. Tasks

**V1. Dataset loaders.** Loaders for GRAB and ARCTIC next to the existing `evaluation/datasets/wlasl.py`, returning body-model parameters, object identifiers and poses, and, for GRAB, contact annotations. Record each dataset's license and terms before downloading. The datasets are not redistributed with this repository.
*Done when* a loaded sequence, passed through the body model, reproduces the joints or vertices the dataset provides, with the error reported.

**V2. Floor and object metrics.** As in section 2.
*Done when* each metric has a test on a constructed sequence with a known answer, and the three floor metrics reproduce a published number.

**V3. Contact precision and recall.** Including the mapping between the dataset's contact annotation and our hand parts.
*Open:* at what granularity contact is compared (per vertex, per hand part, per hand), and per frame or per grasp. Decide before measuring, and report the choice with the numbers.

**V4. Measurements 3.1 and 3.2.** A script that writes one table per dataset.

**V5. Measurement 3.3.** Which perturbations, at what strengths, and how many sequences.
*Open:* the perturbation set. It should cover the operators a user would actually apply and, for KineticSynth, the kinds of error a generator makes (shifted hands, missed grasps, penetrations).

**V6. Report.** A short document in `docs/` with the tables, the thresholds chosen, and the answer to the go/no-go question, in the style of the existing findings documents.

---

### Next Up

➡️ **Next: [Glossary](./GLOSSARY.md)**
