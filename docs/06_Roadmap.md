# 06: Roadmap

**KineticAugment is at an early stage of development.** It is a research codebase, not a production-ready library: the API may change, parts of the constraint system are untested, and much of what has been validated so far comes from one application (sign language). This document lists what exists today and the planned next phases. It replaces the original plan, which is kept for history in [legacy/06_Development_Roadmap.md](./legacy/06_Development_Roadmap.md) and overstates how complete the framework is.

## Where things stand

What exists and has been exercised:

*   **Body model round trip:** MediaPipe landmarks to SMPL-X and back, including hands (MANO) and face. See the [V1 Findings](./V1_Findings.md), [V1.2 Findings](./V1_2_Findings.md), [V1.3 Findings](./V1_3_Findings.md) and [Week 2 Findings](./Week_2_Findings.md).
*   **Tier 1 augmentations**, grounded in evidence about what preserves the meaning of a sign. See [Augmentation Grounding](./Augmentation_Grounding.md).
*   **`Pipeline`**, the library's augmentation API, configured from YAML profiles. The registered operations are listed in the [README](../README.md#available-augmentations).
*   **A constraint system** (joint limits, velocity limits, self-collision) and evaluation code. These exist but are not yet covered by unit tests, and constraint enforcement does nothing in `mode='landmark'`.

## Planned phases

The next four phases harden what exists, extend the framework from the body in isolation to the body in the world (objects, floor, scene), and make its constraints usable beyond the augmentation pipeline. They also remove what is left of the framework's origin as a sign language tool from its core. The companion project [KineticSynth](https://github.com/imics-lab/KineticSynth), which generates human motion, depends on Phases 7 to 9.

| Phase | Description | Status | Specification |
|-------|-------------|--------|---------------|
| Phase 6 | Hardening and generalization | In progress | This document |
| Phase 7 | Object-aware constraints | Planned | [07](./07_Object_Aware_Constraints.md) |
| Phase 8 | Differentiable losses and the violation report | Planned | [08](./08_Differentiable_Losses_and_Violation_Report.md) |
| Phase 9 | Plausibility metrics and validation on recorded grasps | Planned | [09](./09_Plausibility_Metrics_and_Validation.md) |

### Phase 6: Hardening and generalization

Other code is about to depend on the constraint system, so it has to be tested automatically and free of assumptions that hold for one task only.

#### 6.1 Tests under pytest
- [x] One command runs the test suite: `pytest tests/`
- [ ] Port the checks in the old phase scripts (now smoke scripts in `scripts/legacy/phaseN_smoke.py`) that are still relevant into `tests/`
- [ ] Add unit tests for each constraint on its own: joint limits, velocity limits, self-collision, for both checking and enforcing
- [ ] Mark every test that needs the body-model files or PyBullet so that it is skipped, not failed, where those are missing (done for the SMPL-X tests)

#### 6.2 Continuous integration
- [ ] A workflow that runs the tests on every pull request, on CPU

#### 6.3 Explicit frame rate
- [ ] The velocity code assumes a time step of 1/30 s by default. Make the frame rate an explicit argument wherever a sequence is processed, and test that the same motion at two frame rates gives the same violations

#### 6.4 Generality audit
- [ ] Find defaults, names, example paths, and documentation in the core package that assume sign language, and move task-specific values into task profiles. A known example: the higher velocity limit for hands used for signing
- [ ] Add at least one worked example that is not sign language (the `har_running_profile.yaml` profile is a starting point)

#### 6.5 Packaging
- [ ] Replace the placeholder author email in `pyproject.toml`
- [x] Replace the placeholder clone address in the README
- [ ] Decide how the optional dependencies of Phases 7 and 8 are grouped into extras

### Phases 7 to 9

Each has its own specification:

*   **[07 - Object-Aware Constraints](./07_Object_Aware_Constraints.md):** contact and penetration against object meshes, the floor, and the scene.
*   **[08 - Differentiable Losses and the Violation Report](./08_Differentiable_Losses_and_Violation_Report.md):** the constraints as losses and as a per-frame, per-joint report.
*   **[09 - Plausibility Metrics and Validation](./09_Plausibility_Metrics_and_Validation.md):** floor and object metrics, and validation on recorded grasps.

---

**Next: [07 - Object-Aware Constraints](./07_Object_Aware_Constraints.md)** or **[Glossary](./GLOSSARY.md)**
