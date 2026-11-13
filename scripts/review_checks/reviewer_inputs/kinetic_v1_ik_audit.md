# V1 IK scope audit

## Verdict

The analytic swing IK is a sound minimal solution for matching the *directions of the four observed arm segments* under a documented zero-twist convention. The current scope overclaims what that proves. It does not determine unique/anatomically true SMPL-X parameters, and exact segment directions do not guarantee low joint-position error. The scope should narrow its claim and correct its error attribution before Step 7 is treated as measurement-gated shape fitting.

## Evidence and mathematical audit

- The scope §4 (lines 103–143) still defines 12 landmarks and says 8 joints are constrained, with torso orientation in `global_orient`. Later Finding 5 (§ around lines 330–355) establishes hips only visible 32%, legs 0%, only four arm joints constrainable, and proposes a shoulder/nose torso frame. Actual code is arm-only: `SMPLX_ARM_BONES` lists shoulder→elbow and elbow→wrist for each side (`swing_ik.py:29–37`). The proposed Step 4 checklist at lines 517–523 is also stale: it says derive `global_orient`, while the completed implementation explicitly defers estimation (lines 488–499).
- Direction matching is exactly the solver's objective: it reads normalized landmark differences only (`swing_ik.py:87–91, 127–135`). Parent-relative composition is coherent for those directions. Twist is underdetermined by a terminal segment: rotating about its own axis leaves its endpoint unchanged. The tests explicitly demonstrate this (`tests/test_swing_ik.py:108–120`). Thus “genuine valid inputs” is supportable; “correct actual parameters” or anatomical plausibility is not. Zero twist is a choice in the equivalence class.
- The 200-pose check is a position check, not a parameter recovery check: `scripts/measure_round_trip.py:80–101` generates random rotations but reports only the worst positions for six arm joints. The SMPL-X unit test also compares positions, expressly because twist is unobservable (`tests/test_swing_ik.py:154–168`). It cannot establish rotation recovery or parameter truth.
- “Direction error exact ⇒ all position error is bone length” is false with frozen torso/root. A repeatable same-skeleton counterexample is in `/private/tmp/kinetic_v1_root_counterexample.py`; run from the repository root with `.venv/bin/python /private/tmp/kinetic_v1_root_counterexample.py`. Captured output is `/private/tmp/kinetic_v1_root_counterexample.txt`. Rotating the synthetic body root by 0.7 rad, then fitting with default root, leaves fitted root zero. Segment direction error is max 1.21e-6° (mean 5.15e-7°), while six-joint position error is max 0.31915 m / mean 0.31689 m raw and max 0.07918 m / mean 0.04470 m after the harness's per-frame similarity alignment. Segment lengths are identical by construction. Shoulder anchors are wrong because their torso/root chain remained frozen. This contradicts the implication in scope lines 482–499 and 544–565. Code confirms the solver's root is zero unless explicitly passed and estimates no torso frame (`swing_ik.py:113–135`); the harness calls it without `global_orient` (`measure_round_trip.py:123–136`).
- Translation/scale/rotation are discarded by direction fitting; `solve_swing_ik` says it is invariant to scale and translation (`swing_ik.py:87–91`). The scope itself says a similarity transform must be round-trip state (`V1_Problem_Scope.md:449–460`), but the current harness only aligns for scoring. A parameter output alone therefore cannot reconstruct source coordinates unless that transform is part of the delivered contract.

## Verification

Ran `.venv/bin/pytest -q tests/test_swing_ik.py tests/test_kinematics.py`: 27 passed. Ran the read-only root-rotation counterexample; exact output is saved at `/private/tmp/kinetic_v1_root_counterexample.txt`. No repository files were changed.

## Recommended minimum scope edits

1. Recast the core guarantee as “fit the four observed arm-bone directions with zero twist; other rotations and root/torso pose are policy/defaults,” and state parameters are valid but not uniquely recovered or guaranteed anatomically plausible.
2. Fix stale §4 and Step 4 checklist to match actual observed arm-only data and implementation; remove “global_orient for those limbs” from the deliverable unless root/torso estimation is brought back into scope.
3. Replace the claim that direction exactness isolates length error with a decomposition that includes frozen-anchor/torso/root error. Add the root-rotation same-skeleton case to synthetic validation.
4. Make source-to-model similarity transform part of the conversion state/output contract, and distinguish normalized/aligned position error from a true direct round trip.
5. Reassess Step 7 only after a valid coordinate/anchor-aware measurement; shape coefficients cannot repair an unmodeled global torso orientation or positional anchor error.
