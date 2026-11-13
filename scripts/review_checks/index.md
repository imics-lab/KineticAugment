# `scripts/review_checks/` — evidence for the V1 review assessment

**What this is:** the verification scripts and captured outputs behind
[`docs/V1_Review_Assessment.md`](../../docs/V1_Review_Assessment.md). That report empirically tests an external
review of the V1 MediaPipe ↔ SMPL-X plan ([`reviewer_inputs/`](reviewer_inputs/index.md)) against
[`docs/V1_Problem_Scope.md`](../../docs/V1_Problem_Scope.md).

**What this is not:** project code or tests. Nothing in `src/` or `tests/` imports these files. They are frozen
research evidence, not maintained tooling, so don't refactor them, and don't add them to CI.

## ⚠️ Read before trusting any number here

- All captured outputs (`*.txt`, `*.out`) were produced on a **snapshot taken 2026-10-04/05, before Step 8 landed**
  (at that time: Step 7 complete, suite 66 passed / 13 failed [Step 8 red-phase] / 2 xfailed).
- **Step 8 has since been completed** (suite 81 passed / 2 xfailed on 2026-10-05). Re-running these scripts on current code
  gives different results where Step 8 changed behaviour. Example: `main-check/upside.py` reported the fitted torso 164° from
  the data's up-axis (head above the shoulders 0/173 frames) on the snapshot. On current code it reports 6.4° (173/173).
- Re-extracted landmark `.npz` files and large arrays were **not** copied. Scripts that read
  `scripts/review_checks/*/extract/...` need re-extraction first (see `noise-floor/reextract.py`,
  `external-literature/reextract.py`).
- Some scripts have no docstring; their purpose is given by the cluster and by the report section that cites them.

## How to run

From the repo root, with the project venv:

```bash
PYTHONPATH=src:scripts/review_checks/<folder> .venv/bin/python scripts/review_checks/<folder>/<script>.py
```

Folders that ship a `common.py`/`vcommon.py`/`armlib.py` helper need that folder on `PYTHONPATH` (as above).

## Layout

Each claim cluster has two folders: a **tester** (first-pass measurement) and a **`verify-`** folder (an independent agent
that rewrote the key computations to try to refute the tester). Where they disagreed, the report follows the one with
reproducible computation. Cluster IDs (A001–A007) refer to the review's assumption ledger.

| Cluster | Tester | Verifier | Topic |
| --- | --- | --- | --- |
| A001 / A007 | [`fk-and-wiring/`](fk-and-wiring/index.md) | [`verify-fk-and-wiring/`](verify-fk-and-wiring/index.md) | FK vs real SMPL-X; legacy converter wiring; Step 7 contract claims |
| A002 | [`root-counterexample/`](root-counterexample/index.md) | [`verify-root-counterexample/`](verify-root-counterexample/index.md) | Root orientation; nose vs hip torso frame |
| A003 | [`twist-identifiability/`](twist-identifiability/index.md) | [`verify-twist-identifiability/`](verify-twist-identifiability/index.md) | Twist null space; joint plausibility; elbow limit axes |
| A004 | [`noise-floor/`](noise-floor/index.md) | [`verify-noise-floor/`](verify-noise-floor/index.md) | Noise floor; MediaPipe smoothing; frame rate |
| A005 | [`real-data-numbers/`](real-data-numbers/index.md) | [`verify-real-data-numbers/`](verify-real-data-numbers/index.md) | Headline numbers; Step 6 oracle decomposition |
| A006 | [`shape-identifiability/`](shape-identifiability/index.md) | [`verify-shape-identifiability/`](verify-shape-identifiability/index.md) | Betas from arm lengths; non-uniqueness; held-out |
| External | [`external-literature/`](external-literature/index.md) | [`verify-external-literature/`](verify-external-literature/index.md) | MediaPipe docs; HybrIK adaptive IK; extractor state leak |
| Doc | [`doc-consistency/`](doc-consistency/index.md) | [`verify-doc-consistency/`](verify-doc-consistency/index.md) | Plan-doc contradictions and stale numbers |
| Direction | [`direction-and-blocker/`](direction-and-blocker/index.md) | [`verify-direction-and-blocker/`](verify-direction-and-blocker/index.md) | Wiring blocker; camera / delta transfer; augmentation reachability |
| Bugs | [`missed-bugs/`](missed-bugs/index.md) | [`verify-missed-bugs/`](verify-missed-bugs/index.md) | Independent bug hunt |
| — | [`main-check/`](main-check/index.md) | — | Coordinator spot-checks (upside-down torso, elbow hinge axis) |
| — | [`reviewer_inputs/`](reviewer_inputs/index.md) | — | The review under test and its audits |

- `index.md` — This index.
