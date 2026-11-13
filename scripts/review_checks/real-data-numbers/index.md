# `real-data-numbers/`

**Tester** (first-pass measurement) for cluster **real-data-numbers** — A005: reproduces real-clip headline numbers and the Step 6 oracle decomposition.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/real-data-numbers .venv/bin/python scripts/review_checks/real-data-numbers/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `a005_serialized.out` | Captured output (from the script matching `a005_serialized*`). |
| `a005_serialized.py` | A005: can the scored output be rebuilt from saved parameters alone? |
| `aligned_p95.out` | Captured output (from the script matching `aligned_p95*`). |
| `aligned_p95.py` | Aligned per-joint p95 (doc's older Step 5 table) + IK-frame direction error at full precision. |
| `baseline_frames.out` | Captured output (from the script matching `baseline_frames*`). |
| `baseline_frames.py` | Reproduce headline numbers and examine frames / scale / directions / lengths. |
| `betas_contract.out` | Captured output (from the script matching `betas_contract*`). |
| `betas_contract.py` | Is BodyFit self-describing when the skeleton carries a non-default shape? |
| `by_count.out` | Captured output (from the script matching `by_count*`). |
| `by_count.py` | Direct (converter) error stratified by how many arm landmarks were valid in the frame: |
| `common.py` | Shared helpers for the real-data-numbers audit (read-only use of the snapshot). |
| `decomp.py` | Oracle decomposition of the real-data round-trip error, re-derived from scratch. |
| `decomp_analyse.out` | Captured output (from the script matching `decomp_analyse*`). |
| `decomp_analyse.py` | 3-factor Shapley with shoulder width |
| `decomp_extra.out` | Captured output (from the script matching `decomp_extra*`). |
| `decomp_extra.py` | Follow-ups on the decomposition: bootstrap CIs, scale spread under oracle shape, |
| `decomp_run.py` | Parallel driver for decomp.py: every (mode, root, len, sw) config x clip as a task. |
| `insample.out` | Captured output (from the script matching `insample*`). |
| `insample.py` | How much does the converter's per-frame rigid fit (now serialised as global_orient/transl) absorb? |
| `length_jitter.out` | Captured output (from the script matching `length_jitter*`). |
| `length_jitter.py` | How much do MediaPipe world bone lengths vary frame-to-frame within a clip? |
| `measure_round_trip.out` | Captured output (from the script matching `measure_round_trip*`). |
| `oracles.py` | Attempt to reconstruct the doc's Step 5/6 oracle decomposition (no script for it exists in the repo). |
| `torso_orientation.out` | Captured output (from the script matching `torso_orientation*`). |
| `torso_orientation.py` | Is the converter's SMPL-X torso oriented consistently with the observed (unscored) torso landmarks? |
| `index.md` | This index. |
