# `shape-identifiability/`

**Tester** (first-pass measurement) for cluster **shape-identifiability** — A006: length Jacobian w.r.t. betas, shape non-uniqueness, held-out shape fits.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/shape-identifiability .venv/bin/python scripts/review_checks/shape-identifiability/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `common.py` | Shared linear SMPL-X shape model (exact: joints are linear in betas). |
| `map_fits.npy` | Cached array produced by a script in this folder and loaded by another. |
| `pair_height_rand.npy` | Cached array produced by a script in this folder and loaded by another. |
| `pair_height_tmpl.npy` | Cached array produced by a script in this folder and loaded by another. |
| `pair_leg_rand.npy` | Cached array produced by a script in this folder and loaded by another. |
| `pair_leg_tmpl.npy` | Cached array produced by a script in this folder and loaded by another. |
| `pair_volume_rand.npy` | Cached array produced by a script in this folder and loaded by another. |
| `pair_volume_tmpl.npy` | Cached array produced by a script in this folder and loaded by another. |
| `s0_crosscheck.py` | Sanity: the exact-linear model in common.py == the repo's SMPLXSkeleton and the |
| `s0_crosscheck.txt` | Captured output (from the script matching `s0_crosscheck*`). |
| `s10_ratio_reach.py` | Doc line ~724 claims the 13% fore/upper discrepancy (obs 1.066 vs SMPL-X 0.946) is |
| `s10_ratio_reach.txt` | Captured output (from the script matching `s10_ratio_reach*`). |
| `s11_heldout.py` | Q5: does a sequence-level shape fit generalise within a clip? Fit betas on the |
| `s11_heldout.txt` | Captured output (from the script matching `s11_heldout*`). |
| `s12_nose.py` | Step 8 adds the nose (rigid to the head under V1's neck/head=0 policy). If the |
| `s12_nose.txt` | Captured output (from the script matching `s12_nose*`). |
| `s1_jacobian.py` | Q1/Q2: rank, singular values and conditioning of observed-length Jacobian. |
| `s1_jacobian.txt` | Captured output (from the script matching `s1_jacobian*`). |
| `s1b_random_betas_only.py` | Supplement to s1: 10-beta-only Jacobian (scale fixed, as converter.py) at random betas, |
| `s1b_random_betas_only.txt` | Captured output (from the script matching `s1b_random_betas_only*`). |
| `s2_nonunique.py` | Q3: two clearly different bodies with (near-)identical observed lengths. |
| `s2_nonunique.txt` | Captured output (from the script matching `s2_nonunique*`). |
| `s3_pipeline_check.py` | Q3 via the repo's own code + the real smplx forward pass. |
| `s3_pipeline_check.txt` | Captured output (from the script matching `s3_pipeline_check*`). |
| `s3_rerun.txt` | Captured output (from the script matching `s3_rerun*`). |
| `s4_frames.py` | Q4: do many frames add shape information? Full real-SMPL-X-forward Jacobians. |
| `s4_frames.txt` | Captured output (from the script matching `s4_frames*`). |
| `s5_real.py` | Q4/Q5 on real clips: length time-series statistics, and what per-clip |
| `s5_real.txt` | Captured output (from the script matching `s5_real*`). |
| `s6_real_roundtrip.py` | Q5: what does a per-clip, sequence-level shape fit buy on the real round trip? |
| `s6_real_roundtrip.txt` | Captured output (from the script matching `s6_real_roundtrip*`). |
| `s6_rerun.txt` | Captured output (from the script matching `s6_rerun*`). |
| `s7_posterior.py` | Q2/Q5: what the 5 lengths actually teach about shape, under the SMPL-X prior |
| `s7_posterior.txt` | Captured output (from the script matching `s7_posterior*`). |
| `s8_pose_dependence.py` | Q4 on real clips: are frame-to-frame length variations iid noise (would average |
| `s8_pose_dependence.txt` | Captured output (from the script matching `s8_pose_dependence*`). |
| `s8b_shift_null.py` | Autocorrelation-respecting null for s8: circularly shift the length series |
| `s8b_shift_null.txt` | Captured output (from the script matching `s8b_shift_null*`). |
| `s9_betas_contract.py` | New-issue check: if Step 10 hands fit_body a shaped skeleton, are the returned |
| `s9_betas_contract.txt` | Captured output (from the script matching `s9_betas_contract*`). |
| `index.md` | This index. |
