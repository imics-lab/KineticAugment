# `doc-consistency/`

**Tester** (first-pass measurement) for cluster **doc-consistency** — Plan-doc internal consistency checks and number reproduction.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/doc-consistency .venv/bin/python scripts/review_checks/doc-consistency/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `betas_contract_check.py` | Does BodyFit really carry everything needed to reconstruct? (doc 819-831: 'Reconstruction takes |
| `betas_contract_check_out.txt` | Captured output (from the script matching `betas_contract_check*`). |
| `check_doc_numbers.py` | Check the numeric claims in docs/V1_Problem_Scope.md and presentation_one.md. |
| `check_doc_numbers_out.txt` | Captured output (from the script matching `check_doc_numbers*`). |
| `check_doc_numbers_out_v2.txt` | Captured output (from the script matching `check_doc_numbers_out_v2*`). |
| `check_doc_numbers_time.txt` | Captured output (from the script matching `check_doc_numbers_time*`). |
| `doc_contract_checks.py` | Empirical checks behind doc-consistency claims in V1_Problem_Scope.md / presentation_one.md. |
| `doc_contract_checks_out.txt` | Captured output (from the script matching `doc_contract_checks*`). |
| `p1_similarity_variants.py` | P1 robustness: is the similarity failure an artefact of handedness or of z-scale alone? |
| `p1_similarity_variants_out.txt` | Captured output (from the script matching `p1_similarity_variants*`). |
| `p1_zscale.py` | Fitted pseudo-depth scale: image z (x image width) relative to x/y, from world->image fit. |
| `p1_zscale_out.txt` | Captured output (from the script matching `p1_zscale*`). |
| `round_trip_out.txt` | Captured output (from the script matching `round_trip*`). |
| `round_trip_out_v2.txt` | Captured output (from the script matching `round_trip_out_v2*`). |
| `unobserved_output_check.py` | For frames where a modelled landmark (wrist) was INVALID, what does reconstruct_body emit for it? |
| `unobserved_output_check_out.txt` | Captured output (from the script matching `unobserved_output_check*`). |
| `index.md` | This index. |
