# `direction-and-blocker/`

**Tester** (first-pass measurement) for cluster **direction-and-blocker** — Step 7 wiring blocker, camera/delta-transfer feasibility, augmentation reachability.

Run from the repo root: `PYTHONPATH=src:scripts/review_checks/direction-and-blocker .venv/bin/python scripts/review_checks/direction-and-blocker/<script>.py`.

Outputs (`*.txt`, `*.out`) were captured against the **pre-Step-8 snapshot (2026-10-04/05)**; re-running on current code may differ. See [`../index.md`](../index.md).

| File | Description |
| --- | --- |
| `e10_legacy_capabilities.py` | E10: are the six retired legacy tests a 'rejected design' or the adopted contract? |
| `e10_out.txt` | Captured output (from the script matching `e10*`). |
| `e11_hand_detachment.py` | E11: what happens to the image-space HANDS under each integration design, at ZERO augmentation? |
| `e11_out.txt` | Captured output (from the script matching `e11*`). |
| `e1_out.txt` | Captured output (from the script matching `e1*`). |
| `e1_pipeline_reachability.py` | E1: Is SMPL-X conversion reachable from the framework's augmentation pipeline at all? |
| `e2_out.txt` | Captured output (from the script matching `e2*`). |
| `e2_zero_noise_new_converter.py` | E2: Does the new converter make zero-noise intrinsic augmentation a no-op? |
| `e2b_clamp_after_step8.py` | E2b: Would Step 8 (a correct torso root) make zero-noise clamping a no-op? |
| `e2b_out.txt` | Captured output (from the script matching `e2b*`). |
| `e3a_extract_both.py` | E3a: one-pass extraction of image-space pose + hands and world-space pose from the SAME |
| `e3a_out.txt` | Captured output (from the script matching `e3a*`). |
| `e3b_camera_and_transfer.py` | E3b: Is 'no camera model exists to project world-space fits back' a real blocker? |
| `e3b_out.txt` | Captured output (from the script matching `e3b*`). |
| `e3c_debug_camera.py` | E3c: why does weak perspective fit worse than affine? Inspect affine singular values and |
| `e3d_weakpersp_check.py` | E3d: verify the weak-perspective fit with a generic nonlinear least-squares solver. |
| `e3e_out.txt` | Captured output (from the script matching `e3e*`). |
| `e3e_transfer_fixed.py` | E3e (supersedes E3b part 2, whose alternating weak-perspective solver was buggy; see E3d): |
| `e3f_consistency.py` | E3f: separate 'transfer method error' from 'MediaPipe world/image heads disagree'. |
| `e3f_out.txt` | Captured output (from the script matching `e3f*`). |
| `e3g_camera_residual.py` | E3g: per-frame camera fit between MediaPipe WORLD landmarks and IMAGE landmarks of the same |
| `e3g_out.txt` | Captured output (from the script matching `e3g*`). |
| `e4_out.txt` | Captured output (from the script matching `e4*`). |
| `e4_world_image_agreement.py` | E4: sanity check on E3f -- do MediaPipe's world and image heads agree about 2-D segment geometry? |
| `e5_elbow_local_axes.py` | E5: Are SMPL-X *local* elbow angles from the V1 converter anatomically meaningful? |
| `e5_out.txt` | Captured output (from the script matching `e5*`). |
| `e6_out.txt` | Captured output (from the script matching `e6*`). |
| `e6_smplx_vs_landmark_native.py` | E6: Does an SMPL-X round trip buy anything for an elbow-angle augmentation, versus doing it |
| `e7_fitted_body_facing.py` | E7: follow-up to E6. Which way does the CURRENT converter's fitted SMPL-X body face, and with |
| `e7_out.txt` | Captured output (from the script matching `e7*`). |
| `e8_mesh_plausibility_real_forward.py` | E8: confirm E7 with the REAL SMPL-X forward pass (not the repo's FK): run smplx on each |
| `e8_out.txt` | Captured output (from the script matching `e8*`). |
| `e9_after_step8_root.py` | E9: does the Step 8 design (root from shoulders+nose) repair local-angle semantics? |
| `e9_out.txt` | Captured output (from the script matching `e9*`). |
| `legacy_tests/` | Subfolder (see its index.md) |
| `index.md` | This index. |
