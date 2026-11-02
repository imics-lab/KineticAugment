# Tests index

- `__init__.py` — Marks the automated test directory as a Python package.
- `test_alignment.py` — Tests least-squares similarity alignment between coordinate spaces.
- `test_body_model.py` — Tests joint mappings, SMPL-X wrapper, and both landmark converters.
- `test_converter.py` — Tests the V1 converter contract and its round trip through SMPL-X.
- `test_extract_world_landmarks.py` — Tests that a clip's extracted landmarks do not depend on the clip processed before it.
- `test_kinematics.py` — Tests skeleton sources and forward kinematics against the real SMPL-X model.
- `test_shape.py` — Tests the per-clip shape fit against the converter's own round-trip error and SMPL-X's shape prior.
- `test_swing_ik.py` — Tests analytic swing inverse kinematics against known synthetic ground truth.
- `hand_assets.py` — Shared model-asset paths, skip marks and a cached hand-model loader for the hand tests.
- `test_hand_model.py` — Tests the 21-point hand operator against the full MANO/SMPL-X forward pass and hand anatomy.
- `test_hand_converter.py` — Tests the hand fit: analytic start, real-model refinement, decoder, serialisation, shape and per-frame scale.
- `test_tier1.py` — Tests the Tier 1 augmentations: sampling ranges, transforms, re-timing, the camera fit and the hands-in-frame limit.
- `test_hand_observations.py` — Tests the left/right assignment and the hand clip file format.
- `test_extract_hand_landmarks.py` — Runs the hand extractor on pilot clips: clip-order independence, units, sides.
- `index.md` — Index of automated test files.
