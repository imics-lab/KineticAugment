# Body model index

- `__init__.py` — Provides lazy accessors for SMPL-X and MediaPipe conversion classes.
- `alignment.py` — Fits the similarity between two coordinate spaces for comparing poses.
- `converter.py` — Converts world body landmarks to SMPL-X parameters and back.
- `joint_mapping.py` — Defines SMPL-X/MediaPipe joint indices, chains, and angle limits.
- `kinematics.py` — Composes local joint rotations down the chain into world joint positions.
- `shape.py` — Fits one plausible SMPL-X body shape per clip by minimising the round-trip error.
- `skeleton.py` — Supplies rest-pose joints, parent trees, and SMPL-X's linear shape space for fitting.
- `mp_to_smplx.py` — Converts MediaPipe landmarks into SMPL-X pose parameters.
- `smplx_to_mp.py` — Converts SMPL-X pose parameters back to MediaPipe landmarks.
- `swing_ik.py` — Solves SMPL-X joint rotations analytically from observed bone directions.
- `smplx_wrapper.py` — Loads SMPL-X assets and runs model inference and parameter handling.
- `hand_model.py` — Evaluates MediaPipe's 21 hand landmarks on MANO or SMPL-X (joints plus fingertip vertices).
- `hand_converter.py` — Fits MANO / SMPL-X hand parameters to MediaPipe hand landmarks and rebuilds them from parameters alone; optionally keeps a per-frame scale residual for MediaPipe's size drift.
- `hand_observations.py` — Holds per-clip MediaPipe hand observations and assigns each detected hand its anatomical side.
- `index.md` — Index of body-model mappings, converters, and wrapper code.
