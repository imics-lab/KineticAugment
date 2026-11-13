# body_model/

V1 to V1.3 MediaPipe <-> SMPL-X stack. (The old image-space converters moved to `../legacy/`.)

| File | Description |
|---|---|
| `__init__.py` | Exports the V1 API; lazy getters for `SMPLXWrapper` and the legacy converters |
| `converter.py` | V1 body converter: `fit_body` / `reconstruct_body` between world landmarks and SMPL-X parameters |
| `skeleton.py` | Rest-pose joints, parent trees, and SMPL-X's linear shape space for fitting |
| `shape.py` | Fits one plausible SMPL-X body shape (betas) per clip by minimising the round-trip error |
| `kinematics.py` | Composes local joint rotations down the chain into world joint positions |
| `swing_ik.py` | Solves SMPL-X joint rotations analytically from observed bone directions |
| `alignment.py` | Similarity (Umeyama) alignment between coordinate spaces for comparing poses |
| `assembly.py` | Poses one SMPL-X body from the body, hand and head fits |
| `hand_model.py` | Evaluates MediaPipe's 21 hand landmarks on MANO or SMPL-X (joints plus fingertip vertices) |
| `hand_converter.py` | Fits MANO / SMPL-X hand parameters to MediaPipe hand landmarks and rebuilds them from parameters alone; optionally keeps a per-frame scale residual for MediaPipe's size drift |
| `hand_observations.py` | Holds per-clip MediaPipe hand observations and assigns each detected hand its anatomical side |
| `hand_depth.py` | Estimates each wrist's 3-D position relative to the nose from the image (pinhole depth from the head and hand scales) so the body can be re-fitted (V1.3) |
| `head_converter.py` | Fits the SMPL-X neck, head and jaw to MediaPipe face landmarks |
| `head_placement.py` | Spike (08925 head placement): swings the neck so the head joint lands on the weighted mean of per-point votes from several observed face points; keeps the head's world rotation exact |
| `face_correspondence.py` | MediaPipe face-mesh groups and their positions on the SMPL-X surface, via the FLAME embedding |
| `face_regions.py` | SMPL-X head and neck surface labelled with ASL-LEX location regions and subregions from landmark, normal and skinning rules (V1.3) |
| `face_stream.py` | Reversibly attaches observed 478-point face streams to posed head frames using rigid-point similarity alignment |
| `face_observations.py` | Per-clip MediaPipe FaceLandmarker observations (478 landmarks, 52 blendshapes, head matrix) and their `_face.npz` file format |
| `face_features.py` | Validates aligned hand and face inputs and reads/writes versioned feature archives with provenance metadata |
| `face_pipeline.py` | Runs one clip end to end (body, hands, head, hand-to-face geometry) and writes the per-frame `_face_features.npz` (V1.3) |
| `face_metrics.py` | Gate metrics that need more than a mean: fitted head rotation vs FaceLandmarker's head pose, and the correspondence floor (V1.3) |
| `hand_face_geometry.py` | Signed distances from each hand's wrist, fingertips and palm centre to the head and neck surface, with region labels and contact (V1.3) |
| `image_proximity.py` | Projects the fitted head into the image and measures hand points against it in 2-D: overlap, distance at head depth, region, near; Neck is excluded on purpose (V1.3) |
| `joint_mapping.py` | SMPL-X <-> MediaPipe joint indices, chains, correspondence tables and angle limits |
| `smplx_wrapper.py` | `SMPLXWrapper` around the `smplx` library: loads SMPL-X assets, runs inference, handles parameters; used by constraints, the registry API and the week-4 outputs |
| `index.md` | This index |
