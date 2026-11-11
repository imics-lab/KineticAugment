# body_model/

V1 to V1.3 MediaPipe <-> SMPL-X stack. (The old image-space converters moved to `../legacy/`.)

| File | Description |
|---|---|
| `__init__.py` | Exports the V1 API; lazy getters for `SMPLXWrapper` and the legacy converters |
| `converter.py` | V1 body converter: `fit_body` / `reconstruct_body` between world landmarks and SMPL-X parameters |
| `skeleton.py` | SMPL-X rest-pose skeleton sources for kinematics |
| `shape.py` | Per-clip body shape (betas) fit |
| `kinematics.py` | Forward kinematics over a rest-pose skeleton |
| `swing_ik.py` | Analytic swing inverse kinematics from landmarks to rotations |
| `alignment.py` | Similarity (Umeyama) alignment between coordinate spaces |
| `assembly.py` | Assemble one SMPL-X pose from the body, hand and head fits |
| `hand_model.py` | MediaPipe's 21 hand landmarks evaluated on MANO / SMPL-X |
| `hand_converter.py` | V1.2 hand landmarks <-> MANO / SMPL-X hand parameters |
| `hand_observations.py` | MediaPipe hand observation loading for the hand round trip |
| `hand_depth.py` | Wrist position relative to the head recovered from the image |
| `head_converter.py` | SMPL-X neck, head and jaw from FaceLandmarker landmarks |
| `face_correspondence.py` | Where MediaPipe face-mesh points sit on the SMPL-X head surface |
| `face_regions.py` | SMPL-X head/neck surface labelled with ASL-LEX location codes |
| `face_stream.py` | Reversible attachment of observed face landmarks to a posed head |
| `face_observations.py` | MediaPipe FaceLandmarker observations (landmarks, blendshapes, transform) |
| `face_features.py` | Versioned serialization/validation of face feature archives |
| `face_pipeline.py` | One clip end to end: body, hands, head, hand-to-face geometry |
| `face_metrics.py` | V1.3 gate metrics (head rotation vs FaceLandmarker, correspondence floor) |
| `hand_face_geometry.py` | Signed hand-to-head/neck distances with region labels |
| `image_proximity.py` | Hand-to-face proximity measured in the image |
| `joint_mapping.py` | SMPL-X <-> MediaPipe joint correspondence tables and limits |
| `smplx_wrapper.py` | `SMPLXWrapper` around the `smplx` library; used by constraints and the registry API |
