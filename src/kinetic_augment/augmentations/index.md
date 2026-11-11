# augmentations/

| File | Description |
|---|---|
| `__init__.py` | Exports the registered augmentation classes and the week-4 entry points (`AugmentationChain`, `load_profile`, `ClipData`, `build_outputs`) |
| `base.py` | Base classes (`BaseAugmentation`, landmark/SMPL-X variants, `ClipAugmentation`, region tags) and `AugmentationRegistry` |
| `ranges.py` | Seeded per-clip RNG, range parsing (uniform and log-uniform), truncated-normal sampling |
| `clip.py` | Week-4 clip container (`ClipData`, `SceneTransform`, `ChainContext`), building it from a `run_clip` result, slerp re-timing and frame selection |
| `outputs.py` | MediaPipe-layout outputs from a `ClipData`: SMPL-X skinning on edited rest joints (`pose_mesh_rest`), pose arms, hands, carried face, scene transform, mirroring and camera projection |
| `arm_ik.py` | Analytic two-bone arm IK (wrist to a world target, swivel plane and hand world orientation kept), local `world_pose` FK and the torso frame |
| `body.py` | Week-4 body augmentations: coupled arm-joint noise (`body.joint_coupled_noise`), the Tier 1 arm-joint wrapper (`body.tier1_arm_joints`), limb length scaling with wrist retargeting (`body.limb_length_scaling`), movement-size scaling (`body.movement_size`) and location jitter (`body.location_jitter`), plus the shared `retarget_wrists` helper |
| `handshape.py` | Handshape features (signed finger bend, finger states, spread, thumb abduction), `features_changed`, per-joint flexion axes and the mirrored finger-offset helper |
| `fingers.py` | Week-4 finger augmentations: coupled finger flexion noise (`fingers.joint_coupled_noise`) and handshape-preserving per-joint jitter (`fingers.handshape_jitter`) |
| `face.py` | Week-4 face augmentation: blendshape groups, the per-clip ridge expression map (`fit_expression_map`) and expression jitter (`face.expression_jitter`) |
| `face_mirror.py` | Generated 478-point MediaPipe face mirror table (`FACE_MIRROR`, an involution); produced by `scripts/make_face_mirror.py` |
| `scene.py` | Week-4 scene augmentations composed into `clip.scene`: mirror, view, scale, uncentre (`scene.*`), plus `fit_clip_camera` and the re-exported `blendshape_mirror` |
| `profile.py` | Week-4 YAML profile loader (`load_profile`, with enable/disable overrides and range-key validation) and `tier1_ranges` |
| `chain.py` | `AugmentationChain`: runs the enabled profile entries in a fixed `ORDER` with per-entry seeded draws; returns `ChainResult` |
| `extrinsic.py` | Landmark-space transforms: rotation, scaling, flip, trajectory jitter, noise |
| `intrinsic.py` | SMPL-X parameter-space joint, hand, coupled and global-orientation perturbations |
| `temporal.py` | Time warping, speed variation, frame dropping |
| `timing.py` | Week-4 temporal augmentations: time warp modes (uniform, varied speed, slowed segment), frame dropping (rate or random with a gap limit; none/hold/interpolate fill) and hand-dropout runs |
| `tier1.py` | Tier 1 augmentations (Augmentation_Grounding.md §3: mirror, view, duration, arm joints, scale, uncentre) with a weak-perspective camera that keeps the hands in frame; not in the registry or `Pipeline` |
| `report.py` | `measure(base, aug, aligned)`: wrist movement, bone-length change, wrist jitter ratio, face rigidity, closest hand-to-face distance and in-frame share for one augmented run |
| `index.md` | This index |
