# Augmentations index

- `__init__.py` — Exports augmentation interfaces and concrete transformation classes.
- `base.py` — Defines augmentation configuration, base classes (incl. `ClipAugmentation`, region tags), and registry.
- `ranges.py` — Seeded per-clip RNG, range parsing (uniform and log-uniform), truncated-normal sampling.
- `clip.py` — Week-4 clip container (`ClipData`, `SceneTransform`, `ChainContext`), building it from a `run_clip` result, slerp re-timing and frame selection.
- `outputs.py` — MediaPipe-layout outputs from a `ClipData`: SMPL-X skinning on edited rest joints (`pose_mesh_rest`), pose arms, hands, carried face, scene transform, mirroring and camera projection.
- `arm_ik.py` — Analytic two-bone arm IK (wrist to a world target, swivel plane and hand world orientation kept), local `world_pose` FK and the torso frame.
- `body.py` — Week-4 body augmentations: coupled arm-joint noise (`body.joint_coupled_noise`) the Tier 1 arm-joint wrapper (`body.tier1_arm_joints`), limb length scaling with wrist retargeting (`body.limb_length_scaling`), movement-size scaling (`body.movement_size`) and location jitter (`body.location_jitter`), plus the shared `retarget_wrists` helper.
- `handshape.py` — Handshape features (signed finger bend, finger states, spread, thumb abduction), `features_changed`, per-joint flexion axes and the mirrored finger-offset helper.
- `fingers.py` — Week-4 finger augmentations: coupled finger flexion noise (`fingers.joint_coupled_noise`) and handshape-preserving per-joint jitter (`fingers.handshape_jitter`).
- `extrinsic.py` — Implements global landmark-space transforms and noise.
- `intrinsic.py` — Implements SMPL-X joint, hand, limb, and orientation perturbations.
- `temporal.py` — Implements time warping, speed variation, and frame dropping.
- `tier1.py` — Tier 1 augmentations (Augmentation_Grounding.md §3): per-clip sampling from the evidence ranges, view/mirror/scale transforms, uniform duration change, and a weak-perspective camera that keeps the hands in frame when uncentring.
- `index.md` — Index of modular augmentation implementations.
