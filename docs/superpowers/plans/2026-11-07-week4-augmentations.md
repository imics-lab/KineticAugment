# Week 4 Augmentations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add nine per-clip augmentations tagged face / body / fingers / temporal / scene, a
chain that runs them on the V1.3 body + hands + face round trip, and three YAML profiles whose
ranges sit about 25% beyond the proven literature ranges.

**Architecture:** A clip container (`ClipData`) holds one round-trip clip's SMPL-X rotations,
rest joints, carried face stream, blendshapes, masks, timing, weak camera and a pending scene
transform. Each augmentation is a `ClipAugmentation` with `sample(rng, clip, ctx)` (one draw per
clip) and `apply(target, draw, ctx)`. Intrinsic augmentations edit `ClipData`; scene
augmentations compose into `ClipData.scene`; temporal ones re-time `ClipData`; hand dropout
edits the final `ClipOutputs`. `build_outputs` turns a `ClipData` into MediaPipe-layout
landmarks through a custom linear-blend-skinning pass that honours edited rest joints. A YAML
loader builds an `AugmentationChain` that runs them in a fixed order.

**Tech Stack:** Python 3.9+, numpy, scipy (`Rotation`, `Slerp`), smplx + torch (via the existing
`SMPLXSkeleton`), pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-week4-augmentations-design.md`. The interface facts
below were checked against the code on 2026-11-07.

## Spec corrections found while planning

These change how, not what, and are the plan's calls:

1. `tier1.resample` is plain linear interpolation. Rotations are resampled with a new
   `slerp_resample` (Task 2). Points still use `tier1.resample_points`.
2. `assembly.pose_mesh` re-regresses joints from `betas`, so edited rest joints never reach the
   mesh. Task 2 adds `pose_mesh_rest`, a copy of `smplx.lbs.lbs` that takes the joints as input.
   All outputs go through it, so limb scaling reaches the mesh, the hand points and the tips.
3. No wrist-target IK exists (`swing_ik` matches directions only). Task 3 adds an analytic
   two-bone IK.
4. Blendshapes are not in `ClipFaceResult`; `clip_from_result` takes
   `face.blendshapes[result.frames]`.
5. Mirroring has no valid 478-point left/right table (`utils/landmark_maps.py` is wrong). Task 9
   generates one from MediaPipe's canonical face model. Use `tier1.POSE_MIRROR` for the pose.
6. The cache is 20 clips in `data/round_trip_eval` + 5 in `data/test_output`. Measured hand-miss
   runs (20 eval clips, interior gaps): 41 runs, median 1, max 6 frames, 0.032 run starts per
   detected hand-frame. Task 10 reproduces these numbers.
7. `JointCoupledNoise` and `FrameDropping` are already registered names. New augmentations
   register as `"<region>.<key>"` (e.g. `"body.joint_coupled_noise"`), and the registry refuses a
   name that is already registered to a different class.

## Global Constraints

- Python 3.9 compatible: `Optional[...]`/`Tuple[...]` from `typing`, no `X | Y`, no `match`.
- Black, line length 100.
- Code under `src/kinetic_augment/augmentations/`, tests under `tests/`, profiles under `configs/`.
- Do not change `tier1.TIER1`, `tier1.Tier1Ranges` defaults, the old classes in `intrinsic.py`,
  `temporal.py`, `extrinsic.py`, or the old profiles. `tests/test_tier1.py` must keep passing.
- Every new or renamed file is added to its folder's `index.md`.
- Commit messages carry no AI attribution or co-author lines (user rule).
- Run tests with `.venv/bin/python -m pytest`. Tests needing SMPL-X use
  `tests.hand_assets.requires_smplx`; tests needing a real clip skip when
  `data/test_output/01991_hands.npz` is absent.
- Draws are once per clip. No use of global `np.random`. Zero strength reproduces the input
  exactly (`np.array_equal(..., equal_nan=True)`).
- Every range value in the YAML carries a `# proven: ...` comment (§5.0 of the spec).

## Review Focus

1. **Frames where the body was not fitted** (`body_ok` false, NaN rotations): every augmentation
   leaves those frames NaN and never spreads NaN into fitted frames. Tests in Tasks 2, 4, 6.
2. **A hand that was never detected** (`hand_valid` false for the whole clip): finger
   augmentations and hand dropout skip it; mirror swaps its validity with the other hand. Tests in
   Tasks 7, 9, 10.
3. **Very short clips** (2–4 frames, fewer than `num_knots`): time warping and frame dropping
   return ≥ 2 frames and do not raise. Test in Task 10.
4. **A clip with no face frames**: expression jitter records the fallback and changes nothing;
   `build_outputs` returns an all-NaN face. Tests in Tasks 2 and 8.
5. **An IK target out of reach** (limb shortened, arm nearly straight): IK fully extends towards
   the target, returns the residual, never NaN. Tests in Tasks 3 and 5.

---

### Task 1: Regions, the clip-augmentation base class and sampling helpers

**Files:**
- Modify: `src/kinetic_augment/augmentations/base.py` (BaseAugmentation attrs ~:35, AugmentationRegistry :292-334)
- Create: `src/kinetic_augment/augmentations/ranges.py`
- Test: `tests/test_week4_base.py`
- Modify: `src/kinetic_augment/augmentations/index.md`

**Interfaces:**
- Produces (`base.py`):
  - `REGIONS: Tuple[str, ...] = ("face", "body", "fingers", "temporal", "scene")`
  - `BaseAugmentation.regions: Tuple[str, ...] = ()` (class attribute)
  - `AugmentationRegistry.register(name)` raises `ValueError` when `name` is already registered
    to a class with a different `(__module__, __qualname__)`.
  - `AugmentationRegistry.list_by_region(region: str) -> List[str]`
  - `class ClipAugmentation(BaseAugmentation)`: class attrs `key: str`, `regions`,
    `stage: str = "clip"` (or `"outputs"`), `DEFAULT_RANGES: Dict[str, Any]`,
    `DISTRIBUTIONS: Dict[str, str]` (key → `"uniform"`/`"log_uniform"` for `[lo, hi]` keys).
    `__init__(self, ranges: Optional[Dict[str, Any]] = None, probability: float = 1.0)`; sets
    `self.ranges` (defaults merged with given), `self.probability`, `self.defaulted: List[str]`
    (keys that fell back to the default). Unknown range key → `ValueError` naming the key.
    Abstract: `sample(self, rng, clip, ctx) -> Dict[str, Any]`,
    `apply(self, target, draw: Dict[str, Any], ctx)`, `identity_draw(self) -> Dict[str, Any]`.
- Produces (`ranges.py`):
  - `clip_rng(seed: int, clip_id: str, key: str) -> np.random.Generator` —
    `default_rng(SeedSequence([seed, zlib.crc32(clip_id.encode()), zlib.crc32(key.encode())]))`.
  - `@dataclass(frozen=True) RangeSpec(lo: float, hi: float, distribution: str)` with
    `draw(self, rng) -> float`.
  - `parse_range(value, default_distribution: str = "uniform") -> RangeSpec` — `value` is
    `[lo, hi]` or `{"range": [lo, hi], "distribution": "uniform"|"log_uniform"}`; `lo > hi`,
    non-positive bounds with `log_uniform`, or another shape → `ValueError`.
  - `truncated_normal(rng, sigma: float, limit: float, size=None) -> np.ndarray` — rejection
    resampling until `|x| <= limit`; `sigma == 0` returns zeros without consuming draws.

- [ ] **Step 1: Write the failing tests** in `tests/test_week4_base.py`:
  - `test_clip_rng_is_deterministic_and_independent`: same `(seed, clip, key)` → equal first 5
    draws; changing `key` → different draws.
  - `test_parse_range_forms`: `parse_range([0.8, 1.2]).distribution == "uniform"`;
    `parse_range({"range": [0.6, 1.65], "distribution": "log_uniform"})` draws stay in
    `[0.6, 1.65]` for 1000 draws; `parse_range([2, 1])`, `parse_range([0, 1]` with log_uniform),
    `parse_range(3.0)` raise `ValueError`.
  - `test_log_uniform_is_symmetric_in_log`: 20000 draws from `[0.6, 1/0.6]`, `abs(mean(log x)) < 0.01`.
  - `test_truncated_normal_respects_limit`: 10000 draws, σ=2.5, limit=5 → `max |x| <= 5`, std
    within 10% of the truncated-normal std (≈2.20); σ=0 → zeros.
  - `test_registry_refuses_clash`: registering a fresh class under `"JointCoupledNoise"` raises
    `ValueError`; re-registering the same class object does not.
  - `test_clip_augmentation_rejects_unknown_key` with a minimal subclass
    (`DEFAULT_RANGES={"a": 1.0}`): `ranges={"b": 2}` raises; `ranges={}` gives
    `defaulted == ["a"]`.
  - `test_list_by_region`: a subclass registered as `"face.dummy"` with `regions=("face",)`
    appears in `list_by_region("face")` and not in `list_by_region("body")`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_base.py -v` — expect import / attribute failures.
- [ ] **Step 3: Implement** the interfaces above. Leave `category` and the old classes untouched.
- [ ] **Step 4: Run** the new tests and `tests/test_tier1.py` — all pass.
- [ ] **Step 5: Commit** `feat: add augmentation regions, clip-augmentation base and range sampling`.

---

### Task 2: Clip container, rotation resampling and outputs with custom skinning

**Files:**
- Create: `src/kinetic_augment/augmentations/clip.py`, `src/kinetic_augment/augmentations/outputs.py`
- Create: `tests/clip_fixtures.py`
- Test: `tests/test_week4_clip.py`

**Interfaces:**
- Consumes: `tier1.WeakCamera` (`.resampled(times)`), `tier1.resample_points`,
  `face_stream.from_head_frame(points, rotation, position)`, `assembly.global_rotations`,
  `assembly.HEAD = 15`, `export_round_trip_demo.mesh_hand_points` logic (wrist = joint 20/21,
  `out[POSE_SLOT_LANDMARKS[k]] = joints[first + k]`, tips from `SMPLX_TIP_VERTICES`).
- Produces (`clip.py`):
  - `@dataclass(frozen=True) SceneTransform(mirror: bool = False, rotation: np.ndarray = I3,
    scale: float = 1.0, shift: np.ndarray = zeros(3), pivot: np.ndarray = zeros(3))` with
    `apply_points(points (..., 3)) -> (..., 3)`: reflect x about `pivot` if `mirror`, then
    `tier1.about(p, pivot, rotation * scale)`, then `+ shift`. `is_identity() -> bool`.
  - `@dataclass(frozen=True) ClipData` fields: `clip_id: str`, `rotations (T,55,3)`,
    `transl (T,3)`, `betas (300,)`, `rest_joints (55,3)`, `face_local (T,478,3)`,
    `face_valid (T,) bool`, `blendshapes (T,52)`, `blendshape_names: Tuple[str, ...]`,
    `body_ok (T,) bool`, `hand_valid (T,2) bool`, `times_s (T,)`, `fps: float`,
    `camera: Optional[WeakCamera]`, `scene: SceneTransform`. Methods: `frames -> int` (property),
    `replace(**changes) -> ClipData`.
  - `clip_from_result(clip_id: str, result: ClipFaceResult, hands: HandClip, face: FaceClip,
    skeleton: SMPLXSkeleton) -> ClipData` — `rest_joints = skeleton.shaped(result.betas).rest_joints`,
    `face_local = result.face_stream_local`, `face_valid = result.face_attachment_valid`,
    `blendshapes = face.blendshapes[result.frames]`,
    `hand_valid = result.hand_present & result.geometry_valid`,
    `times_s = (result.frames - result.frames[0]) / hands.fps`, `camera=None`, identity scene.
  - `slerp_resample(rotations (T,J,3), times (T',)) -> (T',J,3)` — per joint, scipy `Slerp`
    between the two neighbouring frames; a NaN neighbour takes the finite one; both NaN → NaN.
  - `resample_clip(clip, times) -> ClipData` — rotations by `slerp_resample`; `transl`,
    `blendshapes`, `face_local` by `tier1.resample_points`; boolean masks by nearest frame;
    `times_s` by `np.interp`; `camera.resampled(times)`.
  - `select_frames(clip, keep: np.ndarray) -> ClipData` — `keep` is a bool (T,) or int index
    array; every per-frame field and the camera offset are indexed.
- Produces (`outputs.py`):
  - `@dataclass(frozen=True) ClipOutputs`: `pose_world (T,33,3)` (only 11–16 finite),
    `hands_world (T,2,21,3)` (index 0 = left), `face_world (T,478,3)`, `head_world (T,3)`,
    `blendshapes (T,52)`, `blendshape_names`, `pose_image (T,33,2) | None`,
    `hands_image (T,2,21,2) | None`, `face_image (T,478,2) | None`, `hand_valid (T,2)`,
    `face_valid (T,)`, `body_ok (T,)`, `times_s (T,)`, `vertices: Optional[List[Optional[np.ndarray]]]`.
  - `pose_mesh_rest(skeleton, rotations (55,3), betas, transl (3,), rest_joints (55,3)) -> PosedMesh`
    — copy of `smplx.lbs.lbs` (shape blend from `skeleton.model.shapedirs` with `betas`, pose
    blend from `posedirs`, `batch_rigid_transform` with **the given** `rest_joints` instead of
    `J_regressor @ v_shaped`, `lbs_weights` skinning), expression zero, plus `transl`.
  - `build_outputs(clip: ClipData, skeleton: SMPLXSkeleton, keep_vertices: bool = False) -> ClipOutputs`
    — per `body_ok` frame: `pose_mesh_rest`; pose 11–16 from joints 16–21; hands from the mesh
    (left, right); face via `from_head_frame(face_local[t], global_rotations(...)[HEAD],
    joints[HEAD])` where `face_valid`; then `clip.scene.apply_points` on every point set. When
    `scene.mirror`, also: pose rows by `tier1.POSE_MIRROR`, hands swapped (and `hand_valid`
    columns), face rows by `face_mirror.FACE_MIRROR` (Task 9; until then import lazily and fail
    loudly if mirror is set), blendshape columns swapped `*Left`↔`*Right`. Image arrays through
    `clip.camera.project` when a camera exists, else `None`.

- [ ] **Step 1: Write `tests/clip_fixtures.py`**: `synthetic_clip(frames: int = 30, seed: int = 0) -> ClipData`
  (loads `SMPLXSkeleton("models")` once, zero rotations plus a 0–40° sinusoidal elbow bend on
  joints 18/19 and 20° MCP flexion on joints 25–54, `face_local` = 478 random points within
  ±8 cm, blendshapes uniform [0, 1], `fps=30`, `body_ok`/`face_valid`/`hand_valid` all true) and
  `real_clip() -> Tuple[ClipData, HandClip, FaceClip]` (`data/test_output/01991`; signing span =
  first to last frame where pose 13, 14, 15, 16 all have visibility ≥ 0.5; `run_clip`, cached in a module global; `pytest.skip` when
  the npz is absent).
- [ ] **Step 2: Write the failing tests** (`requires_smplx`) in `tests/test_week4_clip.py`:
  - `test_pose_mesh_rest_matches_pose_mesh`: for 3 random small poses, `pose_mesh_rest(...,
    skeleton.rest_joints)` vertices and joints equal `assembly.pose_mesh` within 1e-5 m.
  - `test_pose_mesh_rest_honours_scaled_forearm`: moving joint 20 and descendants 5 cm along
    the forearm in `rest_joints` changes the posed elbow→wrist distance by 5 cm ± 1 mm.
  - `test_slerp_resample_midpoint`: two frames, joint rotations 0° and 90° about z; time 0.5 →
    45° about z (within 1e-6); identity times → input exactly; NaN frame next to a finite one →
    the finite one; both NaN → NaN.
  - `test_resample_clip_identity_times_is_exact`: `np.arange(T)` → every field equal (NaN-equal).
  - `test_select_frames_shapes`: keep every other frame → every per-frame field has `ceil(T/2)` rows.
  - `test_build_outputs_identity_scene`: shapes as declared; pose rows other than 11–16 NaN;
    `hands_world[:,0,0]` equals the left wrist joint.
  - `test_build_outputs_unfitted_frames_stay_nan` (Review Focus 1): set frame 3 rotations to NaN
    and `body_ok[3]=False` → frame 3 all NaN, frames 2 and 4 finite.
  - `test_build_outputs_no_face` (Review Focus 4): `face_valid` all false → `face_world` all NaN, no error.
- [ ] **Step 3: Run** `.venv/bin/python -m pytest tests/test_week4_clip.py -v` — fails on imports.
- [ ] **Step 4: Implement** `clip.py` and `outputs.py`.
- [ ] **Step 5: Run** the tests — pass.
- [ ] **Step 6: Commit** `feat: add ClipData, slerp resampling and skinning that honours edited rest joints`.

---

### Task 3: Two-bone arm IK and the torso frame

**Files:**
- Create: `src/kinetic_augment/augmentations/arm_ik.py`
- Test: `tests/test_week4_arm_ik.py`

**Interfaces:**
- Consumes: `demo_smplx_augmentation`-style FK — reimplement locally as
  `world_pose(rotations (55,3), rest_joints, parents) -> (rot (55,3,3), pos (55,3))` (do not
  import from `scripts/`); `tier1.SMPLX_ARM`.
- Produces:
  - `world_pose(rotations, rest_joints, parents) -> Tuple[np.ndarray, np.ndarray]`
  - `two_bone_ik(rotations (55,3), rest_joints (55,3), parents, side: str, target (3,)) -> Tuple[np.ndarray, float]`
    returns new rotations and the residual distance (m) from the wrist to `target`.
  - `torso_frame(pos (55,3)) -> Tuple[np.ndarray, np.ndarray]` — origin = mid-shoulder
    (16, 17), axes `(3,3)` rows x = unit(right shoulder 17 − left 16), y = up = unit(neck 12 −
    spine3 9) orthogonalised against x, z = x × y.

Algorithm for `two_bone_ik` (the signature does not determine it):

```
S, E, W = shoulder, elbow, wrist world positions; a = |E-S|, b = |W-E| (from rest bones)
d = target - S; L = clip(|d|, |a-b| + 1e-6, a + b - 1e-6); u = d/|d|
pole = (E - S) minus its component along u; if |pole| < 1e-6 use the torso z axis; p = unit(pole)
cos_alpha = (a² + L² - b²) / (2aL); E' = S + a(cos_alpha·u + sin_alpha·p); W' = S + L·u
shoulder: world swing from unit(E-S) to unit(E'-S), applied with rotate_in_world semantics
elbow:    recompute FK, world swing from unit(W-E) to unit(W'-E')
wrist:    set the wrist's local rotation so its global rotation equals the original global
residual = |W' - target|
```

The hand keeps its world orientation; finger local rotations are untouched.

- [ ] **Step 1: Write the failing tests** (`requires_smplx`, real SMPL-X skeleton, rest pose
  with elbows bent 60°):
  - `test_ik_reaches_reachable_target`: target = wrist + (3, -2, 4) cm → residual < 1e-6;
    upper-arm and forearm lengths unchanged within 1e-9; wrist global rotation unchanged within 1e-9.
  - `test_ik_keeps_swivel_plane`: the elbow stays on the same side of the shoulder–target line
    (sign of the dot product with the original pole is positive).
  - `test_ik_unreachable_target_extends` (Review Focus 5): target 1 m beyond reach → residual ≈
    distance beyond full extension (± 1 mm), rotations all finite.
  - `test_ik_zero_move_is_identity`: target = current wrist → rotations equal within 1e-9.
  - `test_torso_frame_is_orthonormal`: `axes @ axes.T ≈ I`, determinant +1.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_arm_ik.py -v` — fails.
- [ ] **Step 3: Implement** `arm_ik.py`.
- [ ] **Step 4: Run** — pass.
- [ ] **Step 5: Commit** `feat: add analytic two-bone arm IK and torso frame`.

---

### Task 4: Body — joint coupled noise (arms) and the Tier 1 arm-joint wrapper

**Files:**
- Create: `src/kinetic_augment/augmentations/body.py`
- Test: `tests/test_week4_body_noise.py`

**Interfaces:**
- Consumes: `ClipAugmentation`, `truncated_normal`, `ClipData`, `tier1.ArmJointDraw`,
  `tier1.perturb_arm_joints(rotations, rest_joints, parents, joints)` (one frame).
- Produces:
  - `@AugmentationRegistry.register("body.joint_coupled_noise") class ArmCoupledNoise(ClipAugmentation)`,
    `key = "joint_coupled_noise"`, `regions = ("body",)`. `DEFAULT_RANGES` (spec §5.0 used values):
    `sigma_deg=2.5, truncate_deg=5.0, elbow_sigma_deg=4.75, elbow_truncate_deg=9.5,
    chain_coupling=0.5, side_coupling=0.7, side_mode="mirror", drift_deg=0.0`.
    `sample` returns `{"joints": [ArmJointDraw as dict, ...], "drift": Optional[array (T,6)]}`.
    `apply(clip, draw, ctx) -> ClipData`.
  - `@AugmentationRegistry.register("body.tier1_arm_joints") class Tier1ArmJoints(ClipAugmentation)`
    — `DEFAULT_RANGES = {"p": 0.3, "shoulder_wrist_deg": 5.0, "elbow_deg": 9.5}`; draws come from
    `ctx.tier1_draw.arm_joints` (Task 11), applied with `perturb_arm_joints`.

Sampling for `ArmCoupledNoise` (the coupling is the design's choice):

```
per side s, for shoulder and wrist: 3-vector v = sqrt(c)·common_s + sqrt(1-c)·indep_j, each ~ N(0, σ²I)
elbow: scalar e = sqrt(c)·common_s[0]·(σ_e/σ) + sqrt(1-c)·N(0, σ_e²)
right side: v_R = k·mirror(v_L) + sqrt(1-k²)·indep, mirror(x,y,z) = (x,-y,-z); "synchronous" uses v_L,
            "independent" uses k = 0
truncate: shrink any vector whose norm exceeds its limit to the limit; clip e to ±elbow limit
ArmJointDraw(angle_deg=|v|, axis=v/|v|) (angle 0 → skipped by perturb_arm_joints)
```

`drift_deg > 0` adds per-frame angles from Gaussian noise low-passed at 1 Hz (scipy
`butter(2, 1.0, fs=clip.fps)` + `filtfilt`), scaled to std `drift_deg`.

- [ ] **Step 1: Write the failing tests** (`requires_smplx`, `synthetic_clip`):
  - `test_identity_draw_is_exact`: `apply(clip, aug.identity_draw(), ctx)` rotations array-equal.
  - `test_draws_respect_limits`: 500 clip seeds → shoulder/wrist angles ≤ 5°, elbow ≤ 9.5°.
  - `test_mirror_coupling_correlates_sides`: over 2000 draws with `side_coupling=0.7`, the
    correlation of `v_L.x` with `v_R.x` is 0.7 ± 0.05 and of `v_L.y` with `−v_R.y` is 0.7 ± 0.05.
  - `test_constant_over_clip`: the local rotation delta of joint 16 is identical in every frame
    (drift 0).
  - `test_only_arm_joints_change`: joints other than 16–21 unchanged; bone lengths from FK unchanged.
  - `test_nan_frames_untouched` (Review Focus 1): NaN rotations at frame 5 stay NaN; frames 4 and 6 change.
  - `test_drift_is_smooth`: `drift_deg=1` → per-frame angle change between consecutive frames < 0.5°.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_body_noise.py -v` — fails.
- [ ] **Step 3: Implement** `ArmCoupledNoise` and `Tier1ArmJoints` in `body.py`.
- [ ] **Step 4: Run** — pass.
- [ ] **Step 5: Commit** `feat: add coupled arm-joint noise and the Tier 1 arm-joint wrapper`.

---

### Task 5: Body — limb length scaling

**Files:**
- Modify: `src/kinetic_augment/augmentations/body.py`
- Test: `tests/test_week4_limb_length.py`

**Interfaces:**
- Consumes: `two_bone_ik`, `world_pose` (Task 3), `ClipData.rest_joints`.
- Produces:
  - `scale_bone(rest_joints (55,3), parents, joint: int, factor: float) -> (55,3)` — moves
    `joint` and all its descendants by `(factor − 1) · (rest[joint] − rest[parent[joint]])`.
    For `bone="hand"` the caller scales every finger joint about the wrist instead.
  - `@register("body.limb_length_scaling") class LimbLengthScaling(ClipAugmentation)` —
    `DEFAULT_RANGES = {"factor": [0.93, 1.07], "bones": ["upper_arm", "forearm"], "retarget": True}`;
    bones: `upper_arm` → joints 18/19, `forearm` → 20/21, `hand` → fingers about 20/21. One factor
    per bone, shared by both sides. `sample` → `{"factors": {bone: float}}`.
    `apply` → `clip.replace(rest_joints=..., rotations=...)`; with `retarget`, each `body_ok`
    frame's wrist is IK'd to its world position under the original rest joints; the max residual
    goes into `draw["residual_m"]` (the chain records it).

- [ ] **Step 1: Write the failing tests** (`requires_smplx`):
  - `test_identity_draw_is_exact` (factors 1.0) — rotations and rest joints array-equal.
  - `test_bone_lengths_scaled`: factor 1.07 on forearm → rest forearm length × 1.07 (1e-9); upper arm unchanged.
  - `test_retarget_keeps_wrists`: with `retarget=True`, `build_outputs` wrist positions move < 1 mm
    on `synthetic_clip` (elbows bent, so reachable).
  - `test_no_retarget_moves_wrists`: `retarget=False` → wrists move > 5 mm at factor 1.07.
  - `test_unreachable_records_residual` (Review Focus 5): straight arm (bend 0°), factor 0.93 on
    both bones, retarget → finite rotations, `residual_m > 0`.
  - `test_draws_in_range`: 500 seeds, factors within [0.93, 1.07].
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_limb_length.py -v` — fails.
- [ ] **Step 3: Implement** `scale_bone` and `LimbLengthScaling`.
- [ ] **Step 4: Run** — pass.
- [ ] **Step 5: Commit** `feat: add limb length scaling with wrist retargeting`.

---

### Task 6: Body — movement-size scaling and location jitter

**Files:**
- Modify: `src/kinetic_augment/augmentations/body.py`
- Test: `tests/test_week4_location.py`

**Interfaces:**
- Consumes: `two_bone_ik`, `world_pose`, `torso_frame` (Task 3).
- Produces:
  - `wrist_paths_torso(clip) -> Tuple[np.ndarray (T,2,3), np.ndarray (T,3), np.ndarray (T,3,3)]`
    — each fitted frame's wrists in its own torso frame, plus the frame origins and axes (NaN rows
    where `body_ok` is false).
  - `retarget_wrists(clip, targets_world (T,2,3)) -> Tuple[ClipData, float]` — IK per fitted frame
    and side; returns the max residual.
  - `@register("body.movement_size") class MovementSizeScaling` — `DEFAULT_RANGES =
    {"factor": [0.8, 1.2]}`, `DISTRIBUTIONS = {"factor": "log_uniform"}`; scales each wrist's
    torso-frame path about its own centroid (over fitted frames) by one factor shared by both hands.
  - `@register("body.location_jitter") class LocationJitter` — `DEFAULT_RANGES =
    {"sigma_mm": 12.5, "truncate_mm": 25.0}`; one 3-vector offset (per-axis truncated normal) in the
    torso frame, added to both wrists in every frame.
  - Both record `start_end_displacement_mm` (first/last fitted frame wrist moves) in the draw.

- [ ] **Step 1: Write the failing tests** (`requires_smplx`, `synthetic_clip`):
  - `test_movement_identity_is_exact` and `test_location_identity_is_exact`.
  - `test_movement_scales_path_extent`: factor 1.2 → the torso-frame wrist path's RMS distance
    from its centroid grows by 1.2 ± 0.01 (targets reachable).
  - `test_location_offset_constant`: torso-frame wrist displacement equal in every fitted frame
    (± 0.5 mm) and ≤ 25 mm per axis.
  - `test_hand_distance_kept_by_location`: left–right wrist distance unchanged ± 0.5 mm.
  - `test_unfitted_frames_untouched` (Review Focus 1).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_location.py -v` — fails.
- [ ] **Step 3: Implement** the two augmentations and helpers.
- [ ] **Step 4: Run** — pass.
- [ ] **Step 5: Commit** `feat: add movement-size scaling and location jitter`.

---

### Task 7: Fingers — handshape features, finger coupled noise, handshape jitter

**Files:**
- Create: `src/kinetic_augment/augmentations/handshape.py`, `src/kinetic_augment/augmentations/fingers.py`
- Test: `tests/test_week4_fingers.py`

**Interfaces:**
- Consumes: `world_pose` (Task 3), SMPL-X hand layout (left 25–39, right 40–54; per hand
  offsets: index 0–2, middle 3–5, pinky 6–8, ring 9–11, thumb 12–14; MCP→PIP→DIP).
- Produces (`handshape.py`):
  - `FINGERS = ("thumb", "index", "middle", "ring", "pinky")`, `FINGER_SLOTS: Dict[str, Tuple[int,int,int]]`.
  - `@dataclass HandshapeFeatures(bend_deg (T,5), state (T,5) int 0=extended 1=curved 2=flexed,
    spread_deg (T,), thumb_abduction_deg (T,))`.
  - `handshape_features(rotations (T,55,3), rest_joints, parents, side) -> HandshapeFeatures` —
    bend = sum of the angles between consecutive bones wrist→MCP→PIP→DIP; thresholds
    `EXTENDED_MAX_DEG = 40`, `FLEXED_MIN_DEG = 110`; spread = angle between the index and pinky
    proximal bones; thumb abduction = angle between the thumb proximal bone and the index
    metacarpal. Document the thresholds in the module docstring; the builder may tune them on
    the cached clips and must say so in the docstring.
  - `selected_fingers(features) -> Tuple[bool, ...]` (5) — fingers whose median state is
    extended or curved; if none, the index finger.
  - `features_changed(before, after, margin_deg: float = 5.0) -> bool` — compares state, spread
    class (threshold 25°) and thumb class (threshold 45°); frames where `before` is within
    `margin_deg` of a threshold are skipped; a change counts only when `after` crosses by more
    than `margin_deg`.
  - `flexion_axes(rest_joints, side) -> (15,3)` — per finger joint, unit(bone × palm normal) in
    the parent frame at rest (palm normal from the wrist, index MCP and pinky MCP).
- Produces (`fingers.py`):
  - `@register("fingers.joint_coupled_noise") class FingerCoupledNoise` — `regions=("fingers",)`,
    `DEFAULT_RANGES = {"sigma_deg": 3.75, "truncate_deg": 7.5, "finger_coupling": 0.5,
    "dip_ratio": 0.667}`; flexion offsets on MCP and PIP about `flexion_axes`, DIP = `dip_ratio`·PIP;
    neighbour coupling along index→middle→ring→pinky as `x_f = c·x_{f−1} + sqrt(1−c²)·z_f`;
    thumb independent; the same offsets on both hands.
  - `@register("fingers.handshape_jitter") class HandshapeJitter` — `DEFAULT_RANGES =
    {"nonselected_sigma_deg": 7.5, "selected_sigma_deg": 3.75, "selected_flexion_sigma_deg": 2.0,
    "truncate_sigmas": 2.0, "max_redraws": 5, "feature_margin_deg": 5.0}`. Selected fingers get
    `selected_flexion_sigma_deg` about the flexion axis and `selected_sigma_deg` on the other two
    axes; non-selected fingers and the thumb get `nonselected_sigma_deg` on all three. Same
    offsets on both hands when their median feature states match (symmetric). Redraw while
    `features_changed`; after `max_redraws` return the identity draw with
    `draw["fallback"] = "no_valid_redraw"`. Hands with `hand_valid` false in every frame are skipped.

- [ ] **Step 1: Write the failing tests** (`requires_smplx`):
  - `test_features_flat_hand_extended`: zero finger rotations → all states 0.
  - `test_features_fist_flexed`: 90° about the flexion axis at each MCP/PIP → fingers 1–4 state 2.
  - `test_features_changed_margin`: a bend 2° below `FLEXED_MIN_DEG` moved 4° above → not changed;
    moved 8° above from 20° below → changed.
  - `test_finger_noise_identity_exact`; `test_finger_noise_dip_follows_pip` (DIP angle = 0.667 × PIP ± 1e-9);
    `test_finger_noise_limits` (500 seeds, ≤ 7.5°).
  - `test_handshape_jitter_preserves_features`: on `synthetic_clip` (20° MCP flexion) 50 seeds →
    `features_changed` false for every non-fallback draw.
  - `test_handshape_jitter_fallback`: a hand posed exactly on a threshold with margin 0 and σ huge
    → `fallback == "no_valid_redraw"`, rotations unchanged.
  - `test_missing_hand_skipped` (Review Focus 2): `hand_valid[:,1] = False` → right-hand rotations unchanged.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_fingers.py -v` — fails.
- [ ] **Step 3: Implement** `handshape.py`, then `fingers.py`.
- [ ] **Step 4: Run** — pass. Then run `handshape_features` on the 20 cached clips (via
  `real_clip`-style loading in a scratch script) and confirm fewer than 20% of handshape-jitter
  draws fall back; adjust thresholds if not and note it in the docstring.
- [ ] **Step 5: Commit** `feat: add handshape features, coupled finger noise and handshape jitter`.

---

### Task 8: Face — expression jitter

**Files:**
- Create: `src/kinetic_augment/augmentations/face.py`
- Test: `tests/test_week4_face.py`

**Interfaces:**
- Consumes: `ClipData.face_local`, `blendshapes`, `blendshape_names`,
  `face_correspondence.RIGID`.
- Produces:
  - `BLENDSHAPE_GROUPS: Dict[str, Callable[[str], bool]]` — `brows`: startswith `"brow"`;
    `eyes`: startswith `"eye"`; `cheeks_nose`: startswith `"cheekSquint"` or `"noseSneer"`;
    `mouth_jaw`: startswith `"jaw"` or `"mouth"`, or in `("cheekPuff", "tongueOut")`.
    `"_neutral"` is in no group.
  - `fit_expression_map(face_local (T,478,3), blendshapes (T,52), valid (T,), alpha: float = 1e-2)
    -> Tuple[np.ndarray (51, 478*3), np.ndarray (478,3)]` — ridge regression from the 51
    non-neutral scores (centred) to the flattened offsets from the median face; RIGID columns zeroed.
  - `@register("face.expression_jitter") class ExpressionJitter` — `DEFAULT_RANGES =
    {"brows": {"enabled": True, "sigma": 0.0625, "truncate": 0.125}, "eyes": {... same},
    "cheeks_nose": {"enabled": True, "sigma": 0.0375, "truncate": 0.075},
    "mouth_jaw": {"enabled": False, "sigma": 0.0375, "truncate": 0.075},
    "frame_noise": 0.0, "min_face_frames": 30, "ridge_alpha": 0.01}`. One offset per score;
    `frame_noise > 0` adds per-frame Gaussian noise of that std to the enabled scores, low-passed
    at 2 Hz like the arm drift;
    scores clipped to [0, 1]; `Δpoints = (new − old scores) @ W` added to `face_local` on valid
    frames. Fewer than `min_face_frames` valid frames → scores only, `draw["fallback"] =
    "too_few_face_frames"`; zero valid frames → nothing changes, same fallback.

- [ ] **Step 1: Write the failing tests** (no assets needed except the names; use the real
  52-name tuple from the spec of `FaceClip`):
  - `test_groups_partition_names`: each of the 51 non-neutral names is in exactly one group.
  - `test_identity_exact`.
  - `test_mouth_off_by_default`: default draw leaves every `mouth_jaw` column unchanged.
  - `test_scores_stay_in_unit_interval` over 200 seeds.
  - `test_rigid_points_fixed`: RIGID rows of `face_local` unchanged.
  - `test_map_recovers_synthetic_basis`: build `face_local = median + B @ W_true` from random
    scores; fitted `W` predicts held-out offsets within 5% relative error.
  - `test_no_face_fallback` (Review Focus 4): `face_valid` all false → fallback recorded,
    `face_local` and blendshapes unchanged.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_face.py -v` — fails.
- [ ] **Step 3: Implement** `face.py`.
- [ ] **Step 4: Run** — pass.
- [ ] **Step 5: Commit** `feat: add facial expression jitter driven by a per-clip blendshape map`.

---

### Task 9: Scene — face mirror table, mirror, view, scale, uncentre, camera fit

**Files:**
- Create: `scripts/make_face_mirror.py`, `src/kinetic_augment/augmentations/face_mirror.py`,
  `src/kinetic_augment/augmentations/scene.py`
- Test: `tests/test_week4_scene.py`
- Modify: `scripts/index.md`

**Interfaces:**
- Consumes: `tier1.view_rotation`, `tier1.fit_weak_camera`, `tier1.uncentre_shift`,
  `tier1.Tier1Draw`, `SceneTransform`, `build_outputs`.
- Produces:
  - `face_mirror.FACE_MIRROR: Tuple[int, ...]` (478) — generated by `scripts/make_face_mirror.py`
    from `models/mediapipe/canonical_face_model.obj`: reflect x, nearest neighbour (Hungarian
    assignment via `scipy.optimize.linear_sum_assignment` so it is a permutation), symmetrised.
    Iris points 468–477 pair 468–472 ↔ 473–477 by the same rule.
  - `fit_clip_camera(clip, hands: HandClip, skeleton) -> WeakCamera` — like
    `export_round_trip_demo.fit_camera` (:759): arms 11–16 plus 42 hand points from
    `build_outputs`, against `hands.pose_image`/`hands.image` (frames `result.frames`), with
    `pixel_size=(hands.width, hands.height)`.
  - Scene augmentations (`regions=("scene",)`, all read their draw from `ctx.tier1_draw`):
    `@register("scene.mirror") Mirror` (`{"p": 0.5}`), `@register("scene.view") View`
    (`{"yaw_deg": 25.0, "pitch_deg": 6.0}`), `@register("scene.scale") Scale`
    (`{"factor": [0.75, 1.25]}`), `@register("scene.uncentre") Uncentre` (`{"limit": 0.125}`).
    Each `apply` composes into `clip.scene` (pivot = `ctx.pivot`). `Uncentre.apply` builds the
    current outputs, projects both hands' 21 points plus pose wrists 15 and 16, and sets the
    shift from `tier1.uncentre_shift(uv, draw.uncentre_u, limit)` via `camera.world_shift`.
  - `blendshape_mirror(names) -> np.ndarray` — column permutation swapping `*Left`↔`*Right`.

- [ ] **Step 1: Write `scripts/make_face_mirror.py`**, run it, commit the generated
  `face_mirror.py` (a header comment says it is generated and how).
- [ ] **Step 2: Write the failing tests**:
  - `test_face_mirror_is_involution`: `FACE_MIRROR[FACE_MIRROR[i]] == i` for all i;
    `FACE_MIRROR[33] == 263`, `FACE_MIRROR[133] == 362`, `FACE_MIRROR[1] == 1`.
  - `test_blendshape_mirror`: `browDownLeft` ↔ `browDownRight`; `jawOpen` fixed; involution.
  - (`requires_smplx`) `test_mirror_swaps_hands`: mirrored outputs' left hand equals the reflected
    original right hand; `hand_valid` columns swap (Review Focus 2).
  - (`requires_smplx`) `test_scene_identity_exact`: identity draws → outputs array-equal.
  - (`requires_smplx`) `test_view_keeps_relative_positions`: wrist-to-wrist distance unchanged (1e-9).
  - (real clip) `test_uncentre_keeps_hands_in_frame`: limit 0.125, 20 seeds → `tier1.frame_loss == 0`.
- [ ] **Step 3: Run** `.venv/bin/python -m pytest tests/test_week4_scene.py -v` — fails.
- [ ] **Step 4: Implement** `scene.py`; wire `FACE_MIRROR` into `build_outputs` (Task 2).
- [ ] **Step 5: Run** the tests and `tests/test_week4_clip.py` — pass.
- [ ] **Step 6: Commit** `feat: add scene augmentations on clips and a 478-point face mirror table`.

---

### Task 10: Temporal — time warping, frame dropping, hand dropout

**Files:**
- Create: `src/kinetic_augment/augmentations/timing.py`, `scripts/measure_hand_dropout.py`
- Test: `tests/test_week4_timing.py`
- Modify: `scripts/index.md`

**Interfaces:**
- Consumes: `resample_clip`, `select_frames`, `tier1.duration_times`, `ClipOutputs`.
- Produces:
  - `times_from_rates(durations (T−1,)) -> np.ndarray` — `durations[i]` = output length of source
    interval i; cumulative sum gives each source frame's output time; return source times at
    output frames `0..round(total)` by `np.interp`; at least 2 frames; endpoints kept.
  - `varied_durations(count, knot_rates, protect_end, total_factor) -> (count−1,)` — knot rates
    interpolated (cubic, monotone via `PchipInterpolator`) over the first `1 − protect_end`, a
    constant rate over the rest, then normalised so the sum = `total_factor · (count − 1)`.
  - `segment_durations(count, start_frac, width_frac, factor, uniform, blend_frames=3) -> (count−1,)`
    — `uniform` everywhere, `uniform · factor` inside the window, linear blend over
    `blend_frames` at each edge.
  - `@register("temporal.time_warp") TimeWarp` — `DEFAULT_RANGES = {"duration": [0.6, 1.65],
    "mode_weights": {"none": 1, "varied": 0, "segment": 0}, "num_knots": 4, "knot_rate":
    [0.7, 1.43], "protect_end": 0.2, "segment_width": [0.2, 0.6], "segment_factor": [0.6, 1.65]}`;
    `DISTRIBUTIONS`: duration, knot_rate, segment_factor log-uniform. The uniform duration comes
    from `ctx.tier1_draw.duration`.
  - `@register("temporal.frame_drop") FrameDrop` — `DEFAULT_RANGES = {"mode_weights": {"rate": 0,
    "random": 1}, "target_fps": [12, 25], "min_fps": 8, "p": [0.0, 0.2], "max_gap_ms": 125,
    "fill": "none"}`. `rate` keeps frames nearest to a grid at `max(target, min_fps)` (capped at
    the source fps); `random` drops each frame with p unless that would make a gap over
    `max_gap_ms`; first and last frames always kept. `fill`: `none` → `select_frames`; `hold` →
    each dropped frame copies the previous kept one; `interpolate` → `resample_clip` of the kept
    frames back onto the original times.
  - `@register("temporal.hand_dropout") HandDropout` — `stage = "outputs"`, `DEFAULT_RANGES =
    {"start_rate": [0.0, 0.04], "run_length_mean": 1.9, "max_run_frames": 10}`; per hand, runs
    start at detected frames with probability `start_rate`, lengths geometric with the mean,
    truncated at `max_run_frames`; within a run, that hand's `hands_world`/`hands_image` rows
    become NaN and `hand_valid` false. `apply(outputs, draw, ctx) -> ClipOutputs`.
  - `scripts/measure_hand_dropout.py` — prints, over `data/round_trip_eval/*_hands.npz`, the
    interior run count, median, p90, max and run starts per detected hand-frame.

- [ ] **Step 1: Write the failing tests** (`synthetic_clip`, `requires_smplx`):
  - `test_times_from_rates_uniform`: all durations 1.5 on 21 frames → 31 output frames, times
    `linspace(0, 20, 31)`.
  - `test_uniform_matches_tier1`: `mode none`, duration 1.25 → times equal `tier1.duration_times(T, 1.25)`.
  - `test_varied_monotonic_and_total`: 200 seeds → strictly increasing times, output length =
    `round((T−1)·duration)+1`, last `protect_end` portion has constant step (± 1e-9).
  - `test_segment_only_changes_window`: outside the window, the step equals `1/uniform` (± 1e-9).
  - `test_short_clip` (Review Focus 3): T = 2, 3, 4 with every mode and every frame-drop mode →
    ≥ 2 frames, no exception.
  - `test_frame_drop_gap_limit`: 500 seeds at 30 fps → no gap between kept timestamps over 125 ms;
    first/last kept.
  - `test_frame_drop_fill_modes`: `hold` and `interpolate` keep T frames; `none` has fewer.
  - `test_hand_dropout_runs`: runs ≤ 10 frames; only previously detected frames masked; a hand
    with `hand_valid` all false is untouched (Review Focus 2); identity draw exact.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_timing.py -v` — fails.
- [ ] **Step 3: Implement** `timing.py` and `measure_hand_dropout.py`.
- [ ] **Step 4: Run** the tests — pass. Run `.venv/bin/python scripts/measure_hand_dropout.py` —
  expect 41 interior runs, median 1, max 6, start rate ≈ 0.032.
- [ ] **Step 5: Commit** `feat: add time warping modes, frame dropping and hand-dropout runs`.

---

### Task 11: Profiles, the chain, range docs

**Files:**
- Create: `src/kinetic_augment/augmentations/profile.py`, `src/kinetic_augment/augmentations/chain.py`
- Create: `configs/slr_week4_default.yaml`, `configs/slr_week4_ambitious.yaml`, `configs/slr_week4_literature.yaml`
- Create: `docs/Augmentation_Ranges.md`
- Modify: `README.md` (insert "## Augmentation ranges" before "## Documentation", line 52),
  `docs/index.md`, `docs/Augmentation_Grounding.md` (§3 intro: one line linking the ranges page),
  `configs/index.md`, `src/kinetic_augment/augmentations/index.md`, `src/kinetic_augment/augmentations/__init__.py`
- Test: `tests/test_week4_profile.py`

**Interfaces:**
- Consumes: every registered `"<region>.<key>"` augmentation; `tier1.sample_tier1`,
  `tier1.Tier1Ranges`; `build_outputs`.
- Produces (`profile.py`):
  - `@dataclass ProfileEntry(region: str, key: str, enabled: bool, probability: float, ranges: Dict)`
  - `@dataclass Profile(name: str, seed: int, schema_version: int, entries: List[ProfileEntry])`
  - `load_profile(path, enable: Sequence[str] = (), disable: Sequence[str] = ()) -> Profile` —
    `enable`/`disable` items are `"region.key"`. Errors (`ValueError`): schema_version ≠ 1, unknown
    region, unknown key in a region, an override naming a missing entry, both
    `body.joint_coupled_noise` and `body.tier1_arm_joints` enabled. Prints each defaulted range key.
  - `tier1_ranges(profile) -> tier1.Tier1Ranges` — from `scene.*`, `temporal.time_warp.duration`
    and `body.tier1_arm_joints`.
- Produces (`chain.py`):
  - `ORDER: Tuple[str, ...] = ("fingers.joint_coupled_noise", "fingers.handshape_jitter",
    "body.tier1_arm_joints", "body.joint_coupled_noise", "body.limb_length_scaling",
    "body.movement_size", "body.location_jitter", "face.expression_jitter", "scene.mirror",
    "scene.view", "scene.scale", "scene.uncentre", "temporal.time_warp", "temporal.frame_drop",
    "temporal.hand_dropout")`
  - `@dataclass ChainContext(skeleton, parents, pivot (3,), tier1_draw: Tier1Draw, seed: int)`
  - `@dataclass ChainResult(clip: ClipData, outputs: ClipOutputs, draws: Dict[str, Dict],
    applied: List[str], fallbacks: Dict[str, str])`
  - `class AugmentationChain: __init__(self, profile: Profile, skeleton)`;
    `run(self, clip: ClipData) -> ChainResult` — `tier1_draw = sample_tier1(default_rng([seed,
    int(clip_id) if clip_id.isdigit() else zlib.crc32(clip_id.encode())]), tier1_ranges(profile))`;
    pivot = median mid-shoulder of `build_outputs(clip)`; for each enabled entry in `ORDER`:
    `rng = clip_rng(seed, clip_id, "region.key")`, apply with its probability; scene entries only
    act when their tier1 value is non-identity (mirror only if drawn); then `build_outputs`, then
    `stage == "outputs"` entries.
- YAML: the three profiles follow spec §7. Every range key has `# proven: ... (source)` or
  `# proven: none — <reason>`. `default`: scene.*, temporal.time_warp (`mode_weights` none only),
  body.joint_coupled_noise, temporal.frame_drop (`random` only) enabled; everything else present
  with `enabled: false`. `ambitious`: all enabled, `time_warp.mode_weights {none: 1, varied: 1,
  segment: 1}`, `frame_drop.mode_weights {rate: 1, random: 1}`, `face.expression_jitter` with
  `mouth_jaw.enabled: false`. `literature`: the default profile's entries at the proven ranges
  (yaw 20, pitch 5, duration [0.67, 1.5], scale [0.8, 1.2], uncentre 0.1, arm σ 2/limit 4,
  elbow σ 3.75/limit 7.5, frame_drop min_fps 10 / max_gap_ms 100); entries with no proven range
  disabled.
- Docs: `docs/Augmentation_Ranges.md` has the spec §5.0 table, the 25% rule, sources and notes;
  README section has the same table, a one-line explanation, and a link to the page.

- [ ] **Step 1: Write the failing tests** (`tests/test_week4_profile.py`):
  - `test_shipped_profiles_load`: all three load without error.
  - `test_unknown_key_fails`: a temp YAML with `body.joint_coupled_noise.ranges.bogus` → `ValueError`.
  - `test_override_enable`: default + `enable=["face.expression_jitter"]` → that entry enabled.
  - `test_both_arm_noises_rejected`.
  - `test_default_ranges_match_spec_table`: default profile values equal the §5.0 used values
    (yaw 25, pitch 6, duration [0.6, 1.65], scale [0.75, 1.25], uncentre 0.125, location σ 12.5 /
    25, limb [0.93, 1.07], movement [0.8, 1.2], min_fps 8, max_gap_ms 125, max_run_frames 10).
  - `test_every_range_has_proven_comment`: each line under a `ranges:` block in each shipped YAML
    that sets a value has `# proven:`.
  - (`requires_smplx`) `test_chain_disabled_is_identity`: a profile with everything disabled →
    outputs array-equal to `build_outputs(clip)`.
  - (`requires_smplx`) `test_chain_is_deterministic`: two runs of the ambitious profile on
    `synthetic_clip` → equal draws and outputs.
  - (`requires_smplx`) `test_adding_entry_keeps_other_draws`: default vs default+`body.location_jitter`
    → the `body.joint_coupled_noise` draw is equal.
  - `test_tier1_stream_unchanged`: with `tier1_ranges` set to `TIER1` values, the chain's
    `tier1_draw` equals `sample_tier1(default_rng([seed, int(clip)]))`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_profile.py -v` — fails.
- [ ] **Step 3: Implement** `profile.py`, `chain.py`, the three YAMLs, then the docs and
  index updates; export `AugmentationChain`, `load_profile`, `ClipData`, `build_outputs` from
  `augmentations/__init__.py`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests -q` — all pass (or skip on missing assets).
- [ ] **Step 5: Commit** `feat: add week 4 profiles, augmentation chain and range documentation`.

---

### Task 12: Visual trial and per-augmentation report

**Files:**
- Create: `src/kinetic_augment/augmentations/report.py`, `scripts/augment_week4_trials.py`
- Test: `tests/test_week4_report.py`
- Modify: `scripts/index.md`, `src/kinetic_augment/augmentations/index.md`, `data/augment_trials/index.md`

**Interfaces:**
- Consumes: `AugmentationChain`, `load_profile`, `clip_from_result`, `fit_clip_camera`,
  `run_clip`, `augment_face_trials.signing_span`, `view_round_trip.VideoSink`.
- Produces:
  - `measure(base: ClipOutputs, aug: ClipOutputs, aligned: bool) -> Dict[str, float]` — keys
    `wrist_moved_mm_max`, `bone_length_change_mm_max` (11–13, 13–15, 12–14, 14–16),
    `jitter_ratio` (mean |second difference| of the wrists, aug / base), `face_rigidity_mm`
    (max change of face-point-to-head distances), `hand_face_cm_min`, `in_frame_pct` (when
    image arrays exist). With `aligned=False` (temporal), only the frame-free keys.
  - `scripts/augment_week4_trials.py` — `--profile configs/slr_week4_ambitious.yaml`,
    `--clips` (default `57949 12328 63669 63229`), `--landmark-dir data/round_trip_eval`,
    `--output-dir data/augment_trials`, `--enable`/`--disable region.key` (passed to
    `load_profile`). For each clip: load, `run_clip` on the signing span,
    `clip_from_result`, camera fit; then one run per enabled entry with only that entry enabled
    (probability forced to 1), plus one run of the whole profile. Writes
    `<clip>_week4_augmentations.mp4` (one panel per run: front-view 2-D projection of arms, hands
    and face points over the base in grey) and `augment_week4_trials.json` (draws, fallbacks,
    `measure` per run).

- [ ] **Step 1: Write the failing tests**: `test_measure_identity_zero` (same outputs → zeros and
  `jitter_ratio == 1`); `test_measure_detects_wrist_move` (shift left hand 1 cm →
  `wrist_moved_mm_max == 10 ± 1e-6`); `test_measure_unaligned_skips_frame_keys`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_week4_report.py -v` — fails.
- [ ] **Step 3: Implement** `report.py` and the trial script.
- [ ] **Step 4: Run** the tests — pass. Run
  `.venv/bin/python scripts/augment_week4_trials.py --clips 57949` — writes the MP4 and JSON; in
  the JSON, `body.limb_length_scaling` with retarget has `wrist_moved_mm_max < 2`, scene runs have
  `bone_length_change_mm_max < 0.01`, and no run has a Python error.
- [ ] **Step 5: Commit** `feat: add week 4 visual trial and per-augmentation report`.
