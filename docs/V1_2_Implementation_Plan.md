# V1.2 Hand Round Trip — Feasibility Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit genuine MANO and SMPL-X hand parameters to MediaPipe's 21 hand world landmarks, rebuild the landmarks from those parameters alone, and measure the round-trip error on the five pilot clips.

**Architecture:** `HandModel` evaluates exactly the 21 MediaPipe points of the real model (16 skeleton joints + 5 fingertip vertices) through `smplx.lbs`, for MANO or SMPL-X, either side. Fitting starts analytically, the way the V1 body converter does (palm rotation by Kabsch, finger bones by swing IK), then refines all rotations with L-BFGS through the real forward pass; one shape per hand track is fitted inside the prior's plausible region, and a per-frame-shape diagnostic measures how much error MediaPipe's frame-to-frame hand-size changes cause. A paired extractor records MediaPipe hand world landmarks with a geometric left/right assignment, and a measurement script reports an error ladder that separates optimiser error from model mismatch.

**Tech Stack:** Python ≥ 3.9, NumPy, SciPy, PyTorch (CPU, float64), smplx 0.1.28, MediaPipe 0.10.21 legacy solutions, OpenCV, pytest.

**Source documents:** `docs/V1_2_Hand_Roundtrip_Plan.md` (the research plan: P0–P6), `docs/V1_2_Assumptions.md` (assumption ledger A001–A019) and `docs/V1_2_Feasibility_Probes.md` (measurements taken before this plan). This plan implements the feasibility slice P0–P2 plus the P3 pilot measurement. Out of scope here: SMPL-X wrist attachment to the V1 arm (P4 — needs the V1 converter, which is not on this branch), temporal regularisation and a sealed final test set (P5), the image bridge (P6).

## Feasibility, measured before planning

From `docs/V1_2_Feasibility_Probes.md`:

- The MANO npz assets load in `smplx` 0.1.28; the landmark mapping and tip vertices are verified on both sides; MANO's left hand is an exact mirror of its right. SMPL-X's hands are nearly, not exactly, mirrored (5.9 mm).
- Only legacy `mp.solutions.hands.Hands` gives metric hand world landmarks. Its handedness label names the opposite hand on this unmirrored video; side is assigned from the pose wrists instead. Hand-world and pose-world axes agree.
- MediaPipe's hands are not rigid: bone lengths vary about 16% within a clip and hand size about ±30%. A model-free rigid-hand estimate reproduces them to about 15 mm (wrist-relative mean). So the provisional real-data target (5 mm mean) may be out of reach for a hand with one shape per track. That is a result to measure and report, not a reason to change the metric.

## As built: deviations made in review

The code below is the plan as written. Task reviews changed it in these places; the committed code, not this text, is authoritative:

- Every hand test module begins with `pytest.importorskip("torch")` / `("smplx")`. The Task 2 review found an unguarded import that aborted the whole suite without torch, and the Global Constraint governs over the brief's code.
- `load_hand_fits` and `load_hand_clip` read each column once inside `with np.load(path) as data:`. The Task 3 review found quadratic time and memory, plus an open file handle.
- The clip-independence test extracts 01992, then 01991, and compares 01991 with a lone run. The Task 5 review showed that the original order could not detect a `Hands` detector shared between clips.
- The measurement script reports two held-out rows, `held-out shape (time halves)` and `held-out shape (interleaved frames)`, and records `half_span_ratio` for each track. Observability is summarised as `weak_counts` and `joint_weak_weight`. The Task 6 review found that the time-halves row alone is confounded by MediaPipe's drift in hand size, and that the "weakest joint" label was arbitrary.
- From Task 3's fix round on, implementers' commits carry their own model's co-author trailer.

## Global Constraints

- Work only inside the checkout you are given (a git worktree). Run Python from its root as
  `PYTHONPATH=src /Users/bemnetbeshah/Workspace/cv-research/KineticAugment/.venv/bin/python …`
  — the shared virtualenv's editable install points at the main checkout; `PYTHONPATH=src` makes this checkout's code win. Run pytest the same way: `PYTHONPATH=src <python> -m pytest …`.
- Python ≥ 3.9 syntax: `typing.Optional/List/Tuple/Sequence`, no `X | Y` annotations, no `match`.
- Black-compatible formatting with 100-character lines; `flake8 --max-line-length 100` clean on every Python file you touch.
- Library code under `src/kinetic_augment/body_model/`, tests under `tests/`, runnable scripts under `scripts/`.
- No new dependencies. numpy, scipy, torch, smplx 0.1.28, mediapipe 0.10.21 and opencv are installed.
- Never commit licensed model weights. `models/**/*.npz` and `*.pkl` are git-ignored: never `git add -f` them, never copy them into tracked paths. Commit with explicit paths (`git add <file> …`); never `git add -A` or `git add .`.
- Update the `index.md` of every folder where you add a file: one line per file, matching the folder's existing bullet style.
- Tests that need torch/smplx, model assets or mediapipe must skip cleanly when those are missing — and here they must run, not skip. A skipped asset-backed test is not a pass.
- Units: metres in every array and API. Millimetres only in reported metrics and inside fitting losses.
- MediaPipe hand landmark order: 0 wrist; 1–4 thumb CMC, MCP, IP, tip; 5–8 index MCP, PIP, DIP, tip; 9–12 middle; 13–16 ring; 17–20 pinky.
- Model pose convention: `use_pca=False`, `flat_hand_mean=True`, 45 axis-angle values per hand; slot order index, middle, pinky, ring, thumb (three joints each, proximal first).
- Sides: index 0 = the signer's anatomical left, 1 = right; names `"left"` / `"right"`.
- The decoder (`reconstruct_hand`) takes fitted parameters and the model only — never observed landmarks, cached arrays or residuals.
- One MediaPipe detector instance per clip; tracking state never carries between clips.
- Fitting runs in float64.
- Synthetic gates (mean ≤ 0.5 mm, p95 ≤ 1 mm on model-generated hands) are pass/fail acceptance criteria: never loosen them. Real-data targets (mean ≤ 5 mm, tips ≤ 8 mm, p95 ≤ 15 mm) are provisional and reported as measured: never tune toward them by dropping frames, rescaling predictions or changing the metric.
- Hand world landmarks come only from legacy `mp.solutions.hands.Hands`; hand-world axes equal pose-world axes, so no basis change is applied.

## File structure

| File | Responsibility |
| --- | --- |
| `src/kinetic_augment/body_model/hand_model.py` (new) | Landmark topology and the 21-point observation operator `HandModel` over MANO / SMPL-X. |
| `src/kinetic_augment/body_model/hand_converter.py` (new) | Analytic start, refinement, `HandFit`, parameter-only decoder, serialisation, shape fit, the primary error metric. |
| `src/kinetic_augment/body_model/hand_observations.py` (new) | Hand clip record (save/load) and geometric left/right assignment. |
| `scripts/extract_hand_landmarks.py` (new) | Runs MediaPipe Hands + Holistic per clip; writes `data/test_output/<clip>_hands.npz`. |
| `scripts/measure_hand_round_trip.py` (new) | Error ladder on the pilot clips; writes `data/test_output/hand_round_trip.json`. |
| `tests/hand_assets.py` (new) | Shared asset paths, skip marks and a cached `HandModel` loader for the hand tests. |
| `tests/test_hand_model.py`, `tests/test_hand_converter.py`, `tests/test_hand_observations.py`, `tests/test_extract_hand_landmarks.py` (new) | Tests per module. |
| `docs/V1_2_Findings.md` (new) + updates to `docs/V1_2_Assumptions.md`, `docs/V1_2_Hand_Roundtrip_Plan.md` | Measured result and assumption status. |
| `docs/V1_2_Implementation_Plan.md`, `docs/V1_2_Feasibility_Probes.md` | This plan and the measurements behind it (committed before Task 1). |

Task order: 1 `HandModel` → 2 analytic start → 3 refinement + decoder → 4 shape → 5 extraction → 6 measurement → 7 findings. Tasks 1–4 need no MediaPipe data; Task 6 needs Tasks 1–5.

---
### Task 1: `HandModel` — the 21-point observation operator

The single definition of "MediaPipe's 21 hand landmarks on a model": skeleton joints for the wrist and the fifteen finger joints, mesh vertices for the five tips, evaluated with the real model's shape blend shapes, pose correctives and skinning (plan §4, P1; assumptions A002, A006, A007).

**Files:**
- Create: `src/kinetic_augment/body_model/hand_model.py`
- Create: `tests/hand_assets.py`, `tests/test_hand_model.py`
- Modify: `src/kinetic_augment/body_model/index.md`, `tests/index.md`

**Interfaces:**
- Consumes: the installed `smplx` (`smplx.create`, `smplx.lbs.batch_rodrigues`, `smplx.lbs.batch_rigid_transform`) and the local assets `models/mano/MANO_{LEFT,RIGHT}.npz`, `models/smplx/SMPLX_NEUTRAL.npz`.
- Produces (every later task relies on these exact names):
  - constants `NUM_HAND_LANDMARKS = 21`, `WRIST = 0`, `TIPS = (4, 8, 12, 16, 20)`, `FINGERS` (dict name → 4 landmarks), `LANDMARK_NAMES` (21 strings), `HAND_PARENTS` (np.ndarray of 21 ints, wrist −1), `POSE_SLOT_LANDMARKS` (15 ints), `NUM_FINGER_JOINTS = 15`
  - `HandModel(backend: str, side: str, model_dir: str = "models", num_betas: int = 10, dtype=torch.float64)` with attributes `backend`, `side`, `dtype`, `num_betas` and methods:
    - `points(wrist_orient, hand_pose, betas) -> torch.Tensor` `(B, 21, 3)`, metres, MediaPipe order, untranslated, differentiable; each argument may be batched `(B, n)` or single `(n,)`
    - `rest_points(betas=None) -> np.ndarray` `(21, 3)` — zero pose
    - `skeleton(betas=None)` → object with `rest_joints` `(21, 3)`, `parents` (= `HAND_PARENTS`), `num_joints` (= 21); the shape `solve_swing_ik` and `forward_kinematics` accept
  - `tests/hand_assets.py`: `MODEL_DIR` (absolute path string), `HAVE_MANO`, `HAVE_SMPLX`, `requires_mano`, `requires_smplx`, `BACKENDS` (pytest params `"mano"`, `"smplx"` with skip marks), `SIDES = ["left", "right"]`, `hand(backend, side) -> HandModel` (cached), `training_poses(side, count, seed) -> (count, 45)` real MANO training poses (needs the MANO assets)

- [ ] **Step 1: Write the shared test helpers**

Create `tests/hand_assets.py`:

```python
"""
Shared helpers for the hand tests: where the licensed model assets live, skip
marks for when they are absent, and a cache so each HandModel loads once.
"""

from pathlib import Path

import pytest

MODEL_DIR = str(Path(__file__).resolve().parents[1] / "models")
HAVE_MANO = all(
    (Path(MODEL_DIR) / "mano" / f"MANO_{side}.npz").exists() for side in ("LEFT", "RIGHT")
)
HAVE_SMPLX = (Path(MODEL_DIR) / "smplx" / "SMPLX_NEUTRAL.npz").exists()

try:
    import smplx  # noqa: F401
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

requires_mano = pytest.mark.skipif(not (HAVE_TORCH and HAVE_MANO), reason="MANO npz assets absent")
requires_smplx = pytest.mark.skipif(
    not (HAVE_TORCH and HAVE_SMPLX), reason="SMPL-X assets absent"
)
BACKENDS = [
    pytest.param("mano", marks=requires_mano),
    pytest.param("smplx", marks=requires_smplx),
]
SIDES = ["left", "right"]

_cache = {}


def training_poses(side, count, seed):
    """
    `count` real hand poses from MANO's training set (`hands_coeffs`), in the
    flat-hand-mean convention: hands_mean + coefficients @ hands_components.
    The SMPL-X hand pose space is the same (identical mean and components).
    """
    import numpy as np

    raw = np.load(Path(MODEL_DIR) / "mano" / f"MANO_{side.upper()}.npz")
    rows = np.random.default_rng(seed).choice(len(raw["hands_coeffs"]), count, replace=False)
    return raw["hands_mean"] + raw["hands_coeffs"][rows] @ raw["hands_components"]


def hand(backend, side):
    """A HandModel per (backend, side), loaded once per test session."""
    from kinetic_augment.body_model.hand_model import HandModel

    if (backend, side) not in _cache:
        _cache[(backend, side)] = HandModel(backend, side, model_dir=MODEL_DIR)
    return _cache[(backend, side)]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_hand_model.py`. Every check on the landmark mapping compares against an independent source — the full `smplx` forward pass with index lists written out here, the raw model arrays, or anatomy true of any hand — so a wrong table in `hand_model.py` cannot validate itself.

```python
"""
Tests for the 21-point hand observation operator (V1.2 plan, P1).

Mapping checks compare against independent sources -- the full smplx forward
pass with index lists written out below, the raw model arrays, or anatomy that
holds for any human hand -- so a wrong table in hand_model.py cannot pass.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.hand_assets import BACKENDS, MODEL_DIR, SIDES, hand, requires_mano

torch = pytest.importorskip("torch")
smplx = pytest.importorskip("smplx")

from kinetic_augment.body_model.hand_model import FINGERS, TIPS  # noqa: E402

# MediaPipe landmark -> model joint, written out independently of hand_model.py.
MANO_JOINTS = {0: 0, 1: 13, 2: 14, 3: 15, 5: 1, 6: 2, 7: 3, 9: 4, 10: 5, 11: 6,
               13: 10, 14: 11, 15: 12, 17: 7, 18: 8, 19: 9}
SMPLX_JOINTS = {
    "left": {0: 20, 1: 37, 2: 38, 3: 39, 5: 25, 6: 26, 7: 27, 9: 28, 10: 29, 11: 30,
             13: 34, 14: 35, 15: 36, 17: 31, 18: 32, 19: 33},
    "right": {0: 21, 1: 52, 2: 53, 3: 54, 5: 40, 6: 41, 7: 42, 9: 43, 10: 44, 11: 45,
              13: 49, 14: 50, 15: 51, 17: 46, 18: 47, 19: 48},
}
TIP_VERTICES = {
    ("mano", "left"): (744, 320, 443, 554, 671),
    ("mano", "right"): (744, 320, 443, 554, 671),
    ("smplx", "left"): (5361, 4933, 5058, 5169, 5286),
    ("smplx", "right"): (8079, 7669, 7794, 7905, 8022),
}
MIRROR = np.array([-1.0, 1.0, 1.0])  # left and right hands reflect across the x = 0 plane
MIRROR_TOLERANCE_M = {"mano": 1e-6, "smplx": 0.015}


def reference_points(backend, side, wrist_orient, hand_pose, betas):
    """The same 21 points read from the full smplx forward pass."""
    batch = len(wrist_orient)

    def t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float64)

    if backend == "mano":
        model = smplx.create(MODEL_DIR, model_type="mano", is_rhand=side == "right",
                             use_pca=False, flat_hand_mean=True, num_betas=10, ext="npz",
                             batch_size=batch, dtype=torch.float64)
        out = model(betas=t(betas), global_orient=t(wrist_orient), hand_pose=t(hand_pose))
        joints = MANO_JOINTS
    else:
        model = smplx.create(MODEL_DIR, model_type="smplx", gender="neutral", use_pca=False,
                             flat_hand_mean=True, num_betas=10, batch_size=batch,
                             dtype=torch.float64)
        joints = SMPLX_JOINTS[side]
        body_pose = torch.zeros(batch, 63, dtype=torch.float64)
        wrist = joints[0]
        body_pose[:, 3 * (wrist - 1):3 * wrist] = t(wrist_orient)
        out = model(betas=t(betas), body_pose=body_pose, **{f"{side}_hand_pose": t(hand_pose)})
    expected = np.zeros((batch, 21, 3))
    for landmark, joint in joints.items():
        expected[:, landmark] = out.joints[:, joint].detach().numpy()
    tips = list(TIP_VERTICES[(backend, side)])
    expected[:, list(TIPS)] = out.vertices[:, tips].detach().numpy()
    return expected


def mirrored(rotations):
    """Axis-angle rotations reflected across x = 0: (x, y, z) -> (x, -y, -z)."""
    r = np.asarray(rotations, dtype=float)
    return (r.reshape(-1, 3) * [1.0, -1.0, -1.0]).reshape(r.shape)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_points_equal_the_full_model_forward(backend, side):
    rng = np.random.default_rng(0)
    wrist_orient = rng.normal(0.0, 1.0, (4, 3))
    hand_pose = rng.normal(0.0, 0.4, (4, 45))
    betas = rng.normal(0.0, 1.0, (4, 10))
    got = hand(backend, side).points(wrist_orient, hand_pose, betas).detach().numpy()
    want = reference_points(backend, side, wrist_orient, hand_pose, betas)
    assert np.abs(got - want).max() < 1e-9


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_pose_is_the_flat_template_hand(backend, side):
    """flat_hand_mean=True: zero pose is the raw template, not MANO's curled mean."""
    if backend == "mano":
        raw = np.load(Path(MODEL_DIR) / "mano" / f"MANO_{side.upper()}.npz", allow_pickle=True)
        joints = MANO_JOINTS
    else:
        raw = np.load(Path(MODEL_DIR) / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
        joints = SMPLX_JOINTS[side]
    template = np.asarray(raw["v_template"], dtype=np.float64)
    regressed = np.asarray(raw["J_regressor"], dtype=np.float64) @ template
    expected = np.zeros((21, 3))
    for landmark, joint in joints.items():
        expected[landmark] = regressed[joint]
    expected[list(TIPS)] = template[list(TIP_VERTICES[(backend, side)])]
    np.testing.assert_allclose(hand(backend, side).rest_points(), expected, atol=1e-9)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_landmarks_follow_hand_anatomy(backend, side):
    """At rest, for any hand: each chain runs outward and the knuckles run thumb to pinky."""
    rest = hand(backend, side).rest_points()
    from_wrist = np.linalg.norm(rest - rest[0], axis=1)
    for chain in FINGERS.values():
        assert np.all(np.diff(from_wrist[list(chain)]) > 0)
    knuckles = [5, 9, 13, 17]
    assert np.all(np.diff(np.linalg.norm(rest[knuckles] - rest[5], axis=1)) > 0)
    assert np.linalg.norm(rest[1] - rest[5]) < np.linalg.norm(rest[1] - rest[17])
    assert int(np.argmax(from_wrist[list(TIPS)])) == 2  # the middle fingertip reaches farthest


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_tips_extend_their_distal_bone(backend, side):
    rest = hand(backend, side).rest_points()
    for chain in FINGERS.values():
        pip, dip, tip = rest[chain[1]], rest[chain[2]], rest[chain[3]]
        length = np.linalg.norm(tip - dip)
        cosine = np.dot(dip - pip, tip - dip) / (np.linalg.norm(dip - pip) * length)
        assert 0.015 < length < 0.040  # probe: 20-35 mm (MANO), 21-34 mm (SMPL-X)
        assert cosine > np.cos(np.deg2rad(25.0))  # probe: at most 12.6 deg (MANO), 20.3 (SMPL-X)


@pytest.mark.parametrize("backend", BACKENDS)
def test_left_hand_is_the_mirror_of_the_right(backend):
    """
    Same shape and mirrored rotations give mirrored hands -- the side conventions agree.
    MANO's left model is an exact mirror of its right; SMPL-X's body is only nearly
    symmetric (about 6 mm at rest), far below what a side or rotation-sign error causes.
    """
    tolerance = MIRROR_TOLERANCE_M[backend]
    rng = np.random.default_rng(1)
    wrist_orient = rng.normal(0.0, 1.0, (3, 3))
    hand_pose = rng.normal(0.0, 0.4, (3, 45))
    for betas in (np.zeros(10), 2.0 * np.eye(10)[0], 2.0 * np.eye(10)[1]):
        right = hand(backend, "right").points(wrist_orient, hand_pose, betas).detach().numpy()
        left = hand(backend, "left").points(
            mirrored(wrist_orient), mirrored(hand_pose), betas).detach().numpy()
        assert np.abs(left - right * MIRROR).max() < tolerance


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_joint_moves_only_its_own_finger(backend):
    model = hand(backend, "right")
    pose = np.zeros(45)
    pose[0:3] = [0.0, 0.0, np.deg2rad(30.0)]  # slot 0: the index MCP (landmark 5)
    moved = model.points(np.zeros(3), pose, np.zeros(10))[0].detach().numpy()
    shift = np.linalg.norm(moved - model.rest_points(), axis=1)
    for landmark in range(21):
        if landmark in (6, 7, 8):
            assert shift[landmark] > 1e-3
        elif landmark in TIPS:
            assert shift[landmark] < 1e-3  # pose correctives nudge other tips (MANO probe: 0.01 mm)
        else:
            assert shift[landmark] < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradients_are_exact(backend):
    model = hand(backend, "right")
    rng = np.random.default_rng(2)
    inputs = tuple(
        torch.tensor(rng.normal(0.0, scale, size), dtype=torch.float64, requires_grad=True)
        for scale, size in ((0.5, 3), (0.3, 45), (1.0, 10))
    )
    assert torch.autograd.gradcheck(lambda w, p, b: model.points(w, p, b), inputs,
                                    eps=1e-6, atol=1e-7)


@requires_mano
def test_rejects_an_unknown_side_or_backend():
    from kinetic_augment.body_model.hand_model import HandModel

    with pytest.raises(ValueError):
        HandModel("mano", "middle", model_dir=MODEL_DIR)
    with pytest.raises(ValueError):
        HandModel("flame", "left", model_dir=MODEL_DIR)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_model.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'kinetic_augment.body_model.hand_model'`.

- [ ] **Step 4: Implement `hand_model.py`**

Create `src/kinetic_augment/body_model/hand_model.py`:

```python
"""
MediaPipe's 21 hand landmarks evaluated on MANO or SMPL-X (V1.2).

MediaPipe reports the wrist, three joints per finger and the five fingertips.
Both models have the wrist and the fifteen finger joints as skeleton joints but
no fingertip joints, so the tips are read from fixed mesh vertices.
`HandModel.points` computes exactly those 21 points of the real model -- shape
blend shapes, pose correctives and linear blend skinning, through `smplx.lbs`
-- without skinning the rest of the mesh. It equals the full forward pass.

Conventions:
  * Hand pose is the full 45-value axis-angle pose of the fifteen finger joints
    with `flat_hand_mean=True`, so zero is the flat template hand. Slots follow
    the models' shared order -- index, middle, pinky, ring, thumb; proximal
    joint first -- and `POSE_SLOT_LANDMARKS` names the landmark at each joint.
    A joint's rotation orients the bone to its child landmark.
  * `wrist_orient` is the wrist joint's world rotation. In MANO that is the
    root rotation. In SMPL-X it is the wrist's local rotation with every other
    body joint at rest, so the two coincide; attaching the hand to a posed arm
    is a separate step.
  * Positions are metres in the model frame, before any translation.
"""

from types import SimpleNamespace

import numpy as np
import torch

NUM_HAND_LANDMARKS = 21
NUM_FINGER_JOINTS = 15
WRIST = 0
TIPS = (4, 8, 12, 16, 20)
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
LANDMARK_NAMES = (
    "wrist",
    "thumb CMC", "thumb MCP", "thumb IP", "thumb tip",
    "index MCP", "index PIP", "index DIP", "index tip",
    "middle MCP", "middle PIP", "middle DIP", "middle tip",
    "ring MCP", "ring PIP", "ring DIP", "ring tip",
    "pinky MCP", "pinky PIP", "pinky DIP", "pinky tip",
)

# Parent of each MediaPipe hand landmark; the wrist has none.
HAND_PARENTS = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

# The landmark at the joint whose local rotation fills each 3-value slot of the
# 45-value hand pose (index, middle, pinky, ring, thumb; proximal first).
POSE_SLOT_LANDMARKS = (5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3)

# Fingertip vertices in MediaPipe tip order: thumb, index, middle, ring, pinky.
MANO_TIP_VERTICES = (744, 320, 443, 554, 671)
SMPLX_TIP_VERTICES = {
    "left": (5361, 4933, 5058, 5169, 5286),
    "right": (8079, 7669, 7794, 7905, 8022),
}
# Each SMPL-X hand's wrist joint and first finger joint (index1).
SMPLX_WRIST_JOINT = {"left": 20, "right": 21}
SMPLX_FIRST_FINGER_JOINT = {"left": 25, "right": 40}


class HandModel:
    """The 21 MediaPipe hand points of one side of MANO or SMPL-X."""

    def __init__(self, backend: str, side: str, model_dir: str = "models", num_betas: int = 10,
                 dtype=torch.float64):
        import smplx

        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if backend == "mano":
            model = smplx.create(model_dir, model_type="mano", is_rhand=side == "right",
                                 use_pca=False, flat_hand_mean=True, num_betas=num_betas,
                                 ext="npz", dtype=dtype)
            wrist, first_finger, tips = 0, 1, MANO_TIP_VERTICES
        elif backend == "smplx":
            model = smplx.create(model_dir, model_type="smplx", gender="neutral",
                                 use_pca=False, flat_hand_mean=True, num_betas=num_betas,
                                 dtype=dtype)
            wrist = SMPLX_WRIST_JOINT[side]
            first_finger = SMPLX_FIRST_FINGER_JOINT[side]
            tips = SMPLX_TIP_VERTICES[side]
        else:
            raise ValueError(f"backend must be 'mano' or 'smplx', got {backend!r}")
        if torch.any(model.pose_mean != 0):
            raise ValueError("expected flat_hand_mean: zero pose must be the template hand")

        self.backend, self.side, self.dtype = backend, side, dtype
        self.num_betas = int(model.shapedirs.shape[-1])
        self._wrist = wrist
        self._fingers = slice(first_finger, first_finger + NUM_FINGER_JOINTS)
        self._parents = model.parents
        self._num_joints = int(model.J_regressor.shape[0])

        with torch.no_grad():
            regressor = model.J_regressor.to(dtype)
            self._joint_template = regressor @ model.v_template
            self._joint_shapedirs = torch.einsum("jv,vck->jck", regressor, model.shapedirs)
            tip_ids = torch.tensor(tips)
            self._tip_template = model.v_template[tip_ids]
            self._tip_shapedirs = model.shapedirs[tip_ids]
            columns = (3 * tip_ids[:, None] + torch.arange(3)).reshape(-1)
            self._tip_posedirs = model.posedirs[:, columns]
            self._tip_weights = model.lbs_weights[tip_ids]

        joints = {WRIST: wrist}
        for slot, landmark in enumerate(POSE_SLOT_LANDMARKS):
            joints[landmark] = first_finger + slot
        order = sorted(joints)
        self._joint_landmarks = torch.tensor(order)
        self._landmark_joints = torch.tensor([joints[i] for i in order])

    def points(self, wrist_orient, hand_pose, betas) -> torch.Tensor:
        """
        Model-frame positions of the 21 MediaPipe hand points.

        Args:
            wrist_orient: (B, 3) or (3,) axis-angle wrist rotation.
            hand_pose: (B, 45) or (45,) finger joint rotations, axis-angle.
            betas: (B, K) or (K,) shape coefficients, K = num_betas.

        Returns:
            (B, 21, 3) metres in MediaPipe order, untranslated. Differentiable.
        """
        from smplx.lbs import batch_rigid_transform, batch_rodrigues

        wrist_orient = torch.as_tensor(wrist_orient, dtype=self.dtype).reshape(-1, 3)
        hand_pose = torch.as_tensor(hand_pose, dtype=self.dtype).reshape(-1, 3 * NUM_FINGER_JOINTS)
        betas = torch.as_tensor(betas, dtype=self.dtype).reshape(-1, self.num_betas)
        batch = max(len(wrist_orient), len(hand_pose), len(betas))
        betas = betas.expand(batch, -1)

        pose = wrist_orient.new_zeros(batch, self._num_joints, 3)
        pose[:, self._wrist] = wrist_orient.expand(batch, -1)
        pose[:, self._fingers] = hand_pose.expand(batch, -1).reshape(batch, NUM_FINGER_JOINTS, 3)
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch, -1, 3, 3)

        rest = self._joint_template + torch.einsum("jck,bk->bjc", self._joint_shapedirs, betas)
        joints, transforms = batch_rigid_transform(rotations, rest, self._parents,
                                                   dtype=self.dtype)

        corrective = (rotations[:, 1:] - torch.eye(3, dtype=self.dtype)).reshape(batch, -1)
        tips = (self._tip_template
                + torch.einsum("vck,bk->bvc", self._tip_shapedirs, betas)
                + (corrective @ self._tip_posedirs).reshape(batch, -1, 3))
        skin = torch.einsum("vj,bjrc->bvrc", self._tip_weights, transforms)
        tips = torch.einsum("bvrc,bvc->bvr", skin[..., :3, :3], tips) + skin[..., :3, 3]

        out = joints.new_zeros(batch, NUM_HAND_LANDMARKS, 3)
        out[:, self._joint_landmarks] = joints[:, self._landmark_joints]
        out[:, list(TIPS)] = tips
        return out

    def rest_points(self, betas=None) -> np.ndarray:
        """(21, 3) points of the flat template hand in shape `betas` (neutral by default)."""
        betas = np.zeros(self.num_betas) if betas is None else betas
        with torch.no_grad():
            return self.points(np.zeros(3), np.zeros(3 * NUM_FINGER_JOINTS), betas)[0].numpy()

    def skeleton(self, betas=None):
        """Rest skeleton over the 21 landmarks, in the form `solve_swing_ik` and FK take."""
        return SimpleNamespace(rest_joints=self.rest_points(betas), parents=HAND_PARENTS,
                               num_joints=NUM_HAND_LANDMARKS)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_model.py -q`
Expected: all pass, none skipped (the MANO and SMPL-X assets are present here). If a mapping or mirror test fails, do not edit the expected tables in the test: report which landmark and by how much, with status DONE_WITH_CONCERNS.

- [ ] **Step 6: Update the indexes**

Add to `src/kinetic_augment/body_model/index.md` (before the `index.md` line):
`- \`hand_model.py\` — Evaluates MediaPipe's 21 hand landmarks on MANO or SMPL-X (joints plus fingertip vertices).`

Add to `tests/index.md` (before the `index.md` line):
`- \`hand_assets.py\` — Shared model-asset paths, skip marks and a cached hand-model loader for the hand tests.`
`- \`test_hand_model.py\` — Tests the 21-point hand operator against the full MANO/SMPL-X forward pass and hand anatomy.`

- [ ] **Step 7: Lint, run the whole suite once, commit**

```bash
<python> -m flake8 --max-line-length 100 src/kinetic_augment/body_model/hand_model.py tests/hand_assets.py tests/test_hand_model.py
PYTHONPATH=src <python> -m pytest -q
git add src/kinetic_augment/body_model/hand_model.py tests/hand_assets.py tests/test_hand_model.py src/kinetic_augment/body_model/index.md tests/index.md
git commit -m "feat(hands): evaluate MediaPipe's 21 hand landmarks on MANO and SMPL-X"
```

---

### Task 2: Analytic start — palm rotation and finger swing

The V1 body converter's approach carried to the hand: the palm fixes the wrist rotation (MANO's palm is rigid: thumb CMC and the four MCPs are children of the wrist), then each finger bone is swung onto its observed direction with zero twist. It reproduces every joint landmark of a hand the model can represent exactly; only the tips, which are skinned vertices, are approximate (plan §5.1 steps 2–3).

**Files:**
- Create: `src/kinetic_augment/body_model/hand_converter.py`
- Create: `tests/test_hand_converter.py`
- Modify: `src/kinetic_augment/body_model/index.md`, `tests/index.md`

**Interfaces:**
- Consumes: `HAND_PARENTS`, `NUM_HAND_LANDMARKS`, `POSE_SLOT_LANDMARKS`, `WRIST`, `TIPS`, `HandModel.skeleton/points` (Task 1); `solve_swing_ik(landmarks, skeleton, bones, global_orient)` from `kinetic_augment.body_model.swing_ik` and `forward_kinematics(rotations, skeleton)` from `kinetic_augment.body_model.kinematics` (existing).
- Produces (Tasks 3 and 6 rely on these):
  - `PALM_LANDMARKS = (1, 5, 9, 13, 17)`, `FINGER_BONES` (15 tuples)
  - `palm_rotation(rest: (21,3), observed: (21,3), valid: (21,) bool) -> Optional[(3,3)]` — both inputs wrist-relative
  - `initial_pose(observed: (21,3), valid: (21,) bool, skeleton) -> Optional[Tuple[(3,), (45,)]]` — `(wrist_orient, hand_pose)`
  - `node_rotations(wrist_orient, hand_pose) -> (21, 3)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hand_converter.py`:

```python
"""
Tests for the V1.2 hand converter (plan P1-P2).

Synthetic hands come from a skeleton or the real model with known parameters,
so every target is exactly representable and any error is the converter's.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.hand_converter import initial_pose, node_rotations, palm_rotation
from kinetic_augment.body_model.hand_model import HAND_PARENTS, TIPS
from kinetic_augment.body_model.kinematics import forward_kinematics
from tests.hand_assets import BACKENDS, SIDES, hand

# A hand-shaped rest skeleton in metres -- right hand, fingers along +y -- so the
# analytic start can be tested without licensed model assets.
SYNTHETIC_REST = np.array([
    [0.000, 0.000, 0.000],
    [-0.025, 0.020, 0.005], [-0.045, 0.045, 0.008], [-0.058, 0.070, 0.010], [-0.068, 0.092, 0.012],
    [-0.020, 0.085, 0.000], [-0.022, 0.125, 0.000], [-0.023, 0.150, 0.000], [-0.024, 0.172, 0.000],
    [0.000, 0.090, 0.000], [0.000, 0.135, 0.000], [0.000, 0.163, 0.000], [0.000, 0.187, 0.000],
    [0.018, 0.085, 0.000], [0.020, 0.127, 0.000], [0.021, 0.153, 0.000], [0.022, 0.175, 0.000],
    [0.034, 0.075, 0.000], [0.037, 0.105, 0.000], [0.039, 0.125, 0.000], [0.040, 0.143, 0.000],
])
SYNTHETIC = SimpleNamespace(rest_joints=SYNTHETIC_REST, parents=HAND_PARENTS, num_joints=21)
ALL = np.ones(21, dtype=bool)


def posed_synthetic(seed):
    """The synthetic skeleton under random rotations -- twist included -- and a random offset."""
    rng = np.random.default_rng(seed)
    wrist = Rotation.random(random_state=seed).as_rotvec()
    pose = rng.normal(0.0, 0.5, 45)
    return forward_kinematics(node_rotations(wrist, pose), SYNTHETIC) + rng.normal(0.0, 0.1, 3)


class TestPalmRotation:
    def test_recovers_a_known_rotation(self):
        rest = SYNTHETIC_REST - SYNTHETIC_REST[0]
        truth = Rotation.from_rotvec([0.3, -1.1, 0.7]).as_matrix()
        np.testing.assert_allclose(palm_rotation(rest, rest @ truth.T, ALL), truth, atol=1e-12)

    def test_needs_two_palm_points_off_one_line(self):
        rest = SYNTHETIC_REST - SYNTHETIC_REST[0]
        one = np.zeros(21, dtype=bool)
        one[[0, 5]] = True
        assert palm_rotation(rest, rest, one) is None
        line = rest.copy()
        line[[1, 5, 9, 13, 17]] = np.outer([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.02, 0.0])
        assert palm_rotation(line, line, ALL) is None


class TestInitialPose:
    def test_is_exact_on_a_rigid_skeleton(self):
        for seed in range(5):
            observed = posed_synthetic(seed)
            wrist, pose = initial_pose(observed, ALL, SYNTHETIC)
            rebuilt = forward_kinematics(node_rotations(wrist, pose), SYNTHETIC)
            np.testing.assert_allclose(rebuilt - rebuilt[0], observed - observed[0], atol=1e-12)

    def test_invalid_landmarks_never_influence_it(self):
        observed = posed_synthetic(7)
        valid = ALL.copy()
        valid[[8, 14]] = False
        corrupted = observed.copy()
        corrupted[[8, 14]] += 0.05
        a, b = initial_pose(observed, valid, SYNTHETIC), initial_pose(corrupted, valid, SYNTHETIC)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_without_a_wrist_or_a_palm_there_is_no_start(self):
        observed = posed_synthetic(8)
        no_wrist = ALL.copy()
        no_wrist[0] = False
        assert initial_pose(observed, no_wrist, SYNTHETIC) is None
        no_palm = ALL.copy()
        no_palm[[1, 5, 9, 13, 17]] = False
        assert initial_pose(observed, no_palm, SYNTHETIC) is None


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_on_the_real_model_joints_are_exact_and_tips_close(backend, side):
    """
    Joints are pure forward kinematics, so the swing start reproduces them exactly. Tips
    are mesh vertices moved by pose correctives the start ignores (MANO probe: up to
    11.8 mm over its 1554 training poses), so they are only close.
    """
    model = hand(backend, side)
    rng = np.random.default_rng(11)
    wrist = Rotation.random(4, random_state=11).as_rotvec()
    pose = rng.normal(0.0, 0.3, (4, 45))
    observed = model.points(wrist, pose, np.zeros(10)).detach().numpy()
    skeleton = model.skeleton()
    joints = [i for i in range(21) if i not in TIPS]
    for frame in observed:
        start = initial_pose(frame, ALL, skeleton)
        rebuilt = model.points(start[0], start[1], np.zeros(10))[0].detach().numpy()
        gap_mm = 1000.0 * np.linalg.norm(rebuilt - frame, axis=1)
        assert gap_mm[joints].max() < 1e-4  # exact; smplx's Rodrigues adds 1e-8 rad
        assert gap_mm[list(TIPS)].max() < 10.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_converter.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'kinetic_augment.body_model.hand_converter'`.

- [ ] **Step 3: Implement**

Create `src/kinetic_augment/body_model/hand_converter.py`:

```python
"""
V1.2 MediaPipe hand landmarks <-> MANO / SMPL-X hand parameters.

The analytic start follows the V1 body converter. MANO's palm is rigid -- the
thumb CMC and the four finger MCPs are children of the wrist -- so a Kabsch
rotation about the wrist over those five points fixes the wrist rotation. Each
finger bone is then swung onto its observed direction with zero twist
(`solve_swing_ik`), parents before children. On a hand the model can represent
exactly this reproduces every joint landmark; only the fingertips, which are
skinned mesh vertices, are approximate.

Refinement and shape fitting follow in later sections of this module.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.hand_model import (
    HAND_PARENTS,
    NUM_HAND_LANDMARKS,
    POSE_SLOT_LANDMARKS,
    WRIST,
)
from kinetic_augment.body_model.swing_ik import solve_swing_ik

PALM_LANDMARKS = (1, 5, 9, 13, 17)

# (driven node, child node, observed parent, observed child): each finger joint's
# rotation aims the bone to its child. Bones leaving the wrist form the palm,
# which `palm_rotation` places instead.
FINGER_BONES = [(int(p), c, int(p), c) for c, p in enumerate(HAND_PARENTS) if p > 0]


def palm_rotation(rest: np.ndarray, observed: np.ndarray,
                  valid: np.ndarray) -> Optional[np.ndarray]:
    """
    Proper rotation about the wrist best carrying the rest palm onto the observed one.

    Args:
        rest, observed: (21, 3) positions relative to their own wrist.
        valid: (21,) bool.

    Returns:
        (3, 3) rotation, or None when the valid palm points cannot fix one:
        fewer than two of them, or all on one line through the wrist.
    """
    idx = [i for i in PALM_LANDMARKS if valid[i]]
    if len(idx) < 2:
        return None
    u, s, vt = np.linalg.svd(rest[idx].T @ observed[idx])
    if s[1] <= 1e-6 * s[0]:
        return None
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, flip]) @ u.T


def initial_pose(observed, valid, skeleton) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Analytic start for one frame: (wrist_orient (3,), hand_pose (45,)), axis-angle.

    Args:
        observed: (21, 3) landmark positions, any origin.
        valid: (21,) bool; invalid landmarks never influence the result.
        skeleton: rest skeleton from `HandModel.skeleton(betas)`.

    Returns None when the wrist is invalid or the palm cannot fix a rotation.
    """
    observed = np.asarray(observed, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(observed).all(axis=1)
    if not valid[WRIST]:
        return None
    rest = np.asarray(skeleton.rest_joints, dtype=np.float64)
    rotation = palm_rotation(rest - rest[WRIST], observed - observed[WRIST], valid)
    if rotation is None:
        return None
    # Swing IK reads only directions, and takes an all-zero point as missing, so it
    # gets the observations as given rather than wrist-relative.
    masked = np.where(valid[:, None], observed, np.nan)
    local = solve_swing_ik(masked, skeleton, FINGER_BONES,
                           global_orient=Rotation.from_matrix(rotation).as_rotvec())
    return local[WRIST], local[list(POSE_SLOT_LANDMARKS)].reshape(-1)


def node_rotations(wrist_orient, hand_pose) -> np.ndarray:
    """(21, 3) local rotation per landmark node -- the layout `forward_kinematics` takes."""
    nodes = np.zeros((NUM_HAND_LANDMARKS, 3))
    nodes[WRIST] = wrist_orient
    nodes[list(POSE_SLOT_LANDMARKS)] = np.asarray(hand_pose, dtype=np.float64).reshape(-1, 3)
    return nodes
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_converter.py -q`
Expected: all pass, none skipped.

- [ ] **Step 5: Update the indexes**

Add to `src/kinetic_augment/body_model/index.md`:
`- \`hand_converter.py\` — Fits MANO / SMPL-X hand parameters to MediaPipe hand landmarks and rebuilds them from parameters alone.`

Add to `tests/index.md`:
`- \`test_hand_converter.py\` — Tests the hand fit: analytic start, real-model refinement, decoder, serialisation and shape.`

- [ ] **Step 6: Lint, run the whole suite once, commit**

```bash
<python> -m flake8 --max-line-length 100 src/kinetic_augment/body_model/hand_converter.py tests/test_hand_converter.py
PYTHONPATH=src <python> -m pytest -q
git add src/kinetic_augment/body_model/hand_converter.py tests/test_hand_converter.py src/kinetic_augment/body_model/index.md tests/index.md
git commit -m "feat(hands): analytic hand start from palm rotation and finger swing"
```

---
### Task 3: Real-model refinement, `HandFit`, parameter-only decoder and serialisation

Turns the analytic start into a fit through the real forward pass, defines the fit record and the decoder, and proves the decoder needs nothing but parameters (plan P1–P2; assumptions A008, A009).

**Files:**
- Modify: `src/kinetic_augment/body_model/hand_converter.py` (extend the docstring and imports as shown; append the code)
- Modify: `tests/test_hand_converter.py` (append)

**Interfaces:**
- Consumes: `HandModel.points / skeleton / rest_points / num_betas / backend / side` (Task 1); `initial_pose`, `node_rotations` (Task 2).
- Produces (Tasks 4 and 6 rely on these exact names):
  - `HandFit` dataclass: `backend: str, side: str, wrist_orient: (3,), hand_pose: (45,), betas: (K,), transl: (3,), status: str = "ok", rms_mm: float = nan`
  - `fit_hands(observed: (T,21,3), valid: (T,21) bool, model: HandModel, betas: Optional[(K,)] = None) -> List[HandFit]`
  - `reconstruct_hand(fit: HandFit, model: HandModel) -> np.ndarray` `(21, 3)` metres
  - `landmark_errors_mm(predicted, observed) -> np.ndarray` `(..., 20)` — the plan's primary metric
  - `save_hand_fits(path, fits)`, `load_hand_fits(path) -> List[HandFit]`
  - internal helpers Task 4 extends: `_prepared(observed, valid)`, `_rms_mm(residual, mask)`, `_optimise(model, observed, valid, start, betas)`
  - constants `FIT_DTYPE`, `POSE_PRIOR`, `MAX_ITERATIONS`, `FRAMES_PER_BATCH`, `FIT_STATUSES`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hand_converter.py`. Add to its imports `import os`, `import subprocess`, `import sys`, `from pathlib import Path`; extend the converter import to
`from kinetic_augment.body_model.hand_converter import (fit_hands, initial_pose, landmark_errors_mm, load_hand_fits, node_rotations, palm_rotation, reconstruct_hand, save_hand_fits)`; and extend the helpers import to `from tests.hand_assets import BACKENDS, MODEL_DIR, SIDES, hand, requires_mano, training_poses`.

```python
SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")


def synthetic_track(model, count, seed, spread=0.3, betas=None, poses=None):
    """
    Model-generated landmarks with known parameters, offset the way hand-world origins
    are. Finger poses are random (N(0, spread) per value) unless `poses` is given.
    """
    rng = np.random.default_rng(seed)
    wrist = Rotation.random(count, random_state=seed).as_rotvec()
    pose = rng.normal(0.0, spread, (count, 45)) if poses is None else np.asarray(poses)
    shape = np.zeros(model.num_betas) if betas is None else np.asarray(betas, dtype=float)
    points = model.points(wrist, pose, shape).detach().numpy()
    return points + rng.normal(0.0, 0.05, (count, 1, 3))


def rebuild(fits, model):
    return np.stack([reconstruct_hand(f, model) for f in fits])


def all_valid(observed):
    return np.ones(observed.shape[:2], dtype=bool)


class TestLandmarkErrors:
    def test_scores_wrist_relative_distances_in_millimetres(self):
        observed = np.zeros((1, 21, 3))
        predicted = observed.copy()
        predicted[0, :, 0] += 0.5             # a pure translation is not an error
        predicted[0, 8, 1] += 0.003           # 3 mm at the index tip
        errors = landmark_errors_mm(predicted, observed)
        assert errors.shape == (1, 20)
        np.testing.assert_allclose(errors[0, 7], 3.0)
        np.testing.assert_allclose(np.delete(errors[0], 7), 0.0, atol=1e-12)


@pytest.mark.parametrize("poses", ["random", pytest.param("training", marks=requires_mano)])
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_recovers_hands_the_model_can_represent(backend, side, poses):
    """Plan Section 6 synthetic gate: mean <= 0.5 mm, p95 <= 1 mm."""
    model = hand(backend, side)
    real = training_poses(side, 12, seed=20) if poses == "training" else None
    observed = synthetic_track(model, 12, seed=21, poses=real)
    fits = fit_hands(observed, all_valid(observed), model)
    assert [f.status for f in fits] == ["ok"] * 12
    rebuilt = rebuild(fits, model)
    errors = landmark_errors_mm(rebuilt, observed)
    assert errors.mean() <= 0.5
    assert np.percentile(errors, 95) <= 1.0
    np.testing.assert_allclose(rebuilt[:, 0], observed[:, 0], atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rotating_the_observations_rotates_the_reconstruction(backend):
    model = hand(backend, "right")
    observed = synthetic_track(model, 3, seed=22)
    turn = Rotation.from_rotvec([0.4, -0.9, 1.3]).as_matrix()
    plain = rebuild(fit_hands(observed, all_valid(observed), model), model)
    turned = rebuild(fit_hands(observed @ turn.T, all_valid(observed), model), model)
    # Both runs converge to the same hand to well under 0.1 mm; a frame-convention bug
    # would differ by centimetres.
    np.testing.assert_allclose(turned, plain @ turn.T, atol=1e-4)


@requires_mano
def test_invalid_landmarks_never_influence_the_fit():
    model = hand("mano", "right")
    observed = synthetic_track(model, 2, seed=23)
    valid = all_valid(observed)
    valid[:, [8, 16]] = False
    corrupted = observed.copy()
    corrupted[:, [8, 16]] += 0.03
    for a, b in zip(fit_hands(observed, valid, model), fit_hands(corrupted, valid, model)):
        np.testing.assert_array_equal(a.wrist_orient, b.wrist_orient)
        np.testing.assert_array_equal(a.hand_pose, b.hand_pose)


@requires_mano
def test_frames_it_cannot_fit_are_reported_not_invented():
    model = hand("mano", "right")
    observed = synthetic_track(model, 3, seed=24)
    valid = all_valid(observed)
    valid[1, [1, 5, 9, 13, 17]] = False
    valid[2, 0] = False
    fits = fit_hands(observed, valid, model)
    assert [f.status for f in fits] == ["ok", "no_palm", "no_wrist"]
    assert np.isnan(reconstruct_hand(fits[1], model)).all()
    assert np.isnan(fits[2].hand_pose).all()


@requires_mano
def test_reconstruction_refuses_a_model_of_the_other_side():
    model = hand("mano", "right")
    observed = synthetic_track(model, 1, seed=25)
    fit = fit_hands(observed, all_valid(observed), model)[0]
    with pytest.raises(ValueError):
        reconstruct_hand(fit, hand("mano", "left"))


@requires_mano
def test_a_fresh_process_rebuilds_the_hands_from_saved_parameters(tmp_path):
    """Plan A008: the decoder needs only parameters and the model -- no landmarks, no cache."""
    model = hand("mano", "right")
    observed = synthetic_track(model, 3, seed=26)
    fits = fit_hands(observed, all_valid(observed), model)
    saved, out = tmp_path / "fits.npz", tmp_path / "rebuilt.npy"
    save_hand_fits(saved, fits)
    code = (
        "import numpy as np\n"
        "from kinetic_augment.body_model.hand_converter import load_hand_fits, reconstruct_hand\n"
        "from kinetic_augment.body_model.hand_model import HandModel\n"
        f"model = HandModel('mano', 'right', model_dir={MODEL_DIR!r})\n"
        f"fits = load_hand_fits({str(saved)!r})\n"
        f"np.save({str(out)!r}, np.stack([reconstruct_hand(f, model) for f in fits]))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC_DIR, os.environ.get("PYTHONPATH", "")]))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    np.testing.assert_allclose(np.load(out), rebuild(fits, model), atol=1e-12)
    assert [f.status for f in load_hand_fits(saved)] == [f.status for f in fits]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_converter.py -q`
Expected: collection error — `ImportError: cannot import name 'fit_hands'`.

- [ ] **Step 3: Implement**

In `src/kinetic_augment/body_model/hand_converter.py`:

1. Replace the docstring's last paragraph (`Refinement and shape fitting follow in later sections of this module.`) with:

```text
Contract, per frame and hand:
    21 world landmarks (metres) + per-landmark validity
        -> genuine model parameters (wrist_orient, hand_pose, betas, transl)
        -> 21 landmarks rebuilt from those parameters alone

`fit_hands` refines the analytic start through the real model forward pass
(L-BFGS on the summed squared wrist-relative landmark error), which settles the
fingertips -- skinned mesh vertices the analytic start only approximates -- and
whatever the model cannot represent exactly. `reconstruct_hand` reads nothing
but a fit's parameters and the model.
```

2. Change the imports to:

```python
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.hand_model import (
    HAND_PARENTS,
    NUM_HAND_LANDMARKS,
    POSE_SLOT_LANDMARKS,
    WRIST,
)
from kinetic_augment.body_model.swing_ik import solve_swing_ik
```

3. Add below `FINGER_BONES`:

```python
FIT_DTYPE = torch.float64
POSE_PRIOR = 1e-4  # mm^2 per rad^2: only settles directions the landmarks leave free
MAX_ITERATIONS = 1000  # dry run: 200 stops short of convergence (0.07 mm); 1000 reaches 0.02 mm
FRAMES_PER_BATCH = 32
FIT_STATUSES = ("ok", "no_wrist", "no_palm", "diverged")


@dataclass
class HandFit:
    """
    Genuine model parameters for one hand in one frame.

    MANO takes them as they are: global_orient=wrist_orient, hand_pose, betas,
    transl. In SMPL-X, wrist_orient is the wrist joint's local rotation in
    body_pose with every other body joint at rest, and hand_pose is
    left_hand_pose or right_hand_pose. Both use the flat hand mean and no PCA.

    `transl` puts the model's wrist on the observed wrist. A frame without a fit
    has its reason in `status` (one of FIT_STATUSES) and NaN parameters.
    """

    backend: str
    side: str
    wrist_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    transl: np.ndarray
    status: str = "ok"
    rms_mm: float = float("nan")
```

4. Append at the end of the module:

```python
def landmark_errors_mm(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """
    The plan's primary metric: distance in millimetres between predicted and
    observed wrist-relative positions of landmarks 1-20, with no rotation or scale
    registration. Shape (..., 20); NaN where either side is missing.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    gap = (predicted - predicted[..., :1, :]) - (observed - observed[..., :1, :])
    return 1000.0 * np.linalg.norm(gap, axis=-1)[..., 1:]


def _prepared(observed, valid):
    """float64 observations, and validity that also excludes non-finite landmarks."""
    observed = np.asarray(observed, dtype=np.float64)
    return observed, np.asarray(valid, dtype=bool) & np.isfinite(observed).all(axis=-1)


def _unfitted(model, status: str) -> HandFit:
    return HandFit(
        model.backend, model.side, np.full(3, np.nan), np.full(45, np.nan),
        np.full(model.num_betas, np.nan), np.full(3, np.nan), status,
    )


def _rms_mm(residual: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Per-frame RMS of the masked landmark residuals."""
    squared = (residual ** 2).sum(dim=-1) * mask
    return torch.sqrt(squared.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).numpy()


def _optimise(model, observed, valid, start, betas):
    """
    L-BFGS through the real forward pass on the wrist-relative landmark error (mm).
    The frames are independent and their summed loss separates, so one run fits them all.

    Args:
        observed, valid: (F, 21, 3) and (F, 21) for the frames being fitted.
        start: (F, 48) starting wrist_orient and hand_pose.
        betas: (K,) shape.

    Returns (params (F, 48), rms_mm (F,)).
    """
    kept = np.where(valid[..., None], observed, 0.0)
    target = torch.as_tensor(1000.0 * (kept - kept[:, :1]), dtype=FIT_DTYPE)
    mask = torch.tensor(valid)
    mask[:, WRIST] = False  # zero by construction
    shape = torch.as_tensor(betas, dtype=FIT_DTYPE)
    params = torch.tensor(start, dtype=FIT_DTYPE, requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [params], lr=1.0, max_iter=MAX_ITERATIONS, tolerance_grad=1e-12,
        tolerance_change=1e-15, history_size=100, line_search_fn="strong_wolfe",
    )

    def residuals():
        points = 1000.0 * model.points(params[:, :3], params[:, 3:], shape)
        return (points - points[:, :1]) - target

    def closure():
        optimiser.zero_grad()
        loss = (residuals()[mask] ** 2).sum() + POSE_PRIOR * (params[:, 3:] ** 2).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        return params.detach().numpy().copy(), _rms_mm(residuals(), mask)


def fit_hands(observed, valid, model, betas=None) -> List[HandFit]:
    """
    Fit every frame of one hand track independently.

    Args:
        observed: (T, 21, 3) world landmarks of one hand, metres.
        valid: (T, 21) bool; invalid or non-finite landmarks never influence a fit.
        model: `HandModel` of the hand's side.
        betas: (K,) shape used for every frame; neutral when omitted.

    Returns one `HandFit` per frame, in order.
    """
    observed, valid = _prepared(observed, valid)
    betas = np.zeros(model.num_betas) if betas is None else np.asarray(betas, dtype=np.float64)
    skeleton = model.skeleton(betas)

    fits: List[Optional[HandFit]] = [None] * len(observed)
    starts = {}
    for t in range(len(observed)):
        start = initial_pose(observed[t], valid[t], skeleton)
        if start is None:
            fits[t] = _unfitted(model, "no_palm" if valid[t, WRIST] else "no_wrist")
        else:
            starts[t] = np.concatenate(start)

    frames = sorted(starts)
    for first in range(0, len(frames), FRAMES_PER_BATCH):
        batch = frames[first:first + FRAMES_PER_BATCH]
        params, rms = _optimise(
            model, observed[batch], valid[batch], np.stack([starts[t] for t in batch]), betas
        )
        for k, t in enumerate(batch):
            if not (np.isfinite(rms[k]) and np.isfinite(params[k]).all()):
                fits[t] = _unfitted(model, "diverged")
                continue
            fits[t] = HandFit(
                model.backend, model.side, params[k, :3], params[k, 3:], betas.copy(),
                observed[t, WRIST] - skeleton.rest_joints[WRIST], "ok", float(rms[k]),
            )
    return fits


def reconstruct_hand(fit: HandFit, model) -> np.ndarray:
    """
    Rebuild the 21 world landmarks from a fit's parameters alone.

    Returns (21, 3) metres; all NaN when the frame has no fit.

    Raises:
        ValueError: `model` is a different backend or side from the fit's.
    """
    if (fit.backend, fit.side) != (model.backend, model.side):
        raise ValueError(
            f"fit is {fit.backend}/{fit.side} but the model is {model.backend}/{model.side}"
        )
    if fit.status != "ok":
        return np.full((NUM_HAND_LANDMARKS, 3), np.nan)
    with torch.no_grad():
        points = model.points(fit.wrist_orient, fit.hand_pose, fit.betas)[0].numpy()
    return points + fit.transl


def save_hand_fits(path, fits: Sequence[HandFit]) -> None:
    """Write one hand track's fits to an .npz; every fit must share backend and side."""
    tracks = {(f.backend, f.side) for f in fits}
    if len(tracks) != 1:
        raise ValueError(f"one hand track per file, got {sorted(tracks)}")
    (backend, side), = tracks
    np.savez_compressed(
        path,
        backend=np.array(backend),
        side=np.array(side),
        wrist_orient=np.stack([f.wrist_orient for f in fits]),
        hand_pose=np.stack([f.hand_pose for f in fits]),
        betas=np.stack([f.betas for f in fits]),
        transl=np.stack([f.transl for f in fits]),
        status=np.array([f.status for f in fits]),
        rms_mm=np.array([f.rms_mm for f in fits], dtype=np.float64),
    )


def load_hand_fits(path) -> List[HandFit]:
    """Read fits written by `save_hand_fits`."""
    data = np.load(path)
    backend, side = str(data["backend"]), str(data["side"])
    return [
        HandFit(
            backend, side, data["wrist_orient"][t], data["hand_pose"][t], data["betas"][t],
            data["transl"][t], str(data["status"][t]), float(data["rms_mm"][t]),
        )
        for t in range(len(data["status"]))
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_converter.py -q`
Expected: all pass, none skipped. If the synthetic gate (`test_recovers_hands_the_model_can_represent`) misses 0.5 mm / 1 mm for any backend, side or pose set, do NOT loosen it and do not raise iteration counts blindly: report the measured mean/p95 per case and return DONE_WITH_CONCERNS. That gate is a plan acceptance criterion.

- [ ] **Step 5: Lint, run the whole suite once, commit**

```bash
<python> -m flake8 --max-line-length 100 src/kinetic_augment/body_model/hand_converter.py tests/test_hand_converter.py
PYTHONPATH=src <python> -m pytest -q
git add src/kinetic_augment/body_model/hand_converter.py tests/test_hand_converter.py
git commit -m "feat(hands): refine hand fits through the real model; parameter-only decoder"
```

---

### Task 4: Hand shape — one per track, and a per-frame diagnostic

One shape per hand track, fitted jointly with every frame's pose and confined to the plausible region of the shape prior — the bound V1 used for the body, `|betas| ≤ sqrt(chi2.ppf(0.95, K))` (plan §5.1 step 5; assumption A012). Plus a diagnostic that lets every frame take its own plausible shape: the feasibility probe found MediaPipe's hand size changes by roughly ±30% from frame to frame, and comparing the two measures how much of the round-trip error that accounts for. The per-track shape remains the pipeline's contract.

**Files:**
- Modify: `src/kinetic_augment/body_model/hand_converter.py`
- Modify: `tests/test_hand_converter.py` (append)

**Interfaces:**
- Consumes: `fit_hands`, `_prepared`, `_rms_mm`, `_optimise`, `HandFit`, `POSE_PRIOR`, `FIT_DTYPE`, `MAX_ITERATIONS` (Task 3); `HandModel.points / rest_points / num_betas` (Task 1).
- Produces (Task 6 relies on these):
  - `plausible_radius(num_betas: int, plausibility: float = SHAPE_PLAUSIBILITY) -> float`
  - `fit_hand_shape(observed: (T,21,3), valid: (T,21), model, plausibility=SHAPE_PLAUSIBILITY) -> np.ndarray` `(K,)`
  - `fit_hands_per_frame_shape(observed, valid, model, plausibility=SHAPE_PLAUSIBILITY) -> List[HandFit]` — each fit carries its own betas
  - `_optimise(model, observed, valid, start, betas=None, shape=None, plausibility=SHAPE_PLAUSIBILITY) -> (params (F, 48), betas (1 or F, K), rms_mm (F,))` — replaces Task 3's version
  - constants `SHAPE_PLAUSIBILITY = 0.95`, `SHAPE_ITERATIONS = 1000`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hand_converter.py`; extend the converter import with `fit_hand_shape, fit_hands_per_frame_shape, plausible_radius`.

```python
def test_plausible_radius_is_the_chi_square_quantile():
    assert plausible_radius(10) == pytest.approx(4.2786732, abs=1e-6)
    assert plausible_radius(2) == pytest.approx(2.4477468, abs=1e-6)


def unit_shape(model, seed, norm=2.0):
    betas = np.random.default_rng(seed).normal(size=model.num_betas)
    return betas * norm / np.linalg.norm(betas)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_fitted_shape_reproduces_a_plausible_hand(backend):
    model = hand(backend, "right")
    observed = synthetic_track(model, 8, seed=32, betas=unit_shape(model, 31))
    betas = fit_hand_shape(observed, all_valid(observed), model)
    rebuilt = rebuild(fit_hands(observed, all_valid(observed), model, betas), model)
    errors = landmark_errors_mm(rebuilt, observed)
    assert errors.mean() <= 0.5
    assert np.percentile(errors, 95) <= 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_shape_never_leaves_the_plausible_region(backend):
    model = hand(backend, "right")
    radius = plausible_radius(model.num_betas)
    extreme = np.zeros(model.num_betas)
    extreme[0] = 3.0 * radius
    observed = synthetic_track(model, 6, seed=33, betas=extreme)
    assert np.linalg.norm(fit_hand_shape(observed, all_valid(observed), model)) <= radius


@requires_mano
def test_invalid_landmarks_never_influence_the_shape():
    model = hand("mano", "right")
    observed = synthetic_track(model, 4, seed=34)
    valid = all_valid(observed)
    valid[:, [4, 12]] = False
    corrupted = observed.copy()
    corrupted[:, [4, 12]] -= 0.02
    np.testing.assert_array_equal(
        fit_hand_shape(observed, valid, model), fit_hand_shape(corrupted, valid, model)
    )


@requires_mano
def test_with_nothing_to_fit_the_shape_stays_neutral():
    model = hand("mano", "right")
    observed = synthetic_track(model, 2, seed=35)
    no_landmarks = np.zeros(observed.shape[:2], dtype=bool)
    np.testing.assert_array_equal(fit_hand_shape(observed, no_landmarks, model), np.zeros(10))


@pytest.mark.parametrize("backend", BACKENDS)
def test_per_frame_shapes_follow_a_hand_that_changes_size(backend):
    model = hand(backend, "right")
    observed = np.concatenate(
        [synthetic_track(model, 1, seed=40 + k, betas=unit_shape(model, 50 + k)) for k in range(4)]
    )
    fits = fit_hands_per_frame_shape(observed, all_valid(observed), model)
    assert [f.status for f in fits] == ["ok"] * 4
    errors = landmark_errors_mm(rebuild(fits, model), observed)
    assert errors.mean() <= 0.5
    assert np.percentile(errors, 95) <= 1.0
    radius = plausible_radius(model.num_betas)
    assert all(np.linalg.norm(f.betas) <= radius for f in fits)
    assert len({tuple(np.round(f.betas, 6)) for f in fits}) == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_converter.py -q`
Expected: collection error — `ImportError: cannot import name 'fit_hand_shape'`.

- [ ] **Step 3: Implement**

In `hand_converter.py`:

1. Add `from scipy.stats import chi2` to the imports, and next to the other constants:

```python
SHAPE_PLAUSIBILITY = 0.95
SHAPE_ITERATIONS = 1000
```

2. Add this paragraph to the module docstring, after the `fit_hands` paragraph:

```text
Shape is per hand track (`fit_hand_shape`): one plausible betas vector fitted
jointly with every frame's pose. `fit_hands_per_frame_shape` lets each frame
take its own plausible shape; it is a diagnostic of how much error comes from
the detector's frame-to-frame size changes, not the pipeline's contract.
```

3. Replace Task 3's `_optimise` with this version, which can also fit the shape:

```python
def plausible_radius(num_betas: int, plausibility: float = SHAPE_PLAUSIBILITY) -> float:
    """The |betas| bound holding `plausibility` of the shape prior N(0, I)."""
    return float(np.sqrt(chi2.ppf(plausibility, num_betas)))


def _optimise(model, observed, valid, start, betas=None, shape=None,
              plausibility=SHAPE_PLAUSIBILITY):
    """
    L-BFGS through the real forward pass on the wrist-relative landmark error (mm).
    The frames are independent unless they share a fitted shape.

    Args:
        observed, valid: (F, 21, 3) and (F, 21) for the frames being fitted.
        start: (F, 48) starting wrist_orient and hand_pose.
        betas: (K,) fixed shape, used when `shape` is None.
        shape: None, "shared" (one fitted shape for all frames) or "per_frame" (one
            fitted shape per frame). Fitted shapes stay inside |betas| <= plausible_radius
            by construction: betas = r u / sqrt(1 + |u|^2) for unconstrained u.

    Returns (params (F, 48), betas (1 or F, K), rms_mm (F,)).
    """
    kept = np.where(valid[..., None], observed, 0.0)
    target = torch.as_tensor(1000.0 * (kept - kept[:, :1]), dtype=FIT_DTYPE)
    mask = torch.tensor(valid)
    mask[:, WRIST] = False  # zero by construction
    params = torch.tensor(start, dtype=FIT_DTYPE, requires_grad=True)

    if shape is None:
        fixed = torch.as_tensor(betas, dtype=FIT_DTYPE).reshape(1, -1)
        variables, iterations = [params], MAX_ITERATIONS

        def current_betas():
            return fixed
    else:
        radius = plausible_radius(model.num_betas, plausibility)
        rows = len(start) if shape == "per_frame" else 1
        free = torch.zeros(rows, model.num_betas, dtype=FIT_DTYPE, requires_grad=True)
        variables, iterations = [params, free], SHAPE_ITERATIONS

        def current_betas():
            return radius * free / torch.sqrt(1.0 + (free ** 2).sum(dim=1, keepdim=True))

    optimiser = torch.optim.LBFGS(
        variables, lr=1.0, max_iter=iterations, tolerance_grad=1e-12,
        tolerance_change=1e-15, history_size=100, line_search_fn="strong_wolfe",
    )

    def residuals():
        points = 1000.0 * model.points(params[:, :3], params[:, 3:], current_betas())
        return (points - points[:, :1]) - target

    def closure():
        optimiser.zero_grad()
        loss = (residuals()[mask] ** 2).sum() + POSE_PRIOR * (params[:, 3:] ** 2).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        fitted = params.detach().numpy().copy()
        return fitted, current_betas().numpy().copy(), _rms_mm(residuals(), mask)
```

4. In `fit_hands`, change the `_optimise` call to unpack three values:

```python
        params, _, rms = _optimise(
            model, observed[batch], valid[batch], np.stack([starts[t] for t in batch]), betas
        )
```

5. Append:

```python
def _starts_from(fits, frames) -> np.ndarray:
    return np.stack([np.concatenate([fits[t].wrist_orient, fits[t].hand_pose]) for t in frames])


def fit_hand_shape(observed, valid, model, plausibility=SHAPE_PLAUSIBILITY) -> np.ndarray:
    """
    One shape for a whole hand track: the plausible betas minimising the summed
    landmark error of every frame, fitted jointly with the frames' poses (started
    from neutral-shape fits). Frames the neutral fit cannot place are left out.

    Returns (K,) betas; neutral when no frame can be fitted.
    """
    observed, valid = _prepared(observed, valid)
    neutral = fit_hands(observed, valid, model)
    frames = [t for t, fit in enumerate(neutral) if fit.status == "ok"]
    if not frames:
        return np.zeros(model.num_betas)
    _, betas, _ = _optimise(model, observed[frames], valid[frames], _starts_from(neutral, frames),
                            shape="shared", plausibility=plausibility)
    return betas[0]


def fit_hands_per_frame_shape(observed, valid, model,
                              plausibility=SHAPE_PLAUSIBILITY) -> List[HandFit]:
    """
    Diagnostic, not the pipeline's contract: like `fit_hands`, but every frame takes
    its own plausible shape. Comparing it with a per-track shape measures how much
    of the round-trip error is the detector's frame-to-frame change of hand size.
    """
    observed, valid = _prepared(observed, valid)
    fits = fit_hands(observed, valid, model)
    frames = [t for t, fit in enumerate(fits) if fit.status == "ok"]
    if not frames:
        return fits
    params, betas, rms = _optimise(model, observed[frames], valid[frames],
                                   _starts_from(fits, frames), shape="per_frame",
                                   plausibility=plausibility)
    for k, t in enumerate(frames):
        if not (np.isfinite(rms[k]) and np.isfinite(params[k]).all()):
            fits[t] = _unfitted(model, "diverged")
            continue
        wrist = model.rest_points(betas[k])[WRIST]
        fits[t] = HandFit(model.backend, model.side, params[k, :3], params[k, 3:], betas[k],
                          observed[t, WRIST] - wrist, "ok", float(rms[k]))
    return fits
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_converter.py -q`
Expected: all pass, none skipped. As in Task 3, do not loosen the 0.5 mm / 1 mm gates; report measured values with DONE_WITH_CONCERNS if one misses.

- [ ] **Step 5: Lint, run the whole suite once, commit**

```bash
<python> -m flake8 --max-line-length 100 src/kinetic_augment/body_model/hand_converter.py tests/test_hand_converter.py
PYTHONPATH=src <python> -m pytest -q
git add src/kinetic_augment/body_model/hand_converter.py tests/test_hand_converter.py
git commit -m "feat(hands): fit one plausible hand shape per track, with a per-frame diagnostic"
```

---
### Task 5: Hand observation extraction (plan P0)

Records what MediaPipe sees of each hand, per frame and anatomical side, with the paired pose (plan §3.1; assumptions A003, A004, A018). Facts this task relies on, measured by the feasibility probe (`docs/V1_2_Feasibility_Probes.md`):

- Only legacy `mp.solutions.hands.Hands` returns metric hand world landmarks (`multi_hand_world_landmarks`); legacy `Holistic` has none, and the Tasks API needs `.task` model files that are not installed.
- `multi_hand_landmarks[i]`, `multi_hand_world_landmarks[i]` and `multi_handedness[i]` are index-aligned.
- Hand-world axes equal pose-world axes (x right, y down, z away from the camera); the hand-world origin is near the knuckles, not the wrist.
- On this unmirrored video the detector's handedness label names the other hand in 98.7% of detections — it assumes selfie input.
- A detection's own image wrist lies 0.1–3% of the image from its pose wrist and 7–27% from the other; a few frames hold two detections of one hand.

**Files:**
- Create: `src/kinetic_augment/body_model/hand_observations.py`
- Create: `scripts/extract_hand_landmarks.py`
- Create: `tests/test_hand_observations.py`, `tests/test_extract_hand_landmarks.py`
- Create (generated, committed): `data/test_output/{01986,01987,01988,01991,01992}_hands.npz`
- Modify: `src/kinetic_augment/body_model/index.md`, `scripts/index.md`, `tests/index.md`, `data/test_output/index.md`

**Interfaces:**
- Consumes: `NUM_HAND_LANDMARKS` from `hand_model` (Task 1), in the extractor.
- Produces (Task 6 relies on these):
  - `HandClip` dataclass — fields `world (T,2,21,3)`, `image (T,2,21,3)`, `present (T,2)`, `handedness_score (T,2)`, `label_agrees (T,2)`, `unassigned (T,)`, `pose_world (T,33,3)`, `pose_image (T,33,3)`, `pose_visibility (T,33)`, `timestamp_ms (T,)`, `width`, `height`, `fps`, `meta`
  - `save_hand_clip(path, clip)`, `load_hand_clip(path) -> HandClip`
  - `assign_sides(hand_wrists: (D,2), pose_wrists: (2,2), max_gap: float, margin: float) -> (D,) int`
  - constants `LEFT = 0`, `RIGHT = 1`, `SIDES = ("left", "right")`, `POSE_WRISTS = (15, 16)`, `MAX_WRIST_GAP = 0.10`, `SIDE_MARGIN = 0.03`, `LABEL_SIDE_UNMIRRORED`, `HAND_CLIP_SCHEMA = 1`, `NUM_POSE_LANDMARKS = 33`
  - `scripts/extract_hand_landmarks.py`: `extract_all(videos, output_dir)`, `extract_hands(video_path, hands, holistic) -> HandClip`, `observe_frame(hand_result, holistic_result, aspect) -> dict`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_hand_observations.py`:

```python
"""Tests for the hand clip record and the geometric left/right assignment (V1.2 plan, P0)."""

import json

import numpy as np
import pytest

from kinetic_augment.body_model.hand_observations import (
    HAND_CLIP_SCHEMA,
    LEFT,
    RIGHT,
    HandClip,
    assign_sides,
    load_hand_clip,
    save_hand_clip,
)

# The signer faces the camera, so their left wrist is on the image's right.
POSE = np.array([[0.70, 0.60], [0.30, 0.60]])


def sides(hands, pose=POSE):
    return assign_sides(np.array(hands, dtype=float), pose, max_gap=0.10, margin=0.03).tolist()


class TestAssignSides:
    def test_a_hand_takes_the_side_of_the_nearest_pose_wrist(self):
        assert sides([[0.71, 0.61]]) == [LEFT]
        assert sides([[0.29, 0.58]]) == [RIGHT]

    def test_detection_order_carries_no_side(self):
        assert sides([[0.31, 0.60], [0.69, 0.60]]) == [RIGHT, LEFT]
        assert sides([[0.69, 0.60], [0.31, 0.60]]) == [LEFT, RIGHT]

    def test_a_second_detection_of_one_hand_is_not_assigned(self):
        assert sides([[0.72, 0.60], [0.705, 0.60]]) == [-1, LEFT]

    def test_hands_too_close_to_tell_apart_are_not_assigned(self):
        together = np.array([[0.52, 0.60], [0.48, 0.60]])
        assert sides([[0.505, 0.60]], together) == [-1]

    def test_a_hand_far_from_every_pose_wrist_is_not_assigned(self):
        assert sides([[0.70, 0.10]]) == [-1]

    def test_only_a_visible_pose_wrist_can_take_a_hand(self):
        one_wrist = np.array([[0.70, 0.60], [np.nan, np.nan]])
        assert sides([[0.70, 0.60], [0.30, 0.60]], one_wrist) == [LEFT, -1]
        assert sides([[0.70, 0.60]], np.full((2, 2), np.nan)) == [-1]

    def test_no_detections(self):
        assert sides(np.zeros((0, 2))) == []


def tiny_clip(frames=3):
    rng = np.random.default_rng(0)
    world = np.full((frames, 2, 21, 3), np.nan, dtype=np.float32)
    world[:, RIGHT] = rng.normal(0.0, 0.05, (frames, 21, 3))
    image = np.full_like(world, np.nan)
    image[:, RIGHT] = rng.uniform(0.0, 1.0, (frames, 21, 3))
    present = np.zeros((frames, 2), dtype=bool)
    present[:, RIGHT] = True
    score = np.where(present, 0.9, np.nan).astype(np.float32)
    return HandClip(
        world=world, image=image, present=present, handedness_score=score,
        label_agrees=present.copy(), unassigned=np.zeros(frames, dtype=np.int8),
        pose_world=rng.normal(0.0, 0.3, (frames, 33, 3)).astype(np.float32),
        pose_image=rng.uniform(0.0, 1.0, (frames, 33, 3)).astype(np.float32),
        pose_visibility=np.ones((frames, 33), dtype=np.float32),
        timestamp_ms=np.arange(frames) * 40.0, width=320, height=240, fps=25.0,
        meta={"source": "test"},
    )


class TestHandClipFile:
    ARRAYS = ("world", "image", "present", "handedness_score", "label_agrees", "unassigned",
              "pose_world", "pose_image", "pose_visibility", "timestamp_ms")

    def test_a_saved_clip_loads_back_unchanged(self, tmp_path):
        clip = tiny_clip()
        save_hand_clip(tmp_path / "x_hands.npz", clip)
        back = load_hand_clip(tmp_path / "x_hands.npz")
        for name in self.ARRAYS:
            np.testing.assert_array_equal(getattr(back, name), getattr(clip, name))
        assert (back.width, back.height, back.fps) == (320, 240, 25.0)
        assert back.meta == {"source": "test", "schema": HAND_CLIP_SCHEMA}

    def test_an_unknown_schema_is_refused(self, tmp_path):
        path = tmp_path / "x_hands.npz"
        save_hand_clip(path, tiny_clip())
        raw = dict(np.load(path))
        raw["meta"] = np.array(json.dumps({"schema": HAND_CLIP_SCHEMA + 1}))
        np.savez(path, **raw)
        with pytest.raises(ValueError):
            load_hand_clip(path)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_observations.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'kinetic_augment.body_model.hand_observations'`.

- [ ] **Step 3: Implement `hand_observations.py`**

Create `src/kinetic_augment/body_model/hand_observations.py`:

```python
"""
MediaPipe hand observations for the V1.2 hand round trip.

A hand clip holds, for every decoded video frame and each anatomical side
(LEFT = the signer's left, RIGHT = their right), MediaPipe's 21 hand world
landmarks -- metres, camera-aligned axes (x right, y down, z away from the
camera), origin near the knuckles -- and its 21 image landmarks, with the pose
landmarks of the same frame. Missing data is NaN, never zero.

A detection's side comes from geometry: the pose wrist (15 = left, 16 = right)
nearest the detection's own image wrist. Detection order carries no side, and
the legacy detector's handedness label assumes mirrored (selfie) input -- on
unmirrored video it names the other hand -- so the label is kept and compared
(`label_agrees`) but never used to assign.
"""

import json
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

HAND_CLIP_SCHEMA = 1
NUM_POSE_LANDMARKS = 33
LEFT, RIGHT = 0, 1
SIDES = ("left", "right")
POSE_WRISTS = (15, 16)  # the pose's left and right wrists
MAX_WRIST_GAP = 0.10    # fraction of image height
SIDE_MARGIN = 0.03      # fraction of image height
# What the legacy detector's label means on unmirrored video: the opposite hand.
LABEL_SIDE_UNMIRRORED = {"Left": RIGHT, "Right": LEFT}

_ARRAYS = ("world", "image", "present", "handedness_score", "label_agrees", "unassigned",
           "pose_world", "pose_image", "pose_visibility", "timestamp_ms")


@dataclass
class HandClip:
    """One clip's hand observations; T = decoded frames, sides indexed LEFT / RIGHT."""

    world: np.ndarray             # (T, 2, 21, 3) float32 metres; NaN where that side has no hand
    image: np.ndarray             # (T, 2, 21, 3) float32: x / width, y / height, relative z
    present: np.ndarray           # (T, 2) bool: a detection was assigned to that side
    handedness_score: np.ndarray  # (T, 2) float32 detector handedness score; NaN where absent
    label_agrees: np.ndarray      # (T, 2) bool: the label, read as unmirrored, named that side
    unassigned: np.ndarray        # (T,) int8: detections given no side
    pose_world: np.ndarray        # (T, 33, 3) float32 metres, hip-centred; NaN where no pose
    pose_image: np.ndarray        # (T, 33, 3) float32 normalised image coordinates; NaN, no pose
    pose_visibility: np.ndarray   # (T, 33) float32; 0 where no pose
    timestamp_ms: np.ndarray      # (T,) float64, from the video decoder
    width: int
    height: int
    fps: float
    meta: Dict = field(default_factory=dict)  # detector version and settings, source hash


def assign_sides(hand_wrists, pose_wrists, max_gap: float, margin: float) -> np.ndarray:
    """
    Anatomical side of each detected hand, from where its wrist lies in the image.

    Pose wrists take their nearest detections, nearest pair first. A detection stays
    unassigned (-1) when it is farther than `max_gap` from that pose wrist, when it
    is not at least `margin` nearer to it than to the other wrist (hands too close to
    tell apart), or when that side is already taken -- a second detection of one hand.

    Args:
        hand_wrists: (D, 2) image positions of the detections' wrists.
        pose_wrists: (2, 2) image positions of pose wrists 15 (left) and 16 (right);
            NaN where the pose is missing.
        max_gap, margin: distances in the same isotropic units as the positions.

    Returns:
        (D,) int: LEFT, RIGHT or -1.
    """
    hand_wrists = np.asarray(hand_wrists, dtype=np.float64).reshape(-1, 2)
    pose_wrists = np.asarray(pose_wrists, dtype=np.float64).reshape(2, 2)
    distance = np.linalg.norm(hand_wrists[:, None, :] - pose_wrists[None, :, :], axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)

    sides = np.full(len(hand_wrists), -1)
    for flat in np.argsort(distance, axis=None, kind="stable"):
        detection, side = np.unravel_index(flat, distance.shape)
        if sides[detection] != -1 or side in sides or distance[detection, side] > max_gap:
            continue
        if distance[detection, 1 - side] - distance[detection, side] < margin:
            continue
        sides[detection] = side
    return sides


def save_hand_clip(path, clip: HandClip) -> None:
    """Write a clip to .npz, stamping the schema version into its metadata."""
    meta = dict(clip.meta, schema=HAND_CLIP_SCHEMA)
    np.savez_compressed(
        path,
        **{name: getattr(clip, name) for name in _ARRAYS},
        width=np.array(clip.width),
        height=np.array(clip.height),
        fps=np.array(clip.fps),
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )


def load_hand_clip(path) -> HandClip:
    """
    Read a clip written by `save_hand_clip`.

    Raises:
        ValueError: the file's schema is not HAND_CLIP_SCHEMA.
    """
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    if meta.get("schema") != HAND_CLIP_SCHEMA:
        raise ValueError(f"{path}: hand clip schema {meta.get('schema')}, "
                         f"expected {HAND_CLIP_SCHEMA}")
    return HandClip(
        **{name: data[name] for name in _ARRAYS},
        width=int(data["width"]), height=int(data["height"]), fps=float(data["fps"]), meta=meta,
    )
```


- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `PYTHONPATH=src <python> -m pytest tests/test_hand_observations.py -q`
Expected: all pass.

- [ ] **Step 5: Write the failing integration tests**

Create `tests/test_extract_hand_landmarks.py`:

```python
"""
Integration tests for the hand extractor: real MediaPipe on the pilot clips.
Skipped when mediapipe or the videos are absent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mediapipe")
pytest.importorskip("torch")  # the extractor imports hand_model, which needs torch

REPO = Path(__file__).resolve().parents[1]
VIDEOS = REPO / "data" / "sign_videos"
sys.path.insert(0, str(REPO / "scripts"))

from extract_hand_landmarks import extract_all  # noqa: E402

from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402

needs_videos = pytest.mark.skipif(
    not (VIDEOS / "01991.mp4").exists() or not (VIDEOS / "01992.mp4").exists(),
    reason="pilot videos absent",
)


@needs_videos
def test_a_clip_does_not_depend_on_the_clip_before_it(tmp_path):
    extract_all([VIDEOS / "01991.mp4", VIDEOS / "01992.mp4"], tmp_path / "after")
    extract_all([VIDEOS / "01992.mp4"], tmp_path / "alone")
    after = load_hand_clip(tmp_path / "after" / "01992_hands.npz")
    alone = load_hand_clip(tmp_path / "alone" / "01992_hands.npz")
    for name in ("world", "image", "present", "pose_world"):
        np.testing.assert_array_equal(getattr(after, name), getattr(alone, name))


@needs_videos
def test_hands_are_metric_sided_and_missing_as_nan(tmp_path):
    extract_all([VIDEOS / "01992.mp4"], tmp_path)
    clip = load_hand_clip(tmp_path / "01992_hands.npz")
    frames = len(clip.present)
    assert clip.world.shape == (frames, 2, 21, 3)
    assert clip.image.shape == (frames, 2, 21, 3)
    assert clip.present.any(axis=0).all()                 # both hands appear in this clip
    assert np.isnan(clip.world[~clip.present]).all()      # an absent side is NaN, never zeros
    assert np.isfinite(clip.world[clip.present]).all()
    seen = clip.world[clip.present]
    span = np.linalg.norm(seen[:, 9] - seen[:, 0], axis=-1)
    assert 0.05 < np.median(span) < 0.13                  # wrist to middle MCP, in metres
    assert clip.label_agrees[clip.present].mean() > 0.9   # the label, read as unmirrored
    assert clip.meta["mediapipe"] and clip.meta["source"] == "01992.mp4"
```

- [ ] **Step 6: Run them to verify they fail**

Run: `PYTHONPATH=src <python> -m pytest tests/test_extract_hand_landmarks.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'extract_hand_landmarks'`.

- [ ] **Step 7: Implement the extractor**

Create `scripts/extract_hand_landmarks.py`:

```python
#!/usr/bin/env python3
"""
Extract MediaPipe hand landmarks for the V1.2 hand round trip.

Per clip, a fresh legacy `Hands` instance (video mode, up to two hands) gives each
detected hand's 21 world landmarks -- the only installed MediaPipe API returning
metric hand coordinates -- and its 21 image landmarks. A fresh `Holistic`
instance on the same frame gives the pose, whose image wrists decide each hand's
anatomical side (`hand_observations.assign_sides`). Fresh instances per clip keep
a clip's landmarks independent of the clip processed before it.

Output: data/test_output/<clip>_hands.npz (see `HandClip`). The V1 body caches are
not touched.

Usage:
    python scripts/extract_hand_landmarks.py
"""

import hashlib
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # noqa: E402 -- must follow the log-level setting above
import numpy as np  # noqa: E402

from kinetic_augment.body_model.hand_model import NUM_HAND_LANDMARKS  # noqa: E402
from kinetic_augment.body_model.hand_observations import (  # noqa: E402
    LABEL_SIDE_UNMIRRORED,
    MAX_WRIST_GAP,
    NUM_POSE_LANDMARKS,
    POSE_WRISTS,
    SIDE_MARGIN,
    SIDES,
    HandClip,
    assign_sides,
    save_hand_clip,
)

VIDEO_DIR = Path("data/sign_videos")
OUTPUT_DIR = Path("data/test_output")
HANDS_SETTINGS = dict(static_image_mode=False, max_num_hands=2, model_complexity=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
HOLISTIC_SETTINGS = dict(static_image_mode=False, model_complexity=1)


def _points(landmark_list) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmark_list.landmark], dtype=np.float32)


def observe_frame(hand_result, holistic_result, aspect: float) -> dict:
    """One frame's hand and pose observations, each hand on its anatomical side."""
    obs = {
        "world": np.full((2, NUM_HAND_LANDMARKS, 3), np.nan, dtype=np.float32),
        "image": np.full((2, NUM_HAND_LANDMARKS, 3), np.nan, dtype=np.float32),
        "present": np.zeros(2, dtype=bool),
        "handedness_score": np.full(2, np.nan, dtype=np.float32),
        "label_agrees": np.zeros(2, dtype=bool),
        "unassigned": 0,
        "pose_world": np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32),
        "pose_image": np.full((NUM_POSE_LANDMARKS, 3), np.nan, dtype=np.float32),
        "pose_visibility": np.zeros(NUM_POSE_LANDMARKS, dtype=np.float32),
    }
    if holistic_result.pose_landmarks is not None:
        obs["pose_image"] = _points(holistic_result.pose_landmarks)
        obs["pose_visibility"] = np.array(
            [p.visibility for p in holistic_result.pose_landmarks.landmark], dtype=np.float32)
    if holistic_result.pose_world_landmarks is not None:
        obs["pose_world"] = _points(holistic_result.pose_world_landmarks)

    images = hand_result.multi_hand_landmarks or []
    worlds = hand_result.multi_hand_world_landmarks or []
    labels = hand_result.multi_handedness or []
    square = np.array([aspect, 1.0])  # normalised x, y -> fractions of image height
    wrists = np.array([[h.landmark[0].x, h.landmark[0].y] for h in images]).reshape(-1, 2)
    pose_wrists = obs["pose_image"][list(POSE_WRISTS), :2].astype(np.float64)
    sides = assign_sides(wrists * square, pose_wrists * square, MAX_WRIST_GAP, SIDE_MARGIN)

    for image, world, label, side in zip(images, worlds, labels, sides):
        if side < 0:
            obs["unassigned"] += 1
            continue
        best = label.classification[0]
        obs["world"][side] = _points(world)
        obs["image"][side] = _points(image)
        obs["present"][side] = True
        obs["handedness_score"][side] = best.score
        obs["label_agrees"][side] = LABEL_SIDE_UNMIRRORED.get(best.label) == side
    return obs


def extract_hands(video_path: Path, hands, holistic) -> HandClip:
    """Run both detectors over every frame of one clip."""
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    rows, stamps = [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        stamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rows.append(observe_frame(hands.process(rgb), holistic.process(rgb), width / height))
    capture.release()
    if not rows:
        raise ValueError(f"{video_path}: no frames decoded")

    stacked = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    return HandClip(
        world=stacked["world"], image=stacked["image"], present=stacked["present"],
        handedness_score=stacked["handedness_score"], label_agrees=stacked["label_agrees"],
        unassigned=stacked["unassigned"].astype(np.int8), pose_world=stacked["pose_world"],
        pose_image=stacked["pose_image"], pose_visibility=stacked["pose_visibility"],
        timestamp_ms=np.array(stamps, dtype=np.float64), width=width, height=height, fps=fps,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_all(videos, output_dir) -> None:
    """Extract each clip with its own Hands and Holistic instances."""
    import mediapipe as mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'clip':<8}{'frames':>8}{'left':>7}{'right':>7}{'unassigned':>12}{'label agrees':>14}")
    for video in videos:
        video = Path(video)
        with mp.solutions.hands.Hands(**HANDS_SETTINGS) as hands, \
                mp.solutions.holistic.Holistic(**HOLISTIC_SETTINGS) as holistic:
            clip = extract_hands(video, hands, holistic)
        clip.meta = {
            "source": video.name,
            "source_sha256": _sha256(video),
            "mediapipe": mp.__version__,
            "hands_settings": HANDS_SETTINGS,
            "holistic_settings": HOLISTIC_SETTINGS,
            "side_rule": {"max_wrist_gap": MAX_WRIST_GAP, "side_margin": SIDE_MARGIN,
                          "units": "fraction of image height"},
            "detector_state": "fresh Hands and Holistic instances per clip",
        }
        save_hand_clip(output_dir / f"{video.stem}_hands.npz", clip)
        agrees = clip.label_agrees[clip.present].mean() if clip.present.any() else float("nan")
        print(f"{video.stem:<8}{len(clip.present):>8}{clip.present[:, 0].sum():>7}"
              f"{clip.present[:, 1].sum():>7}{clip.unassigned.sum():>12}{100 * agrees:>13.1f}%")


def main() -> None:
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No videos found under {VIDEO_DIR}")
    extract_all(videos, OUTPUT_DIR)
    print(f"\nWrote {len(videos)} files to {OUTPUT_DIR}/*_hands.npz (sides: {', '.join(SIDES)})")


if __name__ == "__main__":
    main()
```


- [ ] **Step 8: Run the integration tests to verify they pass, then extract the pilot clips**

Run: `PYTHONPATH=src <python> -m pytest tests/test_extract_hand_landmarks.py -q`
Expected: 2 passed (allow about a minute).

Then, from the worktree root: `PYTHONPATH=src <python> scripts/extract_hand_landmarks.py`
Expected: a table with one row per clip (01986, 01987, 01988, 01991, 01992); 410 frames in total; `label agrees` near 100% except 01987 (about 90%); and five new `data/test_output/*_hands.npz` files. Paste the table into your report.

- [ ] **Step 9: Update the indexes**

- `src/kinetic_augment/body_model/index.md`: `- \`hand_observations.py\` — Holds per-clip MediaPipe hand observations and assigns each detected hand its anatomical side.`
- `scripts/index.md`: `- \`extract_hand_landmarks.py\` — Extracts MediaPipe hand world and image landmarks with the paired pose and a geometric left/right assignment.`
- `tests/index.md`: `- \`test_hand_observations.py\` — Tests the left/right assignment and the hand clip file format.` and `- \`test_extract_hand_landmarks.py\` — Runs the hand extractor on pilot clips: clip-order independence, units, sides.`
- `data/test_output/index.md`: `- \`*_hands.npz\` — MediaPipe hand world/image landmarks per anatomical side with the paired pose, for the V1.2 hand round trip.`

- [ ] **Step 10: Lint, run the whole suite once, commit**

```bash
<python> -m flake8 --max-line-length 100 src/kinetic_augment/body_model/hand_observations.py scripts/extract_hand_landmarks.py tests/test_hand_observations.py tests/test_extract_hand_landmarks.py
PYTHONPATH=src <python> -m pytest -q
git add src/kinetic_augment/body_model/hand_observations.py scripts/extract_hand_landmarks.py tests/test_hand_observations.py tests/test_extract_hand_landmarks.py data/test_output/01986_hands.npz data/test_output/01987_hands.npz data/test_output/01988_hands.npz data/test_output/01991_hands.npz data/test_output/01992_hands.npz src/kinetic_augment/body_model/index.md scripts/index.md tests/index.md data/test_output/index.md
git commit -m "feat(hands): extract MediaPipe hand landmarks with geometric left/right"
```

---
### Task 6: Pilot round-trip measurement (plan P3)

Runs the full round trip on every pilot clip and side, for both backends, and reports an error ladder that separates optimiser error from model mismatch (plan §6 "Separate optimizer error from representational mismatch"; assumptions A009–A012).

**Files:**
- Create: `scripts/measure_hand_round_trip.py`
- Create (generated, committed): `data/test_output/hand_round_trip.json`
- Modify: `scripts/index.md`, `data/test_output/index.md`

**Interfaces:**
- Consumes: `HandModel`, `FINGERS`, `HAND_PARENTS`, `LANDMARK_NAMES`, `POSE_SLOT_LANDMARKS`, `TIPS` (Task 1); `initial_pose` (Task 2); `fit_hands`, `reconstruct_hand`, `landmark_errors_mm` (Task 3); `fit_hand_shape`, `fit_hands_per_frame_shape`, `plausible_radius` (Task 4); `palm_rotation` (Task 2); `SIDES`, `load_hand_clip` and `HandClip.world / present` (Task 5).
- Produces: the report `data/test_output/hand_round_trip.json` (Task 7 reads it) with keys `environment`, `targets_mm`, `tracks` (per clip-side entries) and `pooled`.

The ladder, all scored with `landmark_errors_mm` on identical frames:

| Row | Meaning |
| --- | --- |
| `swing start` | analytic start only, neutral shape |
| `refined` | real-model refinement, neutral shape |
| `held-out shape` | shape fitted on one half of the track, scored on the other half (both ways) |
| `in-sample shape` | shape fitted on the frames it is scored on — optimistic |
| `per-frame shape (diagnostic)` | every frame its own plausible shape — how much error MediaPipe's frame-to-frame size change causes |
| `rigid reference` | model-free: per-track rigid palm template (wrist-anchored) + median bone lengths along the observed bone directions. A reference, not a lower bound |
| `rigid reference, per-frame scale` | the same, scaled per frame by its least-squares factor |

- [ ] **Step 1: Write the script**

Create `scripts/measure_hand_round_trip.py`:

```python
#!/usr/bin/env python3
"""
V1.2 pilot: MediaPipe hand world landmarks -> MANO / SMPL-X -> landmarks.

For every clip and anatomical side, fits each frame's hand, rebuilds its 21
landmarks from the fitted parameters alone, and scores the plan's primary metric:
millimetre distance between predicted and observed wrist-relative positions of
landmarks 1-20, with no rotation or scale registration
(docs/V1_2_Hand_Roundtrip_Plan.md, Section 6).

The ladder separates optimiser error from model mismatch:
    swing start      analytic palm rotation + finger swing, neutral shape
    refined          real-model refinement, neutral shape
    held-out shape   shape fitted on the other half of the track
    in-sample shape  shape fitted on the frames it is scored on (optimistic)
    per-frame shape  every frame its own plausible shape (diagnostic): how much
                     error MediaPipe's frame-to-frame change of hand size causes
    rigid reference  a per-track rigid palm, anchored at the wrist like the fits,
                     and median bone lengths along the observed bone directions;
                     model-free, not a lower bound; also scaled per frame

Also reported per track and backend: how many rotation directions the landmarks
barely constrain (observability), and the error when the other side's model is
used (chirality). Five clips are a pilot, not a held-out test set.

Usage:
    python scripts/extract_hand_landmarks.py   # once
    python scripts/measure_hand_round_trip.py
"""

import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kinetic_augment.body_model.hand_converter import (
    fit_hand_shape,
    fit_hands,
    fit_hands_per_frame_shape,
    initial_pose,
    landmark_errors_mm,
    palm_rotation,
    plausible_radius,
    reconstruct_hand,
)
from kinetic_augment.body_model.hand_model import (
    FINGERS,
    HAND_PARENTS,
    LANDMARK_NAMES,
    POSE_SLOT_LANDMARKS,
    TIPS,
    WRIST,
    HandModel,
)
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip

DATA_DIR = Path("data/test_output")
MODEL_DIR = "models"
OUTPUT = DATA_DIR / "hand_round_trip.json"
BACKENDS = ("mano", "smplx")
TARGETS_MM = {"mean": 5.0, "tips": 8.0, "p95": 15.0}  # plan Section 6, provisional
PALM = [1, 5, 9, 13, 17]  # wrist-relative, so the wrist is the origin
MIDDLE_MCP = 9
OBSERVABILITY_STRIDE = 5
WEAK_FRACTION = 0.01


def tracks():
    """Yield (clip, side, landmarks (n, 21, 3), frames_in_clip) per clip-side, in time order."""
    for path in sorted(DATA_DIR.glob("*_hands.npz")):
        clip = load_hand_clip(path)
        for s, side in enumerate(SIDES):
            frames = np.flatnonzero(clip.present[:, s])
            if len(frames) >= 2:
                landmarks = clip.world[frames, s].astype(np.float64)
                yield path.stem.replace("_hands", ""), side, landmarks, len(clip.present)


def rebuild(fits, model):
    return np.stack([reconstruct_hand(fit, model) for fit in fits])


def start_points(model, landmarks):
    """Landmarks rebuilt from the analytic start alone, wrists placed on the observed ones."""
    skeleton = model.skeleton()
    out = np.full_like(landmarks, np.nan)
    for t, frame in enumerate(landmarks):
        start = initial_pose(frame, np.ones(len(frame), dtype=bool), skeleton)
        if start is not None:
            with torch.no_grad():
                points = model.points(start[0], start[1], np.zeros(model.num_betas))[0].numpy()
            out[t] = points - points[WRIST] + frame[WRIST]
    return out


def rigid_reference(landmarks, per_frame_scale=False, iterations=10):
    """
    A rigid hand built from the track itself, anchored at the wrist like the model fits:
    a wrist-relative palm template (rotation-only Procrustes) and every finger bone at
    its median length along the observed direction. With `per_frame_scale` the whole
    hand is also scaled each frame by its least-squares factor. Model-free; a
    reference, not a lower bound.
    """
    every = np.ones(landmarks.shape[1], dtype=bool)
    relative = landmarks - landmarks[:, :1]
    template = relative[0].copy()
    for _ in range(iterations):
        template = np.mean(
            [frame @ palm_rotation(frame, template, every).T for frame in relative], axis=0)
    rebuilt = np.zeros_like(relative)
    for t, frame in enumerate(relative):
        rebuilt[t, PALM] = template[PALM] @ palm_rotation(template, frame, every).T
    for parent, child in [(int(p), c) for c, p in enumerate(HAND_PARENTS) if p > 0]:
        offset = relative[:, child] - relative[:, parent]
        length = np.median(np.linalg.norm(offset, axis=-1))
        unit = offset / np.linalg.norm(offset, axis=-1, keepdims=True)
        rebuilt[:, child] = rebuilt[:, parent] + length * unit
    if per_frame_scale:
        factor = (rebuilt * relative).sum(axis=(1, 2)) / (rebuilt ** 2).sum(axis=(1, 2))
        rebuilt *= factor[:, None, None]
    return rebuilt + landmarks[:, :1]


def summary(errors_mm, scale_mm):
    """Pooled statistics of (n, 20) errors; `scale_mm` (n,) is each frame's hand scale."""
    finite = np.isfinite(errors_mm)
    tips = [t - 1 for t in TIPS]
    out = {
        "frames": int(np.isfinite(errors_mm).any(axis=1).sum()),
        "mean_mm": float(np.mean(errors_mm[finite])),
        "tips_mm": float(np.nanmean(errors_mm[:, tips])),
        "p95_mm": float(np.percentile(errors_mm[finite], 95)),
        "mean_pct_hand": float(np.nanmean(100.0 * errors_mm / scale_mm[:, None])),
    }
    for name, chain in FINGERS.items():
        out[f"{name}_mm"] = float(np.nanmean(errors_mm[:, [i - 1 for i in chain]]))
    return out


def ladder(model, landmarks):
    """Per-row (n, 20) errors for one track, plus the fitted shapes."""
    valid = np.ones(landmarks.shape[:2], dtype=bool)
    rows = {"swing start": landmark_errors_mm(start_points(model, landmarks), landmarks)}
    neutral = fit_hands(landmarks, valid, model)
    rows["refined"] = landmark_errors_mm(rebuild(neutral, model), landmarks)

    half = len(landmarks) // 2
    held = np.full_like(rows["refined"], np.nan)
    held_norms = []
    halves = (slice(0, half), slice(half, None))
    for fit_on, score_on in (halves, halves[::-1]):
        betas = fit_hand_shape(landmarks[fit_on], valid[fit_on], model)
        held_norms.append(float(np.linalg.norm(betas)))
        fits = fit_hands(landmarks[score_on], valid[score_on], model, betas)
        held[score_on] = landmark_errors_mm(rebuild(fits, model), landmarks[score_on])
    rows["held-out shape"] = held

    in_sample = fit_hand_shape(landmarks, valid, model)
    fits = fit_hands(landmarks, valid, model, in_sample)
    rows["in-sample shape"] = landmark_errors_mm(rebuild(fits, model), landmarks)
    per_frame = fit_hands_per_frame_shape(landmarks, valid, model)
    rows["per-frame shape (diagnostic)"] = landmark_errors_mm(rebuild(per_frame, model), landmarks)
    shapes = {
        "radius": plausible_radius(model.num_betas),
        "held_out_norms": held_norms,
        "in_sample_norm": float(np.linalg.norm(in_sample)),
        "failed_neutral_fits": int(sum(f.status != "ok" for f in neutral)),
    }
    return rows, neutral, shapes


def observability(model, fits):
    """How many of the 48 rotation directions barely move the landmarks; where the weakest is."""
    names = ["wrist"] + [LANDMARK_NAMES[i] for i in POSE_SLOT_LANDMARKS]
    counts, weakest = [], {}
    for fit in [f for f in fits if f.status == "ok"][::OBSERVABILITY_STRIDE]:
        betas = torch.as_tensor(fit.betas, dtype=torch.float64)

        def relative(x):
            points = model.points(x[:3], x[3:], betas)[0]
            return (points[1:] - points[:1]).reshape(-1)

        x0 = torch.as_tensor(np.concatenate([fit.wrist_orient, fit.hand_pose]), dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(relative, x0).numpy()
        _, singular, vt = np.linalg.svd(jacobian)
        counts.append(int((singular < WEAK_FRACTION * singular[0]).sum()))
        joint = names[int(np.argmax(np.linalg.norm(vt[-1].reshape(16, 3), axis=1)))]
        weakest[joint] = weakest.get(joint, 0) + 1
    return {
        "frames": len(counts),
        "median_weak_directions": float(np.median(counts)) if counts else float("nan"),
        "weakest_direction_joint": weakest,
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    assets = [Path(MODEL_DIR) / "mano" / f"MANO_{s}.npz" for s in ("LEFT", "RIGHT")]
    assets.append(Path(MODEL_DIR) / "smplx" / "SMPLX_NEUTRAL.npz")
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "torch", "smplx")},
        "assets": {str(p): sha256(p) for p in assets},
        "data": {str(p): sha256(p) for p in sorted(DATA_DIR.glob("*_hands.npz"))},
    }


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'row':<36}{'frames':>7}{'mean':>8}{'tips':>8}{'p95':>8}{'% hand':>8}")
    for name, s in rows.items():
        print(f"  {name:<36}{s['frames']:>7}{s['mean_mm']:>8.2f}{s['tips_mm']:>8.2f}"
              f"{s['p95_mm']:>8.2f}{s['mean_pct_hand']:>8.2f}")


def main():
    began = time.time()
    found = list(tracks())
    if not found:
        raise SystemExit("No data/test_output/*_hands.npz; run scripts/extract_hand_landmarks.py")

    models = {(b, s): HandModel(b, s, model_dir=MODEL_DIR) for b in BACKENDS for s in SIDES}
    pooled_errors, pooled_scale = {}, {}
    report = {"environment": environment(), "targets_mm": TARGETS_MM, "tracks": [], "pooled": {}}

    for clip, side, landmarks, total in found:
        span = np.linalg.norm(landmarks[:, MIDDLE_MCP] - landmarks[:, WRIST], axis=-1)
        scale = np.full(len(landmarks), 1000.0 * np.median(span))
        entry = {"clip": clip, "side": side, "frames_in_clip": int(total),
                 "frames_with_hand": len(landmarks), "hand_scale_mm": float(scale[0]), "rows": {}}
        errors = {
            "rigid reference": landmark_errors_mm(rigid_reference(landmarks), landmarks),
            "rigid reference, per-frame scale": landmark_errors_mm(
                rigid_reference(landmarks, per_frame_scale=True), landmarks),
        }
        other = "right" if side == "left" else "left"
        for backend in BACKENDS:
            model = models[(backend, side)]
            rows, neutral, shapes = ladder(model, landmarks)
            for row, value in rows.items():
                errors[f"{backend}: {row}"] = value
            valid = np.ones(landmarks.shape[:2], dtype=bool)
            wrong = fit_hands(landmarks, valid, models[(backend, other)])
            entry[backend] = {
                "shape": shapes,
                "observability": observability(model, neutral),
                "chirality_mean_mm": {
                    "own_side": float(np.nanmean(rows["refined"])),
                    "other_side": float(np.nanmean(
                        landmark_errors_mm(rebuild(wrong, models[(backend, other)]), landmarks))),
                },
            }
        for row, value in errors.items():
            entry["rows"][row] = summary(value, scale)
            pooled_errors.setdefault(row, []).append(value)
            pooled_scale.setdefault(row, []).append(scale)
        report["tracks"].append(entry)
        print_table(f"{clip} {side}  ({len(landmarks)}/{total} frames, hand "
                    f"{scale[0]:.0f} mm)", entry["rows"])

    for row in pooled_errors:
        report["pooled"][row] = summary(
            np.concatenate(pooled_errors[row]), np.concatenate(pooled_scale[row]))
        report["pooled"][row]["track_mean_mm"] = float(np.mean(
            [t["rows"][row]["mean_mm"] for t in report["tracks"]]))
    print_table("POOLED over all tracks (mm; % of wrist-to-middle-MCP length)", report["pooled"])
    print(f"\nProvisional targets (plan Section 6): mean <= {TARGETS_MM['mean']} mm, "
          f"tips <= {TARGETS_MM['tips']} mm, p95 <= {TARGETS_MM['p95']} mm")

    OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUTPUT} in {time.time() - began:.0f} s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and keep the output**

Run (from the worktree root): `PYTHONPATH=src <python> scripts/measure_hand_round_trip.py | tee /tmp/hand_round_trip.txt`
Expected: one table per clip-side, then a POOLED table; `data/test_output/hand_round_trip.json` written. Every ladder row appears for both backends. The script must finish without exceptions; it is expected to take a few minutes. Paste the full printed output into your report.

Sanity checks before committing — report each one's outcome:
- `mano: refined` mean is no worse than `mano: swing start` mean on every track (refinement never makes a track worse). Same for `smplx`.
- `in-sample shape` mean ≤ `refined` mean on every track for each backend (a fitted shape includes the neutral one).
- `per-frame shape (diagnostic)` mean ≤ `in-sample shape` mean on every track for each backend (per-frame shapes include the shared one).
- Every `held_out_norms` and `in_sample_norm` is ≤ `radius`.
- `failed_neutral_fits` is 0 on every track, or explain which frames failed and why.
If a check fails, do not edit thresholds or drop frames: report it with the numbers (DONE_WITH_CONCERNS).

- [ ] **Step 3: Update the indexes**

Add to `scripts/index.md`:
`- \`measure_hand_round_trip.py\` — Measures the V1.2 hand round trip (MediaPipe hand → MANO / SMPL-X → landmarks) as an error ladder on the pilot clips.`

Add to `data/test_output/index.md`:
`- \`hand_round_trip.json\` — Per-track and pooled V1.2 hand round-trip errors, observability and environment hashes from \`scripts/measure_hand_round_trip.py\`.`

- [ ] **Step 4: Lint, run the whole suite once, commit**

```bash
<python> -m flake8 --max-line-length 100 scripts/measure_hand_round_trip.py
PYTHONPATH=src <python> -m pytest -q
git add scripts/measure_hand_round_trip.py data/test_output/hand_round_trip.json scripts/index.md data/test_output/index.md
git commit -m "feat(hands): measure the hand round trip on the pilot clips"
```

---

### Task 7: V1.2 findings and assumption-ledger update

Turns the measurements into the record the project keeps: what was proven, what was disproven, what stays open (plan §7 gate for P0–P3; the assumption ledger says "update this ledger from actual check output and attach report paths").

**Files:**
- Create: `docs/V1_2_Findings.md`
- Modify: `docs/V1_2_Assumptions.md`, `docs/V1_2_Hand_Roundtrip_Plan.md` (status line only), `docs/index.md`

**Interfaces:**
- Consumes: `data/test_output/hand_round_trip.json` and its printed tables (Task 6), the extraction summary printed by `scripts/extract_hand_landmarks.py` and its `HandClip.meta` (Task 5), the test suites of Tasks 1–5, and `docs/V1_2_Feasibility_Probes.md` (pre-implementation probe results, committed with this plan).
- Produces: documentation only.

- [ ] **Step 1: Gather the evidence**

Run, from the worktree root, and keep the outputs:
```bash
PYTHONPATH=src <python> -m pytest tests/test_hand_model.py tests/test_hand_converter.py tests/test_hand_observations.py tests/test_extract_hand_landmarks.py -q
PYTHONPATH=src <python> -c "import json; r = json.load(open('data/test_output/hand_round_trip.json')); print(json.dumps(r['pooled'], indent=1))"
```

- [ ] **Step 2: Write `docs/V1_2_Findings.md`**

Use exactly these sections, with every number taken from the evidence above (cite the file each comes from; never round a failing number into a passing one):

1. **The answer** — two or three sentences: the pooled result per backend against the provisional targets (mean ≤ 5 mm, tips ≤ 8 mm, p95 ≤ 15 mm), given as the bracket between the two held-out rows; whether the synthetic gates passed; and how far the per-frame-shape diagnostic gets.
2. **Error ladder** — one table per backend: rows `swing start`, `refined`, `held-out shape (time halves)`, `held-out shape (interleaved frames)`, `in-sample shape`, `per-frame shape (diagnostic)`, plus both rigid references; columns mean, tips, p95, % of hand; pooled and equal-track (`track_mean_mm`).
3. **Per finger and per track** — per-finger means for both held-out rows and the per-track means, with each track's `half_span_ratio`; name the worst finger and the worst track.
4. **What limits the error** — classify the remaining error, with the evidence for each: optimiser (compare `refined` with the synthetic gate results); MediaPipe's change of hand size, within a clip (`half_span_ratio` against the time-halves penalty) and frame to frame (per-frame-shape diagnostic; rigid reference with and without per-frame scale); model mismatch (the ladder against the rigid reference; non-rigid palm, bone-length variation); shape (the two held-out rows against in-sample, norms against the radius); observability (`weak_counts` per threshold, `joint_weak_weight`); side assignment (chirality, own side against the other).
5. **Data** — per-clip hand coverage and the side-assignment statistics from the extraction (assigned, unassigned and why, label agreement), and the hand-world vs pose-world frame finding from the probes.

Interpretation rules — established by the task reviews; the findings must respect them:
- The two held-out rows bracket generalisation. `time halves` is pessimistic: it includes MediaPipe's drift in hand size between the halves of a clip (`half_span_ratio` 0.61–1.55, which correlates 0.89 with the penalty for MANO and 0.73 for SMPL-X). `interleaved frames` is optimistic, because it scores frames adjacent to the ones it was fitted on. Never attribute the time-halves penalty to shape overfitting.
- The fitter minimises summed squared error. A fitted shape therefore never raises that sum, but it can raise the mean distance (2 of 18 track-backends). State the "a fitted shape includes the neutral one" check in squared-error or RMS terms, and do not call those 2 cases optimiser failures.
- Two rows are not achievable pipeline results. `rigid reference, per-frame scale` is an oracle: each frame's scale is chosen from the observations it is scored on. `per-frame shape (diagnostic)` adds 10 shape values per frame, on top of 48 pose values, against 60 observed values. Label both as such. Only the 1-DoF scale reference isolates the effect of size change.
- Every fitted shape, MANO and SMPL-X, sits at 99.57–99.97% of the plausibility bound (all 90 norms), so the shape fits are bound-limited. The interleaved row lands within 0.04 mm of `in-sample shape` on every track, so in practice the lower end of the bracket is the in-sample result. SMPL-X betas are not identifiable from hand landmarks: report them as "a plausible shape", never as a body-shape estimate.
- Observability: exactly 10 rotation directions are numerically null on every track. The weak counts at larger thresholds depend on the threshold. The weak subspace spreads across the finger joints; no single joint is "the weakest".
- Chirality: the other side's model fits only 1.02–1.31× worse on 16 of 18 track-backends. Fit quality cannot verify which hand is which; sides rest on the geometric assignment.
- Batched L-BFGS couples frames. On the synthetic gate data (right hand, random poses), a frame's reconstruction in a batch of 12 differs from its reconstruction alone by at most 0.134 mm (MANO) and 0.139 mm (SMPL-X) in any single coordinate; as a Euclidean distance, a landmark moves by up to 0.172 mm and 0.181 mm (`docs/V1_2_Implementation_Evidence.md` §2, which records the source of each figure). Runs end at the iteration cap. Status `ok` is not a convergence certificate; `rms_mm` is the quality signal.
6. **Assumptions** — a table of every assumption whose status changed (A002–A012, A018, A019 as applicable): ID, new status (`verified` / `open` / `broken`), one-line evidence with the test name or report path.
7. **Limits** — five pilot clips, four people (01991 and 01992 show the same signer), no sealed test set; no wrist attachment (P4); no temporal model (P5); no multi-start best-found reference, renders or robust loss (squared error only); what a small round-trip error does and does not mean (plan §1).
8. **Next steps** — P4 (attachment to the V1 arm via `R_parentᵀ R_wrist`, needs the V1 converter on this branch), corpus growth per plan §8, and only then augmentation and training.
9. **Reproduce** — the exact commands, from extraction to measurement and tests.

- [ ] **Step 3: Update the ledger and the plan**

In `docs/V1_2_Assumptions.md`, change the `Confidence / status` cell of each assumption the evidence settles and append the evidence pointer (test name, report path) to its `Evidence` cell. Do not mark an assumption verified because a test was written for it — only because the test or measurement ran and passed. Add a dated line under the title: `**Updated:** 2026-10-06 from [V1.2 findings](V1_2_Findings.md).`

In `docs/V1_2_Hand_Roundtrip_Plan.md`, replace the line `**Status:** Proposed; no converter implementation performed for this plan.` with `**Status:** P0–P3 pilot implemented; see [V1.2 findings](V1_2_Findings.md).`

Add to `docs/index.md`: `- \`V1_2_Findings.md\` — Measured V1.2 hand round-trip results, assumption status and next steps.`

- [ ] **Step 4: Commit**

```bash
git add docs/V1_2_Findings.md docs/V1_2_Assumptions.md docs/V1_2_Hand_Roundtrip_Plan.md docs/index.md
git commit -m "docs(hands): record the V1.2 pilot findings and assumption status"
```
