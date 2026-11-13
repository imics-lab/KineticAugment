# V1.3 Face Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit the SMPL-X neck, head and jaw from MediaPipe FaceLandmarker output, measure signed hand-to-face distances with ASL-LEX location labels, and (phase C) carry MediaPipe's 478 face points and 52 blendshapes attached to the fitted head.

**Architecture:** A new FaceLandmarker extraction writes `<clip>_face.npz`. MPI's 105-point MediaPipe→FLAME embedding is transferred to SMPL-X through the SMPL-X↔FLAME vertex map. The head's rotation comes from a weighted, occlusion-aware Umeyama fit on 17 rigid points; the neck swings to put the head on the observed ear midpoint; the jaw is a 1-D pitch lookup. One assembly module poses SMPL-X from the body, hand and head fits, and a geometry module measures hand points against the posed head surface.

**Tech Stack:** Python 3.9+, numpy, scipy, torch, `smplx`, `mediapipe==0.10.21` (Tasks FaceLandmarker), matplotlib, pytest.

**Spec:** [V1_3_Face_Design.md](V1_3_Face_Design.md). Section references (§) below point there.

## Global Constraints

- Support Python 3.9 and newer; Black-compatible formatting at 100 characters per line.
- Implementation code goes under `src/kinetic_augment`, tests under `tests`, scripts under `scripts`.
- Never commit licensed assets. `models/.gitignore` already ignores `*.npz` and `*.pkl`; add `*.task` and `*.obj`.
- Update the nearest `index.md` whenever a file or folder is added, removed or renamed.
- Missing data is NaN, never zero (the convention of `HandClip` and `BodyFit`).
- Frames are fitted independently: no temporal smoothing (§1).
- SMPL-X `expression` stays zero; blendshapes are never converted to SMPL-X expression (§1).
- The verified arm fit (`fit_body`) and the legacy 543-landmark layout in `utils/data_formats.py` are not changed.
- Tests needing licensed assets skip without them, using the marks in `tests/hand_assets.py` and the new `tests/face_assets.py`.
- Run tests with `.venv/bin/python -m pytest`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Before you start (user actions)

1. **Commit or stash the work in progress.** At planning time these were uncommitted:
   `data/index.md`, `docs/V1_Problem_Scope.md`, `scripts/index.md`,
   `scripts/view_body_hand_round_trip.py`, `scripts/view_round_trip.py`, and the untracked
   `data/round_trip_eval/`, `docs/Week_2_Findings.md`, `scripts/evaluate_round_trip.py`. Tasks 1, 2
   and 10 edit some of these files, and their commits must not sweep up unrelated changes.
2. **Download FLAME 2020** from <https://flame.is.tue.mpg.de/> (register, accept the licence) and
   put `generic_model.pkl` at `models/flame/generic_model.pkl`. Only its triangle list is read.
3. **Download the MediaPipe→FLAME embedding** `mediapipe_landmark_embedding.npz` from
   <https://github.com/Zielon/metrical-tracker/tree/master/flame/mediapipe> (MPI non-commercial
   licence) to `models/flame/mediapipe_landmark_embedding.npz`.
4. Task 1's `scripts/download_models.py --face` fetches the two Apache-2.0 MediaPipe assets.

## File map

| File | Status | Responsibility |
| --- | --- | --- |
| `src/kinetic_augment/body_model/face_correspondence.py` | new | MediaPipe index groups; FLAME triangle loader; embedding transfer to SMPL-X |
| `src/kinetic_augment/body_model/assembly.py` | new | One posed SMPL-X from body, hand and head fits (moved out of the viewer script) |
| `src/kinetic_augment/body_model/face_observations.py` | new | `FaceClip`: per-clip FaceLandmarker output, save/load, pixel conversion |
| `src/kinetic_augment/body_model/alignment.py` | modify | Weighted Umeyama |
| `src/kinetic_augment/body_model/head_converter.py` | new | Occlusion, head rotation, jaw, neck/head split, `fit_head` |
| `src/kinetic_augment/body_model/face_regions.py` | new | Head + neck surface and ASL-LEX region labels |
| `src/kinetic_augment/body_model/hand_face_geometry.py` | new | Point-to-triangle distance, signed hand-to-face distances, regions, contact |
| `src/kinetic_augment/body_model/face_pipeline.py` | new | Runs one clip end to end; saves the feature file |
| `src/kinetic_augment/body_model/face_metrics.py` | new | Head-pose matrix comparison and the correspondence floor |
| `src/kinetic_augment/body_model/face_stream.py` | new | Phase C: 478 points in the head frame |
| `scripts/extract_face_landmarks.py` | new | FaceLandmarker pass → `<clip>_face.npz` |
| `scripts/render_face_regions.py` | new | Renders the region segmentation for visual review |
| `scripts/extract_face_features.py` | new | Writes `<clip>_face_features.npz` |
| `scripts/evaluate_face.py` | new | The §7 gate → `face_report.json` |
| `scripts/download_models.py` | modify | Face assets |
| `scripts/view_body_hand_round_trip.py` | modify | Uses `assembly.py` |
| `tests/face_assets.py` | new | Skip marks for face assets |
| `tests/test_face_correspondence.py`, `test_assembly.py`, `test_face_observations.py`, `test_extract_face_landmarks.py`, `test_head_converter.py`, `test_face_regions.py`, `test_hand_face_geometry.py`, `test_face_pipeline.py`, `test_face_metrics.py`, `test_face_stream.py` | new | Tests |
| `tests/test_alignment.py` | modify | Weighted Umeyama tests |
| `docs/V1_3_Findings.md` | new | Gate results |

## Shared names (every task's implementer should know these)

- SMPL-X joint indices: `NECK = 12`, `HEAD = 15`, `JAW = 22`, left/right eye `23`/`24`, wrists
  `20` (left) / `21` (right), middle-finger knuckle `28` (left) / `43` (right). Hand pose slices:
  left `25:40`, right `40:55`. `rotations` always means a `(55, 3)` array of local axis-angle
  rotations in this order; index 0 is `global_orient`.
- SMPL-X rest frame: y up, x toward the subject's left, z forward (the face looks along +z).
- MediaPipe pose world frame and FaceLandmarker pixel frame: x right, y down, z away from the
  camera. The body fit's `global_orient` maps SMPL-X rest into the pose world frame.
- MediaPipe pose ears: `7` (left) and `8` (right). Visibility threshold `0.5`.
- MediaPipe face indices used by name: `33` right eye outer, `133` right eye inner, `362` left eye
  inner, `263` left eye outer, `4` nose tip, `2` subnasale, `13`/`14` inner-lip midpoints, `0`/`17`
  outer-lip top/bottom, `61`/`291` mouth corners.

---

### Task 1: Face correspondence (MediaPipe → SMPL-X surface)

**Files:**
- Create: `src/kinetic_augment/body_model/face_correspondence.py`
- Create: `tests/face_assets.py`, `tests/test_face_correspondence.py`
- Create: `models/flame/index.md`, `models/mediapipe/index.md`
- Modify: `models/.gitignore`, `models/index.md`, `scripts/download_models.py`,
  `src/kinetic_augment/body_model/index.md`, `tests/index.md`
- Modify: `docs/V1_Problem_Scope.md` (§12 ledger rows; §3 out-of-scope table)

**Interfaces:**
- Produces:
  - Index tuples `RIGHT_EYEBROW, LEFT_EYEBROW, RIGHT_EYE, LEFT_EYE, OUTER_LIP, INNER_LIP, NOSE`
    (`OUTER_LIP`/`INNER_LIP` are ordered closed loops), `EYE_CORNERS = (33, 133, 362, 263)`,
    `RIGID = NOSE + EYE_CORNERS` (17 points), `JAW_PAIR = (13, 14)`, `RIGHT_EYE_OUTER = 33`,
    `LEFT_EYE_OUTER = 263`, `NOSE_TIP = 4`, `SUBNASALE = 2`, `UPPER_LIP_TOP = 0`,
    `LOWER_LIP_BOTTOM = 17`, `MOUTH_CORNERS = (61, 291)`.
  - `FaceCorrespondence(mp_indices: (N,) int64, vertex_ids: (N, 3) int64, bary: (N, 3) float64)`
    with `.rows(mp_indices) -> (K,) int` and `.points(vertices, mp_indices=None) -> (..., K, 3)`.
  - `load_flame_faces(path) -> (F, 3) int64`
  - `transfer_embedding(face_idx, bary, mp_indices, flame_faces, flame_to_smplx) -> FaceCorrespondence`
  - `load_face_correspondence(model_dir="models") -> FaceCorrespondence`
  - `load_flame_to_smplx(model_dir="models") -> (5023,) int64`
  - `tests/face_assets.py`: `HAVE_FACE_ASSETS`, `requires_face_assets`, `MEDIAPIPE_DIR`,
    `HAVE_FACE_LANDMARKER`.

- [ ] **Step 1: Create the asset skip marks**

`tests/face_assets.py`:

```python
"""
Skip marks for the face tests: the licensed FLAME assets and the MediaPipe face assets.
"""

from pathlib import Path

import pytest

from tests.hand_assets import HAVE_SMPLX, HAVE_TORCH, MODEL_DIR

FLAME_DIR = Path(MODEL_DIR) / "flame"
MEDIAPIPE_DIR = Path(MODEL_DIR) / "mediapipe"
HAVE_FACE_ASSETS = (
    HAVE_TORCH
    and HAVE_SMPLX
    and (FLAME_DIR / "generic_model.pkl").exists()
    and (FLAME_DIR / "mediapipe_landmark_embedding.npz").exists()
    and (Path(MODEL_DIR) / "mano" / "SMPL-X__FLAME_vertex_ids.npy").exists()
)
HAVE_FACE_LANDMARKER = (MEDIAPIPE_DIR / "face_landmarker.task").exists()
HAVE_CANONICAL_FACE = (MEDIAPIPE_DIR / "canonical_face_model.obj").exists()

requires_face_assets = pytest.mark.skipif(
    not HAVE_FACE_ASSETS, reason="FLAME / embedding / SMPL-X assets absent"
)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_face_correspondence.py`:

```python
"""
Tests for the MediaPipe -> SMPL-X face correspondence (design §3, assumption F1).
"""

import numpy as np
import pytest

from kinetic_augment.body_model.face_correspondence import (
    EYE_CORNERS, INNER_LIP, JAW_PAIR, LEFT_EYE, LEFT_EYEBROW, NOSE, OUTER_LIP, RIGHT_EYE,
    RIGHT_EYEBROW, RIGID,
)
from tests.face_assets import MODEL_DIR, requires_face_assets

GROUPS = (RIGHT_EYEBROW, LEFT_EYEBROW, RIGHT_EYE, LEFT_EYE, OUTER_LIP, INNER_LIP, NOSE)


def test_groups_match_mediapipe_connection_sets():
    connections = pytest.importorskip("mediapipe.python.solutions.face_mesh_connections")

    def points(edges):
        return {i for edge in edges for i in edge}

    assert set(LEFT_EYEBROW) == points(connections.FACEMESH_LEFT_EYEBROW)
    assert set(RIGHT_EYEBROW) == points(connections.FACEMESH_RIGHT_EYEBROW)
    assert set(LEFT_EYE) == points(connections.FACEMESH_LEFT_EYE)
    assert set(RIGHT_EYE) == points(connections.FACEMESH_RIGHT_EYE)
    assert set(OUTER_LIP) | set(INNER_LIP) == points(connections.FACEMESH_LIPS)
    edges = {frozenset(e) for e in connections.FACEMESH_LIPS}
    for loop in (OUTER_LIP, INNER_LIP):
        for a, b in zip(loop, loop[1:] + loop[:1]):
            assert frozenset((a, b)) in edges, (a, b)


def test_rigid_set_and_jaw_pair_are_disjoint_and_sized():
    assert len(RIGID) == 17 and len(set(RIGID)) == 17
    assert set(EYE_CORNERS) <= set(RIGHT_EYE) | set(LEFT_EYE)
    assert not set(JAW_PAIR) & set(RIGID)
    assert sum(len(g) for g in GROUPS) == 105


def test_transfer_maps_triangles_through_the_vertex_map():
    from kinetic_augment.body_model.face_correspondence import transfer_embedding

    flame_faces = np.array([[0, 1, 2], [2, 1, 3]])
    flame_to_smplx = np.array([10, 11, 12, 13])
    corr = transfer_embedding(np.array([1]), np.array([[0.2, 0.3, 0.5]]), np.array([4]),
                              flame_faces, flame_to_smplx)
    np.testing.assert_array_equal(corr.vertex_ids, [[12, 11, 13]])
    vertices = np.zeros((14, 3))
    vertices[[11, 12, 13]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    np.testing.assert_allclose(corr.points(vertices), [[0.3, 0.2, 0.5]])
    np.testing.assert_allclose(corr.points(vertices, (4,)), [[0.3, 0.2, 0.5]])
    with pytest.raises(KeyError):
        corr.rows((5,))


@pytest.fixture(scope="module")
def corr():
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    return load_face_correspondence(MODEL_DIR)


@pytest.fixture(scope="module")
def smplx_npz():
    return np.load(f"{MODEL_DIR}/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)


@requires_face_assets
def test_embedding_covers_exactly_the_named_groups(corr):
    assert len(corr.mp_indices) == 105
    assert set(corr.mp_indices.tolist()) == {i for g in GROUPS for i in g}
    np.testing.assert_allclose(corr.bary.sum(axis=1), 1.0, atol=1e-5)


@requires_face_assets
def test_every_flame_triangle_is_a_smplx_triangle(smplx_npz):
    from kinetic_augment.body_model.face_correspondence import (
        load_flame_faces, load_flame_to_smplx,
    )

    flame_faces = load_flame_faces(f"{MODEL_DIR}/flame/generic_model.pkl")
    mapped = np.sort(load_flame_to_smplx(MODEL_DIR)[flame_faces], axis=1)
    smplx_faces = {tuple(f) for f in np.sort(smplx_npz["f"].astype(np.int64), axis=1)}
    missing = [tuple(f) for f in mapped if tuple(f) not in smplx_faces]
    assert not missing, f"{len(missing)} of {len(mapped)} FLAME triangles have no SMPL-X match"


@requires_face_assets
def test_f1_eye_corners_and_nose_tip_match_smplx_landmarks(corr, smplx_npz):
    """
    Assumption F1. SMPL-X's own 51 landmarks are iBUG-68 minus the jaw contour: 13 is the nose
    tip (iBUG 30), 19/22 the right eye's outer/inner corner (iBUG 36/39), 25/28 the left eye's
    inner/outer corner (iBUG 42/45).
    """
    v = smplx_npz["v_template"].astype(np.float64)
    f = smplx_npz["f"].astype(np.int64)
    lmk = np.einsum("nkc,nk->nc", v[f[smplx_npz["lmk_faces_idx"]]],
                    smplx_npz["lmk_bary_coords"].astype(np.float64))
    pairs = {33: 19, 133: 22, 362: 25, 263: 28}
    ours = corr.points(v, tuple(pairs))
    distance = np.linalg.norm(ours - lmk[list(pairs.values())], axis=1)
    print("eye-corner distances (mm):", np.round(1000 * distance, 2))
    assert (distance < 0.006).all()
    swapped = np.linalg.norm(ours - lmk[[28, 25, 22, 19]], axis=1)
    assert (swapped > distance).all(), "left and right are exchanged"
    nose = np.linalg.norm(corr.points(v, (4,))[0] - lmk[13])
    print(f"nose-tip distance: {1000 * nose:.2f} mm")
    assert nose < 0.008
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_face_correspondence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kinetic_augment.body_model.face_correspondence'`.

- [ ] **Step 4: Implement the module**

`src/kinetic_augment/body_model/face_correspondence.py`:

```python
"""
Where MediaPipe's face-mesh points sit on the SMPL-X surface.

MPI's metrical-tracker embeds 105 MediaPipe points -- eyebrows, eyes, lips and nose -- on the
FLAME head mesh, each as a FLAME triangle plus barycentric weights. SMPL-X's head has FLAME's
topology and SMPL-X ships the vertex map between them, so each embedded point becomes a fixed
weighted sum of three SMPL-X vertices. This replaces the community MediaPipe-to-iBUG-68 tables,
which disagree with each other (design §2).

Assets (licensed; never committed):
    models/flame/generic_model.pkl                 FLAME 2020; only its triangle list is read
    models/flame/mediapipe_landmark_embedding.npz  metrical-tracker, MPI non-commercial
    models/mano/SMPL-X__FLAME_vertex_ids.npy       SMPL-X download
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# MediaPipe face-mesh index groups (mediapipe.solutions.face_mesh_connections). "Right" is the
# subject's right. The lip groups are ordered closed loops.
RIGHT_EYEBROW = (46, 52, 53, 55, 63, 65, 66, 70, 105, 107)
LEFT_EYEBROW = (276, 282, 283, 285, 293, 295, 296, 300, 334, 336)
RIGHT_EYE = (7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246)
LEFT_EYE = (249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466)
OUTER_LIP = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
             291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
INNER_LIP = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95)
NOSE = (168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358)

EYE_CORNERS = (33, 133, 362, 263)  # right outer, right inner, left inner, left outer
RIGHT_EYE_OUTER, LEFT_EYE_OUTER = 33, 263
NOSE_TIP, SUBNASALE = 4, 2
UPPER_LIP_TOP, LOWER_LIP_BOTTOM = 0, 17
MOUTH_CORNERS = (61, 291)
# Points that move with the skull rather than with expression: the head-rotation fit uses these.
RIGID = NOSE + EYE_CORNERS
# Inner-lip midpoints, upper then lower: their gap measures the jaw.
JAW_PAIR = (13, 14)

FLAME_VERTICES = 5023
_EMBEDDING_KEYS = {
    "faces": ("lmk_face_idx", "lmk_faces_idx"),
    "bary": ("lmk_b_coords", "lmk_bary_coords"),
    "mp": ("landmark_indices",),
}


@dataclass(frozen=True)
class FaceCorrespondence:
    """MediaPipe face-mesh points as barycentric combinations of SMPL-X vertices."""

    mp_indices: np.ndarray  # (N,) MediaPipe face-mesh indices
    vertex_ids: np.ndarray  # (N, 3) SMPL-X vertices of each point's triangle
    bary: np.ndarray        # (N, 3) barycentric weights

    def rows(self, mp_indices: Sequence[int]) -> np.ndarray:
        """Row of each MediaPipe index in this correspondence. KeyError if one is not embedded."""
        lookup = {int(m): r for r, m in enumerate(self.mp_indices)}
        missing = [int(m) for m in mp_indices if int(m) not in lookup]
        if missing:
            raise KeyError(f"MediaPipe points not embedded: {missing}")
        return np.array([lookup[int(m)] for m in mp_indices], dtype=np.int64)

    def points(self, vertices: np.ndarray,
               mp_indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Positions of the embedded points on a mesh.

        Args:
            vertices: (..., V, 3) SMPL-X vertices.
            mp_indices: which MediaPipe points, in this order; None for all of them.

        Returns:
            (..., K, 3).
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        ids, bary = self.vertex_ids, self.bary
        if mp_indices is not None:
            rows = self.rows(mp_indices)
            ids, bary = ids[rows], bary[rows]
        return np.einsum("...nkc,nk->...nc", vertices[..., ids, :], bary)


class _Ignored:
    """Stands in for the chumpy objects inside FLAME's pickle; only plain arrays are read."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _FlameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            return _Ignored
        return super().find_class(module, name)


def load_flame_faces(path) -> np.ndarray:
    """FLAME's (F, 3) triangle list, read without needing chumpy installed."""
    with open(path, "rb") as handle:
        model = _FlameUnpickler(handle, encoding="latin1").load()
    faces = np.asarray(model["f"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.max() >= FLAME_VERTICES:
        raise ValueError(f"{path}: not a FLAME triangle list, shape {faces.shape}")
    return faces


def load_flame_to_smplx(model_dir="models") -> np.ndarray:
    """(5023,) SMPL-X vertex id of each FLAME vertex."""
    ids = np.load(Path(model_dir) / "mano" / "SMPL-X__FLAME_vertex_ids.npy").astype(np.int64)
    if ids.shape != (FLAME_VERTICES,):
        raise ValueError(f"expected {FLAME_VERTICES} FLAME vertex ids, got {ids.shape}")
    return ids


def transfer_embedding(face_idx, bary, mp_indices, flame_faces,
                       flame_to_smplx) -> FaceCorrespondence:
    """Carry a FLAME surface embedding onto SMPL-X: same weights, mapped triangle vertices."""
    face_idx = np.asarray(face_idx, dtype=np.int64).ravel()
    vertex_ids = np.asarray(flame_to_smplx, dtype=np.int64)[np.asarray(flame_faces)[face_idx]]
    return FaceCorrespondence(
        mp_indices=np.asarray(mp_indices, dtype=np.int64).ravel(),
        vertex_ids=vertex_ids,
        bary=np.asarray(bary, dtype=np.float64).reshape(-1, 3),
    )


def _read(data, names):
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"embedding has none of {names}; it has {sorted(data.files)}")


def load_face_correspondence(model_dir="models") -> FaceCorrespondence:
    """The 105-point MediaPipe embedding, transferred to SMPL-X."""
    root = Path(model_dir)
    with np.load(root / "flame" / "mediapipe_landmark_embedding.npz", allow_pickle=True) as data:
        face_idx = _read(data, _EMBEDDING_KEYS["faces"])
        bary = _read(data, _EMBEDDING_KEYS["bary"])
        mp_indices = _read(data, _EMBEDDING_KEYS["mp"])
    flame_faces = load_flame_faces(root / "flame" / "generic_model.pkl")
    return transfer_embedding(face_idx, bary, mp_indices, flame_faces,
                              load_flame_to_smplx(model_dir))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_face_correspondence.py -v -s`
Expected: all PASS, and the printed eye-corner and nose-tip distances (keep them: they go into the
ledger). If `test_every_flame_triangle_is_a_smplx_triangle` fails, the FLAME file and the vertex
map disagree on topology: **stop and report**; do not loosen the test. If the embedding npz has
other key names, the `KeyError` lists them; add the names to `_EMBEDDING_KEYS`. If F1 fails,
report the printed distances; do not loosen the thresholds.

- [ ] **Step 6: Face asset download support**

In `scripts/download_models.py`, add the MediaPipe assets and a `--face` flag. Add near the top:

```python
import urllib.request

MEDIAPIPE_FACE_ASSETS = {
    # Apache-2.0, published by Google.
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/1/face_landmarker.task",
    "canonical_face_model.obj": "https://raw.githubusercontent.com/google-ai-edge/mediapipe/"
                                "master/mediapipe/modules/face_geometry/data/"
                                "canonical_face_model.obj",
}
FLAME_FILES = ["generic_model.pkl", "mediapipe_landmark_embedding.npz"]
FLAME_INSTRUCTIONS = """
FLAME assets (licensed, manual):
  1. Register at https://flame.is.tue.mpg.de/ and download FLAME 2020;
     put generic_model.pkl in models/flame/.
  2. Download mediapipe_landmark_embedding.npz from
     https://github.com/Zielon/metrical-tracker/tree/master/flame/mediapipe
     (MPI non-commercial licence) into models/flame/.
"""


def fetch_mediapipe_face_assets(models_dir: Path) -> None:
    """Download the Apache-2.0 MediaPipe face assets into models/mediapipe/."""
    target = models_dir / "mediapipe"
    target.mkdir(parents=True, exist_ok=True)
    for name, url in MEDIAPIPE_FACE_ASSETS.items():
        path = target / name
        if path.exists():
            print(f"  present: {path}")
            continue
        print(f"  fetching {name} ...")
        urllib.request.urlretrieve(url, path)
    missing = [f for f in FLAME_FILES if not (models_dir / "flame" / f).exists()]
    if missing:
        print(FLAME_INSTRUCTIONS)
```

Then in the script's argument parser add `parser.add_argument("--face", action="store_true",
help="fetch MediaPipe face assets and check the FLAME files")` and, where arguments are handled,
`if args.face: fetch_mediapipe_face_assets(Path("models")); return`. Read the existing `main()`
first and follow its structure.

Run: `.venv/bin/python scripts/download_models.py --face`
Expected: both MediaPipe files present in `models/mediapipe/`; no FLAME instructions printed if
the user actions were done.

- [ ] **Step 7: Ignore the new binaries and index the folders**

Append to `models/.gitignore` (above `!.gitignore`):

```
*.task
*.obj
```

`models/flame/index.md`:

```markdown
# FLAME assets index

Licensed assets, never committed (see `models/.gitignore`).

- `generic_model.pkl` — FLAME 2020 head model; only its triangle list is read (`face_correspondence.load_flame_faces`).
- `mediapipe_landmark_embedding.npz` — metrical-tracker's 105-point MediaPipe → FLAME surface embedding (MPI non-commercial).
- `index.md` — This index.
```

`models/mediapipe/index.md`:

```markdown
# MediaPipe assets index

Downloaded by `scripts/download_models.py --face` (Apache-2.0), not committed.

- `face_landmarker.task` — MediaPipe Tasks FaceLandmarker model (478 landmarks, 52 blendshapes, head-pose matrix).
- `canonical_face_model.obj` — MediaPipe's canonical 468-vertex face, in cm; used for the correspondence floor (V1.3 metric 1).
- `index.md` — This index.
```

In `models/index.md` add two lines:
`- \`flame/\` — FLAME triangle list and the MediaPipe → FLAME landmark embedding (licensed, local only).` and
`- \`mediapipe/\` — MediaPipe face-landmarker model and canonical face mesh (downloaded).`

In `src/kinetic_augment/body_model/index.md` add:
`- \`face_correspondence.py\` — MediaPipe face-mesh groups and their positions on the SMPL-X surface, via the FLAME embedding.`

In `tests/index.md` add `face_assets.py` and `test_face_correspondence.py` with one-line descriptions
following the file's existing style.

- [ ] **Step 8: Ledger rows and scope table**

In `docs/V1_Problem_Scope.md` §12, under `### Open`, append (use the next free `O` numbers; at
planning time the last was O9):

```markdown
| O10 | F1: the MediaPipe→FLAME embedding transfers to SMPL-X correctly (V1.3 design §8) | Triangle check passed; eye corners <X> mm, nose tip <Y> mm from smplx's landmarks (Task 1) |
| O11 | F2: FaceLandmarker pixel axes (x·W, y·H, z·W) match the pose world axes | Untested |
| O12 | F3: the 17-point rigid set gives head rotation under expression and partial occlusion | Untested |
| O13 | F4: the neck swing to the observed ear midpoint fixes head placement without touching the arms | Untested |
| O14 | F5: lip gap is a usable proxy for jaw pitch with expression at zero | Untested |
| O15 | F6: landmark rules on the rest template give sensible ASL-LEX regions | Untested |
```

Fill `<X>`/`<Y>` with Step 5's printed numbers. In §3's "Explicitly out of scope for V1" table,
change the Face row's reason to:
`Moved into scope by V1.3 ([design](V1_3_Face_Design.md)): head, neck and jaw from FaceLandmarker; expression stays out.`

- [ ] **Step 9: Commit**

```bash
git add src/kinetic_augment/body_model/face_correspondence.py tests/face_assets.py \
  tests/test_face_correspondence.py scripts/download_models.py models/.gitignore models/index.md \
  models/flame/index.md models/mediapipe/index.md src/kinetic_augment/body_model/index.md \
  tests/index.md docs/V1_Problem_Scope.md
git commit -m "feat(face): MediaPipe to SMPL-X face correspondence via the FLAME embedding"
```

---

### Task 2: Assembly — one posed SMPL-X from the separate fits

**Files:**
- Create: `src/kinetic_augment/body_model/assembly.py`, `tests/test_assembly.py`
- Modify: `scripts/view_body_hand_round_trip.py` (`BodyFigure.pose`, `_kabsch`, `_global_rotations`)
- Modify: `src/kinetic_augment/body_model/index.md`, `tests/index.md`

**Interfaces:**
- Consumes: `BodyFit` (`converter.py`), `HandFit` (`hand_converter.py`), `HandModel`
  (`hand_model.py`), `SMPLXSkeleton` (`skeleton.py`; `.model` is the `smplx` model with
  `num_betas=300`, `flat_hand_mean=True`, `use_pca=False`).
- Produces:
  - Constants `NUM_JOINTS = 55`, `NECK = 12`, `HEAD = 15`, `JAW = 22`,
    `WRIST = {"left": 20, "right": 21}`, `HAND_SLICE = {"left": slice(25, 40), "right": slice(40, 55)}`,
    `PALM = (0, 5, 9, 13, 17)`.
  - `PosedMesh(vertices: (V, 3), joints: (55, 3))`, float64, MediaPipe world frame.
  - `kabsch(source, target) -> (3, 3)`
  - `global_rotations(rotations, parents) -> (J, 3, 3)`
  - `body_rotations(body_fit) -> (55, 3)`
  - `attach_hands(rotations, parents, hand_fits, hand_recon, hand_models) -> (rotations, posed: Dict[str, bool])` (returns a new array)
  - `attach_head(rotations, head_fit) -> rotations` (new array; reads `head_fit.neck_pose`, `.head_pose`, `.jaw_pose`)
  - `pose_mesh(skeleton, rotations, betas, transl) -> PosedMesh`

- [ ] **Step 1: Record the viewer's current output (regression baseline)**

Only if `data/test_output/01988_world.npz` and `data/test_output/01988_hands.npz` exist; skip
otherwise. Save to the scratch directory, not the repo:

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "scripts")
import numpy as np
from pathlib import Path
from view_body_hand_round_trip import BodyFigure, hand_round_trip, load_observations
from kinetic_augment.body_model.converter import fit_body
from kinetic_augment.body_model.shape import fit_shape
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
pose, ok, hands = load_observations(Path("data/test_output"), "01988")
skel = SMPLXSkeleton("models"); body = skel.shaped(fit_shape(pose, ok, skel))
fits, recon, _ = hand_round_trip(hands)
t = 40
bf = fit_body(pose[t], ok[t], body)
v, j, posed = BodyFigure(skel).pose(bf, {s: fits[s][t] for s in fits}, {s: recon[s][t] for s in recon})
np.savez("/tmp/assembly_baseline.npz", vertices=v, joints=j)
print(posed)
EOF
```

(Replace `/tmp` with the session scratchpad path when one is available.)

- [ ] **Step 2: Write the failing tests**

`tests/test_assembly.py`:

```python
"""
Tests for assembly: one SMPL-X pose from the body, hand and head fits.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.hand_assets import MODEL_DIR, requires_smplx

BETAS = np.zeros(300)
BETAS[[0, 1, 40]] = [1.0, -0.5, 1.5]


@pytest.fixture(scope="module")
def skeleton():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton
    return SMPLXSkeleton(model_path=MODEL_DIR)


def test_kabsch_recovers_a_rotation():
    from kinetic_augment.body_model.assembly import kabsch

    rng = np.random.default_rng(0)
    source = rng.normal(size=(6, 3))
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.9]).as_matrix()
    np.testing.assert_allclose(kabsch(source, source @ rotation.T + 5.0), rotation, atol=1e-10)


def test_body_rotations_layout():
    from kinetic_augment.body_model.assembly import body_rotations
    from kinetic_augment.body_model.converter import BodyFit

    fit = BodyFit(body_pose=np.arange(63.0), global_orient=np.array([1.0, 2.0, 3.0]),
                  transl=np.zeros(3))
    rotations = body_rotations(fit)
    assert rotations.shape == (55, 3)
    np.testing.assert_array_equal(rotations[0], [1, 2, 3])
    np.testing.assert_array_equal(rotations[1:22].ravel(), np.arange(63.0))
    assert not rotations[22:].any()


def test_attach_head_writes_neck_head_and_jaw():
    from kinetic_augment.body_model.assembly import attach_head

    head = SimpleNamespace(neck_pose=np.full(3, 0.1), head_pose=np.full(3, 0.2),
                           jaw_pose=np.array([0.3, 0, 0]))
    rotations = attach_head(np.zeros((55, 3)), head)
    np.testing.assert_array_equal(rotations[12], 0.1)
    np.testing.assert_array_equal(rotations[15], 0.2)
    np.testing.assert_array_equal(rotations[22], [0.3, 0, 0])


@requires_smplx
def test_pose_mesh_joints_match_forward_kinematics(skeleton):
    from kinetic_augment.body_model.assembly import pose_mesh
    from kinetic_augment.body_model.kinematics import forward_kinematics

    rng = np.random.default_rng(1)
    rotations = 0.3 * rng.normal(size=(55, 3))
    transl = np.array([0.1, -0.2, 0.3])
    mesh = pose_mesh(skeleton, rotations, BETAS, transl)
    expected = forward_kinematics(rotations, skeleton.shaped(BETAS)) + transl
    np.testing.assert_allclose(mesh.joints, expected, atol=1e-4)
    assert mesh.vertices.shape == (10475, 3)


@requires_smplx
@pytest.mark.parametrize("side", ["left", "right"])
def test_attach_hands_turns_the_palm_onto_the_target(skeleton, side):
    from kinetic_augment.body_model.assembly import (
        HAND_SLICE, PALM, WRIST, attach_hands, global_rotations,
    )
    from kinetic_augment.body_model.hand_converter import HandFit
    from kinetic_augment.body_model.hand_model import HandModel

    other = "right" if side == "left" else "left"
    models = {s: HandModel("smplx", s, model_dir=MODEL_DIR) for s in ("left", "right")}
    import torch

    hand_pose = 0.2 * np.random.default_rng(2).normal(size=45)
    with torch.no_grad():
        rest = models[side].points(np.zeros(3), hand_pose, np.zeros(10))[0].numpy()
    world = Rotation.from_rotvec([0.4, 1.1, -0.3]).as_matrix()
    target = rest @ world.T + np.array([0.3, 0.1, 0.0])
    fits = {
        side: HandFit("smplx", side, np.zeros(3), hand_pose, np.zeros(10), np.zeros(3)),
        other: HandFit("smplx", other, np.full(3, np.nan), np.full(45, np.nan),
                       np.full(10, np.nan), np.full(3, np.nan), status="no_wrist"),
    }
    recon = {side: target, other: np.full((21, 3), np.nan)}
    body = 0.2 * np.random.default_rng(3).normal(size=(55, 3))
    body[22:] = 0.0
    rotations, posed = attach_hands(body, skeleton.parents, fits, recon, models)
    assert posed == {side: True, other: False}
    wrist_world = global_rotations(rotations, skeleton.parents)[WRIST[side]]
    np.testing.assert_allclose(wrist_world, world, atol=1e-6)
    np.testing.assert_allclose(rotations[HAND_SLICE[side]].ravel(), hand_pose)
    assert not rotations[HAND_SLICE[other]].any()
    np.testing.assert_array_equal(rotations[WRIST[other]], body[WRIST[other]])
    assert PALM == (0, 5, 9, 13, 17)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_assembly.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kinetic_augment.body_model.assembly'`.

- [ ] **Step 4: Implement `assembly.py`**

`src/kinetic_augment/body_model/assembly.py`:

```python
"""
One posed SMPL-X body from the separate fits.

The body fit gives global_orient, body_pose, betas and transl; each hand fit gives a finger pose
and, through the palm of its reconstruction, the wrist's world rotation; the head fit gives the
neck, head and jaw. SMPL-X takes all of them as one (55, 3) array of local axis-angle rotations
in its joint order -- the order `forward_kinematics` uses -- so assembly fills that array and
runs the model.

Hands are attached as the V1.2 viewer did: the SMPL-X hand at the fitted finger pose, with the
wrist unrotated, is turned so its palm (wrist and knuckles) lines up with the reconstruction's
palm, and that rotation is written into the wrist relative to its parent.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

NUM_JOINTS = 55
NECK, HEAD, JAW = 12, 15, 22
WRIST = {"left": 20, "right": 21}
HAND_SLICE = {"left": slice(25, 40), "right": slice(40, 55)}
PALM = (0, 5, 9, 13, 17)  # MediaPipe hand points: the wrist and the four knuckles


@dataclass
class PosedMesh:
    vertices: np.ndarray  # (V, 3) metres, MediaPipe world frame
    joints: np.ndarray    # (55, 3) model joints, same frame


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R source - target| over centred point sets."""
    u, _, vt = np.linalg.svd((source - source.mean(0)).T @ (target - target.mean(0)))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def global_rotations(rotations: np.ndarray, parents) -> np.ndarray:
    """(J, 3, 3) world rotation of every joint from (J, 3) local axis-angle rotations."""
    local = Rotation.from_rotvec(np.asarray(rotations, dtype=np.float64)).as_matrix()
    world = np.empty_like(local)
    for j, parent in enumerate(parents):
        world[j] = local[j] if parent < 0 else world[parent] @ local[j]
    return world


def body_rotations(body_fit) -> np.ndarray:
    """(55, 3) rotations holding the body fit; every other joint at rest."""
    rotations = np.zeros((NUM_JOINTS, 3))
    rotations[0] = body_fit.global_orient
    rotations[1:22] = np.asarray(body_fit.body_pose).reshape(21, 3)
    return rotations


def attach_hands(rotations, parents, hand_fits, hand_recon,
                 hand_models) -> Tuple[np.ndarray, Dict[str, bool]]:
    """
    Rotations with each fitted hand's finger pose and wrist rotation written in.

    Args:
        rotations: (55, 3); not modified.
        parents: (55,) kinematic parents.
        hand_fits: side -> HandFit for this frame.
        hand_recon: side -> (21, 3) reconstructed hand landmarks for this frame (NaN if none).
        hand_models: side -> SMPL-X `HandModel`, used for the unrotated SMPL-X palm.

    Returns:
        (new rotations, side -> whether that hand was posed). An unposed hand keeps a flat
        finger pose and the body fit's wrist.
    """
    rotations = np.array(rotations, dtype=np.float64)
    posed = {}
    for side in ("left", "right"):
        fit, target = hand_fits[side], np.asarray(hand_recon[side])
        posed[side] = fit.status == "ok" and bool(np.isfinite(target).all())
        if not posed[side]:
            rotations[HAND_SLICE[side]] = 0.0
            continue
        rotations[HAND_SLICE[side]] = np.asarray(fit.hand_pose).reshape(15, 3)
        with torch.no_grad():
            rest = hand_models[side].points(np.zeros(3), fit.hand_pose,
                                            np.zeros(10))[0].numpy()
        world_wrist = kabsch(rest[list(PALM)], target[list(PALM)])
        wrist = WRIST[side]
        parent_world = global_rotations(rotations, parents)[parents[wrist]]
        rotations[wrist] = Rotation.from_matrix(parent_world.T @ world_wrist).as_rotvec()
    return rotations, posed


def attach_head(rotations, head_fit) -> np.ndarray:
    """Rotations with the head fit's neck, head and jaw written in."""
    rotations = np.array(rotations, dtype=np.float64)
    rotations[NECK] = head_fit.neck_pose
    rotations[HEAD] = head_fit.head_pose
    rotations[JAW] = head_fit.jaw_pose
    return rotations


def pose_mesh(skeleton, rotations, betas, transl) -> PosedMesh:
    """
    Run SMPL-X with every joint rotation, the body shape and the translation; expression zero.

    `skeleton` is an `SMPLXSkeleton`; its model's shape space must hold `len(betas)` values.
    """
    r = np.asarray(rotations, dtype=np.float32)

    def tensor(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, -1)

    with torch.no_grad():
        out = skeleton.model(
            betas=tensor(betas), global_orient=tensor(r[0]), body_pose=tensor(r[1:22]),
            jaw_pose=tensor(r[22]), leye_pose=tensor(r[23]), reye_pose=tensor(r[24]),
            left_hand_pose=tensor(r[25:40]), right_hand_pose=tensor(r[40:55]),
            transl=tensor(transl),
        )
    return PosedMesh(vertices=out.vertices[0].numpy().astype(np.float64),
                     joints=out.joints[0, :NUM_JOINTS].numpy().astype(np.float64))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_assembly.py -v`
Expected: all PASS.

- [ ] **Step 6: Rewire the viewer**

In `scripts/view_body_hand_round_trip.py`: delete `_kabsch` and `_global_rotations`; add
`from kinetic_augment.body_model.assembly import attach_hands, body_rotations, pose_mesh  # noqa: E402`;
replace the body of `BodyFigure.pose` with:

```python
        rotations, posed = attach_hands(body_rotations(body_fit), self.skeleton.parents,
                                        hand_fits, hand_recon, self.hand_models)
        mesh = pose_mesh(self.skeleton, rotations, body_fit.betas, body_fit.transl)
        return mesh.vertices, mesh.joints, posed
```

Remove imports that become unused (`torch`, `Rotation`, `PALM`, `SMPLX_WRIST`) only if nothing
else in the script uses them (`grep` first). If Step 1 ran, re-run its snippet writing
`/tmp/assembly_after.npz` and compare:

```bash
.venv/bin/python -c "import numpy as np; a=np.load('/tmp/assembly_baseline.npz'); b=np.load('/tmp/assembly_after.npz'); print(abs(a['vertices']-b['vertices']).max(), abs(a['joints']-b['joints']).max())"
```

Expected: both maxima below `1e-6`. Then `.venv/bin/python scripts/view_body_hand_round_trip.py --help` exits 0.

- [ ] **Step 7: Index and commit**

Add to `src/kinetic_augment/body_model/index.md`:
`- \`assembly.py\` — Poses one SMPL-X body from the body, hand and head fits.` and add
`test_assembly.py` to `tests/index.md`.

```bash
git add src/kinetic_augment/body_model/assembly.py tests/test_assembly.py \
  scripts/view_body_hand_round_trip.py src/kinetic_augment/body_model/index.md tests/index.md
git commit -m "refactor(assembly): move body + hand stitching from the viewer into src"
```

---

### Task 3: Face observations and the FaceLandmarker extraction

**Files:**
- Create: `src/kinetic_augment/body_model/face_observations.py`, `scripts/extract_face_landmarks.py`
- Create: `tests/test_face_observations.py`, `tests/test_extract_face_landmarks.py`
- Modify: `src/kinetic_augment/body_model/index.md`, `scripts/index.md`, `tests/index.md`

**Interfaces:**
- Produces:
  - `FACE_CLIP_SCHEMA = 1`, `NUM_FACE_POINTS = 478`, `NUM_BLENDSHAPES = 52`
  - `FaceClip(landmarks (T,478,3) f32, blendshapes (T,52) f32, head_matrix (T,4,4) f32, valid (T,) bool, timestamp_ms (T,) f64, width, height, fps, blendshape_names: Tuple[str, ...], meta)`
    with `.pixels() -> (T, 478, 3) float64`.
  - `save_face_clip(path, clip)`, `load_face_clip(path) -> FaceClip`
  - Script function `extract_face(video_path, model_path) -> FaceClip` and
    `extract_all(videos, output_dir, model_path)` writing `<stem>_face.npz`.

- [ ] **Step 1: Write the failing unit tests**

`tests/test_face_observations.py`:

```python
"""Tests for FaceClip: pixel conversion and file round trip."""

import numpy as np
import pytest

from kinetic_augment.body_model.face_observations import (
    FaceClip, load_face_clip, save_face_clip,
)


def clip(frames=2):
    landmarks = np.zeros((frames, 478, 3), dtype=np.float32)
    landmarks[:, 0] = [0.5, 0.25, -0.1]
    return FaceClip(
        landmarks=landmarks,
        blendshapes=np.zeros((frames, 52), dtype=np.float32),
        head_matrix=np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)),
        valid=np.ones(frames, dtype=bool),
        timestamp_ms=np.arange(frames, dtype=np.float64) * 40.0,
        width=640, height=480, fps=25.0,
        blendshape_names=tuple(f"b{i}" for i in range(52)),
    )


def test_pixels_scale_z_by_width_like_x():
    np.testing.assert_allclose(clip().pixels()[0, 0], [320.0, 120.0, -64.0])


def test_save_load_round_trip(tmp_path):
    original = clip()
    save_face_clip(tmp_path / "c_face.npz", original)
    loaded = load_face_clip(tmp_path / "c_face.npz")
    np.testing.assert_array_equal(loaded.landmarks, original.landmarks)
    np.testing.assert_array_equal(loaded.valid, original.valid)
    assert loaded.blendshape_names == original.blendshape_names
    assert (loaded.width, loaded.height, loaded.fps) == (640, 480, 25.0)


def test_wrong_schema_is_refused(tmp_path):
    save_face_clip(tmp_path / "c_face.npz", clip())
    data = dict(np.load(tmp_path / "c_face.npz"))
    data["meta"] = np.array('{"schema": 99}')
    np.savez(tmp_path / "bad_face.npz", **data)
    with pytest.raises(ValueError, match="schema"):
        load_face_clip(tmp_path / "bad_face.npz")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_face_observations.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `face_observations.py`**

```python
"""
MediaPipe FaceLandmarker observations for the V1.3 face work.

Per decoded video frame: 478 landmarks (468 mesh points plus 10 iris points), the 52 blendshape
scores, and the facial transformation matrix. Landmark x and y are normalised by the image width
and height; z is relative depth normalised by the image width, like x (MediaPipe
geometry_pipeline.cc). Missing data is NaN, never zero.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

FACE_CLIP_SCHEMA = 1
NUM_FACE_POINTS = 478
NUM_BLENDSHAPES = 52
_ARRAYS = ("landmarks", "blendshapes", "head_matrix", "valid", "timestamp_ms")


@dataclass
class FaceClip:
    """One clip's face observations; T = decoded frames."""

    landmarks: np.ndarray     # (T, 478, 3) float32: x / width, y / height, z / width; NaN, no face
    blendshapes: np.ndarray   # (T, 52) float32 scores in [0, 1]; NaN where no face
    head_matrix: np.ndarray   # (T, 4, 4) float32 facial transformation (OpenGL axes, cm); NaN
    valid: np.ndarray         # (T,) bool: a face was detected
    timestamp_ms: np.ndarray  # (T,) float64, from the video decoder
    width: int
    height: int
    fps: float
    blendshape_names: Tuple[str, ...] = ()
    meta: Dict = field(default_factory=dict)

    def pixels(self) -> np.ndarray:
        """
        (T, 478, 3) landmarks in pixels on all three axes: (x·W, y·H, z·W), so the axes share
        one scale. Axes: x right, y down, z away from the camera.
        """
        scale = np.array([self.width, self.height, self.width], dtype=np.float64)
        return self.landmarks.astype(np.float64) * scale


def save_face_clip(path, clip: FaceClip) -> None:
    """Write a clip to .npz, stamping the schema version into its metadata."""
    meta = dict(clip.meta, schema=FACE_CLIP_SCHEMA,
                blendshape_names=list(clip.blendshape_names))
    np.savez_compressed(
        path,
        **{name: getattr(clip, name) for name in _ARRAYS},
        width=np.array(clip.width), height=np.array(clip.height), fps=np.array(clip.fps),
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )


def load_face_clip(path) -> FaceClip:
    """
    Read a clip written by `save_face_clip`.

    Raises:
        ValueError: the file's schema is not FACE_CLIP_SCHEMA.
    """
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("schema") != FACE_CLIP_SCHEMA:
            raise ValueError(f"{path}: face clip schema {meta.get('schema')}, "
                             f"expected {FACE_CLIP_SCHEMA}")
        names = tuple(meta.pop("blendshape_names", ()))
        return FaceClip(
            **{name: data[name] for name in _ARRAYS},
            width=int(data["width"]), height=int(data["height"]), fps=float(data["fps"]),
            blendshape_names=names, meta=meta,
        )
```

- [ ] **Step 4: Run the unit tests**

Run: `.venv/bin/python -m pytest tests/test_face_observations.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the failing integration test**

`tests/test_extract_face_landmarks.py`:

```python
"""
Integration tests for the face extractor: real FaceLandmarker on a pilot clip.
Skipped when mediapipe, the model or the video is absent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mediapipe")

from tests.face_assets import HAVE_FACE_LANDMARKER, MEDIAPIPE_DIR  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VIDEO = REPO / "data" / "sign_videos" / "01991.mp4"
HANDS = REPO / "data" / "test_output" / "01991_hands.npz"
sys.path.insert(0, str(REPO / "scripts"))

pytestmark = pytest.mark.skipif(not (HAVE_FACE_LANDMARKER and VIDEO.exists()),
                                reason="face landmarker model or pilot video absent")


@pytest.fixture(scope="module")
def face():
    from extract_face_landmarks import extract_face
    return extract_face(VIDEO, MEDIAPIPE_DIR / "face_landmarker.task")


def test_a_face_is_found_on_most_frames(face):
    assert face.landmarks.shape[1:] == (478, 3)
    assert face.valid.mean() > 0.9
    assert "jawOpen" in face.blendshape_names and len(face.blendshape_names) == 52


def test_frames_line_up_with_the_hand_clip(face):
    if not HANDS.exists():
        pytest.skip("pilot hand clip absent")
    from kinetic_augment.body_model.hand_observations import load_hand_clip
    hands = load_hand_clip(HANDS)
    np.testing.assert_array_equal(face.timestamp_ms, hands.timestamp_ms)


def test_head_matrix_puts_the_face_in_front_of_the_camera(face):
    """OpenGL camera looks down -Z, so translation z is negative; this also pins the layout."""
    tz = face.head_matrix[face.valid, 2, 3]
    assert np.median(tz) < 0
    np.testing.assert_allclose(face.head_matrix[face.valid, 3], [0, 0, 0, 1], atol=1e-5)
```

- [ ] **Step 6: Implement `scripts/extract_face_landmarks.py`**

```python
#!/usr/bin/env python3
"""
Extract MediaPipe FaceLandmarker output for the V1.3 face work.

Per clip, a fresh FaceLandmarker (Tasks API, VIDEO mode, one face) gives 478 landmarks, 52
blendshapes and the facial transformation matrix per frame. Frames are decoded with OpenCV
exactly as `extract_hand_landmarks.py` decodes them, so frame t here is frame t of
`<clip>_hands.npz`.

Output: <output-dir>/<clip>_face.npz (see `FaceClip`).

Usage:
    python scripts/extract_face_landmarks.py
    python scripts/extract_face_landmarks.py --clips-from data/round_trip_eval \
        --video-dir ~/Downloads/raw_videos --output-dir data/round_trip_eval
"""

import argparse
import hashlib
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # noqa: E402 -- must follow the log-level setting above
import numpy as np  # noqa: E402

from kinetic_augment.body_model.face_observations import (  # noqa: E402
    NUM_BLENDSHAPES, NUM_FACE_POINTS, FaceClip, save_face_clip,
)

VIDEO_DIR = Path("data/sign_videos")
OUTPUT_DIR = Path("data/test_output")
MODEL = Path("models/mediapipe/face_landmarker.task")
SETTINGS = dict(num_faces=1, min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True)


def observe_frame(result) -> dict:
    """One frame's arrays; NaN when no face was found."""
    if not result.face_landmarks:
        return dict(landmarks=np.full((NUM_FACE_POINTS, 3), np.nan, np.float32),
                    blendshapes=np.full(NUM_BLENDSHAPES, np.nan, np.float32),
                    head_matrix=np.full((4, 4), np.nan, np.float32), valid=False, names=None)
    points = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
    categories = result.face_blendshapes[0]
    return dict(
        landmarks=points,
        blendshapes=np.array([c.score for c in categories], dtype=np.float32),
        head_matrix=np.asarray(result.facial_transformation_matrixes[0],
                               dtype=np.float32).reshape(4, 4),
        valid=True,
        names=tuple(c.category_name for c in categories),
    )


def extract_face(video_path: Path, model_path: Path = MODEL) -> FaceClip:
    """Run a fresh FaceLandmarker over every frame of one clip."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )

    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                    running_mode=RunningMode.VIDEO, **SETTINGS)
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    rows, stamps, last = [], [], -1
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            stamps.append(stamp)
            # VIDEO mode needs strictly increasing integer timestamps.
            last = max(int(round(stamp)), last + 1)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rows.append(observe_frame(landmarker.detect_for_video(image, last)))
    capture.release()
    if not rows:
        raise ValueError(f"{video_path}: no frames decoded")

    names = next((row["names"] for row in rows if row["names"]), ())
    return FaceClip(
        landmarks=np.stack([row["landmarks"] for row in rows]),
        blendshapes=np.stack([row["blendshapes"] for row in rows]),
        head_matrix=np.stack([row["head_matrix"] for row in rows]),
        valid=np.array([row["valid"] for row in rows], dtype=bool),
        timestamp_ms=np.array(stamps, dtype=np.float64),
        width=width, height=height, fps=fps, blendshape_names=tuple(names),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_all(videos, output_dir, model_path: Path = MODEL) -> None:
    import mediapipe as mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'clip':<8}{'frames':>8}{'face %':>8}")
    for video in videos:
        video = Path(video)
        clip = extract_face(video, model_path)
        clip.meta = {"source": video.name, "source_sha256": _sha256(video),
                     "mediapipe": mp.__version__, "settings": SETTINGS,
                     "model": Path(model_path).name,
                     "detector_state": "fresh FaceLandmarker per clip"}
        save_face_clip(output_dir / f"{video.stem}_face.npz", clip)
        print(f"{video.stem:<8}{len(clip.valid):>8}{100 * clip.valid.mean():>7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("videos", nargs="*", type=Path, help="videos (default: data/sign_videos)")
    parser.add_argument("--clips-from", type=Path,
                        help="take clip ids from the *_hands.npz files in this directory")
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    if args.videos:
        videos = args.videos
    elif args.clips_from:
        stems = sorted(p.name[: -len("_hands.npz")] for p in args.clips_from.glob("*_hands.npz"))
        videos = [args.video_dir.expanduser() / f"{stem}.mp4" for stem in stems]
    else:
        videos = sorted(args.video_dir.glob("*.mp4"))
    missing = [v for v in videos if not Path(v).exists()]
    if missing or not videos:
        raise SystemExit(f"No videos, or missing: {missing[:5]}")
    extract_all(videos, args.output_dir, args.model)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the integration test**

Run: `.venv/bin/python -m pytest tests/test_extract_face_landmarks.py -v`
Expected: PASS. If `test_head_matrix_puts_the_face_in_front_of_the_camera` fails because the
translation sits in the bottom row instead of the last column, the binding returned a
column-major matrix: transpose it in `observe_frame` (`.reshape(4, 4).T`) and re-run.

- [ ] **Step 8: Extract the pilot clips, index, commit**

Run: `.venv/bin/python scripts/extract_face_landmarks.py`
Expected: one row per pilot clip, face % near 100.

Add `face_observations.py` to `src/kinetic_augment/body_model/index.md`,
`extract_face_landmarks.py` to `scripts/index.md`, the two test files to `tests/index.md`, and the
`<clip>_face.npz` pattern to `data/test_output/index.md` (follow each file's style).

```bash
git add src/kinetic_augment/body_model/face_observations.py scripts/extract_face_landmarks.py \
  tests/test_face_observations.py tests/test_extract_face_landmarks.py \
  src/kinetic_augment/body_model/index.md scripts/index.md tests/index.md data/test_output/index.md
git commit -m "feat(face): FaceLandmarker extraction into per-clip face files"
```

(Do not commit the `*_face.npz` files if `data/test_output` ignores npz caches; check
`git status` and follow what is done for `*_hands.npz`.)

---

### Task 4: Weighted Umeyama, occlusion and head rotation

**Files:**
- Modify: `src/kinetic_augment/body_model/alignment.py`, `tests/test_alignment.py`
- Create: `src/kinetic_augment/body_model/head_converter.py`, `tests/test_head_converter.py`
- Modify: `src/kinetic_augment/body_model/index.md`, `tests/index.md`

**Interfaces:**
- Consumes: `similarity_transform`, `apply_similarity` (`alignment.py`).
- Produces (in `head_converter.py`):
  - Constants `OCCLUSION_PAD = 0.15`, `OUTLIER_FACTOR = 2.5`, `RESIDUAL_FLOOR = 0.01`,
    `MIN_RIGID_POINTS = 8`, `MAX_RMS_FRACTION = 0.10`.
  - `hand_outlines(hand_image: (2, 21, 3), width, height) -> List[(K, 2)]` (pixels)
  - `occluded(points: (N, 2), outlines, pad) -> (N,) bool`
  - `HeadRotation(rotation (3,3), rms_fraction, points_used, weights (N,))`
  - `head_rotation(template (N,3), observed (N,3), weights (N,), reference_length) -> HeadRotation`
- `similarity_transform(source, target, allow_scale=True, weights=None)`

- [ ] **Step 1: Write the failing weighted-Umeyama tests**

Append to `tests/test_alignment.py`:

```python
def test_weights_none_matches_uniform_weights():
    from kinetic_augment.body_model.alignment import similarity_transform

    rng = np.random.default_rng(5)
    source, target = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    plain = similarity_transform(source, target)
    weighted = similarity_transform(source, target, weights=np.full(8, 3.0))
    np.testing.assert_allclose(plain[0], weighted[0])
    np.testing.assert_allclose(plain[1], weighted[1], atol=1e-12)
    np.testing.assert_allclose(plain[2], weighted[2], atol=1e-12)


def test_zero_weight_points_are_ignored():
    from kinetic_augment.body_model.alignment import similarity_transform
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(6)
    source = rng.normal(size=(10, 3))
    rotation = Rotation.from_rotvec([0.2, 0.5, -0.1]).as_matrix()
    target = 2.0 * source @ rotation.T + 1.0
    target[:3] += 50.0  # gross outliers
    weights = np.ones(10)
    weights[:3] = 0.0
    scale, fitted, translation = similarity_transform(source, target, weights=weights)
    np.testing.assert_allclose(scale, 2.0)
    np.testing.assert_allclose(fitted, rotation, atol=1e-10)


def test_too_few_weighted_points_raise():
    from kinetic_augment.body_model.alignment import similarity_transform

    with pytest.raises(ValueError, match="positively weighted"):
        similarity_transform(np.eye(3) * [1, 2, 3] + [[0, 0, 0]] * 3,
                             np.eye(3), weights=np.array([1.0, 1.0, 0.0]))
```

(Check the top of `tests/test_alignment.py` imports `numpy as np` and `pytest`; add them if not.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: the three new tests FAIL with `TypeError: ... unexpected keyword argument 'weights'`.

- [ ] **Step 3: Implement weights in `similarity_transform`**

Change the signature to
`def similarity_transform(source, target, allow_scale: bool = True, weights=None)`, document
`weights: (N,) non-negative per-point weights; None weights all points equally. Points with zero
weight do not influence the fit.` in the docstring, and replace the body from `src_mean = ...`
through the translation with:

```python
    if weights is None:
        w = np.full(source.shape[0], 1.0 / source.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != (source.shape[0],) or (w < 0).any() or not np.isfinite(w).all():
            raise ValueError("weights must be (N,), finite and non-negative")
        if (w > 0).sum() < 3:
            raise ValueError("need at least 3 positively weighted points")
        w = w / w.sum()

    src_mean = w @ source
    tgt_mean = w @ target
    src_centred = source - src_mean
    tgt_centred = target - tgt_mean

    covariance = (tgt_centred * w[:, None]).T @ src_centred
    u, singular_values, vt = np.linalg.svd(covariance)

    # Flip the least-significant axis if the raw solution is a reflection.
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
        singular_values = singular_values.copy()
        singular_values[2] = -singular_values[2]

    rotation = u @ correction @ vt

    if allow_scale:
        source_variance = float(w @ (src_centred ** 2).sum(axis=1))
        scale = 1.0 if source_variance == 0 else singular_values.sum() / source_variance
    else:
        scale = 1.0
    translation = tgt_mean - scale * rotation @ src_mean
```

Uniform weights `1/N` reproduce the old formulas exactly.

- [ ] **Step 4: Run all alignment tests**

Run: `.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: all PASS (old and new).

- [ ] **Step 5: Write the failing head-rotation tests**

`tests/test_head_converter.py`:

```python
"""
Tests for the head converter (design §4): occlusion, head rotation, jaw, neck/head split, fit_head.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.head_converter import (
    HeadRotation, head_rotation, occluded,
)

PIXELS_PER_METRE = 4000.0


def synthetic_face(seed=0, noise_px=0.0):
    rng = np.random.default_rng(seed)
    template = 0.04 * rng.normal(size=(17, 3))
    rotation = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    observed = PIXELS_PER_METRE * template @ rotation.T + [320.0, 200.0, 50.0]
    observed += noise_px * rng.normal(size=observed.shape)
    return template, observed, rotation


def angle_deg(a, b):
    return np.degrees(Rotation.from_matrix(a.T @ b).magnitude())


def test_head_rotation_recovers_the_rotation():
    template, observed, rotation = synthetic_face(noise_px=0.5)
    fit = head_rotation(template, observed, np.ones(17), reference_length=250.0)
    assert isinstance(fit, HeadRotation)
    assert angle_deg(fit.rotation, rotation) < 0.5
    assert fit.points_used == 17
    assert fit.rms_fraction < 0.01


def test_outliers_are_dropped_in_the_robust_pass():
    template, observed, rotation = synthetic_face(noise_px=0.5)
    observed[[2, 7]] += 30.0
    fit = head_rotation(template, observed, np.ones(17), reference_length=250.0)
    assert fit.points_used == 15
    assert angle_deg(fit.rotation, rotation) < 0.5


def test_noise_free_points_are_all_kept():
    template, observed, _ = synthetic_face()
    assert head_rotation(template, observed, np.ones(17), 250.0).points_used == 17


def test_too_few_points_give_nan():
    template, observed, _ = synthetic_face()
    weights = np.zeros(17)
    weights[:2] = 1.0
    fit = head_rotation(template, observed, weights, 250.0)
    assert np.isnan(fit.rotation).all() and fit.points_used == 2


def test_occluded_marks_inside_and_near_points():
    square = [np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])]
    points = np.array([[50.0, 50.0], [105.0, 50.0], [120.0, 50.0], [-3.0, -3.0]])
    np.testing.assert_array_equal(occluded(points, square, pad=10.0),
                                  [True, True, False, True])
    assert not occluded(points, [], pad=10.0).any()
```

- [ ] **Step 6: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_head_converter.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 7: Implement the first part of `head_converter.py`**

```python
"""
SMPL-X neck, head and jaw from MediaPipe FaceLandmarker landmarks (V1.3 design §4).

Head orientation: a weighted Umeyama fit from the 17 rigid face points on the SMPL-X rest mesh to
the observed points in pixels (x·W, y·H, z·W). At rest every SMPL-X rotation is the identity, so
the fitted rotation is the head's global rotation; scale and translation are discarded. Rigid
points under either hand's padded image outline get zero weight, and one robust pass drops
outliers, because hands over the face are the failure mode prior sign-language work reports.

Using the rotation in the pose world frame assumes the pixel axes match the pose world axes
(assumption F2): both are x right, y down, z away from the camera.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from matplotlib.path import Path as Polygon
from scipy.spatial import ConvexHull, QhullError

from .alignment import apply_similarity, similarity_transform

OCCLUSION_PAD = 0.15     # padding around a hand outline, fraction of the outer-eye-corner distance
OUTLIER_FACTOR = 2.5     # residual above this many weighted medians is dropped
RESIDUAL_FLOOR = 0.01    # ... but never below this fraction of the outer-eye-corner distance
MIN_RIGID_POINTS = 8     # fewer surviving rigid points: the frame is unreliable
MAX_RMS_FRACTION = 0.10  # provisional; revisited once the floor (metric 1) is measured


def hand_outlines(hand_image: np.ndarray, width: int, height: int) -> List[np.ndarray]:
    """
    Each present hand's convex outline in pixels.

    Args:
        hand_image: (2, 21, 3) one frame of `HandClip.image` (x / width, y / height, z); NaN
            for an absent hand.
    """
    outlines = []
    for side in range(hand_image.shape[0]):
        points = hand_image[side, :, :2].astype(np.float64) * [width, height]
        if not np.isfinite(points).all():
            continue
        try:
            outlines.append(points[ConvexHull(points).vertices])
        except QhullError:  # degenerate (collinear) detection: fall back to its bounding box
            lo, hi = points.min(axis=0), points.max(axis=0)
            outlines.append(np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]))
    return outlines


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def occluded(points: np.ndarray, outlines: List[np.ndarray], pad: float) -> np.ndarray:
    """(N,) True where a 2-D point is inside an outline or within `pad` pixels of one."""
    points = np.asarray(points, dtype=np.float64)
    hidden = np.zeros(len(points), dtype=bool)
    for outline in outlines:
        inside = Polygon(outline).contains_points(points)
        edge = np.min([_segment_distance(points, outline[i], outline[(i + 1) % len(outline)])
                       for i in range(len(outline))], axis=0)
        hidden |= inside | (edge <= pad)
    return hidden


@dataclass
class HeadRotation:
    rotation: np.ndarray  # (3, 3) head's global rotation in camera axes; NaN if not fitted
    rms_fraction: float   # weighted RMS residual / reference_length
    points_used: int      # points with non-zero final weight
    weights: np.ndarray   # (N,) final weights after the outlier pass


def head_rotation(template, observed, weights, reference_length: float) -> HeadRotation:
    """
    Head rotation from rigid points, robust to occlusion and a few outliers.

    Args:
        template: (N, 3) the rigid points on the SMPL-X rest mesh, metres.
        observed: (N, 3) the same points observed, in pixels.
        weights: (N,) initial weights; 0 for occluded points.
        reference_length: the observed outer-eye-corner distance, pixels.
    """
    template = np.asarray(template, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    if (w > 0).sum() < 3:
        return HeadRotation(np.full((3, 3), np.nan), float("nan"), int((w > 0).sum()), w)

    def fit(w):
        s, r, t = similarity_transform(template, observed, weights=w)
        return r, np.linalg.norm(apply_similarity(template, s, r, t) - observed, axis=1)

    rotation, residual = fit(w)
    cut = max(OUTLIER_FACTOR * float(np.median(residual[w > 0])),
              RESIDUAL_FLOOR * reference_length)
    trimmed = np.where(residual > cut, 0.0, w)
    if (trimmed > 0).sum() >= 3 and (trimmed != w).any():
        w = trimmed
        rotation, residual = fit(w)
    rms = float(np.sqrt((w @ residual ** 2) / w.sum())) / reference_length
    return HeadRotation(rotation, rms, int((w > 0).sum()), w)
```

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_head_converter.py tests/test_alignment.py -v`
Expected: all PASS.

- [ ] **Step 9: Index and commit**

Add `head_converter.py` to `src/kinetic_augment/body_model/index.md`
(`— Fits the SMPL-X neck, head and jaw to MediaPipe face landmarks.`) and
`test_head_converter.py` to `tests/index.md`.

```bash
git add src/kinetic_augment/body_model/alignment.py src/kinetic_augment/body_model/head_converter.py \
  tests/test_alignment.py tests/test_head_converter.py src/kinetic_augment/body_model/index.md tests/index.md
git commit -m "feat(face): occlusion-aware weighted head rotation"
```

---

### Task 5: Jaw pitch from the lip gap

**Files:**
- Modify: `src/kinetic_augment/body_model/head_converter.py`, `tests/test_head_converter.py`

**Interfaces:**
- Consumes: `pose_mesh`, `JAW` (`assembly.py`); `FaceCorrespondence`, `JAW_PAIR`,
  `RIGHT_EYE_OUTER`, `LEFT_EYE_OUTER` (`face_correspondence.py`).
- Produces: `JAW_GRID_DEG`, `JAW_AXIS = np.array([1.0, 0.0, 0.0])`,
  `lip_gap_ratio(upper, lower, right_outer, left_outer) -> float`,
  `JawCurve(pitch_rad (K,), gap_ratio (K,))`, `jaw_curve(skeleton, corr) -> JawCurve`,
  `fit_jaw_pitch(ratio, curve) -> float` (radians).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_head_converter.py`:

```python
from tests.face_assets import MODEL_DIR, requires_face_assets  # noqa: E402

BETAS = np.zeros(300)
BETAS[[0, 2, 7]] = [0.8, -0.6, 1.2]


@pytest.fixture(scope="module")
def body():
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton
    return SMPLXSkeleton(model_path=MODEL_DIR).shaped(BETAS)


@pytest.fixture(scope="module")
def corr():
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    return load_face_correspondence(MODEL_DIR)


@pytest.fixture(scope="module")
def curve(body, corr):
    from kinetic_augment.body_model.head_converter import jaw_curve
    return jaw_curve(body, corr)


@requires_face_assets
def test_jaw_curve_opens_with_positive_pitch(curve):
    from kinetic_augment.body_model.head_converter import JAW_GRID_DEG

    assert len(curve.pitch_rad) == len(JAW_GRID_DEG)
    assert np.all(np.diff(curve.gap_ratio) >= 0)
    assert curve.gap_ratio[40] > curve.gap_ratio[0] + 0.05  # 10 degrees opens the lips visibly


@requires_face_assets
def test_fit_jaw_pitch_recovers_a_posed_jaw(body, corr, curve):
    from kinetic_augment.body_model.assembly import JAW, pose_mesh
    from kinetic_augment.body_model.head_converter import JAW_AXIS, fit_jaw_pitch, lip_gap_ratio

    rotations = np.zeros((55, 3))
    rotations[JAW] = np.radians(12.3) * JAW_AXIS
    vertices = pose_mesh(body, rotations, body.betas, np.zeros(3)).vertices
    upper, lower, right, left = corr.points(vertices, (13, 14, 33, 263))
    pitch = fit_jaw_pitch(lip_gap_ratio(upper, lower, right, left), curve)
    assert abs(np.degrees(pitch) - 12.3) < 1.0


@requires_face_assets
def test_fit_jaw_pitch_clamps_to_the_grid(curve):
    from kinetic_augment.body_model.head_converter import fit_jaw_pitch

    assert fit_jaw_pitch(-1.0, curve) == 0.0
    assert np.isclose(np.degrees(fit_jaw_pitch(10.0, curve)), 30.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_head_converter.py -v -k jaw`
Expected: FAIL with `ImportError: cannot import name 'jaw_curve'`.

- [ ] **Step 3: Implement the jaw functions**

Append to `head_converter.py` (and add the imports at the top:
`from .assembly import JAW, pose_mesh` and
`from .face_correspondence import JAW_PAIR, LEFT_EYE_OUTER, RIGHT_EYE_OUTER`):

```python
JAW_GRID_DEG = np.arange(0.0, 30.0 + 1e-9, 0.25)
JAW_AXIS = np.array([1.0, 0.0, 0.0])  # SMPL-X jaw pitch; positive opens the mouth


def lip_gap_ratio(upper, lower, right_outer, left_outer) -> float:
    """Inner-lip gap over the outer-eye-corner distance: scale-free, so pixels or metres."""
    return float(np.linalg.norm(np.asarray(upper) - lower)
                 / np.linalg.norm(np.asarray(right_outer) - left_outer))


@dataclass
class JawCurve:
    pitch_rad: np.ndarray  # (K,) jaw pitch grid
    gap_ratio: np.ndarray  # (K,) lip-gap ratio at each pitch; non-decreasing


def jaw_curve(skeleton, corr) -> JawCurve:
    """Lip-gap ratio of the SMPL-X mesh over the jaw-pitch grid, for this body shape."""
    names = JAW_PAIR + (RIGHT_EYE_OUTER, LEFT_EYE_OUTER)
    gaps = []
    for degrees in JAW_GRID_DEG:
        rotations = np.zeros((55, 3))
        rotations[JAW] = np.radians(degrees) * JAW_AXIS
        vertices = pose_mesh(skeleton, rotations, skeleton.betas, np.zeros(3)).vertices
        gaps.append(lip_gap_ratio(*corr.points(vertices, names)))
    return JawCurve(np.radians(JAW_GRID_DEG), np.maximum.accumulate(np.array(gaps)))


def fit_jaw_pitch(ratio: float, curve: JawCurve) -> float:
    """Jaw pitch (radians) reproducing an observed lip-gap ratio, clamped to the grid."""
    # A tiny ramp makes the abscissa strictly increasing where the curve is flat (closed lips).
    xp = curve.gap_ratio + 1e-12 * np.arange(len(curve.gap_ratio))
    return float(np.interp(ratio, xp, curve.pitch_rad))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_head_converter.py -v`
Expected: all PASS. If `test_jaw_curve_opens_with_positive_pitch` fails because the gap shrinks
or stays flat, the jaw opens with negative pitch on this model: set
`JAW_AXIS = np.array([-1.0, 0.0, 0.0])`, re-run, and note it in the module comment.

- [ ] **Step 5: Commit**

```bash
git add src/kinetic_augment/body_model/head_converter.py tests/test_head_converter.py
git commit -m "feat(face): jaw pitch from the inner-lip gap"
```

---

### Task 6: Neck/head split and `fit_head`

**Files:**
- Modify: `src/kinetic_augment/body_model/head_converter.py`, `tests/test_head_converter.py`

**Interfaces:**
- Consumes: `global_rotations`, `attach_head`, `pose_mesh`, `NECK`, `HEAD` (`assembly.py`);
  `forward_kinematics` (`kinematics.py`); `swing_rotation` (`swing_ik.py`); `RIGID`,
  `JAW_PAIR`, eye-corner constants (`face_correspondence.py`).
- Produces:
  - `NECK_YAW_SHARE = 0.4`, `FALLBACK_SHARES = (0.4, 0.7, 0.5)`, `EULER_ORDER = "YXZ"`,
    `EAR_VERTICES` (SMPL-X `lear`, `rear`).
  - `twist_angle(rotation, axis) -> float`
  - `NeckSplit(neck (3,), head (3,), mode, ear_residual_m, ear_residual_rest_m)`
  - `split_neck_head(rotations, transl, skeleton, ear_rest, head_world, observed_ear_mid=None) -> NeckSplit`
  - `HeadTemplate(skeleton, corr, rigid_rest (17,3), ear_rest (3,), jaw: JawCurve)`,
    `head_template(skeleton, corr) -> HeadTemplate`
  - `HeadFit(neck_pose, head_pose, jaw_pose, head_world (3,3), rigid_weights (17,), valid, reliable, jaw_reliable, rms_fraction, points_used, neck_mode, ear_residual_m, ear_residual_rest_m)`
  - `fit_head(face_px, outlines, rotations, transl, template, observed_ear_mid=None) -> HeadFit`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_head_converter.py`:

```python
SCALE, OFFSET = 3000.0, np.array([300.0, 150.0, 40.0])


def posed_body_rotations():
    """A y-down world like MediaPipe's: the body turned upside down, slightly rotated."""
    rotations = np.zeros((55, 3))
    rotations[0] = (Rotation.from_rotvec([np.pi, 0, 0])
                    * Rotation.from_rotvec([0.05, 0.3, -0.04])).as_rotvec()
    rotations[16:22] = 0.3 * np.random.default_rng(4).normal(size=(6, 3))  # arms
    return rotations


def observe(body, corr, rotations, transl):
    """Pixel landmarks (478, 3; NaN off the embedding) and the world ear midpoint."""
    from kinetic_augment.body_model.assembly import pose_mesh
    from kinetic_augment.body_model.head_converter import EAR_VERTICES

    mesh = pose_mesh(body, rotations, body.betas, transl)
    face_px = np.full((478, 3), np.nan)
    face_px[corr.mp_indices] = SCALE * corr.points(mesh.vertices) + OFFSET
    return face_px, mesh.vertices[list(EAR_VERTICES)].mean(axis=0), mesh


@pytest.fixture(scope="module")
def template(body, corr):
    from kinetic_augment.body_model.head_converter import head_template
    return head_template(body, corr)


@requires_face_assets
@pytest.mark.parametrize("with_ears", [True, False])
def test_fit_head_recovers_head_orientation_and_jaw(body, corr, template, with_ears):
    from kinetic_augment.body_model.assembly import HEAD, attach_head, global_rotations
    from kinetic_augment.body_model.head_converter import fit_head

    truth = posed_body_rotations()
    truth[12] = [0.15, 0.25, -0.05]
    truth[15] = [0.1, -0.2, 0.08]
    truth[22] = [np.radians(9.0), 0, 0]
    transl = np.array([0.02, -0.1, 0.3])
    face_px, ears, _ = observe(body, corr, truth, transl)
    body_only = truth.copy()
    body_only[[12, 15, 22]] = 0.0

    fit = fit_head(face_px, [], body_only, transl, template, ears if with_ears else None)
    assert fit.valid and fit.reliable and fit.jaw_reliable
    assert fit.neck_mode == ("swing" if with_ears else "split")
    got = global_rotations(attach_head(body_only, fit), body.parents)[HEAD]
    want = global_rotations(truth, body.parents)[HEAD]
    assert angle_deg(got, want) < 0.5
    assert abs(np.degrees(fit.jaw_pose[0]) - 9.0) < 1.0
    if with_ears:
        # The ear vertices are partly skinned to the neck and moved by pose correctives, so the
        # rigid-head ear model is approximate: millimetres, not exact.
        assert fit.ear_residual_m < 0.005


@requires_face_assets
def test_neck_swing_leaves_the_arms_alone(body, corr, template):
    from kinetic_augment.body_model.assembly import attach_head
    from kinetic_augment.body_model.head_converter import fit_head
    from kinetic_augment.body_model.kinematics import forward_kinematics

    truth = posed_body_rotations()
    truth[12] = [0.3, 0.0, 0.1]
    face_px, ears, _ = observe(body, corr, truth, np.zeros(3))
    body_only = truth.copy()
    body_only[12] = 0.0
    fit = fit_head(face_px, [], body_only, np.zeros(3), template, ears)
    before = forward_kinematics(body_only, body)[16:22]
    after = forward_kinematics(attach_head(body_only, fit), body)[16:22]
    np.testing.assert_allclose(after, before, atol=1e-12)


@requires_face_assets
def test_no_face_gives_an_invalid_rest_fit(template):
    from kinetic_augment.body_model.head_converter import fit_head

    fit = fit_head(None, [], posed_body_rotations(), np.zeros(3), template)
    assert not fit.valid and not fit.reliable and not fit.jaw_reliable
    assert not fit.neck_pose.any() and not fit.head_pose.any() and not fit.jaw_pose.any()


@requires_face_assets
def test_a_hand_over_the_mouth_keeps_the_jaw_at_rest(body, corr, template):
    from kinetic_augment.body_model.head_converter import fit_head

    truth = posed_body_rotations()
    truth[22] = [np.radians(15.0), 0, 0]
    face_px, ears, _ = observe(body, corr, truth, np.zeros(3))
    mouth = face_px[[13, 14], :2].mean(axis=0)
    square = [mouth + np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])]
    fit = fit_head(face_px, square, truth, np.zeros(3), template, ears)
    assert not fit.jaw_reliable and not fit.jaw_pose.any()


@requires_face_assets
def test_a_hand_over_most_of_the_face_is_unreliable(body, corr, template):
    from kinetic_augment.body_model.head_converter import fit_head

    face_px, ears, _ = observe(body, corr, posed_body_rotations(), np.zeros(3))
    lo, hi = np.nanmin(face_px[:, :2], axis=0), np.nanmax(face_px[:, :2], axis=0)
    cover = [np.array([lo - 50, [hi[0] + 50, lo[1] - 50], hi + 50, [lo[0] - 50, hi[1] + 50]])]
    fit = fit_head(face_px, cover, posed_body_rotations(), np.zeros(3), template, ears)
    assert fit.valid and not fit.reliable
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_head_converter.py -v -k "fit_head or neck or face or hand"`
Expected: FAIL with `ImportError: cannot import name 'head_template'`.

- [ ] **Step 3: Implement the split, the template and `fit_head`**

Replace the `.assembly` and `.face_correspondence` imports added in Task 5, so the top of
`head_converter.py` imports:

```python
from typing import List, Optional

from scipy.spatial.transform import Rotation

from .assembly import HEAD, JAW, NECK, global_rotations, pose_mesh
from .face_correspondence import (
    JAW_PAIR, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, RIGID,
)
from .kinematics import forward_kinematics
from .swing_ik import swing_rotation
```

Append:

```python
NECK_YAW_SHARE = 0.4               # the atlanto-axial joint provides most cervical rotation
FALLBACK_SHARES = (0.4, 0.7, 0.5)  # neck's share of (yaw, pitch, roll) when the ears are missing
EULER_ORDER = "YXZ"                # SMPL-X rest: y up, x to the subject's left, z forward


def _ear_vertices():
    from smplx.vertex_ids import vertex_ids
    return (vertex_ids["smplx"]["lear"], vertex_ids["smplx"]["rear"])


EAR_VERTICES = _ear_vertices()


def twist_angle(rotation: np.ndarray, axis: np.ndarray) -> float:
    """Angle (radians) of the twist component of `rotation` about the unit `axis`."""
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return float(2.0 * np.arctan2(np.dot([x, y, z], axis), w))


@dataclass
class NeckSplit:
    neck: np.ndarray            # (3,) neck local axis-angle
    head: np.ndarray            # (3,) head local axis-angle
    mode: str                   # "swing" (ears observed) or "split" (anatomical shares)
    ear_residual_m: float       # |fitted - observed ear midpoint|; NaN without an observation
    ear_residual_rest_m: float  # the same with the neck at rest and the head still oriented


def split_neck_head(rotations, transl, skeleton, ear_rest, head_world,
                    observed_ear_mid=None) -> NeckSplit:
    """
    Neck and head local rotations that reproduce `head_world` exactly (design §4.3).

    With an observed ear midpoint, the neck swings so the SMPL-X ear midpoint lands on it and
    takes NECK_YAW_SHARE of the head's twist about the neck bone; otherwise it takes
    FALLBACK_SHARES of the relative yaw, pitch and roll. The head takes the exact remainder.
    """
    rot = np.array(rotations, dtype=np.float64)
    rot[[NECK, HEAD]] = 0.0
    parents = skeleton.parents
    torso = global_rotations(rot, parents)[parents[NECK]]
    p_neck = (forward_kinematics(rot, skeleton) + np.asarray(transl))[NECK]
    rest = skeleton.rest_joints
    bone = rest[HEAD] - rest[NECK]
    axis = bone / np.linalg.norm(bone)
    ear_offset = np.asarray(ear_rest) - rest[HEAD]
    relative = torso.T @ head_world

    def ear_mid(neck_local):
        return p_neck + torso @ neck_local @ bone + head_world @ ear_offset

    observed = observed_ear_mid is not None and np.isfinite(observed_ear_mid).all()
    if observed:
        target = np.asarray(observed_ear_mid) - p_neck - head_world @ ear_offset
        swing_world = swing_rotation(torso @ axis, target / np.linalg.norm(target))
        swing = torso.T @ swing_world @ torso
        twist = Rotation.from_rotvec(NECK_YAW_SHARE * twist_angle(relative, axis) * axis)
        neck = swing @ twist.as_matrix()
        mode = "swing"
    else:
        angles = Rotation.from_matrix(relative).as_euler(EULER_ORDER)
        neck = Rotation.from_euler(EULER_ORDER, angles * np.array(FALLBACK_SHARES)).as_matrix()
        mode = "split"
    head = (torso @ neck).T @ head_world

    def residual(neck_local):
        if not observed:
            return float("nan")
        return float(np.linalg.norm(ear_mid(neck_local) - observed_ear_mid))

    return NeckSplit(Rotation.from_matrix(neck).as_rotvec(),
                     Rotation.from_matrix(head).as_rotvec(), mode,
                     residual(neck), residual(np.eye(3)))


@dataclass
class HeadTemplate:
    """Everything `fit_head` needs from one body shape, computed once per clip."""

    skeleton: object         # SMPLXSkeleton in the clip's shape
    corr: object             # FaceCorrespondence
    rigid_rest: np.ndarray   # (17, 3) rigid points on the rest mesh, metres
    ear_rest: np.ndarray     # (3,) SMPL-X ear midpoint at rest
    jaw: JawCurve


def head_template(skeleton, corr) -> HeadTemplate:
    rest = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3)).vertices
    return HeadTemplate(skeleton, corr, corr.points(rest, RIGID),
                        rest[list(EAR_VERTICES)].mean(axis=0), jaw_curve(skeleton, corr))


@dataclass
class HeadFit:
    """
    SMPL-X neck, head and jaw for one frame; expression stays zero.

    `valid` is False when no face was observed (everything at rest). `reliable` is False when too
    few rigid points survived occlusion or the fit residual is too large; the rotation is kept.
    `jaw_reliable` is False when a hand covers the mouth (jaw at rest).
    """

    neck_pose: np.ndarray
    head_pose: np.ndarray
    jaw_pose: np.ndarray
    head_world: np.ndarray = None       # (3, 3) fitted global head rotation; NaN if none
    rigid_weights: np.ndarray = None    # (17,) final rigid-point weights
    valid: bool = True
    reliable: bool = True
    jaw_reliable: bool = True
    rms_fraction: float = float("nan")
    points_used: int = 0
    neck_mode: str = "rest"
    ear_residual_m: float = float("nan")
    ear_residual_rest_m: float = float("nan")


def _rest_fit(**flags) -> HeadFit:
    return HeadFit(np.zeros(3), np.zeros(3), np.zeros(3), head_world=np.full((3, 3), np.nan),
                   rigid_weights=np.zeros(len(RIGID)), **flags)


def fit_head(face_px, outlines, rotations, transl, template: HeadTemplate,
             observed_ear_mid=None) -> HeadFit:
    """
    Fit one frame's neck, head and jaw.

    Args:
        face_px: (478, 3) FaceLandmarker landmarks in pixels (`FaceClip.pixels()[t]`), or None.
            Only the rigid points and the jaw pair must be finite.
        outlines: this frame's hand outlines in pixels (`hand_outlines`).
        rotations: (55, 3) body rotations; neck, head and jaw entries are ignored.
        transl: (3,) body translation.
        template: `head_template` for the clip's body shape.
        observed_ear_mid: (3,) midpoint of the pose-world ears, or None.
    """
    needed = list(RIGID) + list(JAW_PAIR)
    if face_px is None or not np.isfinite(np.asarray(face_px)[needed]).all():
        return _rest_fit(valid=False, reliable=False, jaw_reliable=False)
    face_px = np.asarray(face_px, dtype=np.float64)
    reference = float(np.linalg.norm(face_px[RIGHT_EYE_OUTER] - face_px[LEFT_EYE_OUTER]))
    pad = OCCLUSION_PAD * reference
    rigid = face_px[list(RIGID)]
    weights = (~occluded(rigid[:, :2], outlines, pad)).astype(np.float64)
    fitted = head_rotation(template.rigid_rest, rigid, weights, reference)
    if not np.isfinite(fitted.rotation).all():
        return _rest_fit(valid=True, reliable=False, jaw_reliable=False,
                         points_used=fitted.points_used)

    split = split_neck_head(rotations, transl, template.skeleton, template.ear_rest,
                            fitted.rotation, observed_ear_mid)
    jaw_hidden = occluded(face_px[list(JAW_PAIR), :2], outlines, pad).any()
    if jaw_hidden:
        jaw = np.zeros(3)
    else:
        ratio = lip_gap_ratio(face_px[JAW_PAIR[0]], face_px[JAW_PAIR[1]],
                              face_px[RIGHT_EYE_OUTER], face_px[LEFT_EYE_OUTER])
        jaw = fit_jaw_pitch(ratio, template.jaw) * JAW_AXIS
    return HeadFit(
        neck_pose=split.neck, head_pose=split.head, jaw_pose=jaw,
        head_world=fitted.rotation, rigid_weights=fitted.weights,
        valid=True,
        reliable=fitted.points_used >= MIN_RIGID_POINTS
        and fitted.rms_fraction <= MAX_RMS_FRACTION,
        jaw_reliable=not jaw_hidden,
        rms_fraction=fitted.rms_fraction, points_used=fitted.points_used,
        neck_mode=split.mode, ear_residual_m=split.ear_residual_m,
        ear_residual_rest_m=split.ear_residual_rest_m,
    )
```

Note `test_a_hand_over_the_mouth_keeps_the_jaw_at_rest` passes `truth` with a nonzero jaw as
the body rotations; `fit_head` must ignore those entries (it does: `split_neck_head` zeroes neck
and head, and the jaw entry is never read).

- [ ] **Step 4: Run all head-converter tests**

Run: `.venv/bin/python -m pytest tests/test_head_converter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kinetic_augment/body_model/head_converter.py tests/test_head_converter.py
git commit -m "feat(face): neck swing to the observed ears and fit_head"
```

---

### Task 7: Head surface and ASL-LEX region labels

**Files:**
- Create: `src/kinetic_augment/body_model/face_regions.py`, `tests/test_face_regions.py`,
  `scripts/render_face_regions.py`, `docs/media/face_regions.png` (rendered)
- Modify: `src/kinetic_augment/body_model/index.md`, `scripts/index.md`, `tests/index.md`,
  `docs/media/index.md`

**Interfaces:**
- Consumes: `pose_mesh` (`assembly.py`); `FaceCorrespondence`, index groups, `load_flame_to_smplx`
  (`face_correspondence.py`); `EAR_VERTICES` (`head_converter.py`).
- Produces:
  - `REGIONS = ("HeadTop", "Forehead", "Eye", "CheekNose", "UpperLip", "Mouth", "Chin", "UnderChin", "Neck", "Ear")`,
    `SUBREGIONS = ("Brow", "Nose", "Cheek")`, `NO_LABEL = -1`.
  - `RegionAnchors` dataclass; `region_anchors(rest_vertices, corr) -> RegionAnchors`
  - `vertex_normals(vertices, faces) -> (V, 3)`
  - `label_regions(points, normals, dominant_joint, anchors) -> (region (N,) int8, subregion (N,) int8)`
  - `HeadSurface(vertex_ids (Vh,), faces (Fh,3) local, region (Vh,), subregion (Vh,))`
  - `build_head_surface(skeleton, corr, model_dir="models") -> HeadSurface`

- [ ] **Step 1: Write the failing tests**

`tests/test_face_regions.py`:

```python
"""Tests for the head surface and its ASL-LEX region labels (design §5, assumption F6)."""

import numpy as np
import pytest

from tests.face_assets import MODEL_DIR, requires_face_assets


def test_vertex_normals_of_a_flat_square_point_up():
    from kinetic_augment.body_model.face_regions import vertex_normals

    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    np.testing.assert_allclose(vertex_normals(vertices, faces), [[0, 0, 1]] * 4)


@pytest.fixture(scope="module")
def setup():
    from kinetic_augment.body_model.assembly import pose_mesh
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    from kinetic_augment.body_model.face_regions import build_head_surface
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton

    skeleton = SMPLXSkeleton(model_path=MODEL_DIR)
    corr = load_face_correspondence(MODEL_DIR)
    surface = build_head_surface(skeleton, corr, MODEL_DIR)
    rest = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3)).vertices
    return skeleton, corr, surface, rest


def label_near(setup, point):
    from kinetic_augment.body_model.face_regions import REGIONS, SUBREGIONS

    _, _, surface, rest = setup
    i = np.argmin(np.linalg.norm(rest[surface.vertex_ids] - point, axis=1))
    sub = surface.subregion[i]
    return REGIONS[surface.region[i]], (SUBREGIONS[sub] if sub >= 0 else None)


@requires_face_assets
def test_every_vertex_is_labelled_and_every_region_used(setup):
    from kinetic_augment.body_model.face_regions import REGIONS

    _, _, surface, _ = setup
    assert (surface.region >= 0).all()
    assert set(np.unique(surface.region)) == set(range(len(REGIONS)))
    assert len(surface.faces) > 0 and surface.faces.max() < len(surface.vertex_ids)


@requires_face_assets
def test_anchor_points_get_their_regions(setup):
    from kinetic_augment.body_model.head_converter import EAR_VERTICES

    _, corr, surface, rest = setup

    def p(*mp):
        return corr.points(rest, mp).mean(axis=0)

    brow_centre = p(105, 334)
    assert label_near(setup, p(13, 14))[0] == "Mouth"
    assert label_near(setup, p(4)) == ("CheekNose", "Nose")
    assert label_near(setup, p(17) + [0, -0.025, 0])[0] == "Chin"
    assert label_near(setup, brow_centre + [0, 0.03, 0])[0] == "Forehead"
    assert label_near(setup, p(105)) == ("Forehead", "Brow")
    assert label_near(setup, (p(2) + p(0)) / 2)[0] == "UpperLip"
    assert label_near(setup, rest[EAR_VERTICES[0]])[0] == "Ear"
    top = surface.vertex_ids[np.argmax(rest[surface.vertex_ids, 1])]
    assert label_near(setup, rest[top])[0] == "HeadTop"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_face_regions.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `face_regions.py`**

```python
"""
The SMPL-X head and neck surface, labelled with ASL-LEX location codes (V1.3 design §5).

The surface is every SMPL-X vertex in the FLAME vertex map and every triangle whose three
vertices are in it. Regions follow ASL-LEX 2.0's head locations, plus Neck and Ear, with finer
subregions underneath. They come from rules on the rest mesh -- the mapped MediaPipe landmarks,
vertex normals and SMPL-X skinning weights -- so no further licensed asset is needed. Thresholds
are provisional and are tuned by eye on `scripts/render_face_regions.py`'s image.

Rest frame: y up, x toward the subject's left, z forward.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from matplotlib.path import Path as Polygon
from scipy.spatial import ConvexHull

from .assembly import pose_mesh
from .face_correspondence import (
    LEFT_EYE, LEFT_EYEBROW, LOWER_LIP_BOTTOM, MOUTH_CORNERS, NOSE, OUTER_LIP, RIGHT_EYE,
    RIGHT_EYEBROW, SUBNASALE, UPPER_LIP_TOP, load_flame_to_smplx,
)

REGIONS = ("HeadTop", "Forehead", "Eye", "CheekNose", "UpperLip", "Mouth", "Chin", "UnderChin",
           "Neck", "Ear")
SUBREGIONS = ("Brow", "Nose", "Cheek")
NO_LABEL = -1

NECK_JOINT, JAW_JOINT, EYE_JOINTS = 12, 22, (23, 24)
EYE_RADIUS = 0.012      # metres from the eye contour
EAR_RADIUS = 0.035      # metres from an ear vertex
FRONT_NZ = 0.3          # normal z above this: front-facing
DOWN_NY = -0.5          # normal y below this: facing down
TOP_NY = 0.7            # normal y above this: facing up (scalp, not forehead)
BROW_BAND = 0.012       # metres either side of the mean brow height
FOREHEAD_HEIGHT = 0.07  # metres above the brows that still count as forehead


@dataclass
class RegionAnchors:
    eye_contours: np.ndarray      # (32, 3)
    brows: np.ndarray             # (20, 3)
    outer_lip: np.ndarray         # (20, 3) ordered loop
    nose: np.ndarray              # (13, 3)
    subnasale: np.ndarray         # (3,)
    upper_lip_top: np.ndarray     # (3,)
    lower_lip_bottom: np.ndarray  # (3,)
    mouth_corners: np.ndarray     # (2, 3)
    ears: np.ndarray              # (2, 3)


def region_anchors(rest_vertices, corr) -> RegionAnchors:
    from .head_converter import EAR_VERTICES

    def p(group):
        return corr.points(rest_vertices, group)

    return RegionAnchors(
        eye_contours=p(RIGHT_EYE + LEFT_EYE), brows=p(RIGHT_EYEBROW + LEFT_EYEBROW),
        outer_lip=p(OUTER_LIP), nose=p(NOSE), subnasale=p((SUBNASALE,))[0],
        upper_lip_top=p((UPPER_LIP_TOP,))[0], lower_lip_bottom=p((LOWER_LIP_BOTTOM,))[0],
        mouth_corners=p(MOUTH_CORNERS), ears=np.asarray(rest_vertices)[list(EAR_VERTICES)],
    )


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _nearest(points, anchors):
    return np.min(np.linalg.norm(points[:, None] - anchors[None], axis=-1), axis=1)


def label_regions(points, normals, dominant_joint, a: RegionAnchors) -> Tuple[np.ndarray, np.ndarray]:
    """
    Region and subregion of each rest-frame point. Rules apply in order; the first match wins.
    """
    n = len(points)
    region = np.full(n, NO_LABEL, dtype=np.int8)
    sub = np.full(n, NO_LABEL, dtype=np.int8)
    x, y = points[:, 0], points[:, 1]
    front = normals[:, 2] > FRONT_NZ             # facing forward: lips
    ahead = points[:, 2] > a.ears[:, 2].mean()   # in front of the ears: the sides of the face too

    def assign(mask, name):
        mask = mask & (region == NO_LABEL)
        region[mask] = REGIONS.index(name)
        return mask

    assign(np.isin(dominant_joint, EYE_JOINTS)
           | (_nearest(points, a.eye_contours) < EYE_RADIUS), "Eye")
    assign(dominant_joint == NECK_JOINT, "Neck")
    assign(_nearest(points, a.ears) < EAR_RADIUS, "Ear")
    assign(front & Polygon(a.outer_lip[:, :2]).contains_points(points[:, :2]), "Mouth")
    x_lo, x_hi = np.sort(a.mouth_corners[:, 0])
    assign(front & (x >= x_lo) & (x <= x_hi)
           & (y >= a.upper_lip_top[1]) & (y <= a.subnasale[1]), "UpperLip")
    jaw = dominant_joint == JAW_JOINT
    assign(jaw & (normals[:, 1] < DOWN_NY), "UnderChin")
    assign(jaw & ahead & (y < a.lower_lip_bottom[1]), "Chin")
    brow_y = a.brows[:, 1].mean()
    forehead = assign(ahead & (normals[:, 1] < TOP_NY)
                      & (y >= brow_y - BROW_BAND) & (y <= brow_y + FOREHEAD_HEIGHT), "Forehead")
    brow = (forehead & (np.abs(y - brow_y) <= BROW_BAND)
            & (x >= a.brows[:, 0].min()) & (x <= a.brows[:, 0].max()))
    sub[brow] = SUBREGIONS.index("Brow")
    cheek_nose = assign(ahead & (y < brow_y), "CheekNose")
    hull = a.nose[ConvexHull(a.nose[:, :2]).vertices, :2]
    in_nose = Polygon(hull).contains_points(points[:, :2])
    sub[cheek_nose & in_nose] = SUBREGIONS.index("Nose")
    sub[cheek_nose & ~in_nose] = SUBREGIONS.index("Cheek")
    assign(np.ones(n, dtype=bool), "HeadTop")
    return region, sub


@dataclass(frozen=True)
class HeadSurface:
    vertex_ids: np.ndarray  # (Vh,) SMPL-X vertex ids
    faces: np.ndarray       # (Fh, 3) indices into vertex_ids
    region: np.ndarray      # (Vh,) int8 index into REGIONS
    subregion: np.ndarray   # (Vh,) int8 index into SUBREGIONS, NO_LABEL for none


def build_head_surface(skeleton, corr, model_dir="models") -> HeadSurface:
    """The labelled head and neck surface for one body shape."""
    vertex_ids = np.unique(load_flame_to_smplx(model_dir))
    rest = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3)).vertices
    all_faces = np.asarray(skeleton.model.faces, dtype=np.int64)
    local = np.full(len(rest), -1, dtype=np.int64)
    local[vertex_ids] = np.arange(len(vertex_ids))
    keep = (local[all_faces] >= 0).all(axis=1)
    normals = vertex_normals(rest, all_faces)[vertex_ids]
    dominant = skeleton.model.lbs_weights.detach().numpy()[vertex_ids].argmax(axis=1)
    region, sub = label_regions(rest[vertex_ids], normals, dominant, region_anchors(rest, corr))
    return HeadSurface(vertex_ids, local[all_faces[keep]], region, sub)
```

- [ ] **Step 4: Run the tests and tune by eye**

Run: `.venv/bin/python -m pytest tests/test_face_regions.py -v`
Expected: all PASS. If an anchor test fails, render (Step 5) and adjust only the threshold
constants or the rule order, never the anchors; record each change in the module comment.

- [ ] **Step 5: Render the segmentation**

`scripts/render_face_regions.py`:

```python
#!/usr/bin/env python3
"""
Render the head-region segmentation (V1.3 design §5) for visual review: front and side views of
the SMPL-X head vertices, coloured by region, brows and nose outlined by subregion.

Usage:
    python scripts/render_face_regions.py [--output docs/media/face_regions.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kinetic_augment.body_model.assembly import pose_mesh  # noqa: E402
from kinetic_augment.body_model.face_correspondence import load_face_correspondence  # noqa: E402
from kinetic_augment.body_model.face_regions import (  # noqa: E402
    REGIONS, SUBREGIONS, build_head_surface, vertex_normals,
)
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/media/face_regions.png"))
    args = parser.parse_args()
    skeleton = SMPLXSkeleton("models")
    surface = build_head_surface(skeleton, load_face_correspondence("models"))
    rest = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3)).vertices
    points = rest[surface.vertex_ids]
    normals = vertex_normals(rest, np.asarray(skeleton.model.faces, np.int64))[surface.vertex_ids]
    colours = plt.get_cmap("tab10")(surface.region)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, (axis, facing, title) in zip(axes, [(0, 2, "front"), (2, 0, "side (left)")]):
        visible = normals[:, facing] > 0
        order = np.argsort(points[visible, facing])
        p, c = points[visible][order], colours[visible][order]
        ax.scatter(p[:, axis], p[:, 1], c=c, s=4)
        sub = surface.subregion[visible][order]
        ax.scatter(p[sub == SUBREGIONS.index("Brow"), axis], p[sub == SUBREGIONS.index("Brow"), 1],
                   s=1, c="k")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=plt.get_cmap("tab10")(i), label=name)
               for i, name in enumerate(REGIONS)]
    fig.legend(handles=handles, loc="lower center", ncol=5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python scripts/render_face_regions.py`
Expected: `docs/media/face_regions.png` written. **Open the image and check it**: forehead above
the brows, chin below the lower lip, eyes around the eyes, ears at the sides. This is assumption
F6's evidence; the user reviews it at the Task 10 gate.

- [ ] **Step 6: Index and commit**

Add `face_regions.py`, `render_face_regions.py`, `test_face_regions.py` and
`face_regions.png` (`— Rendered SMPL-X head segmentation into ASL-LEX regions (V1.3).`) to their
folders' `index.md`.

```bash
git add src/kinetic_augment/body_model/face_regions.py scripts/render_face_regions.py \
  tests/test_face_regions.py docs/media/face_regions.png docs/media/index.md \
  src/kinetic_augment/body_model/index.md scripts/index.md tests/index.md
git commit -m "feat(face): head surface labelled with ASL-LEX regions"
```

---

### Task 8: Hand-to-face geometry

**Files:**
- Create: `src/kinetic_augment/body_model/hand_face_geometry.py`, `tests/test_hand_face_geometry.py`
- Modify: `src/kinetic_augment/body_model/index.md`, `tests/index.md`

**Interfaces:**
- Consumes: `PosedMesh`, `WRIST` (`assembly.py`); `HeadSurface`, `NO_LABEL` (`face_regions.py`).
- Produces:
  - `HAND_POINTS = ("wrist", "thumb", "index", "middle", "ring", "pinky", "palm")`,
    `CONTACT_DISTANCE = 0.02`, `SIDES = ("left", "right")`.
  - `hand_points(mesh) -> (2, 7, 3)`
  - `closest_points_on_triangles(points (P,3), triangles (F,3,3)) -> (closest (P,3), face (P,), distance (P,))`
  - `signed_distances(points, vertices, faces) -> (signed (P,), closest (P,3), face (P,))`
  - `HandFaceGeometry(distance (2,7), region (2,7) int8, subregion (2,7) int8, contact (2,7) bool)`
  - `hand_face_geometry(mesh, surface, hands_posed: Dict[str, bool]) -> HandFaceGeometry`

- [ ] **Step 1: Write the failing tests**

`tests/test_hand_face_geometry.py`:

```python
"""Tests for signed hand-to-face distances, regions and contact (design §5)."""

import numpy as np
import pytest

from kinetic_augment.body_model.hand_face_geometry import (
    closest_points_on_triangles, signed_distances,
)
from tests.face_assets import MODEL_DIR, requires_face_assets

TRIANGLE = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


@pytest.mark.parametrize("point, closest", [
    ([0.2, 0.2, 0.5], [0.2, 0.2, 0.0]),     # above the interior
    ([-1.0, -1.0, 0.0], [0.0, 0.0, 0.0]),   # beyond vertex A
    ([2.0, -0.5, 0.0], [1.0, 0.0, 0.0]),    # beyond vertex B
    ([0.5, -1.0, 0.3], [0.5, 0.0, 0.0]),    # beyond edge AB
    ([1.0, 1.0, 0.0], [0.5, 0.5, 0.0]),     # beyond edge BC
    ([-1.0, 0.5, 0.0], [0.0, 0.5, 0.0]),    # beyond edge AC
])
def test_closest_point_regions(point, closest):
    got, face, distance = closest_points_on_triangles(np.array([point]), TRIANGLE)
    np.testing.assert_allclose(got[0], closest, atol=1e-12)
    np.testing.assert_allclose(distance[0], np.linalg.norm(np.subtract(point, closest)))


def test_closest_point_agrees_with_dense_sampling():
    rng = np.random.default_rng(0)
    triangles = rng.normal(size=(5, 3, 3))
    points = rng.normal(size=(30, 3)) * 2
    _, _, distance = closest_points_on_triangles(points, triangles)
    u, v = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    keep = (u + v) <= 1
    u, v = u[keep], v[keep]
    samples = (triangles[:, None, 0] + u[:, None] * (triangles[:, None, 1] - triangles[:, None, 0])
               + v[:, None] * (triangles[:, None, 2] - triangles[:, None, 0])).reshape(-1, 3)
    sampled = np.min(np.linalg.norm(points[:, None] - samples[None], axis=-1), axis=1)
    assert (distance <= sampled + 1e-9).all()
    assert (distance >= sampled - 0.03).all()


def test_sign_follows_the_triangle_normal():
    vertices = TRIANGLE[0]
    faces = np.array([[0, 1, 2]])  # counter-clockwise seen from +z: normal +z
    signed, _, _ = signed_distances(np.array([[0.2, 0.2, 0.4], [0.2, 0.2, -0.1]]), vertices, faces)
    np.testing.assert_allclose(signed, [0.4, -0.1])


@pytest.fixture(scope="module")
def head():
    from kinetic_augment.body_model.assembly import pose_mesh
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    from kinetic_augment.body_model.face_regions import build_head_surface, vertex_normals
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton

    skeleton = SMPLXSkeleton(model_path=MODEL_DIR)
    corr = load_face_correspondence(MODEL_DIR)
    surface = build_head_surface(skeleton, corr, MODEL_DIR)
    mesh = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3))
    normals = vertex_normals(mesh.vertices, np.asarray(skeleton.model.faces, np.int64))
    return skeleton, corr, surface, mesh, normals


@requires_face_assets
def test_points_off_the_forehead_have_the_expected_signed_distance(head):
    _, corr, surface, mesh, normals = head
    brow = corr.points(mesh.vertices, (105, 334)).mean(axis=0) + [0, 0.03, 0]
    ids = surface.vertex_ids
    v = ids[np.argmin(np.linalg.norm(mesh.vertices[ids] - brow, axis=1))]
    outside = mesh.vertices[v] + 0.03 * normals[v]
    inside = mesh.vertices[v] - 0.004 * normals[v]
    signed, _, _ = signed_distances(np.array([outside, inside]), mesh.vertices[ids], surface.faces)
    assert 0.027 < signed[0] <= 0.0301
    assert -0.0045 < signed[1] < 0


@requires_face_assets
def test_unposed_hands_are_nan_and_contact_needs_two_centimetres(head):
    from kinetic_augment.body_model.hand_face_geometry import CONTACT_DISTANCE, hand_face_geometry

    _, _, surface, mesh, _ = head
    geometry = hand_face_geometry(mesh, surface, {"left": False, "right": True})
    assert np.isnan(geometry.distance[0]).all() and (geometry.region[0] == -1).all()
    assert not geometry.contact[0].any()
    right = geometry.distance[1]
    assert np.isfinite(right).all()
    np.testing.assert_array_equal(geometry.contact[1], np.abs(right) < CONTACT_DISTANCE)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_hand_face_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `hand_face_geometry.py`**

```python
"""
Signed distances from hand points to the head and neck surface, with location labels
(V1.3 design §5).

Per hand: the wrist joint, the five fingertip vertices, and the palm centre (midpoint of the
wrist and the middle-finger knuckle). Each point's distance is to the nearest surface triangle,
signed by that triangle's outward normal (negative = inside the head); near concave regions such
as the ear the sign can be wrong for deep penetrations. Contact is |d| < 2 cm (self-contact
work's definition). The region is that of the nearest triangle vertex to the closest point.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .face_regions import NO_LABEL

HAND_POINTS = ("wrist", "thumb", "index", "middle", "ring", "pinky", "palm")
SIDES = ("left", "right")
CONTACT_DISTANCE = 0.02
WRIST_JOINT = {"left": 20, "right": 21}
MIDDLE_KNUCKLE = {"left": 28, "right": 43}


def _tip_vertices():
    from smplx.vertex_ids import vertex_ids
    ids = vertex_ids["smplx"]
    return {side: [ids[f"{side[0]}{finger}"] for finger in
                   ("thumb", "index", "middle", "ring", "pinky")] for side in SIDES}


TIP_VERTICES = _tip_vertices()


def hand_points(mesh) -> np.ndarray:
    """(2, 7, 3) hand points, in HAND_POINTS order, for the left and right hands."""
    out = np.empty((2, len(HAND_POINTS), 3))
    for s, side in enumerate(SIDES):
        wrist = mesh.joints[WRIST_JOINT[side]]
        out[s, 0] = wrist
        out[s, 1:6] = mesh.vertices[TIP_VERTICES[side]]
        out[s, 6] = (wrist + mesh.joints[MIDDLE_KNUCKLE[side]]) / 2.0
    return out


def _dot(u, v):
    return np.einsum("...k,...k->...", u, v)


def closest_points_on_triangles(points, triangles):
    """
    Exact closest point on any of the triangles to each point (Ericson, Real-Time Collision
    Detection §5.1.5, vectorised).

    Returns:
        (closest (P, 3), index of the nearest triangle (P,), distance (P,)).
    """
    p = np.asarray(points, dtype=np.float64)[:, None, :]
    tri = np.asarray(triangles, dtype=np.float64)[None]
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def put(mask, value):
        return np.where(mask[..., None], value, result)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        result = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]  # interior
        # Ericson tests A, B, AB, C, AC, BC in that order; applying them in reverse lets the
        # earlier test win where regions meet.
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        result = put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + t[..., None] * (c - b))
        t = d2 / (d2 - d6)
        result = put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + t[..., None] * ac)
        result = put((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, result.shape))
        t = d1 / (d1 - d3)
        result = put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + t[..., None] * ab)
        result = put((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, result.shape))
        result = put((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, result.shape))
    distance = np.linalg.norm(result - p, axis=-1)
    nearest = np.nanargmin(distance, axis=1)
    rows = np.arange(len(nearest))
    return result[rows, nearest], nearest, distance[rows, nearest]


def signed_distances(points, vertices, faces):
    """Distance to the mesh, negative behind the nearest triangle's outward normal."""
    tri = np.asarray(vertices)[np.asarray(faces)]
    closest, face, distance = closest_points_on_triangles(points, tri)
    normal = np.cross(tri[face, 1] - tri[face, 0], tri[face, 2] - tri[face, 0])
    sign = np.where(_dot(np.asarray(points) - closest, normal) < 0, -1.0, 1.0)
    return sign * distance, closest, face


@dataclass
class HandFaceGeometry:
    distance: np.ndarray   # (2, 7) signed metres; NaN for an unposed hand
    region: np.ndarray     # (2, 7) int8 index into REGIONS; NO_LABEL for an unposed hand
    subregion: np.ndarray  # (2, 7) int8 index into SUBREGIONS; NO_LABEL for none
    contact: np.ndarray    # (2, 7) bool, |distance| < CONTACT_DISTANCE


def hand_face_geometry(mesh, surface, hands_posed: Dict[str, bool]) -> HandFaceGeometry:
    shape = (2, len(HAND_POINTS))
    out = HandFaceGeometry(np.full(shape, np.nan), np.full(shape, NO_LABEL, np.int8),
                           np.full(shape, NO_LABEL, np.int8), np.zeros(shape, bool))
    sides = [s for s, side in enumerate(SIDES) if hands_posed.get(side, False)]
    if not sides:
        return out
    points = hand_points(mesh)[sides].reshape(-1, 3)
    head = mesh.vertices[surface.vertex_ids]
    signed, closest, face = signed_distances(points, head, surface.faces)
    corners = surface.faces[face]
    nearest = corners[np.arange(len(face)),
                      np.argmin(np.linalg.norm(head[corners] - closest[:, None], axis=-1), axis=1)]
    k = len(HAND_POINTS)
    for row, s in enumerate(sides):
        chunk = slice(row * k, (row + 1) * k)
        out.distance[s] = signed[chunk]
        out.region[s] = surface.region[nearest[chunk]]
        out.subregion[s] = surface.subregion[nearest[chunk]]
        out.contact[s] = np.abs(signed[chunk]) < CONTACT_DISTANCE
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_hand_face_geometry.py -v`
Expected: all PASS.

- [ ] **Step 5: Index and commit**

```bash
git add src/kinetic_augment/body_model/hand_face_geometry.py tests/test_hand_face_geometry.py \
  src/kinetic_augment/body_model/index.md tests/index.md
git commit -m "feat(face): signed hand-to-face distances with regions and contact"
```

---

### Task 9: Clip pipeline and the feature file

**Files:**
- Create: `src/kinetic_augment/body_model/face_pipeline.py`, `scripts/extract_face_features.py`,
  `tests/test_face_pipeline.py`
- Modify: `src/kinetic_augment/body_model/index.md`, `scripts/index.md`, `tests/index.md`

**Interfaces:**
- Consumes: everything from Tasks 1–8; `fit_shape` (`shape.py`), `fit_body` (`converter.py`),
  `fit_hand_shape`, `fit_hands`, `reconstruct_hand` (`hand_converter.py`), `HandModel`,
  `HandClip` (`hand_observations.py`; `.pose_world (T,33,3)`, `.pose_visibility (T,33)`,
  `.image (T,2,21,3)`, `.world (T,2,21,3)`, `.width`, `.height`, `.timestamp_ms`).
- Produces:
  - `VISIBILITY_THRESHOLD = 0.5`, `EARS = (7, 8)`
  - `FaceModels(skeleton, corr, mano, smplx_hands, model_dir)`, `load_face_models(model_dir="models")`
  - `hand_near_face(face_px (478,3), outlines, pad) -> bool`
  - `ClipFaceResult(head_fits, rotations (T,55,3), transl (T,3), betas, geometry: List[Optional[HandFaceGeometry]], hand_near_face (T,), body_ok (T,), frames (T,))`
  - `run_clip(hands, face, models, frames=None) -> ClipFaceResult`
  - `save_face_features(path, result, face)`

- [ ] **Step 1: Write the failing tests**

`tests/test_face_pipeline.py`:

```python
"""Tests for the clip pipeline: hand-near-face and a real pilot clip end to end."""

from pathlib import Path

import numpy as np
import pytest

from kinetic_augment.body_model.face_pipeline import hand_near_face
from tests.face_assets import MODEL_DIR, requires_face_assets

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "data" / "test_output" / "01991"


def face_box(lo=100.0, hi=200.0):
    face = np.full((478, 3), np.nan)
    face[:4] = [[lo, lo, 0], [hi, lo, 0], [hi, hi, 0], [lo, hi, 0]]
    return face


def test_hand_near_face_detects_overlap_and_containment():
    square = lambda c, r: np.array([[c - r, c - r], [c + r, c - r], [c + r, c + r], [c - r, c + r]])
    face = face_box()
    assert hand_near_face(face, [square(150.0, 10.0)], pad=5.0)    # hand inside the face box
    assert hand_near_face(face, [square(150.0, 500.0)], pad=5.0)   # hand covers the whole face
    assert hand_near_face(face, [square(208.0, 4.0)], pad=5.0)     # within the padding
    assert not hand_near_face(face, [square(400.0, 10.0)], pad=5.0)
    assert not hand_near_face(face, [], pad=5.0)


@requires_face_assets
def test_run_clip_on_pilot_frames(tmp_path):
    hands_path, face_path = Path(f"{CLIP}_hands.npz"), Path(f"{CLIP}_face.npz")
    if not (hands_path.exists() and face_path.exists()):
        pytest.skip("pilot hand/face clips absent")
    from kinetic_augment.body_model.face_observations import load_face_clip
    from kinetic_augment.body_model.face_pipeline import (
        load_face_models, run_clip, save_face_features,
    )
    from kinetic_augment.body_model.hand_observations import load_hand_clip

    hands, face = load_hand_clip(hands_path), load_face_clip(face_path)
    result = run_clip(hands, face, load_face_models(MODEL_DIR), frames=np.arange(0, 20))
    assert len(result.head_fits) == 20 and result.rotations.shape == (20, 55, 3)
    valid = [f for f in result.head_fits if f.valid]
    assert len(valid) >= 15
    assert np.median([f.rms_fraction for f in valid]) < 0.15
    save_face_features(tmp_path / "x_face_features.npz", result, face)
    data = np.load(tmp_path / "x_face_features.npz")
    assert data["distance"].shape == (20, 2, 7)
    assert data["blendshapes"].shape == (20, 52)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `face_pipeline.py`**

```python
"""
One clip end to end: body, hands and head fitted per frame, assembled into one SMPL-X pose,
and hand-to-face geometry measured on it (V1.3 design §3).

The body is fitted to the pose landmarks stored in the hand clip, so every stream shares the
same decoded frames. Hands are fitted with MANO and attached as the V1.2 viewer attaches them.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from matplotlib.path import Path as Polygon

from .assembly import attach_hands, attach_head, body_rotations, pose_mesh
from .converter import fit_body
from .face_correspondence import LEFT_EYE_OUTER, RIGHT_EYE_OUTER, load_face_correspondence
from .face_regions import REGIONS, SUBREGIONS, build_head_surface
from .hand_converter import fit_hand_shape, fit_hands, reconstruct_hand
from .hand_face_geometry import HAND_POINTS, HandFaceGeometry, hand_face_geometry
from .hand_model import HandModel
from .head_converter import (
    OCCLUSION_PAD, HeadFit, fit_head, hand_outlines, head_template,
)
from .shape import fit_shape
from .skeleton import SMPLXSkeleton

VISIBILITY_THRESHOLD = 0.5  # matches measure_round_trip.py
EARS = (7, 8)               # MediaPipe pose: left ear, right ear
SIDES = ("left", "right")
NECK_MODES = ("rest", "swing", "split")


@dataclass
class FaceModels:
    skeleton: SMPLXSkeleton
    corr: object
    mano: Dict[str, HandModel]         # fitting backend
    smplx_hands: Dict[str, HandModel]  # SMPL-X palms for attaching the wrists
    model_dir: str


def load_face_models(model_dir="models") -> FaceModels:
    return FaceModels(
        skeleton=SMPLXSkeleton(model_dir),
        corr=load_face_correspondence(model_dir),
        mano={s: HandModel("mano", s, model_dir=model_dir) for s in SIDES},
        smplx_hands={s: HandModel("smplx", s, model_dir=model_dir) for s in SIDES},
        model_dir=model_dir,
    )


def hand_near_face(face_px, outlines, pad: float) -> bool:
    """True if a hand outline comes within `pad` pixels of the face's bounding box."""
    points = np.asarray(face_px)[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points) or not outlines:
        return False
    lo, hi = points.min(axis=0) - pad, points.max(axis=0) + pad
    corners = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    for outline in outlines:
        if ((outline >= lo) & (outline <= hi)).all(axis=1).any():
            return True
        if Polygon(outline).contains_points(corners).any():
            return True
    return False


@dataclass
class ClipFaceResult:
    head_fits: List[HeadFit]
    rotations: np.ndarray         # (T, 55, 3); NaN where the body could not be fitted
    transl: np.ndarray            # (T, 3)
    betas: np.ndarray             # (300,)
    geometry: List[Optional[HandFaceGeometry]]
    hand_near_face: np.ndarray    # (T,) bool
    body_ok: np.ndarray           # (T,) bool
    frames: np.ndarray            # (T,) source frame indices


def _fit_hands(hands, mano, frames):
    fits, recon = {}, {}
    for s, side in enumerate(SIDES):
        observed = hands.world[frames, s].astype(np.float64)
        valid = np.isfinite(observed).all(axis=-1)
        betas = fit_hand_shape(observed, valid, mano[side])
        fits[side] = fit_hands(observed, valid, mano[side], betas)
        recon[side] = np.stack([reconstruct_hand(f, mano[side]) for f in fits[side]])
    return fits, recon


def run_clip(hands, face, models: FaceModels, frames=None) -> ClipFaceResult:
    """
    Fit and measure every frame (or `frames`, source indices) of one clip.

    Raises:
        ValueError: the hand and face clips do not share frames.
    """
    if not np.array_equal(hands.timestamp_ms, face.timestamp_ms):
        raise ValueError("hand and face clips are not frame-aligned")
    frames = np.arange(len(face.valid)) if frames is None else np.asarray(frames)
    pose = hands.pose_world.astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (hands.pose_visibility >= VISIBILITY_THRESHOLD)
    betas = fit_shape(pose, ok, models.skeleton)
    body = models.skeleton.shaped(betas)
    template = head_template(body, models.corr)
    surface = build_head_surface(body, models.corr, models.model_dir)
    hand_fits, hand_recon = _fit_hands(hands, models.mano, frames)
    face_px = face.pixels()

    n = len(frames)
    result = ClipFaceResult([], np.full((n, 55, 3), np.nan), np.full((n, 3), np.nan), betas,
                            [], np.zeros(n, bool), np.zeros(n, bool), frames)
    for i, t in enumerate(frames):
        outlines = hand_outlines(hands.image[t], hands.width, hands.height)
        observed_face = face_px[t] if face.valid[t] else None
        if observed_face is not None:
            reference = np.linalg.norm(observed_face[RIGHT_EYE_OUTER]
                                       - observed_face[LEFT_EYE_OUTER])
            result.hand_near_face[i] = hand_near_face(observed_face, outlines,
                                                      OCCLUSION_PAD * reference)
        try:
            body_fit = fit_body(pose[t], ok[t], body)
        except ValueError:
            result.head_fits.append(fit_head(None, [], np.zeros((55, 3)), np.zeros(3), template))
            result.geometry.append(None)
            continue
        result.body_ok[i] = True
        rotations, posed = attach_hands(
            body_rotations(body_fit), body.parents,
            {side: hand_fits[side][i] for side in SIDES},
            {side: hand_recon[side][i] for side in SIDES}, models.smplx_hands)
        ears = pose[t, list(EARS)].mean(axis=0) if ok[t, list(EARS)].all() else None
        head = fit_head(observed_face, outlines, rotations, body_fit.transl, template, ears)
        rotations = attach_head(rotations, head)
        mesh = pose_mesh(body, rotations, betas, body_fit.transl)
        result.head_fits.append(head)
        result.rotations[i], result.transl[i] = rotations, body_fit.transl
        result.geometry.append(hand_face_geometry(mesh, surface, posed))
    return result


def save_face_features(path, result: ClipFaceResult, face) -> None:
    """Write the per-frame face features (design §3) to .npz."""
    fits, n = result.head_fits, len(result.head_fits)
    shape = (n, 2, len(HAND_POINTS))
    distance = np.full(shape, np.nan)
    region = np.full(shape, -1, np.int8)
    subregion = np.full(shape, -1, np.int8)
    contact = np.zeros(shape, bool)
    for i, g in enumerate(result.geometry):
        if g is not None:
            distance[i], region[i], subregion[i], contact[i] = (g.distance, g.region,
                                                                g.subregion, g.contact)
    meta = {"regions": list(REGIONS), "subregions": list(SUBREGIONS),
            "hand_points": list(HAND_POINTS), "neck_modes": list(NECK_MODES),
            "blendshape_names": list(face.blendshape_names)}
    np.savez_compressed(
        path,
        frames=result.frames,
        neck_pose=np.array([f.neck_pose for f in fits]),
        head_pose=np.array([f.head_pose for f in fits]),
        jaw_pose=np.array([f.jaw_pose for f in fits]),
        valid=np.array([f.valid for f in fits]),
        reliable=np.array([f.reliable for f in fits]),
        jaw_reliable=np.array([f.jaw_reliable for f in fits]),
        rms_fraction=np.array([f.rms_fraction for f in fits]),
        points_used=np.array([f.points_used for f in fits]),
        neck_mode=np.array([NECK_MODES.index(f.neck_mode) for f in fits], np.int8),
        ear_residual_m=np.array([f.ear_residual_m for f in fits]),
        distance=distance, region=region, subregion=subregion, contact=contact,
        hand_near_face=result.hand_near_face, body_ok=result.body_ok,
        blendshapes=face.blendshapes[result.frames],
        betas=result.betas,
        meta=np.array(json.dumps(meta)),
    )
```

- [ ] **Step 4: Implement `scripts/extract_face_features.py`**

```python
#!/usr/bin/env python3
"""
Write per-clip face features (V1.3): fitted neck, head and jaw with their flags, signed
hand-to-face distances with ASL-LEX regions and contact, and the blendshapes.

Needs <clip>_hands.npz and <clip>_face.npz in --landmark-dir; writes <clip>_face_features.npz
next to them.

Usage:
    python scripts/extract_face_features.py [--landmark-dir data/test_output] [--clip 01991]
"""

import argparse
from pathlib import Path

from kinetic_augment.body_model.face_observations import load_face_clip
from kinetic_augment.body_model.face_pipeline import load_face_models, run_clip, save_face_features
from kinetic_augment.body_model.hand_observations import load_hand_clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-dir", type=Path, default=Path("data/test_output"))
    parser.add_argument("--clip", action="append", help="clip id; repeatable (default: all)")
    args = parser.parse_args()
    clips = args.clip or sorted(p.name[: -len("_face.npz")]
                                for p in args.landmark_dir.glob("*_face.npz"))
    models = load_face_models()
    for clip in clips:
        hands = load_hand_clip(args.landmark_dir / f"{clip}_hands.npz")
        face = load_face_clip(args.landmark_dir / f"{clip}_face.npz")
        result = run_clip(hands, face, models)
        path = args.landmark_dir / f"{clip}_face_features.npz"
        save_face_features(path, result, face)
        print(f"{clip}: {len(result.frames)} frames -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and the script on the pilot clips**

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py -v`
Expected: all PASS.

Run: `.venv/bin/python scripts/extract_face_features.py`
Expected: one line per pilot clip.

- [ ] **Step 6: Index and commit**

```bash
git add src/kinetic_augment/body_model/face_pipeline.py scripts/extract_face_features.py \
  tests/test_face_pipeline.py src/kinetic_augment/body_model/index.md scripts/index.md tests/index.md
git commit -m "feat(face): per-clip face pipeline and feature file"
```

---

### Task 10: The phase-A gate (metrics, run, findings) — then STOP

**Files:**
- Create: `src/kinetic_augment/body_model/face_metrics.py`, `tests/test_face_metrics.py`,
  `scripts/evaluate_face.py`, `docs/V1_3_Findings.md`
- Modify: `docs/V1_Problem_Scope.md` (ledger O10–O15), `docs/V1_3_Face_Design.md` (status line),
  `docs/index.md`, `src/kinetic_augment/body_model/index.md`, `scripts/index.md`, `tests/index.md`,
  `data/index.md`

**Interfaces:**
- Consumes: `FaceClip.head_matrix`, `HeadFit.head_world`, `ClipFaceResult`, `RIGID`,
  `pose_mesh`, `similarity_transform`.
- Produces:
  - `GL_TO_IMAGE = np.diag([1.0, -1.0, -1.0])`
  - `matrix_rotation(matrix (4,4)) -> (3,3)` (scale removed)
  - `relative_rotations(rotations (T,3,3)) -> (T,3,3)` (each relative to the clip's mean rotation)
  - `head_pose_errors_deg(fitted (T,3,3), head_matrix (T,4,4), mask (T,)) -> (T,)` (NaN off-mask)
  - `load_obj_vertices(path) -> (N,3)`
  - `correspondence_floor(canonical (468,3), rest_rigid (17,3)) -> float` (fraction of the outer-eye-corner distance)

- [ ] **Step 1: Write the failing metric tests**

`tests/test_face_metrics.py`:

```python
"""Tests for the V1.3 gate metrics (design §7)."""

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.face_metrics import (
    GL_TO_IMAGE, correspondence_floor, head_pose_errors_deg, matrix_rotation,
)


def test_matrix_rotation_removes_scale():
    rotation = Rotation.from_rotvec([0.1, 0.4, -0.2]).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = 1.7 * rotation
    np.testing.assert_allclose(matrix_rotation(matrix), rotation, atol=1e-12)


def test_head_pose_errors_vanish_for_a_consistent_pair():
    """
    Image-axis head rotations R_img(t) = F R_gl(t) C for a constant C: the relative rotations
    agree once the OpenGL axes are flipped, whatever C is.
    """
    rng = np.random.default_rng(0)
    gl = Rotation.from_rotvec(0.3 * rng.normal(size=(12, 3))).as_matrix()
    constant = Rotation.from_rotvec([0.5, -1.0, 0.2]).as_matrix()
    fitted = GL_TO_IMAGE @ gl @ constant
    matrices = np.tile(np.eye(4), (12, 1, 1))
    matrices[:, :3, :3] = 3.0 * gl
    errors = head_pose_errors_deg(fitted, matrices, np.ones(12, bool))
    assert np.nanmax(errors) < 1e-6


def test_correspondence_floor_is_zero_for_a_similar_copy():
    rng = np.random.default_rng(1)
    rest = 0.05 * rng.normal(size=(17, 3))
    canonical = np.zeros((468, 3))
    from kinetic_augment.body_model.face_correspondence import RIGID
    canonical[list(RIGID)] = 100.0 * rest @ Rotation.from_rotvec([0, 0.3, 0]).as_matrix().T
    assert correspondence_floor(canonical, rest) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_face_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `face_metrics.py`**

```python
"""
Metrics for the V1.3 gate (design §7) that need more than a mean: comparing the fitted head
rotation with FaceLandmarker's own head pose, and the correspondence floor.

FaceLandmarker's facial transformation matrix maps its canonical face into an OpenGL-style
camera (x right, y up, z toward the viewer) and includes a uniform scale. The fitted rotation is
in image axes (x right, y down, z away). The two differ by a constant on the face side and by
GL_TO_IMAGE on the camera side, so they are compared as rotations relative to each clip's mean:
Δ_image = F Δ_gl F, which cancels the constant.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .alignment import apply_similarity, similarity_transform
from .face_correspondence import LEFT_EYE_OUTER, RIGHT_EYE_OUTER, RIGID

GL_TO_IMAGE = np.diag([1.0, -1.0, -1.0])


def matrix_rotation(matrix) -> np.ndarray:
    """The proper rotation in a 4x4 similarity, with its scale removed."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64)[:3, :3])
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def relative_rotations(rotations) -> np.ndarray:
    mean = Rotation.from_matrix(rotations).mean().as_matrix()
    return np.asarray(rotations) @ mean.T


def head_pose_errors_deg(fitted, head_matrix, mask) -> np.ndarray:
    """Per-frame angle (degrees) between the two relative head rotations; NaN off `mask`."""
    mask = np.asarray(mask, bool) & np.isfinite(fitted).all(axis=(1, 2)) \
        & np.isfinite(head_matrix).all(axis=(1, 2))
    errors = np.full(len(mask), np.nan)
    if mask.sum() < 2:
        return errors
    ours = relative_rotations(np.asarray(fitted)[mask])
    gl = np.stack([matrix_rotation(m) for m in np.asarray(head_matrix)[mask]])
    theirs = GL_TO_IMAGE @ relative_rotations(gl) @ GL_TO_IMAGE
    errors[mask] = np.degrees(
        Rotation.from_matrix(np.transpose(ours, (0, 2, 1)) @ theirs).magnitude())
    return errors


def load_obj_vertices(path) -> np.ndarray:
    """Vertex positions ('v' lines) of an OBJ file."""
    with open(path) as handle:
        return np.array([[float(x) for x in line.split()[1:4]]
                         for line in handle if line.startswith("v ")])


def correspondence_floor(canonical, rest_rigid) -> float:
    """
    Metric 1's floor: the rigid-set residual of a scaled fit from SMPL-X's rest rigid points to
    MediaPipe's canonical face, as a fraction of the canonical outer-eye-corner distance. This
    error remains even with a perfect fit.
    """
    target = np.asarray(canonical, dtype=np.float64)[list(RIGID)]
    s, r, t = similarity_transform(rest_rigid, target)
    residual = np.linalg.norm(apply_similarity(rest_rigid, s, r, t) - target, axis=1)
    reference = np.linalg.norm(np.asarray(canonical)[RIGHT_EYE_OUTER]
                               - np.asarray(canonical)[LEFT_EYE_OUTER])
    return float(np.sqrt(np.mean(residual ** 2)) / reference)
```

Note: the design says "relative to its clip median"; the implementation uses the clip's mean
rotation (`scipy` chordal mean). Record this in the findings.

- [ ] **Step 4: Run the metric tests**

Run: `.venv/bin/python -m pytest tests/test_face_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Implement `scripts/evaluate_face.py`**

```python
#!/usr/bin/env python3
"""
The V1.3 phase-A gate (design §7) over every clip with <clip>_hands.npz and <clip>_face.npz.

Every metric is reported for hand-near-face frames and for the rest:
  1  rigid-set RMS residual, % outer-eye-corner distance, against the correspondence floor
  2  fitted head rotation vs FaceLandmarker's, relative to each clip's mean (degrees)
  3  per-clip Spearman correlation, fitted jaw pitch vs the jawOpen blendshape
  4  ear-midpoint residual, % shoulder width, with the neck swing and with the neck at rest
  5  fraction of frames with a hand point more than 1 cm inside the head
  6  fraction of frames flagged invalid / unreliable / jaw unreliable

Usage:
    python scripts/evaluate_face.py --landmark-dir data/round_trip_eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from kinetic_augment.body_model.assembly import pose_mesh
from kinetic_augment.body_model.face_correspondence import RIGID
from kinetic_augment.body_model.face_metrics import (
    correspondence_floor, head_pose_errors_deg, load_obj_vertices,
)
from kinetic_augment.body_model.face_observations import load_face_clip
from kinetic_augment.body_model.face_pipeline import (
    VISIBILITY_THRESHOLD, load_face_models, run_clip, save_face_features,
)
from kinetic_augment.body_model.hand_observations import load_hand_clip

SHOULDERS = (11, 12)
PENETRATION = -0.01
MIN_JAW_FRAMES, MIN_JAW_SPREAD = 10, 0.02
TARGETS = {"head_pose_median_deg": 5.0, "head_pose_p95_deg": 10.0, "jaw_spearman": 0.7}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "mean": float(values.mean())}


def clip_rows(hands, face, result):
    """Per-frame metric values for one clip, keyed by metric, plus the near-face split."""
    fits = result.head_fits
    pose = hands.pose_world.astype(np.float64)
    ok = np.isfinite(pose).all(-1) & (hands.pose_visibility >= VISIBILITY_THRESHOLD)
    both = ok[:, SHOULDERS[0]] & ok[:, SHOULDERS[1]]
    width = float(np.median(np.linalg.norm(pose[both, 11] - pose[both, 12], axis=-1)))
    frames = result.frames
    valid = np.array([f.valid for f in fits])
    fitted = np.stack([f.head_world for f in fits])
    head_pose = head_pose_errors_deg(fitted, face.head_matrix[frames], valid & face.valid[frames])
    penetrating = np.array([g is not None and bool(np.nanmin(np.append(g.distance, np.inf))
                                                   < PENETRATION) for g in result.geometry])
    posed = np.array([g is not None and np.isfinite(g.distance).any() for g in result.geometry])
    rows = {
        "rigid_rms_pct": 100 * np.array([f.rms_fraction for f in fits]),
        "head_pose_deg": head_pose,
        "ear_swing_pct": 100 * np.array([f.ear_residual_m for f in fits]) / width,
        "ear_rest_pct": 100 * np.array([f.ear_residual_rest_m for f in fits]) / width,
        "penetrating": np.where(posed, penetrating.astype(float), np.nan),
        "invalid": (~valid).astype(float),
        "unreliable": np.array([not f.reliable for f in fits], float),
        "jaw_unreliable": np.array([not f.jaw_reliable for f in fits], float),
    }
    jaw_ok = valid & np.array([f.jaw_reliable for f in fits])
    jaw_open = face.blendshapes[frames, list(face.blendshape_names).index("jawOpen")]
    if jaw_ok.sum() >= MIN_JAW_FRAMES and np.nanstd(jaw_open[jaw_ok]) >= MIN_JAW_SPREAD:
        pitch = np.array([f.jaw_pose[0] for f in fits])
        jaw = float(spearmanr(pitch[jaw_ok], jaw_open[jaw_ok]).correlation)
    else:
        jaw = None
    return rows, result.hand_near_face, jaw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-dir", type=Path, default=Path("data/round_trip_eval"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.landmark_dir / "face_report.json"
    models = load_face_models()
    rest = pose_mesh(models.skeleton, np.zeros((55, 3)), models.skeleton.betas,
                     np.zeros(3)).vertices
    floor = correspondence_floor(load_obj_vertices("models/mediapipe/canonical_face_model.obj"),
                                 models.corr.points(rest, RIGID))
    pooled = {"near": {}, "far": {}}
    jaw, clips = {}, sorted(p.name[: -len("_face.npz")]
                            for p in args.landmark_dir.glob("*_face.npz"))
    for clip in clips:
        hands = load_hand_clip(args.landmark_dir / f"{clip}_hands.npz")
        face = load_face_clip(args.landmark_dir / f"{clip}_face.npz")
        result = run_clip(hands, face, models)
        save_face_features(args.landmark_dir / f"{clip}_face_features.npz", result, face)
        rows, near, jaw[clip] = clip_rows(hands, face, result)
        for key, values in rows.items():
            pooled["near"].setdefault(key, []).extend(values[near])
            pooled["far"].setdefault(key, []).extend(values[~near])
        print(f"{clip}: {len(result.frames)} frames, {int(near.sum())} hand-near-face")
    report = {
        "clips": clips,
        "floor_rigid_rms_pct": 100 * floor,
        "targets": TARGETS,
        "jaw_spearman": jaw,
        "jaw_spearman_summary": stats([v for v in jaw.values() if v is not None]),
        "metrics": {split: {key: stats(v) for key, v in values.items()}
                    for split, values in pooled.items()},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("floor_rigid_rms_pct", "jaw_spearman_summary")},
                     indent=2))
    for split in ("far", "near"):
        print(f"\n{split}:")
        for key, s in report["metrics"][split].items():
            if s.get("n"):
                print(f"  {key:<16} n={s['n']:<6} median={s['median']:.3f} "
                      f"p95={s['p95']:.3f} mean={s['mean']:.3f}")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the gate on the 20 clips**

Run (extract faces for the evaluation clips, then evaluate):

```bash
.venv/bin/python scripts/extract_face_landmarks.py --clips-from data/round_trip_eval \
  --video-dir ~/Downloads/raw_videos --output-dir data/round_trip_eval
.venv/bin/python scripts/evaluate_face.py --landmark-dir data/round_trip_eval
```

Expected: 20 face files, then the printed metrics and `data/round_trip_eval/face_report.json`.
If the videos are not at `~/Downloads/raw_videos`, stop and ask the user where they are.

- [ ] **Step 7: Write the findings and update the ledger**

`docs/V1_3_Findings.md` follows `docs/Week_2_Findings.md`'s style: date and branch line, a
sources list naming `face_report.json` and the commands above, then one table per metric with
the near-face and other-frames rows side by side against the §7 targets, the floor, the jaw
correlations per clip, and a verdict per assumption F1–F6 (F6 cites
`docs/media/face_regions.png`). State plainly which targets were met and which were not, and
note that metric 2 uses the clip mean rotation rather than the median.

In `docs/V1_Problem_Scope.md` §12, move each of O10–O15 to `### Proven` (next `P` number) or
`### Disproven, retracted or superseded` (next `D` number) with its evidence, or keep it open with
an updated status. In `docs/V1_3_Face_Design.md`, change the status line to
`**Status:** phase A implemented; gate results in [V1.3 findings](V1_3_Findings.md).`

Index: `face_metrics.py`, `evaluate_face.py`, `test_face_metrics.py`, `V1_3_Findings.md`, and in
`data/index.md` the `round_trip_eval/` entry gains `face_report.json` and the
`*_face.npz` / `*_face_features.npz` caches.

- [ ] **Step 8: Commit**

```bash
git add src/kinetic_augment/body_model/face_metrics.py tests/test_face_metrics.py \
  scripts/evaluate_face.py docs/V1_3_Findings.md docs/V1_Problem_Scope.md docs/V1_3_Face_Design.md \
  docs/index.md src/kinetic_augment/body_model/index.md scripts/index.md tests/index.md data/index.md
git commit -m "feat(face): phase-A gate metrics and findings"
```

(Commit `face_report.json` only if `round_trip_report.json` is committed; it was not at planning
time because it derives from WLASL footage outside the repo.)

- [ ] **Step 9: STOP — user decision**

Report the gate results to the user. Phase C (Task 11) starts only after the user accepts phase A
(design §1). If a target failed, the design names the first suspects: F2 (axis conventions), then
the rigid set.

---

### Task 11: Phase C — the face stream

**Files:**
- Create: `src/kinetic_augment/body_model/face_stream.py`, `tests/test_face_stream.py`
- Modify: `src/kinetic_augment/body_model/face_pipeline.py` (stream in `run_clip`, `save_face_features`),
  `scripts/evaluate_face.py` (metric C1), `docs/V1_3_Findings.md`, `docs/V1_3_Face_Design.md`,
  `src/kinetic_augment/body_model/index.md`, `tests/index.md`

**Interfaces:**
- Consumes: `similarity_transform`, `apply_similarity`; `HeadFit.rigid_weights`, `.valid`;
  `global_rotations`, `HEAD`, `PosedMesh`; `RIGID`.
- Produces:
  - `align_face_stream(face_px (478,3), rigid_weights (17,), posed_rigid (17,3)) -> (478,3)` world metres
  - `to_head_frame(points, head_rotation (3,3), head_position (3,)) -> (N,3)`
  - `from_head_frame(local, head_rotation, head_position) -> (N,3)`
  - `ClipFaceResult.face_stream_local: (T, 478, 3)` (NaN where no face); feature key `face_stream_local`.

- [ ] **Step 1: Write the failing tests**

`tests/test_face_stream.py`:

```python
"""Tests for the phase-C face stream (design §6)."""

import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.body_model.face_correspondence import RIGID
from kinetic_augment.body_model.face_stream import (
    align_face_stream, from_head_frame, to_head_frame,
)


def test_alignment_recovers_world_points():
    rng = np.random.default_rng(0)
    world = 0.05 * rng.normal(size=(478, 3)) + [0.1, -0.5, 0.3]
    rotation = Rotation.from_rotvec([0.2, -0.4, 0.1]).as_matrix()
    face_px = 3000.0 * world @ rotation.T + [320, 200, 10]
    aligned = align_face_stream(face_px, np.ones(17), world[list(RIGID)])
    np.testing.assert_allclose(aligned, world, atol=1e-9)


def test_reposing_the_head_moves_the_stream_rigidly():
    rng = np.random.default_rng(1)
    world = rng.normal(size=(478, 3))
    r1 = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    p1 = np.array([0.0, -0.6, 0.1])
    local = to_head_frame(world, r1, p1)
    r2 = Rotation.from_rotvec([-0.3, 0.5, 0.0]).as_matrix()
    p2 = np.array([0.05, -0.55, 0.12])
    moved = from_head_frame(local, r2, p2)
    np.testing.assert_allclose(moved, (world - p1) @ (r2 @ r1.T).T + p2, atol=1e-12)
    np.testing.assert_allclose(from_head_frame(local, r1, p1), world, atol=1e-12)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_face_stream.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `face_stream.py`**

```python
"""
Phase C: MediaPipe's 478 face points attached to the fitted SMPL-X head (V1.3 design §6).

A scaled Umeyama fit from the observed rigid points (pixels) to the posed SMPL-X rigid points
(metres), with the head fit's occlusion weights, places all 478 points in the body's metric frame
at the scale of the fitted head. They are stored in the head joint's frame, so re-posing the head
carries them along rigidly. Blendshapes are kept unchanged as a separate stream.
"""

import numpy as np

from .alignment import apply_similarity, similarity_transform
from .face_correspondence import RIGID


def align_face_stream(face_px, rigid_weights, posed_rigid) -> np.ndarray:
    """(478, 3) observed face points carried into the posed body's frame, metres."""
    face_px = np.asarray(face_px, dtype=np.float64)
    s, r, t = similarity_transform(face_px[list(RIGID)], posed_rigid, weights=rigid_weights)
    return apply_similarity(face_px, s, r, t)


def to_head_frame(points, head_rotation, head_position) -> np.ndarray:
    return (np.asarray(points) - head_position) @ np.asarray(head_rotation)


def from_head_frame(local, head_rotation, head_position) -> np.ndarray:
    return np.asarray(local) @ np.asarray(head_rotation).T + head_position
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_face_stream.py -v`
Expected: all PASS.

- [ ] **Step 5: Add the stream and metric C1 to the pipeline**

In `face_pipeline.py`: import `HEAD`, `global_rotations` from `.assembly`, `RIGID` from
`.face_correspondence`, and `align_face_stream`, `to_head_frame` from `.face_stream`. Add three
fields at the end of `ClipFaceResult`, with defaults:

```python
    face_stream_local: np.ndarray = None  # (T, 478, 3) head-frame metres; NaN where no face
    c1_rigid_mm: np.ndarray = None        # (T,) metric C1 over the rigid embedded points
    c1_nonrigid_mm: np.ndarray = None     # (T,) metric C1 over the other embedded points
```

In `run_clip`, after building `result`, initialise them:

```python
    result.face_stream_local = np.full((n, 478, 3), np.nan)
    result.c1_rigid_mm = np.full(n, np.nan)
    result.c1_nonrigid_mm = np.full(n, np.nan)
    rigid_rows = np.isin(models.corr.mp_indices, RIGID)
```

and after `mesh = pose_mesh(...)` add:

```python
        if head.valid and (head.rigid_weights > 0).sum() >= 3:
            world = align_face_stream(face_px[t], head.rigid_weights,
                                      models.corr.points(mesh.vertices, RIGID))
            head_rotation = global_rotations(rotations, body.parents)[HEAD]
            result.face_stream_local[i] = to_head_frame(world, head_rotation, mesh.joints[HEAD])
            # C1: aligned MediaPipe points against their SMPL-X positions.
            diff = np.linalg.norm(world[models.corr.mp_indices]
                                  - models.corr.points(mesh.vertices), axis=1)
            result.c1_rigid_mm[i] = 1000 * np.sqrt(np.mean(diff[rigid_rows] ** 2))
            result.c1_nonrigid_mm[i] = 1000 * np.sqrt(np.mean(diff[~rigid_rows] ** 2))
```

In `save_face_features` add `face_stream_local=result.face_stream_local.astype(np.float32)`.
Extend `test_run_clip_on_pilot_frames` in `tests/test_face_pipeline.py` with:

```python
    stream = data["face_stream_local"][data["valid"]]
    assert np.isfinite(stream).all()
    assert np.nanmedian(result.c1_rigid_mm) < np.nanmedian(result.c1_nonrigid_mm)
```

Run: `.venv/bin/python -m pytest tests/test_face_pipeline.py tests/test_face_stream.py -v`
Expected: all PASS.

- [ ] **Step 6: Report C1 in the gate**

In `scripts/evaluate_face.py`'s `clip_rows`, add to `rows`:

```python
        "c1_rigid_mm": result.c1_rigid_mm,
        "c1_nonrigid_mm": result.c1_nonrigid_mm,
```

Re-run Task 10 Step 6's evaluation command.
Expected: `c1_rigid_mm` is small, since it measures the alignment on the points it was fitted to;
`c1_nonrigid_mm` is larger and reflects expression, which SMPL-X does not model here.

- [ ] **Step 7: Findings, design status, index, commit**

Add a "Phase C" section to `docs/V1_3_Findings.md` with metric C1. Change the design's status
line to `**Status:** implemented (phases A and C); results in [V1.3 findings](V1_3_Findings.md).`
Add `face_stream.py` and `test_face_stream.py` to their indexes.

```bash
git add src/kinetic_augment/body_model/face_stream.py tests/test_face_stream.py \
  src/kinetic_augment/body_model/face_pipeline.py tests/test_face_pipeline.py scripts/evaluate_face.py \
  docs/V1_3_Findings.md docs/V1_3_Face_Design.md src/kinetic_augment/body_model/index.md tests/index.md
git commit -m "feat(face): phase C face stream attached to the fitted head"
```

- [ ] **Step 8: Full suite**

Run: `.venv/bin/python -m pytest`
Expected: everything passes or skips for absent assets; no new failures against the pre-task
baseline.
