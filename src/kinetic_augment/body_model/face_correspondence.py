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
        if module.startswith("scipy.sparse."):  # FLAME's pickle names a deprecated submodule
            module = "scipy.sparse"
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
