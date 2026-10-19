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
