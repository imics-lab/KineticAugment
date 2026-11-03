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
