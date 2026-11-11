"""Which model files the Lab needs, where they are, and the saved Lab settings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Relative to the model folder; must match what `load_face_models` and the face extractor read.
REQUIRED_MODELS = (
    "smplx/SMPLX_NEUTRAL.npz",
    "mano/MANO_LEFT.npz",
    "mano/MANO_RIGHT.npz",
    "mano/SMPL-X__FLAME_vertex_ids.npy",
    "flame/generic_model.pkl",
    "flame/mediapipe_landmark_embedding.npz",
    "mediapipe/face_landmarker.task",
)

MODEL_LINKS: Dict[str, str] = {
    "smplx": "https://smpl-x.is.tue.mpg.de/",
    "flame": "https://flame.is.tue.mpg.de/",
    "mediapipe": "run: python scripts/download_models.py --face (fetches face_landmarker.task)",
    "flame_embedding": "https://github.com/Zielon/metrical-tracker/tree/master/flame/mediapipe",
}


def check_models(model_dir: Path) -> List[str]:
    """Relative paths in `REQUIRED_MODELS` that are not files under `model_dir`."""
    root = Path(model_dir)
    return [rel for rel in REQUIRED_MODELS if not (root / rel).is_file()]


def repo_model_dir() -> Path:
    """`<repo root>/models` (the repo root found from this file, as `server.CONFIGS_DIR` is)."""
    return Path(__file__).resolve().parents[3] / "models"


def lab_home() -> Path:
    """`$KINETIC_LAB_HOME`, else `~/.kinetic_lab`."""
    override = os.environ.get("KINETIC_LAB_HOME")
    return Path(override) if override else Path.home() / ".kinetic_lab"


@dataclass
class LabConfig:
    model_dir: Optional[str] = None

    @classmethod
    def path(cls) -> Path:
        return lab_home() / "config.json"

    @classmethod
    def load(cls) -> "LabConfig":
        try:
            data = json.loads(cls.path().read_text())
            value = data.get("model_dir")
            return cls(model_dir=str(value) if value else None)
        except (OSError, ValueError, AttributeError):
            return cls()

    def save(self) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model_dir": self.model_dir}))
