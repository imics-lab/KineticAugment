"""The Lab's clip store: cached round-trip fits and their video frames, one folder per clip."""

from __future__ import annotations

import pickle
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from kinetic_augment.lab.pane import PreparedClip, prepare_clip

DEFAULT_ROOT = Path("demo/round_trip/clips")
LRU_SIZE = 4


class ClipStore:
    """
    Clips under `root/<id>/`: `fit.pkl` (pickle of `(result, hands, face)`) and `frames/NNNN.jpg`.

    A clip exported earlier (`hands.npz` + `face.npz`, no `fit.pkl`) is refitted with `run_clip`
    on first `get` and `fit.pkl` is written. Prepared clips stay in memory (LRU of 4).
    """

    def __init__(self, root: Optional[Path] = None, models: Any = None):
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.models = models
        self._cache: "OrderedDict[str, PreparedClip]" = OrderedDict()

    def _dir(self, clip_id: str) -> Path:
        if not clip_id or "/" in clip_id or "\\" in clip_id or clip_id.startswith("."):
            raise KeyError(clip_id)
        return self.root / clip_id

    def list(self) -> List[Dict[str, Any]]:
        """Every clip folder with a fit or exported caches: id, frame count, source."""
        out: List[Dict[str, Any]] = []
        if not self.root.exists():
            return out
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            fit = (d / "fit.pkl").exists()
            if not (fit or (d / "hands.npz").exists()):
                continue
            frames = len(list((d / "frames").glob("*.jpg"))) if (d / "frames").is_dir() else 0
            out.append({"id": d.name, "frames": frames, "source": "fit" if fit else "exported"})
        return out

    def get(self, clip_id: str) -> PreparedClip:
        """The prepared clip; raises KeyError when the clip is unknown."""
        if clip_id in self._cache:
            self._cache.move_to_end(clip_id)
            return self._cache[clip_id]
        d = self._dir(clip_id)
        fit = d / "fit.pkl"
        if fit.exists():
            with open(fit, "rb") as handle:
                result, hands, face = pickle.load(handle)
        elif (d / "hands.npz").exists() and (d / "face.npz").exists():
            from kinetic_augment.body_model.face_observations import load_face_clip
            from kinetic_augment.body_model.face_pipeline import run_clip
            from kinetic_augment.body_model.hand_observations import load_hand_clip

            hands, face = load_hand_clip(d / "hands.npz"), load_face_clip(d / "face.npz")
            result = run_clip(hands, face, self.models)
            self._write_fit(d, result, hands, face)
        else:
            raise KeyError(clip_id)
        prep = prepare_clip(clip_id, result, hands, face, self.models)
        self._cache[clip_id] = prep
        while len(self._cache) > LRU_SIZE:
            self._cache.popitem(last=False)
        return prep

    def frame_path(self, clip_id: str, i: int) -> Path:
        """Path of frame `i` (0-based); raises KeyError when it does not exist."""
        path = self._dir(clip_id) / "frames" / f"{int(i):04d}.jpg"
        if not path.exists():
            raise KeyError(f"{clip_id}/{i}")
        return path

    @staticmethod
    def _write_fit(d: Path, result, hands, face) -> None:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "fit.pkl.tmp"
        with open(tmp, "wb") as handle:
            pickle.dump((result, hands, face), handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(d / "fit.pkl")

    def add(self, clip_id: str, result, hands, face, frames_dir: Path) -> None:
        """Store a fit and copy its `*.jpg` frames in."""
        d = self._dir(clip_id)
        self._write_fit(d, result, hands, face)
        out = d / "frames"
        out.mkdir(parents=True, exist_ok=True)
        for jpg in sorted(Path(frames_dir).glob("*.jpg")):
            shutil.copy2(jpg, out / jpg.name)
        self._cache.pop(clip_id, None)
