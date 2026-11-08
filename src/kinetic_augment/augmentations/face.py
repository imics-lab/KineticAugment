"""
Week-4 face augmentation: expression jitter.

`ExpressionJitter` shifts the MediaPipe blendshape scores of the enabled groups by one constant
offset per score (optionally plus slow per-frame noise) and moves the face points with them. The
link from scores to points is a ridge regression fitted on the clip itself (`fit_expression_map`),
so the points follow the clip's own face. The rigid skull points never move.

The draw holds the 51 offsets (zero for disabled groups) and the optional noise as plain lists.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, freqz

from ..body_model.face_correspondence import RIGID
from .base import AugmentationRegistry, ClipAugmentation
from .clip import ChainContext, ClipData
from .ranges import truncated_normal

BLENDSHAPE_GROUPS: Dict[str, Callable[[str], bool]] = {
    "brows": lambda n: n.startswith("brow"),
    "eyes": lambda n: n.startswith("eye"),
    "cheeks_nose": lambda n: n.startswith("cheekSquint") or n.startswith("noseSneer"),
    "mouth_jaw": lambda n: n.startswith("jaw")
    or n.startswith("mouth")
    or n in ("cheekPuff", "tongueOut"),
}
_GROUP_KEYS = ("enabled", "sigma", "truncate")
_NUM_POINTS = 478


def fit_expression_map(
    face_local: np.ndarray, blendshapes: np.ndarray, valid: np.ndarray, alpha: float = 1e-2
) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge map from the 51 non-neutral scores to face point offsets.

    Returns `(W, median_face)`: `W` is (51, 478*3) and maps centred scores to the flattened offset
    from the median face; rigid point columns are zero. Fitted on valid frames with finite scores.
    """
    scores = np.asarray(blendshapes, dtype=np.float64)[:, 1:]
    points = np.asarray(face_local, dtype=np.float64)
    use = np.asarray(valid, dtype=bool) & np.isfinite(scores).all(axis=1)
    use &= np.isfinite(points.reshape(points.shape[0], -1)).all(axis=1)
    n_scores = scores.shape[1]
    if not use.any():
        return np.zeros((n_scores, _NUM_POINTS * 3)), np.zeros((_NUM_POINTS, 3))
    median = np.median(points[use], axis=0)
    x = scores[use] - scores[use].mean(axis=0)
    y = (points[use] - median).reshape(int(use.sum()), -1)
    w = np.linalg.solve(x.T @ x + float(alpha) * np.eye(n_scores), x.T @ y)
    for p in RIGID:
        w[:, 3 * p : 3 * p + 3] = 0.0
    return w, median


def _lowpass_noise(rng: np.random.Generator, frames: int, cols: int, fps: float) -> np.ndarray:
    """Unit-std Gaussian noise low-passed at 2 Hz (zero phase), shape (frames, cols)."""
    if fps <= 4.0:
        raise ValueError(f"frame_noise needs fps > 4 for the 2 Hz low-pass, got fps={fps}")
    b, a = butter(2, 2.0, fs=fps)
    pad = int(np.ceil(3.0 * fps))
    noise = rng.normal(size=(frames + 2 * pad, cols))
    _, h = freqz(b, a, worN=4096)
    gain = float(np.sqrt(np.mean(np.abs(h) ** 4)))
    return filtfilt(b, a, noise, axis=0)[pad : pad + frames] / gain


@AugmentationRegistry.register("face.expression_jitter")
class ExpressionJitter(ClipAugmentation):
    """Constant per-score offsets on the enabled blendshape groups, carried to the face points."""

    key = "expression_jitter"
    regions = ("face",)
    DEFAULT_RANGES = {
        "brows": {"enabled": True, "sigma": 0.0625, "truncate": 0.125},
        "eyes": {"enabled": True, "sigma": 0.0625, "truncate": 0.125},
        "cheeks_nose": {"enabled": True, "sigma": 0.0375, "truncate": 0.075},
        "mouth_jaw": {"enabled": False, "sigma": 0.0375, "truncate": 0.075},
        "frame_noise": 0.0,
        "min_face_frames": 30,
        "ridge_alpha": 0.01,
    }

    def __init__(self, ranges: Optional[Dict[str, Any]] = None, probability: float = 1.0):
        given = dict(ranges or {})
        for group in BLENDSHAPE_GROUPS:
            if group in given:
                value = given[group]
                if not isinstance(value, dict):
                    raise ValueError(f"{group} must be a mapping, got {value!r}")
                for k in value:
                    if k not in _GROUP_KEYS:
                        raise ValueError(f"Unknown key {k!r} in group {group!r}")
                given[group] = {**self.DEFAULT_RANGES[group], **value}
        super().__init__(ranges=given, probability=probability)
        for group in BLENDSHAPE_GROUPS:
            g = self.ranges[group]
            if float(g["sigma"]) < 0 or float(g["truncate"]) < 0:
                raise ValueError(f"{group}: sigma and truncate must be >= 0")
        for k in ("frame_noise", "ridge_alpha"):
            if float(self.ranges[k]) < 0:
                raise ValueError(f"{k} must be >= 0, got {self.ranges[k]!r}")
        if int(self.ranges["min_face_frames"]) < 0:
            raise ValueError("min_face_frames must be >= 0")

    def _enabled_mask(self, names: Tuple[str, ...]) -> np.ndarray:
        """Boolean (51,) mask of the non-neutral scores in enabled groups."""
        mask = np.zeros(len(names) - 1, dtype=bool)
        for group, member in BLENDSHAPE_GROUPS.items():
            if bool(self.ranges[group]["enabled"]):
                mask |= np.array([member(n) for n in names[1:]], dtype=bool)
        return mask

    @staticmethod
    def _valid_frames(clip: ClipData) -> np.ndarray:
        return np.asarray(clip.face_valid, dtype=bool) & np.isfinite(clip.blendshapes).all(axis=1)

    def sample(self, rng: np.random.Generator, clip: ClipData, ctx: Any) -> Dict[str, Any]:
        names = tuple(clip.blendshape_names)
        offsets = np.zeros(len(names) - 1)
        for group, member in BLENDSHAPE_GROUPS.items():
            g = self.ranges[group]
            if not bool(g["enabled"]):
                continue
            idx = [i for i, n in enumerate(names[1:]) if member(n)]
            offsets[idx] = truncated_normal(rng, float(g["sigma"]), float(g["truncate"]), len(idx))
        draw: Dict[str, Any] = {"offsets": [float(v) for v in offsets], "noise": None}
        std = float(self.ranges["frame_noise"])
        if std > 0:
            mask = self._enabled_mask(names)
            noise = np.zeros((clip.blendshapes.shape[0], len(offsets)))
            noise[:, mask] = _lowpass_noise(rng, noise.shape[0], int(mask.sum()), clip.fps) * std
            draw["noise"] = noise.tolist()
        n_valid = int(self._valid_frames(clip).sum())
        if n_valid == 0 or n_valid < int(self.ranges["min_face_frames"]):
            draw["fallback"] = "too_few_face_frames"
        return draw

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        valid = self._valid_frames(target)
        if not valid.any():
            draw.setdefault("fallback", "too_few_face_frames")
            return target
        delta = np.asarray(draw["offsets"], dtype=np.float64)[None, :]
        if draw.get("noise") is not None:
            delta = delta + np.asarray(draw["noise"], dtype=np.float64)
        old = np.asarray(target.blendshapes, dtype=np.float64)
        new = old.copy()
        new[valid, 1:] = np.clip(
            old[valid, 1:] + (delta if delta.shape[0] == 1 else delta[valid]), 0.0, 1.0
        )
        changes: Dict[str, Any] = {"blendshapes": new}
        if draw.get("fallback") != "too_few_face_frames":
            w, _ = fit_expression_map(
                target.face_local,
                old,
                target.face_valid,
                float(self.ranges["ridge_alpha"]),
            )
            face = np.array(target.face_local, dtype=np.float64, copy=True)
            shift = (new[valid, 1:] - old[valid, 1:]) @ w
            face[valid] += shift.reshape(-1, _NUM_POINTS, 3)
            changes["face_local"] = face
        elif not draw.get("fallback"):
            draw["fallback"] = "too_few_face_frames"
        return target.replace(**changes)

    def identity_draw(self) -> Dict[str, Any]:
        return {"offsets": [0.0] * 51, "noise": None}
