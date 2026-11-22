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

from ..body_model.face_correspondence import (
    LEFT_EYE,
    LEFT_EYEBROW,
    RIGHT_EYE,
    RIGHT_EYEBROW,
    RIGID,
)
from .base import AugmentationRegistry, ClipAugmentation
from .clip import ChainContext, ClipData
from .ranges import check_non_negative, check_positive, check_truncation, truncated_normal

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

# A score whose std over the fitted frames is below this carries no signal (MediaPipe emits a flat
# 0 for cheekSquint/noseSneer and some scores barely move), so the ridge cannot map it to motion:
# its offset would vanish or explode. Such scores are left out of the fit and never jittered.
MIN_SCORE_STD = 0.01

# Landmark region each group moves. Index sets are MediaPipe's face-mesh connection groups
# (mediapipe.solutions.face_mesh_connections: LIPS, FACE_OVAL, NOSE, irises); eyes, brows and the
# eye/brow rings are reused from body_model.face_correspondence. A group's jitter reaches only
# these landmarks, so brow/eye offsets cannot leak into the lips.
_LIPS = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
)  # fmt: skip
# Lower half of FACE_OVAL (chin and jaw line), below the mouth corners.
_JAW_LINE = (
    58, 132, 136, 148, 149, 150, 152, 172, 176, 288, 361, 365, 377, 378, 379, 397, 400,
)  # fmt: skip
_NOSE_WINGS = (
    1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327,
    344, 440,
)  # fmt: skip
_IRISES = tuple(range(468, 478))
GROUP_REGIONS: Dict[str, Tuple[int, ...]] = {
    "brows": tuple(sorted(RIGHT_EYEBROW + LEFT_EYEBROW)),
    "eyes": tuple(sorted(RIGHT_EYE + LEFT_EYE + _IRISES)),
    "cheeks_nose": tuple(sorted(_NOSE_WINGS)),
    "mouth_jaw": tuple(sorted(_LIPS + _JAW_LINE)),
}


def constant_scores(blendshapes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Boolean (51,) mask of the non-neutral scores whose std on the usable frames is too small.

    Usable frames are valid with finite scores. With none, nothing is flagged.
    """
    scores = np.asarray(blendshapes, dtype=np.float64)[:, 1:]
    use = np.asarray(valid, dtype=bool) & np.isfinite(scores).all(axis=1)
    if not use.any():
        return np.zeros(scores.shape[1], dtype=bool)
    return scores[use].std(axis=0) < MIN_SCORE_STD


def region_mask(group: str) -> np.ndarray:
    """Boolean (478,) mask of the landmarks `group` may move (rigid points excluded)."""
    mask = np.zeros(_NUM_POINTS, dtype=bool)
    mask[list(GROUP_REGIONS[group])] = True
    mask[list(RIGID)] = False
    return mask


def fit_expression_map(
    face_local: np.ndarray, blendshapes: np.ndarray, valid: np.ndarray, alpha: float = 1e-2
) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge map from the 51 non-neutral scores to face point offsets.

    Returns `(W, median_face)`: `W` is (51, 478*3) and maps centred scores to the flattened offset
    from the median face; rigid point columns are zero. Fitted on valid frames with finite scores.
    Scores with std below `MIN_SCORE_STD` are left out of the fit and get zero rows.
    """
    scores = np.asarray(blendshapes, dtype=np.float64)[:, 1:]
    points = np.asarray(face_local, dtype=np.float64)
    use = np.asarray(valid, dtype=bool) & np.isfinite(scores).all(axis=1)
    use &= np.isfinite(points.reshape(points.shape[0], -1)).all(axis=1)
    n_scores = scores.shape[1]
    if not use.any():
        return np.zeros((n_scores, _NUM_POINTS * 3)), np.zeros((_NUM_POINTS, 3))
    median = np.median(points[use], axis=0)
    live = ~constant_scores(blendshapes, valid)
    x = scores[use][:, live] - scores[use][:, live].mean(axis=0)
    y = (points[use] - median).reshape(int(use.sum()), -1)
    w = np.zeros((n_scores, _NUM_POINTS * 3))
    if live.any():
        w[live] = np.linalg.solve(x.T @ x + float(alpha) * np.eye(int(live.sum())), x.T @ y)
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

    def validate_ranges(self) -> None:
        for group in BLENDSHAPE_GROUPS:
            g = self.ranges[group]
            for k in ("sigma", "truncate"):
                check_non_negative(g[k], self.label(f"{group}.{k}"))
            check_truncation(g["sigma"], g["truncate"], self.label(f"{group}.truncate"))
        check_non_negative(self.ranges["frame_noise"], self.label("frame_noise"))
        check_positive(self.ranges["ridge_alpha"], self.label("ridge_alpha"))
        if check_non_negative(self.ranges["min_face_frames"], self.label("min_face_frames")) % 1:
            raise ValueError(f"{self.label('min_face_frames')} must be a whole number")

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
        dead = constant_scores(clip.blendshapes, self._valid_frames(clip))
        inactive = []
        for group, member in BLENDSHAPE_GROUPS.items():
            g = self.ranges[group]
            if not bool(g["enabled"]):
                continue
            idx = [i for i, n in enumerate(names[1:]) if member(n)]
            # Draw for every member (keeps the rng stream stable), then drop the dead scores.
            offsets[idx] = truncated_normal(rng, float(g["sigma"]), float(g["truncate"]), len(idx))
            if all(dead[i] for i in idx):
                inactive.append(group)
        offsets[dead] = 0.0
        draw: Dict[str, Any] = {"offsets": [float(v) for v in offsets], "noise": None}
        if inactive:
            draw["inactive"] = inactive
        std = float(self.ranges["frame_noise"])
        if std > 0:
            mask = self._enabled_mask(names) & ~dead
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
            noise = np.asarray(draw["noise"], dtype=np.float64)
            if noise.shape[0] != target.blendshapes.shape[0]:
                raise ValueError(
                    f"draw noise has {noise.shape[0]} frames, clip has {target.blendshapes.shape[0]}"
                )
            delta = delta + noise
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
            change = new[valid, 1:] - old[valid, 1:]
            names = tuple(target.blendshape_names)
            for group, member in BLENDSHAPE_GROUPS.items():
                members = np.array([member(n) for n in names[1:]], dtype=bool)
                shift = (change * members) @ w
                shift = shift.reshape(-1, _NUM_POINTS, 3) * region_mask(group)[None, :, None]
                face[valid] += shift
            changes["face_local"] = face
        return target.replace(**changes)

    def identity_draw(self) -> Dict[str, Any]:
        return {"offsets": [0.0] * 51, "noise": None}
