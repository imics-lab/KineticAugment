"""
The clip container that every week-4 augmentation edits.

A `ClipData` holds one clip as SMPL-X parameters (local axis-angle rotations, translation, body
shape and the rest joints the skinning uses) plus the observed face in the head joint's frame,
the blendshapes, the validity masks, the timing, the clip's weak camera and a pending scene
transform (mirror, view rotation, scale, shift) that `outputs.build_outputs` applies to every
point set at the end. Augmentations return edited copies; nothing here mutates its input.

Rotations are re-timed by spherical interpolation (`slerp_resample`), not by the linear
interpolation `tier1.resample` applies to axis-angle values.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from . import tier1
from .tier1 import Tier1Draw, WeakCamera

if TYPE_CHECKING:  # pragma: no cover - heavy imports for annotations only
    from ..body_model.face_observations import FaceClip
    from ..body_model.face_pipeline import ClipFaceResult
    from ..body_model.hand_observations import HandClip
    from ..body_model.skeleton import SMPLXSkeleton


@dataclass(frozen=True, eq=False)
class SceneTransform:
    """
    A whole-scene transform applied to the output points, in MediaPipe's world frame.

    Order: reflect x about `pivot` (if `mirror`), then rotate and scale about `pivot`
    (`tier1.about(p, pivot, rotation * scale)`), then add `shift`. Swapping the sides after a
    reflection is `build_outputs`' job, not this one's.
    """

    mirror: bool = False
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: float = 1.0
    shift: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def is_identity(self) -> bool:
        """True when `apply_points` leaves every point where it is (the pivot is irrelevant)."""
        return (
            not self.mirror
            and np.array_equal(self.rotation, np.eye(3))
            and self.scale == 1.0
            and not np.any(self.shift)
        )

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        """(..., 3) points through the transform; the identity returns an exact copy."""
        points = np.array(points, dtype=np.float64, copy=True)
        if self.is_identity():
            return points
        pivot = np.asarray(self.pivot, dtype=np.float64)
        if self.mirror:
            points = tier1.reflect_x(points, pivot[0])
        points = tier1.about(points, pivot, np.asarray(self.rotation) * self.scale)
        return points + np.asarray(self.shift, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class ClipData:
    """
    One clip as SMPL-X parameters plus the carried face, ready for per-clip augmentation.

    T is the frame count. Rotations are NaN where the body was not fitted (`body_ok` false);
    `face_local` is NaN where no face was attached (`face_valid` false).
    """

    clip_id: str
    rotations: np.ndarray  # (T, 55, 3) local axis-angle
    transl: np.ndarray  # (T, 3) metres
    betas: np.ndarray  # (300,)
    rest_joints: np.ndarray  # (55, 3) rest joints the skinning uses (editable for bone lengths)
    face_local: np.ndarray  # (T, 478, 3) observed face points in the head joint's frame
    face_valid: np.ndarray  # (T,) bool
    blendshapes: np.ndarray  # (T, 52)
    blendshape_names: Tuple[str, ...]
    body_ok: np.ndarray  # (T,) bool
    hand_valid: np.ndarray  # (T, 2) bool, index 0 = left
    times_s: np.ndarray  # (T,) seconds from the first frame
    fps: float
    camera: Optional[WeakCamera]
    scene: SceneTransform

    @property
    def frames(self) -> int:
        return len(self.rotations)

    def replace(self, **changes) -> "ClipData":
        """A copy with the given fields changed (arrays not named are shared, not copied)."""
        return dataclasses.replace(self, **changes)


@dataclass
class ChainContext:
    """What an augmentation chain shares between its steps for one clip."""

    skeleton: Any  # SMPLXSkeleton
    parents: np.ndarray  # (55,) kinematic parents
    pivot: np.ndarray  # (3,) scene pivot (the median shoulder centre)
    tier1_draw: Optional[Tier1Draw] = None
    seed: int = 0


def clip_from_result(
    clip_id: str,
    result: "ClipFaceResult",
    hands: "HandClip",
    face: "FaceClip",
    skeleton: "SMPLXSkeleton",
) -> ClipData:
    """A `ClipData` from one `run_clip` result, with no camera and the identity scene."""
    frames = np.asarray(result.frames)
    return ClipData(
        clip_id=str(clip_id),
        rotations=np.array(result.rotations, dtype=np.float64),
        transl=np.array(result.transl, dtype=np.float64),
        betas=np.array(result.betas, dtype=np.float64),
        rest_joints=np.array(skeleton.shaped(result.betas).rest_joints, dtype=np.float64),
        face_local=np.array(result.face_stream_local, dtype=np.float64),
        face_valid=np.array(result.face_attachment_valid, dtype=bool),
        blendshapes=np.array(face.blendshapes[frames], dtype=np.float64),
        blendshape_names=tuple(face.blendshape_names),
        body_ok=np.array(result.body_ok, dtype=bool),
        hand_valid=np.asarray(result.hand_present, dtype=bool)
        & np.asarray(result.geometry_valid, dtype=bool),
        times_s=(frames - frames[0]) / float(hands.fps),
        fps=float(hands.fps),
        camera=None,
        scene=SceneTransform(),
    )


# --------------------------------------------------------------------------- re-timing


def _neighbours(count: int, times: np.ndarray):
    """Lower and upper source frame and the weight of the upper one, per output time."""
    times = np.clip(np.asarray(times, dtype=np.float64), 0.0, count - 1)
    lo = np.floor(times).astype(int)
    hi = np.minimum(lo + 1, count - 1)
    return lo, hi, times - lo


def slerp_resample(rotations: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    (T, J, 3) axis-angle rotations at fractional frame `times` -> (T', J, 3).

    Per joint, spherical interpolation between the two neighbouring frames: the same rotation
    as a scipy `Slerp` over those two keyframes (`a * exp(w * log(a^-1 b))`), vectorised over
    joints. An integer time returns the input row exactly. A joint missing (non-finite) in one
    neighbour takes the finite one; missing in both stays NaN.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    count, joints = rotations.shape[:2]
    lo, hi, w = _neighbours(count, times)
    out = np.full((len(lo), joints, 3), np.nan)
    finite = np.isfinite(rotations).all(axis=-1)  # (T, J)
    for k, (a, b, weight) in enumerate(zip(lo, hi, w)):
        if weight == 0.0:
            out[k] = rotations[a]
            continue
        both = finite[a] & finite[b]
        only_a, only_b = finite[a] & ~finite[b], finite[b] & ~finite[a]
        out[k, only_a] = rotations[a, only_a]
        out[k, only_b] = rotations[b, only_b]
        if both.any():
            ra = Rotation.from_rotvec(rotations[a, both])
            rb = Rotation.from_rotvec(rotations[b, both])
            step = Rotation.from_rotvec(weight * (ra.inv() * rb).as_rotvec())
            out[k, both] = (ra * step).as_rotvec()
    return out


def resample_clip(clip: ClipData, times: np.ndarray) -> ClipData:
    """
    The clip at fractional source frame `times` (as from `tier1.duration_times`).

    Rotations by `slerp_resample`; translation, blendshapes (as whole vectors) and face points
    by `tier1.resample_points`; the boolean masks from the nearest frame; `times_s` by linear
    interpolation; the camera by `WeakCamera.resampled`.
    """
    times = np.asarray(times, dtype=np.float64)
    count = clip.frames
    nearest = np.clip(np.rint(times), 0, count - 1).astype(int)
    source = np.arange(count, dtype=np.float64)
    return clip.replace(
        rotations=slerp_resample(clip.rotations, times),
        transl=tier1.resample_points(clip.transl[:, None, :], times)[:, 0],
        face_local=tier1.resample_points(clip.face_local, times),
        blendshapes=tier1.resample_points(clip.blendshapes[:, None, :], times)[:, 0],
        face_valid=clip.face_valid[nearest].copy(),
        body_ok=clip.body_ok[nearest].copy(),
        hand_valid=clip.hand_valid[nearest].copy(),
        times_s=np.interp(times, source, clip.times_s),
        camera=None if clip.camera is None else clip.camera.resampled(times),
    )


def select_frames(clip: ClipData, keep: np.ndarray) -> ClipData:
    """The clip's frames picked by a (T,) boolean mask or an integer index array."""
    keep = np.asarray(keep)
    index = np.flatnonzero(keep) if keep.dtype == bool else keep.astype(int)
    camera = clip.camera
    if camera is not None:
        camera = WeakCamera(camera.scale, np.asarray(camera.offset)[index], camera.residual)
    return clip.replace(
        rotations=clip.rotations[index],
        transl=clip.transl[index],
        face_local=clip.face_local[index],
        face_valid=clip.face_valid[index],
        blendshapes=clip.blendshapes[index],
        body_ok=clip.body_ok[index],
        hand_valid=clip.hand_valid[index],
        times_s=clip.times_s[index],
        camera=camera,
    )
