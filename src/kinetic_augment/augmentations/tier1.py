"""
Tier 1 augmentations: the ones safe for every sign (Augmentation_Grounding.md §3).

Every parameter is drawn once per clip (V1 Problem Scope §11) by `sample_tier1`, then applied
to every frame:

    mirror       left/right reflection with the sides swapped, p = 0.5
    view         yaw about the vertical axis ±20°, plus pitch ±5°, about the shoulder centre
    duration     the clip's duration scaled by one factor, 0.8-1.25×, log-uniform; no cropping
    arm joints   shoulder, elbow and wrist on each side, each with p = 0.3, ±4°
    scale        the whole scene about the shoulder centre, 0.85-1.15 (camera distance)
    uncentre     the whole scene shifted by up to ±0.1 of the frame, keeping the hands in it

The arm-joint draw is applied to SMPL-X parameters by the caller; everything else here acts
on landmark and vertex arrays in MediaPipe's world frame (x right, y down, metres).

Uncentring needs to know where the frame is. `fit_weak_camera` fits a weak-perspective camera
per clip, from the reconstructed arm landmarks to MediaPipe's image landmarks: one scale per
axis for the clip and one offset per frame. `uncentre_range` then limits the shift so that no
hand point inside the frame is moved out of it. Points already outside it are ignored: many
clips start and end with the hands below the frame.
"""

import dataclasses
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclasses.dataclass(frozen=True)
class Tier1Ranges:
    """The sampling ranges, from the evidence in Augmentation_Grounding.md §2-§3."""

    mirror_p: float = 0.5
    yaw_deg: float = 20.0  # ±, uniform (Spoken2Sign)
    pitch_deg: float = 5.0  # ±, uniform ("a few degrees of pitch")
    duration: Tuple[float, float] = (0.8, 1.25)  # log-uniform, so 0.8 and 1.25 are symmetric
    arm_joint_p: float = 0.3  # per joint (SPOTER)
    arm_joint_deg: float = 4.0  # ±, uniform (SPOTER)
    scale: Tuple[float, float] = (0.85, 1.15)
    uncentre: float = 0.1  # ±, fraction of the frame on each axis (GISLR)


TIER1 = Tier1Ranges()
ARM_JOINTS = ("shoulder", "elbow", "wrist")
SIDES = ("left", "right")


@dataclasses.dataclass(frozen=True)
class ArmJointDraw:
    """
    One arm joint's rotation, constant over the clip.

    `axis` is a unit vector in the parent joint's frame. The elbow ignores it and flexes
    within its current bending plane instead (positive = more bent), as the elbow is a hinge.
    """

    side: str
    joint: str
    angle_deg: float
    axis: Tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Tier1Draw:
    """One clip's Tier 1 parameters."""

    mirror: bool
    yaw_deg: float
    pitch_deg: float
    duration: float
    arm_joints: Tuple[ArmJointDraw, ...]
    scale: float
    uncentre_u: Tuple[float, float]  # uniform in [0, 1); placed in the allowed range per clip

    def as_dict(self):
        return dataclasses.asdict(self)


def sample_tier1(rng: np.random.Generator, ranges: Tier1Ranges = TIER1) -> Tier1Draw:
    """Draw one clip's parameters. Every value is drawn, in a fixed order, whether used or not."""
    mirror = bool(rng.random() < ranges.mirror_p)
    yaw = float(rng.uniform(-ranges.yaw_deg, ranges.yaw_deg))
    pitch = float(rng.uniform(-ranges.pitch_deg, ranges.pitch_deg))
    lo, hi = np.log(ranges.duration)
    duration = float(np.exp(rng.uniform(lo, hi)))
    joints = []
    for side in SIDES:
        for joint in ARM_JOINTS:
            chosen = rng.random() < ranges.arm_joint_p
            angle = float(rng.uniform(-ranges.arm_joint_deg, ranges.arm_joint_deg))
            axis = rng.normal(size=3)
            axis = tuple(float(a) for a in axis / np.linalg.norm(axis))
            if chosen:
                joints.append(ArmJointDraw(side, joint, angle, axis))
    scale = float(rng.uniform(*ranges.scale))
    uncentre_u = (float(rng.random()), float(rng.random()))
    return Tier1Draw(mirror, yaw, pitch, duration, tuple(joints), scale, uncentre_u)


# --------------------------------------------------------------------------- spatial

# MediaPipe pose landmarks swapped by a left/right reflection.
_POSE_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]
POSE_MIRROR = np.arange(33)
for _a, _b in _POSE_PAIRS:
    POSE_MIRROR[_a], POSE_MIRROR[_b] = _b, _a


def view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Yaw about the vertical (y) axis, then pitch about the fixed horizontal (x) axis."""
    return Rotation.from_euler("yx", [yaw_deg, pitch_deg], degrees=True).as_matrix()


def about(points: np.ndarray, pivot: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a linear map (rotation, scale) about `pivot` to (..., 3) points."""
    return (points - pivot) @ np.asarray(matrix).T + pivot


def reflect_x(points: np.ndarray, centre_x: float) -> np.ndarray:
    """Reflect (..., 3) points in the vertical plane x = centre_x. Sides are not swapped here."""
    out = np.array(points, dtype=np.float64, copy=True)
    out[..., 0] = 2 * centre_x - out[..., 0]
    return out


# --------------------------------------------------------------------------- temporal


def duration_times(count: int, factor: float) -> np.ndarray:
    """
    Source times for a clip whose duration is scaled by `factor` (> 1 is slower).

    The first and last frames are kept, so nothing is cropped; factor 1 gives 0, 1, ..., T-1.
    """
    new_count = max(int(round((count - 1) * factor)) + 1, 2) if count > 1 else count
    if new_count == count:
        return np.arange(count, dtype=np.float64)
    return np.linspace(0.0, count - 1, new_count)


def resample(frames: Sequence, times: np.ndarray) -> list:
    """
    Linear interpolation of a frame sequence at fractional `times`.

    A frame is missing if it is None or not all finite. Between a missing frame and a present
    one the present one is used; between two missing frames, the nearer one.
    """
    out = []
    for tau in times:
        lo = int(np.floor(tau))
        hi = min(lo + 1, len(frames) - 1)
        w = tau - lo
        a, b = frames[lo], frames[hi]
        a_ok = a is not None and np.isfinite(a).all()
        b_ok = b is not None and np.isfinite(b).all()
        if w == 0.0:
            out.append(a)
        elif a_ok and b_ok:
            out.append((1 - w) * a + w * b)
        elif a_ok or b_ok:
            out.append(a if a_ok else b)
        else:
            out.append(a if w < 0.5 else b)
    return out


def resample_points(seq: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Per-point resampling of (T, N, 3), so a point missing in one frame stays local."""
    out = np.full((len(times),) + seq.shape[1:], np.nan)
    for j in range(seq.shape[1]):
        column = resample(list(seq[:, j]), times)
        out[:, j] = np.stack([np.full(seq.shape[2:], np.nan) if p is None else p for p in column])
    return out


# --------------------------------------------------------------------------- camera


@dataclasses.dataclass(frozen=True)
class WeakCamera:
    """
    image = scale * world[..., :2] + offset[t], in MediaPipe's normalised image units.

    One scale per axis for the clip (its sign carries any flip between the frames), one offset
    per frame because MediaPipe's world landmarks are re-centred on the hips every frame.
    """

    scale: Tuple[float, float]
    offset: np.ndarray  # (T, 2)
    residual: float  # median distance of fit points from the image, normalised

    def project(self, points: np.ndarray) -> np.ndarray:
        """(T, ..., 3) world points to (T, ..., 2) normalised image points."""
        points = np.asarray(points, dtype=np.float64)
        shape = (len(points),) + (1,) * (points.ndim - 2) + (2,)
        return np.asarray(self.scale) * points[..., :2] + self.offset.reshape(shape)

    def resampled(self, times: np.ndarray) -> "WeakCamera":
        """The camera re-timed with a clip (for the duration augmentation)."""
        frames = np.arange(len(self.offset))
        offset = np.stack([np.interp(times, frames, self.offset[:, a]) for a in range(2)], -1)
        return WeakCamera(self.scale, offset, self.residual)

    def world_shift(self, image_shift: Sequence[float]) -> np.ndarray:
        """The world translation (z = 0) that moves the projection by `image_shift`."""
        return np.array([image_shift[0] / self.scale[0], image_shift[1] / self.scale[1], 0.0])


def fit_weak_camera(
    world: np.ndarray, image: np.ndarray, pixel_size: Optional[Tuple[int, int]] = None
) -> WeakCamera:
    """
    Least-squares weak-perspective camera from (T, N, 3) world to (T, N, 2) image points.

    Without `pixel_size` each axis gets its own scale. With the frame's (width, height) in
    pixels, one scale serves both axes, as it should for square pixels; that holds up better
    when the camera is used far from the points it was fitted on.

    Points not finite in both are ignored. A frame with fewer than two usable points gets the
    offset of the nearest frame that has them.
    """
    world = np.asarray(world, dtype=np.float64)[..., :2]
    image = np.asarray(image, dtype=np.float64)[..., :2]
    size = np.ones(2) if pixel_size is None else np.asarray(pixel_size, dtype=np.float64)
    image_px = image * size
    ok = np.isfinite(world).all(-1) & np.isfinite(image).all(-1)
    usable = ok.sum(axis=1) >= 2
    if not usable.any():
        raise ValueError("no frame has two points visible in both world and image")
    num, den = np.zeros(2), np.zeros(2)
    for t in np.flatnonzero(usable):
        w, i = world[t, ok[t]], image_px[t, ok[t]]
        w, i = w - w.mean(axis=0), i - i.mean(axis=0)
        num += (w * i).sum(axis=0)
        den += (w * w).sum(axis=0)
    scale_px = num / den if pixel_size is None else np.full(2, num.sum() / den.sum())
    offset = np.full((len(world), 2), np.nan)
    for t in np.flatnonzero(usable):
        offset[t] = (image_px[t, ok[t]] - scale_px * world[t, ok[t]]).mean(axis=0) / size
    have = np.flatnonzero(usable)
    for t in np.flatnonzero(~usable):
        offset[t] = offset[have[np.argmin(np.abs(have - t))]]
    scale = scale_px / size
    projected = scale * world + offset[:, None]
    residual = float(np.median(np.linalg.norm(projected - image, axis=-1)[ok]))
    return WeakCamera((float(scale[0]), float(scale[1])), offset, residual)


def in_frame(uv: np.ndarray) -> np.ndarray:
    """Boolean mask of (..., 2) normalised image points inside the frame. NaN is outside."""
    uv = np.asarray(uv)
    return ((uv >= 0.0) & (uv <= 1.0)).all(axis=-1)


def uncentre_range(uv: np.ndarray, limit: float) -> np.ndarray:
    """
    Per axis, the shift interval [lo, hi] within ±limit that keeps every in-frame point in.

    Returns (2, 2): rows are x and y. Always contains 0.
    """
    uv = np.asarray(uv).reshape(-1, 2)
    inside = uv[in_frame(uv)]
    bounds = np.array([[-limit, limit], [-limit, limit]], dtype=np.float64)
    if len(inside):
        bounds[:, 0] = np.maximum(bounds[:, 0], -inside.min(axis=0))
        bounds[:, 1] = np.minimum(bounds[:, 1], 1.0 - inside.max(axis=0))
    return bounds


def uncentre_shift(uv: np.ndarray, u: Sequence[float], limit: float) -> np.ndarray:
    """The image shift for draw `u` in [0, 1)^2: uniform over the allowed range."""
    bounds = uncentre_range(uv, limit)
    return bounds[:, 0] + np.asarray(u) * (bounds[:, 1] - bounds[:, 0])


def frame_loss(uv_before: np.ndarray, uv_after: np.ndarray) -> Optional[float]:
    """Fraction of points inside the frame before that are outside it after; None if none were."""
    before, after = in_frame(uv_before), in_frame(uv_after)
    if not before.any():
        return None
    return float((before & ~after).sum() / before.sum())


# --------------------------------------------------------------------------- arm joints

# SMPL-X joint indices of the arm joints the draw can rotate, and the torso joint used when an
# arm is straight. The first 22 joints are the body; every arm joint's chain lies inside them.
SMPLX_ARM = {
    "left": {"shoulder": 16, "elbow": 18, "wrist": 20},
    "right": {"shoulder": 17, "elbow": 19, "wrist": 21},
}
SMPLX_SPINE3 = 9
NUM_BODY_JOINTS = 22
MIN_BEND, MAX_BEND = 5.0, 150.0  # degrees between upper arm and forearm; kept anatomical


def _world(rotations, rest_joints, parents):
    """World rotations and positions of the body joints (rest pose at the origin)."""
    local = Rotation.from_rotvec(rotations[:NUM_BODY_JOINTS]).as_matrix()
    rest = np.asarray(rest_joints, dtype=np.float64)[:NUM_BODY_JOINTS]
    rot, pos = np.zeros_like(local), np.zeros_like(rest)
    for j in range(NUM_BODY_JOINTS):
        p = parents[j]
        if p < 0:
            rot[j], pos[j] = local[j], rest[j]
        else:
            rot[j] = rot[p] @ local[j]
            pos[j] = rot[p] @ (rest[j] - rest[p]) + pos[p]
    return rot, pos


def _angle_deg(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def perturb_arm_joints(
    rotations: np.ndarray, rest_joints: np.ndarray, parents, joints: Sequence[ArmJointDraw]
) -> np.ndarray:
    """
    One frame's SMPL-X local rotations (J, 3), J >= 22, with the arm-joint draw applied.

    A shoulder or wrist turns about its drawn axis, fixed in the parent joint's frame, so it sits
    the same few degrees differently relative to its parent in every frame. An elbow bends within
    its current bending plane, kept between 5° and 150°. Joints below the wrist (the hand) keep
    their local rotations, so the hand turns with the wrist. `rest_joints` and `parents` are the
    shaped body's; only the first 22 are read.
    """
    out = np.array(rotations, dtype=np.float64, copy=True)
    parents = np.asarray(parents, dtype=int)
    for j in joints:
        index = SMPLX_ARM[j.side][j.joint]
        if j.joint != "elbow":
            delta = Rotation.from_rotvec(np.asarray(j.axis) * np.radians(j.angle_deg))
            out[index] = (delta * Rotation.from_rotvec(out[index])).as_rotvec()
            continue
        arm = SMPLX_ARM[j.side]
        rot, pos = _world(out, rest_joints, parents)
        upper = pos[arm["elbow"]] - pos[arm["shoulder"]]
        fore = pos[arm["wrist"]] - pos[arm["elbow"]]
        axis = _unit(np.cross(upper, fore))
        if axis is None:  # straight arm: bend toward the body's front
            axis = _unit(np.cross(upper, rot[SMPLX_SPINE3] @ np.array([0.0, 0.0, 1.0])))
        if axis is None:
            continue
        bend = _angle_deg(upper, fore)
        step = np.clip(bend + j.angle_deg, MIN_BEND, MAX_BEND) - bend
        delta = Rotation.from_rotvec(axis * np.radians(step)).as_matrix()
        parent = rot[arm["shoulder"]]
        local = Rotation.from_rotvec(out[index]).as_matrix()
        out[index] = Rotation.from_matrix(parent.T @ delta @ parent @ local).as_rotvec()
    return out
