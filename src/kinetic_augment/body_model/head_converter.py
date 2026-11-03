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
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from .alignment import apply_similarity, similarity_transform
from .assembly import HEAD, JAW, NECK, global_rotations, pose_mesh
from .face_correspondence import JAW_PAIR, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, RIGID
from .kinematics import forward_kinematics
from .swing_ik import swing_rotation

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


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Even-odd ray-casting point-in-polygon test for any simple polygon (convex or not).

    Args:
        points: (N, 2) query points.
        polygon: (K, 2) polygon vertices in order, not repeating the first vertex.

    Returns:
        (N,) bool, True for points inside; points exactly on the boundary may go either way.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64)
    x, y = points[:, 0:1], points[:, 1:2]
    x0, y0 = polygon[:, 0][None, :], polygon[:, 1][None, :]
    x1, y1 = np.roll(polygon[:, 0], -1)[None, :], np.roll(polygon[:, 1], -1)[None, :]
    straddles = (y0 > y) != (y1 > y)  # edge crosses the horizontal line through the point
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (straddles & (x < x_cross)).sum(axis=1) % 2 == 1


def occluded(points: np.ndarray, outlines: List[np.ndarray], pad: float) -> np.ndarray:
    """(N,) True where a 2-D point is inside an outline or within `pad` pixels of one."""
    points = np.asarray(points, dtype=np.float64)
    hidden = np.zeros(len(points), dtype=bool)
    for outline in outlines:
        inside = points_in_polygon(points, outline)
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
