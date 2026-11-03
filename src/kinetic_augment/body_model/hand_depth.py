"""
Wrist positions relative to the head, recovered from the image (V1.3 task 14).

MediaPipe's pose-world wrists can sit 15-25 cm from the face while the video shows fingers at
the chin (ledger O16). The image carries the missing depth cue: a hand's world landmarks are
metric and its image landmarks are in pixels, so their ratio is the hand's own pixels per metre,
and a pinhole camera turns pixels per metre into depth (`Z = f / scale`). The head's scale comes
from the head projection of task 13. With both depths and both image positions, the wrist can be
placed relative to the head in camera axes.

Assumptions: pinhole camera, principal point at the image centre, a nominal focal length from a
vertical field of view (MediaPipe's face-geometry default, 63 degrees), a hand and a head that
are each shallow compared with their distance from the camera (weak perspective per object), and
camera axes aligned with the pose-world axes (ledger O11, measured about 1 degree).

Reach. The body is re-fitted with fixed bone lengths, MediaPipe's shoulder and elbow directions
and the first pass's translation, so the SMPL-X wrist may not get to the target; `wrist_reached`
(residual within `REACH_TOLERANCE`) says whether it did. 3-D contact on a corrected hand that did not reach is not trustworthy: the arm
cannot reach the target, so the SMPL-X hand ends up somewhere else (V1.3 task 14, 01986 frames
14-21 show chin contact with 11-14 cm residual while the image has the hand 9-16 cm away).
A wrist can only be placed when the fit has the same side's shoulder and elbow; a hand without
them is left uncorrected.
"""

from dataclasses import dataclass

import numpy as np

from .alignment import similarity_transform

DEFAULT_VERTICAL_FOV_DEG = 63.0
DEPTH_RATIO_RANGE = (0.5, 1.5)  # plausible Z_hand / Z_head; outside it the correction is skipped
REACH_TOLERANCE = 0.03          # m; residual up to this: the SMPL-X wrist reached the target
SIDES = 2


def focal_px(height, fov_deg=DEFAULT_VERTICAL_FOV_DEG) -> float:
    """Focal length in pixels for a vertical field of view over an image `height` pixels tall."""
    return float((height / 2.0) / np.tan(np.radians(fov_deg) / 2.0))


def camera_point(u, v, depth, focal, cx, cy) -> np.ndarray:
    """(3,) camera-frame point (x right, y down, z away) of pixel (u, v) at `depth` metres."""
    return np.array([(u - cx) * depth / focal, (v - cy) * depth / focal, depth])


def hand_scale_px_per_m(hand_world, hand_px) -> float:
    """
    Pixels per metre of one hand: the scale of a 2-D similarity from its world landmarks' x, y
    (metres) to its image landmarks (pixels).

    Args:
        hand_world: (21, 3) world landmarks, metres.
        hand_px: (21, 2) the same landmarks in pixels.

    Returns:
        The scale, or NaN when either input has a non-finite value or the fit is degenerate.
    """
    world = np.asarray(hand_world, dtype=np.float64)[:, :2]
    pixels = np.asarray(hand_px, dtype=np.float64)[:, :2]
    if not (np.isfinite(world).all() and np.isfinite(pixels).all()):
        return float("nan")
    zeros = np.zeros((len(world), 1))
    try:
        scale, _, _ = similarity_transform(np.hstack([world, zeros]),
                                           np.hstack([pixels, zeros]), allow_scale=True)
    except ValueError:
        return float("nan")
    return scale if np.isfinite(scale) and scale > 0 else float("nan")


@dataclass
class WristTargets:
    target: np.ndarray       # (2, 3) world wrist targets; NaN where not corrected
    corrected: np.ndarray    # (2,) bool
    depth_ratio: np.ndarray  # (2,) Z_hand / Z_head; NaN where unavailable


def wrist_targets(head_px, head_world, head_scale, hand_world, hand_px, width, height,
                  fov_deg=DEFAULT_VERTICAL_FOV_DEG, arm_valid=None) -> WristTargets:
    """
    World wrist targets from the image: each wrist's offset from the head, in camera axes, added
    to the head's world position.

    Depth and position refer to matching points. The head scale (a similarity of the rigid
    points) is the depth of the rigid points' centroid, and the hand scale (a similarity of its
    21 landmarks) is the depth of the hand's centroid. So the head reference is that centroid,
    and the hand's centroid is back-projected at `Z_hand`; the wrist is then that point plus the
    wrist's offset from the hand centroid in the hand's own metric world landmarks. Using the nose
    tip or the wrist directly would apply a centroid depth to a point a few cm nearer.

    Args:
        head_px: (2,) or (3,) image position of the rigid-point centroid, pixels.
        head_world: (3,) world position of the same centroid on the first-pass posed mesh.
        head_scale: pixels per metre at the head (`ProjectedHead.scale`).
        hand_world: (2, 21, 3) world landmarks per side, metres; NaN where absent.
        hand_px: (2, 21, 2) image landmarks per side in pixels; NaN where absent.
        width, height: image size in pixels.
        arm_valid: (2,) bool, whether each side's shoulder and elbow are usable in the pose
            (`fit_body` cannot place a wrist without them); None means both are.

    A hand is left uncorrected (target NaN) when any input it needs is missing or its depth ratio
    `Z_hand / Z_head` falls outside `DEPTH_RATIO_RANGE`, or its `arm_valid` is False;
    the ratio is still reported when both depths exist.
    """
    out = WristTargets(np.full((SIDES, 3), np.nan), np.zeros(SIDES, bool), np.full(SIDES, np.nan))
    head_px = np.asarray(head_px, dtype=np.float64)[:2]
    head_world = np.asarray(head_world, dtype=np.float64)
    if not (np.isfinite(head_px).all() and np.isfinite(head_world).all()
            and np.isfinite(head_scale) and head_scale > 0):
        return out
    focal, cx, cy = focal_px(height, fov_deg), width / 2.0, height / 2.0
    z_head = focal / head_scale
    head_cam = camera_point(head_px[0], head_px[1], z_head, focal, cx, cy)
    for s in range(SIDES):
        scale = hand_scale_px_per_m(hand_world[s], hand_px[s])
        if not np.isfinite(scale):
            continue
        z_hand = focal / scale
        out.depth_ratio[s] = z_hand / z_head
        if arm_valid is not None and not arm_valid[s]:
            continue
        if not DEPTH_RATIO_RANGE[0] <= out.depth_ratio[s] <= DEPTH_RATIO_RANGE[1]:
            continue
        world = np.asarray(hand_world[s], dtype=np.float64)
        centre_px = np.asarray(hand_px[s], dtype=np.float64)[:, :2].mean(axis=0)
        centre_cam = camera_point(centre_px[0], centre_px[1], z_hand, focal, cx, cy)
        wrist_cam = centre_cam + (world[0] - world.mean(axis=0))
        out.target[s] = head_world + (wrist_cam - head_cam)
        out.corrected[s] = True
    return out
