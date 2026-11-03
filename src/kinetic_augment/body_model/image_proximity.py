"""
Hand-to-face proximity measured in the image (V1.3 task 13).

MediaPipe's pose-world 3-D can put a hand 15-25 cm from the face while the video shows the
fingers on the chin. The hand image landmarks and the FaceLandmarker mesh come from the same
frame, so in the image they are consistent. This module projects the fitted, region-labelled
SMPL-X head into the image and measures each hand point against it in 2-D.

The projection is a weighted similarity from the posed head's rigid points (metres) to the
observed ones (pixels), so it is weak perspective about the head: `scale` is pixels per metre at
the head's depth, and 2-D pixel gaps convert to metres only at that depth.

Every measure here is 2-D: an overlap does not prove touching, because a hand can be held in
front of the face (or behind it) and still cover the same pixels. Treat it as the image-space
"where is the hand relative to the face" cue, not as contact.

The Neck surface is left out on purpose. It runs some 10 cm below the chin to the shoulder base,
and a 2-D test accepts any depth, so a hand in front of the throat or upper chest (ordinary
signing space) would read as touching the neck. Overlap, distance, region and near therefore
describe the face and head proper; a hand over the neck gets its true distance to the chin,
under-chin and so on. Throat and neck locations are left to the 3-D `region` measure.
"""

from dataclasses import dataclass

import numpy as np

from .alignment import apply_similarity, similarity_transform
from .face_regions import NO_LABEL, REGIONS, HeadSurface
from .hand_face_geometry import CONTACT_DISTANCE, HAND_POINTS, closest_points_on_triangles

# MediaPipe hand landmarks in HAND_POINTS order (palm is the wrist/middle-knuckle midpoint).
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
MIDDLE_KNUCKLE = 9
NECK = REGIONS.index("Neck")
_AREA_EPS = 1e-9  # pixels^2: edge-on triangles carry no 2-D area


@dataclass
class ProjectedHead:
    points: np.ndarray    # (Vh, 3) head-surface vertices in the pixel frame
    visible: np.ndarray   # (Fh,) bool, triangles facing the camera
    scale: float          # pixels per metre at the head's depth
    rotation_deg: float   # angle of the world->pixel similarity rotation (O11 check)


def project_head(head_vertices, faces, rigid_world, rigid_px, rigid_weights) -> ProjectedHead:
    """
    Project the posed head surface into the pixel frame (x right, y down, z away).

    Args:
        head_vertices: (Vh, 3) posed head-surface vertices in world metres.
        faces: (Fh, 3) triangles over `head_vertices`, wound outward.
        rigid_world: (K, 3) posed SMPL-X rigid points in world metres.
        rigid_px: (K, 3) the same points observed in pixels (`FaceClip.pixels()`).
        rigid_weights: (K,) fit weights (`HeadFit.rigid_weights`); zero drops a point.

    The rotation angle measures how far the world axes are from the pixel axes; world and pixel
    axes are assumed aligned (ledger O11), so a large angle says the assumption fails.

    Raises:
        ValueError: fewer than three points carry weight.
    """
    scale, rotation, translation = similarity_transform(
        rigid_world, rigid_px, allow_scale=True, weights=rigid_weights)
    points = apply_similarity(head_vertices, scale, rotation, translation)
    tri = points[np.asarray(faces)]
    normal_z = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / 2.0  # sin(angle) * unit axis
    angle = np.arctan2(np.linalg.norm(axis), (np.trace(rotation) - 1.0) / 2.0)
    return ProjectedHead(points, normal_z < 0, float(scale), float(np.degrees(angle)))


def image_hand_points(hand_image, width, height) -> np.ndarray:
    """
    (2, 7, 2) hand points in pixels, in `HAND_POINTS` order, from one frame's normalised image
    landmarks `(2, 21, >=2)`. NaN for an absent hand.
    """
    image = np.asarray(hand_image, dtype=np.float64)[..., :2] * np.array([width, height], float)
    tips = [WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    out = np.empty((2, len(HAND_POINTS), 2))
    out[:, :6] = image[:, tips]
    out[:, 6] = (image[:, WRIST] + image[:, MIDDLE_KNUCKLE]) / 2.0
    return out


@dataclass
class ImageProximity:
    distance: np.ndarray   # (2, 7) metres at head depth; 0 when overlapping; NaN if hand absent
    overlap: np.ndarray    # (2, 7) bool
    region: np.ndarray     # (2, 7) int8, NO_LABEL if absent
    subregion: np.ndarray  # (2, 7) int8
    near: np.ndarray       # (2, 7) bool, distance < CONTACT_DISTANCE


def _inside(points, tri):
    """(P, F) bool: 2-D points inside 2-D triangles (any winding; edge-on triangles never)."""
    a, b, c = tri[None, :, 0], tri[None, :, 1], tri[None, :, 2]
    p = points[:, None]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    area = cross(b - a, c - a)
    s0, s1, s2 = cross(b - a, p - a), cross(c - b, p - b), cross(a - c, p - c)
    same = np.where(area < 0, -1.0, 1.0)
    inside = (s0 * same >= 0) & (s1 * same >= 0) & (s2 * same >= 0)
    return inside & (np.abs(area) > _AREA_EPS)


def image_proximity(hand_px, projected: ProjectedHead, surface: HeadSurface) -> ImageProximity:
    """
    Measure each hand point against the camera-facing head triangles, in 2-D.

    `distance` is 0 for a point inside a visible triangle, otherwise the pixel distance to the
    nearest visible triangle divided by `projected.scale` (metres at the head's depth). The
    region of an overlapping point is that of the vertex nearest it in the front-most (smallest
    z) containing triangle; of a point outside, the vertex of the nearest triangle nearest the
    closest point. Triangles whose three vertices are all Neck are excluded (see the module
    docstring). With no usable triangle nothing is measured. This is a 2-D measure: overlap does
    not prove touching, since a hand can be in front of the face.
    """
    shape = (2, len(HAND_POINTS))
    out = ImageProximity(np.full(shape, np.nan), np.zeros(shape, bool),
                         np.full(shape, NO_LABEL, np.int8), np.full(shape, NO_LABEL, np.int8),
                         np.zeros(shape, bool))
    points = np.asarray(hand_px, dtype=np.float64).reshape(-1, 2)
    present = np.isfinite(points).all(axis=1)
    all_faces = np.asarray(surface.faces)
    neck = (np.asarray(surface.region)[all_faces] == NECK).all(axis=1)
    faces = all_faces[np.asarray(projected.visible) & ~neck]
    if not present.any() or not len(faces):
        return out
    pixels = np.asarray(projected.points, dtype=np.float64)
    tri = pixels[faces]                                  # (F, 3, 3)
    flat = tri[..., :2]
    query = points[present]
    inside = _inside(query, flat)                        # (P, F)
    depth = tri[:, :, 2].mean(axis=1)                    # (F,)
    overlap = inside.any(axis=1)

    closest, nearest, gap = closest_points_on_triangles(
        np.column_stack([query, np.zeros(len(query))]),
        np.concatenate([flat, np.zeros((len(flat), 3, 1))], axis=2))
    front = np.argmin(np.where(inside, depth[None], np.inf), axis=1)
    chosen = np.where(overlap, front, nearest)
    anchor = np.where(overlap[:, None], query, closest[:, :2])
    corners = faces[chosen]                              # (P, 3) indices into the head
    spread = np.linalg.norm(pixels[corners][..., :2] - anchor[:, None], axis=-1)
    vertex = corners[np.arange(len(corners)), spread.argmin(axis=1)]
    metres = np.where(overlap, 0.0, gap / projected.scale)

    rows = np.flatnonzero(present)
    distance, flat_overlap = out.distance.reshape(-1), out.overlap.reshape(-1)
    region, subregion = out.region.reshape(-1), out.subregion.reshape(-1)
    distance[rows], flat_overlap[rows] = metres, overlap
    region[rows], subregion[rows] = surface.region[vertex], surface.subregion[vertex]
    out.near = np.isfinite(out.distance) & (out.distance < CONTACT_DISTANCE)
    return out
