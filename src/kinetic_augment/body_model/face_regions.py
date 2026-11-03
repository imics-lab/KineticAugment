"""
The SMPL-X head and neck surface, labelled with ASL-LEX location codes (V1.3 design §5).

The surface is every SMPL-X vertex in the FLAME vertex map, except the eyeball meshes, and every
triangle whose three vertices are kept. Regions follow ASL-LEX 2.0's head locations, plus Neck and Ear, with finer
subregions underneath. They come from rules on the rest mesh -- the mapped MediaPipe landmarks,
vertex normals and SMPL-X skinning weights -- so no further licensed asset is needed. Thresholds
are provisional and are tuned by eye on `scripts/render_face_regions.py`'s image.

Tuning record (first render, thresholds as first written all passed the anchor tests):
- Neck was `dominant_joint == 12` only. The render showed the 117 lowest vertices, skinned mostly
  to spine3 (joint 9), falling through to CheekNose (front) and HeadTop (back). Neck is now every
  vertex whose dominant joint is not head (15), jaw (22) or an eye (23, 24).
- Mouth required front-facing normals, but the inner-lip and mouth-interior vertices are not, so
  112 vertices inside the outer-lip outline fell through to CheekNose. Mouth is now inside the
  outer-lip outline (xy) and ahead of the ears, with no normal test.
- The FLAME vertex map includes the eyeball meshes (1092 vertices, dominant joint 23 or 24).
  They sit inside the skull and their outward normals point into it, so the signed distance of
  a point inside the head behind the eye came out positive (+0.006 m at 2 cm, +0.036 m at
  5 cm). They are now dropped from the surface; the skin around the eyes is still labelled Eye by
  the eye-contour radius rule.

Rest frame: y up, x toward the subject's left, z forward.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import ConvexHull

from .assembly import pose_mesh
from .face_correspondence import (
    LEFT_EYE, LEFT_EYEBROW, LOWER_LIP_BOTTOM, MOUTH_CORNERS, NOSE, OUTER_LIP, RIGHT_EYE,
    RIGHT_EYEBROW, SUBNASALE, UPPER_LIP_TOP, load_flame_to_smplx,
)
from .head_converter import EAR_VERTICES, points_in_polygon

REGIONS = ("HeadTop", "Forehead", "Eye", "CheekNose", "UpperLip", "Mouth", "Chin", "UnderChin",
           "Neck", "Ear")
SUBREGIONS = ("Brow", "Nose", "Cheek")
NO_LABEL = -1

HEAD_JOINT, JAW_JOINT, EYE_JOINTS = 15, 22, (23, 24)
EYE_RADIUS = 0.012      # metres from the eye contour
EAR_RADIUS = 0.035      # metres from an ear vertex
FRONT_NZ = 0.3          # normal z above this: front-facing
DOWN_NY = -0.5          # normal y below this: facing down
TOP_NY = 0.7            # normal y above this: facing up (scalp, not forehead)
BROW_BAND = 0.012       # metres either side of the mean brow height
FOREHEAD_HEIGHT = 0.07  # metres above the brows that still count as forehead


@dataclass
class RegionAnchors:
    eye_contours: np.ndarray      # (32, 3)
    brows: np.ndarray             # (20, 3)
    outer_lip: np.ndarray         # (20, 3) ordered loop
    nose: np.ndarray              # (13, 3)
    subnasale: np.ndarray         # (3,)
    upper_lip_top: np.ndarray     # (3,)
    lower_lip_bottom: np.ndarray  # (3,)
    mouth_corners: np.ndarray     # (2, 3)
    ears: np.ndarray              # (2, 3)


def region_anchors(rest_vertices, corr) -> RegionAnchors:
    def p(group):
        return corr.points(rest_vertices, group)

    return RegionAnchors(
        eye_contours=p(RIGHT_EYE + LEFT_EYE), brows=p(RIGHT_EYEBROW + LEFT_EYEBROW),
        outer_lip=p(OUTER_LIP), nose=p(NOSE), subnasale=p((SUBNASALE,))[0],
        upper_lip_top=p((UPPER_LIP_TOP,))[0], lower_lip_bottom=p((LOWER_LIP_BOTTOM,))[0],
        mouth_corners=p(MOUTH_CORNERS), ears=np.asarray(rest_vertices)[list(EAR_VERTICES)],
    )


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _nearest(points, anchors):
    return np.min(np.linalg.norm(points[:, None] - anchors[None], axis=-1), axis=1)


def label_regions(
    points, normals, dominant_joint, a: RegionAnchors
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Region and subregion of each rest-frame point. Rules apply in order; the first match wins.
    """
    n = len(points)
    region = np.full(n, NO_LABEL, dtype=np.int8)
    sub = np.full(n, NO_LABEL, dtype=np.int8)
    x, y = points[:, 0], points[:, 1]
    front = normals[:, 2] > FRONT_NZ             # facing forward: lips
    ahead = points[:, 2] > a.ears[:, 2].mean()   # in front of the ears: the sides of the face too

    def assign(mask, name):
        mask = mask & (region == NO_LABEL)
        region[mask] = REGIONS.index(name)
        return mask

    assign(np.isin(dominant_joint, EYE_JOINTS)
           | (_nearest(points, a.eye_contours) < EYE_RADIUS), "Eye")
    assign(~np.isin(dominant_joint, (HEAD_JOINT, JAW_JOINT) + EYE_JOINTS), "Neck")
    assign(_nearest(points, a.ears) < EAR_RADIUS, "Ear")
    assign(ahead & points_in_polygon(points[:, :2], a.outer_lip[:, :2]), "Mouth")
    x_lo, x_hi = np.sort(a.mouth_corners[:, 0])
    assign(front & (x >= x_lo) & (x <= x_hi)
           & (y >= a.upper_lip_top[1]) & (y <= a.subnasale[1]), "UpperLip")
    jaw = dominant_joint == JAW_JOINT
    assign(jaw & (normals[:, 1] < DOWN_NY), "UnderChin")
    assign(jaw & ahead & (y < a.lower_lip_bottom[1]), "Chin")
    brow_y = a.brows[:, 1].mean()
    forehead = assign(ahead & (normals[:, 1] < TOP_NY)
                      & (y >= brow_y - BROW_BAND) & (y <= brow_y + FOREHEAD_HEIGHT), "Forehead")
    brow = (forehead & (np.abs(y - brow_y) <= BROW_BAND)
            & (x >= a.brows[:, 0].min()) & (x <= a.brows[:, 0].max()))
    sub[brow] = SUBREGIONS.index("Brow")
    cheek_nose = assign(ahead & (y < brow_y), "CheekNose")
    hull = a.nose[ConvexHull(a.nose[:, :2]).vertices, :2]
    in_nose = points_in_polygon(points[:, :2], hull)
    sub[cheek_nose & in_nose] = SUBREGIONS.index("Nose")
    sub[cheek_nose & ~in_nose] = SUBREGIONS.index("Cheek")
    assign(np.ones(n, dtype=bool), "HeadTop")
    return region, sub


@dataclass(frozen=True)
class HeadSurface:
    vertex_ids: np.ndarray  # (Vh,) SMPL-X vertex ids
    faces: np.ndarray       # (Fh, 3) indices into vertex_ids
    region: np.ndarray      # (Vh,) int8 index into REGIONS
    subregion: np.ndarray   # (Vh,) int8 index into SUBREGIONS, NO_LABEL for none


def build_head_surface(skeleton, corr, model_dir="models") -> HeadSurface:
    """The labelled head and neck surface for one body shape."""
    vertex_ids = np.unique(load_flame_to_smplx(model_dir))
    weights = skeleton.model.lbs_weights.detach().numpy()
    vertex_ids = vertex_ids[~np.isin(weights[vertex_ids].argmax(axis=1), EYE_JOINTS)]
    rest = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3)).vertices
    all_faces = np.asarray(skeleton.model.faces, dtype=np.int64)
    local = np.full(len(rest), -1, dtype=np.int64)
    local[vertex_ids] = np.arange(len(vertex_ids))
    keep = (local[all_faces] >= 0).all(axis=1)
    normals = vertex_normals(rest, all_faces)[vertex_ids]
    dominant = weights[vertex_ids].argmax(axis=1)
    region, sub = label_regions(rest[vertex_ids], normals, dominant, region_anchors(rest, corr))
    return HeadSurface(vertex_ids, local[all_faces[keep]], region, sub)
