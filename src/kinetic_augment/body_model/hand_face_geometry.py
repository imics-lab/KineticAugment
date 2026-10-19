"""
Signed distances from hand points to the head and neck surface, with location labels
(V1.3 design §5).

Per hand: the wrist joint, the five fingertip vertices, and the palm centre (midpoint of the
wrist and the middle-finger knuckle). Each point's distance is to the nearest surface triangle,
signed by that triangle's outward normal (negative = inside the head); near concave regions such
as the ear the sign can be wrong for deep penetrations, and a point nearest an open edge of
the surface (the neck rim and the mouth opening) is positive (see `signed_distances`). Contact
is |d| < 2 cm (self-contact work's definition). The region is that of the nearest triangle
vertex to the closest point.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .assembly import WRIST
from .face_regions import NO_LABEL

HAND_POINTS = ("wrist", "thumb", "index", "middle", "ring", "pinky", "palm")
SIDES = ("left", "right")
CONTACT_DISTANCE = 0.02
MIDDLE_KNUCKLE = {"left": 28, "right": 43}


def _tip_vertices():
    from smplx.vertex_ids import vertex_ids
    ids = vertex_ids["smplx"]
    return {side: [ids[f"{side[0]}{finger}"] for finger in
                   ("thumb", "index", "middle", "ring", "pinky")] for side in SIDES}


TIP_VERTICES = _tip_vertices()


def hand_points(mesh) -> np.ndarray:
    """(2, 7, 3) hand points, in HAND_POINTS order, for the left and right hands."""
    out = np.empty((2, len(HAND_POINTS), 3))
    for s, side in enumerate(SIDES):
        wrist = mesh.joints[WRIST[side]]
        out[s, 0] = wrist
        out[s, 1:6] = mesh.vertices[TIP_VERTICES[side]]
        out[s, 6] = (wrist + mesh.joints[MIDDLE_KNUCKLE[side]]) / 2.0
    return out


def _dot(u, v):
    return np.einsum("...k,...k->...", u, v)


def closest_points_on_triangles(points, triangles):
    """
    Exact closest point on any of the triangles to each point (Ericson, Real-Time Collision
    Detection §5.1.5, vectorised).

    Returns:
        (closest (P, 3), index of the nearest triangle (P,), distance (P,)).
    """
    p = np.asarray(points, dtype=np.float64)[:, None, :]
    tri = np.asarray(triangles, dtype=np.float64)[None]
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def put(mask, value):
        return np.where(mask[..., None], value, result)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        result = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]  # interior
        # Ericson tests A, B, AB, C, AC, BC in that order; applying them in reverse lets the
        # earlier test win where regions meet.
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        result = put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + t[..., None] * (c - b))
        t = d2 / (d2 - d6)
        result = put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + t[..., None] * ac)
        result = put((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, result.shape))
        t = d1 / (d1 - d3)
        result = put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + t[..., None] * ab)
        result = put((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, result.shape))
        result = put((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, result.shape))
    distance = np.linalg.norm(result - p, axis=-1)
    nearest = np.nanargmin(distance, axis=1)
    rows = np.arange(len(nearest))
    return result[rows, nearest], nearest, distance[rows, nearest]


def _boundary_edges(faces) -> np.ndarray:
    """(E, 2) vertex pairs of the edges used by exactly one triangle."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _on_segments(points, a, b, tolerance=1e-9) -> np.ndarray:
    """(P,) True where a point lies within `tolerance` of any segment a[i]-b[i]."""
    if not len(a):
        return np.zeros(len(points), dtype=bool)
    ab = b - a
    t = np.clip(_dot(points[:, None] - a[None], ab[None]) / np.maximum(_dot(ab, ab), 1e-24)[None],
                0.0, 1.0)
    nearest = a[None] + t[..., None] * ab[None]
    return (np.linalg.norm(points[:, None] - nearest, axis=-1) < tolerance).any(axis=1)


def signed_distances(points, vertices, faces):
    """
    Distance to the mesh, negative behind the nearest triangle's outward normal, except that a
    point whose closest point lies on an open (boundary) edge of the mesh is positive.

    The head surface is open at the bottom of the neck and at the mouth opening. Beyond such
    an edge the nearest triangle's normal says nothing about inside or outside (a hand hanging
    at the side would read as deep inside the head), so a point nearest an open edge is outside
    the surface's coverage and counts as outside.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    tri = vertices[np.asarray(faces)]
    closest, face, distance = closest_points_on_triangles(points, tri)
    normal = np.cross(tri[face, 1] - tri[face, 0], tri[face, 2] - tri[face, 0])
    inside = _dot(points - closest, normal) < 0
    rim = _boundary_edges(faces)
    inside &= ~_on_segments(closest, vertices[rim[:, 0]], vertices[rim[:, 1]])
    return np.where(inside, -1.0, 1.0) * distance, closest, face


@dataclass
class HandFaceGeometry:
    distance: np.ndarray   # (2, 7) signed metres; NaN for an unposed hand
    region: np.ndarray     # (2, 7) int8 index into REGIONS; NO_LABEL for an unposed hand
    subregion: np.ndarray  # (2, 7) int8 index into SUBREGIONS; NO_LABEL for none
    contact: np.ndarray    # (2, 7) bool, |distance| < CONTACT_DISTANCE


def hand_face_geometry(mesh, surface, hands_posed: Dict[str, bool]) -> HandFaceGeometry:
    shape = (2, len(HAND_POINTS))
    out = HandFaceGeometry(np.full(shape, np.nan), np.full(shape, NO_LABEL, np.int8),
                           np.full(shape, NO_LABEL, np.int8), np.zeros(shape, bool))
    sides = [s for s, side in enumerate(SIDES) if hands_posed.get(side, False)]
    if not sides:
        return out
    points = hand_points(mesh)[sides].reshape(-1, 3)
    head = mesh.vertices[surface.vertex_ids]
    signed, closest, face = signed_distances(points, head, surface.faces)
    corners = surface.faces[face]
    nearest = corners[np.arange(len(face)),
                      np.argmin(np.linalg.norm(head[corners] - closest[:, None], axis=-1), axis=1)]
    k = len(HAND_POINTS)
    for row, s in enumerate(sides):
        chunk = slice(row * k, (row + 1) * k)
        out.distance[s] = signed[chunk]
        out.region[s] = surface.region[nearest[chunk]]
        out.subregion[s] = surface.subregion[nearest[chunk]]
        out.contact[s] = np.abs(signed[chunk]) < CONTACT_DISTANCE
    return out
