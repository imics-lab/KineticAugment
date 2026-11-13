"""Tests for signed hand-to-face distances, regions and contact (design §5)."""

import numpy as np
import pytest

from kinetic_augment.body_model.hand_face_geometry import (
    closest_points_on_triangles, signed_distances,
)
from tests.face_assets import MODEL_DIR, requires_face_assets

TRIANGLE = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


@pytest.mark.parametrize("point, closest", [
    ([0.2, 0.2, 0.5], [0.2, 0.2, 0.0]),     # above the interior
    ([-1.0, -1.0, 0.0], [0.0, 0.0, 0.0]),   # beyond vertex A
    ([2.0, -0.5, 0.0], [1.0, 0.0, 0.0]),    # beyond vertex B
    ([0.5, -1.0, 0.3], [0.5, 0.0, 0.0]),    # beyond edge AB
    ([1.0, 1.0, 0.0], [0.5, 0.5, 0.0]),     # beyond edge BC
    ([-1.0, 0.5, 0.0], [0.0, 0.5, 0.0]),    # beyond edge AC
])
def test_closest_point_regions(point, closest):
    got, face, distance = closest_points_on_triangles(np.array([point]), TRIANGLE)
    np.testing.assert_allclose(got[0], closest, atol=1e-12)
    np.testing.assert_allclose(distance[0], np.linalg.norm(np.subtract(point, closest)))


def test_closest_point_agrees_with_dense_sampling():
    rng = np.random.default_rng(0)
    triangles = rng.normal(size=(5, 3, 3))
    points = rng.normal(size=(30, 3)) * 2
    _, _, distance = closest_points_on_triangles(points, triangles)
    u, v = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    keep = (u + v) <= 1
    u, v = u[keep], v[keep]
    samples = (triangles[:, None, 0] + u[:, None] * (triangles[:, None, 1] - triangles[:, None, 0])
               + v[:, None] * (triangles[:, None, 2] - triangles[:, None, 0])).reshape(-1, 3)
    sampled = np.min(np.linalg.norm(points[:, None] - samples[None], axis=-1), axis=1)
    assert (distance <= sampled + 1e-9).all()
    assert (distance >= sampled - 0.03).all()


def test_sign_follows_the_triangle_normal():
    vertices = TRIANGLE[0]
    faces = np.array([[0, 1, 2]])  # counter-clockwise seen from +z: normal +z
    signed, _, _ = signed_distances(np.array([[0.2, 0.2, 0.4], [0.2, 0.2, -0.1]]), vertices, faces)
    np.testing.assert_allclose(signed, [0.4, -0.1])


@pytest.fixture(scope="module")
def head():
    from kinetic_augment.body_model.assembly import pose_mesh
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    from kinetic_augment.body_model.face_regions import build_head_surface, vertex_normals
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton

    skeleton = SMPLXSkeleton(model_path=MODEL_DIR)
    corr = load_face_correspondence(MODEL_DIR)
    surface = build_head_surface(skeleton, corr, MODEL_DIR)
    mesh = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3))
    normals = vertex_normals(mesh.vertices, np.asarray(skeleton.model.faces, np.int64))
    return skeleton, corr, surface, mesh, normals


@requires_face_assets
def test_points_off_the_forehead_have_the_expected_signed_distance(head):
    _, corr, surface, mesh, normals = head
    brow = corr.points(mesh.vertices, (105, 334)).mean(axis=0) + [0, 0.03, 0]
    ids = surface.vertex_ids
    v = ids[np.argmin(np.linalg.norm(mesh.vertices[ids] - brow, axis=1))]
    outside = mesh.vertices[v] + 0.03 * normals[v]
    inside = mesh.vertices[v] - 0.004 * normals[v]
    signed, _, _ = signed_distances(np.array([outside, inside]), mesh.vertices[ids], surface.faces)
    assert 0.027 < signed[0] <= 0.0301
    assert -0.0045 < signed[1] < 0


@requires_face_assets
def test_unposed_hands_are_nan_and_contact_needs_two_centimetres(head):
    from kinetic_augment.body_model.hand_face_geometry import CONTACT_DISTANCE, hand_face_geometry

    _, _, surface, mesh, _ = head
    geometry = hand_face_geometry(mesh, surface, {"left": False, "right": True})
    assert np.isnan(geometry.distance[0]).all() and (geometry.region[0] == -1).all()
    assert not geometry.contact[0].any()
    right = geometry.distance[1]
    assert np.isfinite(right).all()
    np.testing.assert_array_equal(geometry.contact[1], np.abs(right) < CONTACT_DISTANCE)


@requires_face_assets
def test_points_inside_the_head_behind_the_eye_are_negative(head):
    from kinetic_augment.body_model.face_correspondence import LEFT_EYE, RIGHT_EYE
    from kinetic_augment.body_model.face_regions import REGIONS

    _, corr, surface, mesh, normals = head
    ids = surface.vertex_ids
    for eye in (LEFT_EYE, RIGHT_EYE):
        centre = corr.points(mesh.vertices, tuple(eye)).mean(axis=0)
        behind = np.array([centre - [0, 0, d] for d in (0.02, 0.03, 0.05)])
        signed, _, _ = signed_distances(behind, mesh.vertices[ids], surface.faces)
        assert (signed < 0).all(), signed
    eye_skin = ids[surface.region == REGIONS.index("Eye")]
    pushed = mesh.vertices[eye_skin] - 0.01 * normals[eye_skin]
    signed, _, _ = signed_distances(pushed, mesh.vertices[ids], surface.faces)
    assert (signed < 0).mean() >= 0.9


def test_point_nearest_an_open_edge_is_outside_whatever_side_of_the_plane():
    faces = np.array([[0, 1, 2]])
    points = np.array([[0.5, -1.0, -0.3], [0.5, -1.0, 0.3], [0.2, 0.2, -0.1]])
    signed, _, _ = signed_distances(points, TRIANGLE[0], faces)
    assert signed[0] > 0 and signed[1] > 0   # closest point on the open edge AB
    assert signed[2] < 0                      # closest point interior: normal sign kept


def test_edge_shared_by_two_triangles_is_not_open():
    # a tent: the shared edge's neighbours fold, so a point below the ridge is inside
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0.3], [0.5, -1, 0.3]], float)
    faces = np.array([[0, 1, 2], [1, 0, 3]])  # shared edge 0-1; both normals point up (+z)
    signed, _, _ = signed_distances(np.array([[0.5, 0.0, -0.2]]), vertices, faces)
    assert signed[0] < 0


@requires_face_assets
def test_points_below_the_open_neck_rim_are_outside(head):
    _, _, surface, mesh, _ = head
    ids = surface.vertex_ids
    lowest = mesh.vertices[ids][np.argmin(mesh.vertices[ids][:, 1])]
    points = lowest + np.array([[0, -d, 0] for d in (0.10, 0.30, 0.50)])
    signed, _, _ = signed_distances(points, mesh.vertices[ids], surface.faces)
    assert (signed > 0).all(), signed
