"""Tests for hand-to-face proximity measured in the image (V1.3 task 13)."""

import numpy as np
import pytest

from kinetic_augment.body_model.alignment import apply_similarity
from kinetic_augment.body_model.face_regions import NO_LABEL, HeadSurface
from kinetic_augment.body_model.hand_face_geometry import CONTACT_DISTANCE, HAND_POINTS
from kinetic_augment.body_model.image_proximity import (
    ProjectedHead, image_hand_points, image_proximity, project_head,
)

SCALE = 1000.0  # pixels per metre

# Unit-square corners wound so that the normal points to -z (toward the camera).
SQUARE_FACES = np.array([[0, 1, 2], [1, 3, 2]])


def square(z=0.0, size=0.1):
    return np.array([[0, 0, z], [0, size, z], [size, 0, z], [size, size, z]], float)


def projected_from(vertices, faces, visible=None):
    pts = np.asarray(vertices, float) * SCALE
    if visible is None:
        tri = pts[faces]
        normal_z = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
        visible = normal_z < 0
    return ProjectedHead(pts, np.asarray(visible), SCALE, 0.0)


def hand_at(point, side=0, index=0):
    hand = np.full((2, len(HAND_POINTS), 2), np.nan)
    hand[side, index] = point
    return hand


def surface_for(n, region):
    return HeadSurface(np.arange(n), SQUARE_FACES, np.asarray(region, np.int8),
                       np.full(n, NO_LABEL, np.int8))


def test_square_faces_the_camera():
    head = projected_from(square(), SQUARE_FACES)
    assert head.visible.all()


def test_inside_point_overlaps_with_zero_distance_and_nearest_vertex_region():
    head = projected_from(square(), SQUARE_FACES)
    out = image_proximity(hand_at([10.0, 90.0]), head, surface_for(4, [0, 1, 2, 3]))
    assert out.overlap[0, 0] and out.distance[0, 0] == 0.0 and out.near[0, 0]
    assert out.region[0, 0] == 1  # nearest vertex is (0, 100)
    assert out.region[1, 0] == NO_LABEL and np.isnan(out.distance[1]).all()
    assert not out.overlap[1].any() and not out.near[1].any()


def test_outside_point_distance_is_pixel_gap_over_scale():
    head = projected_from(square(), SQUARE_FACES)
    gap = 30.0
    out = image_proximity(hand_at([100.0 + gap, 50.0]), head, surface_for(4, [0, 1, 2, 3]))
    assert not out.overlap[0, 0]
    assert out.distance[0, 0] == pytest.approx(gap / SCALE)
    assert not out.near[0, 0]  # 3 cm > 2 cm
    assert out.region[0, 0] in (2, 3)
    near = image_proximity(hand_at([100.0 + 15.0, 50.0]), head, surface_for(4, [0, 1, 2, 3]))
    assert near.near[0, 0] and near.distance[0, 0] < CONTACT_DISTANCE


def test_back_facing_triangles_are_ignored():
    back = square()[:, [0, 1, 2]]
    faces = SQUARE_FACES[:, [0, 2, 1]]  # reversed winding: faces +z
    head = projected_from(back, faces)
    assert not head.visible.any()
    out = image_proximity(hand_at([50.0, 50.0]), head, surface_for(4, [0, 1, 2, 3]))
    assert not out.overlap[0, 0] and np.isnan(out.distance[0, 0])
    assert out.region[0, 0] == NO_LABEL


def test_back_facing_copy_does_not_change_the_answer():
    vertices = np.concatenate([square(z=0.0), square(z=0.05)])
    faces = np.concatenate([SQUARE_FACES, SQUARE_FACES[:, [0, 2, 1]] + 4])
    head = projected_from(vertices, faces)
    assert head.visible.tolist() == [True, True, False, False]
    surface = HeadSurface(np.arange(8), faces, np.array([1, 1, 1, 1, 2, 2, 2, 2], np.int8),
                          np.full(8, NO_LABEL, np.int8))
    out = image_proximity(hand_at([50.0, 50.0]), head, surface)
    assert out.overlap[0, 0] and out.region[0, 0] == 1


def test_front_most_overlapping_square_sets_the_region():
    vertices = np.concatenate([square(z=0.0), square(z=0.05)])  # second is farther (z away)
    faces = np.concatenate([SQUARE_FACES, SQUARE_FACES + 4])
    head = projected_from(vertices, faces)
    assert head.visible.all()
    surface = HeadSurface(np.arange(8), faces, np.array([1, 1, 1, 1, 2, 2, 2, 2], np.int8),
                          np.full(8, NO_LABEL, np.int8))
    out = image_proximity(hand_at([50.0, 50.0]), head, surface)
    assert out.region[0, 0] == 1
    swapped = HeadSurface(np.arange(8), faces, np.array([2, 2, 2, 2, 1, 1, 1, 1], np.int8),
                          np.full(8, NO_LABEL, np.int8))
    assert image_proximity(hand_at([50.0, 50.0]), head, swapped).region[0, 0] == 2


def test_subregion_follows_region_vertex():
    head = projected_from(square(), SQUARE_FACES)
    surface = HeadSurface(np.arange(4), SQUARE_FACES, np.zeros(4, np.int8),
                          np.array([0, 1, 2, 2], np.int8))
    out = image_proximity(hand_at([90.0, 5.0]), head, surface)
    assert out.subregion[0, 0] == 2  # nearest vertex (100, 0) is vertex 2


def test_image_hand_points_order_nan_and_palm():
    image = np.full((2, 21, 3), np.nan, np.float32)
    rng = np.random.default_rng(1)
    image[1] = rng.uniform(0.1, 0.9, (21, 3))
    points = image_hand_points(image, 200, 100)
    assert points.shape == (2, 7, 2) and np.isnan(points[0]).all()
    scale = np.array([200.0, 100.0])
    for k, mp in enumerate((0, 4, 8, 12, 16, 20)):
        np.testing.assert_allclose(points[1, k], image[1, mp, :2] * scale, rtol=1e-6)
    np.testing.assert_allclose(points[1, 6], (image[1, 0, :2] + image[1, 9, :2]) / 2 * scale,
                               rtol=1e-6)


def rotation_about(axis, degrees):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def test_project_head_recovers_scale_rotation_and_applies_to_the_head():
    rng = np.random.default_rng(2)
    rigid_world = rng.normal(scale=0.05, size=(17, 3))
    rotation = rotation_about([1, 2, 3], 12.0)
    shift = np.array([320.0, 240.0, 5.0])
    rigid_px = apply_similarity(rigid_world, 900.0, rotation, shift)
    head = rng.normal(scale=0.1, size=(30, 3))
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    out = project_head(head, faces, rigid_world, rigid_px, np.ones(17))
    assert out.scale == pytest.approx(900.0)
    assert out.rotation_deg == pytest.approx(12.0, abs=1e-6)
    np.testing.assert_allclose(out.points, apply_similarity(head, 900.0, rotation, shift),
                               atol=1e-6)
    assert out.visible.shape == (2,)


def test_project_head_ignores_zero_weight_points():
    rng = np.random.default_rng(3)
    rigid_world = rng.normal(scale=0.05, size=(17, 3))
    rigid_px = apply_similarity(rigid_world, 800.0, np.eye(3), np.zeros(3))
    rigid_px[0] += 500.0  # a corrupted (occluded) point
    weights = np.ones(17)
    weights[0] = 0.0
    out = project_head(rng.normal(size=(6, 3)), np.array([[0, 1, 2]]), rigid_world, rigid_px,
                       weights)
    assert out.scale == pytest.approx(800.0) and out.rotation_deg == pytest.approx(0.0, abs=1e-6)


def test_neck_triangles_are_left_out_of_the_image_test():
    from kinetic_augment.body_model.face_regions import REGIONS

    chin, neck = REGIONS.index("Chin"), REGIONS.index("Neck")
    # One face triangle (0..100 px) and a Neck triangle directly below it (150..250 px in y).
    vertices = np.array([[0, 0, 0], [0, 0.1, 0], [0.1, 0, 0],
                         [0, 0.15, 0], [0, 0.25, 0], [0.1, 0.15, 0]], float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    head = projected_from(vertices, faces)
    assert head.visible.all()
    surface = HeadSurface(np.arange(6), faces, np.array([chin] * 3 + [neck] * 3, np.int8),
                          np.full(6, NO_LABEL, np.int8))
    out = image_proximity(hand_at([20.0, 170.0]), head, surface)  # over the Neck triangle
    assert not out.overlap[0, 0] and not out.near[0, 0]
    assert out.distance[0, 0] == pytest.approx(_face_gap([20.0, 170.0]) / SCALE)
    assert out.region[0, 0] == chin


def _face_gap(point):
    """Pixel distance from a point to the face triangle (0,0)-(0,100)-(100,0)."""
    a, b = np.array([0.0, 100.0]), np.array([100.0, 0.0])
    p = np.asarray(point)
    t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))
