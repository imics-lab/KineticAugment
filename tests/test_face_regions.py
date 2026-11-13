"""Tests for the head surface and its ASL-LEX region labels (design §5, assumption F6)."""

import numpy as np
import pytest

from tests.face_assets import MODEL_DIR, requires_face_assets


def test_vertex_normals_of_a_flat_square_point_up():
    from kinetic_augment.body_model.face_regions import vertex_normals

    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    np.testing.assert_allclose(vertex_normals(vertices, faces), [[0, 0, 1]] * 4)


@pytest.fixture(scope="module")
def setup():
    from kinetic_augment.body_model.assembly import pose_mesh
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    from kinetic_augment.body_model.face_regions import build_head_surface
    from kinetic_augment.body_model.skeleton import SMPLXSkeleton

    skeleton = SMPLXSkeleton(model_path=MODEL_DIR)
    corr = load_face_correspondence(MODEL_DIR)
    surface = build_head_surface(skeleton, corr, MODEL_DIR)
    rest = pose_mesh(skeleton, np.zeros((55, 3)), skeleton.betas, np.zeros(3)).vertices
    return skeleton, corr, surface, rest


def label_near(setup, point):
    from kinetic_augment.body_model.face_regions import REGIONS, SUBREGIONS

    _, _, surface, rest = setup
    i = np.argmin(np.linalg.norm(rest[surface.vertex_ids] - point, axis=1))
    sub = surface.subregion[i]
    return REGIONS[surface.region[i]], (SUBREGIONS[sub] if sub >= 0 else None)


@requires_face_assets
def test_every_vertex_is_labelled_and_every_region_used(setup):
    from kinetic_augment.body_model.face_regions import REGIONS

    _, _, surface, _ = setup
    assert (surface.region >= 0).all()
    assert set(np.unique(surface.region)) == set(range(len(REGIONS)))
    assert len(surface.faces) > 0 and surface.faces.max() < len(surface.vertex_ids)


@requires_face_assets
def test_anchor_points_get_their_regions(setup):
    from kinetic_augment.body_model.head_converter import EAR_VERTICES

    _, corr, surface, rest = setup

    def p(*mp):
        return corr.points(rest, mp).mean(axis=0)

    brow_centre = p(105, 334)
    assert label_near(setup, p(13, 14))[0] == "Mouth"
    assert label_near(setup, p(4)) == ("CheekNose", "Nose")
    assert label_near(setup, p(17) + [0, -0.025, 0])[0] == "Chin"
    assert label_near(setup, brow_centre + [0, 0.03, 0])[0] == "Forehead"
    assert label_near(setup, p(105)) == ("Forehead", "Brow")
    assert label_near(setup, (p(2) + p(0)) / 2)[0] == "UpperLip"
    assert label_near(setup, rest[EAR_VERTICES[0]])[0] == "Ear"
    top = surface.vertex_ids[np.argmax(rest[surface.vertex_ids, 1])]
    assert label_near(setup, rest[top])[0] == "HeadTop"


@requires_face_assets
def test_no_cheek_vertex_inside_the_lip_outline(setup):
    """Inner-lip and mouth-interior vertices are not front-facing but are still Mouth."""
    from kinetic_augment.body_model.face_regions import REGIONS, region_anchors
    from kinetic_augment.body_model.head_converter import points_in_polygon

    _, corr, surface, rest = setup
    a = region_anchors(rest, corr)
    points = rest[surface.vertex_ids]
    ahead = points[:, 2] > a.ears[:, 2].mean()
    inside = points_in_polygon(points[:, :2], a.outer_lip[:, :2]) & ahead
    assert inside.any()
    assert not (surface.region[inside] == REGIONS.index("CheekNose")).any()


@requires_face_assets
def test_no_surface_vertex_is_eyeball_dominated(setup):
    """Eyeball meshes sit inside the skull and would flip signed distances behind the eyes."""
    from kinetic_augment.body_model.face_regions import EYE_JOINTS, REGIONS

    skeleton, _, surface, _ = setup
    weights = skeleton.model.lbs_weights.detach().numpy()[surface.vertex_ids]
    assert not np.isin(weights.argmax(axis=1), EYE_JOINTS).any()
    assert (surface.region == REGIONS.index("Eye")).any()
