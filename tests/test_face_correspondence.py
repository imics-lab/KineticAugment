"""
Tests for the MediaPipe -> SMPL-X face correspondence (design §3, assumption F1).
"""

import numpy as np
import pytest

from kinetic_augment.body_model.face_correspondence import (
    EYE_CORNERS, INNER_LIP, JAW_PAIR, LEFT_EYE, LEFT_EYEBROW, NOSE, OUTER_LIP, RIGHT_EYE,
    RIGHT_EYEBROW, RIGID,
)
from tests.face_assets import MODEL_DIR, requires_face_assets

GROUPS = (RIGHT_EYEBROW, LEFT_EYEBROW, RIGHT_EYE, LEFT_EYE, OUTER_LIP, INNER_LIP, NOSE)


def test_groups_match_mediapipe_connection_sets():
    connections = pytest.importorskip("mediapipe.python.solutions.face_mesh_connections")

    def points(edges):
        return {i for edge in edges for i in edge}

    assert set(LEFT_EYEBROW) == points(connections.FACEMESH_LEFT_EYEBROW)
    assert set(RIGHT_EYEBROW) == points(connections.FACEMESH_RIGHT_EYEBROW)
    assert set(LEFT_EYE) == points(connections.FACEMESH_LEFT_EYE)
    assert set(RIGHT_EYE) == points(connections.FACEMESH_RIGHT_EYE)
    assert set(OUTER_LIP) | set(INNER_LIP) == points(connections.FACEMESH_LIPS)
    edges = {frozenset(e) for e in connections.FACEMESH_LIPS}
    for loop in (OUTER_LIP, INNER_LIP):
        for a, b in zip(loop, loop[1:] + loop[:1]):
            assert frozenset((a, b)) in edges, (a, b)


def test_rigid_set_and_jaw_pair_are_disjoint_and_sized():
    assert len(RIGID) == 17 and len(set(RIGID)) == 17
    assert set(EYE_CORNERS) <= set(RIGHT_EYE) | set(LEFT_EYE)
    assert not set(JAW_PAIR) & set(RIGID)
    assert sum(len(g) for g in GROUPS) == 105


def test_transfer_maps_triangles_through_the_vertex_map():
    from kinetic_augment.body_model.face_correspondence import transfer_embedding

    flame_faces = np.array([[0, 1, 2], [2, 1, 3]])
    flame_to_smplx = np.array([10, 11, 12, 13])
    corr = transfer_embedding(np.array([1]), np.array([[0.2, 0.3, 0.5]]), np.array([4]),
                              flame_faces, flame_to_smplx)
    np.testing.assert_array_equal(corr.vertex_ids, [[12, 11, 13]])
    vertices = np.zeros((14, 3))
    vertices[[11, 12, 13]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    np.testing.assert_allclose(corr.points(vertices), [[0.3, 0.2, 0.5]])
    np.testing.assert_allclose(corr.points(vertices, (4,)), [[0.3, 0.2, 0.5]])
    with pytest.raises(KeyError):
        corr.rows((5,))


@pytest.fixture(scope="module")
def corr():
    from kinetic_augment.body_model.face_correspondence import load_face_correspondence
    return load_face_correspondence(MODEL_DIR)


@pytest.fixture(scope="module")
def smplx_npz():
    return np.load(f"{MODEL_DIR}/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)


@requires_face_assets
def test_embedding_covers_exactly_the_named_groups(corr):
    assert len(corr.mp_indices) == 105
    assert set(corr.mp_indices.tolist()) == {i for g in GROUPS for i in g}
    np.testing.assert_allclose(corr.bary.sum(axis=1), 1.0, atol=1e-5)


@requires_face_assets
def test_every_flame_triangle_is_a_smplx_triangle(smplx_npz):
    from kinetic_augment.body_model.face_correspondence import (
        load_flame_faces, load_flame_to_smplx,
    )

    flame_faces = load_flame_faces(f"{MODEL_DIR}/flame/generic_model.pkl")
    mapped = np.sort(load_flame_to_smplx(MODEL_DIR)[flame_faces], axis=1)
    smplx_faces = {tuple(f) for f in np.sort(smplx_npz["f"].astype(np.int64), axis=1)}
    missing = [tuple(f) for f in mapped if tuple(f) not in smplx_faces]
    assert not missing, f"{len(missing)} of {len(mapped)} FLAME triangles have no SMPL-X match"


@requires_face_assets
def test_f1_eye_corners_and_nose_tip_match_smplx_landmarks(corr, smplx_npz):
    """
    Assumption F1. SMPL-X's own 51 landmarks are iBUG-68 minus the jaw contour: 13 is the nose
    tip (iBUG 30), 19/22 the right eye's outer/inner corner (iBUG 36/39), 25/28 the left eye's
    inner/outer corner (iBUG 42/45).
    """
    v = smplx_npz["v_template"].astype(np.float64)
    f = smplx_npz["f"].astype(np.int64)
    lmk = np.einsum("nkc,nk->nc", v[f[smplx_npz["lmk_faces_idx"]]],
                    smplx_npz["lmk_bary_coords"].astype(np.float64))
    pairs = {33: 19, 133: 22, 362: 25, 263: 28}
    ours = corr.points(v, tuple(pairs))
    distance = np.linalg.norm(ours - lmk[list(pairs.values())], axis=1)
    print("eye-corner distances (mm):", np.round(1000 * distance, 2))
    assert (distance < 0.006).all()
    swapped = np.linalg.norm(ours - lmk[[28, 25, 22, 19]], axis=1)
    assert (swapped > distance).all(), "left and right are exchanged"
    nose = np.linalg.norm(corr.points(v, (4,))[0] - lmk[13])
    print(f"nose-tip distance: {1000 * nose:.2f} mm")
    assert nose < 0.008
