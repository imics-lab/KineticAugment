"""Tests for the week-4 clip container, rotation resampling and the skinned outputs."""

import dataclasses
import math
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kinetic_augment.augmentations.clip import (
    ClipData,
    SceneTransform,
    resample_clip,
    select_frames,
    slerp_resample,
)
from kinetic_augment.augmentations.outputs import (
    blendshape_mirror,
    build_outputs,
    pose_mesh_rest,
)
from kinetic_augment.augmentations.tier1 import POSE_MIRROR, WeakCamera
from tests.hand_assets import requires_smplx

PER_FRAME = (
    "rotations",
    "transl",
    "face_local",
    "face_valid",
    "blendshapes",
    "body_ok",
    "hand_valid",
    "times_s",
)


def assert_clips_equal(a: ClipData, b: ClipData):
    for f in dataclasses.fields(ClipData):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert x.shape == y.shape, f.name
            assert np.array_equal(x, y, equal_nan=True), f.name
        elif isinstance(x, WeakCamera):
            assert x.scale == y.scale and x.residual == y.residual
            assert np.array_equal(x.offset, y.offset)
        elif isinstance(x, SceneTransform):
            assert x.mirror == y.mirror and x.scale == y.scale
            for name in ("rotation", "shift", "pivot"):
                assert np.array_equal(getattr(x, name), getattr(y, name))
        else:
            assert x == y, f.name


def descendants(parents, joint):
    out = [joint]
    for j in range(len(parents)):
        p = parents[j]
        while p >= 0:
            if p == joint:
                out.append(j)
                break
            p = parents[p]
    return out


# --------------------------------------------------------------------------- pure


def test_slerp_resample_midpoint():
    rotations = np.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, math.pi / 2]]])
    mid = slerp_resample(rotations, np.array([0.5]))
    np.testing.assert_allclose(mid[0, 0], [0.0, 0.0, math.pi / 4], atol=1e-6)
    assert np.array_equal(slerp_resample(rotations, np.arange(2.0)), rotations)

    gaps = np.zeros((2, 2, 3))
    gaps[0, 0] = np.nan
    gaps[1, 0] = [0.1, 0.2, 0.3]
    gaps[:, 1] = np.nan
    out = slerp_resample(gaps, np.array([0.5, 0.25]))
    np.testing.assert_array_equal(out[:, 0], [[0.1, 0.2, 0.3]] * 2)
    assert np.isnan(out[:, 1]).all()


def test_slerp_resample_matches_rotation_composition():
    rng = np.random.default_rng(3)
    rotations = rng.normal(scale=0.6, size=(4, 5, 3))
    out = slerp_resample(rotations, np.array([1.25]))
    a = Rotation.from_rotvec(rotations[1])
    b = Rotation.from_rotvec(rotations[2])
    expected = a * Rotation.from_rotvec(0.25 * (a.inv() * b).as_rotvec())
    got = Rotation.from_rotvec(out[0])
    assert np.allclose((got.inv() * expected).magnitude(), 0.0, atol=1e-9)


def test_scene_transform_identity_and_order():
    points = np.random.default_rng(0).normal(size=(4, 7, 3))
    identity = SceneTransform(pivot=np.array([0.3, -0.1, 2.0]))
    assert identity.is_identity()
    assert np.array_equal(identity.apply_points(points), points)

    rotation = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    pivot = np.array([0.2, 0.1, 1.5])
    scene = SceneTransform(True, rotation, 1.1, np.array([0.05, 0.0, 0.0]), pivot)
    assert not scene.is_identity()
    reflected = points.copy()
    reflected[..., 0] = 2 * pivot[0] - reflected[..., 0]
    expected = (reflected - pivot) @ (1.1 * rotation).T + pivot + [0.05, 0.0, 0.0]
    np.testing.assert_allclose(scene.apply_points(points), expected, atol=1e-12)


def test_blendshape_mirror_swaps_left_right():
    names = ("_neutral", "browDownLeft", "browDownRight", "jawOpen", "mouthLeft", "mouthRight")
    perm = blendshape_mirror(names)
    np.testing.assert_array_equal(perm, [0, 2, 1, 3, 5, 4])


# --------------------------------------------------------------------------- SMPL-X


@requires_smplx
def test_pose_mesh_rest_matches_pose_mesh():
    from kinetic_augment.body_model.assembly import pose_mesh
    from tests.clip_fixtures import get_skeleton

    skeleton = get_skeleton()
    rng = np.random.default_rng(1)
    for _ in range(3):
        rotations = rng.normal(scale=0.2, size=(55, 3))
        betas = np.zeros(300)
        betas[:10] = rng.normal(scale=0.8, size=10)
        transl = rng.normal(scale=0.3, size=3)
        body = skeleton.shaped(betas)
        mine = pose_mesh_rest(skeleton, rotations, betas, transl, body.rest_joints)
        ref = pose_mesh(skeleton, rotations, betas, transl)
        np.testing.assert_allclose(mine.vertices, ref.vertices, atol=1e-5)
        np.testing.assert_allclose(mine.joints, ref.joints, atol=1e-5)


@requires_smplx
def test_pose_mesh_rest_honours_scaled_forearm():
    from kinetic_augment.body_model.hand_model import SMPLX_TIP_VERTICES
    from tests.clip_fixtures import get_skeleton

    skeleton = get_skeleton()
    rng = np.random.default_rng(2)
    rotations = rng.normal(scale=0.2, size=(55, 3))
    betas = np.zeros(300)
    rest = skeleton.rest_joints.copy()
    forearm = rest[20] - rest[18]
    edited = rest.copy()
    edited[descendants(skeleton.parents, 20)] += 0.05 * forearm / np.linalg.norm(forearm)

    base = pose_mesh_rest(skeleton, rotations, betas, np.zeros(3), rest)
    longer = pose_mesh_rest(skeleton, rotations, betas, np.zeros(3), edited)
    before = np.linalg.norm(base.joints[20] - base.joints[18])
    after = np.linalg.norm(longer.joints[20] - longer.joints[18])
    assert after - before == pytest.approx(0.05, abs=1e-3)
    tip = SMPLX_TIP_VERTICES["left"][1]
    moved = np.linalg.norm(longer.vertices[tip] - base.vertices[tip])
    assert moved == pytest.approx(0.05, abs=1e-3)  # the edit reaches the mesh, not just FK
    # the hand mesh rides on the moved wrist: the fingertip keeps its distance to the wrist joint
    reach = [np.linalg.norm(m.vertices[tip] - m.joints[20]) for m in (base, longer)]
    assert reach[1] == pytest.approx(reach[0], abs=1e-6)


@requires_smplx
def test_resample_clip_identity_times_is_exact():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=8)
    rotations = clip.rotations.copy()
    rotations[3, 18] = np.nan
    face = clip.face_local.copy()
    face[5, :10] = np.nan
    clip = clip.replace(rotations=rotations, face_local=face)
    assert_clips_equal(resample_clip(clip, np.arange(clip.frames, dtype=float)), clip)


@requires_smplx
def test_resample_clip_interpolates():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=8)
    out = resample_clip(clip, np.array([0.0, 2.5, 7.0]))
    assert out.frames == 3
    np.testing.assert_allclose(out.times_s, [0.0, 2.5 / 30, 7 / 30])
    np.testing.assert_allclose(
        out.rotations[1, 18], 0.5 * (clip.rotations[2, 18] + clip.rotations[3, 18]), atol=1e-9
    )
    np.testing.assert_allclose(out.blendshapes[1], clip.blendshapes[2:4].mean(0))
    np.testing.assert_allclose(out.camera.offset[1], clip.camera.offset[2:4].mean(0))
    assert out.body_ok.shape == (3,) and out.hand_valid.shape == (3, 2)


@requires_smplx
def test_select_frames_shapes():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=7)
    keep = np.arange(clip.frames) % 2 == 0
    for selector in (keep, np.flatnonzero(keep)):
        out = select_frames(clip, selector)
        for name in PER_FRAME:
            assert len(getattr(out, name)) == math.ceil(clip.frames / 2), name
        assert len(out.camera.offset) == math.ceil(clip.frames / 2)
        np.testing.assert_array_equal(out.rotations, clip.rotations[::2])
        np.testing.assert_array_equal(out.camera.offset, clip.camera.offset[::2])
        assert out.frames == 4


@requires_smplx
def test_build_outputs_identity_scene():
    from kinetic_augment.body_model.assembly import pose_mesh
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    skeleton = get_skeleton()
    clip = synthetic_clip(frames=5)
    out = build_outputs(clip, skeleton, keep_vertices=True)
    T = clip.frames
    assert out.pose_world.shape == (T, 33, 3)
    assert out.hands_world.shape == (T, 2, 21, 3)
    assert out.face_world.shape == (T, 478, 3)
    assert out.head_world.shape == (T, 3)
    assert out.blendshapes.shape == (T, 52) and out.blendshape_names == clip.blendshape_names
    assert out.pose_image.shape == (T, 33, 2)
    assert out.hands_image.shape == (T, 2, 21, 2)
    assert out.face_image.shape == (T, 478, 2)
    assert out.hand_valid.shape == (T, 2) and out.face_valid.shape == (T,)
    assert out.body_ok.shape == (T,) and out.times_s.shape == (T,)
    assert len(out.vertices) == T and out.vertices[0].shape == (10475, 3)

    others = [i for i in range(33) if not 11 <= i <= 16]
    assert np.isnan(out.pose_world[:, others]).all()
    assert np.isfinite(out.pose_world[:, 11:17]).all()
    assert np.isfinite(out.hands_world).all() and np.isfinite(out.face_world).all()
    np.testing.assert_array_equal(out.blendshapes, clip.blendshapes)

    for t in range(T):
        ref = pose_mesh(skeleton, clip.rotations[t], clip.betas, clip.transl[t])
        np.testing.assert_allclose(out.hands_world[t, 0, 0], ref.joints[20], atol=1e-5)
        np.testing.assert_allclose(out.hands_world[t, 1, 0], ref.joints[21], atol=1e-5)
        np.testing.assert_allclose(out.pose_world[t, 11:17], ref.joints[16:22], atol=1e-5)
        np.testing.assert_allclose(out.head_world[t], ref.joints[15], atol=1e-5)
        np.testing.assert_allclose(out.vertices[t], ref.vertices, atol=1e-5)
    # the face rides rigidly on the head joint
    radii = np.linalg.norm(out.face_world - out.head_world[:, None], axis=-1)
    np.testing.assert_allclose(radii, np.linalg.norm(clip.face_local, axis=-1), atol=1e-9)
    np.testing.assert_allclose(out.pose_image, clip.camera.project(out.pose_world))


@requires_smplx
def test_build_outputs_unfitted_frames_stay_nan():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=6)
    rotations, body_ok = clip.rotations.copy(), clip.body_ok.copy()
    rotations[3] = np.nan
    body_ok[3] = False
    out = build_outputs(clip.replace(rotations=rotations, body_ok=body_ok), get_skeleton())
    for name in ("pose_world", "hands_world", "face_world", "head_world"):
        value = getattr(out, name)
        assert np.isnan(value[3]).all(), name
        for t in (2, 4):
            assert np.isfinite(value[t]).any(), name
    assert np.isfinite(out.pose_world[[2, 4], 11:17]).all()
    assert np.isfinite(out.hands_world[[2, 4]]).all()
    assert np.isfinite(out.face_world[[2, 4]]).all()
    assert not out.face_valid[3] and not out.hand_valid[3].any()
    assert out.vertices is None


@requires_smplx
def test_build_outputs_no_face():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    clip = synthetic_clip(frames=4)
    clip = clip.replace(face_valid=np.zeros(clip.frames, dtype=bool))
    out = build_outputs(clip, get_skeleton())
    assert np.isnan(out.face_world).all()
    assert np.isnan(out.face_image).all()
    assert np.isfinite(out.hands_world).all()


@requires_smplx
def test_build_outputs_mirror_without_face_table_fails_loudly(monkeypatch):
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    monkeypatch.setitem(sys.modules, "kinetic_augment.augmentations.face_mirror", None)
    clip = synthetic_clip(frames=2)
    clip = clip.replace(scene=SceneTransform(mirror=True))
    with pytest.raises(RuntimeError, match="face mirror table"):
        build_outputs(clip, get_skeleton())


@requires_smplx
def test_build_outputs_mirror_swaps_sides(monkeypatch):
    import types

    from tests.clip_fixtures import get_skeleton, synthetic_clip

    fake = types.ModuleType("kinetic_augment.augmentations.face_mirror")
    fake.FACE_MIRROR = np.arange(478)[::-1].copy()
    monkeypatch.setitem(sys.modules, "kinetic_augment.augmentations.face_mirror", fake)
    skeleton = get_skeleton()
    clip = synthetic_clip(frames=3)
    hand_valid = clip.hand_valid.copy()
    hand_valid[:, 1] = False
    clip = clip.replace(hand_valid=hand_valid)
    base = build_outputs(clip, skeleton)
    pivot = np.array([0.1, 0.0, 0.0])
    mirrored = build_outputs(clip.replace(scene=SceneTransform(mirror=True, pivot=pivot)), skeleton)

    def reflect(p):
        p = p.copy()
        p[..., 0] = 2 * pivot[0] - p[..., 0]
        return p

    np.testing.assert_allclose(mirrored.pose_world, reflect(base.pose_world)[:, POSE_MIRROR])
    np.testing.assert_allclose(mirrored.hands_world, reflect(base.hands_world)[:, ::-1])
    np.testing.assert_array_equal(mirrored.hand_valid, base.hand_valid[:, ::-1])
    np.testing.assert_allclose(mirrored.face_world, reflect(base.face_world)[:, ::-1])
    perm = blendshape_mirror(clip.blendshape_names)
    np.testing.assert_array_equal(mirrored.blendshapes, base.blendshapes[:, perm])
    left = clip.blendshape_names.index("eyeBlinkLeft")
    right = clip.blendshape_names.index("eyeBlinkRight")
    np.testing.assert_array_equal(mirrored.blendshapes[:, left], base.blendshapes[:, right])
