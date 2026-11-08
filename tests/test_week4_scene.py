"""Week-4 scene augmentations: face mirror table, mirror, view, scale, uncentre, camera fit."""

import numpy as np
import pytest

from kinetic_augment.augmentations import tier1
from kinetic_augment.augmentations.base import AugmentationRegistry
from kinetic_augment.augmentations.clip import ChainContext, SceneTransform
from kinetic_augment.augmentations.face_mirror import FACE_MIRROR
from kinetic_augment.augmentations.outputs import build_outputs
from kinetic_augment.augmentations.scene import (
    Mirror,
    Scale,
    Uncentre,
    View,
    blendshape_mirror,
    fit_clip_camera,
)
from tests.hand_assets import requires_smplx

PIVOT = np.array([0.05, -0.02, 0.1])


def make_ctx(draw=None, pivot=PIVOT):
    from tests.clip_fixtures import get_skeleton

    skeleton = get_skeleton()
    return ChainContext(
        skeleton=skeleton, parents=skeleton.parents, pivot=np.array(pivot), tier1_draw=draw
    )


def make_draw(mirror=False, yaw=0.0, pitch=0.0, scale=1.0, u=(0.5, 0.5)):
    return tier1.Tier1Draw(mirror, yaw, pitch, 1.0, (), scale, u)


def test_face_mirror_is_involution():
    assert len(FACE_MIRROR) == 478
    perm = np.array(FACE_MIRROR)
    assert sorted(perm) == list(range(478))
    np.testing.assert_array_equal(perm[perm], np.arange(478))
    assert FACE_MIRROR[33] == 263 and FACE_MIRROR[133] == 362 and FACE_MIRROR[1] == 1
    assert FACE_MIRROR[468] == 473 and FACE_MIRROR[469] == 476 and FACE_MIRROR[472] == 477


def test_blendshape_mirror():
    from tests.clip_fixtures import BLENDSHAPE_NAMES

    perm = blendshape_mirror(BLENDSHAPE_NAMES)
    names = list(BLENDSHAPE_NAMES)
    assert perm[names.index("browDownLeft")] == names.index("browDownRight")
    assert perm[names.index("browDownRight")] == names.index("browDownLeft")
    assert perm[names.index("jawOpen")] == names.index("jawOpen")
    np.testing.assert_array_equal(perm[perm], np.arange(len(names)))


def test_registered_keys_and_defaults():
    for name, cls in (
        ("scene.mirror", Mirror),
        ("scene.view", View),
        ("scene.scale", Scale),
        ("scene.uncentre", Uncentre),
    ):
        assert AugmentationRegistry._registry[name] is cls
        assert cls.regions == ("scene",)
    assert Mirror().ranges == {"p": 0.5}
    assert View().ranges == {"yaw_deg": 25.0, "pitch_deg": 6.0}
    assert Scale().ranges == {"factor": [0.75, 1.25]}
    assert Uncentre().ranges == {"limit": 0.125}


def test_sample_reads_tier1_draw():
    draw = make_draw(mirror=True, yaw=7.0, pitch=-2.0, scale=1.1, u=(0.2, 0.9))
    ctx = make_ctx(draw)
    rng = np.random.default_rng(0)
    assert Mirror().sample(rng, None, ctx) == {"mirror": True}
    assert View().sample(rng, None, ctx) == {"yaw_deg": 7.0, "pitch_deg": -2.0}
    assert Scale().sample(rng, None, ctx) == {"factor": 1.1}
    assert Uncentre().sample(rng, None, ctx) == {"u": [0.2, 0.9], "limit": 0.125}
    with pytest.raises(ValueError):
        Mirror().sample(rng, None, make_ctx(None))


def test_composition_matches_sequential_points():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=2)
    rng = np.random.default_rng(1)
    points = rng.normal(size=(40, 3))
    ctx = make_ctx(None)
    out = View().apply(clip, {"yaw_deg": 20.0, "pitch_deg": -5.0}, ctx)
    out = Scale().apply(out, {"factor": 1.2}, ctx)
    first = SceneTransform(rotation=tier1.view_rotation(20.0, -5.0), pivot=PIVOT)
    second = SceneTransform(scale=1.2, pivot=PIVOT)
    expected = second.apply_points(first.apply_points(points))
    np.testing.assert_allclose(out.scene.apply_points(points), expected, atol=1e-12)


def test_composition_with_mirror_and_shift():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=2)
    points = np.random.default_rng(2).normal(size=(40, 3))
    ctx = make_ctx(None)
    view = SceneTransform(rotation=tier1.view_rotation(15.0, 4.0), pivot=PIVOT)
    shifted = SceneTransform(shift=np.array([0.1, -0.2, 0.0]), pivot=PIVOT)
    start = clip.replace(scene=SceneTransform(shift=np.array([0.1, -0.2, 0.0]), pivot=PIVOT))
    out = View().apply(start, {"yaw_deg": 15.0, "pitch_deg": 4.0}, ctx)
    np.testing.assert_allclose(
        out.scene.apply_points(points), view.apply_points(shifted.apply_points(points)), atol=1e-12
    )
    # A mirror drawn after a rotation: reflect the already transformed points.
    start = clip.replace(scene=view)
    out = Mirror().apply(start, {"mirror": True}, ctx)
    reflected = tier1.reflect_x(view.apply_points(points), PIVOT[0])
    np.testing.assert_allclose(out.scene.apply_points(points), reflected, atol=1e-12)
    assert out.scene.mirror


def test_identity_draws_return_the_clip():
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=2)
    ctx = make_ctx(None)
    for aug in (Mirror(), View(), Scale(), Uncentre()):
        assert aug.apply(clip, aug.identity_draw(), ctx).scene.is_identity()
    assert Mirror().apply(clip, {"mirror": False}, ctx) is clip


@requires_smplx
def test_mirror_swaps_hands():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    skeleton = get_skeleton()
    clip = synthetic_clip(frames=4)
    hand_valid = clip.hand_valid.copy()
    hand_valid[:, 1] = False
    clip = clip.replace(hand_valid=hand_valid)
    base = build_outputs(clip, skeleton)
    mirrored = build_outputs(Mirror().apply(clip, {"mirror": True}, make_ctx(None)), skeleton)
    original = base.hands_world.copy()
    original[..., 0] = 2 * PIVOT[0] - original[..., 0]
    np.testing.assert_allclose(mirrored.hands_world[:, 0], original[:, 1], atol=1e-12)
    np.testing.assert_allclose(mirrored.hands_world[:, 1], original[:, 0], atol=1e-12)
    np.testing.assert_array_equal(mirrored.hand_valid, base.hand_valid[:, ::-1])
    face = base.face_world.copy()
    face[..., 0] = 2 * PIVOT[0] - face[..., 0]
    np.testing.assert_allclose(mirrored.face_world, face[:, FACE_MIRROR], atol=1e-12)


@requires_smplx
def test_scene_identity_exact():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    skeleton = get_skeleton()
    clip = synthetic_clip(frames=4)
    base = build_outputs(clip, skeleton)
    out = clip
    for aug in (Mirror(), View(), Scale(), Uncentre()):
        out = aug.apply(out, aug.identity_draw(), make_ctx(None))
    after = build_outputs(out, skeleton)
    for name in ("pose_world", "hands_world", "face_world", "pose_image", "hands_image"):
        np.testing.assert_array_equal(getattr(base, name), getattr(after, name))


@requires_smplx
def test_view_keeps_relative_positions():
    from tests.clip_fixtures import get_skeleton, synthetic_clip

    skeleton = get_skeleton()
    clip = synthetic_clip(frames=4)
    base = build_outputs(clip, skeleton)
    viewed = build_outputs(
        View().apply(clip, {"yaw_deg": 22.0, "pitch_deg": -5.0}, make_ctx(None)), skeleton
    )

    def wrist_gap(o):
        return np.linalg.norm(o.pose_world[:, 15] - o.pose_world[:, 16], axis=-1)

    np.testing.assert_allclose(wrist_gap(viewed), wrist_gap(base), atol=1e-9)
    assert not np.allclose(viewed.pose_world[:, 15], base.pose_world[:, 15])


@requires_smplx
def test_fit_clip_camera_recovers_synthetic_camera():
    from types import SimpleNamespace

    from tests.clip_fixtures import get_skeleton, synthetic_clip

    skeleton = get_skeleton()
    clip = synthetic_clip(frames=6)
    outputs = build_outputs(clip.replace(camera=None), skeleton)
    width, height = 640, 480
    camera = tier1.WeakCamera((400.0 / width, 400.0 / height), clip.camera.offset, 0.0)
    pose_image = np.full((6, 33, 3), np.nan)
    pose_image[..., :2] = camera.project(outputs.pose_world)
    image = np.zeros((6, 2, 21, 3))
    image[..., :2] = camera.project(outputs.hands_world)
    hands = SimpleNamespace(
        pose_image=pose_image,
        pose_visibility=np.ones((6, 33)),
        image=image,
        width=width,
        height=height,
    )
    fitted = fit_clip_camera(clip, hands, skeleton)
    assert fitted.residual < 1e-9
    np.testing.assert_allclose(fitted.offset, camera.offset, atol=1e-9)
    np.testing.assert_allclose(fitted.scale, camera.scale, atol=1e-9)
    with pytest.raises(ValueError):
        fit_clip_camera(synthetic_clip(frames=5), hands, skeleton)


@requires_smplx
def test_uncentre_keeps_hands_in_frame():
    from tests.clip_fixtures import get_skeleton, real_clip

    clip, hands, _ = real_clip()
    skeleton = get_skeleton()
    camera = fit_clip_camera(clip, hands, skeleton)
    clip = clip.replace(camera=camera)

    def points(c):
        o = build_outputs(c, skeleton)
        hand = np.where(o.hand_valid[:, :, None, None], o.hands_image, np.nan)
        return np.concatenate([hand.reshape(len(hand), 42, 2), o.pose_image[:, [15, 16]]], axis=1)

    before = points(clip)
    moved = 0
    for seed in range(20):
        u = tuple(float(v) for v in np.random.default_rng(seed).random(2))
        draw = Uncentre().sample(np.random.default_rng(0), clip, make_ctx(make_draw(u=u)))
        out = Uncentre().apply(clip, draw, make_ctx(None, pivot=np.zeros(3)))
        loss = tier1.frame_loss(before, points(out))
        assert loss == 0.0 or loss is None
        moved += int(out.scene.shift.any())
    assert moved > 0
