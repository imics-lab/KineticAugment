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
    assert Mirror().ranges == {"p": 0.5, "blacklist": "configs/mirror_blacklist.yaml"}
    assert View().ranges == {"yaw_deg": 25.0, "pitch_deg": 6.0}
    assert Scale().ranges == {"factor": [0.75, 1.25]}
    assert Uncentre().ranges == {"limit": 0.125}


def test_sample_reads_tier1_draw():
    draw = make_draw(mirror=True, yaw=7.0, pitch=-2.0, scale=1.1, u=(0.2, 0.9))
    ctx = make_ctx(draw)
    rng = np.random.default_rng(0)
    # no clip means no gloss: the decision is the Tier 1 one, plus the audit flag
    assert Mirror().sample(rng, None, ctx) == {"mirror": True, "blacklist": "gloss_unknown"}
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
    assert fitted.frame_size == (width, height)
    np.testing.assert_allclose(fitted.offset, camera.offset, atol=1e-9)
    np.testing.assert_allclose(fitted.scale, camera.scale, atol=1e-9)
    with pytest.raises(ValueError):
        fit_clip_camera(synthetic_clip(frames=5), hands, skeleton)
    # no observed z in this fixture and no face clip: every stream falls back, and says so
    assert fitted.depth_scales == {}
    assert fitted.depth_fallback == ("pose", "left_hand", "right_hand", "face")


@requires_smplx
def test_fit_clip_camera_fits_a_depth_scale_per_stream():
    from types import SimpleNamespace

    from tests.clip_fixtures import get_skeleton, synthetic_clip

    skeleton = get_skeleton()
    clip = synthetic_clip(frames=6)
    out = build_outputs(clip.replace(camera=None), skeleton)
    width, height = 640, 480
    camera = tier1.WeakCamera((400.0 / width, 400.0 / height), clip.camera.offset, 0.0)
    truth = {"pose": 2.2, "left_hand": 0.35, "right_hand": 0.5, "face": 0.75}
    pose_image = np.full((6, 33, 3), np.nan)
    pose_image[..., :2] = camera.project(out.pose_world)
    pose_image[:, 11:17, 2] = truth["pose"] * (
        out.pose_world[:, 11:17, 2] - out.hip_centre_world[:, None, 2]
    )
    pose_image[:, [23, 24], 2] = 0.0  # MediaPipe's pose z origin is the hip midpoint
    image = np.zeros((6, 2, 21, 3))
    image[..., :2] = camera.project(out.hands_world)
    for side, stream in enumerate(("left_hand", "right_hand")):
        dz = out.hands_world[:, side, :, 2] - out.hands_world[:, side, :1, 2]
        image[:, side, :, 2] = truth[stream] * dz + 0.01  # MediaPipe's wrist z is not exactly 0
    face_z = out.face_world[..., 2]
    landmarks = np.zeros((6, 478, 3))
    landmarks[..., 2] = truth["face"] * face_z + 0.2  # MediaPipe's face z has its own origin
    hands = SimpleNamespace(
        pose_image=pose_image,
        pose_visibility=np.ones((6, 33)),
        image=image,
        width=width,
        height=height,
    )
    face = SimpleNamespace(landmarks=landmarks)
    fitted = fit_clip_camera(clip, hands, skeleton, face=face)
    assert set(fitted.depth_scales) == set(truth) and fitted.depth_fallback == ()
    for stream, value in truth.items():
        assert fitted.depth_scales[stream] == pytest.approx(value, rel=1e-9), stream
    # x and y are fitted exactly as before
    plain = fit_clip_camera(clip, hands, skeleton)
    np.testing.assert_array_equal(fitted.offset, plain.offset)
    assert fitted.scale == plain.scale
    assert plain.depth_fallback == ("face",)
    # a side whose hand is never valid has nothing to fit and falls back
    hand_valid = clip.hand_valid.copy()
    hand_valid[:, 0] = False
    one_hand = fit_clip_camera(clip.replace(hand_valid=hand_valid), hands, skeleton, face=face)
    assert "left_hand" not in one_hand.depth_scales
    assert one_hand.depth_fallback == ("left_hand",)


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
        hand, wrists = hand[..., :2], o.pose_image[:, [15, 16], :2]
        return np.concatenate([hand.reshape(len(hand), 42, 2), wrists], axis=1)

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


def drift_clip(frames=12):
    """A synthetic clip whose camera offset drifts linearly: x by -0.09, y by +0.04 in total."""
    from tests.clip_fixtures import synthetic_clip

    clip = synthetic_clip(frames=frames)
    ramp = np.linspace(0.0, 1.0, frames)[:, None]
    offset = 0.5 + ramp * np.array([-0.09, 0.04])
    return clip.replace(camera=tier1.WeakCamera(clip.camera.scale, offset, 0.0))


def shoulder_centre_drift(clip, skeleton):
    out = build_outputs(clip, skeleton)
    track = out.pose_image[:, [11, 12]].mean(axis=1)
    return track - track.mean(axis=0)


@requires_smplx
def test_mirror_negates_image_drift_in_x():
    from tests.clip_fixtures import get_skeleton

    skeleton, clip = get_skeleton(), drift_clip()
    base = shoulder_centre_drift(clip, skeleton)
    mirrored = shoulder_centre_drift(
        Mirror().apply(clip, {"mirror": True}, make_ctx(None)), skeleton
    )
    assert base[-1, 0] < -0.04
    np.testing.assert_allclose(mirrored[:, 0], -base[:, 0], atol=1e-9)
    np.testing.assert_allclose(mirrored[:, 1], base[:, 1], atol=1e-9)


@requires_smplx
def test_scale_multiplies_image_drift():
    from tests.clip_fixtures import get_skeleton

    skeleton, clip = get_skeleton(), drift_clip()
    base = shoulder_centre_drift(clip, skeleton)
    scaled = Scale().apply(clip, {"factor": 1.2}, make_ctx(None))
    np.testing.assert_allclose(shoulder_centre_drift(scaled, skeleton), 1.2 * base, atol=1e-9)


@requires_smplx
def test_camera_drift_involutions():
    clip, ctx = drift_clip(), make_ctx(None)
    twice = Mirror().apply(Mirror().apply(clip, {"mirror": True}, ctx), {"mirror": True}, ctx)
    np.testing.assert_allclose(twice.scene.apply_camera(twice.camera).offset, clip.camera.offset)
    back = Scale().apply(Scale().apply(clip, {"factor": 1.3}, ctx), {"factor": 1 / 1.3}, ctx)
    np.testing.assert_allclose(
        back.scene.apply_camera(back.camera).offset, clip.camera.offset, atol=1e-12
    )
    assert clip.scene.apply_camera(clip.camera) is clip.camera


@requires_smplx
def test_uncentre_after_mirror_scale_and_drift_keeps_hands_in_frame():
    from tests.clip_fixtures import get_skeleton

    skeleton, clip, ctx = get_skeleton(), drift_clip(), make_ctx(None)
    clip = Scale().apply(Mirror().apply(clip, {"mirror": True}, ctx), {"factor": 1.25}, ctx)
    draw = {"u": [0.0, 1.0], "limit": 0.125}
    shifted = Uncentre().apply(clip, draw, ctx)
    before, after = build_outputs(clip, skeleton), build_outputs(shifted, skeleton)
    np.testing.assert_allclose(
        after.hands_image[..., :2] - before.hands_image[..., :2],
        np.broadcast_to(draw["image_shift"], before.hands_image[..., :2].shape),
        atol=1e-9,
    )
    # the shift is in the image plane: relative depth does not move
    np.testing.assert_allclose(after.hands_image[..., 2], before.hands_image[..., 2], atol=1e-12)
    uv_before = before.hands_image[..., :2].reshape(-1, 2)
    uv_after = after.hands_image[..., :2].reshape(-1, 2)
    was_in = tier1.in_frame(uv_before)
    assert was_in.any()
    assert tier1.in_frame(uv_after)[was_in].all()


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


@requires_smplx
def test_relative_z_agrees_with_mediapipe_on_the_pilot_clip():
    """The round trip's z has MediaPipe's sign: arms vs the hips, hand points vs the wrist."""
    from tests.clip_fixtures import get_skeleton, real_clip

    clip, hands, _ = real_clip()
    skeleton = get_skeleton()
    camera = fit_clip_camera(clip, hands, skeleton)
    assert camera.frame_size == (hands.width, hands.height)
    out = build_outputs(clip.replace(camera=camera), skeleton)
    mp_pose = np.asarray(hands.pose_image, dtype=np.float64)
    mp_hips = 0.5 * (mp_pose[:, 23, 2] + mp_pose[:, 24, 2])
    arms = list(range(11, 17))
    ours = out.pose_image[:, arms, 2]
    theirs = mp_pose[:, arms, 2] - mp_hips[:, None]
    assert _corr(ours.ravel(), theirs.ravel()) > 0.5
    mp_hand = np.asarray(hands.image, dtype=np.float64)[..., 2]
    valid = out.hand_valid[:, :, None]
    ours = np.where(valid, out.hands_image[..., 2], np.nan)
    theirs = np.where(valid, mp_hand - mp_hand[..., :1], np.nan)
    assert _corr(ours.ravel(), theirs.ravel()) > 0.3
    np.testing.assert_array_equal(
        out.pixels("hands"), out.hands_image * np.array([hands.width, hands.height, hands.width])
    )


def _magnitude_ratio(ours, theirs):
    ok = np.isfinite(ours) & np.isfinite(theirs)
    return float(np.median(np.abs(ours[ok])) / np.median(np.abs(theirs[ok])))


@requires_smplx
def test_fitted_z_magnitudes_match_mediapipe_on_the_pilot_clip():
    """Fitted per-stream z scales give MediaPipe-sized z: median |z| within 0.8-1.25."""
    from tests.clip_fixtures import get_skeleton, real_clip

    clip, hands, face = real_clip()
    skeleton = get_skeleton()
    camera = fit_clip_camera(clip, hands, skeleton, face=face)
    out = build_outputs(clip.replace(camera=camera), skeleton)
    assert "pose" in camera.depth_scales and "face" in camera.depth_scales

    mp_pose = np.asarray(hands.pose_image, dtype=np.float64)
    visible = np.asarray(hands.pose_visibility)[:, 11:17] >= 0.5
    theirs = mp_pose[:, 11:17, 2] - 0.5 * (mp_pose[:, 23, 2] + mp_pose[:, 24, 2])[:, None]
    ratio = _magnitude_ratio(out.pose_image[:, 11:17, 2], np.where(visible, theirs, np.nan))
    assert 0.8 <= ratio <= 1.25, ratio

    mp_hand = np.asarray(hands.image, dtype=np.float64)[..., 2]
    fitted_hands = 0
    for side, stream in enumerate(("left_hand", "right_hand")):
        valid = out.hand_valid[:, side, None]
        ours = np.where(valid, out.hands_image[:, side, 1:, 2], np.nan)
        theirs = np.where(valid, mp_hand[:, side, 1:] - mp_hand[:, side, :1], np.nan)
        if stream in camera.depth_fallback:
            assert not (np.isfinite(ours) & np.isfinite(theirs)).any(), stream
            continue
        fitted_hands += 1
        ratio = _magnitude_ratio(ours, theirs)
        assert 0.8 <= ratio <= 1.25, (stream, ratio)
    assert fitted_hands >= 1

    mp_face = np.asarray(face.landmarks, dtype=np.float64)[out.face_valid, :, 2]
    theirs = mp_face - np.nanmean(mp_face, axis=1, keepdims=True)
    ratio = _magnitude_ratio(out.face_image[out.face_valid, :, 2], theirs)
    assert 0.8 <= ratio <= 1.25, ratio
