"""
Week-4 scene augmentations: mirror, view rotation, scale and un-centring of the whole clip.

Each one composes into the clip's pending `SceneTransform` (`clip.scene`), which
`outputs.build_outputs` applies to every point set at the end. The draws come from the Tier 1
draw the chain hands over in `ChainContext.tier1_draw`; the `DEFAULT_RANGES` are what a profile
configures (the chain builds the Tier 1 ranges from them). The chain applies them in the order
mirror, view, scale, un-centre.

Composition keeps `SceneTransform.apply_points` exact. A new rotation/scale `R2 s2` (about the
scene's pivot) after `x -> R s F^m (x - c) + c + t` gives rotation `R2 R`, scale `s s2` and shift
`R2 s2 t`. A reflection `F = diag(-1, 1, 1)` after it gives rotation `F R F`, shift `F t` and the
mirror flag toggled. The pivot is `ctx.pivot` while the scene is still the identity and is kept
afterwards, so later steps compose about the same point.

`fit_clip_camera` fits the clip's weak camera from MediaPipe's image points, as the round-trip
demo does. `blendshape_mirror` is `outputs.blendshape_mirror`, re-exported here.
"""

import dataclasses
from typing import Any, Dict, Optional, Sequence

import numpy as np

from . import tier1
from .base import AugmentationRegistry, ClipAugmentation
from .blacklist import DEFAULT_BLACKLIST, blacklist_category, load_blacklist, resolve_path
from .clip import ChainContext, ClipData, SceneTransform
from .outputs import blendshape_mirror, build_outputs, face_centre
from .ranges import check_non_negative, check_range, check_unit

__all__ = [
    "Mirror",
    "View",
    "Scale",
    "Uncentre",
    "blendshape_mirror",
    "fit_clip_camera",
]

_F = np.diag([-1.0, 1.0, 1.0])
VISIBILITY = 0.5
ARM_MP = (11, 12, 13, 14, 15, 16)
WRIST_MP = (15, 16)
HIPS_MP = (23, 24)


def _draw_of(ctx: ChainContext) -> tier1.Tier1Draw:
    if ctx is None or ctx.tier1_draw is None:
        raise ValueError("scene augmentations need ChainContext.tier1_draw")
    return ctx.tier1_draw


def _pivot(clip: ClipData, ctx: ChainContext) -> np.ndarray:
    if clip.scene.is_identity():
        return np.array(ctx.pivot, dtype=np.float64)
    return np.asarray(clip.scene.pivot, dtype=np.float64)


def _compose_linear(clip: ClipData, matrix: np.ndarray, scale: float, ctx) -> ClipData:
    """The scene followed by a rotation `matrix` and uniform `scale` about the pivot."""
    scene = clip.scene
    new = SceneTransform(
        mirror=scene.mirror,
        rotation=matrix @ np.asarray(scene.rotation),
        scale=scene.scale * scale,
        shift=(matrix * scale) @ np.asarray(scene.shift),
        pivot=_pivot(clip, ctx),
    )
    return clip.replace(scene=new)


@AugmentationRegistry.register("scene.mirror")
class Mirror(ClipAugmentation):
    """Reflect the whole clip left/right about the pivot; sides are relabelled on output."""

    key = "mirror"
    regions = ("scene",)
    DEFAULT_RANGES = {"p": 0.5, "blacklist": DEFAULT_BLACKLIST}

    def validate_ranges(self) -> None:
        check_unit(self.ranges["p"], self.label("p"))
        path = self.ranges["blacklist"]
        if path is None:
            return
        if not isinstance(path, str):
            raise ValueError(
                f"{self.label('blacklist')} must be a path string or null, got {path!r}"
            )
        if not resolve_path(path).is_file():
            raise ValueError(f"{self.label('blacklist')} file not found: {path}")
        try:
            load_blacklist(path)
        except Exception as exc:  # malformed YAML or missing keys fail at load, not mid-run
            raise ValueError(
                f"{self.label('blacklist')} is not a valid blacklist: {exc!r}"
            ) from exc

    def sample(self, rng: np.random.Generator, clip: Any, ctx: ChainContext) -> Dict[str, Any]:
        """The shared Tier 1 mirror draw, forced off for a blacklisted gloss (draw unchanged)."""
        mirror = bool(_draw_of(ctx).mirror)
        path = self.ranges["blacklist"]
        gloss = getattr(clip, "gloss", None)
        category = blacklist_category(gloss, path) if path else None
        if category is None:
            if path and not gloss:  # unchecked: recorded so the coverage gap is auditable
                return {"mirror": mirror, "blacklist": "gloss_unknown"}
            return {"mirror": mirror}
        return {"mirror": False, "blocked": f"blacklist:{category}"}

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        if not draw["mirror"]:
            return target
        scene = target.scene
        new = SceneTransform(
            mirror=not scene.mirror,
            rotation=_F @ np.asarray(scene.rotation) @ _F,
            scale=scene.scale,
            shift=_F @ np.asarray(scene.shift),
            pivot=_pivot(target, ctx),
        )
        return target.replace(scene=new)

    def identity_draw(self) -> Dict[str, Any]:
        return {"mirror": False}


@AugmentationRegistry.register("scene.view")
class View(ClipAugmentation):
    """Rotate the clip about the pivot: yaw about the vertical axis, then pitch."""

    key = "view"
    regions = ("scene",)
    DEFAULT_RANGES = {"yaw_deg": 25.0, "pitch_deg": 6.0}

    def validate_ranges(self) -> None:
        # a scalar is the symmetric half-width: the draw is uniform in [-x, x], so x >= 0
        for k in ("yaw_deg", "pitch_deg"):
            check_non_negative(self.ranges[k], self.label(k))

    def sample(self, rng: np.random.Generator, clip: Any, ctx: ChainContext) -> Dict[str, Any]:
        tier = _draw_of(ctx)
        return {"yaw_deg": float(tier.yaw_deg), "pitch_deg": float(tier.pitch_deg)}

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        yaw, pitch = float(draw["yaw_deg"]), float(draw["pitch_deg"])
        if yaw == 0.0 and pitch == 0.0:
            return target
        return _compose_linear(target, tier1.view_rotation(yaw, pitch), 1.0, ctx)

    def identity_draw(self) -> Dict[str, Any]:
        return {"yaw_deg": 0.0, "pitch_deg": 0.0}


@AugmentationRegistry.register("scene.scale")
class Scale(ClipAugmentation):
    """Scale the clip about the pivot (the signer's apparent size)."""

    key = "scale"
    regions = ("scene",)
    DEFAULT_RANGES = {"factor": [0.75, 1.25]}

    def validate_ranges(self) -> None:
        check_range(self.ranges["factor"], self.label("factor"), low=0.0, low_open=True)

    def sample(self, rng: np.random.Generator, clip: Any, ctx: ChainContext) -> Dict[str, Any]:
        return {"factor": float(_draw_of(ctx).scale)}

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        factor = float(draw["factor"])
        if factor == 1.0:
            return target
        return _compose_linear(target, np.eye(3), factor, ctx)

    def identity_draw(self) -> Dict[str, Any]:
        return {"factor": 1.0}


@AugmentationRegistry.register("scene.uncentre")
class Uncentre(ClipAugmentation):
    """
    Shift the clip in the image, within +-`limit` of the frame, keeping both hands in frame.

    `apply` builds the clip's current outputs, projects the fitted hands' 21 points and pose
    wrists 15 and 16, picks the shift from `tier1.uncentre_shift` and converts it to a world
    translation with the clip's camera. The image shift is written to `draw["image_shift"]`.
    """

    key = "uncentre"
    regions = ("scene",)
    DEFAULT_RANGES = {"limit": 0.125}

    def validate_ranges(self) -> None:
        check_non_negative(self.ranges["limit"], self.label("limit"))

    def sample(self, rng: np.random.Generator, clip: Any, ctx: ChainContext) -> Dict[str, Any]:
        u = _draw_of(ctx).uncentre_u
        return {"u": [float(u[0]), float(u[1])], "limit": float(self.ranges["limit"])}

    def apply(self, target: ClipData, draw: Dict[str, Any], ctx: ChainContext) -> ClipData:
        if target.camera is None:
            raise ValueError("scene.uncentre needs the clip's camera (fit_clip_camera)")
        limit = float(draw["limit"])
        if limit == 0.0:
            draw["image_shift"] = [0.0, 0.0]
            return target
        outputs = build_outputs(target, ctx.skeleton)
        hands = np.where(outputs.hand_valid[:, :, None, None], outputs.hands_image, np.nan)
        hands, wrists = hands[..., :2], outputs.pose_image[:, list(WRIST_MP), :2]  # x, y only
        uv = np.concatenate([hands.reshape(len(hands), 42, 2), wrists], axis=1)
        shift = tier1.uncentre_shift(uv, draw["u"], limit)
        draw["image_shift"] = [float(shift[0]), float(shift[1])]
        offset = target.camera.world_shift(shift)
        if not offset.any():
            return target
        scene = target.scene
        new = SceneTransform(
            mirror=scene.mirror,
            rotation=scene.rotation,
            scale=scene.scale,
            shift=np.asarray(scene.shift, dtype=np.float64) + offset,
            pivot=_pivot(target, ctx),
        )
        return target.replace(scene=new)

    def identity_draw(self) -> Dict[str, Any]:
        return {"u": [0.5, 0.5], "limit": 0.0}


def fit_clip_camera(
    clip: ClipData, hands, skeleton, frames: Optional[Sequence[int]] = None, face=None
) -> tier1.WeakCamera:
    """
    The clip's weak-perspective camera, fitted from the round trip to MediaPipe's image points.

    World points are `build_outputs` of the clip with no scene transform: pose arms 11-16 and
    both hands' 42 points (hand points only where `hand_valid`). Image points are
    `hands.pose_image` (arms with visibility below 0.5 dropped) and `hands.image`, normalised
    to [0, 1]; `pixel_size=(hands.width, hands.height)` gives one scale for both axes and is
    recorded as the camera's `frame_size`.

    The camera also gets one depth scale per stream (`tier1.fit_depth_scale`), from the same
    world points to MediaPipe's own relative z, each against its own reference: the visible
    arms against the hip midpoint (`hip_centre_world`; MediaPipe's 23, 24), each valid hand's
    20 points against its wrist, and, with `face` (the `FaceClip`), the face points against
    their per-frame mean. A stream with too little observed z (or no `face`) is listed in
    `depth_fallback` and keeps the x-scale rule. x and y do not depend on any of this.

    `ClipData` does not carry its source frame indices, so `frames` (the `result.frames` that
    built the clip) picks the matching image rows of `hands` and `face`. It defaults to every
    row of `hands`, which must then have `clip.frames` of them, else ValueError.
    """
    frames = np.arange(len(hands.image)) if frames is None else np.asarray(frames, dtype=int)
    if len(frames) != clip.frames:
        raise ValueError(
            f"{len(frames)} image frames for a clip of {clip.frames}; pass `frames` "
            "(the run_clip result's frame indices)"
        )
    outputs = build_outputs(clip.replace(camera=None, scene=SceneTransform()), skeleton)
    n = clip.frames
    arm_world = outputs.pose_world[:, list(ARM_MP)]
    hand_world = np.where(outputs.hand_valid[:, :, None, None], outputs.hands_world, np.nan)
    world = np.concatenate([arm_world, hand_world.reshape(n, 42, 3)], axis=1)
    arm_image = np.asarray(hands.pose_image)[frames][:, list(ARM_MP), :2].astype(np.float64)
    visibility = np.asarray(hands.pose_visibility)[frames][:, list(ARM_MP)]
    arm_image[visibility < VISIBILITY] = np.nan
    hand_image = np.asarray(hands.image)[frames][..., :2].astype(np.float64).reshape(n, 42, 2)
    image = np.concatenate([arm_image, hand_image], axis=1)
    camera = tier1.fit_weak_camera(world, image, pixel_size=(hands.width, hands.height))
    scales = _fit_depth_scales(outputs, hands, face, frames, visibility >= VISIBILITY)
    fallback = tuple(s for s in tier1.DEPTH_STREAMS if s not in scales)
    return dataclasses.replace(camera, depth_scales=scales, depth_fallback=fallback)


def _fit_depth_scales(outputs, hands, face, frames, arm_visible) -> Dict[str, float]:
    """Per-stream depth scales from the unaugmented outputs to MediaPipe's z; fitted ones only."""
    pairs = {}
    pose_z = np.asarray(hands.pose_image, dtype=np.float64)[frames, :, 2]
    hips_z = pose_z[:, list(HIPS_MP)].mean(axis=1, keepdims=True)
    world = outputs.pose_world[:, list(ARM_MP), 2] - outputs.hip_centre_world[:, None, 2]
    pairs["pose"] = (world, np.where(arm_visible, pose_z[:, list(ARM_MP)] - hips_z, np.nan))
    hand_z = np.asarray(hands.image, dtype=np.float64)[frames][..., 2]
    for side, stream in enumerate(("left_hand", "right_hand")):
        valid = outputs.hand_valid[:, side, None]
        world = outputs.hands_world[:, side, 1:, 2] - outputs.hands_world[:, side, :1, 2]
        observed = hand_z[:, side, 1:] - hand_z[:, side, :1]
        pairs[stream] = (np.where(valid, world, np.nan), observed)
    if face is not None:
        face_z = np.asarray(face.landmarks, dtype=np.float64)[frames]
        valid = outputs.face_valid[:, None]
        world = outputs.face_world[..., 2] - face_centre(outputs.face_world)[..., 2]
        observed = face_z[..., 2] - face_centre(face_z)[..., 2]
        pairs["face"] = (np.where(valid, world, np.nan), observed)
    scales = {}
    for stream, (world, observed) in pairs.items():
        scale = tier1.fit_depth_scale(world, observed)
        if scale is not None:
            scales[stream] = scale
    return scales
