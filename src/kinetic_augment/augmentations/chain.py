"""
The week-4 augmentation chain: every enabled profile entry, in one fixed order.

`AugmentationChain.run(clip)` draws the shared Tier 1 values once (the stream
`default_rng([seed, clip id])`, as `tier1.sample_tier1` does elsewhere), then walks `ORDER`. Each
enabled entry samples and applies on the clip as it stands at that point, from its own stream
`ranges.clip_rng(seed, clip_id, "region.key")`, so adding or removing one entry never changes
another's draw. The first number of that stream is the probability gate (`< probability` keeps
the entry); a gated-out entry records `{"gated": True}` in `draws` and is not applied. The scene
entries read their values from the Tier 1 draw and count as applied only when that value is not
the identity. `scene.uncentre` needs the clip's camera: without one it is skipped and recorded in
`fallbacks` as "no_camera". After the clip stages, `build_outputs` runs and the
`stage == "outputs"` entries (hand dropout) edit those outputs.

An optional `draw_hook(name, draw, aug) -> dict` runs on each entry that passes the gate, after
`sample` and before `apply`; the dict it returns is applied and recorded (the Augmentation Lab
uses it to pin or widen draws). Without a hook nothing changes.

Every draw dict is recorded after `apply`, so it holds the diagnostics `apply` added. A
`"fallback"` in it is also copied to `ChainResult.fallbacks`. Each clip-stage entry that keeps
the frame count (every one but the temporal ones) also gets `"wrist_moved_mm_max"`: the largest
movement of either wrist over the fitted frames, in mm, from forward kinematics before and after
it (body frame, before the scene transform, so the scene entries record 0.0).
"""

import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import AugmentationRegistry
from .body import wrist_world
from .clip import ChainContext, ClipData
from .outputs import ClipOutputs, build_outputs
from .profile import Profile, tier1_ranges
from .ranges import clip_rng
from .tier1 import Tier1Draw, sample_tier1

ORDER: Tuple[str, ...] = (
    "fingers.joint_coupled_noise",
    "fingers.handshape_jitter",
    "body.tier1_arm_joints",
    "body.joint_coupled_noise",
    "body.limb_length_scaling",
    "body.movement_size",
    "body.location_jitter",
    "face.expression_jitter",
    "scene.mirror",
    "scene.view",
    "scene.scale",
    "scene.uncentre",
    "temporal.time_warp",
    "temporal.frame_drop",
    "temporal.hand_dropout",
)

_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12


def _wrist_moved_mm_max(before: np.ndarray, after: np.ndarray) -> float:
    """Largest per-frame wrist movement between two (T, 2, 3) `wrist_world` arrays, in mm."""
    move = np.linalg.norm(np.asarray(after) - np.asarray(before), axis=-1)
    move = move[np.isfinite(move)]
    return float(move.max() * 1000.0) if move.size else 0.0


@dataclass
class ChainResult:
    clip: ClipData
    outputs: ClipOutputs
    draws: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    fallbacks: Dict[str, str] = field(default_factory=dict)


def _non_identity(name: str, draw: Dict[str, Any]) -> bool:
    """Whether a draw changed anything; only the scene entries and Tier 1 arm joints can be 0."""
    if name == "scene.mirror":
        return bool(draw["mirror"])
    if name == "scene.view":
        return bool(draw["yaw_deg"] or draw["pitch_deg"])
    if name == "scene.scale":
        return float(draw["factor"]) != 1.0
    if name == "scene.uncentre":
        return any(draw.get("image_shift", [0.0, 0.0]))
    if name == "body.tier1_arm_joints":
        return bool(draw["joints"])
    return True


class AugmentationChain:
    """
    Run a profile's enabled entries on clips, in `ORDER` (see the module docstring).

    Set `clip.camera` first, with `scene.fit_clip_camera(clip, hands, skeleton,
    frames=result.frames)`: without a camera `scene.uncentre` is skipped (fallback "no_camera")
    and the image-space outputs are None.
    """

    def __init__(
        self,
        profile: Profile,
        skeleton,
        draw_hook: Optional[Callable[[str, Dict[str, Any], Any], Dict[str, Any]]] = None,
    ):
        self.profile = profile
        self.skeleton = skeleton
        self.draw_hook = draw_hook
        self.augmentations = {
            e.name: AugmentationRegistry.get(e.name)(ranges=e.ranges, probability=e.probability)
            for e in profile.entries
            if e.enabled
        }

    def tier1_draw_for(self, clip: ClipData) -> Tier1Draw:
        """The shared Tier 1 draw for `clip`: independent of which entries are enabled."""
        cid = clip.clip_id
        key = int(cid) if cid.isdigit() else zlib.crc32(cid.encode())
        rng = np.random.default_rng([self.profile.seed, key])
        return sample_tier1(rng, tier1_ranges(self.profile))

    def _pivot(self, clip: ClipData) -> np.ndarray:
        pose = build_outputs(clip, self.skeleton).pose_world
        mid = (pose[:, _LEFT_SHOULDER] + pose[:, _RIGHT_SHOULDER]) / 2.0
        mid = mid[np.isfinite(mid).all(axis=1)]
        return np.median(mid, axis=0) if len(mid) else np.zeros(3)

    def run(self, clip: ClipData) -> ChainResult:
        seed = self.profile.seed
        ctx = ChainContext(
            skeleton=self.skeleton,
            parents=np.asarray(self.skeleton.parents),
            pivot=self._pivot(clip),
            tier1_draw=self.tier1_draw_for(clip),
            seed=seed,
        )
        draws: Dict[str, Dict[str, Any]] = {}
        applied: List[str] = []
        fallbacks: Dict[str, str] = {}
        target: Any = clip
        wrists: Dict[str, Any] = {"clip": None, "world": None}

        def wrists_of(c: ClipData) -> np.ndarray:
            if wrists["clip"] is not c:  # cached: the next step's "before" is this one's "after"
                wrists["clip"], wrists["world"] = c, wrist_world(c, ctx.parents)
            return wrists["world"]

        def step(name: str, target: Any) -> Any:
            aug = self.augmentations[name]
            if name == "scene.uncentre" and getattr(target, "camera", True) is None:
                fallbacks[name] = "no_camera"
                return target
            rng = clip_rng(seed, clip.clip_id, name)
            if not rng.random() < aug.probability:
                draws[name] = {"gated": True}
                return target
            draw = aug.sample(rng, target, ctx)
            if self.draw_hook is not None:
                draw = self.draw_hook(name, draw, aug)
            measure = aug.stage == "clip" and "temporal" not in aug.regions
            before = wrists_of(target) if measure else None
            out = aug.apply(target, draw, ctx)
            if measure and out.frames == target.frames:
                draw["wrist_moved_mm_max"] = _wrist_moved_mm_max(before, wrists_of(out))
            draws[name] = draw
            if draw.get("fallback"):
                fallbacks[name] = str(draw["fallback"])
            if _non_identity(name, draw):
                applied.append(name)
            return out

        for name in ORDER:
            if name in self.augmentations and self.augmentations[name].stage == "clip":
                target = step(name, target)
        outputs = build_outputs(target, self.skeleton)
        for name in ORDER:
            if name in self.augmentations and self.augmentations[name].stage == "outputs":
                outputs = step(name, outputs)
        return ChainResult(target, outputs, draws, applied, fallbacks)
