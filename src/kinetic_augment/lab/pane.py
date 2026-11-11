"""Run one augmentation for one Lab pane on a prepared clip and return its blob payload."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from kinetic_augment.augmentations.chain import AugmentationChain
from kinetic_augment.augmentations.clip import ClipData, clip_from_result
from kinetic_augment.augmentations.outputs import build_outputs
from kinetic_augment.augmentations.profile import Profile, profile_from_dict, resolved_ranges
from kinetic_augment.augmentations.report import measure
from kinetic_augment.augmentations.scene import fit_clip_camera
from kinetic_augment.lab.bundle import SCALE_I16, bundle_arrays, jsonable, source_frames
from kinetic_augment.lab.extreme import extreme_allowed, make_hook
from kinetic_augment.lab.runs import RUNS, merge

WHOLE_PREFIXES = ("combined.", "saved:")


@dataclass(frozen=True)
class PaneRequest:
    """One pane: an entry (or a whole profile) on one clip, with its knob values and mode."""

    clip_id: str
    entry_id: str
    values: Dict[str, Any] = field(default_factory=dict)
    mode: str = "random"
    reroll: int = 0
    edge: str = "high"
    profile: str = "default"

    def key(self) -> str:
        """Canonical JSON (sorted keys) for caching and share manifests."""
        return json.dumps(jsonable(dict(self.__dict__)), sort_keys=True, separators=(",", ":"))

    @property
    def whole(self) -> bool:
        return self.entry_id.startswith(WHOLE_PREFIXES)


@dataclass
class PreparedClip:
    clip_id: str
    clip: ClipData
    frames: np.ndarray
    base_meta: Dict[str, Any]
    base_arrays: Dict[str, Tuple[np.ndarray, str]]
    n_images: int = 0
    base_outputs: Any = None


def _set_dotted(ranges: Dict[str, Any], dotted: str, value: Any) -> None:
    *path, last = dotted.split(".")
    node = ranges
    for part in path:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[last] = value


def _to_dict(profile: Profile) -> Dict[str, Any]:
    regions: Dict[str, Any] = {}
    for e in profile.entries:
        regions.setdefault(e.region, {})[e.key] = {
            "enabled": e.enabled,
            "probability": e.probability,
            "ranges": copy.deepcopy(e.ranges),
        }
    return {
        "profile_name": profile.name,
        "seed": profile.seed,
        "schema_version": profile.schema_version,
        "regions": regions,
    }


def pane_profile(base: Profile, req: PaneRequest) -> Profile:
    """
    A copy of `base` for one pane: only the run's entry on, at probability 1, with its ranges.

    Ranges are the resolved ranges of `base` merged with the run's overrides and `req.values`
    (dotted keys such as `brows.sigma` set nested ranges). `seed = base.seed + req.reroll`.
    A whole-profile request (`combined.*`, `saved:*`) keeps `base`'s entries as they are.

    Raises:
        ValueError: unknown entry id, or a value the augmentation rejects.
    """
    data = _to_dict(base)
    data["seed"] = base.seed + req.reroll
    if not req.whole:
        run = RUNS.get(req.entry_id)
        if run is None or run.entry is None:
            raise ValueError(f"unknown lab entry {req.entry_id!r}")
        if base.entry(run.entry) is None:
            raise ValueError(f"profile {base.name!r} has no entry {run.entry!r}")
        ranges = merge(copy.deepcopy(resolved_ranges(base, run.entry)), run.overrides)
        for dotted, value in req.values.items():
            _set_dotted(ranges, dotted, value)
        for region in data["regions"].values():
            for body in region.values():
                body["enabled"] = False
        region, key = run.entry.split(".", 1)
        data["regions"][region][key] = {"enabled": True, "probability": 1.0, "ranges": ranges}
    return profile_from_dict(data, source=base.name)


def prepare_clip(clip_id: str, result, hands, face, models) -> PreparedClip:
    """Turn a `run_clip` result into a clip with its fitted camera and the unaugmented outputs."""
    skeleton, mp_indices = models.skeleton, np.asarray(models.corr.mp_indices, dtype=int)
    clip = clip_from_result(clip_id, result, hands, face, skeleton)
    camera = fit_clip_camera(clip, hands, skeleton, frames=result.frames)
    clip = clip.replace(camera=camera)
    base_full = build_outputs(clip, skeleton, keep_vertices=True)
    base_out = build_outputs(clip, skeleton)
    frames = np.asarray(result.frames, dtype=np.float64)
    arrays, ok = bundle_arrays(base_full, base_full.vertices, frames, mp_indices)
    arrays = {k: v for k, v in arrays.items() if not k.startswith("img_")}
    arrays["mesh_faces"] = (np.asarray(skeleton.model.faces), "uint16")
    meta = {
        "clip": clip_id,
        "n": clip.frames,
        "fps": clip.fps,
        "scale_i16": SCALE_I16,
        "body_ok": ok.astype(int).tolist(),
        "camera": {"scale_per_m": list(camera.scale), "residual_frame_pct": 100 * camera.residual},
    }
    return PreparedClip(
        clip_id, clip, frames, meta, arrays, int(clip.frames), base_outputs=base_out
    )


def run_pane(
    prep: PreparedClip, req: PaneRequest, base: Profile, models
) -> Tuple[Dict[str, Any], Dict[str, Tuple[np.ndarray, str]]]:
    """
    Run `req` on `prep` and return `(meta, arrays)` for `bundle.encode_pane`.

    For `combined.*` / `saved:*` ids `base` is the whole profile to run (the caller resolves the
    id to it). Extreme mode needs an entry with an extreme switch (`extreme_allowed`).

    Raises:
        ValueError: bad values, unknown entry, or extreme requested where there is no extreme.
    """
    if req.mode not in ("random", "extreme"):
        raise ValueError(f"mode must be 'random' or 'extreme', got {req.mode!r}")
    run = None if req.whole else RUNS.get(req.entry_id)
    if req.mode == "extreme" and (req.whole or not extreme_allowed(req.entry_id)):
        raise ValueError("no extreme for this one")
    profile = pane_profile(base, req)
    skeleton = models.skeleton
    mp_indices = np.asarray(models.corr.mp_indices, dtype=int)
    clip = prep.clip
    hook = make_hook(req.edge, profile, clip=clip) if req.mode == "extreme" else None
    chain = AugmentationChain(profile, skeleton, draw_hook=hook)
    res = chain.run(clip)
    full = build_outputs(res.clip, skeleton, keep_vertices=True)
    outputs = res.outputs  # hand dropout edits these after the clip stages
    times = prep.frames[0] + source_frames(chain, res, clip.frames)
    temporal = any(a.startswith("temporal.time_warp") for a in res.applied) or any(
        a.startswith("temporal.frame_drop") for a in res.applied
    )
    # A mirror swaps the sides, so frame-by-frame wrist / bone measures need an unmirrored run.
    aligned = res.clip.frames == clip.frames and not temporal and not res.clip.scene.mirror
    metrics = measure(prep.base_outputs, outputs, aligned)
    arrays, ok = bundle_arrays(outputs, full.vertices, times, mp_indices)
    arrays = {k: v for k, v in arrays.items() if not k.startswith("img_")}
    if run is not None:
        ranges = {run.entry: resolved_ranges(profile, run.entry)}
    else:
        ranges = {e.name: resolved_ranges(profile, e.name) for e in profile.entries if e.enabled}
    meta = {
        "clip": prep.clip_id,
        "entry": req.entry_id,
        "profile": profile.name,
        "n": res.clip.frames,
        "fps": res.clip.fps,
        "mirrored": bool(res.clip.scene.mirror),
        "scale_i16": SCALE_I16,
        "body_ok": ok.astype(int).tolist(),
        "draws": res.draws,
        "applied": res.applied,
        "fallbacks": res.fallbacks,
        "ranges": ranges,
        "metrics": metrics,
        "reroll": req.reroll,
        "mode": req.mode,
        "edge": req.edge,
    }
    return meta, arrays
