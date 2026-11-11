"""The week-4 run table and the helper that turns a run into an augmentation profile."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Run:
    """One exported week-4 run: an entry alone (`entry`) or a whole profile (`entry` None)."""

    group: str
    label: str
    description: str
    entry: Optional[str] = None
    profile: str = "ambitious"
    overrides: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


RUNS: Dict[str, Run] = {
    "body.joint_coupled_noise": Run(
        "body",
        "arm joint noise",
        "shoulder, elbow and wrist of both arms turned by one coupled draw per clip; "
        "shoulder/wrist σ 2.5° (≤ 5°), elbow σ 4.75° (≤ 9.5°)",
        "body.joint_coupled_noise",
    ),
    "body.limb_length_scaling": Run(
        "body",
        "limb length",
        "upper arm and forearm lengths scaled 0.93–1.07; the arm is re-solved so each wrist "
        "stays where it was",
        "body.limb_length_scaling",
    ),
    "body.movement_size": Run(
        "body",
        "movement size",
        "each wrist's path scaled about its own centre, 0.8–1.2 (log-uniform); the arm is "
        "re-solved to follow it",
        "body.movement_size",
    ),
    "body.location_jitter": Run(
        "body",
        "location jitter",
        "the whole signing space moved by one 3-D offset in the torso frame, σ 12.5 mm, "
        "≤ 25 mm; the arms are re-solved to follow",
        "body.location_jitter",
    ),
    "fingers.joint_coupled_noise": Run(
        "fingers",
        "finger joint noise",
        "the knuckle (MCP) and middle (PIP) bend of each finger changed by one coupled draw, "
        "σ 3.75°, ≤ 7.5°; the end joint (DIP) follows the PIP",
        "fingers.joint_coupled_noise",
    ),
    "fingers.handshape_jitter": Run(
        "fingers",
        "handshape jitter",
        "per-joint finger jitter that keeps the handshape class: non-selected fingers and thumb "
        "σ 7.5°, selected σ 3.75°, selected flexion σ 2°, all ≤ 2σ",
        "fingers.handshape_jitter",
    ),
    "face.expression_jitter": Run(
        "face",
        "expression jitter",
        "brow, eye and cheek/nose blendshapes offset once per clip (σ 0.0625, 0.0625, 0.0375) "
        "and the face points moved to match; mouth and jaw stay off",
        "face.expression_jitter",
    ),
    "temporal.time_warp.uniform": Run(
        "temporal",
        "time warp: uniform",
        "the whole clip played faster or slower by one factor, 0.6–1.65× (log-uniform)",
        "temporal.time_warp",
        overrides={"mode_weights": {"none": 1, "varied": 0, "segment": 0}},
    ),
    "temporal.time_warp.varied": Run(
        "temporal",
        "time warp: varied",
        "the speed changes smoothly over the clip through 4 knots of 0.7–1.43× each, on top of "
        "the uniform factor; the last 20% is left unwarped",
        "temporal.time_warp",
        overrides={"mode_weights": {"none": 0, "varied": 1, "segment": 0}},
    ),
    "temporal.time_warp.segment": Run(
        "temporal",
        "time warp: segment",
        "one window of 20–60% of the clip slowed or sped up by 0.6–1.65×, on top of the uniform "
        "factor",
        "temporal.time_warp",
        overrides={"mode_weights": {"none": 0, "varied": 0, "segment": 1}},
    ),
    "temporal.frame_drop.rate": Run(
        "temporal",
        "frame drop: rate",
        "frames kept on a lower frame-rate grid of 12–25 fps (never below 8 fps), no gap over "
        "125 ms",
        "temporal.frame_drop",
        overrides={"mode_weights": {"rate": 1, "random": 0}},
    ),
    "temporal.frame_drop.random": Run(
        "temporal",
        "frame drop: random",
        "frames dropped at random with p 0–0.2 per frame, no gap over 125 ms",
        "temporal.frame_drop",
        overrides={"mode_weights": {"rate": 0, "random": 1}},
    ),
    "temporal.hand_dropout": Run(
        "temporal",
        "hand dropout",
        "runs of 1–10 frames where a hand goes missing, as when MediaPipe loses it; start rate "
        "0–0.05 per frame",
        "temporal.hand_dropout",
        overrides={"start_rate": [0.05, 0.05]},
        note="This demo uses the top of the start-rate range (0.05), so that a run shows.",
    ),
    "scene.mirror": Run(
        "scene",
        "mirror (week 4)",
        "left/right reflection with the sides swapped, p 0.5",
        "scene.mirror",
        overrides={"p": 1.0},
        note="This demo always mirrors (p = 1), so that the mirror shows.",
    ),
    "scene.view": Run(
        "scene",
        "view (week 4)",
        "yaw ±25° about the vertical axis through the shoulders, pitch ±6°",
        "scene.view",
    ),
    "scene.scale": Run(
        "scene",
        "scale (week 4)",
        "the whole scene about the shoulders, 0.75–1.25 (camera distance)",
        "scene.scale",
    ),
    "scene.uncentre": Run(
        "scene",
        "uncentre (week 4)",
        "the whole scene shifted up to ±0.125 of the frame, hands kept in it",
        "scene.uncentre",
    ),
    "combined.default": Run(
        "combined",
        "default profile",
        "slr_week4_default.yaml as one sample: mirror, view, scale, uncentre, uniform duration, "
        "arm and finger joint noise, random frame drop",
        profile="default",
    ),
    "combined.ambitious": Run(
        "combined",
        "ambitious profile",
        "slr_week4_ambitious.yaml as one sample: every entry on at the used ranges",
        profile="ambitious",
    ),
    "combined.literature": Run(
        "combined",
        "literature profile",
        "slr_week4_literature.yaml as one sample: the default entries at the proven ranges",
        profile="literature",
    ),
}


def merge(base: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def run_profile(run: Run, profiles):
    """The profile a run executes: a whole profile, or a copy with only `run.entry` on."""
    profile = copy.deepcopy(profiles[run.profile])
    if run.entry is None:
        return profile
    for entry in profile.entries:
        entry.enabled = entry.name == run.entry
        if entry.enabled:
            entry.probability = 1.0
            entry.ranges = merge(entry.ranges, run.overrides)
    return profile
