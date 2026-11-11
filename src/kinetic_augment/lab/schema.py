"""
The Lab's knob schema: the hand-written plain-English text in `knobs.yaml` merged with a profile.

`knobs.yaml` is keyed by lab entry id (the `RUNS` keys that have an entry). Each entry has a
title, an explanation (`does`, `changes`, `keeps`, `proven`, `used`, `note`), an `extreme` flag
and its `knobs`. `build_schema(profile)` adds, per entry, the run's overrides and each knob's
`default`: the profile's resolved range merged with the run's overrides.

A knob is `{key, kind, plain, technical, unit, proven, limits, step, share, active, default}` plus
`options` and `multi` for an enum and `integer` for a whole-number scalar. `key` is a range key,
dotted for nested ones (`brows.sigma`). `proven` is the literature range or None, `limits` bound
the slider, `share` marks the knobs that get stops in share mode and `active` is False for a knob
the run's mode does not use (for instance `segment_width` in the uniform time warp).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..augmentations.base import AugmentationRegistry
from ..augmentations.profile import Profile, resolved_ranges
from .runs import RUNS, merge

PROFILES = ["default", "ambitious", "literature"]
_KNOBS_PATH = Path(__file__).with_name("knobs.yaml")
_cache: Dict[str, Any] = {}


def load_knobs() -> Dict[str, Any]:
    """The parsed `knobs.yaml`: lab entry id -> its text, flag and knob list."""
    if "knobs" not in _cache:
        with open(_KNOBS_PATH, "r", encoding="utf-8") as handle:
            _cache["knobs"] = yaml.safe_load(handle)
    return _cache["knobs"]


def _dig(ranges: Dict[str, Any], dotted: str) -> Any:
    node: Any = ranges
    for part in dotted.split("."):
        node = node[part]
    return list(node) if isinstance(node, (list, tuple)) else node


def _entry_ranges(profile: Profile, chain_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    ranges = resolved_ranges(profile, chain_name)
    if ranges is None:  # the profile does not list the entry: the augmentation's code defaults
        ranges = dict(AugmentationRegistry.get(chain_name).DEFAULT_RANGES)
    return merge(ranges, overrides)


def build_schema(profile: Profile) -> Dict[str, Any]:
    """The schema the browser UI renders for `profile`: every lab entry with each knob's default."""
    knobs = load_knobs()
    entries: List[Dict[str, Any]] = []
    for lab_id, run in RUNS.items():
        if run.entry is None:
            continue
        text = knobs[lab_id]
        ranges = _entry_ranges(profile, run.entry, run.overrides)
        entry = {
            "id": lab_id,
            "entry": run.entry,
            "group": run.group,
            "title": text["title"],
            "does": text["does"],
            "changes": list(text["changes"]),
            "keeps": list(text["keeps"]),
            "proven": text["proven"],
            "used": text["used"],
            "note": text.get("note", ""),
            "extreme": bool(text["extreme"]),
            "overrides": dict(run.overrides),
            "knobs": [dict(k, default=_dig(ranges, k["key"])) for k in text["knobs"]],
        }
        entries.append(entry)
    return {
        "profile": profile.name,
        "seed": profile.seed,
        "entries": entries,
        "profiles": list(PROFILES),
    }
