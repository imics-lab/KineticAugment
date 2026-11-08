"""
Week-4 YAML profiles: the loader and the Tier 1 range mapping.

A profile file (see `configs/slr_week4_*.yaml`) holds `profile_name`, `seed`, `schema_version: 1`
and `regions: {region: {key: {enabled, probability, ranges}}}`. `load_profile` checks every entry
against the registered augmentation, so an unknown region, key or range key is a `ValueError`, not
something silently ignored. A range key a profile leaves out takes the augmentation's code default
and the loader prints it. `enable` / `disable` override single entries (`"region.key"`), which is
how one ablation arm is made from the default profile without a file of its own.

`tier1_ranges` turns the scene entries, the uniform duration of `temporal.time_warp` and
`body.tier1_arm_joints` into the `tier1.Tier1Ranges` the chain samples its shared draw from.
`body.tier1_arm_joints` has `shoulder_wrist_deg` and `elbow_deg`, but `Tier1Ranges` has one
`arm_joint_deg`: it takes `shoulder_wrist_deg` and the separate elbow value is ignored in the
Tier 1 draw.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from . import body, face, fingers, scene, timing  # noqa: F401  (registers the augmentations)
from .base import REGIONS, AugmentationRegistry
from .ranges import parse_range
from .tier1 import TIER1, Tier1Ranges

SCHEMA_VERSION = 1
ARM_NOISES = ("body.joint_coupled_noise", "body.tier1_arm_joints")


@dataclass
class ProfileEntry:
    region: str
    key: str
    enabled: bool
    probability: float
    ranges: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.region}.{self.key}"


@dataclass
class Profile:
    name: str
    seed: int
    schema_version: int
    entries: List[ProfileEntry] = field(default_factory=list)

    def entry(self, name: str) -> Optional[ProfileEntry]:
        for e in self.entries:
            if e.name == name:
                return e
        return None


def _parse_entries(regions: Any) -> List[ProfileEntry]:
    if not isinstance(regions, dict):
        raise ValueError("'regions' must be a mapping of region name to augmentations")
    entries: List[ProfileEntry] = []
    for region, keys in regions.items():
        if region not in REGIONS:
            raise ValueError(f"Unknown region {region!r}; expected one of {REGIONS}")
        for key, body_ in (keys or {}).items():
            name = f"{region}.{key}"
            cls = AugmentationRegistry.get(name)
            if cls is None:
                raise ValueError(f"Unknown augmentation {key!r} in region {region!r}")
            body_ = body_ or {}
            ranges = dict(body_.get("ranges") or {})
            probability = float(body_.get("probability", 1.0))
            # Building the instance validates the range keys (and their values) right now.
            instance = cls(ranges=ranges, probability=probability)
            for k in instance.defaulted:
                print(
                    f"[profile] {name}: range {k!r} not given, using default {instance.ranges[k]!r}"
                )
            entries.append(
                ProfileEntry(region, key, bool(body_.get("enabled", False)), probability, ranges)
            )
    return entries


def load_profile(
    path: Union[str, Path], enable: Sequence[str] = (), disable: Sequence[str] = ()
) -> Profile:
    """
    Read a profile file, apply the `enable` / `disable` overrides and validate it.

    Raises:
        ValueError: schema_version is not 1; an unknown region, key or range key; an override
            that names an entry the profile does not have; or both arm-noise entries enabled.
    """
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {data.get('schema_version')!r}"
        )
    profile = Profile(
        name=str(data.get("profile_name", Path(path).stem)),
        seed=int(data.get("seed", 0)),
        schema_version=SCHEMA_VERSION,
        entries=_parse_entries(data.get("regions")),
    )
    for names, value in ((enable, True), (disable, False)):
        for name in names:
            entry = profile.entry(name)
            if entry is None:
                raise ValueError(
                    f"Override names {name!r}, which is not an entry of {profile.name}"
                )
            entry.enabled = value
    on = [n for n in ARM_NOISES if (profile.entry(n) is not None and profile.entry(n).enabled)]
    if len(on) == len(ARM_NOISES):
        raise ValueError(f"{ARM_NOISES[0]} and {ARM_NOISES[1]} cannot both be enabled")
    return profile


def resolved_ranges(profile: Profile, name: str) -> Optional[Dict[str, Any]]:
    """The entry's ranges over the augmentation's code defaults, or None without the entry."""
    entry = profile.entry(name)
    if entry is None:
        return None
    merged = dict(AugmentationRegistry.get(name).DEFAULT_RANGES)
    merged.update(entry.ranges)
    return merged


def tier1_ranges(profile: Profile) -> Tier1Ranges:
    """
    The `Tier1Ranges` the profile's scene, duration and arm-joint entries describe.

    A field whose entry is missing keeps `tier1.TIER1`'s value. `arm_joint_deg` is the entry's
    `shoulder_wrist_deg`; its `elbow_deg` is not used by the Tier 1 draw.
    """
    fields: Dict[str, Any] = {}
    r = resolved_ranges(profile, "scene.mirror")
    if r is not None:
        fields["mirror_p"] = float(r["p"])
    r = resolved_ranges(profile, "scene.view")
    if r is not None:
        fields["yaw_deg"] = float(r["yaw_deg"])
        fields["pitch_deg"] = float(r["pitch_deg"])
    r = resolved_ranges(profile, "scene.scale")
    if r is not None:
        spec = parse_range(r["factor"])
        fields["scale"] = (spec.lo, spec.hi)
    r = resolved_ranges(profile, "scene.uncentre")
    if r is not None:
        fields["uncentre"] = float(r["limit"])
    r = resolved_ranges(profile, "temporal.time_warp")
    if r is not None:
        spec = parse_range(r["duration"], "log_uniform")
        fields["duration"] = (spec.lo, spec.hi)
    r = resolved_ranges(profile, "body.tier1_arm_joints")
    if r is not None:
        fields["arm_joint_p"] = float(r["p"])
        fields["arm_joint_deg"] = float(r["shoulder_wrist_deg"])
    return Tier1Ranges(**{**TIER1.__dict__, **fields}) if fields else TIER1
