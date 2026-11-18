"""
Mirror blacklist: glosses whose sign changes under a left/right flip, and the clip -> gloss map.

`configs/mirror_blacklist.yaml` is authoritative (the config the user edits).
`data/wlasl/clip_glosses.csv` supplies the clip id -> gloss mapping (the CSV's own `mirror_safe`
column is not consulted). Both load once per path and are cached. Relative paths resolve against
the repository root, so they work from any working directory.
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BLACKLIST = "configs/mirror_blacklist.yaml"
CLIP_GLOSSES = "data/wlasl/clip_glosses.csv"

__all__ = [
    "DEFAULT_BLACKLIST",
    "blacklist_category",
    "gloss_for_clip",
    "load_blacklist",
    "resolve_path",
]


def resolve_path(path: Union[str, Path]) -> Path:
    """`path` as given if absolute, else relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_blacklist(path: Union[str, Path]) -> Dict[str, str]:
    """Return {gloss (lower case): category} for every gloss on the blacklist."""
    return dict(_load_blacklist(resolve_path(path)))


@lru_cache(maxsize=None)
def _load_blacklist(path: Path) -> Dict[str, str]:
    config = yaml.safe_load(path.read_text())
    return {
        gloss.lower(): category
        for category, entry in config["categories"].items()
        for gloss in entry["glosses"]
    }


def blacklist_category(gloss: Optional[str], path: Union[str, Path]) -> Optional[str]:
    """The blacklist category of `gloss` (case-insensitive), or None if allowed or unknown."""
    if not gloss:
        return None
    return _load_blacklist(resolve_path(path)).get(gloss.lower())


@lru_cache(maxsize=None)
def _clip_glosses(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["clip_id"]: row["gloss"] for row in csv.DictReader(f)}


def gloss_for_clip(clip_id: str, path: Union[str, Path] = CLIP_GLOSSES) -> Optional[str]:
    """The WLASL gloss of `clip_id` (e.g. "01991"), or None when the id is not in the map."""
    return _clip_glosses(resolve_path(path)).get(str(clip_id))
