"""Saved Lab settings: profile YAML files kept in `configs/viewer_saved/`."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from kinetic_augment.augmentations.profile import Profile, profile_from_dict

DEFAULT_FOLDER = Path(__file__).resolve().parents[3] / "configs" / "viewer_saved"
INDEX_HEADER = (
    "# Saved lab settings index\n\n- `index.md` — Index of this folder.\n\n"
    "Saved Lab settings; each file is a full profile `load_profile` reads.\n\n"
)


def slugify(name: str) -> str:
    """Lowercase `[a-z0-9_-]`, other runs collapsed to one `-`, at most 60 characters."""
    slug = re.sub(r"[^a-z0-9_]+", "-", str(name).lower()).strip("-")[:60].strip("-")
    if not slug:
        raise ValueError("name must contain at least one letter or digit")
    return slug


def validate_profile_text(text: str) -> Profile:
    """Parse YAML text as a profile; raises ValueError for bad YAML or an invalid profile."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"not valid YAML: {exc}") from exc
    return profile_from_dict(data if data is not None else {}, source="saved")


def save_setting(text: str, name: str, folder: Path, overwrite: bool) -> Path:
    """Write `folder/<slug>.yaml` and list it in `folder/index.md`; the text must be valid."""
    validate_profile_text(text)
    folder = Path(folder)
    path = folder / f"{slugify(name)}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    index = folder / "index.md"
    body = index.read_text() if index.exists() else INDEX_HEADER
    if f"- `{path.name}`" not in body:
        body = body if body.endswith("\n") else body + "\n"
        body += f"- `{path.name}` — Saved lab setting.\n"
    index.write_text(body)
    return path
