"""Saved Lab settings: names, validation and the folder index."""

from pathlib import Path

import pytest

from kinetic_augment.lab.saving import save_setting, slugify, validate_profile_text

REPO = Path(__file__).resolve().parents[1]
GOOD = (REPO / "configs" / "slr_week4_default.yaml").read_text()


def test_slugify():
    assert slugify("My run / v2") == "my-run-v2"
    assert slugify("A_b-C") == "a_b-c"
    assert len(slugify("x" * 200)) == 60


@pytest.mark.parametrize("bad", ["", "   ", "///", "é"])
def test_slugify_rejects_empty(bad):
    with pytest.raises(ValueError):
        slugify(bad)


def test_validate_profile_text():
    assert validate_profile_text(GOOD).name
    with pytest.raises(ValueError):
        validate_profile_text("schema_version: 1\n")  # no seed
    with pytest.raises(ValueError):
        validate_profile_text("a: [unclosed")
    with pytest.raises(ValueError):
        validate_profile_text("just a string")


def test_save_writes_file_and_index(tmp_path):
    path = save_setting(GOOD, "Run One", tmp_path / "saved", overwrite=False)
    assert path == tmp_path / "saved" / "run-one.yaml"
    assert path.read_text() == GOOD
    index = (tmp_path / "saved" / "index.md").read_text()
    assert index.count("`run-one.yaml`") == 1
    with pytest.raises(FileExistsError):
        save_setting(GOOD, "run one", tmp_path / "saved", overwrite=False)
    save_setting(GOOD, "run one", tmp_path / "saved", overwrite=True)
    assert (tmp_path / "saved" / "index.md").read_text().count("`run-one.yaml`") == 1
    save_setting(GOOD, "two", tmp_path / "saved", overwrite=False)
    assert "`two.yaml`" in (tmp_path / "saved" / "index.md").read_text()


def test_save_invalid_writes_nothing(tmp_path):
    with pytest.raises(ValueError):
        save_setting("seed: nope\nbogus: 1\n", "x", tmp_path / "saved", overwrite=False)
    assert not (tmp_path / "saved").exists()
