"""The Lab's files are indexed, the setup page is a plain classic page, and the docs exist."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "src" / "kinetic_augment" / "lab"
FOLDERS = [LAB, LAB / "static", LAB / "static" / "vendor"]


def test_every_lab_file_is_indexed():
    for folder in FOLDERS:
        index = (folder / "index.md").read_text()
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.name not in ("__pycache__",) and path.suffix != ".pyc":
                assert f"`{path.name}`" in index, f"{path.relative_to(REPO)} missing from index.md"
        for sub in sorted(p for p in folder.iterdir() if p.is_dir() and p.name != "__pycache__"):
            assert f"`{sub.name}/`" in index, f"{sub.relative_to(REPO)}/ missing from index.md"


def test_agents_lists_the_lab_indexes_and_command():
    agents = (REPO / "AGENTS.md").read_text()
    for folder in FOLDERS + [LAB / "static" / "data"]:
        rel = (folder / "index.md").relative_to(REPO).as_posix()
        assert f"]({rel})" in agents, rel
    assert "kinetic-lab" in agents
    assert "kinetic-lab" in (REPO / "agent.md").read_text()


def test_setup_page_is_a_classic_page_with_the_steps():
    html = (LAB / "static" / "setup.html").read_text()
    bare = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    assert 'type="module"' not in bare and "import " not in bare
    assert 'href="style.css"' in bare
    for needle in (
        "pip install -e '.[smplx,lab]'",
        "kinetic-lab",
        "https://smpl-x.is.tue.mpg.de/",
        "https://flame.is.tue.mpg.de/",
        "download_models.py --face",
        "convert_mano.py",
        "/api/status",
        "/api/models",
    ):
        assert needle in bare, needle
    # every fetch sits in the live-mode branch, after the file:// check
    assert bare.index("file:") < bare.index("fetch(")
    # no external scripts
    assert not re.search(r"<script[^>]+src=[\"']https?:", bare)


def test_setup_page_lists_every_required_model():
    from kinetic_augment.lab.models import REQUIRED_MODELS

    html = (LAB / "static" / "setup.html").read_text()
    for rel in REQUIRED_MODELS:
        assert rel in html, rel


def test_lab_doc_exists_and_is_linked():
    doc = (REPO / "docs" / "augmentation-lab.md").read_text()
    for needle in ("kinetic-lab", "build_lab_share.py", "viewer_saved", "load_profile"):
        assert needle in doc, needle
    assert "augmentation-lab.md" in (REPO / "docs" / "index.md").read_text()
    assert "docs/augmentation-lab.md" in (REPO / "README.md").read_text()
