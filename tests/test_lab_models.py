import json

from kinetic_augment.lab.models import MODEL_LINKS, REQUIRED_MODELS, LabConfig, check_models


def test_check_models_lists_missing(tmp_path):
    assert check_models(tmp_path) == list(REQUIRED_MODELS)
    for rel in REQUIRED_MODELS[:-1]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert check_models(tmp_path) == [REQUIRED_MODELS[-1]]
    path = tmp_path / REQUIRED_MODELS[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    assert check_models(tmp_path) == []


def test_required_models_and_links():
    assert "smplx/SMPLX_NEUTRAL.npz" in REQUIRED_MODELS
    assert "mediapipe/face_landmarker.task" in REQUIRED_MODELS
    assert len(REQUIRED_MODELS) == 7
    assert MODEL_LINKS["smplx"] == "https://smpl-x.is.tue.mpg.de/"
    assert MODEL_LINKS["flame"] == "https://flame.is.tue.mpg.de/"
    assert MODEL_LINKS["flame_embedding"].startswith("https://github.com/Zielon/metrical-tracker")
    assert "download_models.py --face" in MODEL_LINKS["mediapipe"]


def test_config_round_trip(tmp_path, monkeypatch):
    home = tmp_path / "lab_home"
    monkeypatch.setenv("KINETIC_LAB_HOME", str(home))
    assert LabConfig.load().model_dir is None
    LabConfig(model_dir="/some/models").save()
    assert json.loads((home / "config.json").read_text()) == {"model_dir": "/some/models"}
    assert LabConfig.load().model_dir == "/some/models"


def test_config_tolerates_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setenv("KINETIC_LAB_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text("{not json")
    assert LabConfig.load().model_dir is None
