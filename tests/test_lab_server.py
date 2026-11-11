"""The Lab server routes (fake store and models, no model files needed) and the command."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kinetic_augment.augmentations.profile import load_profile
from kinetic_augment.lab import server as server_mod
from kinetic_augment.lab.bundle import decode_pane
from kinetic_augment.lab.cli import main
from kinetic_augment.lab.models import REQUIRED_MODELS, LabConfig
from kinetic_augment.lab.server import create_app

REPO = Path(__file__).resolve().parents[1]
GOOD = (REPO / "configs" / "slr_week4_default.yaml").read_text()


class FakeStore:
    def __init__(self, root):
        self.root = Path(root)
        self.models = None
        self.prep = SimpleNamespace(
            base_meta={"clip": "c1", "n": 2},
            base_arrays={"x": (np.arange(6, dtype=np.float32).reshape(2, 3), "float32")},
        )
        (self.root / "c1" / "frames").mkdir(parents=True)
        (self.root / "c1" / "frames" / "0000.jpg").write_bytes(b"jpg")

    def list(self):
        return [{"id": "c1", "frames": 1, "source": "fit"}]

    def get(self, clip_id):
        if clip_id != "c1":
            raise KeyError(clip_id)
        return self.prep

    def frame_path(self, clip_id, i):
        p = self.root / clip_id / "frames" / f"{int(i):04d}.jpg"
        if not p.exists():
            raise KeyError(clip_id)
        return p


class FakeQueue:
    def __init__(self):
        self.submitted = []

    def submit(self, video, clip_id):
        if clip_id == "busy":
            raise ValueError("in flight")
        self.submitted.append((Path(video), clip_id))
        return "job1"

    def status(self, job_id):
        if job_id != "job1":
            raise KeyError(job_id)
        return {"state": "waiting", "step": "", "progress": 0.0, "error": None}


def _model_dir(tmp_path, complete):
    root = tmp_path / "models"
    for rel in REQUIRED_MODELS if complete else REQUIRED_MODELS[:2]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    return root


@pytest.fixture
def lab_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINETIC_LAB_HOME", str(tmp_path / "home"))


def _app(tmp_path, complete=True, queue=None):
    static = tmp_path / "static"
    static.mkdir(exist_ok=True)
    (static / "index.html").write_text("<p>lab</p>")
    store = FakeStore(tmp_path / "clips")
    models = SimpleNamespace(tag="fake")
    app = create_app(
        store,
        LabConfig(model_dir=str(_model_dir(tmp_path, complete))),
        models_loader=lambda d: models,
        saved_dir=tmp_path / "saved",
        static_dir=static,
        job_queue=queue,
    )
    return TestClient(app), store, models


def test_status_setup_when_models_missing(tmp_path, lab_home):
    client, _, _ = _app(tmp_path, complete=False)
    body = client.get("/api/status").json()
    assert body["mode"] == "setup"
    assert body["missing"] and body["links"]["smplx"]
    assert body["model_dir"]


def test_status_live(tmp_path, lab_home):
    client, store, models = _app(tmp_path)
    body = client.get("/api/status").json()
    assert body["mode"] == "live" and body["missing"] == []
    assert store.models is models  # the store fits clips with the loaded models


def test_model_routes_503_in_setup(tmp_path, lab_home):
    client, _, _ = _app(tmp_path, complete=False)
    for method, url in [
        ("get", "/api/clips"),
        ("get", "/api/clips/c1/base"),
        ("get", "/api/clips/c1/frames/0.jpg"),
        ("post", "/api/pane"),
        ("get", "/api/jobs/x"),
    ]:
        kw = {"json": {"clip_id": "c1", "entry_id": "a"}} if method == "post" else {}
        r = getattr(client, method)(url, **kw)
        assert r.status_code == 503 and r.json() == {"detail": "models missing"}, url
    assert client.post("/api/videos", files={"file": ("a.mp4", b"v")}).status_code == 503
    # still available
    assert client.get("/api/saved").status_code == 200
    assert client.get("/api/status").status_code == 200
    assert client.get("/").text == "<p>lab</p>"


def test_post_models_goes_live(tmp_path, lab_home):
    client, _, _ = _app(tmp_path, complete=False)
    full = _model_dir(tmp_path / "other", complete=True)
    body = client.post("/api/models", json={"model_dir": str(full)}).json()
    assert body["mode"] == "live" and body["missing"] == []
    assert LabConfig.load().model_dir == str(full)
    assert client.get("/api/clips").json()[0]["id"] == "c1"
    still = client.post("/api/models", json={"model_dir": str(tmp_path / "nope")}).json()
    assert still["mode"] == "live"  # a bad folder does not unload working models
    assert still["missing"]


def test_clips_base_and_frames(tmp_path, lab_home):
    client, _, _ = _app(tmp_path)
    assert client.get("/api/clips").json() == [{"id": "c1", "frames": 1, "source": "fit"}]
    r = client.get("/api/clips/c1/base")
    assert r.headers["content-type"] == "application/octet-stream"
    meta, arrays = decode_pane(r.content)
    assert meta["clip"] == "c1" and arrays["x"].shape == (2, 3)
    assert client.get("/api/clips/zzz/base").status_code == 404
    assert client.get("/api/clips/c1/frames/0.jpg").content == b"jpg"
    assert client.get("/api/clips/c1/frames/9.jpg").status_code == 404


def test_schema_and_profile_text(tmp_path, lab_home):
    client, _, _ = _app(tmp_path)
    assert client.get("/api/schema?profile=default").json()["entries"]
    assert client.get("/api/profile-text?profile=default").text == GOOD
    assert client.get("/api/profile-text?profile=ambitious").status_code == 200
    assert client.get("/api/profile-text?profile=nonsense").status_code == 404


def _fake_run_pane(captured):
    def run(prep, req, base, models):
        captured.append((req, base, models))
        if req.values.get("bad"):
            raise ValueError("bad value for knob")
        return {"clip": req.clip_id, "entry": req.entry_id}, {
            "y": (np.ones((2, 2), dtype=np.float32), "float32")
        }

    return run


def test_pane_returns_blob(tmp_path, lab_home, monkeypatch):
    seen = []
    monkeypatch.setattr(server_mod, "run_pane", _fake_run_pane(seen))
    client, _, models = _app(tmp_path)
    r = client.post(
        "/api/pane", json={"clip_id": "c1", "entry_id": "body.location_jitter", "values": {}}
    )
    assert r.status_code == 200
    meta, arrays = decode_pane(r.content)
    assert meta["entry"] == "body.location_jitter" and arrays["y"].shape == (2, 2)
    req, base, used = seen[0]
    assert used is models and req.mode == "random" and base.name  # the default profile
    assert client.post("/api/pane", json={"entry_id": "x"}).status_code == 422
    r = client.post("/api/pane", json={"clip_id": "nope", "entry_id": "a"})
    assert r.status_code == 404


def test_pane_resolves_whole_profiles(tmp_path, lab_home, monkeypatch):
    seen = []
    monkeypatch.setattr(server_mod, "run_pane", _fake_run_pane(seen))
    client, _, _ = _app(tmp_path)
    ok = client.post("/api/pane", json={"clip_id": "c1", "entry_id": "combined.ambitious"})
    assert ok.status_code == 200
    assert seen[-1][1].name == load_profile(REPO / "configs" / "slr_week4_ambitious.yaml").name
    client.post("/api/saved", json={"name": "mine", "text": GOOD, "overwrite": False})
    assert (
        client.post("/api/pane", json={"clip_id": "c1", "entry_id": "saved:mine"}).status_code
        == 200
    )
    for entry in ("combined.nonsense", "saved:ghost"):
        r = client.post("/api/pane", json={"clip_id": "c1", "entry_id": entry})
        assert r.status_code == 422, entry


def test_pane_rejects_bad_value(tmp_path, lab_home, monkeypatch):
    monkeypatch.setattr(server_mod, "run_pane", _fake_run_pane([]))
    client, _, _ = _app(tmp_path)
    r = client.post("/api/pane", json={"clip_id": "c1", "entry_id": "a", "values": {"bad": 1}})
    assert r.status_code == 422 and r.json() == {"detail": "bad value for knob"}


def test_save_name_slug_and_conflict(tmp_path, lab_home):
    client, _, _ = _app(tmp_path)
    body = {"name": "My run / v2", "text": GOOD, "overwrite": False}
    r = client.post("/api/saved", json=body)
    assert r.status_code == 200 and r.json()["path"].endswith("my-run-v2.yaml")
    assert (tmp_path / "saved" / "my-run-v2.yaml").read_text() == GOOD
    assert client.post("/api/saved", json=body).status_code == 409
    assert client.post("/api/saved", json=dict(body, overwrite=True)).status_code == 200
    assert client.get("/api/saved").json() == [{"name": "my-run-v2"}]
    assert client.get("/api/saved/my-run-v2").text == GOOD
    assert client.get("/api/saved/ghost").status_code == 404
    assert client.get("/api/saved/..%2Fx").status_code in (404, 422)


def test_save_rejects_invalid_profile(tmp_path, lab_home):
    client, _, _ = _app(tmp_path)
    r = client.post("/api/saved", json={"name": "x", "text": "bogus: 1\n", "overwrite": False})
    assert r.status_code == 422 and r.json()["detail"]
    r = client.post("/api/saved", json={"name": "///", "text": GOOD, "overwrite": False})
    assert r.status_code == 422
    assert not (tmp_path / "saved" / "x.yaml").exists()


def test_upload_video(tmp_path, lab_home):
    queue = FakeQueue()
    client, store, _ = _app(tmp_path, queue=queue)
    r = client.post("/api/videos", files={"file": ("My Sign!.mp4", b"vid")})
    assert r.json() == {"job_id": "job1", "clip_id": "my-sign"}
    video, _ = queue.submitted[0]
    assert video.parent.parent == store.root and video.suffix == ".mp4"
    assert video.read_bytes() == b"vid"
    # taken by the store -> suffixed; in flight -> suffixed again
    assert client.post("/api/videos", files={"file": ("c1.mp4", b"v")}).json()["clip_id"] == "c1-2"
    assert client.post("/api/videos", files={"file": ("busy.mp4", b"v")}).json()["clip_id"] == (
        "busy-2"
    )
    assert client.get("/api/jobs/job1").json()["state"] == "waiting"
    assert client.get("/api/jobs/zzz").status_code == 404
    assert client.post("/api/videos", files={"file": ("...", b"v")}).status_code == 422


def test_cli_refuses_public_host():
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0", "--no-browser"])
    with pytest.raises(SystemExit):
        main(["--host", "192.168.1.5", "--no-browser"])


def test_cli_starts_uvicorn(tmp_path, lab_home, monkeypatch):
    import kinetic_augment.lab.cli as cli

    calls = {}
    monkeypatch.setattr(cli, "build_app", lambda args: "APP")
    monkeypatch.setattr(
        cli.uvicorn, "run", lambda app, host, port, **kw: calls.update(h=host, p=port, a=app)
    )
    monkeypatch.setattr(cli, "_open_browser", lambda url: calls.update(url=url))
    main(["--port", "8123", "--no-browser"])
    assert calls == {"h": "127.0.0.1", "p": 8123, "a": "APP"}
    calls.clear()
    main(["--port", "8123"])
    assert calls["url"] == "http://127.0.0.1:8123/"


@pytest.mark.parametrize("complete", [True, False])
def test_cli_falls_back_to_repo_models_without_saving(tmp_path, lab_home, monkeypatch, complete):
    import kinetic_augment.lab.cli as cli

    repo_models = _model_dir(tmp_path, complete)
    monkeypatch.setattr(cli, "repo_model_dir", lambda: repo_models)
    config = cli.startup_config(None)
    assert config.model_dir == (str(repo_models) if complete else None)
    assert not LabConfig.path().exists()  # used for this run only


def test_cli_saved_and_given_model_dir_win_over_repo_models(tmp_path, lab_home, monkeypatch):
    import kinetic_augment.lab.cli as cli

    monkeypatch.setattr(cli, "repo_model_dir", lambda: _model_dir(tmp_path, True))
    LabConfig(model_dir="/saved/models").save()
    assert cli.startup_config(None).model_dir == "/saved/models"
    assert cli.startup_config("/given").model_dir == "/given"
    assert LabConfig.load().model_dir == "/given"


def test_repo_model_dir_is_the_repo_models_folder():
    from kinetic_augment.lab.models import repo_model_dir

    assert repo_model_dir() == REPO / "models"


@pytest.mark.parametrize(
    "payload",
    [
        {"clip_id": "c1", "entry_id": "a", "reroll": "a"},
        {"clip_id": "c1", "entry_id": "a", "reroll": -1},
        {"clip_id": "c1", "entry_id": "a", "values": [1]},
        {"clip_id": 5, "entry_id": "a"},
        {"clip_id": "c1", "entry_id": "a", "mode": 3},
        {"clip_id": "c1", "entry_id": "a", "surprise": 1},
    ],
)
def test_pane_malformed_payload_is_422(tmp_path, lab_home, monkeypatch, payload):
    monkeypatch.setattr(server_mod, "run_pane", _fake_run_pane([]))
    client, _, _ = _app(tmp_path)
    r = client.post("/api/pane", json=payload)
    assert r.status_code == 422 and isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_pane_type_error_in_run_is_422(tmp_path, lab_home, monkeypatch):
    def boom(prep, req, base, models):
        raise TypeError("'<' not supported between 'str' and 'float'")

    monkeypatch.setattr(server_mod, "run_pane", boom)
    client, _, _ = _app(tmp_path)
    r = client.post("/api/pane", json={"clip_id": "c1", "entry_id": "a", "values": {"k": "text"}})
    assert r.status_code == 422 and "not supported" in r.json()["detail"]


def test_upload_rejects_non_video(tmp_path, lab_home):
    queue = FakeQueue()
    client, store, _ = _app(tmp_path, queue=queue)
    for name in ("notes.txt", "clip", "x.mp4.exe"):
        r = client.post("/api/videos", files={"file": (name, b"v")})
        assert r.status_code == 422, name
    assert client.post("/api/videos", files={"file": ("A.MOV", b"v")}).status_code == 200
    assert not queue.submitted[0][0].name.endswith((".txt", ".exe"))


def test_upload_deleted_when_submit_fails(tmp_path, lab_home):
    class Broken(FakeQueue):
        def submit(self, video, clip_id):
            raise RuntimeError("queue down")

    client, store, _ = _app(tmp_path, queue=Broken())
    with pytest.raises(RuntimeError):
        client.post("/api/videos", files={"file": ("a.mp4", b"v")})
    assert list((store.root / ".uploads").iterdir()) == []


@pytest.mark.parametrize("fails", [False, True])
def test_upload_deleted_when_job_ends(tmp_path, lab_home, monkeypatch, fails):
    import time

    def fake_ingest(video, clip_id, store, models, progress):
        assert Path(video).exists()
        if fails:
            raise RuntimeError("no hands")
        return clip_id

    monkeypatch.setattr(server_mod, "ingest_video", fake_ingest)
    client, store, _ = _app(tmp_path)  # default JobQueue with the real wrapper
    job = client.post("/api/videos", files={"file": ("a.mp4", b"v")}).json()["job_id"]
    for _ in range(100):
        if client.get(f"/api/jobs/{job}").json()["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert client.get(f"/api/jobs/{job}").json()["state"] == ("failed" if fails else "done")
    assert list((store.root / ".uploads").iterdir()) == []
