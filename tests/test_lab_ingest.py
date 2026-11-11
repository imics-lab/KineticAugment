import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinetic_augment.lab import ingest
from kinetic_augment.lab.clipstore import ClipStore
from kinetic_augment.lab.ingest import JobQueue
from kinetic_augment.lab.models import check_models
from tests.hand_assets import MODEL_DIR, requires_smplx


def wait_for(queue, job, state, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if queue.status(job)["state"] == state:
            return queue.status(job)
        time.sleep(0.01)
    raise AssertionError(f"{job} never reached {state}: {queue.status(job)}")


def test_jobs_run_one_at_a_time():
    release = threading.Event()
    running = []
    order = []

    def slow(video, clip_id, progress):
        running.append(clip_id)
        assert len(running) == 1
        progress("fitting", 0.5)
        release.wait(5)
        order.append(clip_id)
        running.remove(clip_id)
        return clip_id

    queue = JobQueue(runner=slow)
    first = queue.submit(Path("a.mp4"), "a")
    second = queue.submit(Path("b.mp4"), "b")
    wait_for(queue, first, "running")
    assert queue.status(second)["state"] == "waiting"
    assert queue.status(first)["step"] == "fitting"
    assert queue.status(first)["progress"] == 0.5
    release.set()
    assert wait_for(queue, first, "done")["progress"] == 1.0
    wait_for(queue, second, "done")
    assert order == ["a", "b"]


def test_failed_job_reports_error():
    calls = []

    def runner(video, clip_id, progress):
        calls.append(clip_id)
        if clip_id == "x":
            raise ValueError("no landmarks")

    queue = JobQueue(runner=runner)
    status = wait_for(queue, queue.submit(Path("x.mp4"), "x"), "failed")
    assert "no landmarks" in status["error"]
    # the worker survives a failure
    wait_for(queue, queue.submit(Path("y.mp4"), "y"), "done")
    assert calls == ["x", "y"]


def test_unknown_job_raises_key_error():
    with pytest.raises(KeyError):
        JobQueue(runner=lambda *a: None).status("nope")


def test_ingest_needs_scripts_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "scripts_dir", lambda: tmp_path / "missing")
    with pytest.raises(RuntimeError, match="scripts"):
        ingest.ingest_video(Path("v.mp4"), "c", None, None, lambda s, f: None)


def test_submit_refuses_a_clip_already_in_flight():
    release = threading.Event()

    def blocked(video, clip_id, progress):
        release.wait(5)

    queue = JobQueue(runner=blocked)
    first = queue.submit(Path("a.mp4"), "a")
    with pytest.raises(ValueError):
        queue.submit(Path("a2.mp4"), "a")  # running or waiting
    wait_for(queue, first, "running")
    with pytest.raises(ValueError):
        queue.submit(Path("a2.mp4"), "a")
    queue.submit(Path("b.mp4"), "b")
    release.set()
    wait_for(queue, first, "done")
    wait_for(queue, queue.submit(Path("a3.mp4"), "a"), "done")  # allowed once finished


def test_ingest_passes_the_configured_face_model(tmp_path, monkeypatch):
    seen = {}

    class FakeDemo:
        @staticmethod
        def extract(video, cache, reextract, face_model=None):
            seen["face_model"] = face_model
            return SimpleNamespace(), SimpleNamespace(valid=[1, 2])

        @staticmethod
        def run_with_mano(hands, face, models):
            return "result", None

        @staticmethod
        def read_frames(video, out_dir, n):
            seen["n"] = n
            Path(out_dir).mkdir(parents=True)

    class FakeStore:
        def add(self, *args):
            seen["added"] = args[0]

    monkeypatch.setattr(ingest, "_load_export_module", lambda: FakeDemo)
    steps = []
    models = SimpleNamespace(model_dir=str(tmp_path / "m"))
    ingest.ingest_video(Path("v.mp4"), "c", FakeStore(), models, lambda step, f: steps.append(step))
    assert seen["face_model"] == tmp_path / "m" / "mediapipe" / "face_landmarker.task"
    assert seen["n"] == 2 and seen["added"] == "c"
    assert steps == ["extracting landmarks", "fitting", "saving frames", "done"]


def _can_run_real_ingest():
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return not check_models(Path(MODEL_DIR))


@requires_smplx
@pytest.mark.skipif(not _can_run_real_ingest(), reason="needs cv2, mediapipe and every model file")
def test_real_ingest_end_to_end(tmp_path):
    from kinetic_augment.body_model.face_pipeline import load_face_models

    video = Path(__file__).resolve().parents[1] / "data" / "sign_videos" / "01991.mp4"
    if not video.exists():
        pytest.skip("sign video missing")
    models = load_face_models(MODEL_DIR)
    store = ClipStore(tmp_path / "clips", models)
    steps = []
    start = time.time()
    ingest.ingest_video(video, "01991", store, models, lambda step, f: steps.append(step))
    print(f"real ingest took {time.time() - start:.1f} s")
    assert steps == ["extracting landmarks", "fitting", "saving frames", "done"]
    assert store.frame_path("01991", 0).exists()
    assert store.get("01991").frames.shape[0] > 0
