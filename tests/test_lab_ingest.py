import threading
import time
from pathlib import Path

import pytest

from kinetic_augment.lab import ingest
from kinetic_augment.lab.ingest import JobQueue


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
