"""Turn a dropped-in sign video into a stored clip, one at a time."""

from __future__ import annotations

import queue
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

Progress = Callable[[str, float], None]


def scripts_dir() -> Path:
    """The checkout's `scripts/` folder (three levels above `src/kinetic_augment/lab`)."""
    return Path(__file__).resolve().parents[3] / "scripts"


def _load_export_module():
    folder = scripts_dir()
    if not folder.is_dir():
        raise RuntimeError(
            f"video ingest needs the repository's scripts folder, which is missing ({folder})"
        )
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import export_round_trip_demo  # pulls cv2 and mediapipe, so import it lazily

    return export_round_trip_demo


def ingest_video(video: Path, clip_id: str, store, models, progress: Progress) -> str:
    """Extract landmarks, fit SMPL-X, save the frames and add the clip to `store`."""
    demo = _load_export_module()
    video = Path(video)
    with tempfile.TemporaryDirectory(prefix="kinetic_lab_") as tmp:
        work = Path(tmp)
        progress("extracting landmarks", 0.0)
        face_model = Path(models.model_dir) / "mediapipe" / "face_landmarker.task"
        hands, face = demo.extract(video, work, True, face_model=face_model)
        progress("fitting", 0.4)
        result, _recon = demo.run_with_mano(hands, face, models)
        progress("saving frames", 0.8)
        frames = work / "frames"
        demo.read_frames(video, frames, len(face.valid))
        store.add(clip_id, result, hands, face, frames)
    progress("done", 1.0)
    return clip_id


class JobQueue:
    """FIFO queue with one daemon worker; `runner(video, clip_id, progress)` does each job."""

    def __init__(
        self,
        store: Any = None,
        models: Any = None,
        runner: Optional[Callable[[Path, str, Progress], Any]] = None,
    ):
        if runner is None:

            def runner(video, clip_id, progress):
                return ingest_video(video, clip_id, store, models, progress)

        self._runner = runner
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None

    def submit(self, video: Path, clip_id: str) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            for other in self._jobs.values():
                if other["clip_id"] == clip_id and other["state"] in ("waiting", "running"):
                    raise ValueError(f"clip {clip_id!r} is already being ingested")
            self._jobs[job_id] = {
                "video": Path(video),
                "clip_id": clip_id,
                "state": "waiting",
                "step": "",
                "progress": 0.0,
                "error": None,
            }
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, daemon=True)
                self._worker.start()
        self._queue.put(job_id)
        return job_id

    def status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]  # KeyError for an unknown id
            return {k: job[k] for k in ("state", "step", "progress", "error")}

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs[job_id]
                video, clip_id = job["video"], job["clip_id"]
            self._update(job_id, state="running")

            def progress(step: str, fraction: float, _id=job_id) -> None:
                self._update(_id, step=step, progress=float(fraction))

            try:
                self._runner(video, clip_id, progress)
            except Exception as exc:  # report any failure and keep the worker alive
                self._update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
            else:
                self._update(job_id, state="done", step="done", progress=1.0)
