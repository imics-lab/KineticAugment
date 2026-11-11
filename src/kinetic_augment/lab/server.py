"""The Lab's local web server: clips, live panes, video uploads and saved settings."""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from kinetic_augment.augmentations.profile import Profile, load_profile
from kinetic_augment.lab.bundle import encode_pane
from kinetic_augment.lab.clipstore import ClipStore
from kinetic_augment.lab.ingest import JobQueue
from kinetic_augment.lab.models import MODEL_LINKS, REQUIRED_MODELS, LabConfig, check_models
from kinetic_augment.lab.pane import PaneRequest, run_pane
from kinetic_augment.lab.saving import (
    DEFAULT_FOLDER,
    save_setting,
    slugify,
    validate_profile_text,
)
from kinetic_augment.lab.schema import build_schema

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
PROFILE_NAMES = ("default", "ambitious", "literature")
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = ".uploads"
OCTET = "application/octet-stream"


def _default_loader(model_dir):
    from kinetic_augment.body_model.face_pipeline import load_face_models

    return load_face_models(model_dir)


def shipped_profile_path(name: str) -> Path:
    if name not in PROFILE_NAMES:
        raise KeyError(name)
    return CONFIGS_DIR / f"slr_week4_{name}.yaml"


def create_app(
    store: Optional[ClipStore],
    config: LabConfig,
    models_loader: Callable[[Any], Any] = _default_loader,
    saved_dir: Path = DEFAULT_FOLDER,
    static_dir: Path = STATIC_DIR,
    job_queue: Optional[Any] = None,
) -> FastAPI:
    """
    Build the app. Models load now when `config.model_dir` holds every required file, else the
    app starts in setup mode until `POST /api/models` supplies a complete folder.
    """
    app = FastAPI(title="KineticAugment Lab")
    store = store if store is not None else ClipStore()
    saved_dir = Path(saved_dir)
    state: Dict[str, Any] = {"models": None, "jobs": job_queue}
    lock = threading.Lock()

    def missing(folder: Optional[str]) -> List[str]:
        return check_models(Path(folder)) if folder else list(REQUIRED_MODELS)

    def try_load() -> Optional[str]:
        """Load the models if the folder is complete; return an error text on failure."""
        if config.model_dir is None or check_models(Path(config.model_dir)):
            return None
        try:
            models = models_loader(config.model_dir)
        except Exception as exc:  # a bad model file must not take the server down
            return f"{type(exc).__name__}: {exc}"
        with lock:
            state["models"] = models
            store.models = models
            if state["jobs"] is None:
                state["jobs"] = JobQueue(store, models)
        return None

    def status(error: Optional[str] = None, folder: Optional[str] = None) -> Dict[str, Any]:
        folder = folder or config.model_dir
        body = {
            "mode": "live" if state["models"] is not None else "setup",
            "missing": missing(folder),
            "model_dir": folder,
            "links": MODEL_LINKS,
        }
        if error:
            body["error"] = error
        return body

    def need_models() -> Any:
        if state["models"] is None:
            raise HTTPException(503, "models missing")
        return state["models"]

    def saved_path(name: str) -> Path:
        try:
            slug = slugify(name)
        except ValueError:
            raise HTTPException(404, "no such saved setting")
        path = saved_dir / f"{slug}.yaml"
        if slug != name or not path.is_file():
            raise HTTPException(404, "no such saved setting")
        return path

    def profile_path(name: str) -> Path:
        """File for `default|ambitious|literature` or `saved:<name>`; 404 when unknown."""
        if name.startswith("saved:"):
            return saved_path(name[len("saved:") :])
        try:
            return shipped_profile_path(name)
        except KeyError:
            raise HTTPException(404, f"unknown profile {name!r}")

    def base_for(req: PaneRequest) -> Profile:
        """R4: the whole profile for `combined.*` / `saved:*` ids, else the pane's profile."""
        eid = req.entry_id
        try:
            if eid.startswith("combined."):
                return load_profile(shipped_profile_path(eid[len("combined.") :]))
            if eid.startswith("saved:"):
                return load_profile(saved_path(eid[len("saved:") :]))
        except (KeyError, HTTPException):
            raise ValueError(f"unknown profile in {eid!r}")
        try:
            return load_profile(profile_path(req.profile))
        except HTTPException:
            raise ValueError(f"unknown profile {req.profile!r}")

    # ---- status and setup -------------------------------------------------------------------
    @app.get("/api/status")
    def get_status():
        return status()

    @app.post("/api/models")
    def post_models(body: Dict[str, Any] = Body(...)):
        folder = str(body.get("model_dir") or "").strip()
        if not folder:
            raise HTTPException(422, "model_dir is required")
        complete = not check_models(Path(folder))
        if complete or state["models"] is None:
            config.model_dir = folder
            try:
                config.save()
            except OSError:
                pass  # not being able to remember the folder is not fatal
        # A folder that is not complete never unloads models that already work.
        error = try_load() if state["models"] is None else None
        return status(error, folder)

    # ---- profiles ---------------------------------------------------------------------------
    @app.get("/api/schema")
    def get_schema(profile: str = Query("default")):
        return build_schema(load_profile(profile_path(profile)))

    @app.get("/api/profile-text")
    def get_profile_text(profile: str = Query("default")):
        return PlainTextResponse(profile_path(profile).read_text())

    # ---- clips and panes --------------------------------------------------------------------
    @app.get("/api/clips")
    def get_clips():
        need_models()
        return store.list()

    @app.get("/api/clips/{clip_id}/base")
    def get_base(clip_id: str):
        need_models()
        try:
            prep = store.get(clip_id)
        except KeyError:
            raise HTTPException(404, "no such clip")
        return Response(encode_pane(prep.base_meta, prep.base_arrays), media_type=OCTET)

    @app.get("/api/clips/{clip_id}/frames/{index}.jpg")
    def get_frame(clip_id: str, index: int):
        need_models()
        try:
            return FileResponse(store.frame_path(clip_id, index), media_type="image/jpeg")
        except KeyError:
            raise HTTPException(404, "no such frame")

    @app.post("/api/pane")
    def post_pane(body: Dict[str, Any] = Body(...)):
        models = need_models()
        fields = {"clip_id", "entry_id", "values", "mode", "reroll", "edge", "profile"}
        extra = sorted(set(body) - fields)
        if extra or "clip_id" not in body or "entry_id" not in body:
            raise HTTPException(422, "a pane needs clip_id and entry_id (and only known fields)")
        try:
            req = PaneRequest(**body)
            prep = store.get(req.clip_id)
        except KeyError:
            raise HTTPException(404, "no such clip")
        except TypeError as exc:
            raise HTTPException(422, str(exc))
        try:
            meta, arrays = run_pane(prep, req, base_for(req), models)
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=422)
        return Response(encode_pane(meta, arrays), media_type=OCTET)

    # ---- uploads ----------------------------------------------------------------------------
    @app.post("/api/videos")
    def post_video(file: UploadFile = File(...)):
        need_models()
        stem = Path(file.filename or "").stem
        try:
            base_id = slugify(stem)
        except ValueError:
            raise HTTPException(422, "the file name has no letters or digits to name the clip")
        taken = {c["id"] for c in store.list()}
        updir = store.root / UPLOAD_DIR
        updir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=updir, suffix=Path(file.filename or "").suffix or ".mp4", delete=False
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
        n = 1
        while True:
            clip_id = base_id if n == 1 else f"{base_id}-{n}"
            n += 1
            if clip_id in taken:
                continue
            try:
                job_id = state["jobs"].submit(Path(tmp.name), clip_id)
            except ValueError:  # that id is already in flight
                continue
            return {"job_id": job_id, "clip_id": clip_id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        need_models()
        try:
            return state["jobs"].status(job_id)
        except KeyError:
            raise HTTPException(404, "no such job")

    # ---- saved settings ---------------------------------------------------------------------
    @app.get("/api/saved")
    def list_saved():
        if not saved_dir.is_dir():
            return []
        return [{"name": p.stem} for p in sorted(saved_dir.glob("*.yaml"))]

    @app.get("/api/saved/{name}")
    def get_saved(name: str):
        return PlainTextResponse(saved_path(name).read_text())

    @app.post("/api/saved")
    def post_saved(body: Dict[str, Any] = Body(...)):
        name, text = body.get("name"), body.get("text")
        if not isinstance(name, str) or not isinstance(text, str):
            raise HTTPException(422, "name and text are required")
        try:
            validate_profile_text(text)
            slugify(name)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        try:
            path = save_setting(text, name, saved_dir, bool(body.get("overwrite", False)))
        except FileExistsError:
            raise HTTPException(409, "a saved setting with that name already exists")
        return {"path": str(path), "name": path.stem}

    # ---- start-up and static files ----------------------------------------------------------
    try_load()
    if Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app
