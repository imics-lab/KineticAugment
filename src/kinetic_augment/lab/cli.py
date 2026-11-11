"""`kinetic-lab`: start the Augmentation Lab on this machine and open it in the browser."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path
from typing import List, Optional

import uvicorn

from kinetic_augment.lab.clipstore import DEFAULT_ROOT, ClipStore
from kinetic_augment.lab.models import LabConfig, check_models, repo_model_dir
from kinetic_augment.lab.server import create_app

LOCAL_HOST = "127.0.0.1"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kinetic-lab", description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default=LOCAL_HOST, help="only 127.0.0.1 is allowed")
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=None, help="folder with the model files (saved)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.host != LOCAL_HOST:
        parser.error(f"the lab serves this machine only; --host must be {LOCAL_HOST}")
    return args


def startup_config(model_dir: Optional[str] = None) -> LabConfig:
    """
    The Lab settings for this run. `--model-dir` is saved; with no folder given or saved, the
    repo's own `models/` is used when it is complete (for this run only, never saved).
    """
    config = LabConfig.load()
    if model_dir:
        config.model_dir = str(model_dir)
        config.save()
    elif config.model_dir is None:
        repo = repo_model_dir()
        if repo.is_dir() and not check_models(repo):
            config.model_dir = str(repo)
    return config


def build_app(args: argparse.Namespace):
    return create_app(ClipStore(args.clips_dir), startup_config(args.model_dir), port=args.port)


def _open_browser(url: str) -> None:
    """Open `url` shortly after the server starts listening."""
    threading.Timer(1.0, webbrowser.open, args=(url,)).start()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    app = build_app(args)
    url = f"http://{LOCAL_HOST}:{args.port}/"
    if not args.no_browser:
        _open_browser(url)
    uvicorn.run(app, host=LOCAL_HOST, port=args.port)


if __name__ == "__main__":
    main()
