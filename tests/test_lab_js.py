"""Runs the Lab frontend's Node tests (`tests/js/*.test.js`) on freshly written fixtures."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JS_DIR = Path(__file__).resolve().parent / "js"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_lab_js(tmp_path):
    fixtures = tmp_path / "fixtures"
    env = dict(os.environ)
    src = str(JS_DIR.parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(JS_DIR / "make_fixture.py"), str(fixtures)], check=True, env=env
    )
    env["LAB_FIXTURE_DIR"] = str(fixtures)
    # Node 22+ takes globs, not folders: pass each test file.
    files = sorted(str(p) for p in JS_DIR.glob("*.test.js"))
    assert files, "no Node tests found"
    run = subprocess.run(
        [NODE, "--test", *files], capture_output=True, text=True, env=env, timeout=120
    )
    assert run.returncode == 0, run.stdout + run.stderr
