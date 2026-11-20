"""Front-end behaviour that does not need a browser: stored state and API errors."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STATIC_JS = HERE.parent / "pixlang" / "editor" / "static" / "js"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_storage_helper_and_api_wrapper():
    done = subprocess.run(
        ["node", str(HERE / "js" / "client_state.mjs"), str(STATIC_JS / "util.js")],
        capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr or done.stdout


def test_no_module_reads_storage_unvalidated():
    for js in STATIC_JS.glob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert "store.get(" not in text, f"{js.name} uses the removed unvalidated store.get"
        assert "JSON.parse(localStorage" not in text, js.name
