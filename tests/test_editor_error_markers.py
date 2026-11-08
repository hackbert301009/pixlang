"""An error inside an imported module is marked at the call site in the edited file."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("flask")

from pixlang.editor.server import create_app  # noqa: E402

TOKEN = "marker-token"

# the failing RESIZE is on line 3 of the module, and line 3 of main is BLUR
LIB = "# helper\nEXPORT FUNC BOOM()\n  RESIZE 0 0\nENDFUNC\n"
MAIN = 'IMPORT "lib/util.pxl" AS u\nLOAD "in.png"\nBLUR 3\nu.BOOM\nGRAYSCALE\n'
CALL_SITE = 4


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    (root / "lib").mkdir(parents=True)
    cv2.imwrite(str(root / "in.png"), np.full((20, 30, 3), 90, np.uint8))
    (root / "lib" / "util.pxl").write_text(LIB)
    (root / "main.pxl").write_text(MAIN)
    return root


@pytest.fixture
def client(ws):
    app = create_app(ws, token=TOKEN, run_timeout=10.0)
    with app.test_client() as c:
        c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
        yield c


@pytest.mark.parametrize("body", [{"path": "main.pxl"}, {}], ids=["saved", "unsaved"])
def test_run_marks_the_call_site(client, ws, body):
    res = client.post("/api/run", json={"code": MAIN, **body}).get_json()
    info = res["error_info"]
    assert info["line"] == CALL_SITE
    assert info["file"] == "lib/util.pxl"
    assert info["file_line"] == 3
    assert info["col"] is None
    assert str(ws) not in info["message"]
    assert 3 in res["visited_lines"] and 4 in res["visited_lines"]


def test_error_in_the_edited_file_keeps_its_own_line(client):
    res = client.post("/api/run", json={
        "code": 'LOAD "in.png"\nBLUR 3\nRESIZE 0 0\n', "path": "main.pxl"}).get_json()
    info = res["error_info"]
    assert info["line"] == 3
    assert "file_line" not in info


def test_same_named_file_in_another_directory_is_not_the_edited_file(client, ws):
    (ws / "lib" / "main.pxl").write_text(LIB)
    code = 'IMPORT "lib/main.pxl" AS u\nLOAD "in.png"\nBLUR 3\nu.BOOM\n'
    res = client.post("/api/run", json={"code": code, "path": "main.pxl"}).get_json()
    assert res["error_info"]["line"] == 4
    assert res["error_info"]["file"] == "lib/main.pxl"


def test_debugger_reports_the_call_site_too(client):
    view = client.post("/api/debug/start", json={
        "code": MAIN, "path": "main.pxl", "stop_on_entry": False}).get_json()
    for _ in range(50):
        if view["state"] in ("finished", "error", "stopped"):
            break
        view = client.get(f"/api/debug/{view['session_id']}/state?wait={view['seq']}&timeout=1"
                          ).get_json()
    assert view["state"] == "error"
    assert view["result"]["error_info"]["line"] == CALL_SITE
    assert view["location"]["line"] == CALL_SITE
    assert view["stack"][0]["file"].endswith("lib/util.pxl")
