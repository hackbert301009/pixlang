"""The editor must start however mangled its saved browser state is.

Needs Playwright and a Chromium build; skipped when either is missing.
"""
from __future__ import annotations

import json
import threading

import pytest

pytest.importorskip("flask")
sync_api = pytest.importorskip("playwright.sync_api")

from werkzeug.serving import make_server  # noqa: E402

from pixlang.editor.server import create_app  # noqa: E402

STORAGE_KEYS = ["autoRun", "flowScale", "session", "layout", "ui", "bottomTab", "previewMode",
                "dbgCollapsed", "breakpoints", "watch", "theme"]
GARBAGE = ["5", "-5", "{}", "[]", "null", '"text"', "true", '{"a":{"b":[1]}}',
           '{"file":{"kind":"workspace","path":5,"name":"x"},"draft":7}', "[1,{},null]"]


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except sync_api.Error as exc:
            pytest.skip(f"Chromium is not available: {str(exc).splitlines()[0]}")
        yield b
        b.close()


@pytest.fixture
def base_url(tmp_path):
    app = create_app(tmp_path, token="boot-token")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.mark.parametrize("value", GARBAGE)
def test_boots_with_garbage_in_every_key(browser, base_url, value):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    seed = {f"pixlang.{k}": value for k in STORAGE_KEYS}
    page.add_init_script(
        f"for (const [k, v] of Object.entries({json.dumps(seed)})) localStorage.setItem(k, v);")
    page.goto(base_url)
    page.wait_for_selector(".cm-editor", timeout=15000)
    assert page.is_visible("#btn-run")
    assert errors == []
    page.close()
