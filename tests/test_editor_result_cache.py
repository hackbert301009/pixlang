"""The editor keeps only display-sized copies of recent results."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("flask")

from pixlang.editor.server import (  # noqa: E402
    EDITOR_LIMITS,
    FULL_MAX_SIDE,
    RESULT_CACHE_RUNS,
    create_app,
)

TOKEN = "cache-token"


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, token=TOKEN, run_timeout=20.0)
    with app.test_client() as c:
        c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
        yield c


def run(client, code):
    return client.post("/api/run", json={"code": code}).get_json()


def fetch(client, ref):
    data = client.get(ref["url"]).data
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def test_editor_refuses_arrays_beyond_its_limit(client):
    res = run(client, "ALLOC b 30000 30000 3 u8 7")
    assert not res["ok"]
    assert "exceeds the limit" in res["error"]
    assert EDITOR_LIMITS.max_pixels < 64_000_000


def test_large_result_is_served_downscaled(client):
    res = run(client, "USE ZEROS(5000, 5000, 1, f32)")
    assert res["ok"], res["error"]
    assert res["image"]["width"] == 5000
    shown = fetch(client, res["image"])
    assert max(shown.shape[:2]) == FULL_MAX_SIDE


def test_only_the_last_runs_are_kept(client):
    first = run(client, "USE ZEROS(8, 8, 3)")
    for _ in range(RESULT_CACHE_RUNS):
        assert run(client, "USE ZEROS(8, 8, 3)")["ok"]
    assert client.get(first["image"]["url"]).status_code == 404
