"""A command that never returns must not wedge the editor."""
from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("flask")

from pixlang.commands import registry as global_registry  # noqa: E402
from pixlang.commands.registry import CommandRegistry  # noqa: E402
from pixlang.editor.rungate import MAX_STUCK_RUNS, RunBlocked, RunGate  # noqa: E402
from pixlang.editor.server import create_app  # noqa: E402

TOKEN = "gate-token"


@pytest.fixture
def blocker():
    release = threading.Event()
    yield release
    release.set()


@pytest.fixture
def client(tmp_path, blocker):
    reg = CommandRegistry()
    reg._commands.update(global_registry._commands)

    @reg.register("HANG", source="plugin")
    def hang(image):
        blocker.wait(30)        # stands in for a native call that cannot be interrupted
        return image

    app = create_app(tmp_path, token=TOKEN, registry=reg, plugins=False, run_timeout=0.2)
    app.config["PIXLANG_GATE"].grace = 0.05
    with app.test_client() as c:
        c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
        yield c


def run(client, code):
    return client.post("/api/run", json={"code": code}).get_json()


def test_run_after_a_stuck_one_is_accepted(client):
    first = run(client, "USE ZEROS(4, 4, 3)\nHANG")
    assert first["error_info"]["kind"] == "timeout"
    assert "could not be stopped" in first["error"]
    second = run(client, "USE ZEROS(4, 4, 3)")
    assert second["ok"], second["error"]


def test_cwd_is_not_changed_by_a_stuck_run(client):
    before = os.getcwd()
    run(client, "USE ZEROS(4, 4, 3)\nHANG")
    assert os.getcwd() == before


def test_too_many_stuck_runs_say_so_instead_of_in_progress(client):
    for _ in range(MAX_STUCK_RUNS):
        assert run(client, "USE ZEROS(4, 4, 3)\nHANG")["error_info"]["kind"] == "timeout"
    refused = run(client, "USE ZEROS(4, 4, 3)")
    assert refused["error_info"]["kind"] == "stuck"
    assert "restart the editor" in refused["error"]


def test_debugger_start_reports_stuck_runs_too(client):
    for _ in range(MAX_STUCK_RUNS):
        run(client, "USE ZEROS(4, 4, 3)\nHANG")
    r = client.post("/api/debug/start", json={"code": "USE ZEROS(4, 4, 3)"})
    assert r.status_code == 409
    assert "restart the editor" in r.get_json()["error"]


def test_released_ticket_frees_the_gate_once():
    gate = RunGate()
    ticket = gate.acquire(0.1)
    assert gate.acquire(0.05) is None
    ticket.release()
    ticket.release()
    again = gate.acquire(0.1)
    assert again is not None
    assert gate.acquire(0.05) is None


def test_abandoned_threads_are_forgotten_when_they_finish():
    gate = RunGate(max_stuck=1)
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, daemon=True)
    worker.start()
    gate.acquire(0.1).abandon(worker)
    with pytest.raises(RunBlocked):
        gate.acquire(0.1)
    stop.set()
    worker.join()
    assert gate.acquire(0.1) is not None
