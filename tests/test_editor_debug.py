# tests/test_editor_debug.py
"""Editor step debugger: session lifecycle, breakpoints, stepping, state
exposure, watch evaluation, timeouts, security."""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("flask")

from pixlang.editor import debugger as dbgmod  # noqa: E402
from pixlang.editor.server import create_app  # noqa: E402
from pixlang.parser.errors import PixRuntimeError  # noqa: E402

TOKEN = "dbg-token-123"

# line numbers in these snippets are referenced by the tests below
FUNC_CODE = """\
FUNC add_two(n)
    SET r $n + 2
    RETURN $r
ENDFUNC
FUNC SHOW(i)
    SET t $i * 10
    SET u $t + 1
ENDFUNC
SET a 1
SET b add_two($a)
SHOW 4
SET c 3
"""

LOOP_CODE = """\
SET total 0
REPEAT 3 AS i
    SET total $total + $i
END
SET done 1
"""

FLAT_CODE = """\
SET a 1
SET b 2
SET c 3
SET d 4
"""


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _make_app(ws, **kw):
    kw.setdefault("run_timeout", 5.0)
    app = create_app(ws, token=TOKEN, **kw)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def app(ws):
    a = _make_app(ws)
    yield a
    a.config["PIXLANG_DEBUG"].stop_all()


@pytest.fixture
def client(app):
    with app.test_client() as c:
        c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
        yield c


@pytest.fixture
def raw(app):
    with app.test_client() as c:
        yield c


def start(client, code, **body):
    r = client.post("/api/debug/start", json={"code": code, **body})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def ctl(client, snap, action, **extra):
    r = client.post(f"/api/debug/{snap['session_id']}/control", json={"action": action, **extra})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def wait_for(client, snap, pred, timeout=8.0):
    """Long-poll until pred(state) holds."""
    deadline = time.monotonic() + timeout
    while True:
        if pred(snap):
            return snap
        left = deadline - time.monotonic()
        assert left > 0, f"timed out waiting; last state={snap['state']} line={snap.get('location')}"
        r = client.get(f"/api/debug/{snap['session_id']}/state"
                       f"?wait={snap['seq']}&timeout={min(left, 2):.2f}")
        assert r.status_code == 200
        snap = r.get_json()


def line_of(snap):
    return (snap.get("location") or {}).get("line")


def names(vars_):
    return {v["name"]: v for v in vars_}


def debug_threads():
    return [t for t in threading.enumerate() if t.name.startswith("pixlang-debug-")]


def wait_threads_gone(timeout=3.0):
    deadline = time.monotonic() + timeout
    while debug_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    return not debug_threads()


# ── security ────────────────────────────────────────────────────────────────

class TestSecurity:
    ENDPOINTS = [
        ("post", "/api/debug/start", {"code": "SET a 1"}),
        ("post", "/api/debug/stop", {}),
        ("get", "/api/debug/" + "0" * 32 + "/state", None),
        ("post", "/api/debug/" + "0" * 32 + "/control", {"action": "stop"}),
        ("post", "/api/debug/" + "0" * 32 + "/breakpoints", {"lines": [1]}),
        ("post", "/api/debug/" + "0" * 32 + "/watch", {"exprs": ["1"]}),
        ("post", "/api/debug/" + "0" * 32 + "/evaluate", {"expr": "1"}),
    ]

    @pytest.mark.parametrize("method,path,body", ENDPOINTS)
    def test_token_required(self, raw, method, path, body):
        r = raw.post(path, json=body) if method == "post" else raw.get(path)
        assert r.status_code == 403

    @pytest.mark.parametrize("method,path,body", ENDPOINTS)
    def test_wrong_token(self, raw, method, path, body):
        h = {"X-PixLang-Token": "nope"}
        r = raw.post(path, json=body, headers=h) if method == "post" else raw.get(path, headers=h)
        assert r.status_code == 403

    def test_cross_origin_rejected(self, client):
        r = client.post("/api/debug/start", json={"code": "SET a 1"},
                        headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
        assert client.application.config["PIXLANG_DEBUG"].active() is None

    def test_json_only(self, client):
        r = client.post("/api/debug/start", data="code=SET+a+1",
                        content_type="application/x-www-form-urlencoded")
        assert r.status_code == 415

    def test_unknown_session(self, client):
        assert client.get("/api/debug/" + "a" * 32 + "/state").status_code == 404
        assert client.get("/api/debug/not-a-session/state").status_code == 404
        r = client.post("/api/debug/" + "a" * 32 + "/control", json={"action": "stop"})
        assert r.status_code == 404

    def test_bad_input(self, client):
        assert client.post("/api/debug/start", json={"code": 5}).status_code == 400
        r = client.post("/api/debug/start", json={"code": "SET a 1", "breakpoints": "x"})
        assert r.status_code == 400
        r = client.post("/api/debug/start", json={"code": "SET a 1", "breakpoints": [-3]})
        assert r.status_code == 400
        r = client.post("/api/debug/start", json={"code": "SET a 1", "watch": [1]})
        assert r.status_code == 400
        r = client.post("/api/debug/start", json={"code": "SET a 1", "stop_on_entry": "yes"})
        assert r.status_code == 400
        snap = start(client, FLAT_CODE)
        r = client.post(f"/api/debug/{snap['session_id']}/control", json={"action": "explode"})
        assert r.status_code == 400
        r = client.post(f"/api/debug/{snap['session_id']}/breakpoints", json={"breakpoints": [0]})
        assert r.status_code == 400

    def test_workspace_sandbox_for_include(self, client, ws):
        outside = ws.parent / "outside.pxl"
        outside.write_text("SET leaked 1\n")
        snap = start(client, 'INCLUDE "../outside.pxl"\n', path="a.pxl")
        snap = wait_for(client, snap, lambda s: s["state"] in ("finished", "error", "stopped")
                        or s["state"] == "paused")
        if snap["state"] == "paused":
            snap = ctl(client, snap, "continue")
            snap = wait_for(client, snap, lambda s: s["state"] in ("finished", "error"))
        assert snap["state"] == "error"
        assert "outside the editor workspace" in snap["result"]["error"]

    def test_non_pxl_path_ignored(self, client):
        snap = start(client, FLAT_CODE, path="../../etc/passwd")
        assert snap["state"] == "paused"          # an invalid path is ignored, not an error


# ── lifecycle ───────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_pauses_on_entry_without_breakpoints(self, client):
        snap = start(client, FLAT_CODE)
        assert snap["state"] == "paused"
        assert snap["reason"] == "entry"
        assert line_of(snap) == 1
        assert [f["name"] for f in snap["stack"]] == ["main"]
        assert snap["session_id"] and snap["seq"] >= 2

    def test_stop_on_entry_override(self, client):
        snap = start(client, FLAT_CODE, stop_on_entry=False)
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert snap["result"]["ok"] is True

    def test_breakpoint_start_does_not_stop_on_entry(self, client):
        snap = start(client, FLAT_CODE, breakpoints=[3])
        assert snap["state"] == "paused" and line_of(snap) == 3
        assert snap["reason"] == "breakpoint"

    def test_continue_to_end_returns_run_shaped_result(self, client):
        code = "ALLOC buf 8 6 1 u8 9\nUSE buf\nCHECKPOINT \"cp\"\nSET x 1\n"
        snap = start(client, code)
        snap = ctl(client, snap, "continue")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        r = snap["result"]
        for key in ("run_id", "ok", "image", "before", "checkpoints", "stats", "stdout",
                    "visited_lines", "error", "error_info", "image_b64"):
            assert key in r
        assert r["ok"] is True and r["error_info"] is None
        assert r["image"]["width"] == 8 and r["image"]["height"] == 6
        assert any(c["name"] == "cp" for c in r["checkpoints"])
        assert r["stats"]["commands_executed"] >= 1
        # the final image is served by the regular result route
        assert client.get(r["image"]["url"]).mimetype == "image/png"

    def test_state_endpoint_returns_immediately_without_wait(self, client):
        snap = start(client, FLAT_CODE)
        t0 = time.monotonic()
        r = client.get(f"/api/debug/{snap['session_id']}/state").get_json()
        assert time.monotonic() - t0 < 1.0
        assert r["seq"] == snap["seq"] and r["state"] == "paused"

    def test_long_poll_returns_on_change(self, client):
        snap = start(client, FLAT_CODE)
        seq = snap["seq"]
        out = {}

        def poll():
            out["r"] = client.application.test_client()
            c = out["r"]
            c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
            out["snap"] = c.get(f"/api/debug/{snap['session_id']}/state?wait={seq}&timeout=5").get_json()

        t = threading.Thread(target=poll)
        t.start()
        time.sleep(0.2)
        assert t.is_alive()                       # blocked: nothing changed yet
        ctl(client, snap, "step_over")
        t.join(3)
        assert not t.is_alive()
        assert out["snap"]["seq"] > seq

    def test_long_poll_timeout_returns_current_state(self, client):
        snap = start(client, FLAT_CODE)
        t0 = time.monotonic()
        r = client.get(f"/api/debug/{snap['session_id']}/state"
                       f"?wait={snap['seq']}&timeout=0.3").get_json()
        assert 0.25 <= time.monotonic() - t0 < 2.0
        assert r["seq"] == snap["seq"]

    def test_control_when_not_paused_is_conflict(self, client):
        snap = start(client, FLAT_CODE)
        snap = ctl(client, snap, "continue")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        r = client.post(f"/api/debug/{snap['session_id']}/control", json={"action": "continue"})
        assert r.status_code == 409

    def test_syntax_error_start(self, client):
        snap = start(client, "SET a\nIF\n")
        assert snap["state"] == "error"
        assert snap["result"]["error_info"]["kind"] == "syntax"
        assert snap["result"]["error_info"]["line"]
        assert not debug_threads()

    def test_finished_session_still_readable(self, client):
        snap = start(client, FLAT_CODE, stop_on_entry=False)
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        again = client.get(f"/api/debug/{snap['session_id']}/state").get_json()
        assert again["state"] == "finished" and again["result"]["ok"]
        gl = names(again["variables"]["globals"])
        assert gl["d"]["value"] == "4"

    def test_visited_lines_and_timing_exposed(self, client):
        snap = start(client, "ALLOC buf 4 4 1 u8 1\nUSE buf\nBLUR 3\nSET a 1\n", breakpoints=[4])
        assert snap["state"] == "paused"
        assert set(snap["visited_lines"]) >= {1, 2, 3, 4}
        assert snap["stats"]["commands_executed"] == 1
        assert "3" in snap["stats"]["line_times"]


# ── breakpoints ─────────────────────────────────────────────────────────────

class TestBreakpoints:
    def test_hit_in_order(self, client):
        snap = start(client, FLAT_CODE, breakpoints=[4, 2])        # unsorted on purpose
        seen = [line_of(snap)]
        for _ in range(1):
            snap = ctl(client, snap, "continue")
            seen.append(line_of(snap))
        assert seen == [2, 4]
        snap = ctl(client, snap, "continue")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert snap["state"] == "finished"

    def test_unverified_line_reported_and_never_hit(self, client):
        code = "# a comment\nSET a 1\n\nSET b 2\n"
        snap = start(client, code, breakpoints=[1, 3, 4])
        bps = {b["line"]: b for b in snap["breakpoints"]}
        assert bps[1]["verified"] is False and bps[3]["verified"] is False
        assert bps[4]["verified"] is True
        assert line_of(snap) == 4          # comment / blank lines can't pause

    def test_loop_body_breakpoint_hits_every_iteration(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[3])
        seen = []
        while snap["state"] == "paused":
            frame_vars = names(snap["variables"]["frames"][0])
            seen.append((line_of(snap), frame_vars["i"]["value"]))
            snap = ctl(client, snap, "continue")
        assert seen == [(3, "0"), (3, "1"), (3, "2")]
        assert snap["state"] == "finished"

    def test_live_add_and_remove(self, client):
        snap = start(client, FLAT_CODE)                           # paused on entry (line 1)
        r = client.post(f"/api/debug/{snap['session_id']}/breakpoints",
                        json={"lines": [3]}).get_json()
        assert [b["line"] for b in r["breakpoints"]] == [3]
        snap = ctl(client, snap, "continue")
        assert snap["state"] == "paused" and line_of(snap) == 3
        r = client.post(f"/api/debug/{snap['session_id']}/breakpoints",
                        json={"breakpoints": []}).get_json()
        assert r["breakpoints"] == []
        snap = ctl(client, snap, "continue")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert snap["state"] == "finished"

    def test_conditional_breakpoint(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[{"line": 3, "condition": "$i == 2"}])
        assert snap["state"] == "paused" and line_of(snap) == 3
        assert names(snap["variables"]["frames"][0])["i"]["value"] == "2"
        assert snap["breakpoints"][0]["condition"] == "$i == 2"
        snap = ctl(client, snap, "continue")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert snap["state"] == "finished"

    def test_condition_error_pauses_with_message(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[{"line": 3, "condition": "$nope == 1"}])
        assert snap["state"] == "paused"
        assert snap["reason"] == "condition_error"
        assert "not defined" in snap["message"] or "Undefined" in snap["message"]

    def test_condition_cannot_mutate(self, client):
        code = "SET a 1\nSET b 2\n"
        snap = start(client, code, breakpoints=[{"line": 2, "condition": "SET a 5"}])
        assert snap["reason"] == "condition_error"
        assert snap["breakpoints"][0]["error"]
        snap = ctl(client, snap, "step_over")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert names(snap["variables"]["globals"])["a"]["value"] == "1"


# ── stepping ────────────────────────────────────────────────────────────────

class TestStepping:
    def test_step_over_skips_function_call(self, client):
        snap = start(client, FUNC_CODE, breakpoints=[10])         # SET b add_two($a)
        assert line_of(snap) == 10
        snap = ctl(client, snap, "step_over")
        assert line_of(snap) == 11
        assert [f["name"] for f in snap["stack"]] == ["main"]
        assert names(snap["variables"]["globals"])["b"]["value"] == "3"

    def test_step_into_enters_function(self, client):
        snap = start(client, FUNC_CODE, breakpoints=[10])
        snap = ctl(client, snap, "step_into")
        assert line_of(snap) == 2
        assert [f["name"] for f in snap["stack"]] == ["add_two(n=1)", "main"]
        assert snap["stack"][1]["line"] == 10                       # call site
        local = names(snap["variables"]["frames"][0])
        assert local["n"]["value"] == "1"
        assert "r" not in local
        snap = ctl(client, snap, "step_into")
        assert line_of(snap) == 3
        assert names(snap["variables"]["frames"][0])["r"]["value"] == "3"

    def test_step_into_command_function(self, client):
        snap = start(client, FUNC_CODE, breakpoints=[11])          # SHOW 4
        snap = ctl(client, snap, "step_into")
        assert line_of(snap) == 6
        assert snap["stack"][0]["name"] == "SHOW(i=4)"
        assert snap["stack"][0]["kind"] == "func"

    def test_step_out_returns_to_caller(self, client):
        snap = start(client, FUNC_CODE, breakpoints=[2])
        assert [f["name"] for f in snap["stack"]] == ["add_two(n=1)", "main"]
        snap = ctl(client, snap, "step_out")
        assert [f["name"] for f in snap["stack"]] == ["main"]
        assert line_of(snap) == 11                                  # after the call finished

    def test_step_out_at_top_level_leaves_loop(self, client):
        snap = start(client, LOOP_CODE)                           # entry: line 1
        snap = ctl(client, snap, "step_into")                     # REPEAT
        snap = ctl(client, snap, "step_into")                     # first body statement
        assert line_of(snap) == 3
        snap = ctl(client, snap, "step_out")                      # runs the remaining iterations
        assert line_of(snap) == 5
        assert names(snap["variables"]["globals"])["total"]["value"] == "3"

    def test_breakpoint_wins_over_step_out(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[3])
        snap = ctl(client, snap, "step_out")
        assert line_of(snap) == 3 and snap["reason"] == "breakpoint"

    def test_step_out_at_top_level_outside_loop_runs_to_end(self, client):
        snap = start(client, FLAT_CODE)
        snap = ctl(client, snap, "step_out")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert snap["state"] == "finished"

    def test_step_over_in_loop_body_goes_to_next_iteration(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[3])
        snap = ctl(client, snap, "step_over")
        assert line_of(snap) == 3
        assert names(snap["variables"]["frames"][0])["i"]["value"] == "1"
        assert snap["reason"] == "breakpoint"                       # bp wins over "step"

    def test_step_over_loop_header_runs_whole_loop(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[2])
        snap = ctl(client, snap, "step_over")
        assert line_of(snap) == 5
        assert names(snap["variables"]["globals"])["total"]["value"] == "3"

    def test_step_into_loop_header_enters_body(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[2])
        snap = ctl(client, snap, "step_into")
        assert line_of(snap) == 3
        assert snap["reason"] == "step"

    def test_loop_iterates_with_step_into(self, client):
        snap = start(client, LOOP_CODE)                            # entry: line 1
        lines = [line_of(snap)]
        while snap["state"] == "paused":
            snap = ctl(client, snap, "step_into")
            if snap["state"] == "paused":
                lines.append(line_of(snap))
        assert lines == [1, 2, 3, 3, 3, 5]
        assert snap["state"] == "finished"

    def test_step_over_last_statement_finishes(self, client):
        snap = start(client, "SET a 1\n")
        snap = ctl(client, snap, "step_over")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert snap["state"] == "finished"

    def test_recursion_stack(self, client):
        code = ("FUNC fact(n)\n"
                "    IF $n <= 1\n"
                "        RETURN 1\n"
                "    ENDIF\n"
                "    RETURN $n * fact($n - 1)\n"
                "ENDFUNC\n"
                "SET x fact(4)\n")
        snap = start(client, code, breakpoints=[3])
        assert [f["name"] for f in snap["stack"]] == [
            "fact(n=1)", "fact(n=2)", "fact(n=3)", "fact(n=4)", "main"]
        # variables are per frame
        assert [names(v)["n"]["value"] for v in snap["variables"]["frames"][:4]] == ["1", "2", "3", "4"]

    def test_pause_running_program(self, client):
        code = "SET n 0\nREPEAT 3000000\n    SET n $n + 1\nEND\n"
        snap = start(client, code, stop_on_entry=False, breakpoints=[])
        assert snap["state"] in ("running", "paused")
        snap = ctl(client, snap, "pause")
        snap = wait_for(client, snap, lambda s: s["state"] == "paused")
        assert snap["reason"] == "pause"
        assert snap["state"] == "paused"
        ctl(client, snap, "stop")

    def test_stepping_state_is_consistent_after_many_steps(self, client):
        snap = start(client, FLAT_CODE)
        for expected in (2, 3, 4):
            snap = ctl(client, snap, "step_over")
            assert line_of(snap) == expected


# ── state exposure ──────────────────────────────────────────────────────────

class TestState:
    def test_variable_descriptions(self, client):
        code = ('SET num 7\nSET frac 2.5\nSET flag true\nSET txt "hi"\n'
                "ALLOC buf 4 3 1 u8 5\nSET ptr &buf\nSET after 1\n")
        snap = start(client, code, breakpoints=[7])
        g = names(snap["variables"]["globals"])
        assert (g["num"]["type"], g["num"]["value"], g["num"]["kind"]) == ("int", "7", "number")
        assert (g["frac"]["type"], g["frac"]["value"]) == ("float", "2.5")
        assert (g["flag"]["type"], g["flag"]["value"]) == ("bool", "true")
        assert (g["txt"]["type"], g["txt"]["value"]) == ("str", '"hi"')
        assert g["ptr"]["kind"] == "pointer" and g["ptr"]["value"].startswith("&buf")

    def test_buffers_and_thumbnails(self, client):
        code = ("ALLOC small 8 6 1 u8 20\nALLOC big 32 16 3 u8 200\nUSE big\nSET x 1\n")
        snap = start(client, code, breakpoints=[4])
        bufs = {b["name"]: b for b in snap["buffers"]}
        assert set(bufs) == {"small", "big"}
        assert bufs["small"]["shape"] == [6, 8] and bufs["small"]["dtype"] == "u8"
        assert bufs["big"]["shape"] == [16, 32, 3]
        assert bufs["small"]["new"] is True
        for b in bufs.values():
            r = client.get(b["thumb_url"])
            assert r.status_code == 200 and r.mimetype == "image/png"
        # the current image `_`
        assert snap["image"]["width"] == 32 and snap["image"]["height"] == 16
        assert client.get(snap["image"]["url"]).status_code == 200

    def test_unchanged_buffers_reuse_the_cached_thumbnail(self, client):
        code = "ALLOC a 8 8 1 u8 5\nSET x 1\nSET y 2\nALLOC b 8 8 1 u8 90\nSET z 3\n"
        snap = start(client, code, breakpoints=[2])
        first = {b["name"]: b for b in snap["buffers"]}
        snap = ctl(client, snap, "step_over")
        second = {b["name"]: b for b in snap["buffers"]}
        assert second["a"]["thumb_url"] == first["a"]["thumb_url"]
        assert second["a"]["changed"] is False
        snap = ctl(client, snap, "step_over")                    # y
        snap = ctl(client, snap, "step_over")                    # now past ALLOC b
        third = {b["name"]: b for b in snap["buffers"]}
        assert third["a"]["thumb_url"] == first["a"]["thumb_url"]
        assert third["b"]["new"] is True and third["b"]["changed"] is True

    def test_buffer_change_detected(self, client):
        code = "ALLOC a 4 4 1 u8 0\nSET marker 1\nSET a[1, 1] 200\nSET after 1\n"
        snap = start(client, code, breakpoints=[2])
        before = {b["name"]: b for b in snap["buffers"]}["a"]
        snap = ctl(client, snap, "step_over")                    # marker
        mid = {b["name"]: b for b in snap["buffers"]}["a"]
        assert mid["changed"] is False
        snap = ctl(client, snap, "step_over")                    # the POKE
        after = {b["name"]: b for b in snap["buffers"]}["a"]
        assert after["changed"] is True
        assert after["thumb_url"] != before["thumb_url"]

    def test_stdout_so_far(self, client):
        code = "ALLOC buf 4 4 1 u8 1\nUSE buf\nPRINT_INFO\nSET a 1\n"
        snap = start(client, code, breakpoints=[4])
        assert "4" in snap["stdout"]
        snap = ctl(client, snap, "continue")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert snap["result"]["stdout"].strip()

    def test_variables_remain_inspectable_after_finish(self, client):
        snap = start(client, LOOP_CODE, stop_on_entry=False)
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        g = names(snap["variables"]["globals"])
        assert g["total"]["value"] == "3" and g["done"]["value"] == "1"
        assert snap["stack"] == []

    def test_list_values_expand(self, client):
        snap = start(client, "SET v len(\"abcd\")\nSET w 1\n", breakpoints=[2])
        assert names(snap["variables"]["globals"])["v"]["value"] == "4"
        # direct unit coverage for list rendering lives below (describe_value)


# ── watch & evaluation ──────────────────────────────────────────────────────

class TestWatch:
    def test_watch_values_at_each_pause(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[3], watch=["$total + 100", "$i * 2"])
        w = {x["expr"]: x for x in snap["watch"]}
        assert w["$total + 100"]["value"] == "100" and w["$total + 100"]["ok"]
        assert w["$i * 2"]["value"] == "0"
        snap = ctl(client, snap, "continue")
        w = {x["expr"]: x for x in snap["watch"]}
        assert w["$total + 100"]["value"] == "100" and w["$i * 2"]["value"] == "2"
        snap = ctl(client, snap, "continue")
        assert {x["expr"]: x for x in snap["watch"]}["$total + 100"]["value"] == "101"

    def test_watch_errors_are_inline_and_do_not_crash(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[3],
                     watch=["$missing + 1", "1 / 0", "(", "nope", "$i"])
        w = {x["expr"]: x for x in snap["watch"]}
        assert w["$missing + 1"]["ok"] is False and w["$missing + 1"]["error"]
        assert w["1 / 0"]["ok"] is False
        assert w["("]["ok"] is False
        assert w["nope"]["ok"] is False and "not defined" in w["nope"]["error"]
        assert w["$i"]["ok"] is True
        snap = ctl(client, snap, "continue")                      # session is still healthy
        assert snap["state"] == "paused"

    @pytest.mark.parametrize("expr", [
        "SET a 5", "a = 5", "$a = 5", "LOAD \"x.png\"", "BLUR 3", "IF 1 == 1",
        "FREE buf", "ALLOC z 1 1 1 u8 0", "INCLUDE \"x.pxl\"", "GLOBAL a",
        "ZEROS(100000, 100000)",
    ])
    def test_mutations_and_commands_are_rejected(self, client, expr):
        code = "SET a 1\nALLOC buf 4 4 1 u8 7\nSET b 2\n"
        snap = start(client, code, breakpoints=[3], watch=[expr])
        w = snap["watch"][0]
        assert w["ok"] is False and w["error"]
        g = names(snap["variables"]["globals"])
        assert g["a"]["value"] == "1"
        assert [b["name"] for b in snap["buffers"]] == ["buf"]        # nothing allocated / freed
        snap = ctl(client, snap, "continue")
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        assert snap["result"]["ok"] is True

    def test_user_functions_cannot_be_called(self, client):
        code = ("GLOBAL hits\nSET hits 0\n"
                "FUNC bump()\n    GLOBAL hits\n    SET hits $hits + 1\n    RETURN $hits\nENDFUNC\n"
                "SET x 1\n")
        snap = start(client, code, breakpoints=[7], watch=["bump()"])
        assert snap["watch"][0]["ok"] is False
        assert "FUNC" in snap["watch"][0]["error"]
        assert names(snap["variables"]["globals"])["hits"]["value"] == "0"

    def test_builtin_functions_and_buffer_attrs(self, client):
        code = "ALLOC buf 8 4 1 u8 7\nSET x 1\n"
        snap = start(client, code, breakpoints=[2],
                     watch=["max(3, 9)", "buf.width", "len(\"abc\")", "buf[0, 0]", "&buf"])
        w = {x["expr"]: x for x in snap["watch"]}
        assert w["max(3, 9)"]["value"] == "9"
        assert w["buf.width"]["value"] == "8"
        assert w["len(\"abc\")"]["value"] == "3"
        assert w["buf[0, 0]"]["value"] == "7"
        assert w["&buf"]["kind"] == "pointer"

    def test_update_watch_while_paused(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[3])
        r = client.post(f"/api/debug/{snap['session_id']}/watch",
                        json={"exprs": ["$i + 40", "$i + 40", " "]}).get_json()
        assert [w["expr"] for w in r["watch"]] == ["$i + 40"]       # de-duplicated, blanks dropped
        assert r["watch"][0]["value"] == "40"
        assert r["seq"] > snap["seq"]

    def test_evaluate_endpoint(self, client):
        snap = start(client, LOOP_CODE, breakpoints=[3])
        ok = client.post(f"/api/debug/{snap['session_id']}/evaluate",
                         json={"expr": "$i + 1"}).get_json()
        assert ok["ok"] is True and ok["value"] == "1"
        bad = client.post(f"/api/debug/{snap['session_id']}/evaluate",
                          json={"expr": "SET i 9"}).get_json()
        assert bad["ok"] is False
        snap2 = client.get(f"/api/debug/{snap['session_id']}/state").get_json()
        assert names(snap2["variables"]["frames"][0])["i"]["value"] == "0"
        assert client.post(f"/api/debug/{snap['session_id']}/evaluate",
                           json={"expr": 5}).status_code == 400

    def test_evaluate_requires_paused(self, client):
        snap = start(client, FLAT_CODE, stop_on_entry=False)
        snap = wait_for(client, snap, lambda s: s["state"] == "finished")
        r = client.post(f"/api/debug/{snap['session_id']}/evaluate", json={"expr": "1"}).get_json()
        assert r["ok"] is False and "paused" in r["error"]

    def test_watch_in_function_frame_sees_locals(self, client):
        snap = start(client, FUNC_CODE, breakpoints=[3], watch=["$n * 10", "$a"])
        w = {x["expr"]: x for x in snap["watch"]}
        assert w["$n * 10"]["value"] == "10"
        assert w["$a"]["value"] == "1"        # globals are visible from the function frame


# ── stop, one-at-a-time, errors ─────────────────────────────────────────────

class TestStopAndErrors:
    def test_stop_mid_pause_frees_thread_and_lock(self, client):
        before = len(debug_threads())
        snap = start(client, FLAT_CODE)
        assert len(debug_threads()) > before
        snap = ctl(client, snap, "stop")
        assert snap["state"] == "stopped"
        assert snap["result"]["stopped"] is True and snap["result"]["error_info"] is None
        assert snap["result"]["stop_reason"] == "user"
        assert wait_threads_gone()
        # the interpreter is free again: a normal run works right away
        r = client.post("/api/run", json={"code": "ALLOC b 2 2 1 u8 1\nUSE b\n"}).get_json()
        assert r["ok"] is True

    def test_stop_keeps_last_state(self, client):
        snap = start(client, "ALLOC buf 6 4 1 u8 9\nUSE buf\nSET x 1\n", breakpoints=[3])
        snap = ctl(client, snap, "stop")
        assert snap["state"] == "stopped"
        assert snap["result"]["image"]["width"] == 6

    def test_stop_endpoint_stops_active_session(self, client):
        snap = start(client, FLAT_CODE)
        r = client.post("/api/debug/stop", json={}).get_json()
        assert r["ok"] and r["session_id"] == snap["session_id"]
        assert client.get(f"/api/debug/{snap['session_id']}/state").get_json()["state"] == "stopped"
        assert client.post("/api/debug/stop", json={}).get_json()["session_id"] is None

    def test_stop_is_idempotent(self, client):
        snap = start(client, FLAT_CODE)
        ctl(client, snap, "stop")
        assert ctl(client, snap, "stop")["state"] == "stopped"

    def test_only_one_session_at_a_time(self, client):
        a = start(client, FLAT_CODE)
        b = start(client, FLAT_CODE)
        assert a["session_id"] != b["session_id"]
        assert b["state"] == "paused"
        old = client.get(f"/api/debug/{a['session_id']}/state").get_json()
        assert old["state"] == "stopped"
        assert len([t for t in debug_threads() if "wd" not in t.name]) == 1
        # controlling the stale session is refused
        r = client.post(f"/api/debug/{a['session_id']}/control", json={"action": "continue"})
        assert r.status_code == 409

    def test_new_session_after_finished_one(self, client):
        a = start(client, FLAT_CODE, stop_on_entry=False)
        wait_for(client, a, lambda s: s["state"] == "finished")
        b = start(client, FLAT_CODE)
        assert b["state"] == "paused"

    def test_runtime_error_ends_in_error_state_with_line(self, client):
        code = ("FUNC boom(n)\n    SET q 10 / $n\n    RETURN $q\nENDFUNC\n"
                "SET z 1\nSET y boom(0)\nSET after 1\n")
        snap = start(client, code, stop_on_entry=False)
        snap = wait_for(client, snap, lambda s: s["state"] == "error")
        ei = snap["result"]["error_info"]
        assert ei["kind"] == "runtime" and ei["line"] == 2
        assert "division by zero" in snap["result"]["error"]
        assert snap["result"]["ok"] is False
        # the failing frames survive for inspection
        assert [f["name"] for f in snap["stack"]] == ["boom(n=0)", "main"]
        assert snap["stack"][0]["line"] == 2 and snap["stack"][1]["line"] == 6
        assert names(snap["variables"]["frames"][0])["n"]["value"] == "0"
        assert wait_threads_gone()

    def test_error_after_stepping(self, client):
        snap = start(client, "SET a 1\nSET b 1 / 0\nSET c 3\n")
        snap = ctl(client, snap, "step_over")
        assert line_of(snap) == 2
        snap = ctl(client, snap, "step_over")
        snap = wait_for(client, snap, lambda s: s["state"] == "error")
        assert snap["result"]["error_info"]["line"] == 2

    def test_assert_failure_reports_assert_kind(self, client):
        snap = start(client, "SET a 1\nASSERT $a == 2 \"a must be 2\"\n", stop_on_entry=False)
        snap = wait_for(client, snap, lambda s: s["state"] == "error")
        assert snap["result"]["error_info"]["kind"] == "assert"
        assert snap["result"]["error_info"]["line"] == 2

    def test_cwd_and_stdout_restored(self, client):
        import os
        import sys
        cwd, out = os.getcwd(), sys.stdout
        snap = start(client, FLAT_CODE)
        ctl(client, snap, "stop")
        assert wait_threads_gone()
        assert os.getcwd() == cwd and sys.stdout is out


# ── time limits ─────────────────────────────────────────────────────────────

class TestTimeouts:
    def test_pause_time_does_not_count_toward_run_timeout(self, ws):
        app = _make_app(ws, run_timeout=0.5)
        try:
            with app.test_client() as c:
                c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
                snap = start(c, FLAT_CODE)
                time.sleep(1.2)                         # paused for > 2x the limit
                assert c.get(f"/api/debug/{snap['session_id']}/state").get_json()["state"] == "paused"
                snap = ctl(c, snap, "continue")
                snap = wait_for(c, snap, lambda s: s["state"] == "finished")
                assert snap["result"]["ok"] is True and snap["result"]["error_info"] is None
        finally:
            app.config["PIXLANG_DEBUG"].stop_all()

    def test_running_time_is_limited(self, ws):
        app = _make_app(ws, run_timeout=0.4)
        try:
            with app.test_client() as c:
                c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
                code = "SET n 0\nREPEAT 100000000\n    SET n $n + 1\nEND\n"
                snap = start(c, code, stop_on_entry=False)
                snap = wait_for(c, snap, lambda s: s["state"] == "error", timeout=10)
                assert snap["result"]["error_info"]["kind"] == "timeout"
                assert wait_threads_gone()
                # lock released: the editor is usable again
                r = c.post("/api/run", json={"code": "SET a 1\n"}).get_json()
                assert r["ok"] is True
        finally:
            app.config["PIXLANG_DEBUG"].stop_all()

    def test_paused_time_resets_budget_between_steps(self, ws):
        app = _make_app(ws, run_timeout=0.6)
        try:
            with app.test_client() as c:
                c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
                snap = start(c, FLAT_CODE)
                for _ in range(3):
                    time.sleep(0.4)
                    snap = ctl(c, snap, "step_over")
                    assert snap["state"] == "paused"    # 1.2 s elapsed overall; still alive
        finally:
            app.config["PIXLANG_DEBUG"].stop_all()

    def test_idle_session_is_stopped_and_frees_resources(self, ws):
        app = _make_app(ws, debug_idle_timeout=0.4)
        try:
            with app.test_client() as c:
                c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
                snap = start(c, FLAT_CODE)
                assert snap["state"] == "paused"
                snap = wait_for(c, snap, lambda s: s["state"] == "stopped", timeout=6)
                assert snap["result"]["stop_reason"] == "idle"
                assert snap["result"]["error_info"]["kind"] == "idle"
                assert wait_threads_gone()
                r = c.post("/api/run", json={"code": "SET a 1\n"}).get_json()
                assert r["ok"] is True
        finally:
            app.config["PIXLANG_DEBUG"].stop_all()

    def test_activity_restarts_the_idle_clock(self, ws):
        app = _make_app(ws, debug_idle_timeout=0.7)
        try:
            with app.test_client() as c:
                c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
                snap = start(c, FLAT_CODE)
                for _ in range(3):
                    time.sleep(0.4)
                    snap = ctl(c, snap, "step_over")
                assert snap["state"] == "paused"
        finally:
            app.config["PIXLANG_DEBUG"].stop_all()


# ── executor hooks & helpers (unit) ─────────────────────────────────────────

class _Recorder:
    """Stand-in session: records (line, call depth, step depth) per statement."""

    def __init__(self):
        self.seen = []
        self.stop_reason = None

    def on_statement(self, ex, stmt):
        self.seen.append((stmt.line, len(ex._frames), ex.step_depth))


def _run_recorded(code):
    from pixlang.commands import registry
    from pixlang.parser import parse
    rec = _Recorder()
    ex = dbgmod.DebugExecutor(registry, session=rec)
    ex.run(parse(code))
    return rec.seen, ex


class TestDebugExecutor:
    def test_step_depth_counts_loops_and_functions(self):
        seen, _ = _run_recorded(FUNC_CODE + "REPEAT 2\n    SET q 1\nEND\n")
        by_line = {}
        for line, frames, depth in seen:
            by_line.setdefault(line, set()).add((frames, depth))
        assert by_line[10] == {(1, 0)}                 # call statement in main
        assert by_line[2] == {(2, 1)}                  # first statement inside add_two
        assert by_line[6] == {(2, 1)}                  # inside SHOW
        assert by_line[14] == {(1, 1)}                 # loop body: same frame, deeper block

    def test_depth_returns_to_zero_after_error(self):
        from pixlang.commands import registry
        from pixlang.parser import parse
        rec = _Recorder()
        ex = dbgmod.DebugExecutor(registry, session=rec)
        with pytest.raises(RuntimeError):
            ex.run(parse("REPEAT 2\n    SET a 1 / 0\nEND\n"))
        assert ex.step_depth == 0
        assert ex.error_frames and ex.error_node.line == 2

    def test_evaluation_does_not_record_errors(self):
        from pixlang.commands import registry
        from pixlang.parser import parse
        rec = _Recorder()
        ex = dbgmod.DebugExecutor(registry, session=rec)
        ex.run(parse("SET a 1\n"))
        with pytest.raises(PixRuntimeError):
            ex.evaluate(dbgmod.parse_safe_expression("$nope + 1"))
        assert ex.error_frames is None                 # probing must not look like a crash

    def test_capturing_executor_unchanged(self):
        from pixlang.commands import registry
        from pixlang.editor.server import _CapturingExecutor
        from pixlang.parser import parse
        ex = _CapturingExecutor(registry)
        ex.run(parse("ALLOC b 4 4 1 u8 3\nUSE b\n"))
        assert ex.final_image is not None and ex.visited == {1, 2}


class TestHelpers:
    def test_normalize_breakpoints(self):
        assert dbgmod.normalize_breakpoints([3, {"line": 1, "condition": " $x > 1 "}, 3]) == \
            [(1, "$x > 1"), (3, None)]
        assert dbgmod.normalize_breakpoints(None) == []
        for bad in ("3", [True], [0], [1.5], [{"line": "2"}], [{"line": 1, "condition": 5}],
                    list(range(1, 600))):
            with pytest.raises(dbgmod.DebugError):
                dbgmod.normalize_breakpoints(bad)

    def test_normalize_watch(self):
        assert dbgmod.normalize_watch([" a ", "a", "", "b"]) == ["a", "b"]
        with pytest.raises(dbgmod.DebugError):
            dbgmod.normalize_watch([1])
        with pytest.raises(dbgmod.DebugError):
            dbgmod.normalize_watch(["x"] * 1 + [str(i) for i in range(100)])

    def test_parse_safe_expression_accepts_expressions(self):
        for ok in ("$a + 1", "max(1, 2)", "buf[0, 1]", "*p", "&buf", "_.width", "a == 3 and b"):
            assert dbgmod.parse_safe_expression(ok) is not None

    def test_parse_safe_expression_rejects_the_rest(self):
        for bad in ("", "  ", "SET a 1", "a = 1", "LOAD \"x\"", "x\ny", "ZEROS(1, 1)", "(" ):
            with pytest.raises(dbgmod.WatchError):
                dbgmod.parse_safe_expression(bad)

    def test_describe_value_shapes(self):
        import numpy as np
        d = dbgmod.describe_value("x", [1, 2, 3])
        assert d["kind"] == "list" and d["length"] == 3 and len(d["children"]) == 3
        assert d["children"][0]["name"] == "[0]" and d["value"] == "[1, 2, 3]"
        big = dbgmod.describe_value("big", list(range(100)))
        assert len(big["children"]) == dbgmod.MAX_CHILDREN and big["truncated"] is True
        nested = dbgmod.describe_value("n", [[1, [2, [3]]]])
        assert nested["children"][0]["kind"] == "list"
        arr = dbgmod.describe_value("a", np.zeros((4, 5, 3), dtype=np.uint8))
        assert arr["kind"] == "buffer" and arr["value"] == "4×5×3 · u8" and arr["type"] == "buffer(u8)"
        assert dbgmod.describe_value("s", "a\nb")["value"] == '"a\\nb"'
        assert dbgmod.describe_value("l", "x" * 500)["value"].endswith('…')
        assert dbgmod.describe_value("f", np.float32(0.5))["value"] == "0.5"
        assert dbgmod.describe_value("z", None)["value"] == "none"
        assert dbgmod.describe_value("o", object())["kind"] == "other"
