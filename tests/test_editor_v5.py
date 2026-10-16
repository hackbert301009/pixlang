# tests/test_editor_v5.py
"""Editor v5: security hardening, language metadata, structured errors,
files API, result images, generic flow tree."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")

from pixlang.editor import server as srv  # noqa: E402
from pixlang.editor.server import create_app  # noqa: E402

TOKEN = "test-token-123"


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def app(ws):
    a = create_app(ws, token=TOKEN, run_timeout=5.0)
    a.config["TESTING"] = True
    return a


@pytest.fixture
def client(app):
    with app.test_client() as c:
        c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
        yield c


@pytest.fixture
def raw(app):
    """Client without the token header."""
    with app.test_client() as c:
        yield c


def _img(path: Path, shape=(30, 40, 3)):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full(shape, 90, dtype=np.uint8))
    return path


# ── security ────────────────────────────────────────────────────────────────

class TestSecurity:
    def test_index_injects_token(self, raw):
        r = raw.get("/")
        assert r.status_code == 200
        assert TOKEN.encode() in r.data
        assert b"{{PIXLANG_TOKEN}}" not in r.data

    def test_missing_token_rejected(self, raw):
        r = raw.post("/api/run", json={"code": ""})
        assert r.status_code == 403

    def test_wrong_token_rejected(self, raw):
        r = raw.post("/api/run", json={"code": ""}, headers={"X-PixLang-Token": "nope"})
        assert r.status_code == 403

    def test_get_api_requires_token(self, raw):
        assert raw.get("/api/files").status_code == 403

    def test_text_plain_rejected(self, client):
        r = client.post("/api/run", data='{"code": ""}', content_type="text/plain")
        assert r.status_code == 415

    def test_form_post_rejected(self, client):
        r = client.post("/api/files/save", data={"path": "x.pxl", "code": "X"})
        assert r.status_code == 415

    def test_bad_origin_rejected(self, client):
        r = client.post("/api/run", json={"code": ""},
                        headers={"Origin": "http://evil.example"})
        assert r.status_code == 403

    def test_null_origin_rejected(self, client):
        r = client.post("/api/run", json={"code": ""}, headers={"Origin": "null"})
        assert r.status_code == 403

    def test_local_origin_allowed(self, client):
        r = client.post("/api/run", json={"code": ""},
                        headers={"Origin": "http://127.0.0.1:7478"})
        assert r.status_code == 200

    def test_cross_site_fetch_rejected(self, client):
        r = client.post("/api/run", json={"code": ""},
                        headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403

    def test_dns_rebinding_host_rejected(self, client):
        r = client.get("/", headers={"Host": "attacker.example:7478"})
        assert r.status_code == 403
        r = client.post("/api/run", json={"code": ""}, headers={"Host": "evil.test"})
        assert r.status_code == 403

    def test_path_escape_rejected(self, client, ws):
        for bad in ("../outside.pxl", "/etc/passwd", "sub/../../x.pxl", "a.txt"):
            r = client.post("/api/files/save", json={"path": bad, "code": "X"})
            assert r.status_code == 400, bad
            r = client.post("/api/files/read", json={"path": bad})
            assert r.status_code == 404, bad
        assert not (ws.parent / "outside.pxl").exists()

    def test_include_outside_workspace_does_not_leak(self, client, tmp_path):
        secret = tmp_path / "secret.pxl"
        secret.write_text("TOPSECRET_VALUE here\n")
        r = client.post("/api/run", json={"code": f'INCLUDE "{secret}"\n'})
        d = r.get_json()
        assert d["error"]
        assert "TOPSECRET" not in r.get_data(as_text=True)
        r = client.post("/api/run", json={"code": 'INCLUDE "/etc/hostname"\n'})
        assert "outside" in r.get_json()["error"] or "pxl" in r.get_json()["error"]

    def test_oversized_body_rejected(self, client):
        r = client.post("/api/lint", json={"code": "#" * (3 * 1024 * 1024)})
        assert r.status_code == 413

    def test_security_headers(self, client):
        r = client.get("/")
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
        assert r.headers["X-Content-Type-Options"] == "nosniff"


class TestTimeout:
    def test_run_timeout_reported(self, ws, monkeypatch):
        from pixlang.commands import registry as base
        from pixlang.commands.registry import CommandRegistry
        reg = CommandRegistry()
        reg._commands = dict(base._commands)

        @reg.register("SLOW_TEST")
        def _slow(image):
            time.sleep(0.3)
            return image

        a = create_app(ws, token=TOKEN, run_timeout=0.5, registry=reg, plugins=False)
        with a.test_client() as c:
            c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
            code = "REPEAT 50\nSLOW_TEST\nEND\n"
            t0 = time.time()
            d = c.post("/api/run", json={"code": code}).get_json()
            assert time.time() - t0 < 3
            assert d["error_info"]["kind"] == "timeout"
            assert d["ok"] is False
            # the worker stops cooperatively, so the next run works
            time.sleep(0.5)
            d2 = c.post("/api/run", json={"code": ""}).get_json()
            assert d2["error"] is None


# ── language / commands ─────────────────────────────────────────────────────

class TestLanguage:
    def test_language_payload(self, client):
        d = client.get("/api/language").get_json()
        from pixlang.parser.lexer import KEYWORDS
        assert set(d["keywords"]) == set(KEYWORDS)
        cmds = {c["name"]: c for c in d["commands"]}
        assert "LOAD" in cmds
        for f in ("usage", "params", "category", "summary", "doc"):
            assert f in cmds["RESIZE"]
        assert cmds["RESIZE"]["usage"].startswith("RESIZE")
        assert "==" in d["operators"]

    def test_usage_has_no_description(self, client):
        for c in client.get("/api/language").get_json()["commands"]:
            assert " — " not in c["usage"], c["name"]

    def test_commands_example_is_runnable_usage(self, client):
        cmds = {c["name"]: c for c in client.get("/api/commands").get_json()["commands"]}
        assert cmds["GRAYSCALE"]["example"] == "GRAYSCALE"
        assert "—" not in cmds["LOAD"]["example"]


# ── structured errors / run feedback ────────────────────────────────────────

class TestRunFeedback:
    def test_parse_error_line(self, client):
        d = client.post("/api/run", json={"code": 'LOAD "a.png"\nGRAYSCALE\nSET 5 5\n'}).get_json()
        assert d["error_info"]["kind"] == "syntax"
        assert d["error_info"]["line"] == 3
        assert d["error_info"]["col"] >= 1

    def test_lint_parse_error_has_line(self, client):
        d = client.post("/api/lint", json={"code": "GRAYSCALE\nIF $$$\n"}).get_json()
        assert d["issues"][0]["line"] == 2

    def test_runtime_error_partial_image(self, client, ws):
        _img(ws / "in.png")
        code = 'LOAD "in.png"\nGRAYSCALE\nLOAD "missing.png"\nSAVE "x.png"\n'
        d = client.post("/api/run", json={"code": code, "path": "p.pxl"}).get_json()
        assert d["ok"] is False
        assert d["error_info"]["kind"] == "runtime"
        assert d["error_info"]["line"] == 3
        assert d["partial"] is True
        assert d["image"]["width"] == 40
        assert 1 in d["visited_lines"] and 4 not in d["visited_lines"]

    def test_relative_paths_resolve_against_file_dir(self, client, ws):
        _img(ws / "sub" / "in.png")
        code = 'LOAD "in.png"\nSAVE "out/res.png"\n'
        d = client.post("/api/run", json={"code": code, "path": "sub/p.pxl"}).get_json()
        assert d["error"] is None, d["error"]
        assert (ws / "sub" / "out" / "res.png").exists()

    def test_result_images_served(self, client, ws):
        _img(ws / "in.png")
        code = 'LOAD "in.png"\nCHECKPOINT "a"\nGRAYSCALE\n'
        d = client.post("/api/run", json={"code": code}).get_json()
        assert d["before"]["channels"] == 3
        assert d["image"]["channels"] == 1
        assert d["checkpoints"][0]["name"] == "a"
        r = client.get(d["image"]["url"])
        assert r.status_code == 200 and r.data[:4] == b"\x89PNG"
        r = client.get(d["checkpoints"][0]["thumb_url"])
        assert r.status_code == 200
        assert client.get("/api/result/" + "0" * 32 + "/final.png").status_code == 404

    def test_line_times(self, client, ws):
        _img(ws / "in.png")
        d = client.post("/api/run", json={"code": 'LOAD "in.png"\n\nBLUR 3\n'}).get_json()
        lt = d["stats"]["line_times"]
        assert lt["3"]["name"] == "BLUR" and lt["3"]["count"] == 1

    def test_default_template_runs(self, client, ws):
        code = client.post("/api/template", json={"path": "new/pipe.pxl"}).get_json()["code"]
        d = client.post("/api/run", json={"code": code, "path": "new/pipe.pxl"}).get_json()
        if not srv.SAMPLE_IMAGE.exists():
            pytest.skip("sample image not bundled")
        assert d["error"] is None, d["error"]
        assert (ws / "new" / "out" / "result.png").exists()


# ── files ───────────────────────────────────────────────────────────────────

class TestFiles:
    def test_save_list_read(self, client, ws):
        r = client.post("/api/files/save", json={"path": "a/b.pxl", "code": "GRAYSCALE\n"})
        assert r.get_json()["path"] == "a/b.pxl"
        files = client.get("/api/files").get_json()["files"]
        assert [f["path"] for f in files] == ["a/b.pxl"]
        d = client.post("/api/files/read", json={"path": "a/b.pxl"}).get_json()
        assert d["code"] == "GRAYSCALE\n"

    def test_create_refuses_overwrite(self, client):
        client.post("/api/files/save", json={"path": "x.pxl", "code": "A"})
        r = client.post("/api/files/save", json={"path": "x.pxl", "code": "B", "create": True})
        assert r.status_code == 409


# ── flow tree ───────────────────────────────────────────────────────────────

class TestFlowTree:
    def test_labels_have_no_python_repr(self, client):
        code = 'SET w 640\nSET m "x"\nRESIZE $w h\nIF m == "x"\nBLUR 3\nENDIF\n'
        d = client.post("/api/parse", json={"code": code}).get_json()
        text = str(d["tree"]) + str(d["nodes"])
        assert "VarRef" not in text and "BareIdent" not in text
        resize = d["tree"][2]
        assert resize["args"] == "$w h"
        assert d["tree"][1]["args"] == 'm = "x"'

    def test_nested_blocks(self, client):
        code = "REPEAT 2\nIF a == 1\nBLUR 3\nENDIF\nEND\n"
        tree = client.post("/api/parse", json={"code": code}).get_json()["tree"]
        rep = tree[0]
        assert rep["kind"] == "block" and rep["loop"]
        inner = rep["branches"][0]["items"][0]
        assert inner["type"] == "if"
        assert [b["label"] for b in inner["branches"]] == ["then", "else"]
        assert inner["branches"][0]["items"][0]["name"] == "BLUR"

    def test_generic_unknown_node(self):
        from dataclasses import dataclass, field
        from typing import Any, List

        from pixlang.parser.ast_nodes import ASTNode, Command, VarRef

        @dataclass
        class WhileBlock(ASTNode):
            cond: Any = None
            body: List[Any] = field(default_factory=list)

        @dataclass
        class BinOp(ASTNode):
            op: str = "<"
            left: Any = None
            right: Any = None

        node = WhileBlock(line=1, cond=BinOp(op="<", left=VarRef(var_name="i"), right=10),
                          body=[Command(name="BLUR", args=[3], line=2)])
        item = srv.FlowBuilder().tree([node])[0]
        assert item["name"] == "WHILE"
        assert item["args"] == "$i < 10"
        assert item["loop"] is True
        assert item["branches"][0]["items"][0]["name"] == "BLUR"

    def test_fmt_value(self):
        from pixlang.parser.ast_nodes import BareIdent, VarRef
        assert srv.fmt_value(VarRef(var_name="x")) == "$x"
        assert srv.fmt_value(BareIdent(name="jet")) == "jet"
        assert srv.fmt_value('a"b') == '"a\\"b"'
        assert srv.fmt_value(1.5) == "1.5"


class TestErrorInfoContract:
    def test_uses_attributes_when_present(self):
        class PixSyntaxError(SyntaxError):
            pass
        e = PixSyntaxError("bad")
        e.lineno, e.offset = 4, 7
        info = srv.error_info(e, "a\nb\nc\nd\n", "syntax")
        assert (info["line"], info["col"]) == (4, 7)

        class PixRuntimeError(RuntimeError):
            pass
        e2 = PixRuntimeError("boom")
        e2.line, e2.col, e2.file = 2, 3, "x.pxl"
        info = srv.error_info(e2, "a\nb\n", "runtime")
        assert (info["line"], info["col"], info["file"]) == (2, 3, "x.pxl")

    def test_regex_fallback(self):
        info = srv.error_info(RuntimeError("[X] failed (line 5)"), "\n" * 6, "runtime")
        assert info["line"] == 5
