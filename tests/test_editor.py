# tests/test_editor.py
"""
Test suite for the PixLang visual editor API.
Uses the Flask test client — no real HTTP server, all in-process.

All tests are skipped gracefully when Flask is not installed.
"""
import base64
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

flask = pytest.importorskip("flask")

# Patch WORKSPACE_FILE before importing server so tests never touch ~/.pixlang…
import pixlang.editor.server as _srv  # noqa: E402 (import after importorskip)
from pixlang.editor.server import app, EXAMPLES_DIR  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(_srv, "WORKSPACE_FILE", tmp_path / "test_workspace.json")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_image(tmp_path, name="img.png", shape=(40, 40, 3)):
    import cv2
    p = tmp_path / name
    cv2.imwrite(str(p), np.zeros(shape, dtype=np.uint8))
    return p


# ══════════════════════════════════════════════════════════════════════════════
# 1. Serve index
# ══════════════════════════════════════════════════════════════════════════════

class TestServeIndex:

    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_returns_html(self, client):
        r = client.get("/")
        assert b"PixLang" in r.data
        assert b"<!DOCTYPE html>" in r.data.lower() or b"<!doctype html>" in r.data.lower()

    def test_static_css_served(self, client):
        r = client.get("/static/css/editor.css")
        assert r.status_code == 200
        assert b"toolbar" in r.data.lower() or b"#toolbar" in r.data

    def test_static_js_app_served(self, client):
        r = client.get("/static/js/app.js")
        assert r.status_code == 200
        assert b"runPipeline" in r.data or b"boot" in r.data


# ══════════════════════════════════════════════════════════════════════════════
# 2. /api/run
# ══════════════════════════════════════════════════════════════════════════════

class TestRunEndpoint:

    def test_response_has_required_fields(self, client):
        r = client.post("/api/run", json={"code": "", "filename": "t.pxl"})
        assert r.status_code == 200
        data = r.get_json()
        for field in ("image_b64", "stdout", "stats", "error"):
            assert field in data, f"Missing field: {field}"

    def test_empty_pipeline_no_crash(self, client):
        r = client.post("/api/run", json={"code": "", "filename": "t.pxl"})
        data = r.get_json()
        assert data["error"] is None
        assert data["image_b64"] is None

    def test_syntax_error_returns_error_field(self, client):
        r = client.post("/api/run", json={"code": "$$$invalid", "filename": "t.pxl"})
        data = r.get_json()
        assert data["error"] is not None
        assert data["image_b64"] is None

    def test_missing_file_returns_error_not_crash(self, client):
        code = 'LOAD "/nonexistent/path/img.png"\nGRAYSCALE\n'
        r = client.post("/api/run", json={"code": code, "filename": "t.pxl"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["error"] is not None

    def test_valid_pipeline_returns_image_b64(self, client, tmp_path):
        img_path = _make_image(tmp_path)
        code = f'LOAD "{img_path}"\nGRAYSCALE\n'
        r = client.post("/api/run", json={"code": code, "filename": "t.pxl"})
        data = r.get_json()
        assert data["error"] is None
        assert data["image_b64"] is not None
        decoded = base64.b64decode(data["image_b64"])
        assert decoded[:4] == b"\x89PNG", "image_b64 must be a valid PNG"

    def test_valid_pipeline_stats_structure(self, client, tmp_path):
        img_path = _make_image(tmp_path)
        code = f'LOAD "{img_path}"\nGRAYSCALE\n'
        r = client.post("/api/run", json={"code": code, "filename": "t.pxl"})
        stats = r.get_json()["stats"]
        assert "commands_executed" in stats
        assert stats["commands_executed"] >= 2
        assert "total_ms" in stats
        assert isinstance(stats["total_ms"], float)

    def test_stdout_captured(self, client, tmp_path):
        img_path = _make_image(tmp_path)
        code = f'LOAD "{img_path}"\nPRINT_INFO\n'
        r = client.post("/api/run", json={"code": code, "filename": "t.pxl"})
        data = r.get_json()
        assert "shape=" in data["stdout"] or len(data["stdout"]) > 0

    def test_grayscale_image_encoded_as_bgr_png(self, client, tmp_path):
        img_path = _make_image(tmp_path)
        code = f'LOAD "{img_path}"\nGRAYSCALE\n'
        r = client.post("/api/run", json={"code": code, "filename": "t.pxl"})
        data = r.get_json()
        assert data["image_b64"] is not None
        # Must still be a valid PNG (grayscale→BGR conversion in server)
        decoded = base64.b64decode(data["image_b64"])
        assert decoded[:4] == b"\x89PNG"

    def test_missing_json_body_no_crash(self, client):
        r = client.post("/api/run", data=b"", content_type="application/json")
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# 3. /api/lint
# ══════════════════════════════════════════════════════════════════════════════

class TestLintEndpoint:

    def test_clean_pipeline_zero_issues(self, client):
        code = 'LOAD "img.png"\nGRAYSCALE\nSAVE "out.png"\n'
        r = client.post("/api/lint", json={"code": code})
        data = r.get_json()
        assert data["issues"] == []

    def test_px001_no_load(self, client):
        r = client.post("/api/lint", json={"code": "GRAYSCALE\n"})
        codes = [i["code"] for i in r.get_json()["issues"]]
        assert "PX001" in codes

    def test_px002_no_save(self, client):
        r = client.post("/api/lint", json={"code": 'LOAD "x.png"\nGRAYSCALE\n'})
        codes = [i["code"] for i in r.get_json()["issues"]]
        assert "PX002" in codes

    def test_px010_threshold_before_grayscale(self, client):
        code = 'LOAD "x.png"\nTHRESHOLD 128\nGRAYSCALE\nSAVE "o.png"\n'
        r = client.post("/api/lint", json={"code": code})
        codes = [i["code"] for i in r.get_json()["issues"]]
        assert "PX010" in codes

    def test_issue_has_all_required_fields(self, client):
        r = client.post("/api/lint", json={"code": "GRAYSCALE\n"})
        issue = r.get_json()["issues"][0]
        for field in ("line", "severity", "code", "message"):
            assert field in issue

    def test_severity_values_are_valid(self, client):
        r = client.post("/api/lint", json={"code": "GRAYSCALE\n"})
        for issue in r.get_json()["issues"]:
            assert issue["severity"] in ("info", "warning", "error")

    def test_syntax_error_returns_parse_issue(self, client):
        r = client.post("/api/lint", json={"code": "IF $$$\n"})
        data = r.get_json()
        assert len(data["issues"]) > 0
        assert data["issues"][0]["severity"] == "error"

    def test_empty_code_returns_issues_list(self, client):
        r = client.post("/api/lint", json={"code": ""})
        assert r.status_code == 200
        assert "issues" in r.get_json()


# ══════════════════════════════════════════════════════════════════════════════
# 4. /api/parse
# ══════════════════════════════════════════════════════════════════════════════

class TestParseEndpoint:

    def test_empty_code_returns_empty_graph(self, client):
        r = client.post("/api/parse", json={"code": ""})
        data = r.get_json()
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_simple_commands_produce_nodes(self, client):
        r = client.post("/api/parse", json={"code": 'LOAD "x.png"\nGRAYSCALE\n'})
        data = r.get_json()
        assert len(data["nodes"]) >= 2
        assert len(data["edges"]) >= 1

    def test_node_has_required_fields(self, client):
        r = client.post("/api/parse", json={"code": 'LOAD "x.png"\n'})
        node = r.get_json()["nodes"][0]
        for f in ("id", "type", "label", "line", "category"):
            assert f in node, f"Missing node field: {f}"

    def test_edges_reference_valid_node_ids(self, client):
        r = client.post("/api/parse",
                        json={"code": 'LOAD "x.png"\nGRAYSCALE\nBLUR 5\n'})
        data = r.get_json()
        ids = {n["id"] for n in data["nodes"]}
        for edge in data["edges"]:
            assert edge["from"] in ids
            assert edge["to"]   in ids

    def test_if_block_creates_header_and_footer(self, client):
        code = 'SET x 1\nIF x == 1\nGRAYSCALE\nENDIF\n'
        data = client.post("/api/parse", json={"code": code}).get_json()
        types = [n["type"] for n in data["nodes"]]
        assert "if_header" in types
        assert "if_footer" in types

    def test_repeat_block_creates_header_and_footer(self, client):
        code = 'REPEAT 3\nBLUR 5\nEND\n'
        data = client.post("/api/parse", json={"code": code}).get_json()
        types = [n["type"] for n in data["nodes"]]
        assert "repeat_header" in types
        assert "repeat_footer" in types

    def test_roi_block(self, client):
        code = 'LOAD "x.png"\nROI 0 0 100 100\nGRAYSCALE\nROI_RESET\n'
        data = client.post("/api/parse", json={"code": code}).get_json()
        types = [n["type"] for n in data["nodes"]]
        assert "roi_header" in types
        assert "roi_footer" in types

    def test_setvar_node(self, client):
        data = client.post("/api/parse", json={"code": "SET width 640\n"}).get_json()
        assert any(n["type"] == "setvar" for n in data["nodes"])

    def test_include_node(self, client):
        data = client.post("/api/parse",
                           json={"code": 'INCLUDE "sub.pxl"\n'}).get_json()
        assert any(n["type"] == "include" for n in data["nodes"])

    def test_assert_node(self, client):
        data = client.post("/api/parse",
                           json={"code": "ASSERT width == 640\n"}).get_json()
        assert any(n["type"] == "assert" for n in data["nodes"])

    def test_syntax_error_returns_error_key(self, client):
        r = client.post("/api/parse", json={"code": "IF $$$\n"})
        data = r.get_json()
        assert "error" in data

    def test_command_node_category_set(self, client):
        r = client.post("/api/parse", json={"code": 'LOAD "x.png"\n'})
        node = r.get_json()["nodes"][0]
        assert node["category"] != ""

    def test_node_ids_are_unique(self, client):
        r = client.post("/api/parse",
                        json={"code": 'LOAD "x.png"\nGRAYSCALE\nBLUR 5\nSAVE "o.png"\n'})
        ids = [n["id"] for n in r.get_json()["nodes"]]
        assert len(ids) == len(set(ids))


# ══════════════════════════════════════════════════════════════════════════════
# 5. /api/commands
# ══════════════════════════════════════════════════════════════════════════════

class TestCommandsEndpoint:

    def test_returns_200(self, client):
        assert client.get("/api/commands").status_code == 200

    def test_returns_35_commands(self, client):
        data = client.get("/api/commands").get_json()
        assert len(data["commands"]) == 35

    def test_command_has_required_fields(self, client):
        cmd = client.get("/api/commands").get_json()["commands"][0]
        for f in ("name", "category", "signature", "doc", "example"):
            assert f in cmd

    def test_commands_sorted_alphabetically(self, client):
        names = [c["name"] for c in client.get("/api/commands").get_json()["commands"]]
        assert names == sorted(names)

    def test_load_in_io_category(self, client):
        cmds = {c["name"]: c for c in client.get("/api/commands").get_json()["commands"]}
        assert cmds["LOAD"]["category"] == "IO"

    def test_all_categories_valid(self, client):
        valid = {"IO", "Geometry", "Color", "Threshold", "Filter",
                 "Morphology", "Analysis", "Composition", "Annotation", "Other"}
        for cmd in client.get("/api/commands").get_json()["commands"]:
            assert cmd["category"] in valid, f"{cmd['name']} has invalid category"

    def test_signature_contains_command_name(self, client):
        for cmd in client.get("/api/commands").get_json()["commands"]:
            assert cmd["name"] in cmd["signature"], \
                f"{cmd['name']} signature missing name"


# ══════════════════════════════════════════════════════════════════════════════
# 6. /api/examples
# ══════════════════════════════════════════════════════════════════════════════

class TestExamplesEndpoint:

    def test_returns_12_examples(self, client):
        data = client.get("/api/examples").get_json()
        assert len(data["examples"]) == 12

    def test_example_has_name_and_description(self, client):
        ex = client.get("/api/examples").get_json()["examples"][0]
        assert "name"        in ex
        assert "description" in ex

    def test_get_example_returns_code(self, client):
        r = client.get("/api/examples/edge_detection")
        assert r.status_code == 200
        data = r.get_json()
        assert "code" in data
        assert len(data["code"]) > 10

    def test_all_12_examples_load_successfully(self, client):
        examples = client.get("/api/examples").get_json()["examples"]
        for ex in examples:
            r = client.get(f"/api/examples/{ex['name']}")
            assert r.status_code == 200, f"Failed: {ex['name']}"
            assert len(r.get_json()["code"]) > 0

    def test_examples_contain_load_command(self, client):
        examples = client.get("/api/examples").get_json()["examples"]
        for ex in examples:
            code = client.get(f"/api/examples/{ex['name']}").get_json()["code"]
            assert "LOAD" in code, f"{ex['name']} has no LOAD command"

    def test_unknown_example_returns_404(self, client):
        r = client.get("/api/examples/ghost_pipeline_xyz")
        assert r.status_code == 404

    def test_path_traversal_blocked(self, client):
        r = client.get("/api/examples/../../cli")
        assert r.status_code in (400, 404)


# ══════════════════════════════════════════════════════════════════════════════
# 7. /api/workspace
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceEndpoint:

    def test_load_when_no_file_returns_null(self, client):
        data = client.get("/api/workspace/load").get_json()
        assert data["workspace"] is None

    def test_save_returns_ok(self, client):
        r = client.post("/api/workspace/save",
                        json={"workspace": {"code": "LOAD 'x.png'\n"}})
        assert r.get_json()["ok"] is True

    def test_save_and_load_roundtrip(self, client):
        ws = {"code": "GRAYSCALE\n", "filename": "test.pxl", "cursor": {"from": 0}}
        client.post("/api/workspace/save", json={"workspace": ws})
        loaded = client.get("/api/workspace/load").get_json()["workspace"]
        assert loaded["code"]     == ws["code"]
        assert loaded["filename"] == ws["filename"]

    def test_overwrite_keeps_latest(self, client):
        client.post("/api/workspace/save", json={"workspace": {"code": "v1"}})
        client.post("/api/workspace/save", json={"workspace": {"code": "v2"}})
        loaded = client.get("/api/workspace/load").get_json()["workspace"]
        assert loaded["code"] == "v2"

    def test_workspace_file_is_valid_json(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(_srv, "WORKSPACE_FILE", tmp_path / "ws.json")
        ws = {"code": "BLUR 5\n", "x": 42}
        client.post("/api/workspace/save", json={"workspace": ws})
        raw = (tmp_path / "ws.json").read_text()
        parsed = json.loads(raw)
        assert parsed["code"] == ws["code"]


# ══════════════════════════════════════════════════════════════════════════════
# 8. _CapturingExecutor + _image_to_b64
# ══════════════════════════════════════════════════════════════════════════════

class TestCapturingExecutor:

    def test_final_image_none_on_empty_pipeline(self):
        from pixlang.editor.server import _CapturingExecutor
        from pixlang.commands import registry
        from pixlang.parser import parse
        ex = _CapturingExecutor(registry=registry)
        ex.run(parse(""))
        assert ex.final_image is None

    def test_final_image_captured_after_grayscale(self, tmp_path):
        from pixlang.commands.builtin import _unwrap
        from pixlang.editor.server import _CapturingExecutor
        from pixlang.commands import registry
        from pixlang.parser import parse
        img_path = _make_image(tmp_path, shape=(20, 20, 3))
        ex = _CapturingExecutor(registry=registry)
        ex.run(parse(f'LOAD "{img_path}"\nGRAYSCALE\n'))
        assert ex.final_image is not None
        img, _ = _unwrap(ex.final_image)
        assert img.shape == (20, 20)

    def test_image_to_b64_bgr_image(self):
        from pixlang.editor.server import _image_to_b64
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        b64 = _image_to_b64(img)
        assert b64 is not None
        assert base64.b64decode(b64)[:4] == b"\x89PNG"

    def test_image_to_b64_grayscale(self):
        from pixlang.editor.server import _image_to_b64
        img = np.zeros((10, 10), dtype=np.uint8)
        b64 = _image_to_b64(img)
        assert b64 is not None

    def test_image_to_b64_none(self):
        from pixlang.editor.server import _image_to_b64
        assert _image_to_b64(None) is None

    def test_image_to_b64_large_image_resized(self):
        from pixlang.editor.server import _image_to_b64
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        b64 = _image_to_b64(img)
        assert b64 is not None
        # Decode and check it's smaller than the original 2000x2000
        import cv2
        buf    = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert max(decoded.shape[:2]) <= 1200


# ══════════════════════════════════════════════════════════════════════════════
# 9. _serialize_stmts (flow diagram unit tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestFlowDiagram:

    def _parse_and_serialize(self, code):
        from pixlang.parser import parse
        from pixlang.editor.server import _serialize_stmts, _reset_id
        _reset_id()
        pipeline = parse(code)
        nodes, edges, first, last = _serialize_stmts(pipeline.commands)
        return nodes, edges, first, last

    def test_command_node_type(self):
        nodes, _, _, _ = self._parse_and_serialize('LOAD "x.png"\n')
        assert nodes[0]["type"] == "command"
        assert nodes[0]["name"] == "LOAD"

    def test_setvar_node_type(self):
        nodes, _, _, _ = self._parse_and_serialize("SET width 640\n")
        assert nodes[0]["type"] == "setvar"

    def test_if_block_nodes(self):
        nodes, edges, _, _ = self._parse_and_serialize(
            "SET x 1\nIF x == 1\nGRAYSCALE\nENDIF\n"
        )
        types = [n["type"] for n in nodes]
        assert "if_header" in types
        assert "if_footer" in types

    def test_if_block_false_edge_label(self):
        _, edges, _, _ = self._parse_and_serialize(
            "SET x 1\nIF x == 1\nGRAYSCALE\nENDIF\n"
        )
        labels = [e.get("label") for e in edges]
        assert "false" in labels

    def test_repeat_block_has_loop_edge(self):
        _, edges, _, _ = self._parse_and_serialize("REPEAT 3\nBLUR 5\nEND\n")
        labels = [e.get("label") for e in edges]
        assert "loop" in labels

    def test_roi_block_nodes(self):
        nodes, _, _, _ = self._parse_and_serialize(
            'LOAD "x.png"\nROI 0 0 100 100\nGRAYSCALE\nROI_RESET\n'
        )
        types = [n["type"] for n in nodes]
        assert "roi_header" in types
        assert "roi_footer" in types

    def test_include_node_type(self):
        nodes, _, _, _ = self._parse_and_serialize('INCLUDE "sub.pxl"\n')
        assert nodes[0]["type"] == "include"

    def test_assert_node_type(self):
        nodes, _, _, _ = self._parse_and_serialize("ASSERT width == 640\n")
        assert nodes[0]["type"] == "assert"

    def test_sequential_edges_connect_nodes(self):
        nodes, edges, first, last = self._parse_and_serialize(
            'LOAD "x.png"\nGRAYSCALE\nSAVE "o.png"\n'
        )
        assert len(nodes) == 3
        assert len(edges) == 2
        assert first == nodes[0]["id"]
        assert last  == nodes[2]["id"]

    def test_node_ids_are_unique(self):
        nodes, _, _, _ = self._parse_and_serialize(
            'LOAD "x.png"\nGRAYSCALE\nBLUR 5\nSAVE "o.png"\n'
        )
        ids = [n["id"] for n in nodes]
        assert len(ids) == len(set(ids))
