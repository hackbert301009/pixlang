"""Hostile or unlucky requests get a clear answer, never a 500."""
from __future__ import annotations

import inspect
import io
import os

import pytest

pytest.importorskip("flask")

from pixlang.commands import registry as global_registry  # noqa: E402
from pixlang.editor.server import MAX_CONTENT_LENGTH, create_app  # noqa: E402

TOKEN = "robust-token"


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


@pytest.fixture
def client(ws):
    app = create_app(ws, token=TOKEN, run_timeout=10.0)
    with app.test_client() as c:
        c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
        yield c


class TestFilesListing:
    def test_symlink_to_a_pxl_outside_does_not_break_the_listing(self, client, ws, tmp_path):
        outside = tmp_path / "elsewhere.pxl"
        outside.write_text("LOAD x\n")
        (ws / "mine.pxl").write_text("LOAD y\n")
        os.symlink(outside, ws / "leak.pxl")
        r = client.get("/api/files")
        assert r.status_code == 200
        assert [f["path"] for f in r.get_json()["files"]] == ["mine.pxl"]

    def test_symlink_inside_the_workspace_is_listed(self, client, ws):
        (ws / "real.pxl").write_text("LOAD y\n")
        os.symlink(ws / "real.pxl", ws / "alias.pxl")
        names = [f["path"] for f in client.get("/api/files").get_json()["files"]]
        assert names == ["alias.pxl", "real.pxl"]

    def test_dangling_symlink_is_skipped(self, client, ws):
        os.symlink(ws / "gone.pxl", ws / "dangling.pxl")
        assert client.get("/api/files").get_json()["files"] == []


class TestToken:
    def test_non_ascii_token_header_is_a_403(self, client):
        r = client.get("/api/info", headers={"X-PixLang-Token": "töken€"})
        assert r.status_code == 403

    def test_right_token_still_works(self, client):
        assert client.get("/api/info").status_code == 200


class TestDeepNesting:
    @pytest.mark.parametrize("endpoint", ["/api/run", "/api/parse", "/api/lint",
                                          "/api/debug/start"])
    @pytest.mark.parametrize("code", [
        "SET x " + "(" * 5000 + "1" + ")" * 5000,
        "".join("IF 1 == 1\n" for _ in range(3000)) + "GRAYSCALE\n" + "ENDIF\n" * 3000,
    ], ids=["parens", "ifs"])
    def test_is_a_400_with_a_message(self, client, endpoint, code):
        r = client.post(endpoint, json={"code": code})
        assert r.status_code == 400, r.get_data(as_text=True)[:200]
        assert "too deeply" in r.get_json()["error"]


class TestBody:
    def test_malformed_json_is_a_400(self, client):
        r = client.post("/api/run", data="{not json", content_type="application/json")
        assert r.status_code == 400
        assert "JSON" in r.get_json()["error"]

    def test_json_that_is_not_an_object_is_a_400(self, client):
        r = client.post("/api/lint", data="[1, 2]", content_type="application/json")
        assert r.status_code == 400

    def test_empty_body_is_still_an_empty_object(self, client):
        assert client.post("/api/debug/stop", data="", content_type="application/json"
                           ).status_code == 200

    def test_oversized_body_without_content_length_is_a_413(self, client):
        big = b'{"code": "' + b"x" * (MAX_CONTENT_LENGTH + 10) + b'"}'
        r = client.post("/api/run", input_stream=io.BytesIO(big),
                        content_type="application/json")
        assert r.status_code == 413

    def test_oversized_body_with_content_length_is_a_413(self, client):
        big = b'{"code": "' + b"x" * (MAX_CONTENT_LENGTH + 10) + b'"}'
        r = client.post("/api/lint", data=big, content_type="application/json")
        assert r.status_code == 413


def test_csp_forbids_base_forms_and_plugins(client):
    csp = client.get("/").headers["Content-Security-Policy"]
    for directive in ("base-uri 'none'", "form-action 'none'", "object-src 'none'"):
        assert directive in csp


class TestLanguageMatchesSignatures:
    def test_every_command_agrees_with_inspect(self, client):
        served = {c["name"]: c for c in client.get("/api/language").get_json()["commands"]}
        assert set(served) == set(global_registry.list_commands())
        for info in global_registry.list_info():
            expected = []
            for name, p in inspect.signature(info.fn).parameters.items():
                if name in ("image", "_ctx"):
                    continue
                variadic = p.kind is p.VAR_POSITIONAL
                expected.append((name + ("…" if variadic else ""),
                                 variadic or p.default is not inspect.Parameter.empty))
            got = [(p["name"], p["optional"]) for p in served[info.name]["params"]]
            assert got == expected, info.name

    @pytest.mark.parametrize("cmd, param", [
        ("ADD_ALPHA", "value"), ("CONVERT_DEPTH", "scale"), ("EXPORT_C_ARRAY", "fmt"),
        ("IN_RANGE", "l3"), ("MERGE_CHANNELS", "a"),
    ])
    def test_none_defaults_are_optional(self, client, cmd, param):
        cmds = {c["name"]: c for c in client.get("/api/language").get_json()["commands"]}
        entry = next(p for p in cmds[cmd]["params"] if p["name"] == param)
        assert entry["optional"] is True
