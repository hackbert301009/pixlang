"""The editor may only touch files inside its workspace (and read the bundled
examples); every command that takes a path has to declare it."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("flask")

from pixlang.commands import registry as global_registry  # noqa: E402
from pixlang.commands.confine import PATH_PARAM_NAMES, PathConfinementError, confine  # noqa: E402
from pixlang.commands.registry import CommandRegistry  # noqa: E402
from pixlang.editor.server import create_app  # noqa: E402

TOKEN = "confine-token"
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
OUTSIDE_MSG = "outside the editor workspace"


@pytest.fixture
def dirs(tmp_path):
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    cv2.imwrite(str(ws / "in.png"), np.full((20, 30, 3), 90, np.uint8))
    cv2.imwrite(str(outside / "secret.png"), np.full((20, 30, 3), 200, np.uint8))
    np.save(outside / "lut.npy", np.arange(256, dtype=np.uint8))
    (outside / "raw.bin").write_bytes(bytes(30 * 20))
    return ws, outside


@pytest.fixture
def client(dirs):
    app = create_app(dirs[0], token=TOKEN, run_timeout=5.0)
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
        yield c


def run(client, code, path=None):
    body = {"code": code}
    if path:
        body["path"] = path
    return client.post("/api/run", json=body).get_json()


def tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


class TestWritesStayInside:
    @pytest.mark.parametrize("stmt", [
        'SAVE "../outside/x.png"',
        'SAVE "{out}/x.png"',
        'SAVE_RAW "{out}/raw2.bin" gray8',
        'EXPORT_C_ARRAY "{out}/img.h" img gray8',
        'HISTOGRAM_SAVE "{out}/h.png"',
        'SAVE "sub/../../outside/x.png"',
    ])
    def test_rejected_and_nothing_written(self, client, dirs, stmt):
        ws, outside = dirs
        before = tree(outside)
        res = run(client, 'LOAD "in.png"\n' + stmt.format(out=outside))
        assert not res["ok"]
        assert OUTSIDE_MSG in res["error"]
        assert res["error_info"]["line"] == 2
        assert tree(outside) == before

    def test_symlinked_directory_is_followed_before_checking(self, client, dirs):
        ws, outside = dirs
        os.symlink(outside, ws / "link")
        res = run(client, 'LOAD "in.png"\nSAVE "link/x.png"')
        assert not res["ok"]
        assert not (outside / "x.png").exists()

    def test_save_each_template_is_checked_after_formatting(self, client, dirs):
        ws, outside = dirs
        res = run(client, f'SET d "{outside}"\nLOAD_GLOB "*.png"\nSAVE_EACH "{{d}}/{{stem}}_o.png"')
        assert not res["ok"]
        assert OUTSIDE_MSG in res["error"]
        assert not list(outside.glob("*_o.png"))

    def test_relative_writes_land_next_to_the_pipeline(self, client, dirs):
        ws, _ = dirs
        (ws / "proj").mkdir()
        res = run(client, 'LOAD "../in.png"\nSAVE "out/r.png"\nHISTOGRAM_SAVE', path="proj/p.pxl")
        assert res["ok"], res["error"]
        assert (ws / "proj" / "out" / "r.png").is_file()
        assert (ws / "proj" / "histogram.png").is_file()

    def test_working_directory_is_untouched(self, client):
        before = os.getcwd()
        run(client, 'LOAD "in.png"\nSAVE "o.png"')
        assert os.getcwd() == before


class TestReadsStayInside:
    @pytest.mark.parametrize("stmt", [
        'LOAD "{out}/secret.png"',
        'LOAD "../outside/secret.png"',
        'LOAD "in.png"\nLUT "{out}/lut.npy"',
        'LOAD_RAW "{out}/raw.bin" 30 20 gray8',
        'LOAD_GLOB "{out}/*.png"',
        'LOAD_GLOB "../outside/*.png"',
    ])
    def test_rejected(self, client, dirs, stmt):
        ws, outside = dirs
        res = run(client, stmt.format(out=outside))
        assert not res["ok"]
        assert OUTSIDE_MSG in res["error"]

    def test_symlinked_file_pointing_out(self, client, dirs):
        ws, outside = dirs
        os.symlink(outside / "secret.png", ws / "s.png")
        res = run(client, 'LOAD "s.png"')
        assert not res["ok"]
        assert OUTSIDE_MSG in res["error"]

    def test_glob_matches_that_are_symlinks_out_are_dropped(self, client, dirs):
        ws, outside = dirs
        os.symlink(outside / "secret.png", ws / "aa.png")
        res = run(client, 'LOAD_GLOB "*.png"')
        assert res["ok"], res["error"]
        assert res["stats"]["commands_executed"] == 1
        assert res["image"]["width"] == 30

    def test_bundled_examples_are_readable(self, client):
        res = run(client, f'LOAD "{EXAMPLES / "sample.jpg"}"\nPRINT_INFO')
        assert res["ok"], res["error"]

    def test_bundled_examples_are_not_writable(self, client):
        target = EXAMPLES / "zz_nope.png"
        res = run(client, f'LOAD "{EXAMPLES / "sample.jpg"}"\nSAVE "{target}"')
        assert not res["ok"]
        assert not target.exists()

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
    def test_fifo_is_refused_instead_of_hanging(self, client, dirs):
        ws, _ = dirs
        os.mkfifo(ws / "pipe.png")
        done = threading.Event()
        result = {}

        def go():       # its own client: a hang must fail the test, not freeze it
            app = create_app(ws, token=TOKEN, run_timeout=5.0)
            with app.test_client() as c:
                c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
                result["res"] = run(c, 'LOAD "pipe.png"')
            done.set()

        threading.Thread(target=go, daemon=True).start()
        assert done.wait(4), "LOAD of a FIFO blocked"
        assert not result["res"]["ok"]
        assert "not a regular file" in result["res"]["error"]
        assert run(client, 'LOAD "in.png"')["ok"]


class TestDebuggerIsConfinedToo:
    def test_debug_run_cannot_write_outside(self, client, dirs):
        ws, outside = dirs
        view = client.post("/api/debug/start", json={
            "code": 'LOAD "in.png"\nSAVE "../outside/dbg.png"',
            "stop_on_entry": False}).get_json()
        for _ in range(50):
            if view["state"] in ("finished", "error", "stopped"):
                break
            view = client.get(f"/api/debug/{view['session_id']}/state?wait={view['seq']}&timeout=1"
                              ).get_json()
        assert view["state"] == "error"
        assert OUTSIDE_MSG in view["result"]["error"]
        assert not (outside / "dbg.png").exists()


class TestConfineHelper:
    def test_relative_to_base(self, tmp_path):
        root = tmp_path / "r"
        root.mkdir()
        (root / "a.png").write_bytes(b"x")
        assert confine("a.png", [root], "read", base=root) == (root / "a.png").resolve()

    def test_dotdot_and_absolute(self, tmp_path):
        root = tmp_path / "r"
        root.mkdir()
        with pytest.raises(PathConfinementError):
            confine("../x", [root], "write", base=root)
        with pytest.raises(PathConfinementError):
            confine(str(tmp_path / "x"), [root], "write", base=root)

    def test_missing_read_target_is_left_to_the_command(self, tmp_path):
        root = tmp_path / "r"
        root.mkdir()
        assert confine("nope.png", [root], "read", base=root).name == "nope.png"

    def test_directory_is_not_a_file(self, tmp_path):
        root = tmp_path / "r"
        (root / "d").mkdir(parents=True)
        for mode in ("read", "write"):
            with pytest.raises(PathConfinementError, match="not a regular file"):
                confine("d", [root], mode, base=root)

    def test_nul_byte(self, tmp_path):
        with pytest.raises(PathConfinementError):
            confine("a\0b", [tmp_path], "read", base=tmp_path)

    @pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs /proc")
    def test_proc_is_outside(self, tmp_path):
        with pytest.raises(PathConfinementError):
            confine("/proc/self/environ", [tmp_path], "read", base=tmp_path)


class TestEveryPathParameterIsDeclared:
    def test_registry_marks_all_path_like_parameters(self):
        missing = []
        for info in global_registry.list_info():
            for name, _type, _default in info.params:
                if name.lstrip("*") in PATH_PARAM_NAMES and name not in info.paths:
                    missing.append(f"{info.name}.{name}")
        assert not missing, f"declare these with paths={{...}}: {missing}"

    def test_documented_names_are_covered(self):
        assert {"path", "file", "filename", "template", "pattern", "model",
                "labels"} <= PATH_PARAM_NAMES

    def test_marker_must_name_a_real_parameter(self):
        r = CommandRegistry()
        with pytest.raises(ValueError, match="no parameter 'pth'"):
            @r.register("X", paths={"pth": "read"})
            def x(image, path: str):
                return image

    def test_marker_mode_is_validated(self):
        r = CommandRegistry()
        with pytest.raises(ValueError, match="mode"):
            @r.register("X", paths={"path": "append"})
            def x(image, path: str):
                return image

    def test_undeclared_plugin_command_is_refused_in_the_editor(self, dirs):
        ws, outside = dirs
        reg = CommandRegistry()

        @reg.register("DUMP", source="plugin")
        def dump(image, path: str):
            Path(path).write_text("x")
            return image

        app = create_app(ws, token=TOKEN, registry=reg, plugins=False)
        with app.test_client() as c:
            c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
            res = run(c, f'DUMP "{outside}/p.txt"')
        assert not res["ok"]
        assert "paths=" in res["error"]
        assert not (outside / "p.txt").exists()

    def test_declared_plugin_command_is_confined(self, dirs):
        ws, outside = dirs
        reg = CommandRegistry()

        @reg.register("DUMP", source="plugin", paths={"path": "write"})
        def dump(image, path: str):
            Path(path).write_text("x")
            return image

        app = create_app(ws, token=TOKEN, registry=reg, plugins=False)
        with app.test_client() as c:
            c.environ_base["HTTP_X_PIXLANG_TOKEN"] = TOKEN
            bad = run(c, f'DUMP "{outside}/p.txt"')
            good = run(c, 'DUMP "p.txt"')
        assert not bad["ok"]
        assert OUTSIDE_MSG in bad["error"]
        assert good["ok"], good["error"]
        assert (ws / "p.txt").is_file()
        assert not (outside / "p.txt").exists()


class TestLinkerRoots:
    @pytest.mark.parametrize("stmt", [
        'IMPORT "{out}/lib.pxl" AS lib',
        'IMPORT "../outside/lib.pxl" AS lib',
        'FROM "../outside/lib.pxl" IMPORT f',
    ])
    def test_import_outside_is_rejected(self, client, dirs, stmt):
        ws, outside = dirs
        (outside / "lib.pxl").write_text("EXPORT FUNC f()\n  RETURN 1\nENDFUNC\n")
        res = run(client, stmt.format(out=outside) + '\nLOAD "in.png"', path="m.pxl")
        assert not res["ok"]
        assert OUTSIDE_MSG in res["error"]

    def test_non_pxl_module_is_rejected_and_not_echoed(self, client, dirs):
        ws, _ = dirs
        (ws / "notes.txt").write_text("TOP-SECRET-MARKER")
        res = run(client, 'IMPORT "notes.txt" AS n\nLOAD "in.png"', path="m.pxl")
        assert not res["ok"]
        assert "TOP-SECRET-MARKER" not in str(res)

    def test_import_inside_still_works(self, client, dirs):
        ws, _ = dirs
        (ws / "lib.pxl").write_text("EXPORT FUNC f()\n  RETURN 1\nENDFUNC\n")
        res = run(client, 'IMPORT "lib.pxl" AS lib\nLOAD "in.png"\nSET x lib.f()', path="m.pxl")
        assert res["ok"], res["error"]
