# tests/test_cmd_watcher.py
"""Watcher: no busy re-runs after failure, dependency tracking, plugin reuse."""
import os

import cv2
import numpy as np
import pytest

from pixlang.commands.builtin import register_all
from pixlang.commands.registry import CommandRegistry
from pixlang.watcher import Watcher
from pixlang.watcher.engine import format_syntax_error, scan_dependencies, strip_comment


@pytest.fixture
def reg():
    r = CommandRegistry()
    register_all(r)
    return r


def bump(path, text=None):
    if text is not None:
        path.write_text(text)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def test_failed_run_does_not_retrigger(tmp_path, reg):
    pxl = tmp_path / "p.pxl"
    pxl.write_text('LOAD "missing.png"\n')
    w = Watcher(pxl, reg, lint=False)
    assert w.run_once() is False
    assert not w._changed()
    bump(pxl)
    assert w._changed()


def test_syntax_error_records_state(tmp_path, reg):
    pxl = tmp_path / "p.pxl"
    pxl.write_text("@@@\n")
    w = Watcher(pxl, reg, lint=False)
    assert w.run_once() is False
    assert not w._changed()


def test_missing_file_detected_when_created(tmp_path, reg):
    pxl = tmp_path / "later.pxl"
    w = Watcher(pxl, reg, lint=False)
    assert w.run_once() is False
    assert not w._changed()
    pxl.write_text("")
    assert w._changed()


def test_include_dependencies_tracked(tmp_path, reg):
    sub = tmp_path / "lib"
    sub.mkdir()
    (sub / "inner.pxl").write_text("GRAYSCALE\n")
    (sub / "mid.pxl").write_text('INCLUDE "inner.pxl"\n# INCLUDE "ignored.pxl"\n')
    pxl = tmp_path / "p.pxl"
    pxl.write_text('IMPORT "lib/mid.pxl"\nFROM "x.pxl" IMPORT y\n')
    deps = scan_dependencies(pxl)
    assert (sub / "mid.pxl").resolve() in deps
    assert (sub / "inner.pxl").resolve() in deps
    assert (tmp_path / "x.pxl").resolve() in deps
    assert not any(p.name == "ignored.pxl" for p in deps)

    w = Watcher(pxl, reg, lint=False)
    w.run_once()
    assert not w._changed()
    bump(sub / "inner.pxl", "BLUR 3\n")
    assert w._changed()


def test_plugin_file_tracked_and_loader_reused(tmp_path, reg):
    cv2.imwrite(str(tmp_path / "i.png"), np.zeros((8, 8, 3), np.uint8))
    pxl = tmp_path / "p.pxl"
    plug = tmp_path / "p.plugins.py"
    plug.write_text(
        "def register(registry):\n"
        "    @registry.register('MYCMD', source='t')\n"
        "    def _m(image):\n        return image\n"
    )
    pxl.write_text(f'LOAD "{tmp_path / "i.png"}"\nMYCMD\n')
    w = Watcher(pxl, reg, lint=False, trust_plugins=True)
    assert w.run_once() is True
    assert w.run_once() is True          # second run: no ConflictError
    assert plug in w.watched_files
    bump(plug, plug.read_text().replace("MYCMD", "MYCMD2"))
    assert w._changed()
    pxl.write_text(f'LOAD "{tmp_path / "i.png"}"\nMYCMD2\n')
    assert w.run_once() is True
    assert "MYCMD2" in reg and "MYCMD" not in reg


def test_variables_injected(tmp_path, reg):
    cv2.imwrite(str(tmp_path / "i.png"), np.zeros((8, 8, 3), np.uint8))
    pxl = tmp_path / "p.pxl"
    pxl.write_text(f'LOAD "{tmp_path / "i.png"}"\nBLUR $k\n')
    assert Watcher(pxl, reg, lint=False, variables={"k": 3}).run_once() is True


def test_strip_comment_keeps_hash_in_strings():
    assert strip_comment('LOAD "a#b.png"  # c') == 'LOAD "a#b.png"  '


def test_format_syntax_error_uses_attributes(tmp_path):
    e = SyntaxError("bad token")
    e.filename, e.lineno, e.offset = "x.pxl", 3, 7
    assert format_syntax_error(e) == "x.pxl:3:7: bad token"
    assert "recursion" in format_syntax_error(RecursionError(), tmp_path / "a.pxl")
