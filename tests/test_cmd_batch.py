# tests/test_cmd_batch.py
"""Batch engine: static discovery, per-file overrides, variables, no dry run."""
import cv2
import numpy as np
import pytest

from pixlang.batch.engine import (
    BatchRunner, find_glob_commands, iter_commands, static_glob_pattern,
)
from pixlang.commands import registry
from pixlang.parser import parse
from pixlang.parser.ast_nodes import ASTNode, Command, Pipeline


def _images(d, n=3):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"f{i}.png"), np.full((8, 8), 10 * (i + 1), np.uint8))


def test_distinct_outputs_and_no_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _images(tmp_path / "in")
    pxl = tmp_path / "p.pxl"
    # A plain SAVE would be executed by a dry run; it must only appear 3 times
    # (once per file), and SAVE_EACH must see each file's own pixels.
    pxl.write_text('LOAD_GLOB "in/*.png"\nSAVE_EACH "out/{stem}_{index1}.png"\n')
    res = BatchRunner(parse(pxl.read_text()), registry, pipeline_path=pxl).run()
    assert res == {"total": 3, "ok": 3, "failed": 0, "files": res["files"]}
    vals = sorted(int(cv2.imread(str(p), 0)[0, 0]) for p in (tmp_path / "out").glob("*.png"))
    assert vals == [10, 20, 30]
    assert (tmp_path / "out" / "f2_3.png").exists()


def test_discovery_does_not_execute(tmp_path):
    _images(tmp_path)
    pxl = tmp_path / "p.pxl"
    pxl.write_text('SAVE "should_not_exist.png"\nLOAD_GLOB "*.png"\n')
    runner = BatchRunner(parse(pxl.read_text()), registry, pipeline_path=pxl)
    assert len(runner.discover()) == 3
    assert not (tmp_path / "should_not_exist.png").exists()


def test_variables_applied_and_usable_in_pattern_and_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _images(tmp_path / "frames")
    pxl = tmp_path / "p.pxl"
    pxl.write_text('LOAD_GLOB $src\nSAVE_EACH "o/{stem}_{tag}.png"\n')
    runner = BatchRunner(parse(pxl.read_text()), registry, pipeline_path=pxl,
                         variables={"src": "frames/*.png", "tag": "x"})
    assert runner.run()["ok"] == 3
    assert (tmp_path / "o" / "f0_x.png").exists()


def test_set_literal_pattern(tmp_path):
    _images(tmp_path)
    pipeline = parse('SET pat "*.png"\nLOAD_GLOB $pat\n')
    assert static_glob_pattern(pipeline) == "*.png"


def test_glob_found_in_nested_blocks():
    pipeline = parse('REPEAT 1\nIF x == 1\nLOAD_GLOB "a/*.png"\nENDIF\nEND\n')
    assert [c.args for c in find_glob_commands(pipeline)] == [["a/*.png"]]


def test_iter_commands_is_generic_over_new_block_types():
    from dataclasses import dataclass, field

    @dataclass
    class FutureBlock(ASTNode):
        body: list = field(default_factory=list)
        else_body: list = field(default_factory=list)
        elifs: list = field(default_factory=list)

    inner = Command(name="LOAD_GLOB", args=["*.jpg"])
    blk = FutureBlock(body=[Command(name="BLUR")], elifs=[("cond", [inner])])
    names = [c.name for c in iter_commands(Pipeline(commands=[blk]).commands)]
    assert names == ["BLUR", "LOAD_GLOB"]


def test_plugin_loader_passed_to_each_run(tmp_path):
    _images(tmp_path, 2)

    class Loader:
        calls = 0
        manifests = []

        def load_entrypoints(self):
            Loader.calls += 1

        def load_local(self, path):
            pass

        def load_directory(self):
            pass

        def summary(self):
            return ""

    pxl = tmp_path / "p.pxl"
    pxl.write_text('LOAD_GLOB "*.png"\n')
    BatchRunner(parse(pxl.read_text()), registry, pipeline_path=pxl,
                plugin_loader=Loader()).run()
    assert Loader.calls == 2


def test_failures_counted(tmp_path):
    _images(tmp_path, 2)
    pxl = tmp_path / "p.pxl"
    pxl.write_text('LOAD_GLOB "*.png"\nCROP 100 100 5 5\n')
    res = BatchRunner(parse(pxl.read_text()), registry, pipeline_path=pxl).run()
    assert res["failed"] == 2 and "overlap" in res["files"][0]["error"]


def test_no_match_raises(tmp_path):
    pxl = tmp_path / "p.pxl"
    pxl.write_text('LOAD_GLOB "*.nothing"\n')
    with pytest.raises(FileNotFoundError, match="matched no files"):
        BatchRunner(parse(pxl.read_text()), registry, pipeline_path=pxl).run()
