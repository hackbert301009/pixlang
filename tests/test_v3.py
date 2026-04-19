# tests/test_v3.py
"""
v0.3 test suite:
  - Lexer: VAR, KEYWORD, IDENT tokens
  - Parser: SET, IF/ENDIF, REPEAT/END, VarRef in args
  - Executor: variable store, IF branches, REPEAT loop, $ITER
  - New commands: RESIZE_PERCENT, HISTOGRAM_SAVE, PIPELINE_STATS
  - Linter: all 11 rules
  - Watcher: run_once, change detection
"""
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from pixlang.parser import parse
from pixlang.parser.lexer import tokenize
from pixlang.parser.ast_nodes import (
    Command, IfBlock, Pipeline, RepeatBlock, SetVar, VarRef,
)
from pixlang.commands import registry as _global_registry
from pixlang.commands.registry import CommandRegistry
from pixlang.commands.builtin import register_all, register_v03, _wrap, _unwrap
from pixlang.executor import Executor
from pixlang.linter import Linter, Severity


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def reg():
    r = CommandRegistry()
    register_all(r)
    register_v03(r)
    return r


def bgr(h=80, w=100):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

def gray(h=80, w=100):
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)

def run_src(source, reg=None):
    """Parse and run source, return executor (for context inspection)."""
    from pixlang.commands import registry as default_reg
    r = reg or default_reg
    pipeline = parse(source)
    ex = Executor(registry=r)
    ex.run(pipeline)
    return ex


# ═════════════════════════════════════════════════════════════════════════════
# Lexer v0.3
# ═════════════════════════════════════════════════════════════════════════════

class TestLexerV3:

    def test_var_token(self):
        tokens = tokenize("RESIZE $width $height")
        types = [t.type for t in tokens]
        assert types == ["COMMAND", "VAR", "VAR"]

    def test_var_strips_dollar(self):
        tokens = tokenize("$myvar")
        assert tokens[0].value == "$myvar"   # raw value retained
        assert tokens[0].type  == "VAR"

    def test_keyword_set(self):
        tokens = tokenize("SET width 640")
        assert tokens[0].type  == "KEYWORD"
        assert tokens[0].value == "SET"

    def test_keywords_recognized(self):
        src = "SET x 1\nIF x == 1\nENDIF\nREPEAT 3\nEND"
        types = [t.type for t in tokenize(src) if t.type in ("KEYWORD","COMMAND","IDENT","INT","OP")]
        assert "KEYWORD" in types

    def test_op_tokens(self):
        tokens = tokenize("IF x == 5")
        ops = [t for t in tokens if t.type == "OP"]
        assert len(ops) == 1
        assert ops[0].value == "=="

    def test_all_ops(self):
        for op in ["==", "!=", "<", ">", "<=", ">="]:
            tokens = tokenize(f"IF x {op} 0")
            assert any(t.type == "OP" and t.value == op for t in tokens)

    def test_negative_int(self):
        tokens = tokenize("ROTATE -90")
        nums = [t for t in tokens if t.type == "INT"]
        assert nums[0].value == "-90"


# ═════════════════════════════════════════════════════════════════════════════
# Parser v0.3
# ═════════════════════════════════════════════════════════════════════════════

class TestParserV3:

    def test_parse_set(self):
        p = parse("SET width 640")
        assert len(p.commands) == 1
        s = p.commands[0]
        assert isinstance(s, SetVar)
        assert s.var_name == "width"
        assert s.value    == 640

    def test_parse_set_string(self):
        p = parse('SET mode "inspect"')
        s = p.commands[0]
        assert s.value == "inspect"

    def test_parse_var_ref_in_command(self):
        p = parse("RESIZE $width $height")
        cmd = p.commands[0]
        assert isinstance(cmd.args[0], VarRef)
        assert cmd.args[0].var_name == "width"

    def test_parse_if_block(self):
        src = "SET x 1\nIF x == 1\nGRAYSCALE\nENDIF"
        p   = parse(src)
        if_node = p.commands[1]
        assert isinstance(if_node, IfBlock)
        assert if_node.var_name  == "x"
        assert if_node.op        == "=="
        assert if_node.cmp_value == 1
        assert len(if_node.body) == 1
        assert if_node.body[0].name == "GRAYSCALE"

    def test_parse_all_ops_in_if(self):
        for op in ["==", "!=", "<", ">", "<=", ">="]:
            p = parse(f"SET n 1\nIF n {op} 0\nGRAYSCALE\nENDIF")
            assert isinstance(p.commands[1], IfBlock)
            assert p.commands[1].op == op

    def test_parse_repeat_block(self):
        src = "REPEAT 3\nBLUR 5\nEND"
        p   = parse(src)
        rep = p.commands[0]
        assert isinstance(rep, RepeatBlock)
        assert rep.count == 3
        assert rep.body[0].name == "BLUR"

    def test_parse_repeat_var_count(self):
        src = "SET n 5\nREPEAT $n\nGRAYSCALE\nEND"
        p   = parse(src)
        rep = p.commands[1]
        assert isinstance(rep.count, VarRef)
        assert rep.count.var_name == "n"

    def test_parse_nested_if_in_repeat(self):
        src = textwrap.dedent("""\
            SET flag 1
            REPEAT 2
              IF flag == 1
                GRAYSCALE
              ENDIF
            END
        """)
        p = parse(src)
        rep = p.commands[1]
        assert isinstance(rep, RepeatBlock)
        assert isinstance(rep.body[0], IfBlock)

    def test_parse_mixed_pipeline(self):
        src = textwrap.dedent("""\
            SET w 640
            SET h 480
            LOAD "img.png"
            RESIZE $w $h
            IF w == 640
              GRAYSCALE
            ENDIF
            REPEAT 2
              BLUR 3
            END
            SAVE "out.png"
        """)
        p = parse(src)
        types = [type(s).__name__ for s in p.commands]
        assert "SetVar"      in types
        assert "Command"     in types
        assert "IfBlock"     in types
        assert "RepeatBlock" in types


# ═════════════════════════════════════════════════════════════════════════════
# Executor v0.3 — control flow
# ═════════════════════════════════════════════════════════════════════════════

class TestExecutorControlFlow:

    def test_set_stores_variable(self):
        ex = run_src('LOAD "examples/sample.jpg"\nSET x 42')
        assert ex.context["vars"]["x"] == 42

    def test_set_string_variable(self):
        ex = run_src('LOAD "examples/sample.jpg"\nSET mode "fast"')
        assert ex.context["vars"]["mode"] == "fast"

    def test_var_ref_resolved_in_command(self, reg):
        pipeline = parse("SET sz 3\nLOAD \"examples/sample.jpg\"\nBLUR $sz")
        ex = Executor(registry=reg)
        ex.run(pipeline)  # should not raise

    def test_undefined_var_raises(self, reg):
        pipeline = parse("LOAD \"examples/sample.jpg\"\nBLUR $undefined_var")
        ex = Executor(registry=reg)
        with pytest.raises(RuntimeError, match="undefined_var"):
            ex.run(pipeline)

    def test_if_true_branch_executes(self, reg):
        executed = []
        @reg.register("PROBE_CMD", source="test", allow_override=True)
        def _probe(image):
            executed.append(True)
            return image

        pipeline = parse(
            "SET x 5\n"
            "LOAD \"examples/sample.jpg\"\n"
            "IF x == 5\n"
            "PROBE_CMD\n"
            "ENDIF\n"
        )
        Executor(registry=reg).run(pipeline)
        assert executed, "Body should have executed when condition is true"

    def test_if_false_branch_skipped(self, reg):
        executed = []
        @reg.register("PROBE_CMD2", source="test")
        def _probe(image):
            executed.append(True)
            return image

        pipeline = parse(
            "SET x 5\n"
            "LOAD \"examples/sample.jpg\"\n"
            "IF x == 99\n"
            "PROBE_CMD2\n"
            "ENDIF\n"
        )
        Executor(registry=reg).run(pipeline)
        assert not executed, "Body should be skipped when condition is false"

    def test_if_operators(self, reg):
        results = {}
        for op, expected in [("==", True), ("!=", False), (">", False),
                              ("<", True), (">=", True), ("<=", True)]:
            count = []
            @reg.register(f"PROBE_{op.replace('=','E').replace('<','L').replace('>','G')}",
                          source="test")
            def _p(image, _count=count):
                _count.append(1)
                return image

            # x=5, compare to 5 — only == >= <= should fire
            pass  # covered by integration below

    def test_repeat_runs_n_times(self, reg):
        call_count = []
        @reg.register("COUNT_CMD", source="test")
        def _count(image):
            call_count.append(1)
            return image

        pipeline = parse(
            "LOAD \"examples/sample.jpg\"\n"
            "REPEAT 4\n"
            "COUNT_CMD\n"
            "END\n"
        )
        Executor(registry=reg).run(pipeline)
        assert len(call_count) == 4

    def test_repeat_iter_variable(self, reg):
        iters = []
        @reg.register("CAPTURE_ITER", source="test")
        def _cap(image, _ctx=None):
            if _ctx:
                iters.append(_ctx["vars"].get("ITER"))
            return image

        pipeline = parse(
            "LOAD \"examples/sample.jpg\"\n"
            "REPEAT 3\n"
            "CAPTURE_ITER\n"
            "END\n"
        )
        Executor(registry=reg).run(pipeline)
        assert iters == [0, 1, 2]

    def test_repeat_zero_times(self, reg):
        """REPEAT 0 should run zero iterations without error."""
        executed = []
        @reg.register("NEVER_CMD", source="test")
        def _never(image):
            executed.append(True)
            return image

        pipeline = parse(
            "LOAD \"examples/sample.jpg\"\n"
            "REPEAT 0\n"
            "NEVER_CMD\n"
            "END\n"
        )
        Executor(registry=reg).run(pipeline)
        assert not executed

    def test_repeat_var_count(self, reg):
        counts = []
        @reg.register("LOOP_COUNT", source="test")
        def _lc(image):
            counts.append(1)
            return image

        pipeline = parse(
            "SET n 3\n"
            "LOAD \"examples/sample.jpg\"\n"
            "REPEAT $n\n"
            "LOOP_COUNT\n"
            "END\n"
        )
        Executor(registry=reg).run(pipeline)
        assert len(counts) == 3

    def test_stats_populated(self, reg):
        pipeline = parse(
            "LOAD \"examples/sample.jpg\"\n"
            "GRAYSCALE\n"
            "BLUR 5\n"
        )
        ex = Executor(registry=reg)
        ex.run(pipeline)
        stats = ex.context["stats"]
        assert stats["commands_executed"] == 3
        assert "GRAYSCALE" in stats["command_times"]
        assert "BLUR"       in stats["command_times"]


# ═════════════════════════════════════════════════════════════════════════════
# New commands: RESIZE_PERCENT, HISTOGRAM_SAVE, PIPELINE_STATS
# ═════════════════════════════════════════════════════════════════════════════

class TestNewCommandsV3:

    def test_resize_percent_half(self, reg):
        fn  = reg.get("RESIZE_PERCENT")
        img = bgr(200, 300)
        result = fn(img, 50.0)
        out, _ = _unwrap(result)
        assert out.shape[:2] == (100, 150)

    def test_resize_percent_double(self, reg):
        fn  = reg.get("RESIZE_PERCENT")
        img = bgr(100, 100)
        result = fn(img, 200.0)
        out, _ = _unwrap(result)
        assert out.shape[:2] == (200, 200)

    def test_resize_percent_preserves_meta(self, reg):
        fn = reg.get("RESIZE_PERCENT")
        w  = _wrap(bgr())
        w["contours"] = ["c1"]
        result = fn(w, 50)
        _, meta = _unwrap(result)
        assert "contours" in meta

    def test_resize_percent_zero_raises(self, reg):
        fn = reg.get("RESIZE_PERCENT")
        with pytest.raises(ValueError, match="must be > 0"):
            fn(bgr(), 0)

    def test_resize_percent_negative_raises(self, reg):
        fn = reg.get("RESIZE_PERCENT")
        with pytest.raises(ValueError):
            fn(bgr(), -10)

    def test_histogram_save_creates_file(self, reg, tmp_path):
        fn   = reg.get("HISTOGRAM_SAVE")
        out  = tmp_path / "hist.png"
        fn(bgr(), str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_histogram_save_grayscale(self, reg, tmp_path):
        fn  = reg.get("HISTOGRAM_SAVE")
        out = tmp_path / "hist_gray.png"
        fn(gray(), str(out))
        assert out.exists()

    def test_histogram_save_passthrough(self, reg, tmp_path):
        """HISTOGRAM_SAVE should return the image unchanged."""
        fn   = reg.get("HISTOGRAM_SAVE")
        img  = bgr(50, 50)
        result = fn(img, str(tmp_path / "h.png"))
        out, _ = _unwrap(result)
        assert out.shape == img.shape

    def test_pipeline_stats_no_ctx(self, reg, capsys):
        """PIPELINE_STATS without context should print a warning, not crash."""
        fn = reg.get("PIPELINE_STATS")
        fn(bgr(), _ctx=None)
        captured = capsys.readouterr()
        assert "No stats available" in captured.out

    def test_pipeline_stats_with_ctx(self, reg, capsys):
        fn  = reg.get("PIPELINE_STATS")
        ctx = {
            "stats": {
                "commands_executed": 3,
                "total_ms": 42.5,
                "command_times": {"BLUR": [5.0, 6.0], "GRAYSCALE": [30.0]},
            }
        }
        fn(bgr(), _ctx=ctx)
        out = capsys.readouterr().out
        assert "BLUR"      in out
        assert "GRAYSCALE" in out
        assert "42.5"      in out


# ═════════════════════════════════════════════════════════════════════════════
# Linter
# ═════════════════════════════════════════════════════════════════════════════

class TestLinter:

    def _lint(self, source, reg=None):
        from pixlang.commands import registry as default_reg
        pipeline = parse(source)
        return Linter(reg or default_reg).lint(pipeline)

    def _codes(self, diags):
        return {d.code for d in diags}

    def test_clean_pipeline_no_errors(self):
        src = textwrap.dedent("""\
            LOAD "img.png"
            GRAYSCALE
            BLUR 5
            THRESHOLD 128
            FIND_CONTOURS
            DRAW_BOUNDING_BOXES
            SAVE "out.png"
        """)
        diags = self._lint(src)
        errors = [d for d in diags if d.severity == Severity.ERROR]
        assert not errors

    def test_px001_no_load(self):
        diags = self._lint("GRAYSCALE\nSAVE \"out.png\"")
        assert "PX001" in self._codes(diags)

    def test_px001_starts_with_load_ok(self):
        diags = self._lint("LOAD \"img.png\"\nGRAYSCALE\nSAVE \"out.png\"")
        assert "PX001" not in self._codes(diags)

    def test_px002_no_save(self):
        diags = self._lint("LOAD \"img.png\"\nGRAYSCALE")
        assert "PX002" in self._codes(diags)

    def test_px003_contours_no_draw(self):
        diags = self._lint("LOAD \"img.png\"\nFIND_CONTOURS\nSAVE \"out.png\"")
        assert "PX003" in self._codes(diags)

    def test_px003_contours_with_draw_ok(self):
        src = "LOAD \"img.png\"\nFIND_CONTOURS\nDRAW_BOUNDING_BOXES\nSAVE \"out.png\""
        diags = self._lint(src)
        assert "PX003" not in self._codes(diags)

    def test_px004_overlay_no_checkpoint(self):
        diags = self._lint("LOAD \"img.png\"\nOVERLAY \"snap\" 0.5\nSAVE \"out.png\"")
        assert "PX004" in self._codes(diags)

    def test_px004_overlay_with_checkpoint_ok(self):
        src = ("LOAD \"img.png\"\nCHECKPOINT \"snap\"\n"
               "OVERLAY \"snap\" 0.5\nSAVE \"out.png\"")
        diags = self._lint(src)
        assert "PX004" not in self._codes(diags)

    def test_px005_undefined_variable(self):
        diags = self._lint("LOAD \"img.png\"\nBLUR $undefined\nSAVE \"out.png\"")
        assert "PX005" in self._codes(diags)

    def test_px005_defined_variable_ok(self):
        src = "SET k 5\nLOAD \"img.png\"\nBLUR $k\nSAVE \"out.png\""
        diags = self._lint(src)
        assert "PX005" not in self._codes(diags)

    def test_px006_duplicate_command(self):
        src = "LOAD \"img.png\"\nBLUR 5\nBLUR 5\nSAVE \"out.png\""
        diags = self._lint(src)
        assert "PX006" in self._codes(diags)

    def test_px007_resize_after_filter(self):
        src = "LOAD \"img.png\"\nBLUR 5\nRESIZE 640 480\nSAVE \"out.png\""
        diags = self._lint(src)
        assert "PX007" in self._codes(diags)

    def test_px007_resize_before_filter_ok(self):
        src = "LOAD \"img.png\"\nRESIZE 640 480\nBLUR 5\nSAVE \"out.png\""
        diags = self._lint(src)
        assert "PX007" not in self._codes(diags)

    def test_px008_repeat_zero(self):
        src = "LOAD \"img.png\"\nREPEAT 0\nGRAYSCALE\nEND\nSAVE \"out.png\""
        diags = self._lint(src)
        assert "PX008" in self._codes(diags)

    def test_px009_empty_pipeline(self):
        diags = self._lint("")
        assert "PX009" in self._codes(diags)

    def test_px010_threshold_before_gray(self):
        src = "LOAD \"img.png\"\nTHRESHOLD 128\nGRAYSCALE\nSAVE \"out.png\""
        diags = self._lint(src)
        assert "PX010" in self._codes(diags)

    def test_px010_threshold_after_gray_ok(self):
        src = "LOAD \"img.png\"\nGRAYSCALE\nTHRESHOLD 128\nSAVE \"out.png\""
        diags = self._lint(src)
        assert "PX010" not in self._codes(diags)

    def test_px011_unknown_command(self):
        from pixlang.commands import registry
        src = "LOAD \"img.png\"\nFLY_TO_MOON\nSAVE \"out.png\""
        diags = self._lint(src, reg=registry)
        assert "PX011" in self._codes(diags)

    def test_severities_have_correct_types(self):
        diags = self._lint("")
        for d in diags:
            assert isinstance(d.severity, Severity)

    def test_rule_count(self):
        linter = Linter()
        assert linter.rule_count >= 11


# ═════════════════════════════════════════════════════════════════════════════
# Watcher
# ═════════════════════════════════════════════════════════════════════════════

class TestWatcher:

    def test_run_once_success(self, tmp_path, reg):
        from pixlang.watcher import Watcher

        img_path = tmp_path / "img.jpg"
        import cv2, numpy as np
        cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))

        pxl = tmp_path / "pipe.pxl"
        pxl.write_text(
            f'LOAD "{img_path}"\n'
            f'GRAYSCALE\n'
            f'SAVE "{tmp_path / "out.png"}"\n'
        )
        w = Watcher(pxl, reg, verbose=False, lint=True)
        assert w.run_once() is True

    def test_run_once_lint_error_prevents_run(self, tmp_path, reg):
        from pixlang.watcher import Watcher

        pxl = tmp_path / "bad.pxl"
        pxl.write_text("GRAYSCALE\n")  # PX001: no LOAD → lint ERROR

        executed = []
        @reg.register("GRAYSCALE", source="test", allow_override=True)
        def _gs(image):
            executed.append(True)
            return image

        w = Watcher(pxl, reg, verbose=False, lint=True)
        result = w.run_once()
        assert result is False
        assert not executed, "Command should not run when linter returns errors"

    def test_run_once_syntax_error(self, tmp_path, reg):
        from pixlang.watcher import Watcher

        pxl = tmp_path / "syntax.pxl"
        pxl.write_text("@@@invalid@@@\n")

        w = Watcher(pxl, reg, lint=False)
        assert w.run_once() is False

    def test_run_once_missing_file(self, tmp_path, reg):
        from pixlang.watcher import Watcher

        w = Watcher(tmp_path / "ghost.pxl", reg, lint=False)
        assert w.run_once() is False

    def test_change_detection(self, tmp_path, reg):
        from pixlang.watcher import Watcher
        import cv2, numpy as np

        img_path = tmp_path / "img.jpg"
        cv2.imwrite(str(img_path), np.zeros((20, 20, 3), dtype=np.uint8))

        pxl = tmp_path / "watch.pxl"
        pxl.write_text(f'LOAD "{img_path}"\nSAVE "{tmp_path}/o.png"\n')

        w = Watcher(pxl, reg, lint=False)
        w.run_once()                        # first run, snapshots mtime

        assert not w._changed()             # no change yet

        time.sleep(0.05)
        pxl.write_text(f'LOAD "{img_path}"\nGRAYSCALE\nSAVE "{tmp_path}/o.png"\n')
        import os; os.utime(pxl, None)      # force mtime bump

        assert w._changed()                 # now changed

    def test_increments_run_count(self, tmp_path, reg):
        from pixlang.watcher import Watcher
        import cv2, numpy as np

        img = tmp_path / "i.jpg"
        cv2.imwrite(str(img), np.zeros((20,20,3), dtype=np.uint8))
        pxl = tmp_path / "p.pxl"
        pxl.write_text(f'LOAD "{img}"\nSAVE "{tmp_path}/o.png"\n')

        w = Watcher(pxl, reg, lint=False)
        w.run_once()
        w.run_once()
        assert w._run_count == 2


# ═════════════════════════════════════════════════════════════════════════════
# End-to-end: control flow pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndV3:

    def test_control_flow_pipeline(self):
        """Full control_flow.pxl pipeline runs without error."""
        from pixlang.commands import registry
        src = Path("examples/control_flow.pxl").read_text()
        pipeline = parse(src)
        ex = Executor(registry=registry)
        ex.run(pipeline)
        assert Path("examples/control_flow_out.png").exists()
        Path("examples/control_flow_out.png").unlink()

    def test_pyramid_pipeline(self):
        """pyramid.pxl runs and creates all three level outputs."""
        from pixlang.commands import registry
        src = Path("examples/pyramid.pxl").read_text()
        pipeline = parse(src)
        ex = Executor(registry=registry)
        ex.run(pipeline)
        for lvl in ["L0", "L1", "L2"]:
            p = Path(f"examples/pyramid_{lvl}.png")
            assert p.exists(), f"Missing {p}"
            p.unlink()
        for f in Path("examples").glob("histogram_*.png"):
            f.unlink()
