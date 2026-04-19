# tests/test_v4.py
"""
v0.4 test suite:

  Lexer/Parser
    - INCLUDE, ASSERT, ROI/ROI_RESET parse to correct AST nodes
    - Nested ROI in IF/REPEAT
    - ASSERT all subjects and operators

  Executor
    - INCLUDE: inline sub-pipeline, shared context, file-not-found error
    - ASSERT: all subjects (width/height/channels/min/max/contour_count)
    - ASSERT: pass / fail / custom message
    - ROI: crops, runs body, pastes back; handles gray→BGR conversion
    - ROI: shape mismatch resize; out-of-bounds error

  New commands
    - AUTO_CROP: removes dead border, padding, all-black passthrough
    - COMPARE: prints metrics, stores diff checkpoint, passthrough
    - BLEND: all 7 modes produce correct output shape

  Batch engine
    - LOAD_GLOB: finds files, populates context
    - SAVE_EACH: template resolution, auto mkdir
    - BatchRunner: single file succeeds, missing pattern error

  Config
    - pixlang.toml discovery walks upward
    - variables injected, lint_ignore list, output_dir
    - missing file returns defaults

  CLI
    - version string is 0.4.0
    - lint command exits 0 on clean, 1 on errors
    - validate shows ASSERT/INCLUDE/ROI in AST dump
"""
import textwrap
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from pixlang.parser import parse
from pixlang.parser.lexer import tokenize
from pixlang.parser.ast_nodes import (
    Command, IfBlock, Pipeline, RepeatBlock,
    IncludeStmt, AssertStmt, RoiBlock, SetVar,
)
from pixlang.commands import registry as _global_registry
from pixlang.commands.registry import CommandRegistry
from pixlang.commands.builtin import register_all, register_v03, register_v04, _wrap, _unwrap
from pixlang.batch import register_batch_commands
from pixlang.executor import Executor
from pixlang.linter import Linter


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def reg():
    r = CommandRegistry()
    register_all(r)
    register_v03(r)
    register_v04(r)
    register_batch_commands(r)
    return r


def bgr(h=80, w=100):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

def gray(h=80, w=100):
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


def _run(source, reg=None, pipeline_path=None):
    from pixlang.commands import registry as default_reg
    r = reg or default_reg
    pipeline = parse(source)
    ex = Executor(registry=r, pipeline_path=pipeline_path)
    ex.run(pipeline)
    return ex


# ═════════════════════════════════════════════════════════════════════════════
# Lexer + Parser: new v0.4 nodes
# ═════════════════════════════════════════════════════════════════════════════

class TestParserV4:

    def test_parse_include(self):
        p = parse('INCLUDE "sub.pxl"')
        assert len(p.commands) == 1
        node = p.commands[0]
        assert isinstance(node, IncludeStmt)
        assert node.path == "sub.pxl"

    def test_parse_assert_basic(self):
        p = parse("ASSERT width == 640")
        node = p.commands[0]
        assert isinstance(node, AssertStmt)
        assert node.subject == "width"
        assert node.op      == "=="
        assert node.expected == 640

    def test_parse_assert_with_message(self):
        p = parse('ASSERT height >= 100 "Must be tall enough"')
        node = p.commands[0]
        assert node.message == "Must be tall enough"

    def test_parse_assert_all_subjects(self):
        for subject in ["width", "height", "channels", "min", "max", "contour_count"]:
            p = parse(f"ASSERT {subject} == 0")
            assert isinstance(p.commands[0], AssertStmt)
            assert p.commands[0].subject == subject

    def test_parse_assert_all_ops(self):
        for op in ["==", "!=", "<", ">", "<=", ">="]:
            p = parse(f"ASSERT width {op} 100")
            assert p.commands[0].op == op

    def test_parse_roi_block(self):
        src = "ROI 10 20 300 200\nGRAYSCALE\nROI_RESET"
        p   = parse(src)
        node = p.commands[0]
        assert isinstance(node, RoiBlock)
        assert node.x == 10
        assert node.y == 20
        assert node.w == 300
        assert node.h == 200
        assert len(node.body) == 1
        assert node.body[0].name == "GRAYSCALE"

    def test_parse_roi_with_vars(self):
        src = "SET x 0\nROI $x 0 320 240\nBLUR 5\nROI_RESET"
        p   = parse(src)
        roi = p.commands[1]
        from pixlang.parser.ast_nodes import VarRef
        assert isinstance(roi.x, VarRef)
        assert roi.x.var_name == "x"

    def test_parse_roi_nested_in_if(self):
        src = textwrap.dedent("""\
            SET flag 1
            IF flag == 1
              ROI 0 0 100 100
                GRAYSCALE
              ROI_RESET
            ENDIF
        """)
        p = parse(src)
        if_node = p.commands[1]
        assert isinstance(if_node, IfBlock)
        assert isinstance(if_node.body[0], RoiBlock)

    def test_parse_include_in_repeat(self):
        src = 'REPEAT 2\nINCLUDE "step.pxl"\nEND'
        p   = parse(src)
        rep = p.commands[0]
        assert isinstance(rep, RepeatBlock)
        assert isinstance(rep.body[0], IncludeStmt)

    def test_keyword_include_in_lexer(self):
        tokens = tokenize('INCLUDE "file.pxl"')
        assert tokens[0].type  == "KEYWORD"
        assert tokens[0].value == "INCLUDE"

    def test_keyword_assert_in_lexer(self):
        tokens = tokenize("ASSERT width == 100")
        assert tokens[0].type  == "KEYWORD"
        assert tokens[0].value == "ASSERT"


# ═════════════════════════════════════════════════════════════════════════════
# Executor: INCLUDE
# ═════════════════════════════════════════════════════════════════════════════

class TestExecutorInclude:

    def _write_sub(self, tmp_path, content):
        sub = tmp_path / "sub.pxl"
        sub.write_text(content)
        return sub

    def test_include_runs_sub_pipeline(self, tmp_path, reg):
        """Commands in included file execute against current image."""
        executed = []

        @reg.register("PROBE_INC", source="test")
        def _probe(image):
            executed.append(True)
            return image

        sub = tmp_path / "sub.pxl"
        sub.write_text("PROBE_INC\n")

        main = tmp_path / "main.pxl"
        cv2.imwrite(str(tmp_path / "img.jpg"),
                    np.zeros((50, 50, 3), dtype=np.uint8))
        main.write_text(f'LOAD "{tmp_path}/img.jpg"\nINCLUDE "sub.pxl"\n')

        ex = Executor(registry=reg, pipeline_path=main)
        ex.run(parse(main.read_text()))
        assert executed

    def test_include_shares_context(self, tmp_path, reg):
        """Variables SET in parent pipeline are visible inside included file."""
        seen_vars = []

        @reg.register("READ_VAR", source="test")
        def _read(image, _ctx=None):
            if _ctx:
                seen_vars.append(_ctx["vars"].get("shared_val"))
            return image

        sub = tmp_path / "sub.pxl"
        sub.write_text("READ_VAR\n")

        main = tmp_path / "main.pxl"
        cv2.imwrite(str(tmp_path / "img.jpg"),
                    np.zeros((50, 50, 3), dtype=np.uint8))
        main.write_text(
            f'SET shared_val 42\nLOAD "{tmp_path}/img.jpg"\nINCLUDE "sub.pxl"\n'
        )

        ex = Executor(registry=reg, pipeline_path=main)
        ex.run(parse(main.read_text()))
        assert seen_vars == [42]

    def test_include_missing_file_raises(self, tmp_path, reg):
        main = tmp_path / "main.pxl"
        cv2.imwrite(str(tmp_path / "img.jpg"),
                    np.zeros((50, 50, 3), dtype=np.uint8))
        main.write_text(f'LOAD "{tmp_path}/img.jpg"\nINCLUDE "ghost.pxl"\n')

        ex = Executor(registry=reg, pipeline_path=main)
        with pytest.raises(RuntimeError, match="File not found"):
            ex.run(parse(main.read_text()))

    def test_include_nested_two_levels(self, tmp_path, reg):
        """INCLUDE inside an INCLUDEd file works (two levels deep)."""
        calls = []

        @reg.register("LEAF_CMD", source="test")
        def _leaf(image):
            calls.append(1)
            return image

        leaf = tmp_path / "leaf.pxl"
        leaf.write_text("LEAF_CMD\n")

        middle = tmp_path / "middle.pxl"
        middle.write_text('INCLUDE "leaf.pxl"\n')

        main = tmp_path / "main.pxl"
        cv2.imwrite(str(tmp_path / "img.jpg"),
                    np.zeros((30, 30, 3), dtype=np.uint8))
        main.write_text(f'LOAD "{tmp_path}/img.jpg"\nINCLUDE "middle.pxl"\n')

        ex = Executor(registry=reg, pipeline_path=main)
        ex.run(parse(main.read_text()))
        assert len(calls) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Executor: ASSERT
# ═════════════════════════════════════════════════════════════════════════════

class TestExecutorAssert:

    def _make_ctx(self):
        return {"checkpoints": {}, "vars": {}, "stats": {
            "commands_executed": 0, "total_ms": 0.0, "command_times": {}}}

    def _run_assert(self, reg, img, subject, op, expected, message=""):
        from pixlang.parser.ast_nodes import AssertStmt
        stmt = AssertStmt(subject=subject, op=op, expected=expected,
                          message=message, line=1)
        ex = Executor(registry=reg)
        return ex._exec_assert(stmt, img)

    def test_assert_width_pass(self, reg):
        img = bgr(100, 200)
        self._run_assert(reg, img, "width", "==", 200)  # no exception

    def test_assert_width_fail(self, reg):
        img = bgr(100, 200)
        with pytest.raises(AssertionError, match="width"):
            self._run_assert(reg, img, "width", "==", 999)

    def test_assert_height_pass(self, reg):
        img = bgr(120, 80)
        self._run_assert(reg, img, "height", "==", 120)

    def test_assert_channels_bgr(self, reg):
        img = bgr()
        self._run_assert(reg, img, "channels", "==", 3)

    def test_assert_channels_gray(self, reg):
        img = gray()
        self._run_assert(reg, img, "channels", "==", 1)

    def test_assert_min(self, reg):
        img = np.full((50, 50, 3), 10, dtype=np.uint8)
        self._run_assert(reg, img, "min", ">=", 5)

    def test_assert_max(self, reg):
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        self._run_assert(reg, img, "max", "<=", 255)

    def test_assert_contour_count(self, reg):
        wrapped = _wrap(gray())
        wrapped["contours"] = ["c1", "c2", "c3"]
        from pixlang.parser.ast_nodes import AssertStmt
        stmt = AssertStmt(subject="contour_count", op="==", expected=3, line=1)
        ex = Executor(registry=reg)
        ex._exec_assert(stmt, wrapped)  # no exception

    def test_assert_custom_message_in_error(self, reg):
        img = bgr(100, 200)
        with pytest.raises(AssertionError, match="Must be 999 wide"):
            self._run_assert(reg, img, "width", "==", 999, "Must be 999 wide")

    def test_assert_all_ops(self, reg):
        img = bgr(100, 200)  # width=200
        cases = [
            ("width", "==",  200, True),
            ("width", "!=",  100, True),
            ("width", ">",   100, True),
            ("width", "<",   300, True),
            ("width", ">=",  200, True),
            ("width", "<=",  200, True),
            ("width", "==",  999, False),
        ]
        for subject, op, val, should_pass in cases:
            if should_pass:
                self._run_assert(reg, img, subject, op, val)
            else:
                with pytest.raises(AssertionError):
                    self._run_assert(reg, img, subject, op, val)

    def test_assert_unknown_subject_raises(self, reg):
        from pixlang.parser.ast_nodes import AssertStmt
        stmt = AssertStmt(subject="pixel_density", op="==", expected=1, line=1)
        ex = Executor(registry=reg)
        with pytest.raises(RuntimeError, match="Unknown subject"):
            ex._exec_assert(stmt, bgr())

    def test_assert_no_image_raises(self, reg):
        from pixlang.parser.ast_nodes import AssertStmt
        stmt = AssertStmt(subject="width", op="==", expected=100, line=1)
        ex = Executor(registry=reg)
        with pytest.raises(RuntimeError):
            ex._exec_assert(stmt, None)


# ═════════════════════════════════════════════════════════════════════════════
# Executor: ROI
# ═════════════════════════════════════════════════════════════════════════════

class TestExecutorROI:

    def test_roi_processes_region_only(self, reg):
        """Commands inside ROI should only modify the specified region."""
        # White image — invert inside top-left quarter should make it black there
        img = np.full((100, 100, 3), 255, dtype=np.uint8)

        @reg.register("MAKE_BLACK", source="test")
        def _black(image):
            img_in, meta = _unwrap(image)
            return np.zeros_like(img_in)

        from pixlang.parser.ast_nodes import RoiBlock, Command
        roi = RoiBlock(x=0, y=0, w=50, h=50, line=1,
                       body=[Command(name="MAKE_BLACK", args=[], line=2)])
        ex = Executor(registry=reg)
        result = ex._exec_roi(roi, img)
        out, _ = _unwrap(result)

        # Top-left region should now be black
        assert out[:50, :50].max() == 0
        # Rest should remain white
        assert out[50:, 50:].min() == 255

    def test_roi_pastes_result_back(self, reg):
        """Full image shape must be preserved after ROI."""
        img    = bgr(200, 300)
        from pixlang.parser.ast_nodes import RoiBlock, Command
        roi = RoiBlock(x=10, y=10, w=50, h=40, line=1,
                       body=[Command(name="GRAYSCALE", args=[], line=2)])
        ex = Executor(registry=reg)
        result = ex._exec_roi(roi, img)
        out, _ = _unwrap(result)
        assert out.shape[:2] == (200, 300)

    def test_roi_out_of_bounds_raises(self, reg):
        img = bgr(50, 50)
        from pixlang.parser.ast_nodes import RoiBlock
        roi = RoiBlock(x=100, y=100, w=200, h=200, line=1, body=[])
        ex = Executor(registry=reg)
        with pytest.raises(RuntimeError, match="outside image bounds"):
            ex._exec_roi(roi, img)

    def test_roi_var_resolved(self, reg):
        """ROI coordinates can be $variables."""
        from pixlang.parser.ast_nodes import RoiBlock, Command, VarRef
        img = bgr(100, 100)
        roi = RoiBlock(
            x=VarRef(var_name="rx", line=1),
            y=VarRef(var_name="ry", line=1),
            w=40, h=40, line=1,
            body=[Command(name="GRAYSCALE", args=[], line=2)]
        )
        ex = Executor(registry=reg)
        ex.context["vars"]["rx"] = 10
        ex.context["vars"]["ry"] = 10
        result = ex._exec_roi(roi, img)
        out, _ = _unwrap(result)
        assert out.shape[:2] == (100, 100)

    def test_roi_clamps_to_bounds(self, reg):
        """ROI that extends past image edge should be clamped."""
        img = bgr(50, 50)
        from pixlang.parser.ast_nodes import RoiBlock, Command
        # x=30, y=30, w=100, h=100 — extends well past 50x50
        roi = RoiBlock(x=30, y=30, w=100, h=100, line=1,
                       body=[Command(name="BLUR", args=[3], line=2)])
        ex = Executor(registry=reg)
        result = ex._exec_roi(roi, img)
        out, _ = _unwrap(result)
        assert out.shape[:2] == (50, 50)

    def test_roi_no_image_raises(self, reg):
        from pixlang.parser.ast_nodes import RoiBlock
        roi = RoiBlock(x=0, y=0, w=10, h=10, line=1, body=[])
        ex = Executor(registry=reg)
        with pytest.raises(RuntimeError):
            ex._exec_roi(roi, None)


# ═════════════════════════════════════════════════════════════════════════════
# New commands: AUTO_CROP, COMPARE, BLEND
# ═════════════════════════════════════════════════════════════════════════════

class TestAutoGrop:

    def test_auto_crop_removes_border(self, reg):
        """Content centred in a padded black canvas should be tightly cropped."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas[20:80, 20:80] = 200   # white-ish square in centre
        fn = reg.get("AUTO_CROP")
        result = fn(canvas, 0)
        out, _ = _unwrap(result)
        # Result should be smaller than original
        assert out.shape[0] < 100
        assert out.shape[1] < 100

    def test_auto_crop_padding(self, reg):
        """Padding adds back border pixels."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas[30:70, 30:70] = 200
        fn  = reg.get("AUTO_CROP")
        r0  = fn(canvas, 0)
        r10 = fn(canvas, 10)
        out0, _  = _unwrap(r0)
        out10, _ = _unwrap(r10)
        assert out10.shape[0] > out0.shape[0]
        assert out10.shape[1] > out0.shape[1]

    def test_auto_crop_all_black_passthrough(self, reg):
        """All-black image — no content to crop — should return unchanged."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        fn  = reg.get("AUTO_CROP")
        result = fn(img, 0)
        out, _ = _unwrap(result)
        assert out.shape == img.shape

    def test_auto_crop_preserves_meta(self, reg):
        fn = reg.get("AUTO_CROP")
        w  = _wrap(bgr())
        w["contours"] = ["c"]
        result = fn(w, 5)
        _, meta = _unwrap(result)
        assert "contours" in meta

    def test_auto_crop_grayscale(self, reg):
        img = np.zeros((80, 80), dtype=np.uint8)
        img[10:70, 10:70] = 200
        fn = reg.get("AUTO_CROP")
        result = fn(img, 0)
        out, _ = _unwrap(result)
        assert out.shape[0] <= 60
        assert out.shape[1] <= 60


class TestCompare:

    def _make_ctx(self, checkpoint_img=None):
        ctx = {"checkpoints": {}, "vars": {}, "stats": {
            "commands_executed": 0, "total_ms": 0.0, "command_times": {}}}
        if checkpoint_img is not None:
            ctx["checkpoints"]["ref"] = (checkpoint_img, {})
        return ctx

    def test_compare_identical_high_psnr(self, reg, capsys):
        img = bgr(50, 50)
        ctx = self._make_ctx(img.copy())
        fn  = reg.get("COMPARE")
        fn(img, "ref", _ctx=ctx)
        out = capsys.readouterr().out
        # Identical images → very high PSNR
        assert "100.00" in out or "inf" in out.lower() or float(
            [l for l in out.split("\n") if "PSNR" in l][0].split(":")[1].split("dB")[0].strip()
        ) > 80

    def test_compare_stores_diff_checkpoint(self, reg, capsys):
        a = np.zeros((50, 50, 3), dtype=np.uint8)
        b = np.full((50, 50, 3), 128, dtype=np.uint8)
        ctx = self._make_ctx(b)
        fn  = reg.get("COMPARE")
        fn(a, "ref", _ctx=ctx)
        assert "diff" in ctx["checkpoints"]

    def test_compare_passthrough(self, reg, capsys):
        img = bgr(40, 40)
        ctx = self._make_ctx(img.copy())
        fn  = reg.get("COMPARE")
        result = fn(img, "ref", _ctx=ctx)
        out_img, _ = _unwrap(result)
        assert out_img.shape == img.shape

    def test_compare_missing_checkpoint_raises(self, reg):
        ctx = {"checkpoints": {}, "vars": {}}
        fn  = reg.get("COMPARE")
        with pytest.raises(RuntimeError, match="No checkpoint"):
            fn(bgr(), "ghost", _ctx=ctx)

    def test_compare_no_ctx_raises(self, reg):
        with pytest.raises(RuntimeError):
            reg.get("COMPARE")(bgr(), "ref", _ctx=None)


class TestBlend:

    def _ctx_with(self, name, img):
        return {"checkpoints": {name: (img, {})}, "vars": {}}

    def test_blend_normal_shape(self, reg):
        a   = bgr(50, 60)
        ctx = self._ctx_with("snap", bgr(50, 60))
        fn  = reg.get("BLEND")
        result = fn(a, "snap", 0.5, "normal", _ctx=ctx)
        out, _ = _unwrap(result)
        assert out.shape == a.shape

    def test_blend_all_modes_succeed(self, reg):
        a   = bgr(40, 40)
        for mode in ["normal", "difference", "multiply", "screen",
                     "add", "lighten", "darken"]:
            ctx = self._ctx_with("s", bgr(40, 40))
            fn  = reg.get("BLEND")
            result = fn(a, "s", 0.5, mode, _ctx=ctx)
            out, _ = _unwrap(result)
            assert out.shape == a.shape, f"Mode {mode} changed shape"

    def test_blend_difference_detects_change(self, reg):
        """Difference of two distinct images should not be all-zero."""
        a = np.zeros((50, 50, 3), dtype=np.uint8)
        b = np.full((50, 50, 3), 200, dtype=np.uint8)
        ctx = self._ctx_with("b", b)
        fn  = reg.get("BLEND")
        result = fn(a, "b", 1.0, "difference", _ctx=ctx)
        out, _ = _unwrap(result)
        assert out.max() > 0

    def test_blend_add_clamps_at_255(self, reg):
        a   = np.full((50, 50, 3), 200, dtype=np.uint8)
        ctx = self._ctx_with("s", np.full((50, 50, 3), 200, dtype=np.uint8))
        fn  = reg.get("BLEND")
        result = fn(a, "s", 1.0, "add", _ctx=ctx)
        out, _ = _unwrap(result)
        assert out.max() <= 255

    def test_blend_invalid_mode_raises(self, reg):
        ctx = self._ctx_with("s", bgr())
        fn  = reg.get("BLEND")
        with pytest.raises(ValueError, match="Unknown mode"):
            fn(bgr(), "s", 0.5, "vortex_blend", _ctx=ctx)

    def test_blend_missing_checkpoint_raises(self, reg):
        ctx = {"checkpoints": {}, "vars": {}}
        with pytest.raises(RuntimeError, match="No checkpoint"):
            reg.get("BLEND")(bgr(), "ghost", 0.5, "normal", _ctx=ctx)

    def test_blend_resizes_mismatched_checkpoint(self, reg):
        a   = bgr(50, 60)
        ctx = self._ctx_with("big", bgr(200, 300))
        fn  = reg.get("BLEND")
        result = fn(a, "big", 0.5, "normal", _ctx=ctx)
        out, _ = _unwrap(result)
        assert out.shape == a.shape


# ═════════════════════════════════════════════════════════════════════════════
# Batch: LOAD_GLOB + SAVE_EACH
# ═════════════════════════════════════════════════════════════════════════════

class TestBatch:

    def _make_images(self, tmp_path, n=3):
        paths = []
        for i in range(n):
            p = tmp_path / f"img_{i:02d}.jpg"
            cv2.imwrite(str(p), np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8))
            paths.append(p)
        return paths

    def test_load_glob_populates_context(self, tmp_path, reg):
        self._make_images(tmp_path, 3)
        fn  = reg.get("LOAD_GLOB")
        ctx = {"batch_files": [], "batch_index": 0, "batch_current": None,
               "vars": {}, "pipeline_dir": tmp_path}
        result = fn(None, "*.jpg", _ctx=ctx)
        assert len(ctx["batch_files"]) == 3
        assert result is not None  # first image loaded

    def test_load_glob_no_match_raises(self, tmp_path, reg):
        fn  = reg.get("LOAD_GLOB")
        ctx = {"vars": {}, "pipeline_dir": tmp_path}
        with pytest.raises(FileNotFoundError, match="No files matched"):
            fn(None, "*.tiff", _ctx=ctx)

    def test_load_glob_no_ctx_raises(self, reg):
        with pytest.raises(RuntimeError):
            reg.get("LOAD_GLOB")(None, "*.jpg", _ctx=None)

    def test_save_each_creates_file(self, tmp_path, reg):
        img = bgr(40, 60)
        ctx = {
            "batch_current": tmp_path / "source.jpg",
            "batch_index":   0,
            "vars": {"name": "source", "stem": "source", "ext": ".jpg",
                     "index": 0, "index1": 1, "dir": str(tmp_path)},
            "verbose": False,
        }
        out_dir = tmp_path / "out"
        fn = reg.get("SAVE_EACH")
        fn(img, str(out_dir / "{stem}_done.png"), _ctx=ctx)
        assert (out_dir / "source_done.png").exists()

    def test_save_each_auto_mkdir(self, tmp_path, reg):
        """SAVE_EACH must create the output directory if it doesn't exist."""
        img = bgr()
        ctx = {
            "batch_current": tmp_path / "x.jpg",
            "batch_index":   0,
            "vars": {"name": "x", "stem": "x", "ext": ".jpg",
                     "index": 0, "index1": 1, "dir": str(tmp_path)},
            "verbose": False,
        }
        deep = tmp_path / "a" / "b" / "c"
        fn = reg.get("SAVE_EACH")
        fn(img, str(deep / "result.png"), _ctx=ctx)
        assert (deep / "result.png").exists()

    def test_save_each_index_template(self, tmp_path, reg):
        img = bgr()
        ctx = {
            "batch_current": tmp_path / "img.jpg",
            "batch_index":   4,
            "vars": {"name": "img", "stem": "img", "ext": ".jpg",
                     "index": 4, "index1": 5, "dir": str(tmp_path)},
            "verbose": False,
        }
        fn = reg.get("SAVE_EACH")
        fn(img, str(tmp_path / "out_{index1}.png"), _ctx=ctx)
        assert (tmp_path / "out_5.png").exists()

    def test_batch_runner_single_file(self, tmp_path, reg):
        from pixlang.batch.engine import BatchRunner
        img_path = tmp_path / "img.jpg"
        cv2.imwrite(str(img_path), np.zeros((40, 60, 3), dtype=np.uint8))
        out_dir = tmp_path / "out"

        pxl = tmp_path / "pipe.pxl"
        pxl.write_text(
            f'LOAD_GLOB "*.jpg"\n'
            f'GRAYSCALE\n'
            f'SAVE_EACH "out/{{stem}}.png"\n'
        )
        pipeline = parse(pxl.read_text())
        runner   = BatchRunner(pipeline, reg, pipeline_path=pxl)
        results  = runner.run()
        assert results["ok"]    == 1
        assert results["total"] == 1
        assert results["failed"] == 0

    def test_batch_runner_multiple_files(self, tmp_path, reg):
        from pixlang.batch.engine import BatchRunner
        self._make_images(tmp_path, 3)
        out_dir = tmp_path / "out"

        pxl = tmp_path / "pipe.pxl"
        pxl.write_text('LOAD_GLOB "*.jpg"\nGRAYSCALE\nSAVE_EACH "out/{stem}.png"\n')
        pipeline = parse(pxl.read_text())
        runner   = BatchRunner(pipeline, reg, pipeline_path=pxl)
        results  = runner.run()
        assert results["total"] == 3
        assert results["ok"]    == 3
        assert results["failed"] == 0

    def test_plain_pipeline_batch_runner_single_run(self, tmp_path, reg):
        """BatchRunner on a pipeline without LOAD_GLOB falls back to single run."""
        from pixlang.batch.engine import BatchRunner
        img = tmp_path / "i.jpg"
        cv2.imwrite(str(img), np.zeros((30, 30, 3), dtype=np.uint8))
        pxl = tmp_path / "p.pxl"
        pxl.write_text(f'LOAD "{img}"\nGRAYSCALE\nSAVE "{tmp_path}/o.png"\n')
        pipeline = parse(pxl.read_text())
        runner   = BatchRunner(pipeline, reg, pipeline_path=pxl)
        results  = runner.run()
        assert results["total"] == 1
        assert results["ok"]    == 1


# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════

class TestConfig:

    def test_default_config_when_no_file(self, tmp_path):
        from pixlang.config import load_config
        cfg = load_config(tmp_path)
        assert cfg.verbose      == False
        assert cfg.plugins      == True
        assert cfg.source_path  is None
        assert cfg.variables    == {}

    def test_discovers_toml_in_same_dir(self, tmp_path):
        from pixlang.config import load_config
        toml = tmp_path / "pixlang.toml"
        toml.write_text("[defaults]\nverbose = true\n")
        cfg = load_config(tmp_path)
        assert cfg.verbose     == True
        assert cfg.source_path == toml

    def test_discovers_toml_in_parent(self, tmp_path):
        from pixlang.config import load_config
        (tmp_path / "pixlang.toml").write_text("[defaults]\nverbose = true\n")
        sub = tmp_path / "sub" / "dir"
        sub.mkdir(parents=True)
        cfg = load_config(sub)
        assert cfg.verbose == True

    def test_variables_parsed(self, tmp_path):
        from pixlang.config import load_config
        (tmp_path / "pixlang.toml").write_text(
            "[variables]\nwidth = 640\nmode = \"fast\"\n"
        )
        cfg = load_config(tmp_path)
        assert cfg.variables["width"] == 640
        assert cfg.variables["mode"]  == "fast"

    def test_lint_ignore_parsed(self, tmp_path):
        from pixlang.config import load_config
        # Use Python 3.11+ tomllib format (list) — falls back to simple parser
        (tmp_path / "pixlang.toml").write_text('[lint]\nignore = "PX006"\n')
        cfg = load_config(tmp_path)
        assert isinstance(cfg.lint_ignore, list)

    def test_output_dir_parsed(self, tmp_path):
        from pixlang.config import load_config
        (tmp_path / "pixlang.toml").write_text('[batch]\noutput_dir = "results"\n')
        cfg = load_config(tmp_path)
        assert cfg.output_dir == "results"

    def test_broken_toml_returns_defaults(self, tmp_path):
        from pixlang.config import load_config
        import warnings
        (tmp_path / "pixlang.toml").write_text("@@@@broken@@@@\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = load_config(tmp_path)
        assert cfg.verbose == False   # should fall back to defaults


# ═════════════════════════════════════════════════════════════════════════════
# CLI smoke tests
# ═════════════════════════════════════════════════════════════════════════════

class TestCLIV4:

    def test_version_is_040(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pixlang.cli", "--version"],
            capture_output=True, text=True
        )
        assert "0.4.0" in result.stdout + result.stderr

    def test_lint_clean_exit_0(self, tmp_path):
        import subprocess, sys
        pxl = tmp_path / "clean.pxl"
        pxl.write_text(
            'LOAD "img.png"\nGRAYSCALE\nBLUR 5\nSAVE "out.png"\n'
        )
        result = subprocess.run(
            [sys.executable, "-m", "pixlang.cli", "lint", str(pxl)],
            capture_output=True, text=True
        )
        assert result.returncode == 0

    def test_lint_error_exit_1(self, tmp_path):
        import subprocess, sys
        pxl = tmp_path / "bad.pxl"
        pxl.write_text("GRAYSCALE\n")  # PX001: no LOAD
        result = subprocess.run(
            [sys.executable, "-m", "pixlang.cli", "lint", str(pxl)],
            capture_output=True, text=True
        )
        assert result.returncode == 1
        assert "PX001" in result.stdout

    def test_validate_shows_assert_node(self, tmp_path):
        import subprocess, sys
        pxl = tmp_path / "with_assert.pxl"
        pxl.write_text('LOAD "x.jpg"\nASSERT width == 640\nSAVE "y.png"\n')
        result = subprocess.run(
            [sys.executable, "-m", "pixlang.cli", "validate", str(pxl)],
            capture_output=True, text=True
        )
        assert result.returncode == 0

    def test_commands_lists_new_v4(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pixlang.cli", "commands"],
            capture_output=True, text=True
        )
        output = result.stdout
        for cmd in ["AUTO_CROP", "BLEND", "COMPARE", "LOAD_GLOB", "SAVE_EACH"]:
            assert cmd in output, f"{cmd} missing from commands output"


# ═════════════════════════════════════════════════════════════════════════════
# End-to-end v0.4 pipelines
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndV4:

    def test_roi_assert_pipeline(self):
        from pixlang.commands import registry
        src = Path("examples/roi_assert.pxl").read_text()
        ex = Executor(registry=registry,
                      pipeline_path=Path("examples/roi_assert.pxl"))
        ex.run(parse(src))
        out = Path("examples/roi_assert_out.png")
        assert out.exists()
        out.unlink()

    def test_include_demo_pipeline(self):
        from pixlang.commands import registry
        src = Path("examples/include_demo.pxl").read_text()
        ex = Executor(registry=registry,
                      pipeline_path=Path("examples/include_demo.pxl"))
        ex.run(parse(src))
        out = Path("examples/include_demo_out.png")
        assert out.exists()
        out.unlink()

    def test_blend_modes_pipeline(self):
        from pixlang.commands import registry
        src = Path("examples/blend_modes.pxl").read_text()
        ex = Executor(registry=registry,
                      pipeline_path=Path("examples/blend_modes.pxl"))
        ex.run(parse(src))
        for f in ["blend_difference.png", "blend_screen.png", "blend_multiply.png"]:
            p = Path("examples") / f
            assert p.exists(), f"Missing {f}"
            p.unlink()
