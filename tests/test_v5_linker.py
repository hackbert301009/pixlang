# tests/test_v5_linker.py
"""
v0.5 language-core tests, part 3: functions, modules and the linker.

  - FUNC (command-style and expression), frames, GLOBAL, RETURN
  - recursion cap with call-stack trace
  - IMPORT ... AS / FROM ... IMPORT, EXPORT visibility
  - linker diagnostics: cycles, duplicates, undefined symbols, unused imports,
    not exported, shadowing, search paths, INCLUDE splicing
  - LinkedProgram runs without reading files; `pixlang link` CLI
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pixlang.commands import registry as default_registry
from pixlang.commands.registry import CommandRegistry
from pixlang.executor import Executor
from pixlang.linker import LinkedProgram, Linker, link
from pixlang.parser import PixLinkError, PixRuntimeError, PixSyntaxError, parse
from pixlang.parser.ast_nodes import FuncDef, ImportStmt


def run(src, reg=None, path=None):
    ex = Executor(registry=reg or default_registry, pipeline_path=path)
    ex.run(parse(src, filename=str(path) if path else None))
    return ex


def write(tmp_path, name, text):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))
    return p


@pytest.fixture
def probe_reg():
    """Registry with a PROBE command recording (image-id, args)."""
    from pixlang.commands.builtin import register_all
    reg = CommandRegistry()
    register_all(reg)
    log = []

    @reg.register("PROBE", source="test")
    def _probe(image, *args):
        log.append(args)
        return image

    reg.log = log
    return reg


# ═════════════════════════════════════════════════════════════════════════════
# Functions
# ═════════════════════════════════════════════════════════════════════════════

class TestFunctions:

    def test_parse_func(self):
        p = parse("EXPORT FUNC SHARPEN_TWICE(k)\n  SHARPEN\n  BLUR $k\nENDFUNC")
        fn = p.commands[0]
        assert isinstance(fn, FuncDef) and fn.exported and fn.is_command
        assert fn.params == ["k"] and [c.name for c in fn.body] == ["SHARPEN", "BLUR"]
        assert not parse("FUNC f()\nRETURN 1\nENDFUNC").commands[0].is_command

    def test_expression_function(self):
        src = """\
            FUNC clamp(v, lo, hi)
                IF $v < $lo
                    RETURN $lo
                ENDIF
                RETURN min($v, $hi)
            ENDFUNC
            SET a clamp(-5, 0, 255)
            SET b clamp(300, 0, 255)
            SET c clamp(7, 0, 255)
        """
        v = run(textwrap.dedent(src)).context["vars"]
        assert (v["a"], v["b"], v["c"]) == (0, 255, 7)

    def test_command_function_receives_and_returns_image(self, probe_reg):
        import cv2
        src = textwrap.dedent("""\
            FUNC DARKEN(k)
                PROBE $k
                INVERT
            ENDFUNC
            USE ZEROS(4, 3)
            DARKEN 7
        """)
        ex = run(src, reg=probe_reg)
        assert probe_reg.log == [(7,)]
        assert ex.image.max() == 255 and ex.image.shape == (3, 4)

    def test_command_func_before_registry(self, probe_reg):
        src = "FUNC GRAYSCALE()\n  PROBE 1\nENDFUNC\nUSE ZEROS(2, 2, 3)\nGRAYSCALE"
        ex = run(src, reg=probe_reg)
        assert probe_reg.log == [(1,)] and ex.image.ndim == 3   # builtin not called

    def test_locals_do_not_leak(self):
        src = textwrap.dedent("""\
            SET x 1
            FUNC f(a)
                SET x 99
                SET tmp $a
                RETURN $x + $tmp
            ENDFUNC
            SET r f(1)
        """)
        v = run(src).context["vars"]
        assert v == {"x": 1, "r": 100}

    def test_global_writes(self):
        src = textwrap.dedent("""\
            SET counter 0
            FUNC BUMP()
                GLOBAL counter
                SET counter $counter + 1
            ENDFUNC
            REPEAT 3
                BUMP
            END
        """)
        assert run(src).context["vars"]["counter"] == 3

    def test_reads_fall_back_to_globals(self):
        assert run("SET g 5\nFUNC f()\nRETURN $g * 2\nENDFUNC\nSET r f()").context["vars"]["r"] == 10

    def test_scalars_by_value_buffers_by_reference(self):
        src = textwrap.dedent("""\
            ALLOC b 4 4 1 u8 0
            FUNC poke(buf, n)
                SET buf[0, 0] 9
                SET n 100
                RETURN $n
            ENDFUNC
            SET n 1
            SET r poke(b, $n)
            SET v b[0, 0]
        """)
        v = run(src).context["vars"]
        assert v["n"] == 1 and v["r"] == 100 and v["v"] == 9

    def test_recursion(self):
        src = "FUNC fact(n)\nIF $n <= 1\nRETURN 1\nENDIF\nRETURN $n * fact($n - 1)\nENDFUNC\nSET r fact(10)"
        assert run(src).context["vars"]["r"] == 3628800

    def test_recursion_cap_and_trace(self, tmp_path):
        main = write(tmp_path, "rec.pxl", """\
            FUNC down(n)
                RETURN down($n + 1)
            ENDFUNC
            SET r down(0)
        """)
        with pytest.raises(PixRuntimeError) as info:
            run(main.read_text(), path=main)
        e = info.value
        assert "maximum call depth (256)" in str(e)
        assert len(e.trace) > 200
        assert "rec.pxl:4:7: in <main>: call down" in e.trace[0]
        assert "rec.pxl:2:12" in e.trace[1]

    def test_error_inside_function_has_trace(self, tmp_path):
        main = write(tmp_path, "m.pxl", """\
            FUNC bad(x)
                RETURN $x / 0
            ENDFUNC
            SET r bad(1)
        """)
        with pytest.raises(PixRuntimeError) as info:
            run(main.read_text(), path=main)
        e = info.value
        assert e.line == 2 and "division by zero" in str(e)
        assert "call stack" in str(e) and "in bad" in str(e)

    def test_arity_error(self):
        with pytest.raises(PixRuntimeError, match=r"takes 2 arguments \(a, b\), got 1"):
            run("FUNC f(a, b)\nRETURN 1\nENDFUNC\nSET r f(1)")

    def test_no_return_value_is_none(self):
        assert run("FUNC f()\nSET q 1\nENDFUNC\nSET r f()").context["vars"]["r"] is None

    def test_command_func_return_image(self):
        ex = run("FUNC MAKE()\nRETURN ZEROS(5, 2)\nENDFUNC\nMAKE")
        assert ex.image.shape == (2, 5)

    def test_command_func_in_expression_rejected(self):
        with pytest.raises(PixLinkError, match="cannot be called in an expression"):
            run("FUNC MAKE()\nENDFUNC\nSET r MAKE()")

    @pytest.mark.parametrize("src,msg", [
        ("REPEAT 2\nFUNC f()\nENDFUNC\nEND", "only allowed at the top level"),
        ("RETURN 1", "RETURN outside of a FUNC"),
        ("BREAK", "BREAK outside of a loop"),
        ("FUNC f(a, a)\nENDFUNC", "duplicate parameter"),
        ("FUNC f()\nREPEAT 2\nENDFUNC", "expected END"),
        ("IF 1\nIMPORT \"x.pxl\"\nENDIF", "only allowed at the top level"),
    ])
    def test_function_syntax_errors(self, src, msg):
        with pytest.raises(PixSyntaxError, match=msg):
            parse(src)

    def test_break_inside_function_loop(self):
        src = "FUNC f()\nFOR i = 1 TO 10\nIF $i == 3\nBREAK\nENDIF\nEND\nRETURN $i\nENDFUNC"
        parse(src)


# ═════════════════════════════════════════════════════════════════════════════
# Modules
# ═════════════════════════════════════════════════════════════════════════════

LIB = """\
    SET lib_k 3
    EXPORT FUNC SHARPEN_TWICE(k)
        PROBE "sharpen" $k
    ENDFUNC
    EXPORT FUNC clamp(v, lo, hi)
        RETURN max($lo, min($v, $hi))
    ENDFUNC
    FUNC hidden()
        RETURN 1
    ENDFUNC
"""


class TestModules:

    def test_parse_imports(self):
        p = parse('IMPORT "lib/f.pxl" AS f\nFROM "lib/f.pxl" IMPORT A, b')
        a, b = p.commands
        assert isinstance(a, ImportStmt) and a.alias == "f" and a.path == "lib/f.pxl"
        assert b.names == ["A", "b"] and b.name_locs[1] == (2, 28)

    def test_from_import(self, tmp_path, probe_reg):
        write(tmp_path, "lib/filters.pxl", LIB)
        main = write(tmp_path, "main.pxl", """\
            FROM "lib/filters.pxl" IMPORT SHARPEN_TWICE, clamp
            SET y clamp(300, 0, 255)
            SHARPEN_TWICE $lib_k
        """)
        ex = run(main.read_text(), reg=probe_reg, path=main)
        assert ex.context["vars"]["y"] == 255
        assert probe_reg.log == [("sharpen", 3)]

    def test_alias_import(self, tmp_path, probe_reg):
        write(tmp_path, "lib/filters.pxl", LIB)
        main = write(tmp_path, "main.pxl", """\
            IMPORT "lib/filters.pxl" AS f
            SET y f.clamp(-3, 0, 9)
            f.SHARPEN_TWICE 5
        """)
        ex = run(main.read_text(), reg=probe_reg, path=main)
        assert ex.context["vars"]["y"] == 0
        assert probe_reg.log == [("sharpen", 5)]

    def test_default_alias_is_stem(self, tmp_path, probe_reg):
        write(tmp_path, "filters.pxl", LIB)
        main = write(tmp_path, "main.pxl", 'IMPORT "filters.pxl"\nSET y filters.clamp(5, 0, 1)\n')
        assert run(main.read_text(), reg=probe_reg, path=main).context["vars"]["y"] == 1

    def test_import_relative_to_importing_file(self, tmp_path, probe_reg):
        write(tmp_path, "lib/inner.pxl", "EXPORT FUNC two()\nRETURN 2\nENDFUNC\n")
        write(tmp_path, "lib/outer.pxl", 'FROM "inner.pxl" IMPORT two\n'
                                         "EXPORT FUNC four()\nRETURN two() * 2\nENDFUNC\n")
        main = write(tmp_path, "main.pxl", 'FROM "lib/outer.pxl" IMPORT four\nSET r four()\n')
        assert run(main.read_text(), reg=probe_reg, path=main).context["vars"]["r"] == 4

    def test_module_functions_resolve_in_their_module(self, tmp_path, probe_reg):
        # helper() exists in both files; each module's code sees its own
        write(tmp_path, "m.pxl", "FUNC helper()\nRETURN 1\nENDFUNC\n"
                                 "EXPORT FUNC api()\nRETURN helper()\nENDFUNC\n")
        main = write(tmp_path, "main.pxl", 'FROM "m.pxl" IMPORT api\n'
                                          "FUNC helper()\nRETURN 2\nENDFUNC\n"
                                          "SET a api()\nSET b helper()\n")
        v = run(main.read_text(), reg=probe_reg, path=main).context["vars"]
        assert (v["a"], v["b"]) == (1, 2)

    def test_search_paths(self, tmp_path, probe_reg):
        write(tmp_path, "shared/util.pxl", "EXPORT FUNC one()\nRETURN 1\nENDFUNC\n")
        write(tmp_path, "proj/pixlang.toml", '[link]\nsearch_paths = ["../shared"]\n')
        main = write(tmp_path, "proj/main.pxl", 'FROM "util.pxl" IMPORT one\nSET r one()\n')
        assert run(main.read_text(), reg=probe_reg, path=main).context["vars"]["r"] == 1
        prog = link(main, registry=probe_reg)
        assert any(k.endswith("util.pxl") for k in prog.units)

    def test_module_top_level_restricted(self, tmp_path):
        write(tmp_path, "bad.pxl", "GRAYSCALE\nEXPORT FUNC f()\nENDFUNC\n")
        main = write(tmp_path, "main.pxl", 'FROM "bad.pxl" IMPORT f\nf\n'.replace("f\n", "", 1)
                     + "SET r 1\n")
        with pytest.raises(PixLinkError, match="only FUNC, SET, GLOBAL and IMPORT"):
            link(main)


# ═════════════════════════════════════════════════════════════════════════════
# Linker diagnostics
# ═════════════════════════════════════════════════════════════════════════════

class TestLinkerDiagnostics:

    def diag(self, path, code=None, reg=default_registry):
        prog = Linker(reg).link(path, raise_on_error=False)
        ds = prog.diagnostics
        return [d for d in ds if code is None or d.code == code]

    def test_import_cycle_full_chain(self, tmp_path):
        write(tmp_path, "a.pxl", 'FROM "b.pxl" IMPORT fb\nEXPORT FUNC fa()\nRETURN 1\nENDFUNC\n')
        write(tmp_path, "b.pxl", 'FROM "c.pxl" IMPORT fc\nEXPORT FUNC fb()\nRETURN fc()\nENDFUNC\n')
        write(tmp_path, "c.pxl", 'FROM "a.pxl" IMPORT fa\nEXPORT FUNC fc()\nRETURN fa()\nENDFUNC\n')
        main = write(tmp_path, "main.pxl", 'FROM "a.pxl" IMPORT fa\nSET r fa()\n')
        with pytest.raises(PixLinkError) as info:
            link(main)
        assert "import cycle: main.pxl → a.pxl → b.pxl → c.pxl → a.pxl" in str(info.value)
        d = next(d for d in info.value.diagnostics if d.code == "LNK003")
        assert d.file.endswith("c.pxl") and (d.line, d.col) == (1, 1)

    def test_include_cycle_at_link_time(self, tmp_path):
        write(tmp_path, "a.pxl", 'INCLUDE "b.pxl"\n')
        write(tmp_path, "b.pxl", 'INCLUDE "a.pxl"\n')
        with pytest.raises(PixLinkError, match="INCLUDE cycle: a.pxl → b.pxl → a.pxl"):
            link(tmp_path / "a.pxl")

    def test_duplicate_definition(self, tmp_path):
        main = write(tmp_path, "m.pxl", "FUNC f()\nENDFUNC\n\nFUNC f()\nENDFUNC\n")
        d = self.diag(main, "LNK002")
        assert d and "duplicate definition of FUNC 'f'" in d[0].message and d[0].line == 4

    def test_duplicate_import_name(self, tmp_path):
        write(tmp_path, "lib.pxl", "EXPORT FUNC f()\nENDFUNC\n")
        main = write(tmp_path, "m.pxl", 'FROM "lib.pxl" IMPORT f\nFUNC f()\nENDFUNC\n')
        assert self.diag(main, "LNK002")

    def test_undefined_function_location(self, tmp_path):
        main = write(tmp_path, "m.pxl", "SET a 1\nSET r  nothere($a)\n")
        d = self.diag(main, "LNK010")
        assert d and "undefined function 'nothere'" in d[0].message
        assert (d[0].line, d[0].col) == (2, 8) and d[0].file.endswith("m.pxl")
        assert "m.pxl:2:8: error:" in str(d[0])

    def test_unknown_alias_member(self, tmp_path):
        write(tmp_path, "lib.pxl", "EXPORT FUNC f()\nENDFUNC\n")
        main = write(tmp_path, "m.pxl", 'IMPORT "lib.pxl" AS l\nSET r l.g()\nl.NOPE 1\n')
        msgs = [d.message for d in self.diag(main, "LNK010")]
        assert any("no exported function 'g'" in m for m in msgs)
        assert any("no exported command 'NOPE'" in m for m in msgs)

    def test_not_exported(self, tmp_path):
        write(tmp_path, "lib.pxl", LIB)
        main = write(tmp_path, "m.pxl", 'FROM "lib.pxl" IMPORT hidden\nSET r hidden()\n')
        d = self.diag(main, "LNK008")
        assert d and "not exported" in d[0].message and (d[0].line, d[0].col) == (1, 23)

    def test_missing_symbol(self, tmp_path):
        write(tmp_path, "lib.pxl", LIB)
        main = write(tmp_path, "m.pxl", 'FROM "lib.pxl" IMPORT clmap\n')
        d = self.diag(main, "LNK007")
        assert d and "did you mean clamp" in d[0].message

    def test_module_not_found(self, tmp_path):
        main = write(tmp_path, "m.pxl", '\nIMPORT "ghost.pxl" AS g\n')
        d = self.diag(main, "LNK001")
        assert d and d[0].line == 2

    def test_unused_imports(self, tmp_path):
        write(tmp_path, "lib.pxl", LIB)
        main = write(tmp_path, "m.pxl", 'FROM "lib.pxl" IMPORT clamp, SHARPEN_TWICE\n'
                                       'IMPORT "lib.pxl" AS l\nSET r clamp(1, 0, 2)\n')
        msgs = [d.message for d in self.diag(main, "LNK013")]
        assert any("'SHARPEN_TWICE'" in m for m in msgs)
        assert any("alias 'l'" in m for m in msgs)
        assert not any("'clamp'" in m for m in msgs)
        assert all(d.severity == "warning" for d in self.diag(main, "LNK013"))

    def test_shadowing_builtin_warns(self, tmp_path):
        main = write(tmp_path, "m.pxl", "FUNC BLUR(k)\nSHARPEN\nENDFUNC\n")
        d = self.diag(main, "LNK012")
        assert d and d[0].severity == "warning" and "shadows" in d[0].message

    def test_unknown_command_warning(self, tmp_path):
        main = write(tmp_path, "m.pxl", "FLY_TO_MOON\n")
        d = self.diag(main, "LNK011")
        assert d and d[0].severity == "warning"

    def test_include_spliced_with_file_tags(self, tmp_path):
        write(tmp_path, "sub/part.pxl", "BLUR 3\nINCLUDE \"deeper.pxl\"\n")
        write(tmp_path, "sub/deeper.pxl", "SHARPEN\n")
        main = write(tmp_path, "m.pxl", 'LOAD "x.png"\nINCLUDE "sub/part.pxl"\n')
        prog = link(main)
        names = [(s.name, Path(s.file).name, s.line) for s in prog.body]
        assert names == [("LOAD", "m.pxl", 1), ("BLUR", "part.pxl", 1),
                         ("SHARPEN", "deeper.pxl", 1)]

    def test_include_missing_link_time(self, tmp_path):
        main = write(tmp_path, "m.pxl", 'INCLUDE "ghost.pxl"\n')
        with pytest.raises(PixLinkError, match="File not found"):
            link(main)

    def test_object_unit_metadata(self, tmp_path):
        import hashlib
        write(tmp_path, "lib.pxl", LIB)
        main = write(tmp_path, "m.pxl", 'FROM "lib.pxl" IMPORT clamp\nSET r clamp(1, 0, 2)\nPROBE 1\n')
        prog = link(main, registry=None)
        lib = next(u for k, u in prog.units.items() if k.endswith("lib.pxl"))
        assert lib.sha256 == hashlib.sha256((tmp_path / "lib.pxl").read_bytes()).hexdigest()
        assert set(lib.exports) == {"SHARPEN_TWICE", "clamp"}
        assert set(lib.funcs) == {"SHARPEN_TWICE", "clamp", "hidden"}
        entry = prog.units[prog.entry]
        kinds = {(r.name, r.kind) for r in entry.refs}
        assert ("clamp", "call") in kinds and ("PROBE", "command") in kinds
        call = next(r.node for r in entry.refs if r.name == "clamp")
        assert call.func_ref.endswith("lib.pxl::clamp")
        assert [g.var_name for g in lib.globals] == ["lib_k"]

    def test_symbol_map(self, tmp_path):
        write(tmp_path, "lib.pxl", LIB)
        main = write(tmp_path, "m.pxl", 'FROM "lib.pxl" IMPORT clamp\nSET r clamp(1, 0, 2)\n')
        text = link(main).symbol_map()
        assert "m.pxl (entry)" in text and "export function clamp(v, lo, hi)" in text
        assert "clamp" in text and "-> " in text


# ═════════════════════════════════════════════════════════════════════════════
# LinkedProgram execution
# ═════════════════════════════════════════════════════════════════════════════

class TestLinkedProgram:

    def test_run_linked_program_without_files(self, tmp_path, probe_reg):
        write(tmp_path, "lib.pxl", LIB)
        write(tmp_path, "part.pxl", "PROBE \"included\"\n")
        main = write(tmp_path, "main.pxl", """\
            FROM "lib.pxl" IMPORT SHARPEN_TWICE, clamp
            INCLUDE "part.pxl"
            SET y clamp(9, 0, 5)
            SHARPEN_TWICE $lib_k
        """)
        prog = link(main, registry=probe_reg)
        assert isinstance(prog, LinkedProgram)
        for f in tmp_path.glob("*.pxl"):
            f.unlink()                       # nothing is read at run time
        ex = Executor(registry=probe_reg)
        ex.run(prog)
        assert probe_reg.log == [("included",), ("sharpen", 3)]
        assert ex.context["vars"]["y"] == 5

    def test_run_returns_final_image(self):
        img = Executor(registry=default_registry).run(parse("USE ZEROS(3, 2)\nINVERT"))
        assert img.shape == (2, 3) and img.min() == 255

    def test_runtime_include_registers_funcs(self, tmp_path, probe_reg):
        write(tmp_path, "defs.pxl", "FUNC HELLO(x)\nPROBE \"hi\" $x\nENDFUNC\n")
        main = write(tmp_path, "main.pxl", 'INCLUDE "defs.pxl"\nHELLO 4\n')
        run(main.read_text(), reg=probe_reg, path=main)
        assert probe_reg.log == [("hi", 4)]

    def test_legacy_run_body_entry_point(self, probe_reg):
        ex = Executor(registry=probe_reg)
        stmts = parse("FUNC P2()\nPROBE 2\nENDFUNC\nP2").commands
        ex._run_body(stmts, None)
        assert probe_reg.log == [(2,)]


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

class TestLinkCLI:

    def cli(self, *args):
        import os
        env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
        return subprocess.run([sys.executable, "-m", "pixlang.cli", *args],
                              capture_output=True, text=True, env=env)

    def test_link_ok_with_map(self, tmp_path):
        write(tmp_path, "lib.pxl", LIB)
        main = write(tmp_path, "m.pxl", 'FROM "lib.pxl" IMPORT clamp\nSET r clamp(1, 0, 2)\n')
        r = self.cli("link", str(main), "--map")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "clamp" in r.stdout and "(entry)" in r.stdout

    def test_link_errors_exit_1(self, tmp_path):
        main = write(tmp_path, "m.pxl", "SET r nothere(1)\n")
        r = self.cli("link", str(main))
        assert r.returncode == 1
        assert "undefined function 'nothere'" in r.stdout
        assert "m.pxl:1:7" in r.stdout

    def test_run_reports_syntax_error_with_caret(self, tmp_path):
        main = write(tmp_path, "m.pxl", "BLUR 3\nEND\n")
        r = self.cli("run", str(main))
        assert r.returncode == 1
        assert "END without REPEAT" in r.stdout and "^" in r.stdout
