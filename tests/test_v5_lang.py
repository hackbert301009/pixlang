# tests/test_v5_lang.py
"""
v0.5 language-core tests, part 1: regression tests for the v0.4 bugs.

  - parser termination / unterminated blocks
  - INCLUDE cycles and include-relative resolution
  - CRLF, number literals, binary minus, word classification
  - newline-terminated statements, string escapes
  - error locations (unknown command), loop-variable scoping
  - IF TypeError / non-integer REPEAT
  - linter fixes, config loader fixes
"""
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pixlang.commands import registry as default_registry
from pixlang.executor import Executor
from pixlang.linter import Linter, Severity
from pixlang.parser import (
    KEYWORDS, PixRuntimeError, PixSyntaxError, parse, tokenize,
)
from pixlang.parser.ast_nodes import Command, IfBlock, RepeatBlock, SetVar


def run(src, reg=None, path=None, **kw):
    ex = Executor(registry=reg or default_registry, pipeline_path=path, **kw)
    ex.run(parse(src, filename=str(path) if path else None))
    return ex


def syntax_error(src):
    with pytest.raises(PixSyntaxError) as info:
        parse(src)
    return info.value


@pytest.fixture
def img_file(tmp_path):
    import cv2
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), np.full((40, 60, 3), 100, dtype=np.uint8))
    return p


# ═════════════════════════════════════════════════════════════════════════════
# 1. Stray terminators / unterminated blocks
# ═════════════════════════════════════════════════════════════════════════════

class TestBlockTermination:

    @pytest.mark.parametrize("src,needle", [
        ("BLUR 3\nEND\nGRAYSCALE", "END without REPEAT"),
        ("BLUR 3\nENDIF", "ENDIF without IF"),
        ("ROI_RESET", "ROI_RESET without ROI"),
        ("ENDFUNC", "ENDFUNC without FUNC"),
        ("ELSE", "ELSE without IF"),
    ])
    def test_stray_terminator(self, src, needle):
        err = syntax_error(src)
        assert needle in err.msg
        assert isinstance(err, SyntaxError)

    def test_stray_end_location(self):
        err = syntax_error("BLUR 3\nEND\nGRAYSCALE")
        assert err.lineno == 2 and err.offset == 1

    @pytest.mark.parametrize("src,opener,closer", [
        ("REPEAT 3\nBLUR 3\n", "REPEAT", "END"),
        ("SET x 1\nIF x == 1\nBLUR 3", "IF", "ENDIF"),
        ("ROI 0 0 5 5\nBLUR 3", "ROI", "ROI_RESET"),
        ("WHILE true\nBLUR 3", "WHILE", "END"),
    ])
    def test_unterminated_block_names_opener(self, src, opener, closer):
        err = syntax_error(src)
        assert f"unterminated {opener}" in err.msg
        assert closer in err.msg
        assert f"line {src.splitlines().index(next(l for l in src.splitlines() if l.startswith(opener))) + 1}" in err.msg

    def test_wrong_terminator(self):
        err = syntax_error("SET x 1\nIF x == 1\nBLUR 3\nEND")
        assert "expected ENDIF" in err.msg and "found END" in err.msg
        assert err.lineno == 4


# ═════════════════════════════════════════════════════════════════════════════
# 2–3. INCLUDE cycles and relative resolution
# ═════════════════════════════════════════════════════════════════════════════

class TestIncludeFixes:

    def test_include_cycle_detected(self, tmp_path, img_file):
        (tmp_path / "a.pxl").write_text('INCLUDE "b.pxl"\n')
        (tmp_path / "b.pxl").write_text('INCLUDE "a.pxl"\n')
        main = tmp_path / "a.pxl"
        with pytest.raises(PixRuntimeError, match=r"a\.pxl → b\.pxl → a\.pxl"):
            run(main.read_text(), path=main)

    def test_include_self_cycle(self, tmp_path):
        main = tmp_path / "self.pxl"
        main.write_text('INCLUDE "self.pxl"\n')
        with pytest.raises(PixRuntimeError, match="cycle"):
            run(main.read_text(), path=main)

    def test_nested_include_relative_to_including_file(self, tmp_path, img_file):
        calls = []
        from pixlang.commands.registry import CommandRegistry
        from pixlang.commands.builtin import register_all
        reg = CommandRegistry()
        register_all(reg)

        @reg.register("LEAF_V5", source="test")
        def _leaf(image):
            calls.append(1)
            return image

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "middle.pxl").write_text('INCLUDE "leaf.pxl"\n')
        (sub / "leaf.pxl").write_text("LEAF_V5\n")
        # a decoy leaf.pxl next to the root must NOT be picked up
        (tmp_path / "leaf.pxl").write_text("GRAYSCALE\n")
        main = tmp_path / "main.pxl"
        main.write_text(f'LOAD "{img_file}"\nINCLUDE "sub/middle.pxl"\n')
        run(main.read_text(), reg=reg, path=main)
        assert calls == [1]

    def test_include_without_cycle_twice_ok(self, tmp_path, img_file):
        (tmp_path / "step.pxl").write_text("SET n $n + 1\n")
        main = tmp_path / "main.pxl"
        main.write_text('SET n 0\nINCLUDE "step.pxl"\nINCLUDE "step.pxl"\n')
        ex = run(main.read_text(), path=main)
        assert ex.context["vars"]["n"] == 2


# ═════════════════════════════════════════════════════════════════════════════
# 4–8. Lexer fixes
# ═════════════════════════════════════════════════════════════════════════════

class TestLexerFixes:

    def test_crlf(self):
        p = parse('LOAD "a.png"\r\nBLUR 3\r\nGRAYSCALE\r\n')
        assert [c.name for c in p.commands] == ["LOAD", "BLUR", "GRAYSCALE"]
        assert p.commands[1].line == 2

    def test_cr_only(self):
        assert len(parse("BLUR 3\rGRAYSCALE").commands) == 2

    @pytest.mark.parametrize("text,value", [
        ("-1.5", -1.5), (".5", 0.5), ("1e3", 1000.0), ("2.5e-1", 0.25),
        ("0xFF", 255), ("0b1010", 10), ("-7", -7), ("0x1_0", 16),
    ])
    def test_number_literals(self, text, value):
        p = parse(f"SET v {text}")
        assert p.commands[0].value == value

    def test_binary_minus_not_negative_literal(self):
        toks = tokenize("5-3")
        assert [(t.type, t.value) for t in toks] == [("INT", "5"), ("OP", "-"), ("INT", "3")]
        assert run("SET v 5-3").context["vars"]["v"] == 2

    def test_spaced_minus_in_expression(self):
        assert run("SET a 10\nSET v $a -3").context["vars"]["v"] == 7

    def test_negative_args_still_negative(self):
        assert parse("ROTATE -90").commands[0].args == [-90]
        assert parse("BLUR -1").commands[0].args == [-1]
        assert parse("RESIZE $w -1").commands[0].args[1] == -1

    @pytest.mark.parametrize("name", ["Foo_bar", "X", "_tmp", "camelCase", "ABC_1"])
    def test_any_word_in_name_positions(self, name):
        p = parse(f"SET {name} 3\nFOR {name} = 1 TO 2\nEND")
        assert p.commands[0].var_name == name
        assert p.commands[1].var_name == name

    def test_word_classification(self):
        types = {t.value: t.type for t in tokenize("Foo_bar X width GRAYSCALE SET")}
        assert types == {"Foo_bar": "IDENT", "X": "COMMAND", "width": "IDENT",
                         "GRAYSCALE": "COMMAND", "SET": "KEYWORD"}

    def test_newline_terminates_statement(self):
        err = syntax_error("BLUR 3\n5")
        assert err.lineno == 2

    def test_continuation_still_works(self):
        p = parse("RESIZE \\\n  640 \\\n  480\nGRAYSCALE")
        assert p.commands[0].args == [640, 480]
        assert p.commands[1].line == 4

    def test_examples_parse(self):
        for f in Path("examples").glob("*.pxl"):
            parse(f.read_text(), filename=str(f))

    @pytest.mark.parametrize("src,expected", [
        (r'"a\"b"', 'a"b'), (r'"a\\b"', "a\\b"), (r'"x\ny"', "x\ny"),
        (r'"t\tt"', "t\tt"), (r'"C:\data\img.png"', "C:\\data\\img.png"),
    ])
    def test_string_escapes(self, src, expected):
        assert parse(f"LOAD {src}").commands[0].args == [expected]

    def test_unterminated_string(self):
        err = syntax_error('BLUR 3\nLOAD "abc\n')
        assert "unterminated string" in err.msg
        assert (err.lineno, err.offset) == (2, 6)

    def test_multiline_string_line_counting(self):
        p = parse('DRAW_TEXT "a\nb" 1 2\nGRAYSCALE')
        assert p.commands[0].args[0] == "a\nb"
        assert p.commands[1].line == 3

    def test_columns(self):
        toks = tokenize("BLUR  $k")
        assert [t.col for t in toks] == [1, 7]

    def test_bad_character_location(self):
        err = syntax_error("BLUR 3\nBLUR @")
        assert (err.lineno, err.offset) == (2, 6)

    def test_syntax_error_attrs(self):
        with pytest.raises(PixSyntaxError) as info:
            parse("BLUR 3\nEND", filename="x.pxl")
        e = info.value
        assert e.filename == "x.pxl" and e.line == 2 and e.col == 1 and e.file == "x.pxl"
        assert "x.pxl:2:1: error:" in str(e)
        pretty = e.pretty()
        assert "    END" in pretty and "^" in pretty

    def test_keywords_complete(self):
        for kw in ["SET", "IF", "ELIF", "ELSE", "ENDIF", "REPEAT", "END", "WHILE",
                   "FOR", "TO", "STEP", "BREAK", "CONTINUE", "FUNC", "ENDFUNC",
                   "RETURN", "GLOBAL", "EXPORT", "IMPORT", "FROM", "AS", "ALLOC",
                   "BUF", "USE", "FREE", "WITH", "and", "or", "not", "true", "false"]:
            assert kw in KEYWORDS
        for t in tokenize(" ".join(sorted(KEYWORDS))):
            assert t.type == "KEYWORD"


# ═════════════════════════════════════════════════════════════════════════════
# 9–11. Runtime error fixes
# ═════════════════════════════════════════════════════════════════════════════

class TestRuntimeFixes:

    def test_unknown_command_has_location(self, img_file):
        with pytest.raises(PixRuntimeError) as info:
            run(f'LOAD "{img_file}"\n\n  FLY_TO_MOON 3')
        e = info.value
        assert e.line == 3 and e.col == 3
        assert "line 3" in str(e) and "FLY_TO_MOON" in str(e)

    def test_command_error_keeps_prefix(self):
        with pytest.raises(RuntimeError, match=r"\[PixLang\] Error at line 1"):
            run("GRAYSCALE")

    def test_iter_innermost_and_restored(self):
        ex = run(textwrap.dedent("""\
            SET seen ""
            REPEAT 2
                REPEAT 3
                    SET seen $seen + $ITER
                END
                SET seen $seen + "|" + $ITER
            END
        """))
        assert ex.context["vars"]["seen"] == "012|0012|1"
        assert "ITER" not in ex.context["vars"]
        assert "ITER1" not in ex.context["vars"]

    def test_iter_visible_to_commands(self):
        from pixlang.commands.registry import CommandRegistry
        reg = CommandRegistry()
        seen = []

        @reg.register("CAP")
        def _cap(image, _ctx=None):
            seen.append((_ctx["vars"]["ITER"], _ctx["vars"]["ITER1"]))
            return image

        run("REPEAT 2\nREPEAT 2\nCAP\nEND\nCAP\nEND", reg=reg)
        assert seen == [(0, 1), (1, 2), (0, 1), (0, 1), (1, 2), (1, 2)]

    def test_if_type_error_is_reported(self):
        with pytest.raises(PixRuntimeError, match="cannot compare str") as info:
            run('SET mode "fast"\nIF mode > 3\nENDIF')
        assert info.value.line == 2

    def test_if_string_number_coercion_still_works(self):
        ex = run('SET t "120"\nSET hit 0\nIF t > 100\nSET hit 1\nENDIF')
        assert ex.context["vars"]["hit"] == 1

    def test_if_undefined_variable(self):
        with pytest.raises(RuntimeError, match=r"Variable '\$nope' is not defined"):
            run("IF nope == 1\nENDIF")

    @pytest.mark.parametrize("count", ["2.9", "2.0", '"abc"', "true"])
    def test_repeat_non_integer(self, count):
        with pytest.raises(PixRuntimeError, match="must be an integer"):
            run(f"REPEAT {count}\nEND")

    def test_repeat_negative(self):
        with pytest.raises(PixRuntimeError, match="≥ 0"):
            run("REPEAT -1\nEND")


# ═════════════════════════════════════════════════════════════════════════════
# 12. Linter fixes
# ═════════════════════════════════════════════════════════════════════════════

class TestLinterFixes:

    def lint(self, src, path=None, **kw):
        return Linter(default_registry, **kw).lint(
            parse(src, filename=str(path) if path else None), path=path)

    def codes(self, diags):
        return {d.code for d in diags}

    def test_flatten_descends_into_roi(self):
        d = self.lint('LOAD "a.png"\nROI 0 0 5 5\nFLY_X\nROI_RESET\nSAVE "o.png"')
        assert "PX011" in self.codes(d)

    def test_flatten_descends_into_else_and_loops(self):
        d = self.lint('LOAD "a.png"\nSET x 1\nIF x == 1\nELSE\nFIND_CONTOURS\nENDIF\n'
                      'SAVE "o.png"')
        assert "PX003" in self.codes(d)

    def test_px001_load_glob_ok(self):
        d = self.lint('LOAD_GLOB "*.jpg"\nGRAYSCALE\nSAVE_EACH "o/{stem}.png"')
        assert "PX001" not in self.codes(d)
        assert "PX002" not in self.codes(d)

    def test_px001_include_first_ok(self, tmp_path):
        (tmp_path / "load.pxl").write_text('LOAD "a.png"\n')
        main = tmp_path / "m.pxl"
        main.write_text('INCLUDE "load.pxl"\nGRAYSCALE\nSAVE "o.png"\n')
        d = self.lint(main.read_text(), path=main)
        assert "PX001" not in self.codes(d)
        d2 = self.lint('INCLUDE "unknown.pxl"\nGRAYSCALE\nSAVE "o.png"')
        assert "PX001" not in self.codes(d2)

    def test_px001_included_file_checked(self, tmp_path):
        (tmp_path / "proc.pxl").write_text("GRAYSCALE\n")
        main = tmp_path / "m.pxl"
        main.write_text('INCLUDE "proc.pxl"\nSAVE "o.png"\n')
        assert "PX001" in self.codes(self.lint(main.read_text(), path=main))

    def test_px005_order_aware(self):
        d = self.lint('LOAD "a.png"\nBLUR $k\nSET k 3\nSAVE "o.png"')
        msgs = [x.message for x in d if x.code == "PX005"]
        assert msgs and "SET later" in msgs[0]

    def test_px005_toml_vars(self, tmp_path):
        (tmp_path / "pixlang.toml").write_text("[variables]\nk = 5\n")
        main = tmp_path / "m.pxl"
        main.write_text('LOAD "a.png"\nBLUR $k\nSAVE "o.png"\n')
        assert "PX005" not in self.codes(self.lint(main.read_text(), path=main))
        assert "PX005" in self.codes(self.lint(main.read_text()))

    def test_px005_included_definitions(self, tmp_path):
        (tmp_path / "vars.pxl").write_text("SET k 3\n")
        main = tmp_path / "m.pxl"
        main.write_text('INCLUDE "vars.pxl"\nLOAD "a.png"\nBLUR $k\nSAVE "o.png"\n')
        assert "PX005" not in self.codes(self.lint(main.read_text(), path=main))

    def test_px005_loop_vars(self):
        src = ('LOAD "a.png"\nREPEAT 2 AS i\nBLUR $i\nBLUR $ITER\nEND\n'
               'FOR j = 1 TO 2\nBLUR $j\nEND\nSAVE "o.png"')
        assert "PX005" not in self.codes(self.lint(src))
        assert "PX005" in self.codes(self.lint('LOAD "a.png"\nBLUR $ITER\nSAVE "o.png"'))

    def test_px004_no_repr_leak(self):
        d = self.lint('LOAD "a.png"\nOVERLAY snap 0.5\nSAVE "o.png"')
        msg = next(x.message for x in d if x.code == "PX004")
        assert "'snap'" in msg and "BareIdent" not in msg

    def test_px004_buffers_count(self):
        d = self.lint('LOAD "a.png"\nBUF snap\nOVERLAY "snap" 0.5\nSAVE "o.png"')
        assert "PX004" not in self.codes(d)

    def test_px008_zero_is_warning(self):
        d = self.lint('LOAD "a.png"\nREPEAT 0\nEND\nSAVE "o.png"')
        px8 = [x for x in d if x.code == "PX008"]
        assert px8 and px8[0].severity == Severity.WARNING
        d = self.lint('LOAD "a.png"\nREPEAT -2\nEND\nSAVE "o.png"')
        assert [x for x in d if x.code == "PX008"][0].severity == Severity.ERROR

    def test_internal_rule_error_reported(self, monkeypatch):
        from pixlang.linter import engine

        def broken(ctx, flat, registry):
            raise ValueError("boom")
        monkeypatch.setattr(engine, "_RULES", engine._RULES + [broken])
        d = self.lint('LOAD "a.png"\nSAVE "o.png"')
        assert any(x.code == "PX000" and "boom" in x.message for x in d)

    def test_lint_ignore(self, tmp_path):
        (tmp_path / "pixlang.toml").write_text('[lint]\nignore = ["PX002"]\n')
        main = tmp_path / "m.pxl"
        main.write_text('LOAD "a.png"\nGRAYSCALE\n')
        assert "PX002" not in self.codes(self.lint(main.read_text(), path=main))
        assert "PX002" not in self.codes(self.lint(main.read_text(), ignore=["PX002"]))

    def test_px012_typo_warning(self):
        d = self.lint('SET width 640\nLOAD "a.png"\nRESIZE widht 480\nSAVE "o.png"')
        px = [x for x in d if x.code == "PX012"]
        assert px and "$width" in px[0].message and px[0].line == 3
        d = self.lint('LOAD "a.png"\nGRAYSCALE\nHEATMAP jet\nSAVE "o.png"')
        assert "PX012" not in self.codes(d)

    def test_diagnostic_has_col(self):
        d = self.lint('LOAD "a.png"\n  FLY_X\nSAVE "o.png"')
        px = next(x for x in d if x.code == "PX011")
        assert (px.line, px.col) == (2, 3)

    def test_statement_alias_unique(self):
        import pixlang.parser.ast_nodes as an
        src = Path(an.__file__).read_text()
        assert src.count("Statement = Union") == 1


# ═════════════════════════════════════════════════════════════════════════════
# 13. Config loader
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigFixes:

    def test_minimal_parser_lists_and_comments(self, tmp_path):
        from pixlang.config.loader import _simple_toml_parse
        f = tmp_path / "pixlang.toml"
        f.write_text(textwrap.dedent('''\
            [variables]
            width = 640   # px
            name = "a # b"
            esc = "q\\"x"
            lst = [1, 2.5, "x", true]
            hexv = 0xFF
            [lint]
            ignore = ["PX006", "PX009"]  # trailing
            [link]
            search_paths = ['lib']
        '''))
        data = _simple_toml_parse(f)
        assert data["variables"] == {"width": 640, "name": "a # b", "esc": 'q"x',
                                     "lst": [1, 2.5, "x", True], "hexv": 255}
        assert data["lint"]["ignore"] == ["PX006", "PX009"]
        assert data["link"]["search_paths"] == ["lib"]

    def test_minimal_parser_rejects_garbage(self, tmp_path):
        from pixlang.config.loader import _simple_toml_parse
        f = tmp_path / "pixlang.toml"
        f.write_text("@@@@broken@@@@\n")
        with pytest.raises(ValueError):
            _simple_toml_parse(f)

    def test_search_paths_and_loop_cap(self, tmp_path):
        from pixlang.config import load_config
        (tmp_path / "pixlang.toml").write_text(
            '[link]\nsearch_paths = ["lib"]\n[run]\nmax_loop_iterations = 50\n')
        cfg = load_config(tmp_path)
        assert cfg.link_search_paths == [(tmp_path / "lib").resolve()]
        assert cfg.max_loop_iterations == 50
