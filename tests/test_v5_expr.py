# tests/test_v5_expr.py
"""
v0.5 language-core tests, part 2: foundation.

  - source locations on nodes, caret diagnostics
  - Pratt expression parser: precedence, associativity, literals
  - expressions in SET / IF / ELIF / WHILE / ASSERT / FOR / command args
  - legacy forms (bare idents, IF name op value, ASSERT subject op value)
"""
import textwrap

import numpy as np
import pytest

from pixlang.commands import registry as default_registry
from pixlang.executor import Executor
from pixlang.parser import (
    PixAssertionError, PixRuntimeError, PixSyntaxError, format_diagnostic,
    parse, parse_expression,
)
from pixlang.parser.ast_nodes import (
    AssertStmt, Attr, BareIdent, Binary, Call, Command, IfBlock, Index, Literal,
    SetVar, Unary, VarRef, walk_statements,
)


def run(src, **kw):
    ex = Executor(registry=default_registry, **kw)
    ex.run(parse(src))
    return ex


def ev(expr, setup=""):
    """Evaluate *expr* via SET and return the value."""
    ex = run(f"{setup}\nSET __r {expr}")
    return ex.context["vars"]["__r"]


# ═════════════════════════════════════════════════════════════════════════════
# Locations & diagnostics
# ═════════════════════════════════════════════════════════════════════════════

class TestLocations:

    def test_every_node_has_line_col_file(self):
        src = textwrap.dedent("""\
            SET x 1
            IF $x > 0
              REPEAT 2
                BLUR (3 + $x)
              END
            ENDIF
        """)
        p = parse(src, filename="f.pxl")
        for stmt in walk_statements(p.commands):
            assert stmt.line > 0 and stmt.col > 0 and stmt.file == "f.pxl"
        blur = p.commands[1].body[0].body[0]
        assert (blur.line, blur.col) == (4, 5)
        arg = blur.args[0]
        assert isinstance(arg, Binary) and arg.left.col == 11

    def test_caret_rendering(self):
        text = format_diagnostic("error", "boom", None, 2, 6,
                                 source="BLUR 3\nBLUR (3 + )\n")
        lines = text.splitlines()
        assert lines[0] == "<input>:2:6: error: boom"
        assert lines[1] == "    BLUR (3 + )"
        assert lines[2] == "         ^"

    def test_syntax_error_caret_points_at_token(self):
        with pytest.raises(PixSyntaxError) as info:
            parse("SET x 1\nSET y (2 + )")
        e = info.value
        assert (e.lineno, e.offset) == (2, 12)
        assert e.pretty().splitlines()[-1].index("^") == 4 + 11

    def test_runtime_error_location(self):
        with pytest.raises(PixRuntimeError) as info:
            run("SET a 1\nSET b $a / 0")
        e = info.value
        assert e.line == 2 and "division by zero" in str(e)
        assert e.col == 10          # the operator


# ═════════════════════════════════════════════════════════════════════════════
# Expression parser
# ═════════════════════════════════════════════════════════════════════════════

class TestExpressionParser:

    @pytest.mark.parametrize("expr,value", [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("10 - 4 - 3", 3),              # left associative
        ("2 * 3 % 4", 2),
        ("7 // 2", 3),
        ("7 / 2", 3.5),
        ("1 << 2 + 1", 8),             # + binds tighter than <<
        ("1 | 2 ^ 3 & 4", 1 | (2 ^ (3 & 4))),
        ("6 & 3 << 1", 6 & (3 << 1)),
        ("1 + 2 == 3", True),
        ("1 < 2 and 2 < 1", False),
        ("1 < 2 or 2 < 1", True),
        ("not 1 == 2", True),          # not binds looser than ==
        ("not true and false", False), # (not true) and false
        ("true or false and false", True),
        ("-2 * 3", -6),
        ("-(2 + 3)", -5),
        ("~0", -1),
        ("~0xF0 & 0xFF", 0x0F),
        ("0b1010 | 0x05", 15),
        ("2.5e1 + .5", 25.5),
        ('"a" + 1 + true', "a1true"),
        ("--3", 3),
    ])
    def test_precedence_and_values(self, expr, value):
        assert ev(expr) == value

    def test_ast_shapes(self):
        e = parse_expression("a.width + f(1, $x)[2] * -b")
        assert isinstance(e, Binary) and e.op == "+"
        assert isinstance(e.left, Attr) and e.left.name == "width"
        mul = e.right
        assert isinstance(mul.left, Index) and isinstance(mul.left.target, Call)
        assert isinstance(mul.right, Unary) and mul.right.op == "-"

    def test_pointer_prefix_ops(self):
        e = parse_expression("&buf[1:2, ::2]")
        assert isinstance(e, Unary) and e.op == "&"
        assert isinstance(e.operand, Index)
        e = parse_expression("*p")
        assert isinstance(e, Unary) and e.op == "*"

    def test_namespaced_call(self):
        e = parse_expression("f.clamp(1, 2, 3)")
        assert isinstance(e, Call) and e.func == "f.clamp"

    @pytest.mark.parametrize("src", ["(1 + ", "1 +", ")", "[1]", "1 2", "f(1,", "a[1"])
    def test_bad_expressions(self, src):
        with pytest.raises(PixSyntaxError):
            parse_expression(src)

    def test_to_source_roundtrip(self):
        e = parse_expression("(1 + $x) * f(2)")
        assert parse_expression(e.to_source()).to_source() == e.to_source()


# ═════════════════════════════════════════════════════════════════════════════
# Expressions in statements
# ═════════════════════════════════════════════════════════════════════════════

class TestExpressionStatements:

    def test_set_expression(self):
        ex = run("SET w 640\nSET half $w / 2\nSET q = $half // 2")
        assert ex.context["vars"]["half"] == 320.0
        assert ex.context["vars"]["q"] == 160

    def test_command_args_need_parens(self):
        p = parse("RESIZE ($w / 2) 480")
        arg = p.commands[0].args[0]
        assert isinstance(arg, Binary) and p.commands[0].args[1] == 480
        assert parse("RESIZE 640 480").commands[0].args == [640, 480]
        with pytest.raises(PixSyntaxError, match="parentheses"):
            parse("RESIZE $w / 2 480")

    def test_paren_args_evaluated(self):
        from pixlang.commands.registry import CommandRegistry
        reg = CommandRegistry()
        got = []

        @reg.register("PROBE")
        def _p(image, *args):
            got.append(args)
            return image

        ex = Executor(registry=reg)
        ex.run(parse('SET w 640\nPROBE ($w / 2) ($w > 100) ("a" + "b") (7)'))
        assert got == [(320.0, True, "ab", 7)]

    def test_legacy_bare_ident_args(self):
        from pixlang.commands.registry import CommandRegistry
        reg = CommandRegistry()
        got = []

        @reg.register("PROBE")
        def _p(image, *args):
            got.append(args)
            return image

        Executor(registry=reg).run(parse("SET k 5\nPROBE k jet"))
        assert got == [(5, "jet")]

    def test_if_elif_else(self):
        src = textwrap.dedent("""\
            SET out ""
            FOR v = 1 TO 4
                IF $v == 1
                    SET out $out + "a"
                ELIF $v == 2 or $v == 3
                    SET out $out + "b"
                ELSE
                    SET out $out + "c"
                ENDIF
            END
        """)
        assert run(src).context["vars"]["out"] == "abbc"

    def test_if_ast_fields(self):
        p = parse("SET x 1\nIF x == 1\nGRAYSCALE\nELIF $x > 3 and $x < 9\nBLUR 3\nELSE\nSHARPEN\nENDIF")
        node = p.commands[1]
        assert isinstance(node, IfBlock)
        assert (node.var_name, node.op, node.cmp_value) == ("x", "==", 1)
        assert node.cond_text == "x == 1"
        assert len(node.elifs) == 1 and node.elifs[0].body[0].name == "BLUR"
        assert node.else_body[0].name == "SHARPEN"

    def test_if_complex_condition_legacy_fields_empty(self):
        node = parse("IF $a > 1 and $b\nENDIF").commands[0]
        assert node.var_name == "" and node.op == "" and node.cond is not None

    def test_legacy_if_still_works(self):
        ex = run('SET mode "inspect"\nSET hit 0\nIF mode == "inspect"\nSET hit 1\nENDIF')
        assert ex.context["vars"]["hit"] == 1

    def test_assert_expression(self):
        run("SET a 3\nASSERT $a * 2 == 6\nASSERT $a > 1 and $a < 5 \"in range\"")
        with pytest.raises(PixAssertionError, match="in range"):
            run("SET a 9\nASSERT $a > 1 and $a < 5 \"in range\"")

    def test_assert_variable_before_image_subject(self):
        # variable named width wins over the image subject
        run("SET width 640\nASSERT width == 640")

    def test_assert_image_subject(self):
        import cv2
        ex = Executor(registry=default_registry)
        img = np.zeros((20, 30, 3), np.uint8)
        node = parse("ASSERT width == 30").commands[0]
        assert isinstance(node, AssertStmt) and node.subject == "width"
        ex._exec_assert(node, img)
        with pytest.raises(AssertionError, match="but got 30"):
            ex._exec_assert(parse("ASSERT width == 31").commands[0], img)

    def test_assert_unknown_subject(self):
        with pytest.raises(RuntimeError, match="Unknown subject"):
            run("ASSERT bogus == 1")

    def test_assert_error_attrs(self):
        with pytest.raises(PixAssertionError) as info:
            run("SET x 1\n  ASSERT $x == 2")
        assert info.value.line == 2 and info.value.col == 3

    def test_while_expression(self):
        assert run("SET i 0\nWHILE $i * $i < 50\nSET i $i + 1\nEND").context["vars"]["i"] == 8

    def test_true_false_literals(self):
        assert ev("true") is True and ev("false") is False

    def test_undefined_var_message(self):
        with pytest.raises(RuntimeError, match=r"Undefined variable '\$nope'"):
            run("SET x $nope + 1")

    def test_string_comparison(self):
        assert ev('"abc" < "abd"') is True

    def test_bitwise_on_floats_rejected(self):
        with pytest.raises(PixRuntimeError, match="needs integers"):
            ev("1.5 & 1")

    def test_builtin_functions(self):
        assert ev("min(3, 1, 2)") == 1
        assert ev("max(3, 9)") == 9
        assert ev("abs(-4)") == 4
        assert ev("int(\"0x10\")") == 16
        assert ev("float(3)") == 3.0
        assert ev("str(5) + \"!\"") == "5!"
        assert ev('len("abcd")') == 4
        assert ev("sqrt(16)") == 4.0
        assert ev("clamp(300, 0, 255)") == 255
        assert ev("round(2.6)") == 3
        assert ev("floor(2.6)") == 2 and ev("ceil(2.1)") == 3
        assert ev("hex(255)") == "0xff" and ev("bin(5)") == "0b101"

    def test_unknown_function(self):
        with pytest.raises(PixRuntimeError, match="(?i)did you mean:? clamp"):
            ev("clmap(1, 2, 3)")

    def test_builtin_wrong_arity(self):
        with pytest.raises(PixRuntimeError, match="clamp"):
            ev("clamp(1)")
