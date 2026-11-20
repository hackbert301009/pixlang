"""Python-style chained comparisons: a < b < c means a < b and b < c."""
import pytest

from pixlang.commands import registry
from pixlang.commands.registry import CommandRegistry
from pixlang.executor import Executor
from pixlang.parser import PixAssertionError, PixRuntimeError, parse
from pixlang.parser.ast_nodes import Binary


def run(src):
    ex = Executor(registry=registry)
    ex.run(parse(src))
    return ex.context["vars"]


def value(expr, setup=""):
    return run(f"{setup}\nSET r {expr}")["r"]


COUNTER_FUNC = """
SET calls 0
FUNC mid()
    GLOBAL calls
    SET calls ($calls + 1)
    RETURN 5
ENDFUNC
"""


@pytest.mark.parametrize("expr,expected", [
    ("0 < 5 < 10", True),
    ("0 < 50 < 10", False),
    ("10 > 5 > 0", True),
    ("10 > 5 > 7", False),
    ("1 <= 1 <= 1", True),
    ("1 < 2 == 2", True),
    ("1 < 2 == 3", False),
    ("0 < 5 <= 10 != 7", True),
    ("0 < 7 <= 10 != 10", False),
    ("1 == 1 == 1", True),
    ("1 != 2 != 1", True),
])
def test_chain_value(expr, expected):
    assert value(expr) is expected


def test_failing_chain_assert_raises():
    with pytest.raises(PixAssertionError):
        run('SET n 50\nASSERT 0 < $n < 10 "should FAIL"')


def test_passing_chain_assert():
    run('SET n 5\nASSERT 0 < $n < 10')


def test_chain_in_if_elif_while():
    vars_ = run("""
SET n 50
SET hit ""
IF 0 < $n < 10
    SET hit "if"
ELIF 10 < $n < 100
    SET hit "elif"
ENDIF
SET i 0
WHILE 0 <= $i < 3
    SET i ($i + 1)
END
""")
    assert vars_["hit"] == "elif"
    assert vars_["i"] == 3


def test_chain_inside_parenthesised_command_arg():
    reg = CommandRegistry()
    seen = []

    @reg.register("SINK", source="test")
    def sink(image, flag):
        seen.append(flag)
        return image

    Executor(registry=reg).run(parse("SET n 5\nSINK (0 < $n < 10)\nSINK (0 < $n < 3)"))
    assert seen == [True, False]


def test_middle_operand_evaluated_once():
    vars_ = run(COUNTER_FUNC + "\nSET r (0 < mid() < 10)")
    assert vars_["r"] is True
    assert vars_["calls"] == 1


def test_middle_operand_evaluated_once_in_long_chain():
    vars_ = run(COUNTER_FUNC + "\nSET r (0 < mid() <= 5 < 10)")
    assert vars_["r"] is True
    assert vars_["calls"] == 1


def test_short_circuit_skips_later_operands():
    vars_ = run(COUNTER_FUNC + "\nSET r (9 < 1 < mid())")
    assert vars_["r"] is False
    assert vars_["calls"] == 0


def test_and_or_not_precedence():
    assert value("0 < 5 < 10 and 1 < 2 < 3") is True
    assert value("0 < 50 < 10 or 1 < 2 < 3") is True
    assert value("not 0 < 50 < 10") is True
    assert value("not 0 < 5 < 10") is False
    assert value("1 < 2 < 3 and 5 < 4 < 6") is False


def test_arithmetic_binds_tighter_than_chain():
    assert value("0 < 2 + 3 < 2 * 5") is True
    assert value("0 < 2 + 3 < 2 * 2") is False


def test_parenthesised_comparison_is_not_a_chain():
    with pytest.raises(PixRuntimeError):
        value("(1 < 2) < 3")


def test_parenthesised_equality_with_bool_stays_valid():
    assert value("(1 < 2) == true") is True


def test_single_comparison_keeps_binary_shape():
    node = parse("SET r 1 < 2\n").commands[0].value
    assert isinstance(node, Binary)


def test_legacy_assert_and_if_forms():
    run('SET n 3\nASSERT n == 3\nASSERT $n >= 3\nIF n == 3\n  SET ok 1\nENDIF')


def test_string_numeric_coercion_in_chain():
    assert value('"1" < 2 < 3') is True


@pytest.mark.parametrize("expr", ["true < 10", "10 > true", "false <= 0.5", "1 < true < 3"])
def test_bool_vs_number_ordering_is_an_error(expr):
    with pytest.raises(PixRuntimeError, match="bool"):
        value(expr)


def test_bool_equality_with_number_still_allowed():
    assert value("true == 1") is True
    assert value("true != 2") is True


def test_chain_text_appears_in_assert_message():
    with pytest.raises(PixAssertionError) as exc:
        run("SET n 50\nASSERT 0 < $n < 10")
    assert "0 < $n < 10" in str(exc.value)
