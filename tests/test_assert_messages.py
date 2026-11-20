"""ASSERT message interpolation, number formatting and message syntax errors."""
import pytest

from pixlang.commands import registry
from pixlang.executor import Executor
from pixlang.parser import PixAssertionError, PixSyntaxError, parse


def failure(source):
    ex = Executor(registry=registry)
    with pytest.raises(PixAssertionError) as exc:
        ex.run(parse(source))
    return str(exc.value)


def test_message_interpolates_variables_and_format_specs():
    text = failure('SET i 3\nSET slot_mean 12.3456\nSET dark_min 50\n'
                   'ASSERT $slot_mean > $dark_min "slot $i empty (mean $slot_mean:.2f)"')
    assert text.endswith("slot 3 empty (mean 12.35)")


def test_message_names_the_failing_iteration():
    text = failure('SET bad 2\nREPEAT 5 AS i\n  ASSERT $i != $bad "iteration $i failed"\nEND')
    assert "iteration 2 failed" in text


def test_dollar_dollar_is_a_literal_dollar():
    assert failure('ASSERT 1 == 2 "costs $$5"').endswith("costs $5")


def test_braced_names_delimit_adjacent_text():
    text = failure('SET i 4\nASSERT 1 == 2 "slot${i}th and ${i:03d}"')
    assert text.endswith("slot4th and 004")


def test_plain_strings_are_unchanged():
    assert failure('ASSERT 1 == 2 "plain message"').endswith("plain message")


def test_dollar_not_followed_by_a_name_stays_literal():
    assert failure('ASSERT 1 == 2 "5$ and $ alone and $5"').endswith("5$ and $ alone and $5")


def test_colon_without_a_format_spec_is_text():
    text = failure('SET i 1\nASSERT 1 == 2 "slot $i: empty"')
    assert text.endswith("slot 1: empty")


def test_undefined_variable_leaves_the_placeholder_and_keeps_the_failure():
    text = failure('ASSERT 1 == 2 "value $nope here"')
    assert "Pipeline assertion failed" in text
    assert text.endswith("value $nope here")


def test_unusable_format_spec_falls_back_to_plain_text():
    text = failure('SET name "abc"\nASSERT 1 == 2 "name $name:.2f"')
    assert text.endswith("name abc")


def test_bool_and_float_values_render_readably():
    text = failure('SET flag true\nSET f 0.1\nASSERT 1 == 2 "$flag $f"')
    assert text.endswith("true 0.1")


def test_got_part_rounds_floats_to_six_significant_digits():
    text = failure('SET x (113.0 * 3 - 0.00000000000006)\nASSERT $x == 400')
    assert "but got 339" in text
    assert "338.99999" not in text


def test_got_part_keeps_integers_exact():
    assert "but got 1234567" in failure('SET x 1234567\nASSERT $x == 1')


def test_message_must_be_a_string_literal():
    with pytest.raises(PixSyntaxError) as exc:
        parse('SET i 1\nASSERT 1 == 2 "slot " + $i\n')
    assert "string literal" in str(exc.value)
    assert "interpolation" in str(exc.value)


def test_variable_as_message_is_rejected_with_the_same_hint():
    with pytest.raises(PixSyntaxError, match="string literal"):
        parse('SET m "x"\nASSERT 1 == 2 $m\n')
