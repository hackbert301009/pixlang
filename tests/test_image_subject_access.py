"""Reading image properties in ordinary expressions: `_.width`, `_.contour_count`."""
import cv2
import numpy as np
import pytest

from pixlang.commands import registry
from pixlang.executor import Executor
from pixlang.linter import Linter, Severity
from pixlang.parser import PixRuntimeError, parse


@pytest.fixture
def image_path(tmp_path):
    canvas = np.zeros((40, 60), np.uint8)
    cv2.rectangle(canvas, (5, 5), (15, 15), 255, -1)
    cv2.rectangle(canvas, (30, 10), (50, 30), 255, -1)
    path = tmp_path / "two_blobs.png"
    cv2.imwrite(str(path), canvas)
    return path


def run_vars(source):
    ex = Executor(registry=registry)
    ex.run(parse(source))
    return ex.context["vars"]


def test_contour_count_via_current_image(image_path):
    vars_ = run_vars(f'LOAD "{image_path}"\nFIND_CONTOURS\nSET n _.contour_count\n')
    assert vars_["n"] == 2


def test_contour_count_is_zero_before_find_contours(image_path):
    assert run_vars(f'LOAD "{image_path}"\nSET n _.contour_count\n')["n"] == 0


def test_image_extremes_via_current_image(image_path):
    vars_ = run_vars(f'LOAD "{image_path}"\nSET lo _.min\nSET hi _.max\n')
    assert (vars_["lo"], vars_["hi"]) == (0, 255)


def test_geometry_attributes_still_work(image_path):
    vars_ = run_vars(f'LOAD "{image_path}"\nSET w _.width\nSET h _.height\n')
    assert (vars_["w"], vars_["h"]) == (60, 40)


def test_contour_count_can_be_compared_in_an_expression(image_path):
    run_vars(f'LOAD "{image_path}"\nFIND_CONTOURS\nASSERT _.contour_count == 2\n')


def test_contour_count_without_an_image_is_a_clear_error():
    with pytest.raises(PixRuntimeError, match="image"):
        run_vars("SET n _.contour_count\n")


def lint(source):
    return Linter(registry).lint(parse(source))


@pytest.mark.parametrize("word", ["width", "height", "channels", "depth", "min", "max",
                                  "contour_count", "ndim"])
def test_bare_image_subject_in_set_is_flagged(word):
    diags = [d for d in lint(f'LOAD "a.png"\nSET n {word}\nSAVE "o.png"\n')
             if d.code == "PX013"]
    assert len(diags) == 1
    assert diags[0].severity is Severity.WARNING
    assert f"_.{word}" in diags[0].message
    assert diags[0].line == 2


def test_bare_command_name_in_set_is_flagged():
    diags = [d for d in lint('LOAD "a.png"\nSET mode GRAYSCALE\nSAVE "o.png"\n')
             if d.code == "PX013"]
    assert len(diags) == 1
    assert "GRAYSCALE" in diags[0].message
    assert "quote" in diags[0].message.lower()


def test_bare_word_inside_a_set_expression_is_flagged():
    diags = [d for d in lint('LOAD "a.png"\nSET n (contour_count + 1)\nSAVE "o.png"\n')
             if d.code == "PX013"]
    assert len(diags) == 1


def test_defined_variable_with_subject_name_is_not_flagged():
    source = 'LOAD "a.png"\nSET width 5\nSET n width\nSAVE "o.png"\n'
    assert not [d for d in lint(source) if d.code == "PX013"]


def test_buffer_with_subject_name_is_not_flagged():
    source = 'LOAD "a.png"\nALLOC width 4 4\nSET n width\nSAVE "o.png"\n'
    assert not [d for d in lint(source) if d.code == "PX013"]


def test_quoted_strings_and_attribute_access_are_not_flagged():
    source = 'LOAD "a.png"\nSET a "width"\nSET b _.width\nSET c $a\nSAVE "o.png"\n'
    assert not [d for d in lint(source) if d.code == "PX013"]
