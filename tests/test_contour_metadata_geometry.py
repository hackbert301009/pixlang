"""Contours found before a geometry change must not outlive the geometry they describe."""
import cv2
import numpy as np
import pytest

from pixlang.commands import registry
from pixlang.executor import Executor
from pixlang.parser import PixAssertionError, parse

GEOMETRY_COMMANDS = [
    "RESIZE 30 20",
    "CROP 0 0 20 20",
    "ROTATE 90",
    "PAD 2 2 2 2",
    "FLIP 1",
    "ROTATE90",
    "WARP_PERSPECTIVE 0 0 59 0 59 39 0 39 30 20",
]


@pytest.fixture
def image_path(tmp_path):
    canvas = np.zeros((40, 60), np.uint8)
    cv2.rectangle(canvas, (10, 10), (30, 30), 255, -1)
    path = tmp_path / "blob.png"
    cv2.imwrite(str(path), canvas)
    return path


def contour_count_after(image_path, step):
    ex = Executor(registry=registry)
    ex.run(parse(f'LOAD "{image_path}"\nFIND_CONTOURS\n{step}\nSET n _.contour_count\n'))
    return ex.context["vars"]["n"]


def test_contours_survive_a_non_geometric_command(image_path):
    assert contour_count_after(image_path, "BLUR 3") == 1


@pytest.mark.xfail(strict=True, reason="geometry commands keep the contours of the old image")
@pytest.mark.parametrize("step", GEOMETRY_COMMANDS)
def test_contours_are_dropped_after_geometry_change(image_path, step):
    assert contour_count_after(image_path, step) == 0
