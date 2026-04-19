# tests/test_executor.py
"""
Integration tests for the PixLang execution engine.
Uses synthetic numpy arrays — no disk I/O required.
"""
import numpy as np
import pytest
from pixlang.commands.builtin import _wrap, _unwrap
from pixlang.commands import registry
from pixlang.executor import Executor
from pixlang.parser.ast_nodes import Command, Pipeline


def _pipeline(*cmds) -> Pipeline:
    """Helper: build a Pipeline from (name, *args) tuples."""
    p = Pipeline()
    for i, (name, *args) in enumerate(cmds):
        p.commands.append(Command(name=name, args=list(args), line=i + 1))
    return p


def _bgr_image(h=100, w=100):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _gray_image(h=100, w=100):
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


def _run(pipeline: Pipeline):
    Executor(registry=registry).run(pipeline)


# ── Direct command tests ───────────────────────────────────────────────────────

def test_grayscale_returns_2d():
    img = _bgr_image()
    fn = registry.get("GRAYSCALE")
    result = fn(img)
    assert len(result.shape) == 2


def test_resize_shape():
    img = _bgr_image(200, 300)
    fn = registry.get("RESIZE")
    result = fn(img, 50, 80)
    assert result.shape[:2] == (80, 50)


def test_threshold_binary():
    img = _gray_image()
    fn = registry.get("THRESHOLD")
    result = fn(img, 128)
    assert set(np.unique(result)).issubset({0, 255})


def test_invert():
    img = np.zeros((10, 10), dtype=np.uint8)
    fn = registry.get("INVERT")
    result = fn(img)
    assert result.max() == 255


def test_blur_shape_preserved():
    img = _bgr_image()
    fn = registry.get("BLUR")
    result = fn(img, 5)
    assert result.shape == img.shape


def test_find_contours_returns_meta():
    img = np.zeros((50, 50), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.rectangle(img, (10, 10), (20, 20), 255, -1)
    fn = registry.get("FIND_CONTOURS")
    result = fn(img)
    _, meta = _unwrap(result)
    assert "contours" in meta
    assert len(meta["contours"]) >= 1


def test_draw_bounding_boxes_rgb_output():
    img = np.zeros((50, 50), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.rectangle(img, (10, 10), (20, 20), 255, -1)
    fn_fc = registry.get("FIND_CONTOURS")
    fn_db = registry.get("DRAW_BOUNDING_BOXES")
    after_contours = fn_fc(img)
    result = fn_db(after_contours)
    out_img, _ = _unwrap(result)
    # BGR output expected
    assert len(out_img.shape) == 3


def test_canny_2d_output():
    img = _bgr_image()
    fn = registry.get("CANNY")
    result = fn(img, 50, 150)
    assert len(result.shape) == 2


def test_no_image_raises():
    fn = registry.get("GRAYSCALE")
    with pytest.raises(RuntimeError):
        fn(None)
