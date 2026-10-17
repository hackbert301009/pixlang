# tests/test_cmd_builtin_robust.py
"""Built-in robustness: histogram, metadata, dtype/alpha handling, validation."""
import cv2
import numpy as np
import pytest

from pixlang.commands import registry
from pixlang.commands.builtin import _unwrap, _wrap


def run(name, image, *args, ctx=None):
    return registry.invoke(name, image, list(args), ctx if ctx is not None else
                           {"checkpoints": {}, "vars": {}})


def arr(x):
    return _unwrap(x)[0]


def bgr(h=40, w=50, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def bgra(h=40, w=50):
    img = np.dstack([bgr(h, w), np.full((h, w), 77, np.uint8)])
    return img


# ── Task 1: HISTOGRAM_SAVE ────────────────────────────────────────────────────

@pytest.mark.parametrize("img", [
    bgr(), bgr()[:, :, 0], bgra(),
    (bgr().astype(np.uint16) * 257),
    bgr().astype(np.float32) / 255.0,
])
def test_histogram_save_depths(tmp_path, img):
    out = tmp_path / "deep" / "dir" / "h.png"
    res = run("HISTOGRAM_SAVE", img, str(out))
    assert out.exists()
    assert arr(res).shape == img.shape


def test_histogram_save_bad_ext_raises(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        run("HISTOGRAM_SAVE", bgr(), str(tmp_path / "noext"))


# ── Task 2: metadata preservation ────────────────────────────────────────────

META_CMDS = [
    ("RESIZE", 20, 10), ("CROP", 0, 0, 10, 10), ("ROTATE", 30), ("FLIP", 1),
    ("GRAYSCALE",), ("INVERT",), ("NORMALIZE",), ("THRESHOLD", 100),
    ("THRESHOLD_OTSU",), ("ADAPTIVE_THRESHOLD", 11, 2), ("BLUR", 3),
    ("MEDIAN_BLUR", 3), ("SHARPEN",), ("CANNY",), ("DILATE",), ("ERODE",),
    ("EQUALIZE_HIST",),
]


@pytest.mark.parametrize("spec", META_CMDS, ids=[s[0] for s in META_CMDS])
def test_commands_preserve_meta(spec):
    name, *args = spec
    wrapped = _wrap(bgr(), {"contours": ["c"], "tag": 1})
    out = registry.get(name)(wrapped, *args)    # direct call, no invoke
    assert isinstance(out, dict), name
    assert out["contours"] == ["c"] and out["tag"] == 1


def test_wrap_does_not_mutate_input():
    src = {"_image": bgr(), "a": 1}
    out = _wrap(src)
    out["b"] = 2
    assert "b" not in src
    out2 = registry.get("FIND_CONTOURS")(src)
    assert "contours" in out2 and "contours" not in src


def test_load_drops_meta(tmp_path):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), bgr())
    out = run("LOAD", {"_image": bgr(), "contours": [1]}, str(p))
    assert isinstance(out, np.ndarray)


# ── Task 3: dtype + alpha ─────────────────────────────────────────────────────

U16 = (bgr().astype(np.uint16) * 257)
F32 = bgr().astype(np.float32) / 255.0


@pytest.mark.parametrize("img", [U16, F32], ids=["u16", "f32"])
@pytest.mark.parametrize("spec", [
    ("ADAPTIVE_THRESHOLD",), ("MEDIAN_BLUR", 3), ("CANNY",), ("EQUALIZE_HIST",),
    ("FIND_CONTOURS",), ("DRAW_TEXT", "hi"), ("THRESHOLD_OTSU",), ("NORMALIZE",),
    ("INVERT",), ("THRESHOLD", 0.5), ("BLUR", 3), ("DRAW_BOUNDING_BOXES",),
], ids=lambda s: s[0] if isinstance(s, tuple) else None)
def test_deep_images_supported(img, spec):
    name, *args = spec
    out = arr(run(name, img, *args))
    assert out.size > 0


def test_invert_respects_dtype_and_alpha():
    assert arr(run("INVERT", np.array([[0, 65535]], np.uint16))).tolist() == [[65535, 0]]
    f = arr(run("INVERT", np.array([[0.25]], np.float32)))
    assert f.dtype == np.float32 and abs(f[0, 0] - 0.75) < 1e-6
    out = arr(run("INVERT", bgra()))
    assert (out[:, :, 3] == 77).all()
    assert (out[:, :, :3] == 255 - bgra()[:, :, :3]).all()


def test_threshold_u16_and_modes():
    img = np.array([[100, 40000]], np.uint16)
    assert arr(run("THRESHOLD", img, 1000)).tolist() == [[0, 65535]]
    assert arr(run("THRESHOLD", img, 1000, "inv")).tolist() == [[65535, 0]]
    g = np.array([[10, 200]], np.uint8)
    assert arr(run("THRESHOLD", g, 100, "tozero")).tolist() == [[0, 200]]
    with pytest.raises(ValueError, match="mode"):
        run("THRESHOLD", g, 100, "weird")


def test_normalize_u16_full_range():
    out = arr(run("NORMALIZE", np.array([[10, 20]], np.uint16)))
    assert out.tolist() == [[0, 65535]]


def test_median_large_kernel_u16_errors_naming_dtype():
    with pytest.raises(ValueError, match="uint16"):
        run("MEDIAN_BLUR", U16, 7)


def test_save_float_and_u16(tmp_path):
    run("SAVE", F32, str(tmp_path / "f.png"))
    back = cv2.imread(str(tmp_path / "f.png"), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint8 and back.max() > 200
    run("SAVE", U16, str(tmp_path / "u.png"))
    assert cv2.imread(str(tmp_path / "u.png"), cv2.IMREAD_UNCHANGED).dtype == np.uint16
    run("SAVE", U16, str(tmp_path / "u.jpg"))
    assert (tmp_path / "u.jpg").exists()


def test_draw_on_bgra_uses_opaque_alpha():
    img = np.zeros((60, 200, 4), np.uint8)
    out = arr(run("DRAW_TEXT", img, "XX", 5, 40))
    assert out.shape[2] == 4 and out[:, :, 3].max() == 255


def test_overlay_blend_compare_mixed_alpha_and_depth():
    ctx = {"checkpoints": {}, "vars": {}}
    run("CHECKPOINT", bgr(), "c", ctx=ctx)
    out = arr(run("OVERLAY", bgra(), "c", 0.5, ctx=ctx))
    assert out.shape[2] == 4 and (out[:, :, 3] == 77).all()
    out = arr(run("BLEND", U16, "c", 0.5, "screen", ctx=ctx))
    assert out.dtype == np.uint16
    run("COMPARE", bgra(), "c", ctx=ctx)
    assert "diff" in ctx["checkpoints"]


def test_buffer_accepts_raw_and_wrapped():
    ctx = {"checkpoints": {"raw": bgr(), "wrapped": _wrap(bgr())}, "vars": {}}
    run("OVERLAY", bgr(), "raw", 0.5, ctx=ctx)
    run("BLEND", bgr(), "wrapped", 0.5, ctx=ctx)


# ── Task 4: validation ───────────────────────────────────────────────────────

def test_crop_clamps_and_errors():
    img = bgr(40, 50)
    assert arr(run("CROP", img, -5, -5, 20, 20)).shape[:2] == (15, 15)
    assert arr(run("CROP", img, 40, 30, 100, 100)).shape[:2] == (10, 10)
    with pytest.raises(ValueError, match="overlap"):
        run("CROP", img, 100, 100, 10, 10)
    with pytest.raises(ValueError, match="> 0"):
        run("CROP", img, 0, 0, -1, 10)


@pytest.mark.parametrize("spec", [
    ("RESIZE", 0, 10), ("BLUR", 0), ("MEDIAN_BLUR", -1), ("ADAPTIVE_THRESHOLD", 1),
    ("DILATE", 0), ("ERODE", 3, 0), ("FLIP", 2), ("AUTO_CROP", -1),
], ids=lambda s: f"{s[0]}{s[1:]}")
def test_invalid_args_raise(spec):
    name, *args = spec
    with pytest.raises(ValueError):
        run(name, bgr(), *args)


def test_even_kernels_round_up():
    run("BLUR", bgr(), 4)
    run("ADAPTIVE_THRESHOLD", bgr(), 4)


def test_save_needs_extension_and_creates_dirs(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        run("SAVE", bgr(), str(tmp_path / "out"))
    run("SAVE", bgr(), str(tmp_path / "a" / "b" / "out.png"))
    assert (tmp_path / "a" / "b" / "out.png").exists()


def test_rotate_lossless_and_expand():
    img = bgr(40, 50)
    r90 = arr(run("ROTATE", img, 90))
    assert r90.shape[:2] == (50, 40)
    np.testing.assert_array_equal(r90, np.rot90(img))
    assert arr(run("ROTATE", img, 30)).shape[:2] == (40, 50)
    ex = arr(run("ROTATE", img, 30, "expand"))
    assert ex.shape[0] > 40 and ex.shape[1] > 50
    np.testing.assert_array_equal(arr(run("ROTATE", img, 360)), img)


def test_coercion_through_invoke():
    assert arr(run("BLUR", bgr(), "5")).shape == (40, 50, 3)
    with pytest.raises(TypeError, match="ksize"):
        run("BLUR", bgr(), 2.5)
