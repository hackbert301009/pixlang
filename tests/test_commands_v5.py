# tests/test_commands_v5.py
"""v0.5 hardware-near commands on synthetic images."""
import time

import cv2
import numpy as np
import pytest

from pixlang.commands import registry
from pixlang.commands.builtin import _unwrap, _wrap
from pixlang.executor import Executor
from pixlang.parser import parse


def ctx_with(**bufs):
    return {"checkpoints": dict(bufs), "vars": {}}


def run(name, image, *args, ctx=None):
    return registry.invoke(name, image, list(args), ctx if ctx is not None else ctx_with())


def arr(x):
    return _unwrap(x)[0]


def bgr(h=24, w=32, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def grad(h=48, w=64):
    return np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))


# ── Depth ─────────────────────────────────────────────────────────────────────

def test_convert_depth_conventions():
    u8 = np.array([[0, 128, 255]], np.uint8)
    u16 = arr(run("CONVERT_DEPTH", u8, "u16"))
    assert u16.dtype == np.uint16 and u16.tolist() == [[0, 32896, 65535]]
    f = arr(run("CONVERT_DEPTH", u8, "f32"))
    assert f.dtype == np.float32 and f.max() == pytest.approx(1.0)
    back = arr(run("CONVERT_DEPTH", f, "u8"))
    np.testing.assert_array_equal(back, u8)
    assert arr(run("CONVERT_DEPTH", u16, "u8")).tolist() == u8.tolist()
    assert arr(run("CONVERT_DEPTH", u8, "u8", 2)).tolist() == [[0, 255, 255]]
    with pytest.raises(ValueError, match="depth"):
        run("CONVERT_DEPTH", u8, "u12")


def test_convert_scale_saturates():
    img = np.array([[10, 200]], np.uint8)
    assert arr(run("CONVERT_SCALE", img, 2, 10)).tolist() == [[30, 255]]


def test_byteswap():
    img = np.array([[0x0102]], np.uint16)
    assert arr(run("BYTESWAP", img))[0, 0] == 0x0201
    np.testing.assert_array_equal(arr(run("BYTESWAP", bgr())), bgr())


# ── Bitwise ───────────────────────────────────────────────────────────────────

def test_bitwise_constant_and_buffer():
    img = np.array([[0b10101010, 0xFF]], np.uint8)
    assert arr(run("BITWISE_AND", img, 0x0F)).tolist() == [[0b1010, 0x0F]]
    assert arr(run("BITWISE_OR", img, "0x01")).tolist() == [[0b10101011, 0xFF]]
    other = np.array([[0xFF, 0xFF]], np.uint8)
    ctx = ctx_with(m=_wrap(other))
    assert arr(run("BITWISE_XOR", img, "m", ctx=ctx)).tolist() == [[0b01010101, 0]]
    with pytest.raises(ValueError, match="fit"):
        run("BITWISE_AND", img, 300)
    with pytest.raises(ValueError, match="float32"):
        run("BITWISE_AND", img.astype(np.float32), 1)


def test_bitwise_buffer_size_mismatch():
    ctx = ctx_with(m=np.zeros((5, 5), np.uint8))
    with pytest.raises(ValueError, match="sizes must match"):
        run("BITWISE_AND", bgr(), "m", ctx=ctx)


def test_not_shift_bitplane():
    img = np.array([[0b00000011, 0b10000000]], np.uint8)
    assert arr(run("BITWISE_NOT", img)).tolist() == [[0b11111100, 0b01111111]]
    assert arr(run("SHIFT_LEFT", img, 1)).tolist() == [[6, 0]]
    assert arr(run("SHIFT_RIGHT", img, 7)).tolist() == [[0, 1]]
    assert arr(run("BIT_PLANE", img, 7)).tolist() == [[0, 255]]
    with pytest.raises(ValueError):
        run("SHIFT_LEFT", img, 8)


def test_mask():
    img = bgr()
    m = np.zeros(img.shape[:2], np.uint8)
    m[:, :10] = 255
    out = arr(run("MASK", img, "m", ctx=ctx_with(m=(m, {}))))
    assert (out[:, 10:] == 0).all() and (out[:, :10] == img[:, :10]).all()


# ── LUT ───────────────────────────────────────────────────────────────────────

def test_gamma_and_posterize():
    img = np.array([[0, 64, 255]], np.uint8)
    out = arr(run("GAMMA", img, 2.0))
    assert out[0, 0] == 0 and out[0, 2] == 255 and out[0, 1] > 64
    assert arr(run("POSTERIZE", img, 2)).tolist() == [[0, 64, 192]]
    u16 = arr(run("GAMMA", img.astype(np.uint16) * 257, 1.0))
    assert u16.dtype == np.uint16


def test_lut_file(tmp_path):
    table = (255 - np.arange(256)).astype(np.uint8)
    p = tmp_path / "inv.npy"
    np.save(p, table)
    img = bgr()
    np.testing.assert_array_equal(arr(run("LUT", img, str(p))), 255 - img)
    with pytest.raises(ValueError, match="entries"):
        np.save(tmp_path / "bad.npy", np.arange(10))
        run("LUT", img, str(tmp_path / "bad.npy"))


def test_dither_two_levels_preserves_mean_and_is_fast():
    img = np.full((480, 640), 64, np.uint8)
    t0 = time.perf_counter()
    out = arr(run("DITHER", img))
    elapsed = time.perf_counter() - t0
    assert set(np.unique(out)) <= {0, 255}
    assert abs(out.mean() - 64) < 2
    assert elapsed < 2.0
    four = arr(run("DITHER", bgr(), 4))
    assert set(np.unique(four)) <= {0, 85, 170, 255}


def test_dither_matches_reference_scan():
    img = grad(6, 9)
    out = arr(run("DITHER", img))
    # Reference sequential Floyd–Steinberg
    f = img.astype(np.float64)
    h, w = f.shape
    ref = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            old = f[y, x]
            new = np.clip(np.rint(old / 255) * 255, 0, 255)
            ref[y, x] = new
            e = old - new
            if x + 1 < w:
                f[y, x + 1] += e * 7 / 16
            if y + 1 < h:
                if x > 0:
                    f[y + 1, x - 1] += e * 3 / 16
                f[y + 1, x] += e * 5 / 16
                if x + 1 < w:
                    f[y + 1, x + 1] += e * 1 / 16
    np.testing.assert_array_equal(out, ref)


# ── Channels ──────────────────────────────────────────────────────────────────

def test_channel_ops():
    img = bgr()
    assert arr(run("EXTRACT_CHANNEL", img, 2)).tolist() == img[:, :, 2].tolist()
    swapped = arr(run("SWAP_RB", img))
    np.testing.assert_array_equal(swapped, img[:, :, ::-1])
    rgba = arr(run("ADD_ALPHA", img))
    assert rgba.shape[2] == 4 and (rgba[:, :, 3] == 255).all()
    assert arr(run("ADD_ALPHA", img[:, :, 0], 10)).shape[2] == 4
    assert arr(run("DROP_ALPHA", rgba)).shape[2] == 3
    ctx = ctx_with(b=img[:, :, 0], g=(img[:, :, 1], {}), r=_wrap(img[:, :, 2]))
    merged = arr(run("MERGE_CHANNELS", None, "b", "g", "r", ctx=ctx))
    np.testing.assert_array_equal(merged, img)
    with pytest.raises(ValueError, match="out of range"):
        run("EXTRACT_CHANNEL", img, 3)


# ── Raw / embedded ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt", ["gray8", "rgb888", "bgr888", "rgba8888", "rgb565"])
@pytest.mark.parametrize("endian", ["le", "be"])
def test_raw_roundtrip(tmp_path, fmt, endian):
    img = bgr(8, 10)
    p = tmp_path / "sub" / "f.bin"
    run("SAVE_RAW", img, str(p), fmt, endian)
    bpp = {"gray8": 1, "rgb888": 3, "bgr888": 3, "rgba8888": 4, "rgb565": 2}[fmt]
    assert p.stat().st_size == 8 * 10 * bpp
    back = arr(run("LOAD_RAW", None, str(p), 10, 8, fmt, endian))
    if fmt == "gray8":
        np.testing.assert_array_equal(back, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    elif fmt == "rgb565":
        assert np.abs(back.astype(int) - img.astype(int)).max() <= 7
    else:
        np.testing.assert_array_equal(back[:, :, :3], img)


def test_rgb565_byte_layout(tmp_path):
    img = np.zeros((1, 1, 3), np.uint8)
    img[0, 0] = (0, 0, 255)   # pure red in BGR → 0xF800
    run("SAVE_RAW", img, str(tmp_path / "le.bin"), "rgb565", "le")
    run("SAVE_RAW", img, str(tmp_path / "be.bin"), "rgb565", "be")
    assert (tmp_path / "le.bin").read_bytes() == b"\x00\xf8"
    assert (tmp_path / "be.bin").read_bytes() == b"\xf8\x00"


def test_load_raw_size_mismatch(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="needs"):
        run("LOAD_RAW", None, str(p), 4, 4, "gray8")


def test_export_c_array(tmp_path):
    img = bgr(3, 4)
    p = tmp_path / "logo.h"
    run("EXPORT_C_ARRAY", img, str(p), "logo")
    text = p.read_text()
    assert "#define LOGO_WIDTH  4" in text and "#define LOGO_HEIGHT 3" in text
    assert '#define LOGO_FORMAT "rgb565"' in text
    assert "const uint16_t logo[12]" in text
    run("EXPORT_C_ARRAY", img[:, :, 0], str(tmp_path / "g.h"), "g")
    assert "const uint8_t g[12]" in (tmp_path / "g.h").read_text()
    with pytest.raises(ValueError, match="identifier"):
        run("EXPORT_C_ARRAY", img, str(p), "1bad")


# ── Stats & analysis ─────────────────────────────────────────────────────────

def test_stats_into_vars():
    ctx = ctx_with()
    img = np.array([[0, 100], [200, 100]], np.uint8)
    out = run("STATS", img, "s", ctx=ctx)
    assert out is img
    v = ctx["vars"]
    assert v["s_mean"] == 100 and v["s_min"] == 0 and v["s_max"] == 200
    assert v["s_width"] == 2 and v["s_height"] == 2 and v["s_channels"] == 1


def test_stats_usable_in_pipeline(tmp_path):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.full((10, 20, 3), 50, np.uint8))
    ex = Executor(registry=registry)
    ex.run(parse(f'LOAD "{p}"\nSTATS img\n'))
    assert ex.context["vars"]["img_width"] == 20


def test_connected_components_meta():
    img = np.zeros((20, 20), np.uint8)
    img[2:5, 2:5] = 255
    img[10:15, 10:12] = 255
    out = run("CONNECTED_COMPONENTS", img)
    assert out["component_count"] == 2
    assert {c["area"] for c in out["components"]} == {9, 10}


def test_absdiff():
    a = np.array([[10, 200]], np.uint8)
    b = np.array([[30, 100]], np.uint8)
    assert arr(run("ABSDIFF", a, "b", ctx=ctx_with(b=b))).tolist() == [[20, 100]]


# ── Classic CV ────────────────────────────────────────────────────────────────

def test_cvt_color_roundtrip_and_in_range():
    img = np.zeros((4, 4, 3), np.uint8)
    img[:, :2] = (0, 255, 0)  # green
    hsv = run("CVT_COLOR", img, "hsv")
    assert hsv["colorspace"] == "hsv"
    mask = arr(run("IN_RANGE", hsv, 35, 50, 50, 85, 255, 255))
    assert mask[:, :2].min() == 255 and mask[:, 2:].max() == 0
    back = arr(run("CVT_COLOR", hsv, "bgr"))
    np.testing.assert_array_equal(back, img)
    assert arr(run("IN_RANGE", grad(), 100, 200)).dtype == np.uint8


def test_clahe_filter_sobel_laplacian():
    assert arr(run("CLAHE", bgr())).shape == (24, 32, 3)
    assert arr(run("CLAHE", grad(), 3, 4)).shape == (48, 64)
    box = arr(run("FILTER2D", grad(), "1 1 1 1 1 1 1 1 1", "true"))
    assert box.dtype == np.uint8
    with pytest.raises(ValueError, match="square"):
        run("FILTER2D", grad(), "1 2 3")
    sob = arr(run("SOBEL", grad()))
    assert sob.dtype == np.uint8 and sob[:, 5:-5].mean() > 0
    assert arr(run("LAPLACIAN", grad(), 3)).dtype == np.uint8


def test_morph_pad_rotate90_warp():
    img = grad()
    for op in ("open", "close", "gradient", "tophat", "blackhat"):
        assert arr(run("MORPH", img, op, 3, "ellipse")).shape == img.shape
    padded = arr(run("PAD", img, 1, 2, 3, 4, "constant", 9))
    assert padded.shape == (51, 71) and padded[0, 0] == 9
    np.testing.assert_array_equal(arr(run("ROTATE90", img, -1)), np.rot90(img, -1))
    warped = arr(run("WARP_PERSPECTIVE", img, 0, 0, 63, 0, 63, 47, 0, 47, 32, 24))
    assert warped.shape == (24, 32)


def test_new_commands_have_categories_and_usage():
    for name in ("CONVERT_DEPTH", "BITWISE_AND", "DITHER", "SAVE_RAW", "STATS", "MORPH"):
        info = registry.get_info(name)
        assert info.category and info.usage.startswith(name)
