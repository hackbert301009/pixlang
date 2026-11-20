"""One image or buffer may not grow past the configured size."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from pixlang.commands import registry
from pixlang.commands.limits import ArrayLimits, check_array, enforce, read_image_size
from pixlang.executor.engine import Executor
from pixlang.parser import parse


def run_code(code: str, **kwargs):
    ex = Executor(registry, **kwargs)
    return ex.run(parse(code))


def png_header_only(path: Path, width: int, height: int) -> Path:
    """A PNG whose header claims a huge image; the pixel data is not there."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(
            ">I", zlib.crc32(kind + body))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", zlib.compress(b"\0" * 64)) + chunk(b"IEND", b""))
    return path


@pytest.mark.parametrize("code", [
    "ALLOC b 30000 30000 3 u8 7",
    "USE ZEROS(30000, 30000, 3)",
    "USE ZEROS(10, 10)\nRESIZE 30000 30000",
    "USE ZEROS(10, 10)\nRESIZE_PERCENT 900000",
    "USE ZEROS(10, 10)\nPAD 20000 20000 20000 20000",
    "USE ZEROS(10, 10)\nWARP_PERSPECTIVE 0 0 9 0 9 9 0 9 30000 30000",
])
def test_oversized_allocation_is_refused(code):
    with pytest.raises(RuntimeError, match="exceeds the limit"):
        run_code(code)


def test_limit_is_configurable():
    run_code("ALLOC b 100 100 1 u8 0")
    with pytest.raises(RuntimeError, match="exceeds the limit"):
        run_code("ALLOC b 100 100 1 u8 0", array_limits=ArrayLimits(max_pixels=1000))
    with pytest.raises(RuntimeError, match="exceeds the limit"):
        run_code("ALLOC b 100 100 1 f64 0", array_limits=ArrayLimits(max_bytes=50_000))


def test_conversion_to_a_wider_depth_counts_the_result():
    with pytest.raises(RuntimeError, match="exceeds the limit"):
        run_code("USE ZEROS(100, 100, 3)\nCONVERT_DEPTH f64",
                 array_limits=ArrayLimits(max_bytes=100_000))


def test_message_names_size_and_limit():
    with pytest.raises(ValueError) as err:
        with enforce(ArrayLimits(max_pixels=10, max_bytes=10)):
            check_array("RESIZE", 100, 200, 3, 1)
    assert "200x100" in str(err.value) and "per image" in str(err.value)


def test_limits_do_not_leak_out_of_a_run():
    with pytest.raises(ValueError):
        with enforce(ArrayLimits(max_pixels=1)):
            check_array("X", 2, 2)
    check_array("X", 2, 2)


def test_png_bomb_is_rejected_from_its_header(tmp_path):
    bomb = png_header_only(tmp_path / "bomb.png", 30000, 30000)
    assert read_image_size(str(bomb)) == (30000, 30000)
    with pytest.raises(RuntimeError, match="exceeds the limit"):
        run_code(f'LOAD "{bomb}"')


def test_headers_of_common_formats(tmp_path):
    img = np.zeros((7, 11, 3), np.uint8)
    for ext in (".png", ".jpg", ".bmp"):
        p = tmp_path / f"a{ext}"
        cv2.imwrite(str(p), img)
        assert read_image_size(str(p)) == (11, 7), ext
    assert read_image_size(str(tmp_path / "missing.png")) is None


def test_load_glob_checks_the_header_too(tmp_path):
    png_header_only(tmp_path / "bomb.png", 30000, 30000)
    with pytest.raises(RuntimeError, match="exceeds the limit"):
        run_code(f'LOAD_GLOB "{tmp_path}/*.png"')


def test_normal_pipeline_is_unaffected(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((50, 60, 3), np.uint8))
    out = run_code(f'LOAD "{tmp_path / "a.png"}"\nRESIZE 120 100')
    assert out.shape[:2] == (100, 120)
