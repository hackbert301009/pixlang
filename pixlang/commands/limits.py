"""Upper bound on the size of any one image or buffer a run may create.

The limit is carried in a context variable that ``Executor.run`` sets, so
commands that allocate (RESIZE, PAD, ALLOC, ...) can check it without the
executor passing it through every signature.
"""
from __future__ import annotations

import contextlib
import struct
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

MIB = 1024 * 1024


@dataclass(frozen=True)
class ArrayLimits:
    max_pixels: int = 64_000_000
    max_bytes: int = 512 * MIB


DEFAULT_LIMITS = ArrayLimits()

_current: ContextVar[ArrayLimits] = ContextVar("pixlang_array_limits", default=DEFAULT_LIMITS)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_NO_SIZE_MARKERS = {0xC4, 0xC8, 0xCC}          # DHT, JPG, DAC share the SOF range
_JPEG_HEADER_SCAN_BYTES = 64 * 1024


@contextlib.contextmanager
def enforce(limits: ArrayLimits) -> Iterator[None]:
    token = _current.set(limits)
    try:
        yield
    finally:
        _current.reset(token)


def check_array(cmd: str, height: int, width: int, channels: int = 1, itemsize: int = 1) -> None:
    """Raise ``ValueError`` when an array of this shape would exceed the limits."""
    limits = _current.get()
    pixels = int(height) * int(width)
    nbytes = pixels * int(channels) * int(itemsize)
    if pixels > limits.max_pixels or nbytes > limits.max_bytes:
        raise ValueError(
            f"[{cmd}] a {int(width)}x{int(height)} image ({pixels / 1e6:.1f} Mpx, "
            f"{nbytes / MIB:.0f} MiB) exceeds the limit of {limits.max_pixels / 1e6:g} Mpx / "
            f"{limits.max_bytes // MIB} MiB per image")


def check_image_file(cmd: str, path: str) -> None:
    """Reject an oversized image from its header, before the decoder allocates it."""
    size = read_image_size(path)
    if size is not None:
        check_array(cmd, size[1], size[0])


def read_image_size(path: str) -> Optional[Tuple[int, int]]:
    """``(width, height)`` from the file header of a PNG, JPEG, GIF or BMP, else ``None``."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head.startswith(_PNG_SIGNATURE) and head[12:16] == b"IHDR":
                return struct.unpack(">II", head[16:24])
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return struct.unpack("<HH", head[6:10])
            if head[:2] == b"BM":
                w, h = struct.unpack("<ii", head[18:26])
                return w, abs(h)
            if head[:2] == b"\xff\xd8":
                return _jpeg_size(head + f.read(_JPEG_HEADER_SCAN_BYTES))
    except (OSError, struct.error):
        pass
    return None


def _jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
        elif marker in (0x01, *range(0xD0, 0xDA)):
            i += 2
        elif 0xC0 <= marker <= 0xCF and marker not in _JPEG_NO_SIZE_MARKERS:
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        else:
            i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None
