# pixlang/commands/lowlevel.py
"""
v0.5 commands — hardware-near image manipulation.

Groups:
  Depth     CONVERT_DEPTH, CONVERT_SCALE
  Bitwise   BITWISE_AND/OR/XOR, BITWISE_NOT, SHIFT_LEFT, SHIFT_RIGHT, BIT_PLANE, MASK
  LUT       GAMMA, POSTERIZE, LUT, DITHER
  Channels  EXTRACT_CHANNEL, MERGE_CHANNELS, SWAP_RB, DROP_ALPHA, ADD_ALPHA
  Raw       SAVE_RAW, LOAD_RAW, EXPORT_C_ARRAY, BYTESWAP
  Analysis  STATS, CONNECTED_COMPONENTS
  Classic   CVT_COLOR, IN_RANGE, CLAHE, FILTER2D, SOBEL, LAPLACIAN, MORPH, PAD,
            ROTATE90, WARP_PERSPECTIVE, ABSDIFF

Buffer arguments (BITWISE_*, MASK, MERGE_CHANNELS, ABSDIFF) name an entry of
the shared buffer store ``ctx["checkpoints"]`` (CHECKPOINT snapshots or typed
buffers) holding a raw ndarray, a wrapped dict or a ``(img, meta)`` tuple.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .builtin import (
    CATEGORIES, _cast, _channels, _command, _cv, _dtype_max, _ensure_gray_raw,
    _get_buffer, _match_channels, _merge_alpha, _require_image, _split_alpha, _to_u8,
    _unwrap, _value_max, _wrap,
)

CATEGORIES.update({
    "CONVERT_DEPTH": "Depth", "CONVERT_SCALE": "Depth", "BYTESWAP": "Depth",
    "BITWISE_AND": "Bitwise", "BITWISE_OR": "Bitwise", "BITWISE_XOR": "Bitwise",
    "BITWISE_NOT": "Bitwise", "SHIFT_LEFT": "Bitwise", "SHIFT_RIGHT": "Bitwise",
    "BIT_PLANE": "Bitwise", "MASK": "Bitwise",
    "GAMMA": "LUT", "POSTERIZE": "LUT", "LUT": "LUT", "DITHER": "LUT",
    "EXTRACT_CHANNEL": "Channels", "MERGE_CHANNELS": "Channels", "SWAP_RB": "Channels",
    "DROP_ALPHA": "Channels", "ADD_ALPHA": "Channels",
    "SAVE_RAW": "Raw", "LOAD_RAW": "Raw", "EXPORT_C_ARRAY": "Raw",
    "STATS": "Analysis", "CONNECTED_COMPONENTS": "Analysis",
    "CVT_COLOR": "Color", "IN_RANGE": "Threshold", "CLAHE": "Color",
    "FILTER2D": "Filter", "SOBEL": "Filter", "LAPLACIAN": "Filter",
    "MORPH": "Morphology", "PAD": "Geometry", "ROTATE90": "Geometry",
    "WARP_PERSPECTIVE": "Geometry", "ABSDIFF": "Analysis",
})

_DEPTHS = {
    "u8": np.uint8, "uint8": np.uint8, "8u": np.uint8,
    "u16": np.uint16, "uint16": np.uint16, "16u": np.uint16,
    "f32": np.float32, "float32": np.float32, "32f": np.float32, "float": np.float32,
    "f64": np.float64, "float64": np.float64, "64f": np.float64, "double": np.float64,
}

RAW_FORMATS = ("gray8", "rgb565", "rgb888", "bgr888", "rgba8888")
_BPP = {"gray8": 1, "rgb565": 2, "rgb888": 3, "bgr888": 3, "rgba8888": 4}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _require_int_image(cmd: str, img: np.ndarray) -> None:
    if not (np.issubdtype(img.dtype, np.integer) or img.dtype == np.bool_):
        raise ValueError(
            f"[{cmd}] needs an integer image, got dtype {img.dtype}. "
            f"Use CONVERT_DEPTH u8 (or u16) first."
        )


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        try:
            return int(s, 0)        # accepts 0xFF / 0b1010 / 255
        except ValueError:
            return None
    return None


def _operand(cmd: str, img: np.ndarray, value: Any, ctx: Optional[dict]) -> Any:
    """An int constant or a same-sized buffer matched to *img*'s channels/dtype."""
    const = _int_or_none(value)
    if const is not None:
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            if not info.min <= const <= info.max:
                raise ValueError(f"[{cmd}] constant {const} does not fit dtype {img.dtype}")
        return img.dtype.type(const)
    other, _ = _get_buffer(ctx, value, cmd)
    if other.shape[:2] != img.shape[:2]:
        raise ValueError(
            f"[{cmd}] buffer '{value}' is {other.shape[1]}x{other.shape[0]}, "
            f"image is {img.shape[1]}x{img.shape[0]} — sizes must match."
        )
    other = _match_channels(other, _channels(img))
    if other.dtype != img.dtype:
        raise ValueError(
            f"[{cmd}] buffer '{value}' has dtype {other.dtype}, image has {img.dtype}. "
            f"Use CONVERT_DEPTH to match."
        )
    return other


def _gray_buffer(cmd: str, ctx, name, shape) -> np.ndarray:
    buf, _ = _get_buffer(ctx, name, cmd)
    buf = _ensure_gray_raw(buf)
    if buf.shape[:2] != tuple(shape[:2]):
        raise ValueError(
            f"[{cmd}] buffer '{name}' is {buf.shape[1]}x{buf.shape[0]}, "
            f"expected {shape[1]}x{shape[0]}."
        )
    return buf


def _apply_lut_color(img: np.ndarray, fn) -> np.ndarray:
    """Apply an elementwise transform to the colour channels only."""
    color, alpha = _split_alpha(img)
    return _merge_alpha(fn(color), alpha)


def _bits(dtype) -> int:
    return np.dtype(dtype).itemsize * 8


def _ensure_parent(path: str, cmd: str) -> Path:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    """uint8 RGB(A?) helper: returns (rgb uint8 HxWx3, alpha uint8 or None)."""
    u8 = _to_u8(img)
    color, alpha = _split_alpha(u8)
    color = _match_channels(color, 3)
    return color[:, :, ::-1], alpha


def _endian(cmd: str, endian: str) -> str:
    e = str(endian).lower()
    if e not in ("le", "be"):
        raise ValueError(f"[{cmd}] endianness must be 'le' or 'be', got {endian!r}")
    return "<" if e == "le" else ">"


def _raw_fmt(cmd: str, fmt: str) -> str:
    f = str(fmt).lower()
    if f not in RAW_FORMATS:
        raise ValueError(f"[{cmd}] unknown raw format {fmt!r}. Valid: {', '.join(RAW_FORMATS)}")
    return f


def _encode_raw(img: np.ndarray, fmt: str, order: str = "<") -> np.ndarray:
    """Encode to a flat array of uint8 (or uint16 for rgb565, with byte order)."""
    if fmt == "gray8":
        return _to_u8(_ensure_gray_raw(img)).ravel()
    rgb, alpha = _to_rgb_u8(img)
    if fmt == "rgb888":
        return rgb.ravel()
    if fmt == "bgr888":
        return rgb[:, :, ::-1].ravel()
    if fmt == "rgba8888":
        if alpha is None:
            alpha = np.full(rgb.shape[:2], 255, np.uint8)
        return np.dstack([rgb, alpha]).ravel()
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    packed = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    return packed.ravel().astype(np.dtype(np.uint16).newbyteorder(order))


def _decode_raw(data: bytes, w: int, h: int, fmt: str, order: str) -> np.ndarray:
    if fmt == "gray8":
        return np.frombuffer(data, np.uint8).reshape(h, w).copy()
    if fmt in ("rgb888", "bgr888"):
        px = np.frombuffer(data, np.uint8).reshape(h, w, 3)
        return (px[:, :, ::-1] if fmt == "rgb888" else px).copy()
    if fmt == "rgba8888":
        px = np.frombuffer(data, np.uint8).reshape(h, w, 4)
        return px[:, :, [2, 1, 0, 3]].copy()
    v = np.frombuffer(data, np.dtype(np.uint16).newbyteorder(order)).reshape(h, w)
    v = v.astype(np.uint16)
    r = ((v >> 11) & 0x1F).astype(np.uint16)
    g = ((v >> 5) & 0x3F).astype(np.uint16)
    b = (v & 0x1F).astype(np.uint16)
    # Expand to 8 bits by bit replication (0x1F → 0xFF)
    r8 = ((r << 3) | (r >> 2)).astype(np.uint8)
    g8 = ((g << 2) | (g >> 4)).astype(np.uint8)
    b8 = ((b << 3) | (b >> 2)).astype(np.uint8)
    return np.dstack([b8, g8, r8])


def floyd_steinberg(img: np.ndarray, levels: int = 2) -> np.ndarray:
    """Floyd–Steinberg error diffusion to *levels* evenly spaced values.

    Vectorised along anti-diagonals: pixel (y, x) depends only on pixels with
    a smaller ``t = x + 2y``, so all pixels with the same t are processed in
    one NumPy step (≈ W + 2H steps instead of W·H).
    """
    squeeze = img.ndim == 2
    src = img[:, :, None] if squeeze else img
    h, w, c = src.shape
    top = float(_value_max(src))
    step = top / (levels - 1)
    buf = np.zeros((h + 1, w + 2, c), np.float64)
    buf[:h, 1:w + 1] = src
    out = np.empty((h, w, c), np.float64)
    ys_all = np.arange(h)
    for t in range(w + 2 * (h - 1)):
        y0 = max(0, (t - w + 2) // 2)
        y1 = min(h - 1, t // 2)
        if y0 > y1:
            continue
        ys = ys_all[y0:y1 + 1]
        xs = t - 2 * ys
        old = buf[ys, xs + 1]
        new = np.clip(np.rint(old / step) * step, 0, top)
        out[ys, xs] = new
        err = old - new
        buf[ys, xs + 2] += err * (7 / 16)
        buf[ys + 1, xs] += err * (3 / 16)
        buf[ys + 1, xs + 1] += err * (5 / 16)
        buf[ys + 1, xs + 2] += err * (1 / 16)
    out = _cast(out, src.dtype)
    return out[:, :, 0] if squeeze else out


def register_v05(r):
    """Register the v0.5 hardware-near command set."""

    # ── Depth ─────────────────────────────────────────────────────────────────

    @_command(r, "CONVERT_DEPTH")
    def cmd_convert_depth(image, depth: str, scale: float = None):
        """CONVERT_DEPTH <u8|u16|f32|f64> [scale]  — change the pixel depth.

        Without scale the value convention is kept: u8→u16 ×257, u16→u8 ÷257,
        integer→float maps to [0,1], float [0,1]→integer maps to full scale.
        With scale, values are multiplied by scale and saturated (like
        OpenCV's convertTo).

        Examples:
            CONVERT_DEPTH f32
            CONVERT_DEPTH u16
            CONVERT_DEPTH u8 0.25
        """
        _require_image("CONVERT_DEPTH", image)
        img, _ = _unwrap(image)
        key = str(depth).lower()
        if key not in _DEPTHS:
            raise ValueError(f"[CONVERT_DEPTH] unknown depth {depth!r}. Valid: u8, u16, f32, f64")
        target = np.dtype(_DEPTHS[key])
        f = img.astype(np.float64)
        if scale is not None:
            return _cast(f * float(scale), target)
        if target == img.dtype:
            return img.copy()
        src_top = float(_value_max(img))
        dst_top = 1.0 if np.issubdtype(target, np.floating) else float(_dtype_max(target))
        return _cast(f * (dst_top / src_top), target)

    @_command(r, "CONVERT_SCALE")
    def cmd_convert_scale(image, alpha: float, beta: float = 0.0):
        """CONVERT_SCALE <alpha> [beta]  — out = saturate(in × alpha + beta), same depth.

        Examples:
            CONVERT_SCALE 1.5 0      # +50% contrast
            CONVERT_SCALE 1 -40      # darken by 40
        """
        _require_image("CONVERT_SCALE", image)
        img, _ = _unwrap(image)
        return _apply_lut_color(
            img, lambda c: _cast(c.astype(np.float64) * alpha + beta, c.dtype))

    @_command(r, "BYTESWAP")
    def cmd_byteswap(image):
        """BYTESWAP  — swap the byte order of every pixel value (no-op for 8-bit).

        Useful when raw 16-bit sensor data was read with the wrong endianness.
        """
        _require_image("BYTESWAP", image)
        img, _ = _unwrap(image)
        if img.dtype.itemsize == 1:
            return img.copy()
        return img.byteswap()

    # ── Bitwise ───────────────────────────────────────────────────────────────

    def _bitwise(cmd, op):
        def fn(image, operand, _ctx: dict = None):
            _require_image(cmd, image)
            img, _ = _unwrap(image)
            _require_int_image(cmd, img)
            return op(img, _operand(cmd, img, operand, _ctx)).astype(img.dtype)
        return fn

    for _name, _op, _word in (("BITWISE_AND", np.bitwise_and, "AND"),
                              ("BITWISE_OR", np.bitwise_or, "OR"),
                              ("BITWISE_XOR", np.bitwise_xor, "XOR")):
        _fn = _bitwise(_name, _op)
        _fn.__name__ = f"cmd_{_name.lower()}"
        _fn.__doc__ = (
            f"{_name} <buffer|const>  — per-pixel bitwise {_word} with a buffer or an integer.\n\n"
            f"        The constant may be decimal, hex (0xF0) or binary (0b1010). A buffer\n"
            f"        must have the same size and dtype (channels are matched).\n\n"
            f"        Examples:\n"
            f"            {_name} 0xF0\n"
            f"            {_name} mask\n"
        )
        _command(r, _name)(_fn)

    @_command(r, "BITWISE_NOT")
    def cmd_bitwise_not(image):
        """BITWISE_NOT  — flip every bit of every channel (including alpha); integer images only."""
        _require_image("BITWISE_NOT", image)
        img, _ = _unwrap(image)
        _require_int_image("BITWISE_NOT", img)
        return np.invert(img)

    @_command(r, "SHIFT_LEFT")
    def cmd_shift_left(image, n: int):
        """SHIFT_LEFT n  — shift pixel bits left by n (bits shifted out are lost)."""
        _require_image("SHIFT_LEFT", image)
        img, _ = _unwrap(image)
        _require_int_image("SHIFT_LEFT", img)
        if not 0 <= n < _bits(img.dtype):
            raise ValueError(f"[SHIFT_LEFT] n must be in 0..{_bits(img.dtype) - 1}, got {n}")
        return np.left_shift(img, img.dtype.type(n)).astype(img.dtype)

    @_command(r, "SHIFT_RIGHT")
    def cmd_shift_right(image, n: int):
        """SHIFT_RIGHT n  — shift pixel bits right by n (logical for unsigned)."""
        _require_image("SHIFT_RIGHT", image)
        img, _ = _unwrap(image)
        _require_int_image("SHIFT_RIGHT", img)
        if not 0 <= n < _bits(img.dtype):
            raise ValueError(f"[SHIFT_RIGHT] n must be in 0..{_bits(img.dtype) - 1}, got {n}")
        return np.right_shift(img, img.dtype.type(n)).astype(img.dtype)

    @_command(r, "BIT_PLANE")
    def cmd_bit_plane(image, n: int):
        """BIT_PLANE n  — extract bit n (0 = LSB) as a uint8 0/255 image (gray first)."""
        _require_image("BIT_PLANE", image)
        img, _ = _unwrap(image)
        _require_int_image("BIT_PLANE", img)
        gray = _ensure_gray_raw(img)
        if not 0 <= n < _bits(gray.dtype):
            raise ValueError(f"[BIT_PLANE] n must be in 0..{_bits(gray.dtype) - 1}, got {n}")
        return (((gray >> n) & 1) * 255).astype(np.uint8)

    @_command(r, "MASK")
    def cmd_mask(image, buffer: str, _ctx: dict = None):
        """MASK <buffer>  — keep pixels where the buffer is non-zero, zero elsewhere."""
        _require_image("MASK", image)
        img, _ = _unwrap(image)
        m = _gray_buffer("MASK", _ctx, buffer, img.shape) != 0
        out = img.copy()
        out[~m] = 0
        return out

    # ── LUT ───────────────────────────────────────────────────────────────────

    @_command(r, "GAMMA")
    def cmd_gamma(image, g: float):
        """GAMMA g  — gamma correction out = max·(in/max)^(1/g); g > 1 brightens.

        Uses a lookup table for 8/16-bit images; alpha is untouched.
        """
        _require_image("GAMMA", image)
        if g <= 0:
            raise ValueError(f"[GAMMA] g must be > 0, got {g}")
        img, _ = _unwrap(image)

        def apply(c):
            if c.dtype in (np.uint8, np.uint16):
                top = _dtype_max(c.dtype)
                lut = _cast(top * (np.arange(top + 1) / top) ** (1.0 / g), c.dtype)
                return lut[c]
            top = _value_max(c)
            return _cast(top * (np.clip(c, 0, None) / top) ** (1.0 / g), c.dtype)
        return _apply_lut_color(img, apply)

    @_command(r, "POSTERIZE")
    def cmd_posterize(image, bits: int):
        """POSTERIZE bits  — keep only the top *bits* bits of each channel.

        Integer images are masked (value & ~(2^(depth-bits)-1)), floats are
        quantised to 2^bits levels. Alpha is untouched.
        """
        _require_image("POSTERIZE", image)
        img, _ = _unwrap(image)

        def apply(c):
            if np.issubdtype(c.dtype, np.integer):
                depth = _bits(c.dtype)
                if not 1 <= bits <= depth:
                    raise ValueError(f"[POSTERIZE] bits must be in 1..{depth}, got {bits}")
                keep = (~((1 << (depth - bits)) - 1)) & int(np.iinfo(c.dtype).max)
                return np.bitwise_and(c, c.dtype.type(keep))
            if not 1 <= bits <= 16:
                raise ValueError(f"[POSTERIZE] bits must be in 1..16, got {bits}")
            top = _value_max(c)
            n = 2 ** bits
            return _cast(np.floor(np.clip(c / top, 0, 1) * (n - 1e-9)) * top / n, c.dtype)
        return _apply_lut_color(img, apply)

    @_command(r, "LUT")
    def cmd_lut(image, path: str):
        """LUT "table.npy"  — map pixel values through a NumPy lookup table.

        The table has 256 entries (uint8 images) or 65536 entries (uint16),
        optionally with a trailing channel axis (256×C) for per-channel tables.
        The output dtype is the table's dtype. Alpha is untouched.
        """
        _require_image("LUT", image)
        img, _ = _unwrap(image)
        try:
            table = np.load(str(path), allow_pickle=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"[LUT] table not found: {path!r}") from None
        except Exception as exc:
            raise ValueError(f"[LUT] cannot read {path!r}: {exc}") from None
        if img.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"[LUT] needs a uint8 or uint16 image, got {img.dtype}")
        need = _dtype_max(img.dtype) + 1
        table = table.reshape(table.shape[0], -1) if table.ndim > 1 else table
        if table.shape[0] != need:
            raise ValueError(f"[LUT] table has {table.shape[0]} entries, {img.dtype} needs {need}")

        def apply(c):
            if table.ndim == 1:
                return table[c]
            if table.shape[1] != _channels(c):
                raise ValueError(f"[LUT] table has {table.shape[1]} channels, image has {_channels(c)}")
            if c.ndim == 2:
                return table[c, 0]
            return np.stack([table[c[:, :, i], i] for i in range(c.shape[2])], axis=-1)
        color, alpha = _split_alpha(img)
        out = apply(color)
        return _merge_alpha(out, alpha) if alpha is not None else out

    @_command(r, "DITHER")
    def cmd_dither(image, levels: int = 2):
        """DITHER [levels]  — Floyd–Steinberg error diffusion (default 2 levels = 1-bit).

        Each colour channel is reduced to *levels* evenly spaced values over the
        image's full range (2 → 0/255 for uint8). Alpha is untouched.

        Examples:
            GRAYSCALE
            DITHER          # 1-bit black/white
            DITHER 4        # 2-bit per channel
        """
        _require_image("DITHER", image)
        if not 2 <= levels <= 65536:
            raise ValueError(f"[DITHER] levels must be in 2..65536, got {levels}")
        img, _ = _unwrap(image)
        return _apply_lut_color(img, lambda c: floyd_steinberg(c, levels))

    # ── Channels ──────────────────────────────────────────────────────────────

    @_command(r, "EXTRACT_CHANNEL")
    def cmd_extract_channel(image, n: int):
        """EXTRACT_CHANNEL n  — keep channel n (0=B, 1=G, 2=R, 3=A) as a gray image."""
        _require_image("EXTRACT_CHANNEL", image)
        img, _ = _unwrap(image)
        c = _channels(img)
        if not 0 <= n < c:
            raise ValueError(f"[EXTRACT_CHANNEL] channel {n} out of range (image has {c})")
        return img.copy() if img.ndim == 2 else img[:, :, n].copy()

    @_command(r, "MERGE_CHANNELS")
    def cmd_merge_channels(image, b: str, g: str, r: str, a: str = None, _ctx: dict = None):
        """MERGE_CHANNELS <b> <g> <r> [a]  — build a BGR(A) image from named gray buffers."""
        names = [b, g, r] + ([a] if a is not None else [])
        planes = []
        for name in names:
            buf, _ = _get_buffer(_ctx, name, "MERGE_CHANNELS")
            planes.append(_ensure_gray_raw(buf))
        shape, dtype = planes[0].shape, planes[0].dtype
        for name, p in zip(names, planes):
            if p.shape != shape or p.dtype != dtype:
                raise ValueError(
                    f"[MERGE_CHANNELS] buffer '{name}' is {p.shape[1]}x{p.shape[0]} {p.dtype}; "
                    f"expected {shape[1]}x{shape[0]} {dtype}"
                )
        return np.dstack(planes)

    @_command(r, "SWAP_RB")
    def cmd_swap_rb(image):
        """SWAP_RB  — swap the red and blue channels (BGR ↔ RGB); alpha kept."""
        _require_image("SWAP_RB", image)
        img, _ = _unwrap(image)
        if _channels(img) < 3:
            return img.copy()
        out = img.copy()
        out[:, :, [0, 2]] = img[:, :, [2, 0]]
        return out

    @_command(r, "DROP_ALPHA")
    def cmd_drop_alpha(image):
        """DROP_ALPHA  — remove the alpha channel (BGRA → BGR); no-op otherwise."""
        _require_image("DROP_ALPHA", image)
        img, _ = _unwrap(image)
        return img[:, :, :3].copy() if _channels(img) == 4 else img

    @_command(r, "ADD_ALPHA")
    def cmd_add_alpha(image, value: float = None):
        """ADD_ALPHA [value]  — add an alpha channel (default fully opaque); gray → BGRA."""
        _require_image("ADD_ALPHA", image)
        img, _ = _unwrap(image)
        color, _a = _split_alpha(img)
        color = _match_channels(color, 3)
        top = _value_max(img)
        v = top if value is None else value
        if not 0 <= v <= top:
            raise ValueError(f"[ADD_ALPHA] value must be in 0..{top}, got {value}")
        return np.dstack([color, np.full(color.shape[:2], v, dtype=color.dtype)])

    # ── Raw / embedded ────────────────────────────────────────────────────────

    @_command(r, "SAVE_RAW")
    def cmd_save_raw(image, path: str, fmt: str = "rgb888", endian: str = "le"):
        """SAVE_RAW "file.bin" [gray8|rgb565|rgb888|bgr888|rgba8888] [le|be]  — headerless dump.

        Pixels are written row-major with no header (data is converted to
        8-bit first). Endianness only affects rgb565.

        Examples:
            SAVE_RAW "frame.bin" rgb565 le
            SAVE_RAW "mask.bin" gray8
        """
        _require_image("SAVE_RAW", image)
        img, _ = _unwrap(image)
        f = _raw_fmt("SAVE_RAW", fmt)
        data = _encode_raw(img, f, _endian("SAVE_RAW", endian))
        _ensure_parent(path, "SAVE_RAW").write_bytes(data.tobytes())
        return image

    def _load_raw(image, path: str, w: int, h: int, fmt: str = "rgb888", endian: str = "le"):
        """LOAD_RAW "file.bin" <w> <h> [gray8|rgb565|rgb888|bgr888|rgba8888] [le|be]  — read a headerless dump.

        The inverse of SAVE_RAW; returns gray (gray8), BGR, or BGRA (rgba8888).
        The file size must equal w × h × bytes-per-pixel.
        """
        f = _raw_fmt("LOAD_RAW", fmt)
        if w <= 0 or h <= 0:
            raise ValueError(f"[LOAD_RAW] width and height must be > 0, got {w}x{h}")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"[LOAD_RAW] file not found: {path!r}")
        data = p.read_bytes()
        need = w * h * _BPP[f]
        if len(data) != need:
            raise ValueError(
                f"[LOAD_RAW] {path!r} has {len(data)} bytes, but {w}x{h} {f} needs {need}"
            )
        return _decode_raw(data, w, h, f, _endian("LOAD_RAW", endian))
    _command(r, "LOAD_RAW", fresh=True)(_load_raw)

    @_command(r, "EXPORT_C_ARRAY")
    def cmd_export_c_array(image, path: str, name: str, fmt: str = None):
        """EXPORT_C_ARRAY "file.h" <name> [fmt]  — write a C header for microcontrollers.

        Emits NAME_WIDTH / NAME_HEIGHT / NAME_FORMAT / NAME_BYTES_PER_PIXEL
        defines and ``const uint8_t name[]`` (``uint16_t`` for rgb565).
        Default format: gray8 for single-channel images, rgb565 otherwise.

        Examples:
            EXPORT_C_ARRAY "logo.h" logo rgb565
            EXPORT_C_ARRAY "icon.h" icon gray8
        """
        _require_image("EXPORT_C_ARRAY", image)
        img, _ = _unwrap(image)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)):
            raise ValueError(f"[EXPORT_C_ARRAY] {name!r} is not a valid C identifier")
        f = _raw_fmt("EXPORT_C_ARRAY", fmt or ("gray8" if _channels(img) == 1 else "rgb565"))
        data = _encode_raw(img, f, "=")
        h, w = img.shape[:2]
        up = str(name).upper()
        ctype, width = ("uint16_t", 6) if f == "rgb565" else ("uint8_t", 4)
        per_line = 12 if f == "rgb565" else 16
        lines = []
        for i in range(0, data.size, per_line):
            chunk = data[i:i + per_line]
            lines.append("    " + ", ".join(f"0x{int(v):0{width - 2}X}" for v in chunk) + ",")
        guard = f"{up}_H"
        text = "\n".join([
            f"/* Generated by PixLang EXPORT_C_ARRAY — {w}x{h} {f} */",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#include <stdint.h>",
            "",
            f"#define {up}_WIDTH  {w}",
            f"#define {up}_HEIGHT {h}",
            f'#define {up}_FORMAT "{f}"',
            f"#define {up}_BYTES_PER_PIXEL {_BPP[f]}",
            f"#define {up}_SIZE   {data.size}",
            "",
            f"const {ctype} {name}[{data.size}] = {{",
            *lines,
            "};",
            "",
            f"#endif /* {guard} */",
            "",
        ])
        _ensure_parent(path, "EXPORT_C_ARRAY").write_text(text)
        return image

    # ── Analysis ──────────────────────────────────────────────────────────────

    @_command(r, "STATS")
    def cmd_stats(image, prefix: str = "stats", _ctx: dict = None):
        """STATS [prefix]  — store image statistics into variables.

        Sets $<prefix>_mean, _std, _min, _max (over all colour channels),
        _width, _height and _channels. Passes the image through.

        Example:
            STATS img
            IF img_mean < 60
                GAMMA 1.8
            ENDIF
        """
        _require_image("STATS", image)
        if _ctx is None:
            raise RuntimeError("[STATS] Requires executor context.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(prefix)):
            raise ValueError(f"[STATS] invalid variable prefix {prefix!r}")
        img, _ = _unwrap(image)
        color, _a = _split_alpha(img)
        vals = color.astype(np.float64)
        v = _ctx.setdefault("vars", {})
        v[f"{prefix}_mean"] = float(vals.mean())
        v[f"{prefix}_std"] = float(vals.std())
        mn, mx = vals.min(), vals.max()
        is_int = np.issubdtype(img.dtype, np.integer)
        v[f"{prefix}_min"] = int(mn) if is_int else float(mn)
        v[f"{prefix}_max"] = int(mx) if is_int else float(mx)
        v[f"{prefix}_width"] = int(img.shape[1])
        v[f"{prefix}_height"] = int(img.shape[0])
        v[f"{prefix}_channels"] = _channels(img)
        return image

    @_command(r, "CONNECTED_COMPONENTS")
    def cmd_connected_components(image, connectivity: int = 8):
        """CONNECTED_COMPONENTS [4|8]  — label blobs of non-zero pixels.

        Passes the image through and stores metadata:
            components       list of {label, x, y, w, h, area, cx, cy}
            component_count  number of components (background excluded)
            labels           int32 label image
        """
        _require_image("CONNECTED_COMPONENTS", image)
        if connectivity not in (4, 8):
            raise ValueError(f"[CONNECTED_COMPONENTS] connectivity must be 4 or 8, got {connectivity}")
        img, meta = _unwrap(image)
        binary = (_ensure_gray_raw(img) != 0).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
        comps = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i][:5])
            comps.append({"label": i, "x": x, "y": y, "w": w, "h": h, "area": area,
                          "cx": float(cents[i][0]), "cy": float(cents[i][1])})
        out = _wrap(image)
        out.update({"components": comps, "component_count": len(comps), "labels": labels})
        return out

    @_command(r, "ABSDIFF")
    def cmd_absdiff(image, buffer: str, _ctx: dict = None):
        """ABSDIFF <buffer>  — per-pixel |image − buffer| (same size; channels/dtype matched)."""
        _require_image("ABSDIFF", image)
        img, _ = _unwrap(image)
        other, _m = _get_buffer(_ctx, buffer, "ABSDIFF")
        if other.shape[:2] != img.shape[:2]:
            raise ValueError(
                f"[ABSDIFF] buffer '{buffer}' is {other.shape[1]}x{other.shape[0]}, "
                f"image is {img.shape[1]}x{img.shape[0]}"
            )
        other = _match_channels(other, _channels(img))
        if other.dtype != img.dtype:
            other = _cast(other.astype(np.float64) * (_value_max(img) / _value_max(other)),
                          img.dtype)
        return _cv("ABSDIFF", img, cv2.absdiff, img, other)

    # ── Classic CV ────────────────────────────────────────────────────────────

    _TO_BGR = {"hsv": cv2.COLOR_HSV2BGR, "lab": cv2.COLOR_LAB2BGR,
               "ycrcb": cv2.COLOR_YCrCb2BGR, "rgb": cv2.COLOR_RGB2BGR}
    _FROM_BGR = {"hsv": cv2.COLOR_BGR2HSV, "lab": cv2.COLOR_BGR2LAB,
                 "ycrcb": cv2.COLOR_BGR2YCrCb, "rgb": cv2.COLOR_BGR2RGB}

    @_command(r, "CVT_COLOR")
    def cmd_cvt_color(image, space: str):
        """CVT_COLOR <hsv|lab|ycrcb|rgb|bgr|gray>  — convert colour space.

        The current colour space is tracked in metadata ("colorspace",
        default bgr), so ``CVT_COLOR bgr`` converts back from hsv/lab/...
        Alpha is dropped.
        """
        _require_image("CVT_COLOR", image)
        img, meta = _unwrap(image)
        dst = str(space).lower()
        if dst not in ("hsv", "lab", "ycrcb", "rgb", "bgr", "gray"):
            raise ValueError(f"[CVT_COLOR] unknown space {space!r}. "
                             f"Valid: hsv, lab, ycrcb, rgb, bgr, gray")
        src = str(meta.get("colorspace", "bgr")).lower()
        if img.dtype not in (np.uint8, np.uint16, np.float32):
            img = img.astype(np.float32)
        color, _a = _split_alpha(img)
        if _channels(color) == 1:
            bgr = _match_channels(color, 3)
        elif src in _TO_BGR:
            bgr = _cv("CVT_COLOR", color, cv2.cvtColor, color, _TO_BGR[src])
        else:
            bgr = color
        if dst == "bgr":
            out = bgr
        elif dst == "gray":
            out = _ensure_gray_raw(bgr)
        else:
            out = _cv("CVT_COLOR", bgr, cv2.cvtColor, bgr, _FROM_BGR[dst])
        meta["colorspace"] = dst
        return _wrap(out, meta)

    @_command(r, "IN_RANGE")
    def cmd_in_range(image, l1: float, l2: float, l3: float = None,
                     h1: float = None, h2: float = None, h3: float = None):
        """IN_RANGE l1 l2 l3 h1 h2 h3  — uint8 mask of pixels inside [low, high] per channel.

        For single-channel images use ``IN_RANGE low high``.

        Examples:
            CVT_COLOR hsv
            IN_RANGE 35 50 50 85 255 255    # green
            IN_RANGE 100 200                # gray band
        """
        _require_image("IN_RANGE", image)
        img, _ = _unwrap(image)
        color, _a = _split_alpha(img)
        given = [v for v in (l3, h1, h2, h3) if v is not None]
        if not given:
            gray = _ensure_gray_raw(color)
            return cv2.inRange(gray.astype(np.float32), float(l1), float(l2))
        if len(given) != 4:
            raise ValueError("[IN_RANGE] give 2 values (gray) or 6 values (3 channels)")
        c3 = _match_channels(color, 3).astype(np.float32)
        return cv2.inRange(c3, np.array([l1, l2, l3], np.float32), np.array([h1, h2, h3], np.float32))

    @_command(r, "CLAHE")
    def cmd_clahe(image, clip: float = 2.0, tile: int = 8):
        """CLAHE [clip] [tile]  — contrast-limited adaptive histogram equalisation.

        Gray images are equalised directly, colour images on the L channel of
        Lab. Supports uint8 and uint16 (gray).
        """
        _require_image("CLAHE", image)
        if clip <= 0 or tile < 1:
            raise ValueError(f"[CLAHE] clip must be > 0 and tile ≥ 1, got {clip}, {tile}")
        img, _ = _unwrap(image)
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
        color, alpha = _split_alpha(img)
        if _channels(color) == 1:
            if color.dtype not in (np.uint8, np.uint16):
                raise ValueError(f"[CLAHE] needs uint8/uint16, got {color.dtype}")
            return _merge_alpha(clahe.apply(color), alpha)
        if color.dtype != np.uint8:
            raise ValueError(f"[CLAHE] colour images must be uint8, got {color.dtype}")
        lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return _merge_alpha(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), alpha)

    @_command(r, "FILTER2D")
    def cmd_filter2d(image, kernel: str, normalize: bool = False):
        """FILTER2D "k11 k12 ... knn" [normalize]  — convolve with a square kernel.

        Values are separated by spaces, commas or semicolons; the count must be
        a perfect square. With normalize, the kernel is divided by its sum.

        Examples:
            FILTER2D "0 -1 0 -1 5 -1 0 -1 0"
            FILTER2D "1 2 1 2 4 2 1 2 1" true
        """
        _require_image("FILTER2D", image)
        img, _ = _unwrap(image)
        try:
            vals = [float(v) for v in re.split(r"[\s,;]+", str(kernel).strip()) if v]
        except ValueError:
            raise ValueError(f"[FILTER2D] kernel must contain only numbers: {kernel!r}") from None
        n = int(round(len(vals) ** 0.5))
        if not vals or n * n != len(vals):
            raise ValueError(f"[FILTER2D] kernel needs a square number of values, got {len(vals)}")
        k = np.array(vals, np.float32).reshape(n, n)
        if normalize:
            s = float(k.sum())
            if s == 0:
                raise ValueError("[FILTER2D] cannot normalise a kernel that sums to 0")
            k /= s
        return _apply_lut_color(img, lambda c: _cv("FILTER2D", c, cv2.filter2D, c, -1, k))

    def _derivative_out(cmd, gray, res):
        if np.issubdtype(gray.dtype, np.floating):
            return np.abs(res).astype(np.float32)
        return cv2.convertScaleAbs(res, alpha=255.0 / _dtype_max(gray.dtype))

    @_command(r, "SOBEL")
    def cmd_sobel(image, dx: int = 1, dy: int = 0, ksize: int = 3):
        """SOBEL [dx dy ksize]  — Sobel derivative of the gray image (|result|, uint8 for int input)."""
        _require_image("SOBEL", image)
        if dx < 0 or dy < 0 or dx + dy == 0:
            raise ValueError(f"[SOBEL] need dx, dy ≥ 0 and dx + dy > 0, got {dx}, {dy}")
        if ksize not in (1, 3, 5, 7):
            raise ValueError(f"[SOBEL] ksize must be 1, 3, 5 or 7, got {ksize}")
        img, _ = _unwrap(image)
        gray = _ensure_gray_raw(img)
        res = _cv("SOBEL", gray, cv2.Sobel, gray.astype(np.float32), cv2.CV_32F, dx, dy, ksize=ksize)
        return _derivative_out("SOBEL", gray, res)

    @_command(r, "LAPLACIAN")
    def cmd_laplacian(image, ksize: int = 1):
        """LAPLACIAN [ksize]  — Laplacian of the gray image (|result|, uint8 for int input)."""
        _require_image("LAPLACIAN", image)
        if ksize not in (1, 3, 5, 7):
            raise ValueError(f"[LAPLACIAN] ksize must be 1, 3, 5 or 7, got {ksize}")
        img, _ = _unwrap(image)
        gray = _ensure_gray_raw(img)
        res = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=ksize)
        return _derivative_out("LAPLACIAN", gray, res)

    _MORPH_OPS = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE,
                  "gradient": cv2.MORPH_GRADIENT, "tophat": cv2.MORPH_TOPHAT,
                  "blackhat": cv2.MORPH_BLACKHAT}
    _SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}

    @_command(r, "MORPH")
    def cmd_morph(image, op: str, k: int = 3, shape: str = "rect"):
        """MORPH <open|close|gradient|tophat|blackhat> [k] [rect|ellipse|cross]  — morphology."""
        _require_image("MORPH", image)
        o, s = str(op).lower(), str(shape).lower()
        if o not in _MORPH_OPS:
            raise ValueError(f"[MORPH] unknown op {op!r}. Valid: {', '.join(_MORPH_OPS)}")
        if s not in _SHAPES:
            raise ValueError(f"[MORPH] unknown shape {shape!r}. Valid: {', '.join(_SHAPES)}")
        if k < 1:
            raise ValueError(f"[MORPH] k must be ≥ 1, got {k}")
        img, _ = _unwrap(image)
        kernel = cv2.getStructuringElement(_SHAPES[s], (k, k))
        return _cv("MORPH", img, cv2.morphologyEx, img, _MORPH_OPS[o], kernel)

    _BORDERS = {"constant": cv2.BORDER_CONSTANT, "reflect": cv2.BORDER_REFLECT,
                "reflect101": cv2.BORDER_REFLECT_101, "replicate": cv2.BORDER_REPLICATE,
                "wrap": cv2.BORDER_WRAP}

    @_command(r, "PAD")
    def cmd_pad(image, top: int, bottom: int, left: int, right: int,
                mode: str = "constant", value: float = 0):
        """PAD top bottom left right [constant|reflect|reflect101|replicate|wrap] [value]  — add a border."""
        _require_image("PAD", image)
        if min(top, bottom, left, right) < 0:
            raise ValueError("[PAD] padding values must be ≥ 0")
        m = str(mode).lower()
        if m not in _BORDERS:
            raise ValueError(f"[PAD] unknown mode {mode!r}. Valid: {', '.join(_BORDERS)}")
        img, _ = _unwrap(image)
        return _cv("PAD", img, cv2.copyMakeBorder, img, top, bottom, left, right,
                   _BORDERS[m], value=[float(value)] * 4)

    @_command(r, "ROTATE90")
    def cmd_rotate90(image, k: int = 1):
        """ROTATE90 [k]  — rotate k × 90° counter-clockwise, losslessly (negative k = clockwise)."""
        _require_image("ROTATE90", image)
        img, _ = _unwrap(image)
        k %= 4
        if k == 0:
            return img.copy()
        code = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180,
                3: cv2.ROTATE_90_CLOCKWISE}[k]
        return cv2.rotate(img, code)

    @_command(r, "WARP_PERSPECTIVE")
    def cmd_warp_perspective(image, x1: float, y1: float, x2: float, y2: float,
                             x3: float, y3: float, x4: float, y4: float,
                             w: int, h: int):
        """WARP_PERSPECTIVE x1 y1 x2 y2 x3 y3 x4 y4 w h  — map a quad (TL, TR, BR, BL) to a w×h rectangle."""
        _require_image("WARP_PERSPECTIVE", image)
        if w <= 0 or h <= 0:
            raise ValueError(f"[WARP_PERSPECTIVE] w and h must be > 0, got {w}x{h}")
        img, _ = _unwrap(image)
        src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        if not np.all(np.isfinite(M)):
            raise ValueError("[WARP_PERSPECTIVE] the four points are degenerate")
        return _cv("WARP_PERSPECTIVE", img, cv2.warpPerspective, img, M, (w, h))
