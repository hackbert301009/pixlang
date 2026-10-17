# pixlang/commands/builtin.py
"""
Built-in PixLang commands backed by OpenCV + NumPy.

Each function signature:
    fn(image: np.ndarray | dict | None, *args[, _ctx]) -> np.ndarray | dict

Conventions shared by all built-ins (v0.5):

* **Metadata** — an image may travel as a "wrapped" dict ``{"_image": arr, ...meta}``
  (e.g. ``contours`` from FIND_CONTOURS). Every built-in that returns a bare
  array is decorated so the input's metadata is re-attached automatically;
  only loaders (LOAD, LOAD_GLOB, LOAD_RAW) start with fresh metadata.
* **Depth** — uint8, uint16 and float images are accepted. Floats are
  treated as [0, 1] when their max is <= 1, otherwise as [0, 255]. Commands
  that OpenCV only implements for 8-bit data convert to uint8 internally
  and say so in their docstring. Unsupported combinations raise a
  ``ValueError`` naming the dtype.
* **Alpha** — BGRA images keep their alpha channel: colour operations
  (INVERT, NORMALIZE, OVERLAY, BLEND, GAMMA, ...) only touch B, G, R and
  drawing commands draw with alpha = opaque.
* **Kernel sizes** — BLUR / MEDIAN_BLUR / ADAPTIVE_THRESHOLD need odd
  kernels. Even sizes are rounded *up* to the next odd number (4 -> 5);
  sizes below the minimum raise an error.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .registry import CommandRegistry, _carry_meta

# Command name → category, used by `pixlang commands` and the editor.
CATEGORIES = {
    "LOAD": "I/O", "SAVE": "I/O", "HISTOGRAM_SAVE": "I/O",
    "RESIZE": "Geometry", "RESIZE_PERCENT": "Geometry", "CROP": "Geometry",
    "ROTATE": "Geometry", "FLIP": "Geometry", "AUTO_CROP": "Geometry",
    "GRAYSCALE": "Color", "INVERT": "Color", "NORMALIZE": "Color",
    "HEATMAP": "Color", "EQUALIZE_HIST": "Color",
    "THRESHOLD": "Threshold", "THRESHOLD_OTSU": "Threshold",
    "ADAPTIVE_THRESHOLD": "Threshold",
    "BLUR": "Filter", "MEDIAN_BLUR": "Filter", "SHARPEN": "Filter", "CANNY": "Filter",
    "DILATE": "Morphology", "ERODE": "Morphology",
    "FIND_CONTOURS": "Analysis", "PRINT_INFO": "Analysis", "PIPELINE_STATS": "Analysis",
    "COMPARE": "Analysis",
    "DRAW_BOUNDING_BOXES": "Annotation", "DRAW_CONTOURS": "Annotation",
    "DRAW_TEXT": "Annotation",
    "CHECKPOINT": "Composition", "OVERLAY": "Composition", "BLEND": "Composition",
}


def _command(r: CommandRegistry, name: str, *, fresh: bool = False):
    """Register a built-in, preserving input metadata on bare-array results.

    ``fresh=True`` marks loaders whose result must not inherit metadata.
    """
    def deco(fn):
        if fresh:
            fn.fresh_image = True
            target = fn
        else:
            @functools.wraps(fn)
            def target(image, *args, **kwargs):
                return _carry_meta(fn, image, fn(image, *args, **kwargs))
        r.register(name, category=CATEGORIES.get(name, ""))(target)
        return target
    return deco


def register_all(r: CommandRegistry):
    """Wire all built-ins into the given registry."""

    # ── I/O ──────────────────────────────────────────────────────────────────

    @_command(r, "LOAD", fresh=True)
    def cmd_load(image, path: str):
        """LOAD "path/to/image.png"  — loads image from disk (any depth, keeps alpha)."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            hint = "" if Path(path).exists() else " (file does not exist)"
            raise FileNotFoundError(f"[LOAD] Could not read image: {path!r}{hint}")
        return img

    @_command(r, "SAVE")
    def cmd_save(image, path: str):
        """SAVE "output.png"  — writes current image to disk (creates parent dirs).

        Depth is adapted to the format: float images are written as float
        only to .tif/.tiff/.exr/.hdr/.pfm, otherwise scaled to 8-bit;
        16-bit is kept for PNG/TIFF/PNM/JP2 and reduced to 8-bit for JPEG etc.
        """
        _require_image("SAVE", image)
        img, _ = _unwrap(image)
        _write_image("SAVE", str(path), img)
        return image  # pass through unchanged

    # ── Geometry ─────────────────────────────────────────────────────────────

    @_command(r, "RESIZE")
    def cmd_resize(image, width: int, height: int):
        """RESIZE 640 480  — resize to width × height pixels (both > 0)."""
        _require_image("RESIZE", image)
        img, _ = _unwrap(image)
        if width <= 0 or height <= 0:
            raise ValueError(f"[RESIZE] width and height must be > 0, got {width}x{height}")
        return _cv("RESIZE", img, cv2.resize, img, (width, height),
                   interpolation=cv2.INTER_AREA)

    @_command(r, "CROP")
    def cmd_crop(image, x: int, y: int, w: int, h: int):
        """CROP x y width height  — crops a rectangular region (clamped to the image).

        Negative x/y start outside the image and are clamped to 0 (the region
        shrinks accordingly). width/height must be > 0. An empty region raises.
        """
        _require_image("CROP", image)
        img, _ = _unwrap(image)
        if w <= 0 or h <= 0:
            raise ValueError(f"[CROP] width and height must be > 0, got {w}x{h}")
        ih, iw = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"[CROP] Region x={x} y={y} w={w} h={h} does not overlap the "
                f"{iw}x{ih} image."
            )
        return img[y1:y2, x1:x2].copy()

    @_command(r, "ROTATE")
    def cmd_rotate(image, angle: float, mode: str = "crop"):
        """ROTATE angle [expand]  — rotate counter-clockwise by angle degrees.

        Multiples of 90° use a lossless transpose (width/height swap for
        90/270). Other angles rotate about the exact image centre; by default
        the canvas size is kept (corners are cut off), with ``expand`` the
        canvas grows to fit the whole rotated image.

        Examples:
            ROTATE 90
            ROTATE 30 expand
        """
        _require_image("ROTATE", image)
        img, _ = _unwrap(image)
        mode = str(mode).lower()
        if mode not in ("crop", "expand"):
            raise ValueError(f"[ROTATE] mode must be 'crop' or 'expand', got {mode!r}")
        a = float(angle) % 360.0
        if a == 0:
            return img.copy()
        if a % 90 == 0:
            code = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_CLOCKWISE}[int(a)]
            return cv2.rotate(img, code)
        h, w = img.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        out_w, out_h = w, h
        if mode == "expand":
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            out_w = int(math.ceil(h * sin + w * cos))
            out_h = int(math.ceil(h * cos + w * sin))
            M[0, 2] += (out_w - 1) / 2.0 - center[0]
            M[1, 2] += (out_h - 1) / 2.0 - center[1]
        return _cv("ROTATE", img, cv2.warpAffine, img, M, (out_w, out_h))

    @_command(r, "FLIP")
    def cmd_flip(image, axis: int):
        """FLIP 0=vertical  1=horizontal  -1=both"""
        _require_image("FLIP", image)
        img, _ = _unwrap(image)
        if axis not in (-1, 0, 1):
            raise ValueError(
                f"[FLIP] axis must be 0 (vertical), 1 (horizontal) or -1 (both), got {axis}"
            )
        return cv2.flip(img, axis)

    # ── Color ─────────────────────────────────────────────────────────────────

    @_command(r, "GRAYSCALE")
    def cmd_grayscale(image):
        """GRAYSCALE  — converts to single-channel grayscale."""
        _require_image("GRAYSCALE", image)
        img, _ = _unwrap(image)
        return _ensure_gray_raw(img)

    @_command(r, "INVERT")
    def cmd_invert(image):
        """INVERT  — invert colour values (max - x); alpha is left untouched.

        uint8 → 255-x, uint16 → 65535-x, float in [0,1] → 1-x,
        float in [0,255] → 255-x.
        """
        _require_image("INVERT", image)
        img, _ = _unwrap(image)
        color, alpha = _split_alpha(img)
        if color.dtype == np.bool_:
            inv = ~color
        elif np.issubdtype(color.dtype, np.unsignedinteger):
            inv = np.iinfo(color.dtype).max - color
        elif np.issubdtype(color.dtype, np.floating):
            if color.size and color.min() < 0:
                raise ValueError("[INVERT] float images must be non-negative to invert")
            inv = (_float_max(color) - color).astype(color.dtype)
        else:
            raise ValueError(f"[INVERT] Unsupported dtype {color.dtype}; "
                             f"use CONVERT_DEPTH u8 first.")
        return _merge_alpha(inv, alpha)

    @_command(r, "NORMALIZE")
    def cmd_normalize(image):
        """NORMALIZE  — stretch values to the full range (uint8 0-255, uint16 0-65535, float 0-1)."""
        _require_image("NORMALIZE", image)
        img, _ = _unwrap(image)
        color, alpha = _split_alpha(img)
        if color.dtype == np.uint8:
            out = cv2.normalize(color, None, 0, 255, cv2.NORM_MINMAX)
        else:
            if not (np.issubdtype(color.dtype, np.integer)
                    or np.issubdtype(color.dtype, np.floating)):
                raise ValueError(f"[NORMALIZE] Unsupported dtype {color.dtype}")
            f = color.astype(np.float64)
            lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 0.0)
            top = 1.0 if np.issubdtype(color.dtype, np.floating) else _dtype_max(color.dtype)
            if hi > lo:
                f = (f - lo) * (top / (hi - lo))
            else:
                f = np.zeros_like(f)
            out = _cast(f, color.dtype)
        return _merge_alpha(out, alpha)

    # ── Thresholding ──────────────────────────────────────────────────────────

    @_command(r, "THRESHOLD")
    def cmd_threshold(image, value: float, mode: str = "binary"):
        """THRESHOLD value [mode]  — threshold at value (mode: binary|inv|trunc|tozero|tozero_inv).

        The value is in the image's own units (0-255 for uint8, 0-65535 for
        uint16, 0-1 for normalised floats). Output keeps the input depth;
        "on" pixels get the depth's maximum.

        Examples:
            THRESHOLD 120
            THRESHOLD 120 inv
        """
        _require_image("THRESHOLD", image)
        gray = _ensure_gray(image)
        m = str(mode).lower()
        if m not in _THRESH_MODES:
            raise ValueError(f"[THRESHOLD] Unknown mode '{mode}'. "
                             f"Valid: {', '.join(sorted(set(_THRESH_MODES) - {'binary_inv'}))}")
        maxval = _value_max(gray)
        if gray.dtype in (np.uint8, np.float32):
            _, result = cv2.threshold(gray, float(value), maxval, _THRESH_MODES[m])
            return result
        v = gray.astype(np.float64)
        t = float(value)
        if m == "binary":
            out = np.where(v > t, maxval, 0)
        elif m in ("inv", "binary_inv"):
            out = np.where(v > t, 0, maxval)
        elif m == "trunc":
            out = np.minimum(v, t)
        elif m == "tozero":
            out = np.where(v > t, v, 0)
        else:
            out = np.where(v > t, 0, v)
        return _cast(out, gray.dtype)

    @_command(r, "THRESHOLD_OTSU")
    def cmd_threshold_otsu(image):
        """THRESHOLD_OTSU  — automatic Otsu binarisation (output uint8)."""
        _require_image("THRESHOLD_OTSU", image)
        gray = _to_u8(_ensure_gray(image))
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    @_command(r, "ADAPTIVE_THRESHOLD")
    def cmd_adaptive_threshold(image, block_size: int = 11, C: int = 2):
        """ADAPTIVE_THRESHOLD 11 2  — local adaptive binarisation (block odd ≥ 3, output uint8)."""
        _require_image("ADAPTIVE_THRESHOLD", image)
        k = _odd_kernel("ADAPTIVE_THRESHOLD", block_size, minimum=3)
        gray = _to_u8(_ensure_gray(image))
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, k, C
        )

    @_command(r, "BLUR")
    def cmd_blur(image, ksize: int = 5):
        """BLUR 5  — Gaussian blur with kernel size ksize×ksize (even sizes round up)."""
        _require_image("BLUR", image)
        img, _ = _unwrap(image)
        k = _odd_kernel("BLUR", ksize)
        return _cv("BLUR", img, cv2.GaussianBlur, img, (k, k), 0)

    @_command(r, "MEDIAN_BLUR")
    def cmd_median_blur(image, ksize: int = 5):
        """MEDIAN_BLUR 5  — salt-and-pepper noise removal (even sizes round up).

        uint8 supports any size; uint16/float support only 3 and 5.
        """
        _require_image("MEDIAN_BLUR", image)
        img, _ = _unwrap(image)
        k = _odd_kernel("MEDIAN_BLUR", ksize)
        if img.dtype == np.float64:
            img = img.astype(np.float32)
        if img.dtype != np.uint8:
            if img.dtype not in (np.uint16, np.float32):
                raise ValueError(f"[MEDIAN_BLUR] Unsupported dtype {img.dtype}; "
                                 f"use CONVERT_DEPTH u8 first.")
            if k > 5:
                raise ValueError(
                    f"[MEDIAN_BLUR] ksize {k} needs a uint8 image (got {img.dtype}, "
                    f"which supports only 3 or 5). Use CONVERT_DEPTH u8 first."
                )
        return _cv("MEDIAN_BLUR", img, cv2.medianBlur, img, k)

    @_command(r, "SHARPEN")
    def cmd_sharpen(image):
        """SHARPEN  — unsharp mask sharpening."""
        _require_image("SHARPEN", image)
        img, _ = _unwrap(image)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return _cv("SHARPEN", img, cv2.filter2D, img, -1, kernel)

    @_command(r, "CANNY")
    def cmd_canny(image, low: int = 50, high: int = 150):
        """CANNY 50 150  — Canny edge detector (thresholds in 8-bit units, output uint8)."""
        _require_image("CANNY", image)
        gray = _to_u8(_ensure_gray(image))
        return cv2.Canny(gray, low, high)

    @_command(r, "DILATE")
    def cmd_dilate(image, ksize: int = 3, iterations: int = 1):
        """DILATE 3 1  — morphological dilation (ksize ≥ 1, iterations ≥ 1)."""
        _require_image("DILATE", image)
        img, _ = _unwrap(image)
        kernel = _morph_kernel("DILATE", ksize, iterations)
        return _cv("DILATE", img, cv2.dilate, img, kernel, iterations=iterations)

    @_command(r, "ERODE")
    def cmd_erode(image, ksize: int = 3, iterations: int = 1):
        """ERODE 3 1  — morphological erosion (ksize ≥ 1, iterations ≥ 1)."""
        _require_image("ERODE", image)
        img, _ = _unwrap(image)
        kernel = _morph_kernel("ERODE", ksize, iterations)
        return _cv("ERODE", img, cv2.erode, img, kernel, iterations=iterations)

    @_command(r, "EQUALIZE_HIST")
    def cmd_equalize_hist(image):
        """EQUALIZE_HIST  — histogram equalisation for contrast enhancement (output uint8 gray)."""
        _require_image("EQUALIZE_HIST", image)
        gray = _to_u8(_ensure_gray(image))
        return cv2.equalizeHist(gray)

    @_command(r, "FIND_CONTOURS")
    def cmd_find_contours(image):
        """FIND_CONTOURS  — detects contours; stores them in pipeline context."""
        _require_image("FIND_CONTOURS", image)
        gray = _to_u8(_ensure_gray(image))
        found = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[-2]   # OpenCV 3 returned 3 values
        wrapper = _wrap(image)
        wrapper["contours"] = contours
        return wrapper

    @_command(r, "DRAW_BOUNDING_BOXES")
    def cmd_draw_bounding_boxes(image):
        """DRAW_BOUNDING_BOXES  — draws green boxes around detected contours."""
        _require_image("DRAW_BOUNDING_BOXES", image)
        img, meta = _unwrap(image)
        contours = meta.get("contours", [])
        canvas = _color_canvas(img)
        colour = _color(canvas, 0, 255, 80)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
        return _wrap(canvas, meta)

    @_command(r, "DRAW_CONTOURS")
    def cmd_draw_contours(image):
        """DRAW_CONTOURS  — draws all contours in magenta."""
        _require_image("DRAW_CONTOURS", image)
        img, meta = _unwrap(image)
        contours = meta.get("contours", [])
        canvas = _color_canvas(img)
        cv2.drawContours(canvas, contours, -1, _color(canvas, 255, 0, 200), 2)
        return _wrap(canvas, meta)

    @_command(r, "PRINT_INFO")
    def cmd_print_info(image):
        """PRINT_INFO  — prints image shape and dtype to stdout."""
        _require_image("PRINT_INFO", image)
        img, meta = _unwrap(image)
        print(f"  shape={img.shape}  dtype={img.dtype}  "
              f"min={img.min()}  max={img.max()}")
        return image

    # ── HEATMAP ───────────────────────────────────────────────────────────────

    _COLORMAPS = {
        "jet":     cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma":  cv2.COLORMAP_PLASMA,
        "hot":     cv2.COLORMAP_HOT,
        "cool":    cv2.COLORMAP_COOL,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "bone":    cv2.COLORMAP_BONE,
    }

    @_command(r, "HEATMAP")
    def cmd_heatmap(image, colormap: str = "inferno"):
        """HEATMAP [colormap]  — apply a false-color heatmap to a grayscale image.

        Converts intensity values to vivid colour for visual inspection of
        gradient fields, depth maps, and threshold proximity.

        Supported colormaps: jet, inferno (default), plasma, hot, cool,
                             rainbow, viridis, bone

        Examples:
            HEATMAP
            HEATMAP jet
            HEATMAP plasma
        """
        _require_image("HEATMAP", image)
        img, meta = _unwrap(image)
        cmap_key = colormap.lower()
        if cmap_key not in _COLORMAPS:
            valid = ", ".join(sorted(_COLORMAPS))
            raise ValueError(
                f"[HEATMAP] Unknown colormap '{colormap}'. Valid options: {valid}"
            )
        gray = _ensure_gray_raw(img).astype(np.float32)
        normalised = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colored = cv2.applyColorMap(normalised, _COLORMAPS[cmap_key])
        return _wrap(colored, meta)

    # ── DRAW_TEXT ─────────────────────────────────────────────────────────────

    _FONT_MAP = {
        "simplex":    cv2.FONT_HERSHEY_SIMPLEX,
        "plain":      cv2.FONT_HERSHEY_PLAIN,
        "duplex":     cv2.FONT_HERSHEY_DUPLEX,
        "complex":    cv2.FONT_HERSHEY_COMPLEX,
        "triplex":    cv2.FONT_HERSHEY_TRIPLEX,
        "small":      cv2.FONT_HERSHEY_COMPLEX_SMALL,
        "script":     cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    }

    @_command(r, "DRAW_TEXT")
    def cmd_draw_text(
        image,
        text:       str,
        x:          int   = 10,
        y:          int   = 30,
        scale:      float = 1.0,
        r_col:      int   = 255,
        g_col:      int   = 255,
        b_col:      int   = 255,
        thickness:  int   = 2,
        font:       str   = "simplex",
    ):
        """DRAW_TEXT "label" x y [scale] [R G B] [thickness] [font]

        Annotate the current image with a text string at pixel position (x, y).
        All parameters after the text are optional and positional.
        Colours are given in 8-bit units and scaled for 16-bit/float images.

        Parameters:
            text       — quoted string to render
            x, y       — top-left anchor in pixels  (default: 10 30)
            scale      — font scale factor           (default: 1.0)
            R G B      — colour components 0-255     (default: 255 255 255 = white)
            thickness  — stroke width in pixels      (default: 2)
            font       — one of: simplex (default), plain, duplex, complex,
                         triplex, small, script

        Examples:
            DRAW_TEXT "Pass" 20 40
            DRAW_TEXT "Defect" 50 80 1.5
            DRAW_TEXT "OK" 10 30 1.0 0 255 0
            DRAW_TEXT "FAIL" 10 30 1.2 255 0 0 3 duplex
        """
        _require_image("DRAW_TEXT", image)
        img, meta = _unwrap(image)
        font_key = font.lower()
        if font_key not in _FONT_MAP:
            valid = ", ".join(sorted(_FONT_MAP))
            raise ValueError(f"[DRAW_TEXT] Unknown font '{font}'. Valid: {valid}")
        canvas = _color_canvas(img)
        colour = _color(canvas, b_col, g_col, r_col)
        args = (str(text), (int(x), int(y)), _FONT_MAP[font_key], float(scale))
        if canvas.dtype == np.uint8:
            cv2.putText(canvas, *args, colour, max(1, int(thickness)), cv2.LINE_AA)
        else:
            # Some OpenCV builds only rasterise text into 8-bit images: draw an
            # anti-aliased 8-bit coverage mask and composite it at full depth.
            mask = np.zeros(canvas.shape[:2], np.uint8)
            cv2.putText(mask, *args, 255, max(1, int(thickness)), cv2.LINE_AA)
            cov = (mask.astype(np.float64) / 255.0)[:, :, None]
            col = np.array(colour, dtype=np.float64)[None, None, :]
            canvas = _cast(canvas.astype(np.float64) * (1 - cov) + col * cov, canvas.dtype)
        return _wrap(canvas, meta)

    # ── CHECKPOINT / OVERLAY ──────────────────────────────────────────────────

    @_command(r, "CHECKPOINT")
    def cmd_checkpoint(image, name: str = "default", _ctx: dict = None):
        """CHECKPOINT [name]  — save a named snapshot of the current image state.

        Stores the current image in the pipeline context under the given name.
        Use OVERLAY / BLEND / COMPARE (or any command taking a buffer name)
        to use it later.

        Examples:
            CHECKPOINT
            CHECKPOINT original
            CHECKPOINT before_threshold
        """
        _require_image("CHECKPOINT", image)
        if _ctx is None:
            raise RuntimeError("[CHECKPOINT] No executor context available.")
        img, meta = _unwrap(image)
        _ctx.setdefault("checkpoints", {})[name] = (img.copy(), dict(meta))
        return image   # pass through unchanged

    @_command(r, "OVERLAY")
    def cmd_overlay(image, name: str = "default", alpha: float = 0.5, _ctx: dict = None):
        """OVERLAY [name] [alpha]  — blend a saved checkpoint onto the current image.

        Composites a previously saved CHECKPOINT back onto the current pipeline
        image using alpha blending:
            result = alpha * checkpoint + (1 - alpha) * current
        The current image's depth and alpha channel are preserved.

        Parameters:
            name   — checkpoint name to blend (default: "default")
            alpha  — weight of the checkpoint layer, 0.0–1.0 (default: 0.5)

        Examples:
            OVERLAY
            OVERLAY original 0.3
            OVERLAY before_threshold 0.7
        """
        _require_image("OVERLAY", image)
        if _ctx is None:
            raise RuntimeError("[OVERLAY] No executor context available.")
        saved_img, saved_meta = _get_buffer(_ctx, name, "OVERLAY", label="checkpoint named")
        current_img, current_meta = _unwrap(image)
        a, b, base_alpha, like = _blend_operands(current_img, saved_img)
        al = float(alpha)
        out = _from_float01(al * b + (1.0 - al) * a, like)
        merged = dict(saved_meta)
        merged.update(current_meta)
        return _wrap(_merge_alpha(out, base_alpha), merged)


# ── Helpers ────────────────────────────────────────────────────────────────────

_THRESH_MODES = {
    "binary": cv2.THRESH_BINARY, "inv": cv2.THRESH_BINARY_INV,
    "binary_inv": cv2.THRESH_BINARY_INV, "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO, "tozero_inv": cv2.THRESH_TOZERO_INV,
}

_FLOAT_FORMATS = {".tif", ".tiff", ".exr", ".hdr", ".pfm"}
_U16_FORMATS = {".png", ".tif", ".tiff", ".pgm", ".ppm", ".pnm", ".jp2"}
_NO_ALPHA_FORMATS = {".jpg", ".jpeg", ".jpe", ".ppm", ".pgm", ".pnm"}


def _require_image(cmd: str, image):
    if image is None:
        raise RuntimeError(
            f"[{cmd}] No image loaded. Did you forget LOAD?"
        )


def _ensure_gray(image) -> np.ndarray:
    """Unwrap and convert to grayscale."""
    img, _ = _unwrap(image)
    return _ensure_gray_raw(img)


def _ensure_gray_raw(img: np.ndarray) -> np.ndarray:
    """Convert a raw ndarray (gray, BGR or BGRA, any depth) to grayscale."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Cannot convert image of shape {img.shape} to grayscale")
    if img.dtype in (np.uint8, np.uint16, np.float32):
        code = cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(img, code)
    weights = np.array([0.114, 0.587, 0.299])
    return _cast(img[:, :, :3].astype(np.float64) @ weights, img.dtype)


def _wrap(image, meta: Optional[dict] = None) -> dict:
    """Wrap an ndarray in a thin dict so metadata can travel alongside it.

    Never mutates its input: a wrapped dict is shallow-copied.
    """
    if isinstance(image, dict):
        out = dict(image)
    else:
        out = {"_image": image}
    if meta:
        for k, v in meta.items():
            if k != "_image":
                out[k] = v
    return out


def _unwrap(image):
    """Extract (ndarray, metadata_dict) from either a raw array or a wrapped dict."""
    if isinstance(image, dict):
        img = image["_image"]
        meta = {k: v for k, v in image.items() if k != "_image"}
        return img, meta
    return image, {}


def _get_buffer(ctx: Optional[dict], name: Any, cmd: str,
                label: str = "buffer/checkpoint") -> Tuple[np.ndarray, dict]:
    """Fetch a named buffer from ``ctx["checkpoints"]`` as (ndarray, meta).

    Accepts the CHECKPOINT tuple form ``(img, meta)``, a raw ndarray, a
    wrapped dict ``{"_image": ...}`` or any object exposing ``.data`` /
    ``.array`` / ``.image``.
    """
    if ctx is None:
        raise RuntimeError(f"[{cmd}] Requires executor context.")
    store = ctx.get("checkpoints", {})
    key = str(name)
    if key not in store:
        available = list(store.keys()) or ["(none)"]
        raise RuntimeError(
            f"[{cmd}] No {label} '{key}'. Available: {', '.join(map(str, available))}"
        )
    return _as_array(store[key], cmd, key)


def _as_array(entry: Any, cmd: str = "", key: str = "") -> Tuple[np.ndarray, dict]:
    if isinstance(entry, np.ndarray):
        return entry, {}
    if isinstance(entry, dict) and "_image" in entry:
        return _unwrap(entry)
    if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], np.ndarray):
        meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        return entry[0], dict(meta)
    for attr in ("data", "array", "image"):
        val = getattr(entry, attr, None)
        if isinstance(val, np.ndarray):
            return val, dict(getattr(entry, "meta", {}) or {})
    raise TypeError(f"[{cmd}] Buffer '{key}' does not hold an image ({type(entry).__name__})")


def _cv(cmd: str, img: np.ndarray, fn, *args, **kwargs):
    """Call an OpenCV function, turning cv2.error into a readable ValueError."""
    try:
        return fn(*args, **kwargs)
    except cv2.error as exc:
        ch = 1 if img.ndim == 2 else img.shape[2]
        detail = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
        raise ValueError(
            f"[{cmd}] OpenCV cannot process this image (dtype {img.dtype}, "
            f"{ch} channel(s)). Try CONVERT_DEPTH u8 first. ({detail})"
        ) from None


def _odd_kernel(cmd: str, k: int, minimum: int = 1) -> int:
    """Validate a kernel size: >= minimum; even values round up to the next odd."""
    k = int(k)
    if k < minimum:
        raise ValueError(f"[{cmd}] kernel size must be ≥ {minimum}, got {k}")
    return k if k % 2 == 1 else k + 1


def _morph_kernel(cmd: str, ksize: int, iterations: int):
    if ksize < 1:
        raise ValueError(f"[{cmd}] ksize must be ≥ 1, got {ksize}")
    if iterations < 1:
        raise ValueError(f"[{cmd}] iterations must be ≥ 1, got {iterations}")
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


def _dtype_max(dtype) -> float:
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return 1
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    return 1.0


def _float_max(img: np.ndarray) -> float:
    """Assumed full-scale value of a float image: 1.0 if max <= 1 else 255.0."""
    if img.size == 0:
        return 1.0
    return 1.0 if float(np.nanmax(img)) <= 1.0 else 255.0


def _value_max(img: np.ndarray) -> float:
    """Full-scale value of *img* (255, 65535, 1.0 or 255.0 for floats)."""
    if np.issubdtype(img.dtype, np.floating):
        return _float_max(img)
    return _dtype_max(img.dtype)


def _cast(values: np.ndarray, dtype) -> np.ndarray:
    """Round/clip float values into *dtype* (saturating like OpenCV)."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    if dtype == np.bool_:
        return values > 0
    return np.asarray(values).astype(dtype)


def _to_u8(img: np.ndarray) -> np.ndarray:
    """Scale any depth to uint8 by its convention (uint16 /257, float [0,1] *255)."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    if img.dtype == np.uint16:
        return ((img.astype(np.uint32) + 128) // 257).astype(np.uint8)
    if np.issubdtype(img.dtype, np.floating):
        scale = 255.0 if _float_max(img) <= 1.0 else 1.0
        return _cast(np.nan_to_num(img.astype(np.float64)) * scale, np.uint8)
    if np.issubdtype(img.dtype, np.integer):
        return _cast(img.astype(np.float64), np.uint8)
    raise ValueError(f"Unsupported dtype {img.dtype}")


def _to_float01(img: np.ndarray) -> np.ndarray:
    """Return float32 values in [0, 1] according to the depth convention."""
    return (img.astype(np.float32) / np.float32(_value_max(img)))


def _from_float01(values: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Inverse of _to_float01 using *like*'s dtype/scale."""
    values = np.clip(values, 0.0, 1.0)
    return _cast(values * _value_max(like), like.dtype)


def _channels(img: np.ndarray) -> int:
    return 1 if img.ndim == 2 else img.shape[2]


def _split_alpha(img: np.ndarray):
    """(colour part, alpha or None) for BGRA images; others pass through."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    return img, None


def _merge_alpha(color: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    if alpha is None:
        return color
    if color.ndim == 2:
        color = np.stack([color] * 3, axis=-1)
    if alpha.dtype != color.dtype:
        alpha = _from_float01(_to_float01(alpha), color)
    return np.dstack([color, alpha])


def _match_channels(img: np.ndarray, n: int) -> np.ndarray:
    """Convert *img* to n channels (1, 3 or 4) keeping its dtype."""
    c = _channels(img)
    if c == n:
        return img
    if n == 1:
        return _ensure_gray_raw(img)
    color, alpha = _split_alpha(img)
    if color.ndim == 2:
        color = np.stack([color] * 3, axis=-1)
    elif color.shape[2] == 1:
        color = np.repeat(color, 3, axis=2)
    if n == 3:
        return color
    if alpha is None:
        alpha = np.full(color.shape[:2], _value_max(img), dtype=img.dtype)
    return np.dstack([color, alpha])


def _blend_operands(current: np.ndarray, layer: np.ndarray):
    """Prepare two images for blending.

    Returns (a, b, base_alpha, like): float [0,1] colour planes of the current
    image (a) and the layer (b, resized/channel-matched), the current alpha
    channel (or None) and the current colour array (for dtype restoration).
    """
    base_color, base_alpha = _split_alpha(current)
    layer_color, _ = _split_alpha(layer)
    n = 3 if (_channels(base_color) >= 3 or _channels(layer_color) >= 3) else 1
    base_color = _match_channels(base_color, n)
    layer_color = _match_channels(layer_color, n)
    if layer_color.shape[:2] != base_color.shape[:2]:
        layer_color = cv2.resize(_to_float01(layer_color),
                                 (base_color.shape[1], base_color.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        b = layer_color
    else:
        b = _to_float01(layer_color)
    return _to_float01(base_color), b, base_alpha, base_color


def _color_canvas(img: np.ndarray) -> np.ndarray:
    """A drawable copy: gray → 3 channels (same dtype); BGR/BGRA copied."""
    if img.ndim == 2 or img.shape[2] == 1:
        return _match_channels(img, 3).copy()
    return img.copy()


def _color(canvas: np.ndarray, b: float, g: float, r: float) -> tuple:
    """A BGR(A) colour tuple given in 8-bit units, scaled to the canvas depth."""
    scale = _value_max(canvas) / 255.0
    col = [float(b) * scale, float(g) * scale, float(r) * scale]
    if np.issubdtype(canvas.dtype, np.integer):
        col = [int(round(c)) for c in col]
    if _channels(canvas) == 4:
        top = _value_max(canvas)
        col.append(int(top) if np.issubdtype(canvas.dtype, np.integer) else float(top))
    return tuple(col)


def _write_image(cmd: str, path: str, img: np.ndarray) -> Path:
    """imwrite with parent-dir creation, depth adaptation and clear errors."""
    p = Path(path)
    ext = p.suffix.lower()
    if not ext:
        raise ValueError(f"[{cmd}] Output path {path!r} has no file extension "
                         f"(e.g. .png, .jpg, .tif).")
    out = img
    if out.dtype == np.bool_:
        out = out.astype(np.uint8) * 255
    elif np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float32) if ext in _FLOAT_FORMATS else _to_u8(out)
    elif out.dtype == np.uint16:
        if ext not in _U16_FORMATS:
            out = _to_u8(out)
    elif out.dtype != np.uint8:
        out = _to_u8(out)
    if ext in _NO_ALPHA_FORMATS and out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imwrite(str(p), out)
    except cv2.error as exc:
        raise OSError(f"[{cmd}] Cannot write {path!r}: unsupported format "
                      f"or data ({str(exc).strip().splitlines()[-1]})") from None
    if not ok:
        raise OSError(f"[{cmd}] Failed to write: {path!r}")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# v0.3 — utility, statistics, percentage resizing
# ══════════════════════════════════════════════════════════════════════════════

def register_v03(r):
    """Register v0.3 utility commands into an existing registry."""

    @_command(r, "RESIZE_PERCENT")
    def cmd_resize_percent(image, pct: float):
        """RESIZE_PERCENT 50  — resize to a percentage of current dimensions.

        Unlike RESIZE which requires absolute pixel dimensions, RESIZE_PERCENT
        scales proportionally. Useful inside REPEAT loops.

        Examples:
            RESIZE_PERCENT 50       # half size
            RESIZE_PERCENT 200      # double size
            RESIZE_PERCENT $scale   # variable scale
        """
        _require_image("RESIZE_PERCENT", image)
        img, meta = _unwrap(image)
        factor = float(pct) / 100.0
        if factor <= 0:
            raise ValueError(f"[RESIZE_PERCENT] Percentage must be > 0, got {pct}")
        h, w = img.shape[:2]
        new_w = max(1, int(w * factor))
        new_h = max(1, int(h * factor))
        interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        resized = _cv("RESIZE_PERCENT", img, cv2.resize, img, (new_w, new_h),
                      interpolation=interp)
        return _wrap(resized, meta)

    @_command(r, "HISTOGRAM_SAVE")
    def cmd_histogram_save(image, path: str = "histogram.png"):
        """HISTOGRAM_SAVE ["output.png"]  — save a histogram plot of the current image.

        Generates a 256-bin pixel-intensity histogram over the full range of
        the image depth (0-255, 0-65535, or 0-1/0-255 for floats) and saves it
        as a PNG. Grayscale → single curve, BGR(A) → B/G/R curves (alpha is
        ignored). Passes the image through unchanged; parent dirs are created.

        Examples:
            HISTOGRAM_SAVE
            HISTOGRAM_SAVE "reports/channel_hist.png"
        """
        _require_image("HISTOGRAM_SAVE", image)
        img, meta = _unwrap(image)

        hist_h, hist_w = 256, 512
        canvas = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        color, _ = _split_alpha(img)
        top = float(_value_max(color))

        if color.ndim == 2 or color.shape[2] == 1:
            channels = [(color.reshape(color.shape[:2]), (200, 200, 200))]
        else:
            channels = [(color[:, :, 0], (255, 80, 80)),
                        (color[:, :, 1], (80, 255, 80)),
                        (color[:, :, 2], (80, 80, 255))]

        for ch_img, colour in channels:
            hist, _ = np.histogram(ch_img.ravel(), bins=256, range=(0.0, top + 1e-9
                                   if np.issubdtype(color.dtype, np.floating) else top + 1))
            hist = hist.astype(np.float64).ravel()
            peak = hist.max() if hist.size else 0
            if peak > 0:
                hist = hist * ((hist_h - 20) / peak)
            pts = np.array(
                [[int(x * (hist_w / 256)), hist_h - int(hist[x])] for x in range(256)],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)

        # Axes
        cv2.line(canvas, (0, hist_h - 1), (hist_w, hist_h - 1), (60, 60, 60), 1)
        cv2.line(canvas, (0, 0), (0, hist_h), (60, 60, 60), 1)

        _write_image("HISTOGRAM_SAVE", str(path), canvas)
        return _wrap(img, meta)

    @_command(r, "PIPELINE_STATS")
    def cmd_pipeline_stats(image, _ctx: dict = None):
        """PIPELINE_STATS  — print execution statistics to stdout.

        Displays a formatted table of per-command timing totals drawn from
        the executor's stats context. Useful for identifying bottlenecks.

        Example:
            PIPELINE_STATS
        """
        _require_image("PIPELINE_STATS", image)
        if _ctx is None or "stats" not in _ctx:
            print("  [PIPELINE_STATS] No stats available (not running via Executor).")
            return image

        stats = _ctx["stats"]
        cmd_times = stats.get("command_times", {})

        print(f"\n  {'─' * 60}")
        print(f"  {'PIPELINE STATS':^60}")
        print(f"  {'─' * 60}")
        print(f"  {'Command':<26} {'Calls':>5}  {'Total ms':>10}  {'Avg ms':>8}")
        print(f"  {'─' * 60}")

        rows = []
        for name, times in cmd_times.items():
            total = sum(times)
            avg   = total / len(times)
            rows.append((name, len(times), total, avg))

        rows.sort(key=lambda row: row[2], reverse=True)
        for name, calls, total, avg in rows:
            print(f"  {name:<26} {calls:>5}  {total:>10.1f}  {avg:>8.2f}")

        print(f"  {'─' * 60}")
        print(f"  {'TOTAL':<26} {stats['commands_executed']:>5}  "
              f"{stats['total_ms']:>10.1f}")
        print(f"  {'─' * 60}\n")

        return image


# ══════════════════════════════════════════════════════════════════════════════
# v0.4
# ══════════════════════════════════════════════════════════════════════════════

def register_v04(r):
    """Register v0.4 commands into an existing registry."""

    @_command(r, "COMPARE")
    def cmd_compare(image, checkpoint_name: str = "default", _ctx: dict = None):
        """COMPARE ["checkpoint"]  — compute diff metrics vs a named checkpoint.

        Calculates and prints (on a common 0-255 scale, alpha ignored):
          - PSNR (Peak Signal-to-Noise Ratio) — higher = more similar
          - MAD  (mean absolute difference)
          - Pixel-level difference image is stored as checkpoint "diff"

        Examples:
            CHECKPOINT "reference"
            <processing...>
            COMPARE "reference"
        """
        _require_image("COMPARE", image)
        if _ctx is None:
            raise RuntimeError("[COMPARE] Requires executor context (_ctx).")
        saved_img, _ = _get_buffer(_ctx, checkpoint_name, "COMPARE", label="checkpoint")
        current_img, _ = _unwrap(image)

        a = _to_float01(_match_channels(_split_alpha(current_img)[0], 3))
        b = _to_float01(_match_channels(_split_alpha(saved_img)[0], 3))
        if b.shape[:2] != a.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))

        diff_f = np.abs(a - b)
        mse = float(np.mean(diff_f ** 2))
        psnr = 100.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)
        mad = float(np.mean(diff_f)) * 255.0

        diff = _cast(np.clip(diff_f * 4 * 255, 0, 255), np.uint8)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _ctx["checkpoints"]["diff"] = (diff_gray, {})

        print(f"\n  ┌─ COMPARE vs '{checkpoint_name}' ──────────────────────┐")
        print(f"  │  PSNR : {psnr:7.2f} dB   (∞ = identical)            │")
        print(f"  │  MAD  : {mad:7.2f}      (0 = identical)              │")
        print("  │  diff checkpoint saved as 'diff'                     │")
        print("  └──────────────────────────────────────────────────────┘\n")

        return image  # passthrough

    @_command(r, "AUTO_CROP")
    def cmd_auto_crop(image, padding: int = 5):
        """AUTO_CROP [padding]  — automatically crop to the bounding box of non-zero content.

        Finds the tight bounding box around all non-zero (colour) pixels and
        crops to it, with optional padding in pixels (≥ 0). An all-black
        image is passed through unchanged.

        Examples:
            AUTO_CROP
            AUTO_CROP 10
        """
        _require_image("AUTO_CROP", image)
        if padding < 0:
            raise ValueError(f"[AUTO_CROP] padding must be ≥ 0, got {padding}")
        img, meta = _unwrap(image)
        color, _ = _split_alpha(img)
        mask = color != 0
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0:
            return image
        ih, iw = img.shape[:2]
        y1 = max(0, int(rows[0]) - padding)
        y2 = min(ih, int(rows[-1]) + 1 + padding)
        x1 = max(0, int(cols[0]) - padding)
        x2 = min(iw, int(cols[-1]) + 1 + padding)
        return _wrap(img[y1:y2, x1:x2].copy(), meta)

    @_command(r, "BLEND")
    def cmd_blend(image, checkpoint_name: str, alpha: float = 0.5,
                  mode: str = "normal", _ctx: dict = None):
        """BLEND "checkpoint" [alpha] [mode]  — blend two images with a compositing mode.

        Unlike OVERLAY (which always uses linear alpha), BLEND supports multiple
        compositing modes. Works on any depth (computed in [0,1] floats and
        converted back to the current image's depth); the current alpha
        channel is preserved.

        Modes:
            normal     — standard alpha composite (same as OVERLAY)
            difference — absolute pixel difference
            multiply   — pixel-by-pixel multiply (darkens)
            screen     — inverted multiply (brightens)
            add        — additive blend (clamps at full scale)
            lighten    — max per-pixel value
            darken     — min per-pixel value

        Examples:
            BLEND "original" 0.5
            BLEND "original" 0.3 "difference"
            BLEND "mask" 1.0 "multiply"
        """
        _require_image("BLEND", image)
        if _ctx is None:
            raise RuntimeError("[BLEND] Requires executor context.")
        saved_img, _ = _get_buffer(_ctx, checkpoint_name, "BLEND", label="checkpoint")
        current_img, current_meta = _unwrap(image)
        a, b, base_alpha, like = _blend_operands(current_img, saved_img)
        al = float(alpha)

        mode = mode.lower()
        if mode == "normal":
            result_f = al * b + (1.0 - al) * a
        elif mode == "difference":
            result_f = np.abs(a - b)
        elif mode == "multiply":
            result_f = al * (a * b) + (1.0 - al) * a
        elif mode == "screen":
            result_f = 1.0 - (1.0 - a) * (1.0 - b)
            result_f = al * result_f + (1.0 - al) * a
        elif mode == "add":
            result_f = np.clip(a + al * b, 0, 1)
        elif mode == "lighten":
            result_f = al * np.maximum(a, b) + (1.0 - al) * a
        elif mode == "darken":
            result_f = al * np.minimum(a, b) + (1.0 - al) * a
        else:
            valid = "normal, difference, multiply, screen, add, lighten, darken"
            raise ValueError(f"[BLEND] Unknown mode '{mode}'. Valid: {valid}")

        out = _merge_alpha(_from_float01(result_f, like), base_alpha)
        return _wrap(out, current_meta)
