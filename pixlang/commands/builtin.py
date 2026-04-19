# pixlang/commands/builtin.py
"""
Built-in PixLang commands backed by OpenCV + NumPy.

Each function signature:
    fn(image: np.ndarray | None, *args) -> np.ndarray
"""
import cv2
import numpy as np
from .registry import CommandRegistry


def register_all(r: CommandRegistry):
    """Wire all built-ins into the given registry."""

    # ── I/O ──────────────────────────────────────────────────────────────────

    @r.register("LOAD")
    def cmd_load(image, path: str):
        """LOAD "path/to/image.png"  — loads image from disk."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"[LOAD] Could not read image: {path!r}")
        return img

    @r.register("SAVE")
    def cmd_save(image, path: str):
        """SAVE "output.png"  — writes current image to disk."""
        _require_image("SAVE", image)
        img, _ = _unwrap(image)
        ok = cv2.imwrite(path, img)
        if not ok:
            raise IOError(f"[SAVE] Failed to write: {path!r}")
        return image  # pass through unchanged

    # ── Geometry ─────────────────────────────────────────────────────────────

    @r.register("RESIZE")
    def cmd_resize(image, width: int, height: int):
        """RESIZE 640 480"""
        _require_image("RESIZE", image)
        img, _ = _unwrap(image)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    @r.register("CROP")
    def cmd_crop(image, x: int, y: int, w: int, h: int):
        """CROP x y width height  — crops a rectangular region."""
        _require_image("CROP", image)
        img, _ = _unwrap(image)
        return img[y:y+h, x:x+w]

    @r.register("ROTATE")
    def cmd_rotate(image, angle: float):
        """ROTATE 90  — rotates by angle degrees (counter-clockwise)."""
        _require_image("ROTATE", image)
        img, _ = _unwrap(image)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    @r.register("FLIP")
    def cmd_flip(image, axis: int):
        """FLIP 0=vertical  1=horizontal  -1=both"""
        _require_image("FLIP", image)
        img, _ = _unwrap(image)
        return cv2.flip(img, axis)

    # ── Color ─────────────────────────────────────────────────────────────────

    @r.register("GRAYSCALE")
    def cmd_grayscale(image):
        """GRAYSCALE  — converts to single-channel grayscale."""
        _require_image("GRAYSCALE", image)
        img, _ = _unwrap(image)
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @r.register("INVERT")
    def cmd_invert(image):
        """INVERT  — bitwise-NOT on all pixel values."""
        _require_image("INVERT", image)
        img, _ = _unwrap(image)
        return cv2.bitwise_not(img)

    @r.register("NORMALIZE")
    def cmd_normalize(image):
        """NORMALIZE  — stretch histogram to [0, 255]."""
        _require_image("NORMALIZE", image)
        img, _ = _unwrap(image)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # ── Thresholding ──────────────────────────────────────────────────────────

    @r.register("THRESHOLD")
    def cmd_threshold(image, value: int):
        """THRESHOLD 120  — binary threshold at given value."""
        _require_image("THRESHOLD", image)
        gray = _ensure_gray(image)
        _, result = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
        return result

    @r.register("THRESHOLD_OTSU")
    def cmd_threshold_otsu(image):
        """THRESHOLD_OTSU  — automatic Otsu binarisation."""
        _require_image("THRESHOLD_OTSU", image)
        gray = _ensure_gray(image)
        _, result = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return result

    @r.register("ADAPTIVE_THRESHOLD")
    def cmd_adaptive_threshold(image, block_size: int = 11, C: int = 2):
        """ADAPTIVE_THRESHOLD 11 2  — local adaptive binarisation."""
        _require_image("ADAPTIVE_THRESHOLD", image)
        gray = _ensure_gray(image)
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, C
        )

    @r.register("BLUR")
    def cmd_blur(image, ksize: int = 5):
        """BLUR 5  — Gaussian blur with kernel size ksize×ksize."""
        _require_image("BLUR", image)
        img, _ = _unwrap(image)
        k = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    @r.register("MEDIAN_BLUR")
    def cmd_median_blur(image, ksize: int = 5):
        """MEDIAN_BLUR 5  — salt-and-pepper noise removal."""
        _require_image("MEDIAN_BLUR", image)
        img, _ = _unwrap(image)
        k = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(img, k)

    @r.register("SHARPEN")
    def cmd_sharpen(image):
        """SHARPEN  — unsharp mask sharpening."""
        _require_image("SHARPEN", image)
        img, _ = _unwrap(image)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    @r.register("CANNY")
    def cmd_canny(image, low: int = 50, high: int = 150):
        """CANNY 50 150  — Canny edge detector."""
        _require_image("CANNY", image)
        gray = _ensure_gray(image)
        return cv2.Canny(gray, low, high)

    @r.register("DILATE")
    def cmd_dilate(image, ksize: int = 3, iterations: int = 1):
        """DILATE 3 1  — morphological dilation."""
        _require_image("DILATE", image)
        img, _ = _unwrap(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.dilate(img, kernel, iterations=iterations)

    @r.register("ERODE")
    def cmd_erode(image, ksize: int = 3, iterations: int = 1):
        """ERODE 3 1  — morphological erosion."""
        _require_image("ERODE", image)
        img, _ = _unwrap(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.erode(img, kernel, iterations=iterations)

    @r.register("EQUALIZE_HIST")
    def cmd_equalize_hist(image):
        """EQUALIZE_HIST  — histogram equalisation for contrast enhancement."""
        _require_image("EQUALIZE_HIST", image)
        gray = _ensure_gray(image)
        return cv2.equalizeHist(gray)

    @r.register("FIND_CONTOURS")
    def cmd_find_contours(image):
        """FIND_CONTOURS  — detects contours; stores them in pipeline context."""
        _require_image("FIND_CONTOURS", image)
        gray = _ensure_gray(image)
        contours, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        wrapper = _wrap(image)
        wrapper["contours"] = contours
        return wrapper

    @r.register("DRAW_BOUNDING_BOXES")
    def cmd_draw_bounding_boxes(image):
        """DRAW_BOUNDING_BOXES  — draws green boxes around detected contours."""
        _require_image("DRAW_BOUNDING_BOXES", image)
        img, meta = _unwrap(image)
        contours = meta.get("contours", [])
        if len(img.shape) == 2:
            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            canvas = img.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 80), 2)
        result = _wrap(canvas)
        result["contours"] = contours
        return result

    @r.register("DRAW_CONTOURS")
    def cmd_draw_contours(image):
        """DRAW_CONTOURS  — draws all contours in magenta."""
        _require_image("DRAW_CONTOURS", image)
        img, meta = _unwrap(image)
        contours = meta.get("contours", [])
        if len(img.shape) == 2:
            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            canvas = img.copy()
        cv2.drawContours(canvas, contours, -1, (255, 0, 200), 2)
        result = _wrap(canvas)
        result["contours"] = contours
        return result

    @r.register("PRINT_INFO")
    def cmd_print_info(image):
        """PRINT_INFO  — prints image shape and dtype to stdout."""
        _require_image("PRINT_INFO", image)
        img, meta = _unwrap(image)
        print(f"  shape={img.shape}  dtype={img.dtype}  "
              f"min={img.min()}  max={img.max()}")
        return image

    # ══════════════════════════════════════════════════════════════════════════
    # NEW COMMANDS (v0.2)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Command 1: HEATMAP ────────────────────────────────────────────────────

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

    @r.register("HEATMAP")
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

        # Normalise to [0, 255] uint8 before applying colormap
        gray = _ensure_gray_raw(img)
        normalised = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        cmap_key = colormap.lower()
        if cmap_key not in _COLORMAPS:
            valid = ", ".join(sorted(_COLORMAPS))
            raise ValueError(
                f"[HEATMAP] Unknown colormap '{colormap}'. Valid options: {valid}"
            )

        colored = cv2.applyColorMap(normalised, _COLORMAPS[cmap_key])

        result = _wrap(colored)
        result.update({k: v for k, v in meta.items()})   # preserve upstream metadata
        return result

    # ── Command 2: DRAW_TEXT ──────────────────────────────────────────────────

    _FONT_MAP = {
        "simplex":    cv2.FONT_HERSHEY_SIMPLEX,
        "plain":      cv2.FONT_HERSHEY_PLAIN,
        "duplex":     cv2.FONT_HERSHEY_DUPLEX,
        "complex":    cv2.FONT_HERSHEY_COMPLEX,
        "triplex":    cv2.FONT_HERSHEY_TRIPLEX,
        "small":      cv2.FONT_HERSHEY_COMPLEX_SMALL,
        "script":     cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    }

    @r.register("DRAW_TEXT")
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

        # Ensure BGR canvas
        if len(img.shape) == 2:
            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            canvas = img.copy()

        font_key = font.lower()
        if font_key not in _FONT_MAP:
            valid = ", ".join(sorted(_FONT_MAP))
            raise ValueError(f"[DRAW_TEXT] Unknown font '{font}'. Valid: {valid}")

        cv2.putText(
            canvas,
            str(text),
            (int(x), int(y)),
            _FONT_MAP[font_key],
            float(scale),
            (int(b_col), int(g_col), int(r_col)),   # OpenCV uses BGR
            int(thickness),
            cv2.LINE_AA,
        )

        result = _wrap(canvas)
        result.update({k: v for k, v in meta.items()})
        return result

    # ── Command 3: OVERLAY ────────────────────────────────────────────────────

    @r.register("CHECKPOINT")
    def cmd_checkpoint(image, name: str = "default", _ctx: dict = None):
        """CHECKPOINT [name]  — save a named snapshot of the current image state.

        Stores the current image in the pipeline context under the given name.
        Use OVERLAY to blend a checkpoint back onto the current image.

        Examples:
            CHECKPOINT
            CHECKPOINT original
            CHECKPOINT before_threshold
        """
        _require_image("CHECKPOINT", image)
        if _ctx is None:
            raise RuntimeError("[CHECKPOINT] No executor context available.")
        img, meta = _unwrap(image)
        _ctx["checkpoints"][name] = (img.copy(), dict(meta))
        return image   # pass through unchanged

    @r.register("OVERLAY")
    def cmd_overlay(image, name: str = "default", alpha: float = 0.5, _ctx: dict = None):
        """OVERLAY [name] [alpha]  — blend a saved checkpoint onto the current image.

        Composites a previously saved CHECKPOINT back onto the current pipeline
        image using alpha blending:
            result = alpha * checkpoint + (1 - alpha) * current

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

        checkpoints = _ctx.get("checkpoints", {})
        if name not in checkpoints:
            available = list(checkpoints.keys()) or ["(none)"]
            raise RuntimeError(
                f"[OVERLAY] No checkpoint named '{name}'. "
                f"Available: {', '.join(available)}"
            )

        saved_img, saved_meta = checkpoints[name]
        current_img, current_meta = _unwrap(image)

        # Normalise both images to the same colour depth for blending
        def to_bgr(arr):
            if len(arr.shape) == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return arr.copy()

        base  = to_bgr(current_img)
        layer = to_bgr(saved_img)

        # Resize checkpoint to match current image if shapes differ
        if layer.shape[:2] != base.shape[:2]:
            layer = cv2.resize(layer, (base.shape[1], base.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        blended = cv2.addWeighted(layer, float(alpha), base, 1.0 - float(alpha), 0)

        result = _wrap(blended)
        # Merge metadata: current wins on key conflicts
        merged_meta = dict(saved_meta)
        merged_meta.update(current_meta)
        result.update(merged_meta)
        return result


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    """Convert a raw ndarray to grayscale (no unwrap)."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _wrap(image) -> dict:
    """Wrap an ndarray in a thin dict so metadata can travel alongside it."""
    if isinstance(image, dict):
        image["_image"] = image.get("_image", image)
        return image
    return {"_image": image}


def _unwrap(image):
    """Extract (ndarray, metadata_dict) from either a raw array or a wrapped dict."""
    if isinstance(image, dict):
        img = image["_image"]
        meta = {k: v for k, v in image.items() if k != "_image"}
        return img, meta
    return image, {}


# ══════════════════════════════════════════════════════════════════════════════
# NEW COMMANDS (v0.3) — utility, statistics, percentage resizing
# These are appended here; a future refactor can split into sub-modules.
# ══════════════════════════════════════════════════════════════════════════════

def register_v03(r):
    """Register v0.3 utility commands into an existing registry."""

    @r.register("RESIZE_PERCENT")
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
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        result = _wrap(resized)
        result.update(meta)
        return result

    @r.register("HISTOGRAM_SAVE")
    def cmd_histogram_save(image, path: str = "histogram.png"):
        """HISTOGRAM_SAVE ["output.png"]  — save a histogram plot of the current image.

        Generates a pixel-intensity histogram and saves it as a PNG.
        Works on both grayscale (single curve) and BGR (R/G/B channels).
        Passes the image through unchanged.

        Examples:
            HISTOGRAM_SAVE
            HISTOGRAM_SAVE "reports/channel_hist.png"
        """
        _require_image("HISTOGRAM_SAVE", image)
        img, meta = _unwrap(image)

        # Build a simple histogram image using only numpy + cv2 (no matplotlib)
        hist_h, hist_w = 256, 512
        canvas = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

        if len(img.shape) == 2:
            channels = [(img, (200, 200, 200))]
        else:
            b, g, r = cv2.split(img)
            channels = [(b, (255, 80, 80)), (g, (80, 255, 80)), (r, (80, 80, 255))]

        for ch_img, colour in channels:
            hist = cv2.calcHist([ch_img], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h - 20, cv2.NORM_MINMAX)
            pts = []
            for x in range(256):
                bar_x = int(x * (hist_w / 256))
                bar_h = int(hist[x][0])
                pts.append((bar_x, hist_h - bar_h))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], colour, 1, cv2.LINE_AA)

        # Axes
        cv2.line(canvas, (0, hist_h - 1), (hist_w, hist_h - 1), (60, 60, 60), 1)
        cv2.line(canvas, (0, 0), (0, hist_h), (60, 60, 60), 1)

        cv2.imwrite(path, canvas)
        result = _wrap(img)
        result.update(meta)
        return result

    @r.register("PIPELINE_STATS")
    def cmd_pipeline_stats(image, _ctx: dict = None):
        """PIPELINE_STATS  — print execution statistics to stdout.

        Displays a formatted table of per-command timing totals drawn from
        the executor's stats context. Useful for identifying bottlenecks.

        Outputs:
            - Total commands executed so far
            - Per-command cumulative and average time
            - Slowest command

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

        rows.sort(key=lambda r: r[2], reverse=True)
        for name, calls, total, avg in rows:
            print(f"  {name:<26} {calls:>5}  {total:>10.1f}  {avg:>8.2f}")

        print(f"  {'─' * 60}")
        print(f"  {'TOTAL':<26} {stats['commands_executed']:>5}  "
              f"{stats['total_ms']:>10.1f}")
        print(f"  {'─' * 60}\n")

        return image


def register_v04(r):
    """Register v0.4 commands into an existing registry."""

    @r.register("COMPARE")
    def cmd_compare(image, checkpoint_name: str = "default", _ctx: dict = None):
        """COMPARE ["checkpoint"]  — compute diff metrics vs a named checkpoint.

        Calculates and prints:
          - PSNR (Peak Signal-to-Noise Ratio) — higher = more similar
          - SSIM-approximation (mean structural difference)
          - Pixel-level difference image is stored as checkpoint "diff"

        Useful for quality-checking pipeline results against a reference.

        Examples:
            CHECKPOINT "reference"
            <processing...>
            COMPARE "reference"
        """
        _require_image("COMPARE", image)
        if _ctx is None:
            raise RuntimeError("[COMPARE] Requires executor context (_ctx).")

        checkpoints = _ctx.get("checkpoints", {})
        if checkpoint_name not in checkpoints:
            available = list(checkpoints.keys()) or ["(none)"]
            raise RuntimeError(
                f"[COMPARE] No checkpoint '{checkpoint_name}'. "
                f"Available: {', '.join(available)}"
            )

        current_img, current_meta = _unwrap(image)
        saved_img,   _            = checkpoints[checkpoint_name]

        # Normalise to same size and type
        def _prep(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img.copy()

        a = _prep(current_img).astype(np.float32)
        b = _prep(saved_img)
        if b.shape[:2] != a.shape[:2]:
            b = cv2.resize(b.astype(np.float32), (a.shape[1], a.shape[0]))
        else:
            b = b.astype(np.float32)

        # PSNR
        mse  = np.mean((a - b) ** 2)
        psnr = 100.0 if mse < 1e-10 else 10 * np.log10(255.0 ** 2 / mse)

        # Mean absolute difference (simplified SSIM proxy)
        mad = float(np.mean(np.abs(a - b)))

        # Difference image — amplified for visibility
        diff = np.clip(np.abs(a - b) * 4, 0, 255).astype(np.uint8)
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff

        # Store diff as a checkpoint
        _ctx["checkpoints"]["diff"] = (diff_gray, {})

        print(f"\n  ┌─ COMPARE vs '{checkpoint_name}' ──────────────────────┐")
        print(f"  │  PSNR : {psnr:7.2f} dB   (∞ = identical)            │")
        print(f"  │  MAD  : {mad:7.2f}      (0 = identical)              │")
        print(f"  │  diff checkpoint saved as 'diff'                     │")
        print(f"  └──────────────────────────────────────────────────────┘\n")

        return image  # passthrough

    @r.register("AUTO_CROP")
    def cmd_auto_crop(image, padding: int = 5):
        """AUTO_CROP [padding]  — automatically crop to the bounding box of non-zero content.

        Finds the tight bounding box around all non-background pixels and
        crops to it, with optional padding in pixels.

        Useful after THRESHOLD to remove dead border space, or to tightly
        frame a detected object after FIND_CONTOURS + DRAW_BOUNDING_BOXES.

        Examples:
            AUTO_CROP
            AUTO_CROP 10
        """
        _require_image("AUTO_CROP", image)
        img, meta = _unwrap(image)

        # Convert to grayscale for mask detection
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()

        # Find bounding rect of non-zero pixels
        non_zero = cv2.findNonZero(gray_img)
        if non_zero is None:
            # Entirely black — return as-is
            return image

        x, y, w, h = cv2.boundingRect(non_zero)
        ih, iw = img.shape[:2]

        # Apply padding, clamp to image
        x1 = max(0,  x - padding)
        y1 = max(0,  y - padding)
        x2 = min(iw, x + w + padding)
        y2 = min(ih, y + h + padding)

        cropped = img[y1:y2, x1:x2]
        result  = _wrap(cropped)
        result.update(meta)
        return result

    @r.register("BLEND")
    def cmd_blend(image, checkpoint_name: str, alpha: float = 0.5,
                  mode: str = "normal", _ctx: dict = None):
        """BLEND "checkpoint" [alpha] [mode]  — blend two images with a compositing mode.

        Unlike OVERLAY (which always uses linear alpha), BLEND supports multiple
        compositing modes for creative and analytical use.

        Modes:
            normal     — standard alpha composite (same as OVERLAY)
            difference — absolute pixel difference, amplified
            multiply   — pixel-by-pixel multiply (darkens)
            screen     — inverted multiply (brightens)
            add        — additive blend (clamps to 255)
            lighten    — max per-pixel value
            darken     — min per-pixel value

        Parameters:
            checkpoint_name — which CHECKPOINT to blend with
            alpha           — blend weight for the checkpoint layer (0–1)
            mode            — compositing mode (default: normal)

        Examples:
            BLEND "original" 0.5
            BLEND "original" 0.3 "difference"
            BLEND "mask" 1.0 "multiply"
        """
        _require_image("BLEND", image)
        if _ctx is None:
            raise RuntimeError("[BLEND] Requires executor context.")

        checkpoints = _ctx.get("checkpoints", {})
        if checkpoint_name not in checkpoints:
            available = list(checkpoints.keys()) or ["(none)"]
            raise RuntimeError(
                f"[BLEND] No checkpoint '{checkpoint_name}'. "
                f"Available: {', '.join(available)}"
            )

        current_img, current_meta = _unwrap(image)
        saved_img, _              = checkpoints[checkpoint_name]

        def to_bgr(arr):
            if len(arr.shape) == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return arr.copy()

        a = to_bgr(current_img).astype(np.float32) / 255.0
        b = to_bgr(saved_img)
        if b.shape[:2] != a.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        b = b.astype(np.float32) / 255.0
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
            result_f = np.maximum(a, b)
            result_f = al * result_f + (1.0 - al) * a
        elif mode == "darken":
            result_f = np.minimum(a, b)
            result_f = al * result_f + (1.0 - al) * a
        else:
            valid = "normal, difference, multiply, screen, add, lighten, darken"
            raise ValueError(f"[BLEND] Unknown mode '{mode}'. Valid: {valid}")

        out = np.clip(result_f * 255, 0, 255).astype(np.uint8)
        result = _wrap(out)
        result.update(current_meta)
        return result
