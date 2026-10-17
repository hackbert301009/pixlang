# Changelog

All notable changes to PixLang are documented here.
Format: [Semantic Versioning](https://semver.org)

## [Unreleased] — 0.5.1

### Added
- **Editor step debugger.** Breakpoints (gutter click / `F9`, optional condition as a PixLang expression), Continue / Step Over / Step Into / Step Out / Pause / Stop (`F8`, `F10`, `F11`, `Shift+F11`, `F6`, `Shift+F5`; start with `Ctrl/Cmd+Shift+D`). A Debug side view shows the call stack, typed variables, user Watch expressions (read-only evaluation: assignments, commands and FUNC calls are rejected), buffers with change-aware thumbnails and breakpoints; the current line, preview image, flow diagram, Output and Timing follow each pause. Sessions run under the normal sandbox, count only running time towards the run timeout and are stopped after 10 idle minutes.
- Editor API: `POST /api/debug/start`, `GET /api/debug/<id>/state?wait=<seq>&timeout=<s>` (long-poll), `POST /api/debug/<id>/{control,breakpoints,watch,evaluate}`, `POST /api/debug/stop`. The final state carries the same payload as `/api/run`.

### Changed
- `pixlang.editor.server.app` is now created on first access instead of at import time.

## [0.5.0] — Hardware-near language, linker & new editor

### Language
- Expressions (`+ - * / // % & | ^ ~ << >>`, comparisons, `and`/`or`/`not`) in `SET`, `IF`, `WHILE`, `FOR`, `ASSERT`, `RETURN`; parenthesised expressions in command arguments
- `IF/ELIF/ELSE`, `WHILE`, `FOR … TO … STEP`, `REPEAT n AS i`, `BREAK`, `CONTINUE`
- `FUNC`/`RETURN`/`GLOBAL` — command-style (ALL_CAPS) and expression functions, call-stack traces
- **Linker:** `IMPORT`/`FROM … IMPORT`/`EXPORT`, `pixlang link [--map]`; reports cycles, duplicate exports, undefined symbols, unused imports
- **Typed buffers & pointers:** `ALLOC`, `BUF`, `USE`, `FREE`, pixel PEEK/POKE, `&`/`*` live views, `WITH`, `AS`/`CAST`/`SIZEOF`, hardware-style integer wraparound, use-after-free detection
- Compiler-style diagnostics with `file:line:col` and a caret; `PixSyntaxError`, `PixRuntimeError`, `PixAssertionError`, `PixLinkError`
- Newlines now terminate statements (use `\` to continue a command)

### Commands
- 36 new commands: `CONVERT_DEPTH`, `BITWISE_*`, `SHIFT_*`, `BIT_PLANE`, `MASK`, `GAMMA`, `POSTERIZE`, `LUT`, `DITHER`, `EXTRACT_CHANNEL`, `MERGE_CHANNELS`, `SWAP_RB`, `DROP_ALPHA`, `ADD_ALPHA`, `SAVE_RAW`, `LOAD_RAW`, `EXPORT_C_ARRAY`, `BYTESWAP`, `STATS`, `CVT_COLOR`, `IN_RANGE`, `CLAHE`, `FILTER2D`, `SOBEL`, `LAPLACIAN`, `MORPH`, `PAD`, `ROTATE90`, `WARP_PERSPECTIVE`, `CONNECTED_COMPONENTS`, `ABSDIFF`, …
- `THRESHOLD` modes, `ROTATE … expand`; arguments are type-coerced and validated with usage hints and typo suggestions
- uint16/float32/BGRA images supported; metadata (contours, …) survives every built-in
- `pixlang batch`, `--var k=v`, `pixlang commands --category`

### Editor
- Complete redesign: dark/light theme, resizable panes, command palette, autocomplete, hover docs, lint gutter, Problems/Output/Timing panels, compare slider, nested flow diagram
- Hardened server: token + Origin/Host checks, strict JSON, workspace confinement, run timeout, CSP

### Changed
- **Local (`*.plugins.py`) and directory plugins are opt-in:** `--trust-plugins` or `[plugins] trust_local = true`
- Errors go to stderr; colour/banner respect `NO_COLOR` and non-TTY output
- `LOAD`/`SAVE` etc. unchanged; `INCLUDE` resolves relative to the including file

### Fixed
- Parser hang on stray `END`/`ENDIF`/`ROI_RESET`; `INCLUDE` cycles and nested relative paths; CRLF sources; negative/hex/exponent numbers; string escapes; `$ITER` clobbering in nested loops
- Batch mode processed only the first file N times; batch dry run executed the pipeline
- `HISTOGRAM_SAVE` on OpenCV 5; watch mode re-running forever after an error
- Many linter false positives (ROI/INCLUDE bodies, PX001/PX005); ignored `lint.ignore` config
- Editor CSRF/DNS-rebinding exposure and `INCLUDE` file disclosure

---

## [0.4.0] — Pipeline Composition & Validation

### Added
- `INCLUDE "other.pxl"` — inline sub-pipelines with shared executor context
- `ASSERT subject op value ["message"]` — runtime image validation (width, height, channels, min, max, contour_count)
- `ROI x y w h … ROI_RESET` — process a masked region, auto-paste result back
- `LOAD_GLOB "pattern"` — batch multi-file input with glob expansion
- `SAVE_EACH "template"` — batch output with `{stem}`, `{index}`, `{index1}`, `{ext}`, `{dir}` templates
- `AUTO_CROP [padding]` — crop to bounding box of non-zero content
- `COMPARE ["checkpoint"]` — compute PSNR + MAD vs saved reference; stores diff
- `BLEND "checkpoint" [alpha] [mode]` — 7 compositing modes: normal, difference, multiply, screen, add, lighten, darken
- `pixlang.toml` — project-level config (variables, lint ignore, defaults)
- `BatchRunner` — automatic multi-file execution when `LOAD_GLOB` is present
- `pixlang run --batch` flag for explicit batch mode

### Changed
- CLI version → 0.4.0
- `pixlang run` auto-detects batch mode from pipeline AST

---

## [0.3.0] — Language Control Flow

### Added
- `SET name value` — named variables
- `$name` — variable references in command arguments
- `IF var op value … ENDIF` — conditional execution
- `REPEAT n … END` — loop with `$ITER` / `$ITER1` built-ins
- `RESIZE_PERCENT pct` — proportional resize
- `HISTOGRAM_SAVE ["path"]` — save intensity histogram as PNG
- `PIPELINE_STATS` — print per-command timing table
- `pixlang lint` — 11-rule static analyser (PX001–PX011)
- `pixlang watch` — file-change polling with auto-rerun
- `pixlang new <name>` — project scaffold generator
- Linter integrated into watch mode

---

## [0.2.0] — Plugin Architecture

### Added
- Three-tier plugin discovery: entry-points, local `.plugins.py`, directory
- `CommandInfo` with source tracking on every registration
- `ConflictError` on duplicate command names
- `allow_override=True` escape hatch
- `HEATMAP [colormap]` — false-color intensity visualization (8 colormaps)
- `DRAW_TEXT "label" x y …` — full text annotation (7 fonts, RGB color)
- `CHECKPOINT ["name"]` / `OVERLAY ["name"] alpha` — named snapshots + alpha composite
- `pixlang plugins` — list discovered plugins
- `pixlang commands --source` — filter by origin
- `--no-plugins` run flag
- Example plugin: `pixlang-denoise` (NLM_DENOISE, BILATERAL)

---

## [0.1.0] — Initial MVP

### Added
- Line-based DSL: `LOAD`, `SAVE`, `RESIZE`, `GRAYSCALE`, `THRESHOLD`, `THRESHOLD_OTSU`, `ADAPTIVE_THRESHOLD`, `BLUR`, `MEDIAN_BLUR`, `SHARPEN`, `CANNY`, `DILATE`, `ERODE`, `NORMALIZE`, `EQUALIZE_HIST`, `INVERT`, `CROP`, `ROTATE`, `FLIP`, `FIND_CONTOURS`, `DRAW_BOUNDING_BOXES`, `DRAW_CONTOURS`, `PRINT_INFO`
- Recursive tokenizer + AST-based parser
- Command registry with `@register` decorator
- OpenCV-backed execution engine
- `pixlang run`, `pixlang validate`, `pixlang commands`
- 24 tests (parser + executor)
