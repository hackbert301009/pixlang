# Changelog

All notable changes to PixLang are documented here.
Format: [Semantic Versioning](https://semver.org)

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
