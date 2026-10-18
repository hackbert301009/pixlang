# PixLang

<p align="center">
  <img src="docs/logo.svg" alt="PixLang logo" width="480"/>
</p>

**A minimal, human-readable DSL for computer vision pipelines.**

[![CI](https://github.com/hackbert301009/pixlang/actions/workflows/ci.yml/badge.svg)](https://github.com/hackbert301009/pixlang/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-799%20passing-brightgreen)]()

---

## What is PixLang?

PixLang is a line-based domain-specific language for describing image processing pipelines. Write readable, reproducible, versionable vision workflows — no Python boilerplate required.

```
# inspection.pxl
SET width  640
SET height 480

LOAD "part_photo.jpg"
RESIZE $width $height
ASSERT width == 640

CHECKPOINT "original"
INCLUDE "preprocess.pxl"

FIND_CONTOURS
DRAW_BOUNDING_BOXES
OVERLAY "original" 0.3

DRAW_TEXT "INSPECTION v1.2" 12 35 1.0 255 255 80 2 "duplex"
COMPARE "original"
SAVE "output/result.png"
```

---

## Features

| Category | Feature |
|---|---|
| **Language** | Expressions (`+ - * / & \| ^ << >>`), `IF/ELIF/ELSE`, `WHILE`, `FOR`, `REPEAT`, `FUNC`/`RETURN`, typed **buffers & pointers** (`ALLOC`, `&buf[..]`, `*p`), **modules + linker** (`IMPORT`/`EXPORT`) — see [docs/LANGUAGE.md](docs/LANGUAGE.md) |
| **Editor** | Browser IDE: autocomplete, hover docs, live lint, compare slider, flow diagram, dark/light theme |
| **Composition** | `INCLUDE` sub-pipelines with shared context |
| **Validation** | `ASSERT` runtime checks (width, height, channels, min, max, contour_count) |
| **ROI** | `ROI x y w h … ROI_RESET` — process masked regions, auto-paste back |
| **Batch** | `LOAD_GLOB` + `SAVE_EACH` — multi-file runs with template filenames |
| **Plugins** | Entry-points (auto), local `.plugins.py` / directory (opt-in: `--trust-plugins` or a confirmed workspace) |
| **Linter** | 12 rules, `pixlang lint`, integrated into watch mode |
| **Watch** | `pixlang watch` — file-change polling with auto-rerun |
| **Config** | `pixlang.toml` — project-level variables, defaults, lint overrides |
| **71 commands** | I/O · Geometry · Color · Threshold · Filter · Morphology · Analysis · Annotation · Depth · Bitwise · LUT · Channels · Raw/embedded |

---

## Installation

```bash
git clone https://github.com/hackbert301009/pixlang.git
cd pixlang
pip install -e .

# Generate a sample image and run:
python scripts/make_sample.py
pixlang run examples/pipeline.pxl --verbose
```

---

## CLI Reference

```bash
pixlang run      pipeline.pxl [--verbose] [--no-plugins] [--trust-plugins] [--var k=v]
pixlang batch    pipeline.pxl [--var k=v]
pixlang link     pipeline.pxl [--map]
pixlang lint     pipeline.pxl
pixlang watch    pipeline.pxl [--verbose] [--interval 0.5]
pixlang validate pipeline.pxl
pixlang commands [--source builtin] [--category Bitwise]
pixlang plugins
pixlang trust    [DIR] [--list] [--remove DIR]
pixlang new      <project-name>
pixlang editor   [--port 7478] [--workspace DIR] [--timeout 30] [--no-plugins]
pixlang --version
```

---

## Language Reference

### Variables
```
SET width  640
SET height 480
SET mode   "inspect"

LOAD "image.jpg"
RESIZE $width $height   # with $ prefix
RESIZE width height     # bare name also works
```

### Line Continuation

Long argument lists can be split across multiple lines using a trailing backslash:

```
DRAW_TEXT \
    "Defect detected" \
    12 35 \
    1.0 255 80 80 2

DRAW_TEXT "Defect detected" 12 35 1.0 255 80 80 2   # identical
```

Any command can use `\` at the end of a line to continue on the next.
Indentation after `\` is ignored.

### Conditionals
```
SET threshold 128

IF threshold > 100
    THRESHOLD_OTSU
ENDIF
```

Operators: `==` `!=` `<` `>` `<=` `>=`

### Loops
```
REPEAT 3
    ERODE 3 1
    DILATE 3 1
END
# $ITER = 0-based, $ITER1 = 1-based counter inside loop
```

### Expressions, Functions & Loops (v0.5)
```
SET w ($width / 2) + 8          # arithmetic, bit ops: & | ^ ~ << >>
IF $w > 100 and $mode == "fast"
    BLUR 3
ELIF $w > 50
    BLUR 5
ELSE
    BLUR 7
ENDIF

FOR i = 0 TO 4 STEP 2
    RESIZE ($w + $i) 240        # expressions in command args need ( )
END

FUNC clamp(v, lo, hi)
    IF $v < $lo
        RETURN $lo
    ENDIF
    RETURN min($v, $hi)
ENDFUNC
```

### Buffers & Pointers (v0.5)
```
ALLOC canvas 640 480 3 u8 0     # typed memory: u8 i8 u16 i16 i32 u32 f32 f64
LOAD "photo.png"
BUF orig                        # keep a copy of the current image
SET px orig[12, 40, 2]          # PEEK a pixel
SET orig[12, 40, 2] 255         # POKE a pixel
SET p &orig[10:110, 20:220]     # pointer = live view into the buffer
WITH *p
    BLUR 5                      # works in place on the region
END
SET mask (orig & 0xF0) | (orig >> 4)   # whole-buffer bit ops, wraparound like hardware
FREE orig                       # a stale pointer now raises "use-after-free"
```

### Modules & Linker (v0.5)
```
FROM "lib/filters.pxl" IMPORT SHARPEN_TWICE, clamp
IMPORT "lib/geometry.pxl" AS geo
SHARPEN_TWICE 3
geo.FIT 640 480
```
`pixlang link main.pxl --map` resolves every import across files and reports cycles, duplicate exports, undefined symbols and unused imports with `file:line:col`.

### Pipeline Composition
```
LOAD "image.jpg"
INCLUDE "preprocess.pxl"   # runs relative to this file's directory
FIND_CONTOURS
SAVE "output.png"
```

### Validation
```
ASSERT width  == 640  "Must resize to 640 first"
ASSERT height >= 100
ASSERT channels == 3
ASSERT contour_count >= 1  "No objects detected"
```

Subjects: `width` `height` `channels` `min` `max` `contour_count` `ndim`

### ROI (Region of Interest)
```
ROI 0 0 320 240
    GRAYSCALE
    THRESHOLD_OTSU
    FIND_CONTOURS
    DRAW_BOUNDING_BOXES
ROI_RESET
```

### Batch Processing
```
LOAD_GLOB "frames/*.jpg"
RESIZE 640 480
GRAYSCALE
THRESHOLD_OTSU
SAVE_EACH "output/{stem}_processed.png"
```

Template vars: `{stem}` `{name}` `{ext}` `{index}` `{index1}` `{dir}`

---

## Command Summary

<details>
<summary>View all commands (run <code>pixlang commands</code> for usage)</summary>

**I/O:** `LOAD` `SAVE` `PRINT_INFO` `LOAD_GLOB` `SAVE_EACH`

**Geometry:** `RESIZE` `RESIZE_PERCENT` `CROP` `ROTATE` `FLIP` `AUTO_CROP`

**Color:** `GRAYSCALE` `INVERT` `NORMALIZE` `EQUALIZE_HIST` `HEATMAP`

**Threshold:** `THRESHOLD` `THRESHOLD_OTSU` `ADAPTIVE_THRESHOLD`

**Filters:** `BLUR` `MEDIAN_BLUR` `SHARPEN` `CANNY`

**Morphology:** `DILATE` `ERODE`

**Analysis:** `FIND_CONTOURS` `COMPARE` `HISTOGRAM_SAVE` `PIPELINE_STATS`

**Annotation:** `DRAW_BOUNDING_BOXES` `DRAW_CONTOURS` `DRAW_TEXT`

**Compositing:** `CHECKPOINT` `OVERLAY` `BLEND`

**Depth / Bitwise / LUT:** `CONVERT_DEPTH` `CONVERT_SCALE` `BITWISE_AND|OR|XOR|NOT` `SHIFT_LEFT|RIGHT` `BIT_PLANE` `MASK` `GAMMA` `POSTERIZE` `LUT` `DITHER`

**Channels:** `EXTRACT_CHANNEL` `MERGE_CHANNELS` `SWAP_RB` `DROP_ALPHA` `ADD_ALPHA` `CVT_COLOR` `IN_RANGE`

**Raw / embedded:** `SAVE_RAW` `LOAD_RAW` `EXPORT_C_ARRAY` `BYTESWAP` — e.g. `DITHER` + `EXPORT_C_ARRAY "logo.h" logo gray8` turns an image into a C header for microcontrollers

**More CV:** `STATS` `CLAHE` `FILTER2D` `SOBEL` `LAPLACIAN` `MORPH` `PAD` `ROTATE90` `WARP_PERSPECTIVE` `CONNECTED_COMPONENTS` `ABSDIFF`

</details>

---

## Plugin System

Three discovery mechanisms, tried in order:

1. **Entry-points** — any installed package declaring `[project.entry-points."pixlang.commands"]`
2. **Local file** — `pipeline.plugins.py` next to the `.pxl` file
3. **Directory** — `~/.pixlang/plugins/` or `$PIXLANG_PLUGIN_DIR`

> **Security:** local and directory plugins run arbitrary Python, so they load only with
> `pixlang run x.pxl --trust-plugins`, or with `[plugins] trust_local = true` in
> `pixlang.toml` **in a directory you confirmed once** with `pixlang trust <dir>`.
> Both steps are needed: a downloaded bundle can ship its own `pixlang.toml`, so the
> setting alone would be the sender's to make. Pip-installed entry-point plugins load
> automatically — installing a package is already an act of trust.
>
> ```bash
> pixlang trust .              # confirm this project (covers subdirectories)
> pixlang trust --list         # show confirmed workspaces
> pixlang trust --remove DIR   # withdraw
> ```
>
> Confirmations live in `~/.pixlang/trusted.toml` (`$PIXLANG_TRUST_FILE` overrides).
> `pixlang new` confirms the project it scaffolds, since you created it yourself.

**Writing a plugin:**
```python
# my_plugin.py
def register(registry):
    @registry.register("MY_COMMAND", source="my-plugin")
    def cmd_my(image, param: int = 10):
        """MY_COMMAND [param] — does something."""
        from pixlang.commands.builtin import _unwrap, _wrap
        img, meta = _unwrap(image)
        # ... process img ...
        result = _wrap(img)
        result.update(meta)
        return result
```

Drop it in `~/.pixlang/plugins/` and run with `--trust-plugins`, or run it from a project
whose confirmed `pixlang.toml` sets `trust_local = true`.

---

## Project Config (`pixlang.toml`)

```toml
[defaults]
verbose = false
plugins = true
lint    = true

[variables]
width  = 640
height = 480
mode   = "production"

[lint]
ignore = ["PX006"]

[batch]
output_dir = "output"
```

PixLang searches for `pixlang.toml` in the pipeline directory and parent directories.

---

## Linter Rules

| Code | Severity | Description |
|---|---|---|
| PX001 | ERROR | Pipeline does not start with LOAD |
| PX002 | WARNING | No SAVE — output never written |
| PX003 | WARNING | FIND_CONTOURS without a draw command |
| PX004 | ERROR | OVERLAY references undefined CHECKPOINT |
| PX005 | ERROR | `$variable` used but never SET |
| PX006 | INFO | Identical consecutive commands |
| PX007 | WARNING | RESIZE after filtering step |
| PX008 | ERROR | REPEAT with count ≤ 0 |
| PX009 | INFO | Empty pipeline |
| PX010 | WARNING | THRESHOLD before GRAYSCALE |
| PX011 | ERROR | Unknown command name |
| PX012 | WARNING | Bare word looks like a typo of a defined variable |

Rule crashes are reported as `PX000`; ignore rules with `[lint] ignore = [...]`.

---

## Architecture

```
pixlang/
├── parser/         lexer → Pratt expression parser → AST, caret diagnostics
├── linker/         object units, symbol resolution, IMPORT/INCLUDE linking
├── commands/       registry (arity, type coercion, usage) + built-in OpenCV handlers
├── executor/       linked-program runner: frames, buffers/pointers, loops, functions
├── plugins/        three-tier discovery loader
├── linter/         11-rule static analyser
├── watcher/        polling file-change detector
├── batch/          LOAD_GLOB / SAVE_EACH / BatchRunner
└── config/         pixlang.toml discovery + parser
```

---

## Visual Editor

`pixlang editor` opens a local, browser-based IDE for the current directory (the *workspace*).

```bash
pip install pixlang[editor]
pixlang editor                      # http://localhost:7478/
pixlang editor --workspace ./proj   # confine file access to a directory
pixlang editor --timeout 60 --no-plugins
```

- **Editor:** syntax highlighting, autocomplete (commands with argument snippets, keywords, `$variables`, buffers, `FUNC` names), hover docs, live lint markers and search (CodeMirror 6).
- **Running:** `Ctrl/Cmd+Enter` or *Auto* re-run. The preview supports zoom/pan, a before/after compare slider and a strip of every `CHECKPOINT`/buffer.
- **Feedback:** Problems panel (lint + runtime errors with line/col, click to jump), Output and per-command Timing.
- **Flow:** live flow diagram with nested `IF/ELIF/ELSE`, loops, `ROI`/`WITH` and `FUNC` blocks and per-step timing.
- **Debugger:** click the gutter (or `F9`) to set breakpoints — right-click for a conditional one — then press *Debug* (`Ctrl/Cmd+Shift+D`). Step with `F10` over / `F11` into / `Shift+F11` out, continue with `F8`, pause with `F6`, stop with `Shift+F5`. A **Debug** side view shows the call stack, variables (typed, expandable), your **Watch** expressions (read-only PixLang expressions evaluated in the paused frame; they persist per file), buffers with thumbnails and the breakpoint list, while the preview, flow diagram, Output and Timing follow the current line. The editor is read-only during a session, run time is counted only while the program is actually running, and an abandoned paused session is stopped after 10 minutes.
- **Files:** open/save `.pxl` files in the workspace (`Ctrl/Cmd+S`), bundled examples, `Ctrl/Cmd+K` command palette, dark and light theme.

Relative `LOAD`/`SAVE`/`INCLUDE` paths resolve against the opened file's folder, like `pixlang run`.
The server listens on 127.0.0.1 only; API calls need a per-session token, foreign origins are rejected,
file access is confined to the workspace and runs time out. Front-end dependencies are vendored —
to rebuild them see `pixlang/editor/static/vendor/BUILD.md`.

---

## Testing

```bash
pytest                      # 903 tests
pytest --cov=pixlang        # with coverage
pytest tests/test_editor.py tests/test_editor_v5.py tests/test_editor_debug.py  # editor + debugger API
pytest tests/test_v4.py     # v0.4 only
```

---

## Roadmap

- [x] v0.5 — expressions, `FUNC`, `WHILE`/`FOR`, typed buffers & pointers, linker, hardware-near commands, editor redesign
- [x] v0.5.1 — editor step debugger: breakpoints, step over/into/out, watch expressions
- [ ] v0.6 — IR/bytecode stage with `pixlang dis`
- [ ] v0.6 — ONNX model inference (`INFER model.onnx`)
- [ ] v0.7 — compile cache for linked modules, tiling for very large images
- [ ] v1.0 — Language spec, stable API, PyPI release

---

## License

MIT — see [LICENSE](LICENSE).
