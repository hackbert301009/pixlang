# PixLang — GitHub Issues (Backlog)

---

## #1 — [Parser] Support inline variable assignment
**Priority:** High
**Label:** `enhancement` `parser`

**Description:**
Allow users to declare named values and reuse them across commands:
```
SET width 640
SET height 480
RESIZE width height
BLUR kernel_size
```
Requires the lexer to recognise lowercase identifiers, a symbol table in the executor, and argument resolution before dispatch.

---

## #2 — [Parser] Multi-line command continuation with `\`
**Priority:** Medium
**Label:** `enhancement` `parser`

**Description:**
Long argument lists (e.g. future `DRAW_TEXT`) can get unwieldy on one line. Support backslash continuation:
```
DRAW_TEXT \
  "Defect detected" \
  x=10 y=10 \
  font_scale=1.2
```

---

## #3 — [Commands] Add `INFER` command for ONNX model inference
**Priority:** High
**Label:** `enhancement` `ml`

**Description:**
Integrate `onnxruntime` to run arbitrary vision models inline:
```
LOAD "frame.jpg"
RESIZE 416 416
INFER "yolov8n.onnx"
DRAW_BOUNDING_BOXES
SAVE "detected.jpg"
```
The command should accept an ONNX file path, run inference, and store detections as pipeline metadata (class, confidence, bbox).

---

## #4 — [Commands] Add `ROI` (Region of Interest) masking
**Priority:** High
**Label:** `enhancement` `commands`

**Description:**
Allow restricting all subsequent analysis to a sub-region:
```
ROI 100 100 400 300   # x y w h
THRESHOLD 128
FIND_CONTOURS
```
`ROI` should crop the active region, and `ROI_RESET` should restore the full frame. Critical for industrial inspection use cases.

---

## #5 — [Performance] Lazy image loading & memory profiling
**Priority:** Medium
**Label:** `performance`

**Description:**
Current `LOAD` reads the full image into RAM immediately. For large batches or high-res images, add:
- Lazy loading: defer decode until first processing command
- Memory usage reporting in `--verbose` mode
- Option to process in tiles for very large images (>50 MP)

---

## #6 — [CLI] Add `pixlang watch` for live pipeline re-execution
**Priority:** Medium
**Label:** `dx` `cli`

**Description:**
Watch a `.pxl` file and its input image(s) for changes and re-run automatically:
```bash
pixlang watch pipeline.pxl --output preview.png
```
Useful during development. Should debounce file events and report execution time diff vs previous run.

---

## #7 — [Testing] Add property-based tests with Hypothesis
**Priority:** Medium
**Label:** `testing`

**Description:**
Current tests use fixed synthetic images. Add Hypothesis-based fuzzing for:
- Arbitrary image shapes (1×1 to 8192×8192)
- Kernel sizes (including edge cases: 0, 1, even numbers)
- Random pixel value ranges
- Chained command sequences to test state threading invariants

---

## #8 — [Commands] Batch processing with `LOAD_GLOB`
**Priority:** Medium
**Label:** `enhancement` `commands`

**Description:**
Process multiple images with a single pipeline:
```
LOAD_GLOB "frames/*.jpg"
RESIZE 640 480
GRAYSCALE
THRESHOLD_OTSU
SAVE_EACH "output/{name}_processed.png"
```
`LOAD_GLOB` iterates over matched files; `SAVE_EACH` uses the source filename as a template variable.

---

## #9 — [DX] VS Code syntax highlighting extension
**Priority:** Low
**Label:** `tooling` `dx`

**Description:**
Create a minimal VS Code language extension for `.pxl` files providing:
- Keyword highlighting for all registered commands
- String and number literal coloring
- Comment support (`#`)
- Hover docs showing command signature and description
- Auto-completion of built-in command names

Publish to the VS Code marketplace as `pixlang-vscode`.

---

## #10 — [Commands] Add `DRAW_TEXT` annotation command
**Priority:** Low
**Label:** `enhancement` `commands`

**Description:**
Annotate images with text overlays:
```
DRAW_TEXT "Pass" 20 30
DRAW_TEXT "Defect count: 3" 20 60 scale=0.8 color=255,0,0
```
Uses `cv2.putText`. Arguments: text string, x, y, optional scale and BGR color tuple.

---

## #11 — [Architecture] Plugin system for third-party command packages
**Priority:** Medium
**Label:** `architecture` `enhancement`

**Description:**
Allow external Python packages to register PixLang commands via a setuptools entry point:
```toml
# third-party plugin's pyproject.toml
[project.entry-points."pixlang.commands"]
my_plugin = "my_package.pixlang_commands:register"
```
The executor discovers and loads registered plugins at startup. Enables the community to publish command packs (e.g. `pixlang-barcode`, `pixlang-depth`).

---

## #12 — [Web UI] Browser-based pipeline editor
**Priority:** Low
**Label:** `enhancement` `ui`

**Description:**
Build a local web UI (FastAPI backend + React frontend) for interactive pipeline editing:
- Left panel: `.pxl` text editor with syntax highlighting
- Right panel: live image preview updating on each keystroke
- Bottom: per-command timing bar chart
- Export button: download the final image or the `.pxl` file
- Drag-and-drop image upload as `LOAD` source

Start as `pixlang serve` subcommand.
