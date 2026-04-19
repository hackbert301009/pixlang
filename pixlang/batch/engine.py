# pixlang/batch/engine.py
"""
PixLang Batch Processing  v0.4
═══════════════════════════════

Two new DSL commands that work as a pair:

    LOAD_GLOB "frames/*.jpg"
    <any processing commands>
    SAVE_EACH "output/{stem}_result.png"

LOAD_GLOB:
  - Expands the glob pattern relative to the pipeline file
  - Stores the matched file list in the executor context
  - Loads the FIRST file as the current image (so normal commands work)
  - The executor's batch loop drives the rest

SAVE_EACH:
  - Uses Python str.format()-style templates with these variables:
      {name}    original filename without extension  (e.g. "frame_001")
      {stem}    alias for {name}
      {ext}     original extension                   (e.g. ".jpg")
      {index}   zero-based batch index               (e.g. "0", "1", …)
      {index1}  one-based batch index                (e.g. "1", "2", …)
      {dir}     original file's parent directory

How the batch loop works:
  The Executor detects LOAD_GLOB in the pipeline and switches to batch mode.
  It re-runs the entire pipeline body for each matched file, injecting the
  current file's image and updating context vars {name}, {index}, etc.
  This keeps the DSL declarative — users don't need explicit loops.

Design decision: LOAD_GLOB / SAVE_EACH are registered as normal commands BUT
the Executor pre-scans the AST for LOAD_GLOB to decide batch mode. This avoids
any special parser machinery.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pixlang.commands.registry import CommandRegistry


def register_batch_commands(registry: "CommandRegistry") -> None:
    """Register LOAD_GLOB and SAVE_EACH into the given registry."""

    @registry.register("LOAD_GLOB")
    def cmd_load_glob(image, pattern: str, _ctx: dict = None):
        """LOAD_GLOB "frames/*.jpg"  — load all files matching a glob pattern.

        Expands the pattern and stores the matched file list in the pipeline
        context. The first matched file is loaded as the current image.

        In the context, sets:
            batch_files   : list of matched Path objects
            batch_index   : current index (0-based)
            batch_current : Path of the current file
            name, stem    : filename without extension
            ext           : file extension
            dir           : parent directory

        Examples:
            LOAD_GLOB "images/*.jpg"
            LOAD_GLOB "frames/frame_*.png"
            LOAD_GLOB "/data/scans/**/*.tiff"
        """
        if _ctx is None:
            raise RuntimeError("[LOAD_GLOB] Requires executor context.")

        # Resolve pattern relative to pipeline directory if set
        base_dir = _ctx.get("pipeline_dir", Path("."))
        resolved = Path(base_dir) / pattern if not Path(pattern).is_absolute() else Path(pattern)

        matched = sorted(Path(p) for p in glob.glob(str(resolved), recursive=True))
        if not matched:
            raise FileNotFoundError(
                f"[LOAD_GLOB] No files matched pattern: '{pattern}'\n"
                f"  Searched from: {base_dir}"
            )

        _ctx["batch_files"]   = matched
        _ctx["batch_index"]   = 0
        _ctx["batch_current"] = matched[0]
        _update_batch_ctx(_ctx, matched[0], 0)

        loaded = cv2.imread(str(matched[0]), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise IOError(f"[LOAD_GLOB] Cannot read: {matched[0]}")

        return loaded

    @registry.register("SAVE_EACH")
    def cmd_save_each(image, template: str, _ctx: dict = None):
        """SAVE_EACH "output/{stem}_result.png"  — save each batch image with template name.

        Template variables:
            {name} / {stem}  — filename without extension
            {ext}            — original file extension (e.g. ".jpg")
            {index}          — zero-based batch index
            {index1}         — one-based batch index
            {dir}            — original file's parent directory

        The output directory is created automatically if it does not exist.

        Examples:
            SAVE_EACH "output/{stem}.png"
            SAVE_EACH "results/{name}_processed{ext}"
            SAVE_EACH "batch_{index1:03d}_{stem}.png"
        """
        from pixlang.commands.builtin import _require_image, _unwrap

        _require_image("SAVE_EACH", image)
        if _ctx is None:
            raise RuntimeError("[SAVE_EACH] Requires executor context.")

        img, _ = _unwrap(image)

        # Build template context
        current = _ctx.get("batch_current", Path("unknown.jpg"))
        idx     = _ctx.get("batch_index",  0)
        fmt_ctx = {
            "name":   current.stem,
            "stem":   current.stem,
            "ext":    current.suffix,
            "index":  str(idx),
            "index1": str(idx + 1),
            "dir":    str(current.parent),
        }
        # Support zero-padded numeric formats like {index1:03d}
        fmt_ctx_typed = {
            "name":   current.stem,
            "stem":   current.stem,
            "ext":    current.suffix,
            "index":  idx,
            "index1": idx + 1,
            "dir":    str(current.parent),
        }

        try:
            out_path = Path(template.format(**fmt_ctx_typed))
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"[SAVE_EACH] Template error in '{template}': {e}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            raise IOError(f"[SAVE_EACH] Failed to write: {out_path}")

        if _ctx.get("verbose"):
            print(f"       → {out_path}")

        return image


def _update_batch_ctx(ctx: dict, file_path: Path, index: int) -> None:
    """Sync batch context variables for the current file."""
    ctx["batch_current"] = file_path
    ctx["batch_index"]   = index
    ctx["vars"]["name"]   = file_path.stem
    ctx["vars"]["stem"]   = file_path.stem
    ctx["vars"]["ext"]    = file_path.suffix
    ctx["vars"]["index"]  = index
    ctx["vars"]["index1"] = index + 1
    ctx["vars"]["dir"]    = str(file_path.parent)


# ── BatchExecutor ─────────────────────────────────────────────────────────────

class BatchRunner:
    """
    Wraps an Executor to run a pipeline repeatedly across a glob-matched set
    of images.

    Usage (internal, called from CLI):
        runner = BatchRunner(pipeline, registry, verbose=True)
        runner.run(pipeline_path)
    """

    def __init__(self, pipeline, registry, verbose=False, plugin_loader=None,
                 pipeline_path=None):
        self.pipeline       = pipeline
        self.registry       = registry
        self.verbose        = verbose
        self.plugin_loader  = plugin_loader
        self.pipeline_path  = pipeline_path

    def run(self) -> dict:
        """
        Execute the batch pipeline.

        If the pipeline contains LOAD_GLOB, runs once per matched file.
        Otherwise falls back to a single normal run.

        Returns a summary dict: {total, ok, failed, files}.
        """
        from pixlang.executor import Executor
        from pixlang.commands.builtin import _unwrap
        from pixlang.parser.ast_nodes import Command

        # Quick scan: does this pipeline use LOAD_GLOB?
        all_commands = _flatten_commands(self.pipeline.commands)
        glob_cmds = [c for c in all_commands if c.name == "LOAD_GLOB"]

        if not glob_cmds:
            # Plain pipeline — single run
            ex = Executor(
                registry=self.registry, verbose=self.verbose,
                plugin_loader=self.plugin_loader,
                pipeline_path=self.pipeline_path,
            )
            ex.run(self.pipeline)
            return {"total": 1, "ok": 1, "failed": 0, "files": []}

        # Batch mode: first run to discover files
        ex = Executor(
            registry=self.registry, verbose=False,
            pipeline_path=self.pipeline_path,
        )
        ex.context["pipeline_dir"] = (
            self.pipeline_path.parent if self.pipeline_path else Path(".")
        )

        # Dry-run just to trigger LOAD_GLOB and populate batch_files
        try:
            ex.run(self.pipeline)
        except Exception:
            pass  # SAVE_EACH may fail on first run if no prior image — that's ok

        batch_files = ex.context.get("batch_files", [])
        if not batch_files:
            raise RuntimeError("[Batch] LOAD_GLOB found no files.")

        results = {"total": len(batch_files), "ok": 0, "failed": 0, "files": []}

        print(f"\n  Batch mode: {len(batch_files)} file(s)\n")

        for idx, file_path in enumerate(batch_files):
            label = f"[{idx+1}/{len(batch_files)}] {file_path.name}"
            try:
                batch_ex = Executor(
                    registry=self.registry,
                    verbose=self.verbose,
                    pipeline_path=self.pipeline_path,
                )
                batch_ex.context["pipeline_dir"] = (
                    self.pipeline_path.parent if self.pipeline_path else Path(".")
                )
                # Pre-inject batch variables so LOAD_GLOB loads the right file
                _update_batch_ctx(batch_ex.context, file_path, idx)
                batch_ex.context["batch_files"]   = batch_files
                batch_ex.context["batch_override"] = file_path  # executor reads this

                batch_ex.run(self.pipeline)
                results["ok"]    += 1
                results["files"].append({"path": str(file_path), "ok": True})
                print(f"  \033[92m✓\033[0m  {label}")
            except Exception as e:
                results["failed"] += 1
                results["files"].append({"path": str(file_path), "ok": False, "error": str(e)})
                print(f"  \033[91m✗\033[0m  {label}  → {e}")

        print(f"\n  Done: {results['ok']} ok, {results['failed']} failed\n")
        return results


def _flatten_commands(stmts) -> list:
    from pixlang.parser.ast_nodes import Command, IfBlock, RepeatBlock, RoiBlock
    out = []
    for s in stmts:
        if isinstance(s, Command):
            out.append(s)
        elif isinstance(s, (IfBlock, RepeatBlock, RoiBlock)):
            out.extend(_flatten_commands(s.body))
    return out
