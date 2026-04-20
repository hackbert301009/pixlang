# pixlang/executor/engine.py
"""
PixLang Execution Engine  v0.3

New capabilities:
  - Variable store: SET / $ref resolution in args
  - IF <var> <op> <value> ... ENDIF  — conditional execution
  - REPEAT <n> ... END               — loop with iteration counter ($ITER)
  - Plugin loading unchanged from v0.2
"""
from __future__ import annotations

import inspect
import operator
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pixlang.commands.builtin import _unwrap
from pixlang.commands.registry import CommandRegistry
from pixlang.parser.ast_nodes import (
    Command, IfBlock, Pipeline, RepeatBlock, SetVar, Statement, VarRef, BareIdent,
    IncludeStmt, AssertStmt, RoiBlock,
)

if TYPE_CHECKING:
    from pixlang.plugins.loader import PluginLoader


# Comparison operators supported by IF
_OPS: Dict[str, Any] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<":  operator.lt,
    ">":  operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
}


class Executor:
    def __init__(
        self,
        registry: CommandRegistry,
        verbose: bool = False,
        plugin_loader: Optional["PluginLoader"] = None,
        pipeline_path: Optional[Path] = None,
    ):
        self.registry      = registry
        self.verbose       = verbose
        self.plugin_loader = plugin_loader
        self.pipeline_path = pipeline_path

        # Shared mutable state threaded through the entire pipeline
        self.context: Dict[str, Any] = {
            "checkpoints": {},   # used by CHECKPOINT / OVERLAY
            "vars":        {},   # used by SET / $ref / IF / REPEAT
            "stats":       {     # used by PIPELINE_STATS
                "commands_executed": 0,
                "total_ms":          0.0,
                "command_times":     {},
            },
        }

    # ── Plugin loading ────────────────────────────────────────────────────────

    def _load_plugins(self) -> None:
        if self.plugin_loader is None:
            return
        loader = self.plugin_loader
        loader.load_entrypoints()
        if self.pipeline_path:
            loader.load_local(self.pipeline_path)
        loader.load_directory()
        if self.verbose and loader.manifests:
            print("  Plugins:")
            print(loader.summary())
            print()

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, pipeline: Pipeline) -> None:
        self._load_plugins()
        image = None
        total_start = time.perf_counter()

        image = self._run_body(pipeline.commands, image)

        total_ms = (time.perf_counter() - total_start) * 1000
        self.context["stats"]["total_ms"] = total_ms
        if self.verbose:
            print(f"\n  Pipeline finished in {total_ms:.1f} ms  "
                  f"({self.context['stats']['commands_executed']} commands)")

    # ── Statement dispatcher ─────────────────────────────────────────────────

    def _run_body(self, stmts: List[Statement], image: Any) -> Any:
        for stmt in stmts:
            if isinstance(stmt, SetVar):
                image = self._exec_set(stmt, image)
            elif isinstance(stmt, IfBlock):
                image = self._exec_if(stmt, image)
            elif isinstance(stmt, RepeatBlock):
                image = self._exec_repeat(stmt, image)
            elif isinstance(stmt, IncludeStmt):
                image = self._exec_include(stmt, image)
            elif isinstance(stmt, AssertStmt):
                image = self._exec_assert(stmt, image)
            elif isinstance(stmt, RoiBlock):
                image = self._exec_roi(stmt, image)
            elif isinstance(stmt, Command):
                image = self._exec_command(stmt, image)
        return image

    # ── SET ───────────────────────────────────────────────────────────────────

    def _exec_set(self, stmt: SetVar, image: Any) -> Any:
        value = self._resolve(stmt.value)
        self.context["vars"][stmt.var_name] = value
        if self.verbose:
            print(f"  {'≡':2} SET {stmt.var_name:<22} = {value!r}")
        return image

    # ── IF ────────────────────────────────────────────────────────────────────

    def _exec_if(self, stmt: IfBlock, image: Any) -> Any:
        var_val = self.context["vars"].get(stmt.var_name)
        if var_val is None:
            raise RuntimeError(
                f"[IF] Variable '${stmt.var_name}' is not defined "
                f"(line {stmt.line}). Use SET first."
            )
        cmp_val  = self._resolve(stmt.cmp_value)
        op_fn    = _OPS.get(stmt.op)
        if op_fn is None:
            raise RuntimeError(f"[IF] Unknown operator '{stmt.op}'")

        try:
            condition = op_fn(_coerce_for_compare(var_val), _coerce_for_compare(cmp_val))
        except TypeError:
            condition = False

        if self.verbose:
            tick = "✓" if condition else "✗"
            print(f"  {tick}  IF ${stmt.var_name} {stmt.op} {cmp_val!r}  "
                  f"→ {'True — executing body' if condition else 'False — skipped'}")

        if condition:
            image = self._run_body(stmt.body, image)
        return image

    # ── REPEAT ────────────────────────────────────────────────────────────────

    def _exec_repeat(self, stmt: RepeatBlock, image: Any) -> Any:
        count = self._resolve(stmt.count)
        try:
            count = int(count)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"[REPEAT] Count must be an integer, got {count!r} "
                f"(line {stmt.line})"
            )
        if count < 0:
            raise RuntimeError(f"[REPEAT] Count must be ≥ 0, got {count}")

        if self.verbose:
            print(f"  ↻  REPEAT {count}")

        for i in range(count):
            self.context["vars"]["ITER"] = i        # $ITER available in body
            self.context["vars"]["ITER1"] = i + 1   # $ITER1 = 1-based
            image = self._run_body(stmt.body, image)

        return image

    # ── COMMAND ───────────────────────────────────────────────────────────────

    def _exec_command(self, cmd: Command, image: Any) -> Any:
        fn = self.registry.get(cmd.name)
        resolved_args = [self._resolve(a) for a in cmd.args]

        t0 = time.perf_counter()
        try:
            sig = inspect.signature(fn)
            if "_ctx" in sig.parameters:
                image = fn(image, *resolved_args, _ctx=self.context)
            else:
                image = fn(image, *resolved_args)
        except Exception as exc:
            raise RuntimeError(
                f"[PixLang] Error at line {cmd.line} "
                f"({cmd.name} {' '.join(str(a) for a in resolved_args)}): {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Update stats
        stats = self.context["stats"]
        stats["commands_executed"] += 1
        stats["total_ms"] += elapsed_ms
        prev = stats["command_times"].get(cmd.name, [])
        prev.append(elapsed_ms)
        stats["command_times"][cmd.name] = prev

        if self.verbose:
            img_raw, _ = _unwrap(image) if image is not None else (None, {})
            shape_str  = str(img_raw.shape) if img_raw is not None else "—"
            args_str   = " ".join(str(a) for a in resolved_args)
            print(
                f"  {'✓':2} {cmd.name:<24} {args_str:<20} "
                f"{shape_str:<18} {elapsed_ms:6.1f} ms"
            )
        return image

    # ── Variable resolution ───────────────────────────────────────────────────

    def _resolve(self, value: Any) -> Any:
        """Resolve a VarRef or BareIdent to its stored value; pass through everything else."""
        if isinstance(value, VarRef):
            name = value.var_name
            if name not in self.context["vars"]:
                raise RuntimeError(
                    f"[PixLang] Undefined variable '${name}'. "
                    f"Available: {', '.join('$'+k for k in self.context['vars'])}"
                )
            return self.context["vars"][name]
        if isinstance(value, BareIdent):
            # Variable lookup with string fallback — allows both:
            #   SET width 640 / RESIZE width height   (variable)
            #   HEATMAP jet                            (string literal)
            name = value.name
            if name in self.context["vars"]:
                return self.context["vars"][name]
            return name
        return value


    # ── INCLUDE ───────────────────────────────────────────────────────────────

    def _exec_include(self, stmt: IncludeStmt, image):
        """Inline-expand another .pxl file at runtime."""
        from pixlang.parser import parse as _parse

        # Resolve relative to the current pipeline file's directory
        base = self.pipeline_path.parent if self.pipeline_path else Path(".")
        inc_path = (base / stmt.path).resolve()

        if not inc_path.exists():
            raise RuntimeError(
                f"[INCLUDE] File not found: '{inc_path}' (line {stmt.line})"
            )
        if self.verbose:
            print(f"  ⤵  INCLUDE {stmt.path}")

        sub_source   = inc_path.read_text()
        sub_pipeline = _parse(sub_source)

        # Run the sub-pipeline using the same executor state (shared vars + ctx)
        image = self._run_body(sub_pipeline.commands, image)
        return image

    # ── ASSERT ────────────────────────────────────────────────────────────────

    _ASSERT_OPS = {
        "==": operator.eq, "!=": operator.ne,
        "<":  operator.lt, ">":  operator.gt,
        "<=": operator.le, ">=": operator.ge,
    }

    def _exec_assert(self, stmt: AssertStmt, image):
        """Validate image properties; raise AssertionError on failure."""
        from pixlang.commands.builtin import _unwrap
        import numpy as np

        if image is None:
            raise RuntimeError(f"[ASSERT] No image loaded (line {stmt.line})")

        img, meta = _unwrap(image)
        subject   = stmt.subject.lower()
        expected  = self._resolve(stmt.expected)

        # Measure the actual value
        if subject == "width":
            actual = img.shape[1]
        elif subject == "height":
            actual = img.shape[0]
        elif subject in ("channels", "depth"):
            actual = img.shape[2] if len(img.shape) == 3 else 1
        elif subject == "min":
            actual = int(img.min())
        elif subject == "max":
            actual = int(img.max())
        elif subject == "contour_count":
            contours = meta.get("contours", [])
            actual = len(contours)
        elif subject == "ndim":
            actual = img.ndim
        else:
            raise RuntimeError(
                f"[ASSERT] Unknown subject '{stmt.subject}'. "
                f"Valid: width, height, channels, min, max, contour_count, ndim"
            )

        op_fn = self._ASSERT_OPS.get(stmt.op)
        if op_fn is None:
            raise RuntimeError(f"[ASSERT] Unknown operator '{stmt.op}'")

        passed = op_fn(_coerce_for_compare(actual), _coerce_for_compare(expected))

        if self.verbose:
            tick = "✓" if passed else "✗"
            print(f"  {tick}  ASSERT {subject} {stmt.op} {expected}  "
                  f"(actual={actual})  {'OK' if passed else 'FAIL'}")

        if not passed:
            custom = f" — {stmt.message}" if stmt.message else ""
            raise AssertionError(
                f"[ASSERT] Pipeline assertion failed (line {stmt.line}): "
                f"{subject} {stmt.op} {expected}, but got {actual}{custom}"
            )
        return image

    # ── ROI ───────────────────────────────────────────────────────────────────

    def _exec_roi(self, stmt: RoiBlock, image):
        """Execute body commands on a cropped region, then paste result back."""
        from pixlang.commands.builtin import _unwrap, _wrap
        import numpy as np

        if image is None:
            raise RuntimeError(f"[ROI] No image loaded (line {stmt.line})")

        img, meta = _unwrap(image)

        x = int(self._resolve(stmt.x))
        y = int(self._resolve(stmt.y))
        w = int(self._resolve(stmt.w))
        h = int(self._resolve(stmt.h))

        # Clamp to image bounds
        ih, iw = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)

        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(
                f"[ROI] Region {x},{y},{w},{h} is outside image bounds "
                f"{iw}×{ih} (line {stmt.line})"
            )

        if self.verbose:
            print(f"  ⌗  ROI {x1},{y1} → {x2},{y2} ({x2-x1}×{y2-y1})")

        # Extract region, run body, paste result back
        region = img[y1:y2, x1:x2].copy()
        roi_image = _wrap(region)
        roi_image.update(meta)

        roi_result = self._run_body(stmt.body, roi_image)
        processed_region, processed_meta = _unwrap(roi_result)

        # Paste back — handle shape mismatches (e.g. gray→BGR inside ROI)
        canvas = img.copy()
        if len(canvas.shape) == 2 and len(processed_region.shape) == 3:
            import cv2
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        elif len(canvas.shape) == 3 and len(processed_region.shape) == 2:
            import cv2
            processed_region = cv2.cvtColor(processed_region, cv2.COLOR_GRAY2BGR)

        # Resize processed region back if body changed its size
        if processed_region.shape[:2] != (y2-y1, x2-x1):
            import cv2
            processed_region = cv2.resize(
                processed_region, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR
            )

        canvas[y1:y2, x1:x2] = processed_region

        result = _wrap(canvas)
        result.update(meta)
        result.update(processed_meta)
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coerce_for_compare(v: Any):
    """Try to make value numeric for comparison; fall back to string."""
    if isinstance(v, (int, float)):
        return v
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        return float(v)
    except (ValueError, TypeError):
        pass
    return str(v)
