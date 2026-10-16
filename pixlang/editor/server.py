# pixlang/editor/server.py
"""
PixLang Visual Editor — Flask backend.

The editor is a local, single-user tool.  Every API call is protected by:

* a Host allow-list (localhost / 127.0.0.1 / ::1)  -> blocks DNS rebinding
* an Origin / Sec-Fetch-Site check                -> blocks cross-site requests
* strict ``application/json`` bodies on POST       -> no "simple" CSRF forms
* a per-process random token, injected into index.html and required in the
  ``X-PixLang-Token`` header                       -> blocks blind CSRF

File operations are confined to a *workspace root* (default: the CWD the
editor was started from).  Pipelines run in a worker thread with a deadline.

Use :func:`create_app` to build an app; ``app`` is a default instance kept for
the CLI and backwards compatibility.
"""
from __future__ import annotations

import base64
import contextlib
import inspect
import io
import json
import os
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import fields as dc_fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from pixlang.commands import registry as pxl_registry
from pixlang.commands.builtin import _unwrap
from pixlang.commands.registry import CommandRegistry
from pixlang.executor.engine import Executor
from pixlang.linter import Linter
from pixlang.parser import parse as pxl_parse
from pixlang.parser.ast_nodes import (
    ASTNode, AssertStmt, BareIdent, Command, IfBlock, IncludeStmt, Pipeline,
    RepeatBlock, RoiBlock, SetVar, VarRef,
)

# ── Path constants ────────────────────────────────────────────────────────────

STATIC_DIR     = Path(__file__).parent / "static"
EXAMPLES_DIR   = Path(__file__).parent.parent.parent / "examples"
WORKSPACE_FILE = Path.home() / ".pixlang_editor_workspace.json"

if not EXAMPLES_DIR.exists():
    EXAMPLES_DIR = Path.cwd() / "examples"

SAMPLE_IMAGE = EXAMPLES_DIR / "sample.jpg"

TOKEN_HEADER       = "X-PixLang-Token"
MAX_CONTENT_LENGTH = 2 * 1024 * 1024        # request body cap
MAX_CODE_CHARS     = 512 * 1024             # pipeline source cap
DEFAULT_TIMEOUT_S  = 30.0
PREVIEW_MAX_SIDE   = 1200                   # legacy image_b64 downscale
FULL_MAX_SIDE      = 4096                   # /api/result full-size cap
THUMB_MAX_SIDE     = 160
RESULT_CACHE_RUNS  = 6
MAX_LISTED_FILES   = 500

_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox",
              ".mypy_cache", ".pytest_cache", "build", "dist", ".idea", ".vscode"}

# ── Command categories (fallback when CommandInfo has no .category) ───────────

_COMMAND_CATEGORIES: dict[str, list[str]] = {
    "IO":          ["LOAD", "SAVE", "PRINT_INFO", "LOAD_GLOB", "SAVE_EACH"],
    "Geometry":    ["RESIZE", "RESIZE_PERCENT", "CROP", "ROTATE", "FLIP", "AUTO_CROP"],
    "Color":       ["GRAYSCALE", "INVERT", "NORMALIZE", "EQUALIZE_HIST", "HEATMAP"],
    "Threshold":   ["THRESHOLD", "THRESHOLD_OTSU", "ADAPTIVE_THRESHOLD"],
    "Filter":      ["BLUR", "MEDIAN_BLUR", "SHARPEN", "CANNY"],
    "Morphology":  ["DILATE", "ERODE"],
    "Analysis":    ["FIND_CONTOURS", "DRAW_BOUNDING_BOXES", "DRAW_CONTOURS",
                    "COMPARE", "HISTOGRAM_SAVE", "PIPELINE_STATS"],
    "Composition": ["CHECKPOINT", "OVERLAY", "BLEND"],
    "Annotation":  ["DRAW_TEXT"],
}
_CMD_TO_CATEGORY: dict[str, str] = {
    cmd: cat for cat, cmds in _COMMAND_CATEGORIES.items() for cmd in cmds
}

# Commands whose first string argument is a file read / written relative to
# the pipeline file.  Used to make relative paths behave like the CLI.
_INPUT_PATH_CMDS  = {"LOAD"}
_OUTPUT_PATH_CMDS = {"SAVE", "HISTOGRAM_SAVE"}

_OPERATORS = ["==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "%",
              "&", "|", "^", "~", "<<", ">>", "=", "&&", "||", "!"]


# ══════════════════════════════════════════════════════════════════════════════
# Value / label formatting (never leak Python reprs into the UI)
# ══════════════════════════════════════════════════════════════════════════════

def fmt_value(v: Any, depth: int = 0) -> str:
    """Render an AST argument the way it would be written in PixLang."""
    if depth > 6:
        return "…"
    if isinstance(v, VarRef):
        return f"${v.var_name}"
    if isinstance(v, BareIdent):
        return v.name
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return str(v)
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(fmt_value(x, depth + 1) for x in v)
    if isinstance(v, ASTNode) or is_dataclass(v):
        return _fmt_expr_node(v, depth + 1)
    return str(v)


def _simple_fields(node: Any) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if not is_dataclass(node):
        return out
    for f in dc_fields(node):
        if f.name in ("line", "col", "end_line", "end_col", "file", "filename"):
            continue
        val = getattr(node, f.name, None)
        if isinstance(val, list) and any(isinstance(x, ASTNode) for x in val):
            continue
        out.append((f.name, val))
    return out


def _fmt_expr_node(node: Any, depth: int) -> str:
    """Best-effort rendering of (possibly unknown) expression nodes."""
    g = lambda *names: next((getattr(node, n) for n in names if hasattr(node, n)), None)  # noqa: E731
    op = g("op", "operator")
    left, right = g("left", "lhs"), g("right", "rhs")
    if op is not None and left is not None and right is not None:
        return f"{fmt_value(left, depth)} {op} {fmt_value(right, depth)}"
    operand = g("operand", "expr", "target")
    if op is not None and operand is not None:
        return f"{op}{fmt_value(operand, depth)}"
    name = g("name", "var_name", "func", "func_name")
    args = g("args", "arguments")
    if isinstance(name, str) and isinstance(args, list):
        return f"{name}(" + ", ".join(fmt_value(a, depth) for a in args) + ")"
    if isinstance(name, str):
        return name
    value = g("value")
    if value is not None:
        return fmt_value(value, depth)
    parts = [fmt_value(v, depth) for _, v in _simple_fields(node)]
    parts = [p for p in parts if p]
    return " ".join(parts) or type(node).__name__


def _kw_from_class(node: Any) -> str:
    name = type(node).__name__
    for suf in ("Statement", "Stmt", "Block", "Node", "Def", "Decl"):
        if name.endswith(suf) and len(name) > len(suf):
            name = name[: -len(suf)]
            break
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


# ══════════════════════════════════════════════════════════════════════════════
# Flow graph builder
# ══════════════════════════════════════════════════════════════════════════════

_BRANCH_LABELS = {"body": "body", "else_body": "else", "orelse": "else",
                  "then_body": "then", "elifs": "elif", "branches": "branch"}


class FlowBuilder:
    """Turns an AST into (a) a nested tree for the UI and (b) a flat node/edge
    graph (kept for API compatibility).  Instances are per-request, so ids are
    thread-safe."""

    def __init__(self) -> None:
        self._n = 0

    def new_id(self, prefix: str = "n") -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    # ── tree ──────────────────────────────────────────────────────────────────

    def tree(self, stmts: Iterable[Any]) -> list[dict]:
        return [self._item(s) for s in stmts if s is not None]

    def _item(self, s: Any) -> dict:
        line = int(getattr(s, "line", 0) or 0)
        end_line = int(getattr(s, "end_line", 0) or 0) or None
        if isinstance(s, Command):
            args = " ".join(fmt_value(a) for a in s.args)
            return {"id": self.new_id("cmd"), "kind": "stmt", "type": "command",
                    "name": s.name, "label": s.name, "args": args, "line": line,
                    "category": self.category(s.name)}
        if isinstance(s, SetVar):
            return {"id": self.new_id("set"), "kind": "stmt", "type": "setvar",
                    "name": "SET", "label": "SET",
                    "args": f"{s.var_name} = {fmt_value(s.value)}",
                    "line": line, "category": "Variable"}
        if isinstance(s, IncludeStmt):
            return {"id": self.new_id("inc"), "kind": "stmt", "type": "include",
                    "name": "INCLUDE", "label": "INCLUDE", "args": fmt_value(s.path),
                    "line": line, "category": "Include"}
        if isinstance(s, AssertStmt):
            a = f"{s.subject} {s.op} {fmt_value(s.expected)}"
            return {"id": self.new_id("assert"), "kind": "stmt", "type": "assert",
                    "name": "ASSERT", "label": "ASSERT", "args": a,
                    "line": line, "category": "Assert"}
        if isinstance(s, IfBlock) and not hasattr(s, "elifs") and not hasattr(s, "else_body"):
            cond = f"{s.var_name} {s.op} {fmt_value(s.cmp_value)}"
            return self._block(s, "if", "IF", cond, "Control", line, end_line,
                               [("then", line, s.body), ("else", None, [])], "ENDIF")
        if isinstance(s, RepeatBlock):
            return self._block(s, "repeat", "REPEAT", fmt_value(s.count), "Control",
                               line, end_line, [("body", line, s.body)], "END", loop=True)
        if isinstance(s, RoiBlock):
            args = f"{fmt_value(s.x)} {fmt_value(s.y)} {fmt_value(s.w)} {fmt_value(s.h)}"
            return self._block(s, "roi", "ROI", args, "ROI", line, end_line,
                               [("region", line, s.body)], "ROI_RESET")
        return self._generic(s, line, end_line)

    def _block(self, s, prefix, kw, args, cat, line, end_line, branches,
               footer, loop=False) -> dict:
        return {
            "id": self.new_id(prefix), "kind": "block", "type": prefix,
            "name": kw, "label": kw, "args": args, "line": line,
            "end_line": end_line, "category": cat, "loop": loop,
            "footer": footer,
            "branches": [{"label": lbl, "line": bl, "items": self.tree(body)}
                         for lbl, bl, body in branches],
        }

    def _generic(self, s: Any, line: int, end_line: int | None) -> dict:
        """Unknown node: label from class name + simple fields; every list
        attribute holding AST nodes becomes a nested branch."""
        kw = _kw_from_class(s)
        name = getattr(s, "name", None) or getattr(s, "func_name", None)
        params = getattr(s, "params", None)
        if isinstance(name, str) and isinstance(params, list):
            args = f"{name}(" + ", ".join(fmt_value(p) if not isinstance(p, str) else p
                                          for p in params) + ")"
        else:
            parts = []
            for _, v in _simple_fields(s):
                t = fmt_value(v)
                if t:
                    parts.append(t)
            args = " ".join(parts)
        branches = []
        if is_dataclass(s):
            for f in dc_fields(s):
                val = getattr(s, f.name, None)
                if not isinstance(val, list) or not val:
                    if isinstance(val, list) and f.name in ("body", "then_body"):
                        branches.append({"label": _BRANCH_LABELS.get(f.name, f.name),
                                         "line": line, "items": []})
                    continue
                if all(isinstance(x, (tuple, list)) for x in val):
                    # e.g. elifs = [(cond, body), ...]
                    for tup in val:
                        cond = next((x for x in tup if not isinstance(x, list)), None)
                        body = next((x for x in tup if isinstance(x, list)), [])
                        branches.append({"label": "elif " + fmt_value(cond) if cond is not None else "elif",
                                         "line": getattr(cond, "line", None),
                                         "items": self.tree(body)})
                    continue
                if not any(isinstance(x, ASTNode) for x in val):
                    continue
                if all(isinstance(x, ASTNode) and isinstance(getattr(x, "body", None), list)
                       and not isinstance(x, (IfBlock, RepeatBlock, RoiBlock))
                       and f.name in ("elifs", "branches", "cases", "arms") for x in val):
                    for br in val:
                        cond = [fmt_value(v) for _, v in _simple_fields(br)]
                        lbl = _BRANCH_LABELS.get(f.name, f.name)
                        branches.append({"label": (lbl + " " + " ".join(c for c in cond if c)).strip(),
                                         "line": getattr(br, "line", None),
                                         "items": self.tree(br.body)})
                    continue
                branches.append({"label": _BRANCH_LABELS.get(f.name, f.name),
                                 "line": line if f.name in ("body", "then_body") else None,
                                 "items": self.tree(val)})
        kind = "block" if branches else "stmt"
        item = {"id": self.new_id(kw.lower() or "node"), "kind": kind,
                "type": kw.lower(), "name": kw, "label": kw, "args": args,
                "line": line, "category": self.category_for_kw(kw)}
        if kind == "block":
            low = kw.lower()
            if low == "if" and not any(b["label"] == "else" for b in branches):
                branches.append({"label": "else", "line": None, "items": []})
            item.update({
                "end_line": end_line, "branches": branches,
                "loop": low in ("while", "for", "repeat", "loop", "foreach"),
                "func": low in ("func", "function", "def"),
                "footer": {"if": "ENDIF", "func": "ENDFUNC", "roi": "ROI_RESET"}.get(low, "END"),
            })
        return item

    @staticmethod
    def category(name: str) -> str:
        return _CMD_TO_CATEGORY.get(name, "Other")

    @staticmethod
    def category_for_kw(kw: str) -> str:
        if kw in ("IF", "ELIF", "ELSE", "WHILE", "FOR", "REPEAT", "BREAK", "CONTINUE",
                  "RETURN", "LOOP"):
            return "Control"
        if kw in ("FUNC", "FUNCTION", "CALL"):
            return "Function"
        return "Statement"

    # ── flat graph (legacy) ───────────────────────────────────────────────────

    def flat(self, stmts: list) -> tuple[list, list, str | None, str | None]:
        return _flatten(self.tree(stmts), self)


def _flatten(items: list[dict], b: FlowBuilder) -> tuple[list, list, str | None, str | None]:
    nodes: list[dict] = []
    edges: list[dict] = []
    first: str | None = None
    last: str | None = None

    def chain(to: str) -> None:
        nonlocal first, last
        if last is not None:
            edges.append({"from": last, "to": to})
        if first is None:
            first = to

    legacy_type = {"if": ("if_header", "if_footer"),
                   "repeat": ("repeat_header", "repeat_footer"),
                   "roi": ("roi_header", "roi_footer")}
    for it in items:
        label = (it["label"] + (" " + it["args"] if it.get("args") else "")).strip()
        if it["kind"] == "stmt":
            node = {"id": it["id"], "type": it["type"], "label": label,
                    "line": it["line"], "category": it["category"]}
            if it["type"] == "command":
                node["name"] = it["name"]
            nodes.append(node)
            chain(it["id"])
            last = it["id"]
            continue
        htype, ftype = legacy_type.get(it["type"], (it["type"] + "_header", it["type"] + "_footer"))
        hid, fid = it["id"], b.new_id("end")
        nodes.append({"id": hid, "type": htype, "label": label, "line": it["line"],
                      "category": it["category"]})
        chain(hid)
        nodes.append({"id": fid, "type": ftype, "label": it.get("footer") or "END",
                      "line": it.get("end_line") or it["line"], "category": it["category"]})
        for br in it["branches"]:
            sn, se, bf, bl = _flatten(br["items"], b)
            nodes.extend(sn)
            edges.extend(se)
            lbl = {"then": "true", "else": "false", "body": "body"}.get(br["label"], br["label"])
            if bf:
                edges.append({"from": hid, "to": bf, "label": lbl})
                edges.append({"from": bl, "to": fid})
                if it.get("loop"):
                    edges.append({"from": fid, "to": bf, "label": "loop"})
            else:
                edges.append({"from": hid, "to": fid, "label": lbl})
        if it.get("loop"):
            edges.append({"from": hid, "to": fid, "label": "exit"})
        last = fid
    return nodes, edges, first, last


# Legacy module-level helpers (thread-local so concurrent requests don't clash)
_tls = threading.local()


def _builder() -> FlowBuilder:
    b = getattr(_tls, "builder", None)
    if b is None:
        b = _tls.builder = FlowBuilder()
    return b


def _new_id(prefix: str = "n") -> str:
    return _builder().new_id(prefix)


def _reset_id() -> None:
    _tls.builder = FlowBuilder()


def _serialize_stmts(stmts: list) -> tuple[list, list, str | None, str | None]:
    return _builder().flat(stmts)


def _pipeline_body(pipeline: Any) -> list:
    for attr in ("commands", "body", "statements"):
        val = getattr(pipeline, attr, None)
        if isinstance(val, list):
            return val
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Error location helpers
# ══════════════════════════════════════════════════════════════════════════════

_LINE_RE = re.compile(r"\b[Ll]ine\s+(\d+)")
_TAG_RE = re.compile(r"^\s*(\[[^\]]+\]\s*)+")


def _clean_message(msg: str) -> str:
    """Strip '[PixLang Parser] Line 3:' noise for display."""
    first = msg.strip().splitlines()[0] if msg.strip() else msg
    first = _TAG_RE.sub("", first)
    first = re.sub(r"^[Ll]ine\s+\d+\s*:\s*", "", first)
    first = re.sub(r"^Error at line \d+\s*(\([^)]*\))?\s*:\s*", "", first)
    first = _TAG_RE.sub("", first)
    return first.strip() or msg.strip()


def _guess_col(code: str, line: int, msg: str) -> int | None:
    """Locate the offending token inside the source line (1-based col)."""
    lines = code.splitlines()
    if not (1 <= line <= len(lines)):
        return None
    text = lines[line - 1]
    for pat in (r"got '[A-Z_]+' '((?:[^'\\]|\\.)*)'", r"[Uu]nexpected character '((?:[^'\\]|\\.)*)'",
                r"token '[A-Z_]+' '((?:[^'\\]|\\.)*)'"):
        m = re.search(pat, msg)
        if m:
            idx = text.find(m.group(1))
            if idx >= 0:
                return idx + 1
    stripped = len(text) - len(text.lstrip())
    return stripped + 1


def error_info(exc: BaseException, code: str, kind: str,
               fallback_line: int | None = None) -> dict:
    msg = str(exc)
    line = getattr(exc, "lineno", None) if isinstance(exc, SyntaxError) else None
    line = line or getattr(exc, "line", None)
    if not isinstance(line, int):
        line = None
    col = getattr(exc, "offset", None) if isinstance(exc, SyntaxError) else None
    col = col or getattr(exc, "col", None)
    if not isinstance(col, int):
        col = None
    efile = getattr(exc, "filename", None) if isinstance(exc, SyntaxError) else None
    efile = efile or getattr(exc, "file", None)
    if not line:
        m = _LINE_RE.search(msg)
        if m:
            line = int(m.group(1))
    if not line and exc.__cause__ is not None:
        m = _LINE_RE.search(str(exc.__cause__))
        if m:
            line = int(m.group(1))
    if not line:
        line = fallback_line
    if line and not col:
        col = _guess_col(code, line, msg)
    if isinstance(exc, AssertionError):
        kind = "assert"
    return {"kind": kind, "message": _clean_message(msg), "raw": msg.strip(),
            "line": line or None, "col": col or None,
            "file": str(efile) if efile else None,
            "type": type(exc).__name__}


# ══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════════════════════════════════════

def _as_array(image: Any) -> Any:
    """Extract an ndarray from ndarray / wrapped dict / (img, meta) tuple."""
    import numpy as np
    if image is None:
        return None
    if isinstance(image, tuple) and image and isinstance(image[0], np.ndarray):
        return image[0]
    try:
        img, _ = _unwrap(image)
    except Exception:
        return None
    return img if isinstance(img, np.ndarray) else None


def _to_display(img: Any, max_side: int) -> Any:
    """Convert to 8-bit BGR(A) for PNG, downscaled to max_side."""
    import cv2
    import numpy as np
    if img.dtype != np.uint8:
        if img.dtype == np.bool_:
            img = img.astype(np.uint8) * 255
        else:
            f = img.astype(np.float32)
            lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
            if 0.0 <= lo and hi <= 1.0 and img.dtype.kind == "f":
                f = f * 255.0
            elif hi > 255 or lo < 0:
                f = (f - lo) / (hi - lo or 1.0) * 255.0
            img = np.clip(f, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img


def _encode_png(img: Any, max_side: int) -> bytes | None:
    import cv2
    try:
        ok, buf = cv2.imencode(".png", _to_display(img, max_side))
    except Exception:
        return None
    return buf.tobytes() if ok else None


def _image_to_b64(image: Any) -> str | None:
    """Convert a pixlang image (ndarray or wrapped dict) to a base64 PNG string."""
    img = _as_array(image)
    if img is None:
        return None
    data = _encode_png(img, PREVIEW_MAX_SIDE)
    return base64.b64encode(data).decode("ascii") if data else None


def _image_meta(img: Any) -> dict:
    ch = 1 if img.ndim == 2 else int(img.shape[2])
    dt = {"uint8": "u8", "uint16": "u16", "float32": "f32", "float64": "f64",
          "int32": "i32", "int16": "i16", "bool": "bool"}.get(str(img.dtype), str(img.dtype))
    return {"width": int(img.shape[1]), "height": int(img.shape[0]),
            "channels": ch, "dtype": dt}


class _ResultStore:
    """Keeps the arrays of the last few runs; PNGs are encoded lazily."""

    def __init__(self, max_runs: int = RESULT_CACHE_RUNS) -> None:
        self._runs: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_runs

    def put(self, run_id: str, images: dict[str, Any]) -> None:
        with self._lock:
            self._runs[run_id] = {"arrays": images, "png": {}}
            while len(self._runs) > self._max:
                self._runs.popitem(last=False)

    def png(self, run_id: str, name: str, thumb: bool) -> bytes | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            key = name + (".thumb" if thumb else "")
            if key in run["png"]:
                return run["png"][key]
            arr = run["arrays"].get(name)
        if arr is None:
            return None
        data = _encode_png(arr, THUMB_MAX_SIDE if thumb else FULL_MAX_SIDE)
        with self._lock:
            if run_id in self._runs and data is not None:
                self._runs[run_id]["png"][key] = data
        return data


# ══════════════════════════════════════════════════════════════════════════════
# Executor used by the editor
# ══════════════════════════════════════════════════════════════════════════════

class RunCancelled(RuntimeError):
    pass


class _TrackedBody(list):
    """List proxy handed to Executor._run_body: records visited lines and
    honours cancellation without depending on executor internals."""

    def __init__(self, stmts: list, ex: "_CapturingExecutor") -> None:
        super().__init__(stmts)
        self._ex = ex

    def __iter__(self):
        for stmt in list.__iter__(self):
            self._ex._check_cancel()
            self._ex._visit(stmt)
            yield stmt


class _CapturingExecutor(Executor):
    """Executor that retains the final / partial / first-LOAD images, per-line
    timing, visited lines, and supports cooperative cancellation."""

    def __init__(self, *args: Any, base_dir: Path | None = None,
                 workspace_root: Path | None = None,
                 include_roots: list[Path] | None = None,
                 cancel: threading.Event | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.final_image: Any = None
        self.last_image: Any = None
        self.before_image: Any = None
        self.base_dir = base_dir
        self.workspace_root = workspace_root
        self.include_roots = include_roots
        self.cancel = cancel or threading.Event()
        self.line_times: dict[int, dict] = {}
        self.visited: set[int] = set()
        self.current_line: int | None = None
        self._include_depth = 0
        self._body_depth = 0

    # hooks ---------------------------------------------------------------
    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise RunCancelled("Run cancelled (time limit reached)")

    def _visit(self, stmt: Any) -> None:
        if self._include_depth == 0:
            line = getattr(stmt, "line", None)
            if isinstance(line, int) and line > 0:
                self.visited.add(line)
                self.current_line = line

    def run(self, pipeline: Pipeline) -> None:
        result = super().run(pipeline)
        if result is not None and self.final_image is None:
            self.final_image = result

    def _run_body(self, stmts, image):  # type: ignore[override]
        self._body_depth += 1
        try:
            result = super()._run_body(_TrackedBody(list(stmts), self), image)
        finally:
            self._body_depth -= 1
        if self._body_depth == 0:
            self.final_image = result
        return result

    def _exec_command(self, cmd: Command, image: Any) -> Any:
        self._check_cancel()
        cmd = self._rewrite_paths(cmd)
        t0 = time.perf_counter()
        result = super()._exec_command(cmd, image)
        ms = (time.perf_counter() - t0) * 1000
        if self._include_depth == 0 and cmd.line:
            rec = self.line_times.setdefault(cmd.line, {"ms": 0.0, "count": 0, "name": cmd.name})
            rec["ms"] += ms
            rec["count"] += 1
        self.last_image = result
        if self.before_image is None and cmd.name == "LOAD" and result is not None:
            self.before_image = result
        return result

    def _exec_include(self, stmt: IncludeStmt, image: Any) -> Any:
        base = self.pipeline_path.parent if self.pipeline_path else (self.base_dir or Path("."))
        target = (base / stmt.path).resolve()
        roots = self.include_roots or []
        if roots and not any(_is_within(target, r) for r in roots):
            raise RuntimeError(
                f"[INCLUDE] '{stmt.path}' is outside the editor workspace (line {stmt.line})")
        if target.suffix.lower() != ".pxl":
            raise RuntimeError(f"[INCLUDE] Only .pxl files can be included (line {stmt.line})")
        self._include_depth += 1
        try:
            return super()._exec_include(stmt, image)
        except SyntaxError as exc:
            raise RuntimeError(
                f"[INCLUDE] Syntax error in '{stmt.path}' (line {stmt.line})") from exc
        finally:
            self._include_depth -= 1

    def _rewrite_paths(self, cmd: Command) -> Command:
        """Relative paths resolve against the pipeline file's directory; inputs
        fall back to the workspace root (matching `pixlang run` from the project
        root); output directories inside the workspace are created."""
        if self.base_dir is None or not cmd.args:
            return cmd
        if cmd.name not in _INPUT_PATH_CMDS and cmd.name not in _OUTPUT_PATH_CMDS:
            return cmd
        raw = cmd.args[0]
        try:
            val = self._resolve(raw)
        except Exception:
            return cmd
        if not isinstance(val, str) or not val or os.path.isabs(val):
            return cmd
        primary = (self.base_dir / val)
        new_val = None
        if cmd.name in _INPUT_PATH_CMDS:
            if not primary.exists() and self.workspace_root is not None:
                alt = self.workspace_root / val
                if alt.exists():
                    new_val = str(alt)
        else:
            parent = primary.resolve().parent
            if self.workspace_root is not None and _is_within(parent, self.workspace_root):
                with contextlib.suppress(OSError):
                    parent.mkdir(parents=True, exist_ok=True)
        if new_val is None:
            return cmd
        return Command(name=cmd.name, args=[new_val] + list(cmd.args[1:]), line=cmd.line)


def _is_within(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _clone_registry(src: CommandRegistry) -> CommandRegistry:
    try:
        clone = type(src)()
        clone._commands = dict(src._commands)  # type: ignore[attr-defined]
        return clone
    except Exception:
        return src


# ══════════════════════════════════════════════════════════════════════════════
# Command metadata
# ══════════════════════════════════════════════════════════════════════════════

def _param_list(info: Any) -> list[dict]:
    params = getattr(info, "params", None)
    if isinstance(params, list) and params:
        out = []
        for p in params:
            if isinstance(p, dict):
                out.append({"name": str(p.get("name", "")), "type": str(p.get("type", "") or ""),
                            "default": None if p.get("default") is None else fmt_default(p.get("default")),
                            "optional": bool(p.get("optional", p.get("default") is not None)),
                            "doc": str(p.get("doc", "") or "")})
            else:
                name = getattr(p, "name", str(p))
                default = getattr(p, "default", None)
                ptype = getattr(p, "type", "") or getattr(p, "annotation", "")
                out.append({"name": str(name),
                            "type": getattr(ptype, "__name__", str(ptype or "")),
                            "default": None if default in (None, inspect.Parameter.empty) else fmt_default(default),
                            "optional": default not in (None, inspect.Parameter.empty),
                            "doc": str(getattr(p, "doc", "") or "")})
        return out
    out = []
    try:
        sig = inspect.signature(info.fn)
    except (TypeError, ValueError):
        return out
    for name, p in sig.parameters.items():
        if name in ("image", "_ctx", "img") or p.kind in (p.VAR_KEYWORD,):
            continue
        ann = p.annotation
        tname = ann.__name__ if (ann is not inspect.Parameter.empty and hasattr(ann, "__name__")) \
            else ("" if ann is inspect.Parameter.empty else str(ann))
        has_default = p.default is not inspect.Parameter.empty
        out.append({"name": name + ("…" if p.kind == p.VAR_POSITIONAL else ""),
                    "type": tname,
                    "default": fmt_default(p.default) if has_default else None,
                    "optional": has_default or p.kind == p.VAR_POSITIONAL, "doc": ""})
    return out


def fmt_default(v: Any) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if v is None:
        return "none"
    return str(v)


def _command_meta(info: Any) -> dict:
    full_doc = inspect.cleandoc(info.fn.__doc__ or "") if getattr(info, "fn", None) else ""
    lines = [ln.strip() for ln in full_doc.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    usage = getattr(info, "usage", None)
    summary = first
    if " — " in first:
        head, _, tail = first.partition(" — ")
        summary = tail.strip()
        if not usage and head.strip().upper().startswith(info.name):
            usage = head.strip()
    elif first.upper().startswith(info.name):
        if not usage:
            usage = first
        summary = ""
    params = _param_list(info)
    if not usage:
        usage = info.name + "".join(
            f" [{p['name']}]" if p["optional"] else f" {p['name']}" for p in params)
    category = getattr(info, "category", None) or _CMD_TO_CATEGORY.get(info.name, "Other")
    doc_body = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return {
        "name": info.name,
        "category": category,
        "usage": usage,
        "summary": summary or (getattr(info, "doc", "") or ""),
        "doc": doc_body,
        "params": params,
        "source": getattr(info, "source", "builtin"),
    }


def _keywords() -> list[str]:
    try:
        from pixlang.parser.lexer import KEYWORDS
        return sorted(KEYWORDS)
    except Exception:  # pragma: no cover
        return ["SET", "IF", "ENDIF", "REPEAT", "END", "INCLUDE", "ASSERT", "ROI", "ROI_RESET"]


# ══════════════════════════════════════════════════════════════════════════════
# Templates
# ══════════════════════════════════════════════════════════════════════════════

def new_file_template(file_dir: Path, workspace_root: Path) -> str:
    """A pipeline that runs out of the box from ``file_dir``."""
    if SAMPLE_IMAGE.exists():
        try:
            rel = os.path.relpath(SAMPLE_IMAGE.resolve(), file_dir.resolve())
        except ValueError:
            rel = str(SAMPLE_IMAGE.resolve())
        src = rel.replace(os.sep, "/")
    else:
        src = "input.jpg"
    return f"""# New pipeline — edit and press Ctrl/Cmd+Enter to run
SET size 640

LOAD "{src}"
RESIZE $size 480
CHECKPOINT "original"

# ── Edges ─────────────────────────────────
GRAYSCALE
BLUR 5
CANNY 60 160
DILATE 3 1

# ── Composite back onto the original ──────
OVERLAY "original" 0.6

SAVE "out/result.png"
"""


# ══════════════════════════════════════════════════════════════════════════════
# App factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app(workspace: str | os.PathLike | None = None, *,
               token: str | None = None,
               run_timeout: float = DEFAULT_TIMEOUT_S,
               plugins: bool = True,
               registry: CommandRegistry | None = None,
               examples_dir: Path | None = None) -> Flask:
    """Build an editor app.

    workspace    — root directory for file browsing / saving (default: CWD)
    token        — API token (default: random per process)
    run_timeout  — seconds before a pipeline run is reported as timed out
    plugins      — load local/entry-point plugins for runs (into a private
                   copy of the registry, never the global one)
    """
    root = Path(workspace or os.getcwd()).resolve()
    ex_dir = (examples_dir or EXAMPLES_DIR).resolve()
    api_token = token or secrets.token_urlsafe(32)
    reg = _clone_registry(registry or pxl_registry) if plugins else (registry or pxl_registry)

    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["PIXLANG_TOKEN"] = api_token
    app.config["PIXLANG_WORKSPACE"] = root
    app.config["PIXLANG_RUN_TIMEOUT"] = float(run_timeout)

    results = _ResultStore()
    run_lock = threading.Lock()

    # ── security ──────────────────────────────────────────────────────────────

    def _host_ok(host: str | None) -> bool:
        if not host:
            return False
        try:
            hostname = urlsplit("//" + host).hostname
        except ValueError:
            return False
        return hostname in _ALLOWED_HOSTS

    @app.before_request
    def _guard() -> Any:
        if not _host_ok(request.host):
            return _err("Host not allowed", 403)
        origin = request.headers.get("Origin")
        if origin is not None:
            if origin == "null" or not _host_ok(urlsplit(origin).netloc):
                return _err("Cross-origin request rejected", 403)
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            return _err("Cross-site request rejected", 403)
        path = request.path
        if path.startswith("/api/") and not path.startswith("/api/result/"):
            if not secrets.compare_digest(request.headers.get(TOKEN_HEADER, ""), api_token):
                return _err("Missing or invalid editor token", 403)
            if request.method == "POST" and request.mimetype != "application/json":
                return _err("Content-Type must be application/json", 415)
        return None

    @app.after_request
    def _headers(resp: Response) -> Response:
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if request.path.startswith("/api/") and not request.path.startswith("/api/result/"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    def _err(msg: str, status: int) -> Any:
        return jsonify({"error": msg}), status

    def _body() -> dict:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    def _code(data: dict) -> str:
        code = data.get("code", "")
        if not isinstance(code, str):
            abort(400)
        if len(code) > MAX_CODE_CHARS:
            abort(413)
        return code

    # ── path helpers ──────────────────────────────────────────────────────────

    def _ws_path(rel: Any, must_exist: bool = False) -> Path | None:
        """Resolve a workspace-relative .pxl path, or None when invalid."""
        if not isinstance(rel, str) or not rel.strip():
            return None
        rel = rel.strip().replace("\\", "/")
        if rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) or "\x00" in rel:
            return None
        p = (root / rel).resolve()
        if not _is_within(p, root) or p.suffix.lower() != ".pxl":
            return None
        if must_exist and not p.is_file():
            return None
        return p

    def _example_path(name: Any) -> Path | None:
        if not isinstance(name, str):
            return None
        safe = "".join(c for c in name if c.isalnum() or c in "_-")
        if not safe or safe != name:
            return None
        p = ex_dir / f"{safe}.pxl"
        return p if p.is_file() else None

    def _rel(p: Path) -> str:
        return p.resolve().relative_to(root).as_posix()

    # ── routes: shell ─────────────────────────────────────────────────────────

    @app.route("/")
    def index() -> Any:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("{{PIXLANG_TOKEN}}", api_token)
        resp = Response(html, mimetype="text/html")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/info")
    def api_info() -> Any:
        return jsonify({"workspace": str(root), "workspace_name": root.name or str(root),
                        "examples_dir": str(ex_dir), "run_timeout": app.config["PIXLANG_RUN_TIMEOUT"],
                        "command_count": len(reg)})

    # ── routes: language ──────────────────────────────────────────────────────

    @app.route("/api/language")
    def api_language() -> Any:
        cmds = [_command_meta(i) for i in reg.list_info()]
        cats = sorted({c["category"] for c in cmds})
        return jsonify({"keywords": _keywords(), "commands": cmds, "categories": cats,
                        "operators": _OPERATORS,
                        "builtin_vars": ["ITER", "ITER1"]})

    @app.route("/api/commands")
    def api_commands() -> Any:
        result = []
        for info in reg.list_info():
            meta = _command_meta(info)
            parts = []
            for p in meta["params"]:
                s = p["name"] + (f": {p['type']}" if p["type"] else "")
                if p["default"] is not None:
                    s += f"={p['default']}"
                parts.append(s)
            sig = info.name + (" " + "  ".join(parts) if parts else "")
            result.append({"name": info.name, "category": meta["category"], "signature": sig,
                           "doc": meta["summary"], "example": meta["usage"],
                           "usage": meta["usage"], "params": meta["params"]})
        return jsonify({"commands": result})

    # ── routes: analysis ──────────────────────────────────────────────────────

    @app.route("/api/lint", methods=["POST"])
    def api_lint() -> Any:
        code = _code(_body())
        try:
            pipeline = pxl_parse(code)
        except SyntaxError as exc:
            info = error_info(exc, code, "syntax")
            return jsonify({"issues": [{
                "line": info["line"] or 1, "col": info["col"], "severity": "error",
                "code": "PARSE", "message": info["message"]}]})
        try:
            diags = Linter(reg).lint(pipeline)
        except Exception as exc:  # linter bug must never break the editor
            return jsonify({"issues": [], "warning": f"linter failed: {exc}"})
        issues = [{"line": d.line, "col": getattr(d, "col", None),
                   "severity": d.severity.value, "code": d.code, "message": d.message}
                  for d in diags]
        return jsonify({"issues": issues})

    @app.route("/api/parse", methods=["POST"])
    def api_parse() -> Any:
        code = _code(_body())
        if not code.strip():
            return jsonify({"nodes": [], "edges": [], "tree": []})
        try:
            pipeline = pxl_parse(code)
        except SyntaxError as exc:
            info = error_info(exc, code, "syntax")
            return jsonify({"error": info["message"], "error_info": info,
                            "nodes": [], "edges": [], "tree": []})
        body = _pipeline_body(pipeline)
        tree = FlowBuilder().tree(body)
        nodes, edges, _, _ = _flatten(tree, FlowBuilder())
        return jsonify({"nodes": nodes, "edges": edges, "tree": tree})

    # ── routes: run ───────────────────────────────────────────────────────────

    @app.route("/api/run", methods=["POST"])
    def api_run() -> Any:
        data = _body()
        code = _code(data)
        run_id = uuid.uuid4().hex

        # Where does this pipeline "live"?
        pipeline_path: Path | None = None
        base_dir = root
        example = _example_path(data.get("example")) if data.get("example") else None
        if example is not None:
            pipeline_path = example
            base_dir = ex_dir.parent          # examples use repo-root paths
        else:
            p = _ws_path(data.get("path") or data.get("filename"))
            if p is not None:
                pipeline_path, base_dir = p, p.parent

        empty = {"run_id": run_id, "ok": False, "image_b64": None, "image": None,
                 "before": None, "checkpoints": [], "partial": False, "stdout": "",
                 "stats": {"commands_executed": 0, "total_ms": 0.0, "command_times": {},
                           "line_times": {}},
                 "visited_lines": [], "error": None, "error_info": None}

        try:
            pipeline = pxl_parse(code)
        except SyntaxError as exc:
            info = error_info(exc, code, "syntax")
            return jsonify({**empty, "error": str(exc), "error_info": info})

        if not run_lock.acquire(timeout=2.0):
            info = {"kind": "busy", "message": "Another run is still in progress",
                    "line": None, "col": None, "file": None, "raw": "", "type": "Busy"}
            return jsonify({**empty, "error": info["message"], "error_info": info})

        loader = None
        if plugins:
            try:
                from pixlang.plugins import PluginLoader
                loader = PluginLoader(reg)
            except Exception:
                loader = None
        cancel = threading.Event()
        executor = _CapturingExecutor(
            registry=reg, verbose=False, plugin_loader=loader,
            pipeline_path=pipeline_path, base_dir=base_dir, workspace_root=root,
            include_roots=[root, ex_dir], cancel=cancel)

        stdout_buf = io.StringIO()
        outcome: dict[str, Any] = {}

        def work() -> None:
            prev_cwd = os.getcwd()
            try:
                if not base_dir.exists() and _is_within(base_dir, root):
                    base_dir.mkdir(parents=True, exist_ok=True)
                os.chdir(base_dir)
                with contextlib.redirect_stdout(stdout_buf):
                    executor.run(pipeline)
            except BaseException as exc:  # noqa: BLE001 — report everything
                outcome["exc"] = exc
            finally:
                with contextlib.suppress(OSError):
                    os.chdir(prev_cwd)
                run_lock.release()

        worker = threading.Thread(target=work, name=f"pixlang-run-{run_id[:8]}", daemon=True)
        t0 = time.perf_counter()
        worker.start()
        timeout = app.config["PIXLANG_RUN_TIMEOUT"]
        worker.join(timeout)
        timed_out = worker.is_alive()
        if timed_out:
            cancel.set()
            worker.join(1.0)
        wall_ms = (time.perf_counter() - t0) * 1000

        err = None
        info = None
        if timed_out:
            info = {"kind": "timeout", "line": executor.current_line, "col": None,
                    "file": None, "type": "Timeout", "raw": "",
                    "message": f"Run exceeded the {timeout:g}s time limit and was stopped"}
            err = info["message"]
        elif "exc" in outcome:
            exc = outcome["exc"]
            info = error_info(exc, code, "runtime", fallback_line=executor.current_line)
            if info.get("file") and pipeline_path is not None and \
                    Path(info["file"]).name != pipeline_path.name:
                info["line"] = executor.current_line
            err = str(exc)

        raw_stats = executor.context.get("stats", {})
        stats = {
            "commands_executed": int(raw_stats.get("commands_executed", 0)),
            "total_ms": float(raw_stats.get("total_ms", 0.0) or wall_ms),
            "wall_ms": wall_ms,
            "command_times": {k: [float(v) for v in vs]
                              for k, vs in raw_stats.get("command_times", {}).items()},
            "line_times": {str(k): {"ms": float(v["ms"]), "count": int(v["count"]),
                                    "name": v["name"]}
                           for k, v in sorted(executor.line_times.items())},
        }

        final = executor.final_image
        if err is not None and executor.last_image is not None:
            final = executor.last_image
        arrays: dict[str, Any] = {}
        final_arr = _as_array(final)
        before_arr = _as_array(executor.before_image)
        if final_arr is not None:
            arrays["final"] = final_arr
        if before_arr is not None:
            arrays["before"] = before_arr
        cps = []
        try:
            cp_items = list(executor.context.get("checkpoints", {}).items())
        except Exception:
            cp_items = []
        for i, (name, val) in enumerate(cp_items[:24]):
            arr = _as_array(val)
            if arr is None:
                continue
            key = f"cp{i}"
            arrays[key] = arr
            cps.append({"name": str(name), "key": key, **_image_meta(arr),
                        "url": f"/api/result/{run_id}/{key}.png",
                        "thumb_url": f"/api/result/{run_id}/{key}.thumb.png"})
        results.put(run_id, arrays)

        def ref(key: str, arr: Any) -> dict | None:
            if arr is None:
                return None
            return {**_image_meta(arr), "url": f"/api/result/{run_id}/{key}.png",
                    "thumb_url": f"/api/result/{run_id}/{key}.thumb.png"}

        plugin_errors = []
        if loader is not None:
            for m in getattr(loader, "manifests", []):
                e = getattr(m, "error", None)
                if e and "already registered" not in str(e):
                    plugin_errors.append(f"{m.name}: {e}")

        return jsonify({
            "run_id": run_id,
            "ok": err is None,
            "image_b64": _image_to_b64(final_arr),
            "image": ref("final", final_arr),
            "before": ref("before", before_arr),
            "checkpoints": cps,
            "partial": err is not None and final_arr is not None,
            "stdout": stdout_buf.getvalue(),
            "stats": stats,
            "visited_lines": sorted(executor.visited),
            "error": err,
            "error_info": info,
            "plugin_errors": plugin_errors,
            "file": _rel(pipeline_path) if pipeline_path is not None and _is_within(pipeline_path, root) else None,
        })

    @app.route("/api/result/<run_id>/<name>")
    def api_result(run_id: str, name: str) -> Any:
        m = re.fullmatch(r"([a-z0-9]{1,16})(\.thumb)?\.png", name)
        if not re.fullmatch(r"[0-9a-f]{32}", run_id) or not m:
            abort(404)
        data = results.png(run_id, m.group(1), bool(m.group(2)))
        if data is None:
            abort(404)
        resp = Response(data, mimetype="image/png")
        resp.headers["Cache-Control"] = "private, max-age=3600, immutable"
        return resp

    # ── routes: files ─────────────────────────────────────────────────────────

    @app.route("/api/files")
    def api_files() -> Any:
        files: list[dict] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if not d.startswith(".") and d not in _SKIP_DIRS)
            for fn in sorted(filenames):
                if fn.lower().endswith(".pxl"):
                    p = Path(dirpath) / fn
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    files.append({"path": _rel(p), "name": fn, "size": st.st_size,
                                  "mtime": st.st_mtime})
                    if len(files) >= MAX_LISTED_FILES:
                        break
            if len(files) >= MAX_LISTED_FILES:
                break
        return jsonify({"root": str(root), "name": root.name or str(root), "files": files,
                        "truncated": len(files) >= MAX_LISTED_FILES})

    @app.route("/api/files/read", methods=["POST"])
    def api_files_read() -> Any:
        p = _ws_path(_body().get("path"), must_exist=True)
        if p is None:
            return _err("File not found or outside the workspace", 404)
        try:
            return jsonify({"path": _rel(p), "code": p.read_text(encoding="utf-8")})
        except (OSError, UnicodeDecodeError) as exc:
            return _err(f"Could not read file: {exc.__class__.__name__}", 400)

    @app.route("/api/files/save", methods=["POST"])
    def api_files_save() -> Any:
        data = _body()
        code = _code(data)
        p = _ws_path(data.get("path"))
        if p is None:
            return _err("Path must be a .pxl file inside the workspace", 400)
        if data.get("create") and p.exists():
            return _err("A file with that name already exists", 409)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(code, encoding="utf-8")
        except OSError as exc:
            return _err(f"Could not save: {exc.strerror or exc}", 500)
        return jsonify({"ok": True, "path": _rel(p)})

    @app.route("/api/template", methods=["POST"])
    def api_template() -> Any:
        rel = _body().get("path")
        p = _ws_path(rel) if rel else None
        d = p.parent if p is not None else root
        return jsonify({"code": new_file_template(d, root)})

    # ── routes: examples ──────────────────────────────────────────────────────

    @app.route("/api/examples")
    def api_examples() -> Any:
        examples = []
        for p in sorted(ex_dir.glob("*.pxl")):
            desc = ""
            for line in p.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    d = stripped.lstrip("#").strip().strip("─").strip()
                    if d and not d.lower().startswith(("file:", "run:")):
                        desc = d
                        break
            examples.append({"name": p.stem, "description": desc})
        return jsonify({"examples": examples})

    @app.route("/api/examples/<name>")
    def api_example_code(name: str) -> Any:
        safe = "".join(c for c in name if c.isalnum() or c == "_")
        path = ex_dir / f"{safe}.pxl"
        if not path.is_file():
            return jsonify({"error": f"Example '{safe}' not found"}), 404
        return jsonify({"code": path.read_text(encoding="utf-8"), "name": safe})

    # ── routes: legacy workspace blob ─────────────────────────────────────────

    @app.route("/api/workspace/save", methods=["POST"])
    def api_workspace_save() -> Any:
        workspace = _body().get("workspace", {})
        try:
            WS = _workspace_file()
            WS.write_text(json.dumps(workspace, indent=2), encoding="utf-8")
            return jsonify({"ok": True})
        except (OSError, TypeError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/workspace/load")
    def api_workspace_load() -> Any:
        WS = _workspace_file()
        if not WS.exists():
            return jsonify({"workspace": None})
        try:
            return jsonify({"workspace": json.loads(WS.read_text(encoding="utf-8"))})
        except (json.JSONDecodeError, OSError):
            return jsonify({"workspace": None})

    @app.errorhandler(413)
    def _too_large(_e: Any) -> Any:
        return _err("Request too large", 413)

    @app.errorhandler(400)
    def _bad(_e: Any) -> Any:
        return _err("Bad request", 400)

    return app


def _workspace_file() -> Path:
    # looked up at call time so tests can monkeypatch WORKSPACE_FILE
    return WORKSPACE_FILE


# Default app (used by `pixlang editor`).  Workspace = CWD at import time.
app = create_app()
