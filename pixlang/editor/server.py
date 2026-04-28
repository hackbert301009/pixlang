# pixlang/editor/server.py
"""
PixLang Visual Editor — Flask backend.

Provides 9 API endpoints for the single-page editor UI.
"""
from __future__ import annotations

import base64
import contextlib
import inspect
import io
import json
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from pixlang.commands import registry as pxl_registry
from pixlang.commands.builtin import _unwrap
from pixlang.executor.engine import Executor
from pixlang.linter import Linter
from pixlang.parser import parse as pxl_parse
from pixlang.parser.ast_nodes import (
    AssertStmt, Command, IfBlock, IncludeStmt, Pipeline,
    RepeatBlock, RoiBlock, SetVar,
)

# ── Path constants ────────────────────────────────────────────────────────────

STATIC_DIR     = Path(__file__).parent / "static"
EXAMPLES_DIR   = Path(__file__).parent.parent.parent / "examples"
WORKSPACE_FILE = Path.home() / ".pixlang_editor_workspace.json"

if not EXAMPLES_DIR.exists():
    EXAMPLES_DIR = Path.cwd() / "examples"

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)

# ── Command category map ──────────────────────────────────────────────────────

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


# ── Image capture executor ────────────────────────────────────────────────────

class _CapturingExecutor(Executor):
    """Executor subclass that retains the final image after run()."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.final_image: Any = None

    def run(self, pipeline: Pipeline) -> None:
        self._load_plugins()
        start = time.perf_counter()
        self.final_image = self._run_body(pipeline.commands, None)
        self.context["stats"]["total_ms"] = (time.perf_counter() - start) * 1000


def _image_to_b64(image: Any) -> str | None:
    """Convert a pixlang image (ndarray or wrapped dict) to a base64 PNG string."""
    import cv2
    if image is None:
        return None
    try:
        img, _ = _unwrap(image)
    except Exception:
        return None
    if img is None:
        return None
    # Convert grayscale to BGR for consistent browser display
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Cap preview at 1200px on longest side (original SAVE output unaffected)
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── AST → graph serialization ─────────────────────────────────────────────────

_id_counter = 0


def _new_id(prefix: str = "n") -> str:
    global _id_counter
    _id_counter += 1
    return f"{prefix}_{_id_counter}"


def _reset_id() -> None:
    global _id_counter
    _id_counter = 0


def _serialize_stmts(stmts: list) -> tuple[list, list, str | None, str | None]:
    """
    Recursively serialize a list of AST statements into graph nodes and edges.
    Returns (nodes, edges, first_node_id, last_node_id).
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    first_id: str | None = None
    last_id:  str | None = None

    def _add_node(node: dict) -> str:
        nodes.append(node)
        return node["id"]

    def _chain(from_id: str | None, to_id: str) -> None:
        if from_id is not None:
            edges.append({"from": from_id, "to": to_id})

    for stmt in stmts:
        if isinstance(stmt, Command):
            nid = _new_id("cmd")
            args_preview = " ".join(str(a) for a in stmt.args[:2]) if stmt.args else ""
            label = f"{stmt.name}" + (f" {args_preview}" if args_preview else "")
            _add_node({
                "id": nid, "type": "command",
                "label": label, "line": stmt.line,
                "category": _CMD_TO_CATEGORY.get(stmt.name, "Other"),
                "name": stmt.name,
            })
            _chain(last_id, nid)
            if first_id is None:
                first_id = nid
            last_id = nid

        elif isinstance(stmt, SetVar):
            nid = _new_id("set")
            _add_node({
                "id": nid, "type": "setvar",
                "label": f"SET {stmt.var_name} = {stmt.value!r}",
                "line": stmt.line, "category": "Variable",
            })
            _chain(last_id, nid)
            if first_id is None:
                first_id = nid
            last_id = nid

        elif isinstance(stmt, IfBlock):
            hid = _new_id("if")
            _add_node({
                "id": hid, "type": "if_header",
                "label": f"IF {stmt.var_name} {stmt.op} {stmt.cmp_value!r}",
                "line": stmt.line, "category": "Control",
            })
            _chain(last_id, hid)
            if first_id is None:
                first_id = hid

            fid = _new_id("endif")
            _add_node({"id": fid, "type": "if_footer", "label": "ENDIF",
                       "line": stmt.line, "category": "Control"})

            if stmt.body:
                sub_nodes, sub_edges, bfirst, blast = _serialize_stmts(stmt.body)
                nodes.extend(sub_nodes)
                edges.extend(sub_edges)
                if bfirst:
                    edges.append({"from": hid, "to": bfirst, "label": "true"})
                if blast:
                    edges.append({"from": blast, "to": fid})
            edges.append({"from": hid, "to": fid, "label": "false"})
            last_id = fid

        elif isinstance(stmt, RepeatBlock):
            hid = _new_id("repeat")
            _add_node({
                "id": hid, "type": "repeat_header",
                "label": f"REPEAT {stmt.count!r}",
                "line": stmt.line, "category": "Control",
            })
            _chain(last_id, hid)
            if first_id is None:
                first_id = hid

            fid = _new_id("end")
            _add_node({"id": fid, "type": "repeat_footer", "label": "END",
                       "line": stmt.line, "category": "Control"})

            if stmt.body:
                sub_nodes, sub_edges, bfirst, blast = _serialize_stmts(stmt.body)
                nodes.extend(sub_nodes)
                edges.extend(sub_edges)
                if bfirst:
                    edges.append({"from": hid, "to": bfirst, "label": "body"})
                if blast:
                    edges.append({"from": blast, "to": fid})
                    edges.append({"from": fid, "to": bfirst, "label": "loop"})
            edges.append({"from": hid, "to": fid, "label": "exit"})
            last_id = fid

        elif isinstance(stmt, IncludeStmt):
            nid = _new_id("inc")
            _add_node({
                "id": nid, "type": "include",
                "label": f'INCLUDE "{stmt.path}"',
                "line": stmt.line, "category": "Include",
            })
            _chain(last_id, nid)
            if first_id is None:
                first_id = nid
            last_id = nid

        elif isinstance(stmt, AssertStmt):
            nid = _new_id("assert")
            _add_node({
                "id": nid, "type": "assert",
                "label": f"ASSERT {stmt.subject} {stmt.op} {stmt.expected}",
                "line": stmt.line, "category": "Assert",
            })
            _chain(last_id, nid)
            if first_id is None:
                first_id = nid
            last_id = nid

        elif isinstance(stmt, RoiBlock):
            hid = _new_id("roi")
            _add_node({
                "id": hid, "type": "roi_header",
                "label": f"ROI {stmt.x},{stmt.y} {stmt.w}×{stmt.h}",
                "line": stmt.line, "category": "ROI",
            })
            _chain(last_id, hid)
            if first_id is None:
                first_id = hid

            fid = _new_id("roi_reset")
            _add_node({"id": fid, "type": "roi_footer", "label": "ROI_RESET",
                       "line": stmt.line, "category": "ROI"})

            if stmt.body:
                sub_nodes, sub_edges, bfirst, blast = _serialize_stmts(stmt.body)
                nodes.extend(sub_nodes)
                edges.extend(sub_edges)
                if bfirst:
                    edges.append({"from": hid, "to": bfirst})
                if blast:
                    edges.append({"from": blast, "to": fid})
            else:
                edges.append({"from": hid, "to": fid})
            last_id = fid

    return nodes, edges, first_id, last_id


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index() -> Any:
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/run", methods=["POST"])
def api_run() -> Any:
    data     = request.get_json(force=True, silent=True) or {}
    code     = data.get("code", "")
    filename = data.get("filename", "untitled.pxl")

    try:
        pipeline = pxl_parse(code)
    except SyntaxError as exc:
        return jsonify({"image_b64": None, "stdout": "", "stats": {}, "error": str(exc)})

    executor = _CapturingExecutor(
        registry=pxl_registry,
        verbose=False,
        pipeline_path=Path(filename) if filename else None,
    )

    stdout_buf = io.StringIO()
    error_msg: str | None = None
    try:
        with contextlib.redirect_stdout(stdout_buf):
            executor.run(pipeline)
    except Exception as exc:
        error_msg = str(exc)

    # Make stats JSON-serialisable (strip numpy types)
    raw_stats = executor.context.get("stats", {})
    stats = {
        "commands_executed": int(raw_stats.get("commands_executed", 0)),
        "total_ms":          float(raw_stats.get("total_ms", 0.0)),
        "command_times":     {
            k: [float(v) for v in vs]
            for k, vs in raw_stats.get("command_times", {}).items()
        },
    }

    return jsonify({
        "image_b64": _image_to_b64(executor.final_image),
        "stdout":    stdout_buf.getvalue(),
        "stats":     stats,
        "error":     error_msg,
    })


@app.route("/api/lint", methods=["POST"])
def api_lint() -> Any:
    data = request.get_json(force=True, silent=True) or {}
    code = data.get("code", "")

    try:
        pipeline = pxl_parse(code)
    except SyntaxError as exc:
        return jsonify({"issues": [
            {"line": 0, "severity": "error", "code": "PARSE", "message": str(exc)}
        ]})

    linter = Linter(pxl_registry)
    diags  = linter.lint(pipeline)
    issues = [
        {"line": d.line, "severity": d.severity.value,
         "code": d.code,  "message": d.message}
        for d in diags
    ]
    return jsonify({"issues": issues})


@app.route("/api/parse", methods=["POST"])
def api_parse() -> Any:
    data = request.get_json(force=True, silent=True) or {}
    code = data.get("code", "")

    if not code.strip():
        return jsonify({"nodes": [], "edges": []})

    try:
        pipeline = pxl_parse(code)
    except SyntaxError as exc:
        return jsonify({"error": str(exc), "nodes": [], "edges": []})

    _reset_id()
    nodes, edges, _, _ = _serialize_stmts(pipeline.commands)
    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/commands")
def api_commands() -> Any:
    result = []
    for info in pxl_registry.list_info():
        sig    = inspect.signature(info.fn)
        params = [
            p for name, p in sig.parameters.items()
            if name not in ("image", "_ctx")
        ]
        if params:
            parts = []
            for p in params:
                ann = p.annotation
                ann_str = ann.__name__ if (ann != inspect.Parameter.empty and hasattr(ann, "__name__")) else ""
                default = (f"={p.default!r}" if p.default != inspect.Parameter.empty else "")
                parts.append(f"{p.name}{': ' + ann_str if ann_str else ''}{default}")
            sig_str = f"{info.name} " + "  ".join(parts)
        else:
            sig_str = info.name

        full_doc = info.fn.__doc__ or ""
        lines    = [ln.strip() for ln in full_doc.strip().splitlines() if ln.strip()]
        doc_line = lines[0] if lines else ""
        example  = next(
            (ln for ln in lines if ln.startswith(info.name)),
            info.name,
        )

        result.append({
            "name":      info.name,
            "category":  _CMD_TO_CATEGORY.get(info.name, "Other"),
            "signature": sig_str,
            "doc":       doc_line,
            "example":   example,
        })
    return jsonify({"commands": result})


@app.route("/api/examples")
def api_examples() -> Any:
    examples = []
    for p in sorted(EXAMPLES_DIR.glob("*.pxl")):
        desc = ""
        for line in p.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                desc = stripped.lstrip("#").strip()
                if desc:
                    break
        examples.append({"name": p.stem, "description": desc})
    return jsonify({"examples": examples})


@app.route("/api/examples/<name>")
def api_example_code(name: str) -> Any:
    safe = "".join(c for c in name if c.isalnum() or c == "_")
    path = EXAMPLES_DIR / f"{safe}.pxl"
    if not path.exists():
        return jsonify({"error": f"Example '{safe}' not found"}), 404
    return jsonify({"code": path.read_text(encoding="utf-8")})


@app.route("/api/workspace/save", methods=["POST"])
def api_workspace_save() -> Any:
    data      = request.get_json(force=True, silent=True) or {}
    workspace = data.get("workspace", {})
    try:
        WORKSPACE_FILE.write_text(json.dumps(workspace, indent=2), encoding="utf-8")
        return jsonify({"ok": True})
    except OSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/workspace/load")
def api_workspace_load() -> Any:
    if not WORKSPACE_FILE.exists():
        return jsonify({"workspace": None})
    try:
        workspace = json.loads(WORKSPACE_FILE.read_text(encoding="utf-8"))
        return jsonify({"workspace": workspace})
    except (json.JSONDecodeError, OSError):
        return jsonify({"workspace": None})
