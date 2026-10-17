# pixlang/editor/debugger.py
"""
PixLang Visual Editor — step debugger.

A debug session runs the pipeline in a worker thread under the same security
model as ``/api/run`` (workspace sandbox, include roots, token/Origin guard) and
pauses *inside the executor's* ``_on_statement`` hook, i.e. before a statement
executes.  While paused the worker is parked on a condition variable; the HTTP
side only ever reads an immutable snapshot that the worker built just before
parking, so request threads never race with the interpreter.

Stepping uses two counters maintained by :class:`DebugExecutor`:

* the call-stack depth (``len(executor._frames)``), and
* a *step depth* that additionally counts nested loop bodies and FUNC bodies.

``step_over`` runs until a statement at the same or a lower step depth,
``step_into`` stops at the very next statement, ``step_out`` leaves the current
FUNC frame (or the innermost loop when called from the top level).

Clocks: the run time limit only counts time spent *running* — it is suspended
while the session is paused.  A paused session that nobody touches is stopped
after ``idle_timeout`` seconds so abandoned tabs cannot pin a thread, the run
lock and the working directory forever.

Debugger-evaluated expressions (watch list, breakpoint conditions) are parsed
with the normal expression grammar and then checked: statements, assignments,
commands and calls to user FUNCs are rejected, so evaluating them can never
change interpreter state.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pixlang.executor import memory
from pixlang.executor.engine import _short
from pixlang.parser import parse_expression
from pixlang.parser.ast_nodes import (
    BareIdent,
    Call,
    ForBlock,
    FuncDef,
    ImportStmt,
    RepeatBlock,
    WhileBlock,
    walk_expr,
    walk_statements,
)
from pixlang.parser.errors import PixSyntaxError

from .server import (
    THUMB_MAX_SIDE,
    _as_array,
    _CapturingExecutor,
    _encode_png,
    _image_meta,
    _is_within,
    _pipeline_body,
    _to_display,
    build_run_payload,
    empty_run_payload,
    error_info,
)

# ── Limits ────────────────────────────────────────────────────────────────────

DEFAULT_IDLE_TIMEOUT_S = 600.0      # paused + untouched for this long -> stopped
MAX_BREAKPOINTS = 500
MAX_WATCH = 32
MAX_EXPR_CHARS = 500
MAX_LINE = 1_000_000
DEBUG_IMAGE_MAX_SIDE = 1200         # pause-time images are downscaled to this
MAX_BUFFERS = 24
MAX_CHILDREN = 16
MAX_STDOUT_CHARS = 256 * 1024
IMAGE_CACHE_ENTRIES = 32
KEPT_SESSIONS = 2
EVAL_TIMEOUT_S = 2.0
STATES_FINAL = ("finished", "error", "stopped")
ACTIONS = ("continue", "step_over", "step_into", "step_out", "pause", "stop")

_DENIED_BUILTINS = {"ZEROS"}        # allocates arbitrary amounts of memory
_PASSIVE = (FuncDef, ImportStmt)    # statements that never "execute" anything


class DebugAbort(BaseException):
    """Unwinds the interpreter when a session is stopped.

    A BaseException on purpose: the executor wraps ordinary exceptions raised
    inside statement handlers into PixRuntimeError, which would turn a user
    "Stop" into a bogus runtime error.
    """

    def __init__(self, reason: str = "user") -> None:
        super().__init__(reason)
        self.reason = reason


class DebugError(ValueError):
    """Bad input to the debug API (maps to HTTP 400)."""


class WatchError(Exception):
    """An expression that must not (or cannot) be evaluated."""


# ══════════════════════════════════════════════════════════════════════════════
# Value description
# ══════════════════════════════════════════════════════════════════════════════

def _buffer_text(arr: np.ndarray) -> str:
    dims = "×".join(str(d) for d in arr.shape) or "scalar"
    return f"{dims} · {memory.dtype_name(arr.dtype)}"


def _num_text(v: float) -> str:
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return str(v)
        return format(v, ".10g")
    return str(v)


def _clip(text: str, n: int = 120) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def describe_value(name: str, v: Any, depth: int = 0) -> dict:
    """Compact, JSON-safe description of an interpreter value."""
    if isinstance(v, np.generic):
        v = memory.to_python(v)
    out: dict[str, Any] = {"name": name}
    if isinstance(v, dict) and isinstance(v.get("_image"), np.ndarray):
        arr = v["_image"]
        out.update(type="image", kind="buffer", value=_buffer_text(arr))
    elif isinstance(v, np.ndarray):
        out.update(type=f"buffer({memory.dtype_name(v.dtype)})", kind="buffer",
                   value=_buffer_text(v))
    elif isinstance(v, memory.Ref):
        try:
            text = v.describe()
        except Exception:
            text = "<ptr>"
        out.update(type="pointer", kind="pointer", value=text)
    elif isinstance(v, bool):
        out.update(type="bool", kind="bool", value="true" if v else "false")
    elif isinstance(v, (int, float)):
        out.update(type=type(v).__name__, kind="number", value=_num_text(v))
    elif isinstance(v, str):
        shown = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        out.update(type="str", kind="string", value=_clip('"' + shown + '"', 100))
    elif v is None:
        out.update(type="none", kind="other", value="none")
    elif isinstance(v, (list, tuple)):
        children = []
        if depth < 2:
            children = [describe_value(f"[{i}]", x, depth + 1)
                        for i, x in enumerate(v[:MAX_CHILDREN])]
        inner = ", ".join(c["value"] for c in children[:6])
        more = ", …" if len(v) > 6 else ""
        open_, close = ("[", "]") if isinstance(v, list) else ("(", ")")
        out.update(type=type(v).__name__, kind="list",
                   value=_clip(f"{open_}{inner}{more}{close}", 80),
                   length=len(v), children=children, truncated=len(v) > MAX_CHILDREN)
    else:
        try:
            text = _short(v)
        except Exception:
            text = object.__repr__(v)
        out.update(type=type(v).__name__, kind="other", value=_clip(text, 100))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Image cache (pause-time images and buffer thumbnails)
# ══════════════════════════════════════════════════════════════════════════════

class _ImageCache:
    """Small LRU of display-size images; PNGs are encoded on first request."""

    def __init__(self, limit: int = IMAGE_CACHE_ENTRIES) -> None:
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._by_sig: dict[tuple, str] = {}
        self._lock = threading.Lock()
        self._limit = limit
        self._n = 0

    def put(self, sig: tuple, disp: np.ndarray) -> str:
        with self._lock:
            key = self._by_sig.get(sig)
            if key is not None and key in self._items:
                self._items.move_to_end(key)
                return key
            self._n += 1
            key = f"d{self._n:x}"
            self._items[key] = {"arr": disp, "sig": sig, "png": {}}
            self._by_sig[sig] = key
            while len(self._items) > self._limit:
                _, old = self._items.popitem(last=False)
                self._by_sig.pop(old["sig"], None)
            return key

    def lookup(self, sig: tuple) -> str | None:
        with self._lock:
            key = self._by_sig.get(sig)
            if key is not None and key in self._items:
                self._items.move_to_end(key)
                return key
        return None

    def png(self, key: str, thumb: bool) -> bytes | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            tag = "t" if thumb else "f"
            if tag in item["png"]:
                return item["png"][tag]
            arr = item["arr"]
        data = _encode_png(arr, THUMB_MAX_SIDE if thumb else DEBUG_IMAGE_MAX_SIDE)
        if data is not None:
            with self._lock:
                if key in self._items:
                    self._items[key]["png"][tag] = data
        return data


def _signature(arr: np.ndarray) -> tuple:
    flat = np.ascontiguousarray(arr)
    return (flat.shape, flat.dtype.str, zlib.crc32(flat.data))


# ══════════════════════════════════════════════════════════════════════════════
# Expression safety
# ══════════════════════════════════════════════════════════════════════════════

_LOC_PREFIX = re.compile(r"^<[^>]*>:\d+:\d+:\s*(error:\s*)?")


def parse_safe_expression(text: Any, registry: Any = None) -> Any:
    """Parse *text* as a single PixLang expression, rejecting anything that is
    not a side-effect free expression.  Raises :class:`WatchError`."""
    if not isinstance(text, str) or not text.strip():
        raise WatchError("Enter an expression")
    text = text.strip()
    if len(text) > MAX_EXPR_CHARS:
        raise WatchError(f"Expression is longer than {MAX_EXPR_CHARS} characters")
    if "\n" in text or "\r" in text:
        raise WatchError("Only a single-line expression can be evaluated")
    from pixlang.parser.lexer import KEYWORDS
    first = re.match(r"[A-Za-z_][A-Za-z_0-9]*", text)
    word = first.group(0) if first else ""
    if word.isupper() and word not in ("AS",) and (
            word in KEYWORDS or (registry is not None and word in registry)):
        raise WatchError("Only expressions can be evaluated — "
                         "statements, commands and assignments are not allowed")
    try:
        node = parse_expression(text)
    except PixSyntaxError as exc:
        msg = _LOC_PREFIX.sub("", str(exc))
        if re.search(r"unexpected op '='", msg):
            raise WatchError("Assignments are not allowed in expressions "
                             "(use == to compare)") from None
        raise WatchError(msg) from None
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise WatchError(str(exc)) from None
    for sub in walk_expr(node):
        if isinstance(sub, Call) and sub.func in _DENIED_BUILTINS:
            raise WatchError(f"{sub.func}() is not available in the debugger")
    return node


# ══════════════════════════════════════════════════════════════════════════════
# Executor
# ══════════════════════════════════════════════════════════════════════════════

class DebugExecutor(_CapturingExecutor):
    """:class:`_CapturingExecutor` that parks in the owning session before
    statements run."""

    def __init__(self, *args: Any, session: DebugSession, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.session = session
        self._deep: set[int] = set()       # ids of loop / FUNC body lists
        self.step_depth = 0
        self._evaluating = 0
        # Frames alive when the first runtime error was raised; the executor
        # pops them while unwinding, but the error state wants to show them.
        self.error_frames: list | None = None
        self.error_node: Any = None

    def _on_statement(self, stmt: Any) -> None:
        super()._on_statement(stmt)
        if isinstance(stmt, (RepeatBlock, WhileBlock, ForBlock)):
            self._deep.add(id(stmt.body))
        self.session.on_statement(self, stmt)

    def _exec_block(self, stmts: list) -> None:
        deep = id(stmts) in self._deep
        if deep:
            self.step_depth += 1
        try:
            super()._exec_block(stmts)
        finally:
            if deep:
                self.step_depth -= 1

    def _call_function(self, fn: FuncDef, qual: str, args: list, node: Any) -> Any:
        self._deep.add(id(fn.body))
        return super()._call_function(fn, qual, args, node)

    def _error(self, msg: str, node: Any = None, raw: bool = False) -> Any:
        if self.error_frames is None and not self._evaluating:
            self.error_frames = list(self._frames)
            self.error_node = node if node is not None else self._current
        return super()._error(msg, node, raw)

    # -- read-only evaluation ------------------------------------------------

    def check_names(self, node: Any) -> None:
        """Reject names / calls that would not be pure in the current frame."""
        for sub in walk_expr(node):
            if isinstance(sub, Call):
                qual = self.program.lookup(self._frame.unit, sub.func) if self.program else None
                if qual is not None:
                    raise WatchError(f"'{sub.func}' is a FUNC — user functions can't be "
                                     "called from watch expressions")
            elif isinstance(sub, BareIdent):
                name = sub.name
                found, _ = self._lookup(name)
                if found:
                    continue
                if name == "_":
                    if self.image is None:
                        raise WatchError("_ (current image) is empty")
                    continue
                if self._slot_array(name) is not None:
                    continue
                raise WatchError(f"'{name}' is not defined here")

    def evaluate(self, node: Any) -> Any:
        """Evaluate a vetted expression in the current frame without leaving
        traces on the executor."""
        self.check_names(node)
        saved = self._current
        self._evaluating += 1
        try:
            return self._eval(node)
        finally:
            self._evaluating -= 1
            self._current = saved


# ══════════════════════════════════════════════════════════════════════════════
# Session
# ══════════════════════════════════════════════════════════════════════════════

class _Breakpoint:
    __slots__ = ("line", "text", "node", "error")

    def __init__(self, line: int, text: str | None, registry: Any) -> None:
        self.line = line
        self.text = text
        self.node = None
        self.error: str | None = None
        if text:
            try:
                self.node = parse_safe_expression(text, registry)
            except WatchError as exc:
                self.error = str(exc)


def normalize_breakpoints(raw: Any) -> list[tuple[int, str | None]]:
    """Accept ``[12, {"line": 14, "condition": "$i == 2"}]`` -> [(12, None), ...]."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise DebugError("breakpoints must be a list")
    if len(raw) > MAX_BREAKPOINTS:
        raise DebugError(f"At most {MAX_BREAKPOINTS} breakpoints are supported")
    out: dict[int, str | None] = {}
    for item in raw:
        cond = None
        line = item
        if isinstance(item, dict):
            line = item.get("line")
            cond = item.get("condition")
        if isinstance(line, bool) or not isinstance(line, int) or not 1 <= line <= MAX_LINE:
            raise DebugError("Breakpoint lines must be positive integers")
        if cond is not None:
            if not isinstance(cond, str) or len(cond) > MAX_EXPR_CHARS:
                raise DebugError("Breakpoint conditions must be short strings")
            cond = cond.strip() or None
        out[line] = cond
    return sorted(out.items())


def normalize_watch(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or any(not isinstance(x, str) for x in raw):
        raise DebugError("watch must be a list of expression strings")
    if len(raw) > MAX_WATCH:
        raise DebugError(f"At most {MAX_WATCH} watch expressions are supported")
    out = []
    for x in raw:
        x = x.strip()
        if len(x) > MAX_EXPR_CHARS:
            raise DebugError(f"Watch expressions are limited to {MAX_EXPR_CHARS} characters")
        if x and x not in out:
            out.append(x)
    return out


class _ThreadStdout:
    """sys.stdout replacement: output of the worker thread goes to the
    session buffer, everything else still reaches the real stdout."""

    def __init__(self, buf: Any, tid: int, fallback: Any) -> None:
        self._buf, self._tid, self._fallback = buf, tid, fallback

    def write(self, s: str) -> int:
        if threading.get_ident() == self._tid:
            if self._buf.tell() < MAX_STDOUT_CHARS:
                self._buf.write(s)
            return len(s)
        return self._fallback.write(s)

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._fallback.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fallback, name)


class DebugSession:
    def __init__(self, manager: DebugManager, *, code: str, pipeline: Any,
                 pipeline_path: Path | None, base_dir: Path,
                 breakpoints: list[tuple[int, str | None]], watch: list[str],
                 stop_on_entry: bool) -> None:
        import io
        self.manager = manager
        self.id = uuid.uuid4().hex
        self.code = code
        self.pipeline = pipeline
        self.pipeline_path = pipeline_path
        self.base_dir = base_dir
        self.cond = threading.Condition()
        self.cancel = threading.Event()
        self.images = _ImageCache()
        self.stdout_buf = io.StringIO()
        self.thread: threading.Thread | None = None
        self.executor: DebugExecutor | None = None
        self.loader: Any = None

        self.state = "starting"
        self.seq = 1
        self.result: dict | None = None
        self.stop_reason: str | None = None          # None | user | idle | timeout
        self._finished = False
        self._snap: dict[str, Any] = {}
        self._resume: dict | None = None
        self._pause_req = False
        self._entry = stop_on_entry
        self._mode = "continue"
        self._target_depth = 0
        self._target_frames = 0
        self._done = threading.Event()
        self._created = time.monotonic()
        self._run_acc = 0.0
        self._seg_start: float | None = None
        self._pauses = 0
        self._touched = time.monotonic()      # last user activity (idle clock)
        self._prev_buffers: dict[str, tuple] = {}
        self._prev_image_sig: tuple | None = None
        self.watch_exprs = list(watch)
        self._watch_results: list[dict] = []
        self.stmt_lines = {s.line for s in walk_statements(_pipeline_body(pipeline))
                           if not isinstance(s, _PASSIVE) and s.line}
        self._bps: dict[int, _Breakpoint] = {}
        self._set_bps(breakpoints)

    # ── breakpoints / watch ────────────────────────────────────────────────

    def _set_bps(self, items: list[tuple[int, str | None]]) -> None:
        reg = self.manager.registry
        self._bps = {line: _Breakpoint(line, cond, reg) for line, cond in items}

    def set_breakpoints(self, items: list[tuple[int, str | None]]) -> None:
        with self.cond:
            self._touched = time.monotonic()
            self._set_bps(items)
            self.seq += 1
            self.cond.notify_all()

    def set_watch(self, exprs: list[str]) -> None:
        with self.cond:
            self._touched = time.monotonic()
            self.watch_exprs = list(exprs)
            if self.state == "paused" and self.executor is not None:
                self._watch_results = self._eval_watch(self.executor, guarded=True)
                self._snap = {**self._snap, "watch": self._watch_results}
            else:
                self._watch_results = [{"expr": e, "pending": True} for e in exprs]
            self.seq += 1
            self.cond.notify_all()

    def _eval_watch(self, ex: DebugExecutor, guarded: bool = False) -> list[dict]:
        return [self.evaluate(ex, e, guarded=guarded) for e in self.watch_exprs]

    def evaluate(self, ex: DebugExecutor, text: str, guarded: bool = False) -> dict:
        """Evaluate one expression; never raises."""
        out: dict[str, Any] = {"expr": text}
        try:
            node = parse_safe_expression(text, self.manager.registry)
            if guarded:
                value = _run_with_timeout(lambda: ex.evaluate(node), EVAL_TIMEOUT_S)
            else:
                value = ex.evaluate(node)
            out.update(describe_value(text, value))
            out["expr"] = text
            out["ok"] = True
        except WatchError as exc:
            out.update(ok=False, error=str(exc))
        except Exception as exc:  # PixRuntimeError & friends: show, never crash
            msg = str(getattr(exc, "msg", None) or exc)
            out.update(ok=False, error=_clip(re.sub(r"\.\s*Available:.*$", "", msg), 240))
        return out

    # ── timing ─────────────────────────────────────────────────────────────

    def running_seconds(self) -> float:
        extra = (time.monotonic() - self._seg_start) if self._seg_start is not None else 0.0
        return self._run_acc + extra

    # ── thread lifecycle ───────────────────────────────────────────────────

    def start_thread(self, executor: DebugExecutor, loader: Any) -> None:
        self.executor = executor
        self.loader = loader
        self.thread = threading.Thread(target=self._work, name=f"pixlang-debug-{self.id[:8]}",
                                       daemon=True)
        with self.cond:
            self.state = "running"
            self._seg_start = time.monotonic()
        self.thread.start()
        threading.Thread(target=self._watchdog, name=f"pixlang-debug-wd-{self.id[:8]}",
                         daemon=True).start()

    def _work(self) -> None:
        ex = self.executor
        assert ex is not None
        prev_cwd = os.getcwd()
        prev_stdout = sys.stdout
        proxy = _ThreadStdout(self.stdout_buf, threading.get_ident(), prev_stdout)
        sys.stdout = proxy  # type: ignore[assignment]
        exc: BaseException | None = None
        try:
            if not self.base_dir.exists() and _is_within(self.base_dir, self.manager.root):
                self.base_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(self.base_dir)
            ex.run(self.pipeline)
        except DebugAbort:
            pass
        except BaseException as e:  # noqa: BLE001 — report everything
            exc = e
        finally:
            with contextlib.suppress(OSError):
                os.chdir(prev_cwd)
            if sys.stdout is proxy:
                sys.stdout = prev_stdout
            try:
                self._finish(exc)
            finally:
                self.manager.run_lock.release()

    def _watchdog(self) -> None:
        tick = self.manager.tick
        while not self._done.wait(tick):
            with self.cond:
                if self._finished:
                    return
                if self.state == "running" and \
                        self.running_seconds() > self.manager.timeout_fn():
                    self.stop_reason = self.stop_reason or "timeout"
                    self.cancel.set()
                    self.cond.notify_all()
                    stuck = True
                else:
                    stuck = False
            if stuck:
                # Give the worker a moment to unwind at its next statement;
                # if it is stuck inside one long command, report right away.
                t = self.thread
                if t is not None:
                    t.join(1.0)
                self._finish(None)
                return

    def _finish(self, exc: BaseException | None) -> None:
        with self.cond:
            if self._finished:
                return
            self._finished = True
            if self._seg_start is not None:
                self._run_acc += time.monotonic() - self._seg_start
                self._seg_start = None
            stop = self.stop_reason
        ex = self.executor
        assert ex is not None
        info_override = None
        timed_out = stop == "timeout"
        if stop == "idle":
            minutes = self.manager.idle_timeout / 60
            span = f"{minutes:g} minutes" if minutes >= 1 else f"{self.manager.idle_timeout:g} s"
            info_override = {"kind": "idle", "line": None, "col": None, "file": None,
                             "type": "Idle", "raw": "",
                             "message": f"Debug session stopped after {span} of inactivity"}
        if stop in ("user", "idle") and ex.final_image is None:
            ex.final_image = ex.image
        use_exc = None if stop else exc
        # Executor.run() reports wall time; a debug session only counts the
        # time it actually spent running (pauses excluded).
        ex.context["stats"]["total_ms"] = round(self._run_acc * 1000, 3)
        try:
            payload = build_run_payload(
                run_id=uuid.uuid4().hex, executor=ex, code=self.code, exc=use_exc,
                timed_out=timed_out, timeout=self.manager.timeout_fn(),
                stdout=self.stdout_buf.getvalue(),
                wall_ms=(time.monotonic() - self._created) * 1000,
                results=self.manager.results, root=self.manager.root,
                pipeline_path=self.pipeline_path, loader=self.loader,
                info_override=info_override)
        except Exception as err:  # never leave a session hanging
            payload = {**empty_run_payload(uuid.uuid4().hex), "ok": False,
                       "error": str(err),
                       "error_info": {"kind": "runtime", "message": str(err), "line": None,
                                      "col": None, "file": None, "raw": "", "type": "Error"}}
        if stop in ("user", "idle"):
            payload["stopped"] = True
        payload["stop_reason"] = stop
        if stop == "user":
            state = "stopped"
        elif stop == "idle":
            state = "stopped"
        elif payload.get("error_info"):
            state = "error"
        else:
            state = "finished"
        final = {}
        if state != "stopped" or ex.image is not None:
            with contextlib.suppress(Exception):
                if state == "error" and ex.error_frames:
                    final = self._build_snapshot(ex, ex.error_node, "error", final=True,
                                                 frames=ex.error_frames)
                else:
                    final = self._build_snapshot(ex, None, "end", final=True)
        with self.cond:
            self.state = state
            self._snap = final
            self.result = payload
            self.seq += 1
            self.cond.notify_all()
        self._done.set()

    # ── executor hook (worker thread) ──────────────────────────────────────

    def on_statement(self, ex: DebugExecutor, stmt: Any) -> None:
        if self.stop_reason:
            raise DebugAbort(self.stop_reason)
        if stmt.file is not None or isinstance(stmt, _PASSIVE):
            return                       # statements of other files never pause
        line = stmt.line
        reason = None
        message = None
        bp = self._bps.get(line)
        if bp is not None:
            if bp.text is None:
                reason = "breakpoint"
            elif bp.node is None:
                reason, message = "condition_error", f"Breakpoint condition: {bp.error}"
            else:
                try:
                    value = ex.evaluate(bp.node)
                    if isinstance(value, np.ndarray):
                        value = bool(value.any())
                    if memory.truth(value):
                        reason = "breakpoint"
                except WatchError as exc:
                    reason, message = "condition_error", f"Breakpoint condition: {exc}"
                except Exception as exc:
                    reason = "condition_error"
                    message = "Breakpoint condition: " + _clip(
                        str(getattr(exc, "msg", None) or exc), 200)
        if reason is None:
            if self._pause_req:
                self._pause_req = False
                reason = "pause"
            elif self._entry:
                reason = "entry"
            elif self._mode == "into":
                reason = "step"
            elif self._mode == "over" and ex.step_depth <= self._target_depth:
                reason = "step"
            elif self._mode == "out_frame" and len(ex._frames) <= self._target_frames:
                reason = "step"
            elif self._mode == "out_loop" and ex.step_depth <= self._target_depth:
                reason = "step"
        if reason is not None:
            self._entry = False
            self._park(ex, stmt, reason, message)

    def _park(self, ex: DebugExecutor, stmt: Any, reason: str, message: str | None) -> None:
        self._pauses += 1
        snap = self._build_snapshot(ex, stmt, reason, message=message)
        idle = self.manager.idle_timeout
        with self.cond:
            if self._seg_start is not None:
                self._run_acc += time.monotonic() - self._seg_start
                self._seg_start = None
            self.state = "paused"
            self._snap = snap
            self._resume = None
            self._touched = time.monotonic()
            self.seq += 1
            self.cond.notify_all()
            while self._resume is None and not self.stop_reason:
                remaining = self._touched + idle - time.monotonic()
                if remaining <= 0:
                    self.stop_reason = "idle"
                    self.cancel.set()
                    break
                self.cond.wait(min(remaining, 1.0))
            if self.stop_reason:
                raise DebugAbort(self.stop_reason)
            action = self._resume
        assert action is not None
        kind = action["action"]
        self._mode = "continue"
        if kind == "step_into":
            self._mode = "into"
        elif kind == "step_over":
            self._mode, self._target_depth = "over", ex.step_depth
        elif kind == "step_out":
            if len(ex._frames) > 1:
                self._mode, self._target_frames = "out_frame", len(ex._frames) - 1
            elif ex.step_depth > 0:
                self._mode, self._target_depth = "out_loop", ex.step_depth - 1

    # ── snapshot (worker thread, or finish) ────────────────────────────────

    def _image_ref(self, arr: Any, label: str | None = None,
                   sig: tuple | None = None) -> dict | None:
        if not isinstance(arr, np.ndarray) or arr.size == 0 or arr.ndim not in (2, 3) \
                or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)):
            return None                  # not displayable as a picture
        sig = sig or _signature(arr)
        key = self.images.lookup(sig)
        if key is None:
            disp = _to_display(arr, DEBUG_IMAGE_MAX_SIDE)
            if np.shares_memory(disp, arr):
                disp = disp.copy()
            key = self.images.put(sig, disp)
        ref = {**_image_meta(arr), "key": key,
               "url": f"/api/result/{self.id}/{key}.png",
               "thumb_url": f"/api/result/{self.id}/{key}.thumb.png"}
        if label:
            ref["label"] = label
        return ref

    def _build_snapshot(self, ex: DebugExecutor, stmt: Any, reason: str, *,
                        message: str | None = None, final: bool = False,
                        frames: list | None = None) -> dict:
        keep_stack = frames is not None or not final
        frames = list(ex._frames) if frames is None else list(frames)
        n = len(frames)
        stack = []
        variables_by_frame = []
        for i in range(n - 1, -1, -1):
            fr = frames[i]
            if i == n - 1 and stmt is not None:
                line, col, file = stmt.line, stmt.col, stmt.file
            elif i + 1 < n:
                cn = frames[i + 1].call_node
                line, col, file = getattr(cn, "line", 0), getattr(cn, "col", 0), \
                    getattr(cn, "file", None)
            else:
                line, col, file = 0, 0, None
            if fr.func is not None:
                args = ", ".join(f"{p}={_clip(_short(fr.locals.get(p)), 24)}"
                                 for p in fr.func.params)
                label = f"{fr.name}({args})"
                kind = "func"
            elif fr.name == "<main>":
                label, kind = "main", "main"
            else:
                label, kind = fr.name.strip("<>"), "init"
            stack.append({"index": len(stack), "name": label, "kind": kind,
                          "line": line or None, "col": col or None, "file": file,
                          "unit": Path(fr.unit).name if fr.unit else None})
            local_vars = []
            if fr.func is not None:
                local_vars += [{**describe_value(k, v), "scope": "local"}
                               for k, v in fr.locals.items()]
            for scope in fr.scopes:
                local_vars += [{**describe_value(k, v), "scope": "loop"}
                               for k, v in scope.items()]
            variables_by_frame.append(local_vars)

        loop_names = {v["name"] for fv in variables_by_frame for v in fv if v["scope"] == "loop"}
        gvars = [describe_value(k, v) for k, v in sorted(ex.context["vars"].items())
                 if k not in loop_names]

        buffers = []
        seen: dict[str, tuple] = {}
        for name, entry in list(ex.context["checkpoints"].items())[:MAX_BUFFERS]:
            arr = memory.slot_array(entry)
            if arr is None:
                continue
            sig = _signature(arr)
            ref = self._image_ref(arr, sig=sig)
            seen[name] = sig
            buffers.append({
                "name": str(name), "type": f"buffer({memory.dtype_name(arr.dtype)})",
                "value": _buffer_text(arr), "shape": list(arr.shape),
                "dtype": memory.dtype_name(arr.dtype),
                "gen": ex.context["buffer_gen"].get(name, 0),
                "changed": self._prev_buffers.get(name) != sig and not final,
                "new": name not in self._prev_buffers and not final,
                **({k: ref[k] for k in ("key", "url", "thumb_url", "width", "height",
                                        "channels")} if ref else {}),
            })
        self._prev_buffers = seen

        image = self._image_ref(_as_array(ex.image), label="_")
        before = self._image_ref(_as_array(ex.before_image), label="Input")
        watch = self._eval_watch(ex)
        self._watch_results = watch
        raw = ex.context.get("stats", {})
        stats = {
            "commands_executed": int(raw.get("commands_executed", 0)),
            "total_ms": float(raw.get("total_ms", 0.0)),
            "line_times": {str(k): {"ms": float(v["ms"]), "count": int(v["count"]),
                                    "name": v["name"]}
                           for k, v in sorted(ex.line_times.items())},
        }
        return {
            "pause": self._pauses,
            "reason": reason,
            "message": message,
            "location": None if stmt is None else {
                "line": stmt.line or None, "col": stmt.col or None,
                "statement": type(stmt).__name__},
            "stack": stack if keep_stack else [],
            "variables": {"globals": gvars, "frames": variables_by_frame},
            "buffers": buffers,
            "image": image,
            "before": before,
            "watch": watch,
            "stats": stats,
            "stdout": self.stdout_buf.getvalue()[-MAX_STDOUT_CHARS:],
            "visited_lines": sorted(ex.visited),
        }

    # ── control (request threads) ──────────────────────────────────────────

    def control(self, action: str, settle: float = 1.0) -> dict:
        if action not in ACTIONS:
            raise DebugError(f"Unknown action '{action}'")
        if action == "stop":
            self.stop()
            return self.view()
        with self.cond:
            if action == "pause":
                if self.state == "running":
                    self._pause_req = True
                    seq = self.seq
                    self._wait_locked(lambda: self.seq != seq or self.state != "running",
                                      settle)
                return self._view_locked()
            if self.state != "paused":
                raise PermissionError("Session is not paused")
            self._resume = {"action": action}
            self.state = "running"
            self._seg_start = time.monotonic()
            self.seq += 1
            self.cond.notify_all()
            self._wait_locked(lambda: self.state != "running", settle)
            return self._view_locked()

    def stop(self, join_timeout: float = 3.0) -> None:
        with self.cond:
            if self.state in STATES_FINAL:
                return
            self.stop_reason = self.stop_reason or "user"
            self.cancel.set()
            self.cond.notify_all()
        t = self.thread
        if t is None:
            self._finish(None)
            return
        if t is not threading.current_thread():
            t.join(join_timeout)
        if t.is_alive():
            self._finish(None)

    def _wait_locked(self, pred: Callable[[], bool], timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        while not pred():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.cond.wait(remaining)

    def wait_for_change(self, after: int, timeout: float) -> dict:
        """Long-poll: block until ``seq > after`` (or the session is final)."""
        with self.cond:
            self._wait_locked(lambda: self.seq > after or self.state in STATES_FINAL, timeout)
            return self._view_locked()

    # ── views ──────────────────────────────────────────────────────────────

    def view(self) -> dict:
        with self.cond:
            return self._view_locked()

    def _view_locked(self) -> dict:
        bps = [{"line": b.line, "condition": b.text, "verified": b.line in self.stmt_lines,
                "error": b.error}
               for b in sorted(self._bps.values(), key=lambda b: b.line)]
        d: dict[str, Any] = {
            "session_id": self.id, "state": self.state, "seq": self.seq,
            "running_ms": round(self.running_seconds() * 1000, 1),
            "idle_timeout": self.manager.idle_timeout,
            "breakpoints": bps,
            "watch": [{"expr": e, "pending": True} for e in self.watch_exprs],
            "pause": 0, "reason": None, "message": None, "location": None, "stack": [],
            "variables": {"globals": [], "frames": []}, "buffers": [], "image": None,
            "before": None, "stats": None, "stdout": "", "visited_lines": [],
        }
        d.update(self._snap)
        if self.result is not None:
            d["result"] = self.result
        return d

    def png(self, name: str, thumb: bool) -> bytes | None:
        return self.images.png(name, thumb)

    def evaluate_paused(self, text: str) -> dict:
        """Evaluate *text* in the paused frame (request threads)."""
        with self.cond:
            if self.state != "paused" or self.executor is None:
                return {"expr": text, "ok": False,
                        "error": "Expressions can only be evaluated while paused"}
            self._touched = time.monotonic()
            return self.evaluate(self.executor, text, guarded=True)


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["v"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["e"] = exc

    t = threading.Thread(target=target, daemon=True, name="pixlang-debug-eval")
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise WatchError(f"Evaluation took longer than {timeout:g}s and was abandoned")
    if "e" in box:
        raise box["e"]
    return box["v"]


# ══════════════════════════════════════════════════════════════════════════════
# Manager (one active session at a time)
# ══════════════════════════════════════════════════════════════════════════════

class DebugManager:
    def __init__(self, *, registry: Any, root: Path, run_lock: threading.Lock,
                 results: Any, timeout_fn: Callable[[], float],
                 idle_timeout: float = DEFAULT_IDLE_TIMEOUT_S, tick: float = 0.1,
                 plugins: bool = True) -> None:
        self.registry = registry
        self.root = root
        self.run_lock = run_lock
        self.results = results
        self.timeout_fn = timeout_fn
        self.idle_timeout = idle_timeout
        self.tick = tick
        self.plugins = plugins
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, DebugSession] = OrderedDict()

    def get(self, sid: str) -> DebugSession | None:
        with self._lock:
            return self._sessions.get(sid)

    def active(self) -> DebugSession | None:
        with self._lock:
            for s in reversed(self._sessions.values()):
                if s.state not in STATES_FINAL:
                    return s
        return None

    def png(self, sid: str, name: str, thumb: bool) -> bytes | None:
        s = self.get(sid)
        return s.png(name, thumb) if s is not None else None

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            s.stop()

    def start(self, *, code: str, pipeline: Any, pipeline_path: Path | None,
              base_dir: Path, include_roots: list[Path],
              breakpoints: list[tuple[int, str | None]], watch: list[str],
              stop_on_entry: bool | None) -> DebugSession:
        """Stop any running session, then start a new one.

        Raises :class:`RuntimeError` when the interpreter is busy with
        something else (a normal run, or a session that will not unwind).
        """
        old = self.active()
        if old is not None:
            old.stop()
        if not self.run_lock.acquire(timeout=3.0):
            raise RuntimeError("Another run is still in progress")
        try:
            entry = (not breakpoints) if stop_on_entry is None else bool(stop_on_entry)
            sess = DebugSession(self, code=code, pipeline=pipeline,
                                pipeline_path=pipeline_path, base_dir=base_dir,
                                breakpoints=breakpoints, watch=watch, stop_on_entry=entry)
            loader = None
            if self.plugins:
                try:
                    from pixlang.plugins import PluginLoader
                    loader = PluginLoader(self.registry)
                except Exception:
                    loader = None
            ex = DebugExecutor(
                registry=self.registry, verbose=False, plugin_loader=loader,
                pipeline_path=pipeline_path, base_dir=base_dir, workspace_root=self.root,
                include_roots=include_roots, cancel=sess.cancel, session=sess)
            with self._lock:
                self._sessions[sess.id] = sess
                self._trim()
            sess.start_thread(ex, loader)       # releases run_lock when done
        except BaseException:
            self.run_lock.release()
            raise
        return sess

    def failed(self, *, code: str, exc: SyntaxError, pipeline: Any = None) -> DebugSession:
        """A session that never started (syntax error): terminal from the get-go."""
        sess = DebugSession(self, code=code, pipeline=pipeline or _EMPTY_PIPELINE(),
                            pipeline_path=None, base_dir=self.root, breakpoints=[],
                            watch=[], stop_on_entry=False)
        info = error_info(exc, code, "syntax")
        payload = {**empty_run_payload(uuid.uuid4().hex), "error": str(exc),
                   "error_info": info, "stop_reason": None}
        with sess.cond:
            sess.state = "error"
            sess.result = payload
            sess._finished = True
            sess.seq += 1
        sess._done.set()
        with self._lock:
            self._sessions[sess.id] = sess
            self._trim()
        return sess

    def _trim(self) -> None:
        finals = [sid for sid, s in self._sessions.items() if s.state in STATES_FINAL]
        while len(finals) > KEPT_SESSIONS:
            self._sessions.pop(finals.pop(0), None)


def _EMPTY_PIPELINE() -> Any:
    from pixlang.parser.ast_nodes import Pipeline
    return Pipeline(commands=[])
