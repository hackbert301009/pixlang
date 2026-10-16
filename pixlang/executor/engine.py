"""
PixLang Execution Engine  v0.5

Runs a LinkedProgram (see pixlang.linker) — or a bare Pipeline, which is
linked on the fly (INCLUDE then stays a run-time statement resolved relative
to the including file, exactly like v0.4).

State
  self.image                 the current-image register ``_``
  context["vars"]            GLOBAL variables (commands may read/write them)
  context["checkpoints"]     named buffer slots (CHECKPOINT, BUF, ALLOC);
                             also reachable as context["buffers"]
  context["buffer_gen"]      slot generation counters (pointer validation)

Scoping
  Each FUNC call gets its own frame: SET writes locals unless the name was
  declared GLOBAL; reads fall back to globals. Loop variables ($ITER, $ITER1,
  FOR / REPEAT ... AS names) live in a per-loop scope and disappear when the
  loop ends; $ITER / $ITER1 always describe the innermost REPEAT.

Errors
  PixRuntimeError (RuntimeError) with .line .col .file and a PixLang call
  stack; failed ASSERTs raise PixAssertionError (AssertionError).
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from pixlang.commands.builtin import _unwrap, _wrap
from pixlang.commands.registry import CommandRegistry
from pixlang.parser.ast_nodes import (
    AllocStmt,
    AssertStmt,
    AsType,
    Attr,
    BareIdent,
    Binary,
    BreakStmt,
    BufStmt,
    Call,
    Command,
    ContinueStmt,
    Expr,
    ForBlock,
    FreeStmt,
    FuncDef,
    GlobalStmt,
    IfBlock,
    ImportStmt,
    IncludeStmt,
    Index,
    Literal,
    Pipeline,
    RepeatBlock,
    ReturnStmt,
    RoiBlock,
    SetVar,
    Slice,
    Statement,
    Unary,
    UseStmt,
    VarRef,
    WhileBlock,
    WithBlock,
    expr_source,
)
from pixlang.parser.errors import PixAssertionError, PixRuntimeError, PixSyntaxError, format_loc

from . import memory
from .builtins import BUILTIN_FUNCTIONS
from .memory import Ref

if TYPE_CHECKING:
    from pixlang.linker import LinkedProgram
    from pixlang.plugins.loader import PluginLoader


# Legacy ASSERT subjects measured on the current image
IMAGE_SUBJECTS = ("width", "height", "channels", "depth", "min", "max",
                  "contour_count", "ndim")

# Kept for backwards compatibility (v0.3 imported these from here)
_OPS = {op: (lambda o: (lambda a, b: memory.compare(o, a, b)))(op)
        for op in ("==", "!=", "<", ">", "<=", ">=")}


# ── Control-flow signals ──────────────────────────────────────────────────────

class _Signal(Exception):
    """Base for non-error control transfer."""


class _BreakSignal(_Signal):
    pass


class _ContinueSignal(_Signal):
    pass


class _ReturnSignal(_Signal):
    def __init__(self, value: Any):
        super().__init__()
        self.value = value


# ── Frames ────────────────────────────────────────────────────────────────────

class _Frame:
    """One activation record."""

    def __init__(self, name: str, unit: Optional[str], top: bool,
                 call_node: Any = None, func: Optional[FuncDef] = None):
        self.name = name
        self.unit = unit
        self.top = top
        self.call_node = call_node
        self.func = func
        self.locals: Dict[str, Any] = {}
        self.globals_declared: set = set()
        self.scopes: List[Dict[str, Any]] = []   # loop-variable scopes


class Executor:
    MAX_CALL_DEPTH = 256
    DEFAULT_MAX_LOOP_ITERATIONS = 1_000_000

    def __init__(
        self,
        registry: CommandRegistry,
        verbose: bool = False,
        plugin_loader: Optional[PluginLoader] = None,
        pipeline_path: Optional[Path] = None,
        max_loop_iterations: Optional[int] = None,
    ):
        self.registry      = registry
        self.verbose       = verbose
        self.plugin_loader = plugin_loader
        self.pipeline_path = Path(pipeline_path) if pipeline_path else None
        self.max_loop_iterations = max_loop_iterations

        # Shared mutable state threaded through the entire pipeline
        self.context: Dict[str, Any] = {
            "checkpoints": {},   # CHECKPOINT / OVERLAY / BUF / ALLOC
            "vars":        {},   # global variables
            "stats":       {     # used by PIPELINE_STATS
                "commands_executed": 0,
                "total_ms":          0.0,
                "command_times":     {},
            },
            "buffer_gen":  {},   # slot -> generation counter
            "buffer_freed": {},  # slot -> line where it was FREEd
        }
        self.context["buffers"] = self.context["checkpoints"]

        self.image: Any = None
        self.program: Optional[LinkedProgram] = None
        self._linker = None
        self._frames: List[_Frame] = []
        self._include_stack: List[str] = []
        self._current: Any = None          # statement being executed

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

    def run(self, pipeline: Union[Pipeline, LinkedProgram]) -> Any:
        """Execute a Pipeline or LinkedProgram; returns the final image."""
        self._load_plugins()
        total_start = time.perf_counter()

        with self._deep_recursion():
            self._prepare(pipeline)
            self.image = None
            self._run_program()

        total_ms = (time.perf_counter() - total_start) * 1000
        self.context["stats"]["total_ms"] = total_ms
        if self.verbose:
            print(f"\n  Pipeline finished in {total_ms:.1f} ms  "
                  f"({self.context['stats']['commands_executed']} commands)")
        return self.image

    def _prepare(self, pipeline: Any) -> None:
        """Link a bare Pipeline (lazily, INCLUDE stays a runtime statement)."""
        from pixlang.linker import LinkedProgram
        cfg = None
        if self.pipeline_path is not None:
            from pixlang.config import load_config
            cfg = load_config(self.pipeline_path.resolve())
        if self.max_loop_iterations is None:
            self.max_loop_iterations = (cfg.max_loop_iterations if cfg
                                        else self.DEFAULT_MAX_LOOP_ITERATIONS)
        linker = self._get_linker(cfg)
        if isinstance(pipeline, LinkedProgram):
            self.program = pipeline
        else:
            self.program = linker.link_pipeline(
                pipeline, path=self.pipeline_path, splice_includes=False)
        self._frames = [_Frame("<main>", self.program.entry, top=True)]

    def _get_linker(self, cfg=None):
        if self._linker is None:
            from pixlang.linker import Linker
            paths = cfg.link_search_paths if cfg else []
            self._linker = Linker(self.registry, paths)
        return self._linker

    def _run_program(self) -> None:
        prog = self.program
        for unit, stmts in prog.init:
            if unit == prog.entry or not stmts:
                continue
            self._frames.append(_Frame(f"<init {Path(unit).name}>", unit, top=True))
            try:
                self._exec_block(stmts)
            finally:
                self._frames.pop()
        self._exec_block(prog.body)

    @contextmanager
    def _deep_recursion(self):
        old = sys.getrecursionlimit()
        if old < 20000:
            sys.setrecursionlimit(20000)
        try:
            yield
        finally:
            sys.setrecursionlimit(old)

    # ── Statement dispatcher ─────────────────────────────────────────────────

    def _run_body(self, stmts: List[Statement], image: Any) -> Any:
        """Run *stmts* with *image* as ``_`` and return the resulting image.

        Kept for v0.4 callers (e.g. the editor's capturing executor). If no
        program has been prepared yet, the statements are linked first.
        """
        if self.program is None:
            with self._deep_recursion():
                self._prepare(Pipeline(commands=list(stmts)))
                self.image = image
                self._run_program()
            return self.image
        if not self._frames:
            self._frames = [_Frame("<main>", self.program.entry, top=True)]
        self.image = image
        self._exec_block(stmts)
        return self.image

    # v0.4-style single-statement entry points: (stmt, image) -> image
    def _exec_set(self, stmt, image=None):     return self._run_body([stmt], image)
    def _exec_if(self, stmt, image=None):      return self._run_body([stmt], image)
    def _exec_repeat(self, stmt, image=None):  return self._run_body([stmt], image)
    def _exec_command(self, stmt, image=None): return self._run_body([stmt], image)
    def _exec_include(self, stmt, image=None): return self._run_body([stmt], image)
    def _exec_assert(self, stmt, image=None):  return self._run_body([stmt], image)
    def _exec_roi(self, stmt, image=None):     return self._run_body([stmt], image)

    def _exec_block(self, stmts: List[Any]) -> None:
        for stmt in stmts:
            self._current = stmt
            self._on_statement(stmt)
            handler = self._HANDLERS.get(type(stmt))
            if handler is None:
                raise self._error(f"cannot execute {type(stmt).__name__}", stmt)
            try:
                handler(self, stmt)
            except (_Signal, PixRuntimeError, PixAssertionError, PixSyntaxError):
                raise
            except RecursionError:
                raise self._error("expression or call nesting too deep", stmt) from None
            except Exception as exc:
                raise self._error(str(exc), stmt) from exc

    # ── Embedder hooks (no-ops by default) ────────────────────────────────────

    def _on_statement(self, stmt: Any) -> None:
        """Called before every statement executes (tracing, cancellation)."""

    def _before_command(self, cmd: Command) -> Command:
        """Called before a registry command runs; may return a rewritten Command."""
        return cmd

    def _after_command(self, cmd: Command, elapsed_ms: float) -> None:
        """Called after a registry command succeeded."""

    # ── Errors ────────────────────────────────────────────────────────────────

    def _trace(self, node: Any = None) -> List[str]:
        """PixLang call stack, outermost first."""
        if len(self._frames) <= 1:
            return []
        out = []
        for i, frame in enumerate(self._frames[1:], start=1):
            call = frame.call_node
            where = format_loc(call.file or self._entry_file(), call.line, call.col) if call else "?"
            caller = self._frames[i - 1].name
            out.append(f"{where}: in {caller}: call {frame.name}")
        if node is not None:
            out.append(f"{format_loc(node.file or self._entry_file(), node.line, node.col)}: "
                       f"in {self._frames[-1].name}")
        return out

    def _entry_file(self) -> Optional[str]:
        return str(self.pipeline_path) if self.pipeline_path else None

    def _error(self, msg: str, node: Any = None, raw: bool = False) -> PixRuntimeError:
        node = node if node is not None else self._current
        line = getattr(node, "line", 0) or 0
        col = getattr(node, "col", 0) or 0
        file = getattr(node, "file", None)
        return PixRuntimeError(msg, line=line, col=col, file=file,
                               trace=self._trace(node), raw=raw)

    # ── Variables ─────────────────────────────────────────────────────────────

    @property
    def _frame(self) -> _Frame:
        if not self._frames:
            self._frames = [_Frame("<main>", self.program.entry if self.program else None,
                                   top=True)]
        return self._frames[-1]

    def _lookup(self, name: str):
        """Return (found, value) following scope → locals → globals."""
        frame = self._frame
        for scope in reversed(frame.scopes):
            if name in scope:
                return True, scope[name]
        if not frame.top and name in frame.locals and name not in frame.globals_declared:
            return True, frame.locals[name]
        g = self.context["vars"]
        if name in g:
            return True, g[name]
        return False, None

    def _assign(self, name: str, value: Any) -> None:
        frame = self._frame
        for scope in reversed(frame.scopes):
            if name in scope:
                scope[name] = value
                if name in ("ITER", "ITER1"):
                    self.context["vars"][name] = value
                return
        if frame.top or name in frame.globals_declared:
            self.context["vars"][name] = value
        else:
            frame.locals[name] = value

    def _visible_names(self) -> List[str]:
        frame = self._frame
        names = set(self.context["vars"])
        if not frame.top:
            names |= set(frame.locals)
        for s in frame.scopes:
            names |= set(s)
        return sorted(names)

    @contextmanager
    def _loop_scope(self, mirror_iter: bool = False):
        """Push a loop-variable scope; restore $ITER/$ITER1 mirrors afterwards."""
        scope: Dict[str, Any] = {}
        g = self.context["vars"]
        saved = {k: g[k] for k in ("ITER", "ITER1") if k in g} if mirror_iter else None
        self._frame.scopes.append(scope)
        try:
            yield scope
        finally:
            self._frame.scopes.pop()
            if mirror_iter:
                for k in ("ITER", "ITER1"):
                    if k in saved:
                        g[k] = saved[k]
                    else:
                        g.pop(k, None)

    # ── SET ───────────────────────────────────────────────────────────────────

    def _do_set(self, stmt: SetVar) -> None:
        value = self._eval(stmt.value)
        if stmt.target is not None:
            self._store(stmt.target, value, stmt)
            if self.verbose:
                print(f"  {'≡':2} SET {expr_source(stmt.target):<22} = {_short(value)}")
            return
        if stmt.var_name == "_":
            self.image = self._as_image(value, stmt)
            return
        if isinstance(value, Ref):
            self._check_ref(value, stmt)
        self._assign(stmt.var_name, value)
        if self.verbose:
            print(f"  {'≡':2} SET {stmt.var_name:<22} = {_short(value)}")

    def _store(self, target: Any, value: Any, stmt: Any) -> None:
        """POKE: write through an index target or a dereferenced pointer."""
        if isinstance(target, Unary) and target.op == "*":
            ref = self._eval(target.operand)
            if not isinstance(ref, Ref):
                raise self._error(f"cannot dereference {memory.type_name(ref)} "
                                  f"'{expr_source(target.operand)}' (not a pointer)", target)
            view = self._deref(ref, target)
            view[...] = memory.store_cast(view, self._plain(value))
            return
        if isinstance(target, Index):
            container = self._eval_container(target.target)
            idx = self._eval_indices(target.indices)
            try:
                container[idx] = memory.store_cast(container, self._plain(value))
            except IndexError as exc:
                raise self._error(f"index out of bounds: {expr_source(target)} "
                                  f"for buffer of shape {container.shape} ({exc})", target)
            return
        raise self._error("invalid assignment target", stmt)

    def _eval_container(self, node: Any) -> np.ndarray:
        """Evaluate the array an index assignment writes into (by reference)."""
        value = self._eval(node)
        if isinstance(value, Ref):
            value = self._deref(value, node)
        if not isinstance(value, np.ndarray):
            raise self._error(f"'{expr_source(node)}' is not a buffer "
                              f"(got {memory.type_name(value)})", node)
        return value

    # ── IF ────────────────────────────────────────────────────────────────────

    def _do_if(self, stmt: IfBlock) -> None:
        cond = stmt.cond
        if cond is None:          # programmatically built v0.3 node
            cond = Binary(op=stmt.op, left=BareIdent(name=stmt.var_name, strict="if",
                                                     line=stmt.line, file=stmt.file),
                          right=stmt.cmp_value, line=stmt.line, col=stmt.col, file=stmt.file)
        branches = [(cond, stmt.body)] + [(c.cond, c.body) for c in stmt.elifs]
        for c, body in branches:
            ok = self._truth(self._eval(c), c)
            if self.verbose:
                tick = "✓" if ok else "✗"
                print(f"  {tick}  IF {_cond_text(c)}  → "
                      f"{'True — executing body' if ok else 'False'}")
            if ok:
                self._exec_block(body)
                return
        if stmt.else_body:
            self._exec_block(stmt.else_body)

    # ── Loops ─────────────────────────────────────────────────────────────────

    def _int_count(self, value: Any, what: str, node: Any) -> int:
        if isinstance(value, str):
            try:
                value = int(value, 0)
            except ValueError:
                pass
        if isinstance(value, (np.integer,)):
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._error(f"[{what}] Count must be an integer, got "
                              f"{memory.type_name(value)} {value!r}", node)
        return value

    def _do_repeat(self, stmt: RepeatBlock) -> None:
        count = self._int_count(self._eval(stmt.count), "REPEAT", stmt)
        if count < 0:
            raise self._error(f"[REPEAT] Count must be ≥ 0, got {count}", stmt)
        if self.verbose:
            print(f"  ↻  REPEAT {count}")
        g = self.context["vars"]
        with self._loop_scope(mirror_iter=True) as scope:
            for i in range(count):
                scope["ITER"] = g["ITER"] = i          # $ITER  (0-based)
                scope["ITER1"] = g["ITER1"] = i + 1    # $ITER1 (1-based)
                if stmt.var_name:
                    scope[stmt.var_name] = i
                try:
                    self._exec_block(stmt.body)
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    continue

    def _do_while(self, stmt: WhileBlock) -> None:
        cap = self.max_loop_iterations or self.DEFAULT_MAX_LOOP_ITERATIONS
        n = 0
        with self._loop_scope():
            while self._truth(self._eval(stmt.cond), stmt.cond):
                n += 1
                if n > cap:
                    raise self._error(
                        f"[WHILE] loop exceeded {cap} iterations — infinite loop? "
                        f"(raise [run] max_loop_iterations in pixlang.toml)", stmt)
                try:
                    self._exec_block(stmt.body)
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    continue

    def _do_for(self, stmt: ForBlock) -> None:
        start = self._number(self._eval(stmt.start), "FOR start", stmt)
        stop = self._number(self._eval(stmt.stop), "FOR end", stmt)
        step = 1 if stmt.step is None else self._number(self._eval(stmt.step), "FOR STEP", stmt)
        if step == 0:
            raise self._error("[FOR] STEP must not be 0", stmt)
        cap = self.max_loop_iterations or self.DEFAULT_MAX_LOOP_ITERATIONS
        with self._loop_scope() as scope:
            i, n = start, 0
            while (i <= stop) if step > 0 else (i >= stop):
                n += 1
                if n > cap:
                    raise self._error(f"[FOR] loop exceeded {cap} iterations", stmt)
                scope[stmt.var_name] = i
                try:
                    self._exec_block(stmt.body)
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    pass
                i = i + step

    def _number(self, v: Any, what: str, node: Any):
        v = memory.to_python(v)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise self._error(f"[{what}] must be a number, got {memory.type_name(v)} {v!r}", node)
        return v

    def _do_break(self, stmt: BreakStmt) -> None:
        raise _BreakSignal()

    def _do_continue(self, stmt: ContinueStmt) -> None:
        raise _ContinueSignal()

    # ── COMMAND ───────────────────────────────────────────────────────────────

    def _do_command(self, cmd: Command) -> None:
        qual = self.program.lookup(self._frame.unit, cmd.name) if self.program else None
        if qual is not None:
            fn = self.program.functions[qual]
            args = [self._resolve(a) for a in cmd.args]
            self._call_function(fn, qual, args, cmd)
            return
        if "." in cmd.name:
            raise self._error(f"Unknown command '{cmd.name}' (no such module function)", cmd)

        cmd = self._before_command(cmd)
        resolved_args = [self._resolve(a) for a in cmd.args]
        t0 = time.perf_counter()
        try:
            self.image = self.registry.invoke(cmd.name, self.image, resolved_args, self.context)
        except NameError as exc:
            if cmd.name.upper() not in self.registry:
                raise self._error(str(exc).replace("[PixLang] ", ""), cmd) from exc
            raise self._command_error(cmd, resolved_args, exc) from exc
        except Exception as exc:
            raise self._command_error(cmd, resolved_args, exc) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = self.context["stats"]
        stats["commands_executed"] += 1
        stats["total_ms"] += elapsed_ms
        stats["command_times"].setdefault(cmd.name, []).append(elapsed_ms)
        self._after_command(cmd, elapsed_ms)

        if self.verbose:
            img_raw, _ = _unwrap(self.image) if self.image is not None else (None, {})
            shape_str  = str(getattr(img_raw, "shape", "—"))
            args_str   = " ".join(_short(a) for a in resolved_args)
            print(f"  {'✓':2} {cmd.name:<24} {args_str:<20} "
                  f"{shape_str:<18} {elapsed_ms:6.1f} ms")

    def _command_error(self, cmd: Command, args: List[Any], exc: Exception) -> PixRuntimeError:
        where = f"line {cmd.line}"
        if cmd.col:
            where += f", col {cmd.col}"
        if cmd.file:
            where += f" in {cmd.file}"
        msg = (f"[PixLang] Error at {where} "
               f"({cmd.name} {' '.join(_short(a) for a in args)}): {exc}")
        return self._error(msg, cmd, raw=True)

    def _resolve(self, value: Any) -> Any:
        """Resolve a command argument (legacy: bare word = variable or string)."""
        if isinstance(value, BareIdent):
            found, v = self._lookup(value.name)
            return v if found else value.name
        if isinstance(value, Expr):
            v = self._eval(value)
            if isinstance(v, Ref):
                v = self._deref(v, value)
            return v
        return value

    # ── Functions ─────────────────────────────────────────────────────────────

    def _call_function(self, fn: FuncDef, qual: str, args: List[Any], node: Any) -> Any:
        if len(args) != len(fn.params):
            raise self._error(
                f"{fn.name}() takes {len(fn.params)} argument"
                f"{'s' if len(fn.params) != 1 else ''} ({', '.join(fn.params)}), "
                f"got {len(args)}", node)
        if len(self._frames) > self.MAX_CALL_DEPTH:
            raise self._error(f"maximum call depth ({self.MAX_CALL_DEPTH}) exceeded "
                              f"calling {fn.name} — runaway recursion?", node)
        frame = _Frame(fn.name, self.program.func_units.get(qual), top=False,
                       call_node=node, func=fn)
        frame.locals = dict(zip(fn.params, args))
        saved_image = self.image
        self._frames.append(frame)
        result = None
        try:
            self._exec_block(fn.body)
        except _ReturnSignal as ret:
            result = ret.value
        finally:
            self._frames.pop()
        if fn.is_command:
            if result is not None:
                self.image = self._as_image(result, node)
            return self.image
        self.image = saved_image
        return result

    def _do_return(self, stmt: ReturnStmt) -> None:
        value = None if stmt.value is None else self._eval(stmt.value)
        raise _ReturnSignal(value)

    def _do_global(self, stmt: GlobalStmt) -> None:
        frame = self._frame
        if frame.top:
            return
        for name in stmt.names:
            frame.globals_declared.add(name)
            frame.locals.pop(name, None)

    def _do_noop(self, stmt: Any) -> None:
        """FUNC / IMPORT are handled by the linker."""

    # ── INCLUDE (run-time, unlinked pipelines) ────────────────────────────────

    def _check_include_path(self, inc_path: Path, stmt: IncludeStmt) -> None:
        """Hook for embedders to restrict which files INCLUDE may read."""

    def _do_include(self, stmt: IncludeStmt) -> None:
        from pixlang.parser import parse as _parse

        if stmt.file:
            base = Path(stmt.file).parent
        else:
            base = self.pipeline_path.parent if self.pipeline_path else Path(".")
        inc_path = (base / stmt.path).resolve()
        key = str(inc_path)

        self._check_include_path(inc_path, stmt)
        if not inc_path.exists():
            raise self._error(f"[INCLUDE] File not found: '{inc_path}'", stmt)
        if key in self._include_stack:
            root = [str(self.pipeline_path.resolve())] if self.pipeline_path else []
            chain = " → ".join(Path(p).name for p in root + self._include_stack + [key])
            raise self._error(f"[INCLUDE] cycle detected: {chain}", stmt)
        if self.verbose:
            print(f"  ⤵  INCLUDE {stmt.path}")

        sub = _parse(inc_path.read_text(encoding="utf-8"), filename=key)
        stmts = self._get_linker().extend(self.program, self._frame.unit, sub, key)
        self._include_stack.append(key)
        try:
            self._exec_block(stmts)
        finally:
            self._include_stack.pop()

    # ── ASSERT ────────────────────────────────────────────────────────────────

    def _image_subject(self, name: str, node: Any) -> Any:
        if self.image is None:
            raise self._error("[ASSERT] No image loaded", node)
        img, meta = _unwrap(self.image)
        if name == "width":
            return img.shape[1]
        if name == "height":
            return img.shape[0]
        if name in ("channels", "depth"):
            return img.shape[2] if len(img.shape) == 3 else 1
        if name == "min":
            return memory.to_python(img.min())
        if name == "max":
            return memory.to_python(img.max())
        if name == "contour_count":
            return len(meta.get("contours", []))
        if name == "ndim":
            return img.ndim
        raise self._error(
            f"[ASSERT] Unknown subject '{name}'. "
            f"Valid: {', '.join(IMAGE_SUBJECTS)} (or a variable)", node)

    def _do_assert(self, stmt: AssertStmt) -> None:
        cond = stmt.cond
        if cond is None:          # programmatically built v0.4 node
            cond = Binary(op=stmt.op,
                          left=BareIdent(name=stmt.subject, strict="assert",
                                         line=stmt.line, file=stmt.file),
                          right=stmt.expected, line=stmt.line, col=stmt.col, file=stmt.file)
        actual = None
        if isinstance(cond, Binary) and cond.op in memory._COMPARE:
            actual = self._eval(cond.left)
            expected = self._eval(cond.right)
            passed = self._truth(memory.compare(cond.op, actual, expected), cond)
            desc = f"{_cond_text(cond.left)} {cond.op} {_short(expected)}"
        else:
            passed = self._truth(self._eval(cond), cond)
            desc = stmt.cond_text or expr_source(cond)

        if self.verbose:
            tick = "✓" if passed else "✗"
            got = f"  (actual={_short(actual)})" if actual is not None else ""
            print(f"  {tick}  ASSERT {desc}{got}  {'OK' if passed else 'FAIL'}")

        if not passed:
            custom = f" — {stmt.message}" if stmt.message else ""
            got = f", but got {_short(actual)}" if actual is not None else ""
            raise PixAssertionError(
                f"[ASSERT] Pipeline assertion failed (line {stmt.line}): "
                f"{desc}{got}{custom}",
                line=stmt.line, col=stmt.col, file=stmt.file, trace=self._trace(stmt))

    # ── ROI ───────────────────────────────────────────────────────────────────

    def _do_roi(self, stmt: RoiBlock) -> None:
        """Execute body commands on a cropped region, then paste result back."""
        if self.image is None:
            raise self._error(f"[ROI] No image loaded (line {stmt.line})", stmt)

        img, meta = _unwrap(self.image)
        try:
            x, y, w, h = (int(self._resolve(v)) for v in (stmt.x, stmt.y, stmt.w, stmt.h))
        except (TypeError, ValueError) as exc:
            raise self._error(f"[ROI] coordinates must be integers ({exc})", stmt)

        ih, iw = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            raise self._error(
                f"[ROI] Region {x},{y},{w},{h} is outside image bounds "
                f"{iw}×{ih} (line {stmt.line})", stmt)

        if self.verbose:
            print(f"  ⌗  ROI {x1},{y1} → {x2},{y2} ({x2-x1}×{y2-y1})")

        region = img[y1:y2, x1:x2].copy()
        roi_image = _wrap(region)
        roi_image.update(meta)

        self.image = roi_image
        self._exec_block(stmt.body)
        processed_region, processed_meta = _unwrap(self.image)

        canvas = img.copy()
        canvas, processed_region = _fit_channels(canvas, processed_region)
        if processed_region.shape[:2] != (y2 - y1, x2 - x1):
            import cv2
            processed_region = cv2.resize(
                processed_region, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        canvas[y1:y2, x1:x2] = processed_region

        result = _wrap(canvas)
        result.update(meta)
        result.update(processed_meta)
        self.image = result

    # ── Buffers & pointers ────────────────────────────────────────────────────

    def _slots(self) -> Dict[str, Any]:
        return self.context["checkpoints"]

    def _slot_array(self, name: str) -> Optional[np.ndarray]:
        entry = self._slots().get(name)
        return memory.slot_array(entry) if entry is not None else None

    def _put_slot(self, name: str, arr: np.ndarray) -> None:
        gens = self.context["buffer_gen"]
        gens[name] = gens.get(name, 0) + 1
        self.context["buffer_freed"].pop(name, None)
        self._slots()[name] = memory.make_slot_entry(arr)

    def _do_alloc(self, stmt: AllocStmt) -> None:
        w = self._int_count(self._resolve(stmt.width), "ALLOC width", stmt)
        h = self._int_count(self._resolve(stmt.height), "ALLOC height", stmt)
        c = self._int_count(self._resolve(stmt.channels), "ALLOC channels", stmt)
        if w <= 0 or h <= 0 or c <= 0:
            raise self._error(f"[ALLOC] dimensions must be > 0, got {w}×{h}×{c}", stmt)
        dt = memory.dtype_of(stmt.dtype)
        fill = self._plain(self._resolve(stmt.fill))
        shape = (h, w) if c == 1 else (h, w, c)
        arr = np.zeros(shape, dtype=dt)
        arr[...] = memory.store_cast(arr, fill)
        self._put_slot(stmt.name, arr)
        if self.verbose:
            print(f"  ▣  ALLOC {stmt.name} {shape} {stmt.dtype}  ({arr.nbytes} bytes)")

    def _do_buf(self, stmt: BufStmt) -> None:
        if stmt.value is None:
            if self.image is None:
                raise self._error("[BUF] No image loaded to store", stmt)
            arr = _unwrap(self.image)[0]
        else:
            arr = self._plain(self._eval(stmt.value))
        if not isinstance(arr, np.ndarray):
            raise self._error(f"[BUF] value must be a buffer/image, got "
                              f"{memory.type_name(arr)}", stmt)
        self._put_slot(stmt.name, arr.copy())
        if self.verbose:
            print(f"  ▣  BUF {stmt.name} {arr.shape} {memory.dtype_name(arr.dtype)}")

    def _do_use(self, stmt: UseStmt) -> None:
        value = self._plain(self._eval(stmt.value))
        if isinstance(value, str):
            raise self._error(f"[USE] no buffer or variable named '{value}'", stmt)
        self.image = self._as_image(value, stmt)

    def _do_free(self, stmt: FreeStmt) -> None:
        slots = self._slots()
        for name in stmt.names:
            if name not in slots:
                if name in self.context["buffer_freed"]:
                    raise self._error(f"[FREE] double free of buffer '{name}' (already "
                                      f"freed at line {self.context['buffer_freed'][name]})", stmt)
                raise self._error(f"[FREE] no buffer named '{name}'", stmt)
            del slots[name]
            gens = self.context["buffer_gen"]
            gens[name] = gens.get(name, 0) + 1
            self.context["buffer_freed"][name] = stmt.line

    def _do_with(self, stmt: WithBlock) -> None:
        target = stmt.target
        if isinstance(target, Unary) and target.op == "*":
            ref = self._eval(target.operand)
            if not isinstance(ref, Ref):
                raise self._error("WITH *p needs a pointer", target)
            view = self._deref(ref, target)
        else:
            value = self._eval(target)
            view = self._deref(value, target) if isinstance(value, Ref) else value
        if not isinstance(view, np.ndarray) or view.ndim < 2:
            raise self._error("WITH needs a 2-D region (pointer to a buffer slice)", stmt)

        saved = self.image
        self.image = _wrap(view.copy())
        try:
            self._exec_block(stmt.body)
            result, _meta = _unwrap(self.image)
        finally:
            produced = self.image
            self.image = saved
        if not isinstance(result, np.ndarray):
            raise self._error("WITH body did not produce an image", stmt)
        _, result = _fit_channels(view, result, promote_canvas=False)
        if result.shape[:2] != view.shape[:2]:
            import cv2
            result = cv2.resize(result, (view.shape[1], view.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            if result.ndim < view.ndim:
                result = result.reshape(view.shape)
        view[...] = memory.store_cast(view, result)
        del produced

    def _make_ref(self, node: Any, at: Any) -> Ref:
        """``&name`` / ``&name[...]`` → pointer."""
        target, indices = node, []
        if isinstance(node, Index):
            target, indices = node.target, node.indices
        if isinstance(target, BareIdent) and target.name == "_":
            raise self._error("cannot take the address of _ (store it with BUF first)", at)
        idx = self._eval_indices(indices) if indices else ()
        if isinstance(target, BareIdent):
            found, value = self._lookup(target.name)
            if not found or isinstance(value, str):
                arr = self._slot_array(target.name)
                if arr is None:
                    raise self._error(f"cannot take the address of '{target.name}': "
                                      "no such buffer", at)
                ref = Ref(target.name, self.context["buffer_gen"].get(target.name, 0),
                          idx, arr, at.line)
                self._deref(ref, at)            # validates the index
                return ref
            base = value
        else:
            base = self._eval(target)
        if isinstance(base, Ref):
            view = self._deref(base, at)
            ref = Ref(base.slot, base.gen, base.index, base.base, at.line)
            if idx:
                # Pointer to a sub-region of a pointed region: compose via view
                ref = Ref(None, 0, idx, view, at.line)
            self._deref(ref, at)
            return ref
        if isinstance(base, np.ndarray):
            ref = Ref(None, 0, idx, base, at.line)
            self._deref(ref, at)
            return ref
        raise self._error(f"cannot take the address of {memory.type_name(base)} "
                          f"'{expr_source(node)}'", at)

    def _check_ref(self, ref: Ref, node: Any) -> None:
        if ref.slot is None:
            return
        freed = self.context["buffer_freed"].get(ref.slot)
        current = self._slot_array(ref.slot)
        gen = self.context["buffer_gen"].get(ref.slot, 0)
        line = getattr(node, "line", 0)
        if current is None and freed is not None:
            raise self._error(
                f"use-after-free (line {line}): pointer {ref.describe()} (taken at line "
                f"{ref.line}) refers to buffer '{ref.slot}' freed at line {freed}", node)
        if current is None or gen != ref.gen or current is not ref.base:
            raise self._error(
                f"dangling pointer (line {line}): buffer '{ref.slot}' was reallocated "
                f"after the pointer was taken at line {ref.line}", node)

    def _deref(self, ref: Ref, node: Any) -> np.ndarray:
        self._check_ref(ref, node)
        try:
            return ref.raw_view()
        except IndexError as exc:
            raise self._error(f"pointer index out of bounds: {ref.describe()} for buffer "
                              f"of shape {ref.base.shape} ({exc})", node)

    def _eval_indices(self, indices: List[Any]) -> tuple:
        out = []
        for item in indices:
            if isinstance(item, Slice):
                parts = []
                for p in (item.start, item.stop, item.step):
                    parts.append(None if p is None else self._index_int(self._eval(p), item))
                out.append(slice(*parts))
            else:
                out.append(self._index_int(self._eval(item), item))
        return tuple(out)

    def _index_int(self, v: Any, node: Any) -> int:
        v = memory.to_python(v)
        if isinstance(v, bool) or not isinstance(v, int):
            raise self._error(f"index must be an integer, got {memory.type_name(v)} {v!r}", node)
        return v

    # ── Expression evaluation ─────────────────────────────────────────────────

    def _eval(self, node: Any) -> Any:
        if not isinstance(node, Expr):
            return node
        method = self._EVAL.get(type(node))
        return method(self, node)

    def _ev_literal(self, node: Literal) -> Any:
        return node.value

    def _ev_varref(self, node: VarRef) -> Any:
        found, value = self._lookup(node.var_name)
        if not found:
            avail = ", ".join("$" + k for k in self._visible_names()) or "(none)"
            raise self._error(f"Undefined variable '${node.var_name}'. Available: {avail}", node)
        return value

    def _ev_bare(self, node: BareIdent) -> Any:
        name = node.name
        found, value = self._lookup(name)
        if found:
            return value
        if name == "_":
            if self.image is None:
                raise self._error("_ (current image) is empty — LOAD or USE first", node)
            return _unwrap(self.image)[0]
        arr = self._slot_array(name)
        if arr is not None:
            return arr
        if node.strict == "if":
            raise self._error(
                f"[IF] Variable '${name}' is not defined (line {node.line}). Use SET first.",
                node)
        if node.strict == "assert":
            if name.lower() in IMAGE_SUBJECTS:
                return self._image_subject(name.lower(), node)
            raise self._error(
                f"[ASSERT] Unknown subject '{name}'. "
                f"Valid: {', '.join(IMAGE_SUBJECTS)} (or a variable)", node)
        if name in self.context["buffer_freed"]:
            raise self._error(f"use-after-free (line {node.line}): buffer '{name}' was "
                              f"freed at line {self.context['buffer_freed'][name]}", node)
        return name          # legacy: bare word is a string literal

    def _ev_unary(self, node: Unary) -> Any:
        if node.op == "&":
            return self._make_ref(node.operand, node)
        if node.op == "*":
            ref = self._eval(node.operand)
            if not isinstance(ref, Ref):
                raise self._error(f"cannot dereference {memory.type_name(ref)} "
                                  f"'{expr_source(node.operand)}' (not a pointer)", node)
            view = self._deref(ref, node)
            return memory.to_python(view) if view.ndim == 0 else view
        value = self._plain(self._eval(node.operand))
        if node.op == "not":
            return not self._truth(value, node) if not isinstance(value, np.ndarray) \
                else np.logical_not(value)
        try:
            return memory.unary(node.op, value)
        except (TypeError, ValueError) as exc:
            raise self._error(str(exc), node)

    def _ev_binary(self, node: Binary) -> Any:
        op = node.op
        if op in ("and", "or"):
            left = self._eval(node.left)
            lt = self._truth(left, node.left)
            if op == "and" and not lt:
                return left
            if op == "or" and lt:
                return left
            return self._eval(node.right)
        left = self._plain(self._eval(node.left), allow_ref=op in ("==", "!="))
        right = self._plain(self._eval(node.right), allow_ref=op in ("==", "!="))
        try:
            return memory.binop(op, left, right)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise self._error(f"{exc} (in '{expr_source(node)}')", node)

    def _ev_index(self, node: Index) -> Any:
        target = self._eval(node.target)
        if isinstance(target, Ref):
            target = self._deref(target, node)
        idx = self._eval_indices(node.indices)
        if isinstance(target, np.ndarray):
            try:
                value = target[idx]
            except IndexError as exc:
                raise self._error(f"index out of bounds: {expr_source(node)} for buffer "
                                  f"of shape {target.shape} ({exc})", node)
            if isinstance(value, np.ndarray):
                return value.copy()           # slices are values; use & for a live view
            return memory.to_python(value)
        if isinstance(target, (str, tuple, list)):
            try:
                return target[idx[0] if len(idx) == 1 else idx]
            except (IndexError, TypeError) as exc:
                raise self._error(f"bad index {expr_source(node)}: {exc}", node)
        raise self._error(f"'{expr_source(node.target)}' is not indexable "
                          f"({memory.type_name(target)})", node)

    def _ev_attr(self, node: Attr) -> Any:
        target = self._eval(node.target)
        if isinstance(target, Ref):
            view = self._deref(target, node)
            if node.name == "offset":
                return view.__array_interface__["data"][0] - \
                    target.base.__array_interface__["data"][0]
            target = view
        if isinstance(target, np.ndarray):
            try:
                return memory.buffer_attr(target, node.name)
            except AttributeError as exc:
                raise self._error(str(exc), node)
        if isinstance(target, str) and isinstance(node.target, BareIdent):
            raise self._error(f"'{node.target.name}' is not a buffer or variable "
                              f"(cannot read .{node.name})", node)
        raise self._error(f"{memory.type_name(target)} has no attribute '{node.name}'", node)

    def _ev_call(self, node: Call) -> Any:
        args = [self._plain(self._eval(a)) for a in node.args]
        qual = self.program.lookup(self._frame.unit, node.func) if self.program else None
        if qual is not None:
            fn = self.program.functions[qual]
            if fn.is_command:
                raise self._error(f"command FUNC '{fn.name}' cannot be used in an "
                                  "expression", node)
            return self._call_function(fn, qual, args, node)
        builtin = BUILTIN_FUNCTIONS.get(node.func)
        if builtin is None:
            import difflib
            pool = list(BUILTIN_FUNCTIONS) + (
                list(self.program.symbols.get(self._frame.unit, {})) if self.program else [])
            close = difflib.get_close_matches(node.func, pool, n=3, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise self._error(f"Unknown function '{node.func}'.{hint}", node)
        try:
            return builtin(*args)
        except TypeError as exc:
            msg = str(exc).replace(f"_{node.func.lower()}() ", "")
            raise self._error(f"{node.func}(): {msg}", node)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise self._error(f"{node.func}(): {exc}", node)

    def _ev_astype(self, node: AsType) -> Any:
        value = self._eval(node.value)
        if isinstance(value, Ref):
            value = self._deref(value, node)
        if not isinstance(value, np.ndarray):
            raise self._error(f"AS {node.dtype} needs a buffer, got "
                              f"{memory.type_name(value)}", node)
        try:
            return memory.reinterpret(value, node.dtype)
        except ValueError as exc:
            raise self._error(str(exc), node)

    def _ev_slice(self, node: Slice) -> Any:
        raise self._error("a slice (a:b) is only valid inside [...]", node)

    # ── Value helpers ─────────────────────────────────────────────────────────

    def _plain(self, value: Any, allow_ref: bool = False) -> Any:
        """Unwrap image dicts; pointers must be dereferenced explicitly."""
        if isinstance(value, dict) and "_image" in value:
            return value["_image"]
        if isinstance(value, Ref) and not allow_ref:
            self._check_ref(value, self._current)
        return value

    def _truth(self, value: Any, node: Any) -> bool:
        try:
            return memory.truth(value)
        except TypeError as exc:
            raise self._error(str(exc), node)

    def _as_image(self, value: Any, node: Any) -> Any:
        if isinstance(value, Ref):
            value = self._deref(value, node)
        if isinstance(value, dict) and "_image" in value:
            return value
        if not isinstance(value, np.ndarray):
            raise self._error(f"expected an image/buffer, got {memory.type_name(value)}", node)
        if value.dtype == np.bool_:
            return value.astype(np.uint8) * 255
        return value.copy()

    # ── Dispatch tables ───────────────────────────────────────────────────────

    _HANDLERS = {
        SetVar: _do_set, IfBlock: _do_if, RepeatBlock: _do_repeat,
        WhileBlock: _do_while, ForBlock: _do_for, BreakStmt: _do_break,
        ContinueStmt: _do_continue, Command: _do_command,
        IncludeStmt: _do_include, AssertStmt: _do_assert, RoiBlock: _do_roi,
        WithBlock: _do_with, ReturnStmt: _do_return, GlobalStmt: _do_global,
        FuncDef: _do_noop, ImportStmt: _do_noop, AllocStmt: _do_alloc,
        BufStmt: _do_buf, UseStmt: _do_use, FreeStmt: _do_free,
    }

    _EVAL = {
        Literal: _ev_literal, VarRef: _ev_varref, BareIdent: _ev_bare,
        Unary: _ev_unary, Binary: _ev_binary, Index: _ev_index, Attr: _ev_attr,
        Call: _ev_call, AsType: _ev_astype, Slice: _ev_slice,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cond_text(node: Any) -> str:
    text = expr_source(node)
    if isinstance(node, Binary) and text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return text


def _short(v: Any) -> str:
    """Compact display of a value (buffers shown by shape/dtype)."""
    if isinstance(v, np.ndarray):
        return f"<buffer {'×'.join(map(str, v.shape))} {memory.dtype_name(v.dtype)}>"
    if isinstance(v, dict) and "_image" in v:
        return _short(v["_image"])
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _fit_channels(canvas: np.ndarray, region: np.ndarray, promote_canvas: bool = True):
    """Make *region* channel-compatible with *canvas* (gray ↔ BGR).

    ROI promotes a gray canvas to BGR (v0.4 behaviour); WITH writes into a
    fixed buffer, so there the region is converted to gray instead.
    """
    import cv2
    if canvas.ndim == 2 and region.ndim == 3:
        if promote_canvas:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        else:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    elif canvas.ndim == 3 and region.ndim == 2:
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
        if canvas.shape[2] == 4:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2BGRA)
    return canvas, region


def _coerce_for_compare(v: Any):
    """Kept for backwards compatibility; see memory.coerce_for_compare."""
    v = memory.coerce_for_compare(v)
    return v if isinstance(v, (int, float)) else str(v)
