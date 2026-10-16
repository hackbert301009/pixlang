"""
PixLang Linter  v0.5

Analyses a parsed Pipeline AST and returns a list of Diagnostics.
Each rule is a small function registered with @_rule.

Built-in rules:
  PX000  WARNING — internal lint error (a rule crashed; reported, not hidden)
  PX001  ERROR   — pipeline does not start with LOAD / LOAD_GLOB / USE / INCLUDE
  PX002  WARNING — SAVE / SAVE_EACH missing (output never written)
  PX003  WARNING — FIND_CONTOURS not followed by a draw command
  PX004  ERROR   — OVERLAY / BLEND / COMPARE use a checkpoint not created yet
  PX005  ERROR   — variable used before it is defined (order-aware)
  PX006  INFO    — repeated identical command (possible copy-paste)
  PX007  WARNING — RESIZE used after a filtering step (unusual order)
  PX008  ERROR   — REPEAT with a negative / non-integer literal count
         WARNING — REPEAT 0 (body never runs)
  PX009  INFO    — pipeline has no commands
  PX010  WARNING — THRESHOLD used before GRAYSCALE
  PX011  ERROR   — command name is not a FUNC and not in the registry
  PX012  WARNING — bare word silently becomes a string but looks like a
                   typo of a defined variable
  LNKxxx         — linker diagnostics (when the pipeline path is known)

Blocks (IF/ELIF/ELSE, REPEAT, WHILE, FOR, ROI, WITH) and INCLUDEd files are
analysed; FUNC bodies are checked for variables and unknown commands.
"""
from __future__ import annotations

import difflib
import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Set

from pixlang.parser.ast_nodes import (
    AllocStmt, AssertStmt, BareIdent, BufStmt, Call, Command, ForBlock, FuncDef,
    IfBlock, ImportStmt, IncludeStmt, Literal, Pipeline, RepeatBlock, SetVar,
    UseStmt, VarRef, expr_source, iter_child_bodies, statement_exprs, walk_expr,
    walk_statements,
)


# ── Data types ────────────────────────────────────────────────────────────────

class Severity(enum.Enum):
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


@dataclass
class Diagnostic:
    code:     str       # e.g. "PX001"
    severity: Severity
    message:  str
    line:     int = 0
    col:      int = 0
    file:     Optional[str] = None

    def __str__(self):
        icon = {"info": "ℹ", "warning": "⚠", "error": "✗"}[self.severity.value]
        loc  = f"line {self.line}  " if self.line else ""
        if self.file and self.line:
            loc = f"{Path(self.file).name}:{self.line}  "
        return f"  {icon}  [{self.code}] {loc}{self.message}"


def _diag(code: str, sev: Severity, msg: str, node: Any = None) -> Diagnostic:
    if node is None:
        return Diagnostic(code, sev, msg)
    return Diagnostic(code, sev, msg, line=node.line, col=getattr(node, "col", 0),
                      file=getattr(node, "file", None))


# ── Rule registry ─────────────────────────────────────────────────────────────

_RULES: List[Callable] = []

def _rule(fn: Callable) -> Callable:
    _RULES.append(fn)
    return fn


# ── Analysis context ──────────────────────────────────────────────────────────

class _Ctx:
    """Pre-computed facts shared by the rules."""

    def __init__(self, pipeline: Pipeline, registry, config_vars: Iterable[str],
                 path: Optional[Path]):
        self.pipeline = pipeline
        self.registry = registry
        self.path = path
        self.config_vars: Set[str] = set(config_vars)
        self.unresolved_include = False
        self.main = self._expand(pipeline.commands, path, [])
        self.funcs = {s.name: s for s in self.main if isinstance(s, FuncDef)}
        self.imported: Set[str] = set()
        self.aliases: Set[str] = set()
        for s in self.main:
            if isinstance(s, ImportStmt):
                if s.names:
                    self.imported.update(s.names)
                else:
                    self.aliases.add(s.alias or Path(s.path).stem)
        self.main_body = [s for s in self.main if not isinstance(s, (FuncDef, ImportStmt))]

    def _expand(self, stmts: List[Any], cur: Optional[Path], stack: List[str]) -> List[Any]:
        """Resolve INCLUDEs (best effort) so rules see included commands."""
        from pixlang.parser import parse
        out: List[Any] = []
        for stmt in stmts:
            if isinstance(stmt, IncludeStmt):
                base = Path(stmt.file).parent if stmt.file else (
                    cur.parent if cur else None)
                target = (base / stmt.path).resolve() if base is not None else None
                if target is None or not target.exists() or str(target) in stack:
                    self.unresolved_include = True
                    out.append(stmt)
                    continue
                try:
                    sub = parse(target.read_text(encoding="utf-8"), filename=str(target))
                except SyntaxError:
                    self.unresolved_include = True
                    out.append(stmt)
                    continue
                out.extend(self._expand(sub.commands, target, stack + [str(target)]))
                continue
            out.append(stmt)
        return out

    def is_user_command(self, name: str) -> bool:
        if name in self.funcs or name in self.imported:
            return True
        return "." in name and name.split(".")[0] in self.aliases


def _walk_main(ctx: _Ctx) -> Iterable[Any]:
    yield from walk_statements(ctx.main_body)


def _flatten(stmts: List[Any]) -> List[Command]:
    """Recursively extract all Command nodes from nested blocks (not FUNCs)."""
    cmds: List[Command] = []
    for stmt in stmts:
        if isinstance(stmt, Command):
            cmds.append(stmt)
        elif isinstance(stmt, FuncDef):
            continue
        else:
            for body in iter_child_bodies(stmt):
                cmds.extend(_flatten(body))
    return cmds


def _arg_text(arg: Any) -> str:
    """Readable text of a command argument (no repr leaks)."""
    if isinstance(arg, str):
        return arg
    if isinstance(arg, BareIdent):
        return arg.name
    return expr_source(arg)


# ── Linter ────────────────────────────────────────────────────────────────────

class Linter:
    """
    Run all registered rules over a Pipeline AST.

    Usage:
        linter = Linter(registry)
        diags  = linter.lint(pipeline)                 # in-memory
        diags  = linter.lint(pipeline, path=pxl_path)  # + INCLUDE / linker / toml
        diags  = linter.lint_file(pxl_path)
    """

    def __init__(self, registry=None, ignore: Optional[Iterable[str]] = None,
                 config_vars: Optional[Iterable[str]] = None):
        self.registry = registry  # optional, used to validate command names
        self.ignore = set(ignore) if ignore is not None else None
        self.config_vars = set(config_vars) if config_vars is not None else None

    def lint(self, pipeline: Pipeline, path: Optional[Path] = None) -> List[Diagnostic]:
        path = Path(path) if path else None
        ignore, config_vars, search = self.ignore, self.config_vars, []
        if path is not None and (ignore is None or config_vars is None):
            from pixlang.config import load_config
            cfg = load_config(path.resolve())
            if ignore is None:
                ignore = set(cfg.lint_ignore)
            if config_vars is None:
                config_vars = set(cfg.variables)
            search = cfg.link_search_paths
        ignore = ignore or set()

        ctx = _Ctx(pipeline, self.registry, config_vars or (), path)
        flat = _flatten(ctx.main_body)
        diags: List[Diagnostic] = []

        for rule_fn in _RULES:
            try:
                results = rule_fn(ctx, flat, self.registry)
                if results:
                    diags.extend(results)
            except Exception as exc:     # report, never hide, a broken rule
                diags.append(Diagnostic(
                    "PX000", Severity.WARNING,
                    f"internal lint error in {rule_fn.__name__}: "
                    f"{type(exc).__name__}: {exc}"))

        if path is not None:
            diags.extend(self._link_diagnostics(pipeline, path, search))

        diags = [d for d in diags if d.code not in ignore]
        return sorted(diags, key=lambda d: (d.line, d.code))

    def lint_file(self, path) -> List[Diagnostic]:
        from pixlang.parser import parse
        path = Path(path)
        pipeline = parse(path.read_text(encoding="utf-8"), filename=str(path.resolve()))
        return self.lint(pipeline, path=path)

    def _link_diagnostics(self, pipeline: Pipeline, path: Path, search) -> List[Diagnostic]:
        import copy
        from pixlang.linker import Linker
        try:
            program = Linker(self.registry, search).link_pipeline(
                copy.deepcopy(pipeline), path=path, splice_includes=True,
                raise_on_error=False)
        except SyntaxError as exc:
            return [Diagnostic("PARSE", Severity.ERROR, str(exc),
                               line=exc.lineno or 0, col=exc.offset or 0,
                               file=exc.filename)]
        out = []
        for d in program.diagnostics:
            if d.code == "LNK011":        # unknown command — PX011 covers it
                continue
            sev = Severity.ERROR if d.severity == "error" else Severity.WARNING
            out.append(Diagnostic(d.code, sev, d.message, d.line, d.col, d.file))
        return out

    @property
    def rule_count(self) -> int:
        return len(_RULES)


# ── Rules ─────────────────────────────────────────────────────────────────────

_IMAGE_SOURCES = {"LOAD", "LOAD_GLOB"}


@_rule
def rule_px001_starts_with_load(ctx, flat, registry):
    """PX001 — pipeline should start by loading an image."""
    for stmt in _walk_main(ctx):
        if isinstance(stmt, (IncludeStmt,)):
            return []                         # unresolved include may load
        if isinstance(stmt, (UseStmt, AllocStmt)):
            return []                         # image comes from a buffer
        if isinstance(stmt, Command):
            if stmt.name in _IMAGE_SOURCES or ctx.is_user_command(stmt.name):
                return []
            return [_diag(
                "PX001", Severity.ERROR,
                f"Pipeline does not start with LOAD (first command is '{stmt.name}'). "
                "No image will be loaded.", stmt)]
    return []


@_rule
def rule_px002_has_save(ctx, flat, registry):
    """PX002 — pipeline should write its output (SAVE / SAVE_EACH)."""
    names = {c.name for c in flat}
    for fn in ctx.funcs.values():
        names |= {c.name for c in _flatten(fn.body)}
    if names & {"SAVE", "SAVE_EACH"} or ctx.unresolved_include:
        return []
    if any(ctx.is_user_command(n) and n not in ctx.funcs for n in names):
        return []                             # an imported FUNC may save
    return [Diagnostic(
        "PX002", Severity.WARNING,
        "Pipeline has no SAVE command. Output will not be written to disk.",
    )]


@_rule
def rule_px003_contours_draw(ctx, flat, registry):
    """PX003 — FIND_CONTOURS should be followed by a draw command."""
    draw_cmds = {"DRAW_BOUNDING_BOXES", "DRAW_CONTOURS"}
    diags = []
    for i, cmd in enumerate(flat):
        if cmd.name == "FIND_CONTOURS":
            following = {c.name for c in flat[i+1:i+4]}
            if not following.intersection(draw_cmds):
                diags.append(_diag(
                    "PX003", Severity.WARNING,
                    "FIND_CONTOURS is not followed by DRAW_BOUNDING_BOXES or DRAW_CONTOURS. "
                    "Detected contours will be discarded.", cmd))
    return diags


@_rule
def rule_px004_overlay_needs_checkpoint(ctx, flat, registry):
    """PX004 — OVERLAY / BLEND / COMPARE need an existing checkpoint/buffer."""
    diags = []
    created: Set[str] = set()
    for stmt in _walk_main(ctx):
        if isinstance(stmt, (BufStmt, AllocStmt)):
            created.add(stmt.name)
            continue
        if not isinstance(stmt, Command):
            continue
        if stmt.name == "CHECKPOINT":
            created.add(_arg_text(stmt.args[0]) if stmt.args else "default")
        elif stmt.name in ("OVERLAY", "BLEND", "COMPARE"):
            if stmt.args and not isinstance(stmt.args[0], (str, BareIdent)):
                continue                      # computed name: cannot check
            name = _arg_text(stmt.args[0]) if stmt.args else "default"
            if name not in created:
                diags.append(_diag(
                    "PX004", Severity.ERROR,
                    f"{stmt.name} references checkpoint '{name}' which has not been "
                    "created yet.", stmt))
            if stmt.name == "COMPARE":
                created.add("diff")
    return diags


def _defs_in(stmts: List[Any]) -> Set[str]:
    """Names any statement in *stmts* (recursively) defines."""
    names: Set[str] = set()
    for s in walk_statements(stmts):
        if isinstance(s, SetVar) and s.target is None and s.var_name:
            names.add(s.var_name)
        elif isinstance(s, ForBlock):
            names.add(s.var_name)
        elif isinstance(s, RepeatBlock) and s.var_name:
            names.add(s.var_name)
    return names


class _Ref:
    """Normalised variable use (VarRef or strict IF bare word)."""
    def __init__(self, name, node):
        self.var_name, self.line = name, node.line
        self.col, self.file = node.col, node.file


def _var_refs(stmt: Any) -> List[Any]:
    out = []
    for v in statement_exprs(stmt):
        for n in walk_expr(v):
            if isinstance(n, VarRef):
                out.append(_Ref(n.var_name, n))
            elif isinstance(n, BareIdent) and n.strict == "if" and n.name != "_":
                out.append(_Ref(n.name, n))
    return out


@_rule
def rule_px005_undefined_variable(ctx, flat, registry):
    """PX005 — $variable used before it is defined (SET / FOR / toml)."""
    diags: List[Diagnostic] = []
    reported: Set[str] = set()
    all_globals = _defs_in(ctx.main_body) | ctx.config_vars

    def check(stmts: List[Any], defined: Set[str]) -> None:
        for stmt in stmts:
            if isinstance(stmt, FuncDef):
                continue
            for ref in _var_refs(stmt):
                if ref.var_name not in defined and ref.var_name not in reported:
                    reported.add(ref.var_name)
                    later = " (it is SET later — move the SET up)" \
                        if ref.var_name in all_globals else ""
                    diags.append(_diag("PX005", Severity.ERROR,
                                       f"Variable '${ref.var_name}' is used but never "
                                       f"defined with SET{later}.", ref))
            if isinstance(stmt, SetVar) and stmt.target is None and stmt.var_name:
                defined.add(stmt.var_name)
            inner = set(defined)
            if isinstance(stmt, ForBlock):
                inner.add(stmt.var_name)
            if isinstance(stmt, RepeatBlock):
                inner |= {"ITER", "ITER1"} | ({stmt.var_name} if stmt.var_name else set())
            if isinstance(stmt, (RepeatBlock, ForBlock)) or type(stmt).__name__ == "WhileBlock":
                inner |= _defs_in(stmt.body)    # defined on a previous iteration
            for body in iter_child_bodies(stmt):
                check(body, inner)
            # a SET inside any branch counts as defined afterwards (lenient)
            for body in iter_child_bodies(stmt):
                defined |= {s.var_name for s in walk_statements(body)
                            if isinstance(s, SetVar) and s.target is None and s.var_name}

    check(ctx.main_body, set(ctx.config_vars))
    for fn in ctx.funcs.values():
        check(fn.body, set(fn.params) | all_globals | _defs_in(fn.body))
    return diags


@_rule
def rule_px006_duplicate_commands(ctx, flat, registry):
    """PX006 — identical consecutive commands (possible copy-paste error)."""
    diags = []
    for i in range(1, len(flat)):
        a, b = flat[i-1], flat[i]
        if a.name == b.name and a.args == b.args and a.name not in (
            "SAVE", "CHECKPOINT", "PRINT_INFO", "ERODE", "DILATE"
        ):
            diags.append(_diag(
                "PX006", Severity.INFO,
                f"'{a.name}' appears twice in a row with the same arguments. "
                "This may be a copy-paste mistake.", b))
    return diags


@_rule
def rule_px007_resize_after_filter(ctx, flat, registry):
    """PX007 — RESIZE after a destructive filter (unusual ordering)."""
    filter_cmds = {"BLUR", "MEDIAN_BLUR", "CANNY", "THRESHOLD", "THRESHOLD_OTSU",
                   "ADAPTIVE_THRESHOLD", "ERODE", "DILATE"}
    diags = []
    seen_filter = False
    for cmd in flat:
        if cmd.name in filter_cmds:
            seen_filter = True
        elif cmd.name == "LOAD":
            seen_filter = False
        elif cmd.name == "RESIZE" and seen_filter:
            diags.append(_diag(
                "PX007", Severity.WARNING,
                "RESIZE appears after a filtering step. "
                "Consider resizing before processing for better performance and results.",
                cmd))
    return diags


@_rule
def rule_px008_repeat_count(ctx, flat, registry):
    """PX008 — REPEAT literal count: negative / non-integer (error), 0 (warning)."""
    diags = []
    for stmt in walk_statements(ctx.main):
        if not isinstance(stmt, RepeatBlock):
            continue
        c = stmt.count
        if isinstance(c, bool) or isinstance(c, str):
            continue
        if isinstance(c, float):
            diags.append(_diag("PX008", Severity.ERROR,
                               f"REPEAT count must be an integer, got {c}.", stmt))
        elif isinstance(c, int) and c < 0:
            diags.append(_diag("PX008", Severity.ERROR,
                               f"REPEAT count must be ≥ 0, got {c}.", stmt))
        elif isinstance(c, int) and c == 0:
            diags.append(_diag("PX008", Severity.WARNING,
                               "REPEAT 0 — the loop body never runs.", stmt))
    return diags


@_rule
def rule_px009_empty_pipeline(ctx, flat, registry):
    """PX009 — pipeline has no commands at all."""
    if not flat and not ctx.funcs and not ctx.main_body:
        return [Diagnostic("PX009", Severity.INFO, "Pipeline is empty.")]
    return []


@_rule
def rule_px010_threshold_before_gray(ctx, flat, registry):
    """PX010 — threshold applied before GRAYSCALE."""
    threshold_cmds = {"THRESHOLD", "THRESHOLD_OTSU", "ADAPTIVE_THRESHOLD"}
    diags = []
    gray_seen = False
    for cmd in flat:
        if cmd.name == "GRAYSCALE":
            gray_seen = True
        elif cmd.name in threshold_cmds and not gray_seen:
            diags.append(_diag(
                "PX010", Severity.WARNING,
                f"{cmd.name} applied before GRAYSCALE. "
                "Thresholding a colour image may give unexpected results.", cmd))
    return diags


@_rule
def rule_px011_unknown_commands(ctx, flat, registry):
    """PX011 — command name is neither a FUNC nor a registered command."""
    if registry is None:
        return []
    diags = []
    cmds = list(flat)
    for fn in ctx.funcs.values():
        cmds.extend(_flatten(fn.body))
    for cmd in cmds:
        if cmd.name not in registry and not ctx.is_user_command(cmd.name):
            pool = list(registry.list_commands()) + list(ctx.funcs)
            close = difflib.get_close_matches(cmd.name, pool, n=3, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            diags.append(_diag("PX011", Severity.ERROR,
                               f"Unknown command '{cmd.name}'.{hint}", cmd))
    return diags


@_rule
def rule_px012_bare_word_typo(ctx, flat, registry):
    """PX012 — bare word that becomes a string but looks like a variable typo."""
    from pixlang.parser.parser import DTYPE_NAMES
    defined = _defs_in(ctx.main_body) | ctx.config_vars
    for fn in ctx.funcs.values():
        defined |= set(fn.params) | _defs_in(fn.body)
    buffers = {s.name for s in walk_statements(ctx.main) if isinstance(s, (BufStmt, AllocStmt))}
    known = defined | buffers | set(DTYPE_NAMES) | {"_"}
    diags = []
    seen: Set[tuple] = set()
    for stmt in walk_statements(ctx.main):
        for value in statement_exprs(stmt):
            for node in walk_expr(value):
                if not isinstance(node, BareIdent) or node.strict or node.name in known:
                    continue
                close = difflib.get_close_matches(node.name, sorted(defined), n=1, cutoff=0.75)
                key = (node.name, node.line)
                if close and key not in seen:
                    seen.add(key)
                    diags.append(_diag(
                        "PX012", Severity.WARNING,
                        f"'{node.name}' is not a variable and will be used as the string "
                        f"\"{node.name}\" — did you mean ${close[0]}?", node))
    return diags
