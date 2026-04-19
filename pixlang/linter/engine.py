# pixlang/linter/engine.py
"""
PixLang Linter  v0.3

Analyses a parsed Pipeline AST and returns a list of Diagnostics.
Each rule is a small function annotated with @rule(code, severity).

Built-in rules:
  PX001  ERROR   — pipeline does not start with LOAD
  PX002  ERROR   — SAVE missing (output never written)
  PX003  WARNING — FIND_CONTOURS not followed by a draw command
  PX004  WARNING — OVERLAY used without a preceding CHECKPOINT
  PX005  WARNING — undefined variable referenced via $name
  PX006  INFO    — repeated identical command (possible copy-paste)
  PX007  WARNING — RESIZE used after a filtering step (unusual order)
  PX008  ERROR   — REPEAT with non-positive count literal
  PX009  INFO    — pipeline has no commands
  PX010  WARNING — THRESHOLD used before GRAYSCALE
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, List, Set

from pixlang.parser.ast_nodes import (
    Command, IfBlock, Pipeline, RepeatBlock, SetVar, Statement, VarRef,
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

    def __str__(self):
        icon = {"info": "ℹ", "warning": "⚠", "error": "✗"}[self.severity.value]
        loc  = f"line {self.line}  " if self.line else ""
        return f"  {icon}  [{self.code}] {loc}{self.message}"


# ── Rule registry ─────────────────────────────────────────────────────────────

_RULES: List[Callable] = []

def _rule(fn: Callable) -> Callable:
    _RULES.append(fn)
    return fn


# ── Linter ────────────────────────────────────────────────────────────────────

class Linter:
    """
    Run all registered rules over a Pipeline AST.

    Usage:
        linter = Linter(registry)
        diags  = linter.lint(pipeline)
        for d in diags:
            print(d)
    """

    def __init__(self, registry=None):
        self.registry = registry  # optional, used to validate command names

    def lint(self, pipeline: Pipeline) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        # Flatten the pipeline to a linear command sequence for simple rules
        flat = _flatten(pipeline.commands)

        for rule_fn in _RULES:
            try:
                results = rule_fn(pipeline, flat, self.registry)
                if results:
                    diags.extend(results)
            except Exception:
                pass  # never let a broken rule crash the linter

        return sorted(diags, key=lambda d: (d.line, d.code))

    @property
    def rule_count(self) -> int:
        return len(_RULES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten(stmts: List[Statement]) -> List[Command]:
    """Recursively extract all Command nodes from nested blocks."""
    cmds: List[Command] = []
    for stmt in stmts:
        if isinstance(stmt, Command):
            cmds.append(stmt)
        elif isinstance(stmt, (IfBlock, RepeatBlock)):
            cmds.extend(_flatten(stmt.body))
    return cmds


def _collect_set_vars(stmts: List[Statement]) -> Set[str]:
    names: Set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, SetVar):
            names.add(stmt.var_name)
        elif isinstance(stmt, (IfBlock, RepeatBlock)):
            names.update(_collect_set_vars(stmt.body))
    return names


def _collect_var_refs(stmts: List[Statement]) -> List[tuple]:
    """Return (var_name, line) for every VarRef in arg lists."""
    refs = []
    for stmt in stmts:
        if isinstance(stmt, Command):
            for arg in stmt.args:
                if isinstance(arg, VarRef):
                    refs.append((arg.var_name, stmt.line))
        elif isinstance(stmt, RepeatBlock):
            if isinstance(stmt.count, VarRef):
                refs.append((stmt.count.var_name, stmt.line))
            refs.extend(_collect_var_refs(stmt.body))
        elif isinstance(stmt, IfBlock):
            refs.extend(_collect_var_refs(stmt.body))
    return refs


# ── Rules ─────────────────────────────────────────────────────────────────────

@_rule
def rule_px001_starts_with_load(pipeline, flat, registry):
    """PX001 — pipeline should start with LOAD."""
    cmds = [s for s in pipeline.commands if isinstance(s, Command)]
    if not cmds:
        return []
    if cmds[0].name != "LOAD":
        return [Diagnostic(
            "PX001", Severity.ERROR,
            f"Pipeline does not start with LOAD (first command is '{cmds[0].name}'). "
            "No image will be loaded.",
            line=cmds[0].line,
        )]
    return []


@_rule
def rule_px002_has_save(pipeline, flat, registry):
    """PX002 — pipeline should end with SAVE."""
    saves = [c for c in flat if c.name == "SAVE"]
    if not saves:
        return [Diagnostic(
            "PX002", Severity.WARNING,
            "Pipeline has no SAVE command. Output will not be written to disk.",
        )]
    return []


@_rule
def rule_px003_contours_draw(pipeline, flat, registry):
    """PX003 — FIND_CONTOURS should be followed by a draw command."""
    draw_cmds = {"DRAW_BOUNDING_BOXES", "DRAW_CONTOURS"}
    diags = []
    for i, cmd in enumerate(flat):
        if cmd.name == "FIND_CONTOURS":
            following = {c.name for c in flat[i+1:i+4]}
            if not following.intersection(draw_cmds):
                diags.append(Diagnostic(
                    "PX003", Severity.WARNING,
                    "FIND_CONTOURS is not followed by DRAW_BOUNDING_BOXES or DRAW_CONTOURS. "
                    "Detected contours will be discarded.",
                    line=cmd.line,
                ))
    return diags


@_rule
def rule_px004_overlay_needs_checkpoint(pipeline, flat, registry):
    """PX004 — OVERLAY must be preceded by a CHECKPOINT."""
    diags = []
    checkpointed_names: Set[str] = set()
    for cmd in flat:
        if cmd.name == "CHECKPOINT":
            name = cmd.args[0] if cmd.args else "default"
            checkpointed_names.add(str(name))
        elif cmd.name == "OVERLAY":
            name = str(cmd.args[0]) if cmd.args else "default"
            if name not in checkpointed_names:
                diags.append(Diagnostic(
                    "PX004", Severity.ERROR,
                    f"OVERLAY references checkpoint '{name}' which has not been created yet.",
                    line=cmd.line,
                ))
    return diags


@_rule
def rule_px005_undefined_variable(pipeline, flat, registry):
    """PX005 — $variable referenced but never SET."""
    defined = _collect_set_vars(pipeline.commands)
    defined.add("ITER")    # built-in loop variable
    defined.add("ITER1")
    refs    = _collect_var_refs(pipeline.commands)
    diags   = []
    seen    = set()
    for name, line in refs:
        if name not in defined and name not in seen:
            seen.add(name)
            diags.append(Diagnostic(
                "PX005", Severity.ERROR,
                f"Variable '${name}' is used but never defined with SET.",
                line=line,
            ))
    return diags


@_rule
def rule_px006_duplicate_commands(pipeline, flat, registry):
    """PX006 — identical consecutive commands (possible copy-paste error)."""
    diags = []
    for i in range(1, len(flat)):
        a, b = flat[i-1], flat[i]
        if a.name == b.name and a.args == b.args and a.name not in (
            "SAVE", "CHECKPOINT", "PRINT_INFO", "ERODE", "DILATE"
        ):
            diags.append(Diagnostic(
                "PX006", Severity.INFO,
                f"'{a.name}' appears twice in a row with the same arguments. "
                "This may be a copy-paste mistake.",
                line=b.line,
            ))
    return diags


@_rule
def rule_px007_resize_after_filter(pipeline, flat, registry):
    """PX007 — RESIZE after a destructive filter (unusual ordering)."""
    filter_cmds = {"BLUR", "MEDIAN_BLUR", "CANNY", "THRESHOLD", "THRESHOLD_OTSU",
                   "ADAPTIVE_THRESHOLD", "ERODE", "DILATE"}
    diags = []
    seen_filter = False
    for cmd in flat:
        if cmd.name in filter_cmds:
            seen_filter = True
        elif cmd.name == "RESIZE" and seen_filter:
            diags.append(Diagnostic(
                "PX007", Severity.WARNING,
                "RESIZE appears after a filtering step. "
                "Consider resizing before processing for better performance and results.",
                line=cmd.line,
            ))
    return diags


@_rule
def rule_px008_repeat_nonpositive(pipeline, flat, registry):
    """PX008 — REPEAT with a literal count <= 0."""
    diags = []
    def _check(stmts):
        for stmt in stmts:
            if isinstance(stmt, RepeatBlock):
                if isinstance(stmt.count, int) and stmt.count <= 0:
                    diags.append(Diagnostic(
                        "PX008", Severity.ERROR,
                        f"REPEAT count must be > 0, got {stmt.count}.",
                        line=stmt.line,
                    ))
                _check(stmt.body)
    _check(pipeline.commands)
    return diags


@_rule
def rule_px009_empty_pipeline(pipeline, flat, registry):
    """PX009 — pipeline has no commands at all."""
    if not flat:
        return [Diagnostic("PX009", Severity.INFO, "Pipeline is empty.")]
    return []


@_rule
def rule_px010_threshold_before_gray(pipeline, flat, registry):
    """PX010 — threshold applied before GRAYSCALE."""
    threshold_cmds = {"THRESHOLD", "THRESHOLD_OTSU", "ADAPTIVE_THRESHOLD"}
    diags = []
    gray_seen = False
    for cmd in flat:
        if cmd.name == "GRAYSCALE":
            gray_seen = True
        elif cmd.name in threshold_cmds and not gray_seen:
            diags.append(Diagnostic(
                "PX010", Severity.WARNING,
                f"{cmd.name} applied before GRAYSCALE. "
                "Thresholding a colour image may give unexpected results.",
                line=cmd.line,
            ))
    return diags


@_rule
def rule_px011_unknown_commands(pipeline, flat, registry):
    """PX011 — command name not in registry."""
    if registry is None:
        return []
    diags = []
    for cmd in flat:
        if cmd.name not in registry:
            diags.append(Diagnostic(
                "PX011", Severity.ERROR,
                f"Unknown command '{cmd.name}'.",
                line=cmd.line,
            ))
    return diags
