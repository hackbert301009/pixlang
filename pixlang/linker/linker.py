# pixlang/linker/linker.py
"""
PixLang Linker  v0.5
═════════════════════

Turns one or more .pxl files into a single LinkedProgram the Executor can run
without touching the file system.

  compile  each file → ObjectUnit
             { path, sha256, funcs, exports, imports, globals, body, refs }
             INCLUDE statements are spliced in here (cycle-checked, resolved
             relative to the *including* file, locations keep their file tag).
  link     DFS from the entry unit over IMPORT / FROM ... IMPORT edges.
             Module paths resolve relative to the importing file, then the
             ``[link] search_paths`` of pixlang.toml.
             Reports: import / include cycles (full chain), duplicate
             definitions, names that are not exported, undefined functions,
             unknown commands (warning), unused imports (warning), command
             FUNCs shadowing builtin commands (warning).
  output   LinkedProgram { functions, symbols (per unit), init (module
             globals, dependencies first), body (entry statements) }.

Command / function lookup at run time: the calling unit's symbol table
(local FUNCs, FROM-imports, ``alias.NAME`` for IMPORT ... AS alias) first,
then the command registry (plugins / builtins) or the builtin expression
functions.
"""
from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pixlang.parser import parse
from pixlang.parser.ast_nodes import (
    Call, Command, FuncDef, GlobalStmt, ImportStmt, IncludeStmt, Pipeline,
    SetVar, statement_exprs, walk_expr, walk_statements,
)
from pixlang.parser.errors import PixLinkError, format_diagnostic

ENTRY_KEY = "<input>"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class LinkDiagnostic:
    """One linker finding with a source location."""
    severity: str          # "error" | "warning"
    code: str              # LNK001 ...
    message: str
    file: Optional[str] = None
    line: int = 0
    col: int = 0

    def __str__(self) -> str:
        return format_diagnostic(self.severity, f"[{self.code}] {self.message}",
                                 self.file, self.line, self.col)


@dataclass
class SymbolRef:
    """A use of a symbol: a command name or a function call."""
    name: str
    kind: str              # "command" | "call"
    node: Any              # the Command / Call node (func_ref is filled in)

    @property
    def file(self):
        return self.node.file

    @property
    def line(self):
        return self.node.line

    @property
    def col(self):
        return self.node.col


@dataclass
class ObjectUnit:
    """One compiled source file."""
    key: str                                   # resolved path or "<input>"
    path: Optional[Path]
    sha256: str
    pipeline: Pipeline
    funcs: Dict[str, FuncDef] = field(default_factory=dict)
    exports: Dict[str, FuncDef] = field(default_factory=dict)
    imports: List[ImportStmt] = field(default_factory=list)
    globals: List[SetVar] = field(default_factory=list)
    body: List[Any] = field(default_factory=list)
    refs: List[SymbolRef] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)

    def qualname(self, name: str) -> str:
        return f"{self.key}::{name}"


@dataclass
class LinkedProgram:
    """Everything the Executor needs; no file access required at run time."""
    entry: str
    units: Dict[str, ObjectUnit] = field(default_factory=dict)
    functions: Dict[str, FuncDef] = field(default_factory=dict)
    func_units: Dict[str, str] = field(default_factory=dict)
    symbols: Dict[str, Dict[str, str]] = field(default_factory=dict)
    aliases: Dict[str, Dict[str, str]] = field(default_factory=dict)
    init: List[Tuple[str, List[Any]]] = field(default_factory=list)
    body: List[Any] = field(default_factory=list)
    diagnostics: List[LinkDiagnostic] = field(default_factory=list)

    @property
    def errors(self) -> List[LinkDiagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> List[LinkDiagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]

    def lookup(self, unit: Optional[str], name: str) -> Optional[str]:
        """Resolve *name* as seen from *unit* to a function qualname."""
        table = self.symbols.get(unit or self.entry)
        if table is None:
            table = self.symbols.get(self.entry, {})
        return table.get(name)

    def symbol_map(self) -> str:
        """Human-readable map of units, functions and imports."""
        out: List[str] = []
        for key, unit in self.units.items():
            tag = " (entry)" if key == self.entry else ""
            out.append(f"{_short(key)}{tag}  sha256={unit.sha256[:12]}")
            for name, fn in unit.funcs.items():
                kind = "command" if fn.is_command else "function"
                exp = "export " if fn.exported or unit.exports.get(name) is fn else ""
                params = ", ".join(fn.params)
                out.append(f"    {exp}{kind:<8} {name}({params})  line {fn.line}")
            for imp in unit.imports:
                if imp.names:
                    out.append(f"    from {imp.path} import {', '.join(imp.names)}")
                else:
                    out.append(f"    import {imp.path} as {imp.alias or Path(imp.path).stem}")
            for inc in unit.includes:
                out.append(f"    include {_short(inc)}")
            for g in unit.globals:
                out.append(f"    global   {g.var_name}  line {g.line}")
            imported = {n: q for n, q in self.symbols.get(key, {}).items()
                        if self.func_units.get(q) != key}
            for n, q in sorted(imported.items()):
                out.append(f"    {n:<24} -> {_short(q)}")
        return "\n".join(out)


def _short(key: str) -> str:
    try:
        return str(Path(key.split("::")[0]).relative_to(Path.cwd())) + (
            "::" + key.split("::", 1)[1] if "::" in key else "")
    except ValueError:
        return key


# ── Linker ────────────────────────────────────────────────────────────────────

class Linker:
    """Compile and link PixLang units.

    Args:
        registry:     CommandRegistry used to check command names (optional)
        search_paths: extra module directories (``[link] search_paths``)
    """

    def __init__(self, registry=None, search_paths: Optional[Sequence[Path]] = None):
        self.registry = registry
        self.search_paths = [Path(p) for p in (search_paths or [])]

    # ── Public API ────────────────────────────────────────────────────────────

    def link(self, path, raise_on_error: bool = True) -> LinkedProgram:
        """Compile *path* and everything it imports/includes into a program."""
        path = Path(path).resolve()
        if not path.exists():
            raise PixLinkError(f"[PixLang] File not found: '{path}'",
                               file=str(path))
        source = path.read_text(encoding="utf-8")
        pipeline = parse(source, filename=str(path))
        return self.link_pipeline(pipeline, path=path, splice_includes=True,
                                  raise_on_error=raise_on_error, source=source)

    def link_pipeline(self, pipeline: Pipeline, path: Optional[Path] = None,
                      splice_includes: bool = True, raise_on_error: bool = True,
                      source: Optional[str] = None) -> LinkedProgram:
        """Link an already-parsed Pipeline (path anchors relative imports)."""
        key = str(Path(path).resolve()) if path else ENTRY_KEY
        program = LinkedProgram(entry=key)
        self._diags = program.diagnostics
        entry = self._compile(pipeline, key, Path(path).resolve() if path else None,
                              splice_includes, source)
        program.units[key] = entry
        self._link_unit(program, entry, [key], is_entry=True)
        program.body = entry.body
        program.init.append((key, entry.globals))
        self._check_refs(program)
        if raise_on_error and program.errors:
            raise PixLinkError(
                "[PixLang] Link failed:\n" + "\n".join(str(d) for d in program.errors),
                diagnostics=program.diagnostics,
                line=program.errors[0].line, col=program.errors[0].col,
                file=program.errors[0].file)
        return program

    def extend(self, program: LinkedProgram, unit_key: str, pipeline: Pipeline,
               file_key: str) -> List[Any]:
        """Runtime INCLUDE of an unlinked pipeline: register its FUNCs and
        IMPORTs into *unit_key*'s symbol table and return its statements."""
        self._diags = program.diagnostics
        before = len(program.diagnostics)
        sub = self._compile(pipeline, file_key, Path(file_key), False, None)
        program.units.setdefault(file_key, sub)
        self._link_unit(program, sub, [file_key], is_entry=True, symbols_key=unit_key)
        errors = [d for d in program.diagnostics[before:] if d.severity == "error"]
        if errors:
            raise PixLinkError("[PixLang] Link failed:\n" + "\n".join(map(str, errors)),
                               diagnostics=errors, line=errors[0].line,
                               col=errors[0].col, file=errors[0].file)
        return sub.body

    # ── Compile ───────────────────────────────────────────────────────────────

    def _error(self, code: str, msg: str, node: Any = None, file=None, line=0, col=0):
        if node is not None:
            file, line, col = node.file, node.line, node.col
        self._diags.append(LinkDiagnostic("error", code, msg, file, line, col))

    def _warn(self, code: str, msg: str, node: Any = None, file=None, line=0, col=0):
        if node is not None:
            file, line, col = node.file, node.line, node.col
        self._diags.append(LinkDiagnostic("warning", code, msg, file, line, col))

    def _compile(self, pipeline: Pipeline, key: str, path: Optional[Path],
                 splice_includes: bool, source: Optional[str]) -> ObjectUnit:
        if source is None and path is not None and path.exists():
            source = path.read_text(encoding="utf-8")
        digest = hashlib.sha256((source or "").encode("utf-8")).hexdigest()
        unit = ObjectUnit(key=key, path=path, sha256=digest, pipeline=pipeline)

        stmts = pipeline.commands
        if splice_includes:
            stmts = self._splice(stmts, path, [key] if path else [], True, unit)

        for stmt in stmts:
            if isinstance(stmt, FuncDef):
                if stmt.name in unit.funcs:
                    first = unit.funcs[stmt.name]
                    self._error("LNK002",
                                f"duplicate definition of FUNC '{stmt.name}' "
                                f"(first defined at {first.file or '<input>'}:{first.line})",
                                stmt)
                    continue
                unit.funcs[stmt.name] = stmt
            elif isinstance(stmt, ImportStmt):
                unit.imports.append(stmt)
            elif isinstance(stmt, SetVar) and stmt.target is None:
                unit.body.append(stmt)
            elif isinstance(stmt, GlobalStmt):
                continue                       # top-level GLOBAL is a no-op
            else:
                unit.body.append(stmt)

        if any(f.exported for f in unit.funcs.values()):
            unit.exports = {n: f for n, f in unit.funcs.items() if f.exported}
        else:
            unit.exports = dict(unit.funcs)    # no EXPORT at all: everything public

        # Symbol references (commands + calls) in the body and all functions
        for stmt in walk_statements(stmts):
            if isinstance(stmt, Command):
                unit.refs.append(SymbolRef(stmt.name, "command", stmt))
            for value in statement_exprs(stmt):
                for node in walk_expr(value):
                    if isinstance(node, Call):
                        unit.refs.append(SymbolRef(node.func, "call", node))
        return unit

    def _splice(self, stmts: List[Any], cur: Optional[Path], stack: List[str],
                top_level: bool, unit: ObjectUnit) -> List[Any]:
        """Replace INCLUDE statements by the included file's statements."""
        out: List[Any] = []
        for stmt in stmts:
            if isinstance(stmt, IncludeStmt):
                base = Path(stmt.file).parent if stmt.file else (
                    cur.parent if cur else Path("."))
                target = (base / stmt.path).resolve()
                tkey = str(target)
                if tkey in stack:
                    chain = " → ".join(Path(s).name for s in stack + [tkey])
                    self._error("LNK003", f"INCLUDE cycle: {chain}", stmt)
                    continue
                if not target.exists():
                    self._error("LNK004", f"[INCLUDE] File not found: '{target}'", stmt)
                    continue
                unit.includes.append(tkey)
                sub = parse(target.read_text(encoding="utf-8"), filename=tkey)
                spliced = self._splice(sub.commands, target, stack + [tkey], top_level, unit)
                if not top_level:
                    for s in spliced:
                        if isinstance(s, (FuncDef, ImportStmt)):
                            self._error("LNK005",
                                        "FUNC / IMPORT in a file INCLUDEd inside a block "
                                        "(INCLUDE it at the top level instead)", s)
                out.extend(spliced)
                continue
            if isinstance(stmt, FuncDef):
                stmt.body = self._splice(stmt.body, cur, stack, False, unit)
            else:
                if isinstance(getattr(stmt, "body", None), list):
                    stmt.body = self._splice(stmt.body, cur, stack, False, unit)
                for clause in getattr(stmt, "elifs", None) or []:
                    clause.body = self._splice(clause.body, cur, stack, False, unit)
                if getattr(stmt, "else_body", None):
                    stmt.else_body = self._splice(stmt.else_body, cur, stack, False, unit)
            out.append(stmt)
        return out

    # ── Link ──────────────────────────────────────────────────────────────────

    def _resolve_module(self, imp: ImportStmt, unit: ObjectUnit) -> Optional[Path]:
        bases = []
        if imp.file:
            bases.append(Path(imp.file).parent)
        elif unit.path:
            bases.append(unit.path.parent)
        else:
            bases.append(Path("."))
        bases.extend(self.search_paths)
        for base in bases:
            cand = (base / imp.path).resolve()
            if cand.exists():
                return cand
        return None

    def _link_unit(self, program: LinkedProgram, unit: ObjectUnit, stack: List[str],
                   is_entry: bool = False, symbols_key: Optional[str] = None) -> None:
        skey = symbols_key or unit.key
        table = program.symbols.setdefault(skey, {})
        aliases = program.aliases.setdefault(skey, {})
        for name, fn in unit.funcs.items():
            q = unit.qualname(name)
            program.functions[q] = fn
            program.func_units[q] = skey
            table[name] = q

        if not is_entry:
            # Imported modules: only definitions and global initialisers
            for stmt in unit.body:
                if not (isinstance(stmt, SetVar) and stmt.target is None):
                    self._error("LNK006",
                                "only FUNC, SET, GLOBAL and IMPORT are allowed at the top "
                                "level of an imported module", stmt)
            unit.globals, unit.body = [s for s in unit.body if isinstance(s, SetVar)], []
        else:
            unit.globals = []

        for imp in unit.imports:
            target = self._resolve_module(imp, unit)
            if target is None:
                where = ", ".join(str(p) for p in self.search_paths)
                self._error("LNK001",
                            f"module not found: '{imp.path}'"
                            + (f" (also searched: {where})" if where else ""), imp)
                continue
            tkey = str(target)
            if tkey in stack:
                chain = " → ".join(Path(s).name for s in stack + [tkey])
                self._error("LNK003", f"import cycle: {chain}", imp)
                continue
            if tkey not in program.units:
                source = target.read_text(encoding="utf-8")
                sub = self._compile(parse(source, filename=tkey), tkey, target, True, source)
                program.units[tkey] = sub
                self._link_unit(program, sub, stack + [tkey])
                if sub.globals:
                    program.init.append((tkey, sub.globals))
            mod = program.units[tkey]

            if imp.names:
                for i, name in enumerate(imp.names):
                    line, col = imp.name_locs[i] if i < len(imp.name_locs) else (imp.line, imp.col)
                    if name not in mod.funcs:
                        hint = _suggest(name, list(mod.funcs))
                        self._error("LNK007", f"'{Path(imp.path).name}' has no symbol "
                                    f"'{name}'{hint}", file=imp.file, line=line, col=col)
                        continue
                    if name not in mod.exports:
                        self._error("LNK008", f"'{name}' is not exported by "
                                    f"'{Path(imp.path).name}' (mark it EXPORT FUNC)",
                                    file=imp.file, line=line, col=col)
                        continue
                    q = mod.qualname(name)
                    if name in table and table[name] != q:
                        self._error("LNK002", f"'{name}' is already defined in this "
                                    "file (local FUNC or another import)",
                                    file=imp.file, line=line, col=col)
                        continue
                    table[name] = q
            else:
                alias = imp.alias or Path(imp.path).stem
                if alias in aliases and aliases[alias] != tkey:
                    self._error("LNK002", f"import alias '{alias}' is already used", imp)
                    continue
                aliases[alias] = tkey
                for name in mod.exports:
                    table[f"{alias}.{name}"] = mod.qualname(name)

    # ── Reference checks ──────────────────────────────────────────────────────

    def _check_refs(self, program: LinkedProgram) -> None:
        from pixlang.executor.builtins import BUILTIN_FUNCTIONS

        for key, unit in program.units.items():
            table = program.symbols.get(key, {})
            used: set = set()
            for ref in unit.refs:
                q = table.get(ref.name)
                if q is not None:
                    used.add(ref.name)
                    fn = program.functions[q]
                    if ref.kind == "call" and fn.is_command:
                        self._error("LNK009", f"command FUNC '{ref.name}' cannot be "
                                    "called in an expression; use it as a statement",
                                    ref.node)
                    ref.node.func_ref = q
                    continue
                if ref.kind == "call":
                    if ref.name in BUILTIN_FUNCTIONS:
                        continue
                    prefix = ref.name.split(".")[0] if "." in ref.name else None
                    pool = list(table) + list(BUILTIN_FUNCTIONS)
                    if prefix and prefix in program.aliases.get(key, {}):
                        self._error("LNK010", f"module '{prefix}' has no exported function "
                                    f"'{ref.name.split('.', 1)[1]}'", ref.node)
                    else:
                        self._error("LNK010", f"undefined function '{ref.name}'"
                                    + _suggest(ref.name, pool), ref.node)
                else:
                    if "." in ref.name:
                        prefix = ref.name.split(".")[0]
                        if prefix in program.aliases.get(key, {}):
                            self._error("LNK010", f"module '{prefix}' has no exported command "
                                        f"'{ref.name.split('.', 1)[1]}'", ref.node)
                        else:
                            self._error("LNK010", f"unknown module alias '{prefix}' in "
                                        f"'{ref.name}'", ref.node)
                    elif self.registry is not None and ref.name not in self.registry:
                        pool = list(table) + list(self.registry.list_commands())
                        self._warn("LNK011", f"unknown command '{ref.name}' (not a FUNC or a "
                                   "registered command)" + _suggest(ref.name, pool), ref.node)

            # Shadowing builtin commands
            if self.registry is not None:
                for name, fn in unit.funcs.items():
                    if fn.is_command and name in self.registry:
                        self._warn("LNK012", f"FUNC '{name}' shadows the registered command "
                                   f"'{name}' in this file", fn)

            # Unused imports
            for imp in unit.imports:
                if imp.names:
                    for i, name in enumerate(imp.names):
                        if name not in used and name in table:
                            line, col = imp.name_locs[i] if i < len(imp.name_locs) else (imp.line, imp.col)
                            self._warn("LNK013", f"unused import '{name}'",
                                       file=imp.file, line=line, col=col)
                else:
                    alias = imp.alias or Path(imp.path).stem
                    if not any(n.startswith(alias + ".") for n in used):
                        self._warn("LNK013", f"unused import '{imp.path}' (alias '{alias}')", imp)

        # Duplicate global initialisers across modules
        seen: Dict[str, SetVar] = {}
        for key, stmts in program.init:
            if key == program.entry:
                continue
            for s in stmts:
                if s.var_name in seen and seen[s.var_name].file != s.file:
                    first = seen[s.var_name]
                    self._warn("LNK014", f"global '{s.var_name}' is also initialised in "
                               f"{first.file}:{first.line}", s)
                seen.setdefault(s.var_name, s)


def _suggest(name: str, pool: Sequence[str]) -> str:
    close = difflib.get_close_matches(name, list(pool), n=3, cutoff=0.6)
    return f" (did you mean {', '.join(close)}?)" if close else ""


# ── Convenience ───────────────────────────────────────────────────────────────

def link(path, registry=None, search_paths=None, raise_on_error: bool = True) -> LinkedProgram:
    """Link the .pxl file at *path*; search paths default to pixlang.toml's
    ``[link] search_paths``."""
    if search_paths is None:
        from pixlang.config import load_config
        search_paths = load_config(Path(path).resolve()).link_search_paths
    return Linker(registry, search_paths).link(path, raise_on_error=raise_on_error)
