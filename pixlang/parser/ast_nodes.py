"""
AST node definitions for PixLang v0.5.

Every node carries a source location: ``line`` (1-based), ``col`` (1-based)
and ``file`` (None for in-memory sources). Nodes holding a statement list
call it ``body`` (plus ``else_body`` / ``elifs[i].body``) so tools can walk
blocks generically — see ``iter_child_bodies``.

Statements
  Command      RESIZE 640 480 / SHARPEN_TWICE 3 / f.SHARPEN_TWICE 3
  SetVar       SET x <expr> / SET buf[y, x, c] <expr> / SET *p <expr> / *p = <expr>
  IfBlock      IF <expr> ... [ELIF <expr> ...]* [ELSE ...] ENDIF
  RepeatBlock  REPEAT <n> [AS i] ... END
  WhileBlock   WHILE <expr> ... END
  ForBlock     FOR i = a TO b [STEP s] ... END
  BreakStmt / ContinueStmt
  RoiBlock     ROI x y w h ... ROI_RESET
  WithBlock    WITH *p ... END
  IncludeStmt  INCLUDE "other.pxl"
  AssertStmt   ASSERT <expr> ["message"]
  FuncDef      [EXPORT] FUNC name(params) ... ENDFUNC
  ReturnStmt   RETURN [<expr>]
  GlobalStmt   GLOBAL a, b
  ImportStmt   IMPORT "lib.pxl" AS f   |   FROM "lib.pxl" IMPORT a, b
  AllocStmt    ALLOC name w h [c] [dtype] [fill]
  BufStmt      BUF name [<expr>]
  UseStmt      USE <expr>
  FreeStmt     FREE a, b

Expressions
  Literal, VarRef ($x), BareIdent (x), Unary, Binary, Index, Slice, Attr,
  Call, AsType (expr AS u32)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple, Union


@dataclass
class ASTNode:
    """Base node: every node knows where it came from."""
    line: int = 0
    col:  int = 0
    file: Optional[str] = None

    @property
    def loc(self):
        from .errors import SourceLoc
        return SourceLoc(self.file, self.line, self.col)


# ── Expressions ───────────────────────────────────────────────────────────────

@dataclass
class Expr(ASTNode):
    """Base class for expression nodes."""

    def to_source(self) -> str:      # pragma: no cover - overridden
        return "?"

    def __str__(self) -> str:
        return self.to_source()


def expr_source(value: Any) -> str:
    """Render an expression node or a plain literal back to PixLang text."""
    if isinstance(value, Expr):
        return value.to_source()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return repr(value)


@dataclass
class Literal(Expr):
    """A constant: int, float, str or bool."""
    value: Any = None

    def to_source(self) -> str:
        return expr_source(self.value)


@dataclass
class VarRef(Expr):
    """$name — a reference to a variable; resolved at runtime."""
    var_name: str = ""

    def to_source(self) -> str:
        return f"${self.var_name}"


@dataclass
class BareIdent(Expr):
    """name — bare identifier.

    Resolution order at runtime:
      1. variable with this name           → its value
      2. buffer slot with this name        → the buffer (by reference)
      3. ``_``                             → the current image register
      4. otherwise                         → the name as a string literal
         (legacy: colormap names etc.). With ``strict`` set (left side of an
         IF / ASSERT comparison) an undefined name is an error instead —
         ASSERT first falls back to the image subjects width/height/...
    """
    name: str = ""
    strict: str = ""        # "" | "if" | "assert"

    def to_source(self) -> str:
        return self.name


@dataclass
class Unary(Expr):
    """op operand — op is one of: - ~ not & *"""
    op: str = "-"
    operand: Any = None

    def to_source(self) -> str:
        sep = " " if self.op == "not" else ""
        return f"{self.op}{sep}{expr_source(self.operand)}"


@dataclass
class Binary(Expr):
    """left op right (arithmetic, bitwise, comparison, and/or)."""
    op: str = "+"
    left: Any = None
    right: Any = None

    def to_source(self) -> str:
        return f"({expr_source(self.left)} {self.op} {expr_source(self.right)})"


@dataclass
class Slice(Expr):
    """start:stop[:step] inside an index."""
    start: Any = None
    stop: Any = None
    step: Any = None

    def to_source(self) -> str:
        s = (expr_source(self.start) if self.start is not None else "") + ":" + \
            (expr_source(self.stop) if self.stop is not None else "")
        if self.step is not None:
            s += ":" + expr_source(self.step)
        return s


@dataclass
class Index(Expr):
    """target[i, j, ...] — PEEK / slicing."""
    target: Any = None
    indices: List[Any] = field(default_factory=list)

    def to_source(self) -> str:
        return f"{expr_source(self.target)}[{', '.join(expr_source(i) for i in self.indices)}]"


@dataclass
class Attr(Expr):
    """target.name — buffer attribute (width, dtype, ...) or module member."""
    target: Any = None
    name: str = ""

    def to_source(self) -> str:
        return f"{expr_source(self.target)}.{self.name}"


@dataclass
class Call(Expr):
    """name(args) — builtin or user expression function; module.name(args)."""
    func: str = ""                          # "clamp" or "f.clamp"
    args: List[Any] = field(default_factory=list)
    func_ref: Optional[str] = None          # set by the linker: qualified name

    def to_source(self) -> str:
        return f"{self.func}({', '.join(expr_source(a) for a in self.args)})"


@dataclass
class AsType(Expr):
    """expr AS dtype — reinterpret the bytes of a buffer (no copy)."""
    value: Any = None
    dtype: str = ""

    def to_source(self) -> str:
        return f"{expr_source(self.value)} AS {self.dtype}"


# ── Statements ────────────────────────────────────────────────────────────────

@dataclass
class Command(ASTNode):
    """
    A single pipeline command, e.g.:
        RESIZE 640 480   -> Command(name="RESIZE", args=[640, 480])
        LOAD "img.png"   -> Command(name="LOAD",   args=["img.png"])
    Literal args are plain Python values; VarRef / BareIdent / parenthesised
    expressions are resolved at runtime. ``func_ref`` is set by the linker
    when the name resolves to a user FUNC.
    """
    name: str = ""
    args: List[Any] = field(default_factory=list)
    func_ref: Optional[str] = None

    def __repr__(self):
        return f"Command({self.name!r}, args={self.args}, line={self.line})"


@dataclass
class SetVar(ASTNode):
    """SET name <expr>  — binds a name; with ``target`` it writes through
    an index (POKE) or a pointer: SET buf[y, x] 255 / SET *p 0."""
    var_name: str = ""
    value: Any = None
    target: Any = None          # Index | Unary('*') | None (plain variable)


@dataclass
class ElifClause(ASTNode):
    """ELIF <cond> + its body (part of an IfBlock)."""
    cond: Any = None
    body: List[Statement] = field(default_factory=list)


@dataclass
class IfBlock(ASTNode):
    """IF <cond> ... [ELIF <cond> ...] [ELSE ...] ENDIF

    var_name / op / cmp_value are filled for the simple legacy form
    ``IF name <op> value`` (empty otherwise); ``cond`` always holds the
    full condition expression and ``cond_text`` its source text.
    """
    var_name: str = ""
    op: str = "=="
    cmp_value: Any = None
    body: List[Statement] = field(default_factory=list)
    cond: Any = None
    cond_text: str = ""
    elifs: List[ElifClause] = field(default_factory=list)
    else_body: List[Statement] = field(default_factory=list)


@dataclass
class RepeatBlock(ASTNode):
    """REPEAT <n> [AS i] ... END  — run the body n times."""
    count: Any = 1          # int or expression
    body: List[Statement] = field(default_factory=list)
    var_name: Optional[str] = None


@dataclass
class WhileBlock(ASTNode):
    """WHILE <cond> ... END"""
    cond: Any = None
    body: List[Statement] = field(default_factory=list)
    cond_text: str = ""


@dataclass
class ForBlock(ASTNode):
    """FOR i = start TO stop [STEP step] ... END  (stop is inclusive)."""
    var_name: str = ""
    start: Any = 0
    stop: Any = 0
    step: Any = None
    body: List[Statement] = field(default_factory=list)


@dataclass
class BreakStmt(ASTNode):
    """BREAK — leave the innermost loop."""


@dataclass
class ContinueStmt(ASTNode):
    """CONTINUE — next iteration of the innermost loop."""


@dataclass
class Pipeline(ASTNode):
    """Root node — an ordered list of statements."""
    commands: List[Statement] = field(default_factory=list)

    @property
    def body(self) -> List[Statement]:
        return self.commands


@dataclass
class IncludeStmt(ASTNode):
    """INCLUDE "other.pxl"  — splice another pipeline in (link time, or at
    run time when executing an unlinked Pipeline)."""
    path: str = ""


@dataclass
class AssertStmt(ASTNode):
    """ASSERT <expr> ["message"]

    Legacy form ``ASSERT subject op value`` fills subject/op/expected, where
    subject may be a variable or an image property:
        width height channels depth min max contour_count ndim
    ``cond`` holds the full expression (None for programmatically built nodes).
    """
    subject:  str = ""
    op:       str = "=="
    expected: Any = None
    message:  str = ""
    cond:     Any = None
    cond_text: str = ""


@dataclass
class RoiBlock(ASTNode):
    """ROI x y w h ... ROI_RESET — execute body within a rectangular region."""
    x: Any = 0
    y: Any = 0
    w: Any = 0
    h: Any = 0
    body: List[Statement] = field(default_factory=list)


@dataclass
class WithBlock(ASTNode):
    """WITH <view> ... END — run the body on a pointed region, in place."""
    target: Any = None
    body: List[Statement] = field(default_factory=list)


@dataclass
class FuncDef(ASTNode):
    """[EXPORT] FUNC name(params) ... ENDFUNC

    ALL_CAPS names are *command functions* (called like a command, operate on
    the caller's current image); other names are *expression functions*.
    """
    name: str = ""
    params: List[str] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)
    exported: bool = False

    @property
    def is_command(self) -> bool:
        return self.name.isupper()


@dataclass
class ReturnStmt(ASTNode):
    """RETURN [<expr>]"""
    value: Any = None


@dataclass
class GlobalStmt(ASTNode):
    """GLOBAL a, b — writes to these names go to the global scope."""
    names: List[str] = field(default_factory=list)


@dataclass
class ImportStmt(ASTNode):
    """IMPORT "path" [AS alias]  |  FROM "path" IMPORT a, b"""
    path: str = ""
    alias: Optional[str] = None
    names: List[str] = field(default_factory=list)     # FROM form
    name_locs: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class AllocStmt(ASTNode):
    """ALLOC name width height [channels] [dtype] [fill]"""
    name: str = ""
    width: Any = 0
    height: Any = 0
    channels: Any = 1
    dtype: str = "u8"
    fill: Any = 0


@dataclass
class BufStmt(ASTNode):
    """BUF name [<expr>] — store a copy of _ (or of expr) in a buffer slot."""
    name: str = ""
    value: Any = None


@dataclass
class UseStmt(ASTNode):
    """USE <expr> — load a copy of a buffer / array into _."""
    value: Any = None


@dataclass
class FreeStmt(ASTNode):
    """FREE a, b — release buffer slots (pointers into them become invalid)."""
    names: List[str] = field(default_factory=list)


# Type alias: anything that can appear in a pipeline body
Statement = Union[
    Command, SetVar, IfBlock, RepeatBlock, WhileBlock, ForBlock, BreakStmt,
    ContinueStmt, IncludeStmt, AssertStmt, RoiBlock, WithBlock, FuncDef,
    ReturnStmt, GlobalStmt, ImportStmt, AllocStmt, BufStmt, UseStmt, FreeStmt,
]


# ── Generic traversal helpers ─────────────────────────────────────────────────

def iter_child_bodies(stmt: Any) -> Iterator[List[Any]]:
    """Yield every nested statement list of *stmt* (body, elif bodies, else)."""
    body = getattr(stmt, "body", None)
    if isinstance(body, list):
        yield body
    for clause in getattr(stmt, "elifs", None) or []:
        yield clause.body
    else_body = getattr(stmt, "else_body", None)
    if else_body:
        yield else_body


def walk_statements(stmts: List[Any]) -> Iterator[Any]:
    """Depth-first pre-order walk over statements and nested blocks."""
    for stmt in stmts:
        yield stmt
        for body in iter_child_bodies(stmt):
            yield from walk_statements(body)


def walk_expr(value: Any) -> Iterator[Any]:
    """Yield every expression node inside *value* (pre-order)."""
    if isinstance(value, Expr):
        yield value
        if isinstance(value, Unary):
            yield from walk_expr(value.operand)
        elif isinstance(value, Binary):
            yield from walk_expr(value.left)
            yield from walk_expr(value.right)
        elif isinstance(value, Index):
            yield from walk_expr(value.target)
            for i in value.indices:
                yield from walk_expr(i)
        elif isinstance(value, Slice):
            for part in (value.start, value.stop, value.step):
                yield from walk_expr(part)
        elif isinstance(value, Attr):
            yield from walk_expr(value.target)
        elif isinstance(value, Call):
            for a in value.args:
                yield from walk_expr(a)
        elif isinstance(value, AsType):
            yield from walk_expr(value.value)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from walk_expr(v)


def statement_exprs(stmt: Any) -> List[Any]:
    """Return the expression-valued fields of a single statement (not bodies)."""
    if isinstance(stmt, Command):
        return list(stmt.args)
    if isinstance(stmt, SetVar):
        return [stmt.value, stmt.target]
    if isinstance(stmt, IfBlock):
        conds = [stmt.cond] if stmt.cond is not None else [stmt.cmp_value]
        return conds + [c.cond for c in stmt.elifs]
    if isinstance(stmt, RepeatBlock):
        return [stmt.count]
    if isinstance(stmt, WhileBlock):
        return [stmt.cond]
    if isinstance(stmt, ForBlock):
        return [stmt.start, stmt.stop, stmt.step]
    if isinstance(stmt, AssertStmt):
        return [stmt.cond] if stmt.cond is not None else [stmt.expected]
    if isinstance(stmt, RoiBlock):
        return [stmt.x, stmt.y, stmt.w, stmt.h]
    if isinstance(stmt, WithBlock):
        return [stmt.target]
    if isinstance(stmt, ReturnStmt):
        return [stmt.value]
    if isinstance(stmt, AllocStmt):
        return [stmt.width, stmt.height, stmt.channels, stmt.fill]
    if isinstance(stmt, (BufStmt, UseStmt)):
        return [stmt.value]
    return []
