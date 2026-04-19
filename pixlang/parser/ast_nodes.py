# pixlang/parser/ast_nodes.py
"""
AST node definitions for PixLang v0.3.

New node types:
  SetVar      — SET width 640
  VarRef      — $width  (resolves to stored value at execution time)
  IfBlock     — IF <expr> ... ENDIF
  RepeatBlock — REPEAT <n> ... END
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Union


@dataclass
class ASTNode:
    """Base node."""
    line: int = 0


@dataclass
class Command(ASTNode):
    """
    A single pipeline command, e.g.:
        RESIZE 640 480   -> Command(name="RESIZE", args=[640, 480])
        LOAD "img.png"   -> Command(name="LOAD",   args=["img.png"])
    Args may contain VarRef nodes that are resolved at runtime.
    """
    name: str = ""
    args: List[Any] = field(default_factory=list)

    def __repr__(self):
        return f"Command({self.name!r}, args={self.args}, line={self.line})"


@dataclass
class SetVar(ASTNode):
    """SET name value  — binds a name to a scalar value."""
    var_name: str = ""
    value: Any = None


@dataclass
class VarRef(ASTNode):
    """$name — a reference to a variable; resolved at runtime."""
    var_name: str = ""


@dataclass
class IfBlock(ASTNode):
    """IF <var> <op> <value> ... ENDIF"""
    var_name: str = ""
    op: str = "=="
    cmp_value: Any = None
    body: List["Statement"] = field(default_factory=list)


@dataclass
class RepeatBlock(ASTNode):
    """REPEAT <n> ... END  — run the body n times."""
    count: Any = 1          # int or VarRef
    body: List["Statement"] = field(default_factory=list)


@dataclass
class Pipeline(ASTNode):
    """Root node — an ordered list of statements."""
    commands: List["Statement"] = field(default_factory=list)


# Type alias: anything that can appear in a pipeline body
Statement = Union[Command, SetVar, IfBlock, RepeatBlock]


@dataclass
class IncludeStmt(ASTNode):
    """INCLUDE "other.pxl"  — inline another pipeline at parse time."""
    path: str = ""


@dataclass
class AssertStmt(ASTNode):
    """ASSERT <check> [message]  — validate pipeline state at runtime.

    check formats:
        width  == 640
        height >= 100
        channels == 3
        min >= 0
        max <= 255
        contour_count >= 1
    """
    subject:  str = ""    # "width" | "height" | "channels" | "min" | "max" | "contour_count"
    op:       str = "=="
    expected: Any = None
    message:  str = ""


@dataclass
class RoiBlock(ASTNode):
    """ROI x y w h ... ROI_RESET — execute body within a masked region."""
    x: Any = 0
    y: Any = 0
    w: Any = 0
    h: Any = 0
    body: List["Statement"] = field(default_factory=list)


# Update Statement union
Statement = Union[Command, SetVar, IfBlock, RepeatBlock, IncludeStmt, AssertStmt, RoiBlock]
