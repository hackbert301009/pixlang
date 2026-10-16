# pixlang/parser/errors.py
"""
PixLang error types and diagnostic rendering  v0.5

Every error raised by the language core carries a source location so tools
(CLI, editor, linter) can point at the offending code:

    PixSyntaxError   (SyntaxError)    — lexer / parser errors
                                        .lineno .offset (1-based col) .filename .msg
                                        plus .line .col .file aliases
    PixRuntimeError  (RuntimeError)   — execution errors
                                        .line .col .file .msg .trace
    PixAssertionError(AssertionError) — failed ASSERT statements
    PixLinkError     (PixRuntimeError)— linker errors (.diagnostics)

Diagnostics render compiler-style:

    pipeline.pxl:3:7: error: unexpected token ')'
        BLUR (3 + )
              ^
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


# ── Source locations ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SourceLoc:
    """A position in a source file (1-based line and column)."""
    file: Optional[str] = None
    line: int = 0
    col:  int = 0

    def __str__(self) -> str:
        return format_loc(self.file, self.line, self.col)


def format_loc(file: Optional[str], line: int, col: int = 0) -> str:
    """Return ``file:line:col`` (``<input>`` when no file is known)."""
    out = file or "<input>"
    if line:
        out += f":{line}"
        if col:
            out += f":{col}"
    return out


def source_line(file: Optional[str], line: int, source: Optional[str] = None) -> Optional[str]:
    """Fetch line *line* from *source* text, or read it from *file*."""
    if not line:
        return None
    text = source
    if text is None and file:
        try:
            text = Path(file).read_text(encoding="utf-8")
        except OSError:
            return None
    if text is None:
        return None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return None


def format_diagnostic(
    severity: str,
    message: str,
    file: Optional[str] = None,
    line: int = 0,
    col: int = 0,
    source: Optional[str] = None,
    notes: Sequence[str] = (),
) -> str:
    """Render ``file:line:col: severity: message`` + source line + caret."""
    out = [f"{format_loc(file, line, col)}: {severity}: {message}"]
    text = source_line(file, line, source)
    if text is not None:
        out.append(f"    {text}")
        if col:
            # Keep tabs so the caret lines up with the echoed source line
            pad = "".join("\t" if ch == "\t" else " " for ch in text[: col - 1])
            out.append(f"    {pad}^")
    for note in notes:
        out.append(f"  note: {note}")
    return "\n".join(out)


# ── Error classes ─────────────────────────────────────────────────────────────

class PixSyntaxError(SyntaxError):
    """A lexer or parser error with standard SyntaxError location attributes."""

    def __init__(self, msg: str, line: int = 0, col: int = 0,
                 file: Optional[str] = None, source: Optional[str] = None):
        text = source_line(file, line, source)
        super().__init__(msg, (file, line, col, text))
        # SyntaxError.__init__ fills msg/filename/lineno/offset/text
        self.source = source

    # Uniform accessors shared with PixRuntimeError
    @property
    def line(self) -> int:
        return self.lineno or 0

    @property
    def col(self) -> int:
        return self.offset or 0

    @property
    def file(self) -> Optional[str]:
        return self.filename

    def pretty(self) -> str:
        return format_diagnostic("error", self.msg, self.filename,
                                 self.line, self.col, self.source)

    def __str__(self) -> str:
        return f"{format_loc(self.filename, self.line, self.col)}: error: {self.msg}"


class _RuntimeLocMixin:
    """Shared formatting for runtime-ish errors."""

    def _setup(self, msg: str, line: int, col: int, file: Optional[str],
               trace: Optional[List[str]], raw: bool) -> str:
        self.msg   = msg
        self.line  = line
        self.col   = col
        self.file  = file
        self.trace = list(trace or [])
        if raw:
            text = msg
        else:
            where = f"line {line}" if line else "unknown line"
            if col:
                where += f", col {col}"
            if file:
                where += f" in {file}"
            text = f"[PixLang] Error at {where}: {msg}"
        if self.trace:
            text += "\n  call stack (innermost last):\n" + "\n".join(
                f"    {t}" for t in self.trace)
        return text

    def pretty(self) -> str:
        return format_diagnostic("error", str(self), self.file, self.line, self.col)


class PixRuntimeError(_RuntimeLocMixin, RuntimeError):
    """A runtime error carrying .line .col .file (and a PixLang call trace)."""

    def __init__(self, msg: str, line: int = 0, col: int = 0,
                 file: Optional[str] = None, trace: Optional[List[str]] = None,
                 raw: bool = False):
        super().__init__(self._setup(msg, line, col, file, trace, raw))


class PixAssertionError(_RuntimeLocMixin, AssertionError):
    """A failed ASSERT statement."""

    def __init__(self, msg: str, line: int = 0, col: int = 0,
                 file: Optional[str] = None, trace: Optional[List[str]] = None):
        super().__init__(self._setup(msg, line, col, file, trace, raw=True))


class PixLinkError(PixRuntimeError):
    """Linking failed; .diagnostics holds every error found."""

    def __init__(self, msg: str, diagnostics=None, line: int = 0, col: int = 0,
                 file: Optional[str] = None):
        self.diagnostics = list(diagnostics or [])
        super().__init__(msg, line=line, col=col, file=file, raw=True)
