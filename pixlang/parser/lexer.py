"""
Lexer for PixLang v0.5.

Token types:
  COMMAND   - ALL-CAPS word: GRAYSCALE, RESIZE, SHARPEN_TWICE, X
  KEYWORD   - reserved word (see KEYWORDS)
  IDENT     - any other word: width, Foo_bar, _ , u8
  VAR       - variable reference: $name
  INT       - 42, -7, 0xFF, 0b1010
  FLOAT     - 1.5, -1.5, .5, 1e3, 2.5e-3
  STRING    - "text" with escapes \\" \\\\ \\n \\t  (value keeps the quotes)
  OP        - operators and punctuation: == != <= >= < > + - * / // % & | ^ ~
              << >> ( ) [ ] , . : =
  NEWLINE   - end of a logical line (only with tokenize(..., newlines=True))

Notes:
  * CRLF / CR line endings are normalised to LF.
  * A backslash at the end of a line continues the logical line.
  * '-' directly followed by a digit is part of a negative number literal
    unless it directly follows (no whitespace) a value such as 5, $x, ) or ].
    So  ROTATE -90  -> INT(-90), but  5-3  -> INT(5) OP(-) INT(3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .errors import PixSyntaxError

# Every reserved word. The editor reads this set for highlighting/completion.
KEYWORDS = {
    # v0.3 / v0.4
    "SET", "IF", "ENDIF", "REPEAT", "END", "INCLUDE", "ASSERT",
    "ROI", "ROI_RESET", "LOAD_GLOB", "SAVE_EACH",
    # v0.5 control flow
    "ELIF", "ELSE", "WHILE", "FOR", "TO", "STEP", "AS", "BREAK", "CONTINUE",
    # v0.5 functions / modules
    "FUNC", "ENDFUNC", "RETURN", "GLOBAL", "EXPORT", "IMPORT", "FROM",
    # v0.5 buffers / memory
    "ALLOC", "BUF", "USE", "FREE", "WITH",
    # v0.5 expression words
    "and", "or", "not", "true", "false",
}

TOKEN_PATTERNS = [
    ("COMMENT",    r"#[^\n]*"),
    ("STRING",     r'"(?:[^"\\]|\\.)*"'),
    ("BADSTRING",  r'"'),                               # unterminated
    ("VAR",        r"\$[A-Za-z_][A-Za-z0-9_]*"),        # $name
    ("BADVAR",     r"\$"),
    ("HEX",        r"-?0[xX][0-9A-Fa-f_]+"),
    ("BIN",        r"-?0[bB][01_]+"),
    ("FLOAT",      r"-?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)"),
    ("INT",        r"-?\d+"),
    ("OP",         r"==|!=|<=|>=|<<|>>|//|[<>+\-*/%&|^~()\[\],.:=]"),
    ("WORD",       r"[A-Za-z_][A-Za-z0-9_]*"),
    ("SKIP",       r"[ \t]+"),
    ("LINE_CONT",  r"\\[ \t]*(?:#[^\n]*)?\n"),          # backslash continuation
    ("NEWLINE",    r"\n"),
    ("MISMATCH",   r"."),
]

MASTER_PATTERN = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_PATTERNS)
)

# Tokens after which a directly-attached '-' is binary minus, not a sign
_VALUE_END_TYPES = {"INT", "FLOAT", "VAR", "IDENT", "STRING", "COMMAND"}
_VALUE_END_OPS   = {")", "]"}


@dataclass
class Token:
    type: str     # COMMAND | KEYWORD | IDENT | STRING | INT | FLOAT | VAR | OP | NEWLINE
    value: str
    line: int
    col: int = 0
    file: Optional[str] = None
    space_before: bool = False   # whitespace (or line start) precedes this token


def normalize_newlines(source: str) -> str:
    """Convert CRLF / CR line endings to LF."""
    return source.replace("\r\n", "\n").replace("\r", "\n")


def decode_string(raw: str) -> str:
    """Strip quotes and process escapes from a STRING token value."""
    body = raw[1:-1]
    if "\\" not in body:
        return body
    out = []
    i = 0
    escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "0": "\0"}
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in escapes:
                out.append(escapes[nxt])
            else:
                out.append("\\" + nxt)   # unknown escape kept verbatim (Windows paths)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def classify_word(value: str) -> str:
    """KEYWORD for reserved words, COMMAND for ALL-CAPS words, else IDENT."""
    if value in KEYWORDS:
        return "KEYWORD"
    if value.isupper():
        return "COMMAND"
    return "IDENT"


def tokenize(source: str, newlines: bool = False, file: Optional[str] = None) -> List[Token]:
    """
    Convert raw source text into a list of tokens.

    Comments, whitespace and line continuations are discarded. With
    ``newlines=True`` a NEWLINE token terminates every non-empty logical line
    (the parser uses this; the default keeps the v0.4 token stream).
    """
    source = normalize_newlines(source)
    tokens: List[Token] = []
    line_num = 1
    line_start = 0
    prev: Optional[Token] = None
    space = True

    for match in MASTER_PATTERN.finditer(source):
        kind  = match.lastgroup
        value = match.group()
        col   = match.start() - line_start + 1

        if kind in ("NEWLINE", "LINE_CONT"):
            if kind == "NEWLINE" and newlines and tokens and tokens[-1].type != "NEWLINE":
                tokens.append(Token("NEWLINE", "\n", line_num, col, file))
            line_num += 1
            line_start = match.end()
            space = True
            if kind == "NEWLINE":
                prev = None
            continue
        if kind in ("SKIP", "COMMENT"):
            space = True
            continue
        if kind == "MISMATCH":
            raise PixSyntaxError(f"unexpected character {value!r}", line_num, col, file, source)
        if kind == "BADSTRING":
            raise PixSyntaxError("unterminated string literal", line_num, col, file, source)
        if kind == "BADVAR":
            raise PixSyntaxError("'$' must be followed by a variable name", line_num, col, file, source)

        if kind in ("HEX", "BIN", "FLOAT", "INT") and value.startswith("-"):
            # '-' glued to a preceding value is binary minus: split it off
            if prev is not None and not space and (
                prev.type in _VALUE_END_TYPES or
                (prev.type == "OP" and prev.value in _VALUE_END_OPS)
            ):
                tok = Token("OP", "-", line_num, col, file, space)
                tokens.append(tok)
                value = value[1:]
                col += 1
                space = False
                prev = tok
        if kind == "FLOAT" and value.lstrip("-").startswith(".") and prev is not None \
                and not space and (prev.type in _VALUE_END_TYPES or
                                   (prev.type == "OP" and prev.value in _VALUE_END_OPS)):
            # "x.5" — attribute-style access, not a float; lex '.' then the int
            tokens.append(Token("OP", ".", line_num, col, file, space))
            kind, value, col = "INT", value[1:], col + 1
            space = False

        if kind in ("HEX", "BIN"):
            kind = "INT"
        elif kind == "WORD":
            kind = classify_word(value)

        tok = Token(kind, value, line_num, col, file, space)
        tokens.append(tok)
        prev = tok
        space = False

        if kind == "STRING" and "\n" in value:
            # Multi-line string: keep line/column tracking accurate
            line_num  += value.count("\n")
            line_start = match.start() + value.rfind("\n") + 1

    if newlines and tokens and tokens[-1].type != "NEWLINE":
        tokens.append(Token("NEWLINE", "\n", line_num, len(source) - line_start + 1, file))
    return tokens
