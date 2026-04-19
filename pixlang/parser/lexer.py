# pixlang/parser/lexer.py
"""
Lexer for PixLang v0.3.

New token types:
  VAR      - variable reference: $name
  IDENT    - lowercase/mixed identifier used in SET/IF/REPEAT: width, thresh
  KEYWORD  - control-flow reserved words: SET IF ENDIF REPEAT END
  OP       - comparison operators: == != < > <= >=
"""
import re
from dataclasses import dataclass
from typing import List

KEYWORDS = {"SET", "IF", "ENDIF", "REPEAT", "END", "INCLUDE", "ASSERT", "ROI", "ROI_RESET", "LOAD_GLOB", "SAVE_EACH"}

TOKEN_PATTERNS = [
    ("COMMENT",  r"#[^\n]*"),
    ("STRING",   r'"[^"]*"'),
    ("VAR",      r"\$[A-Za-z_][A-Za-z0-9_]*"),   # $name
    ("FLOAT",    r"\d+\.\d+"),
    ("INT",      r"-?\d+"),                        # allow negative ints
    ("OP",       r"==|!=|<=|>=|<|>"),
    ("WORD",     r"[A-Za-z_][A-Za-z0-9_]*"),      # commands, keywords, idents
    ("SKIP",     r"[ \t]+"),
    ("NEWLINE",  r"\n"),
    ("MISMATCH", r"."),
]

MASTER_PATTERN = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_PATTERNS)
)


@dataclass
class Token:
    type: str    # COMMAND | KEYWORD | IDENT | STRING | INT | FLOAT | VAR | OP
    value: str
    line: int


def tokenize(source: str) -> List[Token]:
    """
    Convert raw source text into a flat list of meaningful tokens.
    Comments, whitespace, and blank lines are discarded.
    WORD tokens are classified as KEYWORD, COMMAND, or IDENT by case.
    """
    tokens: List[Token] = []
    line_num = 1

    for match in MASTER_PATTERN.finditer(source):
        kind  = match.lastgroup
        value = match.group()

        if kind == "NEWLINE":
            line_num += 1
        elif kind in ("SKIP", "COMMENT"):
            pass
        elif kind == "MISMATCH":
            raise SyntaxError(
                f"[PixLang Lexer] Unexpected character {value!r} on line {line_num}"
            )
        elif kind == "WORD":
            # Classify: reserved keyword > all-caps command > lowercase ident
            if value in KEYWORDS:
                tokens.append(Token("KEYWORD", value, line_num))
            elif value.isupper() or (value[0].isupper() and "_" in value):
                tokens.append(Token("COMMAND", value, line_num))
            elif value[0].isupper() and value.replace("_", "").isalpha():
                # e.g. "GRAYSCALE" — all caps already caught; mixed case = IDENT
                tokens.append(Token("IDENT", value, line_num))
            else:
                tokens.append(Token("IDENT", value, line_num))
        else:
            tokens.append(Token(type=kind, value=value, line=line_num))

    return tokens
