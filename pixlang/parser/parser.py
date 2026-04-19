# pixlang/parser/parser.py
"""
PixLang recursive-descent parser  v0.3

Grammar (simplified BNF):
    pipeline    := statement*
    statement   := set_stmt
                 | if_stmt
                 | repeat_stmt
                 | command_stmt

    set_stmt    := KEYWORD("SET") IDENT value
    if_stmt     := KEYWORD("IF") IDENT OP value
                   statement*
                   KEYWORD("ENDIF")
    repeat_stmt := KEYWORD("REPEAT") (INT | VAR)
                   statement*
                   KEYWORD("END")
    command_stmt:= COMMAND arg*
    arg         := STRING | INT | FLOAT | VAR
    value       := STRING | INT | FLOAT
"""
from __future__ import annotations
from typing import List, Any
from .lexer import Token, tokenize
from .ast_nodes import (
    Command, SetVar, IfBlock, RepeatBlock, Pipeline, VarRef, Statement,
    IncludeStmt, AssertStmt, RoiBlock,
)

# ── Public API ────────────────────────────────────────────────────────────────

def parse(source: str) -> Pipeline:
    tokens = tokenize(source)
    p = _Parser(tokens)
    return p.parse_pipeline()


# ── Parser ────────────────────────────────────────────────────────────────────

class _Parser:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._pos = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _peek(self) -> Token | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, type_: str, value: str | None = None) -> Token:
        tok = self._peek()
        if tok is None:
            raise SyntaxError(
                f"[PixLang Parser] Unexpected end of input; "
                f"expected {type_!r}{(' ' + repr(value)) if value else ''}"
            )
        if tok.type != type_ or (value is not None and tok.value != value):
            raise SyntaxError(
                f"[PixLang Parser] Line {tok.line}: "
                f"expected {type_!r}{(' ' + repr(value)) if value else ''}, "
                f"got {tok.type!r} {tok.value!r}"
            )
        return self._advance()

    def _at_block_end(self) -> bool:
        """True if next token is a block-terminator (ENDIF, END) or EOF."""
        tok = self._peek()
        if tok is None:
            return True
        return tok.type == "KEYWORD" and tok.value in ("ENDIF", "END", "ROI_RESET")

    # ── Grammar productions ───────────────────────────────────────────────────

    def parse_pipeline(self) -> Pipeline:
        pipeline = Pipeline(line=1)
        while self._peek() is not None:
            stmt = self._parse_statement()
            if stmt is not None:
                pipeline.commands.append(stmt)
        return pipeline

    def _parse_body(self) -> List[Statement]:
        """Parse statements until a block-end keyword."""
        body: List[Statement] = []
        while not self._at_block_end():
            if self._peek() is None:
                break
            stmt = self._parse_statement()
            if stmt is not None:
                body.append(stmt)
        return body

    def _parse_statement(self) -> Statement | None:
        tok = self._peek()
        if tok is None:
            return None

        if tok.type == "KEYWORD":
            kw = tok.value
            if kw == "SET":
                return self._parse_set()
            elif kw == "IF":
                return self._parse_if()
            elif kw == "REPEAT":
                return self._parse_repeat()
            elif kw == "INCLUDE":
                return self._parse_include()
            elif kw == "ASSERT":
                return self._parse_assert()
            elif kw == "ROI":
                return self._parse_roi()
            elif kw in ("LOAD_GLOB", "SAVE_EACH"):
                # Treated as commands — handled by executor as batch ops
                return self._parse_command()
            else:
                # ENDIF / END / ROI_RESET — let the caller handle termination
                return None

        if tok.type == "COMMAND":
            return self._parse_command()

        # Unexpected token
        self._advance()
        raise SyntaxError(
            f"[PixLang Parser] Line {tok.line}: unexpected token "
            f"{tok.type!r} {tok.value!r}"
        )

    # ── SET ───────────────────────────────────────────────────────────────────

    def _parse_set(self) -> SetVar:
        kw = self._advance()                       # consume SET
        name_tok = self._expect("IDENT")
        value = self._parse_scalar()
        return SetVar(var_name=name_tok.value, value=value, line=kw.line)

    # ── IF ────────────────────────────────────────────────────────────────────

    def _parse_if(self) -> IfBlock:
        kw    = self._advance()                    # consume IF
        var   = self._expect("IDENT")
        op    = self._expect("OP")
        cmp   = self._parse_scalar()
        body  = self._parse_body()
        self._expect("KEYWORD", "ENDIF")
        return IfBlock(
            var_name=var.value, op=op.value,
            cmp_value=cmp, body=body, line=kw.line
        )

    # ── REPEAT ────────────────────────────────────────────────────────────────

    def _parse_repeat(self) -> RepeatBlock:
        kw    = self._advance()                    # consume REPEAT
        count = self._parse_arg()                  # int or $var
        body  = self._parse_body()
        self._expect("KEYWORD", "END")
        return RepeatBlock(count=count, body=body, line=kw.line)

    # ── COMMAND ───────────────────────────────────────────────────────────────

    def _parse_command(self) -> Command:
        cmd_tok = self._advance()
        args: List[Any] = []

        # Consume args until next COMMAND/KEYWORD/EOF
        while self._peek() is not None:
            nxt = self._peek()
            if nxt.type in ("COMMAND", "KEYWORD"):
                break
            args.append(self._parse_arg())

        return Command(name=cmd_tok.value, args=args, line=cmd_tok.line)

    # ── Argument parsers ──────────────────────────────────────────────────────

    def _parse_arg(self) -> Any:
        """Parse one argument token: STRING | INT | FLOAT | VAR | IDENT-as-string."""
        tok = self._peek()
        if tok is None:
            raise SyntaxError("[PixLang Parser] Expected argument, got end of input")

        if tok.type == "VAR":
            self._advance()
            return VarRef(var_name=tok.value[1:], line=tok.line)  # strip $
        return self._parse_scalar()

    def _parse_scalar(self) -> Any:
        """Parse a literal value (no VarRef)."""
        tok = self._peek()
        if tok is None:
            raise SyntaxError("[PixLang Parser] Expected literal value")

        if tok.type == "FLOAT":
            self._advance(); return float(tok.value)
        if tok.type == "INT":
            self._advance(); return int(tok.value)
        if tok.type == "STRING":
            self._advance(); return tok.value.strip('"')
        if tok.type == "IDENT":
            self._advance(); return tok.value          # bare word (colormap etc)
        if tok.type == "OP":
            self._advance(); return tok.value

        raise SyntaxError(
            f"[PixLang Parser] Line {tok.line}: expected literal, "
            f"got {tok.type!r} {tok.value!r}"
        )

    # ── v0.4 productions ──────────────────────────────────────────────────────

    def _parse_include(self) -> IncludeStmt:
        """INCLUDE "other.pxl" """
        kw   = self._advance()           # consume INCLUDE keyword
        path = self._expect("STRING")
        return IncludeStmt(path=path.value.strip('"'), line=kw.line)

    def _parse_assert(self) -> AssertStmt:
        """ASSERT subject op value ["message"] """
        kw      = self._advance()        # consume ASSERT
        subject = self._expect("IDENT")
        op      = self._expect("OP")
        value   = self._parse_scalar()
        # Optional message string
        msg = ""
        if self._peek() and self._peek().type == "STRING":
            msg = self._advance().value.strip('"')
        return AssertStmt(
            subject=subject.value, op=op.value,
            expected=value, message=msg, line=kw.line
        )

    def _parse_roi(self) -> RoiBlock:
        """ROI x y w h ... ROI_RESET"""
        kw = self._advance()             # consume ROI
        x  = self._parse_arg()
        y  = self._parse_arg()
        w  = self._parse_arg()
        h  = self._parse_arg()
        body = self._parse_body()
        self._expect("KEYWORD", "ROI_RESET")
        return RoiBlock(x=x, y=y, w=w, h=h, body=body, line=kw.line)
