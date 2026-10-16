"""
PixLang recursive-descent + Pratt parser  v0.5

Grammar (simplified EBNF; NL = end of line):

    pipeline    := (statement NL)*
    statement   := command | set | if | repeat | while | for | BREAK | CONTINUE
                 | roi | with | include | assert | func | return | global
                 | import | alloc | buf | use | free | '*' unary '=' expr

    command     := COMMAND ['.' COMMAND] arg*   |   IDENT '.' COMMAND arg*
    arg         := INT | FLOAT | STRING | VAR | word | true | false | '(' expr ')'
    set         := SET target ['='] expr
    target      := word ['[' index_list ']']  |  '*' unary
    if          := IF expr NL block (ELIF expr NL block)* [ELSE NL block] ENDIF
    repeat      := REPEAT expr [AS word] NL block END
    while       := WHILE expr NL block END
    for         := FOR word ['='] expr TO expr [STEP expr] NL block END
    roi         := ROI arg arg arg arg NL block ROI_RESET
    with        := WITH expr NL block END
    include     := INCLUDE STRING
    assert      := ASSERT expr [STRING]
    func        := [EXPORT] FUNC word ['(' [word (',' word)*] ')'] NL block ENDFUNC
    return      := RETURN [expr]
    global      := GLOBAL word (',' word)*
    import      := IMPORT STRING [AS word]  |  FROM STRING IMPORT word (',' word)*
    alloc       := ALLOC word arg arg [arg] [dtype] [arg]
    buf         := BUF word [expr]
    use         := USE expr
    free        := FREE word (',' word)*

Expression precedence (low → high):
    AS (reinterpret) < or < and < not < == != < > <= >= < | < ^ < &
    < << >> < + - < * / // % < unary - ~ & * < postfix [..] .attr (..)

Command arguments are atoms; wrap an expression in parentheses to use it as
an argument:  RESIZE ($w / 2) 480   — so  BLUR -1  and  RESIZE 640 480  keep
their v0.4 meaning.
"""
from __future__ import annotations

from typing import Any, List, Optional, Set

from .errors import PixSyntaxError
from .lexer import Token, decode_string, normalize_newlines, tokenize
from .ast_nodes import (
    AllocStmt, AssertStmt, AsType, Attr, BareIdent, Binary, BreakStmt, BufStmt,
    Call, Command, ContinueStmt, ElifClause, Expr, ForBlock, FreeStmt, FuncDef,
    GlobalStmt, IfBlock, ImportStmt, IncludeStmt, Index, Literal, Pipeline,
    RepeatBlock, ReturnStmt, RoiBlock, SetVar, Slice, Statement, Unary, UseStmt,
    VarRef, WhileBlock, WithBlock,
)

# Numeric dtype names accepted by ALLOC / AS / CAST / SIZEOF
DTYPE_NAMES = ("u8", "i8", "u16", "i16", "i32", "u32", "f32", "f64")

COMPARISON_OPS = ("==", "!=", "<", ">", "<=", ">=")

# Infix binding powers (higher binds tighter)
_INFIX_BP = {
    "or": 10, "and": 20,
    "==": 40, "!=": 40, "<": 40, ">": 40, "<=": 40, ">=": 40,
    "|": 50, "^": 60, "&": 70, "<<": 80, ">>": 80,
    "+": 90, "-": 90, "*": 100, "/": 100, "//": 100, "%": 100,
}
_AS_BP      = 5
_NOT_BP     = 30
_UNARY_BP   = 110

# Which opener each block terminator closes (for helpful errors)
_TERMINATOR_OWNER = {
    "END":       "REPEAT, WHILE, FOR or WITH",
    "ENDIF":     "IF",
    "ELIF":      "IF",
    "ELSE":      "IF",
    "ENDFUNC":   "FUNC",
    "ROI_RESET": "ROI",
}
_ALL_TERMINATORS = set(_TERMINATOR_OWNER)


# ── Public API ────────────────────────────────────────────────────────────────

def parse(source: str, filename: Optional[str] = None) -> Pipeline:
    """Parse PixLang source text into a Pipeline AST.

    Raises PixSyntaxError (a SyntaxError) with .lineno/.offset/.filename set.
    """
    source = normalize_newlines(source)
    tokens = tokenize(source, newlines=True, file=filename)
    return _Parser(tokens, source, filename).parse_pipeline()


def parse_expression(source: str, filename: Optional[str] = None) -> Any:
    """Parse a single expression (used by tools and tests)."""
    source = normalize_newlines(source)
    tokens = tokenize(source, newlines=True, file=filename)
    p = _Parser(tokens, source, filename)
    expr = p._expr()
    p._end_of_statement("expression")
    return expr


def literal_value(node: Any) -> Any:
    """Unwrap a Literal node into its Python value (legacy AST shape)."""
    return node.value if isinstance(node, Literal) else node


# ── Parser ────────────────────────────────────────────────────────────────────

class _Parser:
    def __init__(self, tokens: List[Token], source: str, filename: Optional[str]):
        self._tokens = tokens
        self._pos = 0
        self._source = source
        self._file = filename
        self._func_depth = 0
        self._block_depth = 0
        self._loop_depth = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> Optional[Token]:
        i = self._pos + offset
        return self._tokens[i] if i < len(self._tokens) else None

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _error(self, msg: str, tok: Optional[Token] = None) -> PixSyntaxError:
        if tok is None:
            tok = self._peek()
        if tok is None and self._tokens:
            tok = self._tokens[-1]
        line, col = (tok.line, tok.col) if tok else (1, 1)
        return PixSyntaxError(msg, line, col, self._file, self._source)

    def _loc(self, tok: Token) -> dict:
        return {"line": tok.line, "col": tok.col, "file": self._file}

    @staticmethod
    def _describe(tok: Optional[Token]) -> str:
        if tok is None:
            return "end of input"
        if tok.type == "NEWLINE":
            return "end of line"
        return f"{tok.type.lower()} {tok.value!r}"

    def _is(self, type_: str, value: Optional[str] = None, offset: int = 0) -> bool:
        tok = self._peek(offset)
        return tok is not None and tok.type == type_ and (value is None or tok.value == value)

    def _is_op(self, value: str) -> bool:
        return self._is("OP", value)

    def _is_kw(self, value: str) -> bool:
        return self._is("KEYWORD", value)

    def _expect(self, type_: str, value: Optional[str] = None, what: str = "") -> Token:
        tok = self._peek()
        if tok is None or tok.type != type_ or (value is not None and tok.value != value):
            want = what or (repr(value) if value else type_.lower())
            raise self._error(f"expected {want}, got {self._describe(tok)}", tok)
        return self._advance()

    def _expect_word(self, what: str = "a name") -> Token:
        """Name positions accept any word (IDENT or ALL-CAPS COMMAND)."""
        tok = self._peek()
        if tok is None or tok.type not in ("IDENT", "COMMAND"):
            raise self._error(f"expected {what}, got {self._describe(tok)}", tok)
        return self._advance()

    def _skip_newlines(self) -> None:
        while self._is("NEWLINE"):
            self._pos += 1

    def _end_of_statement(self, what: str) -> None:
        tok = self._peek()
        if tok is None:
            return
        if tok.type != "NEWLINE":
            raise self._error(
                f"unexpected {self._describe(tok)} after {what} "
                f"(each statement must end at the end of the line; use '\\' to continue)",
                tok,
            )
        self._advance()

    # ── Pipeline / blocks ─────────────────────────────────────────────────────

    def parse_pipeline(self) -> Pipeline:
        pipeline = Pipeline(line=1, col=1, file=self._file)
        self._skip_newlines()
        while self._peek() is not None:
            tok = self._peek()
            if tok.type == "KEYWORD" and tok.value in _ALL_TERMINATORS:
                raise self._error(
                    f"{tok.value} without {_TERMINATOR_OWNER[tok.value]}", tok)
            pipeline.commands.append(self._parse_statement())
            self._skip_newlines()
        return pipeline

    def _parse_block(self, terminators: Set[str], opener: Token, opener_name: str,
                     closer: str) -> List[Statement]:
        """Parse statements until one of *terminators* (not consumed)."""
        body: List[Statement] = []
        self._block_depth += 1
        try:
            self._skip_newlines()
            while True:
                tok = self._peek()
                if tok is None:
                    raise self._error(
                        f"unterminated {opener_name} block opened at line "
                        f"{opener.line} (expected {closer})", opener)
                if tok.type == "KEYWORD" and tok.value in _ALL_TERMINATORS:
                    if tok.value in terminators:
                        return body
                    raise self._error(
                        f"expected {closer} to close {opener_name} from line "
                        f"{opener.line}, found {tok.value}", tok)
                body.append(self._parse_statement())
                self._skip_newlines()
        finally:
            self._block_depth -= 1

    def _parse_statement(self) -> Statement:
        tok = self._peek()

        if tok.type == "KEYWORD":
            handler = {
                "SET": self._parse_set, "IF": self._parse_if,
                "REPEAT": self._parse_repeat, "WHILE": self._parse_while,
                "FOR": self._parse_for, "BREAK": self._parse_break,
                "CONTINUE": self._parse_continue, "INCLUDE": self._parse_include,
                "ASSERT": self._parse_assert, "ROI": self._parse_roi,
                "WITH": self._parse_with, "FUNC": self._parse_func,
                "EXPORT": self._parse_func, "RETURN": self._parse_return,
                "GLOBAL": self._parse_global, "IMPORT": self._parse_import,
                "FROM": self._parse_import, "ALLOC": self._parse_alloc,
                "BUF": self._parse_buf, "USE": self._parse_use,
                "FREE": self._parse_free,
            }.get(tok.value)
            if handler is not None:
                return handler()
            if tok.value in ("LOAD_GLOB", "SAVE_EACH"):
                # Treated as commands — handled by the batch runner
                return self._parse_command()
            raise self._error(f"unexpected keyword {tok.value!r}", tok)

        if tok.type == "COMMAND":
            return self._parse_command()

        if tok.type == "IDENT" and self._is("OP", ".", 1) and self._is("COMMAND", None, 2):
            return self._parse_command()          # alias.COMMAND args

        if tok.type == "OP" and tok.value == "*":
            return self._parse_deref_assign()

        if tok.type == "IDENT":
            hint = ""
            if self._is("OP", "=", 1):
                hint = f" (did you mean 'SET {tok.value} ...'?)"
            elif tok.value.upper() == tok.value.lower():
                hint = ""
            else:
                hint = f" (commands are ALL CAPS — did you mean {tok.value.upper()}?)"
            raise self._error(f"unexpected identifier {tok.value!r}{hint}", tok)

        raise self._error(f"unexpected {self._describe(tok)}", tok)

    # ── COMMAND ───────────────────────────────────────────────────────────────

    def _parse_command(self) -> Command:
        cmd_tok = self._advance()
        name = cmd_tok.value
        if self._is_op(".") and not self._peek().space_before:
            self._advance()
            member = self._expect("COMMAND", what="a command name after '.'")
            name = f"{name}.{member.value}"
        args: List[Any] = []
        while not self._is("NEWLINE") and self._peek() is not None:
            args.append(self._parse_arg())
        if self._peek() is not None:
            self._advance()                       # NEWLINE
        return Command(name=name, args=args, **self._loc(cmd_tok))

    def _parse_arg(self) -> Any:
        """One command argument: literal, $var, bare word or (expression)."""
        tok = self._peek()
        if tok.type == "OP" and tok.value == "(":
            return literal_value(self._parse_primary())
        if tok.type in ("INT", "FLOAT", "STRING", "VAR"):
            return literal_value(self._parse_primary())
        if tok.type in ("IDENT", "COMMAND"):
            self._advance()
            return BareIdent(name=tok.value, **self._loc(tok))
        if tok.type == "KEYWORD" and tok.value in ("true", "false"):
            self._advance()
            return tok.value == "true"
        if tok.type == "KEYWORD":
            raise self._error(
                f"unexpected keyword {tok.value!r} in command arguments "
                f"(statements must be on their own line)", tok)
        raise self._error(
            f"unexpected {self._describe(tok)} in command arguments "
            f"(wrap expressions in parentheses, e.g. RESIZE ($w / 2) 480)", tok)

    # ── SET / assignment ──────────────────────────────────────────────────────

    def _parse_set(self) -> SetVar:
        kw = self._advance()
        target = None
        if self._is_op("*"):
            target = self._parse_unary_prefix()
            name = ""
        else:
            name_tok = self._expect_word("a variable name after SET")
            name = name_tok.value
            if self._is_op("[") and not self._peek().space_before:
                target = self._parse_postfix(BareIdent(name=name, **self._loc(name_tok)), 0)
                if not isinstance(target, Index):
                    raise self._error("invalid assignment target", name_tok)
        if self._is_op("="):
            self._advance()
        if self._is("NEWLINE") or self._peek() is None:
            raise self._error(f"SET {name or '*...'} needs a value")
        value = literal_value(self._expr())
        self._end_of_statement("SET value")
        return SetVar(var_name=name, value=value, target=target, **self._loc(kw))

    def _parse_deref_assign(self) -> SetVar:
        star = self._peek()
        target = self._parse_unary_prefix()
        self._expect("OP", "=", what="'=' after pointer target")
        value = literal_value(self._expr())
        self._end_of_statement("assignment")
        return SetVar(var_name="", value=value, target=target, **self._loc(star))

    def _parse_unary_prefix(self) -> Unary:
        star = self._advance()
        operand = self._expr(_UNARY_BP)
        return Unary(op="*", operand=operand, **self._loc(star))

    # ── IF / ELIF / ELSE ──────────────────────────────────────────────────────

    def _parse_if(self) -> IfBlock:
        kw = self._advance()
        cond = self._condition("if")
        self._end_of_statement("IF condition")
        node = IfBlock(cond=cond, cond_text=_text(cond), **self._loc(kw))
        _fill_legacy_compare(node, cond)
        node.body = self._parse_block({"ELIF", "ELSE", "ENDIF"}, kw, "IF", "ENDIF")
        while self._is_kw("ELIF"):
            etok = self._advance()
            econd = self._condition("if")
            self._end_of_statement("ELIF condition")
            body = self._parse_block({"ELIF", "ELSE", "ENDIF"}, kw, "IF", "ENDIF")
            node.elifs.append(ElifClause(cond=econd, body=body, **self._loc(etok)))
        if self._is_kw("ELSE"):
            self._advance()
            self._end_of_statement("ELSE")
            node.else_body = self._parse_block({"ENDIF"}, kw, "IF", "ENDIF")
            if not node.else_body:
                node.else_body = []
        self._expect("KEYWORD", "ENDIF")
        self._end_of_statement("ENDIF")
        return node

    def _condition(self, strict: str) -> Any:
        if self._is("NEWLINE") or self._peek() is None:
            raise self._error("missing condition")
        cond = self._expr()
        if isinstance(cond, Binary) and cond.op in COMPARISON_OPS \
                and isinstance(cond.left, BareIdent):
            cond.left.strict = strict
        elif isinstance(cond, BareIdent):
            cond.strict = strict
        return cond

    # ── Loops ─────────────────────────────────────────────────────────────────

    def _loop_body(self, kw: Token, name: str) -> List[Statement]:
        self._loop_depth += 1
        try:
            body = self._parse_block({"END"}, kw, name, "END")
        finally:
            self._loop_depth -= 1
        self._expect("KEYWORD", "END")
        self._end_of_statement("END")
        return body

    def _parse_repeat(self) -> RepeatBlock:
        kw = self._advance()
        if self._is("NEWLINE") or self._peek() is None:
            raise self._error("REPEAT needs a count", kw)
        count = literal_value(self._expr(allow_as=False))
        var_name = None
        if self._is_kw("AS"):
            self._advance()
            var_name = self._expect_word("a loop variable name after AS").value
        self._end_of_statement("REPEAT count")
        body = self._loop_body(kw, "REPEAT")
        return RepeatBlock(count=count, body=body, var_name=var_name, **self._loc(kw))

    def _parse_while(self) -> WhileBlock:
        kw = self._advance()
        cond = self._condition("")
        self._end_of_statement("WHILE condition")
        body = self._loop_body(kw, "WHILE")
        return WhileBlock(cond=cond, body=body, cond_text=_text(cond), **self._loc(kw))

    def _parse_for(self) -> ForBlock:
        kw = self._advance()
        var = self._expect_word("a loop variable after FOR")
        if self._is_op("="):
            self._advance()
        start = literal_value(self._expr())
        self._expect("KEYWORD", "TO", what="TO")
        stop = literal_value(self._expr())
        step = None
        if self._is_kw("STEP"):
            self._advance()
            step = literal_value(self._expr())
        self._end_of_statement("FOR header")
        body = self._loop_body(kw, "FOR")
        return ForBlock(var_name=var.value, start=start, stop=stop, step=step,
                        body=body, **self._loc(kw))

    def _parse_break(self) -> BreakStmt:
        kw = self._advance()
        if not self._loop_depth:
            raise self._error("BREAK outside of a loop", kw)
        self._end_of_statement("BREAK")
        return BreakStmt(**self._loc(kw))

    def _parse_continue(self) -> ContinueStmt:
        kw = self._advance()
        if not self._loop_depth:
            raise self._error("CONTINUE outside of a loop", kw)
        self._end_of_statement("CONTINUE")
        return ContinueStmt(**self._loc(kw))

    # ── INCLUDE / ASSERT / ROI / WITH ─────────────────────────────────────────

    def _parse_include(self) -> IncludeStmt:
        kw   = self._advance()
        path = self._expect("STRING", what="a file path string after INCLUDE")
        self._end_of_statement("INCLUDE path")
        return IncludeStmt(path=decode_string(path.value), **self._loc(kw))

    def _parse_assert(self) -> AssertStmt:
        kw = self._advance()
        cond = self._condition("assert")
        msg = ""
        if self._is("STRING"):
            msg = decode_string(self._advance().value)
        self._end_of_statement("ASSERT")
        node = AssertStmt(message=msg, cond=cond, cond_text=_text(cond), op="",
                          **self._loc(kw))
        if isinstance(cond, Binary) and cond.op in COMPARISON_OPS \
                and isinstance(cond.left, (BareIdent, VarRef)):
            node.subject = getattr(cond.left, "name", None) or cond.left.var_name
            node.op = cond.op
            node.expected = literal_value(cond.right)
        return node

    def _parse_roi(self) -> RoiBlock:
        kw = self._advance()
        coords = []
        for _ in range(4):
            if self._is("NEWLINE") or self._peek() is None:
                raise self._error("ROI needs 4 arguments: x y w h", kw)
            coords.append(self._parse_arg())
        self._end_of_statement("ROI x y w h")
        body = self._parse_block({"ROI_RESET"}, kw, "ROI", "ROI_RESET")
        self._expect("KEYWORD", "ROI_RESET")
        self._end_of_statement("ROI_RESET")
        x, y, w, h = coords
        return RoiBlock(x=x, y=y, w=w, h=h, body=body, **self._loc(kw))

    def _parse_with(self) -> WithBlock:
        kw = self._advance()
        target = self._expr()
        self._end_of_statement("WITH target")
        body = self._parse_block({"END"}, kw, "WITH", "END")
        self._expect("KEYWORD", "END")
        self._end_of_statement("END")
        return WithBlock(target=target, body=body, **self._loc(kw))

    # ── Functions & modules ───────────────────────────────────────────────────

    def _parse_func(self) -> FuncDef:
        first = self._advance()
        exported = first.value == "EXPORT"
        if exported:
            self._expect("KEYWORD", "FUNC", what="FUNC after EXPORT")
        if self._block_depth or self._func_depth:
            raise self._error("FUNC definitions are only allowed at the top level", first)
        name = self._expect_word("a function name after FUNC")
        params: List[str] = []
        if self._is_op("("):
            self._advance()
            while not self._is_op(")"):
                p = self._expect_word("a parameter name")
                if p.value in params:
                    raise self._error(f"duplicate parameter {p.value!r}", p)
                params.append(p.value)
                if self._is_op(","):
                    self._advance()
                elif not self._is_op(")"):
                    raise self._error(f"expected ',' or ')', got {self._describe(self._peek())}")
            self._advance()
        self._end_of_statement("FUNC header")
        self._func_depth += 1
        saved_loop, self._loop_depth = self._loop_depth, 0
        try:
            body = self._parse_block({"ENDFUNC"}, first, "FUNC", "ENDFUNC")
        finally:
            self._func_depth -= 1
            self._loop_depth = saved_loop
        self._expect("KEYWORD", "ENDFUNC")
        self._end_of_statement("ENDFUNC")
        return FuncDef(name=name.value, params=params, body=body, exported=exported,
                       **self._loc(first))

    def _parse_return(self) -> ReturnStmt:
        kw = self._advance()
        if not self._func_depth:
            raise self._error("RETURN outside of a FUNC", kw)
        value = None
        if not self._is("NEWLINE") and self._peek() is not None:
            value = literal_value(self._expr())
        self._end_of_statement("RETURN")
        return ReturnStmt(value=value, **self._loc(kw))

    def _name_list(self, what: str) -> List[Token]:
        names = [self._expect_word(what)]
        while self._is_op(","):
            self._advance()
            names.append(self._expect_word(what))
        return names

    def _parse_global(self) -> GlobalStmt:
        kw = self._advance()
        names = self._name_list("a variable name after GLOBAL")
        self._end_of_statement("GLOBAL")
        return GlobalStmt(names=[t.value for t in names], **self._loc(kw))

    def _parse_import(self) -> ImportStmt:
        kw = self._advance()
        if self._block_depth or self._func_depth:
            raise self._error(f"{kw.value} is only allowed at the top level", kw)
        path = decode_string(self._expect("STRING", what="a module path string").value)
        node = ImportStmt(path=path, **self._loc(kw))
        if kw.value == "IMPORT":
            if self._is_kw("AS"):
                self._advance()
                node.alias = self._expect_word("a module alias after AS").value
        else:
            self._expect("KEYWORD", "IMPORT", what="IMPORT after FROM \"path\"")
            toks = self._name_list("a name to import")
            node.names = [t.value for t in toks]
            node.name_locs = [(t.line, t.col) for t in toks]
        self._end_of_statement(kw.value)
        return node

    # ── Buffers ───────────────────────────────────────────────────────────────

    def _parse_alloc(self) -> AllocStmt:
        kw = self._advance()
        name = self._expect_word("a buffer name after ALLOC").value
        atoms = []
        while not self._is("NEWLINE") and self._peek() is not None:
            atoms.append((self._peek(), self._parse_arg()))
        if len(atoms) < 2:
            raise self._error("ALLOC needs at least width and height: ALLOC name w h [c] [dtype] [fill]", kw)
        node = AllocStmt(name=name, width=atoms[0][1], height=atoms[1][1], **self._loc(kw))
        rest = atoms[2:]
        if rest and not _is_dtype_atom(rest[0][1]):
            node.channels = rest.pop(0)[1]
        if rest and _is_dtype_atom(rest[0][1]):
            node.dtype = rest.pop(0)[1].name
        if rest:
            node.fill = rest.pop(0)[1]
        if rest:
            raise self._error("too many arguments for ALLOC", rest[0][0])
        self._end_of_statement("ALLOC")
        return node

    def _parse_buf(self) -> BufStmt:
        kw = self._advance()
        name = self._expect_word("a buffer name after BUF").value
        value = None
        if not self._is("NEWLINE") and self._peek() is not None:
            value = literal_value(self._expr())
        self._end_of_statement("BUF")
        return BufStmt(name=name, value=value, **self._loc(kw))

    def _parse_use(self) -> UseStmt:
        kw = self._advance()
        if self._is("NEWLINE") or self._peek() is None:
            raise self._error("USE needs a buffer or expression", kw)
        value = literal_value(self._expr())
        self._end_of_statement("USE")
        return UseStmt(value=value, **self._loc(kw))

    def _parse_free(self) -> FreeStmt:
        kw = self._advance()
        names = self._name_list("a buffer name after FREE")
        self._end_of_statement("FREE")
        return FreeStmt(names=[t.value for t in names], **self._loc(kw))

    # ── Expressions (Pratt) ───────────────────────────────────────────────────

    def _expr(self, min_bp: int = 0, allow_as: bool = True) -> Any:
        left = self._parse_unary()
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok.type == "KEYWORD" and tok.value == "AS":
                if not allow_as or _AS_BP < min_bp:
                    break
                self._advance()
                dtype = self._expect_word("a dtype after AS (u8 i8 u16 i16 i32 u32 f32 f64)")
                left = AsType(value=left, dtype=dtype.value, **self._loc(tok))
                continue
            if tok.type in ("INT", "FLOAT") and tok.value.startswith("-"):
                # "$x -3" inside an expression: binary minus
                bp = _INFIX_BP["-"]
                if bp <= min_bp:
                    break
                self._advance()
                stripped = Token(tok.type, tok.value[1:], tok.line, tok.col + 1, tok.file)
                right = self._number(stripped)
                left = Binary(op="-", left=left, right=right, **self._loc(tok))
                continue
            if tok.type in ("OP", "KEYWORD"):
                op = tok.value
                bp = _INFIX_BP.get(op)
                if bp is None or bp <= min_bp:
                    break
                self._advance()
                right = self._expr(bp, allow_as=False)
                left = Binary(op=op, left=left, right=right, **self._loc(tok))
                continue
            break
        return left

    def _parse_unary(self) -> Any:
        tok = self._peek()
        if tok is None or tok.type == "NEWLINE":
            raise self._error("expected an expression, got " + self._describe(tok), tok)
        if tok.type == "OP" and tok.value in ("-", "~", "&", "*", "+"):
            self._advance()
            operand = self._expr(_UNARY_BP, allow_as=False)
            if tok.value == "+":
                return operand
            if tok.value == "-" and isinstance(operand, Literal) \
                    and isinstance(operand.value, (int, float)) \
                    and not isinstance(operand.value, bool):
                return Literal(value=-operand.value, **self._loc(tok))
            return Unary(op=tok.value, operand=operand, **self._loc(tok))
        if tok.type == "KEYWORD" and tok.value == "not":
            self._advance()
            operand = self._expr(_NOT_BP, allow_as=False)
            return Unary(op="not", operand=operand, **self._loc(tok))
        return self._parse_postfix(self._parse_primary(), 0)

    def _parse_postfix(self, node: Any, _bp: int) -> Any:
        while True:
            tok = self._peek()
            if tok is None or tok.type != "OP":
                return node
            if tok.value == "[":
                self._advance()
                indices = [self._index_item()]
                while self._is_op(","):
                    self._advance()
                    indices.append(self._index_item())
                self._expect("OP", "]", what="']'")
                node = Index(target=node, indices=indices, **self._loc(tok))
            elif tok.value == "." and not tok.space_before:
                self._advance()
                name = self._expect_word("an attribute name after '.'")
                node = Attr(target=node, name=name.value, **self._loc(tok))
            elif tok.value == "(" and not tok.space_before and _callable_name(node):
                self._advance()
                args = []
                if not self._is_op(")"):
                    args.append(self._expr())
                    while self._is_op(","):
                        self._advance()
                        args.append(self._expr())
                self._expect("OP", ")", what="')' to close the call")
                node = Call(func=_callable_name(node), args=args,
                            line=node.line, col=node.col, file=self._file)
            else:
                return node

    def _index_item(self) -> Any:
        start = stop = step = None
        if not self._is_op(":"):
            start = self._expr()
            if not self._is_op(":"):
                return start
        tok = self._advance()                     # ':'
        if not (self._is_op(":") or self._is_op(",") or self._is_op("]")):
            stop = self._expr()
        if self._is_op(":"):
            self._advance()
            if not (self._is_op(",") or self._is_op("]")):
                step = self._expr()
        return Slice(start=start, stop=stop, step=step, **self._loc(tok))

    def _number(self, tok: Token) -> Literal:
        v = tok.value
        if tok.type == "FLOAT":
            value: Any = float(v)
        elif v.lstrip("-")[:2].lower() in ("0x", "0b"):
            value = int(v, 0)
        else:
            value = int(v.replace("_", ""), 10)
        return Literal(value=value, **self._loc(tok))

    def _parse_primary(self) -> Any:
        tok = self._peek()
        if tok is None or tok.type == "NEWLINE":
            raise self._error("expected a value, got " + self._describe(tok), tok)
        if tok.type in ("INT", "FLOAT"):
            self._advance()
            return self._number(tok)
        if tok.type == "STRING":
            self._advance()
            return Literal(value=decode_string(tok.value), **self._loc(tok))
        if tok.type == "VAR":
            self._advance()
            return VarRef(var_name=tok.value[1:], **self._loc(tok))
        if tok.type in ("IDENT", "COMMAND"):
            self._advance()
            return BareIdent(name=tok.value, **self._loc(tok))
        if tok.type == "KEYWORD" and tok.value in ("true", "false"):
            self._advance()
            return Literal(value=(tok.value == "true"), **self._loc(tok))
        if tok.type == "OP" and tok.value == "(":
            self._advance()
            if self._is_op(")"):
                raise self._error("empty parentheses", tok)
            inner = self._expr()
            self._expect("OP", ")", what="')'")
            return inner
        raise self._error(f"expected a value, got {self._describe(tok)}", tok)


# ── Module helpers ────────────────────────────────────────────────────────────

def _text(expr: Any) -> str:
    from .ast_nodes import expr_source
    text = expr_source(expr)
    if isinstance(expr, Binary) and text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return text


def _callable_name(node: Any) -> Optional[str]:
    """'name' or 'alias.name' if *node* can be called, else None."""
    if isinstance(node, BareIdent):
        return node.name
    if isinstance(node, Attr) and isinstance(node.target, BareIdent):
        return f"{node.target.name}.{node.name}"
    return None


def _is_dtype_atom(arg: Any) -> bool:
    return isinstance(arg, BareIdent) and arg.name in DTYPE_NAMES


def _fill_legacy_compare(node: IfBlock, cond: Any) -> None:
    """Populate IfBlock.var_name/op/cmp_value for the simple v0.3 form."""
    if isinstance(cond, Binary) and cond.op in COMPARISON_OPS:
        left = cond.left
        if isinstance(left, BareIdent):
            node.var_name = left.name
        elif isinstance(left, VarRef):
            node.var_name = left.var_name
        else:
            node.var_name = ""
        node.op = cond.op
        node.cmp_value = literal_value(cond.right)
    else:
        node.var_name = ""
        node.op = ""
        node.cmp_value = None
