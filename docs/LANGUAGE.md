# PixLang v0.5 Language Reference

PixLang is a small DSL for OpenCV image pipelines. A program is a list of
statements, one per line. Most statements are *commands* that transform the
current image; v0.5 adds expressions, functions, modules, typed buffers and
pointers.

```
LOAD "examples/sample.jpg"
SET w 640
RESIZE $w ($w * 3 // 4)
GRAYSCALE
SAVE "out.png"
```

Contents: [Lexical structure](#1-lexical-structure) ·
[Values and types](#2-values-and-types) · [Expressions](#3-expressions) ·
[Statements](#4-statements) · [Control flow](#5-control-flow) ·
[Functions](#6-functions) · [Modules and the linker](#7-modules-and-the-linker) ·
[Buffers, pointers and memory](#8-buffers-pointers-and-memory) ·
[Builtin functions](#9-builtin-functions) · [Errors](#10-errors-and-diagnostics) ·
[Linter](#11-linter) · [Configuration](#12-configuration-pixlangtoml) ·
[Grammar](#13-grammar) · [Compatibility](#14-compatibility-with-v04)

---

## 1. Lexical structure

* **Statements end at the end of the line.** A trailing `\` continues the
  statement on the next line. Blank lines and `#` comments are ignored.
  CRLF / CR line endings are accepted.
* **Words** are `[A-Za-z_][A-Za-z0-9_]*`. An ALL-CAPS word is a *command
  name* (`BLUR`, `SHARPEN_TWICE`, `X`); anything else is an identifier
  (`width`, `Foo_bar`). Reserved words (`KEYWORDS` in `pixlang.parser.lexer`)
  cannot be used as names. **Name positions** (after `SET`, `FOR`, `FUNC`,
  parameters, buffer names...) accept any non-reserved word.
* **Numbers**: `42`, `-7`, `1.5`, `-1.5`, `.5`, `1e3`, `2.5e-3`, `0xFF`,
  `0b1010`, underscores in hex/binary (`0x1_0`). A `-` glued to the previous
  value is binary minus: `5-3` is `2`, while `ROTATE -90` passes `-90`.
* **Strings**: `"text"` with escapes `\"` `\\` `\n` `\t` `\r` `\0`. Unknown
  escapes are kept verbatim (so `"C:\data"` works). Strings may span lines.
* **Variables**: `$name` reads a variable.
* **Keywords**:
  `SET IF ELIF ELSE ENDIF REPEAT WHILE FOR TO STEP AS END BREAK CONTINUE
  INCLUDE ASSERT ROI ROI_RESET WITH FUNC ENDFUNC RETURN GLOBAL EXPORT IMPORT
  FROM ALLOC BUF USE FREE LOAD_GLOB SAVE_EACH` and the lowercase expression
  words `and or not true false`.

## 2. Values and types

| Type      | Examples                          | Notes |
|-----------|-----------------------------------|-------|
| int       | `42`, `0xFF`, `-3`                | arbitrary precision (Python int) |
| float     | `1.5`, `1e3`                      | |
| str       | `"inspect"`                       | |
| bool      | `true`, `false`                   | result of comparisons |
| buffer    | `ALLOC`, `BUF`, `CAST(...)`        | numpy array with a dtype; see section 8 |
| pointer   | `&buf[10:20, 0:5]`                | live view into a buffer |
| none      | result of a FUNC without RETURN   | |

**dtypes**: `u8 i8 u16 i16 i32 u32 f32 f64`.

Scalars are copied on assignment and call. Buffers and pointers are passed by
reference (see slice-copy rules in section 8).

**Variables**: `SET name <expr>` writes a variable. Globals live in
`executor.context["vars"]`. Each FUNC call has its own frame; see section 6.

**The current image `_`** is a register holding the image commands operate on.
Read it in expressions as `_` (`_.width`, `max(_)`), replace it with
`USE <buffer>` or `SET _ <buffer>`.

## 3. Expressions

Expressions appear after `SET`, in `IF` / `ELIF` / `WHILE` / `ASSERT` /
`RETURN`, in `FOR` bounds, `REPEAT` counts, index lists, call arguments, and
**inside parentheses in command arguments**:

```
RESIZE ($w / 2) 480          # parenthesised expression as an argument
BLUR -1                      # bare literal; no expression parsing
```

### Precedence (low to high)

| Level | Operators                                  | Assoc |
|-------|--------------------------------------------|-------|
| 1     | `expr AS dtype` (reinterpret)              | left  |
| 2     | `or`                                       | left  |
| 3     | `and`                                      | left  |
| 4     | `not`                                      | prefix|
| 5     | `==  !=  <  >  <=  >=`                     | left  |
| 6     | `\|`                                       | left  |
| 7     | `^`                                        | left  |
| 8     | `&`                                        | left  |
| 9     | `<<  >>`                                   | left  |
| 10    | `+  -`                                     | left  |
| 11    | `*  /  //  %`                              | left  |
| 12    | unary `-  ~  &  *`                         | prefix|
| 13    | postfix `[...]` index/slice, `.attr`, `(...)` call | left |

`and` / `or` short-circuit and return the deciding operand. `/` is true
division, `//` floor division. `+` concatenates when either side is a string.
Comparisons between a numeric string and a number coerce (`"120" > 100` is
true, kept from v0.3); other mismatches raise an error.

### Bare identifiers (legacy)

In command arguments and expressions a bare word `name` resolves, in order:
variable, buffer slot, `_`, else the string `"name"` (`HEATMAP jet`). The left
side of an `IF` comparison must be defined (error otherwise); in `ASSERT` it
falls back to the image properties `width height channels depth min max
contour_count ndim`. The linter warns (PX012) when such a word looks like a
typo of a variable.

## 4. Statements

| Statement | Meaning |
|-----------|---------|
| `COMMAND args...` | run a command (builtin, plugin or user FUNC) |
| `SET name <expr>` / `SET name = <expr>` | assign a variable |
| `SET buf[y, x, c] <expr>` | POKE (section 8) |
| `SET *p <expr>` / `*p = <expr>` | write through a pointer |
| `INCLUDE "file.pxl"` | splice another file (link-time) |
| `ASSERT <expr> ["message"]` | fail the run if false |
| `ROI x y w h` ... `ROI_RESET` | run the body on a crop and paste it back |
| `WITH *p` ... `END` | run the body in place on a pointed region |
| `ALLOC` `BUF` `USE` `FREE` | buffers (section 8) |
| `FUNC` `RETURN` `GLOBAL` `IMPORT` `FROM` | functions and modules |

`ASSERT` keeps the v0.4 forms: `ASSERT width == 640 "msg"` where `width` may
be a variable or an image property.

`ROI` clamps to the image, converts gray/colour mismatches, resizes a body
result back to the region, and merges metadata exactly as in v0.4.

## 5. Control flow

```
IF $mode == "a"            # any expression; legacy `IF name op value` works
    ...
ELIF $x > 3 and $x < 9
    ...
ELSE
    ...
ENDIF

REPEAT 3 [AS i]            # count must be an integer >= 0
WHILE <expr>               # capped at 1,000,000 iterations by default
FOR i = a TO b [STEP s]    # inclusive; STEP may be negative or fractional
    BREAK | CONTINUE
END
```

* `$ITER` (0-based) and `$ITER1` (1-based) describe the **innermost**
  `REPEAT`; they are restored after the loop and never leak. `REPEAT n AS i`
  and `FOR i` bind loop variables in a per-loop scope that disappears when the
  loop ends and never clobbers an outer variable of the same name.
* The `WHILE` / `FOR` cap is `[run] max_loop_iterations` in `pixlang.toml`
  or `Executor(max_loop_iterations=...)`; exceeding it raises an error.
* `BREAK` / `CONTINUE` outside a loop is a syntax error.

## 6. Functions

```
EXPORT FUNC SHARPEN_TWICE(k)     # ALL_CAPS: command-style
    SHARPEN
    BLUR $k
ENDFUNC

FUNC clamp(v, lo, hi)            # lowercase: expression function
    IF $v < $lo
        RETURN $lo
    ENDIF
    RETURN min($v, $hi)
ENDFUNC

SHARPEN_TWICE 3
SET y clamp($x, 0, 255)
```

* **Command FUNCs** (ALL_CAPS) are called like commands. They receive the
  caller's current image as `_` and the caller sees the callee's final image.
  `RETURN <image>` replaces it. They cannot be used inside expressions.
* **Expression FUNCs** (any other name) return a value (`none` without
  `RETURN`) and leave the caller's image untouched. They can also be called as
  statements.
* `FUNC` may appear only at the top level of a file. Arity is checked.
* **Frames**: each call has its own variable frame. Parameters are locals;
  scalars are passed by value, buffers by reference. `SET` inside a function
  writes a local unless the name is declared `GLOBAL name`. Reads fall back to
  globals. Recursion is allowed up to depth 256.
* **Lookup order** for a command name: user FUNC (local, `FROM` imported,
  `alias.NAME`) then the command registry (plugins/builtins). The linker warns
  (LNK012) when a FUNC shadows a registered command.
* Errors raised inside functions carry a call stack
  (`file:line:col: in caller: call callee` per frame).

## 7. Modules and the linker

```
FROM "lib/filters.pxl" IMPORT SHARPEN_TWICE, clamp
IMPORT "lib/filters.pxl" AS f          # alias defaults to the file stem
f.SHARPEN_TWICE 3
SET y f.clamp($x, 0, 255)
```

* Namespaced calls are `alias.NAME args` (statement) and `alias.name(args)`
  (expression); the alias must be a plain word with no space before `.`.
* Only `EXPORT`ed functions are importable. If a file has no `EXPORT` at all,
  all its functions are public.
* Top-level `SET` statements in an imported module are **global initialisers**,
  run once before the entry file, dependencies first. Other executable
  statements are rejected in imported modules (LNK006).
* `IMPORT` / `FROM` must be top-level statements.

### Linking

`pixlang.linker.link(path)` compiles each file into an **ObjectUnit**
(`path`, `sha256`, `funcs`, `exports`, `imports`, `globals`, `body`, `refs`
with locations) and links them by depth-first search from the entry file.
Module paths resolve relative to the importing file, then the
`[link] search_paths` of `pixlang.toml`. `INCLUDE` is a link-time splice:
cycle-checked, resolved relative to the *including* file, with file-tagged
locations. The result, a **LinkedProgram** (function table, per-unit symbol
tables, global initialisers, entry body, diagnostics), runs without touching
the file system:

```python
prog = link("main.pxl")
Executor(registry=registry).run(prog)
```

`Executor.run()` also accepts a bare `Pipeline` (what `parse()` returns); it
is linked on the fly and `INCLUDE`s are resolved at run time relative to the
including file, as before.

| Code   | Severity | Meaning |
|--------|----------|---------|
| LNK001 | error    | module not found |
| LNK002 | error    | duplicate definition / import name clash |
| LNK003 | error    | import or INCLUDE cycle (full chain reported) |
| LNK004 | error    | INCLUDE file not found |
| LNK005 | error    | FUNC / IMPORT in a file INCLUDEd inside a block |
| LNK006 | error    | executable statement in an imported module |
| LNK007 | error    | imported name does not exist (with suggestion) |
| LNK008 | error    | imported name is not exported |
| LNK009 | error    | command FUNC used in an expression |
| LNK010 | error    | undefined function / unknown module member |
| LNK011 | warning  | unknown command (not a FUNC, not registered) |
| LNK012 | warning  | FUNC shadows a registered command |
| LNK013 | warning  | unused import |
| LNK014 | warning  | global initialised by several modules |

CLI: `pixlang link file.pxl [--map]` prints diagnostics (and the symbol map)
and exits 1 on errors.

## 8. Buffers, pointers and memory

### Buffers

Named buffer slots live in `executor.context["checkpoints"]` (also
`context["buffers"]`), the same store `CHECKPOINT`, `OVERLAY`, `BLEND`,
`COMPARE` and `BITWISE_*` use. A slot holds an n-dimensional numpy array,
shape `(height, width)` or `(height, width, channels)`.

```
ALLOC canvas 640 480 3 u8 0   # name width height [channels] [dtype] [fill]
BUF orig                      # store a copy of _ in slot orig
BUF mask (orig & 0xF0)        # store a copy of an expression result
USE canvas                    # _ := a copy of canvas
FREE orig, mask               # release slots
```

`ALLOC name w h` defaults to 1 channel, `u8`, fill 0. A buffer name used in an
expression yields the array itself (by reference); a *variable* of the same
name wins.

### PEEK and POKE

```
SET v orig[12, 40, 2]         # (y, x, channel); integers only; negatives allowed
SET orig[12, 40, 2] 255       # POKE
SET orig[0:2, :] 0            # slice assignment (scalar or buffer)
```

Reading a slice (`SET s orig[0:2, 0:2]`) yields a **copy**; use `&` for a
live view. Stores convert with a C-style cast: **integer stores wrap around**
(`u8` 256 becomes 0, -1 becomes 255; `i8` 200 becomes -56).

### Attributes

`width height channels dtype nbytes stride itemsize size ndim shape addr`.
`stride` is bytes per row (of the parent for views), `itemsize` bytes per
element, `size` the element count, `addr` the address of the first byte
(stable while the buffer lives, for display). Pointers also have `offset`
(bytes from the start of the base buffer).

### Pointers

A pointer is a live numpy view into a buffer, made with `&`:

```
SET p &orig[10:110, 20:220]   # region of a buffer
SET q &orig                   # whole buffer
SET r &p[1:5, 1:5]            # pointer into a pointer
*p = 7                        # or SET *p 7 : write through (broadcasts)
SET v *p                      # read the current contents (a view / scalar)
WITH *p                       # run the body on the region, in place
    BLUR 5
END
```

* Pointers carry `(slot, generation, index)`. `FREE`, re-`ALLOC`, and `BUF`
  over an existing slot bump the slot's generation; using a stale pointer
  raises `use-after-free (line N)` (names the line of the `FREE` and where the
  pointer was taken) or `dangling pointer` (slot reallocated).
* `WITH` runs its body with `_` set to a copy of the region, then writes the
  result back through the view (gray/colour and size mismatches are
  converted). The caller's `_` is untouched.
* `ROI x y w h ... ROI_RESET` behaves exactly as in v0.4 (it crops `_`).
* Pointer arithmetic (`p + n`) is not supported; use slices.
* `&_` is rejected (store `_` with `BUF` first).

### Reinterpreting and converting

```
SET raw orig AS u32           # same bytes, no copy; sizes validated
SET f   CAST(orig, f32)       # converting copy (wrapping C-style cast)
SIZEOF(u16)                   # 2
```

`AS` first tries a numpy view along the last axis, then a flat view of a
contiguous buffer; otherwise it raises `cannot reinterpret N bytes ...`.
Writes through the view are visible in the original.

### Whole-buffer arithmetic

All operators work element-wise on buffers (and buffer/scalar):

```
SET mask (orig & 0xF0) | (orig >> 4)
```

Integer buffers keep their dtype and **wrap like hardware registers**: a
Python int operand is converted to the buffer dtype first, so `u8 250 + 10`
is `4`, `i16 32767 + 1` is `-32768`, `u8 0x81 << 1` is `2`. Bitwise operators
require integer buffers. Comparisons give boolean masks; a multi-element
buffer has no truth value (use `any()` / `all()`). Shape mismatches raise a
clear error.

## 9. Builtin functions

| Function | Notes |
|----------|-------|
| `min(a, b, ...)`, `max(...)` | scalars, buffers (element-wise), or one buffer (reduction) |
| `abs(x)`, `sqrt(x)`, `floor(x)`, `ceil(x)`, `round(x[, n])` | numbers or buffers |
| `clamp(v, lo, hi)` | numbers or buffers |
| `int(x)`, `float(x)`, `str(x)`, `hex(n)`, `bin(n)` | conversions (`int("0x10")` works) |
| `len(x)` | string length / buffer height |
| `sum(buf)`, `mean(buf)`, `any(buf)`, `all(buf)`, `copy(buf)` | reductions (`sum` cannot overflow) |
| `CAST(x, dtype)` | converting copy |
| `SIZEOF(dtype)` | bytes per element |
| `ZEROS(w, h[, c[, dtype]])` | new buffer |

User FUNCs shadow builtins of the same name inside their unit.

## 10. Errors and diagnostics

* **`PixSyntaxError(SyntaxError)`**: `.lineno`, `.offset` (1-based column),
  `.filename`, `.msg`, plus `.line`, `.col`, `.file`, `.pretty()`.
* **`PixRuntimeError(RuntimeError)`**: `.line`, `.col`, `.file`, `.msg`,
  `.trace` (call stack). Message prefix: `[PixLang] Error at line N, col C ...`;
  command failures read `[PixLang] Error at line N ... (CMD args): reason`.
* **`PixAssertionError(AssertionError)`** for failed `ASSERT`;
  **`PixLinkError(PixRuntimeError)`** with `.diagnostics`.

Diagnostics render compiler-style:

```
pipeline.pxl:3:7: error: unexpected int '5' ...
    BLUR 3 5
           ^
```

Common syntax errors: `END without REPEAT, WHILE, FOR or WITH`,
`unterminated IF block opened at line N (expected ENDIF)`, `INCLUDE cycle:
a.pxl → b.pxl → a.pxl`, `unterminated string literal`.

## 11. Linter

Rules PX001-PX012 plus the linker's LNK codes (see `pixlang/linter/engine.py`
for the list). Highlights: blocks and included files are analysed; PX001
accepts `LOAD_GLOB`/`INCLUDE`/`USE` starts; PX005 is order-aware and knows
variables from `pixlang.toml`; PX008 flags `REPEAT 0` as a warning; PX012
warns when a bare word silently becomes a string but resembles a defined
variable; a crashing rule is reported as PX000 instead of being hidden.
`[lint] ignore = [...]` is honoured when a file path is known
(`Linter.lint(pipeline, path=...)` / `lint_file`).

## 12. Configuration (`pixlang.toml`)

```toml
[defaults]   verbose = false
[variables]  width = 640            # injected before the run
[lint]       ignore = ["PX006"]
[link]       search_paths = ["lib"] # relative to the toml file
[run]        max_loop_iterations = 1000000
```

Uses `tomllib` (3.11+), `tomli` when installed, else a built-in reader that
handles tables, strings, numbers, booleans, arrays and trailing comments.

## 13. Grammar

```
pipeline   := (statement NL)*
statement  := command | SET target ['='] expr | '*' unary '=' expr
            | IF expr NL block (ELIF expr NL block)* [ELSE NL block] ENDIF
            | REPEAT expr [AS word] NL block END
            | WHILE expr NL block END
            | FOR word ['='] expr TO expr [STEP expr] NL block END
            | BREAK | CONTINUE
            | ROI arg arg arg arg NL block ROI_RESET
            | WITH expr NL block END
            | INCLUDE STRING | ASSERT expr [STRING]
            | [EXPORT] FUNC word ['(' [word (',' word)*] ')'] NL block ENDFUNC
            | RETURN [expr] | GLOBAL word (',' word)*
            | IMPORT STRING [AS word] | FROM STRING IMPORT word (',' word)*
            | ALLOC word arg arg [arg] [dtype] [arg]
            | BUF word [expr] | USE expr | FREE word (',' word)*
command    := (COMMAND | word '.' COMMAND) arg*
arg        := INT | FLOAT | STRING | VAR | word | true | false | '(' expr ')'
target     := word ['[' index (',' index)* ']'] | '*' unary
index      := expr | [expr] ':' [expr] [':' [expr]]
expr       := Pratt expression; precedence table in section 3
```

## 14. Compatibility with v0.4

* All v0.4 programs that put each statement on its own line (or use `\`) run
  unchanged. **Newlines now end statements**: `BLUR 3` followed by a line
  starting with a number is a syntax error instead of an extra argument.
* In command arguments an expression must be parenthesised.
* `$ITER` / `$ITER1` now refer to the innermost loop and are removed after it.
* Legacy AST shapes (`Command`, `SetVar`, `IfBlock.var_name/op/cmp_value`,
  `RepeatBlock`, `RoiBlock`, `IncludeStmt`, `AssertStmt`) are preserved; v0.5
  adds fields and node types (`ast_nodes.py` docstring). Literal arguments
  and `SET` values are still plain Python values; expressions are `Expr`
  nodes.
* Not in v0.5 (next steps): a bytecode / IR backend, an on-disk compile
  cache, and pointer arithmetic.
