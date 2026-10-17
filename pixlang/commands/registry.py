# pixlang/commands/registry.py
"""
CommandRegistry — maps DSL keyword strings to Python callables.

Every registration records:
  - its *source* (builtin | plugin name | local file label), chosen by the author
  - its *origin* (the loader that registered it, e.g. ``local:/abs/x.plugins.py``),
    which lets a plugin loader unload/reload everything a file registered
  - a cached signature, from which ``usage``, ``params`` and argument
    coercion are derived
  - a ``category`` (I/O, Geometry, Color, ...) used by ``pixlang commands``
    and the editor

Conflict policy: a plugin cannot silently overwrite an existing command;
pass ``allow_override=True`` to explicitly shadow it. Unregistering the
shadowing command restores the previous one.

``invoke(name, image, args, ctx)`` is the single dispatch entry point used by
the executor. It performs arity checking (with a usage line) and coerces
arguments to the annotated parameter types (int / float / str / bool).
"""
from __future__ import annotations

import contextlib
import difflib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

_SKIP_PARAMS = ("image", "_ctx")
_TYPE_NAMES = {int: "int", float: "float", str: "str", bool: "bool"}
_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


@dataclass
class CommandInfo:
    name:   str
    fn:     Callable
    source: str   # "builtin" | plugin package name | "local:<path>"
    doc:    str = ""
    accepts_ctx: bool = False
    signature: Optional[inspect.Signature] = field(default=None, repr=False)
    category: str = ""
    origin: str = ""   # loader-level source used by unregister_source()
    shadowed: Optional[CommandInfo] = field(default=None, repr=False)

    def __repr__(self):
        return f"<CommandInfo {self.name!r} from={self.source!r}>"

    # ── Derived metadata ────────────────────────────────────────────────────

    @property
    def _arg_params(self) -> List[inspect.Parameter]:
        if self.signature is None:
            return []
        return [
            p for name, p in self.signature.parameters.items()
            if name not in _SKIP_PARAMS
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        ]

    @property
    def params(self) -> List[Tuple[str, str, Any]]:
        """``[(name, type_name, default_or_None), ...]`` excluding image/_ctx."""
        out = []
        for p in self._arg_params:
            tname = _type_name(p.annotation)
            if p.kind == p.VAR_POSITIONAL:
                out.append((f"*{p.name}", tname, None))
                continue
            default = None if p.default is inspect.Parameter.empty else p.default
            out.append((p.name, tname, default))
        return out

    @property
    def usage(self) -> str:
        """Clean usage string, e.g. ``THRESHOLD <value:int> [mode:str=binary]``."""
        if self.signature is None:
            return _doc_syntax(self.doc, self.name)
        parts = [self.name]
        for p in self._arg_params:
            tname = _type_name(p.annotation)
            label = f"{p.name}:{tname}" if tname else p.name
            if p.kind == p.VAR_POSITIONAL:
                parts.append(f"[{label}...]")
            elif p.default is inspect.Parameter.empty:
                parts.append(f"<{label}>")
            elif p.default is None:
                parts.append(f"[{label}]")
            else:
                parts.append(f"[{label}={p.default}]")
        return " ".join(parts)

    @property
    def summary(self) -> str:
        """Short description: the docstring part after ' — ' (or the whole line)."""
        for sep in (" — ", " - "):
            if sep in self.doc:
                return self.doc.split(sep, 1)[1].strip()
        return self.doc

    @property
    def arity(self) -> Tuple[int, Optional[int]]:
        """(min_args, max_args or None for variadic)."""
        lo, hi = 0, 0
        for p in self._arg_params:
            if p.kind == p.VAR_POSITIONAL:
                return lo, None
            hi += 1
            if p.default is inspect.Parameter.empty:
                lo += 1
        return lo, hi


class CommandRegistry:
    BUILTIN_SOURCE = "builtin"

    def __init__(self):
        self._commands: Dict[str, CommandInfo] = {}
        self._origin_stack: List[str] = []

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, *, source: str = "builtin", allow_override: bool = False,
                 category: Optional[str] = None):
        """Decorator factory — @registry.register('CMD', source='my_plugin')"""
        def decorator(fn: Callable) -> Callable:
            key = name.upper()
            existing = self._commands.get(key)
            if existing is not None and not allow_override:
                raise ConflictError(
                    f"[PixLang Registry] Command '{key}' is already registered "
                    f"by '{existing.source}'. Pass allow_override=True to shadow it."
                )
            doc = (fn.__doc__ or "").strip().splitlines()[0].strip() if fn.__doc__ else ""
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                sig = None
            accepts_ctx = sig is not None and "_ctx" in sig.parameters
            origin = self._origin_stack[-1] if self._origin_stack else source
            cat = category or ("Plugin" if source != self.BUILTIN_SOURCE else "")
            self._commands[key] = CommandInfo(
                name=key, fn=fn, source=source, doc=doc,
                accepts_ctx=accepts_ctx, signature=sig,
                category=cat, origin=origin, shadowed=existing,
            )
            return fn
        return decorator

    @contextlib.contextmanager
    def origin(self, tag: str) -> Iterator[None]:
        """Attribute every registration inside the block to loader origin *tag*."""
        self._origin_stack.append(tag)
        try:
            yield
        finally:
            self._origin_stack.pop()

    def unregister(self, name: str) -> bool:
        """Remove one command; a command it shadowed (allow_override) is restored."""
        key = name.upper()
        info = self._commands.pop(key, None)
        if info is None:
            return False
        if info.shadowed is not None:
            self._commands[key] = info.shadowed
        return True

    def unregister_source(self, source: str) -> List[str]:
        """Remove every command whose ``source`` or ``origin`` equals *source*.

        Returns the removed command names. Used by the plugin loader to roll
        back a partially failed registration and to reload a changed plugin.
        """
        removed = []
        for key in list(self._commands):
            info = self._commands.get(key)
            while info is not None and source in (info.source, info.origin):
                self.unregister(key)
                removed.append(key)
                info = self._commands.get(key)
        return sorted(set(removed))

    def set_category(self, name: str, category: str) -> None:
        key = name.upper()
        if key in self._commands:
            self._commands[key].category = category

    # ── Lookup ───────────────────────────────────────────────────────────────

    def _unknown(self, key: str) -> UnknownCommandError:
        suggestions = self._suggestions(key)
        hint = f"  Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return UnknownCommandError(
            f"[PixLang] Unknown command: '{key}'.{hint}\n"
            f"  Run 'pixlang commands' to list all {len(self._commands)} commands.",
            name=key, suggestions=suggestions,
        )

    def get(self, name: str) -> Callable:
        key = name.upper()
        if key not in self._commands:
            raise self._unknown(key)
        return self._commands[key].fn

    def get_info(self, name: str) -> CommandInfo:
        key = name.upper()
        if key not in self._commands:
            raise self._unknown(key)
        return self._commands[key]

    def invoke(self, name: str, image: Any, args: List[Any], ctx: Optional[Dict[str, Any]] = None) -> Any:
        """Run command *name* on *image* with already-resolved *args*.

        Single entry point used by the executor:
          1. looks the command up (UnknownCommandError with suggestions),
          2. checks the argument count against the signature (usage on error),
          3. coerces args to the annotated types (int/float/str/bool),
          4. passes ``_ctx`` only to commands that declare it,
          5. re-attaches the input's metadata if the command returned a bare
             array (unless the command is marked ``fresh_image``).
        """
        info = self.get_info(name)
        args = list(args)
        lo, hi = info.arity
        if len(args) < lo or (hi is not None and len(args) > hi):
            want = f"{lo}" if lo == hi else (f"at least {lo}" if hi is None else f"{lo}-{hi}")
            raise ArgumentError(
                f"[{info.name}] expects {want} argument(s), got {len(args)}.\n"
                f"  Usage: {info.usage}"
            )
        args = _coerce_args(info, args)
        if info.accepts_ctx:
            result = info.fn(image, *args, _ctx=ctx if ctx is not None else {})
        else:
            result = info.fn(image, *args)
        return _carry_meta(info.fn, image, result)

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_commands(self) -> List[str]:
        return sorted(self._commands.keys())

    def list_info(self) -> List[CommandInfo]:
        return [self._commands[k] for k in sorted(self._commands)]

    def commands_by_source(self) -> Dict[str, List[CommandInfo]]:
        groups: Dict[str, List[CommandInfo]] = {}
        for info in self._commands.values():
            groups.setdefault(info.source, []).append(info)
        return dict(sorted(groups.items()))

    def commands_by_category(self) -> Dict[str, List[CommandInfo]]:
        groups: Dict[str, List[CommandInfo]] = {}
        for info in self.list_info():
            groups.setdefault(info.category or "Other", []).append(info)
        return dict(sorted(groups.items()))

    def __len__(self) -> int:
        return len(self._commands)

    def __contains__(self, name: str) -> bool:
        return name.upper() in self._commands

    def _suggestions(self, key: str, max_n: int = 3) -> List[str]:
        """Close-match typo suggestions (difflib), falling back to prefix match."""
        names = list(self._commands)
        found = difflib.get_close_matches(key, names, n=max_n, cutoff=0.6)
        if not found and len(key) >= 3:
            found = [k for k in sorted(names) if k.startswith(key[:3])][:max_n]
        return found


# ── Errors ────────────────────────────────────────────────────────────────────

class ConflictError(Exception):
    """Raised when two sources try to register the same command name."""


class UnknownCommandError(NameError):
    """Raised for a command name that is not registered."""

    def __init__(self, msg: str = "", name: str = "", suggestions: Optional[List[str]] = None):
        super().__init__(msg)
        self.command = name
        self.suggestions = list(suggestions or [])


class ArgumentError(TypeError):
    """Wrong number of arguments or an argument that cannot be coerced."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _type_name(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return ""
    if ann in _TYPE_NAMES:
        return _TYPE_NAMES[ann]
    if isinstance(ann, str):          # postponed annotations in plugin files
        return ann.strip().strip("'\"")
    return getattr(ann, "__name__", str(ann))


def _doc_syntax(doc: str, name: str) -> str:
    for sep in (" — ", "  - "):
        if sep in doc:
            return doc.split(sep, 1)[0].strip()
    return doc.strip() or name


def _coerce_args(info: CommandInfo, args: List[Any]) -> List[Any]:
    params = info._arg_params
    out = []
    for i, value in enumerate(args):
        if i < len(params):
            p = params[i]
        elif params and params[-1].kind == params[-1].VAR_POSITIONAL:
            p = params[-1]
        else:
            out.append(value)
            continue
        out.append(_coerce(info.name, p.name, _type_name(p.annotation), value))
    return out


def _coerce(cmd: str, pname: str, tname: str, value: Any) -> Any:
    if value is None or tname not in ("int", "float", "str", "bool"):
        return value

    def fail(extra: str = "") -> ArgumentError:
        return ArgumentError(
            f"[{cmd}] parameter '{pname}' expects {tname}, got {value!r}{extra}"
        )

    if tname == "str":
        return value if isinstance(value, str) else str(value)
    if tname == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        s = str(value).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise fail(" (use true/false)")
    if isinstance(value, bool):
        raise fail()
    if tname == "float":
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except ValueError:
            raise fail() from None
    # int
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise fail(" (not a whole number)")
    s = str(value).strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except ValueError:
        raise fail() from None
    if f.is_integer():
        return int(f)
    raise fail(" (not a whole number)")


def _carry_meta(fn: Callable, image: Any, result: Any) -> Any:
    """If *image* carried metadata and *result* is a bare array, re-wrap it."""
    if getattr(fn, "fresh_image", False):
        return result
    if not isinstance(image, dict) or isinstance(result, dict) or result is None:
        return result
    if not hasattr(result, "shape"):
        return result
    wrapped = {k: v for k, v in image.items() if k != "_image"}
    wrapped["_image"] = result
    return wrapped
