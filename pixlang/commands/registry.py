# pixlang/commands/registry.py
"""
CommandRegistry — maps DSL keyword strings to Python callables.

v0.2 additions:
  - Every registration records its *source* (builtin | plugin name | local file)
  - Conflict policy: by default a plugin cannot silently overwrite a builtin;
    pass allow_override=True to explicitly permit shadowing.
  - list_info() returns rich CommandInfo objects with docs and origin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class CommandInfo:
    name:   str
    fn:     Callable
    source: str   # "builtin" | plugin package name | "local:<path>"
    doc:    str = ""

    def __repr__(self):
        return f"<CommandInfo {self.name!r} from={self.source!r}>"


class CommandRegistry:
    BUILTIN_SOURCE = "builtin"

    def __init__(self):
        self._commands: Dict[str, CommandInfo] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, *, source: str = "builtin", allow_override: bool = False):
        """Decorator factory — @registry.register('CMD', source='my_plugin')"""
        def decorator(fn: Callable) -> Callable:
            key = name.upper()
            if key in self._commands and not allow_override:
                existing = self._commands[key]
                raise ConflictError(
                    f"[PixLang Registry] Command '{key}' is already registered "
                    f"by '{existing.source}'. Pass allow_override=True to shadow it."
                )
            doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
            self._commands[key] = CommandInfo(name=key, fn=fn, source=source, doc=doc)
            return fn
        return decorator

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> Callable:
        key = name.upper()
        if key not in self._commands:
            suggestions = self._suggestions(key)
            hint = f"  Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise NameError(
                f"[PixLang] Unknown command: '{key}'.{hint}\n"
                f"  Run 'pixlang commands' to list all {len(self._commands)} commands."
            )
        return self._commands[key].fn

    def get_info(self, name: str) -> CommandInfo:
        key = name.upper()
        if key not in self._commands:
            raise NameError(f"[PixLang] Unknown command: '{key}'.")
        return self._commands[key]

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

    def __len__(self) -> int:
        return len(self._commands)

    def __contains__(self, name: str) -> bool:
        return name.upper() in self._commands


    def _suggestions(self, key: str, max_n: int = 3) -> List[str]:
        """Simple prefix/substring typo suggestions."""
        candidates = []
        for k in self._commands:
            if k.startswith(key[:3]) or (len(key) >= 4 and key[:4] in k):
                candidates.append(k)
        return candidates[:max_n]


class ConflictError(Exception):
    """Raised when two sources try to register the same command name."""
