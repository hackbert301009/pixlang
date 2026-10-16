# pixlang/_style.py
"""Terminal styling shared by the CLI, batch runner and watcher.

ANSI colours are used only when the stream is a TTY and ``NO_COLOR`` is not
set (https://no-color.org). ``FORCE_COLOR`` forces them on.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

_CODES = {
    "cyan": "\033[96m", "green": "\033[92m", "red": "\033[91m",
    "yellow": "\033[93m", "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m",
}


def use_color(stream: Optional[TextIO] = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def is_tty(stream: Optional[TextIO] = None) -> bool:
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


class Style:
    """Attribute access to ANSI codes that collapse to '' when colour is off."""

    def __init__(self, stream: Optional[TextIO] = None):
        self.enabled = use_color(stream)

    def __getattr__(self, name: str) -> str:
        if name.startswith("_") or name not in _CODES:
            raise AttributeError(name)
        return _CODES[name] if self.enabled else ""

    def paint(self, text: str, *names: str) -> str:
        if not self.enabled:
            return text
        return "".join(_CODES[n] for n in names) + text + _CODES["reset"]
