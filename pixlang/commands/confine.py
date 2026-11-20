"""Keep an embedder's file access inside a few directories.

Commands declare which of their parameters are paths (``paths=`` on
``CommandRegistry.register``). An embedder such as the web editor gives its
executor a :class:`PathGuard`; the executor then runs every declared path
through it before the command sees the value.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

# Parameters with these names must be declared, whether or not the command is
# a built-in: it is how a new INFER-style command cannot slip past the guard.
PATH_PARAM_NAMES = frozenset({
    "path", "file", "filename", "template", "pattern", "model", "labels",
    "dir", "directory", "folder",
})

# read/write: one file; glob: a pattern matched against the disk;
# template: an output path built at run time from variables
PATH_MODES = ("read", "write", "glob", "template")

_GLOB_CHARS = "*?["


class PathConfinementError(PermissionError):
    """A path leaves the allowed directories or is not an ordinary file."""


def confine(path: Union[str, os.PathLike[str]], roots: Iterable[Path], mode: str = "read",
            *, base: Optional[Path] = None, area: str = "the allowed directories") -> Path:
    """Resolve *path* (symlinks included) and make sure it lies under one of *roots*.

    ``mode`` is ``"read"``, ``"write"`` or ``"dir"``. Existing targets of read
    and write must be regular files, which keeps FIFOs, devices and ``/proc``
    entries from blocking or leaking. A missing read target is returned as is:
    the command reports it in its own words.
    """
    text = os.fspath(path)
    if "\0" in text:
        raise PathConfinementError(f"path {text!r} contains a NUL byte")
    candidate = Path(text)
    if base is not None and not candidate.is_absolute():
        candidate = Path(base) / candidate
    try:
        resolved = candidate.resolve()
        allowed = [Path(r).resolve() for r in roots]
    except (OSError, RuntimeError) as exc:
        raise PathConfinementError(f"path {text!r} cannot be resolved ({exc})") from None
    if not any(resolved.is_relative_to(r) for r in allowed):
        raise PathConfinementError(f"path {text!r} is outside {area}")
    if mode != "dir" and resolved.exists() and not resolved.is_file():
        raise PathConfinementError(f"path {text!r} is not a regular file")
    return resolved


class PathGuard:
    """The directories an embedded run may use, and where relative paths start.

    Relative reads start in ``base_dir`` and fall back to ``fallback_dir``;
    relative writes start in ``write_base`` (default ``base_dir``).
    """

    def __init__(self, base_dir: Path, read_roots: Iterable[Path], write_roots: Iterable[Path],
                 *, write_base: Optional[Path] = None, fallback_dir: Optional[Path] = None,
                 area: str = "the editor workspace") -> None:
        self.base_dir = Path(base_dir)
        self.read_roots: List[Path] = [Path(r) for r in read_roots]
        self.write_roots: List[Path] = [Path(r) for r in write_roots]
        self.write_base = Path(write_base) if write_base is not None else self.base_dir
        self.fallback_dir = Path(fallback_dir) if fallback_dir is not None else None
        self.area = area

    def resolve(self, value: str, mode: str) -> str:
        """The confined form of a declared path argument, ready for the command."""
        if mode == "write":
            return str(confine(value, self.write_roots, "write",
                               base=self.write_base, area=self.area))
        if mode == "read":
            return str(confine(self._read_start(value), self.read_roots, "read",
                               base=self.base_dir, area=self.area))
        if mode == "glob":
            return self._glob(value)
        if mode == "template":
            self._template(value)
            return value
        raise ValueError(f"unknown path mode {mode!r}")

    def allows(self, path: Path, mode: str = "read") -> bool:
        roots = self.write_roots if mode == "write" else self.read_roots
        try:
            confine(path, roots, mode, area=self.area)
        except PathConfinementError:
            return False
        return True

    def check_module(self, path: Path) -> Path:
        """Gate for IMPORT / INCLUDE targets: a ``.pxl`` file inside the read roots."""
        resolved = confine(path, self.read_roots, "read", area=self.area)
        if resolved.suffix.lower() != ".pxl":
            raise PathConfinementError(f"path {str(path)!r} is not a .pxl file")
        return resolved

    def _read_start(self, value: str) -> str:
        if self.fallback_dir is not None and not Path(value).is_absolute() \
                and not (self.base_dir / value).exists() and (self.fallback_dir / value).exists():
            return str(self.fallback_dir / value)
        return value

    def _glob(self, pattern: str) -> str:
        if pattern.startswith("~") or ".." in Path(pattern).parts:
            raise PathConfinementError(f"path {pattern!r} is outside {self.area}")
        full = pattern if Path(pattern).is_absolute() else str(self.base_dir / pattern)
        cut = min((i for i in (full.find(c) for c in _GLOB_CHARS) if i >= 0), default=len(full))
        confine(os.path.dirname(full[:cut]) or ".", self.read_roots, "dir", area=self.area)
        return full

    def _template(self, template: str) -> None:
        literal = template.split("{", 1)[0]
        if ".." in Path(template).parts:
            raise PathConfinementError(f"path {template!r} is outside {self.area}")
        if Path(literal).is_absolute():
            confine(os.path.dirname(literal), self.write_roots, "dir", area=self.area)
