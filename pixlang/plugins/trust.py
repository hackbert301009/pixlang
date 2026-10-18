# pixlang/plugins/trust.py
"""
PixLang Workspace Trust  (v0.5)
═══════════════════════════════

``[plugins] trust_local = true`` in a ``pixlang.toml`` is a statement by a
*project* about itself. A downloaded bundle can ship that file alongside its
own ``pipeline.plugins.py``, so on its own it cannot establish trust: it would
let arbitrary Python run from a plain ``pixlang run``.

A directory therefore only grants local-plugin trust once the user has
confirmed it here, outside the project, with ``pixlang trust <dir>``. The list
lives in ``~/.pixlang/trusted.toml`` (override with ``$PIXLANG_TRUST_FILE``):

    # PixLang — directories whose pixlang.toml may trust local plugins.
    [trust]
    paths = [
      "/home/me/projects/edge-detect",
    ]

Confirming a directory also covers everything below it, so trusting a checkout
once is enough for its subdirectories.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

TRUST_FILE_ENV     = "PIXLANG_TRUST_FILE"
DEFAULT_TRUST_FILE = Path.home() / ".pixlang" / "trusted.toml"

HEADER = ("# PixLang — directories whose pixlang.toml may trust local plugins.\n"
          "# Managed by `pixlang trust`; see `pixlang trust --help`.\n")


# ── Store location ─────────────────────────────────────────────────────────────

def trust_file() -> Path:
    """The file holding the confirmed workspaces."""
    env = os.environ.get(TRUST_FILE_ENV)
    return Path(env).expanduser() if env else DEFAULT_TRUST_FILE


def _resolve(directory) -> Path:
    """A comparable absolute path; works for directories that do not exist yet."""
    path = Path(directory).expanduser()
    try:
        return path.resolve()
    except OSError:                     # pragma: no cover - unreadable parent
        return path.absolute()


# ── Reading ────────────────────────────────────────────────────────────────────

def trusted_paths() -> List[Path]:
    """Every confirmed directory, in the order it was added."""
    from pixlang.plugins.loader import _read_toml
    path = trust_file()
    if not path.is_file():
        return []
    data = _read_toml(path)
    section = data.get("trust", {}) if isinstance(data, dict) else {}
    raw = section.get("paths", []) if isinstance(section, dict) else []
    if isinstance(raw, str):            # a single path instead of a list
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [_resolve(entry) for entry in raw if isinstance(entry, str) and entry.strip()]


def is_trusted(directory) -> bool:
    """True if *directory* or one of its parents has been confirmed."""
    target = _resolve(directory)
    for trusted in trusted_paths():
        if target == trusted or trusted in target.parents:
            return True
    return False


# ── Writing ────────────────────────────────────────────────────────────────────

def add_trusted(directory) -> bool:
    """Confirm *directory*. False if it was already covered by an entry."""
    target = _resolve(directory)
    if is_trusted(target):
        return False
    _write(trusted_paths() + [target])
    return True


def remove_trusted(directory) -> bool:
    """Drop the exact entry for *directory*. False if it was not listed."""
    target = _resolve(directory)
    remaining = [p for p in trusted_paths() if p != target]
    if len(remaining) == len(trusted_paths()):
        return False
    _write(remaining)
    return True


def _write(paths: List[Path]) -> None:
    path = trust_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER, "[trust]\n", "paths = [\n"]
    lines += [f'  "{_escape(str(p))}",\n' for p in paths]
    lines.append("]\n")
    path.write_text("".join(lines), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:                     # pragma: no cover - e.g. Windows/FAT
        pass


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


# ── Hints ──────────────────────────────────────────────────────────────────────

def overbroad_warning(directory) -> Optional[str]:
    """A warning when confirming *directory* would cover far more than a project."""
    target = _resolve(directory)
    home = _resolve(Path.home())
    if target == target.parent:
        return "that is the filesystem root — every pixlang.toml on this machine would be trusted"
    if target == home:
        return "that is your home directory — every pixlang.toml below it would be trusted"
    if target in home.parents:
        return f"that is above your home directory ({home}) — far more than one project"
    return None


def trust_hint_for(directory) -> str:
    """The instruction shown when a pixlang.toml asks for trust it has not been given."""
    return (f"its pixlang.toml sets `[plugins] trust_local = true`, but {directory} is not a "
            f"confirmed workspace; run `pixlang trust {directory}` to confirm it, "
            f"or re-run with --trust-plugins for this run only")
