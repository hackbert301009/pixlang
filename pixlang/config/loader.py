# pixlang/config/loader.py
"""
PixLang Project Configuration  v0.5
═════════════════════════════════════

Reads a `pixlang.toml` file from the pipeline directory (or any parent)
and applies project-level defaults to every run.

Example pixlang.toml:
─────────────────────
[pixlang]
version = "0.4"

[defaults]
verbose  = false
plugins  = true
lint     = true

[variables]
# Pre-set variables injected before the pipeline runs
width    = 640
height   = 480
blur_k   = 5
mode     = "production"

[lint]
# Rules to silence (by code)
ignore   = ["PX006", "PX009"]

[batch]
output_dir = "output"

[link]
# Extra directories searched for IMPORT / FROM ... IMPORT modules
# (relative to this file)
search_paths = ["lib"]

[run]
# Safety cap for WHILE loops
max_loop_iterations = 1000000
─────────────────────

Discovery order (first found wins):
  1. Directory of the .pxl file being run
  2. Parent directories up to filesystem root
  3. Current working directory
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PixLangConfig:
    """Parsed pixlang.toml settings."""
    version:     str            = "0.4"
    verbose:     bool           = False
    plugins:     bool           = True
    lint:        bool           = True
    variables:   Dict[str, Any] = field(default_factory=dict)
    lint_ignore: List[str]      = field(default_factory=list)
    output_dir:  str            = "output"
    source_path: Optional[Path] = None    # where the config was found
    link_search_paths:   List[Path] = field(default_factory=list)
    max_loop_iterations: int        = 1_000_000

    @classmethod
    def default(cls) -> "PixLangConfig":
        return cls()


def load_config(start_path: Optional[Path] = None) -> PixLangConfig:
    """
    Walk upward from start_path looking for pixlang.toml.
    Returns PixLangConfig.default() if none is found.
    """
    config_file = _find_config(start_path or Path.cwd())
    if config_file is None:
        return PixLangConfig.default()
    return _parse_toml(config_file)


def _find_config(start: Path) -> Optional[Path]:
    """Search start and each parent directory for pixlang.toml."""
    candidates = [start] if start.is_dir() else [start.parent]
    # walk up
    p = candidates[0]
    for _ in range(20):          # max 20 levels
        candidate = p / "pixlang.toml"
        if candidate.exists():
            return candidate
        parent = p.parent
        if parent == p:
            break
        p = parent
    return None


def _parse_toml(path: Path) -> PixLangConfig:
    """Parse pixlang.toml using tomllib (3.11+), tomli, or a minimal fallback."""
    try:
        toml_mod = None
        if sys.version_info >= (3, 11):
            import tomllib as toml_mod
        else:
            try:
                import tomli as toml_mod          # type: ignore
            except ImportError:
                toml_mod = None
        if toml_mod is not None:
            with open(path, "rb") as f:
                data = toml_mod.load(f)
        else:
            data = _simple_toml_parse(path)
    except Exception as e:
        # Never crash because of a config file
        import warnings
        warnings.warn(f"[PixLang] Could not parse pixlang.toml: {e}")
        return PixLangConfig.default()

    cfg = PixLangConfig(source_path=path)

    defaults  = data.get("defaults", {})
    cfg.verbose = bool(defaults.get("verbose",  cfg.verbose))
    cfg.plugins = bool(defaults.get("plugins",  cfg.plugins))
    cfg.lint    = bool(defaults.get("lint",     cfg.lint))

    cfg.variables   = {str(k): v for k, v in data.get("variables", {}).items()}
    ignore = data.get("lint", {}).get("ignore", [])
    if isinstance(ignore, str):
        ignore = [ignore]
    cfg.lint_ignore = [str(c) for c in ignore]

    search = data.get("link", {}).get("search_paths", [])
    if isinstance(search, str):
        search = [search]
    cfg.link_search_paths = [(path.parent / str(p)).resolve() for p in search]

    try:
        cfg.max_loop_iterations = int(
            data.get("run", {}).get("max_loop_iterations", cfg.max_loop_iterations))
    except (TypeError, ValueError):
        pass
    cfg.output_dir  = str(data.get("batch", {}).get("output_dir", cfg.output_dir))

    return cfg


def _simple_toml_parse(path: Path) -> dict:
    """
    Minimal TOML reader for Python < 3.11 without tomli.

    Handles [sections] / [dotted.sections], key = value with basic strings
    (escapes), literal 'strings', integers (incl. 0x / 0b / underscores),
    floats, booleans, single-line arrays and trailing # comments.
    Raises ValueError on anything it does not understand.
    """
    data:    dict = {}
    section: dict = data

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = data
            for k in line[1:-1].strip().split("."):
                section = section.setdefault(k.strip().strip('"'), {})
            continue
        if "=" not in line:
            raise ValueError(f"line {lineno}: expected 'key = value'")
        key, raw = line.split("=", 1)
        key = key.strip().strip('"')
        if not key:
            raise ValueError(f"line {lineno}: empty key")
        val, rest = _parse_value(raw.strip(), lineno)
        if rest.strip():
            raise ValueError(f"line {lineno}: unexpected text after value: {rest.strip()!r}")
        section[key] = val
    return data


def _strip_comment(line: str) -> str:
    """Remove a trailing # comment that is not inside a string."""
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == "#":
            return line[:i]
        i += 1
    return line


def _parse_value(raw: str, lineno: int):
    """Parse one TOML value from the start of *raw*; return (value, rest)."""
    import re
    if not raw:
        raise ValueError(f"line {lineno}: missing value")
    if raw[0] == '"':
        out, i = [], 1
        esc = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}
        while i < len(raw):
            ch = raw[i]
            if ch == "\\" and i + 1 < len(raw):
                out.append(esc.get(raw[i + 1], raw[i + 1]))
                i += 2
                continue
            if ch == '"':
                return "".join(out), raw[i + 1:]
            out.append(ch)
            i += 1
        raise ValueError(f"line {lineno}: unterminated string")
    if raw[0] == "'":
        end = raw.find("'", 1)
        if end < 0:
            raise ValueError(f"line {lineno}: unterminated string")
        return raw[1:end], raw[end + 1:]
    if raw[0] == "[":
        items, rest = [], raw[1:].lstrip()
        while True:
            if rest.startswith("]"):
                return items, rest[1:]
            val, rest = _parse_value(rest, lineno)
            items.append(val)
            rest = rest.lstrip()
            if rest.startswith(","):
                rest = rest[1:].lstrip()
            elif not rest.startswith("]"):
                raise ValueError(f"line {lineno}: expected ',' or ']' in array")
    m = re.match(r"(true|false)\b", raw)
    if m:
        return m.group(1) == "true", raw[m.end():]
    m = re.match(r"[+-]?(0x[0-9A-Fa-f_]+|0b[01_]+|0o[0-7_]+)", raw)
    if m:
        return int(m.group(0).replace("_", ""), 0), raw[m.end():]
    m = re.match(r"[+-]?\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d+)?", raw)
    if m:
        text = m.group(0).replace("_", "")
        if m.group(1) or m.group(2):
            return float(text), raw[m.end():]
        return int(text), raw[m.end():]
    raise ValueError(f"line {lineno}: cannot parse value {raw!r}")
