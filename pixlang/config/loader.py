# pixlang/config/loader.py
"""
PixLang Project Configuration  v0.4
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
    """Parse pixlang.toml using stdlib tomllib (Python 3.11+) or fallback."""
    try:
        if sys.version_info >= (3, 11):
            import tomllib
            with open(path, "rb") as f:
                data = tomllib.load(f)
        else:
            # Fallback: minimal TOML reader for simple key=value files
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
    cfg.lint_ignore = [str(c) for c in data.get("lint", {}).get("ignore", [])]
    cfg.output_dir  = str(data.get("batch", {}).get("output_dir", cfg.output_dir))

    return cfg


def _simple_toml_parse(path: Path) -> dict:
    """
    Minimal TOML parser for Python < 3.11.
    Handles [sections], key = value, and quoted strings.
    Does NOT handle arrays, inline tables, or multi-line strings.
    """
    import re
    data:    dict = {}
    section: dict = data
    sec_key: str  = ""

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Section header
        m = re.match(r'^\[([^\]]+)\]$', line)
        if m:
            keys = m.group(1).split(".")
            section = data
            for k in keys:
                section = section.setdefault(k, {})
            continue
        # Key = value
        m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if m:
            key, raw = m.group(1), m.group(2).strip()
            if raw.startswith('"') and raw.endswith('"'):
                val: Any = raw[1:-1]
            elif raw.lower() == "true":
                val = True
            elif raw.lower() == "false":
                val = False
            else:
                try:
                    val = int(raw)
                except ValueError:
                    try:
                        val = float(raw)
                    except ValueError:
                        val = raw
            section[key] = val

    return data
