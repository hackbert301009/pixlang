# pixlang/plugins/loader.py
"""
PixLang Plugin Loader  (v0.2)
═════════════════════════════

Three discovery mechanisms, tried in order:

1. **Entry-points**  (installed packages)
   Any Python package that declares:
       [project.entry-points."pixlang.commands"]
       my_plugin = "my_package.pixlang_plugin:register"
   will be discovered automatically when PixLang starts.
   The `register(registry)` function receives the live registry instance.

2. **Local plugin file**  (single .py alongside the pipeline)
   If a file named `<pipeline_stem>.plugins.py` exists next to the .pxl file,
   it is loaded automatically.  The file must expose a top-level
   `register(registry)` function.

3. **Plugin directory**  ($PIXLANG_PLUGIN_DIR or ~/.pixlang/plugins/)
   Every *.py file in this directory is imported. Each must expose `register`.

The loader records which source loaded each plugin and surfaces that
in `pixlang plugins` CLI output.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pixlang.commands.registry import CommandRegistry

ENTRY_POINT_GROUP = "pixlang.commands"
PLUGIN_DIR_ENV    = "PIXLANG_PLUGIN_DIR"
DEFAULT_PLUGIN_DIR = Path.home() / ".pixlang" / "plugins"


# ── Plugin manifest ────────────────────────────────────────────────────────────

@dataclass
class PluginManifest:
    """Describes a single loaded plugin."""
    name:          str           # human-readable plugin name
    source_type:   str           # "entrypoint" | "local" | "directory"
    source_path:   str           # package name or file path
    commands:      List[str] = field(default_factory=list)   # command names it registered
    error:         Optional[str] = None                       # non-None if load failed

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Loader ─────────────────────────────────────────────────────────────────────

class PluginLoader:
    """
    Discovers and loads PixLang plugins into a CommandRegistry.

    Usage:
        loader = PluginLoader(registry)
        loader.load_entrypoints()
        loader.load_local(pipeline_path)
        loader.load_directory()          # reads $PIXLANG_PLUGIN_DIR
        print(loader.manifests)          # inspect what was loaded
    """

    def __init__(self, registry: "CommandRegistry"):
        self.registry = registry
        self.manifests: List[PluginManifest] = []

    # ── Discovery methods ─────────────────────────────────────────────────────

    def load_entrypoints(self) -> List[PluginManifest]:
        """
        Discover all installed packages that expose the 'pixlang.commands'
        entry-point group and call their register() function.
        """
        loaded = []
        try:
            eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception:
            return loaded

        for ep in eps:
            manifest = PluginManifest(
                name=ep.name,
                source_type="entrypoint",
                source_path=ep.value,
            )
            try:
                register_fn = ep.load()
                self._call_register(register_fn, manifest)
            except Exception as exc:
                manifest.error = str(exc)

            self.manifests.append(manifest)
            loaded.append(manifest)

        return loaded

    def load_local(self, pipeline_path: Path) -> Optional[PluginManifest]:
        """
        Look for <pipeline_stem>.plugins.py alongside the pipeline file.
        Example: if running `my_pipeline.pxl`, loads `my_pipeline.plugins.py`.
        """
        candidate = pipeline_path.parent / f"{pipeline_path.stem}.plugins.py"
        if not candidate.exists():
            return None

        manifest = PluginManifest(
            name=candidate.stem,
            source_type="local",
            source_path=str(candidate),
        )
        try:
            register_fn = _load_module_from_path(candidate).register
            self._call_register(register_fn, manifest)
        except Exception as exc:
            manifest.error = str(exc)

        self.manifests.append(manifest)
        return manifest

    def load_directory(self, directory: Optional[Path] = None) -> List[PluginManifest]:
        """
        Load every *.py file from $PIXLANG_PLUGIN_DIR (or ~/.pixlang/plugins/).
        """
        plugin_dir = directory or _resolve_plugin_dir()
        if not plugin_dir.is_dir():
            return []

        loaded = []
        for py_file in sorted(plugin_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            manifest = PluginManifest(
                name=py_file.stem,
                source_type="directory",
                source_path=str(py_file),
            )
            try:
                mod = _load_module_from_path(py_file)
                if not hasattr(mod, "register"):
                    raise AttributeError(
                        f"Plugin file '{py_file.name}' has no top-level register(registry) function."
                    )
                self._call_register(mod.register, manifest)
            except Exception as exc:
                manifest.error = str(exc)

            self.manifests.append(manifest)
            loaded.append(manifest)

        return loaded

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _call_register(self, fn, manifest: PluginManifest):
        """
        Call register(registry), then diff the registry to find new commands.
        """
        before = set(self.registry.list_commands())
        fn(self.registry)
        after  = set(self.registry.list_commands())
        manifest.commands = sorted(after - before)

    def summary(self) -> str:
        lines = []
        for m in self.manifests:
            status = "✓" if m.ok else "✗"
            cmds   = ", ".join(m.commands) if m.commands else "—"
            lines.append(
                f"  {status}  [{m.source_type:<11}] {m.name:<28} commands: {cmds}"
            )
            if not m.ok:
                lines.append(f"       ERROR: {m.error}")
        return "\n".join(lines) if lines else "  (no plugins loaded)"


# ── Module utilities ───────────────────────────────────────────────────────────

def _load_module_from_path(path: Path):
    """Import a .py file as a module by absolute path."""
    spec = importlib.util.spec_from_file_location(
        f"pixlang_plugin_{path.stem}", str(path.resolve())
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_plugin_dir() -> Path:
    env = os.environ.get(PIXLANG_PLUGIN_DIR_ENV := PLUGIN_DIR_ENV)
    return Path(env) if env else DEFAULT_PLUGIN_DIR
