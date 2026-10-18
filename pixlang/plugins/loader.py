# pixlang/plugins/loader.py
"""
PixLang Plugin Loader  (v0.5)
═════════════════════════════

Three discovery mechanisms:

1. **Entry-points**  (installed packages) — always loaded
   Any Python package that declares:
       [project.entry-points."pixlang.commands"]
       my_plugin = "my_package.pixlang_plugin:register"
   is discovered automatically. Installing a package is an explicit act of
   trust, so these load without further opt-in.

2. **Local plugin file**  (``<pipeline_stem>.plugins.py`` next to the .pxl)
3. **Plugin directory**  ($PIXLANG_PLUGIN_DIR or ~/.pixlang/plugins/)

   Local and directory plugins are arbitrary Python code that would run just
   by *running a pipeline* (e.g. one you downloaded). They are therefore
   **opt-in**: pass ``--trust-plugins`` on the command line, or set

       [plugins]
       trust_local = true

   in pixlang.toml **and** confirm that directory once with ``pixlang trust``.
   The second step matters: a downloaded bundle can ship its own pixlang.toml,
   so the flag alone would be the attacker's to set (see ``trust.py``). When
   such a file exists but is not trusted, one hint line is printed to stderr.
   (Calling ``load_directory(path)`` with an explicit path is an explicit
   request and always loads.)

Loading is idempotent: loading the same file again is a no-op unless the
file changed on disk, in which case its commands are unregistered and the
file is re-executed (used by watch mode). If ``register()`` fails half-way,
everything it registered is rolled back. Load errors are always reported on
stderr, not only in verbose mode.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple

if TYPE_CHECKING:
    from pixlang.commands.registry import CommandRegistry

ENTRY_POINT_GROUP  = "pixlang.commands"
PLUGIN_DIR_ENV     = "PIXLANG_PLUGIN_DIR"
DEFAULT_PLUGIN_DIR = Path.home() / ".pixlang" / "plugins"
TRUST_HINT = ("re-run with --trust-plugins, or set `[plugins] trust_local = true` in "
              "pixlang.toml and confirm the directory once with `pixlang trust`")


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
        loader = PluginLoader(registry, trust_local=args.trust_plugins)
        loader.load_entrypoints()
        loader.load_local(pipeline_path)   # only if trusted
        loader.load_directory()            # only if trusted
        print(loader.manifests)

    *trust_hint* replaces the generic instruction in the skip message when the
    caller knows a more specific one -- e.g. that a pixlang.toml did ask for
    trust and only the confirmation is missing.
    """

    def __init__(self, registry: CommandRegistry, trust_local: bool = False,
                 stream: Optional[TextIO] = None, trust_hint: Optional[str] = None):
        self.registry = registry
        self.trust_local = bool(trust_local)
        self.stream = stream
        self.trust_hint = trust_hint or TRUST_HINT
        self.manifests: List[PluginManifest] = []
        self.skipped: List[Path] = []               # untrusted files that were not loaded
        self._loaded: Dict[str, Tuple[Optional[float], PluginManifest]] = {}
        self._hinted: set = set()

    # ── Discovery methods ─────────────────────────────────────────────────────

    def load_entrypoints(self) -> List[PluginManifest]:
        """Load every installed package exposing the 'pixlang.commands' entry-point group."""
        loaded = []
        try:
            eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:   # Python < 3.10 API
            eps = importlib.metadata.entry_points().get(ENTRY_POINT_GROUP, [])
        except Exception:
            return loaded

        for ep in eps:
            tag = f"entrypoint:{ep.name}"
            if tag in self._loaded:
                loaded.append(self._loaded[tag][1])
                continue
            manifest = PluginManifest(name=ep.name, source_type="entrypoint",
                                      source_path=ep.value)
            try:
                register_fn = ep.load()
            except Exception as exc:
                manifest.error = f"{type(exc).__name__}: {exc}"
                self._warn(manifest)
            else:
                self._register(register_fn, manifest, tag)
            self._remember(tag, None, manifest)
            loaded.append(manifest)
        return loaded

    def load_local(self, pipeline_path: Path, trusted: Optional[bool] = None
                   ) -> Optional[PluginManifest]:
        """Load ``<pipeline_stem>.plugins.py`` next to *pipeline_path* if trusted."""
        candidate = local_plugin_path(Path(pipeline_path))
        if not candidate.exists():
            return None
        if not (self.trust_local if trusted is None else trusted):
            self._hint(candidate, "local plugin")
            return None
        return self._load_file(candidate, "local", candidate.name[: -len(".py")])

    def load_directory(self, directory: Optional[Path] = None) -> List[PluginManifest]:
        """Load every ``*.py`` in *directory*, or (if trusted) in the user plugin dir."""
        explicit = directory is not None
        plugin_dir = Path(directory) if explicit else _resolve_plugin_dir()
        if not plugin_dir.is_dir():
            return []
        files = [p for p in sorted(plugin_dir.glob("*.py")) if not p.name.startswith("_")]
        if not explicit and not self.trust_local:
            for f in files:
                self._hint(f, "plugin")
            return []
        return [self._load_file(f, "directory", f.stem) for f in files]

    def reload_changed(self) -> List[PluginManifest]:
        """Re-execute any loaded plugin file whose mtime changed."""
        out = []
        for mtime, manifest in list(self._loaded.values()):
            if manifest.source_type == "entrypoint":
                continue
            path = Path(manifest.source_path)
            if path.exists() and _mtime(path) != mtime:
                out.append(self._load_file(path, manifest.source_type, manifest.name))
        return out

    def unload_all(self) -> None:
        """Unregister every command this loader registered."""
        for tag in list(self._loaded):
            self.registry.unregister_source(tag)
        self._loaded.clear()
        self.manifests.clear()

    @property
    def watched_files(self) -> List[Path]:
        return [Path(m.source_path) for m in self.manifests if m.source_type != "entrypoint"]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_file(self, path: Path, source_type: str, name: str) -> PluginManifest:
        path = path.resolve()
        tag = f"{source_type}:{path}"
        mtime = _mtime(path)
        if tag in self._loaded:
            old_mtime, old = self._loaded[tag]
            if old_mtime == mtime:
                return old
            self.registry.unregister_source(tag)        # reload a changed file
        manifest = PluginManifest(name=name, source_type=source_type, source_path=str(path))
        try:
            module = _load_module_from_path(path)
            register_fn = getattr(module, "register", None)
            if not callable(register_fn):
                raise AttributeError(
                    f"Plugin file '{path.name}' has no top-level register(registry) function."
                )
        except Exception as exc:
            manifest.error = f"{type(exc).__name__}: {exc}"
            self._warn(manifest)
        else:
            self._register(register_fn, manifest, tag)
        self._remember(tag, mtime, manifest)
        return manifest

    def _register(self, fn, manifest: PluginManifest, tag: str) -> None:
        """Call register(registry) attributing commands to *tag*; roll back on failure."""
        try:
            with self.registry.origin(tag):
                fn(self.registry)
        except Exception as exc:
            rolled = self.registry.unregister_source(tag)
            manifest.error = f"{type(exc).__name__}: {exc}"
            if rolled:
                manifest.error += f" (rolled back: {', '.join(rolled)})"
            self._warn(manifest)
            return
        manifest.commands = sorted(
            info.name for info in self.registry.list_info() if info.origin == tag
        )

    def _remember(self, tag: str, mtime: Optional[float], manifest: PluginManifest) -> None:
        if tag in self._loaded:
            old = self._loaded[tag][1]
            self.manifests = [m for m in self.manifests if m is not old]
        self._loaded[tag] = (mtime, manifest)
        self.manifests.append(manifest)

    def _out(self) -> TextIO:
        return self.stream if self.stream is not None else sys.stderr

    def _warn(self, manifest: PluginManifest) -> None:
        print(f"pixlang: warning: plugin '{manifest.name}' ({manifest.source_path}) "
              f"failed to load: {manifest.error}", file=self._out())

    def _hint(self, path: Path, what: str) -> None:
        key = str(path.resolve())
        if key not in self.skipped_keys:
            self.skipped.append(path)
        if key in self._hinted:
            return
        self._hinted.add(key)
        print(f"pixlang: hint: {what} '{path}' was not loaded (untrusted code); "
              f"{self.trust_hint}.", file=self._out())

    @property
    def skipped_keys(self) -> set:
        return {str(p.resolve()) for p in self.skipped}

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


# ── Trust configuration ────────────────────────────────────────────────────────

def local_plugin_path(pipeline_path: Path) -> Path:
    return pipeline_path.parent / f"{pipeline_path.stem}.plugins.py"


def config_trust_request(start: Optional[Path] = None) -> Optional[Path]:
    """
    The nearest pixlang.toml that sets ``[plugins] trust_local``, or None.

    Finding one says only that the *project* asks for local-plugin trust. It
    does not grant it -- the project file may have come with the pipeline.
    """
    from pixlang.config.loader import _find_config
    start = Path(start) if start is not None else Path.cwd()
    try:
        cfg_file = _find_config(start.resolve())
    except Exception:
        return None
    if cfg_file is None:
        return None
    data = _read_toml(cfg_file)
    section = data.get("plugins", {}) if isinstance(data, dict) else {}
    value = section.get("trust_local", False) if isinstance(section, dict) else False
    if isinstance(value, str):
        value = value.strip().lower() in ("true", "yes", "1", "on")
    return cfg_file if bool(value) else None


def trust_from_config(start: Optional[Path] = None) -> bool:
    """
    True if a pixlang.toml asks for local-plugin trust **and** its directory
    has been confirmed with ``pixlang trust`` (see ``pixlang.plugins.trust``).
    """
    from pixlang.plugins.trust import is_trusted
    cfg_file = config_trust_request(start)
    return cfg_file is not None and is_trusted(cfg_file.parent)


def _read_toml(path: Path) -> dict:
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:   # pragma: no cover - exercised on Python < 3.11 only
            try:
                import tomli as tomllib
            except ImportError:
                from pixlang.config.loader import _simple_toml_parse
                return _simple_toml_parse(path)
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


# ── Module utilities ───────────────────────────────────────────────────────────

def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _module_name(path: Path) -> str:
    """A module name unique per absolute path (two 'utils.py' never collide)."""
    resolved = str(path.resolve())
    stem = re.sub(r"\W", "_", path.stem)
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]
    return f"pixlang_plugin_{stem}_{digest}"


def _load_module_from_path(path: Path):
    """Import a .py file as a module by absolute path (fresh execution each call)."""
    name = _module_name(path)
    spec = importlib.util.spec_from_file_location(name, str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plugin file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _resolve_plugin_dir() -> Path:
    env = os.environ.get(PLUGIN_DIR_ENV)
    return Path(env) if env else DEFAULT_PLUGIN_DIR
