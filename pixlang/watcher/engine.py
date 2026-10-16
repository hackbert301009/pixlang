# pixlang/watcher/engine.py
"""
PixLang File Watcher  v0.5

Polls a .pxl pipeline file and everything it depends on for modification.
On change, re-parses, lints, and re-executes the pipeline.

Watched files:
  - the pipeline itself
  - its local plugin file ``<stem>.plugins.py`` (also when it appears later)
  - INCLUDE / IMPORT / FROM "..." files, recursively, resolved relative to
    the including file
  - LOAD "..." images (relative to the CWD and to the pipeline directory)

The file state is recorded *before* every run, so a failing run does not
re-trigger every poll interval; only a real change does. One PluginLoader is
kept for the whole session, so plugins are not re-registered on each run
(changed plugin files are reloaded in place).

Uses pure polling (no OS-level inotify dependency) so it works
cross-platform without extra packages.

Usage:
    watcher = Watcher(pipeline_path, registry, verbose=True, interval=0.5)
    watcher.run()        # blocks — Ctrl-C to stop
"""
from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from pixlang._style import Style

if TYPE_CHECKING:
    from pixlang.commands.registry import CommandRegistry
    from pixlang.plugins.loader import PluginLoader

_DEP_RE  = re.compile(r'\b(?:INCLUDE|IMPORT|FROM)\s+"([^"]+)"')
_LOAD_RE = re.compile(r'\b(?:LOAD|LOAD_RAW)\s+"([^"]+)"')

FileState = Optional[Tuple[int, int]]   # (mtime_ns, size) or None if missing


def strip_comment(line: str) -> str:
    """Remove a trailing ``# comment`` that is not inside a string literal."""
    in_str = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_str = not in_str
        elif ch == "#" and not in_str:
            return line[:i]
    return line


def scan_dependencies(path: Path, _seen: Optional[Set[Path]] = None) -> List[Path]:
    """Recursively find INCLUDE/IMPORT/FROM "..." files of *path* (resolved)."""
    seen = _seen if _seen is not None else set()
    out: List[Path] = []
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        for rel in _DEP_RE.findall(strip_comment(line)):
            dep = (path.parent / rel).resolve()
            if dep in seen:
                continue
            seen.add(dep)
            out.append(dep)
            if dep.exists():
                out.extend(scan_dependencies(dep, seen))
    return out


def scan_loads(path: Path) -> List[Path]:
    try:
        text = path.read_text()
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        for rel in _LOAD_RE.findall(strip_comment(line)):
            p = Path(rel)
            found.append(p if p.is_absolute() else Path.cwd() / p)
            if not p.is_absolute():
                found.append(path.parent / p)
    return found


def _state(path: Path) -> FileState:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


class Watcher:
    """
    Poll a pipeline file for changes and re-run it automatically.

    Args:
        pipeline_path:  path to the .pxl file
        registry:       CommandRegistry to execute against
        verbose:        pass --verbose to each run
        interval:       polling interval in seconds (default 0.5)
        lint:           run linter before executing (default True)
        plugin_loader:  loader reused across runs (default: a new one honouring
                        trust_plugins); pass ``False`` to disable plugins
        trust_plugins:  allow local/directory plugins when creating the loader
        variables:      variables injected before each run (config / --var)
    """

    def __init__(
        self,
        pipeline_path: Path,
        registry: CommandRegistry,
        verbose: bool = False,
        interval: float = 0.5,
        lint: bool = True,
        plugin_loader: Optional[PluginLoader] = None,
        trust_plugins: bool = False,
        variables: Optional[dict] = None,
        stream=None,
    ):
        self.pipeline_path = Path(pipeline_path)
        self.registry      = registry
        self.verbose       = verbose
        self.interval      = interval
        self.lint          = lint
        self.variables     = dict(variables or {})
        self.stream        = stream
        if plugin_loader is None:
            from pixlang.plugins import PluginLoader
            plugin_loader = PluginLoader(registry, trust_local=trust_plugins)
        self.plugin_loader = plugin_loader or None

        self._mtimes: Dict[Path, FileState] = {}
        self._run_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block forever, re-running pipeline on file changes. Ctrl-C to stop."""
        st = self._style
        self._print(f"\n{st.cyan}{st.bold}  PixLang Watch Mode{st.reset}")
        self._print(f"{st.dim}  Watching: {self.pipeline_path}{st.reset}")
        self._print(f"{st.dim}  Interval: {self.interval}s  —  Ctrl-C to stop{st.reset}\n")

        self._execute()
        try:
            while True:
                time.sleep(self.interval)
                if self._changed():
                    self._execute()
        except KeyboardInterrupt:
            self._print(f"\n{st.dim}  Watch stopped.{st.reset}\n")

    def run_once(self) -> bool:
        """Run the pipeline once. Returns True on success."""
        return self._execute()

    @property
    def watched_files(self) -> List[Path]:
        return list(self._mtimes)

    # ── Internals ─────────────────────────────────────────────────────────────

    @property
    def _style(self) -> Style:
        return Style(self.stream if self.stream is not None else sys.stdout)

    def _print(self, msg: str = "") -> None:
        print(msg, file=self.stream if self.stream is not None else sys.stdout)

    def _execute(self) -> bool:
        from pixlang.executor import Executor
        from pixlang.linter import Linter
        from pixlang.parser import parse

        st = self._style
        self._run_count += 1
        ts = time.strftime("%H:%M:%S")
        self._print(f"  {st.dim}[{ts}]{st.reset}  {st.bold}Run #{self._run_count}{st.reset}  "
                    f"{st.dim}{self.pipeline_path.name}{st.reset}")

        # Record state first: a failed run waits for the next real change.
        self._snapshot_mtimes()

        if not self.pipeline_path.exists():
            self._print(f"  {st.red}✗  File not found: {self.pipeline_path}{st.reset}\n")
            return False

        source = self.pipeline_path.read_text()

        try:
            pipeline = parse(source)
        except (SyntaxError, RecursionError) as e:
            self._print(f"  {st.red}✗  {format_syntax_error(e, self.pipeline_path)}{st.reset}\n")
            return False

        if self.lint:
            diags  = Linter(self.registry).lint(pipeline)
            errors = [d for d in diags if d.severity.value == "error"]
            warns  = [d for d in diags if d.severity.value == "warning"]
            for d in diags:
                colour = st.red if d.severity.value == "error" else st.yellow
                self._print(f"  {colour}{d}{st.reset}")
            if errors:
                self._print(f"  {st.red}✗  {len(errors)} lint error(s) — pipeline not run.{st.reset}\n")
                return False
            if warns:
                self._print(f"  {st.yellow}⚠  {len(warns)} warning(s) — running anyway.{st.reset}")

        t0 = time.perf_counter()
        try:
            from pixlang.batch.engine import BatchRunner, find_glob_commands
            if find_glob_commands(pipeline):
                res = BatchRunner(pipeline, self.registry, verbose=self.verbose,
                                  plugin_loader=self.plugin_loader,
                                  pipeline_path=self.pipeline_path,
                                  variables=self.variables, stream=self.stream).run()
                if res["failed"]:
                    raise RuntimeError(f"{res['failed']} of {res['total']} file(s) failed")
            else:
                executor = Executor(
                    registry=self.registry, verbose=self.verbose,
                    plugin_loader=self.plugin_loader, pipeline_path=self.pipeline_path,
                )
                executor.context["vars"].update(self.variables)
                executor.run(pipeline)
        except Exception as e:
            self._print(f"  {st.red}✗  {e}{st.reset}\n")
            if self.verbose:
                traceback.print_exc()
            self._snapshot_mtimes()   # plugin files may have been discovered
            return False

        elapsed = (time.perf_counter() - t0) * 1000
        self._print(f"  {st.green}✓  Done in {elapsed:.0f} ms{st.reset}\n")
        self._snapshot_mtimes()
        return True

    def _watch_list(self) -> List[Path]:
        from pixlang.plugins.loader import local_plugin_path
        files = [self.pipeline_path, local_plugin_path(self.pipeline_path)]
        if self.pipeline_path.exists():
            files.extend(scan_dependencies(self.pipeline_path.resolve()))
            files.extend(scan_loads(self.pipeline_path))
        if self.plugin_loader is not None:
            files.extend(getattr(self.plugin_loader, "watched_files", []))
        uniq: List[Path] = []
        seen = set()
        for f in files:
            key = str(Path(f).absolute())
            if key not in seen:
                seen.add(key)
                uniq.append(Path(f))
        return uniq

    def _snapshot_mtimes(self, source: Optional[str] = None) -> None:
        """Record (mtime, size) of every watched file (None if missing)."""
        self._mtimes = {p: _state(p) for p in self._watch_list()}

    def _changed(self) -> bool:
        """Return True if any watched file changed, appeared or disappeared."""
        if self.pipeline_path not in self._mtimes:
            return True
        for path, last in list(self._mtimes.items()):
            if _state(path) != last:
                return True
        # New dependencies (e.g. an INCLUDE added in an unchanged-size edit)
        return any(p not in self._mtimes for p in self._watch_list())

    @staticmethod
    def _extract_load_paths(source: str) -> list:
        """Regex extract of LOAD argument paths (comments ignored)."""
        out = []
        for line in source.splitlines():
            out.extend(_LOAD_RE.findall(strip_comment(line)))
        return out


def format_syntax_error(e: BaseException, path: Optional[Path] = None) -> str:
    """``file:line:col: message`` using SyntaxError attributes when present."""
    if isinstance(e, RecursionError):
        where = f"{path}: " if path else ""
        return f"{where}pipeline is nested too deeply (recursion limit reached)"
    filename = getattr(e, "filename", None) or (str(path) if path else None)
    lineno = getattr(e, "lineno", None)
    offset = getattr(e, "offset", None)
    msg = getattr(e, "msg", None) or str(e)
    loc = ":".join(str(x) for x in (filename, lineno, offset) if x)
    return f"{loc}: {msg}" if loc else str(msg)
