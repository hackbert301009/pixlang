# pixlang/watcher/engine.py
"""
PixLang File Watcher  v0.3

Polls a .pxl pipeline file (and any files it LOADs) for modification.
On change, re-parses, lints, and re-executes the pipeline.

Uses pure polling (no OS-level inotify dependency) so it works
cross-platform without extra packages.

Usage:
    watcher = Watcher(pipeline_path, registry, verbose=True, interval=0.5)
    watcher.run()        # blocks — Ctrl-C to stop
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pixlang.commands.registry import CommandRegistry

# ANSI colours
_C = "\033[96m"; _G = "\033[92m"; _R = "\033[91m"; _Y = "\033[93m"
_D = "\033[2m";  _B = "\033[1m";  _E = "\033[0m"


class Watcher:
    """
    Poll a pipeline file for changes and re-run it automatically.

    Args:
        pipeline_path: path to the .pxl file
        registry:      CommandRegistry to execute against
        verbose:       pass --verbose to each run
        interval:      polling interval in seconds (default 0.5)
        lint:          run linter before executing (default True)
    """

    def __init__(
        self,
        pipeline_path: Path,
        registry: "CommandRegistry",
        verbose: bool = False,
        interval: float = 0.5,
        lint: bool = True,
    ):
        self.pipeline_path = Path(pipeline_path)
        self.registry      = registry
        self.verbose       = verbose
        self.interval      = interval
        self.lint          = lint

        self._mtimes: Dict[Path, float] = {}
        self._run_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block forever, re-running pipeline on file changes. Ctrl-C to stop."""
        print(f"\n{_C}{_B}  PixLang Watch Mode{_E}")
        print(f"{_D}  Watching: {self.pipeline_path}{_E}")
        print(f"{_D}  Interval: {self.interval}s  —  Ctrl-C to stop{_E}\n")

        # Run once immediately
        self._execute()

        try:
            while True:
                time.sleep(self.interval)
                if self._changed():
                    self._execute()
        except KeyboardInterrupt:
            print(f"\n{_D}  Watch stopped.{_E}\n")

    def run_once(self) -> bool:
        """Run the pipeline once. Returns True on success."""
        return self._execute()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _execute(self) -> bool:
        from pixlang.parser import parse
        from pixlang.executor import Executor
        from pixlang.linter import Linter
        from pixlang.plugins import PluginLoader

        self._run_count += 1
        ts = time.strftime("%H:%M:%S")
        print(f"  {_D}[{ts}]{_E}  {_B}Run #{self._run_count}{_E}  "
              f"{_D}{self.pipeline_path.name}{_E}")

        if not self.pipeline_path.exists():
            print(f"  {_R}✗  File not found: {self.pipeline_path}{_E}\n")
            return False

        source = self.pipeline_path.read_text()

        # Parse
        try:
            pipeline = parse(source)
        except SyntaxError as e:
            print(f"  {_R}✗  Syntax error: {e}{_E}\n")
            return False

        # Lint
        if self.lint:
            linter  = Linter(self.registry)
            diags   = linter.lint(pipeline)
            errors  = [d for d in diags if d.severity.value == "error"]
            warns   = [d for d in diags if d.severity.value == "warning"]
            if diags:
                for d in diags:
                    colour = _R if d.severity.value == "error" else _Y
                    print(f"  {colour}{d}{_E}")
            if errors:
                print(f"  {_R}✗  {len(errors)} lint error(s) — pipeline not run.{_E}\n")
                return False
            elif warns:
                print(f"  {_Y}⚠  {len(warns)} warning(s) — running anyway.{_E}")

        # Execute
        loader   = PluginLoader(self.registry)
        executor = Executor(
            registry=self.registry,
            verbose=self.verbose,
            plugin_loader=loader,
            pipeline_path=self.pipeline_path,
        )

        t0 = time.perf_counter()
        try:
            executor.run(pipeline)
        except Exception as e:
            print(f"  {_R}✗  {e}{_E}\n")
            if self.verbose:
                traceback.print_exc()
            return False

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  {_G}✓  Done in {elapsed:.0f} ms{_E}\n")

        # Update mtimes after a successful run
        self._snapshot_mtimes(source)
        return True

    def _snapshot_mtimes(self, source: str) -> None:
        """Record current mtime of the pipeline file and any LOAD'd images."""
        self._mtimes[self.pipeline_path] = self.pipeline_path.stat().st_mtime
        for img_path in self._extract_load_paths(source):
            p = Path(img_path)
            if p.exists():
                self._mtimes[p] = p.stat().st_mtime

    def _changed(self) -> bool:
        """Return True if any watched file has been modified since last run."""
        for path, last_mtime in list(self._mtimes.items()):
            try:
                if path.stat().st_mtime != last_mtime:
                    return True
            except FileNotFoundError:
                return True
        # Also detect if the pipeline file itself was never tracked
        if self.pipeline_path not in self._mtimes:
            return True
        return False

    @staticmethod
    def _extract_load_paths(source: str) -> list:
        """Quick regex extract of LOAD argument paths for watch tracking."""
        import re
        return re.findall(r'LOAD\s+"([^"]+)"', source)
