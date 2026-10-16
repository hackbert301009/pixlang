#!/usr/bin/env python3
# pixlang/cli.py  v0.5
"""
PixLang CLI

Commands:
    run      pipeline.pxl [--verbose] [--no-plugins] [--trust-plugins] [--var k=v ...]
    batch    pipeline.pxl [--verbose] [--no-plugins] [--trust-plugins] [--var k=v ...]
    lint     pipeline.pxl
    watch    pipeline.pxl [--verbose] [--interval 0.5] [--trust-plugins] [--var k=v ...]
    validate pipeline.pxl
    commands [--source SRC] [--category CAT]
    plugins  [--trust-plugins]
    new      <name>
    editor   [--port 7478] [--no-browser]
    --version

Output conventions: results go to stdout, errors to stderr. ANSI colours are
used only on a TTY and never when NO_COLOR is set; the banner is shown only
for interactive top-level help and `run`.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from pixlang._style import Style, is_tty


def _version() -> str:
    try:
        from pixlang import __version__
        return __version__
    except Exception:   # pragma: no cover
        return "unknown"


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("file")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-plugins", action="store_true", help="Do not load any plugins")
    p.add_argument("--trust-plugins", action="store_true",
                   help="Allow local <stem>.plugins.py and ~/.pixlang/plugins/*.py "
                        "(arbitrary Python code)")
    p.add_argument("--var", action="append", default=[], metavar="KEY=VALUE",
                   help="Set a pipeline variable (repeatable); overrides pixlang.toml")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pixlang", description="PixLang — Computer Vision Pipeline DSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = ap.add_subparsers(dest="command")

    rp = sub.add_parser("run",      help="Execute a .pxl pipeline")
    _add_run_options(rp)

    bp = sub.add_parser("batch",    help="Run a LOAD_GLOB pipeline once per matched file")
    _add_run_options(bp)

    lp = sub.add_parser("lint",     help="Lint a pipeline without running it")
    lp.add_argument("file")

    wp = sub.add_parser("watch",    help="Re-run pipeline on file changes")
    _add_run_options(wp)
    wp.add_argument("--interval", type=float, default=0.5, metavar="SEC")
    wp.add_argument("--no-lint", action="store_true")

    kp = sub.add_parser("link", help="Link a pipeline and report symbols/diagnostics")
    kp.add_argument("file")
    kp.add_argument("--map", action="store_true", help="print the symbol map")

    vp = sub.add_parser("validate", help="Parse-check without executing")
    vp.add_argument("file")

    cp = sub.add_parser("commands", help="List available DSL commands")
    cp.add_argument("--source", metavar="SRC")
    cp.add_argument("--category", metavar="CAT")

    pp = sub.add_parser("plugins", help="Show discovered plugins")
    pp.add_argument("--trust-plugins", action="store_true",
                    help="Also load ~/.pixlang/plugins/*.py")

    np_ = sub.add_parser("new",     help="Scaffold a new pipeline project")
    np_.add_argument("name", help="Project / pipeline name")

    ep = sub.add_parser("editor",  help="Launch the visual editor in a browser")
    ep.add_argument("--port", type=int, default=7478, metavar="PORT",
                    help="Port to listen on (default: 7478)")
    ep.add_argument("--no-browser", action="store_true",
                    help="Do not open browser automatically")
    ep.add_argument("--workspace", metavar="DIR", default=None,
                    help="Directory the editor may read/write (default: current directory)")
    ep.add_argument("--timeout", type=float, default=30.0, metavar="SECONDS",
                    help="Time limit for a single pipeline run (default: 30)")
    ep.add_argument("--no-plugins", action="store_true",
                    help="Do not load plugins inside the editor")

    args = ap.parse_args(argv)
    if args.command is None:
        _print_banner()
        ap.print_help()
        sys.exit(0)

    if args.command == "run":
        _print_banner()

    handlers = {
        "run":      lambda: _cmd_run(args, force_batch=False),
        "batch":    lambda: _cmd_run(args, force_batch=True),
        "lint":     lambda: _cmd_lint(args.file),
        "watch":    lambda: _cmd_watch(args),
        "validate": lambda: _cmd_validate(args.file),
        "link":     lambda: _cmd_link(args.file, args.map),
        "commands": lambda: _cmd_list_commands(args.source, args.category),
        "plugins":  lambda: _cmd_plugins(args.trust_plugins),
        "new":      lambda: _cmd_new(args.name),
        "editor":   lambda: _cmd_editor(args.port, args.no_browser, args.workspace,
                                       args.timeout, args.no_plugins),
    }
    handler = handlers.get(args.command)
    if handler is None:   # a subcommand registered elsewhere
        ap.print_help()
        sys.exit(2)
    handler()


# ── shared option handling ────────────────────────────────────────────────────

def parse_var(spec: str):
    """``key=value`` → (key, typed value): int, float, true/false, else string."""
    if "=" not in spec:
        _error(f"--var expects KEY=VALUE, got {spec!r}")
    key, raw = spec.split("=", 1)
    key = key.strip().lstrip("$")
    if not key.isidentifier():
        _error(f"--var: invalid variable name {key!r}")
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return key, raw[1:-1]
    low = raw.lower()
    if low in ("true", "false"):
        return key, low == "true"
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    return key, raw


def _variables(cfg, var_specs) -> dict:
    variables = dict(getattr(cfg, "variables", {}) or {})
    for spec in var_specs or []:
        k, v = parse_var(spec)
        variables[k] = v
    return variables


def _make_loader(args, path: Path, cfg):
    """PluginLoader honouring --no-plugins, pixlang.toml and --trust-plugins."""
    from pixlang import registry
    from pixlang.plugins import PluginLoader, trust_from_config
    if args.no_plugins or not getattr(cfg, "plugins", True):
        return None
    trust = bool(args.trust_plugins) or trust_from_config(path.parent)
    return PluginLoader(registry, trust_local=trust)


# ── run / batch ───────────────────────────────────────────────────────────────

def _cmd_run(args, force_batch: bool = False):
    from pixlang import Executor, registry
    from pixlang.batch.engine import BatchRunner, find_glob_commands
    from pixlang.config import load_config

    path = _require_file(args.file)
    cfg  = load_config(path)
    verbose = args.verbose or cfg.verbose
    variables = _variables(cfg, args.var)
    loader = _make_loader(args, path, cfg)

    print(f"  Pipeline  : {path.name}")
    if cfg.source_path:
        print(f"  Config    : {cfg.source_path.name}")
    plugins = "off" if loader is None else ("on (trusted)" if loader.trust_local else "on")
    print(f"  Verbose   : {verbose}  |  Plugins: {plugins}")

    pipeline = _parse_or_die(path)
    has_glob = bool(find_glob_commands(pipeline))
    if force_batch and not has_glob:
        _error("`pixlang batch` needs a pipeline with LOAD_GLOB \"pattern\".")

    if has_glob:
        print("  Mode      : batch\n")
        runner = BatchRunner(pipeline, registry, verbose=verbose,
                             plugin_loader=loader, pipeline_path=path,
                             variables=variables)
        try:
            results = runner.run()
        except Exception as e:
            _fail(e, verbose)
        if results["failed"]:
            _error(f"{results['failed']} of {results['total']} file(s) failed.")
        _success(f"Batch complete: {results['ok']}/{results['total']} succeeded.")
        return

    print("  Mode      : single\n")
    if verbose:
        print(f"  {'CMD':<26}{'ARGS':<22}{'SHAPE':<18}{'TIME':>8}")
        print(f"  {chr(0x2500)*76}")

    ex = Executor(registry=registry, verbose=verbose,
                  plugin_loader=loader, pipeline_path=path)
    ex.context["vars"].update(variables)

    try:
        ex.run(pipeline)
    except Exception as e:
        _fail(e, verbose)

    print()
    _success("Pipeline completed successfully.")


def _cmd_lint(filepath):
    from pixlang import registry
    from pixlang.linter import Linter

    S = Style()
    path     = _require_file(filepath)
    pipeline = _parse_or_die(path)
    linter   = Linter(registry)
    diags    = linter.lint(pipeline, path=path)

    if not diags:
        print(f"  {path.name}  ({linter.rule_count} rules checked)\n")
        _success("No issues found.")
        return

    errors = warns = infos = 0
    for d in diags:
        if d.severity.value == "error":
            print(f"  {S.red}{d}{S.reset}")
            errors += 1
        elif d.severity.value == "warning":
            print(f"  {S.yellow}{d}{S.reset}")
            warns += 1
        else:
            print(f"  {S.dim}{d}{S.reset}")
            infos += 1

    print()
    summary = f"  {errors} error(s)  {warns} warning(s)  {infos} info(s)"
    if errors:
        print(f"{S.red}{summary}{S.reset}")
        sys.exit(1)
    else:
        print(f"{S.yellow}{summary}{S.reset}")


# ── watch ─────────────────────────────────────────────────────────────────────

def _cmd_watch(args):
    from pixlang import registry
    from pixlang.config import load_config
    from pixlang.watcher import Watcher

    path = _require_file(args.file)
    cfg = load_config(path)
    loader = _make_loader(args, path, cfg)
    Watcher(path, registry, verbose=args.verbose or cfg.verbose,
            interval=args.interval, lint=not args.no_lint,
            plugin_loader=loader if loader is not None else False,
            variables=_variables(cfg, args.var)).run()


# ── validate ──────────────────────────────────────────────────────────────────

def _cmd_validate(filepath):
    path     = _require_file(filepath)
    pipeline = _parse_or_die(path)
    print(f"  Statements parsed: {len(pipeline.commands)}\n")
    _dump_stmts(pipeline.commands, indent=4)
    print()
    _success("Syntax OK.")


def _dump_stmts(stmts, indent=4):
    from pixlang.parser.ast_nodes import ASTNode, Command, IfBlock, RepeatBlock, SetVar
    S = Style()
    pad = " " * indent
    for s in stmts:
        line = getattr(s, "line", 0)
        if isinstance(s, Command):
            print(f"{pad}line {line:>3}  {S.cyan}{s.name}{S.reset}  {S.dim}{s.args}{S.reset}")
        elif isinstance(s, SetVar):
            print(f"{pad}line {line:>3}  {S.yellow}SET{S.reset}  {s.var_name} = {s.value!r}")
        elif isinstance(s, IfBlock):
            print(f"{pad}line {line:>3}  {S.yellow}IF{S.reset}  ${s.var_name} {s.op} {s.cmp_value!r}")
            _dump_stmts(s.body, indent + 4)
            print(f"{pad}        {S.yellow}ENDIF{S.reset}")
        elif isinstance(s, RepeatBlock):
            print(f"{pad}line {line:>3}  {S.yellow}REPEAT{S.reset}  {s.count!r}")
            _dump_stmts(s.body, indent + 4)
            print(f"{pad}        {S.yellow}END{S.reset}")
        else:
            label = type(s).__name__.replace("Stmt", "").replace("Block", "").upper()
            print(f"{pad}line {line:>3}  {S.yellow}{label}{S.reset}")
            for value in vars(s).values():
                if isinstance(value, list) and any(isinstance(v, ASTNode) for v in value):
                    _dump_stmts([v for v in value if isinstance(v, ASTNode)], indent + 4)


# ── link ──────────────────────────────────────────────────────────────────────

def _cmd_link(filepath, show_map=False):
    from pixlang import registry
    from pixlang.config import load_config
    from pixlang.linker import Linker

    path = _require_file(filepath)
    cfg  = load_config(path.resolve())
    try:
        prog = Linker(registry, cfg.link_search_paths).link(path, raise_on_error=False)
    except SyntaxError as e:
        _error(e.pretty() if hasattr(e, "pretty") else str(e))
    except RuntimeError as e:
        _error(str(e))

    if show_map:
        print(prog.symbol_map()); print()
    S = Style()
    for d in prog.diagnostics:
        colour = S.red if d.severity == "error" else S.yellow
        print(f"{colour}{d}{S.reset}")
    if prog.diagnostics:
        print()
    print(f"  {len(prog.units)} unit(s), {len(prog.functions)} function(s), "
          f"{len(prog.errors)} error(s), {len(prog.warnings)} warning(s)")
    if prog.errors:
        sys.exit(1)
    _success("Link OK.")


# ── commands ──────────────────────────────────────────────────────────────────

def _cmd_list_commands(source_filter, category_filter=None):
    from pixlang import registry
    S = Style()
    infos = registry.list_info()
    if source_filter:
        infos = [i for i in infos if source_filter.lower() in i.source.lower()]
        if not infos:
            _error(f"No commands from source '{source_filter}'.")
    if category_filter:
        infos = [i for i in infos if category_filter.lower() == (i.category or "").lower()]
        if not infos:
            _error(f"No commands in category '{category_filter}'.")
    groups = {}
    for info in infos:
        groups.setdefault(info.category or "Other", []).append(info)
    print(f"  {len(infos)} commands registered:\n")
    for cat in sorted(groups):
        print(f"  {S.cyan}[{cat}]{S.reset}")
        for info in groups[cat]:
            src = "" if info.source == "builtin" else f"  {S.dim}({info.source}){S.reset}"
            summary = f"  {S.dim}{info.summary}{S.reset}" if info.summary else ""
            print(f"    {info.usage}{src}")
            if summary:
                print(f"      {summary.strip()}")
        print()


# ── plugins ───────────────────────────────────────────────────────────────────

def _cmd_plugins(trust: bool = False):
    from pixlang import registry
    from pixlang.plugins import PluginLoader, trust_from_config
    from pixlang.plugins.loader import ENTRY_POINT_GROUP, _resolve_plugin_dir
    S = Style()
    loader = PluginLoader(registry, trust_local=trust or trust_from_config(Path.cwd()))
    loader.load_entrypoints()
    loader.load_directory()
    if not loader.manifests and not loader.skipped:
        print(f"  {S.dim}No plugins discovered.{S.reset}\n")
        print(f"  Entry-point group : {ENTRY_POINT_GROUP}")
        print(f"  Plugin directory  : {_resolve_plugin_dir()}\n")
        return
    for m in loader.manifests:
        st = f"{S.green}✓{S.reset}" if m.ok else f"{S.red}✗{S.reset}"
        print(f"  {st}  {S.bold}{m.name}{S.reset}  {S.dim}[{m.source_type}]{S.reset}")
        print(f"       path     : {m.source_path}")
        if m.commands:
            print(f"       commands : {', '.join(m.commands)}")
        if not m.ok:
            print(f"       {S.red}error    : {m.error}{S.reset}")
        print()
    for p in loader.skipped:
        print(f"  {S.yellow}-{S.reset}  {S.bold}{p.stem}{S.reset}  {S.dim}[directory, not trusted]{S.reset}")
        print(f"       path     : {p}\n")


# ── new ───────────────────────────────────────────────────────────────────────

def _cmd_new(name: str):
    """Scaffold a minimal but complete project that runs as printed."""
    import cv2
    import numpy as np

    S = Style()
    slug = name.lower().replace(" ", "_")
    root = Path(slug)

    if root.exists():
        _error(f"Directory '{slug}' already exists.")

    (root / "images").mkdir(parents=True)
    (root / "output").mkdir(parents=True)

    # A synthetic starter image so the scaffold runs out of the box.
    canvas = np.full((480, 640, 3), 30, np.uint8)
    cv2.circle(canvas, (180, 240), 90, (60, 180, 255), -1)
    cv2.rectangle(canvas, (360, 140), (560, 340), (255, 200, 60), -1)
    cv2.imwrite(str(root / "images" / "input.png"), canvas)

    safe_title = name.replace('"', "'")
    pipeline_content = f"""\
# {name}
# Generated by: pixlang new {name}
# Usage (from this directory): pixlang run pipeline.pxl --verbose
# Paths are relative to the directory you run pixlang from.

LOAD "images/input.png"
RESIZE 640 480
PRINT_INFO

# ── Pre-processing ────────────────────────────────────────────────────────────
GRAYSCALE
BLUR 5

# ── Analysis ──────────────────────────────────────────────────────────────────
THRESHOLD_OTSU
FIND_CONTOURS
DRAW_BOUNDING_BOXES

# ── Custom command from pipeline.plugins.py ──────────────────────────────────
CUSTOM_CMD

# ── Annotation ────────────────────────────────────────────────────────────────
DRAW_TEXT "{safe_title}" 12 35 1.0 255 255 255 2 "duplex"

# ── Output ────────────────────────────────────────────────────────────────────
SAVE "output/result.png"
"""

    plugin_content = f"""\
# {slug}/pipeline.plugins.py
# Local plugin — loaded alongside pipeline.pxl because pixlang.toml sets
# [plugins] trust_local = true (or run with --trust-plugins).
# Add custom commands for this project here.

def register(registry):
    @registry.register("CUSTOM_CMD", source="{slug}-local")
    def cmd_custom(image):
        \"\"\"CUSTOM_CMD — replace with your own processing logic.\"\"\"
        return image
"""

    toml_content = """\
# pixlang.toml — project configuration
[defaults]
verbose = false

[plugins]
# This project's own pipeline.plugins.py is trusted code.
trust_local = true

[variables]
# Injected into every run; override with --var key=value
"""

    readme_content = f"""\
# {name}

A PixLang computer vision pipeline.

## Run

```bash
cd {slug}
pixlang run pipeline.pxl --verbose
```

Replace `images/input.png` with your own image (keep the name or edit
`pipeline.pxl`). The result is written to `output/result.png`.

## Lint / watch

```bash
pixlang lint pipeline.pxl
pixlang watch pipeline.pxl
```

## Custom commands

Edit `pipeline.plugins.py`. It is loaded because `pixlang.toml` sets
`[plugins] trust_local = true`.
"""

    (root / "pipeline.pxl").write_text(pipeline_content)
    (root / "pipeline.plugins.py").write_text(plugin_content)
    (root / "pixlang.toml").write_text(toml_content)
    (root / "README.md").write_text(readme_content)

    print(f"  {S.green}✓{S.reset}  Project '{name}' created at {S.cyan}./{slug}/{S.reset}\n")
    print(f"  {S.dim}Structure:{S.reset}")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        indent = "  " * len(rel.parts)
        print(f"  {S.dim}{indent}{rel.name}{S.reset}")
    print()
    print(f"  {S.dim}Next steps:{S.reset}")
    print(f"  1. {S.cyan}cd {slug}{S.reset}")
    print(f"  2. {S.cyan}pixlang run pipeline.pxl --verbose{S.reset}")
    print("  3. Replace images/input.png with your own image")


# ── editor ────────────────────────────────────────────────────────────────────

def _cmd_editor(port: int = 7478, no_browser: bool = False, workspace=None,
                timeout: float = 30.0, no_plugins: bool = False):
    try:
        from flask import Flask  # noqa: F401
    except ImportError:
        _error(
            "Flask is required for the editor.\n"
            "  Install with:  pip install pixlang[editor]\n"
            "  Or directly:   pip install flask>=3.0"
        )

    from pixlang.editor.server import EXAMPLES_DIR, STATIC_DIR, create_app

    ws = Path(workspace).resolve() if workspace else Path.cwd()
    if not ws.is_dir():
        _error(f"Workspace is not a directory: {ws}")
    app = create_app(workspace=ws, run_timeout=timeout, plugins=not no_plugins)

    S = Style()
    print(f"  Editor    : {S.cyan}http://localhost:{port}/{S.reset}")
    print(f"  Workspace : {S.dim}{ws}{S.reset}")
    print(f"  Examples  : {S.dim}{EXAMPLES_DIR}{S.reset}")
    print(f"  Static    : {S.dim}{STATIC_DIR}{S.reset}")
    print(f"  Press {S.bold}Ctrl+C{S.reset} to stop\n")

    if not no_browser:
        import threading
        import time as _time
        import webbrowser

        def _open():
            _time.sleep(1.0)
            webbrowser.open(f"http://localhost:{port}/")
        threading.Thread(target=_open, daemon=True).start()

    # use_reloader=False prevents double-import of the command registry
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


# ── Shared utilities ──────────────────────────────────────────────────────────

_BANNER = """
  ██████╗ ██╗██╗  ██╗██╗      █████╗ ███╗   ██╗ ██████╗
  ██╔══██╗██║╚██╗██╔╝██║     ██╔══██╗████╗  ██║██╔════╝
  ██████╔╝██║ ╚███╔╝ ██║     ███████║██╔██╗ ██║██║  ███╗
  ██╔═══╝ ██║ ██╔██╗ ██║     ██╔══██║██║╚██╗██║██║   ██║
  ██║     ██║██╔╝ ██╗███████╗██║  ██║██║ ╚████║╚██████╔╝
  ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝"""


def _print_banner():
    if not is_tty(sys.stdout):
        return
    S = Style()
    print(f"{S.cyan}{S.bold}{_BANNER}{S.reset}")
    print(f"{S.dim}  Computer Vision Pipeline DSL  v{_version()}{S.reset}\n")


def _require_file(filepath):
    p = Path(filepath)
    if not p.exists():
        _error(f"File not found: {filepath}")
    return p


def _parse_or_die(path: Path):
    from pixlang import parse
    from pixlang.watcher.engine import format_syntax_error
    try:
        source = path.read_text()
    except OSError as e:
        _error(f"Cannot read {path}: {e}")
    try:
        return parse(source, filename=str(path))
    except (SyntaxError, RecursionError) as e:
        if isinstance(e, SyntaxError) and not getattr(e, "filename", None):
            try:
                e.filename = str(path)
            except Exception:
                pass
        _error(e.pretty() if hasattr(e, "pretty") else format_syntax_error(e, path))


def _fail(exc: BaseException, verbose: bool = False):
    from pixlang.watcher.engine import format_syntax_error
    if isinstance(exc, (SyntaxError, RecursionError)):
        msg = format_syntax_error(exc)
    else:
        msg = str(exc) or type(exc).__name__
    if verbose:
        traceback.print_exc()
    _error(msg)


def _error(msg):
    S = Style(sys.stderr)
    print(f"\n  {S.red}✗ Error:{S.reset} {msg}\n", file=sys.stderr)
    sys.exit(1)


def _success(msg):
    S = Style()
    print(f"  {S.green}✓{S.reset} {msg}")


if __name__ == "__main__":
    main()
