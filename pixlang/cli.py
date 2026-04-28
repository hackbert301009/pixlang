#!/usr/bin/env python3
# pixlang/cli.py  v0.4
"""
PixLang CLI

Commands:
    run      pipeline.pxl [--verbose] [--no-plugins]
    lint     pipeline.pxl
    watch    pipeline.pxl [--verbose] [--interval 0.5]
    validate pipeline.pxl
    commands [--source SRC]
    plugins
    new      <name>
    --version
"""
import argparse, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        prog="pixlang", description="PixLang — Computer Vision Pipeline DSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version="%(prog)s 0.4.0")
    sub = ap.add_subparsers(dest="command")

    rp = sub.add_parser("run",      help="Execute a .pxl pipeline")
    rp.add_argument("file"); rp.add_argument("--verbose","-v",action="store_true")
    rp.add_argument("--no-plugins", action="store_true")

    lp = sub.add_parser("lint",     help="Lint a pipeline without running it")
    lp.add_argument("file")

    wp = sub.add_parser("watch",    help="Re-run pipeline on file changes")
    wp.add_argument("file"); wp.add_argument("--verbose","-v",action="store_true")
    wp.add_argument("--interval", type=float, default=0.5, metavar="SEC")
    wp.add_argument("--no-lint", action="store_true")

    vp = sub.add_parser("validate", help="Parse-check without executing")
    vp.add_argument("file")

    cp = sub.add_parser("commands", help="List available DSL commands")
    cp.add_argument("--source", metavar="SRC")

    sub.add_parser("plugins", help="Show discovered plugins")

    np_ = sub.add_parser("new",     help="Scaffold a new pipeline project")
    np_.add_argument("name", help="Project / pipeline name")

    ep = sub.add_parser("editor",  help="Launch the visual editor in a browser")
    ep.add_argument("--port", type=int, default=7478, metavar="PORT",
                    help="Port to listen on (default: 7478)")
    ep.add_argument("--no-browser", action="store_true",
                    help="Do not open browser automatically")

    args = ap.parse_args()
    if args.command is None:
        ap.print_help(); sys.exit(0)

    _print_banner()

    {
        "run":      lambda: _cmd_run(args.file, args.verbose, args.no_plugins, getattr(args,"batch",False)),
        "lint":     lambda: _cmd_lint(args.file),
        "watch":    lambda: _cmd_watch(args.file, args.verbose, args.interval, args.no_lint),
        "validate": lambda: _cmd_validate(args.file),
        "commands": lambda: _cmd_list_commands(getattr(args,"source",None)),
        "plugins":  _cmd_plugins,
        "new":      lambda: _cmd_new(args.name),
        "editor":   lambda: _cmd_editor(args.port, args.no_browser),
    }[args.command]()


# ── run ───────────────────────────────────────────────────────────────────────

def _cmd_run(filepath, verbose, no_plugins, batch=False):
    from pixlang import parse, Executor, registry
    from pixlang.plugins import PluginLoader
    from pixlang.config import load_config
    from pixlang.batch.engine import BatchRunner, _flatten_commands

    path = _require_file(filepath)
    cfg  = load_config(path)

    verbose = verbose or cfg.verbose

    print(f"  Pipeline  : {path.name}")
    if cfg.source_path:
        print(f"  Config    : {cfg.source_path.name}")
    print(f"  Verbose   : {verbose}  |  Plugins: {'off' if no_plugins else 'on'}")

    pipeline = _parse_or_die(path.read_text())
    loader   = None if no_plugins else PluginLoader(registry)

    flat_cmds = _flatten_commands(pipeline.commands)
    has_glob  = any(c.name == "LOAD_GLOB" for c in flat_cmds)

    if batch or has_glob:
        print(f"  Mode      : batch\n")
        runner = BatchRunner(pipeline, registry, verbose=verbose,
                             plugin_loader=loader, pipeline_path=path)
        try:
            results = runner.run()
        except Exception as e:
            _error(str(e))
        if results["failed"]:
            _error(f"{results['failed']} file(s) failed.")
        _success(f"Batch complete: {results['ok']}/{results['total']} succeeded.")
        return

    print(f"  Mode      : single\n")
    if verbose:
        print(f"  {'CMD':<26}{'ARGS':<22}{'SHAPE':<18}{'TIME':>8}")
        print(f"  {chr(0x2500)*76}")

    ex = Executor(registry=registry, verbose=verbose,
                  plugin_loader=loader, pipeline_path=path)
    ex.context["vars"].update(cfg.variables)

    try:
        ex.run(pipeline)
    except (RuntimeError, FileNotFoundError, NameError, AssertionError) as e:
        _error(str(e))

    print(); _success("Pipeline completed successfully.")


def _cmd_lint(filepath):
    from pixlang import registry
    from pixlang.linter import Linter

    path     = _require_file(filepath)
    pipeline = _parse_or_die(path.read_text())
    linter   = Linter(registry)
    diags    = linter.lint(pipeline)

    if not diags:
        print(f"  {path.name}  ({linter.rule_count} rules checked)\n")
        _success("No issues found.")
        return

    errors = warns = infos = 0
    for d in diags:
        if d.severity.value == "error":
            print(f"  {RED}{d}{RESET}"); errors += 1
        elif d.severity.value == "warning":
            print(f"  {YEL}{d}{RESET}"); warns += 1
        else:
            print(f"  {DIM}{d}{RESET}"); infos += 1

    print()
    summary = f"  {errors} error(s)  {warns} warning(s)  {infos} info(s)"
    if errors:
        print(f"{RED}{summary}{RESET}"); sys.exit(1)
    else:
        print(f"{YEL}{summary}{RESET}")


# ── watch ─────────────────────────────────────────────────────────────────────

def _cmd_watch(filepath, verbose, interval, no_lint):
    from pixlang import registry
    from pixlang.watcher import Watcher

    path = _require_file(filepath)
    Watcher(path, registry, verbose=verbose,
            interval=interval, lint=not no_lint).run()


# ── validate ──────────────────────────────────────────────────────────────────

def _cmd_validate(filepath):
    path     = _require_file(filepath)
    pipeline = _parse_or_die(path.read_text())
    print(f"  Statements parsed: {len(pipeline.commands)}\n")
    _dump_stmts(pipeline.commands, indent=4)
    print(); _success("Syntax OK.")


def _dump_stmts(stmts, indent=4):
    from pixlang.parser.ast_nodes import Command, SetVar, IfBlock, RepeatBlock
    pad = " " * indent
    for s in stmts:
        if isinstance(s, Command):
            print(f"{pad}line {s.line:>3}  {CYAN}{s.name}{RESET}  {DIM}{s.args}{RESET}")
        elif isinstance(s, SetVar):
            print(f"{pad}line {s.line:>3}  {YEL}SET{RESET}  {s.var_name} = {s.value!r}")
        elif isinstance(s, IfBlock):
            print(f"{pad}line {s.line:>3}  {YEL}IF{RESET}  ${s.var_name} {s.op} {s.cmp_value!r}")
            _dump_stmts(s.body, indent + 4)
            print(f"{pad}        {YEL}ENDIF{RESET}")
        elif isinstance(s, RepeatBlock):
            print(f"{pad}line {s.line:>3}  {YEL}REPEAT{RESET}  {s.count!r}")
            _dump_stmts(s.body, indent + 4)
            print(f"{pad}        {YEL}END{RESET}")


# ── commands ──────────────────────────────────────────────────────────────────

def _cmd_list_commands(source_filter):
    from pixlang import registry
    by_src = registry.commands_by_source()
    if source_filter:
        by_src = {k: v for k, v in by_src.items() if source_filter.lower() in k.lower()}
        if not by_src: _error(f"No commands from source '{source_filter}'.")
    print(f"  {len(registry)} commands registered:\n")
    for src, infos in by_src.items():
        print(f"  {CYAN}[{src}]{RESET}")
        for info in sorted(infos, key=lambda i: i.name):
            doc = f"  {DIM}{info.doc}{RESET}" if info.doc else ""
            print(f"    {info.name:<32}{doc}")
        print()


# ── plugins ───────────────────────────────────────────────────────────────────

def _cmd_plugins():
    from pixlang import registry
    from pixlang.plugins import PluginLoader
    from pixlang.plugins.loader import DEFAULT_PLUGIN_DIR
    loader = PluginLoader(registry)
    loader.load_entrypoints()
    loader.load_directory()
    if not loader.manifests:
        print(f"  {DIM}No plugins discovered.{RESET}\n")
        print(f"  Entry-point group : pixlang.commands")
        print(f"  Plugin directory  : {DEFAULT_PLUGIN_DIR}\n")
        return
    for m in loader.manifests:
        st = f"{GREEN}✓{RESET}" if m.ok else f"{RED}✗{RESET}"
        print(f"  {st}  {BOLD}{m.name}{RESET}  {DIM}[{m.source_type}]{RESET}")
        print(f"       path     : {m.source_path}")
        if m.commands: print(f"       commands : {', '.join(m.commands)}")
        if not m.ok:   print(f"       {RED}error    : {m.error}{RESET}")
        print()


# ── new ───────────────────────────────────────────────────────────────────────

def _cmd_new(name: str):
    """Scaffold a minimal but complete project layout for a new pipeline."""
    slug = name.lower().replace(" ", "_")
    root = Path(slug)

    if root.exists():
        _error(f"Directory '{slug}' already exists.")

    (root / "images").mkdir(parents=True)
    (root / "output").mkdir(parents=True)

    pipeline_content = f"""\
# {name}
# Generated by: pixlang new {name}
# Usage: pixlang run {slug}/pipeline.pxl --verbose

LOAD "images/input.jpg"
RESIZE 640 480
PRINT_INFO

# ── Pre-processing ────────────────────────────────────────────────────────────
GRAYSCALE
BLUR 5

# ── Analysis ──────────────────────────────────────────────────────────────────
THRESHOLD_OTSU
FIND_CONTOURS
DRAW_BOUNDING_BOXES

# ── Annotation ────────────────────────────────────────────────────────────────
DRAW_TEXT "{name}" 12 35 1.0 255 255 255 2 "duplex"

# ── Output ────────────────────────────────────────────────────────────────────
SAVE "output/result.png"
"""

    plugin_content = f"""\
# {slug}/pipeline.plugins.py
# Local plugin — auto-loaded alongside pipeline.pxl
# Add custom commands for this project here.

def register(registry):
    @registry.register("CUSTOM_CMD", source="{slug}-local")
    def cmd_custom(image):
        \"\"\"CUSTOM_CMD — replace with your own processing logic.\"\"\"
        return image
"""

    readme_content = f"""\
# {name}

A PixLang computer vision pipeline.

## Run

```bash
# Place your input image at images/input.jpg
pixlang run pipeline.pxl --verbose
```

## Lint

```bash
pixlang lint pipeline.pxl
```

## Watch mode (auto-rerun on save)

```bash
pixlang watch pipeline.pxl
```

## Custom commands

Edit `pipeline.plugins.py` to add project-specific DSL commands.
"""

    (root / "pipeline.pxl").write_text(pipeline_content)
    (root / "pipeline.plugins.py").write_text(plugin_content)
    (root / "README.md").write_text(readme_content)

    print(f"  {GREEN}✓{RESET}  Project '{name}' created at {CYAN}./{slug}/{RESET}\n")
    print(f"  {DIM}Structure:{RESET}")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        indent = "  " * len(rel.parts)
        print(f"  {DIM}{indent}{rel.name}{RESET}")
    print()
    print(f"  {DIM}Next steps:{RESET}")
    print(f"  1. Add your image  →  {slug}/images/input.jpg")
    print(f"  2. {CYAN}pixlang run {slug}/pipeline.pxl --verbose{RESET}")
    print(f"  3. {CYAN}pixlang lint {slug}/pipeline.pxl{RESET}")


# ── editor ────────────────────────────────────────────────────────────────────

def _cmd_editor(port: int = 7478, no_browser: bool = False):
    try:
        from flask import Flask  # noqa: F401
    except ImportError:
        _error(
            "Flask is required for the editor.\n"
            "  Install with:  pip install pixlang[editor]\n"
            "  Or directly:   pip install flask>=3.0"
        )

    from pixlang.editor.server import app, STATIC_DIR, EXAMPLES_DIR

    print(f"  Editor    : {CYAN}http://localhost:{port}/{RESET}")
    print(f"  Examples  : {DIM}{EXAMPLES_DIR}{RESET}")
    print(f"  Static    : {DIM}{STATIC_DIR}{RESET}")
    print(f"  Press {BOLD}Ctrl+C{RESET} to stop\n")

    if not no_browser:
        import threading, webbrowser, time as _time
        def _open():
            _time.sleep(1.0)
            webbrowser.open(f"http://localhost:{port}/")
        threading.Thread(target=_open, daemon=True).start()

    # use_reloader=False prevents double-import of the command registry
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


# ── Shared utilities ──────────────────────────────────────────────────────────

CYAN  = "\033[96m"; GREEN = "\033[92m"; RED   = "\033[91m"
YEL   = "\033[93m"; DIM   = "\033[2m";  RESET = "\033[0m"; BOLD  = "\033[1m"


def _print_banner():
    print(f"""
{CYAN}{BOLD}
  ██████╗ ██╗██╗  ██╗██╗      █████╗ ███╗   ██╗ ██████╗
  ██╔══██╗██║╚██╗██╔╝██║     ██╔══██╗████╗  ██║██╔════╝
  ██████╔╝██║ ╚███╔╝ ██║     ███████║██╔██╗ ██║██║  ███╗
  ██╔═══╝ ██║ ██╔██╗ ██║     ██╔══██║██║╚██╗██║██║   ██║
  ██║     ██║██╔╝ ██╗███████╗██║  ██║██║ ╚████║╚██████╔╝\n  ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝
{RESET}{DIM}  Computer Vision Pipeline DSL  v0.4.0{RESET}
""")


def _require_file(filepath):
    p = Path(filepath)
    if not p.exists(): _error(f"File not found: {filepath}")
    return p


def _parse_or_die(source):
    from pixlang import parse
    try:
        return parse(source)
    except SyntaxError as e:
        _error(str(e))


def _error(msg):
    print(f"\n  {RED}✗ Error:{RESET} {msg}\n"); sys.exit(1)


def _success(msg):
    print(f"  {GREEN}✓{RESET} {msg}")


if __name__ == "__main__":
    main()
