# pixlang/batch/engine.py
"""
PixLang Batch Processing  v0.5
═══════════════════════════════

Two DSL commands that work as a pair:

    LOAD_GLOB "frames/*.jpg"
    <any processing commands>
    SAVE_EACH "output/{stem}_result.png"

LOAD_GLOB:
  - Expands the glob pattern relative to the pipeline file
  - Stores the matched file list in the executor context
  - In batch mode loads the file chosen by the runner (ctx["batch_override"]),
    otherwise the FIRST match

SAVE_EACH:
  - Uses Python str.format()-style templates with these variables:
      {name}    original filename without extension  (e.g. "frame_001")
      {stem}    alias for {name}
      {ext}     original extension                   (e.g. ".jpg")
      {index}   zero-based batch index               (e.g. "0", "1", …)
      {index1}  one-based batch index                (e.g. "1", "2", …)
      {dir}     original file's parent directory
      {<var>}   any pipeline variable

How the batch loop works:
  BatchRunner statically walks the AST (recursively through every block
  type) for a LOAD_GLOB command and expands its literal pattern relative to
  the pipeline file — nothing is executed for discovery. It then runs the
  whole pipeline once per matched file with ``ctx["batch_override"]`` set;
  LOAD_GLOB loads that file instead of re-globbing. Config / --var
  variables are applied to every run and the plugin loader is shared.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import cv2

if TYPE_CHECKING:
    from pixlang.commands.registry import CommandRegistry


def register_batch_commands(registry: CommandRegistry) -> None:
    """Register LOAD_GLOB and SAVE_EACH into the given registry."""

    def cmd_load_glob(image, pattern: str, _ctx: dict = None):
        """LOAD_GLOB "frames/*.jpg"  — load all files matching a glob pattern.

        Expands the pattern (relative to the pipeline file) and stores the
        matched file list in the pipeline context. In batch mode the runner
        selects the file for each run (``batch_override``); otherwise the
        first match is loaded.

        In the context, sets:
            batch_files   : list of matched Path objects
            batch_index   : current index (0-based)
            batch_current : Path of the current file
            name, stem    : filename without extension
            ext           : file extension
            dir           : parent directory

        Examples:
            LOAD_GLOB "images/*.jpg"
            LOAD_GLOB "frames/frame_*.png"
            LOAD_GLOB "/data/scans/**/*.tiff"
        """
        if _ctx is None:
            raise RuntimeError("[LOAD_GLOB] Requires executor context.")
        _ctx.setdefault("vars", {})

        override = _ctx.get("batch_override")
        if override is not None:
            current = Path(override)
            files = list(_ctx.get("batch_files") or [current])
            index = _ctx.get("batch_index")
            if index is None:
                index = files.index(current) if current in files else 0
        else:
            base_dir = _ctx.get("pipeline_dir", Path("."))
            files = expand_pattern(pattern, base_dir)
            if not files:
                raise FileNotFoundError(
                    f"[LOAD_GLOB] No files matched pattern: '{pattern}'\n"
                    f"  Searched from: {base_dir}"
                )
            current, index = files[0], 0

        _ctx["batch_files"] = files
        _update_batch_ctx(_ctx, current, index)

        loaded = cv2.imread(str(current), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise OSError(f"[LOAD_GLOB] Cannot read: {current}")
        return loaded

    cmd_load_glob.fresh_image = True
    registry.register("LOAD_GLOB", category="I/O")(cmd_load_glob)

    @registry.register("SAVE_EACH", category="I/O")
    def cmd_save_each(image, template: str, _ctx: dict = None):
        """SAVE_EACH "output/{stem}_result.png"  — save each batch image with template name.

        Template variables:
            {name} / {stem}  — filename without extension
            {ext}            — original file extension (e.g. ".jpg")
            {index}          — zero-based batch index
            {index1}         — one-based batch index (supports {index1:03d})
            {dir}            — original file's parent directory
            {<var>}          — any pipeline variable

        The output directory is created automatically if it does not exist.

        Examples:
            SAVE_EACH "output/{stem}.png"
            SAVE_EACH "results/{name}_processed{ext}"
            SAVE_EACH "batch_{index1:03d}_{stem}.png"
        """
        from pixlang.commands.builtin import _require_image, _unwrap, _write_image

        _require_image("SAVE_EACH", image)
        if _ctx is None:
            raise RuntimeError("[SAVE_EACH] Requires executor context.")

        img, _ = _unwrap(image)
        current = Path(_ctx.get("batch_current") or "unknown.jpg")
        idx = int(_ctx.get("batch_index", 0))
        fields: Dict[str, Any] = dict(_ctx.get("vars", {}))
        fields.update({
            "name": current.stem, "stem": current.stem, "ext": current.suffix,
            "index": idx, "index1": idx + 1, "dir": str(current.parent),
        })
        try:
            out_path = template.format(**fields)
        except (KeyError, ValueError, IndexError) as e:
            raise RuntimeError(f"[SAVE_EACH] Template error in '{template}': {e}") from None

        written = _write_image("SAVE_EACH", out_path, img)
        if _ctx.get("verbose"):
            print(f"       → {written}")
        return image


def _update_batch_ctx(ctx: dict, file_path: Path, index: int) -> None:
    """Sync batch context variables for the current file."""
    ctx["batch_current"] = file_path
    ctx["batch_index"]   = index
    v = ctx.setdefault("vars", {})
    v["name"]   = file_path.stem
    v["stem"]   = file_path.stem
    v["ext"]    = file_path.suffix
    v["index"]  = index
    v["index1"] = index + 1
    v["dir"]    = str(file_path.parent)


def expand_pattern(pattern: str, base_dir: Any = ".") -> List[Path]:
    """Expand *pattern* (relative to *base_dir* unless absolute); sorted, files only."""
    p = Path(str(pattern)).expanduser()
    resolved = p if p.is_absolute() else Path(base_dir) / p
    return sorted(Path(m) for m in glob.glob(str(resolved), recursive=True)
                  if Path(m).is_file())


# ── Static AST scanning ───────────────────────────────────────────────────────

def iter_commands(nodes: Any) -> Iterator[Any]:
    """Yield every Command node in source order, recursing into nested blocks.

    Any attribute holding an AST node, or a list/tuple containing AST nodes
    (``body``, ``else_body``, ``elifs`` ...), is traversed, so block types
    added later are covered without changes here.
    """
    from pixlang.parser.ast_nodes import ASTNode, Command

    stack = [nodes]
    seen = set()
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
            continue
        if not isinstance(item, ASTNode) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, Command):
            yield item
            continue
        children = [v for v in vars(item).values() if isinstance(v, (ASTNode, list, tuple))]
        stack.extend(reversed(children))


def find_glob_commands(pipeline: Any) -> list:
    root = getattr(pipeline, "commands", pipeline)
    return [c for c in iter_commands(root) if c.name.upper() == "LOAD_GLOB"]


def _literal_vars(pipeline: Any, variables: Optional[dict]) -> dict:
    """Variables known statically: config/--var values, then top-level SET literals."""
    from pixlang.parser.ast_nodes import ASTNode, SetVar
    known = dict(variables or {})
    for stmt in getattr(pipeline, "commands", []):
        if isinstance(stmt, SetVar) and not isinstance(stmt.value, ASTNode):
            known[stmt.var_name] = stmt.value
    return known


def static_glob_pattern(pipeline: Any, variables: Optional[dict] = None) -> Optional[str]:
    """Return the LOAD_GLOB pattern of *pipeline* without executing it (or None)."""
    from pixlang.parser.ast_nodes import BareIdent, VarRef
    cmds = find_glob_commands(pipeline)
    if not cmds:
        return None
    cmd = cmds[0]
    if len({repr(c.args) for c in cmds}) > 1:
        print(f"  warning: {len(cmds)} different LOAD_GLOB commands; batch uses the one "
              f"on line {cmd.line}", file=sys.stderr)
    if len(cmd.args) != 1:
        raise RuntimeError(f"[Batch] LOAD_GLOB (line {cmd.line}) takes exactly one pattern")
    arg = cmd.args[0]
    known = _literal_vars(pipeline, variables)
    if isinstance(arg, VarRef):
        if arg.var_name not in known:
            raise RuntimeError(
                f"[Batch] LOAD_GLOB pattern ${arg.var_name} (line {cmd.line}) must be a "
                f"string literal or a variable set by SET/config/--var."
            )
        arg = known[arg.var_name]
    elif isinstance(arg, BareIdent):
        arg = known.get(arg.name, arg.name)
    return str(arg)


# ── BatchRunner ───────────────────────────────────────────────────────────────

class BatchRunner:
    """
    Run a pipeline once per file matched by its LOAD_GLOB pattern.

    Usage (internal, called from CLI):
        runner = BatchRunner(pipeline, registry, verbose=True,
                             plugin_loader=loader, pipeline_path=path,
                             variables=cfg.variables)
        summary = runner.run()
    """

    def __init__(self, pipeline, registry, verbose=False, plugin_loader=None,
                 pipeline_path=None, variables: Optional[dict] = None, stream=None):
        self.pipeline       = pipeline
        self.registry       = registry
        self.verbose        = verbose
        self.plugin_loader  = plugin_loader
        self.pipeline_path  = Path(pipeline_path) if pipeline_path else None
        self.variables      = dict(variables or {})
        self.stream         = stream

    @property
    def pipeline_dir(self) -> Path:
        return self.pipeline_path.parent if self.pipeline_path else Path(".")

    def discover(self) -> List[Path]:
        """Statically resolve the files this batch will process."""
        pattern = static_glob_pattern(self.pipeline, self.variables)
        if pattern is None:
            return []
        return expand_pattern(pattern, self.pipeline_dir)

    def _executor(self):
        from pixlang.executor import Executor
        ex = Executor(
            registry=self.registry, verbose=self.verbose,
            plugin_loader=self.plugin_loader, pipeline_path=self.pipeline_path,
        )
        ex.context["pipeline_dir"] = self.pipeline_dir
        ex.context["verbose"] = self.verbose
        ex.context["vars"].update(self.variables)
        return ex

    def run(self) -> dict:
        """
        Execute the batch pipeline.

        If the pipeline contains LOAD_GLOB, runs once per matched file.
        Otherwise falls back to a single normal run.

        Returns a summary dict: {total, ok, failed, files}.
        """
        from pixlang._style import Style
        out = self.stream if self.stream is not None else sys.stdout
        st = Style(out)

        if not find_glob_commands(self.pipeline):
            self._executor().run(self.pipeline)
            return {"total": 1, "ok": 1, "failed": 0, "files": []}

        batch_files = self.discover()
        if not batch_files:
            pattern = static_glob_pattern(self.pipeline, self.variables)
            raise FileNotFoundError(
                f"[Batch] LOAD_GLOB pattern '{pattern}' matched no files "
                f"(searched from {self.pipeline_dir})."
            )

        results = {"total": len(batch_files), "ok": 0, "failed": 0, "files": []}
        print(f"\n  Batch mode: {len(batch_files)} file(s)\n", file=out)

        for idx, file_path in enumerate(batch_files):
            label = f"[{idx+1}/{len(batch_files)}] {file_path.name}"
            try:
                ex = self._executor()
                ex.context["batch_files"] = batch_files
                ex.context["batch_override"] = file_path
                _update_batch_ctx(ex.context, file_path, idx)
                ex.run(self.pipeline)
                results["ok"] += 1
                results["files"].append({"path": str(file_path), "ok": True})
                print(f"  {st.green}✓{st.reset}  {label}", file=out)
            except Exception as e:
                results["failed"] += 1
                results["files"].append({"path": str(file_path), "ok": False, "error": str(e)})
                print(f"  {st.red}✗{st.reset}  {label}  → {e}", file=out)

        print(f"\n  Done: {results['ok']} ok, {results['failed']} failed\n", file=out)
        return results


def _flatten_commands(stmts) -> list:
    """Backward-compatible helper: every Command in *stmts*, at any depth."""
    return list(iter_commands(stmts))
