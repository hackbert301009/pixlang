# tests/test_v5_examples.py
"""End-to-end: the v0.5 showcase examples run, link and lint cleanly."""
from pathlib import Path

import pytest

from pixlang.commands import registry
from pixlang.executor import Executor
from pixlang.linker import link
from pixlang.linter import Linter, Severity
from pixlang.parser import parse

EXAMPLES = sorted(Path("examples/v05").glob("*.pxl"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(path):
    out = Path("examples/v05/out") / f"{path.stem}.png"
    out.unlink(missing_ok=True)
    Executor(registry=registry, pipeline_path=path).run(parse(path.read_text(), filename=str(path)))
    assert out.exists()
    out.unlink()


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_links_and_lints(path):
    prog = link(path, registry=registry)
    assert not prog.errors
    errors = [d for d in Linter(registry).lint_file(path) if d.severity == Severity.ERROR]
    assert not errors, [str(e) for e in errors]


def test_linked_example_runs():
    path = Path("examples/v05/functions_modules.pxl")
    prog = link(path, registry=registry)
    ex = Executor(registry=registry)
    ex.run(prog)
    assert ex.context["vars"]["calls"] == 3
    Path("examples/v05/out/functions_modules.png").unlink(missing_ok=True)
