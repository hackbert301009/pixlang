# Contributing to PixLang

## Setup

```bash
git clone https://github.com/hackbert301009/pixlang && cd pixlang
pip install -e ".[dev,editor]"      # dev: pytest, pytest-cov, ruff; editor: flask
```

Python 3.9 to 3.13 are supported and CI runs all of them, so no syntax newer than 3.9.

## Tests and lint

```bash
pytest                              # whole suite; must stay green
pytest tests/test_chained_comparison.py -k middle   # one file / one test
ruff check pixlang tests            # must report nothing
```

CI runs `pytest --cov=pixlang` on every push. Run both commands before you push.

## Code style

- No banner or section-divider comments. They add noise and go stale.
- Comments say why, never what. The code already says what.
- No version numbers in code, names, docstrings or test file names. They are wrong after the next release.
- Docstrings only where the name and signature do not already say it. No `Args:` / `Returns:` blocks.
- No `except Exception` outside a thread or process boundary. Catch what you can handle, so bugs stay visible.
- No `getattr(x, "attr", None)` for attributes the type defines. It hides typos and type changes.
- Search for an existing helper before writing one.
- Name magic numbers (`DISPLAY_DIGITS = 6`). The name says what the number means.

## Public API

What a package exports (its `__init__.py` / `__all__`) is the public API. Names with a leading underscore are private to their module's package: never import them from another package. If you need one, export a proper name instead.

## Error messages

- Say what went wrong and what to do, in one line: `Unknown command 'TRESHOLD'. Did you mean: THRESHOLD?`
- Point at the source: `file:line:col`, or `line N` inside a running pipeline.
- Prefix command errors with the command in brackets (`[LOAD] Could not read image: 'a.png'`).
- Expected failures (a failed ASSERT, a user typo, a bad file) print the message only. A traceback is for bugs in PixLang itself.
- Quote the user's value and show valid choices when there is a fixed set.

## Adding a command

1. Write the function in `pixlang/commands/builtin.py` (or `lowlevel.py` for hardware-near ones) inside `register_all`, decorated with `@_command(r, "NAME")`. The first parameter is the image. Later ones are the DSL arguments; type annotations drive argument coercion and the usage line. Add `_ctx` to receive the executor context.
2. Add the name to `CATEGORIES` in `builtin.py`; it is what `pixlang commands` and the editor group by.
3. Make the docstring's first line `NAME args — description`. It becomes the command summary shown by `pixlang commands` and the editor hover.
4. Parameters that hold file paths are called `path`, as in `LOAD`, `SAVE` and `HISTOGRAM_SAVE`. Write images with `_write_image`, which checks the extension and creates directories.
5. Return a new array and keep metadata (contours, ...); `_command` carries it over for bare arrays.
6. Add tests next to the related ones (`tests/test_cmd_builtin_robust.py`, `tests/test_commands_v5.py`) using `registry.invoke(name, image, args, ctx)`.
7. List the command in `README.md` / `docs/LANGUAGE.md` if users need to know about it, and add a CHANGELOG entry.

Plugins register commands the same way through `registry.register("NAME", source="my_plugin")`; see `plugins_example/`.

## Adding a test

- Name the file after its subject (`test_chained_comparison.py`), never after a release.
- Write the failing regression test first, watch it fail, then fix.
- Assert observable results (values, output, exit codes), not private state. Use the public API: `parse`, `Executor`, `Linter`, `cli.main(argv)`.
- Call the CLI in-process through `pixlang.cli.main(argv)` with `capsys` and `monkeypatch.chdir(tmp_path)`. Keep subprocess tests for smoke checks.
- Build files under `tmp_path` and anchor other paths on `Path(__file__)`. Tests must pass from any working directory.
- Seed randomness (`np.random.default_rng(0)`). A test that fails sometimes is worse than none.
- Mark known bugs with `@pytest.mark.xfail(strict=True, reason=...)` so the fix flips the test.

## Commits and pull requests

- Small, separate commits; each one builds and passes the tests. Do not mix a refactor with a fix.
- A fix comes with its test, in the same commit or the one before it.
- Commit subject in the imperative; the body explains why.
- User-visible changes (syntax, CLI flags, messages, new commands) get an entry in `CHANGELOG.md` under `[Unreleased]` and an update to `docs/LANGUAGE.md` or the README.
- In the pull request, say what changed, why, and how you checked it.
