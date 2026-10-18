# tests/test_cmd_trust.py
"""Workspace trust: the store, ancestor coverage, overbroad paths, the CLI."""
import pytest

from pixlang.cli import main
from pixlang.plugins import (
    add_trusted,
    is_trusted,
    overbroad_warning,
    remove_trusted,
    trust_file,
    trusted_paths,
)


def test_store_starts_empty(tmp_path):
    assert trusted_paths() == []
    assert is_trusted(tmp_path) is False
    assert not trust_file().exists()


def test_add_is_idempotent_and_removable(tmp_path):
    assert add_trusted(tmp_path) is True
    assert add_trusted(tmp_path) is False          # already covered
    assert trusted_paths() == [tmp_path.resolve()]
    assert is_trusted(tmp_path) is True

    assert remove_trusted(tmp_path) is True
    assert remove_trusted(tmp_path) is False
    assert is_trusted(tmp_path) is False


def test_trust_covers_subdirectories_but_not_siblings(tmp_path):
    project = tmp_path / "project"
    (project / "deep" / "nested").mkdir(parents=True)
    (tmp_path / "other").mkdir()

    add_trusted(project)
    assert is_trusted(project / "deep" / "nested") is True
    assert is_trusted(tmp_path / "other") is False
    assert is_trusted(tmp_path) is False           # the parent is not implied


def test_paths_with_quotes_survive_a_round_trip(tmp_path):
    odd = tmp_path / 'we"ird'
    odd.mkdir()
    add_trusted(odd)
    assert trusted_paths() == [odd.resolve()]
    assert is_trusted(odd) is True


def test_store_is_not_world_readable(tmp_path):
    add_trusted(tmp_path)
    assert trust_file().stat().st_mode & 0o077 == 0


def test_corrupt_store_is_ignored_not_fatal(tmp_path):
    trust_file().parent.mkdir(parents=True, exist_ok=True)
    trust_file().write_text("this is not toml [[[\n")
    assert trusted_paths() == []
    assert is_trusted(tmp_path) is False


def test_overbroad_paths_are_named(tmp_path):
    assert overbroad_warning(tmp_path) is None
    assert "home directory" in overbroad_warning(pytest.importorskip("pathlib").Path.home())
    assert "filesystem root" in overbroad_warning("/")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _run(*argv):
    """Run the CLI in-process; returns the SystemExit code (None when it returns)."""
    try:
        main(list(argv))
    except SystemExit as exc:
        return exc.code
    return None


def test_cli_refuses_home_and_root(tmp_path, capsys):
    assert _run("trust", "/") == 1
    assert "filesystem root" in capsys.readouterr().err
    assert trusted_paths() == []


def test_cli_refuses_a_file(tmp_path, capsys):
    target = tmp_path / "pipeline.pxl"
    target.write_text("")
    assert _run("trust", str(target)) == 1
    assert "Not a directory" in capsys.readouterr().err


def test_cli_remove_unknown_path_fails(tmp_path, capsys):
    assert _run("trust", "--remove", str(tmp_path)) == 1
    assert "not a confirmed workspace" in capsys.readouterr().err


def test_cli_add_list_remove(tmp_path, capsys):
    project = tmp_path / "project"
    project.mkdir()

    assert _run("trust", str(project)) is None
    assert "Confirmed" in capsys.readouterr().out

    assert _run("trust", "--list") is None
    assert str(project.resolve()) in capsys.readouterr().out

    assert _run("trust", str(project)) is None
    assert "already covered" in capsys.readouterr().out

    assert _run("trust", "--remove", str(project)) is None
    assert "Withdrew trust" in capsys.readouterr().out
    assert trusted_paths() == []


def test_cli_list_marks_missing_directories(tmp_path, capsys):
    gone = tmp_path / "gone"
    gone.mkdir()
    add_trusted(gone)
    gone.rmdir()
    _run("trust", "--list")
    assert "(missing)" in capsys.readouterr().out
