"""`pixlang lint --var` and pixlang.toml [variables] define variables for PX005."""
import pytest

from pixlang import cli

SOURCE = 'LOAD "in.png"\nRESIZE $w $h\nSAVE "out.png"\n'


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "p.pxl"
    path.write_text(SOURCE)
    return path


def lint(*argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["lint", *argv])
    return exit_info.value.code


def test_undefined_variable_is_reported_without_var(pipeline, capsys):
    assert lint(str(pipeline)) == 1
    assert "PX005" in capsys.readouterr().out


def test_var_option_defines_variables(pipeline, capsys):
    cli.main(["lint", str(pipeline), "--var", "w=10", "--var", "h=20"])
    out = capsys.readouterr().out
    assert "PX005" not in out
    assert "No issues found" in out


def test_only_the_supplied_variables_are_defined(pipeline, capsys):
    assert lint(str(pipeline), "--var", "w=10") == 1
    out = capsys.readouterr().out
    assert "$h" in out and "$w" not in out


def test_toml_variables_define_variables(pipeline, capsys):
    (pipeline.parent / "pixlang.toml").write_text("[variables]\nw = 10\nh = 20\n")
    cli.main(["lint", str(pipeline)])
    assert "PX005" not in capsys.readouterr().out


def test_var_and_toml_variables_combine(pipeline, capsys):
    (pipeline.parent / "pixlang.toml").write_text("[variables]\nw = 10\n")
    cli.main(["lint", str(pipeline), "--var", "h=20"])
    assert "PX005" not in capsys.readouterr().out


def test_bad_var_spec_is_rejected(pipeline, capsys):
    assert lint(str(pipeline), "--var", "nokey") == 1
    assert "KEY=VALUE" in capsys.readouterr().err


def test_validate_accepts_var_and_rejects_bad_spec(pipeline, capsys):
    cli.main(["validate", str(pipeline), "--var", "w=1"])
    assert "Syntax OK" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["validate", str(pipeline), "--var", "nokey"])
    assert "KEY=VALUE" in capsys.readouterr().err
