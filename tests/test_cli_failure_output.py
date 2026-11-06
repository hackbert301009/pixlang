"""How `pixlang run` reports failures: readable message, traceback only for bugs."""
import cv2
import numpy as np
import pytest

from pixlang import cli


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "in.png"), np.full((20, 30, 3), 90, np.uint8))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run_cli(workdir, source, *flags):
    (workdir / "p.pxl").write_text(source)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "p.pxl", *flags])
    return exit_info.value.code


@pytest.mark.parametrize("source,expected", [
    ('LOAD "in.png"\nASSERT width == 99 "wrong width"\n', "wrong width"),
    ('LOAD "in.png"\nSET n 1\nSET n ($n / 0)\n', "division by zero"),
    ('LOAD "in.png"\nIF 1 == 1\n', "ENDIF"),
    ('LOAD "in.png"\nIMPORT "missing.pxl"\n', "missing.pxl"),
])
def test_expected_failures_print_no_traceback_in_verbose_mode(workdir, capsys, source, expected):
    code = run_cli(workdir, source, "--verbose")
    err = capsys.readouterr().err
    assert code == 1
    assert expected in err
    assert "Traceback" not in err


def test_failed_assert_without_verbose_prints_message_only(workdir, capsys):
    code = run_cli(workdir, 'LOAD "in.png"\nASSERT width == 99\n')
    err = capsys.readouterr().err
    assert code == 1
    assert "ASSERT" in err and "Traceback" not in err


def test_unexpected_internal_error_keeps_traceback_in_verbose_mode(
        workdir, capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyError("internal bug")

    monkeypatch.setattr("pixlang.executor.Executor.run", boom)
    code = run_cli(workdir, 'LOAD "in.png"\n', "--verbose")
    err = capsys.readouterr().err
    assert code == 1
    assert "Traceback" in err and "internal bug" in err


def test_unexpected_internal_error_hides_traceback_without_verbose(
        workdir, capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyError("internal bug")

    monkeypatch.setattr("pixlang.executor.Executor.run", boom)
    code = run_cli(workdir, 'LOAD "in.png"\n')
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err
