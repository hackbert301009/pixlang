"""`run` / `batch` reject unknown commands and link errors before executing anything."""
import cv2
import numpy as np
import pytest

from pixlang import cli


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def fails(*argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(list(argv))
    return exit_info.value.code


def test_unknown_command_is_reported_before_the_first_command_runs(workdir, capsys):
    (workdir / "p.pxl").write_text('LOAD "missing.png"\nTRESHOLD 128\n')
    assert fails("run", "p.pxl") == 1
    err = capsys.readouterr().err
    assert "p.pxl:2:1" in err
    assert "Unknown command 'TRESHOLD'" in err
    assert "THRESHOLD" in err
    assert "missing.png" not in err


def test_every_unknown_command_is_listed(workdir, capsys):
    (workdir / "p.pxl").write_text('LOAD "missing.png"\nBLURR 3\nTRESHOLD 1\n')
    assert fails("run", "p.pxl") == 1
    err = capsys.readouterr().err
    assert "p.pxl:2:1" in err and "p.pxl:3:1" in err


def test_unknown_command_inside_a_block_and_a_func(workdir, capsys):
    (workdir / "p.pxl").write_text(
        'LOAD "missing.png"\nFUNC HELPER\n  BLURR 3\nENDFUNC\n'
        'IF 1 == 1\n  GRAYSCAL\nENDIF\n')
    assert fails("run", "p.pxl") == 1
    err = capsys.readouterr().err
    assert "p.pxl:3:3" in err and "p.pxl:6:3" in err


def test_link_error_is_reported_before_execution(workdir, capsys):
    (workdir / "p.pxl").write_text('LOAD "missing.png"\nIMPORT "nothing_here.pxl"\n')
    assert fails("run", "p.pxl") == 1
    err = capsys.readouterr().err
    assert "nothing_here.pxl" in err
    assert "missing.png" not in err


def test_no_preflight_restores_run_time_discovery(workdir, capsys):
    (workdir / "p.pxl").write_text('LOAD "missing.png"\nTRESHOLD 128\n')
    assert fails("run", "p.pxl", "--no-preflight") == 1
    assert "missing.png" in capsys.readouterr().err


def test_valid_pipeline_still_runs(workdir, capsys):
    cv2.imwrite(str(workdir / "in.png"), np.full((8, 8), 90, np.uint8))
    (workdir / "p.pxl").write_text('LOAD "in.png"\nTHRESHOLD 50\nSAVE "out.png"\n')
    cli.main(["run", "p.pxl"])
    assert (workdir / "out.png").exists()


def test_batch_runs_the_preflight_too(workdir, capsys):
    (workdir / "p.pxl").write_text('LOAD_GLOB "in/*.png"\nTRESHOLD 1\n')
    assert fails("batch", "p.pxl") == 1
    assert "p.pxl:2:1" in capsys.readouterr().err


def test_user_funcs_and_variables_are_not_flagged(workdir, capsys):
    cv2.imwrite(str(workdir / "in.png"), np.full((8, 8), 90, np.uint8))
    (workdir / "p.pxl").write_text(
        'LOAD "in.png"\nFUNC SHRINK\n  RESIZE 4 4\nENDFUNC\nSHRINK\nSAVE "out.png"\n')
    cli.main(["run", "p.pxl"])
    assert cv2.imread(str(workdir / "out.png")).shape[:2] == (4, 4)
