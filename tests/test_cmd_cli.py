# tests/test_cmd_cli.py
"""CLI: stderr errors, no ANSI off-TTY, --var, plugin trust, scaffold, commands."""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def cli(*args, cwd, env_extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("FORCE_COLOR", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "pixlang.cli", *args], cwd=cwd,
                          capture_output=True, text=True, env=env)


@pytest.fixture
def img(tmp_path):
    cv2.imwrite(str(tmp_path / "in.png"), np.full((20, 30, 3), 90, np.uint8))
    return tmp_path / "in.png"


def test_no_ansi_and_no_banner_when_piped(tmp_path, img):
    (tmp_path / "p.pxl").write_text('LOAD "in.png"\nSAVE "out.png"\n')
    r = cli("run", "p.pxl", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "\033[" not in r.stdout and "██" not in r.stdout
    assert (tmp_path / "out.png").exists()


def test_errors_go_to_stderr(tmp_path, img):
    (tmp_path / "p.pxl").write_text('LOAD "in.png"\nTRESHOLD 5\n')
    r = cli("run", "p.pxl", cwd=tmp_path)
    assert r.returncode == 1
    assert "THRESHOLD" in r.stderr and "Error" not in r.stdout


def test_syntax_error_reported_cleanly(tmp_path):
    (tmp_path / "p.pxl").write_text("@@@\n")
    r = cli("run", "p.pxl", cwd=tmp_path)
    assert r.returncode == 1 and "Traceback" not in r.stderr
    assert "p.pxl" in r.stderr


def test_var_overrides(tmp_path, img):
    (tmp_path / "pixlang.toml").write_text("[variables]\nw = 5\n")
    (tmp_path / "p.pxl").write_text('LOAD "in.png"\nRESIZE $w $h\nSAVE "o.png"\n')
    r = cli("run", "p.pxl", "--var", "w=12", "--var", "h=7", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert cv2.imread(str(tmp_path / "o.png")).shape[:2] == (7, 12)


def test_bad_var_spec(tmp_path, img):
    (tmp_path / "p.pxl").write_text('LOAD "in.png"\n')
    r = cli("run", "p.pxl", "--var", "nokey", cwd=tmp_path)
    assert r.returncode == 1 and "KEY=VALUE" in r.stderr


PLUGIN = ("def register(registry):\n"
          "    @registry.register('LOCALCMD', source='t')\n"
          "    def _c(image):\n        return image\n")


def test_local_plugin_requires_trust(tmp_path, img):
    (tmp_path / "p.pxl").write_text('LOAD "in.png"\nLOCALCMD\n')
    (tmp_path / "p.plugins.py").write_text(PLUGIN)
    r = cli("run", "p.pxl", cwd=tmp_path)
    assert r.returncode == 1
    assert "--trust-plugins" in r.stderr
    r = cli("run", "p.pxl", "--trust-plugins", cwd=tmp_path)
    assert r.returncode == 0, r.stderr


def test_bundled_toml_alone_does_not_grant_trust(tmp_path, img):
    """A downloaded bundle ships pipeline + plugin + pixlang.toml. It must not run."""
    (tmp_path / "p.pxl").write_text('LOAD "in.png"\nLOCALCMD\n')
    (tmp_path / "p.plugins.py").write_text(PLUGIN)
    (tmp_path / "pixlang.toml").write_text("[plugins]\ntrust_local = true\n")

    r = cli("run", "p.pxl", cwd=tmp_path)
    assert r.returncode == 1
    assert "pixlang trust" in r.stderr and "not a confirmed workspace" in r.stderr

    assert cli("trust", str(tmp_path), cwd=tmp_path).returncode == 0
    assert cli("run", "p.pxl", cwd=tmp_path).returncode == 0


def test_trust_list_and_remove(tmp_path, img):
    (tmp_path / "p.pxl").write_text('LOAD "in.png"\nLOCALCMD\n')
    (tmp_path / "p.plugins.py").write_text(PLUGIN)
    (tmp_path / "pixlang.toml").write_text("[plugins]\ntrust_local = true\n")

    cli("trust", ".", cwd=tmp_path)
    assert str(tmp_path.resolve()) in cli("trust", "--list", cwd=tmp_path).stdout
    assert cli("run", "p.pxl", cwd=tmp_path).returncode == 0

    assert cli("trust", "--remove", str(tmp_path), cwd=tmp_path).returncode == 0
    assert cli("run", "p.pxl", cwd=tmp_path).returncode == 1
    assert "No confirmed workspaces" in cli("trust", "--list", cwd=tmp_path).stdout


def test_batch_subcommand_and_mode(tmp_path):
    (tmp_path / "in").mkdir()
    for i in range(2):
        cv2.imwrite(str(tmp_path / "in" / f"{i}.png"), np.full((4, 4), i * 50, np.uint8))
    (tmp_path / "p.pxl").write_text('LOAD_GLOB "in/*.png"\nSAVE_EACH "o/{stem}.png"\n')
    r = cli("batch", "p.pxl", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == ["0.png", "1.png"]
    (tmp_path / "q.pxl").write_text('LOAD "in/0.png"\n')
    assert cli("batch", "q.pxl", cwd=tmp_path).returncode == 1


def test_new_scaffold_runs_as_printed(tmp_path):
    r = cli("new", "Demo Project", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    proj = tmp_path / "demo_project"
    assert "cd demo_project" in r.stdout
    r = cli("run", "pipeline.pxl", cwd=proj)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (proj / "output" / "result.png").exists()


def test_commands_shows_usage_and_category(tmp_path):
    r = cli("commands", cwd=tmp_path)
    assert "[Threshold]" in r.stdout
    assert "THRESHOLD <value:float> [mode:str=binary]" in r.stdout
    r = cli("commands", "--category", "bitwise", cwd=tmp_path)
    assert "BITWISE_AND" in r.stdout and "BLUR" not in r.stdout
