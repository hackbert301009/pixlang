# tests/test_cmd_plugins.py
"""Plugin loader: opt-in trust, idempotence/reload, rollback, visible errors."""
import os
import textwrap
import time

import pytest

from pixlang.commands.builtin import register_all
from pixlang.commands.registry import CommandRegistry
from pixlang.plugins import PluginLoader, add_trusted, config_trust_request, trust_from_config


@pytest.fixture
def reg():
    r = CommandRegistry()
    register_all(r)
    return r


def plugin_src(*names, fail=False, source="p"):
    body = "".join(
        f"    @registry.register({n!r}, source={source!r})\n"
        f"    def _c{i}(image):\n        return image\n"
        for i, n in enumerate(names)
    )
    if fail:
        body += "    raise RuntimeError('boom')\n"
    return "def register(registry):\n" + body


def _touch_later(path, text):
    path.write_text(text)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def test_local_plugin_not_loaded_without_trust(tmp_path, reg, capsys):
    pxl = tmp_path / "pipe.pxl"
    pxl.write_text("")
    (tmp_path / "pipe.plugins.py").write_text(plugin_src("LOCAL_X"))
    loader = PluginLoader(reg)
    assert loader.load_local(pxl) is None
    assert "LOCAL_X" not in reg
    err = capsys.readouterr().err
    assert "--trust-plugins" in err and "pipe.plugins.py" in err
    loader.load_local(pxl)
    assert capsys.readouterr().err == ""          # hint printed once


def test_local_plugin_loaded_with_trust(tmp_path, reg):
    pxl = tmp_path / "pipe.pxl"
    (tmp_path / "pipe.plugins.py").write_text(plugin_src("LOCAL_X"))
    m = PluginLoader(reg, trust_local=True).load_local(pxl)
    assert m.ok and m.commands == ["LOCAL_X"] and "LOCAL_X" in reg


def test_default_dir_needs_trust(tmp_path, reg, monkeypatch, capsys):
    monkeypatch.setenv("PIXLANG_PLUGIN_DIR", str(tmp_path))
    (tmp_path / "dirplug.py").write_text(plugin_src("DIR_X"))
    assert PluginLoader(reg).load_directory() == []
    assert "DIR_X" not in reg and "hint" in capsys.readouterr().err
    ms = PluginLoader(reg, trust_local=True).load_directory()
    assert ms[0].ok and "DIR_X" in reg


def test_loading_twice_is_idempotent(tmp_path, reg):
    pxl = tmp_path / "pipe.pxl"
    (tmp_path / "pipe.plugins.py").write_text(plugin_src("LOCAL_X"))
    loader = PluginLoader(reg, trust_local=True)
    m1 = loader.load_local(pxl)
    m2 = loader.load_local(pxl)
    assert m1 is m2 and m2.ok and len(loader.manifests) == 1


def test_changed_file_is_reloaded(tmp_path, reg):
    pxl = tmp_path / "pipe.pxl"
    plug = tmp_path / "pipe.plugins.py"
    plug.write_text(plugin_src("OLD_CMD"))
    loader = PluginLoader(reg, trust_local=True)
    loader.load_local(pxl)
    _touch_later(plug, plugin_src("NEW_CMD"))
    m = loader.load_local(pxl)
    assert m.ok and "NEW_CMD" in reg and "OLD_CMD" not in reg
    assert len(loader.manifests) == 1


def test_two_loaders_do_not_conflict_after_unload(tmp_path, reg):
    pxl = tmp_path / "pipe.pxl"
    (tmp_path / "pipe.plugins.py").write_text(plugin_src("LOCAL_X"))
    a = PluginLoader(reg, trust_local=True)
    a.load_local(pxl)
    a.unload_all()
    assert "LOCAL_X" not in reg
    assert PluginLoader(reg, trust_local=True).load_local(pxl).ok


def test_partial_registration_rolled_back(tmp_path, reg, capsys):
    (tmp_path / "bad.py").write_text(plugin_src("HALF_A", "HALF_B", fail=True))
    m = PluginLoader(reg).load_directory(tmp_path)[0]
    assert not m.ok and "boom" in m.error
    assert "HALF_A" not in reg and "HALF_B" not in reg
    assert "failed to load" in capsys.readouterr().err


def test_override_rollback_restores_builtin(tmp_path, reg):
    orig = reg.get("GRAYSCALE")
    (tmp_path / "ov.py").write_text(textwrap.dedent("""\
        def register(registry):
            @registry.register("GRAYSCALE", allow_override=True)
            def _g(image):
                return image
            raise ValueError("late failure")
    """))
    PluginLoader(reg).load_directory(tmp_path)
    assert reg.get("GRAYSCALE") is orig


def test_same_stem_in_two_dirs_gets_distinct_modules(tmp_path, reg):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "utils.py").write_text("VALUE = 'a'\n" + plugin_src("FROM_A"))
    (b / "utils.py").write_text("VALUE = 'b'\n" + plugin_src("FROM_B"))
    loader = PluginLoader(reg)
    loader.load_directory(a)
    loader.load_directory(b)
    assert "FROM_A" in reg and "FROM_B" in reg


def test_manifest_counts_overrides_via_origin(tmp_path, reg):
    (tmp_path / "ov.py").write_text(textwrap.dedent("""\
        def register(registry):
            @registry.register("BLUR", allow_override=True)
            def _b(image, k=1):
                return image
    """))
    m = PluginLoader(reg).load_directory(tmp_path)[0]
    assert m.commands == ["BLUR"]


def test_trust_from_config_needs_a_confirmed_workspace(tmp_path):
    assert config_trust_request(tmp_path) is None
    assert trust_from_config(tmp_path) is False

    cfg = tmp_path / "pixlang.toml"
    cfg.write_text("[plugins]\ntrust_local = true\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    # The toml asks for trust, but asking is not granting: it may have arrived
    # with the pipeline.
    assert config_trust_request(sub) == cfg
    assert trust_from_config(sub) is False

    add_trusted(tmp_path)
    assert trust_from_config(sub) is True


def test_reload_changed_helper(tmp_path, reg):
    plug = tmp_path / "p.py"
    plug.write_text(plugin_src("V1"))
    loader = PluginLoader(reg)
    loader.load_directory(tmp_path)
    time.sleep(0.01)
    _touch_later(plug, plugin_src("V2"))
    loader.reload_changed()
    assert "V2" in reg and "V1" not in reg
