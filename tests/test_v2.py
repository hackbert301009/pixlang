# tests/test_v2.py
"""
v0.2 test suite — covers:
  - Registry v2 (CommandInfo, ConflictError, sources, suggestions)
  - PluginLoader (local file, directory, entry-points no-op)
  - New commands: HEATMAP, DRAW_TEXT, CHECKPOINT, OVERLAY
  - Executor context injection (_ctx)
  - End-to-end pipeline with new commands
"""
import os
import textwrap
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pixlang.commands import registry as _base_registry
from pixlang.commands.registry import CommandRegistry, ConflictError, CommandInfo
from pixlang.commands.builtin import register_all, _wrap, _unwrap
from pixlang.plugins.loader import PluginLoader
from pixlang.executor import Executor
from pixlang.parser import parse


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_registry():
    """A fresh registry with all builtins loaded."""
    r = CommandRegistry()
    register_all(r)
    return r


def bgr(h=120, w=160):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def gray(h=120, w=160):
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Registry v2
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryV2:

    def test_command_info_has_source(self, fresh_registry):
        info = fresh_registry.get_info("GRAYSCALE")
        assert isinstance(info, CommandInfo)
        assert info.source == "builtin"
        assert info.name == "GRAYSCALE"

    def test_command_info_has_doc(self, fresh_registry):
        info = fresh_registry.get_info("RESIZE")
        assert info.doc  # non-empty first line of docstring

    def test_commands_by_source_groups(self, fresh_registry):
        by_src = fresh_registry.commands_by_source()
        assert "builtin" in by_src
        assert len(by_src["builtin"]) == len(fresh_registry)

    def test_conflict_error_on_duplicate(self, fresh_registry):
        """Registering an existing name without allow_override raises ConflictError."""
        with pytest.raises(ConflictError, match="GRAYSCALE"):
            @fresh_registry.register("GRAYSCALE")
            def _duplicate(image): pass

    def test_allow_override_succeeds(self, fresh_registry):
        called = []

        @fresh_registry.register("GRAYSCALE", allow_override=True, source="test")
        def _override(image):
            called.append(True)
            return image

        fn = fresh_registry.get("GRAYSCALE")
        fn(bgr())
        assert called

    def test_plugin_source_label(self, fresh_registry):
        @fresh_registry.register("MY_PLUGIN_CMD", source="my-plugin")
        def _cmd(image): return image

        info = fresh_registry.get_info("MY_PLUGIN_CMD")
        assert info.source == "my-plugin"

    def test_suggestions_on_typo(self, fresh_registry):
        """NameError message should include a 'Did you mean' hint for close typos."""
        with pytest.raises(NameError, match="GRAYSCAL"):
            fresh_registry.get("GRAYSCAL")

    def test_len(self, fresh_registry):
        assert len(fresh_registry) == 27  # 23 builtins + 4 new v0.2 commands

    def test_contains(self, fresh_registry):
        assert "HEATMAP" in fresh_registry
        assert "NONEXISTENT" not in fresh_registry


# ─────────────────────────────────────────────────────────────────────────────
# New command: HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

class TestHeatmap:

    def test_heatmap_returns_bgr(self, fresh_registry):
        fn = fresh_registry.get("HEATMAP")
        result = fn(gray(), "inferno")
        img, _ = _unwrap(result)
        assert len(img.shape) == 3  # BGR output
        assert img.shape[2] == 3

    def test_heatmap_on_bgr_input(self, fresh_registry):
        """HEATMAP should accept a colour image (converts to gray internally)."""
        fn = fresh_registry.get("HEATMAP")
        result = fn(bgr())
        img, _ = _unwrap(result)
        assert len(img.shape) == 3

    def test_heatmap_all_colormaps(self, fresh_registry):
        fn = fresh_registry.get("HEATMAP")
        for cmap in ["jet", "inferno", "plasma", "hot", "cool", "rainbow", "viridis", "bone"]:
            result = fn(gray(), cmap)
            img, _ = _unwrap(result)
            assert img.shape[2] == 3, f"Failed for colormap: {cmap}"

    def test_heatmap_invalid_colormap(self, fresh_registry):
        fn = fresh_registry.get("HEATMAP")
        with pytest.raises(ValueError, match="Unknown colormap"):
            fn(gray(), "neon_rainbow_unicorn")

    def test_heatmap_preserves_metadata(self, fresh_registry):
        """Upstream metadata (e.g. contours) must survive HEATMAP."""
        wrapped = _wrap(gray())
        wrapped["contours"] = ["fake_contour"]
        fn = fresh_registry.get("HEATMAP")
        result = fn(wrapped)
        _, meta = _unwrap(result)
        assert "contours" in meta

    def test_heatmap_no_image_raises(self, fresh_registry):
        with pytest.raises(RuntimeError, match="HEATMAP"):
            fresh_registry.get("HEATMAP")(None)


# ─────────────────────────────────────────────────────────────────────────────
# New command: DRAW_TEXT
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawText:

    def test_draw_text_returns_bgr(self, fresh_registry):
        fn = fresh_registry.get("DRAW_TEXT")
        result = fn(bgr(), "Hello", 10, 30)
        img, _ = _unwrap(result)
        assert img.shape[2] == 3

    def test_draw_text_on_gray_converts(self, fresh_registry):
        """DRAW_TEXT on a grayscale image should auto-convert to BGR."""
        fn = fresh_registry.get("DRAW_TEXT")
        result = fn(gray(), "test")
        img, _ = _unwrap(result)
        assert len(img.shape) == 3

    def test_draw_text_custom_color(self, fresh_registry):
        fn = fresh_registry.get("DRAW_TEXT")
        # Red text (R=255 G=0 B=0) at large scale — pixel values should change
        canvas = np.zeros((100, 200, 3), dtype=np.uint8)
        result = fn(canvas, "X", 10, 50, 2.0, 255, 0, 0)
        img, _ = _unwrap(result)
        # At least one pixel should be non-zero (text was drawn)
        assert img.max() > 0

    def test_draw_text_all_fonts(self, fresh_registry):
        fn = fresh_registry.get("DRAW_TEXT")
        for font in ["simplex", "plain", "duplex", "complex", "triplex", "small", "script"]:
            result = fn(bgr(), "A", 10, 30, 1.0, 255, 255, 255, 2, font)
            img, _ = _unwrap(result)
            assert img is not None

    def test_draw_text_invalid_font(self, fresh_registry):
        fn = fresh_registry.get("DRAW_TEXT")
        with pytest.raises(ValueError, match="Unknown font"):
            fn(bgr(), "oops", 10, 30, 1.0, 255, 255, 255, 2, "comic_sans")

    def test_draw_text_preserves_shape(self, fresh_registry):
        fn = fresh_registry.get("DRAW_TEXT")
        img_in = bgr(200, 300)
        result = fn(img_in, "label", 10, 30)
        img_out, _ = _unwrap(result)
        assert img_out.shape[:2] == (200, 300)

    def test_draw_text_no_image_raises(self, fresh_registry):
        with pytest.raises(RuntimeError):
            fresh_registry.get("DRAW_TEXT")(None, "hi")


# ─────────────────────────────────────────────────────────────────────────────
# New commands: CHECKPOINT + OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointOverlay:

    def _make_ctx(self):
        return {"checkpoints": {}}

    def test_checkpoint_stores_image(self, fresh_registry):
        ctx = self._make_ctx()
        fn = fresh_registry.get("CHECKPOINT")
        img = bgr()
        fn(img, "snap", _ctx=ctx)
        assert "snap" in ctx["checkpoints"]
        saved, _ = ctx["checkpoints"]["snap"]
        assert saved.shape == img.shape

    def test_checkpoint_default_name(self, fresh_registry):
        ctx = self._make_ctx()
        fn = fresh_registry.get("CHECKPOINT")
        fn(bgr(), _ctx=ctx)  # uses "default"
        assert "default" in ctx["checkpoints"]

    def test_checkpoint_does_not_mutate_image(self, fresh_registry):
        """CHECKPOINT is a passthrough — it returns the image unchanged."""
        ctx = self._make_ctx()
        fn = fresh_registry.get("CHECKPOINT")
        original = bgr()
        result = fn(original, "s", _ctx=ctx)
        assert result is original  # same object returned

    def test_checkpoint_stores_copy(self, fresh_registry):
        """Stored image must be a copy, not a reference."""
        ctx = self._make_ctx()
        fn = fresh_registry.get("CHECKPOINT")
        img = bgr()
        fn(img, "s", _ctx=ctx)
        saved_before = ctx["checkpoints"]["s"][0].copy()
        img[:] = 0  # mutate original
        np.testing.assert_array_equal(ctx["checkpoints"]["s"][0], saved_before)

    def test_overlay_blends(self, fresh_registry):
        ctx = self._make_ctx()
        cp_fn = fresh_registry.get("CHECKPOINT")
        ov_fn = fresh_registry.get("OVERLAY")

        white = np.full((50, 50, 3), 255, dtype=np.uint8)
        black = np.zeros((50, 50, 3), dtype=np.uint8)

        cp_fn(white, "snap", _ctx=ctx)
        result = ov_fn(black, "snap", 0.5, _ctx=ctx)
        img, _ = _unwrap(result)
        # 50% blend of 255 and 0 → ~127 (allow rounding)
        assert 100 < img.mean() < 160

    def test_overlay_alpha_extremes(self, fresh_registry):
        ctx = self._make_ctx()
        cp_fn = fresh_registry.get("CHECKPOINT")
        ov_fn = fresh_registry.get("OVERLAY")

        red   = np.zeros((40, 40, 3), dtype=np.uint8); red[:, :, 2] = 255
        blue  = np.zeros((40, 40, 3), dtype=np.uint8); blue[:, :, 0] = 255

        cp_fn(red, "r", _ctx=ctx)
        # alpha=1.0 → result should be all red
        result = ov_fn(blue, "r", 1.0, _ctx=ctx)
        img, _ = _unwrap(result)
        assert img[:, :, 2].mean() > 200  # red channel dominant

    def test_overlay_missing_checkpoint(self, fresh_registry):
        ctx = self._make_ctx()
        fn = fresh_registry.get("OVERLAY")
        with pytest.raises(RuntimeError, match="No checkpoint named"):
            fn(bgr(), "ghost", 0.5, _ctx=ctx)

    def test_overlay_resizes_mismatched_checkpoint(self, fresh_registry):
        """OVERLAY should resize the checkpoint to match the current image."""
        ctx = self._make_ctx()
        cp_fn = fresh_registry.get("CHECKPOINT")
        ov_fn = fresh_registry.get("OVERLAY")

        cp_fn(bgr(100, 100), "big", _ctx=ctx)
        result = ov_fn(bgr(50, 60), "big", 0.5, _ctx=ctx)
        img, _ = _unwrap(result)
        assert img.shape[:2] == (50, 60)

    def test_checkpoint_no_ctx_raises(self, fresh_registry):
        with pytest.raises(RuntimeError, match="context"):
            fresh_registry.get("CHECKPOINT")(bgr(), "s", _ctx=None)

    def test_overlay_no_ctx_raises(self, fresh_registry):
        with pytest.raises(RuntimeError, match="context"):
            fresh_registry.get("OVERLAY")(bgr(), "s", 0.5, _ctx=None)


# ─────────────────────────────────────────────────────────────────────────────
# Plugin Loader
# ─────────────────────────────────────────────────────────────────────────────

class TestPluginLoader:

    def _make_plugin_source(self, cmd_name="TEST_CMD_PLUGIN"):
        return textwrap.dedent(f"""\
            def register(registry):
                @registry.register("{cmd_name}", source="test-plugin")
                def _cmd(image):
                    "Test plugin command."
                    return image
        """)

    def test_load_local_plugin_file(self, tmp_path, fresh_registry):
        # Create a fake pipeline path and a sibling .plugins.py
        pipeline_file = tmp_path / "my_pipe.pxl"
        pipeline_file.write_text("GRAYSCALE\n")
        plugin_file = tmp_path / "my_pipe.plugins.py"
        plugin_file.write_text(self._make_plugin_source("LOCAL_CMD"))

        loader = PluginLoader(fresh_registry)
        manifest = loader.load_local(pipeline_file)

        assert manifest is not None
        assert manifest.ok
        assert "LOCAL_CMD" in manifest.commands
        assert "LOCAL_CMD" in fresh_registry

    def test_load_directory_plugins(self, tmp_path, fresh_registry):
        plugin_file = tmp_path / "my_plugin.py"
        plugin_file.write_text(self._make_plugin_source("DIR_CMD"))

        loader = PluginLoader(fresh_registry)
        manifests = loader.load_directory(tmp_path)

        assert len(manifests) == 1
        assert manifests[0].ok
        assert "DIR_CMD" in fresh_registry

    def test_directory_skips_underscore_files(self, tmp_path, fresh_registry):
        (tmp_path / "_internal.py").write_text(self._make_plugin_source("SKIP_CMD"))
        loader = PluginLoader(fresh_registry)
        manifests = loader.load_directory(tmp_path)
        assert len(manifests) == 0

    def test_broken_plugin_records_error(self, tmp_path, fresh_registry):
        broken = tmp_path / "broken.py"
        broken.write_text("def register(registry):\n    raise RuntimeError('oops')\n")

        loader = PluginLoader(fresh_registry)
        manifests = loader.load_directory(tmp_path)

        assert len(manifests) == 1
        assert not manifests[0].ok
        assert "oops" in manifests[0].error

    def test_plugin_without_register_fn_errors(self, tmp_path, fresh_registry):
        no_register = tmp_path / "no_fn.py"
        no_register.write_text("x = 42\n")

        loader = PluginLoader(fresh_registry)
        manifests = loader.load_directory(tmp_path)

        assert not manifests[0].ok

    def test_conflict_detection_in_plugin(self, tmp_path, fresh_registry):
        """A plugin registering an existing builtin name should be caught."""
        conflict_plugin = tmp_path / "conflict.py"
        conflict_plugin.write_text(textwrap.dedent("""\
            def register(registry):
                @registry.register("GRAYSCALE")   # already a builtin!
                def _cmd(image): return image
        """))

        loader = PluginLoader(fresh_registry)
        manifests = loader.load_directory(tmp_path)

        assert not manifests[0].ok
        assert "GRAYSCALE" in manifests[0].error

    def test_plugin_with_allow_override(self, tmp_path, fresh_registry):
        """A plugin using allow_override=True should succeed."""
        override_plugin = tmp_path / "override.py"
        override_plugin.write_text(textwrap.dedent("""\
            def register(registry):
                @registry.register("GRAYSCALE", allow_override=True, source="test-override")
                def _cmd(image): return image
        """))

        loader = PluginLoader(fresh_registry)
        manifests = loader.load_directory(tmp_path)
        # override replaces existing — diff is empty (no net new command)
        assert manifests[0].ok

    def test_load_entrypoints_no_crash(self, fresh_registry):
        """Entry-point discovery should run without crashing even with no plugins."""
        loader = PluginLoader(fresh_registry)
        result = loader.load_entrypoints()
        assert isinstance(result, list)

    def test_loader_summary_format(self, tmp_path, fresh_registry):
        plugin_file = tmp_path / "summary_test.py"
        plugin_file.write_text(self._make_plugin_source("SUMMARY_CMD"))
        loader = PluginLoader(fresh_registry)
        loader.load_directory(tmp_path)
        summary = loader.summary()
        assert "summary_test" in summary
        assert "SUMMARY_CMD" in summary

    def test_nonexistent_directory_returns_empty(self, fresh_registry):
        loader = PluginLoader(fresh_registry)
        result = loader.load_directory(Path("/nonexistent/dir/xyz"))
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Executor context injection
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutorContext:

    def test_executor_injects_ctx_to_checkpoint(self):
        """CHECKPOINT in a pipeline should have access to executor context."""
        from pixlang.commands import registry
        pipeline = parse('\n'.join([
            'LOAD "examples/sample.jpg"',
            'GRAYSCALE',
            'CHECKPOINT "before_thresh"',
            'THRESHOLD_OTSU',
            'OVERLAY "before_thresh" 0.4',
            'SAVE "examples/overlay_test.png"',
        ]))
        executor = Executor(registry=registry, verbose=False)
        executor.run(pipeline)

        out = Path("examples/overlay_test.png")
        assert out.exists()
        out.unlink()

    def test_heatmap_end_to_end(self):
        """HEATMAP pipeline runs fully and produces a file."""
        from pixlang.commands import registry
        pipeline = parse('\n'.join([
            'LOAD "examples/sample.jpg"',
            'GRAYSCALE',
            'HEATMAP "inferno"',
            'DRAW_TEXT "HEAT MAP" 10 30 1.0 255 255 255',
            'SAVE "examples/heatmap_test.png"',
        ]))
        executor = Executor(registry=registry, verbose=False)
        executor.run(pipeline)

        out = Path("examples/heatmap_test.png")
        assert out.exists()
        out.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Example plugin: pixlang-denoise
# ─────────────────────────────────────────────────────────────────────────────

class TestDenoisePlugin:

    @pytest.fixture
    def denoise_registry(self):
        r = CommandRegistry()
        register_all(r)
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "plugins_example"))
        from pixlang_denoise.plugin import register
        register(r)
        return r

    def test_bilateral_registered(self, denoise_registry):
        assert "BILATERAL" in denoise_registry
        info = denoise_registry.get_info("BILATERAL")
        assert info.source == "pixlang-denoise"

    def test_nlm_denoise_registered(self, denoise_registry):
        assert "NLM_DENOISE" in denoise_registry

    def test_bilateral_preserves_shape(self, denoise_registry):
        fn = denoise_registry.get("BILATERAL")
        img = bgr(80, 80)
        result = fn(img, 9, 75, 75)
        out, _ = _unwrap(result)
        assert out.shape == img.shape

    def test_nlm_denoise_on_gray(self, denoise_registry):
        fn = denoise_registry.get("NLM_DENOISE")
        img = gray(60, 60)
        result = fn(img, 10.0, 7, 21)
        out, _ = _unwrap(result)
        assert len(out.shape) == 2

    def test_plugin_source_isolated_from_builtins(self, denoise_registry):
        by_src = denoise_registry.commands_by_source()
        assert "pixlang-denoise" in by_src
        assert "builtin" in by_src
        plugin_names = [i.name for i in by_src["pixlang-denoise"]]
        assert "BILATERAL" in plugin_names
        assert "GRAYSCALE" not in plugin_names
