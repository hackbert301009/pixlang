# tests/test_cmd_registry.py
"""Registry: suggestions, UnknownCommandError, arity, coercion, metadata, unregister."""
import numpy as np
import pytest

from pixlang.commands.registry import (
    ArgumentError,
    CommandRegistry,
    ConflictError,
    UnknownCommandError,
)


@pytest.fixture
def reg():
    r = CommandRegistry()

    @r.register("THRESHOLD", category="Threshold")
    def _thr(image, value: int, mode: str = "binary"):
        """THRESHOLD <value> [mode]  — binary threshold."""
        return (image, value, mode)

    @r.register("SCALE")
    def _scale(image, f: float, flag: bool = False, _ctx: dict = None):
        """SCALE f — scale."""
        return (f, flag, _ctx)

    @r.register("ANY")
    def _any(image, *vals):
        """ANY ... — variadic."""
        return vals

    return r


def test_unknown_command_is_nameerror_with_suggestion(reg):
    with pytest.raises(UnknownCommandError) as ei:
        reg.get("TRESHOLD")
    assert isinstance(ei.value, NameError)
    assert "THRESHOLD" in str(ei.value)
    assert ei.value.suggestions == ["THRESHOLD"]


def test_get_info_unknown_raises_nameerror(reg):
    with pytest.raises(NameError):
        reg.get_info("NOPE")


def test_invoke_coerces_int(reg):
    assert reg.invoke("THRESHOLD", None, ["5"]) == (None, 5, "binary")
    assert reg.invoke("THRESHOLD", None, [5.0]) == (None, 5, "binary")


def test_invoke_rejects_fractional_int(reg):
    with pytest.raises(ArgumentError, match=r"THRESHOLD.*value.*int.*5\.5"):
        reg.invoke("THRESHOLD", None, [5.5])


def test_invoke_rejects_garbage(reg):
    with pytest.raises(ArgumentError, match="value"):
        reg.invoke("THRESHOLD", None, ["abc"])


def test_invoke_coerces_str_float_bool(reg):
    f, flag, ctx = reg.invoke("SCALE", None, [2, "yes"], {"x": 1})
    assert f == 2.0 and isinstance(f, float) and flag is True and ctx == {"x": 1}
    assert reg.invoke("THRESHOLD", None, [1, 7])[2] == "7"


def test_arity_errors_show_usage(reg):
    with pytest.raises(ArgumentError, match="Usage: THRESHOLD <value:int>"):
        reg.invoke("THRESHOLD", None, [])
    with pytest.raises(ArgumentError, match="1-2"):
        reg.invoke("THRESHOLD", None, [1, "binary", 3])


def test_variadic(reg):
    assert reg.invoke("ANY", None, [1, 2, 3]) == (1, 2, 3)
    assert reg.get_info("ANY").arity == (0, None)


def test_metadata(reg):
    info = reg.get_info("THRESHOLD")
    assert info.usage == "THRESHOLD <value:int> [mode:str=binary]"
    assert info.params == [("value", "int", None), ("mode", "str", "binary")]
    assert info.category == "Threshold"
    assert info.summary == "binary threshold."


def test_invoke_carries_meta(reg):
    @reg.register("ID")
    def _id(image):
        return np.zeros((2, 2), np.uint8)

    out = reg.invoke("ID", {"_image": np.ones((2, 2), np.uint8), "contours": [1]}, [])
    assert isinstance(out, dict) and out["contours"] == [1]


def test_unregister_source_restores_shadowed(reg):
    orig = reg.get("THRESHOLD")
    with reg.origin("local:/x.py"):
        @reg.register("THRESHOLD", source="mine", allow_override=True)
        def _t(image, v):
            return 1

        @reg.register("NEWCMD", source="mine")
        def _n(image):
            return 2
    assert reg.get_info("NEWCMD").origin == "local:/x.py"
    removed = reg.unregister_source("local:/x.py")
    assert removed == ["NEWCMD", "THRESHOLD"]
    assert reg.get("THRESHOLD") is orig
    assert "NEWCMD" not in reg


def test_conflict_still_raises(reg):
    with pytest.raises(ConflictError):
        @reg.register("THRESHOLD")
        def _x(image):
            return image


def test_global_registry_suggests():
    from pixlang.commands import registry
    with pytest.raises(NameError, match="THRESHOLD"):
        registry.get("TRESHOLD")
