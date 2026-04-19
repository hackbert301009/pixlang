# tests/test_parser.py
"""
Unit tests for the PixLang parser layer.
"""
import pytest
from pixlang.parser import parse
from pixlang.parser.lexer import tokenize


# ── Lexer tests ───────────────────────────────────────────────────────────────

def test_tokenize_basic():
    tokens = tokenize('LOAD "image.png"')
    assert len(tokens) == 2
    assert tokens[0].type == "COMMAND"
    assert tokens[0].value == "LOAD"
    assert tokens[1].type == "STRING"
    assert tokens[1].value == '"image.png"'


def test_tokenize_numbers():
    tokens = tokenize("RESIZE 640 480")
    assert tokens[1].type == "INT"
    assert tokens[1].value == "640"


def test_tokenize_float():
    tokens = tokenize("SCALE 0.5")
    assert tokens[1].type == "FLOAT"


def test_tokenize_ignores_comments():
    tokens = tokenize("# this is a comment\nGRAYSCALE")
    assert len(tokens) == 1
    assert tokens[0].value == "GRAYSCALE"


def test_tokenize_multiline():
    src = "LOAD \"a.png\"\nRESIZE 100 200\nGRAYSCALE"
    tokens = tokenize(src)
    cmds = [t for t in tokens if t.type == "COMMAND"]
    assert [c.value for c in cmds] == ["LOAD", "RESIZE", "GRAYSCALE"]


# ── Parser tests ──────────────────────────────────────────────────────────────

def test_parse_single_command():
    pipeline = parse("GRAYSCALE")
    assert len(pipeline.commands) == 1
    assert pipeline.commands[0].name == "GRAYSCALE"
    assert pipeline.commands[0].args == []


def test_parse_load():
    pipeline = parse('LOAD "photo.jpg"')
    cmd = pipeline.commands[0]
    assert cmd.name == "LOAD"
    assert cmd.args == ["photo.jpg"]   # quotes stripped


def test_parse_resize_args():
    pipeline = parse("RESIZE 640 480")
    cmd = pipeline.commands[0]
    assert cmd.name == "RESIZE"
    assert cmd.args == [640, 480]
    assert all(isinstance(a, int) for a in cmd.args)


def test_parse_float_arg():
    pipeline = parse("BLUR 2.5")
    assert pipeline.commands[0].args == [2.5]


def test_parse_multi_command_pipeline():
    src = """
LOAD "img.png"
RESIZE 640 480
GRAYSCALE
THRESHOLD 128
SAVE "out.png"
"""
    pipeline = parse(src)
    names = [c.name for c in pipeline.commands]
    assert names == ["LOAD", "RESIZE", "GRAYSCALE", "THRESHOLD", "SAVE"]


def test_parse_with_comments():
    src = """
# load the source
LOAD "a.png"
# convert
GRAYSCALE
"""
    pipeline = parse(src)
    assert len(pipeline.commands) == 2


def test_parse_line_numbers():
    src = "LOAD \"a.png\"\n\nGRAYSCALE"
    pipeline = parse(src)
    assert pipeline.commands[0].line == 1
    assert pipeline.commands[1].line == 3


def test_parse_unknown_token_raises():
    with pytest.raises(SyntaxError):
        parse("load 'lowercase'")  # lowercase COMMAND is invalid


# ── Registry tests ────────────────────────────────────────────────────────────

def test_registry_has_core_commands():
    from pixlang.commands import registry
    for cmd in ["LOAD", "SAVE", "RESIZE", "GRAYSCALE", "THRESHOLD",
                "BLUR", "CANNY", "FIND_CONTOURS", "DRAW_BOUNDING_BOXES"]:
        assert registry.get(cmd) is not None


def test_registry_unknown_raises():
    from pixlang.commands import registry
    with pytest.raises(NameError):
        registry.get("UNKNOWN_COMMAND_XYZ")
