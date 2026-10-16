# tests/test_v5_memory.py
"""
v0.5 language-core tests, part 4: buffers, pointers, low-level memory, loops.
"""
import textwrap

import numpy as np
import pytest

from pixlang.commands import registry as default_registry
from pixlang.executor import Executor
from pixlang.executor.memory import Ref
from pixlang.parser import PixRuntimeError, PixSyntaxError, parse
from pixlang.parser.ast_nodes import (
    AllocStmt,
    BufStmt,
    ForBlock,
    FreeStmt,
    RepeatBlock,
    UseStmt,
    WhileBlock,
    WithBlock,
)


def run(src, **kw):
    ex = Executor(registry=default_registry, **kw)
    ex.run(parse(textwrap.dedent(src)))
    return ex


def buf(ex, name):
    entry = ex.context["checkpoints"][name]
    return entry[0] if isinstance(entry, tuple) else entry


# ═════════════════════════════════════════════════════════════════════════════
# ALLOC / BUF / USE / FREE
# ═════════════════════════════════════════════════════════════════════════════

class TestBuffers:

    def test_parse_nodes(self):
        p = parse("ALLOC c 640 480 3 u16 7\nBUF o\nBUF o2 $x\nUSE c\nFREE a, b")
        a, b, b2, u, f = p.commands
        assert isinstance(a, AllocStmt) and (a.name, a.width, a.height, a.channels, a.dtype, a.fill) \
            == ("c", 640, 480, 3, "u16", 7)
        assert isinstance(b, BufStmt) and b.value is None and b2.value is not None
        assert isinstance(u, UseStmt) and isinstance(f, FreeStmt) and f.names == ["a", "b"]

    def test_alloc_defaults(self):
        a = parse("ALLOC c 4 3").commands[0]
        assert (a.channels, a.dtype, a.fill) == (1, "u8", 0)
        a = parse("ALLOC c 4 3 f32").commands[0]
        assert (a.channels, a.dtype) == (1, "f32")
        a = parse("ALLOC c 4 3 f32 0.5").commands[0]
        assert (a.dtype, a.fill) == ("f32", 0.5)

    @pytest.mark.parametrize("dt,np_dt", [
        ("u8", np.uint8), ("i8", np.int8), ("u16", np.uint16), ("i16", np.int16),
        ("i32", np.int32), ("u32", np.uint32), ("f32", np.float32), ("f64", np.float64)])
    def test_alloc_dtypes(self, dt, np_dt):
        ex = run(f"ALLOC c 5 4 3 {dt} 1")
        b = buf(ex, "c")
        assert b.dtype == np_dt and b.shape == (4, 5, 3) and (b == 1).all()

    def test_alloc_gray_is_2d(self):
        assert buf(run("ALLOC c 5 4"), "c").shape == (4, 5)

    def test_alloc_errors(self):
        with pytest.raises(PixRuntimeError, match="dimensions must be > 0"):
            run("ALLOC c 0 4")
        with pytest.raises(PixRuntimeError, match="must be an integer"):
            run("ALLOC c 4.5 4")
        with pytest.raises(PixSyntaxError, match="width and height"):
            parse("ALLOC c 4")

    def test_buf_copies_current_image(self):
        ex = run("""\
            USE ZEROS(4, 3)
            BUF snap
            USE ZEROS(2, 2)
        """)
        assert buf(ex, "snap").shape == (3, 4) and ex.image.shape == (2, 2)

    def test_buf_is_a_copy(self):
        ex = run("USE ZEROS(3, 3)\nBUF a\nSET a[0, 0] 9\nBUF b a\nSET a[1, 1] 5\nSET x b[1, 1]")
        assert ex.context["vars"]["x"] == 0

    def test_buf_without_image(self):
        with pytest.raises(PixRuntimeError, match="No image loaded"):
            run("BUF a")

    def test_use_buffer_copy(self):
        ex = run("ALLOC c 3 2 1 u8 5\nUSE c\nINVERT\nSET v c[0, 0]")
        assert ex.image.max() == 250 and ex.context["vars"]["v"] == 5

    def test_use_unknown_name(self):
        with pytest.raises(PixRuntimeError, match="no buffer or variable named 'ghost'"):
            run("USE ghost")

    def test_current_image_register(self):
        ex = run("USE ZEROS(4, 2)\nSET w _.width\nSET h _.height\nSET m max(_)")
        assert ex.context["vars"] == {"w": 4, "h": 2, "m": 0}
        with pytest.raises(PixRuntimeError, match="current image"):
            run("SET x _.width")

    def test_set_underscore_assigns_current_image(self):
        ex = run("ALLOC c 3 2 1 u8 9\nSET _ c")
        assert ex.image.shape == (2, 3) and ex.image.max() == 9

    def test_shares_checkpoint_store(self):
        """BUF slots are CHECKPOINT slots: OVERLAY / COMPARE / BLEND see them."""
        ex = run("""\
            ALLOC base 8 8 3 u8 100
            USE base
            BUF saved
            CHECKPOINT "cp"
            SET w saved.width
            ASSERT w == 8
            OVERLAY "saved" 0.5
            COMPARE "cp"
        """)
        assert ex.context["checkpoints"] is ex.context["buffers"]
        assert "saved" in ex.context["checkpoints"] and "cp" in ex.context["checkpoints"]

    def test_checkpoint_visible_as_buffer(self):
        ex = run('USE ZEROS(4, 4, 3)\nCHECKPOINT "cp"\nSET w cp.width\nSET v cp[0, 0, 0]')
        assert ex.context["vars"] == {"w": 4, "v": 0}

    def test_free(self):
        ex = run("ALLOC a 2 2\nFREE a")
        assert "a" not in ex.context["checkpoints"]

    def test_double_free_and_missing(self):
        with pytest.raises(PixRuntimeError, match="double free of buffer 'a' .*line 2"):
            run("ALLOC a 2 2\nFREE a\nFREE a")
        with pytest.raises(PixRuntimeError, match="no buffer named 'zz'"):
            run("FREE zz")

    def test_use_after_free_by_name(self):
        with pytest.raises(PixRuntimeError, match=r"use-after-free \(line 3\)"):
            run("ALLOC a 2 2\nFREE a\nSET v a[0, 0]")

    def test_verbose_output(self, capsys):
        run("ALLOC a 4 4 3 u8 0\nBUF b a", verbose=True)
        out = capsys.readouterr().out
        assert "ALLOC a" in out and "BUF b" in out


# ═════════════════════════════════════════════════════════════════════════════
# PEEK / POKE / attributes
# ═════════════════════════════════════════════════════════════════════════════

class TestPeekPoke:

    def test_peek_poke(self):
        ex = run("ALLOC b 50 20 3 u8 0\nSET b[12, 40, 2] 255\nSET v b[12, 40, 2]\nSET z b[0, 0, 0]")
        assert ex.context["vars"]["v"] == 255 and ex.context["vars"]["z"] == 0
        assert buf(ex, "b")[12, 40, 2] == 255

    def test_poke_with_assign_syntax_and_expr(self):
        ex = run("ALLOC b 4 4\nSET i 2\nSET b[$i, $i + 1] = 3 * 5\nSET v b[2, 3]")
        assert ex.context["vars"]["v"] == 15

    def test_poke_slice_scalar_and_buffer(self):
        ex = run("ALLOC b 6 4\nSET b[0:2, :] 7\nALLOC t 2 2 1 u8 9\nSET b[2:4, 0:2] t")
        arr = buf(ex, "b")
        assert (arr[0:2] == 7).all() and (arr[2:4, 0:2] == 9).all() and (arr[2:4, 2:] == 0).all()

    def test_poke_wraps_like_hardware(self):
        ex = run("ALLOC b 2 2 1 u8 0\nSET b[0, 0] 256\nSET b[0, 1] -1\nSET b[1, 0] 300")
        assert list(buf(ex, "b").ravel()[:3]) == [0, 255, 44]

    def test_poke_i8_and_float(self):
        ex = run("ALLOC b 2 1 1 i8 0\nSET b[0, 0] 200\nALLOC f 1 1 1 f32 0\nSET f[0, 0] 0.25")
        assert buf(ex, "b")[0, 0] == -56 and buf(ex, "f")[0, 0] == 0.25

    def test_poke_out_of_bounds(self):
        with pytest.raises(PixRuntimeError, match="out of bounds.*shape \\(2, 2\\)"):
            run("ALLOC b 2 2\nSET b[5, 0] 1")

    def test_peek_out_of_bounds(self):
        with pytest.raises(PixRuntimeError, match="index out of bounds"):
            run("ALLOC b 2 2\nSET v b[0, 9]")

    def test_peek_non_integer_index(self):
        with pytest.raises(PixRuntimeError, match="index must be an integer"):
            run("ALLOC b 2 2\nSET v b[0.5, 0]")

    def test_poke_non_buffer(self):
        with pytest.raises(PixRuntimeError, match="not a buffer"):
            run("SET x 3\nSET x[0] 1")

    def test_slice_peek_is_a_copy(self):
        ex = run("ALLOC b 4 4 1 u8 1\nSET s b[0:2, 0:2]\nSET b[0, 0] 9\nSET v s[0, 0]\nSET w s.width")
        assert ex.context["vars"]["v"] == 1 and ex.context["vars"]["w"] == 2

    def test_negative_index(self):
        ex = run("ALLOC b 3 3\nSET b[-1, -1] 8\nSET v b[2, 2]")
        assert ex.context["vars"]["v"] == 8

    def test_attributes(self):
        ex = run("""\
            ALLOC b 10 4 3 u16 0
            SET w b.width
            SET h b.height
            SET c b.channels
            SET d b.dtype
            SET n b.nbytes
            SET st b.stride
            SET isz b.itemsize
            SET sz b.size
            SET ad b.addr
            SET sh b.shape
        """)
        v = ex.context["vars"]
        assert (v["w"], v["h"], v["c"], v["d"]) == (10, 4, 3, "u16")
        assert (v["n"], v["st"], v["isz"], v["sz"]) == (240, 60, 2, 120)
        assert v["ad"] == buf(ex, "b").__array_interface__["data"][0] and v["ad"] > 0
        assert v["sh"] == (4, 10, 3)

    def test_gray_channels_attr(self):
        assert run("ALLOC b 2 2\nSET c b.channels").context["vars"]["c"] == 1

    def test_unknown_attribute(self):
        with pytest.raises(PixRuntimeError, match="no attribute 'bogus'"):
            run("ALLOC b 2 2\nSET x b.bogus")

    def test_attr_on_non_buffer(self):
        with pytest.raises(PixRuntimeError, match="not a buffer or variable"):
            run("SET x nothing.width")
        with pytest.raises(PixRuntimeError, match="has no attribute"):
            run("SET n 3\nSET x $n.width")

    def test_assert_attr(self):
        run("ALLOC orig 640 480 3\nASSERT orig.width == 640 and orig.height == 480")
        with pytest.raises(AssertionError, match="orig.width == 641"):
            run("ALLOC orig 640 480 3\nASSERT orig.width == 641")

    def test_sizeof(self):
        v = run("SET a SIZEOF(u16)\nSET b SIZEOF(f64)\nSET c SIZEOF(u8)\nSET d SIZEOF(i32)").context["vars"]
        assert (v["a"], v["b"], v["c"], v["d"]) == (2, 8, 1, 4)

    def test_sizeof_bad_dtype(self):
        with pytest.raises(PixRuntimeError, match="Unknown dtype 'u7'"):
            run("SET a SIZEOF(u7)")


# ═════════════════════════════════════════════════════════════════════════════
# Whole-buffer arithmetic
# ═════════════════════════════════════════════════════════════════════════════

class TestBufferArithmetic:

    def test_bitwise_mask(self):
        ex = run("""\
            ALLOC orig 4 1 1 u8 0xAB
            SET mask (orig & 0xF0) | (orig >> 4)
        """)
        m = buf(ex, "orig")
        assert (ex.context["vars"]["mask"] == ((0xAB & 0xF0) | (0xAB >> 4))).all()
        assert ex.context["vars"]["mask"].dtype == np.uint8

    def test_u8_wraparound(self):
        v = run("ALLOC b 2 1 1 u8 250\nSET s b + 10\nSET d b - 251\nSET m b * 2").context["vars"]
        assert (v["s"] == 4).all() and (v["d"] == 255).all() and (v["m"] == 244).all()
        assert v["s"].dtype == v["d"].dtype == v["m"].dtype == np.uint8

    def test_shift_left_wraps(self):
        v = run("ALLOC b 1 1 1 u8 0x81\nSET s b << 1").context["vars"]
        assert v["s"][0, 0] == 0x02

    def test_i16_wraps_signed(self):
        v = run("ALLOC b 1 1 1 i16 32767\nSET s b + 1").context["vars"]
        assert v["s"][0, 0] == -32768 and v["s"].dtype == np.int16

    def test_not_and_xor(self):
        v = run("ALLOC b 1 1 1 u8 0x0F\nSET n ~b\nSET x b ^ 0xFF").context["vars"]
        assert v["n"][0, 0] == 0xF0 and v["x"][0, 0] == 0xF0

    def test_buffer_buffer_ops(self):
        v = run("ALLOC a 2 2 1 u8 200\nALLOC b 2 2 1 u8 100\nSET s a + b\nSET d b - a").context["vars"]
        assert (v["s"] == 44).all() and (v["d"] == 156).all()

    def test_shape_mismatch(self):
        with pytest.raises(PixRuntimeError, match="shapes do not match"):
            run("ALLOC a 2 2\nALLOC b 3 3\nSET s a + b")

    def test_float_buffer_division(self):
        v = run("ALLOC f 2 1 1 f32 3\nSET q f / 2").context["vars"]
        assert v["q"].dtype == np.float32 and (v["q"] == 1.5).all()

    def test_bitwise_on_float_buffer_rejected(self):
        with pytest.raises(PixRuntimeError, match="needs integer buffers"):
            run("ALLOC f 2 2 1 f32 1\nSET q f & 1")

    def test_comparison_gives_mask(self):
        v = run("ALLOC b 2 1 1 u8 5\nSET m b > 3").context["vars"]
        assert v["m"].dtype == np.bool_ and v["m"].all()

    def test_ambiguous_truth_value(self):
        with pytest.raises(PixRuntimeError, match="ambiguous"):
            run("ALLOC b 2 2\nIF b\nENDIF")

    def test_cast(self):
        v = run("ALLOC b 2 1 1 u8 200\nSET f CAST(b, f32)\nSET i CAST(b, i8)\nSET s CAST(3.7, u8)").context["vars"]
        assert v["f"].dtype == np.float32 and v["f"][0, 0] == 200.0
        assert v["i"].dtype == np.int8 and v["i"][0, 0] == -56
        assert v["s"] == 3

    def test_cast_is_copy(self):
        ex = run("ALLOC b 2 2\nSET c CAST(b, u8)\nSET b[0, 0] 9\nSET v c[0, 0]")
        assert ex.context["vars"]["v"] == 0

    def test_reductions(self):
        v = run("ALLOC b 2 2 1 u8 4\nSET b[0, 0] 8\nSET s sum(b)\nSET m mean(b)\nSET hi max(b)\nSET lo min(b)").context["vars"]
        assert (v["s"], v["m"], v["hi"], v["lo"]) == (20, 5.0, 8, 4)

    def test_sum_does_not_overflow(self):
        assert run("ALLOC b 100 100 1 u8 255\nSET s sum(b)").context["vars"]["s"] == 2_550_000

    def test_zeros_and_clamp_buffer(self):
        v = run("SET z ZEROS(3, 2, 3, u16)\nALLOC b 2 1 1 u8 250\nSET c clamp(b, 0, 100)").context["vars"]
        assert v["z"].shape == (2, 3, 3) and v["z"].dtype == np.uint16
        assert (v["c"] == 100).all()

    def test_scalar_int_ops_are_python_ints(self):
        assert run("SET x 255 + 1").context["vars"]["x"] == 256


# ═════════════════════════════════════════════════════════════════════════════
# Pointers
# ═════════════════════════════════════════════════════════════════════════════

class TestPointers:

    def test_address_of_region_is_live_view(self):
        ex = run("""\
            ALLOC b 8 8 1 u8 0
            SET p &b[2:4, 1:5]
            *p = 7
        """)
        arr = buf(ex, "b")
        assert (arr[2:4, 1:5] == 7).all() and arr.sum() == 7 * 8
        assert isinstance(ex.context["vars"]["p"], Ref)

    def test_write_through_via_set(self):
        ex = run("ALLOC b 4 4\nSET p &b[0:2, 0:2]\nSET *p 3\nSET q &b[3, 3]\nSET *q 9")
        arr = buf(ex, "b")
        assert (arr[0:2, 0:2] == 3).all() and arr[3, 3] == 9

    def test_deref_reads_current_contents(self):
        ex = run("ALLOC b 4 4\nSET p &b[1, 1]\nSET before *p\nSET b[1, 1] 5\nSET after *p")
        v = ex.context["vars"]
        assert (v["before"], v["after"]) == (0, 5)

    def test_pointer_to_whole_buffer(self):
        ex = run("ALLOC b 3 3\nSET p &b\n*p = 4")
        assert (buf(ex, "b") == 4).all()

    def test_pointer_attributes(self):
        v = run("ALLOC b 100 50 3\nSET p &b[10:30, 20:60]\nSET w p.width\nSET h p.height\n"
                "SET c p.channels\nSET st p.stride\nSET off p.offset").context["vars"]
        assert (v["w"], v["h"], v["c"]) == (40, 20, 3)
        assert v["st"] == 100 * 3                     # parent row stride in bytes
        assert v["off"] == 10 * 300 + 20 * 3

    def test_peek_poke_through_pointer(self):
        ex = run("ALLOC b 8 8\nSET p &b[2:6, 2:6]\nSET p[1, 1] 9\nSET v p[1, 1]")
        assert buf(ex, "b")[3, 3] == 9 and ex.context["vars"]["v"] == 9

    def test_pointer_of_pointer(self):
        ex = run("ALLOC b 8 8\nSET p &b[2:6, 2:6]\nSET q &p[1:3, 1:3]\n*q = 5")
        assert (buf(ex, "b")[3:5, 3:5] == 5).all() and buf(ex, "b").sum() == 20

    def test_pointer_in_function_writes_through(self):
        ex = run("""\
            ALLOC b 4 4
            FUNC fill(ptr, v)
                *ptr = $v
            ENDFUNC
            SET r &b[0:2, 0:2]
            SET z fill($r, 6)
        """)
        assert (buf(ex, "b")[0:2, 0:2] == 6).all()

    def test_use_after_free_pointer(self):
        with pytest.raises(PixRuntimeError, match=r"use-after-free \(line 4\)") as info:
            run("""\
                ALLOC b 4 4
                SET p &b[0:2, 0:2]
                FREE b
                *p = 1
            """)
        assert info.value.line == 4
        assert "freed at line 3" in str(info.value) and "taken at line 2" in str(info.value)

    def test_use_after_free_read(self):
        with pytest.raises(PixRuntimeError, match="use-after-free"):
            run("ALLOC b 4 4\nSET p &b[0, 0]\nFREE b\nSET v *p")

    def test_use_after_free_attr_and_index(self):
        for access in ("SET v p.width", "SET v p[0, 0]", "WITH *p\nEND"):
            with pytest.raises(PixRuntimeError, match="use-after-free"):
                run(f"ALLOC b 4 4\nSET p &b[0:2, 0:2]\nFREE b\n{access}")

    def test_pointer_dangling_after_realloc(self):
        with pytest.raises(PixRuntimeError, match="dangling pointer"):
            run("ALLOC b 4 4\nSET p &b[0:2, 0:2]\nALLOC b 4 4\n*p = 1")

    def test_pointer_dangling_after_buf_overwrite(self):
        with pytest.raises(PixRuntimeError, match="dangling pointer"):
            run("USE ZEROS(4, 4)\nBUF b\nSET p &b[0:2, 0:2]\nBUF b\nSET v *p")

    def test_free_then_new_buffer_same_name_old_pointer_dead(self):
        with pytest.raises(PixRuntimeError, match="dangling pointer|use-after-free"):
            run("ALLOC b 4 4\nSET p &b[0, 0]\nFREE b\nALLOC b 4 4\nSET v *p")

    def test_checkpoint_replacement_invalidates_nothing_for_other_slots(self):
        ex = run("ALLOC a 4 4\nALLOC b 4 4\nSET p &a[0, 0]\nFREE b\n*p = 3")
        assert buf(ex, "a")[0, 0] == 3

    def test_address_of_unknown(self):
        with pytest.raises(PixRuntimeError, match="no such buffer"):
            run("SET p &ghost[0, 0]")

    def test_address_of_non_buffer(self):
        with pytest.raises(PixRuntimeError, match="cannot take the address"):
            run("SET x 3\nSET p &x")

    def test_address_of_current_image_rejected(self):
        with pytest.raises(PixRuntimeError, match="address of _"):
            run("USE ZEROS(2, 2)\nSET p &_")

    def test_pointer_oob_on_creation(self):
        with pytest.raises(PixRuntimeError, match="out of bounds|index"):
            run("ALLOC b 4 4\nSET p &b[9, 9]")

    def test_deref_non_pointer(self):
        with pytest.raises(PixRuntimeError, match="not a pointer"):
            run("SET x 3\nSET v *x")
        with pytest.raises(PixRuntimeError, match="not a pointer"):
            run("SET x 3\n*x = 1")

    def test_pointer_arithmetic_rejected(self):
        with pytest.raises(PixRuntimeError, match="pointer arithmetic is not supported"):
            run("ALLOC b 4 4\nSET p &b[0, 0]\nSET q $p + 1")

    def test_pointer_str_and_repr(self):
        ex = run("ALLOC b 4 4\nSET p &b[1:2, 0:3]")
        s = str(ex.context["vars"]["p"])
        assert s.startswith("<ptr &b[1:2, 0:3] @0x")

    def test_pointer_as_command_arg_is_dereferenced(self):
        from pixlang.commands.registry import CommandRegistry
        reg = CommandRegistry()
        got = []

        @reg.register("PROBE")
        def _p(image, arr):
            got.append(arr)
            return image

        Executor(registry=reg).run(parse("ALLOC b 4 4 1 u8 3\nSET p &b[0:2, 0:2]\nPROBE ($p)"))
        # parenthesised expression arguments are dereferenced to the live view
        assert isinstance(got[0], np.ndarray) and got[0].shape == (2, 2)

    def test_deref_in_expressions(self):
        v = run("ALLOC b 2 2 1 u8 5\nSET p &b[0:1, 0:2]\nSET s *p + 1").context["vars"]
        assert (v["s"] == 6).all() and v["s"].shape == (1, 2)


# ═════════════════════════════════════════════════════════════════════════════
# AS reinterpretation
# ═════════════════════════════════════════════════════════════════════════════

class TestReinterpret:

    def test_parse(self):
        from pixlang.parser.ast_nodes import AsType
        node = parse("SET raw orig AS u32").commands[0].value
        assert isinstance(node, AsType) and node.dtype == "u32"

    def test_view_shares_memory(self):
        ex = run("ALLOC b 8 2 1 u8 0\nSET raw b AS u32\nSET w raw.width\nSET n raw.size\nSET isz raw.itemsize")
        v = ex.context["vars"]
        assert (v["w"], v["n"], v["isz"]) == (2, 4, 4)

    def test_no_copy_writes_visible(self):
        ex = run("ALLOC b 4 1 1 u8 0\nSET raw b AS u32\nSET raw[0, 0] 0x01020304\nSET x b[0, 0]\nSET y b[0, 3]")
        v = ex.context["vars"]
        assert (v["x"], v["y"]) == (4, 1)           # little-endian bytes

    def test_float_view(self):
        v = run("ALLOC b 2 1 1 u16 0\nSET f b AS f32\nSET n f.nbytes").context["vars"]
        assert v["n"] == 4

    def test_size_validation(self):
        with pytest.raises(PixRuntimeError, match="cannot reinterpret 6 bytes"):
            run("ALLOC b 3 2 1 u8 0\nSET raw b AS u32")

    def test_flat_reinterpret_of_contiguous(self):
        ex = run("ALLOC b 4 4 3 u8 1\nSET raw b AS u16\nSET n raw.size")
        assert ex.context["vars"]["n"] == 24

    def test_non_buffer(self):
        with pytest.raises(PixRuntimeError, match="needs a buffer"):
            run("SET x 3\nSET r $x AS u8")

    def test_bad_dtype(self):
        with pytest.raises(PixRuntimeError, match="Unknown dtype"):
            run("ALLOC b 4 4\nSET r b AS u7")


# ═════════════════════════════════════════════════════════════════════════════
# WITH, ROI
# ═════════════════════════════════════════════════════════════════════════════

class TestWithAndRoi:

    def test_with_blurs_region_in_place(self):
        ex = run("""\
            ALLOC b 40 40 3 u8 0
            SET b[10:30, 10:30, 0] 255
            SET before b[9, 9, 0]
            SET p &b[0:20, 0:20]
            WITH *p
                BLUR 9
            END
        """)
        arr = buf(ex, "b")
        assert 0 < arr[10, 10, 0] < 255                    # edge got blurred
        assert (arr[30:, 30:] == 0).all()                  # outside untouched
        assert arr[25, 25, 0] == 255                       # outside the pointer region

    def test_with_does_not_touch_current_image(self):
        ex = run("USE ZEROS(10, 10)\nALLOC b 8 8 1 u8 0\nSET p &b[0:4, 0:4]\nWITH *p\nINVERT\nEND")
        assert ex.image.shape == (10, 10) and ex.image.max() == 0
        assert (buf(ex, "b")[0:4, 0:4] == 255).all() and buf(ex, "b")[5, 5] == 0

    def test_with_gray_region_color_command(self):
        ex = run("ALLOC b 8 8 1 u8 100\nSET p &b[0:4, 0:4]\nWITH *p\nGRAYSCALE\nEND")
        assert (buf(ex, "b") == 100).all()

    def test_with_resizing_body_is_fitted(self):
        ex = run("ALLOC b 8 8 1 u8 100\nSET p &b[0:4, 0:4]\nWITH *p\nRESIZE 2 2\nEND")
        assert (buf(ex, "b") == 100).all()

    def test_with_direct_buffer(self):
        ex = run("ALLOC b 4 4 1 u8 7\nWITH b\nINVERT\nEND")
        assert (buf(ex, "b") == 248).all()

    def test_with_needs_pointer(self):
        with pytest.raises(PixRuntimeError, match="needs a pointer"):
            run("SET x 3\nWITH *x\nEND")
        with pytest.raises(PixRuntimeError, match="2-D region"):
            run("ALLOC b 4 4\nSET p &b[1, 1]\nWITH *p\nEND")

    def test_with_node(self):
        node = parse("WITH *p\nBLUR 3\nEND").commands[0]
        assert isinstance(node, WithBlock) and node.body[0].name == "BLUR"

    def test_with_variables_visible(self):
        ex = run("ALLOC b 4 4\nSET k 2\nSET p &b[0:4, 0:4]\nWITH *p\nSET inner $k\nEND")
        assert ex.context["vars"]["inner"] == 2

    def test_roi_unchanged_semantics(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        ex = Executor(registry=default_registry)
        roi = parse("ROI 0 0 50 50\nINVERT\nROI_RESET").commands[0]
        from pixlang.commands.builtin import _unwrap
        out, _ = _unwrap(ex._exec_roi(roi, img))
        assert out[:50, :50].max() == 0 and out[50:, 50:].min() == 255

    def test_roi_with_buffer_vars(self):
        ex = run("""\
            ALLOC c 20 20 3 u8 255
            USE c
            SET x 5
            ROI $x 5 10 10
                INVERT
            ROI_RESET
            BUF res
            SET a res[7, 7, 0]
            SET b res[0, 0, 0]
        """)
        assert ex.context["vars"]["a"] == 0 and ex.context["vars"]["b"] == 255

    def test_roi_gray_to_color_inside(self):
        ex = run("ALLOC c 20 20 1 u8 100\nUSE c\nROI 0 0 10 10\nHEATMAP jet\nROI_RESET")
        from pixlang.commands.builtin import _unwrap
        img, _ = _unwrap(ex.image)
        assert img.ndim == 3 and img.shape[:2] == (20, 20)


# ═════════════════════════════════════════════════════════════════════════════
# Loops
# ═════════════════════════════════════════════════════════════════════════════

class TestLoops:

    def test_parse_loops(self):
        w = parse("WHILE $i < 3\nSET i $i + 1\nEND").commands[0]
        f = parse("FOR i = 1 TO 10 STEP 2\nEND").commands[0]
        r = parse("REPEAT 3 AS k\nEND").commands[0]
        assert isinstance(w, WhileBlock) and isinstance(f, ForBlock) and isinstance(r, RepeatBlock)
        assert (f.var_name, f.start, f.stop, f.step) == ("i", 1, 10, 2)
        assert r.var_name == "k" and r.count == 3

    def test_while(self):
        assert run("SET i 0\nSET s 0\nWHILE $i < 5\nSET s $s + $i\nSET i $i + 1\nEND").context["vars"]["s"] == 10

    def test_while_cap(self):
        with pytest.raises(PixRuntimeError, match="exceeded 1000 iterations"):
            run("WHILE true\nEND", max_loop_iterations=1000)

    def test_while_default_cap_is_one_million(self):
        assert Executor.DEFAULT_MAX_LOOP_ITERATIONS == 1_000_000

    def test_while_cap_from_config(self, tmp_path):
        (tmp_path / "pixlang.toml").write_text("[run]\nmax_loop_iterations = 25\n")
        main = tmp_path / "m.pxl"
        main.write_text("WHILE true\nEND\n")
        with pytest.raises(PixRuntimeError, match="exceeded 25 iterations"):
            Executor(registry=default_registry, pipeline_path=main).run(parse(main.read_text()))

    def test_for_inclusive(self):
        assert run("SET s 0\nFOR i = 1 TO 5\nSET s $s + $i\nEND").context["vars"]["s"] == 15

    def test_for_step_and_negative(self):
        assert run('SET s ""\nFOR i = 0 TO 10 STEP 5\nSET s $s + $i\nEND').context["vars"]["s"] == "0510"
        assert run('SET s ""\nFOR i = 3 TO 1 STEP -1\nSET s $s + $i\nEND').context["vars"]["s"] == "321"

    def test_for_empty_range(self):
        assert run("SET s 0\nFOR i = 5 TO 1\nSET s 1\nEND").context["vars"]["s"] == 0

    def test_for_float_step(self):
        assert run("SET n 0\nFOR x = 0 TO 1 STEP 0.25\nSET n $n + 1\nEND").context["vars"]["n"] == 5

    def test_for_bounds_are_expressions(self):
        assert run("SET n 4\nSET s 0\nFOR i = $n - 2 TO $n * 2\nSET s $s + 1\nEND").context["vars"]["s"] == 7

    def test_for_errors(self):
        with pytest.raises(PixRuntimeError, match="STEP must not be 0"):
            run("FOR i = 1 TO 3 STEP 0\nEND")
        with pytest.raises(PixRuntimeError, match="must be a number"):
            run('FOR i = "a" TO 3\nEND')

    def test_loop_var_scoped_to_loop(self):
        ex = run("SET i 99\nFOR i = 1 TO 3\nEND\nREPEAT 2 AS j\nEND")
        v = ex.context["vars"]
        assert v == {"i": 99}            # loop vars never leak or clobber

    def test_loop_var_readable_bare_and_dollar(self):
        ex = run("ALLOC b 4 1\nFOR i = 0 TO 3\nSET b[0, i] $i * 10\nEND")
        assert list(buf(ex, "b")[0]) == [0, 10, 20, 30]

    def test_repeat_as(self):
        assert run('SET s ""\nREPEAT 3 AS k\nSET s $s + $k\nEND').context["vars"]["s"] == "012"

    def test_nested_repeat_as(self):
        ex = run("SET n 0\nREPEAT 3 AS a\nREPEAT 2 AS b\nSET n $n + $a * 10 + $b\nEND\nEND")
        assert ex.context["vars"]["n"] == (0 + 1) + (10 + 11) + (20 + 21)

    def test_break_continue(self):
        src = """\
            SET s ""
            FOR i = 1 TO 10
                IF $i == 2
                    CONTINUE
                ENDIF
                IF $i == 5
                    BREAK
                ENDIF
                SET s $s + $i
            END
        """
        assert run(src).context["vars"]["s"] == "134"

    def test_break_only_innermost(self):
        src = """\
            SET n 0
            REPEAT 3
                REPEAT 5
                    BREAK
                END
                SET n $n + 1
            END
        """
        assert run(src).context["vars"]["n"] == 3

    def test_continue_in_while(self):
        src = "SET i 0\nSET s 0\nWHILE $i < 6\nSET i $i + 1\nIF $i % 2 == 1\nCONTINUE\nENDIF\nSET s $s + $i\nEND"
        assert run(src).context["vars"]["s"] == 12

    def test_break_continue_outside_loop(self):
        for kw in ("BREAK", "CONTINUE"):
            with pytest.raises(PixSyntaxError, match="outside of a loop"):
                parse(kw)
            with pytest.raises(PixSyntaxError, match="outside of a loop"):
                parse(f"IF true\n{kw}\nENDIF")

    def test_unterminated_for(self):
        with pytest.raises(PixSyntaxError, match="unterminated FOR block opened at line 2"):
            parse("SET a 1\nFOR i = 1 TO 3\nBLUR 3")

    def test_for_missing_to(self):
        with pytest.raises(PixSyntaxError, match="expected TO"):
            parse("FOR i = 1\nEND")

    def test_loops_run_commands(self):
        ex = run("USE ZEROS(4, 4)\nREPEAT 3\nINVERT\nEND")
        assert ex.image.max() == 255        # odd number of inversions

    def test_iter_not_defined_by_for(self):
        with pytest.raises(PixRuntimeError, match="Undefined variable"):
            run("FOR i = 1 TO 2\nSET x $ITER\nEND")

    def test_loop_in_function_uses_frame_scope(self):
        src = """\
            FUNC total(n)
                SET s 0
                FOR i = 1 TO $n
                    SET s $s + $i
                END
                RETURN $s
            ENDFUNC
            SET i 7
            SET r total(4)
        """
        v = run(src).context["vars"]
        assert v == {"i": 7, "r": 10}
