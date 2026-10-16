# pixlang/executor/memory.py
"""
PixLang memory model  v0.5
═══════════════════════════

Buffers
  Named buffer slots live in ``executor.context["checkpoints"]`` (aliased as
  ``context["buffers"]``) — the same store CHECKPOINT / OVERLAY / BLEND /
  COMPARE use, so every buffer is visible to those commands and vice versa.
  Entries are stored like CHECKPOINT stores them: ``(ndarray, meta_dict)``.
  Readers here accept that tuple form, the wrapped-dict form produced by
  ``builtin._wrap`` and bare ndarrays.

  Each slot has a *generation* counter (``context["buffer_gen"]``) bumped on
  every (re)allocation and on FREE, so stale pointers are detected.

Pointers
  ``Ref`` = (slot name, generation, index tuple, base array). Dereferencing
  validates the generation and the identity of the slot's array, then returns
  a live numpy *view*: writes through a pointer modify the buffer in place.

Arithmetic
  Whole-buffer operators map onto numpy. Integer buffers keep their dtype and
  wrap around on overflow like hardware registers: a Python int operand is
  first converted to the buffer's dtype with a C-style (wrapping) cast.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ── dtypes ────────────────────────────────────────────────────────────────────

DTYPES: Dict[str, np.dtype] = {
    "u8":  np.dtype(np.uint8),   "i8":  np.dtype(np.int8),
    "u16": np.dtype(np.uint16),  "i16": np.dtype(np.int16),
    "u32": np.dtype(np.uint32),  "i32": np.dtype(np.int32),
    "f32": np.dtype(np.float32), "f64": np.dtype(np.float64),
}
_DTYPE_NAMES = {v: k for k, v in DTYPES.items()}
_DTYPE_NAMES[np.dtype(np.int64)]  = "i64"
_DTYPE_NAMES[np.dtype(np.uint64)] = "u64"
_DTYPE_NAMES[np.dtype(np.bool_)]  = "bool"


def dtype_of(name: Any) -> np.dtype:
    """Map a PixLang dtype name (u8 ... f64) to a numpy dtype."""
    if isinstance(name, np.dtype):
        return name
    key = str(name).lower()
    if key not in DTYPES:
        raise ValueError(
            f"Unknown dtype '{name}'. Valid: {', '.join(DTYPES)}")
    return DTYPES[key]


def dtype_name(dt: np.dtype) -> str:
    return _DTYPE_NAMES.get(np.dtype(dt), str(dt))


# ── Slot storage helpers ──────────────────────────────────────────────────────

def slot_array(entry: Any) -> Optional[np.ndarray]:
    """Extract the ndarray from a checkpoint/buffer entry of any form."""
    if isinstance(entry, np.ndarray):
        return entry
    if isinstance(entry, tuple) and entry and isinstance(entry[0], np.ndarray):
        return entry[0]
    if isinstance(entry, dict) and isinstance(entry.get("_image"), np.ndarray):
        return entry["_image"]
    return None


def make_slot_entry(arr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Build a slot entry in the same shape CHECKPOINT uses."""
    return (arr, {})


# ── Pointers ──────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Ref:
    """A pointer: a live view into a buffer slot (or into an array value)."""
    slot: Optional[str]
    gen: int
    index: Tuple[Any, ...]
    base: np.ndarray = field(repr=False)
    line: int = 0                      # where the address was taken

    def raw_view(self) -> np.ndarray:
        """View without validation (callers validate liveness first)."""
        return self.base[self.index + (Ellipsis,)]

    def describe(self) -> str:
        idx = ", ".join(_fmt_index(i) for i in self.index)
        target = self.slot or "<array>"
        return f"&{target}[{idx}]" if idx else f"&{target}"

    def __str__(self) -> str:
        try:
            addr = self.raw_view().__array_interface__["data"][0]
            return f"<ptr {self.describe()} @0x{addr:x}>"
        except Exception:
            return f"<ptr {self.describe()}>"

    __repr__ = __str__


def _fmt_index(i: Any) -> str:
    if isinstance(i, slice):
        s = f"{'' if i.start is None else i.start}:{'' if i.stop is None else i.stop}"
        return s + (f":{i.step}" if i.step is not None else "")
    return str(i)


# ── Attributes ────────────────────────────────────────────────────────────────

BUFFER_ATTRS = ("width", "height", "channels", "dtype", "nbytes", "stride",
                "itemsize", "size", "addr", "ndim", "shape")


def buffer_attr(arr: np.ndarray, name: str) -> Any:
    """Attribute access on a buffer / view."""
    if name == "width":
        return int(arr.shape[1]) if arr.ndim >= 2 else int(arr.shape[0]) if arr.ndim else 1
    if name == "height":
        return int(arr.shape[0]) if arr.ndim >= 1 else 1
    if name in ("channels", "depth"):
        return int(arr.shape[2]) if arr.ndim == 3 else 1
    if name == "dtype":
        return dtype_name(arr.dtype)
    if name == "nbytes":
        return int(arr.nbytes)
    if name == "stride":
        return int(arr.strides[0]) if arr.ndim else int(arr.itemsize)
    if name == "itemsize":
        return int(arr.itemsize)
    if name == "size":
        return int(arr.size)
    if name == "addr":
        return int(arr.__array_interface__["data"][0])
    if name == "ndim":
        return int(arr.ndim)
    if name == "shape":
        return tuple(int(s) for s in arr.shape)
    raise AttributeError(
        f"buffer has no attribute '{name}'. Valid: {', '.join(BUFFER_ATTRS)}")


# ── Scalar / array conversion ─────────────────────────────────────────────────

def to_python(value: Any) -> Any:
    """Convert numpy scalars / 0-d arrays to plain Python values."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def wrap_scalar_for(arr: np.ndarray, value: Any) -> Any:
    """Convert a Python int to *arr*'s integer dtype with wraparound."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return value
    if np.issubdtype(arr.dtype, np.integer):
        with np.errstate(all="ignore"):
            return np.array(int(value) & 0xFFFFFFFFFFFFFFFF, dtype=np.uint64).astype(arr.dtype)
    return value


def store_cast(target: np.ndarray, value: Any) -> Any:
    """Convert *value* for assignment into *target* (C-style wrapping cast)."""
    if isinstance(value, np.ndarray):
        if value.dtype == target.dtype:
            return value
        with np.errstate(all="ignore"):
            return value.astype(target.dtype, casting="unsafe")
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, np.integer)):
        return wrap_scalar_for(target, value)
    if isinstance(value, (float, np.floating)) and np.issubdtype(target.dtype, np.integer):
        with np.errstate(all="ignore"):
            return np.array(value).astype(target.dtype, casting="unsafe")
    return value


# ── Operators ─────────────────────────────────────────────────────────────────

_ARITH = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": operator.truediv, "//": operator.floordiv, "%": operator.mod,
    "&": operator.and_, "|": operator.or_, "^": operator.xor,
    "<<": operator.lshift, ">>": operator.rshift,
}
_COMPARE = {
    "==": operator.eq, "!=": operator.ne, "<": operator.lt,
    ">": operator.gt, "<=": operator.le, ">=": operator.ge,
}
_BITWISE = {"&", "|", "^", "<<", ">>"}


def type_name(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return f"buffer({dtype_name(v.dtype)})"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, Ref):
        return "pointer"
    return type(v).__name__


def coerce_for_compare(v: Any) -> Any:
    """Legacy v0.3 rule: make strings numeric when possible."""
    if isinstance(v, (int, float, np.ndarray, np.generic)):
        return v
    if isinstance(v, str):
        for conv in (int, float):
            try:
                return conv(v)
            except ValueError:
                pass
    return v


def compare(op: str, a: Any, b: Any) -> Any:
    """Comparison with v0.3 string→number coercion; clear error on mismatch."""
    fn = _COMPARE[op]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return fn(a, b)
    ca, cb = coerce_for_compare(a), coerce_for_compare(b)
    if op in ("==", "!="):
        if isinstance(ca, str) != isinstance(cb, str):
            ca, cb = str(a), str(b)
        return fn(ca, cb)
    try:
        return fn(ca, cb)
    except TypeError:
        raise TypeError(
            f"cannot compare {type_name(a)} {a!r} with {type_name(b)} {b!r} using '{op}'"
        ) from None


def binop(op: str, a: Any, b: Any) -> Any:
    """Arithmetic / bitwise operator with PixLang semantics."""
    if op in _COMPARE:
        return compare(op, a, b)
    if isinstance(a, Ref) or isinstance(b, Ref):
        raise TypeError("pointer arithmetic is not supported; dereference with *p first")
    a_arr, b_arr = isinstance(a, np.ndarray), isinstance(b, np.ndarray)

    if not a_arr and not b_arr:
        if op == "+" and (isinstance(a, str) or isinstance(b, str)):
            return _fmt(a) + _fmt(b)
        if isinstance(a, str) or isinstance(b, str):
            raise TypeError(
                f"unsupported operand types for '{op}': {type_name(a)} and {type_name(b)}")
        if op in _BITWISE:
            if not (_is_int(a) and _is_int(b)):
                raise TypeError(
                    f"bitwise '{op}' needs integers, got {type_name(a)} and {type_name(b)}")
            if op in ("<<", ">>") and b < 0:
                raise ValueError(f"negative shift count {b}")
        if op in ("/", "//", "%") and b == 0:
            raise ZeroDivisionError("division by zero")
        return _ARITH[op](a, b)

    # Buffer arithmetic: keep integer dtypes, wrap Python ints like hardware
    if a_arr and not b_arr:
        b = wrap_scalar_for(a, b)
    elif b_arr and not a_arr:
        a = wrap_scalar_for(b, a)
    if op in _BITWISE:
        for v in (a, b):
            dt = v.dtype if isinstance(v, (np.ndarray, np.generic)) else np.asarray(v).dtype
            if not (np.issubdtype(dt, np.integer) or dt == np.bool_):
                raise TypeError(f"bitwise '{op}' needs integer buffers, got {dtype_name(dt)}")
    with np.errstate(all="ignore"):
        try:
            return _ARITH[op](a, b)
        except ValueError as exc:
            raise ValueError(f"buffer shapes do not match for '{op}': {exc}") from None


def unary(op: str, v: Any) -> Any:
    if op == "-":
        if isinstance(v, (str, bool)) or isinstance(v, Ref):
            raise TypeError(f"bad operand for unary '-': {type_name(v)}")
        with np.errstate(all="ignore"):
            return -v
    if op == "~":
        if isinstance(v, np.ndarray):
            if not (np.issubdtype(v.dtype, np.integer) or v.dtype == np.bool_):
                raise TypeError(f"'~' needs an integer buffer, got {dtype_name(v.dtype)}")
            return ~v
        if not _is_int(v):
            raise TypeError(f"'~' needs an integer, got {type_name(v)}")
        return ~v
    raise ValueError(f"unknown unary operator {op!r}")


def truth(v: Any) -> bool:
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return bool(v.reshape(-1)[0])
        raise TypeError(
            "the truth value of a buffer is ambiguous; use any(...) or all(...)")
    if isinstance(v, Ref):
        return True
    return bool(v)


def _is_int(v: Any) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def reinterpret(arr: np.ndarray, dt_name: str) -> np.ndarray:
    """``buf AS dtype`` — view the same bytes with another dtype (no copy)."""
    dt = dtype_of(dt_name)
    try:
        return arr.view(dt)
    except ValueError:
        pass
    if arr.flags.c_contiguous and arr.nbytes % dt.itemsize == 0:
        return arr.reshape(-1).view(dt)          # flat reinterpretation, still no copy
    raise ValueError(
        f"cannot reinterpret {arr.nbytes} bytes of shape {arr.shape} "
        f"({dtype_name(arr.dtype)}) as {dt_name}: the byte count of the last axis "
        f"(or of a contiguous buffer) must be a multiple of {dt.itemsize}")
