# pixlang/executor/builtins.py
"""
Builtin expression functions  v0.5

Scalars:   min max abs int float str len sqrt clamp round floor ceil hex bin
Buffers:   sum mean any all copy        (min / max / abs / clamp work too)
Memory:    CAST(buf, dtype)  SIZEOF(dtype)  ZEROS(w, h, [c], [dtype])

Every function receives already-evaluated argument values. Buffers are numpy
arrays; dtype arguments are names such as u8 or f32.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict

import numpy as np

from .memory import dtype_of, to_python


def _num(v: Any, fn: str) -> Any:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise TypeError(f"{fn}() needs a number, got {type(v).__name__} {v!r}")
    return v


def _min(*args):
    if len(args) == 1:
        a = args[0]
        if isinstance(a, np.ndarray):
            return to_python(a.min())
        return min(a)
    if any(isinstance(a, np.ndarray) for a in args):
        out = args[0]
        for a in args[1:]:
            out = np.minimum(out, a)
        return out
    return min(args)


def _max(*args):
    if len(args) == 1:
        a = args[0]
        if isinstance(a, np.ndarray):
            return to_python(a.max())
        return max(a)
    if any(isinstance(a, np.ndarray) for a in args):
        out = args[0]
        for a in args[1:]:
            out = np.maximum(out, a)
        return out
    return max(args)


def _clamp(v, lo, hi):
    if isinstance(v, np.ndarray):
        return np.clip(v, lo, hi).astype(v.dtype)
    return max(lo, min(v, hi))


def _int(v):
    if isinstance(v, str):
        return int(v, 0)
    return int(_num(v, "int"))


def _float(v):
    if isinstance(v, str):
        return float(v)
    return float(_num(v, "float"))


def _str(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _len(v):
    if isinstance(v, np.ndarray):
        return int(v.shape[0]) if v.ndim else 1
    return len(v)


def _round(v, ndigits=0):
    if isinstance(v, np.ndarray):
        return np.round(v, int(ndigits))
    return round(_num(v, "round"), int(ndigits)) if ndigits else int(round(_num(v, "round")))


def _floor(v):
    return np.floor(v) if isinstance(v, np.ndarray) else math.floor(_num(v, "floor"))


def _ceil(v):
    return np.ceil(v) if isinstance(v, np.ndarray) else math.ceil(_num(v, "ceil"))


def _sqrt(v):
    if isinstance(v, np.ndarray):
        return np.sqrt(v)
    v = _num(v, "sqrt")
    if v < 0:
        raise ValueError(f"sqrt() of negative number {v}")
    return math.sqrt(v)


def _abs(v):
    return np.abs(v) if isinstance(v, np.ndarray) else abs(_num(v, "abs"))


def _need_buffer(v, fn):
    if not isinstance(v, np.ndarray):
        raise TypeError(f"{fn}() needs a buffer, got {type(v).__name__}")
    return v


def _sum(v):
    return to_python(_need_buffer(v, "sum").sum(dtype=np.float64
                     if np.issubdtype(v.dtype, np.floating) else np.int64))


def _mean(v):
    return float(_need_buffer(v, "mean").mean())


def _any(v):
    return bool(np.any(v))


def _all(v):
    return bool(np.all(v))


def _copy(v):
    return v.copy() if isinstance(v, np.ndarray) else v


def _cast(v, dt):
    target = dtype_of(dt)
    if isinstance(v, np.ndarray):
        with np.errstate(all="ignore"):
            return v.astype(target, casting="unsafe")
    with np.errstate(all="ignore"):
        return to_python(np.array(v).astype(target, casting="unsafe"))


def _sizeof(dt):
    if isinstance(dt, np.ndarray):
        return int(dt.nbytes)
    return int(dtype_of(dt).itemsize)


def _zeros(w, h, c=1, dt="u8"):
    shape = (int(h), int(w)) if int(c) == 1 else (int(h), int(w), int(c))
    return np.zeros(shape, dtype=dtype_of(dt))


def _hex(v):
    return hex(int(_num(v, "hex")))


def _bin(v):
    return bin(int(_num(v, "bin")))


BUILTIN_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "min": _min, "max": _max, "abs": _abs, "int": _int, "float": _float,
    "str": _str, "len": _len, "sqrt": _sqrt, "clamp": _clamp, "round": _round,
    "floor": _floor, "ceil": _ceil, "sum": _sum, "mean": _mean, "any": _any,
    "all": _all, "copy": _copy, "hex": _hex, "bin": _bin,
    "CAST": _cast, "SIZEOF": _sizeof, "ZEROS": _zeros,
}
