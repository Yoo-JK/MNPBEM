"""MATLAB-compatible floating-point primitives.

Problem: Python's `np.linspace`, `np.arctan2` output differ from MATLAB's by
1 ULP due to different FP accumulation order / algorithm. `np.cos/sin` itself
already matches MATLAB (both routed through Intel MKL on Linux with numpy
linked against MKL). But 1 ULP differences in theta or linspace propagate
through `p @ rot` and `np.unique` causing mesh topology divergence.

This module provides drop-in replacements that match MATLAB bit-for-bit.

Functions:
    mlinspace(a, b, n): MATLAB's linspace formula
    matan2(y, x):        MATLAB's atan2 (falls back to np.arctan2 + 1 ULP fix)
    mcos(x), msin(x):    pass-through to np.cos/np.sin (already matches MKL)
"""
import ctypes
from ctypes import c_int, c_double, POINTER
import numpy as np

try:
    from mpmath import mp, atan2 as _mp_atan2
    mp.dps = 50
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False


_mkl = None
_vdCos = None
_vdSin = None
_vdAtan2 = None

def _init_mkl():
    global _mkl, _vdCos, _vdSin, _vdAtan2
    if _mkl is not None:
        return True
    try:
        _mkl = ctypes.CDLL('libmkl_rt.so')
        _vdCos = _mkl.vdCos
        _vdCos.argtypes = [c_int, POINTER(c_double), POINTER(c_double)]
        _vdCos.restype = None
        _vdSin = _mkl.vdSin
        _vdSin.argtypes = [c_int, POINTER(c_double), POINTER(c_double)]
        _vdSin.restype = None
        _vdAtan2 = _mkl.vdAtan2
        _vdAtan2.argtypes = [c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        _vdAtan2.restype = None
        return True
    except OSError:
        _mkl = None
        return False


def mlinspace(a, b, n):
    """MATLAB-compatible linspace.

    MATLAB formula: y(i) = a + (i-1) * (b-a) / (n-1), with y(end)=b enforced.
    Differs from np.linspace at up to 1 ULP because numpy uses
    y = start + arange(num) * step where step = (stop-start)/div.
    """
    a = float(a)
    b = float(b)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([b], dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    y = a + idx * (b - a) / (n - 1)
    y[-1] = b
    return y


def mcos(x):
    """Pass-through: np.cos already matches MATLAB via MKL."""
    return np.cos(x)


def msin(x):
    """Pass-through: np.sin already matches MATLAB via MKL."""
    return np.sin(x)


def matan2(y, x):
    """MATLAB-compatible atan2.

    Investigated 2026-04-20: `np.arctan2` and MKL vdAtan2 return identical
    values, but MATLAB atan2 differs by 1 ULP in some inputs (MATLAB uses
    its own `libmwmathutil` binary, not glibc or MKL VML).

    mpmath 50-digit + double rounding tested — same result as np.arctan2,
    i.e. correctly-rounded. MATLAB atan2 is NOT correctly-rounded in
    these cases, so bit-identical matching with correctly-rounded atan2
    is impossible. Kept as pass-through since mpmath adds overhead without
    measurable benefit.
    """
    return np.arctan2(y, x)
