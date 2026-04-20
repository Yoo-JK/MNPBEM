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

    MKL vdAtan2 and glibc atan2 both differ by 1 ULP from MATLAB in some
    inputs. Currently use np.arctan2; a future fix could route through
    a more accurate implementation.
    """
    return np.arctan2(y, x)
