"""MATLAB-compatible floating-point primitives.

Problem: Python's `np.linspace`, `np.arctan2` output differ from MATLAB's by
1 ULP due to different FP accumulation order / algorithm. `np.cos/sin` itself
already matches MATLAB (both routed through Intel MKL on Linux with numpy
linked against MKL). But 1 ULP differences in theta or linspace propagate
through `p @ rot` and `np.unique` causing mesh topology divergence.

This module provides drop-in replacements that match MATLAB bit-for-bit.

Functions:
    mlinspace(a, b, n): MATLAB's linspace formula
    matan2(y, x):       MATLAB's atan2 via libmwmathutil ctypes (bit-identical)
    mcos(x), msin(x):   pass-through to np.cos/np.sin (already matches MKL)
"""
import ctypes
import os
from ctypes import c_int, c_double, c_size_t, POINTER
import numpy as np


# --- MATLAB libmwmathutil (bit-identical atan2) ---
# Investigated 2026-04-22: MATLAB R2025b uses its own atan2 in
# libmwmathutil.so (fdlibm-derived), which differs from np.arctan2 / MKL
# vdAtan2 at 1 ULP for ~17% of random inputs. Directly calling the
# vectorized template `mu::Atan2<double,double,double>` gives bit-identical
# results at ~np.arctan2 speed.
#
# Symbol: _ZN2mu5Atan2IdddEEvPT_PT0_PT1_mmmm
# Signature: void(double* out, double* y, double* x,
#                 size_t stride_out, size_t stride_y, size_t stride_x,
#                 size_t count)
# Scalar variant: muDoubleScalarAtan2(double y, double x) -> double
_MATLAB_LIB_PATH = '/usr/local/MATLAB/R2025b/bin/glnxa64/libmwmathutil.so'
_mathutil = None
_matan2_vec = None
_matan2_scalar = None
_mcos_vec = None
_mcos_scalar = None
_msin_vec = None
_msin_scalar = None


def _init_matlab_atan2():
    """Attempt to load MATLAB's atan2/cos/sin from libmwmathutil.so. Returns
    True on success, False if MATLAB is not installed — callers then fall
    back to np.arctan2 / np.cos / np.sin.
    """
    global _mathutil, _matan2_vec, _matan2_scalar
    global _mcos_vec, _mcos_scalar, _msin_vec, _msin_scalar
    if _mathutil is not None:
        return True
    if not os.path.exists(_MATLAB_LIB_PATH):
        return False
    try:
        _mathutil = ctypes.CDLL(_MATLAB_LIB_PATH)
        _matan2_vec = _mathutil._ZN2mu5Atan2IdddEEvPT_PT0_PT1_mmmm
        _matan2_vec.argtypes = [
            POINTER(c_double), POINTER(c_double), POINTER(c_double),
            c_size_t, c_size_t, c_size_t, c_size_t,
        ]
        _matan2_vec.restype = None
        _matan2_scalar = _mathutil.muDoubleScalarAtan2
        _matan2_scalar.argtypes = [c_double, c_double]
        _matan2_scalar.restype = c_double
        # MATLAB cos/sin — single-argument vector+scalar forms
        _mcos_vec = _mathutil._ZN2mu3CosIdEEvPT_S2_mmm
        _mcos_vec.argtypes = [POINTER(c_double), POINTER(c_double),
                              c_size_t, c_size_t, c_size_t]
        _mcos_vec.restype = None
        _mcos_scalar = _mathutil.muDoubleScalarCos
        _mcos_scalar.argtypes = [c_double]
        _mcos_scalar.restype = c_double
        _msin_vec = _mathutil._ZN2mu3SinIdEEvPT_S2_mmm
        _msin_vec.argtypes = [POINTER(c_double), POINTER(c_double),
                              c_size_t, c_size_t, c_size_t]
        _msin_vec.restype = None
        _msin_scalar = _mathutil.muDoubleScalarSin
        _msin_scalar.argtypes = [c_double]
        _msin_scalar.restype = c_double
        return True
    except (OSError, AttributeError):
        _mathutil = None
        _matan2_vec = None
        _matan2_scalar = None
        _mcos_vec = None
        _mcos_scalar = None
        _msin_vec = None
        _msin_scalar = None
        return False


# Eagerly initialise so tests can query availability.
_MATLAB_ATAN2_AVAILABLE = _init_matlab_atan2()


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
    """MATLAB-compatible cos.

    MATLAB ships its own cos in libmwmathutil (fdlibm-derived) that differs
    from np.cos / Intel MKL at 1 ULP on certain inputs (e.g.
    sin(2*pi/3) = 0.866025403784438486 via MKL vs 0.866025403784438597 via
    MATLAB). 1-ULP drift in sin/cos propagates through p @ rot and breaks
    quadtree/meshpoly topology. Falls back to np.cos when MATLAB is not
    available.
    """
    if not _MATLAB_ATAN2_AVAILABLE:
        return np.cos(x)
    if np.isscalar(x):
        return _mcos_scalar(float(x))
    x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x_arr.ndim == 0:
        return np.float64(_mcos_scalar(float(x_arr)))
    out = np.empty_like(x_arr)
    _mcos_vec(
        out.ctypes.data_as(POINTER(c_double)),
        x_arr.ctypes.data_as(POINTER(c_double)),
        c_size_t(1), c_size_t(1), c_size_t(x_arr.size),
    )
    return out


def msin(x):
    """MATLAB-compatible sin. See mcos() for rationale."""
    if not _MATLAB_ATAN2_AVAILABLE:
        return np.sin(x)
    if np.isscalar(x):
        return _msin_scalar(float(x))
    x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x_arr.ndim == 0:
        return np.float64(_msin_scalar(float(x_arr)))
    out = np.empty_like(x_arr)
    _msin_vec(
        out.ctypes.data_as(POINTER(c_double)),
        x_arr.ctypes.data_as(POINTER(c_double)),
        c_size_t(1), c_size_t(1), c_size_t(x_arr.size),
    )
    return out


def _matan2_impl(y_arr, x_arr):
    """Core vectorized call: y_arr, x_arr are contiguous float64 numpy arrays
    of the same shape. Returns a new array of atan2 values computed by
    MATLAB's libmwmathutil.
    """
    out = np.empty_like(y_arr)
    n = y_arr.size
    if n == 0:
        return out
    _matan2_vec(
        out.ctypes.data_as(POINTER(c_double)),
        y_arr.ctypes.data_as(POINTER(c_double)),
        x_arr.ctypes.data_as(POINTER(c_double)),
        c_size_t(1), c_size_t(1), c_size_t(1), c_size_t(n),
    )
    return out


def matan2(y, x):
    """MATLAB-compatible atan2.

    Loads MATLAB's own atan2 from libmwmathutil.so when available; otherwise
    falls back to np.arctan2. Bit-identical with MATLAB `atan2(y, x)` on the
    full domain including signed zeros, inf, and NaN (verified on 40k
    random + edge-case samples).

    Accepts scalars, numpy arrays, or broadcastable pairs; mirrors
    np.arctan2 return semantics (scalar in -> scalar out).
    """
    if not _MATLAB_ATAN2_AVAILABLE:
        return np.arctan2(y, x)

    # Scalar fast path (mirrors np.arctan2 returning a Python float for
    # 0-dim inputs).
    if np.isscalar(y) and np.isscalar(x):
        return _matan2_scalar(float(y), float(x))

    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    # Broadcast and contiguify
    if y_arr.shape != x_arr.shape:
        y_arr, x_arr = np.broadcast_arrays(y_arr, x_arr)
    y_arr = np.ascontiguousarray(y_arr, dtype=np.float64)
    x_arr = np.ascontiguousarray(x_arr, dtype=np.float64)

    # 0-dim inputs: use scalar routine to match np.arctan2's scalar-out
    # behaviour, matching MATLAB bit-for-bit.
    if y_arr.ndim == 0:
        return np.float64(_matan2_scalar(float(y_arr), float(x_arr)))

    return _matan2_impl(y_arr, x_arr)
