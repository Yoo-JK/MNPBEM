"""GPU dispatch helpers for LU factor / solve.

Provides a unified API that automatically routes large dense linear systems
to a CuPy + cuSOLVER backend when:

- ``cupy`` is importable
- ``MNPBEM_GPU=1`` (default OFF — explicit opt-in)
- the matrix dimension is at least ``MNPBEM_GPU_THRESHOLD`` (default 1500)

Below the threshold the helpers fall back to ``scipy.linalg.lu_factor`` /
``lu_solve`` to avoid host <-> device transfer overhead on small problems.

The factor object is opaque: a tuple whose first element is ``'cpu'`` or
``'gpu'`` indicating where the LU lives.  ``lu_solve_dispatch`` returns a
NumPy array regardless of backend so callers do not need to be aware of the
device.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import numpy as np
from scipy.linalg import lu_factor as _scipy_lu_factor
from scipy.linalg import lu_solve as _scipy_lu_solve

USE_GPU: bool = os.environ.get("MNPBEM_GPU", "0") == "1"
GPU_THRESHOLD: int = int(os.environ.get("MNPBEM_GPU_THRESHOLD", "1500"))

try:
    import cupy as _cp  # type: ignore
    from cupyx.scipy.linalg import lu_factor as _cp_lu_factor  # type: ignore
    from cupyx.scipy.linalg import lu_solve as _cp_lu_solve  # type: ignore
    _CUPY_OK: bool = True
except Exception:
    _cp = None  # type: ignore
    _CUPY_OK = False


def gpu_available() -> bool:
    return _CUPY_OK and USE_GPU


def lu_factor_dispatch(A: np.ndarray, **kwargs: Any) -> Tuple:
    """Factorize A on GPU when beneficial, else CPU.

    Extra ``kwargs`` are forwarded to ``scipy.linalg.lu_factor`` for the CPU
    path; GPU path uses CuPy defaults (``check_finite`` / ``overwrite_a``
    are not exposed by ``cupyx.scipy.linalg.lu_factor`` in the same way).
    """
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        lu_gpu, piv_gpu = _cp_lu_factor(A_gpu, overwrite_a=True)
        return ("gpu", lu_gpu, piv_gpu)
    kwargs.setdefault("check_finite", False)
    lu, piv = _scipy_lu_factor(A, **kwargs)
    return ("cpu", lu, piv)


def lu_solve_dispatch(piv_pkg: Tuple, b: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Solve A x = b given a factorization produced by ``lu_factor_dispatch``.

    Returns a NumPy array on the host irrespective of where the LU lives.
    """
    tag = piv_pkg[0]
    if tag == "gpu":
        b_gpu = _cp.asarray(b)
        x_gpu = _cp_lu_solve((piv_pkg[1], piv_pkg[2]), b_gpu)
        return _cp.asnumpy(x_gpu)
    kwargs.setdefault("check_finite", False)
    return _scipy_lu_solve((piv_pkg[1], piv_pkg[2]), b, **kwargs)


def lu_backend(piv_pkg: Tuple) -> str:
    return piv_pkg[0]
