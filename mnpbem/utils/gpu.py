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
from typing import Any, Optional, Tuple

import numpy as np
from scipy.linalg import lu_factor as _scipy_lu_factor
from scipy.linalg import lu_solve as _scipy_lu_solve
from scipy.linalg import solve as _scipy_solve

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


def lu_solve_native(piv_pkg: Tuple, b: Any, **kwargs: Any):
    """Cupy-passthrough variant of ``lu_solve_dispatch``.

    When the LU package is on GPU and ``b`` is a cupy ndarray, returns a
    cupy ndarray (no host round-trip).  Otherwise behaves like
    ``lu_solve_dispatch``.
    """
    tag = piv_pkg[0]
    if tag == "gpu":
        if _CUPY_OK and isinstance(b, _cp.ndarray):
            return _cp_lu_solve((piv_pkg[1], piv_pkg[2]), b)
        b_gpu = _cp.asarray(b)
        x_gpu = _cp_lu_solve((piv_pkg[1], piv_pkg[2]), b_gpu)
        if _CUPY_OK and isinstance(b, _cp.ndarray):
            return x_gpu
        return _cp.asnumpy(x_gpu)
    # CPU LU: if b is cupy, bring it to host
    if _CUPY_OK and isinstance(b, _cp.ndarray):
        b = _cp.asnumpy(b)
    kwargs.setdefault("check_finite", False)
    return _scipy_lu_solve((piv_pkg[1], piv_pkg[2]), b, **kwargs)


def lu_backend(piv_pkg: Tuple) -> str:
    return piv_pkg[0]


def solve_dispatch(A: np.ndarray, b: np.ndarray, **kwargs: Any) -> np.ndarray:
    """One-shot Ax=b: dense solve on GPU when beneficial, else CPU.

    Used by code paths that build a small dense system on the fly without
    reusing the factorization.  Falls back to ``scipy.linalg.solve`` below
    threshold or when CuPy is unavailable.
    """
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        b_gpu = _cp.asarray(b)
        x_gpu = _cp.linalg.solve(A_gpu, b_gpu)
        return _cp.asnumpy(x_gpu)
    kwargs.setdefault("check_finite", False)
    return _scipy_solve(A, b, **kwargs)


def eigh_dispatch(A: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Hermitian eigendecomposition on GPU when beneficial, else CPU.

    Returns ``(w, v)`` as host NumPy arrays regardless of backend.  Routes
    to ``cupy.linalg.eigh`` when GPU is enabled and the matrix is at least
    ``GPU_THRESHOLD`` rows.
    """
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        w_gpu, v_gpu = _cp.linalg.eigh(A_gpu)
        return _cp.asnumpy(w_gpu), _cp.asnumpy(v_gpu)
    from scipy.linalg import eigh as _scipy_eigh
    kwargs.setdefault("check_finite", False)
    return _scipy_eigh(A, **kwargs)


def matmul_dispatch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Dense matrix product on GPU when beneficial, else CPU.

    Used by field-application code paths where a single big dense GEMM
    dominates and the inputs are still on the host.
    """
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        B_gpu = _cp.asarray(B)
        C_gpu = A_gpu @ B_gpu
        return _cp.asnumpy(C_gpu)
    return A @ B


# ---------------------------------------------------------------------------
# Lane C — layer-Green / Sommerfeld integral GPU helpers
#
# These do NOT participate in the BLAS-level dispatch; they are designed for
# the elementwise-heavy kernels (outer products, propagation factors, weighted
# sum reductions) that dominate ``_intbessel_batch`` / ``_inthankel_batch`` in
# layer_structure.py.  Activation is tied to ``MNPBEM_GPU=1`` AND a separate
# ``MNPBEM_GPU_LAYER`` flag so the BEM-solver dispatch above can be tuned
# independently of the layer-Green path.
# ---------------------------------------------------------------------------

LAYER_GPU: bool = (
    USE_GPU and os.environ.get("MNPBEM_GPU_LAYER", "1").strip() not in ("", "0", "false", "False")
)
LAYER_GPU_THRESHOLD: int = int(os.environ.get("MNPBEM_GPU_LAYER_THRESHOLD", "5000"))


def layer_gpu_available() -> bool:
    return _CUPY_OK and LAYER_GPU


def layer_gpu_active(n_flat: int) -> bool:
    """Decide whether to route a layer-Green elementwise kernel to GPU.

    ``n_flat`` is the size of the flattened (n1*n2) array.  The host->device
    copy and kernel launch overhead make GPU profitable only above a few
    thousand entries; below that NumPy wins.
    """
    return _CUPY_OK and LAYER_GPU and n_flat >= LAYER_GPU_THRESHOLD


def get_layer_xp(n_flat: int):
    """Return (xp, asnumpy, on_gpu) for layer-Green elementwise kernels.

    ``xp`` is either ``cupy`` or ``numpy``; ``asnumpy(arr)`` materializes
    arrays on the host; ``on_gpu`` is a boolean for the active backend.
    """
    if layer_gpu_active(n_flat):
        return _cp, _cp.asnumpy, True
    return np, (lambda a: np.asarray(a)), False
