"""Numba-parallel element-wise kernels for the substrate BEM assembly.

The distributed substrate build (:meth:`BEMRetLayer._init_distributed_precond`)
routes every dense N x N matmul through MKL (already multithreaded) but keeps
the *element-wise* combination steps -- the ``diff = A@B - C`` subtractions,
the ``m_ij`` fused ``A - B - 1j*k*(P + D*nperp)`` tails, and (via
``BEMRetLayer._sub_mat``) the ``G2/G2e`` outer-mixed subtractions -- on the
host as plain numpy.  Numpy element-wise ops are single-threaded (only BLAS
matmul honours MKL_NUM_THREADS), so on a 15072-face substrate each N^2
element-wise combo costs ~7-10 s and dominates the non-matmul share of the
build.

These kernels reproduce those combos with ``numba.prange`` over the matrix
rows.  Only the forms whose numba result is *bit-identical* to numpy are
provided here: subtraction (``sub2``) and the fused ``m_ij`` tails, which do
at most one complex*complex multiply in the same evaluation order as numpy
(verified max relative error = 0 at N=800..15072, fp64).  A *chained*
complex*complex product (e.g. ``1j*k*prod*npar_outer`` for ``Gammapar``)
rounds ~1 ULP differently under numba, so it deliberately stays on numpy in
the caller.  Matmul is never done here -- the ``@`` operands arrive already
computed on the host.

Gate: only used when ``MNPBEM_BEM_ELEM_NUMBA`` is truthy AND numba imports.
Default OFF, so the legacy numpy path stays byte-for-byte unchanged.
"""

import os

import numpy as np

try:
    import numba
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def bem_elem_numba_enabled() -> bool:
    if not _HAS_NUMBA:
        return False
    return os.environ.get('MNPBEM_BEM_ELEM_NUMBA', '').strip() not in (
        '', '0', 'false', 'False')


if _HAS_NUMBA:

    @njit(cache = True, parallel = True, fastmath = False)
    def _sub2(A, B, out):
        n0, n1 = A.shape
        for i in prange(n0):
            for j in range(n1):
                out[i, j] = A[i, j] - B[i, j]

    @njit(cache = True, parallel = True, fastmath = False)
    def _m_full_block(SG, He, GP, diff2, nperp, ik, out):
        # SG - He - 1j*k*(GP + diff2*nperp[:,None])
        n0, n1 = SG.shape
        for i in prange(n0):
            npi = nperp[i]
            for j in range(n1):
                out[i, j] = SG[i, j] - He[i, j] - ik * (GP[i, j] + diff2[i, j] * npi)

    @njit(cache = True, parallel = True, fastmath = False)
    def _m_half_block(SG, H, diff, nperp, ik, out):
        # SG - H - 1j*k*diff*nperp[:,None]
        n0, n1 = SG.shape
        for i in prange(n0):
            npi = nperp[i]
            for j in range(n1):
                out[i, j] = SG[i, j] - H[i, j] - ik * diff[i, j] * npi


def _is_matrix(x) -> bool:
    return isinstance(x, np.ndarray) and x.ndim == 2


def sub2(A, B):
    """A - B, numba-parallel; falls back to numpy when disabled/non-array."""
    if not bem_elem_numba_enabled() or not (_is_matrix(A) and _is_matrix(B)):
        return A - B
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    out = np.empty_like(A)
    _sub2(A, B, out)
    return out


def m_full_block(SG, He, GP, diff2, nperp, k):
    """SG - He - 1j*k*(GP + diff2*nperp[:,None]), numba-parallel."""
    if not bem_elem_numba_enabled() or not (
            _is_matrix(SG) and _is_matrix(He) and _is_matrix(GP) and _is_matrix(diff2)):
        return SG - He - 1j * k * (GP + diff2 * nperp[:, np.newaxis])
    SG = np.ascontiguousarray(SG)
    He = np.ascontiguousarray(He)
    GP = np.ascontiguousarray(GP)
    diff2 = np.ascontiguousarray(diff2)
    nperp = np.ascontiguousarray(nperp)
    out = np.empty_like(SG)
    _m_full_block(SG, He, GP, diff2, nperp, 1j * k, out)
    return out


def m_half_block(SG, H, diff, nperp, k):
    """SG - H - 1j*k*diff*nperp[:,None], numba-parallel."""
    if not bem_elem_numba_enabled() or not (
            _is_matrix(SG) and _is_matrix(H) and _is_matrix(diff)):
        return SG - H - 1j * k * diff * nperp[:, np.newaxis]
    SG = np.ascontiguousarray(SG)
    H = np.ascontiguousarray(H)
    diff = np.ascontiguousarray(diff)
    nperp = np.ascontiguousarray(nperp)
    out = np.empty_like(SG)
    _m_half_block(SG, H, diff, nperp, 1j * k, out)
    return out
