"""
Numba-accelerated kernels for retarded Green function assembly.

Used by GreenRetRefined.eval for the dense G/F/Gp pre-phase fill.
Refinement overlays are applied by the Python caller after these kernels
produce the dense pre-phase matrices.

Activation:
  - default: enabled when numba is importable
  - disable by setting MNPBEM_NUMBA=0
"""

import os
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


_EPS = 2.220446049250313e-16


def numba_enabled():
    """Return True iff numba kernels should be used."""
    if not NUMBA_AVAILABLE:
        return False
    return os.environ.get('MNPBEM_NUMBA', '1') != '0'


if NUMBA_AVAILABLE:

    @njit(parallel = True, fastmath = False, cache = True)
    def _green_ret_dGF(pos1, pos2, nvec1, area2, same):
        """
        Build distance d, pre-phase G and F matrices.

        G_pre[i, j] = area2[j] / d
        F_pre[i, j] = (n_dot_r) * (1j * k - 1/d) / d^2 * area2[j]   [k applied later]

        For numerical efficiency we return only k-independent factors:
          d[i, j]
          inv_d[i, j]            = 1 / d
          n_dot_r[i, j]          = nvec1[i] · (pos1[i] - pos2[j])
        Caller assembles G_pre, F_pre (cheap) then multiplies by exp(1j*k*d).
        Self-block (same and i == j) entries get d = eps so refinement /
        analytical correction can overwrite them safely.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        d_out = np.empty((n1, n2))
        inv_d_out = np.empty((n1, n2))
        ndr_out = np.empty((n1, n2))
        for i in prange(n1):
            p0 = pos1[i, 0]
            p1 = pos1[i, 1]
            p2 = pos1[i, 2]
            nx = nvec1[i, 0]
            ny = nvec1[i, 1]
            nz = nvec1[i, 2]
            for j in range(n2):
                rx = p0 - pos2[j, 0]
                ry = p1 - pos2[j, 1]
                rz = p2 - pos2[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                d = d2 ** 0.5
                if same and i == j:
                    d = _EPS
                    inv_d = 1.0 / _EPS
                    ndotr = 0.0
                else:
                    if d < _EPS:
                        d = _EPS
                    inv_d = 1.0 / d
                    ndotr = nx * rx + ny * ry + nz * rz
                d_out[i, j] = d
                inv_d_out[i, j] = inv_d
                ndr_out[i, j] = ndotr
        return d_out, inv_d_out, ndr_out

    @njit(parallel = True, fastmath = False, cache = True)
    def _green_ret_dGFr(pos1, pos2, nvec1, area2, same):
        """
        Like _green_ret_dGF but additionally returns the relative vector
        components rx, ry, rz needed for cart deriv / Gp.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        d_out = np.empty((n1, n2))
        inv_d_out = np.empty((n1, n2))
        ndr_out = np.empty((n1, n2))
        rx_out = np.empty((n1, n2))
        ry_out = np.empty((n1, n2))
        rz_out = np.empty((n1, n2))
        for i in prange(n1):
            p0 = pos1[i, 0]
            p1 = pos1[i, 1]
            p2 = pos1[i, 2]
            nx = nvec1[i, 0]
            ny = nvec1[i, 1]
            nz = nvec1[i, 2]
            for j in range(n2):
                rx = p0 - pos2[j, 0]
                ry = p1 - pos2[j, 1]
                rz = p2 - pos2[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                d = d2 ** 0.5
                if same and i == j:
                    d = _EPS
                    inv_d = 1.0 / _EPS
                    ndotr = 0.0
                else:
                    if d < _EPS:
                        d = _EPS
                    inv_d = 1.0 / d
                    ndotr = nx * rx + ny * ry + nz * rz
                d_out[i, j] = d
                inv_d_out[i, j] = inv_d
                ndr_out[i, j] = ndotr
                rx_out[i, j] = rx
                ry_out[i, j] = ry
                rz_out[i, j] = rz
        return d_out, inv_d_out, ndr_out, rx_out, ry_out, rz_out


def green_ret_distances(pos1, pos2, nvec1, area2, same, want_r = False):
    """
    Compute distance and inner products for retarded Green function.

    Returns
    -------
    d, inv_d, n_dot_r              (when want_r is False)
    d, inv_d, n_dot_r, rx, ry, rz  (when want_r is True)

    Falls back to numpy broadcasting when numba is unavailable / disabled.
    """
    pos1 = np.ascontiguousarray(pos1, dtype = np.float64)
    pos2 = np.ascontiguousarray(pos2, dtype = np.float64)
    nvec1 = np.ascontiguousarray(nvec1, dtype = np.float64)
    area2 = np.ascontiguousarray(area2, dtype = np.float64)

    if numba_enabled():
        if want_r:
            return _green_ret_dGFr(pos1, pos2, nvec1, area2, same)
        return _green_ret_dGF(pos1, pos2, nvec1, area2, same)

    return _green_ret_distances_numpy(pos1, pos2, nvec1, area2, same, want_r)


def _green_ret_distances_numpy(pos1, pos2, nvec1, area2, same, want_r):
    """Reference numpy implementation (used when MNPBEM_NUMBA=0)."""
    rx = pos1[:, 0:1] - pos2[:, 0]
    ry = pos1[:, 1:2] - pos2[:, 1]
    rz = pos1[:, 2:3] - pos2[:, 2]
    d = np.sqrt(rx * rx + ry * ry + rz * rz)
    d = np.maximum(d, np.finfo(float).eps)
    inv_d = 1.0 / d
    n_dot_r = (nvec1[:, 0:1] * rx +
               nvec1[:, 1:2] * ry +
               nvec1[:, 2:3] * rz)
    if want_r:
        return d, inv_d, n_dot_r, rx, ry, rz
    return d, inv_d, n_dot_r
