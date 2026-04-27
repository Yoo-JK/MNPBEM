"""Numba-accelerated trilinear/bilinear interpolation kernels for layer
Green-function tabulation.

Activated via the ``MNPBEM_NUMBA`` environment variable.  Falls back to
``scipy.interpolate.RegularGridInterpolator`` when numba is unavailable
or the variable is unset.

The interpolator mimics RegularGridInterpolator(method='linear',
bounds_error=False, fill_value=None): inside the grid we use plain
multilinear interpolation, outside we extrapolate with the gradient of
the boundary cell (i.e. clamp the cell index but keep the fractional
weight unbounded).
"""

import os
import numpy as np

try:
    import numba
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _numba_enabled() -> bool:
    return _HAS_NUMBA and os.environ.get('MNPBEM_NUMBA', '').strip() not in ('', '0', 'false', 'False')


# ---------------------------------------------------------------------------
# Helpers for index location (linear search; grids are small, n <= ~50)
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, fastmath=False)
    def _locate_cell(grid: np.ndarray, x: float) -> tuple:
        """Locate the cell (i, t) such that grid[i] + t*(grid[i+1]-grid[i]) = x.

        Returns (i, t) clamped so that 0 <= i <= len(grid)-2, with t allowed
        to lie outside [0, 1] (linear extrapolation outside the grid).
        """
        n = grid.shape[0]
        if n < 2:
            return 0, 0.0

        # Binary search for first index i where grid[i+1] > x
        lo = 0
        hi = n - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if grid[mid + 1] <= x:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        denom = grid[i + 1] - grid[i]
        if denom == 0.0:
            t = 0.0
        else:
            t = (x - grid[i]) / denom
        return i, t

    # Serial inner loop: parallel adds ~1-2 ms launch overhead per call
    # which dominates for the typical BEM call sizes (n ~ a few thousand
    # called 90+ times per solve).  Parallel only wins for n >> 10^5.
    @njit(cache=True, parallel=False, fastmath=False)
    def _trilinear_complex(
            grid_r: np.ndarray,
            grid_z1: np.ndarray,
            grid_z2: np.ndarray,
            data: np.ndarray,
            r_q: np.ndarray,
            z1_q: np.ndarray,
            z2_q: np.ndarray,
            out: np.ndarray,
    ) -> None:
        """Trilinear interpolation of complex 3D data on a regular grid.

        data : (nr, nz1, nz2) complex128
        r_q, z1_q, z2_q : (n,) float64
        out : (n,) complex128 — filled in place.
        """
        n = r_q.shape[0]
        for q in range(n):
            ir, tr = _locate_cell(grid_r, r_q[q])
            iz, tz = _locate_cell(grid_z1, z1_q[q])
            iw, tw = _locate_cell(grid_z2, z2_q[q])

            c000 = data[ir,     iz,     iw]
            c100 = data[ir + 1, iz,     iw]
            c010 = data[ir,     iz + 1, iw]
            c110 = data[ir + 1, iz + 1, iw]
            c001 = data[ir,     iz,     iw + 1]
            c101 = data[ir + 1, iz,     iw + 1]
            c011 = data[ir,     iz + 1, iw + 1]
            c111 = data[ir + 1, iz + 1, iw + 1]

            # Interpolate along r
            c00 = c000 * (1.0 - tr) + c100 * tr
            c10 = c010 * (1.0 - tr) + c110 * tr
            c01 = c001 * (1.0 - tr) + c101 * tr
            c11 = c011 * (1.0 - tr) + c111 * tr
            # Along z1
            c0 = c00 * (1.0 - tz) + c10 * tz
            c1 = c01 * (1.0 - tz) + c11 * tz
            # Along z2
            out[q] = c0 * (1.0 - tw) + c1 * tw

    @njit(cache=True, parallel=False, fastmath=False)
    def _bilinear_complex(
            grid_r: np.ndarray,
            grid_z: np.ndarray,
            data: np.ndarray,
            r_q: np.ndarray,
            z_q: np.ndarray,
            out: np.ndarray,
    ) -> None:
        """Bilinear interpolation of complex 2D data on a regular grid.

        data : (nr, nz) complex128
        r_q, z_q : (n,) float64
        out : (n,) complex128 — filled in place.
        """
        n = r_q.shape[0]
        for q in range(n):
            ir, tr = _locate_cell(grid_r, r_q[q])
            iz, tz = _locate_cell(grid_z, z_q[q])

            c00 = data[ir,     iz]
            c10 = data[ir + 1, iz]
            c01 = data[ir,     iz + 1]
            c11 = data[ir + 1, iz + 1]

            c0 = c00 * (1.0 - tr) + c10 * tr
            c1 = c01 * (1.0 - tr) + c11 * tr
            out[q] = c0 * (1.0 - tz) + c1 * tz


def trilinear_complex(grid, data, points):
    """Public wrapper that dispatches to numba when enabled, else RGI.

    grid   : tuple of 1D ndarrays (axis grids)
    data   : complex array with shape == (len(g) for g in grid)
    points : (n, ndim) ndarray of query coordinates
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    if not _numba_enabled():
        return _scipy_fallback(grid, data, points)

    data_c = np.ascontiguousarray(data, dtype=np.complex128)
    out = np.empty(points.shape[0], dtype=np.complex128)

    if data_c.ndim == 3:
        _trilinear_complex(
            np.ascontiguousarray(grid[0], dtype=np.float64),
            np.ascontiguousarray(grid[1], dtype=np.float64),
            np.ascontiguousarray(grid[2], dtype=np.float64),
            data_c,
            np.ascontiguousarray(points[:, 0], dtype=np.float64),
            np.ascontiguousarray(points[:, 1], dtype=np.float64),
            np.ascontiguousarray(points[:, 2], dtype=np.float64),
            out,
        )
    elif data_c.ndim == 2:
        _bilinear_complex(
            np.ascontiguousarray(grid[0], dtype=np.float64),
            np.ascontiguousarray(grid[1], dtype=np.float64),
            data_c,
            np.ascontiguousarray(points[:, 0], dtype=np.float64),
            np.ascontiguousarray(points[:, 1], dtype=np.float64),
            out,
        )
    else:
        return _scipy_fallback(grid, data, points)
    return out


def _scipy_fallback(grid, data, points):
    """Slow path identical to GreenTabLayer._interp_complex."""
    from scipy.interpolate import RegularGridInterpolator
    val_r = RegularGridInterpolator(
        grid, data.real, method='linear',
        bounds_error=False, fill_value=None)(points)
    val_i = RegularGridInterpolator(
        grid, data.imag, method='linear',
        bounds_error=False, fill_value=None)(points)
    return val_r + 1j * val_i
