"""MATLAB Engine bridge for BEM solver linear solves (Wave 66).

Opt-in helper: when BEMRetLayer is constructed with use_matlab_engine=True,
the dense complex linear solve of the 2n x 2n block matrix and downstream
solves are routed through a MATLAB Engine session. This mirrors MATLAB's
mldivide / lu / solve numerical behavior exactly, eliminating residual
differences from numpy's scipy lu_factor / lu_solve.
"""
import os

import numpy as np


_engine = None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    import matlab.engine as _matlab_engine_mod

    eng = _matlab_engine_mod.start_matlab()
    eng.addpath(eng.genpath('/home/yoojk20/workspace/MNPBEM'), nargout=0)
    helper_dir = os.path.dirname(os.path.abspath(__file__))
    eng.addpath(helper_dir, nargout=0)

    _engine = eng
    return eng


def matlab_solve(M, b):
    """Solve M x = b using MATLAB's mldivide.

    Parameters
    ----------
    M : ndarray (n, n) complex
    b : ndarray (n,) or (n, k) complex

    Returns
    -------
    x : ndarray, same shape as b, complex
    """
    import matlab as _matlab_mod

    eng = get_engine()

    M = np.asarray(M, dtype=complex)
    b = np.asarray(b, dtype=complex)

    one_d = (b.ndim == 1)
    if one_d:
        b2 = b.reshape(-1, 1)
    else:
        b2 = b

    M_r = _matlab_mod.double(np.real(M).tolist())
    M_i = _matlab_mod.double(np.imag(M).tolist())
    b_r = _matlab_mod.double(np.real(b2).tolist())
    b_i = _matlab_mod.double(np.imag(b2).tolist())

    x_r, x_i = eng.mnpbem_bem_solve_helper(
        M_r, M_i, b_r, b_i, nargout=2)

    x_r_arr = np.asarray(x_r, dtype=float)
    x_i_arr = np.asarray(x_i, dtype=float)
    if x_r_arr.ndim == 0:
        x_r_arr = x_r_arr.reshape(1, 1)
    elif x_r_arr.ndim == 1:
        x_r_arr = x_r_arr.reshape(-1, 1)
    if x_i_arr.ndim == 0:
        x_i_arr = x_i_arr.reshape(1, 1)
    elif x_i_arr.ndim == 1:
        x_i_arr = x_i_arr.reshape(-1, 1)

    x = x_r_arr + 1j * x_i_arr

    if one_d:
        return x.ravel()
    return x.reshape(b2.shape)
