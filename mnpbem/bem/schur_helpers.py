"""
Schur complement reduction for cover-layer BEM (MNPBEM v1.2.0).

When a particle carries an artificial nonlocal cover layer (EpsNonlocal +
coverlayer.shift), the BEM mesh face count doubles -- shell faces +
core faces. For the dense direct solver this inflates the BEM matrix to
(2N, 2N) and the LU factor cost by 8x.

The cover layer enters the BEM equations only as a thin boundary
condition; the shell sub-matrix G_ss tends to be small and well
conditioned. Schur complement elimination of the shell variables
collapses the system back to the (M, M) core block

    [G_ss G_sc] [sig_s]   [b_s]
    [G_cs G_cc] [sig_c] = [b_c]

becomes

    G_eff = G_cc - G_cs @ inv(G_ss) @ G_sc
    b_eff = b_c  - G_cs @ inv(G_ss) @ b_s
    sig_c = inv(G_eff) @ b_eff
    sig_s = inv(G_ss) @ (b_s - G_sc @ sig_c)

The reduced solve produces results that are mathematically identical to
the full block solve (up to floating-point round-off), but the dominant
LU factor now operates on an (M, M) matrix instead of (M+N, M+N).
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def schur_eliminate(M_full: np.ndarray,
        shell_indices: np.ndarray,
        core_indices: np.ndarray) -> Tuple[np.ndarray, Callable, Callable]:

    s = np.asarray(shell_indices, dtype = int)
    c = np.asarray(core_indices, dtype = int)

    if s.size == 0:
        # No shell -- nothing to eliminate. Return identity-like wrappers
        # so callers can use the same code path.
        def _identity_rhs(b: np.ndarray) -> np.ndarray:
            return b[c]

        def _identity_recover(sig_core: np.ndarray, b_full: np.ndarray) -> np.ndarray:
            return sig_core

        return M_full[np.ix_(c, c)], _identity_rhs, _identity_recover

    M_ss = M_full[np.ix_(s, s)]
    M_sc = M_full[np.ix_(s, c)]
    M_cs = M_full[np.ix_(c, s)]
    M_cc = M_full[np.ix_(c, c)]

    lu_ss, piv_ss = lu_factor(M_ss, check_finite = False)
    M_inv_sc = lu_solve((lu_ss, piv_ss), M_sc, check_finite = False)
    M_eff = M_cc - M_cs @ M_inv_sc

    def reduce_rhs(b_full: np.ndarray) -> np.ndarray:
        b_s = b_full[s]
        b_c = b_full[c]
        if b_s.ndim == 1:
            corr = M_cs @ lu_solve((lu_ss, piv_ss), b_s, check_finite = False)
        else:
            corr = M_cs @ lu_solve((lu_ss, piv_ss),
                    b_s.reshape(b_s.shape[0], -1), check_finite = False).reshape(b_s.shape)
        return b_c - corr

    def recover_full(sig_core: np.ndarray, b_full: np.ndarray) -> np.ndarray:
        b_s = b_full[s]
        rhs_s = b_s - M_sc @ sig_core
        if rhs_s.ndim == 1:
            sig_s = lu_solve((lu_ss, piv_ss), rhs_s, check_finite = False)
        else:
            sig_s = lu_solve((lu_ss, piv_ss),
                    rhs_s.reshape(rhs_s.shape[0], -1), check_finite = False).reshape(rhs_s.shape)

        out_shape = list(b_full.shape)
        out = np.empty(out_shape, dtype = np.result_type(sig_core, sig_s))
        out[s] = sig_s
        out[c] = sig_core
        return out

    return M_eff, reduce_rhs, recover_full


def detect_shell_core_partition(particle: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:

    from ..materials import EpsNonlocal

    eps_list = getattr(particle, 'eps', None)
    inout = getattr(particle, 'inout_faces', None)

    if eps_list is None or inout is None:
        return None

    # 1-based MATLAB indices in inout. Identify which eps slots are EpsNonlocal.
    nonlocal_eps_idx_1based = set()
    for i, eps in enumerate(eps_list):
        if isinstance(eps, EpsNonlocal):
            nonlocal_eps_idx_1based.add(i + 1)

    if not nonlocal_eps_idx_1based:
        return None

    inout_arr = np.asarray(inout)
    nfaces = inout_arr.shape[0]

    # Convention (matches the EpsNonlocal cover-layer geometry built by
    # ``coverlayer.shift`` + ``make_nonlocal_pair``):
    #
    #   shell particle row in inout : [nonlocal_eps_idx, embed_eps_idx]
    #                                 -> EpsNonlocal sits on the *inside*
    #                                    of the shell face, embed on the
    #                                    outside.
    #   core particle row in inout  : [metal_eps_idx, nonlocal_eps_idx]
    #                                 -> EpsNonlocal sits on the *outside*
    #                                    of the core face, metal on the
    #                                    inside.
    #
    # Schur reduction targets the artificial cover-layer (shell) faces --
    # i.e. those whose *inside* material is EpsNonlocal.  Reducing the
    # core faces would also work mathematically but would defeat the
    # memory savings (core block is the larger one).
    in_col = inout_arr[:, 0].astype(int)
    shell_mask = np.array([int(idx) in nonlocal_eps_idx_1based for idx in in_col])
    shell_indices = np.where(shell_mask)[0]
    core_indices = np.where(~shell_mask)[0]

    if shell_indices.size == 0 or core_indices.size == 0:
        # Either no shell faces (no EpsNonlocal on the inside of any face)
        # or no remaining core faces -- in both cases the reduction is a
        # no-op or degenerate, so return None and let the caller fall back
        # to the full BEM matrix.
        return None

    return shell_indices, core_indices


def schur_memory_estimate(nfaces_total: int, nfaces_shell: int) -> dict:

    nfaces_core = nfaces_total - nfaces_shell

    # Each complex matrix entry is 16 bytes.
    bytes_per_entry = 16

    full_bytes = nfaces_total * nfaces_total * bytes_per_entry
    reduced_bytes = nfaces_core * nfaces_core * bytes_per_entry
    schur_overhead = (nfaces_shell * nfaces_shell + 2 * nfaces_shell * nfaces_core) * bytes_per_entry

    return {
        'nfaces_total': nfaces_total,
        'nfaces_shell': nfaces_shell,
        'nfaces_core': nfaces_core,
        'full_matrix_bytes': full_bytes,
        'reduced_matrix_bytes': reduced_bytes,
        'schur_temp_bytes': schur_overhead,
        'reduction_ratio': reduced_bytes / full_bytes if full_bytes else 0.0,
    }
