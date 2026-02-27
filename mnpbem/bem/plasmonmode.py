import numpy as np
from scipy.sparse.linalg import eigs
from typing import Tuple, Any, Optional

from ..greenfun import CompGreenStat


def plasmonmode(
        p: Any,
        nev: int = 20,
        **options: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB: BEM/plasmonmode.m

    Compute plasmon eigenmodes for a discretized particle surface.

    The eigenvalue problem is solved for the surface derivative F of the
    quasistatic Green function.  Eigenvalues correspond to plasmon
    eigenenergies and eigenvectors to the associated surface charge
    distributions.

    Parameters
    ----------
    p : ComParticle
        Compound of discretized particles (see comparticle).
    nev : int
        Number of eigenmodes to compute.  Defaults to 20.
    **options
        Additional keyword arguments forwarded to CompGreenStat.

    Returns
    -------
    ene : np.ndarray, shape (nev,)
        Plasmon eigenenergies (sorted by ascending real part).
    ur : np.ndarray, shape (n, nev)
        Right eigenvectors (surface charge patterns), columns sorted to
        match *ene*.
    ul : np.ndarray, shape (nev, n)
        Left eigenvectors, rows sorted to match *ene*.
    """

    # Green function and its surface derivative F  (MATLAB: compgreenstat)
    g = CompGreenStat(p, p, **options)
    F = g.F  # (n, n)

    n = F.shape[0]

    # Clamp nev so it does not exceed the matrix size minus 1
    # (scipy.sparse.linalg.eigs requires k < n)
    nev_actual = min(nev, n - 1) if n > 1 else 1

    eigs_opts = dict(which = 'SR', maxiter = 1000)

    if nev_actual < n - 1:
        # sparse eigenvalue solver (same as MATLAB eigs(..., 'sr'))
        # scipy eigs returns (eigenvalues, eigenvectors) -- note the order
        # is reversed compared to MATLAB which returns (V, D).

        # left eigenvectors = eigenvectors of F^T
        _, ul = eigs(F.T, k = nev_actual, **eigs_opts)
        ul = ul.T  # (nev, n)

        # right eigenvectors and eigenvalues
        ene_diag, ur = eigs(F, k = nev_actual, **eigs_opts)
        # ur: (n, nev),  ene_diag: (nev,)
    else:
        # matrix too small for sparse solver -- fall back to dense eig
        ene_all, ur_all = np.linalg.eig(F)
        idx_sort = np.argsort(ene_all.real)[:nev_actual]
        ur = ur_all[:, idx_sort]
        ene_diag = ene_all[idx_sort]

        ene_all_l, ul_all = np.linalg.eig(F.T)
        idx_sort_l = np.argsort(ene_all_l.real)[:nev_actual]
        ul = ul_all[:, idx_sort_l].T  # (nev, n)

    # make eigenvectors bi-orthogonal  (MATLAB: ul = (ul * ur) \ ul)
    overlap = ul @ ur  # (nev, nev)
    ul = np.linalg.solve(overlap, ul)

    # extract eigenvalues and sort by ascending real part
    ene = ene_diag.real if np.all(np.isreal(ene_diag)) else ene_diag
    sort_idx = np.argsort(ene.real)
    ene = ene[sort_idx].real

    ur = ur[:, sort_idx]
    ul = ul[sort_idx, :]

    return ene, ur, ul
