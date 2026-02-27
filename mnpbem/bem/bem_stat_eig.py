"""
BEM solver for quasistatic approximation with eigenmode expansion.

MATLAB: BEM/@bemstateig/
Given an external excitation, BEMStatEig computes the surface charges
such that the boundary conditions of Maxwell's equations in the
quasistatic approximation (using eigenmode expansion) are fulfilled.

Reference:
    Garcia de Abajo and Howie, PRB 65, 115418 (2002)
    Hohenester et al., PRL 103, 106801 (2009)
"""

import numpy as np
from typing import Optional, Tuple, Any

from ..greenfun import CompGreenStat, CompStruct
from .plasmonmode import plasmonmode


class BEMStatEig(object):
    """BEM solver for quasistatic approximation and eigenmode expansion.

    Given an external excitation, BEMStatEig computes the surface
    charges such that the boundary conditions of Maxwell's equations in
    the quasistatic approximation (using eigenmode expansion) are fulfilled.

    MATLAB: @bemstateig

    Parameters
    ----------
    p : ComParticle
        Composite particle (see comparticle)
    nev : int
        Number of eigenmodes to compute.  Defaults to 20.
    enei : float, optional
        Light wavelength in vacuum for pre-initialization

    Properties
    ----------
    name : str
        'bemsolver' (constant)
    needs : dict
        {'sim': 'stat', 'nev': True} (constant)
    p : ComParticle
        Composite particle
    nev : int
        Number of eigenmodes
    ene : ndarray, shape (nev,)
        Plasmon eigenenergies
    ur : ndarray, shape (n, nev)
        Right eigenvectors (surface charge patterns)
    ul : ndarray, shape (nev, n)
        Left eigenvectors
    unit : ndarray, shape (nev^2, np)
        Unit matrices for eigenmode expansion
    enei : float or None
        Light wavelength in vacuum
    mat : ndarray or None
        Resolvent matrix
    g : CompGreenStat
        Green function

    Methods
    -------
    __init__(p, nev=20, enei=None, **options)
        Initialize quasistatic BEM solver with eigenmode expansion
    solve(exc)
        Solve BEM equations for given excitation
    __truediv__(exc)
        Surface charge for given excitation (operator \\)
    __mul__(sig)
        Induced potential for given surface charge (operator *)
    field(sig, inout=2)
        Electric field inside/outside of particle surface
    potential(sig, inout=2)
        Potentials and surface derivatives inside/outside of particle
    __call__(enei)
        Computes resolvent matrix for later use in solve

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.bem import BEMStatEig
    >>>
    >>> # Create gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Create BEM solver
    >>> bem = BEMStatEig(p, nev=20)
    >>>
    >>> # Solve for excitation
    >>> sig, bem = bem.solve(exc)
    """

    name = 'bemsolver'
    needs = {'sim': 'stat', 'nev': True}

    def __init__(self,
            p,
            nev = 20,
            enei = None,
            **options):
        """Initialize quasistatic BEM solver with eigenmode expansion.

        MATLAB: bemstateig.m

        Parameters
        ----------
        p : ComParticle
            Compound of particles (see comparticle)
        nev : int
            Number of eigenmodes to compute.  Defaults to 20.
        enei : float, optional
            Light wavelength in vacuum
        **options : dict
            Additional options passed to CompGreenStat
        """
        self.p = p
        self.nev = nev
        self.enei = None  # type: Optional[float]

        # resolvent matrix
        self.mat = None  # type: Optional[np.ndarray]

        # Green function
        self.g = CompGreenStat(p, p, **options)

        # surface derivative of Green function
        F = self.g.F  # (n, n)

        # eigenmode expansion using plasmonmode
        ene, ur, ul = plasmonmode(p, nev = nev, **options)

        # actual number of eigenmodes (may be less than requested)
        self.nev = len(ene)

        self.ene = np.diag(ene)  # (nev, nev) diagonal matrix
        self.ur = ur  # (n, nev)
        self.ul = ul  # (nev, n)

        # unit matrices for eigenmode expansion
        # MATLAB: unit(:, ip) = reshape(ul(:, ind) * ur(ind, :), nev^2, 1)
        self.unit = np.zeros((self.nev ** 2, p.np), dtype = complex)
        for ip in range(p.np):
            ind = p.index_func(ip + 1)
            chunk = self.ul[:, ind] @ self.ur[ind, :]  # (nev, nev)
            self.unit[:, ip] = chunk.ravel()

        if enei is not None:
            self._init_matrices(enei)

    def _init_matrices(self, enei):
        """Initialize resolvent matrix for BEM solver.

        MATLAB: @bemstateig/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        self : BEMStatEig
            Returns self for chaining
        """
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # dielectric functions per boundary pair
        eps_vals = [eps_func(enei)[0] for eps_func in self.p.eps]

        eps1_arr = np.array([eps_vals[int(self.p.inout[j, 0]) - 1]
                             for j in range(self.p.inout.shape[0])])
        eps2_arr = np.array([eps_vals[int(self.p.inout[j, 1]) - 1]
                             for j in range(self.p.inout.shape[0])])

        # Lambda [Garcia de Abajo, Eq. (23)]
        Lambda = 2 * np.pi * (eps1_arr + eps2_arr) / (eps1_arr - eps2_arr)

        # BEM resolvent matrix from eigenmodes
        # unit @ Lambda gives (nev^2,) vector, reshaped to (nev, nev)
        unit_lambda = self.unit @ Lambda[:]  # (nev^2,)
        unit_lambda_mat = unit_lambda.reshape(self.nev, self.nev)
        resolvent = unit_lambda_mat + self.ene  # (nev, nev)

        # mat = -ur @ inv(resolvent) @ ul
        self.mat = -self.ur @ np.linalg.solve(resolvent, self.ul)

        self.enei = enei
        return self

    def solve(self, exc):
        """Solve BEM equations for given excitation.

        MATLAB: @bemstateig/solve.m

        Parameters
        ----------
        exc : CompStruct
            compstruct with field 'phip' for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with field for surface charge
        obj : BEMStatEig
            Updated solver
        """
        return self.__truediv__(exc)

    def __truediv__(self, exc):
        """Surface charge for given excitation.

        MATLAB: @bemstateig/mldivide.m

        Parameters
        ----------
        exc : CompStruct
            compstruct with field 'phip' for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with field for surface charge
        obj : BEMStatEig
            Updated solver
        """
        self._init_matrices(exc.enei)

        sig_result = _matmul(self.mat, exc.phip)
        sig = CompStruct(self.p, exc.enei, sig = sig_result)

        return sig, self

    def __mul__(self, sig):
        """Induced potential for given surface charge.

        MATLAB: @bemstateig/mtimes.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with fields for surface charge

        Returns
        -------
        phi : CompStruct
            compstruct with fields for induced potential
        """
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
                phi1 = pot1.phi1, phi1p = pot1.phi1p,
                phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def potential(self, sig, inout = 2):
        """Potentials and surface derivatives inside/outside of particle.

        MATLAB: @bemstateig/potential.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with surface charges
        inout : int, optional
            Potential inside (inout=1) or outside (inout=2, default)

        Returns
        -------
        pot : CompStruct
            compstruct object with potentials
        """
        return self.g.potential(sig, inout)

    def field(self, sig, inout = 2):
        """Electric field inside/outside of particle surface.

        MATLAB: @bemstateig/field.m

        Parameters
        ----------
        sig : CompStruct
            COMPSTRUCT object with surface charges
        inout : int, optional
            Electric field inside (inout=1) or outside (inout=2, default)

        Returns
        -------
        field : CompStruct
            COMPSTRUCT object with electric field
        """
        return self.g.field(sig, inout)

    def __call__(self, enei):
        """Computes resolvent matrix for later use in solve.

        MATLAB: @bemstateig/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        self : BEMStatEig
            Returns self for chaining
        """
        return self._init_matrices(enei)

    def __repr__(self):
        """String representation."""
        status = 'enei={}'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatEig(p={}, nev={}, {})'.format(self.p, self.nev, status)


def _matmul(a, x):
    """Generalized matrix multiplication for tensors.

    MATLAB: Misc/matmul.m

    Handles scalar, 1D, 2D, and higher-dimensional inputs.
    For a 2D matrix a and a multi-dimensional x, the multiplication
    is performed along the first axis of x.
    """
    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
        if a == 0:
            return 0
        return a * x
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
        if x == 0:
            return 0
        return a * x

    siza = a.shape
    sizx = x.shape if hasattr(x, 'shape') else (len(x),)

    if len(siza) == 3:
        # a is (n1, 3, n2), x is (n2,) or (n2, ...)
        n1, _, n2 = siza
        if len(sizx) == 1:
            return np.tensordot(a, x, axes = ([2], [0]))
        else:
            a_flat = a.reshape(n1 * 3, n2)
            x_flat = x.reshape(n2, -1)
            y_flat = a_flat @ x_flat
            new_shape = (n1, 3) + sizx[1:]
            return y_flat.reshape(new_shape)
    else:
        # Standard 2D matrix multiplication
        if len(sizx) == 1:
            return a @ x
        else:
            x_flat = x.reshape(sizx[0], -1)
            y_flat = a @ x_flat
            return y_flat.reshape((siza[0],) + sizx[1:])
