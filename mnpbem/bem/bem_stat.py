"""
BEM solver for quasistatic approximation.

Given an external excitation, BEMStat computes the surface charges
such that the boundary conditions of Maxwell's equations in the
quasistatic approximation are fulfilled.

Reference:
    Garcia de Abajo and Howie, PRB 65, 115418 (2002)
    Hohenester et al., PRL 103, 106801 (2009)
"""

import numpy as np
from ..greenfun import CompGreenStat


class BEMStat:
    """
    BEM solver for quasistatic approximation.

    Solves the boundary element method equations in the quasistatic regime
    to find surface charges given an external potential.

    Parameters
    ----------
    p : ComParticle
        Composite particle with geometry and material properties
    enei : float, optional
        Photon energy (eV) or wavelength (nm) for pre-initialization

    Attributes
    ----------
    p : ComParticle
        The particle object
    green : CompGreenStat
        Green function object
    F : ndarray
        Surface derivative of Green function matrix
    enei : float or None
        Current wavelength/energy (None if not initialized)
    mat : ndarray or None
        BEM resolvent matrix: -inv(Λ + F)

    Notes
    -----
    The BEM equation in quasistatic approximation is:
        (Λ + F) · σ = -φₚ

    where:
        Λ[i] = 2π(ε₁ + ε₂)/(ε₁ - ε₂)  (diagonal matrix)
        F = surface derivative of Green function
        σ = surface charge distribution
        φₚ = external potential

    The resolvent matrix is:
        mat = -inv(Λ + F)

    So the solution is:
        σ = mat · φₚ

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.bem import BEMStat
    >>>
    >>> # Create gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Create BEM solver
    >>> bem = BEMStat(p)
    >>>
    >>> # Initialize at specific wavelength
    >>> bem.init(600.0)
    >>>
    >>> # Or initialize during construction
    >>> bem = BEMStat(p, enei=600.0)
    """

    def __init__(self, p, enei=None):
        """
        Initialize BEM solver for quasistatic approximation.

        Parameters
        ----------
        p : ComParticle
            Composite particle
        enei : float, optional
            Photon energy (eV) or wavelength (nm) for pre-initialization
        """
        self.p = p
        self.enei = None
        self.mat = None

        # Create Green function
        self.green = CompGreenStat(p, p)
        self.F = self.green.F

        # Initialize at specific energy if provided
        if enei is not None:
            self.init(enei)

    def init(self, enei):
        """
        Initialize BEM solver for specific wavelength/energy.

        Computes the resolvent matrix mat = -inv(Λ + F) for later use
        in solving the BEM equations.

        Parameters
        ----------
        enei : float
            Photon energy (eV) or wavelength (nm)

        Returns
        -------
        self : BEMStat
            Returns self for chaining
        """
        # Skip if already initialized at this energy
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # Get inside and outside dielectric functions
        eps1 = self.p.eps1(enei)  # (nfaces,)
        eps2 = self.p.eps2(enei)  # (nfaces,)

        # Lambda matrix [Garcia de Abajo, Eq. (23)]
        # Λ = 2π(ε₁ + ε₂)/(ε₁ - ε₂)
        lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        # BEM resolvent matrix: mat = -inv(Λ + F)
        Lambda = np.diag(lambda_diag)
        self.mat = -np.linalg.inv(Lambda + self.F)

        # Save energy
        self.enei = enei

        return self

    def solve(self, enei_or_exc, exc=None):
        """
        Solve BEM equations for given excitation.

        Computes surface charges σ from external potential φₚ:
            σ = mat · φₚ = -inv(Λ + F) · φₚ

        Parameters
        ----------
        enei_or_exc : float or dict
            Either wavelength (nm) when exc is provided separately,
            or excitation dict with 'enei' and 'phip' keys
        exc : dict, optional
            Excitation dict with 'phip' key (if enei_or_exc is wavelength)

        Returns
        -------
        sig : dict
            Dictionary containing:
                - sig : surface charge distribution (array, shape (nfaces,) or (nfaces, npol))
                - enei : wavelength/energy
                - p : particle object

        Examples
        --------
        >>> # Method 1: excitation as dict
        >>> result = bem.solve(exc)
        >>>
        >>> # Method 2: enei and exc separately
        >>> result = bem.solve(enei, exc)
        """
        # Handle both calling conventions
        if exc is None:
            # exc contains both enei and phip
            exc = enei_or_exc
            if isinstance(exc, dict):
                enei = exc['enei']
                phip = exc['phip']
            else:
                enei = exc.enei
                phip = exc.phip
        else:
            # enei_or_exc is the energy, exc is the excitation
            enei = enei_or_exc
            if isinstance(exc, dict):
                phip = exc['phip']
            else:
                phip = exc.phip

        # Initialize at this energy
        self.init(enei)

        # Solve: σ = mat · φₚ
        sig = self.mat @ phip

        return {
            'sig': sig,
            'enei': enei,
            'p': self.p
        }

    def __call__(self, enei):
        """
        Initialize solver at specific energy (alternative syntax).

        Parameters
        ----------
        enei : float
            Photon energy or wavelength

        Returns
        -------
        self : BEMStat
            Returns self for chaining
        """
        return self.init(enei)

    def __repr__(self):
        status = f"λ={self.enei:.1f}nm" if self.enei is not None else "not initialized"
        return f"BEMStat(p: {self.p.nfaces} faces, {status})"

    def __str__(self):
        status = f"Initialized at λ={self.enei:.2f} nm" if self.enei is not None else "Not initialized"
        mat_info = f"  Resolvent matrix: {self.mat.shape}" if self.mat is not None else "  Resolvent matrix: Not computed"

        return (
            f"BEM Solver (Quasistatic):\n"
            f"  Particle: {self.p.nfaces} faces\n"
            f"  F matrix: {self.F.shape}\n"
            f"  Status: {status}\n"
            f"{mat_info}"
        )
