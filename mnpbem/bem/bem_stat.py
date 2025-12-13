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
        # Handle multi-dimensional phip (e.g., from DipoleStat with shape (nfaces, npt, ndip))
        if phip.ndim == 1:
            # (nfaces,) -> (nfaces,)
            sig = self.mat @ phip
        elif phip.ndim == 2:
            # (nfaces, npol) -> (nfaces, npol)
            sig = self.mat @ phip
        else:
            # (nfaces, npt, ndip) or higher dimensions
            # Use einsum: result[i, j, k, ...] = sum_m mat[i, m] * phip[m, j, k, ...]
            original_shape = phip.shape
            nfaces = original_shape[0]
            # Reshape to (nfaces, -1)
            phip_reshaped = phip.reshape(nfaces, -1)
            sig_reshaped = self.mat @ phip_reshaped
            # Reshape back
            sig = sig_reshaped.reshape(original_shape)

        return {
            'sig': sig,
            'enei': enei,
            'p': self.p
        }

    def field(self, sig, inout=2):
        """
        Compute electric field inside/outside of particle surface.

        MATLAB: bemstat/field.m

        Parameters
        ----------
        sig : dict
            Solution containing 'sig' (surface charges), 'enei', 'p'
        inout : int, optional
            1 for inside, 2 for outside (default: 2)

        Returns
        -------
        field : dict
            Dictionary with 'e' electric field, shape (nfaces, 3) or (nfaces, 3, npol)
        """
        surface_charge = sig['sig']
        enei = sig['enei']

        # Compute electric field from Green function derivative
        # MATLAB: e = -matmul(Hp, sig.sig)
        # Hp has shape (n1, 3, n2), sig has shape (n2,) or (n2, npol)
        if inout == 1:
            # Inside: use H1p (derivative toward inside)
            Hp = self.green.H1p()  # (n1, 3, n2)
        else:
            # Outside: use H2p (derivative toward outside)
            Hp = self.green.H2p()  # (n1, 3, n2)

        # Electric field: e = -Hp @ sig
        # Result: (n1, 3) or (n1, 3, npol)
        if surface_charge.ndim == 1:
            # Single polarization: e[i, xyz] = -sum_j Hp[i, xyz, j] * sig[j]
            e = -np.einsum('ijk,k->ij', Hp, surface_charge)
        else:
            # Multiple polarizations: e[i, xyz, pol] = -sum_j Hp[i, xyz, j] * sig[j, pol]
            e = -np.einsum('ijk,kp->ijp', Hp, surface_charge)

        return {
            'e': e,
            'enei': enei,
            'p': self.p
        }

    def potential(self, sig, inout=2):
        """
        Compute potential and surface derivative inside/outside of particle.

        MATLAB: bemstat/potential.m

        Parameters
        ----------
        sig : dict
            Solution containing 'sig' (surface charges), 'enei', 'p'
        inout : int, optional
            1 for inside, 2 for outside (default: 2)

        Returns
        -------
        pot : dict
            Dictionary with potential and surface derivative:
            - phi1/phi2: scalar potential
            - phi1p/phi2p: surface derivative
        """
        surface_charge = sig['sig']
        enei = sig['enei']

        # Get Green function and surface derivative
        G = self.green.G

        if inout == 1:
            H = self.green.H1()
        else:
            H = self.green.H2()

        # Potential and surface derivative
        phi = G @ surface_charge
        phip = H @ surface_charge

        if inout == 1:
            return {
                'phi1': phi,
                'phi1p': phip,
                'enei': enei,
                'p': self.p
            }
        else:
            return {
                'phi2': phi,
                'phi2p': phip,
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
