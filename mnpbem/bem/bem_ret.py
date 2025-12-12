"""
BEM solver for full Maxwell equations (retarded).

Given an external excitation, BEMRet computes the surface charges
and currents such that the boundary conditions of Maxwell's equations
are fulfilled.

Reference:
    Garcia de Abajo and Howie, PRB 65, 115418 (2002)
"""

import numpy as np
from ..greenfun import CompGreenRet


class BEMRet:
    """
    BEM solver for full Maxwell equations (retarded).

    Solves the boundary element method equations in the retarded regime
    to find surface charges and currents given external fields.

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
    green : CompGreenRet
        Green function object
    enei : float or None
        Current wavelength/energy (None if not initialized)
    k : float or None
        Wavenumber in vacuum (2π/λ)

    Notes
    -----
    The BEM equations for retarded case are more complex than quasistatic,
    involving coupled equations for surface charges (σ) and currents (h).

    The implementation follows Garcia de Abajo and Howie, PRB 65, 115418 (2002),
    Equations (19-22).

    Key matrices:
        G1, G2 : Green functions (inside/outside)
        H1, H2 : Surface derivatives with ±2π terms
        L1, L2 : G * ε * G^(-1) matrices
        Sigma1, Sigma2 : H * G^(-1) matrices
        Deltai : inv(Sigma1 - Sigma2)
        Sigmai : Inverse of combined Sigma matrix

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.bem import BEMRet
    >>>
    >>> # Create gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Create BEM solver
    >>> bem = BEMRet(p)
    >>>
    >>> # Initialize at specific wavelength
    >>> bem.init(600.0)
    """

    def __init__(self, p, enei=None):
        """
        Initialize BEM solver for retarded approximation.

        Parameters
        ----------
        p : ComParticle
            Composite particle
        enei : float, optional
            Photon energy (eV) or wavelength (nm) for pre-initialization
        """
        self.p = p
        self.enei = None

        # Green function will be created on demand
        self.green = None

        # BEM matrices (initialized on demand)
        self.k = None
        self.nvec = None
        self.eps1 = None
        self.eps2 = None
        self.G1i = None
        self.G2i = None
        self.L1 = None
        self.L2 = None
        self.Sigma1 = None
        self.Deltai = None
        self.Sigmai = None

        # Initialize at specific energy if provided
        if enei is not None:
            self.init(enei)

    def init(self, enei):
        """
        Initialize BEM solver for specific wavelength/energy.

        Computes all necessary matrices for solving the BEM equations.

        Parameters
        ----------
        enei : float
            Photon energy (eV) or wavelength (nm)

        Returns
        -------
        self : BEMRet
            Returns self for chaining
        """
        # Skip if already initialized at this energy
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        self.enei = enei

        # Create Green function if not exists
        if self.green is None:
            self.green = CompGreenRet(self.p, self.p, enei)
        else:
            # Update Green function for new energy
            self.green = CompGreenRet(self.p, self.p, enei)

        # Outer surface normals
        self.nvec = self.p.nvec

        # Wavenumber in vacuum
        self.k = 2 * np.pi / enei

        # Dielectric functions
        eps1_vals = self.p.eps1(enei)  # (nfaces,)
        eps2_vals = self.p.eps2(enei)  # (nfaces,)

        # Check if all values are the same (can use scalar)
        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(eps2_vals, eps2_vals[0]):
            self.eps1 = eps1_vals[0]
            self.eps2 = eps2_vals[0]
        else:
            self.eps1 = np.diag(eps1_vals)
            self.eps2 = np.diag(eps2_vals)

        # Get Green functions and surface derivatives
        # For single particle (p1 == p2), we have simple case
        G = self.green.G
        H1_mat = self.green.H1()
        H2_mat = self.green.H2()

        # For now, implement simple case (single homogeneous particle)
        # G1 = G (inside to inside)
        # G2 = G (outside to outside)
        # This is simplified - full version would handle connectivity matrix

        G1 = G
        G2 = G

        # Compute inverses
        self.G1i = np.linalg.inv(G1)
        self.G2i = np.linalg.inv(G2)

        # L matrices [Eq. (22)]
        # L1 = G1 * eps1 * G1^(-1)
        # L2 = G2 * eps2 * G2^(-1)
        if np.isscalar(self.eps1):
            self.L1 = self.eps1
            self.L2 = self.eps2
        else:
            self.L1 = G1 @ self.eps1 @ self.G1i
            self.L2 = G2 @ self.eps2 @ self.G2i

        # Sigma matrices [Eq. (21)]
        # Sigma1 = H1 * G1^(-1)
        # Sigma2 = H2 * G2^(-1)
        self.Sigma1 = H1_mat @ self.G1i
        Sigma2 = H2_mat @ self.G2i

        # Inverse Delta matrix
        self.Deltai = np.linalg.inv(self.Sigma1 - Sigma2)

        # Combined Sigma matrix and its inverse
        L = self.L1 - self.L2 if not np.isscalar(self.L1) else (self.L1 - self.L2)

        if np.isscalar(L):
            # Simplified case
            Sigma = self.Sigma1 * self.L1 - Sigma2 * self.L2
        else:
            # Full matrix case with magnetic coupling term
            nvec_outer = self.nvec @ self.nvec.T
            Sigma = (self.Sigma1 @ self.L1 - Sigma2 @ self.L2 +
                     self.k**2 * ((L @ self.Deltai) * nvec_outer) @ L)

        self.Sigmai = np.linalg.inv(Sigma)

        return self

    def solve(self, exc):
        """
        Solve BEM equations for given excitation.

        Computes surface charges and currents from external fields.

        Parameters
        ----------
        exc : dict or object
            Excitation object with fields:
                - enei : wavelength/energy
                - phi, a, alpha, De : excitation potentials/fields

        Returns
        -------
        sig : dict
            Dictionary containing:
                - sig1, sig2 : surface charge distributions (inside/outside)
                - h1, h2 : surface current distributions (inside/outside)
                - enei : wavelength/energy
                - p : particle object

        Notes
        -----
        This implements Equations (19-20) from Garcia de Abajo & Howie (2002).
        """
        # Initialize at excitation energy if needed
        self.init(exc.enei)

        # Extract excitation fields
        phi = exc.phi
        a = exc.a
        alpha = exc.alpha
        De = exc.De

        # Get stored variables
        k = self.k
        nvec = self.nvec
        G1i = self.G1i
        G2i = self.G2i
        L1 = self.L1
        L2 = self.L2
        Sigma1 = self.Sigma1
        Deltai = self.Deltai
        Sigmai = self.Sigmai

        # Modify alpha and De [Eqs. before (19)]
        if np.isscalar(L1):
            L1_phi = L1 * phi
            L1_a = L1 * a
        else:
            L1_phi = L1 @ phi
            L1_a = L1 @ a

        alpha_mod = alpha - Sigma1 @ a + 1j * k * (nvec.T * L1_phi).T
        De_mod = De - Sigma1 @ L1_phi + 1j * k * np.sum(nvec * L1_a[:, None], axis=1)

        # Eq. (19): surface charge
        if np.isscalar(L1):
            L_diff = L1 - L2
        else:
            L_diff = L1 - L2

        nvec_dot_L_diff_Deltai_alpha = np.sum(
            nvec * ((L_diff @ Deltai @ alpha_mod)[:, None] if not np.isscalar(L_diff) else (L_diff * Deltai @ alpha_mod)[:, None]),
            axis=1
        )

        sig2 = Sigmai @ (De_mod + 1j * k * nvec_dot_L_diff_Deltai_alpha)

        # Eq. (20): surface current
        L_diff_sig2 = L_diff @ sig2 if not np.isscalar(L_diff) else L_diff * sig2
        h2 = Deltai @ (1j * k * (nvec.T * L_diff_sig2).T + alpha_mod)

        # Surface charges and currents [from Eqs. (10-11)]
        sig1 = G1i @ (sig2 + phi)
        h1 = G1i @ (h2 + a)
        sig2 = G2i @ sig2
        h2 = G2i @ h2

        return {
            'sig1': sig1,
            'sig2': sig2,
            'h1': h1,
            'h2': h2,
            'enei': exc.enei,
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
        self : BEMRet
            Returns self for chaining
        """
        return self.init(enei)

    def __repr__(self):
        status = f"λ={self.enei:.1f}nm" if self.enei is not None else "not initialized"
        return f"BEMRet(p: {self.p.nfaces} faces, {status})"

    def __str__(self):
        status = f"Initialized at λ={self.enei:.2f} nm" if self.enei is not None else "Not initialized"
        mat_info = f"  Sigmai matrix: {self.Sigmai.shape}" if self.Sigmai is not None else "  Sigmai matrix: Not computed"

        return (
            f"BEM Solver (Retarded/Full Maxwell):\n"
            f"  Particle: {self.p.nfaces} faces\n"
            f"  Status: {status}\n"
            f"{mat_info}\n"
            f"  Wavenumber k: {self.k:.6f}" if self.k is not None else "  Wavenumber k: Not computed"
        )
