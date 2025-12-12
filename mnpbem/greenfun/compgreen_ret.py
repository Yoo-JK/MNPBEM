"""
Composite Green function for retarded (full Maxwell) approximation.
"""

import numpy as np


class CompGreenRet:
    """
    Green function for composite particles with full Maxwell equations.

    Computes the G and F matrices (Green function and surface derivative)
    for boundary element method in the retarded regime.

    Parameters
    ----------
    p1 : ComParticle
        Source points (where we evaluate)
    p2 : ComParticle
        Field points (where Green function is evaluated)
    enei : float
        Photon energy in eV or wavelength in nm

    Attributes
    ----------
    p1, p2 : ComParticle
        Particle objects
    enei : float
        Photon energy/wavelength
    k : complex
        Wavenumber in the medium
    G : ndarray, shape (n1, n2)
        Green function matrix (scalar potential)
    F : ndarray, shape (n1, n2)
        Surface derivative of Green function matrix

    Notes
    -----
    The retarded Green function is:
        G(r, r') = exp(ik|r - r'|) / (4π|r - r'|)

    The surface derivative is:
        F = ∂G/∂n' = n'·∇'G = n'·(r' - r)·(ik - 1/r)/(4π r²) · exp(ikr)

    For closed surfaces, diagonal elements are set to F ± 2π.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.greenfun import CompGreenRet
    >>>
    >>> # Create 10nm gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Compute Green function at 600 nm
    >>> g = CompGreenRet(p, p, 600.0)
    >>> print(f"G matrix shape: {g.G.shape}")
    >>> print(f"F matrix shape: {g.F.shape}")
    """

    def __init__(self, p1, p2, enei):
        """
        Initialize Green function between p1 and p2 at given energy.

        Parameters
        ----------
        p1 : ComParticle
            Source particle
        p2 : ComParticle
            Field particle
        enei : float
            Photon energy (eV) or wavelength (nm)
        """
        self.p1 = p1
        self.p2 = p2
        self.enei = enei

        # Get wavenumber from material properties
        # For now, use the outside medium (assuming p1 and p2 are the same)
        eps_out, k = p1.eps[0](enei)  # Get from first material (usually vacuum)
        self.k = k

        # Compute G and F matrices
        self._compute_GF()

    def _compute_GF(self):
        """
        Compute G and F matrices (Green function and surface derivative).

        Following MATLAB MNPBEM convention:
        G[i,j] = exp(ik·r_ij) / r_ij * area_j
        F[i,j] = n_j · (r_i - r_j) · (ik - 1/r_ij) / r_ij² · exp(ik·r_ij) * area_j

        where r_ij = |r_i - r_j|

        Note: No 4π factor (absorbed into area/BEM formulation)
        """
        # Get positions, normals, and areas
        pos1 = self.p1.pos  # (n1, 3) - field points
        pos2 = self.p2.pos  # (n2, 3) - source points
        nvec2 = self.p2.nvec  # (n2, 3) - normals at source
        area2 = self.p2.area  # (n2,) - areas at source

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Initialize matrices
        G = np.zeros((n1, n2), dtype=complex)
        F = np.zeros((n1, n2), dtype=complex)

        # Compute all pairwise interactions
        for i in range(n1):
            # Vector from source j to field i: r_i - r_j
            r_ij = pos1[i] - pos2  # (n2, 3)

            # Distance
            dist = np.linalg.norm(r_ij, axis=1)  # (n2,)
            dist = np.maximum(dist, np.finfo(float).eps)  # Avoid division by zero

            # Phase factor
            phase = np.exp(1j * self.k * dist)

            # Green function: G = exp(ikr) / r * area
            G[i, :] = phase / dist * area2

            # Surface derivative: F = (n·r)·(ik - 1/r)/r² · exp(ikr) * area
            # Auxiliary quantity: (ik - 1/r) / r²
            f = (1j * self.k - 1.0 / dist) / (dist ** 2)

            # n·r term
            n_dot_r = np.sum(nvec2 * r_ij, axis=1)

            # Complete F matrix
            F[i, :] = n_dot_r * f * phase * area2

        # Handle diagonal elements for self-interaction (p1 == p2)
        # Note: MATLAB sets diagonal to computed value, then adds/subtracts 2π
        # in H1/H2 methods. We do the same.

        self.G = G
        self.F = F

    def H1(self):
        """
        Return H1 matrix: F + 2π on diagonal.

        Used for BEM solver (inside formulation).
        """
        H1 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H1, np.diag(self.F) + 2.0 * np.pi)
        return H1

    def H2(self):
        """
        Return H2 matrix: F - 2π on diagonal.

        Used for BEM solver (outside formulation).
        """
        H2 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H2, np.diag(self.F) - 2.0 * np.pi)
        return H2

    def __repr__(self):
        return (
            f"CompGreenRet(p1: {self.p1.nfaces} faces, "
            f"p2: {self.p2.nfaces} faces, λ={self.enei:.1f}nm)"
        )

    def __str__(self):
        return (
            f"Retarded Green Function:\n"
            f"  Source (p1): {self.p1.nfaces} faces\n"
            f"  Field (p2): {self.p2.nfaces} faces\n"
            f"  Wavelength: {self.enei:.2f} nm\n"
            f"  Wavenumber k: {self.k:.6f}\n"
            f"  G matrix: {self.G.shape}\n"
            f"  F matrix: {self.F.shape}\n"
            f"  |G| range: [{np.abs(self.G).min():.4e}, {np.abs(self.G).max():.4e}]\n"
            f"  |F| range: [{np.abs(self.F).min():.4e}, {np.abs(self.F).max():.4e}]"
        )
