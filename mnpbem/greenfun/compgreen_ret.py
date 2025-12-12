"""
Composite Green function for retarded (full Maxwell) approximation.

Matches MATLAB MNPBEM implementation exactly.
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
        Field points (where we evaluate)
    p2 : ComParticle
        Source points (integration surface)
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
    MATLAB MNPBEM convention (greenret/eval1.m):
        G(r, r') = exp(ik|r - r'|) / |r - r'|  (no 4π factor)

        F[i,j] = n_i · (r_i - r_j) · (ik - 1/r) / r² · exp(ikr) × area_j

    where n_i is the normal at FIELD point (p1), not source point.
    Note: The sign is POSITIVE (unlike quasistatic which is negative).

    For closed surfaces, H1 = F + 2π, H2 = F - 2π on diagonal.

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
            Field particle (evaluation points)
        p2 : ComParticle
            Source particle (integration surface)
        enei : float
            Photon energy (eV) or wavelength (nm)
        """
        self.p1 = p1
        self.p2 = p2
        self.enei = enei

        # Get wavenumber from material properties
        # Use the outside medium (assuming p1 and p2 are the same)
        eps_out, k = p1.eps[0](enei)  # Get from first material (usually vacuum)
        self.k = k

        # Compute G and F matrices
        self._compute_GF()

    def _compute_GF(self):
        """
        Compute G and F matrices following MATLAB MNPBEM exactly.

        MATLAB greenret/eval1.m:
            x = bsxfun(@minus, pos1(:,1), pos2(:,1)');  % r_i - r_j
            nvec = obj.p1.nvec;  % normal at FIELD point (p1)
            G = 1./d * area .* exp(1i*k*d)
            F = (n·r) · (1i*k - 1/d) / d² * area .* exp(1i*k*d)

        Note: F has POSITIVE sign (unlike quasistatic which is negative)
        """
        # Get positions, normals, and areas
        pos1 = self.p1.pos    # (n1, 3) - field points
        pos2 = self.p2.pos    # (n2, 3) - source points
        nvec1 = self.p1.nvec  # (n1, 3) - normals at FIELD points (MATLAB convention!)
        area2 = self.p2.area  # (n2,) - areas at source

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Vectorized computation (matches MATLAB bsxfun)
        # r[i,j,:] = pos1[i,:] - pos2[j,:] = r_i - r_j
        r = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]  # (n1, n2, 3)

        # Distance d[i,j] = |r_i - r_j|
        d = np.linalg.norm(r, axis=2)  # (n1, n2)
        d = np.maximum(d, np.finfo(float).eps)  # Avoid division by zero

        # Phase factor: exp(i*k*d)
        phase = np.exp(1j * self.k * d)  # (n1, n2)

        # G matrix: G[i,j] = exp(ikd)/d * area[j]
        # MATLAB: G = 1./d * area .* exp(1i*k*d)
        G = (phase / d) * area2[np.newaxis, :]  # (n1, n2)

        # F matrix: F[i,j] = nvec1[i]·r[i,j] · (ik - 1/d) / d² · exp(ikd) * area[j]
        # MATLAB: F = (n·r) .* (1i*k - 1./d) ./ d.^2 * area .* exp(1i*k*d)
        # Note: POSITIVE sign (unlike quasistatic)
        n_dot_r = np.sum(nvec1[:, np.newaxis, :] * r, axis=2)  # (n1, n2)
        f_aux = (1j * self.k - 1.0 / d) / (d ** 2)  # (n1, n2)
        F = n_dot_r * f_aux * phase * area2[np.newaxis, :]  # (n1, n2)

        # Handle diagonal elements for self-interaction (p1 == p2)
        # MATLAB: H1 = F + 2*pi*(d==0), H2 = F - 2*pi*(d==0)
        # Diagonal of F itself is left as computed (will be ~0 or small)
        if self.p1 is self.p2:
            np.fill_diagonal(G, 0.0)  # G diagonal is undefined (self-term)
            # F diagonal: MATLAB leaves it as computed, ±2π added in H1/H2

        self.G = G
        self.F = F

    def H1(self):
        """
        Return H1 matrix: F + 2π on diagonal.

        MATLAB greenret/eval1.m:
            H1 = F + 2*pi*(d==0)

        Used for BEM solver (inside formulation).
        """
        H1 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H1, np.diag(self.F) + 2.0 * np.pi)
        return H1

    def H2(self):
        """
        Return H2 matrix: F - 2π on diagonal.

        MATLAB greenret/eval1.m:
            H2 = F - 2*pi*(d==0)

        Used for BEM solver (outside formulation).
        """
        H2 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H2, np.diag(self.F) - 2.0 * np.pi)
        return H2

    def __repr__(self):
        return (
            f"CompGreenRet(p1: {self.p1.nfaces} faces, "
            f"p2: {self.p2.nfaces} faces, k={self.k:.6f})"
        )

    def __str__(self):
        return (
            f"Retarded Green Function:\n"
            f"  Field (p1): {self.p1.nfaces} faces\n"
            f"  Source (p2): {self.p2.nfaces} faces\n"
            f"  Wavelength: {self.enei:.2f} nm\n"
            f"  Wavenumber k: {self.k:.6f}\n"
            f"  G matrix: {self.G.shape}\n"
            f"  F matrix: {self.F.shape}\n"
            f"  |G| range: [{np.abs(self.G).min():.4e}, {np.abs(self.G).max():.4e}]\n"
            f"  |F| range: [{np.abs(self.F).min():.4e}, {np.abs(self.F).max():.4e}]"
        )
