"""
Composite Green function for quasistatic approximation.

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np


class CompGreenStat:
    """
    Green function for composite particles in quasistatic approximation.

    Computes the F matrix (surface derivative of Green function) for
    boundary element method in the quasistatic regime.

    Parameters
    ----------
    p1 : ComParticle
        Field points (where we evaluate)
    p2 : ComParticle
        Source points (integration surface)

    Attributes
    ----------
    p1, p2 : ComParticle
        Particle objects
    F : ndarray, shape (n1, n2)
        Surface derivative of Green function matrix

    Notes
    -----
    MATLAB MNPBEM convention (greenstat/eval1.m):
        G(r, r') = 1 / |r - r'|  (no 4π factor, absorbed in BEM)

        F[i,j] = - n_i · (r_i - r_j) / |r_i - r_j|³ × area_j

    where n_i is the normal at FIELD point (p1), not source point.

    For closed surfaces, diagonal elements are set to -2π
    (Fuchs & Liu, PRB 14, 5521, 1976).

    Examples
    --------
    >>> from mnpbem import EpsConst, trisphere, ComParticle
    >>> from mnpbem.greenfun import CompGreenStat
    >>>
    >>> # Create 10nm gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Compute Green function
    >>> g = CompGreenStat(p, p)
    >>> print(f"F matrix shape: {g.F.shape}")
    """

    def __init__(self, p1, p2):
        """
        Initialize Green function between p1 and p2.

        Parameters
        ----------
        p1 : ComParticle
            Field particle (evaluation points)
        p2 : ComParticle
            Source particle (integration surface)
        """
        self.p1 = p1
        self.p2 = p2

        # Compute G and F matrices
        self._compute_GF()

    def _compute_GF(self):
        """
        Compute G and F matrices following MATLAB MNPBEM exactly.

        MATLAB greenstat/eval1.m:
            x = bsxfun(@minus, pos1(:,1), pos2(:,1)');  % r_i - r_j
            nvec = obj.p1.nvec;  % normal at FIELD point (p1)
            F = -(n·r) / d³ * area
            G = 1 / d * area
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

        # G matrix: G[i,j] = 1/d[i,j] * area[j]
        # MATLAB: G = 1 ./ d * area;
        G = (1.0 / d) * area2[np.newaxis, :]  # (n1, n2)

        # F matrix: F[i,j] = -nvec1[i] · r[i,j] / d[i,j]³ * area[j]
        # MATLAB: F = -(in(x,1) + in(y,2) + in(z,3)) ./ d.^3 * area
        # where in(x,i) = bsxfun(@times, x, nvec(:,i))
        # This computes nvec1[i] · r[i,j,:] for each (i,j)
        n_dot_r = np.sum(nvec1[:, np.newaxis, :] * r, axis=2)  # (n1, n2)
        F = -n_dot_r / (d ** 3) * area2[np.newaxis, :]  # (n1, n2)

        # Handle diagonal elements for self-interaction (p1 == p2)
        # MATLAB compgreenstat/init.m: diag(obj.g, ind, -2*pi*dir - f.')
        if self.p1 is self.p2:
            # For closed surfaces, diagonal = -2π
            np.fill_diagonal(G, 0.0)  # G diagonal is undefined (self-term)
            np.fill_diagonal(F, -2.0 * np.pi)

        self.G = G
        self.F = F

    def H1(self):
        """
        Return H1 matrix: F + 2π on diagonal.

        MATLAB greenstat/eval1.m:
            H1 = F + 2*pi*(d==0)
        """
        H1 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H1, np.diag(self.F) + 2.0 * np.pi)
        return H1

    def H2(self):
        """
        Return H2 matrix: F - 2π on diagonal.

        MATLAB greenstat/eval1.m:
            H2 = F - 2*pi*(d==0)
        """
        H2 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H2, np.diag(self.F) - 2.0 * np.pi)
        return H2

    def _compute_Gp(self):
        """
        Compute Cartesian derivative of Green function.

        MATLAB greenstat/eval1.m (case 'cart'):
            Gp = -[x/d³, y/d³, z/d³] * area
        """
        if hasattr(self, '_Gp'):
            return self._Gp

        pos1 = self.p1.pos
        pos2 = self.p2.pos
        area2 = self.p2.area

        # Position difference: r[i,j,:] = pos1[i] - pos2[j]
        r = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]  # (n1, n2, 3)
        d = np.linalg.norm(r, axis=2)
        d = np.maximum(d, np.finfo(float).eps)

        # Gp[i,j,:] = -r[i,j,:] / d[i,j]³ * area[j]
        # Shape: (n1, n2, 3)
        Gp = -r / (d[:, :, np.newaxis] ** 3) * area2[np.newaxis, :, np.newaxis]

        # Reshape to (n1, 3, n2) to match MATLAB convention
        self._Gp = np.transpose(Gp, (0, 2, 1))

        return self._Gp

    def H1p(self):
        """
        Return H1p matrix: Gp + 2π*outer(nvec, d==0).

        MATLAB greenstat/eval1.m:
            H1p = Gp + 2*pi*outer(nvec, d==0)

        Returns
        -------
        H1p : ndarray, shape (n1, 3, n2)
            Cartesian derivative + 2π term on diagonal
        """
        Gp = self._compute_Gp()
        H1p = Gp.copy()

        if self.p1 is self.p2:
            # Add 2π * nvec[i] for diagonal elements (i == j)
            nvec = self.p1.nvec
            for i in range(len(nvec)):
                H1p[i, :, i] += 2 * np.pi * nvec[i]

        return H1p

    def H2p(self):
        """
        Return H2p matrix: Gp - 2π*outer(nvec, d==0).

        MATLAB greenstat/eval1.m:
            H2p = Gp - 2*pi*outer(nvec, d==0)

        Returns
        -------
        H2p : ndarray, shape (n1, 3, n2)
            Cartesian derivative - 2π term on diagonal
        """
        Gp = self._compute_Gp()
        H2p = Gp.copy()

        if self.p1 is self.p2:
            # Subtract 2π * nvec[i] for diagonal elements (i == j)
            nvec = self.p1.nvec
            for i in range(len(nvec)):
                H2p[i, :, i] -= 2 * np.pi * nvec[i]

        return H2p

    def __repr__(self):
        return (
            f"CompGreenStat(p1: {self.p1.nfaces} faces, "
            f"p2: {self.p2.nfaces} faces)"
        )

    def __str__(self):
        return (
            f"Quasistatic Green Function:\n"
            f"  Field (p1): {self.p1.nfaces} faces\n"
            f"  Source (p2): {self.p2.nfaces} faces\n"
            f"  G matrix: {self.G.shape}\n"
            f"  F matrix: {self.F.shape}\n"
            f"  G range: [{self.G.min():.4e}, {self.G.max():.4e}]\n"
            f"  F range: [{self.F.min():.4e}, {self.F.max():.4e}]"
        )
