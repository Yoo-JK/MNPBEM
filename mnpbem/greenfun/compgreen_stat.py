"""
Composite Green function for quasistatic approximation.
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
        Source points (where we evaluate)
    p2 : ComParticle
        Field points (where Green function is evaluated)

    Attributes
    ----------
    p1, p2 : ComParticle
        Particle objects
    F : ndarray, shape (n1, n2)
        Surface derivative of Green function matrix

    Notes
    -----
    The quasistatic Green function is:
        G(r, r') = 1 / (4π|r - r'|)

    The surface derivative is:
        F = ∂G/∂n' = n'·∇'G = n'·(r' - r) / (4π|r - r'|³)

    For closed surfaces, diagonal elements are set to -2π (Fuchs & Liu, PRB 1976).

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
            Source particle
        p2 : ComParticle
            Field particle
        """
        self.p1 = p1
        self.p2 = p2

        # Compute F matrix
        self._compute_F()

    def _compute_F(self):
        """
        Compute F matrix (surface derivative of Green function).

        Following MATLAB MNPBEM convention:
        F[i,j] = - n_j · (r_i - r_j) / |r_i - r_j|³ * area_j

        Note: No 4π factor (absorbed into area/BEM formulation)
        """
        # Get positions, normals, and areas
        pos1 = self.p1.pos  # (n1, 3) - field points
        pos2 = self.p2.pos  # (n2, 3) - source points
        nvec2 = self.p2.nvec  # (n2, 3) - normals at source
        area2 = self.p2.area  # (n2,) - areas at source

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Initialize F matrix
        F = np.zeros((n1, n2))

        # Compute all pairwise interactions
        for i in range(n1):
            # Vector from source j to field i: r_i - r_j
            r_ij = pos1[i] - pos2  # (n2, 3)

            # Distance
            dist = np.linalg.norm(r_ij, axis=1)  # (n2,)

            # Avoid division by zero for diagonal elements (handled separately)
            with np.errstate(divide='ignore', invalid='ignore'):
                # F = - (n · r) / r³ * area
                # Minus sign is part of MATLAB convention
                dist3 = dist**3
                n_dot_r = np.sum(nvec2 * r_ij, axis=1)
                F[i, :] = - n_dot_r / dist3 * area2

        # Handle diagonal elements for self-interaction (p1 == p2)
        if self.p1 is self.p2:
            # For closed surfaces, diagonal should be -2π
            # This is the discontinuity from the solid angle
            np.fill_diagonal(F, -2.0 * np.pi)

        self.F = F

    def __repr__(self):
        return (
            f"CompGreenStat(p1: {self.p1.nfaces} faces, "
            f"p2: {self.p2.nfaces} faces)"
        )

    def __str__(self):
        return (
            f"Quasistatic Green Function:\n"
            f"  Source (p1): {self.p1.nfaces} faces\n"
            f"  Field (p2): {self.p2.nfaces} faces\n"
            f"  F matrix: {self.F.shape}\n"
            f"  F range: [{self.F.min():.4f}, {self.F.max():.4f}]"
        )
