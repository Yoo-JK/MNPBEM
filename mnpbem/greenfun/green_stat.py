"""
Quasistatic Green function.
"""

import numpy as np
from typing import Optional, Tuple, Union
from scipy import sparse

from ..particles import Particle, Point


class GreenStat:
    """
    Green function for quasistatic approximation (Coulomb).

    The quasistatic Green function is G(r, r') = 1 / (4*pi*|r - r'|).

    Parameters
    ----------
    p1 : Particle or Point
        Source points/particle.
    p2 : Particle
        Target particle (boundary elements).
    rel_cutoff : float
        Cutoff parameter for face integration refinement.

    Attributes
    ----------
    p1 : Particle or Point
        Source points.
    p2 : Particle
        Target particle.
    """

    def __init__(
        self,
        p1: Union[Particle, Point],
        p2: Particle,
        rel_cutoff: float = 3.0,
        deriv: str = 'cart',
        **kwargs
    ):
        """
        Initialize quasistatic Green function.

        Parameters
        ----------
        p1 : Particle or Point
            Source points/particle.
        p2 : Particle
            Target particle.
        rel_cutoff : float
            Cutoff for integration refinement.
        deriv : str
            Derivative mode: 'cart' or 'norm'.
        """
        self.p1 = p1
        self.p2 = p2
        self.rel_cutoff = rel_cutoff
        self.deriv = deriv

        # Cache for computed matrices
        self._G = None
        self._F = None
        self._Gp = None

    @property
    def pos1(self) -> np.ndarray:
        """Source positions."""
        return self.p1.pos

    @property
    def pos2(self) -> np.ndarray:
        """Target positions (face centroids)."""
        return self.p2.pos

    @property
    def n1(self) -> int:
        """Number of source points."""
        return len(self.pos1)

    @property
    def n2(self) -> int:
        """Number of target faces."""
        return self.p2.n_faces

    def _compute_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distances and distance vectors between all point pairs.

        Returns
        -------
        d : ndarray
            Distance matrix, shape (n1, n2).
        dr : ndarray
            Distance vectors, shape (n1, n2, 3).
        """
        # dr[i, j] = pos2[j] - pos1[i]
        dr = self.pos2[np.newaxis, :, :] - self.pos1[:, np.newaxis, :]
        d = np.linalg.norm(dr, axis=2)
        return d, dr

    def G(self) -> np.ndarray:
        """
        Compute Green function matrix.

        G[i, j] = 1 / (4*pi*|r_j - r_i|) * area[j]

        Returns
        -------
        ndarray
            Green function matrix, shape (n1, n2).
        """
        if self._G is not None:
            return self._G

        d, _ = self._compute_distances()

        # Avoid division by zero on diagonal
        d = np.where(d == 0, np.inf, d)

        # Green function: 1 / (4*pi*r)
        G = 1.0 / (4 * np.pi * d)

        # Multiply by face areas for integration
        G = G * self.p2.area[np.newaxis, :]

        self._G = G
        return self._G

    def F(self) -> np.ndarray:
        """
        Compute surface derivative of Green function.

        F[i, j] = n_j . grad_j G(r_i, r_j) * area[j]
                = n_j . (r_j - r_i) / (4*pi*|r_j - r_i|^3) * area[j]

        Returns
        -------
        ndarray
            Surface derivative matrix, shape (n1, n2).
        """
        if self._F is not None:
            return self._F

        d, dr = self._compute_distances()

        # Avoid division by zero
        d = np.where(d == 0, np.inf, d)
        d3 = d ** 3

        # Normal vectors at target faces
        nvec = self.p2.nvec  # (n2, 3)

        # n . (r2 - r1) / (4*pi*r^3)
        # dr[i, j, k] * nvec[j, k] summed over k
        n_dot_dr = np.sum(dr * nvec[np.newaxis, :, :], axis=2)

        F = n_dot_dr / (4 * np.pi * d3)

        # Multiply by face areas
        F = F * self.p2.area[np.newaxis, :]

        self._F = F
        return self._F

    def H1(self) -> np.ndarray:
        """
        F + 2*pi (for inside boundary condition).

        Returns
        -------
        ndarray
            Modified surface derivative, shape (n1, n2).
        """
        return self.F() + 2 * np.pi * np.eye(self.n1, self.n2)

    def H2(self) -> np.ndarray:
        """
        F - 2*pi (for outside boundary condition).

        Returns
        -------
        ndarray
            Modified surface derivative, shape (n1, n2).
        """
        return self.F() - 2 * np.pi * np.eye(self.n1, self.n2)

    def Gp(self) -> np.ndarray:
        """
        Compute gradient of Green function.

        Gp[i, j, :] = grad_j G(r_i, r_j) * area[j]
                    = (r_j - r_i) / (4*pi*|r_j - r_i|^3) * area[j]

        Returns
        -------
        ndarray
            Green function gradient, shape (n1, n2, 3).
        """
        if self._Gp is not None:
            return self._Gp

        d, dr = self._compute_distances()

        # Avoid division by zero
        d = np.where(d == 0, np.inf, d)
        d3 = d ** 3

        # Gradient: (r2 - r1) / (4*pi*r^3)
        Gp = dr / (4 * np.pi * d3[:, :, np.newaxis])

        # Multiply by face areas
        Gp = Gp * self.p2.area[np.newaxis, :, np.newaxis]

        self._Gp = Gp
        return self._Gp

    def potential(self, sig: np.ndarray) -> np.ndarray:
        """
        Compute potential from surface charges.

        phi = G @ sig

        Parameters
        ----------
        sig : ndarray
            Surface charges, shape (n2,) or (n2, n_exc).

        Returns
        -------
        ndarray
            Potential at source points.
        """
        return self.G() @ sig

    def field(self, sig: np.ndarray) -> np.ndarray:
        """
        Compute electric field from surface charges.

        E = -Gp @ sig

        Parameters
        ----------
        sig : ndarray
            Surface charges, shape (n2,) or (n2, n_exc).

        Returns
        -------
        ndarray
            Electric field at source points, shape (n1, 3) or (n1, 3, n_exc).
        """
        Gp = self.Gp()

        if sig.ndim == 1:
            # E[i, k] = -sum_j Gp[i, j, k] * sig[j]
            return -np.einsum('ijk,j->ik', Gp, sig)
        else:
            # Multiple excitations
            return -np.einsum('ijk,jl->ikl', Gp, sig)

    def clear_cache(self) -> None:
        """Clear cached matrices."""
        self._G = None
        self._F = None
        self._Gp = None

    def __repr__(self) -> str:
        return f"GreenStat(n1={self.n1}, n2={self.n2})"
