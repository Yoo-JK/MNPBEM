"""
Point collection class.
"""

import numpy as np
from typing import Optional, Union


class Point:
    """
    Collection of points for field evaluation.

    Parameters
    ----------
    pos : ndarray
        Point positions, shape (n_points, 3).
    vec : ndarray, optional
        Basis vectors at each point, shape (n_points, 3, 3).
        vec[i, 0] = tangent1, vec[i, 1] = tangent2, vec[i, 2] = normal.

    Attributes
    ----------
    pos : ndarray
        Point positions.
    vec : ndarray
        Basis vectors at each point.
    """

    def __init__(
        self,
        pos: np.ndarray,
        vec: Optional[np.ndarray] = None
    ):
        """
        Initialize point collection.

        Parameters
        ----------
        pos : ndarray
            Point positions, shape (n_points, 3).
        vec : ndarray, optional
            Basis vectors at each point.
        """
        self.pos = np.asarray(pos, dtype=float)
        if self.pos.ndim == 1:
            self.pos = self.pos.reshape(1, -1)

        if vec is None:
            # Default: identity basis at each point
            n = len(self.pos)
            self.vec = np.tile(np.eye(3), (n, 1, 1))
        else:
            self.vec = np.asarray(vec, dtype=float)

    @property
    def n_points(self) -> int:
        """Number of points."""
        return len(self.pos)

    @property
    def nvec(self) -> np.ndarray:
        """Normal vectors (3rd column of vec)."""
        return self.vec[:, 2, :]

    @property
    def tvec1(self) -> np.ndarray:
        """First tangent vectors."""
        return self.vec[:, 0, :]

    @property
    def tvec2(self) -> np.ndarray:
        """Second tangent vectors."""
        return self.vec[:, 1, :]

    def shift(self, displacement: np.ndarray) -> 'Point':
        """
        Shift points by given displacement.

        Parameters
        ----------
        displacement : array_like
            Displacement vector [dx, dy, dz].

        Returns
        -------
        Point
            Shifted points.
        """
        displacement = np.asarray(displacement)
        return Point(self.pos + displacement, self.vec.copy())

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return f"Point(n_points={self.n_points})"


def meshgrid_points(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    plane: str = 'xy'
) -> Point:
    """
    Create a grid of points.

    Parameters
    ----------
    x : ndarray
        X coordinates.
    y : ndarray
        Y coordinates.
    z : ndarray or float, optional
        Z coordinates. If scalar, creates 2D grid at that z.
    plane : str
        Plane for 2D grid: 'xy', 'xz', or 'yz'.

    Returns
    -------
    Point
        Grid of points.
    """
    if z is None:
        z = np.array([0.0])
    elif np.isscalar(z):
        z = np.array([z])

    if len(z) == 1:
        # 2D grid
        xx, yy = np.meshgrid(x, y, indexing='ij')
        zz = np.full_like(xx, z[0])
    else:
        # 3D grid
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return Point(pos)
