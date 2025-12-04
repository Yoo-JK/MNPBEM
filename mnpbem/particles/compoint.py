"""
Compound point class for field evaluation points.
"""

import numpy as np
from typing import List, Any

from .point import Point
from .compound import Compound


class ComPoint(Compound):
    """
    Compound of points in a dielectric environment.

    Used for defining field evaluation points with associated
    dielectric properties.

    Parameters
    ----------
    eps : list
        List of dielectric function objects.
    p : list
        List of Point objects.
    inout : ndarray
        Index to medium eps for each point set.

    Examples
    --------
    >>> from mnpbem import EpsConst, ComPoint
    >>> from mnpbem.particles import Point
    >>>
    >>> eps_vacuum = EpsConst(1.0)
    >>> points = Point(np.random.rand(100, 3) * 20 - 10)
    >>> cp = ComPoint([eps_vacuum], [points], [1])
    """

    def __init__(
        self,
        eps: List[Any],
        p: List[Point],
        inout: np.ndarray
    ):
        """
        Initialize compound point.

        Parameters
        ----------
        eps : list
            List of dielectric function objects.
        p : list
            List of Point objects.
        inout : ndarray
            Medium indices for each point set.
        """
        # Convert single point to list
        if isinstance(p, Point):
            p = [p]

        super().__init__(eps, p, inout)

    @property
    def pos(self) -> np.ndarray:
        """All point positions."""
        return self.pc.pos

    @property
    def n_points(self) -> int:
        """Total number of points."""
        return self.pc.n_points

    def shift(self, displacement: np.ndarray) -> 'ComPoint':
        """
        Shift all points by given displacement.

        Parameters
        ----------
        displacement : array_like
            Displacement vector [dx, dy, dz].

        Returns
        -------
        ComPoint
            Shifted compound point.
        """
        new_p = [p.shift(displacement) for p in self.p]
        return ComPoint(self.eps, new_p, self.inout)

    def flip(self, axis: int) -> 'ComPoint':
        """
        Flip all points along axis.

        Parameters
        ----------
        axis : int
            Axis to flip.

        Returns
        -------
        ComPoint
            Flipped compound point.
        """
        new_p = []
        for pt in self.p:
            new_pos = pt.pos.copy()
            new_pos[:, axis] = -new_pos[:, axis]
            new_p.append(Point(new_pos, pt.vec.copy()))
        return ComPoint(self.eps, new_p, self.inout)

    def __repr__(self) -> str:
        return (f"ComPoint(n_eps={len(self.eps)}, "
                f"n_point_sets={len(self.p)}, "
                f"n_points={self.n_points})")
