"""
Point class - Collection of points.

MATLAB: Particles/@point
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Callable


class Point:
    """
    Collection of points.

    MATLAB: @point

    Properties
    ----------
    pos : ndarray, shape (n, 3)
        Coordinates of points
    vec : list of 3 ndarrays, each shape (n, 3)
        Basis vectors {x, y, z} at positions

    Examples
    --------
    >>> pt = Point(np.array([[0, 0, 0], [1, 0, 0]]))
    >>> pt.n
    2
    """

    def __init__(self, pos: np.ndarray, vec: Optional[List[np.ndarray]] = None):
        """
        Initialize Point collection.

        MATLAB: point.m

        Parameters
        ----------
        pos : ndarray, shape (n, 3)
            Coordinates of points
        vec : list of 3 ndarrays, optional
            Basis vectors {x, y, z} at positions.
            If None, uses default Cartesian basis.
        """
        self.pos = np.atleast_2d(pos).astype(float)

        # Basis vectors
        if vec is not None and len(vec) > 0:
            self.vec = vec
        else:
            n = self.pos.shape[0]
            self.vec = [
                np.tile([1.0, 0.0, 0.0], (n, 1)),
                np.tile([0.0, 1.0, 0.0], (n, 1)),
                np.tile([0.0, 0.0, 1.0], (n, 1))
            ]

    @property
    def n(self) -> int:
        """Number of points."""
        return self.pos.shape[0]

    @property
    def size(self) -> int:
        """Number of points (alias for n)."""
        return self.n

    @property
    def nvec(self) -> np.ndarray:
        """
        z-component of vec (normal vector).

        MATLAB: subsref.m - case 'nvec'
        """
        return self.vec[2]

    @property
    def tvec(self) -> List[np.ndarray]:
        """
        x,y-components of vec (tangent vectors).

        MATLAB: subsref.m - case 'tvec'
        """
        return [self.vec[0], self.vec[1]]

    @property
    def tvec1(self) -> np.ndarray:
        """
        x-component of vec (first tangent vector).

        MATLAB: subsref.m - case 'tvec1'
        """
        return self.vec[0]

    @property
    def tvec2(self) -> np.ndarray:
        """
        y-component of vec (second tangent vector).

        MATLAB: subsref.m - case 'tvec2'
        """
        return self.vec[1]

    def select(self, method: str, value: Union[np.ndarray, Callable]) -> Tuple['Point', Optional['Point']]:
        """
        Select points.

        MATLAB: select.m

        Parameters
        ----------
        method : str
            Selection method:
            - 'index' or 'ind': Direct index selection
            - 'carfun' or 'cartfun': Function f(x, y, z) for Cartesian selection
            - 'polfun': Function f(phi, r, z) for cylindrical selection
            - 'sphfun': Function f(phi, theta, r) for spherical selection
        value : ndarray or callable
            Index array or selection function

        Returns
        -------
        obj1 : Point
            Selected points
        obj2 : Point or None
            Complement of selected points (if requested)
        """
        x = self.pos[:, 0]
        y = self.pos[:, 1]
        z = self.pos[:, 2]

        if method in ['ind', 'index']:
            index = np.asarray(value)
        elif method in ['carfun', 'cartfun']:
            index = np.where(value(x, y, z))[0]
        elif method == 'polfun':
            # Cylindrical coordinates
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            index = np.where(value(phi, r, z))[0]
        elif method == 'sphfun':
            # Spherical coordinates
            r = np.sqrt(x**2 + y**2 + z**2)
            phi = np.arctan2(y, x)
            theta = np.arccos(z / np.maximum(r, 1e-10))  # theta from z-axis
            index = np.where(value(phi, theta, r))[0]
        else:
            raise ValueError(f"Unknown selection method: {method}")

        # Selected points
        obj1 = self._compress(index)

        # Complement
        all_indices = np.arange(self.n)
        complement_index = np.setdiff1d(all_indices, index)
        obj2 = self._compress(complement_index) if len(complement_index) > 0 else None

        return obj1, obj2

    def _compress(self, index: np.ndarray) -> 'Point':
        """
        Compress point collection to selected indices.

        MATLAB: select.m - compress function
        """
        new_pos = self.pos[index, :]
        new_vec = [
            self.vec[0][index, :],
            self.vec[1][index, :],
            self.vec[2][index, :]
        ]
        return Point(new_pos, new_vec)

    def __add__(self, other: 'Point') -> 'Point':
        """
        Concatenate points.

        MATLAB: vertcat.m
        """
        return self.vertcat(other)

    def vertcat(self, *others: 'Point') -> 'Point':
        """
        Concatenate points vertically.

        MATLAB: vertcat.m

        Parameters
        ----------
        *others : Point
            Other Point objects to concatenate

        Returns
        -------
        Point
            Concatenated Point object
        """
        new_pos = self.pos.copy()
        new_vec = [v.copy() for v in self.vec]

        for other in others:
            new_pos = np.vstack([new_pos, other.pos])
            for dim in range(3):
                new_vec[dim] = np.vstack([new_vec[dim], other.vec[dim]])

        return Point(new_pos, new_vec)

    def __repr__(self) -> str:
        """Command window display."""
        return f"Point(n={self.n}, pos_range=[{self.pos.min():.2f}, {self.pos.max():.2f}])"

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index) -> 'Point':
        """Index-based access to points."""
        return self._compress(np.atleast_1d(index))
