"""
Polygon class - 2D polygons for use with mesh generation.

MATLAB: Particles/@polygon
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from scipy.interpolate import CubicSpline
from matplotlib.path import Path


class Polygon:
    """
    2D polygons for use with mesh generation.

    MATLAB: @polygon

    Properties
    ----------
    pos : ndarray, shape (n, 2)
        Positions of polygon vertices
    dir : int
        Direction of polygon (1 = counterclockwise, -1 = clockwise)
    sym : str or None
        Symmetry keyword: None, 'x', 'y', or 'xy'

    Examples
    --------
    >>> poly = Polygon(n=6)  # Hexagon
    >>> poly = Polygon(pos=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    """

    def __init__(self, n: Optional[int] = None, pos: Optional[np.ndarray] = None,
                 dir: int = 1, sym: Optional[str] = None, size: Optional[np.ndarray] = None):
        """
        Initialize polygon.

        MATLAB: polygon.m, private/init.m

        Parameters
        ----------
        n : int, optional
            Number of vertices (creates regular polygon)
        pos : ndarray, optional
            Positions of vertices (n, 2)
        dir : int, optional
            Direction (1 = counterclockwise, -1 = clockwise)
        sym : str, optional
            Symmetry keyword: None, 'x', 'y', or 'xy'
        size : array-like, optional
            Scale polygon to [width, height]
        """
        # Initialize positions
        if pos is not None:
            self.pos = np.atleast_2d(pos).astype(float)
        elif n is not None:
            phi = np.arange(n) / n * 2 * np.pi + np.pi / n
            self.pos = np.column_stack([np.cos(phi), np.sin(phi)])
        else:
            raise ValueError("Either n or pos must be provided")

        self.dir = dir
        self.sym = sym

        # Scale if size is provided
        if size is not None:
            size = np.atleast_1d(size)
            if len(size) == 1:
                size = np.array([size[0], size[0]])
            x_range = self.pos[:, 0].max() - self.pos[:, 0].min()
            y_range = self.pos[:, 1].max() - self.pos[:, 1].min()
            if x_range > 0:
                self.pos[:, 0] = size[0] / x_range * self.pos[:, 0]
            if y_range > 0:
                self.pos[:, 1] = size[1] / y_range * self.pos[:, 1]

        # Apply symmetry
        if self.sym is not None:
            self._apply_symmetry()

    @property
    def n(self) -> int:
        """Number of vertices."""
        return self.pos.shape[0]

    def shift(self, vec: np.ndarray) -> 'Polygon':
        """
        Shift polygon by given vector.

        MATLAB: shift.m

        Parameters
        ----------
        vec : array-like, shape (2,)
            Translation vector [dx, dy]

        Returns
        -------
        Polygon
            Shifted polygon
        """
        vec = np.atleast_1d(vec)
        self.pos = self.pos + vec
        return self

    def scale(self, scale_factor: Union[float, np.ndarray]) -> 'Polygon':
        """
        Scale polygon.

        MATLAB: scale.m

        Parameters
        ----------
        scale_factor : float or array-like
            Scaling factor [sx, sy] or single value

        Returns
        -------
        Polygon
            Scaled polygon
        """
        scale_factor = np.atleast_1d(scale_factor)
        if len(scale_factor) == 1:
            scale_factor = np.array([scale_factor[0], scale_factor[0]])
        self.pos = self.pos * scale_factor
        return self

    def rot(self, angle: float) -> 'Polygon':
        """
        Rotate polygon by given angle.

        MATLAB: rot.m

        Parameters
        ----------
        angle : float
            Rotation angle in degrees

        Returns
        -------
        Polygon
            Rotated polygon
        """
        angle_rad = angle / 180 * np.pi
        rot_matrix = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ])
        self.pos = self.pos @ rot_matrix
        return self

    def flip(self, ax: int = 0) -> 'Polygon':
        """
        Flip polygon along given axis.

        MATLAB: flip.m

        Parameters
        ----------
        ax : int
            Axis to flip along (0 = x, 1 = y)

        Returns
        -------
        Polygon
            Flipped polygon
        """
        self.pos[:, ax] = -self.pos[:, ax]
        return self

    def norm(self) -> np.ndarray:
        """
        Normal vector at polygon positions.

        MATLAB: norm.m

        Returns
        -------
        nvec : ndarray, shape (n, 2)
            Normal vectors at each vertex
        """
        pos = self.pos
        n = pos.shape[0]

        # Unit vector function
        def unit(v):
            norms = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
            norms = np.maximum(norms, 1e-10)
            return v / norms

        # Edge vectors
        vec = np.roll(pos, -1, axis=0) - pos

        # Normal vectors (perpendicular to edges)
        nvec = np.column_stack([-vec[:, 1], vec[:, 0]])
        nvec = unit(nvec)

        # Interpolate to polygon positions (average of adjacent edge normals)
        nvec = (nvec + np.roll(nvec, 1, axis=0)) / 2
        nvec = unit(nvec)

        # Check direction of normal vectors
        posp = pos + 1e-6 * nvec
        path = Path(pos)
        in_polygon = path.contains_points(posp)

        if self.dir == 1:
            nvec[in_polygon, :] = -nvec[in_polygon, :]
        elif self.dir == -1:
            nvec[~in_polygon, :] = -nvec[~in_polygon, :]

        # Normal vector at symmetry points
        if self.sym is not None:
            if self.sym in ['x', 'xy']:
                nvec[np.abs(pos[:, 0]) < 1e-10, 0] = 0
            if self.sym in ['y', 'xy']:
                nvec[np.abs(pos[:, 1]) < 1e-10, 1] = 0
            nvec = unit(nvec)

        return nvec

    def dist(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find distance of point positions from polygon.

        MATLAB: dist.m

        Parameters
        ----------
        pos : ndarray, shape (m, 2)
            Query positions

        Returns
        -------
        dmin : ndarray, shape (m,)
            Minimum distance of each position to polygon
        imin : ndarray, shape (m,)
            Nearest polygon vertex index for each position
        """
        pos = np.atleast_2d(pos)
        m = pos.shape[0]

        dmin = np.full(m, 1e10)
        imin = np.zeros(m, dtype=int)

        xa = self.pos[:, 0]
        ya = self.pos[:, 1]
        xb = np.roll(xa, -1)
        yb = np.roll(ya, -1)

        for j in range(m):
            x, y = pos[j, 0], pos[j, 1]

            # Project point onto each edge
            denom = (xb - xa)**2 + (yb - ya)**2
            denom = np.maximum(denom, 1e-20)
            lam = ((xb - xa) * (x - xa) + (yb - ya) * (y - ya)) / denom
            lam = np.clip(lam, 0, 1)

            # Distance to each edge
            d = np.sqrt((xa + lam * (xb - xa) - x)**2 +
                        (ya + lam * (yb - ya) - y)**2)

            ind = np.argmin(d)
            dmin[j] = d[ind]
            imin[j] = ind

        return dmin, imin

    def round(self, rad: Optional[float] = None, nrad: int = 5,
              edge: Optional[np.ndarray] = None) -> 'Polygon':
        """
        Round edges of polygon.

        MATLAB: round.m

        Parameters
        ----------
        rad : float, optional
            Radius of rounded edges (default: 0.1 * max extent)
        nrad : int, optional
            Number of interpolation points for rounded edges
        edge : array-like, optional
            Mask for edges that should be rounded (0-indexed)

        Returns
        -------
        Polygon
            Polygon with rounded edges
        """
        pos = self.pos

        if rad is None:
            rad = 0.1 * np.max(np.abs(pos))
        if edge is None:
            edge = np.arange(pos.shape[0])

        def unit(v):
            norms = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
            norms = np.maximum(norms, 1e-10)
            return v / norms

        # Edge direction vectors
        vec = unit(np.roll(pos, -1, axis=0) - pos)
        # Direction to circle center
        dir_vec = unit(vec - np.roll(vec, 1, axis=0))

        # Angle between edges
        dot_prod = np.sum(np.roll(vec, 1, axis=0) * vec, axis=1)
        beta = np.arccos(np.clip(dot_prod, -1, 1)) / 2

        # Circle centers
        zero = pos + rad * dir_vec / np.cos(beta)[:, np.newaxis]

        # Check if center is inside polygon
        path = Path(pos)
        in_poly = path.contains_points(zero)
        sgn = np.where(in_poly, 1, -1)

        # Build rounded polygon
        new_pos = []

        def rot_matrix(phi):
            return np.array([
                [np.cos(phi), np.sin(phi)],
                [-np.sin(phi), np.cos(phi)]
            ])

        for i in range(pos.shape[0]):
            if i not in edge:
                new_pos.append(pos[i, :])
            else:
                if np.abs(beta[i]) < 1e-3:
                    angles = [0]
                else:
                    angles = beta[i] * np.linspace(-1, 1, nrad) * sgn[i]

                for phi in angles:
                    pt = zero[i, :] - rad * dir_vec[i, :] @ rot_matrix(phi)
                    new_pos.append(pt)

        self.pos = np.array(new_pos)
        return self

    def close(self) -> 'Polygon':
        """
        Close polygon in case of xy-symmetry.

        MATLAB: close.m

        Returns
        -------
        Polygon
            Closed polygon
        """
        if self.sym is None or self.sym != 'xy':
            return self

        self._sort()
        pos = self.pos

        # Add origin to position list if needed
        if (not np.allclose(pos[-1, :], [0, 0], atol=1e-6) and
            np.abs(np.prod(pos[0, :])) < 1e-6 and
            np.abs(np.prod(pos[-1, :])) < 1e-6):
            self.pos = np.vstack([pos, [0, 0]])

        return self

    def _sort(self) -> 'Polygon':
        """
        Sort polygon positions for symmetry.

        MATLAB: sort.m
        """
        if self.sym is None:
            return self

        # Find positions on x and/or y axis
        ind = []
        if self.sym in ['x', 'xy']:
            ind.extend(np.where(np.abs(self.pos[:, 0]) < 1e-6)[0])
        if self.sym in ['y', 'xy']:
            ind.extend(np.where(np.abs(self.pos[:, 1]) < 1e-6)[0])
        ind = np.unique(ind)

        # Shift positions so first/last point are on axis
        if len(ind) >= 2 and ind[0] != 0:
            shift_amount = self.pos.shape[0] - ind[-1]
            self.pos = np.roll(self.pos, shift_amount, axis=0)

        return self

    def _apply_symmetry(self):
        """Apply symmetry transformation."""
        if self.sym is None:
            return

        # Keep only positions in symmetry region
        if self.sym == 'x':
            mask = self.pos[:, 0] >= 0
        elif self.sym == 'y':
            mask = self.pos[:, 1] >= 0
        elif self.sym == 'xy':
            mask = (self.pos[:, 0] >= 0) & (self.pos[:, 1] >= 0)
        else:
            return

        self.pos = self.pos[mask, :]
        self._sort()

    def union(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get positions and connections for mesh generation.

        MATLAB: union.m

        Returns
        -------
        upos : ndarray, shape (n, 2)
            Unified positions
        unet : ndarray, shape (n, 2)
            Edge connectivity (0-indexed)
        """
        n = self.pos.shape[0]
        net = np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])
        return self.pos.copy(), net

    def midpoints(self, same: bool = False) -> 'Polygon':
        """
        Add midpoints for polygon positions using spline interpolation.

        MATLAB: midpoints.m

        Parameters
        ----------
        same : bool, optional
            If True, positions already include midpoints, smooth only

        Returns
        -------
        Polygon
            Polygon with midpoint positions
        """
        if same:
            pos = np.vstack([self.pos[::2, :], self.pos[0, :]])
        else:
            pos = np.vstack([self.pos, self.pos[0, :]])

        n = pos.shape[0] - 1

        # Length of polygon segments
        lengths = np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1))

        # Arc length parameterization
        x = np.concatenate([[0], np.cumsum(lengths)])
        # Midpoint arc lengths
        xi = 0.5 * (x[:-1] + x[1:])

        # Spline interpolation for midpoints
        cs_x = CubicSpline(x, pos[:, 0], bc_type='periodic')
        cs_y = CubicSpline(x, pos[:, 1], bc_type='periodic')
        posi = np.column_stack([cs_x(xi), cs_y(xi)])

        # Interleave original and midpoints
        new_pos = np.zeros((2 * n, 2))
        new_pos[0::2, :] = pos[:-1, :]
        new_pos[1::2, :] = posi

        self.pos = new_pos
        return self

    def interp1(self, pos: np.ndarray) -> Tuple['Polygon', np.ndarray]:
        """
        Make new polygon through given positions using interpolation.

        MATLAB: interp1.m

        Parameters
        ----------
        pos : ndarray, shape (m, 2)
            Positions that lie on the polygon

        Returns
        -------
        Polygon
            New polygon passing through pos
        ipos : ndarray
            Order of positions in new polygon
        """
        pos = np.atleast_2d(pos)
        ipos = np.arange(pos.shape[0])

        # Find points on polygon
        d, inst = self.dist(pos)
        on_polygon = np.abs(d) < 1e-6
        ipos = ipos[on_polygon]
        inst = inst[on_polygon]

        # Distance to nearest vertex
        d2 = np.sqrt((self.pos[inst, 0] - pos[ipos, 0])**2 +
                     (self.pos[inst, 1] - pos[ipos, 1])**2)

        # Sort by distance
        sort_idx = np.argsort(d2)
        ipos = ipos[sort_idx]
        inst = inst[sort_idx]

        # Sort by vertex index
        sort_idx = np.argsort(inst)
        ipos = ipos[sort_idx]

        self.pos = pos[ipos, :]
        return self, ipos

    def copy(self) -> 'Polygon':
        """Create a copy of the polygon."""
        return Polygon(pos=self.pos.copy(), dir=self.dir, sym=self.sym)

    def __repr__(self) -> str:
        """Command window display."""
        return f"Polygon(n={self.n}, dir={self.dir}, sym={self.sym})"


def polygon_union(polygons: List[Polygon]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple polygons for mesh generation.

    MATLAB: union.m (for polygon arrays)

    Parameters
    ----------
    polygons : list of Polygon
        List of polygons to combine

    Returns
    -------
    upos : ndarray
        Combined positions
    unet : ndarray
        Combined connectivity (0-indexed)
    """
    upos = []
    unet = []
    offset = 0

    for poly in polygons:
        n = poly.pos.shape[0]
        net = np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])
        unet.append(net + offset)
        upos.append(poly.pos)
        offset += n

    return np.vstack(upos), np.vstack(unet)
