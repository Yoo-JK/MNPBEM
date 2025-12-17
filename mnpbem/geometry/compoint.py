"""
ComPoint class - Compound of points in a dielectric environment.

MATLAB: Particles/@compoint
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union, Callable
from scipy.spatial import cKDTree
from matplotlib.path import Path

from .point import Point


def distmin3(p, pos: np.ndarray, cutoff: float = np.inf):
    """
    Minimum distance in 3D between particle faces and positions.

    MATLAB: Misc/distmin3.m

    Parameters
    ----------
    p : Particle or ComParticle
        Particle object
    pos : ndarray, shape (npos, 3)
        Positions
    cutoff : float, optional
        Compute distances only correctly for dmin < cutoff

    Returns
    -------
    dmin : ndarray, shape (npos,)
        Minimum distance between particle faces and positions
        (positive if outside, negative if inside)
    ind : ndarray, shape (npos,)
        Index to nearest neighbour face
    """
    # Use KD-tree for fast nearest neighbor search
    tree = cKDTree(p.pos)
    _, ind = tree.query(pos)

    # Distance between centroids and positions along normal direction
    # Sign: positive if point is above face element, negative otherwise
    diff = pos - p.pos[ind, :]
    dmin = np.sum(diff * p.nvec[ind, :], axis=1)

    if cutoff == 0:
        return dmin, ind

    # Refine distance for points close to surface
    for i in np.where(np.abs(dmin) <= cutoff)[0]:
        pos0 = p.pos[ind[i], :]
        nvec = p.nvec[ind[i], :]
        tvec1 = p.tvec1[ind[i], :]
        tvec2 = p.tvec2[ind[i], :]

        # Project position onto plane perpendicular to nvec
        diff_i = pos[i, :] - pos0
        x = np.dot(diff_i, tvec1)
        y = np.dot(diff_i, tvec2)

        # Get face vertices
        face = p.faces[ind[i], :]
        face = face[~np.isnan(face) & (face >= 0)]
        face = face.astype(int)
        verts = p.verts[face, :]

        # Project vertices
        xv = np.sum((verts - pos0) * tvec1, axis=1)
        yv = np.sum((verts - pos0) * tvec2, axis=1)

        # Distance from point to polygon in 2D
        rmin = _p_poly_dist(x, y, xv, yv)

        # Add to dmin if point is located outside of polygon
        if rmin > 0:
            dmin[i] = np.sign(dmin[i]) * np.sqrt(dmin[i]**2 + rmin**2)

    return dmin, ind


def _p_poly_dist(x: float, y: float, xv: np.ndarray, yv: np.ndarray) -> float:
    """
    Distance from point to polygon.

    MATLAB: distmin3.m - p_poly_dist function

    Parameters
    ----------
    x, y : float
        Point coordinates
    xv, yv : ndarray
        Polygon vertices coordinates

    Returns
    -------
    d : float
        Distance (positive if outside, negative if inside)
    """
    xv = np.asarray(xv).flatten()
    yv = np.asarray(yv).flatten()
    Nv = len(xv)

    # Close polygon if not closed
    if xv[0] != xv[-1] or yv[0] != yv[-1]:
        xv = np.append(xv, xv[0])
        yv = np.append(yv, yv[0])
        Nv = Nv + 1

    # Linear parameters of segments
    A = -np.diff(yv)
    B = np.diff(xv)
    C = yv[1:] * xv[:-1] - xv[1:] * yv[:-1]

    # Find projection of point on each rib
    AB = 1.0 / (A**2 + B**2 + 1e-30)  # Avoid division by zero
    vv = A * x + B * y + C
    xp = x - (A * AB) * vv
    yp = y - (B * AB) * vv

    # Find cases where projected point is inside the segment
    idx_x = ((xp >= np.minimum(xv[:-1], xv[1:])) &
             (xp <= np.maximum(xv[:-1], xv[1:])))
    idx_y = ((yp >= np.minimum(yv[:-1], yv[1:])) &
             (yp <= np.maximum(yv[:-1], yv[1:])))
    idx = idx_x & idx_y

    # Distance from point to vertices
    dv = np.sqrt((xv[:-1] - x)**2 + (yv[:-1] - y)**2)

    if not np.any(idx):
        # All projections are outside polygon ribs
        d = np.min(dv)
    else:
        # Distance from point to projection on ribs
        dp = np.sqrt((xp[idx] - x)**2 + (yp[idx] - y)**2)
        d = min(np.min(dv), np.min(dp))

    # Check if point is inside polygon
    path = Path(np.column_stack([xv, yv]))
    if path.contains_point([x, y]):
        d = -d

    return d


class Compound:
    """
    Base class for compound of points or particles.

    MATLAB: Particles/@compound

    Properties
    ----------
    eps : list
        List of dielectric functions
    p : list
        List of Point or Particle objects
    inout : ndarray
        Index to medium eps (n x 2 for particles)
    mask : ndarray
        Mask for points or particles
    """

    def __init__(self, eps: List, p: List, inout: np.ndarray):
        """
        Initialize Compound.

        MATLAB: compound.m

        Parameters
        ----------
        eps : list
            List of dielectric functions
        p : list
            List of Point or Particle objects
        inout : ndarray
            Index to medium eps
        """
        self.eps = eps
        self.p = p if p else []
        self.inout = np.atleast_2d(inout) if inout is not None and len(inout) > 0 else np.array([])
        self.mask = np.arange(len(self.p)) if self.p else np.array([], dtype=int)
        self._pc = None  # Compound of points/particles (lazy evaluation)

    @property
    def pc(self):
        """Compound of all points/particles."""
        if self._pc is None and len(self.p) > 0:
            masked_p = [self.p[i] for i in self.mask]
            if masked_p:
                self._pc = masked_p[0]
                for pt in masked_p[1:]:
                    self._pc = self._pc.vertcat(pt)
        return self._pc

    @pc.setter
    def pc(self, value):
        self._pc = value

    @property
    def n(self) -> int:
        """Total number of points or faces."""
        return sum(self.p[i].n for i in self.mask)

    @property
    def np(self) -> int:
        """Number of point sets or particles."""
        return len(self.mask)

    @property
    def size(self) -> np.ndarray:
        """Number of positions for each point set."""
        return np.array([self.p[i].n for i in self.mask])

    def index(self, i: int) -> np.ndarray:
        """
        Index to positions of given point set or particle.

        MATLAB: subsref.m - case 'index'
        """
        if i < 0 or i >= self.np:
            raise IndexError(f"Index {i} out of range [0, {self.np})")

        start = sum(self.p[self.mask[j]].n for j in range(i))
        end = start + self.p[self.mask[i]].n
        return np.arange(start, end)

    def ipart(self, ind: np.ndarray) -> np.ndarray:
        """
        Particle number for given position index.

        MATLAB: subsref.m - case 'ipart'
        """
        ind = np.atleast_1d(ind)
        result = np.zeros(len(ind), dtype=int)

        cumsum = 0
        for i, mi in enumerate(self.mask):
            n = self.p[mi].n
            in_range = (ind >= cumsum) & (ind < cumsum + n)
            result[in_range] = i
            cumsum += n

        return result

    def eps1(self, enei: float) -> np.ndarray:
        """Inside dielectric function at wavelength enei."""
        result = np.zeros(self.n, dtype=complex)
        idx = 0
        for i in self.mask:
            n = self.p[i].n
            eps_idx = int(self.inout[i, 0]) - 1  # Convert to 0-indexed
            eps_val, _ = self.eps[eps_idx](enei)
            result[idx:idx+n] = eps_val
            idx += n
        return result

    def eps2(self, enei: float) -> np.ndarray:
        """Outside dielectric function at wavelength enei."""
        result = np.zeros(self.n, dtype=complex)
        idx = 0
        for i in self.mask:
            n = self.p[i].n
            eps_idx = int(self.inout[i, 1]) - 1 if self.inout.shape[1] > 1 else int(self.inout[i, 0]) - 1
            eps_val, _ = self.eps[eps_idx](enei)
            result[idx:idx+n] = eps_val
            idx += n
        return result


class ComPoint(Compound):
    """
    Compound of points in a dielectric environment.

    MATLAB: Particles/@compoint

    Given a set of points, ComPoint puts the points into a dielectric
    environment such that they can be used by the CompGreen classes.

    Parameters
    ----------
    pin : ComParticle
        ComParticle object from initialization
    layer : LayerStructure, optional
        Layer structure for substrates
    """

    def __init__(self, p, pos: np.ndarray, mindist: Optional[np.ndarray] = None,
                 medium: Optional[np.ndarray] = None, layer=None):
        """
        Initialize ComPoint.

        MATLAB: compoint.m, init.m

        Parameters
        ----------
        p : ComParticle
            Composite particle defining the dielectric environment
        pos : ndarray, shape (npos, 3)
            Positions of points
        mindist : ndarray, optional
            Minimum distance of points to given particle
        medium : ndarray, optional
            Mask out points in selected media
        layer : LayerStructure, optional
            Layer structure for substrates
        """
        # Initialize base class with empty values
        super().__init__(p.eps, [], np.array([]))

        self.pin = p
        self.layer = layer
        self._npos = pos.shape[0]
        self._ind = {}  # Index to positions for each medium

        # Initialize
        self._init(p, pos, mindist, medium, layer)

    def _init(self, p, pos1: np.ndarray, mindist: Optional[np.ndarray],
              medium: Optional[np.ndarray], layer):
        """
        Initialize ComPoint.

        MATLAB: init.m
        """
        npos = pos1.shape[0]

        # Default minimum distance
        if mindist is None:
            mindist = np.zeros(p.np)
        else:
            mindist = np.atleast_1d(mindist)
            if len(mindist) != p.np:
                mindist = np.full(p.np, mindist[0])

        # Minimal distance between particle and positions
        r, ind2 = distmin3(p, pos1, np.max(mindist))

        # Conversion table between face indices and particle number
        ind2part = np.zeros(p.n, dtype=int)
        for ip in range(p.np):
            indices = p.index(ip)
            ind2part[indices] = ip

        # Keep only positions sufficiently far from boundaries
        far_enough = np.abs(r) >= mindist[ind2part[ind2]]
        ind1 = np.where(far_enough)[0]
        r = r[ind1]
        ind2_filtered = ind2[ind1]

        # Determine whether point is in- or outside the nearest surface
        inout_arr = np.zeros(len(ind1), dtype=int)

        # Inside (r < 0)
        inside_mask = r < 0
        if np.any(inside_mask):
            part_idx = ind2part[ind2_filtered[inside_mask]]
            inout_arr[inside_mask] = p.inout_faces[part_idx, 0]

        # Outside (r >= 0)
        outside_mask = r >= 0
        if np.any(outside_mask):
            part_idx = ind2part[ind2_filtered[outside_mask]]
            inout_arr[outside_mask] = p.inout_faces[part_idx, 1]

        # Handle layer structure if provided
        if layer is not None:
            self.layer = layer
            # Index to points connected to layer structure
            layer_ind = np.array(layer.ind)
            indl = np.any(inout_arr[:, np.newaxis] == layer_ind[np.newaxis, :], axis=1)

            if np.any(indl):
                indl1 = ind1[indl]
                # Points in layer
                in_layer = layer.mindist(pos1[indl1, 2]) < 1e-10
                # Move points into upper layer
                pos1[indl1[in_layer], 2] += 1e-8
                # Assign substrate index to inout
                inout_arr[indl] = layer.ind[layer.indlayer(pos1[indl1, 2])]

        # Group together points by medium
        iotab = np.unique(inout_arr)

        self.p = []
        self.inout = np.zeros((len(iotab), 1), dtype=int)
        self._ind = {}

        for i, io in enumerate(iotab):
            # Pointer to set of points in given medium
            self._ind[i] = ind1[inout_arr == io]
            # Set of points in given medium
            self.p.append(Point(pos1[self._ind[i], :]))
            # Pointer to dielectric function
            self.inout[i, 0] = io

        # Mask points
        if medium is None:
            self.mask = np.arange(len(self.p))
        else:
            medium = np.atleast_1d(medium)
            self.mask = np.array([i for i, io in enumerate(iotab) if io in medium])

        # Update compound
        self._pc = None

    def select(self, method: str, value: Union[np.ndarray, Callable]) -> 'ComPoint':
        """
        Select points in ComPoint object.

        MATLAB: select.m

        Parameters
        ----------
        method : str
            Selection method ('index', 'carfun', 'polfun', 'sphfun')
        value : ndarray or callable
            Index array or selection function

        Returns
        -------
        ComPoint
            Selected points
        """
        if method != 'index':
            # Pass select input to all point objects
            for i in range(len(self.p)):
                self.p[i], _ = self.p[i].select(method, value)
        else:
            # Index-based selection
            ipt = []  # Group index for each point
            ind = []  # Point index within group

            for i, pt in enumerate(self.p):
                ipt.extend([i] * pt.n)
                ind.extend(range(pt.n))

            ipt = np.array(ipt)
            ind = np.array(ind)

            # Get selected indices
            selected = np.atleast_1d(value)
            ind_sel = ind[selected]
            ipt_sel = ipt[selected]

            # Select points in each group
            for i in range(len(self.p)):
                group_mask = ipt_sel == i
                if np.any(group_mask):
                    self.p[i], _ = self.p[i].select('index', ind_sel[group_mask])

        # Keep only non-empty point objects
        non_empty = [i for i, pt in enumerate(self.p) if pt.n > 0]
        self.p = [self.p[i] for i in non_empty]
        self.inout = self.inout[non_empty, :]
        self.mask = np.arange(len(self.p))

        # Update compound
        self._pc = None

        return self

    def flip(self, directions: Union[int, List[int]] = 1) -> 'ComPoint':
        """
        Flip ComPoint object along given directions.

        MATLAB: flip.m

        Parameters
        ----------
        directions : int or list of int
            Directions along which object is flipped (0=x, 1=y, 2=z)

        Returns
        -------
        ComPoint
            Flipped object
        """
        if isinstance(directions, int):
            directions = [directions]

        for d in directions:
            for pt in self.p:
                pt.pos[:, d] = -pt.pos[:, d]

        self._pc = None
        return self

    def __call__(self, valpt: np.ndarray, valdef: float = np.nan) -> np.ndarray:
        """
        Convert between ComPoint indices and original position indices.

        MATLAB: subsref.m - case '()'

        Parameters
        ----------
        valpt : ndarray
            Value array computed for the ComPoint object
        valdef : float, optional
            Default value for unset elements (NaN on default)

        Returns
        -------
        val : ndarray
            Array with same number of elements as original positions
        """
        siz = valpt.shape
        if len(siz) == 1:
            siz = (siz[0], 1)

        # Allocate output array
        if np.isnan(valdef):
            val = np.full((self._npos,) + siz[1:], np.nan)
        else:
            val = np.full((self._npos,) + siz[1:], valdef)

        # Convert between ComPoint and original position indices
        for i in range(self.np):
            idx = self.index(i)
            orig_idx = self._ind[self.mask[i]]
            val[orig_idx, ...] = valpt[idx, ...]

        return val

    @property
    def pos(self) -> np.ndarray:
        """Positions of all points."""
        if self.pc is not None:
            return self.pc.pos
        return np.array([])

    @property
    def nvec(self) -> np.ndarray:
        """Normal vectors of all points."""
        if self.pc is not None:
            return self.pc.nvec
        return np.array([])

    def __repr__(self) -> str:
        """Command window display."""
        return (f"ComPoint(n={self.n}, np={self.np}, "
                f"media={list(self.inout[self.mask, 0])})")
