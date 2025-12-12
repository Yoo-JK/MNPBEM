"""
Compound particle with multiple materials.
"""

import numpy as np
from .particle import Particle


class ComParticle:
    """
    Compound particle with multiple dielectric media.

    Combines multiple particles with different material properties,
    specifying which dielectric function applies inside and outside
    each particle surface.

    Parameters
    ----------
    eps : list of material objects
        List of dielectric functions (EpsConst, EpsTable, etc.)
    particles : list of Particle
        List of particle objects
    inout : list or ndarray, shape (nparticles, 2)
        For each particle: [inside_eps_index, outside_eps_index]
        Indices refer to the eps list (1-indexed like MATLAB)
    closed : list of int, optional
        Indices of particles that form closed surfaces

    Attributes
    ----------
    eps : list
        Dielectric functions
    p : list of Particle
        Particle geometries
    inout : ndarray
        Inside/outside dielectric indices
    nverts : int
        Total number of vertices
    nfaces : int
        Total number of faces
    np : int
        Number of unique material boundaries

    Examples
    --------
    >>> from mnpbem.materials import EpsConst, EpsTable
    >>> from mnpbem.geometry import trisphere
    >>>
    >>> # Materials: vacuum and gold
    >>> eps_vacuum = EpsConst(1.0)
    >>> eps_gold = EpsTable('gold.dat')
    >>> epstab = [eps_vacuum, eps_gold]
    >>>
    >>> # Create sphere
    >>> sphere = trisphere(144, 10.0)
    >>>
    >>> # Compound particle: gold sphere in vacuum
    >>> # inout = [2, 1] means inside=gold (index 2), outside=vacuum (index 1)
    >>> p = ComParticle(epstab, [sphere], [[2, 1]])
    """

    def __init__(self, eps, particles, inout, *closed):
        """
        Initialize compound particle.

        Parameters
        ----------
        eps : list
            Dielectric functions
        particles : list
            Particle objects or list of particles
        inout : array_like
            Inside/outside material indices
        closed : int or list of int, optional
            Closed surface indices
        """
        self.eps = eps

        # Handle particle list (may be nested list from MATLAB style)
        if isinstance(particles, list):
            if len(particles) > 0 and isinstance(particles[0], list):
                # Flatten nested list: [[p1]] -> [p1]
                self.p = [item for sublist in particles for item in sublist]
            else:
                self.p = particles
        else:
            self.p = [particles]

        # Convert inout to numpy array (MATLAB uses 1-indexing)
        self.inout = np.atleast_2d(inout)

        # Closed surfaces
        if len(closed) > 0:
            self.closed = list(closed)
        else:
            # By default, assume all particles are closed
            self.closed = list(range(len(self.p)))

        # Compute auxiliary properties
        self._compute_properties()

    def _compute_properties(self):
        """Compute derived properties."""
        # Total number of vertices and faces
        self.nverts = sum(part.nverts for part in self.p)
        self.nfaces = sum(part.nfaces for part in self.p)

        # Number of unique material boundaries
        # This is the number of unique (eps_in, eps_out) pairs
        unique_pairs = set()
        for pair in self.inout:
            unique_pairs.add(tuple(pair))
        self.np = len(unique_pairs)

        # Create index mapping for boundary elements
        self._create_index()

    def _create_index(self):
        """
        Create index array mapping faces to material boundaries.

        This is similar to MATLAB's compound.index property.
        """
        self.index = np.zeros(self.nfaces, dtype=int)

        # Create mapping from (eps_in, eps_out) to unique index
        unique_pairs = []
        pair_to_idx = {}

        for pair in self.inout:
            pair_tuple = tuple(pair)
            if pair_tuple not in pair_to_idx:
                pair_to_idx[pair_tuple] = len(unique_pairs)
                unique_pairs.append(pair_tuple)

        # Assign indices to each face
        offset = 0
        for i, part in enumerate(self.p):
            pair = tuple(self.inout[i])
            idx = pair_to_idx[pair]
            self.index[offset:offset + part.nfaces] = idx
            offset += part.nfaces

    def eps1(self, enei):
        """
        Get inside dielectric constants at given wavelength.

        Parameters
        ----------
        enei : float or array
            Wavelength in nm

        Returns
        -------
        eps : ndarray
            Inside dielectric constants for each face
        """
        eps_vals = np.zeros(self.nfaces, dtype=complex)
        offset = 0

        for i, part in enumerate(self.p):
            eps_idx = int(self.inout[i, 0]) - 1  # Convert to 0-indexed
            eps_mat = self.eps[eps_idx]
            eps_val, _ = eps_mat(enei)

            # Broadcast to all faces of this particle
            # Handle both scalar and array cases properly
            eps_vals[offset:offset + part.nfaces] = complex(np.asarray(eps_val).flat[0])

            offset += part.nfaces

        return eps_vals

    def eps2(self, enei):
        """
        Get outside dielectric constants at given wavelength.

        Parameters
        ----------
        enei : float or array
            Wavelength in nm

        Returns
        -------
        eps : ndarray
            Outside dielectric constants for each face
        """
        eps_vals = np.zeros(self.nfaces, dtype=complex)
        offset = 0

        for i, part in enumerate(self.p):
            eps_idx = int(self.inout[i, 1]) - 1  # Convert to 0-indexed
            eps_mat = self.eps[eps_idx]
            eps_val, _ = eps_mat(enei)

            # Broadcast to all faces of this particle
            # Handle both scalar and array cases properly
            eps_vals[offset:offset + part.nfaces] = complex(np.asarray(eps_val).flat[0])

            offset += part.nfaces

        return eps_vals

    @property
    def pos(self):
        """Centroid positions of all faces."""
        return np.vstack([part.pos for part in self.p])

    @property
    def nvec(self):
        """Normal vectors of all faces."""
        return np.vstack([part.nvec for part in self.p])

    @property
    def area(self):
        """Areas of all faces."""
        return np.concatenate([part.area for part in self.p])

    @property
    def verts(self):
        """All vertices (concatenated)."""
        return np.vstack([part.verts for part in self.p])

    @property
    def inout_faces(self):
        """
        Get inside/outside material indices for each face.

        Returns array of shape (nfaces, 2) where:
        - Column 0: inside material index (1-indexed like MATLAB)
        - Column 1: outside material index (1-indexed like MATLAB)

        This matches MATLAB's p.inout property expanded to all faces.
        """
        inout_arr = np.zeros((self.nfaces, 2), dtype=int)
        offset = 0

        for i, part in enumerate(self.p):
            inout_arr[offset:offset + part.nfaces, 0] = self.inout[i, 0]
            inout_arr[offset:offset + part.nfaces, 1] = self.inout[i, 1]
            offset += part.nfaces

        return inout_arr

    def get_face_indices(self, medium, side='outside'):
        """
        Get indices of faces where the specified medium is on the given side.

        Parameters
        ----------
        medium : int
            Material index (1-indexed like MATLAB)
        side : str
            'inside' (column 0) or 'outside' (column 1)

        Returns
        -------
        indices : ndarray
            Array of face indices where the medium is on the specified side
        """
        col = 0 if side == 'inside' else 1
        return np.where(self.inout_faces[:, col] == medium)[0]

    def __repr__(self):
        return (
            f"ComParticle(nparticles={len(self.p)}, "
            f"nverts={self.nverts}, nfaces={self.nfaces})"
        )

    def __str__(self):
        parts_info = "\n".join(
            f"  Particle {i+1}: {p.nverts} verts, {p.nfaces} faces"
            for i, p in enumerate(self.p)
        )
        return (
            f"Compound Particle:\n"
            f"  Materials: {len(self.eps)}\n"
            f"  Particles: {len(self.p)}\n"
            f"{parts_info}\n"
            f"  Total vertices: {self.nverts}\n"
            f"  Total faces: {self.nfaces}"
        )
