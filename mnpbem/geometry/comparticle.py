"""
Compound particle with multiple materials.

Matches MATLAB MNPBEM @comparticle implementation exactly.
"""

import numpy as np
from .particle import Particle


class ComParticle:
    """
    Compound particle with multiple dielectric media.

    Combines multiple particles with different material properties,
    specifying which dielectric function applies inside and outside
    each particle surface.

    MATLAB: @comparticle (inherits from @compound)

    Parameters
    ----------
    eps : list of material objects
        List of dielectric functions (EpsConst, EpsTable, etc.)
    particles : list of Particle
        List of particle objects
    inout : list or ndarray, shape (nparticles, 2)
        For each particle: [inside_eps_index, outside_eps_index]
        Indices refer to the eps list (1-indexed like MATLAB)
    closed_args : tuple, optional
        Arguments passed to closed() method
    **kwargs : dict
        Options (interp='curv'/'flat', etc.)

    Attributes
    ----------
    eps : list
        Dielectric functions
    p : list of Particle
        Particle geometries
    inout : ndarray
        Inside/outside dielectric indices
    closed : list
        Closed surface information for each particle
    pc : Particle
        Concatenated particle (vertcat of all particles)
    nverts : int
        Total number of vertices
    nfaces : int
        Total number of faces
    np : int
        Number of unique material boundaries
    """

    def __init__(self, eps, particles, inout, *closed_args, **kwargs):
        """
        Initialize compound particle.

        MATLAB: obj = comparticle(eps, p, inout, varargin)
        """
        self.eps = eps

        # Process input particles and options (MATLAB: getinput.m)
        particles, closed_args = self._getinput(particles, closed_args, kwargs)

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

        # Mask (default: all particles active)
        self._mask = list(range(len(self.p)))

        # Initialize closed surfaces (MATLAB: init.m)
        self.closed = [None] * len(self.p)
        if len(closed_args) > 0:
            self.set_closed(*closed_args)

        # Compute auxiliary properties and create pc
        self._compute_properties()
        self._norm()

    def _getinput(self, particles, closed_args, kwargs):
        """
        Extract options for particles and get closed arguments.

        MATLAB: getinput.m
        """
        # Apply interp option to all particles if specified
        if 'interp' in kwargs:
            interp = kwargs['interp']
            for p in particles:
                if interp == 'curv':
                    p.curved()
                else:
                    p.flat()

        return particles, closed_args

    def _norm(self):
        """
        Compute auxiliary information for discretized particle surface.

        MATLAB: norm(obj)
        """
        # Create concatenated particle (vertcat all particles)
        if len(self.p) > 0:
            self.pc = self.p[0]
            for p in self.p[1:]:
                self.pc = self.pc + p
        else:
            self.pc = Particle(np.array([]).reshape(0, 3),
                              np.array([]).reshape(0, 4))

    def _compute_properties(self):
        """Compute derived properties."""
        # Total number of vertices and faces
        self.nverts = sum(part.nverts for part in self.p)
        self.nfaces = sum(part.nfaces for part in self.p)

        # Number of unique material boundaries
        unique_pairs = set()
        for pair in self.inout:
            unique_pairs.add(tuple(pair))
        self.np = len(unique_pairs)

        # Create index mapping for boundary elements
        self._create_index()

    def _create_index(self):
        """
        Create index array mapping faces to material boundaries.

        MATLAB: compound.index property
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

    # ==================== Closed surfaces ====================

    def set_closed(self, *args):
        """
        Indicate closed surfaces of particles (for use in compgreen).

        MATLAB: closed(obj, varargin)

        Usage
        -----
        obj.set_closed([i1, i2, ...])
            Closed surface of particles i1, i2, ...
        obj.set_closed({i1, p1, p2, ...})
            Closed surface of particle i1 and particles p1, p2, ...
        """
        for arg in args:
            # Input is index to particle(s) stored in obj
            if not isinstance(arg, (list, tuple)) or not isinstance(arg[0], (list, Particle)):
                # Simple list of indices
                if np.isscalar(arg):
                    indices = [arg]
                else:
                    indices = arg

                for ind in indices:
                    ind_abs = abs(ind)
                    # Set closed property if not previously set (1-indexed!)
                    if self.closed[ind_abs - 1] is None:
                        self.closed[ind_abs - 1] = indices
            # Input is an additional particle
            else:
                idx = arg[0] if np.isscalar(arg[0]) else arg[0][0]
                # Vertcat particles
                particles_to_concat = [self.p[idx - 1]] + list(arg[1:])
                combined = particles_to_concat[0]
                for p in particles_to_concat[1:]:
                    combined = combined + p
                self.closed[idx - 1] = combined

    def closedparticle(self, ind):
        """
        Return particle with closed surface for indexed particle.

        MATLAB: [p, dir, loc] = closedparticle(obj, ind)

        Parameters
        ----------
        ind : int
            Particle index (1-indexed like MATLAB)

        Returns
        -------
        p : Particle or None
            Closed particle (None if not closed)
        dir : int or None
            Outer (dir=1) or inner (dir=-1) surface normal
        loc : ndarray or None
            If closed particle is contained in pc, loc points to the
            elements of the closed particle (None otherwise)
        """
        idx = ind - 1  # Convert to 0-indexed

        if self.closed[idx] is None:
            return None, None, None

        elif isinstance(self.closed[idx], Particle):
            return self.closed[idx], 1, None

        else:
            closed_list = self.closed[idx]
            # Find direction
            dir_val = None
            for c in closed_list:
                if abs(c) == ind:
                    dir_val = np.sign(c) if c != 0 else 1
                    break

            if dir_val is None:
                dir_val = 1

            # Put together closed particle surface
            abs_closed = [abs(c) for c in closed_list]
            sign_closed = [np.sign(c) if c != 0 else 1 for c in closed_list]

            particles = [self.p[i - 1] for i in abs_closed]

            # Flip faces where direction doesn't match
            for i, (p, sign) in enumerate(zip(particles, sign_closed)):
                if sign != dir_val:
                    particles[i] = p.flipfaces()

            # Vertcat all particles
            p_combined = particles[0]
            for p in particles[1:]:
                p_combined = p_combined + p

            # Index to closed particle
            if all(c > 0 for c in closed_list):
                # Find locations in pc
                loc = []
                for i, pos in enumerate(p_combined.pos):
                    # Find matching position in pc
                    for j, pc_pos in enumerate(self.pc.pos):
                        if np.allclose(pos, pc_pos):
                            loc.append(j)
                            break
                loc = np.array(loc) if len(loc) == len(p_combined.pos) else None
            else:
                loc = None

            return p_combined, dir_val, loc

    # ==================== Selection ====================

    def select(self, **kwargs):
        """
        Select faces in comparticle object.

        MATLAB: obj = select(obj, 'PropertyName', PropertyValue)

        Parameters
        ----------
        index : array_like, optional
            Index to selected elements
        carfun : callable, optional
            Function f(x, y, z) for selected elements
        polfun : callable, optional
            Function f(phi, r, z) for selected elements
        sphfun : callable, optional
            Function f(phi, theta, r) for selected elements

        Returns
        -------
        obj : ComParticle
            Selected comparticle
        """
        if 'index' not in kwargs:
            # Pass select input to all particle objects
            new_particles = []
            for p in self.p:
                p_selected, _ = p.select(**kwargs)
                new_particles.append(p_selected)
            self.p = new_particles
        else:
            # Index to grouped particles
            index = kwargs['index']

            # Create particle index for each face
            ipt = []
            for i, p in enumerate(self.p):
                ipt.extend([i] * p.nfaces)
            ipt = np.array(ipt)

            # Point index (global face index)
            ind_global = []
            for p in self.p:
                ind_global.extend(range(p.nfaces))
            ind_global = np.array(ind_global)

            # Get selected indices
            ind_selected = ind_global[index]
            ipt_selected = ipt[index]

            # Loop over all particles
            new_particles = []
            for i in range(len(self.p)):
                # Get local indices for this particle
                mask = (ipt_selected == i)
                if mask.any():
                    local_ind = ind_selected[mask]
                    p_selected, _ = self.p[i].select(index=local_ind)
                    new_particles.append(p_selected)

            self.p = new_particles

        # Keep only non-empty particles
        non_empty = [i for i, p in enumerate(self.p) if p.nfaces > 0]
        self.p = [self.p[i] for i in non_empty]
        self.inout = self.inout[non_empty, :]

        # Reset closed arguments
        self.closed = [None] * len(self.p)

        # Update mask
        self._mask = list(range(len(self.p)))

        # Update compound particle
        self._norm()

        return self

    # ==================== Wrapper methods ====================

    def clean(self, *args, **kwargs):
        """
        Apply particle.clean() to all particles.

        MATLAB: clean(obj, varargin)
        """
        self.p = [p.clean(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def flip(self, *args, **kwargs):
        """
        Apply particle.flip() to all particles.

        MATLAB: flip(obj, varargin)
        """
        self.p = [p.flip(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def flipfaces(self, *args, **kwargs):
        """
        Apply particle.flipfaces() to all particles.

        MATLAB: flipfaces(obj, varargin)
        """
        self.p = [p.flipfaces(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def rot(self, *args, **kwargs):
        """
        Apply particle.rot() to all particles.

        MATLAB: rot(obj, varargin)
        """
        self.p = [p.rot(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def scale(self, *args, **kwargs):
        """
        Apply particle.scale() to all particles.

        MATLAB: scale(obj, varargin)
        """
        self.p = [p.scale(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    def shift(self, *args, **kwargs):
        """
        Apply particle.shift() to all particles.

        MATLAB: shift(obj, varargin)
        """
        self.p = [p.shift(*args, **kwargs) for p in self.p]
        self._norm()
        return self

    # ==================== Delegation to pc ====================

    def deriv(self, v):
        """
        Tangential derivative of function defined on surface.

        MATLAB: [v1, v2, t1, t2] = deriv(obj, v)
        """
        return self.pc.deriv(v)

    def interp_values(self, v, method='area'):
        """
        Interpolate values from faces to vertices or vice versa.

        MATLAB: [vi, mat] = interp(obj, v, key)
        """
        return self.pc.interp_values(v, method)

    def curvature(self):
        """
        Curvature of particle.

        MATLAB: curv = curvature(obj, varargin)
        """
        return self.pc.curvature()

    def quad_integration(self, ind=None):
        """
        Integration over boundary elements.

        MATLAB: [pos, w, iface] = quad(obj, ind)
        """
        return self.pc.quad_integration(ind)

    def quadpol(self, ind=None):
        """
        Integration over boundary elements using polar coordinates.

        MATLAB: [pos, weight, row] = quadpol(obj, ind)
        """
        return self.pc.quadpol(ind)

    def vertices(self, ind, close=False):
        """
        Vertices of indexed face.

        MATLAB: v = vertices(obj, ind, 'close')

        Parameters
        ----------
        ind : int
            Global face index
        close : bool
            If True, close the face indices

        Returns
        -------
        v : ndarray
            Vertices of the face
        """
        # Find which particle this face belongs to
        ip, local_ind = self._ipart(ind)
        return self.p[ip].vertices(local_ind, close)

    def _ipart(self, ind):
        """
        Return particle and face index for global face index.

        MATLAB: [ip, ind] = ipart(obj, ind)

        Parameters
        ----------
        ind : int
            Global face index

        Returns
        -------
        ip : int
            Particle index (0-indexed)
        local_ind : int
            Local face index within that particle
        """
        offset = 0
        for i, p in enumerate(self.p):
            if ind < offset + p.nfaces:
                return i, ind - offset
            offset += p.nfaces
        raise IndexError(f"Face index {ind} out of range")

    def plot(self, val=None, **kwargs):
        """
        Plot discretized particle surface.

        MATLAB: plot(obj, val, 'PropertyName', PropertyValue, ...)
        """
        return self.pc.plot(val, **kwargs)

    def plot2(self, val=None, **kwargs):
        """
        Advanced plot of discretized particle surface.

        MATLAB: plot2(obj, val, 'PropertyName', PropertyValue, ...)
        """
        return self.pc.plot2(val, **kwargs)

    # ==================== Dielectric functions ====================

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
            eps_vals[offset:offset + part.nfaces] = complex(np.asarray(eps_val).flat[0])

            offset += part.nfaces

        return eps_vals

    # ==================== Properties ====================

    @property
    def pos(self):
        """Centroid positions of all faces."""
        return self.pc.pos if hasattr(self, 'pc') else np.vstack([part.pos for part in self.p])

    @property
    def nvec(self):
        """Normal vectors of all faces."""
        return self.pc.nvec if hasattr(self, 'pc') else np.vstack([part.nvec for part in self.p])

    @property
    def area(self):
        """Areas of all faces."""
        return self.pc.area if hasattr(self, 'pc') else np.concatenate([part.area for part in self.p])

    @property
    def verts(self):
        """All vertices (concatenated)."""
        return self.pc.verts if hasattr(self, 'pc') else np.vstack([part.verts for part in self.p])

    @property
    def inout_faces(self):
        """
        Get inside/outside material indices for each face.

        Returns array of shape (nfaces, 2) where:
        - Column 0: inside material index (1-indexed like MATLAB)
        - Column 1: outside material index (1-indexed like MATLAB)
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

    @property
    def n(self):
        """Number of positions/faces (alias for nfaces, MATLAB compatibility)."""
        return self.nfaces

    @property
    def mask(self):
        """
        Mask array indicating which particles are active.

        MATLAB: obj.mask
        """
        mask_arr = np.zeros(len(self.p), dtype=bool)
        mask_arr[self._mask] = True
        return mask_arr

    def set_mask(self, ind):
        """
        Mask out particles indicated by ind.

        MATLAB: obj = mask(obj, ind)

        Parameters
        ----------
        ind : array_like
            Indices of particles to keep (1-indexed like MATLAB)
        """
        if np.isscalar(ind):
            ind = [ind]
        self._mask = [i - 1 for i in ind]  # Convert to 0-indexed
        self._norm()
        return self

    def index_func(self, particle_indices):
        """
        Get face indices for given particle indices.

        MATLAB: p.index(i) returns indices of faces belonging to particle i.

        Parameters
        ----------
        particle_indices : int or array
            Particle indices (1-indexed like MATLAB)

        Returns
        -------
        face_indices : ndarray
            Face indices corresponding to the particles
        """
        if np.isscalar(particle_indices):
            particle_indices = [particle_indices]

        face_indices = []
        offset = 0

        for i, part in enumerate(self.p):
            if (i + 1) in particle_indices:  # Convert to 1-indexed
                face_indices.extend(range(offset, offset + part.nfaces))
            offset += part.nfaces

        return np.array(face_indices, dtype=int)

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
