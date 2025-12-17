"""
ComParticleMirror class - Compound of particles with mirror symmetry.

MATLAB: Particles/@comparticlemirror
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict
from .comparticle import ComParticle


class ComParticleMirror(ComParticle):
    """
    Compound of particles with mirror symmetry in a dielectric environment.

    MATLAB: @comparticlemirror (inherits from @bembase and @comparticle)

    Parameters
    ----------
    eps : list
        Cell array of dielectric functions
    p : list of Particle
        Cell array of particles
    inout : ndarray
        Index to medium EPS
    closed_args : tuple, optional
        Arguments passed to closed()
    sym : str
        Symmetry key: 'x', 'y', or 'xy'
    **kwargs : dict
        Additional options

    Properties
    ----------
    sym : str
        Symmetry key
    symtable : ndarray
        Table with symmetry values
    pfull : ComParticle
        Full comparticle produced with mirror symmetry

    Examples
    --------
    >>> from mnpbem.geometry import trisphere, ComParticleMirror
    >>> from mnpbem.material import EpsConst
    >>> eps_air = EpsConst(1.0)
    >>> eps_au = EpsConst(-11.4 + 1.18j)
    >>> p = trisphere(10, 1)
    >>> p = p.shift([0.5, 0, 0])  # Shift to use x-symmetry
    >>> comp = ComParticleMirror([eps_air, eps_au], [p], [[2, 1]], sym='x')
    """

    name = 'bemparticle'
    needs = ['sym']

    def __init__(self, eps, particles, inout, *closed_args, sym: str = None, **kwargs):
        """
        Initialize comparticle with mirror symmetry.

        MATLAB: comparticlemirror.m, init.m

        Parameters
        ----------
        eps : list
            Cell array of dielectric functions
        particles : list of Particle
            Cell array of particles
        inout : ndarray
            Index to medium EPS
        closed_args : tuple, optional
            Arguments passed to closed()
        sym : str
            Symmetry key: 'x', 'y', or 'xy'
        **kwargs : dict
            Additional options
        """
        # Get symmetry from kwargs if not provided
        if sym is None:
            sym = kwargs.pop('sym', None)
        if sym is None:
            raise ValueError("Symmetry key 'sym' must be provided ('x', 'y', or 'xy')")

        self.sym = sym

        # Initialize symmetry table
        # MATLAB: init.m lines 38-46
        if self.sym in ['x', 'y']:
            self.symtable = np.array([[1, 1], [1, -1]])  # '+' ; '-'
        elif self.sym == 'xy':
            self.symtable = np.array([
                [1, 1, 1, 1],    # '++'
                [1, 1, -1, -1],  # '+-'
                [1, -1, 1, -1],  # '-+'
                [1, -1, -1, 1]   # '--'
            ])
        else:
            raise ValueError(f"Unknown symmetry key: {self.sym}")

        # Handle particle list
        if not isinstance(particles, list):
            particles = [particles]

        # Convert inout to numpy array
        inout = np.atleast_2d(inout)

        # Make full particle using mirror symmetry
        # MATLAB: init.m lines 49-61
        p_full = list(particles)
        inout_full = inout.copy()

        # Mirror operations
        mirror = [['x', 'xy'], ['y', 'xy']]

        # Add equivalent particles by applying mirror symmetry operations
        for k in range(2):
            n_particles = len(p_full)
            for i in range(n_particles):
                if self.sym in mirror[k]:
                    # Flip particle along axis k (0=x, 1=y)
                    p_flipped = p_full[i].flip(k)
                    p_full.append(p_flipped)
                    inout_full = np.vstack([inout_full, inout_full[i, :]])

        # Initialize base ComParticle without closed arguments
        super().__init__(eps, particles, inout, **kwargs)

        # Initialize full particle
        self.pfull = ComParticle(eps, p_full, inout_full, **kwargs)

        # Handle closed arguments
        if len(closed_args) > 0:
            self._closed_mirror(*closed_args)

    def full(self) -> ComParticle:
        """
        Full particle produced with mirror symmetry.

        MATLAB: full.m

        Returns
        -------
        p : ComParticle
            Comparticle object of full particle
        """
        return self.pfull

    def closedparticle(self, ind: int) -> Tuple:
        """
        Return particle with closed surface for particle ind.

        MATLAB: closedparticle.m

        Parameters
        ----------
        ind : int
            Index to given particle (1-indexed like MATLAB)

        Returns
        -------
        p : Particle or None
            Closed particle (None if not closed)
        dir : int or None
            Outer (dir=1) or inner (dir=-1) surface normal
        loc : ndarray or None
            Location indices (None for mirror particles)
        """
        return self.pfull.closedparticle(ind)

    def symindex(self, tab: np.ndarray) -> int:
        """
        Index of symmetry values within symmetry table.

        MATLAB: symindex within comparticlemirror.m

        Parameters
        ----------
        tab : ndarray
            Two or four symmetry values

        Returns
        -------
        ind : int
            Index to symmetry table (1-indexed like MATLAB)
        """
        tab = np.atleast_1d(tab)
        for i in range(self.symtable.shape[0]):
            if np.all(self.symtable[i, :] == tab):
                return i + 1  # 1-indexed
        return -1

    def symvalue(self, key: Union[str, List[str]]) -> np.ndarray:
        """
        Symmetry values for given key.

        MATLAB: symvalue.m

        Parameters
        ----------
        key : str or list of str
            Symmetry key: '+', '-' for sym={'x','y'}, and
            '++', '+-', '-+', '--' for sym='xy'

        Returns
        -------
        val : ndarray
            Value array
        """
        if isinstance(key, list):
            vals = []
            for k in key:
                vals.append(self._symvalue_single(k))
            return np.vstack(vals)
        else:
            return self._symvalue_single(key)

    def _symvalue_single(self, key: str) -> np.ndarray:
        """
        Get symmetry value for a single key.

        MATLAB: symvalue.m
        """
        values = {
            '+': np.array([1, 1]),
            '-': np.array([1, -1]),
            '++': np.array([1, 1, 1, 1]),
            '+-': np.array([1, 1, -1, -1]),
            '-+': np.array([1, -1, 1, -1]),
            '--': np.array([1, -1, -1, 1])
        }
        if key not in values:
            raise ValueError(f"Unknown symmetry key: {key}")
        return values[key]

    def _closed_mirror(self, *args):
        """
        Indicate closed surfaces of particles for Green function evaluation.

        MATLAB: closed.m

        Parameters
        ----------
        *args : various
            Either indices or cell arrays with indices and additional particles
        """
        # Index to equivalent particle surfaces
        n_equiv = self.symtable.shape[1]
        n_orig = len(self.p)
        ind = np.arange(len(self.pfull.p)).reshape(n_orig, n_equiv)

        for arg in args:
            if not isinstance(arg, (list, tuple)) or \
               not isinstance(arg[0], (list, tuple)):
                # Input is index to particle(s) stored in pfull
                indices = np.atleast_1d(arg)

                # Build table of equivalent particles
                tab = []
                for i in indices:
                    row = ind[abs(i) - 1, :]  # Convert to 0-indexed
                    if i < 0:
                        row = -row
                    tab.extend(row)
                tab = np.array(tab)

                # Set closed for each equivalent particle
                for j in tab:
                    self.pfull.closed[abs(j)] = tab.tolist()

            else:
                # Input is an additional particle
                # Tab of equivalent particles
                i = arg[0] if np.isscalar(arg[0]) else arg[0][0]
                tab = ind[i - 1, :]  # Convert to 0-indexed

                # Closed particle (vertcat of pfull particles and additional ones)
                particles_list = [self.pfull.p[t] for t in tab]
                for additional in arg[1:]:
                    particles_list.append(additional)

                # Combine particles
                p_combined = particles_list[0]
                for p in particles_list[1:]:
                    p_combined = p_combined + p

                # Add closed particle to equivalent particles
                for j in range(len(tab)):
                    self.pfull.closed[j] = p_combined

    def set_mask(self, ind: Union[int, List[int]]):
        """
        Mask out particles indicated by ind.

        MATLAB: mask.m

        Parameters
        ----------
        ind : int or list of int
            Indices of particles to keep (1-indexed like MATLAB)
        """
        # Call parent mask
        super().set_mask(ind)

        # Index to equivalent particles
        n_equiv = self.symtable.shape[1]
        n_orig = len(self.p)
        ip = np.arange(len(self.pfull.p)).reshape(n_orig, n_equiv)

        # Convert ind to 0-indexed array
        ind = np.atleast_1d(ind) - 1

        # Mask full particle
        mask_indices = ip[ind, :].flatten() + 1  # Convert back to 1-indexed
        self.pfull.set_mask(mask_indices)

        return self

    # ==================== Hidden methods (raise NotImplementedError) ====================

    def clean(self, *args, **kwargs):
        """Not allowed for mirror particles."""
        raise NotImplementedError("clean() not allowed for ComParticleMirror")

    def flip(self, *args, **kwargs):
        """Not allowed for mirror particles."""
        raise NotImplementedError("flip() not allowed for ComParticleMirror")

    def flipfaces(self, *args, **kwargs):
        """Not allowed for mirror particles."""
        raise NotImplementedError("flipfaces() not allowed for ComParticleMirror")

    def norm(self, *args, **kwargs):
        """Not allowed for mirror particles."""
        raise NotImplementedError("norm() not allowed for ComParticleMirror")

    def rot(self, *args, **kwargs):
        """Not allowed for mirror particles."""
        raise NotImplementedError("rot() not allowed for ComParticleMirror")

    def scale(self, *args, **kwargs):
        """Not allowed for mirror particles."""
        raise NotImplementedError("scale() not allowed for ComParticleMirror")

    def select(self, *args, **kwargs):
        """Not allowed for mirror particles."""
        raise NotImplementedError("select() not allowed for ComParticleMirror")

    def shift(self, *args, **kwargs):
        """Not allowed for mirror particles."""
        raise NotImplementedError("shift() not allowed for ComParticleMirror")

    def __repr__(self) -> str:
        """Command window display."""
        return (f"ComParticleMirror(nparticles={len(self.p)}, sym='{self.sym}', "
                f"nverts={self.nverts}, nfaces={self.nfaces})")
