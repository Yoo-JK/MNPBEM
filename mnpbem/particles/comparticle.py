"""
Compound particle class with dielectric functions.
"""

import numpy as np
from typing import List, Optional, Union, Any

from .particle import Particle
from .compound import Compound
from ..misc.options import BEMOptions


class ComParticle(Compound):
    """
    Compound of particles in a dielectric environment.

    This is the main class for defining particle geometries with
    associated dielectric functions.

    Parameters
    ----------
    eps : list
        List of dielectric function objects (e.g., EpsConst, EpsDrude).
    p : list
        List of Particle objects.
    inout : ndarray
        Index array of shape (n_particles, 2) specifying
        [inside_medium_idx, outside_medium_idx] for each particle.
        Indices are 1-based (MATLAB convention).
    closed : int or list, optional
        Indices of particles that form closed surfaces.
    **kwargs : dict
        Additional options.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, ComParticle
    >>> from mnpbem.particles.shapes import trisphere
    >>>
    >>> # Define dielectric functions
    >>> eps_vacuum = EpsConst(1.0)
    >>> eps_gold = EpsTable('gold.dat')
    >>> epstab = [eps_vacuum, eps_gold]
    >>>
    >>> # Create gold sphere in vacuum
    >>> sphere = trisphere(144, 10)  # 144 points, 10 nm diameter
    >>> p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)
    """

    def __init__(
        self,
        eps: List[Any],
        p: List[Particle],
        inout: np.ndarray,
        closed: Optional[Union[int, List[int]]] = None,
        **kwargs
    ):
        """
        Initialize compound particle.

        Parameters
        ----------
        eps : list
            List of dielectric function objects.
        p : list
            List of Particle objects.
        inout : ndarray
            Medium indices for each particle.
        closed : int or list, optional
            Indices of closed surfaces (1-based).
        **kwargs : dict
            Additional options.
        """
        # Convert single particle to list
        if isinstance(p, Particle):
            p = [p]

        # Initialize base class
        super().__init__(eps, p, inout)

        # Handle closed surfaces
        if closed is None:
            self.closed = np.zeros(len(p), dtype=bool)
        elif isinstance(closed, (int, np.integer)):
            self.closed = np.zeros(len(p), dtype=bool)
            if closed > 0:
                self.closed[closed - 1] = True  # Convert to 0-based
        else:
            self.closed = np.zeros(len(p), dtype=bool)
            for idx in closed:
                if idx > 0:
                    self.closed[idx - 1] = True

        # Store options
        self.options = kwargs

    @property
    def pos(self) -> np.ndarray:
        """Face centroids of combined particle."""
        return self.pc.pos

    @property
    def nvec(self) -> np.ndarray:
        """Normal vectors of combined particle."""
        return self.pc.nvec

    @property
    def area(self) -> np.ndarray:
        """Face areas of combined particle."""
        return self.pc.area

    @property
    def n_faces(self) -> int:
        """Total number of faces."""
        return self.pc.n_faces

    @property
    def verts(self) -> np.ndarray:
        """Vertices of combined particle."""
        return self.pc.verts

    @property
    def faces(self) -> np.ndarray:
        """Faces of combined particle."""
        return self.pc.faces

    def closedparticle(self, particle_idx: int) -> bool:
        """
        Check if particle forms a closed surface.

        Parameters
        ----------
        particle_idx : int
            Index of particle (0-based).

        Returns
        -------
        bool
            True if particle is closed.
        """
        return self.closed[particle_idx]

    def lambda_factor(self, enei: float) -> np.ndarray:
        """
        Compute Lambda factor for BEM equations.

        Lambda = (eps1 + eps2) / (eps1 - eps2) / (2 * pi)

        Parameters
        ----------
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            Lambda factor for each face.
        """
        n_faces = self.n_faces
        Lambda = np.zeros(n_faces)

        face_idx = 0
        for i, particle in enumerate(self.p):
            eps_in, eps_out = self.dielectric_inout(enei, i)
            n = particle.n_faces

            # Lambda = (eps1 + eps2) / (eps1 - eps2) / (2 * pi)
            lam = (eps_in + eps_out) / (eps_in - eps_out) / (2 * np.pi)
            Lambda[face_idx:face_idx + n] = lam

            face_idx += n

        return Lambda

    def delta_eps(self, enei: float) -> np.ndarray:
        """
        Compute dielectric function difference for each face.

        Parameters
        ----------
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            eps_in - eps_out for each face.
        """
        n_faces = self.n_faces
        deps = np.zeros(n_faces, dtype=complex)

        face_idx = 0
        for i, particle in enumerate(self.p):
            eps_in, eps_out = self.dielectric_inout(enei, i)
            n = particle.n_faces
            deps[face_idx:face_idx + n] = eps_in - eps_out
            face_idx += n

        return deps

    def shift(self, displacement: np.ndarray) -> 'ComParticle':
        """
        Shift all particles by given displacement.

        Parameters
        ----------
        displacement : array_like
            Displacement vector [dx, dy, dz].

        Returns
        -------
        ComParticle
            Shifted compound particle.
        """
        new_p = [p.shift(displacement) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def scale(self, factor: Union[float, np.ndarray]) -> 'ComParticle':
        """
        Scale all particles.

        Parameters
        ----------
        factor : float or array_like
            Scale factor.

        Returns
        -------
        ComParticle
            Scaled compound particle.
        """
        new_p = [p.scale(factor) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def rot(self, angle: float, axis: int = 2) -> 'ComParticle':
        """
        Rotate all particles.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : int
            Rotation axis.

        Returns
        -------
        ComParticle
            Rotated compound particle.
        """
        new_p = [p.rot(angle, axis) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def flip(self, axis: int) -> 'ComParticle':
        """
        Flip all particles along axis.

        Parameters
        ----------
        axis : int
            Axis to flip.

        Returns
        -------
        ComParticle
            Flipped compound particle.
        """
        new_p = [p.flip(axis) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def clean(self, tol: float = 1e-10) -> 'ComParticle':
        """
        Clean all particles (remove duplicate vertices).

        Parameters
        ----------
        tol : float
            Tolerance for duplicates.

        Returns
        -------
        ComParticle
            Cleaned compound particle.
        """
        new_p = [p.clean(tol) for p in self.p]
        return ComParticle(self.eps, new_p, self.inout, self.closed, **self.options)

    def curvature(self) -> np.ndarray:
        """
        Compute curvature for all faces.

        Returns
        -------
        ndarray
            Curvature at each face.
        """
        return self.pc.curvature()

    def __repr__(self) -> str:
        return (f"ComParticle(n_eps={len(self.eps)}, "
                f"n_particles={len(self.p)}, "
                f"n_faces={self.n_faces})")
