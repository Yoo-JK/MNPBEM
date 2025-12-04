"""
Compound class for collections of particles or points.
"""

import numpy as np
from typing import List, Optional, Union, Any

from .particle import Particle
from .point import Point


class Compound:
    """
    Compound of points or particles within a dielectric environment.

    Base class for ComParticle and ComPoint.

    Parameters
    ----------
    eps : list
        Cell array of dielectric functions.
    p : list
        List of Particle or Point objects.
    inout : ndarray
        Index to medium eps. For particles, n x 2 array with
        [inside_medium, outside_medium] indices.

    Attributes
    ----------
    eps : list
        Dielectric functions.
    p : list
        Particles or points.
    inout : ndarray
        Medium indices.
    mask : ndarray
        Mask for active particles/points.
    """

    def __init__(
        self,
        eps: List[Any],
        p: List[Union[Particle, Point]],
        inout: np.ndarray
    ):
        """
        Initialize compound object.

        Parameters
        ----------
        eps : list
            List of dielectric function objects.
        p : list
            List of Particle or Point objects.
        inout : ndarray
            Index to medium eps.
        """
        self.eps = eps
        self.p = p
        self.inout = np.asarray(inout)

        # Ensure 2D inout array
        if self.inout.ndim == 1:
            self.inout = self.inout.reshape(-1, 1)

        # Mask and combined particle
        self.mask = np.arange(len(p))
        self._pc = self._combine_particles()

    def _combine_particles(self) -> Union[Particle, Point]:
        """Combine all particles/points into one object."""
        if len(self.p) == 0:
            return Particle()

        result = self.p[0]
        for particle in self.p[1:]:
            result = result + particle
        return result

    @property
    def pc(self) -> Union[Particle, Point]:
        """Combined particle/point collection."""
        return self._pc

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return len(self.p)

    def dielectric(self, enei: float, medium: int = 0) -> complex:
        """
        Get dielectric function value at given wavelength.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        medium : int
            Index of medium (0-based).

        Returns
        -------
        complex
            Dielectric function value.
        """
        eps_func = self.eps[medium]
        eps_val, _ = eps_func(enei)
        return eps_val

    def dielectric_inout(self, enei: float, particle_idx: int = 0) -> tuple:
        """
        Get inside and outside dielectric functions for a particle.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        particle_idx : int
            Index of particle.

        Returns
        -------
        tuple
            (eps_in, eps_out) dielectric values.
        """
        in_idx = int(self.inout[particle_idx, 0]) - 1  # Convert to 0-based
        out_idx = int(self.inout[particle_idx, 1]) - 1

        eps_in = self.dielectric(enei, in_idx)
        eps_out = self.dielectric(enei, out_idx)

        return eps_in, eps_out

    def index(self, particle_idx: int) -> np.ndarray:
        """
        Get face indices for a specific particle.

        Parameters
        ----------
        particle_idx : int
            Index of particle.

        Returns
        -------
        ndarray
            Face indices in the combined particle.
        """
        start = 0
        for i, particle in enumerate(self.p):
            if i == particle_idx:
                return np.arange(start, start + particle.n_faces)
            start += particle.n_faces
        raise IndexError(f"Particle index {particle_idx} out of range")

    def ipart(self, face_idx: int) -> int:
        """
        Get particle index for a given face.

        Parameters
        ----------
        face_idx : int
            Index of face in combined particle.

        Returns
        -------
        int
            Index of particle containing this face.
        """
        count = 0
        for i, particle in enumerate(self.p):
            count += particle.n_faces
            if face_idx < count:
                return i
        raise IndexError(f"Face index {face_idx} out of range")

    def expand(self, val: np.ndarray) -> np.ndarray:
        """
        Expand values from masked particles to full array.

        Parameters
        ----------
        val : ndarray
            Values for masked particles only.

        Returns
        -------
        ndarray
            Full array with zeros for masked-out particles.
        """
        full = np.zeros(len(self.p))
        full[self.mask] = val
        return full

    def set_mask(self, mask: np.ndarray) -> None:
        """
        Set mask for active particles.

        Parameters
        ----------
        mask : ndarray
            Boolean mask or index array.
        """
        self.mask = np.asarray(mask)

    def __repr__(self) -> str:
        return (f"Compound(n_eps={len(self.eps)}, "
                f"n_particles={len(self.p)}, "
                f"inout={self.inout.shape})")
