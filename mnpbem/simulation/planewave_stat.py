"""
Plane wave excitation for quasistatic BEM.
"""

import numpy as np
from typing import Optional, Union, Tuple

from ..particles import ComParticle, CompStruct
from ..misc.options import BEMOptions


class PlaneWaveStat:
    """
    Plane wave excitation within quasistatic approximation.

    In the quasistatic limit, a plane wave is simply a uniform electric
    field E = E0 * pol, where pol is the polarization direction.

    Parameters
    ----------
    pol : ndarray
        Light polarization directions, shape (n_exc, 3) or (3,).
    direction : ndarray, optional
        Light propagation directions (ignored in quasistatic limit).
    medium : int
        Index of medium through which particle is excited.

    Examples
    --------
    >>> # Single polarization along x
    >>> exc = PlaneWaveStat([1, 0, 0])
    >>>
    >>> # Two orthogonal polarizations
    >>> exc = PlaneWaveStat([[1, 0, 0], [0, 1, 0]])
    """

    def __init__(
        self,
        pol: np.ndarray,
        direction: Optional[np.ndarray] = None,
        medium: int = 1,
        **kwargs
    ):
        """
        Initialize plane wave excitation.

        Parameters
        ----------
        pol : ndarray
            Polarization directions.
        direction : ndarray, optional
            Propagation directions (ignored in quasistatic).
        medium : int
            Index of excitation medium (1-based).
        """
        self.pol = np.atleast_2d(pol)
        self.direction = direction
        self.medium = medium
        self.options = kwargs

    @property
    def n_exc(self) -> int:
        """Number of excitations (polarizations)."""
        return len(self.pol)

    def __call__(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute external potential for BEM solver.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        return self.potential(p, enei)

    def potential(self, p: ComParticle, enei: float) -> CompStruct:
        """
        Compute potential derivative at particle boundary.

        In quasistatic limit: phip = -E . n = -pol . nvec

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        CompStruct
            Excitation with 'phip' field.
        """
        # phip[i, j] = -pol[j] . nvec[i]
        # Shape: (n_faces, n_exc)
        phip = -p.nvec @ self.pol.T

        return CompStruct(p, enei, phip=phip)

    def field(self, p: ComParticle, enei: float) -> np.ndarray:
        """
        Electric field of plane wave.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Wavelength in nm.

        Returns
        -------
        ndarray
            Electric field at face centroids, shape (n_faces, 3, n_exc).
        """
        # Uniform field: E = pol for each face
        n_faces = p.n_faces
        field = np.zeros((n_faces, 3, self.n_exc))
        for i, pol in enumerate(self.pol):
            field[:, :, i] = pol

        return field

    def scattering(self, sig: CompStruct) -> np.ndarray:
        """
        Compute scattering cross section.

        sca = (8*pi/3) * k^4 * |p|^2

        where p is the induced dipole moment.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.

        Returns
        -------
        ndarray
            Scattering cross section for each polarization.
        """
        return self.sca(sig)

    def sca(self, sig: CompStruct) -> np.ndarray:
        """Alias for scattering."""
        # Induced dipole moment: p = integral(r * sigma * dA)
        area = sig.p.area
        pos = sig.p.pos
        charges = sig.get('sig')

        # dip[k, j] = sum_i pos[i, k] * area[i] * sig[i, j]
        # Shape: (3, n_exc)
        weighted_pos = pos * area[:, np.newaxis]  # (n_faces, 3)
        dip = weighted_pos.T @ charges  # (3, n_exc)

        # Get wavenumber in medium
        eps_func = sig.p.eps[self.medium - 1]  # Convert to 0-based
        _, k = eps_func(sig.enei)

        # Scattering cross section: (8*pi/3) * k^4 * |p|^2
        sca = (8 * np.pi / 3) * k ** 4 * np.sum(np.abs(dip) ** 2, axis=0)

        return np.real(sca)

    def absorption(self, sig: CompStruct) -> np.ndarray:
        """
        Compute absorption cross section.

        abs = 4*pi*k * Im(pol* . p)

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.

        Returns
        -------
        ndarray
            Absorption cross section for each polarization.
        """
        return self.abs(sig)

    def abs(self, sig: CompStruct) -> np.ndarray:
        """Alias for absorption."""
        # Induced dipole moment
        area = sig.p.area
        pos = sig.p.pos
        charges = sig.get('sig')

        weighted_pos = pos * area[:, np.newaxis]
        dip = weighted_pos.T @ charges  # (3, n_exc)

        # Get wavenumber
        eps_func = sig.p.eps[self.medium - 1]
        _, k = eps_func(sig.enei)

        # Absorption: 4*pi*k * Im(pol* . p)
        # pol[j, k] * dip[k, j] summed over k
        pol_dot_dip = np.sum(self.pol.T * dip, axis=0)
        abs_cs = 4 * np.pi * k * np.imag(pol_dot_dip)

        return np.real(abs_cs)

    def extinction(self, sig: CompStruct) -> np.ndarray:
        """
        Compute extinction cross section.

        ext = sca + abs

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.

        Returns
        -------
        ndarray
            Extinction cross section for each polarization.
        """
        return self.ext(sig)

    def ext(self, sig: CompStruct) -> np.ndarray:
        """Alias for extinction."""
        return self.sca(sig) + self.abs(sig)

    def __repr__(self) -> str:
        return f"PlaneWaveStat(pol={self.pol.tolist()}, medium={self.medium})"


def planewave(
    pol: np.ndarray,
    direction: Optional[np.ndarray] = None,
    options: Optional[Union[BEMOptions, dict]] = None,
    **kwargs
) -> PlaneWaveStat:
    """
    Factory function for plane wave excitation.

    Parameters
    ----------
    pol : ndarray
        Polarization directions.
    direction : ndarray, optional
        Propagation directions.
    options : BEMOptions or dict, optional
        Simulation options.
    **kwargs : dict
        Additional options.

    Returns
    -------
    PlaneWaveStat
        Plane wave excitation object.
    """
    # Handle options
    if options is None:
        options = {}
    elif isinstance(options, BEMOptions):
        options = options.extra.copy()

    all_options = {**options, **kwargs}

    # Get medium
    medium = all_options.pop('medium', 1)

    return PlaneWaveStat(pol, direction, medium=medium, **all_options)
