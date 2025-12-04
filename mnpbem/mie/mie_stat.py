"""
Quasistatic Mie theory for spherical particles.
"""

import numpy as np
from typing import Any, Union


class MieStat:
    """
    Mie theory for spherical particles in quasistatic approximation.

    The quasistatic (dipole) approximation is valid when the particle
    is much smaller than the wavelength (diameter << lambda).

    In this limit, the polarizability is:
        alpha = (eps_in - eps_out) / (eps_in + 2*eps_out) * a^3

    Parameters
    ----------
    epsin : dielectric function
        Dielectric function inside the sphere.
    epsout : dielectric function
        Dielectric function outside the sphere.
    diameter : float
        Diameter of the sphere in nm.

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, MieStat
    >>> eps_out = EpsConst(1.0)  # vacuum
    >>> eps_in = EpsTable('gold.dat')
    >>> mie = MieStat(eps_in, eps_out, 10)  # 10 nm gold sphere
    >>> sca = mie.sca(np.linspace(400, 800, 100))
    """

    def __init__(
        self,
        epsin: Any,
        epsout: Any,
        diameter: float,
        **kwargs
    ):
        """
        Initialize quasistatic Mie solver.

        Parameters
        ----------
        epsin : dielectric function
            Dielectric function inside sphere.
        epsout : dielectric function
            Dielectric function outside sphere.
        diameter : float
            Sphere diameter in nm.
        """
        self.epsin = epsin
        self.epsout = epsout
        self.diameter = diameter
        self.options = kwargs

    @property
    def radius(self) -> float:
        """Sphere radius."""
        return self.diameter / 2

    def polarizability(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute dipole polarizability.

        alpha = (eps_in - eps_out) / (eps_in + 2*eps_out) * a^3

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Complex polarizability.
        """
        enei = np.atleast_1d(enei)

        # Get dielectric functions
        eps_in, _ = self.epsin(enei)
        eps_out, _ = self.epsout(enei)

        # Polarizability (Clausius-Mossotti)
        a = self.radius
        alpha = (eps_in - eps_out) / (eps_in + 2 * eps_out) * a ** 3

        return alpha

    def scattering(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute scattering cross section.

        sca = (8*pi/3) * k^4 * |alpha|^2

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Scattering cross section in nm^2.
        """
        return self.sca(enei)

    def sca(self, enei: np.ndarray) -> np.ndarray:
        """Alias for scattering."""
        enei = np.atleast_1d(enei)

        # Get background dielectric and wavenumber
        eps_out, k = self.epsout(enei)

        # Polarizability
        alpha = self.polarizability(enei)

        # Scattering cross section
        sca = (8 * np.pi / 3) * np.real(k) ** 4 * np.abs(alpha) ** 2

        return np.real(sca)

    def absorption(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute absorption cross section.

        abs = 4*pi*k * Im(alpha)

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Absorption cross section in nm^2.
        """
        return self.abs(enei)

    def abs(self, enei: np.ndarray) -> np.ndarray:
        """Alias for absorption."""
        enei = np.atleast_1d(enei)

        # Get wavenumber
        _, k = self.epsout(enei)

        # Polarizability
        alpha = self.polarizability(enei)

        # Absorption cross section
        abs_cs = 4 * np.pi * np.real(k) * np.imag(alpha)

        return np.real(abs_cs)

    def extinction(self, enei: np.ndarray) -> np.ndarray:
        """
        Compute extinction cross section.

        ext = sca + abs

        Parameters
        ----------
        enei : ndarray
            Wavelengths in nm.

        Returns
        -------
        ndarray
            Extinction cross section in nm^2.
        """
        return self.ext(enei)

    def ext(self, enei: np.ndarray) -> np.ndarray:
        """Alias for extinction."""
        return self.sca(enei) + self.abs(enei)

    def __repr__(self) -> str:
        return f"MieStat(diameter={self.diameter})"
