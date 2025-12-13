"""
Drude model dielectric function.

Matches MATLAB MNPBEM @epsdrude implementation exactly.
"""

import numpy as np

# Physical constant: hc in eV*nm
EV2NM = 1240.0  # eV * nm


class EpsDrude:
    """
    Drude model dielectric function.

    Formula:
        eps = eps0 - wp^2 / (w * (w + i*gammad))

    where w is the photon energy in eV.

    Parameters
    ----------
    eps0 : float
        Background dielectric constant (high-frequency limit)
    wp : float
        Plasma frequency in eV
    gammad : float
        Damping rate in eV

    Examples
    --------
    >>> # Gold (approximate Drude parameters)
    >>> eps_au = EpsDrude(9.5, 8.95, 0.069)
    >>> eps_val, k = eps_au(500)  # at 500 nm

    >>> # Use predefined metal
    >>> eps_au = EpsDrude.gold()
    >>> eps_ag = EpsDrude.silver()

    Notes
    -----
    MATLAB equivalent: @epsdrude class

    Common Drude parameters:
    - Gold (Au):   eps0=9.5,  wp=8.95 eV, gammad=0.069 eV
    - Silver (Ag): eps0=3.7,  wp=9.17 eV, gammad=0.021 eV
    - Aluminum (Al): eps0=1.0, wp=15.0 eV, gammad=0.6 eV
    """

    def __init__(self, eps0, wp, gammad, name=None):
        """
        Initialize Drude dielectric function.

        Parameters
        ----------
        eps0 : float
            Background dielectric constant
        wp : float
            Plasma frequency in eV
        gammad : float
            Damping rate in eV
        name : str, optional
            Material name (e.g., 'Au', 'Ag')
        """
        self.eps0 = eps0
        self.wp = wp
        self.gammad = gammad
        self.name = name

    def __call__(self, enei):
        """
        Get dielectric constant and wavenumber.

        MATLAB: subsref.m
            w = eV2nm / enei
            eps = eps0 - wp^2 / (w * (w + 1i*gammad))
            k = 2*pi / enei * sqrt(eps)

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        eps : complex or ndarray
            Drude dielectric function
        k : complex or ndarray
            Wavenumber in medium (1/nm)
        """
        enei = np.asarray(enei, dtype=float)

        # Convert wavelength to photon energy in eV
        # MATLAB: w = eV2nm / enei
        w = EV2NM / enei

        # Drude formula
        # MATLAB: eps = eps0 - wp^2 / (w * (w + 1i*gammad))
        eps = self.eps0 - self.wp**2 / (w * (w + 1j * self.gammad))

        # Wavenumber: k = 2π/λ × √ε
        k = 2 * np.pi / enei * np.sqrt(eps)

        return eps, k

    def wavenumber(self, enei):
        """
        Get wavenumber in medium.

        Parameters
        ----------
        enei : float or array_like
            Light wavelength in vacuum (nm)

        Returns
        -------
        k : complex or ndarray
            Wavenumber in medium (1/nm)
        """
        _, k = self(enei)
        return k

    @classmethod
    def gold(cls):
        """
        Create Drude model for gold (Au).

        Returns
        -------
        EpsDrude
            Gold dielectric function
        """
        # Drude parameters for gold
        # From Johnson & Christy / typical literature values
        return cls(eps0=9.5, wp=8.95, gammad=0.069, name='Au')

    @classmethod
    def silver(cls):
        """
        Create Drude model for silver (Ag).

        Returns
        -------
        EpsDrude
            Silver dielectric function
        """
        return cls(eps0=3.7, wp=9.17, gammad=0.021, name='Ag')

    @classmethod
    def aluminum(cls):
        """
        Create Drude model for aluminum (Al).

        Returns
        -------
        EpsDrude
            Aluminum dielectric function
        """
        return cls(eps0=1.0, wp=15.0, gammad=0.6, name='Al')

    def __repr__(self):
        if self.name:
            return f"EpsDrude('{self.name}', eps0={self.eps0}, wp={self.wp}, gammad={self.gammad})"
        return f"EpsDrude(eps0={self.eps0}, wp={self.wp}, gammad={self.gammad})"

    def __str__(self):
        name_str = f" ({self.name})" if self.name else ""
        return f"Drude dielectric function{name_str}: ε = {self.eps0} - {self.wp}²/(ω(ω+i{self.gammad}))"
