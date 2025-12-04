"""
Drude model for dielectric functions of metals.
"""

import numpy as np
from typing import Tuple, Optional

from .eps_base import EpsBase
from ..misc.units import eV2nm, HARTREE, TUNIT


class EpsDrude(EpsBase):
    """
    Drude dielectric function for metals.

    The Drude model is:
        eps = eps0 - wp^2 / (w * (w + i * gammad))

    where:
        eps0 : background dielectric constant
        wp : plasmon energy (eV)
        gammad : damping rate (eV)
        w : photon energy (eV)

    Parameters
    ----------
    name : str, optional
        Material name: 'Au', 'Ag', or 'Al'.
        If not provided, must set eps0, wp, gammad manually.
    eps0 : float, optional
        Background dielectric constant.
    wp : float, optional
        Plasmon energy in eV.
    gammad : float, optional
        Damping rate in eV.

    Examples
    --------
    >>> # Gold
    >>> eps_gold = EpsDrude('Au')
    >>> eps, k = eps_gold(500)  # At 500 nm

    >>> # Custom Drude parameters
    >>> eps_custom = EpsDrude(eps0=10.0, wp=9.0, gammad=0.1)
    """

    # Predefined material parameters
    MATERIALS = {
        'Au': {'rs': 3.0, 'eps0': 10.0, 'gammad_factor': 10.0},
        'gold': {'rs': 3.0, 'eps0': 10.0, 'gammad_factor': 10.0},
        'Ag': {'rs': 3.0, 'eps0': 3.3, 'gammad_factor': 30.0},
        'silver': {'rs': 3.0, 'eps0': 3.3, 'gammad_factor': 30.0},
        'Al': {'rs': 2.07, 'eps0': 1.0, 'gammad_eV': 1.06},
        'aluminum': {'rs': 2.07, 'eps0': 1.0, 'gammad_eV': 1.06},
    }

    def __init__(
        self,
        name: Optional[str] = None,
        eps0: Optional[float] = None,
        wp: Optional[float] = None,
        gammad: Optional[float] = None,
    ):
        """
        Initialize Drude dielectric function.

        Parameters
        ----------
        name : str, optional
            Material name: 'Au', 'Ag', or 'Al'.
        eps0 : float, optional
            Background dielectric constant.
        wp : float, optional
            Plasmon energy in eV.
        gammad : float, optional
            Damping rate in eV.
        """
        self.name = name

        if name is not None:
            self._init_from_name(name)
        else:
            if eps0 is None or wp is None or gammad is None:
                raise ValueError(
                    "Must provide either 'name' or all of (eps0, wp, gammad)"
                )
            self.eps0 = eps0
            self.wp = wp
            self.gammad = gammad

        # Override with explicit values if provided
        if eps0 is not None and name is not None:
            self.eps0 = eps0
        if wp is not None and name is not None:
            self.wp = wp
        if gammad is not None and name is not None:
            self.gammad = gammad

    def _init_from_name(self, name: str) -> None:
        """Initialize parameters from material name."""
        if name not in self.MATERIALS:
            raise ValueError(
                f"Unknown material '{name}'. Available: {list(self.MATERIALS.keys())}"
            )

        params = self.MATERIALS[name]
        rs = params['rs']
        self.eps0 = params['eps0']

        # Compute gammad
        if 'gammad_eV' in params:
            gammad = params['gammad_eV'] / HARTREE
        else:
            gammad = TUNIT / params['gammad_factor']

        # Compute plasmon energy from electron gas parameter
        # density in atomic units: n = 3 / (4 * pi * rs^3)
        density = 3 / (4 * np.pi * rs ** 3)
        # plasmon frequency: wp = sqrt(4 * pi * n)
        wp = np.sqrt(4 * np.pi * density)

        # Save values in eV
        self.gammad = gammad * HARTREE
        self.wp = wp * HARTREE

    def __call__(self, enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate Drude dielectric function and wavenumber.

        Parameters
        ----------
        enei : array_like
            Light wavelength in vacuum (nm).

        Returns
        -------
        eps : ndarray
            Drude dielectric function (complex).
        k : ndarray
            Wavenumber in medium.
        """
        enei = np.asarray(enei, dtype=float)

        # Convert wavelength to energy (eV)
        w = eV2nm / enei

        # Drude formula: eps = eps0 - wp^2 / (w * (w + i*gammad))
        eps = self.eps0 - self.wp ** 2 / (w * (w + 1j * self.gammad))

        # Wavenumber
        k = 2 * np.pi / enei * np.sqrt(eps)

        return eps, k

    def __repr__(self) -> str:
        if self.name:
            return f"EpsDrude('{self.name}')"
        return f"EpsDrude(eps0={self.eps0}, wp={self.wp}, gammad={self.gammad})"
