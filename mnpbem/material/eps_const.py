"""
Constant dielectric function.
"""

import numpy as np
from typing import Tuple, Union

from .eps_base import EpsBase


class EpsConst(EpsBase):
    """
    Constant dielectric function.

    Parameters
    ----------
    eps : float or complex
        Value of the dielectric constant.

    Examples
    --------
    >>> # Vacuum
    >>> eps_vacuum = EpsConst(1.0)
    >>> # Water (n = 1.33)
    >>> eps_water = EpsConst(1.33**2)
    >>> # Get dielectric function at 500 nm
    >>> eps, k = eps_water(500)
    """

    def __init__(self, eps: Union[float, complex]):
        """
        Initialize constant dielectric function.

        Parameters
        ----------
        eps : float or complex
            Value of the dielectric constant.
        """
        self._eps = complex(eps)

    @property
    def value(self) -> complex:
        """Return the dielectric constant value."""
        return self._eps

    def __call__(self, enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate dielectric function and wavenumber.

        Parameters
        ----------
        enei : array_like
            Light wavelength in vacuum (nm).

        Returns
        -------
        eps : ndarray
            Dielectric constant (same shape as enei).
        k : ndarray
            Wavenumber in medium.
        """
        enei = np.asarray(enei)
        eps = np.full(enei.shape, self._eps, dtype=complex)
        k = 2 * np.pi / enei * np.sqrt(self._eps)
        return eps, k

    def wavenumber(self, enei: np.ndarray) -> np.ndarray:
        """
        Get wavenumber in medium.

        Parameters
        ----------
        enei : array_like
            Light wavelength in vacuum (nm).

        Returns
        -------
        k : ndarray
            Wavenumber in medium.
        """
        enei = np.asarray(enei)
        return 2 * np.pi / enei * np.sqrt(self._eps)

    def __repr__(self) -> str:
        return f"EpsConst(eps={self._eps})"
