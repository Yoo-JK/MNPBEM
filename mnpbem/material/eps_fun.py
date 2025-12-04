"""
User-defined dielectric function.
"""

import numpy as np
from typing import Tuple, Callable, Literal

from .eps_base import EpsBase
from ..misc.units import eV2nm


class EpsFun(EpsBase):
    """
    Dielectric function using user-supplied function.

    Parameters
    ----------
    fun : callable
        Function for evaluation of dielectric function.
        Should take wavelength (or energy) and return complex eps.
    key : {'nm', 'eV'}, optional
        Whether function takes wavelengths (nm) or energies (eV).
        Default is 'nm'.

    Examples
    --------
    >>> # Function taking wavelength in nm
    >>> def eps_func(wavelength):
    ...     return 1 + 0.1 * (wavelength / 500)**2
    >>> eps = EpsFun(eps_func, key='nm')

    >>> # Function taking energy in eV
    >>> def eps_func_ev(energy):
    ...     return 1 + energy**2
    >>> eps = EpsFun(eps_func_ev, key='eV')
    """

    def __init__(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        key: Literal['nm', 'eV'] = 'nm'
    ):
        """
        Initialize user-defined dielectric function.

        Parameters
        ----------
        fun : callable
            Function for evaluation of dielectric function.
        key : {'nm', 'eV'}, optional
            Whether function takes wavelengths (nm) or energies (eV).
        """
        self.fun = fun
        self.key = key

        if key not in ('nm', 'eV'):
            raise ValueError(f"key must be 'nm' or 'eV', got '{key}'")

    def __call__(self, enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate user-defined dielectric function and wavenumber.

        Parameters
        ----------
        enei : array_like
            Light wavelength in vacuum (nm).

        Returns
        -------
        eps : ndarray
            Dielectric function (complex).
        k : ndarray
            Wavenumber in medium.
        """
        enei = np.asarray(enei, dtype=float)

        # Evaluate dielectric function
        if self.key == 'nm':
            eps = self.fun(enei)
        else:  # 'eV'
            energy = eV2nm / enei
            eps = self.fun(energy)

        eps = np.asarray(eps, dtype=complex)

        # Wavenumber
        k = 2 * np.pi / enei * np.sqrt(eps)

        return eps, k

    def __repr__(self) -> str:
        return f"EpsFun(fun={self.fun.__name__}, key='{self.key}')"
