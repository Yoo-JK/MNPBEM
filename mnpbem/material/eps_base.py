"""
Base class for dielectric functions.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class EpsBase(ABC):
    """
    Abstract base class for dielectric functions.

    All dielectric function classes should inherit from this base class
    and implement the __call__ method.
    """

    @abstractmethod
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
            Dielectric function (complex).
        k : ndarray
            Wavenumber in medium.
        """
        pass

    def eps(self, enei: np.ndarray) -> np.ndarray:
        """
        Get dielectric function only.

        Parameters
        ----------
        enei : array_like
            Light wavelength in vacuum (nm).

        Returns
        -------
        eps : ndarray
            Dielectric function (complex).
        """
        return self(enei)[0]

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
        return self(enei)[1]

    def refractive_index(self, enei: np.ndarray) -> np.ndarray:
        """
        Get complex refractive index.

        Parameters
        ----------
        enei : array_like
            Light wavelength in vacuum (nm).

        Returns
        -------
        n : ndarray
            Complex refractive index.
        """
        eps = self.eps(enei)
        return np.sqrt(eps)
