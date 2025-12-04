"""
Compound structure for storing BEM solution data.
"""

import numpy as np
from typing import Optional, Any, Dict, Union

from .comparticle import ComParticle


class CompStruct:
    """
    Structure for storing BEM solution data.

    Used to pass results between BEM solver and excitation classes.

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    enei : float
        Light wavelength in nm.
    **kwargs : dict
        Named data arrays (e.g., sig=surface_charges).

    Attributes
    ----------
    p : ComParticle
        Associated particle.
    enei : float
        Wavelength.
    sig : ndarray, optional
        Surface charges.
    h : ndarray, optional
        Surface currents (retarded).
    phip : ndarray, optional
        External potential.
    """

    def __init__(
        self,
        p: ComParticle,
        enei: float,
        **kwargs
    ):
        """
        Initialize compound structure.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float
            Light wavelength in nm.
        **kwargs : dict
            Named data arrays.
        """
        self.p = p
        self.enei = enei
        self._data: Dict[str, np.ndarray] = {}

        for key, value in kwargs.items():
            self._data[key] = np.asarray(value)
            setattr(self, key, self._data[key])

    def __getitem__(self, key: str) -> np.ndarray:
        """Get data by key."""
        return self._data.get(key, getattr(self, key, None))

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Set data by key."""
        self._data[key] = np.asarray(value)
        setattr(self, key, self._data[key])

    def get(self, key: str, default: Any = None) -> Any:
        """Get data with default."""
        return self._data.get(key, default)

    def __add__(self, other: 'CompStruct') -> 'CompStruct':
        """Add two compound structures."""
        if self.p != other.p:
            raise ValueError("Cannot add CompStruct with different particles")
        if self.enei != other.enei:
            raise ValueError("Cannot add CompStruct with different wavelengths")

        result = CompStruct(self.p, self.enei)
        for key in set(self._data.keys()) | set(other._data.keys()):
            if key in self._data and key in other._data:
                result[key] = self._data[key] + other._data[key]
            elif key in self._data:
                result[key] = self._data[key].copy()
            else:
                result[key] = other._data[key].copy()
        return result

    def __sub__(self, other: 'CompStruct') -> 'CompStruct':
        """Subtract two compound structures."""
        if self.p != other.p:
            raise ValueError("Cannot subtract CompStruct with different particles")
        if self.enei != other.enei:
            raise ValueError("Cannot subtract CompStruct with different wavelengths")

        result = CompStruct(self.p, self.enei)
        for key in set(self._data.keys()) | set(other._data.keys()):
            if key in self._data and key in other._data:
                result[key] = self._data[key] - other._data[key]
            elif key in self._data:
                result[key] = self._data[key].copy()
            else:
                result[key] = -other._data[key]
        return result

    def __mul__(self, scalar: float) -> 'CompStruct':
        """Multiply by scalar."""
        result = CompStruct(self.p, self.enei)
        for key, value in self._data.items():
            result[key] = value * scalar
        return result

    def __rmul__(self, scalar: float) -> 'CompStruct':
        """Right multiply by scalar."""
        return self.__mul__(scalar)

    def __neg__(self) -> 'CompStruct':
        """Negate."""
        return self.__mul__(-1)

    def keys(self):
        """Return data keys."""
        return self._data.keys()

    def __repr__(self) -> str:
        data_keys = list(self._data.keys())
        return f"CompStruct(enei={self.enei}, data={data_keys})"
