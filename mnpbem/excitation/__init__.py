"""
Excitation module for MNPBEM.

Provides plane wave and dipole excitation classes for BEM simulations.
"""

from .planewave_stat import PlaneWaveStat
from .planewave_ret import PlaneWaveRet
from .dipole_stat import DipoleStat
from .dipole_ret import DipoleRet

__all__ = ['PlaneWaveStat', 'PlaneWaveRet', 'DipoleStat', 'DipoleRet']
