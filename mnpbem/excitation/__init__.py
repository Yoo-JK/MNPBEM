"""
Excitation module for MNPBEM.

Provides plane wave excitation classes for BEM simulations.
"""

from .planewave_stat import PlaneWaveStat
from .planewave_ret import PlaneWaveRet

__all__ = ['PlaneWaveStat', 'PlaneWaveRet']
