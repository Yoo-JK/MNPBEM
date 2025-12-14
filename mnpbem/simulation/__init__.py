"""
Simulation module for MNPBEM.

This module provides excitation sources for BEM simulations:
- PlaneWaveStat: Plane wave excitation for quasistatic simulations
- PlaneWaveRet: Plane wave excitation for retarded simulations

Matches MATLAB MNPBEM Simulation module exactly.
"""

from .planewave_stat import PlaneWaveStat
from .planewave_ret import PlaneWaveRet

__all__ = [
    "PlaneWaveStat",
    "PlaneWaveRet",
]
