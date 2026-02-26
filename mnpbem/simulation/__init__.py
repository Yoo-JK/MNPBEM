"""
Simulation module for MNPBEM.

This module provides excitation sources for BEM simulations:
- PlaneWaveStat: Plane wave excitation for quasistatic simulations
- PlaneWaveRet: Plane wave excitation for retarded simulations
- DipoleStat: Dipole excitation for quasistatic simulations
- DipoleRet: Dipole excitation for retarded simulations
- EELSBase: Base class for EELS simulations
- EELSStat: EELS excitation for quasistatic simulations
- EELSRet: EELS excitation for retarded simulations

Matches MATLAB MNPBEM Simulation module exactly.
"""

from .planewave_stat import PlaneWaveStat
from .planewave_ret import PlaneWaveRet
from .dipole_stat import DipoleStat
from .dipole_ret import DipoleRet
from .eels_base import EELSBase
from .eels_stat import EELSStat
from .eels_ret import EELSRet

__all__ = [
    "PlaneWaveStat",
    "PlaneWaveRet",
    "DipoleStat",
    "DipoleRet",
    "EELSBase",
    "EELSStat",
    "EELSRet",
]
