"""
Simulation classes for MNPBEM.

This module provides excitation and spectrum calculation classes:
- PlaneWaveStat: Plane wave excitation (quasistatic)
- DipoleStat: Dipole excitation (quasistatic)
- SpectrumStat: Spectral response calculations (quasistatic)
"""

from .planewave_stat import PlaneWaveStat, planewave
from .dipole_stat import DipoleStat, dipole
from .spectrum_stat import SpectrumStat

__all__ = [
    "PlaneWaveStat",
    "planewave",
    "DipoleStat",
    "dipole",
    "SpectrumStat",
]
