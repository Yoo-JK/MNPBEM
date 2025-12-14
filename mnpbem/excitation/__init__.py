"""
Excitation module for MNPBEM.

Provides plane wave and dipole excitation classes for BEM simulations.

Note: PlaneWave classes are now in the simulation module.
For backward compatibility, they are imported here.
"""

# Import from simulation module (complete implementations with field() method)
from ..simulation import PlaneWaveStat, PlaneWaveRet

# Import dipole excitations from this module
from .dipole_stat import DipoleStat
from .dipole_ret import DipoleRet

__all__ = ['PlaneWaveStat', 'PlaneWaveRet', 'DipoleStat', 'DipoleRet']
