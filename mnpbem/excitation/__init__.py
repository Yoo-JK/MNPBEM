"""
Excitation module for MNPBEM.

Provides plane wave and dipole excitation classes for BEM simulations.

Note: All excitation classes are now in the simulation module.
For backward compatibility, they are imported here.
"""

# Import all excitation classes from simulation module
from ..simulation import PlaneWaveStat, PlaneWaveRet, DipoleStat, DipoleRet

__all__ = ['PlaneWaveStat', 'PlaneWaveRet', 'DipoleStat', 'DipoleRet']
