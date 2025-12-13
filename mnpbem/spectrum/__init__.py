"""
Spectrum module for computing far-fields and scattering cross sections.

Provides:
- SpectrumRet: For full Maxwell equations (retarded case)
- SpectrumStat: For quasistatic approximation
"""

from .spectrum_ret import SpectrumRet
from .spectrum_stat import SpectrumStat

__all__ = ['SpectrumRet', 'SpectrumStat']
