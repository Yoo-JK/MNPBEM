"""
MNPBEM - Metallic Nanoparticle Boundary Element Method
Python implementation of the MATLAB MNPBEM toolbox

Main modules:
- materials: Dielectric functions (EpsConst, EpsTable, EpsDrude)
- geometry: Particle geometries and mesh generation
- greenfun: Green's functions (static and retarded)
- bem: BEM solvers
- excitation: External excitations (plane wave, dipole)
- spectra: Optical spectra calculations
- fields: Field and potential calculations
- visualization: Plotting utilities
"""

__version__ = "0.1.0"

from .materials import EpsConst, EpsTable, EpsDrude
from .geometry import Particle, ComParticle, ComPoint, trisphere
from .greenfun import CompGreenStat, CompGreenRet
from .bem import BEMStat, BEMRet
from .simulation import PlaneWaveStat, PlaneWaveRet, DipoleStat, DipoleRet
from .spectrum import SpectrumRet, SpectrumStat
from .utils.constants import EV2NM

__all__ = [
    "EpsConst",
    "EpsTable",
    "EpsDrude",
    "Particle",
    "ComParticle",
    "ComPoint",
    "trisphere",
    "CompGreenStat",
    "CompGreenRet",
    "BEMStat",
    "BEMRet",
    "PlaneWaveStat",
    "PlaneWaveRet",
    "DipoleStat",
    "DipoleRet",
    "SpectrumRet",
    "SpectrumStat",
    "EV2NM",
]
