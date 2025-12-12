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

from .materials import EpsConst, EpsTable
from .geometry import Particle, ComParticle, trisphere
from .greenfun import CompGreenStat, CompGreenRet
from .bem import BEMStat, BEMRet
from .excitation import PlaneWaveStat, PlaneWaveRet, DipoleStat, DipoleRet
from .utils.constants import EV2NM

__all__ = [
    "EpsConst",
    "EpsTable",
    "Particle",
    "ComParticle",
    "trisphere",
    "CompGreenStat",
    "CompGreenRet",
    "BEMStat",
    "BEMRet",
    "PlaneWaveStat",
    "PlaneWaveRet",
    "DipoleStat",
    "DipoleRet",
    "EV2NM",
]
