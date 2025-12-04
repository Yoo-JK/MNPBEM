"""
MNPBEM - Metallic Nanoparticle Boundary Element Method

A Python library for simulating the optical properties of metallic nanoparticles
using the boundary element method (BEM).

This is a Python port of the original MATLAB MNPBEM toolbox.
"""

__version__ = "1.0.0"

# Core imports
from .misc.options import BEMOptions, bemoptions
from .misc.units import eV2nm, nm2eV

# Material classes
from .material import EpsConst, EpsDrude, EpsTable, EpsFun

# Particle classes
from .particles import Particle, Compound, ComParticle, ComPoint

# Particle shapes
from .particles.shapes import trisphere, tricube, trirod, tritorus

# Green functions
from .greenfun import GreenStat, CompGreenStat

# BEM solvers
from .bem import BEMStat, bemsolver

# Simulation classes
from .simulation import PlaneWaveStat, DipoleStat, SpectrumStat, planewave, dipole

# Mie theory
from .mie import MieStat, MieRet, miesolver

__all__ = [
    # Version
    "__version__",
    # Options
    "BEMOptions",
    "bemoptions",
    # Units
    "eV2nm",
    "nm2eV",
    # Materials
    "EpsConst",
    "EpsDrude",
    "EpsTable",
    "EpsFun",
    # Particles
    "Particle",
    "Compound",
    "ComParticle",
    "ComPoint",
    # Shapes
    "trisphere",
    "tricube",
    "trirod",
    "tritorus",
    # Green functions
    "GreenStat",
    "CompGreenStat",
    # BEM
    "BEMStat",
    "bemsolver",
    # Simulation
    "PlaneWaveStat",
    "DipoleStat",
    "SpectrumStat",
    "planewave",
    "dipole",
    # Mie
    "MieStat",
    "MieRet",
    "miesolver",
]
