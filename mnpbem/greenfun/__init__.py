"""
Green's functions for electromagnetic boundary element method.

Classes:
- GreenStat: Green function (quasistatic)
- GreenRet: Green function (retarded)
- CompGreenStat: Composite Green function (quasistatic)
- CompGreenRet: Composite Green function (retarded)
- CompGreenStatMirror: Composite Green function with mirror symmetry (quasistatic)
- CompGreenRetMirror: Composite Green function with mirror symmetry (retarded)
- GreenRetLayer: Green function for layer structure (retarded)
- CompGreenRetLayer: Composite Green function for layer structure (retarded)
- CompStruct: Structure for compound of points or particles
- CompStructMirror: Structure for compound with mirror symmetry
"""

from .greenstat import GreenStat
from .greenret import GreenRet
from .compgreen_stat import CompGreenStat, CompStruct
from .compgreen_ret import CompGreenRet
from .compgreen_stat_mirror import CompGreenStatMirror, CompStructMirror
from .compgreen_ret_mirror import CompGreenRetMirror
from .greenret_layer import GreenRetLayer
from .compgreen_ret_layer import CompGreenRetLayer

__all__ = [
    # Base Green functions
    "GreenStat",
    "GreenRet",
    # Composite Green functions
    "CompGreenStat",
    "CompGreenRet",
    # Mirror symmetry
    "CompGreenStatMirror",
    "CompGreenRetMirror",
    # Layer structure
    "GreenRetLayer",
    "CompGreenRetLayer",
    # Data structures
    "CompStruct",
    "CompStructMirror",
]
