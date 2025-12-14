"""
Green's functions for electromagnetic boundary element method.

Classes:
- CompGreenStat: Composite Green function (quasistatic)
- CompGreenRet: Composite Green function (retarded)
- CompStruct: Structure for compound of points or particles
"""

from .compgreen_stat import CompGreenStat, CompStruct
from .compgreen_ret import CompGreenRet

__all__ = [
    "CompGreenStat",
    "CompGreenRet",
    "CompStruct",
]
