"""
Green's functions for electromagnetic boundary element method.

Classes:
- CompGreenStat: Composite Green function (quasistatic)
- CompGreenRet: Composite Green function (retarded)
"""

from .compgreen_stat import CompGreenStat
from .compgreen_ret import CompGreenRet

__all__ = [
    "CompGreenStat",
    "CompGreenRet",
]
