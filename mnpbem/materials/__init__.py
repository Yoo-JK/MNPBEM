"""
Material dielectric functions.

Classes:
- EpsConst: Constant dielectric function
- EpsTable: Tabulated dielectric function with interpolation
- EpsDrude: Drude model dielectric function
- EpsFun: User-supplied dielectric function

Functions:
- epsfun: Convenience factory for creating dielectric functions
"""

from .eps_const import EpsConst
from .eps_table import EpsTable
from .eps_drude import EpsDrude
from .epsfun import EpsFun, epsfun

__all__ = [
    "EpsConst",
    "EpsTable",
    "EpsDrude",
    "EpsFun",
    "epsfun",
]
