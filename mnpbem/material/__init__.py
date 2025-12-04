"""
Dielectric function classes for MNPBEM.

This module provides various models for dielectric functions:
- EpsConst: Constant dielectric function
- EpsDrude: Drude model for metals (Au, Ag, Al)
- EpsTable: Tabulated dielectric functions from data files
- EpsFun: User-defined dielectric functions
"""

from .eps_base import EpsBase
from .eps_const import EpsConst
from .eps_drude import EpsDrude
from .eps_table import EpsTable
from .eps_fun import EpsFun

__all__ = [
    "EpsBase",
    "EpsConst",
    "EpsDrude",
    "EpsTable",
    "EpsFun",
]
