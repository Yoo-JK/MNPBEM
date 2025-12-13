"""
Material dielectric functions.

Classes:
- EpsConst: Constant dielectric function
- EpsTable: Tabulated dielectric function with interpolation
- EpsDrude: Drude model dielectric function
"""

from .eps_const import EpsConst
from .eps_table import EpsTable
from .eps_drude import EpsDrude

__all__ = [
    "EpsConst",
    "EpsTable",
    "EpsDrude",
]
