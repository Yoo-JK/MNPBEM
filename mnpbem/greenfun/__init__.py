"""
Green function classes for MNPBEM.

This module provides Green function calculations:
- GreenStat: Quasistatic (Coulomb) Green function
- CompGreenStat: Composite Green function for compound particles
"""

from .green_stat import GreenStat
from .comp_green_stat import CompGreenStat

__all__ = [
    "GreenStat",
    "CompGreenStat",
]
