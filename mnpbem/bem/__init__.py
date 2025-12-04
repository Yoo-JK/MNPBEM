"""
BEM solver classes for MNPBEM.

This module provides boundary element method solvers:
- BEMStat: Quasistatic BEM solver
- bemsolver: Factory function to create appropriate solver
"""

from .bem_base import BEMBase
from .bem_stat import BEMStat
from .factory import bemsolver

__all__ = [
    "BEMBase",
    "BEMStat",
    "bemsolver",
]
