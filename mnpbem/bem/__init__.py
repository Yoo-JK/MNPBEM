"""
BEM solvers for electromagnetic boundary element method.

Classes:
- BEMStat: BEM solver (quasistatic approximation)
- BEMRet: BEM solver (retarded/full Maxwell)
"""

from .bem_stat import BEMStat
from .bem_ret import BEMRet

__all__ = [
    "BEMStat",
    "BEMRet",
]
