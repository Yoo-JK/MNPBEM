"""
Base class for BEM solvers.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any

from ..particles import ComParticle, CompStruct


class BEMBase(ABC):
    """
    Abstract base class for BEM solvers.

    All BEM solver classes should inherit from this base class.
    """

    @abstractmethod
    def solve(self, exc: CompStruct) -> CompStruct:
        """
        Solve BEM equations for given excitation.

        Parameters
        ----------
        exc : CompStruct
            External excitation with 'phip' field.

        Returns
        -------
        CompStruct
            Solution with surface charges 'sig'.
        """
        pass

    @abstractmethod
    def field(self, sig: CompStruct, pts: Any) -> np.ndarray:
        """
        Compute electric field at given points.

        Parameters
        ----------
        sig : CompStruct
            Surface charges from BEM solution.
        pts : Point or ComPoint
            Evaluation points.

        Returns
        -------
        ndarray
            Electric field at points.
        """
        pass

    @abstractmethod
    def potential(self, sig: CompStruct, pts: Any) -> np.ndarray:
        """
        Compute potential at given points.

        Parameters
        ----------
        sig : CompStruct
            Surface charges from BEM solution.
        pts : Point or ComPoint
            Evaluation points.

        Returns
        -------
        ndarray
            Potential at points.
        """
        pass
