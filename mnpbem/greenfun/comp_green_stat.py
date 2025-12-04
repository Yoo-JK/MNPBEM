"""
Composite Green function for compound particles.
"""

import numpy as np
from typing import Optional, Union

from .green_stat import GreenStat
from ..particles import ComParticle, ComPoint


class CompGreenStat:
    """
    Green function for composite particles in quasistatic approximation.

    Wraps GreenStat for use with compound particles that have
    multiple dielectric regions.

    Parameters
    ----------
    p1 : ComParticle or ComPoint
        Source compound.
    p2 : ComParticle
        Target compound particle.
    **kwargs : dict
        Options passed to GreenStat.

    Attributes
    ----------
    p1 : ComParticle or ComPoint
        Source compound.
    p2 : ComParticle
        Target compound particle.
    g : GreenStat
        Underlying Green function object.
    """

    def __init__(
        self,
        p1: Union[ComParticle, ComPoint],
        p2: ComParticle,
        **kwargs
    ):
        """
        Initialize composite Green function.

        Parameters
        ----------
        p1 : ComParticle or ComPoint
            Source compound.
        p2 : ComParticle
            Target compound particle.
        """
        self.p1 = p1
        self.p2 = p2

        # Create underlying Green function using combined particles
        self.g = GreenStat(p1.pc, p2.pc, **kwargs)

    @property
    def G(self) -> np.ndarray:
        """Green function matrix."""
        return self.g.G()

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function."""
        return self.g.F()

    @property
    def H1(self) -> np.ndarray:
        """F + 2*pi."""
        return self.g.H1()

    @property
    def H2(self) -> np.ndarray:
        """F - 2*pi."""
        return self.g.H2()

    @property
    def Gp(self) -> np.ndarray:
        """Gradient of Green function."""
        return self.g.Gp()

    def potential(self, sig: np.ndarray) -> np.ndarray:
        """
        Compute potential from surface charges.

        Parameters
        ----------
        sig : ndarray
            Surface charges.

        Returns
        -------
        ndarray
            Potential at p1 positions.
        """
        return self.g.potential(sig)

    def field(self, sig: np.ndarray) -> np.ndarray:
        """
        Compute electric field from surface charges.

        Parameters
        ----------
        sig : ndarray
            Surface charges.

        Returns
        -------
        ndarray
            Electric field at p1 positions.
        """
        return self.g.field(sig)

    def __repr__(self) -> str:
        return f"CompGreenStat(p1={self.p1}, p2={self.p2})"
