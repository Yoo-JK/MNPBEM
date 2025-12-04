"""
Quasistatic BEM solver.
"""

import numpy as np
from typing import Optional, Union, Tuple

from .bem_base import BEMBase
from ..particles import ComParticle, CompStruct, Point, ComPoint
from ..greenfun import CompGreenStat, GreenStat
from ..misc.options import BEMOptions


class BEMStat(BEMBase):
    """
    BEM solver for quasistatic approximation.

    Solves the boundary integral equation for surface charges such that
    Maxwell's equations are fulfilled in the quasistatic limit.

    The quasistatic BEM equation is:
        (Lambda + F) * sig = -phip

    where:
        Lambda = 2*pi*(eps1 + eps2)/(eps1 - eps2) (diagonal)
        F = surface derivative of Green function
        sig = surface charges
        phip = external potential at boundary

    References
    ----------
    Garcia de Abajo & Howie, PRB 65, 115418 (2002)

    Parameters
    ----------
    p : ComParticle
        Compound particle.
    enei : float, optional
        Initial wavelength for precomputation.
    **kwargs : dict
        Options (rel_cutoff, etc.)

    Examples
    --------
    >>> from mnpbem import ComParticle, EpsConst, EpsTable, BEMStat
    >>> from mnpbem.particles.shapes import trisphere
    >>>
    >>> epstab = [EpsConst(1), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10)
    >>> p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)
    >>> bem = BEMStat(p)
    """

    def __init__(
        self,
        p: ComParticle,
        enei: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize quasistatic BEM solver.

        Parameters
        ----------
        p : ComParticle
            Compound particle.
        enei : float, optional
            Wavelength for precomputation.
        **kwargs : dict
            Options.
        """
        self.p = p
        self.options = kwargs

        # Create Green function
        self.g = CompGreenStat(p, p, **kwargs)

        # Surface derivative of Green function
        self._F = self.g.F

        # Cache for resolvent matrix
        self._mat = None
        self._enei = None

        # Precompute if wavelength given
        if enei is not None:
            self._compute_resolvent(enei)

    @property
    def F(self) -> np.ndarray:
        """Surface derivative of Green function matrix."""
        return self._F

    @property
    def n_faces(self) -> int:
        """Number of boundary elements."""
        return self.p.n_faces

    def _compute_resolvent(self, enei: float) -> None:
        """
        Compute BEM resolvent matrix for given wavelength.

        Parameters
        ----------
        enei : float
            Wavelength in nm.
        """
        if self._enei == enei and self._mat is not None:
            return

        # Compute Lambda factor for each face
        Lambda = self.p.lambda_factor(enei)

        # BEM matrix: Lambda + F
        # Resolvent: -inv(Lambda + F)
        bem_matrix = np.diag(Lambda) + self._F
        self._mat = -np.linalg.inv(bem_matrix)
        self._enei = enei

    def solve(self, exc: CompStruct) -> CompStruct:
        """
        Solve BEM equations for given excitation.

        Parameters
        ----------
        exc : CompStruct
            Excitation with 'phip' field (external potential at boundary).

        Returns
        -------
        CompStruct
            Solution with 'sig' field (surface charges).
        """
        # Ensure resolvent is computed for this wavelength
        self._compute_resolvent(exc.enei)

        # Get external potential
        phip = exc.get('phip')
        if phip is None:
            raise ValueError("Excitation must have 'phip' field")

        # Solve: sig = -inv(Lambda + F) @ phip
        sig = self._mat @ phip

        return CompStruct(self.p, exc.enei, sig=sig)

    def __truediv__(self, exc: CompStruct) -> CompStruct:
        """Allow bem / exc syntax (alternative to solve)."""
        return self.solve(exc)

    def __rtruediv__(self, exc: CompStruct) -> CompStruct:
        """Allow exc / bem syntax."""
        return self.solve(exc)

    def field(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
        """
        Compute electric field from surface charges.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.
        pts : Point or ComPoint, optional
            Evaluation points. If None, evaluates at particle surface.

        Returns
        -------
        ndarray
            Electric field, shape (n_pts, 3).
        """
        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        if pts is None:
            # Field at particle surface
            return self.g.field(charges)
        else:
            # Field at external points
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenStat(pts_obj, self.p.pc, **self.options)
            return g_pts.field(charges)

    def potential(
        self,
        sig: CompStruct,
        pts: Optional[Union[Point, ComPoint]] = None
    ) -> np.ndarray:
        """
        Compute potential from surface charges.

        Parameters
        ----------
        sig : CompStruct
            BEM solution with surface charges.
        pts : Point or ComPoint, optional
            Evaluation points. If None, evaluates at particle surface.

        Returns
        -------
        ndarray
            Potential at points.
        """
        charges = sig.get('sig')
        if charges is None:
            raise ValueError("Solution must have 'sig' field")

        if pts is None:
            return self.g.potential(charges)
        else:
            if isinstance(pts, ComPoint):
                pts_obj = pts.pc
            else:
                pts_obj = pts

            g_pts = GreenStat(pts_obj, self.p.pc, **self.options)
            return g_pts.potential(charges)

    def __repr__(self) -> str:
        return f"BEMStat(p={self.p}, enei={self._enei})"
