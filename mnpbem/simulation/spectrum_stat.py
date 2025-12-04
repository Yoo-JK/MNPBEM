"""
Spectral response calculations for quasistatic BEM.
"""

import numpy as np
from typing import Optional, Callable, List, Union
from tqdm import tqdm

from ..particles import ComParticle, CompStruct
from ..bem import BEMStat


class SpectrumStat:
    """
    Spectral response calculations for quasistatic simulations.

    Computes optical spectra (scattering, extinction, absorption)
    over a range of wavelengths.

    Parameters
    ----------
    bem : BEMStat
        BEM solver.
    exc : callable
        Excitation object (PlaneWaveStat or DipoleStat).
    wavelengths : ndarray
        Array of wavelengths in nm.

    Examples
    --------
    >>> from mnpbem import trisphere, ComParticle, EpsConst, EpsTable
    >>> from mnpbem import BEMStat, PlaneWaveStat, SpectrumStat
    >>>
    >>> epstab = [EpsConst(1), EpsTable('gold.dat')]
    >>> p = ComParticle(epstab, [trisphere(144, 10)], [[2, 1]], closed=1)
    >>> bem = BEMStat(p)
    >>> exc = PlaneWaveStat([1, 0, 0])
    >>> wavelengths = np.linspace(400, 800, 50)
    >>> spec = SpectrumStat(bem, exc, wavelengths)
    >>> sca, ext = spec.compute()
    """

    def __init__(
        self,
        bem: BEMStat,
        exc,
        wavelengths: np.ndarray,
        show_progress: bool = True
    ):
        """
        Initialize spectrum calculator.

        Parameters
        ----------
        bem : BEMStat
            BEM solver.
        exc : callable
            Excitation object.
        wavelengths : ndarray
            Wavelengths in nm.
        show_progress : bool
            Show progress bar.
        """
        self.bem = bem
        self.exc = exc
        self.wavelengths = np.asarray(wavelengths)
        self.show_progress = show_progress

        # Storage for results
        self._sca = None
        self._abs = None
        self._ext = None

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelengths."""
        return len(self.wavelengths)

    def compute(self) -> tuple:
        """
        Compute scattering and extinction spectra.

        Returns
        -------
        sca : ndarray
            Scattering cross section, shape (n_wavelengths, n_exc).
        ext : ndarray
            Extinction cross section, shape (n_wavelengths, n_exc).
        """
        n_exc = self.exc.n_exc

        sca = np.zeros((self.n_wavelengths, n_exc))
        ext = np.zeros((self.n_wavelengths, n_exc))
        abs_cs = np.zeros((self.n_wavelengths, n_exc))

        iterator = enumerate(self.wavelengths)
        if self.show_progress:
            iterator = tqdm(list(iterator), desc="Computing spectrum")

        for i, enei in iterator:
            # Get excitation
            exc_struct = self.exc(self.bem.p, enei)

            # Solve BEM equations
            sig = self.bem.solve(exc_struct)

            # Compute cross sections
            sca[i] = self.exc.sca(sig)
            ext[i] = self.exc.ext(sig)
            abs_cs[i] = self.exc.abs(sig)

        self._sca = sca
        self._abs = abs_cs
        self._ext = ext

        return sca, ext

    @property
    def sca(self) -> np.ndarray:
        """Scattering cross section."""
        if self._sca is None:
            self.compute()
        return self._sca

    @property
    def ext(self) -> np.ndarray:
        """Extinction cross section."""
        if self._ext is None:
            self.compute()
        return self._ext

    @property
    def abs(self) -> np.ndarray:
        """Absorption cross section."""
        if self._abs is None:
            self.compute()
        return self._abs

    def __repr__(self) -> str:
        return f"SpectrumStat(n_wavelengths={self.n_wavelengths})"
