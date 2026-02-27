"""
Mie-Gans theory for ellipsoidal particles (quasistatic approximation).

MATLAB reference: Mie/@miegans/
"""

import numpy as np
from scipy.integrate import quad


def _get_eps_value(eps_func, enei):
    """Extract dielectric constant from eps_func(enei).

    Handles both:
    - Functions returning scalar/array eps value
    - Functions returning (eps, k) tuple
    """
    result = eps_func(enei)
    if isinstance(result, tuple):
        return result[0]
    return result


class MieGans:
    """Mie-Gans theory for ellipsoidal particle (quasistatic approximation).

    MATLAB: @miegans

    Parameters
    ----------
    epsin : callable
        Dielectric function inside ellipsoid. epsin(enei) -> eps or (eps, k).
    epsout : callable
        Dielectric function outside ellipsoid. epsout(enei) -> eps or (eps, k).
    ax : ndarray, shape (3,)
        Ellipsoid semi-axis diameters (full axes, not semi-axes) in nm.
    """

    def __init__(self, epsin, epsout, ax):
        self.epsin = epsin
        self.epsout = epsout
        self.ax = np.asarray(ax, dtype=float)
        self._compute_depolarization()

    def _compute_depolarization(self):
        """Compute depolarization factors L1, L2, L3.

        MATLAB: @miegans/init.m
        van de Hulst, Sec. 6.32
        """
        a, b, c = self.ax / 2  # semi-axes

        def integrand(s, ax_i):
            return (a * b * c / 2
                    / ((s + ax_i**2)**1.5
                       * np.sqrt((s + a**2) * (s + b**2) * (s + c**2))
                       / np.sqrt(s + ax_i**2)))

        # Simpler form of integrand for each axis
        def f1(s):
            return a * b * c / 2 / ((s + a**2)**1.5 * np.sqrt(s + b**2) * np.sqrt(s + c**2))

        def f2(s):
            return a * b * c / 2 / (np.sqrt(s + a**2) * (s + b**2)**1.5 * np.sqrt(s + c**2))

        def f3(s):
            return a * b * c / 2 / (np.sqrt(s + a**2) * np.sqrt(s + b**2) * (s + c**2)**1.5)

        upper = 1e5 * max(self.ax)
        self._L1, _ = quad(f1, 0, upper, limit=200)
        self._L2, _ = quad(f2, 0, upper, limit=200)
        self._L3, _ = quad(f3, 0, upper, limit=200)

    def _polarizabilities(self, enei):
        """Compute per-axis polarizabilities.

        MATLAB: @miegans/scattering.m lines 14-21
        """
        epsb = _get_eps_value(self.epsout, np.array([0.0]))
        if isinstance(epsb, np.ndarray):
            epsb = complex(epsb.ravel()[0])
        epsi = _get_eps_value(self.epsin, enei)
        epsz = epsi / epsb

        vol = 4 * np.pi / 3 * np.prod(self.ax / 2)
        a1 = vol / (4 * np.pi) / (self._L1 + 1.0 / (epsz - 1))
        a2 = vol / (4 * np.pi) / (self._L2 + 1.0 / (epsz - 1))
        a3 = vol / (4 * np.pi) / (self._L3 + 1.0 / (epsz - 1))

        nb = np.sqrt(epsb)
        k = 2 * np.pi / enei * nb
        return a1, a2, a3, k

    def extinction(self, enei, pol):
        """Extinction cross section.

        MATLAB: @miegans/extinction.m
        """
        return self.scattering(enei, pol) + self.absorption(enei, pol)

    def scattering(self, enei, pol):
        """Scattering cross section.

        MATLAB: @miegans/scattering.m
        """
        enei = np.asarray(enei, dtype=float)
        pol = np.asarray(pol, dtype=float)
        a1, a2, a3, k = self._polarizabilities(enei)
        sca = (8 * np.pi / 3 * k**4
               * (np.abs(a1 * pol[0])**2
                  + np.abs(a2 * pol[1])**2
                  + np.abs(a3 * pol[2])**2))
        return np.real(sca)

    def absorption(self, enei, pol):
        """Absorption cross section.

        MATLAB: @miegans/absorption.m
        """
        enei = np.asarray(enei, dtype=float)
        pol = np.asarray(pol, dtype=float)
        a1, a2, a3, k = self._polarizabilities(enei)
        abso = 4 * np.pi * k * np.imag(a1 * pol[0] + a2 * pol[1] + a3 * pol[2])
        return np.real(abso)

    def __repr__(self):
        return "MieGans(ax={})".format(self.ax.tolist())
