import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class GreenTabLayer(object):

    name = 'greentablayer'

    def __init__(self,
            layer: Any,
            tab: Optional[Dict[str, Any]] = None,
            **options: Any) -> None:

        self.layer = layer

        if tab is not None:
            self.r = tab.get('r', None)
            self.z1 = tab.get('z1', None)
            self.z2 = tab.get('z2', None)
            self._Gsav = tab.get('Gsav', None)
            self._Frsav = tab.get('Frsav', None)
            self._Fzsav = tab.get('Fzsav', None)
        else:
            self.r = None
            self.z1 = None
            self.z2 = None
            self._Gsav = None
            self._Frsav = None
            self._Fzsav = None

        self.enei = None
        self.G = None
        self.Fr = None
        self.Fz = None

    def eval(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self._Gsav is not None and self.r is not None:
            return self._interp(enei, r, z1, z2)
        else:
            return self._compute(enei, r, z1, z2)

    def _compute(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        G, Fr, Fz = self.layer.green(enei, r, z1, z2)
        self.G = G
        self.Fr = Fr
        self.Fz = Fz
        self.enei = enei
        return G, Fr, Fz

    def _interp(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.enei is not None and np.isclose(self.enei, enei):
            if self.G is not None:
                return self.G, self.Fr, self.Fz

        r = np.asarray(r, dtype = float)
        z1 = np.asarray(z1, dtype = float)
        z2 = np.asarray(z2, dtype = float)

        shape = r.shape
        r_flat = r.ravel()
        z1_flat = z1.ravel()
        z2_flat = z2.ravel()

        n = len(r_flat)

        G = np.zeros(n, dtype = complex)
        Fr = np.zeros(n, dtype = complex)
        Fz = np.zeros(n, dtype = complex)

        # Compute tabulated values if not cached at this wavelength
        if self._Gsav is None or not np.isclose(self.enei, enei) if self.enei is not None else True:
            self._compute_tab(enei)

        # Interpolate from tabulated values
        for i in range(n):
            G[i], Fr[i], Fz[i] = self._interp_single(
                r_flat[i], z1_flat[i], z2_flat[i])

        G = G.reshape(shape)
        Fr = Fr.reshape(shape)
        Fz = Fz.reshape(shape)

        self.G = G
        self.Fr = Fr
        self.Fz = Fz
        self.enei = enei

        return G, Fr, Fz

    def _compute_tab(self,
            enei: float) -> None:

        nr = len(self.r)
        nz1 = len(self.z1)
        nz2 = len(self.z2)

        self._Gsav = np.zeros((nr, nz1, nz2), dtype = complex)
        self._Frsav = np.zeros((nr, nz1, nz2), dtype = complex)
        self._Fzsav = np.zeros((nr, nz1, nz2), dtype = complex)

        for iz1 in range(nz1):
            for iz2 in range(nz2):
                r_vec = self.r
                z1_vec = np.full_like(r_vec, self.z1[iz1])
                z2_vec = np.full_like(r_vec, self.z2[iz2])
                G, Fr, Fz = self.layer.green(enei, r_vec, z1_vec, z2_vec)
                self._Gsav[:, iz1, iz2] = G
                self._Frsav[:, iz1, iz2] = Fr
                self._Fzsav[:, iz1, iz2] = Fz

        self.enei = enei

    def _interp_single(self,
            r: float,
            z1: float,
            z2: float) -> Tuple[complex, complex, complex]:

        r_idx = np.interp(r, self.r, np.arange(len(self.r)))
        z1_idx = np.interp(z1, self.z1, np.arange(len(self.z1)))
        z2_idx = np.interp(z2, self.z2, np.arange(len(self.z2)))

        ir = int(np.clip(r_idx, 0, len(self.r) - 2))
        iz1 = int(np.clip(z1_idx, 0, len(self.z1) - 2))
        iz2 = int(np.clip(z2_idx, 0, len(self.z2) - 2))

        fr = r_idx - ir
        fz1 = z1_idx - iz1
        fz2 = z2_idx - iz2

        G = self._trilinear_interp(self._Gsav, ir, iz1, iz2, fr, fz1, fz2)
        Fr = self._trilinear_interp(self._Frsav, ir, iz1, iz2, fr, fz1, fz2)
        Fz = self._trilinear_interp(self._Fzsav, ir, iz1, iz2, fr, fz1, fz2)

        return G, Fr, Fz

    def _trilinear_interp(self,
            data: np.ndarray,
            ir: int,
            iz1: int,
            iz2: int,
            fr: float,
            fz1: float,
            fz2: float) -> complex:

        c000 = data[ir, iz1, iz2]
        c100 = data[ir + 1, iz1, iz2]
        c010 = data[ir, iz1 + 1, iz2]
        c110 = data[ir + 1, iz1 + 1, iz2]
        c001 = data[ir, iz1, iz2 + 1]
        c101 = data[ir + 1, iz1, iz2 + 1]
        c011 = data[ir, iz1 + 1, iz2 + 1]
        c111 = data[ir + 1, iz1 + 1, iz2 + 1]

        c00 = c000 * (1 - fr) + c100 * fr
        c10 = c010 * (1 - fr) + c110 * fr
        c01 = c001 * (1 - fr) + c101 * fr
        c11 = c011 * (1 - fr) + c111 * fr

        c0 = c00 * (1 - fz1) + c10 * fz1
        c1 = c01 * (1 - fz1) + c11 * fz1

        return c0 * (1 - fz2) + c1 * fz2

    def __repr__(self) -> str:
        r_info = 'nr={}'.format(len(self.r)) if self.r is not None else 'no table'
        return 'GreenTabLayer({})'.format(r_info)
