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
            # Handle list of tabs (from tabspace with particle argument)
            if isinstance(tab, list):
                tab = self._merge_tabs(tab)
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

        # Per-component caches
        self._enei_comp = None
        self._Gsav_comp = None
        self._Frsav_comp = None
        self._Fzsav_comp = None

    @staticmethod
    def _merge_tabs(tabs):
        """Merge a list of tabspace dicts into a single dict.

        When tabspace is called with a particle, it returns a list of dicts
        (one per layer combination). This merges them by taking the union
        of all r, z1, z2 grids.
        """
        if len(tabs) == 1:
            return tabs[0]

        # Merge grids by taking the union
        all_r = np.concatenate([t['r'] for t in tabs])
        all_z1 = np.concatenate([np.atleast_1d(t['z1']) for t in tabs])
        all_z2 = np.concatenate([np.atleast_1d(t['z2']) for t in tabs])

        r = np.sort(np.unique(all_r))
        z1 = np.sort(np.unique(all_z1))
        z2 = np.sort(np.unique(all_z2))

        return {'r': r, 'z1': z1, 'z2': z2}

    def set(self, enei_arr, **options):
        """Pre-compute Green function table at multiple wavelengths.

        MATLAB: @compgreentablayer/set.m
        Stores 4D arrays (nr, nz1, nz2, n_enei) for interpolation.
        """
        enei_arr = np.atleast_1d(np.asarray(enei_arr, dtype=float))
        n_enei = len(enei_arr)
        nr = len(self.r)
        nz1 = len(self.z1)
        nz2 = len(self.z2)

        # Determine component names
        r_sample = self.r[:1]
        z1_sample = np.full_like(r_sample, self.z1[0])
        z2_sample = np.full_like(r_sample, self.z2[0])
        result = self.layer.green(enei_arr[0], r_sample, z1_sample, z2_sample)
        names = list(result[0].keys())

        # 4D arrays: (nr, nz1, nz2, n_enei)
        self._Gsav_multi = {k: np.zeros((nr, nz1, nz2, n_enei), dtype=complex) for k in names}
        self._Frsav_multi = {k: np.zeros((nr, nz1, nz2, n_enei), dtype=complex) for k in names}
        self._Fzsav_multi = {k: np.zeros((nr, nz1, nz2, n_enei), dtype=complex) for k in names}

        for ie, enei in enumerate(enei_arr):
            for iz1 in range(nz1):
                for iz2 in range(nz2):
                    r_vec = self.r
                    z1_vec = np.full_like(r_vec, self.z1[iz1])
                    z2_vec = np.full_like(r_vec, self.z2[iz2])
                    result = self.layer.green(enei, r_vec, z1_vec, z2_vec)
                    for name in names:
                        self._Gsav_multi[name][:, iz1, iz2, ie] = np.asarray(result[0][name], dtype=complex)
                        self._Frsav_multi[name][:, iz1, iz2, ie] = np.asarray(result[1][name], dtype=complex)
                        self._Fzsav_multi[name][:, iz1, iz2, ie] = np.asarray(result[2][name], dtype=complex)

        self._enei_tab = enei_arr
        return self

    def _interp_wavelength(self, enei):
        """Interpolate 4D multi-wavelength table to 3D at given wavelength."""
        enei_arr = self._enei_tab
        idx = np.searchsorted(enei_arr, enei, side='right') - 1
        idx = np.clip(idx, 0, len(enei_arr) - 2)
        frac = (enei - enei_arr[idx]) / (enei_arr[idx + 1] - enei_arr[idx])

        names = list(self._Gsav_multi.keys())
        self._Gsav_comp = {}
        self._Frsav_comp = {}
        self._Fzsav_comp = {}
        for name in names:
            self._Gsav_comp[name] = (1 - frac) * self._Gsav_multi[name][:, :, :, idx] + frac * self._Gsav_multi[name][:, :, :, idx + 1]
            self._Frsav_comp[name] = (1 - frac) * self._Frsav_multi[name][:, :, :, idx] + frac * self._Frsav_multi[name][:, :, :, idx + 1]
            self._Fzsav_comp[name] = (1 - frac) * self._Fzsav_multi[name][:, :, :, idx] + frac * self._Fzsav_multi[name][:, :, :, idx + 1]
        self._enei_comp = enei

    def eval(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.r is not None:
            return self._interp(enei, r, z1, z2)
        else:
            return self._compute(enei, r, z1, z2)

    def _compute(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        result = self.layer.green(enei, r, z1, z2)
        # layer.green() returns (G_dict, Fr_dict, Fz_dict, pos_dict)
        # where G, Fr, Fz are dicts keyed by reflection names.
        # Sum all components to obtain the total reflected Green function.
        G_dict, Fr_dict, Fz_dict = result[0], result[1], result[2]
        G = self._sum_components(G_dict)
        Fr = self._sum_components(Fr_dict)
        Fz = self._sum_components(Fz_dict)
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

        shape = np.asarray(r).shape

        # Compute tabulated values if not cached at this wavelength
        if self._Gsav is None or (self.enei is not None and not np.isclose(self.enei, enei)):
            self._compute_tab(enei)

        # Query points (clip to grid bounds)
        r_q = np.clip(np.asarray(r, dtype = float).ravel(), self.r[0], self.r[-1])
        z1_q = np.clip(np.asarray(z1, dtype = float).ravel(), self.z1[0], self.z1[-1])
        z2_q = np.clip(np.asarray(z2, dtype = float).ravel(), self.z2[0], self.z2[-1])
        points = np.column_stack([r_q, z1_q, z2_q])

        grid = (self.r, self.z1, self.z2)

        G = RegularGridInterpolator(grid, self._Gsav.real, method = 'linear', bounds_error = False, fill_value = None)(points) \
          + 1j * RegularGridInterpolator(grid, self._Gsav.imag, method = 'linear', bounds_error = False, fill_value = None)(points)
        Fr = RegularGridInterpolator(grid, self._Frsav.real, method = 'linear', bounds_error = False, fill_value = None)(points) \
           + 1j * RegularGridInterpolator(grid, self._Frsav.imag, method = 'linear', bounds_error = False, fill_value = None)(points)
        Fz = RegularGridInterpolator(grid, self._Fzsav.real, method = 'linear', bounds_error = False, fill_value = None)(points) \
           + 1j * RegularGridInterpolator(grid, self._Fzsav.imag, method = 'linear', bounds_error = False, fill_value = None)(points)

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
                result = self.layer.green(enei, r_vec, z1_vec, z2_vec)
                G = self._sum_components(result[0])
                Fr = self._sum_components(result[1])
                Fz = self._sum_components(result[2])
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

    def eval_components(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Evaluate reflected Green function preserving per-component structure.

        Returns (G_dict, Fr_dict, Fz_dict) where each is a dict keyed by
        reflection names ('p', 'ss', 'hh', 'sh', 'hs').
        """
        if self.r is not None:
            return self._interp_components(enei, r, z1, z2)
        else:
            return self._compute_components(enei, r, z1, z2)

    def _compute_components(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        result = self.layer.green(enei, r, z1, z2)
        G_dict = {k: np.asarray(v, dtype=complex) for k, v in result[0].items()}
        Fr_dict = {k: np.asarray(v, dtype=complex) for k, v in result[1].items()}
        Fz_dict = {k: np.asarray(v, dtype=complex) for k, v in result[2].items()}
        return G_dict, Fr_dict, Fz_dict

    def _interp_components(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        shape = np.asarray(r).shape

        if hasattr(self, '_Gsav_multi') and self._Gsav_multi is not None:
            self._interp_wavelength(enei)
        elif self._Gsav_comp is None or (
                self._enei_comp is not None and not np.isclose(self._enei_comp, enei)):
            self._compute_tab_components(enei)

        # Query points (clip to grid bounds)
        r_q = np.clip(np.asarray(r, dtype = float).ravel(), self.r[0], self.r[-1])
        z1_q = np.clip(np.asarray(z1, dtype = float).ravel(), self.z1[0], self.z1[-1])
        z2_q = np.clip(np.asarray(z2, dtype = float).ravel(), self.z2[0], self.z2[-1])
        points = np.column_stack([r_q, z1_q, z2_q])

        grid = (self.r, self.z1, self.z2)
        names = list(self._Gsav_comp.keys())
        G_dict = {}
        Fr_dict = {}
        Fz_dict = {}

        for name in names:
            Gsav = self._Gsav_comp[name]
            Frsav = self._Frsav_comp[name]
            Fzsav = self._Fzsav_comp[name]

            G_arr = RegularGridInterpolator(grid, Gsav.real, method = 'linear', bounds_error = False, fill_value = None)(points) \
                  + 1j * RegularGridInterpolator(grid, Gsav.imag, method = 'linear', bounds_error = False, fill_value = None)(points)
            Fr_arr = RegularGridInterpolator(grid, Frsav.real, method = 'linear', bounds_error = False, fill_value = None)(points) \
                   + 1j * RegularGridInterpolator(grid, Frsav.imag, method = 'linear', bounds_error = False, fill_value = None)(points)
            Fz_arr = RegularGridInterpolator(grid, Fzsav.real, method = 'linear', bounds_error = False, fill_value = None)(points) \
                   + 1j * RegularGridInterpolator(grid, Fzsav.imag, method = 'linear', bounds_error = False, fill_value = None)(points)

            G_dict[name] = G_arr.reshape(shape)
            Fr_dict[name] = Fr_arr.reshape(shape)
            Fz_dict[name] = Fz_arr.reshape(shape)

        return G_dict, Fr_dict, Fz_dict

    def _compute_tab_components(self,
            enei: float) -> None:

        nr = len(self.r)
        nz1 = len(self.z1)
        nz2 = len(self.z2)

        # Determine component names from a sample call
        r_sample = self.r[:1]
        z1_sample = np.full_like(r_sample, self.z1[0])
        z2_sample = np.full_like(r_sample, self.z2[0])
        result = self.layer.green(enei, r_sample, z1_sample, z2_sample)
        names = list(result[0].keys())

        self._Gsav_comp = {k: np.zeros((nr, nz1, nz2), dtype=complex) for k in names}
        self._Frsav_comp = {k: np.zeros((nr, nz1, nz2), dtype=complex) for k in names}
        self._Fzsav_comp = {k: np.zeros((nr, nz1, nz2), dtype=complex) for k in names}

        for iz1 in range(nz1):
            for iz2 in range(nz2):
                r_vec = self.r
                z1_vec = np.full_like(r_vec, self.z1[iz1])
                z2_vec = np.full_like(r_vec, self.z2[iz2])
                result = self.layer.green(enei, r_vec, z1_vec, z2_vec)
                for name in names:
                    self._Gsav_comp[name][:, iz1, iz2] = np.asarray(result[0][name], dtype=complex)
                    self._Frsav_comp[name][:, iz1, iz2] = np.asarray(result[1][name], dtype=complex)
                    self._Fzsav_comp[name][:, iz1, iz2] = np.asarray(result[2][name], dtype=complex)

        self._enei_comp = enei

    def setup_grid(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> None:

        self.r = np.asarray(r, dtype = float)
        self.z1 = np.asarray(z1, dtype = float)
        self.z2 = np.asarray(z2, dtype = float)
        # Invalidate caches
        self._Gsav = None
        self._Frsav = None
        self._Fzsav = None
        self._Gsav_comp = None
        self._Frsav_comp = None
        self._Fzsav_comp = None
        self.enei = None
        self._enei_comp = None

    @staticmethod
    def _sum_components(d):
        """Sum all reflection-component arrays stored in a dict.

        layer.green() returns dicts keyed by reflection names
        ('p', 'ss', 'hs', 'sh', 'hh').  This helper adds them together
        to yield the total reflected Green function as a single array.

        If *d* is already an ndarray (not a dict) it is returned as-is.
        """
        if isinstance(d, dict):
            total = None
            for v in d.values():
                if total is None:
                    total = np.array(v, dtype = complex)
                else:
                    total = total + np.array(v, dtype = complex)
            return total if total is not None else np.zeros(0, dtype = complex)
        return np.asarray(d, dtype = complex)

    def __repr__(self) -> str:
        r_info = 'nr={}'.format(len(self.r)) if self.r is not None else 'no table'
        return 'GreenTabLayer({})'.format(r_info)
