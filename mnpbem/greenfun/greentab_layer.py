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
        self._pos = None

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
        names = list(self._Gsav_multi.keys())
        self._Gsav_comp = {}
        self._Frsav_comp = {}
        self._Fzsav_comp = {}

        if len(enei_arr) == 1:
            # Single wavelength — no interpolation needed
            for name in names:
                self._Gsav_comp[name] = self._Gsav_multi[name][:, :, :, 0]
                self._Frsav_comp[name] = self._Frsav_multi[name][:, :, :, 0]
                self._Fzsav_comp[name] = self._Fzsav_multi[name][:, :, :, 0]
        else:
            idx = np.searchsorted(enei_arr, enei, side='right') - 1
            idx = np.clip(idx, 0, len(enei_arr) - 2)
            frac = (enei - enei_arr[idx]) / (enei_arr[idx + 1] - enei_arr[idx])
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
        self._pos = result[3]
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

        # Query points
        r_q = np.clip(np.asarray(r, dtype=float).ravel(), self.r[0], self.r[-1])
        z1_q = np.asarray(z1, dtype=float).ravel()
        z2_q = np.asarray(z2, dtype=float).ravel()

        if len(self.z2) == 1:
            # Single-z2: fold z2 into z1 via mindist
            z2_ref = self.z2[0]
            mindist_z2, _ = self.layer.mindist(z2_q)
            if z2_ref >= self.layer.z[0]:
                z1_eff = np.clip(z1_q + mindist_z2, self.z1[0], self.z1[-1])
            else:
                z1_eff = np.clip(z1_q - mindist_z2, self.z1[0], self.z1[-1])
            points = np.column_stack([r_q, z1_eff])
            grid = (self.r, self.z1)
            Gsav = self._Gsav[:, :, 0]
            Frsav = self._Frsav[:, :, 0]
            Fzsav = self._Fzsav[:, :, 0]
        else:
            z1_q = np.clip(z1_q, self.z1[0], self.z1[-1])
            z2_q = np.clip(z2_q, self.z2[0], self.z2[-1])
            points = np.column_stack([r_q, z1_q, z2_q])
            grid = (self.r, self.z1, self.z2)
            Gsav = self._Gsav
            Frsav = self._Frsav
            Fzsav = self._Fzsav

        G = RegularGridInterpolator(grid, Gsav.real, method='linear', bounds_error=False, fill_value=None)(points) \
          + 1j * RegularGridInterpolator(grid, Gsav.imag, method='linear', bounds_error=False, fill_value=None)(points)
        Fr = RegularGridInterpolator(grid, Frsav.real, method='linear', bounds_error=False, fill_value=None)(points) \
           + 1j * RegularGridInterpolator(grid, Frsav.imag, method='linear', bounds_error=False, fill_value=None)(points)
        Fz = RegularGridInterpolator(grid, Fzsav.real, method='linear', bounds_error=False, fill_value=None)(points) \
           + 1j * RegularGridInterpolator(grid, Fzsav.imag, method='linear', bounds_error=False, fill_value=None)(points)

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
        self._pos = result[3]
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

        # Query points
        r_q = np.clip(np.asarray(r, dtype=float).ravel(), self.r[0], self.r[-1])
        z1_q = np.asarray(z1, dtype=float).ravel()
        z2_q = np.asarray(z2, dtype=float).ravel()

        names = list(self._Gsav_comp.keys())
        G_dict = {}
        Fr_dict = {}
        Fz_dict = {}

        # Single-z2 case: MATLAB uses mindist to fold z2 into z1
        # tabspace sets z2 = layer.z[0]+1e-10 for uppermost layer
        if len(self.z2) == 1:
            # Fold: z1_eff = z1 + mindist(z2) for uppermost layer
            # or z1_eff = z1 - mindist(z2) for lowermost layer
            z2_ref = self.z2[0]
            mindist_z2, _ = self.layer.mindist(z2_q)
            if z2_ref >= self.layer.z[0]:
                # Uppermost layer: z1_eff = z1 + mindist(z2)
                z1_eff = np.clip(z1_q + mindist_z2, self.z1[0], self.z1[-1])
            else:
                # Lowermost layer
                z1_eff = np.clip(z1_q - mindist_z2, self.z1[0], self.z1[-1])

            # 2D interpolation (r, z1_eff) — squeeze out z2 dimension
            points_2d = np.column_stack([r_q, z1_eff])
            grid_2d = (self.r, self.z1)

            for name in names:
                Gsav = self._Gsav_comp[name][:, :, 0]  # (nr, nz1)
                Frsav = self._Frsav_comp[name][:, :, 0]
                Fzsav = self._Fzsav_comp[name][:, :, 0]

                G_arr = RegularGridInterpolator(grid_2d, Gsav.real, method='linear', bounds_error=False, fill_value=None)(points_2d) \
                      + 1j * RegularGridInterpolator(grid_2d, Gsav.imag, method='linear', bounds_error=False, fill_value=None)(points_2d)
                Fr_arr = RegularGridInterpolator(grid_2d, Frsav.real, method='linear', bounds_error=False, fill_value=None)(points_2d) \
                       + 1j * RegularGridInterpolator(grid_2d, Frsav.imag, method='linear', bounds_error=False, fill_value=None)(points_2d)
                Fz_arr = RegularGridInterpolator(grid_2d, Fzsav.real, method='linear', bounds_error=False, fill_value=None)(points_2d) \
                       + 1j * RegularGridInterpolator(grid_2d, Fzsav.imag, method='linear', bounds_error=False, fill_value=None)(points_2d)

                G_dict[name] = G_arr.reshape(shape)
                Fr_dict[name] = Fr_arr.reshape(shape)
                Fz_dict[name] = Fz_arr.reshape(shape)
        else:
            # Standard 3D interpolation
            z1_q = np.clip(z1_q, self.z1[0], self.z1[-1])
            z2_q = np.clip(z2_q, self.z2[0], self.z2[-1])
            points = np.column_stack([r_q, z1_q, z2_q])
            grid = (self.r, self.z1, self.z2)

            for name in names:
                Gsav = self._Gsav_comp[name]
                Frsav = self._Frsav_comp[name]
                Fzsav = self._Fzsav_comp[name]

                G_arr = RegularGridInterpolator(grid, Gsav.real, method='linear', bounds_error=False, fill_value=None)(points) \
                      + 1j * RegularGridInterpolator(grid, Gsav.imag, method='linear', bounds_error=False, fill_value=None)(points)
                Fr_arr = RegularGridInterpolator(grid, Frsav.real, method='linear', bounds_error=False, fill_value=None)(points) \
                       + 1j * RegularGridInterpolator(grid, Frsav.imag, method='linear', bounds_error=False, fill_value=None)(points)
                Fz_arr = RegularGridInterpolator(grid, Fzsav.real, method='linear', bounds_error=False, fill_value=None)(points) \
                       + 1j * RegularGridInterpolator(grid, Fzsav.imag, method='linear', bounds_error=False, fill_value=None)(points)

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

    def norm(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # MATLAB: @greentablayer/norm.m
        # Multiply Green function with distance-dependent normalization factors.
        assert self.G is not None

        # Tabulated radii and minimal distances from pos
        r = self._pos['r']
        zmin = self._pos['zmin']
        # Distance
        d = np.sqrt(r ** 2 + zmin ** 2)

        G_out = {}
        Fr_out = {}
        Fz_out = {}
        for name in self.G.keys():
            G_out[name] = self.G[name] * d
            Fr_out[name] = self.Fr[name] * d ** 3 / r
            Fz_out[name] = self.Fz[name] * d ** 3 / zmin

        return G_out, Fr_out, Fz_out

    def inside(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: Optional[np.ndarray] = None) -> np.ndarray:
        # MATLAB: @greentablayer/inside.m
        layer = self.layer

        r = np.asarray(r, dtype = float)
        z1 = np.asarray(z1, dtype = float)

        # Round radii and z-values
        r = np.maximum(layer.rmin, r)
        if z2 is not None:
            z2 = np.asarray(z2, dtype = float)
            z1, z2 = layer.round_z(z1, z2)
        else:
            z1, = layer.round_z(z1)

        def fun(x: np.ndarray, limits: np.ndarray) -> np.ndarray:
            return (x >= np.min(limits)) & (x <= np.max(limits))

        # Uppermost or lowermost layer (single z2 value in table)
        if np.atleast_1d(self.z2).size == 1:
            ind1, _ = layer.indlayer(z1)
            ind2, _ = layer.indlayer(z2)

            # Find z-values in uppermost or lowermost layer
            in1 = (ind1 == ind2) & (ind1 == 1)
            in2 = (ind1 == ind2) & (ind1 == layer.n + 1)
            result = in1 | in2

            if np.any(in1):
                mindist_z2, _ = layer.mindist(z2[in1])
                result[in1] = fun(r[in1], self.r) & fun(z1[in1] + mindist_z2, self.z1)
            if np.any(in2):
                mindist_z2, _ = layer.mindist(z2[in2])
                result[in2] = fun(r[in2], self.r) & fun(z1[in2] - mindist_z2, self.z1)
        else:
            result = fun(r, self.r) & fun(z1, self.z1) & fun(z2, self.z2)

        return result

    def ismember(self,
            layer: Any,
            enei: Optional[np.ndarray] = None) -> bool:
        # MATLAB: @greentablayer/ismember.m
        # Check if precomputed table is compatible with given layer and enei.

        # enei not set
        if self.enei is None:
            return False

        # Check wavelength range
        enei_tab = np.atleast_1d(self.enei) if not hasattr(self, '_enei_tab') else np.atleast_1d(self._enei_tab)
        if enei is not None:
            enei = np.atleast_1d(enei)
            if np.min(enei) < np.min(enei_tab) or np.max(enei) > np.max(enei_tab):
                return False

        # Check layer structure compatibility
        if layer.n != self.layer.n:
            return False
        if not np.allclose(layer.z, self.layer.z):
            return False

        # Evaluate dielectric functions and compare
        for eps_new, eps_old in zip(layer.eps, self.layer.eps):
            for e in enei_tab:
                val_new = eps_new(e)
                val_old = eps_old(e)
                # eps functions return (eps, k) tuples
                if isinstance(val_new, tuple):
                    val_new = val_new[0]
                if isinstance(val_old, tuple):
                    val_old = val_old[0]
                if abs(val_new - val_old) > 1e-8:
                    return False

        return True

    def parset(self,
            enei_arr: np.ndarray,
            **options: Any) -> 'GreenTabLayer':
        # MATLAB: @greentablayer/parset.m
        # Same as set() but with parallel computation.
        # Python: sequential fallback (green() may not be thread-safe).
        enei_arr = np.atleast_1d(np.asarray(enei_arr, dtype = float))
        n_enei = len(enei_arr)
        nr = len(self.r)
        nz1 = len(self.z1)
        nz2 = len(self.z2)

        pos_saved = None

        for ien in range(n_enei):
            # Determine component names from a sample call on the first iteration
            if ien == 0:
                r_sample = self.r[:1]
                z1_sample = np.full_like(r_sample, self.z1[0])
                z2_sample = np.full_like(r_sample, self.z2[0])
                result_sample = self.layer.green(enei_arr[0], r_sample, z1_sample, z2_sample)
                names = list(result_sample[0].keys())

                siz = (n_enei, nr, nz1, nz2)
                Gsav = {k: np.zeros(siz, dtype = complex) for k in names}
                Frsav = {k: np.zeros(siz, dtype = complex) for k in names}
                Fzsav = {k: np.zeros(siz, dtype = complex) for k in names}

            for iz1 in range(nz1):
                for iz2 in range(nz2):
                    r_vec = self.r
                    z1_vec = np.full_like(r_vec, self.z1[iz1])
                    z2_vec = np.full_like(r_vec, self.z2[iz2])
                    result = self.layer.green(enei_arr[ien], r_vec, z1_vec, z2_vec)
                    if pos_saved is None:
                        pos_saved = result[3]
                    for name in names:
                        Gsav[name][ien, :, iz1, iz2] = np.asarray(result[0][name], dtype = complex)
                        Frsav[name][ien, :, iz1, iz2] = np.asarray(result[1][name], dtype = complex)
                        Fzsav[name][ien, :, iz1, iz2] = np.asarray(result[2][name], dtype = complex)

        # Store results (MATLAB stores as Gsav, Frsav, Fzsav with shape [n_enei, ...])
        self.enei = enei_arr
        self._pos = pos_saved
        self._Gsav_multi = Gsav
        self._Frsav_multi = Frsav
        self._Fzsav_multi = Fzsav
        self._enei_tab = enei_arr

        return self

    def __repr__(self) -> str:
        r_info = 'nr={}'.format(len(self.r)) if self.r is not None else 'no table'
        return 'GreenTabLayer({})'.format(r_info)
