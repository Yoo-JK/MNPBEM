"""
Dielectric layer structure for stratified media.

MATLAB: Particles/@layerstructure/

Implements dielectric layer structures for BEM simulations with
substrates and multilayer systems. Provides Fresnel coefficients,
reflected Green functions, and BEM equation solvers for layer systems.

Reference:
    M. Paulus et al., PRE 62, 5797 (2000)
    Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
"""

import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.special import jv as besselj
from scipy.special import hankel1
from scipy.integrate import solve_ivp


class LayerStructure(object):
    """
    Dielectric layer structure for stratified media.

    MATLAB: @layerstructure

    The outer surface normals of the layers must point upwards.
    Geometry:

                  eps[0]
        --------------------------  z[0]
                  eps[1]
        --------------------------  z[1]
                   ...
        --------------------------  z[-1]
                  eps[-1]

    Properties
    ----------
    eps : list
        Dielectric functions for each layer
    z : ndarray
        z-positions of layer interfaces
    ind : ndarray
        Index to table of dielectric functions
    ztol : float
        Tolerance for detecting points in layer (default: 2e-2)
    rmin : float
        Minimum radial distance for Green function (default: 1e-2)
    zmin : float
        Minimum distance to layer for Green function (default: 1e-2)
    semi : float
        Imaginary part of semiellipse for complex integration (default: 0.1)
    ratio : float
        z:r ratio which determines integration path (default: 2)

    Methods
    -------
    fresnel(enei, kpar, pos)
        Fresnel reflection/transmission coefficients for potentials
    efresnel(pol, dir, enei)
        Reflected and transmitted electric fields for plane wave
    green(enei, r, z1, z2)
        Reflected Green function via complex integration
    reflection(enei, kpar, pos)
        Reflection coefficients for surface charges and currents
    bemsolve(enei, kpar)
        Solve BEM equations for layer structure
    indlayer(z)
        Find layer index for given z-values
    mindist(z)
        Minimal distance of z-values to layer boundaries
    round_z(*z_args)
        Round z-values to achieve minimal distance to layers
    tabspace(...)
        Generate grids for tabulated Green functions

    Examples
    --------
    >>> from mnpbem import EpsConst
    >>> from mnpbem.geometry import LayerStructure
    >>>
    >>> # Single interface (substrate)
    >>> eps_tab = [EpsConst(1.0), EpsConst(2.25)]
    >>> layer = LayerStructure(eps_tab, [1, 2], [0.0])
    """

    def __init__(self,
            epstab: list,
            ind: Union[list, np.ndarray],
            z: Union[list, np.ndarray],
            **options: Any) -> None:

        ind = np.asarray(ind, dtype = int)
        z = np.asarray(z, dtype = float)

        # eps is indexed from epstab using ind (MATLAB 1-indexed -> Python 0-indexed)
        self.eps = [epstab[i - 1] for i in ind]
        self.ind = ind
        self.z = z

        # Default options for complex integration
        self.ztol = options.get('ztol', 2e-2)
        self.rmin = options.get('rmin', 1e-2)
        self.zmin = options.get('zmin', 1e-2)
        self.semi = options.get('semi', 0.1)
        self.ratio = options.get('ratio', 2.0)

        # ODE integration options
        self.atol = options.get('atol', 1e-6)
        self.initial_step = options.get('initial_step', 1e-3)

    @property
    def n(self) -> int:
        return len(self.z)

    def indlayer(self,
            z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # MATLAB: layers are ordered with decreasing z-values
        # [~, ind] = histc(-z, [-inf, -obj.z, inf])
        z = np.asarray(z, dtype = float)
        shape = z.shape
        z_flat = z.ravel()

        # Bin edges: [-inf, -z[0], -z[1], ..., inf] for -z
        edges = np.empty(len(self.z) + 2, dtype = float)
        edges[0] = -np.inf
        edges[1:-1] = -self.z
        edges[-1] = np.inf

        ind = np.digitize(-z_flat, edges)
        # digitize returns 1-based bin indices, we keep them 1-based for MATLAB compat
        ind = ind.reshape(shape)

        # Is point located in layer?
        zmin_vals, _ = self.mindist(z)
        in_layer = np.abs(zmin_vals) < self.ztol

        return ind, in_layer

    def mindist(self,
            z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype = float)
        shape = z.shape
        z_flat = z.ravel()

        # Compute distance to each interface
        # z_flat: (n,), self.z: (m,) -> diff: (n, m)
        diff = z_flat[:, np.newaxis] - self.z[np.newaxis, :]

        # Find minimum absolute distance
        abs_diff = np.abs(diff)
        ind = np.argmin(abs_diff, axis = 1)
        zmin = abs_diff[np.arange(len(z_flat)), ind]

        # MATLAB uses 1-based indexing for ind
        ind = ind + 1

        return zmin.reshape(shape), ind.reshape(shape)

    def round_z(self, *z_args: np.ndarray) -> Tuple[np.ndarray, ...]:
        # MATLAB: round.m
        results = []

        for z in z_args:
            z = np.asarray(z, dtype = float).copy()
            zmin_vals, ind = self.mindist(z)

            # z-value of nearest layer (1-based ind -> 0-based)
            ztab = self.z[ind - 1]

            # Shift direction
            direction = np.sign(z - ztab)

            # Shift points that are too close to layer
            mask = zmin_vals <= self.zmin
            z[mask] = ztab[mask] + direction[mask] * self.zmin

            results.append(z)

        return tuple(results)

    def _mul(self,
            a: np.ndarray,
            b: np.ndarray) -> np.ndarray:
        # MATLAB: private/mul.m
        a = np.asarray(a)
        b = np.asarray(b)

        if a.shape == b.shape:
            return a * b
        else:
            # Outer product
            a_flat = a.ravel()
            b_flat = b.ravel()
            c = np.outer(a_flat, b_flat)

            # Determine output shape
            siza = list(a.shape)
            sizb = list(b.shape)
            if siza[-1] == 1:
                siza = siza[:-1]
            if sizb[0] == 1:
                sizb = sizb[1:]
            out_shape = siza + sizb
            if len(out_shape) == 0:
                return c.ravel()[0]
            return c.reshape(out_shape)

    def reflection(self,
            enei: float,
            kpar: Union[float, complex],
            pos: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # MATLAB: reflection.m
        # Use simpler equations for substrate in case of single interface
        if len(self.z) == 1:
            return self._reflection_subs(enei, kpar, pos)

        return self._reflection_full(enei, kpar, pos)

    def _reflection_subs(self,
            enei: float,
            kpar: Union[float, complex],
            pos: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # MATLAB: reflectionsubs.m
        # Dielectric functions and wavenumbers in media
        eps_vals = np.empty(len(self.eps), dtype = complex)
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            eps_vals[i], k_vals[i] = eps_func(enei)

        # z-component of wavevector
        kz = np.sqrt(k_vals ** 2 - kpar ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        # Dielectric functions and wavenumbers
        # eps1 is above, eps2 is below
        eps1 = eps_vals[0]
        k1z = kz[0]
        eps2 = eps_vals[1]
        k2z = kz[1]

        # Parallel surface current
        rr_p = (k1z - k2z) / (k2z + k1z)
        r_p = np.array([[rr_p, 1 + rr_p], [1 - rr_p, -rr_p]], dtype = complex)

        # Wavenumber of light in vacuum
        k0 = 2 * np.pi / enei

        # Auxiliary quantity
        Delta = (k2z + k1z) * (eps1 * k2z + eps2 * k1z)

        # Surface charge from surface charge source
        r_ss_11 = (k1z + k2z) * (2 * eps1 * k1z - eps2 * k1z - eps1 * k2z) / Delta
        r_ss_22 = (k2z + k1z) * (2 * eps2 * k2z - eps1 * k2z - eps2 * k1z) / Delta
        r_ss = np.array([[r_ss_11, k1z / k2z * (r_ss_22 + 1)],
                         [k2z / k1z * (r_ss_11 + 1), r_ss_22]], dtype = complex)

        # Induced surface current from surface charge source
        r_hs_11 = -2 * k0 * (eps2 - eps1) * eps1 * k1z / Delta
        r_hs_22 = -2 * k0 * (eps1 - eps2) * eps2 * k2z / Delta
        r_hs = np.array([[r_hs_11, -k1z / k2z * r_hs_22],
                         [k2z / k1z * r_hs_11, -r_hs_22]], dtype = complex)

        # Induced surface charge from surface current source
        r_sh_11 = -2 * k0 * (eps2 - eps1) * k1z / Delta
        r_sh_22 = -2 * k0 * (eps1 - eps2) * k2z / Delta
        r_sh = np.array([[r_sh_11, -k1z / k2z * r_sh_22],
                         [k2z / k1z * r_sh_11, -r_sh_22]], dtype = complex)

        # Surface current from surface current source
        r_hh_11 = (k1z - k2z) * (2 * eps1 * k1z - eps2 * k1z + eps1 * k2z) / Delta
        r_hh_22 = (k2z - k1z) * (2 * eps2 * k2z - eps1 * k2z + eps2 * k1z) / Delta
        r_hh = np.array([[r_hh_11, k1z / k2z * (r_hh_22 + 1)],
                         [k2z / k1z * (r_hh_11 + 1), r_hh_22]], dtype = complex)

        r = {'p': r_p, 'ss': r_ss, 'hs': r_hs, 'sh': r_sh, 'hh': r_hh}
        rz = {}

        # Green function propagation factors
        ind1 = np.atleast_1d(pos['ind1'])
        ind2 = np.atleast_1d(pos['ind2'])
        z1 = np.atleast_1d(pos['z1'])
        z2 = np.atleast_1d(pos['z2'])

        abs_z1 = np.abs(z1[:, np.newaxis] - self.z) if z1.ndim == 1 else np.abs(z1 - self.z)
        abs_z2 = np.abs(z2[:, np.newaxis] - self.z) if z2.ndim == 1 else np.abs(z2 - self.z)
        g1 = np.exp(1j * kz[ind1 - 1][:, np.newaxis] * abs_z1)
        g2 = np.exp(1j * kz[ind2 - 1][:, np.newaxis] * abs_z2)
        # Derivative of Green function wrt z-value
        sign_z1 = np.sign(z1[:, np.newaxis] - self.z) if z1.ndim == 1 else np.sign(z1 - self.z)
        g1z = g1 * sign_z1

        # Apply propagation factors
        r_out = {}
        rz_out = {}

        same_size = (ind1.shape == ind2.shape)

        for name in r.keys():
            rr = r[name]
            if same_size:
                idx = (ind1 - 1, ind2 - 1)
                rr_sel = rr[idx]
                r_out[name] = g1.ravel() * rr_sel * g2.ravel()
                rz_out[name] = g1z.ravel() * rr_sel * g2.ravel()
            else:
                r_out[name] = rr[ind1 - 1][:, ind2 - 1] * np.outer(g1.ravel(), g2.ravel())
                rz_out[name] = rr[ind1 - 1][:, ind2 - 1] * np.outer(g1z.ravel(), g2.ravel())

        return r_out, rz_out

    def _reflection_full(self,
            enei: float,
            kpar: Union[float, complex],
            pos: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # MATLAB: reflection.m (multi-layer case)
        # Wavenumbers in media
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        # Perpendicular component of wavevector
        kz = np.sqrt(k_vals ** 2 - kpar ** 2) + 1e-10j
        kz = kz * np.sign(np.imag(kz))

        ind1 = np.atleast_1d(pos['ind1'])
        ind2 = np.atleast_1d(pos['ind2'])
        z1 = np.atleast_1d(pos['z1'])
        z2 = np.atleast_1d(pos['z2'])

        k1z = kz[ind1 - 1]
        k2z = kz[ind2 - 1]

        # Distance to lower interfaces
        z_lower = np.append(self.z, -1e100)
        z_upper = np.append(1e100, self.z)
        dn1 = np.abs(z1 - z_lower[ind1 - 1])
        dn2 = np.abs(z2 - z_lower[ind2 - 1])
        up1 = np.abs(z1 - z_upper[ind1 - 1])
        up2 = np.abs(z2 - z_upper[ind2 - 1])

        # Excitation matrix
        n_ind2 = len(ind2)
        siz = (2 * len(self.z) + 2, n_ind2)

        fac = 2j * np.pi / k2z

        exc = np.zeros(siz, dtype = complex)
        for j in range(n_ind2):
            exc[2 * ind2[j] - 1, j] += fac[j] * np.exp(1j * k2z[j] * dn2[j])
            exc[2 * ind2[j] - 2, j] += fac[j] * np.exp(1j * k2z[j] * up2[j])

        # Remove layers at infinity
        exc = exc[1:-1, :]

        # BEM solve
        par, perp = self.bemsolve(enei, kpar)

        same_size = (ind1.shape == ind2.shape)

        def _multiply(a, b):
            if same_size:
                return a[np.arange(len(ind1)), :][
                    np.arange(len(ind1)),
                    np.arange(a.shape[1]) if a.shape[1] == len(ind1) else 0
                ] * b if a.ndim == 2 else a * b
            else:
                return a[ind1 - 1, :] * b

        r = {}
        rz = {}

        # Parallel surface current
        y = par @ exc
        n_z = len(self.z)
        zeros_row = np.zeros((1, y.shape[1]), dtype = complex)

        h1_p = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h1_p[0, :] = 0
        h1_p[1:, :] = y[1::2, :]

        h2_p = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h2_p[:-1, :] = y[0::2, :]
        h2_p[-1, :] = 0

        r['p'] = self._layer_multiply(pos, h2_p, np.exp(1j * k1z * dn1), ind1) + \
                 self._layer_multiply(pos, h1_p, np.exp(1j * k1z * up1), ind1)
        rz['p'] = self._layer_multiply(pos, h2_p, np.exp(1j * k1z * dn1), ind1) - \
                  self._layer_multiply(pos, h1_p, np.exp(1j * k1z * up1), ind1)

        # Surface charge
        exc2 = np.zeros((2 * exc.shape[0], exc.shape[1]), dtype = complex)
        exc2[0::2, :] = exc

        y = perp @ exc2

        sig1 = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        sig1[0, :] = 0
        sig1[1:, :] = y[2::4, :]

        sig2 = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        sig2[:-1, :] = y[0::4, :]
        sig2[-1, :] = 0

        r['ss'] = self._layer_multiply(pos, sig2, np.exp(1j * k1z * dn1), ind1) + \
                  self._layer_multiply(pos, sig1, np.exp(1j * k1z * up1), ind1)
        rz['ss'] = self._layer_multiply(pos, sig2, np.exp(1j * k1z * dn1), ind1) - \
                   self._layer_multiply(pos, sig1, np.exp(1j * k1z * up1), ind1)

        h1_s = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h1_s[0, :] = 0
        h1_s[1:, :] = y[3::4, :]

        h2_s = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h2_s[:-1, :] = y[1::4, :]
        h2_s[-1, :] = 0

        r['hs'] = self._layer_multiply(pos, h2_s, np.exp(1j * k1z * dn1), ind1) + \
                  self._layer_multiply(pos, h1_s, np.exp(1j * k1z * up1), ind1)
        rz['hs'] = self._layer_multiply(pos, h2_s, np.exp(1j * k1z * dn1), ind1) - \
                   self._layer_multiply(pos, h1_s, np.exp(1j * k1z * up1), ind1)

        # Perpendicular surface current
        exc2 = np.zeros((2 * exc.shape[0], exc.shape[1]), dtype = complex)
        exc2[1::2, :] = exc

        y = perp @ exc2

        sig1 = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        sig1[0, :] = 0
        sig1[1:, :] = y[2::4, :]

        sig2 = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        sig2[:-1, :] = y[0::4, :]
        sig2[-1, :] = 0

        r['sh'] = self._layer_multiply(pos, sig2, np.exp(1j * k1z * dn1), ind1) + \
                  self._layer_multiply(pos, sig1, np.exp(1j * k1z * up1), ind1)
        rz['sh'] = self._layer_multiply(pos, sig2, np.exp(1j * k1z * dn1), ind1) - \
                   self._layer_multiply(pos, sig1, np.exp(1j * k1z * up1), ind1)

        h1_h = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h1_h[0, :] = 0
        h1_h[1:, :] = y[3::4, :]

        h2_h = np.empty((n_z + 1, y.shape[1]), dtype = complex)
        h2_h[:-1, :] = y[1::4, :]
        h2_h[-1, :] = 0

        r['hh'] = self._layer_multiply(pos, h2_h, np.exp(1j * k1z * dn1), ind1) + \
                  self._layer_multiply(pos, h1_h, np.exp(1j * k1z * up1), ind1)
        rz['hh'] = self._layer_multiply(pos, h2_h, np.exp(1j * k1z * dn1), ind1) - \
                   self._layer_multiply(pos, h1_h, np.exp(1j * k1z * up1), ind1)

        return r, rz

    def _layer_multiply(self,
            pos: Dict[str, Any],
            a: np.ndarray,
            b: np.ndarray,
            ind1: np.ndarray) -> np.ndarray:
        z1 = np.atleast_1d(pos['z1'])
        z2 = np.atleast_1d(pos['z2'])
        ind2 = np.atleast_1d(pos['ind2'])

        same_size = (z1.shape == z2.shape)

        if same_size:
            # Direct product
            return a[ind1 - 1, np.arange(a.shape[1])] * b
        else:
            return a[ind1 - 1, :] * b[:, np.newaxis]

    def fresnel(self,
            enei: float,
            kpar: Union[float, complex],
            pos: Dict[str, Any]) -> Dict[str, Any]:
        # MATLAB: fresnel.m
        # Wavenumber in media
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        # Perpendicular component of wavevector
        kz = np.sqrt(k_vals ** 2 - kpar ** 2) + 1e-10j
        kz = kz * np.sign(np.imag(kz))

        # Perpendicular components
        ind1 = np.atleast_1d(pos['ind1'])
        ind2 = np.atleast_1d(pos['ind2'])
        k1z = kz[ind1 - 1]
        k2z = kz[ind2 - 1]

        # Ratio of z-components
        z1 = np.atleast_1d(pos['z1'])
        z2 = np.atleast_1d(pos['z2'])
        if z1.shape == z2.shape:
            ratio = k2z / k1z
        else:
            ratio = np.outer(1.0 / k1z, k2z)

        # Reflection and transmission coefficients
        r, _ = self.reflection(enei, kpar, pos)

        # Correct: REFLECTION uses surface charges/currents, FRESNEL uses potentials
        for name in r.keys():
            r[name] = r[name] * ratio

        return r

    def efresnel(self,
            pol: np.ndarray,
            dir: np.ndarray,
            enei: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # MATLAB: efresnel.m
        pol = np.atleast_2d(pol)
        dir = np.atleast_2d(dir)

        k0 = 2 * np.pi / enei
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        # Upper and lower layers
        z1_val = self.z[0] + 1e-10
        ind1 = 1
        z2_val = self.z[-1] - 1e-10
        ind2 = 1 + len(self.z)

        n_dir = pol.shape[0]
        ei = np.zeros_like(pol, dtype = complex)
        er = np.zeros_like(pol, dtype = complex)
        et = np.zeros_like(pol, dtype = complex)
        ki = np.zeros_like(pol, dtype = complex)
        kr = np.zeros_like(pol, dtype = complex)
        kt = np.zeros_like(pol, dtype = complex)

        for i in range(n_dir):
            if dir[i, 2] < 0:
                # Excitation through upper medium
                posr = {'r': 0, 'z1': z1_val, 'ind1': ind1, 'z2': z1_val, 'ind2': ind1}
                post = {'r': 0, 'z1': z2_val, 'ind1': ind2, 'z2': z1_val, 'ind2': ind1}
            else:
                # Excitation through lower medium
                posr = {'r': 0, 'z1': z2_val, 'ind1': ind2, 'z2': z2_val, 'ind2': ind2}
                post = {'r': 0, 'z1': z1_val, 'ind1': ind1, 'z2': z2_val, 'ind2': ind2}

            kpar_vec = k_vals[post['ind2'] - 1] * dir[i, 0:2]
            kpar_mag = np.sqrt(np.sum(kpar_vec ** 2))

            kzr = np.sqrt(k_vals[posr['ind1'] - 1] ** 2 - kpar_mag ** 2)
            kzr = kzr * np.sign(np.imag(kzr + 1e-10j))
            kzt = np.sqrt(k_vals[post['ind1'] - 1] ** 2 - kpar_mag ** 2)
            kzt = kzt * np.sign(np.imag(kzt + 1e-10j))

            ki[i, :] = np.array([kpar_vec[0], kpar_vec[1],
                                 np.sign(dir[i, 2]) * kzr])
            kr[i, :] = np.array([kpar_vec[0], kpar_vec[1],
                                 -np.sign(dir[i, 2]) * kzr])
            kt[i, :] = np.array([kpar_vec[0], kpar_vec[1],
                                 np.sign(dir[i, 2]) * kzt])

            # Reflection and transmission coefficients
            r = self.fresnel(enei, kpar_mag, posr)
            t = self.fresnel(enei, kpar_mag, post)

            ei[i, :] = pol[i, :]

            # Reflected and transmitted electric field
            r_p = np.atleast_1d(r['p']).ravel()[0]
            r_hh = np.atleast_1d(r['hh']).ravel()[0]
            r_sh = np.atleast_1d(r['sh']).ravel()[0]
            t_p = np.atleast_1d(t['p']).ravel()[0]
            t_hh = np.atleast_1d(t['hh']).ravel()[0]
            t_sh = np.atleast_1d(t['sh']).ravel()[0]

            er[i, 0:2] = r_p * pol[i, 0:2]
            er[i, 2] = r_hh * pol[i, 2]
            er[i, :] = er[i, :] - kr[i, :] / k0 * r_sh * pol[i, 2]

            et[i, 0:2] = t_p * pol[i, 0:2]
            et[i, 2] = t_hh * pol[i, 2]
            et[i, :] = et[i, :] - kt[i, :] / k0 * t_sh * pol[i, 2]

            # Phase factors
            er[i, :] *= np.exp(-1j * kr[i, 2] * posr['z2'] - 1j * kr[i, 2] * posr['z1'])
            et[i, :] *= np.exp(-1j * kr[i, 2] * post['z2'] - 1j * kt[i, 2] * post['z1'])

        e = {'i': ei, 'r': er, 't': et}
        k = {'i': ki, 'r': kr, 't': kt}
        return e, k

    def green(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        # MATLAB: green.m
        r = np.asarray(r, dtype = float)
        z1 = np.asarray(z1, dtype = float)
        z2 = np.asarray(z2, dtype = float)

        # Round radii and z-values
        r = np.maximum(r, self.rmin)
        z1, z2 = self.round_z(z1, z2)

        # Save positions
        pos = {'r': r, 'z1': z1, 'z2': z2}
        ind1, _ = self.indlayer(z1)
        ind2, _ = self.indlayer(z2)
        pos['ind1'] = ind1
        pos['ind2'] = ind2

        # Expand positions using mul
        r_exp = self._mul(r, self._mul(np.ones_like(z1), np.ones_like(z2)))
        z1_exp = self._mul(np.ones_like(r), self._mul(z1, np.ones_like(z2)))
        z2_exp = self._mul(np.ones_like(r), self._mul(np.ones_like(z1), z2))

        # Minimal distance to layers
        zmin_1, _ = self.mindist(z1_exp.ravel())
        zmin_2, _ = self.mindist(z2_exp.ravel())
        zmin = (zmin_1 + zmin_2).reshape(r_exp.shape)

        # Size of integrand
        n1 = r_exp.size

        # Get number of reflection coefficient names
        test_r, _ = self.reflection(enei, 0, pos)
        names = list(test_r.keys())
        n_names = len(names)

        # Solve ODEs for complex integration
        # Semi-ellipse integration
        y1 = self._integrate_semiellipse(enei, pos, n1, n_names)

        # Determine integration path
        flat_zmin = zmin.ravel()
        flat_r = r_exp.ravel()
        ind_real = np.where(flat_zmin >= flat_r / self.ratio)[0]
        ind_imag = np.where(flat_zmin < flat_r / self.ratio)[0]

        n2 = len(ind_real)
        n3 = len(ind_imag)
        y2 = np.zeros(15 * n2) if n2 > 0 else np.zeros(0)
        y3 = np.zeros(15 * n3) if n3 > 0 else np.zeros(0)

        # Integration along real axis
        if n2 > 0:
            y2 = self._integrate_real(enei, pos, ind_real, n2, n_names)

        # Integration along imaginary axis
        if n3 > 0:
            y3 = self._integrate_imag(enei, pos, ind_imag, n3, n_names)

        # Combine results
        G = {}
        Fr = {}
        Fz = {}

        for iname, name in enumerate(names):
            g = y1[(iname * 3) * n1:(iname * 3 + 1) * n1].copy()
            fr = y1[(iname * 3 + 1) * n1:(iname * 3 + 2) * n1].copy()
            fz = y1[(iname * 3 + 2) * n1:(iname * 3 + 3) * n1].copy()

            if n2 > 0:
                g[ind_real] += y2[(iname * 3) * n2:(iname * 3 + 1) * n2]
                fr[ind_real] += y2[(iname * 3 + 1) * n2:(iname * 3 + 2) * n2]
                fz[ind_real] += y2[(iname * 3 + 2) * n2:(iname * 3 + 3) * n2]

            if n3 > 0:
                g[ind_imag] += y3[(iname * 3) * n3:(iname * 3 + 1) * n3]
                fr[ind_imag] += y3[(iname * 3 + 1) * n3:(iname * 3 + 2) * n3]
                fz[ind_imag] += y3[(iname * 3 + 2) * n3:(iname * 3 + 3) * n3]

            G[name] = np.squeeze(g.reshape(r_exp.shape))
            Fr[name] = np.squeeze(fr.reshape(r_exp.shape))
            Fz[name] = np.squeeze(fz.reshape(r_exp.shape))

        pos_out = {'r': r_exp, 'z1': z1_exp, 'z2': z2_exp, 'zmin': zmin}
        return G, Fr, Fz, pos_out

    def _integrate_semiellipse(self,
            enei: float,
            pos: Dict[str, Any],
            n1: int,
            n_names: int) -> np.ndarray:
        # Integration along semi-ellipse in complex kr-plane
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        k0 = 2 * np.pi / enei
        k1max = np.max(np.real(k_vals)) + k0

        def rhs_semi(x, y):
            kr = k1max * (1 - np.cos(x) - 1j * self.semi * np.sin(x))
            dkr = k1max * (np.sin(x) - 1j * self.semi * np.cos(x))
            integ = self._intbessel(enei, kr, pos)
            return dkr * integ

        y0 = np.zeros(15 * n1, dtype = complex)
        sol = solve_ivp(rhs_semi, [0, np.pi], y0,
                        t_eval = [np.pi],
                        rtol = 1e-6, atol = self.atol,
                        max_step = 0.1)
        return sol.y[:, -1]

    def _integrate_real(self,
            enei: float,
            pos: Dict[str, Any],
            ind: np.ndarray,
            n: int,
            n_names: int) -> np.ndarray:
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        k0 = 2 * np.pi / enei
        k1max = np.max(np.real(k_vals)) + k0

        def rhs_real(x, y):
            if x < 1e-15:
                return np.zeros_like(y)
            kr = 2 * k1max / x
            integ = self._intbessel(enei, kr, pos, ind)
            return -2 * k1max * integ / x ** 2

        y0 = np.zeros(15 * n, dtype = complex)
        sol = solve_ivp(rhs_real, [1, 1e-10], y0,
                        t_eval = [1e-10],
                        rtol = 1e-6, atol = self.atol,
                        max_step = 0.1)
        return sol.y[:, -1]

    def _integrate_imag(self,
            enei: float,
            pos: Dict[str, Any],
            ind: np.ndarray,
            n: int,
            n_names: int) -> np.ndarray:
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        k0 = 2 * np.pi / enei
        k1max = np.max(np.real(k_vals)) + k0

        def rhs_imag(x, y):
            if x < 1e-15:
                return np.zeros_like(y)
            kr = 2 * k1max * (1 - 1j + 1j / x)
            integ = self._inthankel(enei, kr, pos, ind)
            return -2j * k1max * integ / x ** 2

        y0 = np.zeros(15 * n, dtype = complex)
        sol = solve_ivp(rhs_imag, [1, 1e-10], y0,
                        t_eval = [1e-10],
                        rtol = 1e-6, atol = self.atol,
                        max_step = 0.1)
        return sol.y[:, -1]

    def _intbessel(self,
            enei: float,
            kpar: complex,
            pos: Dict[str, Any],
            ind: Optional[np.ndarray] = None) -> np.ndarray:
        # MATLAB: private/intbessel.m
        r_exp = self._mul(pos['r'], self._mul(np.ones_like(pos['z1']), np.ones_like(pos['z2'])))
        r_flat = r_exp.ravel()

        if ind is None:
            ind = np.arange(len(r_flat))

        # Wavenumber in media
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        kz = np.sqrt(k_vals ** 2 - kpar ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        ind1 = np.atleast_1d(pos['ind1'])
        kz_sel = kz[ind1 - 1] if ind1.ndim == 0 else kz[ind1.ravel() - 1]

        # Expand kz to match r shape
        if len(kz_sel) != len(r_flat):
            kz_full = np.ones(len(r_flat), dtype = complex)
            # Replicate kz_sel appropriately
            n_r = np.atleast_1d(pos['r']).size
            n_z1 = np.atleast_1d(pos['z1']).size
            n_z2 = np.atleast_1d(pos['z2']).size
            kz_full = np.tile(np.repeat(kz_sel, n_z2), n_r)[:len(r_flat)]
        else:
            kz_full = kz_sel

        kz_ind = kz_full[ind]

        # Reflection coefficients
        refl, reflz = self.reflection(enei, kpar, pos)

        # Bessel functions
        r_ind = r_flat[ind]
        j0 = besselj(0, np.real(kpar * r_ind) if np.isreal(kpar) else kpar * r_ind)
        j1 = besselj(1, np.real(kpar * r_ind) if np.isreal(kpar) else kpar * r_ind)

        n = len(ind)
        y = np.zeros(15 * n, dtype = complex)

        names = list(refl.keys())
        for iname, name in enumerate(names):
            rr = np.atleast_1d(refl[name]).ravel()
            rrz = np.atleast_1d(reflz[name]).ravel()

            rr_ind = rr[ind] if len(rr) > 1 else rr
            rrz_ind = rrz[ind] if len(rrz) > 1 else rrz

            # G component
            y[(iname * 3) * n:(iname * 3 + 1) * n] = \
                1j * j0 * rr_ind * kpar / kz_ind
            # Fr component
            y[(iname * 3 + 1) * n:(iname * 3 + 2) * n] = \
                1j * j1 * rr_ind * (-kpar ** 2) / kz_ind
            # Fz component
            y[(iname * 3 + 2) * n:(iname * 3 + 3) * n] = \
                1j * j0 * 1j * rrz_ind * kpar

        return np.real(y) if np.all(np.isreal(y)) else y

    def _inthankel(self,
            enei: float,
            kpar: complex,
            pos: Dict[str, Any],
            ind: Optional[np.ndarray] = None) -> np.ndarray:
        # MATLAB: private/inthankel.m
        r_exp = self._mul(pos['r'], self._mul(np.ones_like(pos['z1']), np.ones_like(pos['z2'])))
        r_flat = r_exp.ravel()

        if ind is None:
            ind = np.arange(len(r_flat))

        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            _, k_vals[i] = eps_func(enei)

        kpar1 = kpar
        kpar2 = np.conj(kpar)

        kz1 = np.sqrt(k_vals ** 2 - kpar1 ** 2)
        kz1 = kz1 * np.sign(np.imag(kz1 + 1e-10j))
        kz2 = np.sqrt(k_vals ** 2 - kpar2 ** 2)
        kz2 = kz2 * np.sign(np.imag(kz2 + 1e-10j))

        ind1 = np.atleast_1d(pos['ind1'])
        kz1_sel = kz1[ind1.ravel() - 1] if ind1.size > 1 else kz1[ind1 - 1]
        kz2_sel = kz2[ind1.ravel() - 1] if ind1.size > 1 else kz2[ind1 - 1]

        refl1, refl1z = self.reflection(enei, kpar1, pos)
        refl2, refl2z = self.reflection(enei, kpar2, pos)

        r_ind = r_flat[ind]
        h0 = hankel1(0, kpar * r_ind)
        h1 = hankel1(1, kpar * r_ind)

        n = len(ind)
        y = np.zeros(15 * n, dtype = complex)

        # Expand kz
        if np.atleast_1d(kz1_sel).size != len(r_flat):
            kz1_full = np.tile(kz1_sel, max(1, len(r_flat) // len(np.atleast_1d(kz1_sel))))[:len(r_flat)]
            kz2_full = np.tile(kz2_sel, max(1, len(r_flat) // len(np.atleast_1d(kz2_sel))))[:len(r_flat)]
        else:
            kz1_full = kz1_sel
            kz2_full = kz2_sel

        kz1_ind = np.atleast_1d(kz1_full)[ind] if np.atleast_1d(kz1_full).size > 1 else kz1_full
        kz2_ind = np.atleast_1d(kz2_full)[ind] if np.atleast_1d(kz2_full).size > 1 else kz2_full

        names = list(refl1.keys())
        for iname, name in enumerate(names):
            rr1 = np.atleast_1d(refl1[name]).ravel()
            rr1z = np.atleast_1d(refl1z[name]).ravel()
            rr2 = np.atleast_1d(refl2[name]).ravel()
            rr2z = np.atleast_1d(refl2z[name]).ravel()

            rr1_ind = rr1[ind] if len(rr1) > 1 else rr1
            rr1z_ind = rr1z[ind] if len(rr1z) > 1 else rr1z
            rr2_ind = rr2[ind] if len(rr2) > 1 else rr2
            rr2z_ind = rr2z[ind] if len(rr2z) > 1 else rr2z

            # G component
            y[(iname * 3) * n:(iname * 3 + 1) * n] = \
                0.5j * h0 * rr1_ind * kpar1 / kz1_ind - \
                0.5j * np.conj(h0) * rr2_ind * kpar2 / kz2_ind
            # Fr component
            y[(iname * 3 + 1) * n:(iname * 3 + 2) * n] = \
                0.5j * h1 * rr1_ind * (-kpar1 ** 2) / kz1_ind - \
                0.5j * np.conj(h1) * rr2_ind * (-kpar2 ** 2) / kz2_ind
            # Fz component
            y[(iname * 3 + 2) * n:(iname * 3 + 3) * n] = \
                0.5j * h0 * 1j * rr1z_ind * kpar1 - \
                0.5j * np.conj(h0) * 1j * rr2z_ind * kpar2

        return y

    def bemsolve(self,
            enei: float,
            kpar: Union[float, complex]) -> Tuple[np.ndarray, np.ndarray]:
        # MATLAB: bemsolve.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
        k0 = 2 * np.pi / enei

        eps_vals = np.empty(len(self.eps), dtype = complex)
        k_vals = np.empty(len(self.eps), dtype = complex)
        for i, eps_func in enumerate(self.eps):
            eps_vals[i], k_vals[i] = eps_func(enei)

        kz = np.sqrt(k_vals ** 2 - kpar ** 2)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        n = len(self.z)
        G0 = 2j * np.pi / kz

        # Interlayer Green function
        if n > 1:
            G = 2j * np.pi / kz[1:-1] * np.exp(1j * kz[1:-1] * np.abs(np.diff(self.z)))
        else:
            G = np.array([], dtype = complex)

        # Parallel surface current
        siz = 2 * n
        lhs = np.zeros((siz, siz), dtype = complex)
        rhs_mat = np.zeros((siz, siz), dtype = complex)

        i1 = np.arange(1, 2 * n, 2)  # h2 indices (0-based)
        i2 = np.arange(0, 2 * n, 2)  # h1 indices
        eq1 = np.arange(0, 2 * n, 2)  # continuity of A
        eq2 = np.arange(1, 2 * n, 2)  # continuity of dA

        # Continuity of vector potential [Eq. (13a)]
        for idx in range(n):
            lhs[eq1[idx], i1[idx]] = G0[idx + 1]
            lhs[eq1[idx], i2[idx]] = -G0[idx]

            rhs_mat[eq1[idx], i1[idx]] = -1
            rhs_mat[eq1[idx], i2[idx]] = 1

        if n > 1:
            for idx in range(n - 1):
                lhs[eq1[idx + 1], i1[idx]] = -G[idx]
                lhs[eq1[idx], i2[idx + 1]] = G[idx]

        # Continuity of derivative [Eq. (13b)]
        for idx in range(n):
            lhs[eq2[idx], i1[idx]] = 2j * np.pi
            lhs[eq2[idx], i2[idx]] = 2j * np.pi

            rhs_mat[eq2[idx], i1[idx]] = kz[idx + 1]
            rhs_mat[eq2[idx], i2[idx]] = kz[idx]

        if n > 1:
            for idx in range(n - 1):
                lhs[eq2[idx + 1], i1[idx]] = -kz[idx + 1] * G[idx]
                lhs[eq2[idx], i2[idx + 1]] = -kz[idx + 1] * G[idx]

        par = np.linalg.solve(lhs, rhs_mat)

        # Perpendicular surface current and surface charge
        siz = 4 * n
        lhs = np.zeros((siz, siz), dtype = complex)
        rhs_mat = np.zeros((siz, siz), dtype = complex)

        # Indices for surface charge (i1,j1) and current (i2,j2)
        i1 = np.arange(2, 4 * n, 4)  # sig1
        i2 = np.arange(0, 4 * n, 4)  # sig2
        j1 = np.arange(3, 4 * n, 4)  # h1
        j2 = np.arange(1, 4 * n, 4)  # h2

        eq1 = np.arange(0, 4 * n, 4)  # continuity of phi
        eq2 = np.arange(1, 4 * n, 4)  # continuity of A
        eq3 = np.arange(2, 4 * n, 4)  # continuity of D
        eq4 = np.arange(3, 4 * n, 4)  # continuity of dA

        for idx in range(n):
            # Eq. (14a): continuity of scalar potential
            lhs[eq1[idx], i1[idx]] = G0[idx + 1]
            lhs[eq1[idx], i2[idx]] = -G0[idx]
            rhs_mat[eq1[idx], i1[idx]] = -1
            rhs_mat[eq1[idx], i2[idx]] = 1

            # Eq. (14b): continuity of vector potential
            lhs[eq2[idx], j1[idx]] = G0[idx + 1]
            lhs[eq2[idx], j2[idx]] = -G0[idx]
            rhs_mat[eq2[idx], j1[idx]] = -1
            rhs_mat[eq2[idx], j2[idx]] = 1

            # Eq. (14c): continuity of dielectric displacement
            lhs[eq3[idx], i1[idx]] = 2j * np.pi * eps_vals[idx + 1]
            lhs[eq3[idx], i2[idx]] = 2j * np.pi * eps_vals[idx]
            lhs[eq3[idx], j1[idx]] = k0 * G0[idx + 1] * eps_vals[idx + 1]
            lhs[eq3[idx], j2[idx]] = -k0 * G0[idx] * eps_vals[idx]
            rhs_mat[eq3[idx], i1[idx]] = kz[idx + 1] * eps_vals[idx + 1]
            rhs_mat[eq3[idx], i2[idx]] = kz[idx] * eps_vals[idx]
            rhs_mat[eq3[idx], j1[idx]] = -k0 * eps_vals[idx + 1]
            rhs_mat[eq3[idx], j2[idx]] = k0 * eps_vals[idx]

            # Eq. (14d): continuity of derivative of vector potential
            lhs[eq4[idx], j1[idx]] = 2j * np.pi
            lhs[eq4[idx], j2[idx]] = 2j * np.pi
            lhs[eq4[idx], i1[idx]] = k0 * G0[idx + 1] * eps_vals[idx + 1]
            lhs[eq4[idx], i2[idx]] = -k0 * G0[idx] * eps_vals[idx]
            rhs_mat[eq4[idx], j1[idx]] = kz[idx + 1]
            rhs_mat[eq4[idx], j2[idx]] = kz[idx]
            rhs_mat[eq4[idx], i1[idx]] = -k0 * eps_vals[idx + 1]
            rhs_mat[eq4[idx], i2[idx]] = k0 * eps_vals[idx]

        if n > 1:
            for idx in range(n - 1):
                # Cross-coupling terms
                lhs[eq1[idx + 1], i1[idx]] = -G[idx]
                lhs[eq1[idx], i2[idx + 1]] = G[idx]

                lhs[eq2[idx + 1], j1[idx]] = -G[idx]
                lhs[eq2[idx], j2[idx + 1]] = G[idx]

                lhs[eq3[idx + 1], i1[idx]] = -kz[idx + 1] * eps_vals[idx + 1] * G[idx]
                lhs[eq3[idx], i2[idx + 1]] = -kz[idx + 1] * eps_vals[idx + 1] * G[idx]
                lhs[eq3[idx + 1], j1[idx]] = -k0 * eps_vals[idx + 1] * G[idx]
                lhs[eq3[idx], j2[idx + 1]] = k0 * eps_vals[idx + 1] * G[idx]

                lhs[eq4[idx + 1], j1[idx]] = -kz[idx + 1] * G[idx]
                lhs[eq4[idx], j2[idx + 1]] = -kz[idx + 1] * G[idx]
                lhs[eq4[idx + 1], i1[idx]] = -k0 * eps_vals[idx + 1] * G[idx]
                lhs[eq4[idx], i2[idx + 1]] = k0 * eps_vals[idx + 1] * G[idx]

        perp = np.linalg.solve(lhs, rhs_mat)

        return par, perp

    def tabspace(self,
            *args: Any,
            **options: Any) -> Union[Dict[str, Any], list]:
        # MATLAB: tabspace.m
        if len(args) > 0 and isinstance(args[0], (int, float, np.ndarray)):
            return self._tabspace1(args[0], args[1], args[2], **options)
        else:
            return self._tabspace2(args[0], *args[1:], **options)

    def _tabspace1(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray,
            **options: Any) -> Dict[str, Any]:
        # MATLAB: private/tabspace1.m
        r = np.asarray(r, dtype = float)
        z1 = np.asarray(z1, dtype = float)
        z2 = np.asarray(z2, dtype = float)

        rmod = options.get('rmod', 'log')
        zmod = options.get('zmod', 'log')

        # Slightly reduce zmin for tabulation
        zmin_orig = self.zmin
        self.zmin = 0.999 * self.zmin

        tab = {}
        # Table for radii
        tab['r'] = self._linlogspace(max(r[0], self.rmin), r[1], int(r[2]), rmod)

        if z1.size == 1:
            tab['z1'] = z1
        else:
            z1_sorted = np.sort(np.asarray(self.round_z(z1[0:2])).ravel())
            if np.abs(z1_sorted[0] - z1_sorted[1]) < 1e-3:
                z1_sorted = self._expand_z(z1_sorted)
            tab['z1'] = self._zlinlogspace(z1_sorted[0], z1_sorted[1], int(z1[2]), zmod)

        if z2.size == 1:
            tab['z2'] = z2
        else:
            z2_sorted = np.sort(np.asarray(self.round_z(z2[0:2])).ravel())
            if np.abs(z2_sorted[0] - z2_sorted[1]) < 1e-3:
                z2_sorted = self._expand_z(z2_sorted)
            tab['z2'] = self._zlinlogspace(z2_sorted[0], z2_sorted[1], int(z2[2]), zmod)

        self.zmin = zmin_orig
        return tab

    def _tabspace2(self,
            p: Any,
            *args: Any,
            **options: Any) -> list:
        # MATLAB: private/tabspace2.m
        pt = None
        remaining = list(args)
        if len(remaining) > 0 and not isinstance(remaining[0], str):
            pt = remaining[0]
            remaining = remaining[1:]

        scale = options.get('scale', 1.05)
        nr = options.get('nr', 30)
        nz = options.get('nz', 30)
        n_layers = len(self.z) + 1

        if not isinstance(p, list):
            p = [p]

        # Collect positions from particles
        positions = []
        for particle in p:
            if hasattr(particle, 'verts'):
                positions.append(particle.verts)
            elif hasattr(particle, 'pos'):
                positions.append(particle.pos)

        pos1 = np.vstack(positions)
        pos2 = pos1.copy()

        if pt is not None:
            if hasattr(pt, 'pos'):
                pos1 = np.vstack([pos1, pt.pos])

        # Get limits
        ir, iz1, iz2 = self._limits(pos1, pos2)

        # Generate tables
        tabs = []
        if isinstance(nz, (int, float)):
            nz_arr = [nz] * n_layers
        else:
            nz_arr = list(nz)

        for i1 in range(n_layers):
            for i2 in range(n_layers):
                if iz1[i1][i2] is not None:
                    # Adjust z-values
                    z1_adj, z2_adj = self._adjust(iz1[i1][i2], iz2[i1][i2], scale = scale)

                    # Scale radii
                    r_range = ir[i1][i2]
                    r_mean = np.mean(r_range)
                    r_range = np.maximum(
                        [r_mean + 0.5 * scale * (r_range[0] - r_range[1]),
                         r_mean - 0.5 * scale * (r_range[0] - r_range[1])], 0)

                    tab = self._tabspace1(
                        np.array([r_range[0], r_range[1], nr]),
                        np.array([z1_adj[0], z1_adj[1], nz_arr[i1]]),
                        np.array([z2_adj[0], z2_adj[1], nz_arr[i2]]) if len(z2_adj) > 1
                        else np.array([z2_adj[0]]),
                        **options)
                    tabs.append(tab)

        return tabs

    def _adjust(self,
            z1: np.ndarray,
            z2: np.ndarray,
            scale: float = 1.05,
            range_mode: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        # MATLAB: private/adjust.m
        i1, _ = self.indlayer(np.array([z1[0]]))
        i2, _ = self.indlayer(np.array([z2[0]]))
        i1 = int(i1)
        i2 = int(i2)

        n = len(self.z) + 1
        zlayer = np.empty((n, 2))
        for i in range(n):
            zlayer[i, 0] = self.z[i] if i < len(self.z) else -np.inf
            zlayer[i, 1] = self.z[i - 1] if i > 0 else np.inf

        z1 = self._enlarge(z1, zlayer[i1 - 1, :], scale, range_mode)
        z2 = self._enlarge(z2, zlayer[i2 - 1, :], scale, range_mode)

        # Handle uppermost and lowermost layer
        if i1 == 1 and i2 == 1:
            z1 = z1 + z2 - self.z[0]
            z2 = np.array([self.z[0] + 1e-10])
        elif i1 == n and i2 == n:
            z1 = z1 + z2 - self.z[-1]
            z2 = np.array([self.z[-1] - 1e-10])

        return z1, z2

    def _enlarge(self,
            z: np.ndarray,
            zlayer: np.ndarray,
            scale: float,
            range_mode: Optional[str]) -> np.ndarray:
        z = np.asarray(z, dtype = float).copy()
        if np.abs(z[0] - z[1]) < 1e-10:
            z = np.array([z[0] - 1e-2, z[0] + 1e-2])
        else:
            mean_z = np.mean(z)
            z = np.array([mean_z + 0.5 * scale * (z[0] - z[1]),
                          mean_z - 0.5 * scale * (z[0] - z[1])])

        if z[0] < zlayer[0]:
            z[0] = zlayer[0] + 1e-10
        if z[1] > zlayer[1]:
            z[1] = zlayer[1] - 1e-10

        if range_mode == 'full':
            if not np.isinf(zlayer[0]):
                z[0] = zlayer[0] + 1e-10
            if not np.isinf(zlayer[1]):
                z[1] = zlayer[1] - 1e-10

        return z

    def _expand_z(self,
            z: np.ndarray) -> np.ndarray:
        _, ind = self.mindist(np.array([z[0]]))
        ind = int(ind) - 1
        z_out = np.sort([z[0] + np.sign(z[0] - self.z[ind]) * 0.1 * self.zmin, z[1]])
        return z_out

    def _limits(self,
            pos1: np.ndarray,
            pos2: np.ndarray) -> Tuple[list, list, list]:
        # Radial distance between points
        diff_xy = pos1[:, 0:2][:, np.newaxis, :] - pos2[:, 0:2][np.newaxis, :, :]
        r = np.sqrt(np.sum(diff_xy ** 2, axis = 2))

        z1 = pos1[:, 2]
        ind1, _ = self.indlayer(z1)
        z2 = pos2[:, 2]
        ind2, _ = self.indlayer(z2)

        n_layers = len(self.z) + 1
        ir = [[None] * n_layers for _ in range(n_layers)]
        iz1 = [[None] * n_layers for _ in range(n_layers)]
        iz2 = [[None] * n_layers for _ in range(n_layers)]

        for i1 in range(1, n_layers + 1):
            for i2 in range(1, n_layers + 1):
                mask1 = (ind1 == i1)
                mask2 = (ind2 == i2)
                if np.any(mask1) and np.any(mask2):
                    r_sub = r[mask1][:, mask2]
                    ir[i1 - 1][i2 - 1] = np.array([np.min(r_sub), np.max(r_sub)])
                    iz1[i1 - 1][i2 - 1] = np.array([np.min(z1[mask1]), np.max(z1[mask1])])
                    iz2[i1 - 1][i2 - 1] = np.array([np.min(z2[mask2]), np.max(z2[mask2])])

        return ir, iz1, iz2

    @staticmethod
    def _linlogspace(xmin: float,
            xmax: float,
            n: int,
            key: str,
            x0: float = 0.0) -> np.ndarray:
        if key == 'lin':
            return np.linspace(xmin, xmax, n)
        else:
            return x0 + np.logspace(np.log10(xmin - x0), np.log10(xmax - x0), n)

    def _zlinlogspace(self,
            zmin: float,
            zmax: float,
            n: int,
            key: str) -> np.ndarray:
        if key == 'lin':
            return np.linspace(zmin, zmax, n)

        medium, _ = self.indlayer(np.array([zmin]))
        medium = int(np.atleast_1d(medium).ravel()[0])

        if medium == 1:
            # Upper layer
            return self.z[0] + np.logspace(
                np.log10(zmin - self.z[0]),
                np.log10(zmax - self.z[0]), n)
        elif medium == len(self.z) + 1:
            # Lower layer
            z = self.z[-1] - np.logspace(
                np.log10(self.z[-1] - zmax),
                np.log10(self.z[-1] - zmin), n)
            return z[::-1]
        else:
            # Intermediate layer
            zup = self.z[medium - 2]
            zlo = self.z[medium - 1]
            zmin_scaled = 2 * (zmin - zlo) / (zup - zlo) - 1
            zmax_scaled = 2 * (zmax - zlo) / (zup - zlo) - 1
            z_scaled = np.tanh(np.linspace(np.arctanh(zmin_scaled), np.arctanh(zmax_scaled), n))
            return 0.5 * (zup + zlo) + 0.5 * z_scaled * (zup - zlo)

    @staticmethod
    def options(**kwargs: Any) -> Dict[str, Any]:
        opt = {
            'ztol': 2e-2,
            'rmin': 1e-2,
            'zmin': 1e-2,
            'semi': 0.1,
            'ratio': 2.0,
            'atol': 1e-6,
            'initial_step': 1e-3,
        }
        opt.update(kwargs)
        return opt

    def __repr__(self) -> str:
        return 'LayerStructure(n_layers={}, z={})'.format(len(self.z), self.z.tolist())
