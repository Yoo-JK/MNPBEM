import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct
from ..greenfun.greentab_layer import GreenTabLayer


class DipoleRetLayer(object):

    name = 'dipole'
    needs = {'sim': 'ret'}

    def __init__(self,
            pt: Any,
            layer: Any,
            dip: Optional[np.ndarray] = None,
            full: bool = False,
            medium: int = 1,
            pinfty: Optional[Any] = None,
            **options: Any) -> None:

        self.pt = pt
        self.layer = layer
        self.varargin = options

        self._init(dip, full, medium, pinfty, **options)

    def _init(self,
            dip: Optional[np.ndarray] = None,
            full: bool = False,
            medium: int = 1,
            pinfty: Optional[Any] = None,
            **options: Any) -> None:

        if dip is None:
            dip = np.eye(3)
            full = False

        dip = np.asarray(dip, dtype = float)

        if full:
            if dip.ndim == 2:
                dip = dip.reshape(dip.shape + (1,))
            self.dip = dip
        else:
            if dip.ndim == 1:
                dip = dip.reshape(1, -1)

            dip_reshaped = dip.T.reshape(1, dip.shape[1], dip.shape[0])
            self.dip = np.tile(dip_reshaped, (self.pt.n, 1, 1))

        self._medium = medium

        # Tabulated Green function for reflected contribution
        tab = options.get('tab', None)
        if tab is not None:
            self.tab = GreenTabLayer(self.layer, tab=tab)
        else:
            self.tab = GreenTabLayer(self.layer)

        # Spectrum for radiative decay rate
        from ..spectrum import SpectrumRetLayer
        self.spec = SpectrumRetLayer(pinfty, self.layer, medium=medium)
        self._pinfty = pinfty

    def field(self,
            p: Any,
            enei: float,
            inout: int = 1) -> CompStruct:
        """Electric and magnetic field for dipole excitation.

        MATLAB: @dipoleretlayer/field.m
        E = ik0*A - grad(V), H = curl(A)
        """
        pt = self.pt
        pos1 = p.pos if hasattr(p, 'pos') else p.pc.pos
        pos2 = pt.pos
        ndip = self.dip.shape[2]
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        exc = CompStruct(p, enei)
        exc.e = np.zeros((n1, 3, n2, ndip), dtype=complex)
        exc.h = np.zeros((n1, 3, n2, ndip), dtype=complex)

        # Direct contribution
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        eps_med = eps_vals[self._medium - 1]
        k_med = k_vals[self._medium - 1]

        e_direct, h_direct = self._dipolefield(pos1, pos2, self.dip, eps_med, k_med)
        exc.e += e_direct
        exc.h += h_direct

        # Reflected contribution using Green function derivatives
        k0 = 2 * np.pi / enei

        # Dielectric at dipole positions
        eps2 = np.array([eps_med] * n2, dtype=complex)

        G, F = self._greenderiv(enei, pos1, pos2)
        dip = self.dip                                      # (n2, 3, ndip)
        dip2 = dip / eps2[:, np.newaxis, np.newaxis]        # reduced dipole

        def fun(g, d):
            """g: (n1,n2), d: (n2,ndip) -> (n1,n2,ndip)"""
            if isinstance(g, (int, float)):
                return 0 if g == 0 else g * d[np.newaxis, :, :]
            return g[:, :, np.newaxis] * d[np.newaxis, :, :]

        # Vector potential A
        a1 = -1j * k0 * fun(G.get('p', 0), dip[:, 0, :])
        a2 = -1j * k0 * fun(G.get('p', 0), dip[:, 1, :])
        a3 = (-1j * k0 * fun(G.get('hh', 0), dip[:, 2, :])
              + fun(F[(1, 2)].get('hs', 0), dip2[:, 0, :])
              + fun(F[(1, 3)].get('hs', 0), dip2[:, 1, :])
              + fun(F[(1, 4)].get('hs', 0), dip2[:, 2, :]))

        # E += ik0 * A
        exc.e[:, 0, :, :] += 1j * k0 * a1
        exc.e[:, 1, :, :] += 1j * k0 * a2
        exc.e[:, 2, :, :] += 1j * k0 * a3

        # Derivatives of A for H-field: curl(A)
        # dA3/dy - dA2/dz
        a23 = (-1j * k0 * fun(F[(3, 1)].get('hh', 0), dip[:, 2, :])
              + fun(F[(3, 2)].get('hs', 0), dip2[:, 0, :])
              + fun(F[(3, 3)].get('hs', 0), dip2[:, 1, :])
              + fun(F[(3, 4)].get('hs', 0), dip2[:, 2, :]))
        a32 = -1j * k0 * fun(F[(4, 1)].get('p', 0), dip[:, 1, :])

        # dA1/dz - dA3/dx
        a31 = -1j * k0 * fun(F[(4, 1)].get('p', 0), dip[:, 0, :])
        a13 = (-1j * k0 * fun(F[(4, 1)].get('hh', 0), dip[:, 2, :])
              + fun(F[(4, 2)].get('hs', 0), dip2[:, 0, :])
              + fun(F[(4, 3)].get('hs', 0), dip2[:, 1, :])
              + fun(F[(4, 4)].get('hs', 0), dip2[:, 2, :]))

        # dA2/dx - dA1/dy
        a12 = -1j * k0 * fun(F[(2, 1)].get('p', 0), dip[:, 1, :])
        a21 = -1j * k0 * fun(F[(3, 1)].get('p', 0), dip[:, 0, :])

        # grad(V): scalar potential gradients
        phi1 = (fun(F[(2, 2)].get('ss', 0), dip2[:, 0, :])
              + fun(F[(2, 3)].get('ss', 0), dip2[:, 1, :])
              + fun(F[(2, 4)].get('ss', 0), dip2[:, 2, :])
              - 1j * k0 * fun(F[(2, 1)].get('sh', 0), dip[:, 2, :]))
        phi2 = (fun(F[(3, 2)].get('ss', 0), dip2[:, 0, :])
              + fun(F[(3, 3)].get('ss', 0), dip2[:, 1, :])
              + fun(F[(3, 4)].get('ss', 0), dip2[:, 2, :])
              - 1j * k0 * fun(F[(3, 1)].get('sh', 0), dip[:, 2, :]))
        phi3 = (fun(F[(4, 2)].get('ss', 0), dip2[:, 0, :])
              + fun(F[(4, 3)].get('ss', 0), dip2[:, 1, :])
              + fun(F[(4, 4)].get('ss', 0), dip2[:, 2, :])
              - 1j * k0 * fun(F[(1, 4)].get('sh', 0), dip[:, 2, :]))

        # E -= grad(V)
        exc.e[:, 0, :, :] -= phi1
        exc.e[:, 1, :, :] -= phi2
        exc.e[:, 2, :, :] -= phi3

        # H = curl(A) (sign convention matches MATLAB)
        exc.h[:, 0, :, :] -= (a23 - a32)
        exc.h[:, 1, :, :] -= (a31 - a13)
        exc.h[:, 2, :, :] -= (a12 - a21)

        return exc

    def _greenderiv(self,
            enei: float,
            pos1: np.ndarray,
            pos2: np.ndarray) -> Tuple[Dict, Dict]:
        """Green function and derivatives via finite differences.

        MATLAB: @dipoleretlayer/private/greenderiv.m

        Returns
        -------
        G_dict : dict
            Reflected Green function components, each (n1, n2).
        F : dict
            F[(i,j)][name] = (n1, n2) array of 2nd derivatives.
            Indices: 1=value, 2=x, 3=y, 4=z.
        """
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Handle self-interaction: perturb pos2 to avoid singular limit
        if n1 == n2 and np.allclose(pos1, pos2):
            pos2 = pos2.copy()
            pos2[:, 0] += self.layer.rmin

        # Lateral distances
        dx = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        dy = pos1[:, 1:2] - pos2[:, 1:2].T  # (n1, n2)
        r = np.sqrt(dx ** 2 + dy ** 2)

        # z-components
        z1 = np.repeat(pos1[:, 2:3], n2, axis=1)  # (n1, n2)
        z2 = np.tile(pos2[:, 2:3].T, (n1, 1))     # (n1, n2)

        rmin = self.layer.rmin
        eta = 1e-6

        # Enforce minimum distance
        r = np.maximum(r, rmin)
        # Unit vectors
        xhat = dx / r
        yhat = dy / r

        # Round z-values
        z1_r, z2_r = self.layer.round_z(z1.ravel(), z2.ravel())
        r_flat = np.maximum(r.ravel(), rmin)

        # Baseline: G, Fr, Fz at (r, z1, z2)
        G0, Fr0, Fz0 = self.tab.eval_components(enei, r_flat, z1_r, z2_r)

        # Perturbed in r: (r+eta, z1, z2)
        _, Fr_r, Fz_r = self.tab.eval_components(enei, r_flat + eta, z1_r, z2_r)

        # Perturbed in z2: (r, z1, z2+eta)
        G_z, Fr_z, Fz_z = self.tab.eval_components(enei, r_flat, z1_r, z2_r + eta)

        names = list(G0.keys())
        shape = (n1, n2)

        # Reshape Green function
        G_dict = {}
        for name in names:
            G_dict[name] = G0[name].reshape(shape)

        # Build derivative tensor F[(i,j)][name]
        F = {}
        for key in [(1, 2), (1, 3), (2, 1), (3, 1), (4, 1), (1, 4),
                     (2, 2), (2, 3), (3, 2), (3, 3),
                     (2, 4), (3, 4), (4, 2), (4, 3), (4, 4)]:
            F[key] = {}

        for name in names:
            Fr_val = Fr0[name].reshape(shape)
            Fz_val = Fz0[name].reshape(shape)

            # Finite difference derivatives
            Frr = (Fr_r[name].reshape(shape) - Fr_val) / eta
            Fr1 = (Fz_r[name].reshape(shape) - Fz_val) / eta
            F2 = (G_z[name].reshape(shape) - G_dict[name]) / eta
            Fr2 = (Fr_z[name].reshape(shape) - Fr_val) / eta
            F12 = (Fz_z[name].reshape(shape) - Fz_val) / eta

            # 1st derivatives (Cartesian)
            F[(1, 2)][name] = -Fr_val * xhat       # dG/dx'
            F[(1, 3)][name] = -Fr_val * yhat       # dG/dy'
            F[(2, 1)][name] = Fr_val * xhat        # dG/dx
            F[(3, 1)][name] = Fr_val * yhat        # dG/dy
            F[(4, 1)][name] = Fz_val               # dG/dz1
            F[(1, 4)][name] = F2                   # dG/dz2

            # 2nd derivatives
            F[(2, 2)][name] = -Fr_val * yhat ** 2 / r - Frr * xhat ** 2
            F[(2, 3)][name] = -Fr_val * xhat * yhat / r - Frr * xhat * yhat
            F[(3, 3)][name] = -Fr_val * xhat ** 2 / r - Frr * yhat ** 2
            F[(3, 2)][name] = F[(2, 3)][name]      # symmetric

            # Mixed z derivatives
            F[(2, 4)][name] = Fr2 * xhat            # d2G/dx dz2
            F[(3, 4)][name] = Fr2 * yhat            # d2G/dy dz2
            F[(4, 2)][name] = -Fr1 * xhat           # d2G/dz1 dx'
            F[(4, 3)][name] = -Fr1 * yhat           # d2G/dz1 dy'
            F[(4, 4)][name] = F12                   # d2G/dz1 dz2

        return G_dict, F

    def _dipolefield(self,
            pos1: np.ndarray,
            pos2: np.ndarray,
            dip: np.ndarray,
            eps: complex,
            k: complex) -> Tuple[np.ndarray, np.ndarray]:

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        z = pos1[:, 2:3] - pos2[:, 2].T
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.maximum(r, np.finfo(float).eps)
        x, y, z = x / r, y / r, z / r

        G = np.exp(1j * k * r) / r

        ndip = dip.shape[2]

        e = np.zeros((n1, 3, n2, ndip), dtype = complex)
        h = np.zeros((n1, 3, n2, ndip), dtype = complex)

        for i in range(ndip):
            dx = np.tile(dip[:, 0, i], (n1, 1))
            dy = np.tile(dip[:, 1, i], (n1, 1))
            dz = np.tile(dip[:, 2, i], (n1, 1))

            inner = x * dx + y * dy + z * dz

            fac = k ** 2 * G * (1 - 1 / (1j * k * r)) / np.sqrt(eps)
            h[:, 0, :, i] = fac * (y * dz - z * dy)
            h[:, 1, :, i] = fac * (z * dx - x * dz)
            h[:, 2, :, i] = fac * (x * dy - y * dx)

            fac1 = k ** 2 * G / eps
            fac2 = G * (1 / r ** 2 - 1j * k / r) / eps
            e[:, 0, :, i] = fac1 * (dx - inner * x) + fac2 * (3 * inner * x - dx)
            e[:, 1, :, i] = fac1 * (dy - inner * y) + fac2 * (3 * inner * y - dy)
            e[:, 2, :, i] = fac1 * (dz - inner * z) + fac2 * (3 * inner * z - dz)

        return e, h

    def potential(self,
            p: Any,
            enei: float) -> CompStruct:
        """Potential of dipole excitation for use in BEM.

        MATLAB: @dipoleretlayer/potential.m
        """
        pt = self.pt
        pos1 = p.pos if hasattr(p, 'pos') else p.pc.pos
        nvec = p.nvec if hasattr(p, 'nvec') else p.pc.nvec
        n1 = pos1.shape[0]
        n2 = pt.pos.shape[0]
        ndip = self.dip.shape[2]

        # Direct contribution
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        eps_med = eps_vals[self._medium - 1]
        k_med = k_vals[self._medium - 1]

        exc = CompStruct(p, enei)
        # Initialize with direct potentials (outside boundary = inout 2)
        phi_d, phip_d, a_d, ap_d = self._pot(
            pos1, pt.pos, nvec, self.dip, eps_med, k_med)
        exc.phi2 = phi_d
        exc.phi2p = phip_d
        exc.a2 = a_d
        exc.a2p = ap_d
        exc.phi1 = phi_d.copy()
        exc.phi1p = phip_d.copy()
        exc.a1 = a_d.copy()
        exc.a1p = ap_d.copy()

        # Reflected contribution using Green function derivatives
        k0 = 2 * np.pi / enei

        eps2_arr = np.array([eps_med] * n2, dtype=complex)
        G, F = self._greenderiv(enei, pos1, pt.pos)
        dip = self.dip
        dip2 = dip / eps2_arr[:, np.newaxis, np.newaxis]

        def fun(g, d):
            """g: (n1,n2), d: (n2,ndip) -> (n1,n2,ndip)"""
            if isinstance(g, (int, float)):
                return 0 if g == 0 else g * d[np.newaxis, :, :]
            return g[:, :, np.newaxis] * d[np.newaxis, :, :]

        # Vector potential: a
        a1 = -1j * k0 * fun(G.get('p', 0), dip[:, 0, :])
        a2 = -1j * k0 * fun(G.get('p', 0), dip[:, 1, :])
        a3 = (-1j * k0 * fun(G.get('hh', 0), dip[:, 2, :])
              + fun(F[(1, 2)].get('hs', 0), dip2[:, 0, :])
              + fun(F[(1, 3)].get('hs', 0), dip2[:, 1, :])
              + fun(F[(1, 4)].get('hs', 0), dip2[:, 2, :]))

        # Scalar potential: phi
        phi_r = (fun(F[(1, 2)].get('ss', 0), dip2[:, 0, :])
               + fun(F[(1, 3)].get('ss', 0), dip2[:, 1, :])
               + fun(F[(1, 4)].get('ss', 0), dip2[:, 2, :])
               - 1j * k0 * fun(G.get('sh', 0), dip[:, 2, :]))

        # Add reflected to a2, phi2
        exc.a2[:, 0, :, :] += a1
        exc.a2[:, 1, :, :] += a2
        exc.a2[:, 2, :, :] += a3
        exc.phi2 += phi_r

        # Surface derivatives: nvec dot grad
        def deriv(comp_name, col_idx):
            """Normal derivative: nvec . [F{2,j}, F{3,j}, F{4,j}]"""
            f2 = F[(2, col_idx)].get(comp_name, None)
            f3 = F[(3, col_idx)].get(comp_name, None)
            f4 = F[(4, col_idx)].get(comp_name, None)
            if f2 is None and f3 is None and f4 is None:
                return 0
            result = np.zeros((n1, n2, 1), dtype=complex)
            if f2 is not None:
                result += nvec[:, 0:1, np.newaxis] * f2[:, :, np.newaxis]
            if f3 is not None:
                result += nvec[:, 1:2, np.newaxis] * f3[:, :, np.newaxis]
            if f4 is not None:
                result += nvec[:, 2:3, np.newaxis] * f4[:, :, np.newaxis]
            return result

        def fun3(g, d):
            """g: (n1,n2,1) or 0, d: (n2,ndip) -> (n1,n2,ndip)"""
            if isinstance(g, (int, float)) and g == 0:
                return 0
            return g * d[np.newaxis, :, :]

        # Surface derivative of vector potential: ap
        a1p = -1j * k0 * fun3(deriv('p', 1), dip[:, 0, :])
        a2p = -1j * k0 * fun3(deriv('p', 1), dip[:, 1, :])
        a3p = (-1j * k0 * fun3(deriv('hh', 1), dip[:, 2, :])
              + fun3(deriv('hs', 2), dip2[:, 0, :])
              + fun3(deriv('hs', 3), dip2[:, 1, :])
              + fun3(deriv('hs', 4), dip2[:, 2, :]))

        # Surface derivative of scalar potential: phip
        phip_r = (fun3(deriv('ss', 2), dip2[:, 0, :])
                + fun3(deriv('ss', 3), dip2[:, 1, :])
                + fun3(deriv('ss', 4), dip2[:, 2, :])
                - 1j * k0 * fun3(deriv('sh', 1), dip[:, 2, :]))

        exc.a2p[:, 0, :, :] += a1p
        exc.a2p[:, 1, :, :] += a2p
        exc.a2p[:, 2, :, :] += a3p
        exc.phi2p += phip_r

        return exc

    def _pot(self,
            pos1: np.ndarray,
            pos2: np.ndarray,
            nvec: np.ndarray,
            dip: np.ndarray,
            eps: complex,
            k: complex) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        k0 = k / np.sqrt(eps)

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        z = pos1[:, 2:3] - pos2[:, 2].T
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.maximum(r, np.finfo(float).eps)
        x, y, z = x / r, y / r, z / r

        G = np.exp(1j * k * r) / r
        F = (1j * k - 1 / r) * G

        nx = np.tile(nvec[:, 0:1], (1, n2))
        ny = np.tile(nvec[:, 1:2], (1, n2))
        nz = np.tile(nvec[:, 2:3], (1, n2))
        en = nx * x + ny * y + nz * z

        ndip = dip.shape[2]
        phi = np.zeros((n1, n2, ndip), dtype = complex)
        phip = np.zeros((n1, n2, ndip), dtype = complex)
        a = np.zeros((n1, 3, n2, ndip), dtype = complex)
        ap = np.zeros((n1, 3, n2, ndip), dtype = complex)

        for i in range(ndip):
            dx = np.tile(dip[:, 0, i], (n1, 1))
            dy = np.tile(dip[:, 1, i], (n1, 1))
            dz = np.tile(dip[:, 2, i], (n1, 1))

            ep = x * dx + y * dy + z * dz
            np_dot = nx * dx + ny * dy + nz * dz

            phi[:, :, i] = -ep * F / eps
            phip[:, :, i] = (
                (np_dot - 3 * en * ep) / r ** 2 * (1 - 1j * k * r) * G / eps
                + k ** 2 * ep * en * G / eps)

            a[:, 0, :, i] = -1j * k0 * dx * G
            a[:, 1, :, i] = -1j * k0 * dy * G
            a[:, 2, :, i] = -1j * k0 * dz * G

            ap[:, 0, :, i] = -1j * k0 * dx * en * F
            ap[:, 1, :, i] = -1j * k0 * dy * en * F
            ap[:, 2, :, i] = -1j * k0 * dz * en * F

        return phi, phip, a, ap

    def decayrate(self,
            sig: CompStruct) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        p, enei = sig.p, sig.enei

        from ..greenfun import CompGreenRetLayer
        g = CompGreenRetLayer(self.pt, sig.p, self.layer, **self.varargin)

        field_struct = g.field(sig)
        e = field_struct.e

        k0 = 2 * np.pi / sig.enei
        gamma = 4 / 3 * k0 ** 3

        npt = self.pt.n
        ndip = self.dip.shape[2]
        tot = np.zeros((npt, ndip))
        rad = np.zeros((npt, ndip))
        rad0 = np.zeros((npt, ndip))

        for ipos in range(npt):
            for idip in range(ndip):
                dip = self.dip[ipos, :, idip]
                nb = np.sqrt(self.pt.eps1(sig.enei)[ipos])

                if e.ndim == 4:
                    e_i = e[ipos, :, ipos, idip]
                else:
                    e_i = e[ipos, :]

                tot[ipos, idip] = 1 + np.imag(e_i @ dip) / (0.5 * nb * gamma)
                rad0[ipos, idip] = nb * gamma

        return tot, rad, rad0

    def farfield(self,
            spec: Any,
            enei: float) -> CompStruct:

        dir = spec.pinfty.nvec if hasattr(spec.pinfty, 'nvec') else spec.nvec
        epstab = self.pt.eps
        eps_val, k = epstab[spec.medium - 1](enei) if hasattr(spec, 'medium') else epstab[0](enei)
        nb = np.sqrt(eps_val)

        pt = self.pt
        dip = self.dip.copy()

        n1 = dir.shape[0]
        n2 = dip.shape[0]
        n3 = dip.shape[2]

        e = np.zeros((n1, 3, n2, n3), dtype = complex)
        h = np.zeros((n1, 3, n2, n3), dtype = complex)

        # Direct far-field
        g = np.exp(-1j * k * (dir @ pt.pos.T))
        g = g[:, np.newaxis, :, np.newaxis]
        g = np.tile(g, (1, 3, 1, n3))

        dir_rep = dir[:, :, np.newaxis, np.newaxis]
        dir_rep = np.tile(dir_rep, (1, 1, n2, n3))

        dip_perm = dip.transpose(1, 0, 2)
        dip_rep = dip_perm[np.newaxis, :, :, :]
        dip_rep = np.tile(dip_rep, (n1, 1, 1, 1))

        h_temp = np.cross(dir_rep, dip_rep, axis = 1) * g
        e_temp = np.cross(h_temp, dir_rep, axis = 1)

        e = k ** 2 * e_temp / eps_val
        h = k ** 2 * h_temp / nb

        # Reflected far-field contribution
        # Uses Fresnel coefficients applied to image dipole
        layer = self.layer
        z_layer = layer.z[0]

        eps1, _ = layer.eps[0](enei)
        eps2, _ = layer.eps[1](enei)
        n1_layer = np.sqrt(eps1)
        n2_layer = np.sqrt(eps2)

        # For each direction on the unit sphere, compute reflected contribution
        # via stationary phase with Fresnel coefficients
        for idir in range(n1):
            cos_theta = np.abs(dir[idir, 2])
            sin_theta = np.sqrt(1 - cos_theta ** 2)

            # Fresnel coefficients for this direction
            cos_theta_t = np.sqrt(1 - (n1_layer / n2_layer * sin_theta) ** 2 + 0j)
            rs_val = (n1_layer * cos_theta - n2_layer * cos_theta_t) / (n1_layer * cos_theta + n2_layer * cos_theta_t)
            rp_val = (n2_layer * cos_theta - n1_layer * cos_theta_t) / (n2_layer * cos_theta + n1_layer * cos_theta_t)

            # Phase from reflection
            for ipt in range(n2):
                z_pt = pt.pos[ipt, 2]
                phase_refl = np.exp(-2j * k * (z_pt - z_layer) * cos_theta)

                for idip_idx in range(n3):
                    # Reflected dipole contribution
                    dip_val = dip[ipt, :, idip_idx]

                    # Decompose into TE/TM relative to propagation direction
                    # Simplified: use rs for horizontal, rp for vertical
                    dip_refl = dip_val.copy()
                    dip_refl[0] *= rs_val
                    dip_refl[1] *= rs_val
                    dip_refl[2] *= -rp_val

                    # Add reflected contribution
                    h_refl = np.cross(dir[idir], dip_refl) * phase_refl
                    e_refl = np.cross(h_refl, dir[idir])

                    e[idir, :, ipt, idip_idx] += k ** 2 * e_refl / eps_val
                    h[idir, :, ipt, idip_idx] += k ** 2 * h_refl / nb

        field = CompStruct(spec.pinfty, enei, e = e, h = h)
        return field

    def __call__(self,
            p: Any,
            enei: float) -> CompStruct:

        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'DipoleRetLayer(npt={}, ndip={})'.format(
            self.pt.n, self.dip.shape[2])
