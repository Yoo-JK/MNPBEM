import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct


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

        # Spectrum for radiative decay rate
        if pinfty is not None:
            from ..spectrum import SpectrumRetLayer
            self.spec = SpectrumRetLayer(pinfty, self.layer, medium = medium)
        else:
            self.spec = None
        self._pinfty = pinfty

    def field(self,
            p: Any,
            enei: float,
            inout: int = 1) -> CompStruct:

        pt = self.pt
        pos1 = p.pos if hasattr(p, 'pos') else p.pc.pos
        pos2 = pt.pos
        ndip = self.dip.shape[2]
        n = pos1.shape[0]

        exc = CompStruct(p, enei)
        exc.e = np.zeros((n, 3, pt.n, ndip), dtype = complex)
        exc.h = np.zeros((n, 3, pt.n, ndip), dtype = complex)

        # Direct contribution
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        # Use medium index for direct contribution
        eps_med = eps_vals[self._medium - 1]
        k_med = k_vals[self._medium - 1]

        e_direct, h_direct = self._dipolefield(pos1, pos2, self.dip, eps_med, k_med)
        exc.e += e_direct
        exc.h += h_direct

        # Reflected contribution via layer
        # Image method for retarded case
        z_layer = self.layer.z[0]
        pos2_image = pos2.copy()
        pos2_image[:, 2] = 2 * z_layer - pos2[:, 2]

        # Image dipole with Fresnel-modified moments
        eps1, _ = self.layer.eps[0](enei)
        eps2, _ = self.layer.eps[1](enei)
        n1 = np.sqrt(eps1)
        n2 = np.sqrt(eps2)

        # Simplified image coefficients for retarded case
        rp = (n2 * 1.0 - n1 * 1.0) / (n2 * 1.0 + n1 * 1.0)  # normal incidence approx
        rs = (n1 * 1.0 - n2 * 1.0) / (n1 * 1.0 + n2 * 1.0)

        dip_image = self.dip.astype(complex).copy()
        dip_image[:, 0, :] *= rs  # x (TE-like)
        dip_image[:, 1, :] *= rs  # y (TE-like)
        dip_image[:, 2, :] *= -rp  # z (TM-like, sign flip)

        e_refl, h_refl = self._dipolefield(pos1, pos2_image, dip_image, eps_med, k_med)
        exc.e += e_refl
        exc.h += h_refl

        return exc

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

        pt = self.pt
        pos1 = p.pos if hasattr(p, 'pos') else p.pc.pos
        nvec = p.nvec if hasattr(p, 'nvec') else p.pc.nvec
        n = pos1.shape[0]
        ndip = self.dip.shape[2]

        exc = CompStruct(p, enei)
        exc.phi1 = np.zeros((n, pt.n, ndip), dtype = complex)
        exc.phi1p = np.zeros((n, pt.n, ndip), dtype = complex)
        exc.phi2 = np.zeros((n, pt.n, ndip), dtype = complex)
        exc.phi2p = np.zeros((n, pt.n, ndip), dtype = complex)
        exc.a1 = np.zeros((n, 3, pt.n, ndip), dtype = complex)
        exc.a1p = np.zeros((n, 3, pt.n, ndip), dtype = complex)
        exc.a2 = np.zeros((n, 3, pt.n, ndip), dtype = complex)
        exc.a2p = np.zeros((n, 3, pt.n, ndip), dtype = complex)

        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        eps_med = eps_vals[self._medium - 1]
        k_med = k_vals[self._medium - 1]

        # Direct + reflected potential
        for inout in range(2):
            phi, phip, a, ap = self._pot(
                pos1, pt.pos, nvec, self.dip, eps_med, k_med)

            # Reflected contribution
            z_layer = self.layer.z[0]
            pos2_image = pt.pos.copy()
            pos2_image[:, 2] = 2 * z_layer - pos2_image[:, 2]

            eps1, _ = self.layer.eps[0](enei)
            eps2, _ = self.layer.eps[1](enei)
            n1 = np.sqrt(eps1)
            n2 = np.sqrt(eps2)
            rp = (n2 - n1) / (n2 + n1)
            rs = (n1 - n2) / (n1 + n2)

            dip_image = self.dip.astype(complex).copy()
            dip_image[:, 0, :] *= rs
            dip_image[:, 1, :] *= rs
            dip_image[:, 2, :] *= -rp

            phi_r, phip_r, a_r, ap_r = self._pot(
                pos1, pos2_image, nvec, dip_image, eps_med, k_med)

            phi_total = phi + phi_r
            phip_total = phip + phip_r
            a_total = a + a_r
            ap_total = ap + ap_r

            if inout == 0:
                exc.phi1 = phi_total
                exc.phi1p = phip_total
                exc.a1 = a_total
                exc.a1p = ap_total
            else:
                exc.phi2 = phi_total
                exc.phi2p = phip_total
                exc.a2 = a_total
                exc.a2p = ap_total

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
