import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct


class PlaneWaveStatLayer(object):

    name = 'planewave'
    needs = {'sim': 'stat'}

    def __init__(self,
            pol: np.ndarray,
            layer: Any,
            medium: int = 1,
            **options: Any) -> None:

        self.pol = np.asarray(pol)
        if self.pol.ndim == 1:
            self.pol = self.pol.reshape(1, -1)

        self.layer = layer
        self.medium = options.get('medium', medium)

    def decompose(self,
            pol: np.ndarray,
            dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: planewavestatlayer/decompose.m
        # Decompose polarization into TE and TM components
        # z-axis is the layer normal

        pol = np.asarray(pol)
        dir = np.asarray(dir)

        if pol.ndim == 1:
            pol = pol.reshape(1, -1)
        if dir.ndim == 1:
            dir = dir.reshape(1, -1)

        npol = pol.shape[0]

        # TE: polarization perpendicular to plane of incidence (containing dir and z)
        # TM: polarization in the plane of incidence
        z_hat = np.array([0.0, 0.0, 1.0])

        pol_te = np.zeros_like(pol)
        pol_tm = np.zeros_like(pol)

        for i in range(npol):
            # Plane of incidence normal
            n_inc = np.cross(dir[i], z_hat)
            n_inc_norm = np.linalg.norm(n_inc)

            if n_inc_norm < 1e-10:
                # Normal incidence: TE and TM are degenerate
                pol_te[i] = pol[i]
                pol_tm[i] = np.zeros(3)
            else:
                n_inc = n_inc / n_inc_norm
                # TE component: along n_inc
                pol_te[i] = np.dot(pol[i], n_inc) * n_inc
                # TM component: remainder
                pol_tm[i] = pol[i] - pol_te[i]

        return pol_te, pol_tm, dir

    def fresnel(self,
            dir: np.ndarray,
            enei: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: planewavestatlayer/fresnel.m
        # Compute Fresnel reflection and transmission coefficients

        dir = np.asarray(dir)
        if dir.ndim == 1:
            dir = dir.reshape(1, -1)

        layer = self.layer
        npol = dir.shape[0]

        # Get dielectric functions
        eps1, _ = layer.eps[0](enei)
        eps2, _ = layer.eps[1](enei)

        rp = np.zeros(npol, dtype = complex)
        rs = np.zeros(npol, dtype = complex)
        tp = np.zeros(npol, dtype = complex)
        ts = np.zeros(npol, dtype = complex)

        for i in range(npol):
            # Angle of incidence from z-component
            cos_theta = np.abs(dir[i, 2])
            sin_theta = np.sqrt(1 - cos_theta ** 2)

            # Transmitted angle (Snell's law in quasistatic: sqrt(eps))
            n1 = np.sqrt(eps1)
            n2 = np.sqrt(eps2)
            sin_theta_t = n1 / n2 * sin_theta
            cos_theta_t = np.sqrt(1 - sin_theta_t ** 2 + 0j)

            # Fresnel coefficients
            # s-polarization (TE)
            rs[i] = (n1 * cos_theta - n2 * cos_theta_t) / (n1 * cos_theta + n2 * cos_theta_t)
            ts[i] = 2 * n1 * cos_theta / (n1 * cos_theta + n2 * cos_theta_t)

            # p-polarization (TM)
            rp[i] = (n2 * cos_theta - n1 * cos_theta_t) / (n2 * cos_theta + n1 * cos_theta_t)
            tp[i] = 2 * n1 * cos_theta / (n2 * cos_theta + n1 * cos_theta_t)

        return rp, rs, tp, ts

    def field(self,
            p: Any,
            enei: float) -> CompStruct:

        # MATLAB: planewavestatlayer/field.m
        # Electric field including reflected and transmitted components

        n = p.n if hasattr(p, 'n') else p.nfaces
        npol = self.pol.shape[0]

        layer = self.layer
        z_layer = layer.z[0]

        e = np.zeros((n, 3, npol), dtype = complex)

        # Propagation direction (normal incidence for quasistatic)
        dir = np.zeros((npol, 3))
        dir[:, 2] = -1.0  # downward propagation

        rp, rs, tp, ts = self.fresnel(dir, enei)

        pol_te, pol_tm, _ = self.decompose(self.pol, dir)

        pos = p.pos if hasattr(p, 'pos') else p.pc.pos

        for ipol in range(npol):
            # Determine if each face is above or below layer
            above = pos[:, 2] > z_layer
            below = ~above

            # Above the layer: incident + reflected
            # Incident field = pol
            # Reflected field: TE reflected with rs, TM reflected with rp
            e_inc = self.pol[ipol]
            e_refl = rs[ipol] * pol_te[ipol] + rp[ipol] * pol_tm[ipol]
            e_refl[2] = -e_refl[2]  # z-component flips for reflection

            e[above, :, ipol] = e_inc + e_refl

            # Below the layer: transmitted
            e_trans = ts[ipol] * pol_te[ipol] + tp[ipol] * pol_tm[ipol]
            e[below, :, ipol] = e_trans

        if npol == 1:
            e = e[:, :, 0]

        return CompStruct(p, enei, e = e)

    def potential(self,
            p: Any,
            enei: float) -> CompStruct:

        # MATLAB: planewavestatlayer/potential.m
        # Surface derivative of scalar potential: phip = -nvec . E

        exc = self.field(p, enei)
        e = exc.e

        nvec = p.nvec if hasattr(p, 'nvec') else p.pc.nvec

        if e.ndim == 2:
            phip = -np.sum(nvec * e, axis = 1)
        else:
            npol = e.shape[2]
            phip = np.zeros((nvec.shape[0], npol), dtype = complex)
            for ipol in range(npol):
                phip[:, ipol] = -np.sum(nvec * e[:, :, ipol], axis = 1)

        return CompStruct(p, enei, phip = phip)

    def absorption(self,
            sig: CompStruct) -> np.ndarray:

        # Induced dipole moment
        area_pos = sig.p.area[:, np.newaxis] * sig.p.pos

        if sig.sig.ndim == 1:
            dip = area_pos.T @ sig.sig
            dip = dip.reshape(3, 1)
        else:
            dip = area_pos.T @ sig.sig

        eps_func = sig.p.eps[self.medium - 1]
        eps_val, k = eps_func(sig.enei)

        pol_dot_dip = np.sum(self.pol * dip.T, axis = 1)
        abs_cs = 4 * np.pi * k * np.imag(pol_dot_dip)

        if abs_cs.size == 1:
            return abs_cs[0]
        return abs_cs

    def scattering(self,
            sig: CompStruct) -> np.ndarray:

        # Induced dipole moment
        area_pos = sig.p.area[:, np.newaxis] * sig.p.pos

        if sig.sig.ndim == 1:
            dip = area_pos.T @ sig.sig
            dip = dip.reshape(3, 1)
        else:
            dip = area_pos.T @ sig.sig

        eps_func = sig.p.eps[self.medium - 1]
        eps_val, k = eps_func(sig.enei)
        k = np.real(k)

        sca = 8 * np.pi / 3 * k ** 4 * np.sum(np.abs(dip) ** 2, axis = 0)
        sca = np.real(sca)

        if sca.size == 1:
            return float(sca[0])
        return sca

    def extinction(self,
            sig: CompStruct) -> np.ndarray:

        return self.scattering(sig) + self.absorption(sig)

    def __call__(self,
            p: Any,
            enei: float) -> CompStruct:

        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'PlaneWaveStatLayer(pol={}, medium={})'.format(
            self.pol.tolist(), self.medium)
