import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct


class PlaneWaveRetLayer(object):

    name = 'planewave'
    needs = {'sim': 'ret'}

    def __init__(self,
            pol: np.ndarray,
            dir: np.ndarray,
            layer: Any,
            medium: int = 1,
            **options: Any) -> None:

        self.pol = np.asarray(pol)
        self.dir = np.asarray(dir)

        if self.pol.ndim == 1:
            self.pol = self.pol.reshape(1, -1)
        if self.dir.ndim == 1:
            self.dir = self.dir.reshape(1, -1)

        self.layer = layer
        self.medium = options.get('medium', medium)

        # Spectrum for scattering calculations
        pinfty_arg = options.get('pinfty', None)
        if pinfty_arg is not None:
            from ..spectrum import SpectrumRetLayer
            self.spec = SpectrumRetLayer(pinfty_arg, layer, medium = self.medium)
        else:
            self.spec = None

    def _fresnel_layer(self,
            dir: np.ndarray,
            enei: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        layer = self.layer
        npol = dir.shape[0]

        eps_vals = []
        k_vals = []
        for eps_func in layer.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        k0 = 2 * np.pi / enei

        rp = np.zeros(npol, dtype = complex)
        rs = np.zeros(npol, dtype = complex)
        tp = np.zeros(npol, dtype = complex)
        ts = np.zeros(npol, dtype = complex)

        for i in range(npol):
            cos_theta = np.abs(dir[i, 2])
            sin_theta = np.sqrt(1 - cos_theta ** 2)

            n1 = np.sqrt(eps_vals[0])
            n2 = np.sqrt(eps_vals[1])

            sin_theta_t = n1 / n2 * sin_theta
            cos_theta_t = np.sqrt(1 - sin_theta_t ** 2 + 0j)

            # Fresnel coefficients
            rs[i] = (n1 * cos_theta - n2 * cos_theta_t) / (n1 * cos_theta + n2 * cos_theta_t)
            ts[i] = 2 * n1 * cos_theta / (n1 * cos_theta + n2 * cos_theta_t)

            rp[i] = (n2 * cos_theta - n1 * cos_theta_t) / (n2 * cos_theta + n1 * cos_theta_t)
            tp[i] = 2 * n1 * cos_theta / (n2 * cos_theta + n1 * cos_theta_t)

        return rp, rs, tp, ts

    def _decompose_pol(self,
            pol: np.ndarray,
            dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        npol = pol.shape[0]
        z_hat = np.array([0.0, 0.0, 1.0])

        pol_te = np.zeros_like(pol)
        pol_tm = np.zeros_like(pol)

        for i in range(npol):
            n_inc = np.cross(dir[i], z_hat)
            n_inc_norm = np.linalg.norm(n_inc)

            if n_inc_norm < 1e-10:
                pol_te[i] = pol[i]
                pol_tm[i] = np.zeros(3)
            else:
                n_inc = n_inc / n_inc_norm
                pol_te[i] = np.dot(pol[i], n_inc) * n_inc
                pol_tm[i] = pol[i] - pol_te[i]

        return pol_te, pol_tm

    def field(self,
            p: Any,
            enei: float,
            inout: int = 1) -> CompStruct:

        # Refractive index
        eps_func = p.eps[self.medium - 1]
        eps_val, _ = eps_func(enei)
        nb = np.sqrt(eps_val)

        k0 = 2 * np.pi / enei
        k = k0 * nb

        pol = self.pol
        dir = self.dir
        npol = pol.shape[0]

        n = p.n if hasattr(p, 'n') else p.nfaces
        e = np.zeros((n, 3, npol), dtype = complex)
        h = np.zeros((n, 3, npol), dtype = complex)

        pos = p.pos if hasattr(p, 'pos') else p.pc.pos

        # Fresnel coefficients
        rp, rs, tp, ts = self._fresnel_layer(dir, enei)
        pol_te, pol_tm = self._decompose_pol(pol, dir)

        layer = self.layer
        z_layer = layer.z[0]

        for i in range(npol):
            phase_inc = np.exp(1j * k * pos @ dir[i])

            # Reflected direction
            dir_refl = dir[i].copy()
            dir_refl[2] = -dir_refl[2]

            # Reflected phase
            z_shift = 2 * z_layer * dir[i, 2]
            phase_refl = np.exp(1j * k * (pos @ dir_refl + z_shift))

            # Reflected polarization
            pol_refl_te = rs[i] * pol_te[i]
            pol_refl_tm = rp[i] * pol_tm[i].copy()
            pol_refl_tm[2] = -pol_refl_tm[2]  # z-component flips
            pol_refl = pol_refl_te + pol_refl_tm

            above = pos[:, 2] > z_layer
            below = ~above

            # Above: incident + reflected
            e_inc = phase_inc[above, np.newaxis] * pol[i]
            e_refl = phase_refl[above, np.newaxis] * pol_refl
            e[above, :, i] = e_inc + e_refl

            # Magnetic field
            dir_rep_inc = np.tile(dir[i], (np.sum(above), 1))
            dir_rep_refl = np.tile(dir_refl, (np.sum(above), 1))
            h[above, :, i] = nb * (np.cross(dir_rep_inc, e_inc) +
                np.cross(dir_rep_refl, e_refl))

            # Below: transmitted
            # Transmitted direction
            eps2, _ = layer.eps[1](enei)
            n2 = np.sqrt(eps2)
            k_trans = k0 * n2

            sin_theta = np.sqrt(dir[i, 0] ** 2 + dir[i, 1] ** 2)
            cos_theta_t = np.sqrt(1 - (nb / n2 * sin_theta) ** 2 + 0j)
            dir_trans = dir[i].copy()
            if sin_theta > 1e-10:
                dir_trans[0] = dir[i, 0] * nb / n2
                dir_trans[1] = dir[i, 1] * nb / n2
            dir_trans[2] = -np.real(cos_theta_t) * np.sign(dir[i, 2])

            pol_trans = ts[i] * pol_te[i] + tp[i] * pol_tm[i]
            phase_trans = np.exp(1j * k_trans * pos[below] @ dir_trans)
            e[below, :, i] = phase_trans[:, np.newaxis] * pol_trans

            dir_rep_trans = np.tile(dir_trans, (np.sum(below), 1))
            h[below, :, i] = n2 * np.cross(dir_rep_trans, e[below, :, i])

        if npol == 1:
            e = e[:, :, 0]
            h = h[:, :, 0]

        return CompStruct(p, enei, e = e, h = h)

    def potential(self,
            p: Any,
            enei: float) -> CompStruct:

        eps_func = p.eps[self.medium - 1]
        eps_val, _ = eps_func(enei)
        nb = np.sqrt(eps_val)

        k0 = 2 * np.pi / enei
        k = k0 * nb

        pol = self.pol
        dir = self.dir
        npol = pol.shape[0]
        nfaces = p.nfaces if hasattr(p, 'nfaces') else p.n

        exc = CompStruct(p, enei)

        for inout in range(1, 3):
            a = np.zeros((nfaces, 3, npol), dtype = complex)
            ap = np.zeros((nfaces, 3, npol), dtype = complex)
            phi = np.zeros((nfaces, npol), dtype = complex)
            phip = np.zeros((nfaces, npol), dtype = complex)

            pos = p.pos if hasattr(p, 'pos') else p.pc.pos
            nvec = p.nvec if hasattr(p, 'nvec') else p.pc.nvec

            layer = self.layer
            z_layer = layer.z[0]

            rp, rs, tp, ts = self._fresnel_layer(dir, enei)
            pol_te, pol_tm = self._decompose_pol(pol, dir)

            for i in range(npol):
                # Phase factor for incident wave
                phase = np.exp(1j * k * pos @ dir[i]) / (1j * k0)

                # Vector potential: A = phase * pol
                a[:, :, i] = phase[:, np.newaxis] * pol[i]

                # Surface derivative
                nvec_dot_dir = nvec @ dir[i]
                ap[:, :, i] = (1j * k * nvec_dot_dir)[:, np.newaxis] * phase[:, np.newaxis] * pol[i]

                # Add reflected/transmitted contributions depending on position
                above = pos[:, 2] > z_layer

                # Reflected contribution for faces above layer
                dir_refl = dir[i].copy()
                dir_refl[2] = -dir_refl[2]
                z_shift = 2 * z_layer * dir[i, 2]
                phase_refl = np.exp(1j * k * (pos[above] @ dir_refl + z_shift)) / (1j * k0)

                pol_refl_te = rs[i] * pol_te[i]
                pol_refl_tm = rp[i] * pol_tm[i].copy()
                pol_refl_tm[2] = -pol_refl_tm[2]
                pol_refl = pol_refl_te + pol_refl_tm

                a[above, :, i] += phase_refl[:, np.newaxis] * pol_refl
                nvec_dot_dir_refl = nvec[above] @ dir_refl
                ap[above, :, i] += (1j * k * nvec_dot_dir_refl)[:, np.newaxis] * phase_refl[:, np.newaxis] * pol_refl

            if npol == 1:
                a = a[:, :, 0]
                ap = ap[:, :, 0]

            if inout == 1:
                exc = exc.set(a1 = a, a1p = ap)
            else:
                exc = exc.set(a2 = a, a2p = ap)

        return exc

    def absorption(self,
            sig: CompStruct) -> np.ndarray:

        ext = self.extinction(sig)
        sca, _ = self.scattering(sig)
        return ext - sca

    def scattering(self,
            sig: CompStruct) -> Tuple[np.ndarray, Any]:

        if self.spec is None:
            raise ValueError('[error] Scattering requires spectrum object. '
                'Provide <pinfty> in constructor.')

        sca, dsca = self.spec.scattering(sig)

        eps_func = sig.p.eps[0]
        eps_val, _ = eps_func(sig.enei)
        nb = np.real(np.sqrt(eps_val))

        sca = np.real(sca / (0.5 * nb))

        return sca, dsca

    def extinction(self,
            sig: CompStruct) -> np.ndarray:

        if self.spec is None:
            raise ValueError('[error] Extinction requires spectrum object. '
                'Provide <pinfty> in constructor.')

        field = self.spec.farfield(sig, self.dir)

        _, k = sig.p.eps[self.medium - 1](sig.enei)

        e_forward = field.e[0] if field.e.ndim >= 2 else field.e

        if e_forward.ndim == 1:
            pol_dot_e = np.sum(np.conj(self.pol[0]) * e_forward)
        else:
            pol_dot_e = np.sum(np.conj(self.pol.T) * e_forward, axis = 0)

        ext = 4 * np.pi / k * np.imag(pol_dot_e)

        if np.isscalar(ext) or (isinstance(ext, np.ndarray) and ext.size == 1):
            return float(np.real(ext)) if np.isscalar(ext) else float(np.real(ext.ravel()[0]))
        return np.real(ext)

    def __call__(self,
            p: Any,
            enei: float) -> CompStruct:

        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'PlaneWaveRetLayer(pol={}, dir={}, medium={})'.format(
            self.pol.tolist(), self.dir.tolist(), self.medium)
