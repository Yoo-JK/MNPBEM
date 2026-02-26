import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .spectrum_ret import trisphere_unit, _PinftyStruct
from ..greenfun import CompStruct


class SpectrumStatLayer(object):

    def __init__(self,
            pinfty: Optional[Any] = None,
            layer: Optional[Any] = None,
            medium: int = 1) -> None:

        self.medium = medium
        self.layer = layer

        # Handle different input types
        if pinfty is None:
            _, _, nvec, area = trisphere_unit(256)
            self.pinfty = _PinftyStruct(nvec, area)
        elif isinstance(pinfty, int):
            _, _, nvec, area = trisphere_unit(pinfty)
            self.pinfty = _PinftyStruct(nvec, area)
        elif isinstance(pinfty, np.ndarray):
            nvec = np.atleast_2d(pinfty)
            area = np.full(nvec.shape[0], 4 * np.pi / nvec.shape[0])
            self.pinfty = _PinftyStruct(nvec, area)
        elif hasattr(pinfty, 'nvec') and hasattr(pinfty, 'area'):
            self.pinfty = pinfty
        else:
            _, _, nvec, area = trisphere_unit(256)
            self.pinfty = _PinftyStruct(nvec, area)

        self.nvec = self.pinfty.nvec if hasattr(self.pinfty, 'nvec') else self.pinfty['nvec']
        self.area = self.pinfty.area if hasattr(self.pinfty, 'area') else self.pinfty['area']
        self.ndir = len(self.nvec)

        # Separate into upper and lower hemisphere
        self._init_hemispheres()

    def _init_hemispheres(self) -> None:

        # MATLAB: spectrumstatlayer/init.m
        # Upper hemisphere (z > 0) -> medium above layer
        # Lower hemisphere (z < 0) -> medium below layer

        z = self.nvec[:, 2]
        self.ind_up = np.where(z >= 0)[0]
        self.ind_down = np.where(z < 0)[0]

        self.nvec_up = self.nvec[self.ind_up]
        self.area_up = self.area[self.ind_up]
        self.nvec_down = self.nvec[self.ind_down]
        self.area_down = self.area[self.ind_down]

    def efarfield(self,
            sig: Any,
            enei: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB: spectrumstatlayer/efarfield.m
        # Compute electric far-fields using Novotny & Hecht Eqs. 10.31-38
        # with Fresnel coefficients for upper and lower hemispheres

        if enei is None:
            enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        p = sig.p if hasattr(sig, 'p') else sig['p']
        surface_charge = sig.sig if hasattr(sig, 'sig') else sig['sig']

        if surface_charge.ndim == 1:
            surface_charge = surface_charge[:, np.newaxis]
        npol = surface_charge.shape[1]

        pos = p.pos
        area = p.area

        # Induced dipole moment
        weighted_pos = area[:, np.newaxis] * pos  # (nfaces, 3)
        dip = weighted_pos.T @ surface_charge  # (3, npol)

        layer = self.layer
        if layer is None:
            # No layer: use standard far-field
            return self._farfield_free(dip, enei, npol)

        # Get dielectric functions
        eps1, k1 = layer.eps[0](enei)
        eps2, k2 = layer.eps[1](enei)
        n1 = np.sqrt(eps1)
        n2 = np.sqrt(eps2)

        k0 = 2 * np.pi / enei

        # Electric far-field for each direction
        e_total = np.zeros((self.ndir, 3, npol), dtype = complex)

        # Upper hemisphere: use Fresnel transmission/reflection coefficients
        for idx_dir, idir in enumerate(self.ind_up):
            dir_vec = self.nvec[idir]
            cos_theta = np.abs(dir_vec[2])
            sin_theta = np.sqrt(1 - cos_theta ** 2)

            # Fresnel coefficients for upper medium
            cos_theta_t = np.sqrt(1 - (n1 / n2 * sin_theta) ** 2 + 0j)

            rs = (n1 * cos_theta - n2 * cos_theta_t) / (n1 * cos_theta + n2 * cos_theta_t)
            rp = (n2 * cos_theta - n1 * cos_theta_t) / (n2 * cos_theta + n1 * cos_theta_t)

            for ipol in range(npol):
                dip_i = dip[:, ipol]

                # Far-field: E ~ k^2 * dir x (dir x dip)
                cross1 = np.cross(dir_vec, dip_i)
                e_ff = k1 ** 2 * np.cross(cross1, dir_vec) / eps1

                # Add reflected contribution
                dir_refl = dir_vec.copy()
                dir_refl[2] = -dir_refl[2]

                # Decompose dipole into TE/TM
                dip_te = np.array([0.0, 0.0, 0.0], dtype = complex)
                dip_tm = dip_i.copy()

                if sin_theta > 1e-10:
                    # TE direction perpendicular to plane of incidence
                    te_dir = np.cross(dir_vec, np.array([0, 0, 1]))
                    te_dir = te_dir / np.linalg.norm(te_dir)
                    dip_te = np.dot(dip_i, te_dir) * te_dir
                    dip_tm = dip_i - dip_te

                # Reflected far-field
                dip_refl = rs * dip_te + rp * dip_tm
                dip_refl[2] = -dip_refl[2]

                cross_refl = np.cross(dir_vec, dip_refl)
                e_refl = k1 ** 2 * np.cross(cross_refl, dir_vec) / eps1

                e_total[idir, :, ipol] = e_ff + e_refl

        # Lower hemisphere
        for idx_dir, idir in enumerate(self.ind_down):
            dir_vec = self.nvec[idir]
            cos_theta = np.abs(dir_vec[2])
            sin_theta = np.sqrt(1 - cos_theta ** 2)

            cos_theta_i = np.sqrt(1 - (n2 / n1 * sin_theta) ** 2 + 0j)

            # Transmission coefficients
            ts = 2 * n1 * cos_theta_i / (n1 * cos_theta_i + n2 * cos_theta)
            tp = 2 * n1 * cos_theta_i / (n2 * cos_theta_i + n1 * cos_theta)

            for ipol in range(npol):
                dip_i = dip[:, ipol]

                # Transmitted far-field
                cross1 = np.cross(dir_vec, dip_i)
                e_ff = k2 ** 2 * np.cross(cross1, dir_vec) / eps2

                # Apply transmission coefficients
                if sin_theta > 1e-10:
                    te_dir = np.cross(dir_vec, np.array([0, 0, 1]))
                    te_dir = te_dir / np.linalg.norm(te_dir)
                    dip_te = np.dot(dip_i, te_dir) * te_dir
                    dip_tm = dip_i - dip_te
                else:
                    dip_te = dip_i
                    dip_tm = np.zeros(3, dtype = complex)

                dip_trans = ts * dip_te + tp * dip_tm

                cross_trans = np.cross(dir_vec, dip_trans)
                e_trans = k2 ** 2 * np.cross(cross_trans, dir_vec) / eps2

                e_total[idir, :, ipol] = e_trans

        return e_total, dip

    def _farfield_free(self,
            dip: np.ndarray,
            enei: float,
            npol: int) -> Tuple[np.ndarray, np.ndarray]:

        layer = self.layer
        if layer is not None:
            eps_val, k = layer.eps[0](enei)
        else:
            k = 2 * np.pi / enei
            eps_val = 1.0
        nb = np.sqrt(eps_val)

        e = np.zeros((self.ndir, 3, npol), dtype = complex)

        for ipol in range(npol):
            dip_i = dip[:, ipol]
            dir_expanded = self.nvec
            dip_expanded = np.tile(dip_i, (self.ndir, 1))

            cross1 = np.cross(dir_expanded, dip_expanded)
            e[:, :, ipol] = k ** 2 * np.cross(cross1, dir_expanded) / eps_val

        return e, dip

    def farfield(self,
            sig: Any,
            direction: Optional[np.ndarray] = None) -> CompStruct:

        e_total, dip = self.efarfield(sig)
        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        npol = e_total.shape[2]
        if npol == 1:
            e_total = e_total[:, :, 0]

        # Compute magnetic field from electric field
        # H ~ nb * dir x E
        if self.layer is not None:
            eps_val, k = self.layer.eps[0](enei)
        else:
            k = 2 * np.pi / enei
            eps_val = 1.0
        nb = np.sqrt(eps_val)

        if e_total.ndim == 2:
            h = nb * np.cross(self.nvec, e_total) / (k ** 2 / eps_val) * k ** 2 / nb
        else:
            h = np.zeros_like(e_total)
            for ipol in range(npol):
                h[:, :, ipol] = nb * np.cross(self.nvec, e_total[:, :, ipol])

        field = CompStruct(self.pinfty, enei, e = e_total, h = h,
            nvec = self.nvec, area = self.area, k = k)
        return field

    def scattering(self,
            sig: Any) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB: spectrumstatlayer/scattering.m
        # Integrate |E|^2 weighted by 0.5*k/k0 over the sphere

        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']
        e_total, _ = self.efarfield(sig)

        k0 = 2 * np.pi / enei

        npol = e_total.shape[2]

        dsca = np.zeros((self.ndir, npol))

        for ipol in range(npol):
            e_pol = e_total[:, :, ipol]  # (ndir, 3)

            # Differential scattering: |E|^2
            dsca[:, ipol] = 0.5 * np.real(np.sum(e_pol * np.conj(e_pol), axis = 1))

            # Weight by k/k0 for the appropriate medium
            if self.layer is not None:
                for idir in range(self.ndir):
                    if self.nvec[idir, 2] >= 0:
                        _, k = self.layer.eps[0](enei)
                    else:
                        _, k = self.layer.eps[1](enei)
                    dsca[idir, ipol] *= np.real(k) / k0

        # Total scattering: integrate over sphere
        sca = np.dot(self.area, dsca)

        if npol == 1:
            sca = sca[0]
            dsca = dsca[:, 0]

        return sca, dsca

    def __repr__(self) -> str:
        return 'SpectrumStatLayer(ndir={}, medium={})'.format(self.ndir, self.medium)
