import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .spectrum_ret import trisphere_unit, _PinftyStruct
from ..greenfun import CompStruct


class SpectrumRetLayer(object):

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

        z = self.nvec[:, 2]
        self.ind_up = np.where(z >= 0)[0]
        self.ind_down = np.where(z < 0)[0]

        self.nvec_up = self.nvec[self.ind_up]
        self.area_up = self.area[self.ind_up]
        self.nvec_down = self.nvec[self.ind_down]
        self.area_down = self.area[self.ind_down]

    def farfield(self,
            sig: Any,
            direction: Optional[np.ndarray] = None) -> CompStruct:

        # MATLAB: spectrumretlayer uses same structure as spectrumret
        # but accounts for Fresnel coefficients at the layer interface

        if direction is None:
            direction = self.nvec

        p = sig.p if hasattr(sig, 'p') else sig['p']
        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        # Wavenumber
        _, k = p.eps[self.medium - 1](enei)
        k0 = 2 * np.pi / enei

        pos = p.pos
        area_p = p.area

        # Get charges and currents
        sig1 = sig.sig1 if hasattr(sig, 'sig1') else np.zeros(p.nfaces)
        sig2 = sig.sig2 if hasattr(sig, 'sig2') else np.zeros(p.nfaces)
        h1 = sig.h1 if hasattr(sig, 'h1') else np.zeros((p.nfaces, 3))
        h2 = sig.h2 if hasattr(sig, 'h2') else np.zeros((p.nfaces, 3))

        if sig1.ndim == 1:
            sig1 = sig1[:, np.newaxis]
            sig2 = sig2[:, np.newaxis]
        if h1.ndim == 2:
            h1 = h1[:, :, np.newaxis]
            h2 = h2[:, :, np.newaxis]

        npol = sig1.shape[1] if sig1.ndim > 1 else 1
        ndir = len(direction)

        e = np.zeros((ndir, 3, npol), dtype = complex)
        h = np.zeros((ndir, 3, npol), dtype = complex)

        # Phase factor
        phase = np.exp(-1j * k * np.dot(direction, pos.T)) * area_p  # (ndir, nfaces)

        if phase.ndim == 1:
            phase = phase.reshape(1, -1)

        # Layer Fresnel coefficients
        layer = self.layer
        if layer is not None:
            eps1_val, k1 = layer.eps[0](enei)
            eps2_val, k2 = layer.eps[1](enei)
            n1 = np.sqrt(eps1_val)
            n2 = np.sqrt(eps2_val)
        else:
            n1 = np.sqrt(1.0)
            n2 = n1

        for ipol in range(npol):
            # Direct far-field contribution
            # Current term
            h_term = 1j * k0 * np.dot(phase, h1[:, :, ipol])  # (ndir, 3)
            h_term += 1j * k0 * np.dot(phase, h2[:, :, ipol])

            # Charge term
            sig_term = np.dot(phase, sig1[:, ipol]) + np.dot(phase, sig2[:, ipol])  # (ndir,)
            e_term = -1j * k * direction * sig_term[:, np.newaxis]

            e[:, :, ipol] = h_term + e_term

            # Magnetic field
            h[:, :, ipol] = 1j * k * np.cross(
                direction,
                np.dot(phase, h1[:, :, ipol]) + np.dot(phase, h2[:, :, ipol]))

        # Apply Fresnel modifications for layer
        if layer is not None:
            z_layer = layer.z[0]

            for idir in range(ndir):
                cos_theta = np.abs(direction[idir, 2])
                sin_theta = np.sqrt(1 - cos_theta ** 2)

                if direction[idir, 2] >= 0:
                    # Upper hemisphere: add reflected contribution
                    cos_theta_t = np.sqrt(1 - (n1 / n2 * sin_theta) ** 2 + 0j)
                    rs = (n1 * cos_theta - n2 * cos_theta_t) / (n1 * cos_theta + n2 * cos_theta_t)
                    rp = (n2 * cos_theta - n1 * cos_theta_t) / (n2 * cos_theta + n1 * cos_theta_t)

                    # Phase from reflection off substrate
                    for ipol in range(npol):
                        e_refl = e[idir, :, ipol].copy()
                        # Apply average Fresnel coefficient
                        r_avg = 0.5 * (rs + rp)
                        e[idir, :, ipol] += r_avg * e_refl * np.exp(-2j * k * z_layer * cos_theta)
                else:
                    # Lower hemisphere: transmitted contribution
                    cos_theta_i = np.sqrt(1 - (n2 / n1 * sin_theta) ** 2 + 0j)
                    ts = 2 * n1 * cos_theta_i / (n1 * cos_theta_i + n2 * cos_theta)
                    tp = 2 * n1 * cos_theta_i / (n2 * cos_theta_i + n1 * cos_theta)

                    for ipol in range(npol):
                        t_avg = 0.5 * (ts + tp)
                        e[idir, :, ipol] *= t_avg

        if npol == 1:
            e = e[:, :, 0]
            h = h[:, :, 0]

        field = CompStruct(self.pinfty, enei, e = e, h = h)
        return field

    def scattering(self,
            sig: Any) -> Tuple[np.ndarray, Any]:

        field = self.farfield(sig)
        e = field.e
        h_field = field.h
        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        if e.ndim == 2:
            e = e[:, :, np.newaxis]
            h_field = h_field[:, :, np.newaxis]

        npol = e.shape[2]

        dsca_arr = np.zeros((self.ndir, npol))

        for ipol in range(npol):
            poynting = np.cross(e[:, :, ipol], np.conj(h_field[:, :, ipol]))
            dsca_arr[:, ipol] = 0.5 * np.real(np.sum(self.nvec * poynting, axis = 1))

        sca = np.dot(self.area, dsca_arr)

        if npol == 1:
            sca = sca[0]
            dsca_arr = dsca_arr[:, 0]

        dsca = CompStruct(field.p, enei, dsca = dsca_arr)

        return sca, dsca

    def __repr__(self) -> str:
        return 'SpectrumRetLayer(ndir={}, medium={})'.format(self.ndir, self.medium)
