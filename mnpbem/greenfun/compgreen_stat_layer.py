import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .compgreen_stat import CompGreenStat, CompStruct


class CompGreenStatLayer(object):

    name = 'greenfunction'
    needs = {'sim': 'stat'}

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            **options: Any) -> None:

        self.p1 = p1
        self.p2 = p2
        self.layer = layer
        self.deriv = options.get('deriv', 'norm')

        # BEM solver cache
        self._enei_cache = None
        self._mat_cache = None

        # Initialize direct and reflected Green functions
        self._init(p1, p2, layer, **options)

    def _init(self,
            p1: Any,
            p2: Any,
            layer: Any,
            **options: Any) -> None:

        # MATLAB: compgreenstatlayer uses image charge method
        # Only works for single layer (substrate) with particle in upper medium

        # Direct Green function (free-space)
        self.g = CompGreenStat(p1, p2, **options)

        # Create reflected (image) particle by mirroring across layer interface
        z_layer = layer.z[0]

        # Mirror p2 positions across the layer
        pos2 = p2.pos.copy() if hasattr(p2, 'pos') else p2.pc.pos.copy()
        pos2_reflected = pos2.copy()
        pos2_reflected[:, 2] = 2 * z_layer - pos2_reflected[:, 2]

        # Create mirrored particle for reflected Green function
        self._pos2r = pos2_reflected
        self._z_layer = z_layer

        # Compute image factors from dielectric functions
        # MATLAB: f1 = 2*eps1/(eps2+eps1), f2 = -(eps2-eps1)/(eps2+eps1)
        # These depend on enei and are computed in eval()
        self._gr_G = None
        self._gr_F = None

    def _compute_reflected_green(self,
            enei: float) -> None:

        pos1 = self.p1.pos if hasattr(self.p1, 'pos') else self.p1.pc.pos
        pos2r = self._pos2r
        nvec1 = self.p1.nvec if hasattr(self.p1, 'nvec') else self.p1.pc.nvec
        area2 = self.p2.area if hasattr(self.p2, 'area') else self.p2.pc.area

        n1 = pos1.shape[0]
        n2 = pos2r.shape[0]

        # Distance to reflected particle
        r = pos1[:, np.newaxis, :] - pos2r[np.newaxis, :, :]  # (n1, n2, 3)
        d = np.linalg.norm(r, axis = 2)  # (n1, n2)
        d_safe = np.maximum(d, np.finfo(float).eps)

        # Green function G_refl = 1/d * area
        self._gr_G = (1.0 / d_safe) * area2[np.newaxis, :]  # (n1, n2)

        # Surface derivative F_refl
        if self.deriv == 'norm':
            n_dot_r = np.sum(nvec1[:, np.newaxis, :] * r, axis = 2)  # (n1, n2)
            self._gr_F = -n_dot_r / (d_safe ** 3) * area2[np.newaxis, :]  # (n1, n2)
        else:
            self._gr_F = -r / (d_safe[:, :, np.newaxis] ** 3) * area2[np.newaxis, :, np.newaxis]

    def _image_factors(self,
            enei: float) -> Tuple[complex, complex, complex]:

        layer = self.layer

        # Get dielectric functions of upper and lower layers
        eps1_val, _ = layer.eps[0](enei)
        eps2_val, _ = layer.eps[1](enei)

        # Image charge factors (Jackson Eq. 4.45)
        f1 = 2 * eps1_val / (eps2_val + eps1_val)
        f2 = -(eps2_val - eps1_val) / (eps2_val + eps1_val)
        fl = eps1_val / eps2_val * f1

        return f1, f2, fl

    @property
    def G(self) -> np.ndarray:
        return self.g.G

    @property
    def F(self) -> np.ndarray:
        return self.g.F

    def eval(self,
            enei: float,
            key: str = 'G') -> np.ndarray:

        # Compute reflected Green function if needed
        if self._gr_G is None or (self._enei_cache is not None and not np.isclose(self._enei_cache, enei)):
            self._compute_reflected_green(enei)
            self._enei_cache = enei

        _, f2, _ = self._image_factors(enei)

        if key == 'G':
            return self.g.G + f2 * self._gr_G
        elif key == 'F':
            return self.g.F + f2 * self._gr_F
        elif key == 'H1':
            H1 = self.g._eval_H1()
            return H1 + f2 * self._gr_F
        elif key == 'H2':
            H2 = self.g._eval_H2()
            return H2 + f2 * self._gr_F
        else:
            raise ValueError('[error] Unknown Green function key: <{}>'.format(key))

    def eval_multi(self,
            enei: float,
            *keys: str) -> Tuple[np.ndarray, ...]:

        results = []
        for key in keys:
            results.append(self.eval(enei, key))

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def potential(self,
            sig: Any,
            inout: int = 1) -> CompStruct:

        enei = sig.enei

        H_key = 'H1' if inout == 1 else 'H2'

        G = self.eval(enei, 'G')
        H = self.eval(enei, H_key)

        phi = G @ sig.sig if hasattr(sig, 'sig') else np.zeros(self.p1.pos.shape[0])
        phip = H @ sig.sig if hasattr(sig, 'sig') else np.zeros(self.p1.pos.shape[0])

        if inout == 1:
            return CompStruct(self.p1, enei, phi1 = phi, phi1p = phip)
        else:
            return CompStruct(self.p1, enei, phi2 = phi, phi2p = phip)

    def field(self,
            sig: Any,
            inout: int = 1) -> CompStruct:

        enei = sig.enei
        H_key = 'H1' if inout == 1 else 'H2'

        # Use Cartesian derivative if available
        if self.deriv == 'cart':
            Hp = self.eval(enei, H_key)
            e = -Hp @ sig.sig
            return CompStruct(self.p1, enei, e = e)
        else:
            # Normal derivative: compute E from normal component
            H = self.eval(enei, H_key)
            nvec = self.p1.nvec if hasattr(self.p1, 'nvec') else self.p1.pc.nvec
            H_sig = H @ sig.sig
            if H_sig.ndim == 1:
                e = -nvec * H_sig[:, np.newaxis]
            else:
                e = -nvec[:, :, np.newaxis] * H_sig[:, np.newaxis, :]
            return CompStruct(self.p1, enei, e = e)

    def __repr__(self) -> str:
        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else '?'
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else '?'
        return 'CompGreenStatLayer(p1: {} faces, p2: {} faces)'.format(n1, n2)
