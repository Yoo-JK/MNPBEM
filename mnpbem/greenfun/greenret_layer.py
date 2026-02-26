import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .greentab_layer import GreenTabLayer


class GreenRetLayer(object):

    name = 'greenretlayer'

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            tab: Optional[Dict[str, Any]] = None,
            deriv: str = 'norm',
            **options: Any) -> None:

        self.p1 = p1
        self.p2 = p2
        self.layer = layer
        self.deriv = deriv
        self.enei = None

        self.G = None
        self.F = None
        self.Gp = None

        # Tabulated Green function
        if tab is not None:
            self.tab = GreenTabLayer(layer, tab = tab)
        else:
            self.tab = GreenTabLayer(layer)

        # Compute positions and distances for reflected Green function
        self._init_positions()

    def _init_positions(self) -> None:

        pos1 = self.p1.pos
        pos2 = self.p2.pos

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Radial distance between face pairs
        dx = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        dy = pos1[:, 1:2] - pos2[:, 1:2].T  # (n1, n2)

        self._r = np.sqrt(dx ** 2 + dy ** 2)  # (n1, n2)
        self._z1 = pos1[:, 2]  # (n1,)
        self._z2 = pos2[:, 2]  # (n2,)
        self._dx = dx
        self._dy = dy

    def eval(self,
            enei: float) -> None:

        if self.enei is not None and np.isclose(self.enei, enei):
            return

        self.enei = enei

        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        # Round z-values to avoid being too close to layer interface
        z1, z2 = self.layer.round_z(self._z1, self._z2)

        r_flat = self._r.ravel()
        z1_flat = np.repeat(z1, n2)
        z2_flat = np.tile(z2, n1)

        # Enforce minimum radial distance
        r_flat = np.maximum(r_flat, self.layer.rmin)

        # Compute reflected Green function
        G, Fr, Fz = self.tab.eval(enei, r_flat, z1_flat, z2_flat)

        G = G.reshape(n1, n2)
        Fr = Fr.reshape(n1, n2)
        Fz = Fz.reshape(n1, n2)

        # Store Green function
        self.G = G

        # Compute surface derivative
        if self.deriv == 'norm':
            self._compute_F_norm(G, Fr, Fz)
        else:
            self._compute_F_cart(G, Fr, Fz)

    def _compute_F_norm(self,
            G: np.ndarray,
            Fr: np.ndarray,
            Fz: np.ndarray) -> None:

        nvec1 = self.p1.nvec
        n1 = nvec1.shape[0]
        n2 = self.p2.pos.shape[0]

        r_safe = np.maximum(self._r, np.finfo(float).eps)

        # Normal derivative: F = nvec_x * Fr * dx/r + nvec_y * Fr * dy/r + nvec_z * Fz
        F = np.zeros((n1, n2), dtype = complex)
        F += nvec1[:, 0:1] * Fr * self._dx / r_safe
        F += nvec1[:, 1:2] * Fr * self._dy / r_safe
        F += nvec1[:, 2:3] * Fz

        self.F = F

    def _compute_F_cart(self,
            G: np.ndarray,
            Fr: np.ndarray,
            Fz: np.ndarray) -> None:

        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        r_safe = np.maximum(self._r, np.finfo(float).eps)

        # Cartesian derivative: Gp[:,:,0] = Fr * dx/r, Gp[:,:,1] = Fr * dy/r, Gp[:,:,2] = Fz
        Gp = np.zeros((n1, n2, 3), dtype = complex)
        Gp[:, :, 0] = Fr * self._dx / r_safe
        Gp[:, :, 1] = Fr * self._dy / r_safe
        Gp[:, :, 2] = Fz

        self.Gp = Gp
        self.F = Gp

    def __repr__(self) -> str:
        return 'GreenRetLayer(n1={}, n2={}, deriv={})'.format(
            self.p1.pos.shape[0], self.p2.pos.shape[0], self.deriv)
