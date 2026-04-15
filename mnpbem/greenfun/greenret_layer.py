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
            deriv: str = 'cart',
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

        # Multiply with p2.area (MATLAB initrefl1.m line 36, initrefl2.m line 35)
        area2 = self.p2.area  # (n2,)
        G = G * area2[np.newaxis, :]
        Fr = Fr * area2[np.newaxis, :]
        Fz = Fz * area2[np.newaxis, :]

        # Store Green function
        self.G = G

        # Compute surface derivative
        if self.deriv == 'norm':
            self._compute_F_norm(G, Fr, Fz)
        else:
            self._compute_F_cart(G, Fr, Fz)

    def eval_components(self,
            enei: float) -> None:
        """Compute per-component reflected Green function (G, F, Gp).

        Always computes both normal and Cartesian derivatives regardless
        of self.deriv, since field() needs Gp and potential() needs F.

        After calling, results are stored in:
          self.G_comp  : dict of (n1, n2) arrays
          self.F_comp  : dict of (n1, n2) arrays (normal derivative)
          self.Gp_comp : dict of (n1, 3, n2) arrays (Cartesian derivative)
        """
        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        z1, z2 = self.layer.round_z(self._z1, self._z2)

        r_flat = self._r.ravel()
        z1_flat = np.repeat(z1, n2)
        z2_flat = np.tile(z2, n1)
        r_flat = np.maximum(r_flat, self.layer.rmin)

        G_dict, Fr_dict, Fz_dict = self.tab.eval_components(
            enei, r_flat, z1_flat, z2_flat)

        self.G_comp = {}
        self.F_comp = {}
        self.Gp_comp = {}

        nvec1 = self.p1.nvec
        r_safe = np.maximum(self._r, np.finfo(float).eps)
        area2 = self.p2.area  # (n2,)

        for name in G_dict:
            G = G_dict[name].reshape(n1, n2)
            Fr = Fr_dict[name].reshape(n1, n2)
            Fz = Fz_dict[name].reshape(n1, n2)

            # Multiply with p2.area (MATLAB initrefl1.m line 36, initrefl2.m line 35)
            G = G * area2[np.newaxis, :]
            Fr = Fr * area2[np.newaxis, :]
            Fz = Fz * area2[np.newaxis, :]

            self.G_comp[name] = G

            # Cartesian derivative: Gp (n1, 3, n2) — matches MATLAB shape
            Gp = np.zeros((n1, 3, n2), dtype = complex)
            Gp[:, 0, :] = Fr * self._dx / r_safe
            Gp[:, 1, :] = Fr * self._dy / r_safe
            Gp[:, 2, :] = Fz
            self.Gp_comp[name] = Gp

            # Normal derivative: F = nvec . Gp  (inner product)
            F = np.zeros((n1, n2), dtype = complex)
            F += nvec1[:, 0:1] * Gp[:, 0, :]
            F += nvec1[:, 1:2] * Gp[:, 1, :]
            F += nvec1[:, 2:3] * Gp[:, 2, :]
            self.F_comp[name] = F

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
        """Compute Cartesian derivative Gp and normal derivative F.

        MATLAB: initrefl2.m
        - Gp is the 3D Cartesian derivative, shape (n1, 3, n2)
        - F is the 2D normal derivative: F[i,j] = nvec[i] . Gp[i,:,j]
        """

        nvec1 = self.p1.nvec
        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        r_safe = np.maximum(self._r, np.finfo(float).eps)

        # Cartesian derivative: Gp[:, 0, :] = Fr * dx/r, etc.
        # Shape (n1, 3, n2) to match MATLAB convention
        Gp = np.zeros((n1, 3, n2), dtype = complex)
        Gp[:, 0, :] = Fr * self._dx / r_safe
        Gp[:, 1, :] = Fr * self._dy / r_safe
        Gp[:, 2, :] = Fz

        # Normal derivative: F = inner(nvec, Gp)
        # F[i,j] = nvec[i,0]*Gp[i,0,j] + nvec[i,1]*Gp[i,1,j] + nvec[i,2]*Gp[i,2,j]
        F = np.einsum('ik,ikj->ij', nvec1, Gp)

        self.Gp = Gp
        self.F = F

    def setup_tabulation(self, nr = 30, nz = 20):

        z1, z2 = self.layer.round_z(self._z1, self._z2)

        # r: logarithmic (rmin → max radial distance)
        r_max = max(self._r.max(), self.layer.rmin * 10)
        r_grid = np.geomspace(self.layer.rmin, r_max, nr)

        # z1, z2: linear (face z-coordinate range)
        z_all = np.concatenate([z1, z2])
        z_min, z_max = z_all.min(), z_all.max()
        if np.isclose(z_min, z_max):
            z_max = z_min + 1.0
        z1_grid = np.linspace(z_min, z_max, nz)
        z2_grid = np.linspace(z_min, z_max, nz)

        self.tab.setup_grid(r_grid, z1_grid, z2_grid)

    def __repr__(self) -> str:
        return 'GreenRetLayer(n1={}, n2={}, deriv={})'.format(
            self.p1.pos.shape[0], self.p2.pos.shape[0], self.deriv)
