import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .compgreen_ret import CompGreenRet
from .compgreen_stat import CompStruct
from .greenret_layer import GreenRetLayer


class _StructuredGreen(object):

    def __init__(self,
            ss: Optional[np.ndarray] = None,
            hh: Optional[np.ndarray] = None,
            p: Optional[np.ndarray] = None,
            sh: Optional[np.ndarray] = None,
            hs: Optional[np.ndarray] = None) -> None:

        self.ss = ss if ss is not None else 0
        self.hh = hh if hh is not None else 0
        self.p = p if p is not None else 0
        self.sh = sh if sh is not None else 0
        self.hs = hs if hs is not None else 0


def _matmul_structured(G: _StructuredGreen,
        x: np.ndarray,
        nvec: np.ndarray,
        mode: str = 'sig') -> np.ndarray:

    if mode == 'sig':
        # G * sig: scalar -> scalar
        # Result = G.ss * x + G.sh * (nvec . x_vec) where x_vec is expanded
        if isinstance(G.ss, np.ndarray):
            return G.ss @ x
        elif G.ss != 0:
            return G.ss * x
        else:
            return np.zeros_like(x)
    elif mode == 'h':
        # G * h: vector -> vector
        # Need parallel/perpendicular decomposition
        if isinstance(G.hh, np.ndarray) and x.ndim >= 2:
            n = x.shape[0]
            result = np.zeros_like(x)
            for j in range(3):
                if x.ndim == 2:
                    result[:, j] = G.hh @ x[:, j]
                else:
                    for ipol in range(x.shape[2]):
                        result[:, j, ipol] = G.hh @ x[:, j, ipol]
            return result
        elif G.hh != 0:
            return G.hh * x
        else:
            return np.zeros_like(x)
    else:
        raise ValueError('[error] Unknown matmul mode: <{}>'.format(mode))


class CompGreenRetLayer(object):

    name = 'greenfunction'
    needs = {'sim': 'ret'}

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            **options: Any) -> None:

        self.p1 = p1
        self.p2 = p2
        self.layer = layer
        self.deriv = options.get('deriv', 'norm')

        # Direct (free-space) Green function
        self.g = CompGreenRet(p1, p2, **options)

        # Reflected Green function
        tab = options.get('tab', None)
        self.gr = GreenRetLayer(p1, p2, layer, tab = tab,
            deriv = self.deriv, **options)

        # Indices of faces connected to layer
        self._init_layer_indices()

        # Cache
        self.enei = None
        self._G_cache = {}

    def _init_layer_indices(self) -> None:

        pos1 = self.p1.pos if hasattr(self.p1, 'pos') else self.p1.pc.pos
        pos2 = self.p2.pos if hasattr(self.p2, 'pos') else self.p2.pc.pos

        # All faces are connected to the layer (for substrate geometry)
        self.ind1 = np.arange(pos1.shape[0])
        self.ind2 = np.arange(pos2.shape[0])

    def eval(self,
            i: int,
            j: int,
            key: str,
            enei: float,
            ind: Optional[np.ndarray] = None) -> Any:

        # Compute reflected Green function
        self.gr.eval(enei)

        # Get direct Green function
        g_direct = self.g.eval(i, j, key, enei, ind = ind)

        # Add reflected contribution
        g_refl = self._get_reflected(key)

        if isinstance(g_direct, (int, float)) and g_direct == 0:
            return g_refl
        elif isinstance(g_refl, (int, float)) and g_refl == 0:
            return g_direct
        else:
            return g_direct + g_refl

    def _get_reflected(self,
            key: str) -> Any:

        if key == 'G':
            return self.gr.G if self.gr.G is not None else 0
        elif key in ('F', 'H1', 'H2'):
            return self.gr.F if self.gr.F is not None else 0
        elif key == 'Gp':
            return self.gr.Gp if self.gr.Gp is not None else 0
        else:
            return 0

    def eval_structured(self,
            enei: float) -> _StructuredGreen:

        self.gr.eval(enei)

        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else self.p1.pc.pos.shape[0]
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else self.p2.pc.pos.shape[0]

        G_refl = self.gr.G
        if G_refl is None:
            G_refl = np.zeros((n1, n2), dtype = complex)

        # For substrate: reflected Green function has structured form
        # ss: scalar-scalar coupling
        # hh: vector-vector coupling (parallel to layer)
        # p: coupling through perpendicular component
        # sh: scalar-vector coupling
        # hs: vector-scalar coupling

        # Simple approximation: use scalar reflected Green function for all components
        return _StructuredGreen(
            ss = G_refl,
            hh = G_refl,
            p = G_refl,
            sh = np.zeros((n1, n2), dtype = complex),
            hs = np.zeros((n1, n2), dtype = complex)
        )

    def potential(self,
            sig: Any,
            inout: int = 1) -> CompStruct:

        enei = sig.enei
        self.gr.eval(enei)

        # Get direct potential
        pot_direct = self.g.potential(sig, inout)

        # Add reflected contribution
        sig_val = sig.sig1 if hasattr(sig, 'sig1') else sig.sig
        h_val = sig.h1 if hasattr(sig, 'h1') else np.zeros_like(sig_val)

        G_refl = self.gr.G
        if G_refl is not None and isinstance(G_refl, np.ndarray):
            if sig_val.ndim == 1:
                phi_refl = G_refl @ sig_val
            else:
                phi_refl = G_refl @ sig_val

            F_refl = self.gr.F
            if F_refl is not None and isinstance(F_refl, np.ndarray):
                if sig_val.ndim == 1:
                    phip_refl = F_refl @ sig_val
                else:
                    phip_refl = F_refl @ sig_val
            else:
                phip_refl = np.zeros_like(phi_refl)

            # Add reflected to direct
            if inout == 1:
                phi = pot_direct.phi1 + phi_refl if hasattr(pot_direct, 'phi1') else phi_refl
                phip = pot_direct.phi1p + phip_refl if hasattr(pot_direct, 'phi1p') else phip_refl
                return CompStruct(self.p1, enei, phi1 = phi, phi1p = phip)
            else:
                phi = pot_direct.phi2 + phi_refl if hasattr(pot_direct, 'phi2') else phi_refl
                phip = pot_direct.phi2p + phip_refl if hasattr(pot_direct, 'phi2p') else phip_refl
                return CompStruct(self.p1, enei, phi2 = phi, phi2p = phip)
        else:
            return pot_direct

    def field(self,
            sig: Any,
            inout: int = 1) -> CompStruct:

        enei = sig.enei
        self.gr.eval(enei)

        # Get direct field
        field_direct = self.g.field(sig, inout)

        # Reflected contribution (simplified)
        # Full implementation requires structured Green function multiplication
        return field_direct

    def __repr__(self) -> str:
        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else '?'
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else '?'
        return 'CompGreenRetLayer(p1: {} faces, p2: {} faces)'.format(n1, n2)
