"""
Composite Green function for retarded approximation with layer structure.

MATLAB: Greenfun/@compgreenretlayer/

Implements structured Green function multiplication following MATLAB
matmul2.m and matmul3.m for proper polarization decomposition
(ss, hh, p, sh, hs components).
"""

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


def _safe_matmul(A, x):
    """Matrix multiply handling zero and scalar cases."""
    if isinstance(A, (int, float)):
        if A == 0:
            return 0
        return A * x
    return A @ x


def _matmul2_refl(G_comp, sig, h, mode):
    """Structured matmul2 for reflected Green function (2D components).

    MATLAB: @compgreenretlayer/private/matmul2.m

    Parameters
    ----------
    G_comp : dict
        Keys 'ss','hh','p','sh','hs', each (n1, n2) array.
    sig : ndarray
        Surface charge, shape (n2,) or (n2, npol).
    h : ndarray
        Surface current vector, shape (n2, 3) or (n2, 3, npol).
    mode : str
        'sig' for charge contribution, 'h' for current contribution.

    Returns
    -------
    ndarray
        'sig' mode: (n1,) or (n1, npol)
        'h' mode: (n1, 3) or (n1, 3, npol)
    """
    G_ss = G_comp.get('ss', 0)
    G_hh = G_comp.get('hh', 0)
    G_p = G_comp.get('p', 0)
    G_sh = G_comp.get('sh', 0)
    G_hs = G_comp.get('hs', 0)

    if mode == 'sig':
        # phi = G.ss @ sig + G.sh @ h_z
        h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]
        pot = _safe_matmul(G_ss, sig)
        sh_term = _safe_matmul(G_sh, h_z)
        if isinstance(pot, (int, float)) and pot == 0:
            return sh_term
        if isinstance(sh_term, (int, float)) and sh_term == 0:
            return pot
        return pot + sh_term

    elif mode == 'h':
        # a = [G.p @ h_x, G.p @ h_y, G.hh @ h_z + G.hs @ sig]
        h_x = h[:, 0] if h.ndim == 2 else h[:, 0, :]
        h_y = h[:, 1] if h.ndim == 2 else h[:, 1, :]
        h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]

        pot_x = _safe_matmul(G_p, h_x)
        pot_y = _safe_matmul(G_p, h_y)
        pot_z = _safe_matmul(G_hh, h_z)
        hs_term = _safe_matmul(G_hs, sig)

        if not isinstance(pot_z, (int, float)):
            if not isinstance(hs_term, (int, float)):
                pot_z = pot_z + hs_term
        elif not isinstance(hs_term, (int, float)):
            pot_z = hs_term

        # Stack into vector
        parts = [pot_x, pot_y, pot_z]
        # Determine output shape
        for part in parts:
            if not isinstance(part, (int, float)):
                ref = part
                break
        else:
            return 0

        result_parts = []
        for part in parts:
            if isinstance(part, (int, float)):
                result_parts.append(np.zeros_like(ref))
            else:
                result_parts.append(part)

        return np.stack(result_parts, axis=1)

    else:
        raise ValueError('[error] Unknown matmul2 mode: <{}>'.format(mode))


def _matmul2_refl_3d(Gp_comp, sig, h, mode):
    """Structured matmul2 for reflected Cartesian derivative (3D components).

    MATLAB: matmul2 with Gp/H1p/H2p structured arguments.

    Parameters
    ----------
    Gp_comp : dict
        Keys 'ss','hh','p','sh','hs', each (n1, n2, 3) array.
    sig : ndarray
        Surface charge, (n2,) or (n2, npol).
    h : ndarray
        Surface current, (n2, 3) or (n2, 3, npol).
    mode : str
        'sig' or 'h'.

    Returns
    -------
    ndarray
        Always (n1, 3) or (n1, 3, npol) — vector result.
    """
    if mode == 'sig':
        # grad_phi = [Gp.ss[:,:,j] @ sig + Gp.sh[:,:,j] @ h_z  for j in 0,1,2]
        h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]
        Gp_ss = Gp_comp.get('ss', 0)
        Gp_sh = Gp_comp.get('sh', 0)

        parts = []
        for j in range(3):
            ss_j = Gp_ss[:, :, j] if isinstance(Gp_ss, np.ndarray) else 0
            sh_j = Gp_sh[:, :, j] if isinstance(Gp_sh, np.ndarray) else 0
            val = _safe_matmul(ss_j, sig)
            sh_val = _safe_matmul(sh_j, h_z)
            if isinstance(val, (int, float)) and val == 0:
                val = sh_val
            elif not isinstance(sh_val, (int, float)):
                val = val + sh_val
            parts.append(val)

        # Find reference shape
        for part in parts:
            if not isinstance(part, (int, float)):
                ref = part
                break
        else:
            return 0

        result_parts = []
        for part in parts:
            if isinstance(part, (int, float)):
                result_parts.append(np.zeros_like(ref))
            else:
                result_parts.append(part)

        return np.stack(result_parts, axis=1)

    elif mode == 'h':
        # For each Cartesian direction j:
        # a_j = [Gp.p[:,:,j] @ h_x, Gp.p[:,:,j] @ h_y,
        #        Gp.hh[:,:,j] @ h_z + Gp.hs[:,:,j] @ sig]
        # But this gives (n1, 3, 3) — too complex.
        # Actually, for H-field computation we use _cross_refl_3d instead.
        raise NotImplementedError(
            'matmul2_refl_3d h mode not needed; use _cross_refl_3d')

    else:
        raise ValueError('[error] Unknown matmul2_3d mode: <{}>'.format(mode))


def _cross_refl_3d(Gp_comp, sig, h):
    """Structured cross product for magnetic field from reflected Gp.

    MATLAB: cross() in field.m using matmul3.m

    H = curl(A) where A_j = Gp(:,:,j) @ h
    H_x = Gp(:,:,2)@h_z - Gp(:,:,3)@h_y  (indices: y*z - z*y)
    H_y = Gp(:,:,3)@h_x - Gp(:,:,1)@h_z  (indices: z*x - x*z)
    H_z = Gp(:,:,1)@h_y - Gp(:,:,2)@h_x  (indices: x*y - y*x)

    With structured decomposition (matmul3.m):
    - For parallel components (i2=0,1): use G.p component
    - For z component (i2=2): use G.hh + G.hs @ sig

    Parameters
    ----------
    Gp_comp : dict
        Keys 'p','hh','hs', each (n1, n2, 3) array.
    sig : ndarray
        Surface charge, (n2,) or (n2, npol).
    h : ndarray
        Surface current, (n2, 3) or (n2, 3, npol).

    Returns
    -------
    ndarray
        (n1, 3) or (n1, 3, npol) — magnetic field contribution.
    """
    h_x = h[:, 0] if h.ndim == 2 else h[:, 0, :]
    h_y = h[:, 1] if h.ndim == 2 else h[:, 1, :]
    h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]

    Gp_p = Gp_comp.get('p', 0)
    Gp_hh = Gp_comp.get('hh', 0)
    Gp_hs = Gp_comp.get('hs', 0)

    def matmul3_comp(i1, i2):
        """Compute G(:,:,i1) @ h(:,i2,:) with structured decomposition."""
        if i2 in (0, 1):
            # Parallel component: use G.p
            G_slice = Gp_p[:, :, i1] if isinstance(Gp_p, np.ndarray) else 0
            h_slice = h[:, i2] if h.ndim == 2 else h[:, i2, :]
            return _safe_matmul(G_slice, h_slice)
        else:
            # z component: use G.hh + G.hs
            G_hh_slice = Gp_hh[:, :, i1] if isinstance(Gp_hh, np.ndarray) else 0
            G_hs_slice = Gp_hs[:, :, i1] if isinstance(Gp_hs, np.ndarray) else 0
            hh_term = _safe_matmul(G_hh_slice, h_z)
            hs_term = _safe_matmul(G_hs_slice, sig)
            if isinstance(hh_term, (int, float)) and hh_term == 0:
                return hs_term
            if isinstance(hs_term, (int, float)) and hs_term == 0:
                return hh_term
            return hh_term + hs_term

    # cross product: H = curl(G @ h)
    # H_x = matmul3(1,2) - matmul3(2,1)  (Gy@hz - Gz@hy)
    # H_y = matmul3(2,0) - matmul3(0,2)  (Gz@hx - Gx@hz)
    # H_z = matmul3(0,1) - matmul3(1,0)  (Gx@hy - Gy@hx)
    hx = _sub_safe(matmul3_comp(1, 2), matmul3_comp(2, 1))
    hy = _sub_safe(matmul3_comp(2, 0), matmul3_comp(0, 2))
    hz = _sub_safe(matmul3_comp(0, 1), matmul3_comp(1, 0))

    parts = [hx, hy, hz]
    for part in parts:
        if not isinstance(part, (int, float)):
            ref = part
            break
    else:
        return 0

    result_parts = []
    for part in parts:
        if isinstance(part, (int, float)):
            result_parts.append(np.zeros_like(ref))
        else:
            result_parts.append(part)

    return np.stack(result_parts, axis=1)


def _sub_safe(a, b):
    """Subtract handling zero cases."""
    if isinstance(a, (int, float)) and a == 0:
        if isinstance(b, (int, float)) and b == 0:
            return 0
        return -b
    if isinstance(b, (int, float)) and b == 0:
        return a
    return a - b


def _add_safe(a, b):
    """Add handling zero cases."""
    if isinstance(a, (int, float)) and a == 0:
        return b
    if isinstance(b, (int, float)) and b == 0:
        return a
    return a + b


def _matmul_structured(G: _StructuredGreen,
        x: np.ndarray,
        nvec: np.ndarray,
        mode: str = 'sig') -> np.ndarray:
    """Legacy structured matmul for backward compatibility."""

    if mode == 'sig':
        if isinstance(G.ss, np.ndarray):
            return G.ss @ x
        elif G.ss != 0:
            return G.ss * x
        else:
            return np.zeros_like(x)
    elif mode == 'h':
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
        self.gr = GreenRetLayer(p1, p2, layer, tab=tab,
            deriv=self.deriv, **options)

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
        g_direct = self.g.eval(i, j, key, enei, ind=ind)

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
        """Evaluate structured reflected Green function.

        MATLAB: @compgreenretlayer/private/eval2.m assembly()
        """
        self.gr.eval_components(enei)
        G_comp = self.gr.G_comp

        return _StructuredGreen(
            ss=G_comp.get('ss', 0),
            hh=G_comp.get('hh', 0),
            p=G_comp.get('p', 0),
            sh=G_comp.get('sh', 0),
            hs=G_comp.get('hs', 0)
        )

    def potential(self,
            sig: Any,
            inout: int = 1) -> CompStruct:
        """Potentials and surface derivatives including reflected contribution.

        MATLAB: @compgreenretlayer/potential.m
        """
        enei = sig.enei

        # Get direct potential
        pot_direct = self.g.potential(sig, inout)

        # Compute reflected contribution using structured Green functions
        self.gr.eval_components(enei)
        G_comp = self.gr.G_comp
        F_comp = self.gr.F_comp

        if not G_comp:
            return pot_direct

        # Surface charges and currents
        sig1 = sig.sig1 if hasattr(sig, 'sig1') else getattr(sig, 'sig', None)
        sig2 = sig.sig2 if hasattr(sig, 'sig2') else None
        h1 = sig.h1 if hasattr(sig, 'h1') else None
        h2 = sig.h2 if hasattr(sig, 'h2') else None

        if sig1 is None:
            return pot_direct
        if h1 is None:
            h1 = np.zeros((sig1.shape[0], 3), dtype=complex)
        if sig2 is None:
            sig2 = np.zeros_like(sig1)
        if h2 is None:
            h2 = np.zeros_like(h1)

        # Scalar potential: phi_refl = matmul2(G, sig, 'sig1') + matmul2(G, sig, 'sig2')
        phi_refl = _add_safe(
            _matmul2_refl(G_comp, sig1, h1, 'sig'),
            _matmul2_refl(G_comp, sig2, h2, 'sig'))

        # Surface derivative of scalar potential
        phip_refl = _add_safe(
            _matmul2_refl(F_comp, sig1, h1, 'sig'),
            _matmul2_refl(F_comp, sig2, h2, 'sig'))

        # Vector potential: a_refl = matmul2(G, sig, 'h1') + matmul2(G, sig, 'h2')
        a_refl = _add_safe(
            _matmul2_refl(G_comp, sig1, h1, 'h'),
            _matmul2_refl(G_comp, sig2, h2, 'h'))

        # Surface derivative of vector potential
        ap_refl = _add_safe(
            _matmul2_refl(F_comp, sig1, h1, 'h'),
            _matmul2_refl(F_comp, sig2, h2, 'h'))

        # Combine direct + reflected
        if inout == 1:
            phi = _add_to_attr(pot_direct, 'phi1', phi_refl)
            phip = _add_to_attr(pot_direct, 'phi1p', phip_refl)
            a = _add_to_attr(pot_direct, 'a1', a_refl)
            ap = _add_to_attr(pot_direct, 'a1p', ap_refl)
            return CompStruct(self.p1, enei,
                phi1=phi, phi1p=phip, a1=a, a1p=ap)
        else:
            phi = _add_to_attr(pot_direct, 'phi2', phi_refl)
            phip = _add_to_attr(pot_direct, 'phi2p', phip_refl)
            a = _add_to_attr(pot_direct, 'a2', a_refl)
            ap = _add_to_attr(pot_direct, 'a2p', ap_refl)
            return CompStruct(self.p1, enei,
                phi2=phi, phi2p=phip, a2=a, a2p=ap)

    def field(self,
            sig: Any,
            inout: int = 1) -> CompStruct:
        """Electric and magnetic field including reflected contribution.

        MATLAB: @compgreenretlayer/field.m

        E = i*k*A - grad(phi)
          = i*k*(G@h1 + G@h2) - (Gp@sig1 + Gp@sig2)
        H = curl(A)
          = curl(Gp@h1) + curl(Gp@h2)
        """
        enei = sig.enei
        k = 2 * np.pi / enei

        # Direct field (free-space contribution)
        field_direct = self.g.field(sig, inout)

        # Compute reflected structured Green functions
        self.gr.eval_components(enei)
        G_comp = self.gr.G_comp
        Gp_comp = self.gr.Gp_comp

        if not G_comp:
            return field_direct

        # Surface charges and currents
        sig1 = sig.sig1 if hasattr(sig, 'sig1') else getattr(sig, 'sig', None)
        sig2 = sig.sig2 if hasattr(sig, 'sig2') else None
        h1 = sig.h1 if hasattr(sig, 'h1') else None
        h2 = sig.h2 if hasattr(sig, 'h2') else None

        if sig1 is None:
            return field_direct
        if h1 is None:
            h1 = np.zeros((sig1.shape[0], 3), dtype=complex)
        if sig2 is None:
            sig2 = np.zeros_like(sig1)
        if h2 is None:
            h2 = np.zeros_like(h1)

        # E_refl = i*k*(G_refl @ h) - Gp_refl @ sig
        # Vector potential contribution: i*k*(matmul2(G, sig, 'h1') + matmul2(G, sig, 'h2'))
        ik_A = _add_safe(
            _matmul2_refl(G_comp, sig1, h1, 'h'),
            _matmul2_refl(G_comp, sig2, h2, 'h'))
        if not isinstance(ik_A, (int, float)):
            ik_A = 1j * k * ik_A

        # Gradient of scalar potential: matmul2(Gp, sig, 'sig1') + matmul2(Gp, sig, 'sig2')
        grad_phi = _add_safe(
            _matmul2_refl_3d(Gp_comp, sig1, h1, 'sig'),
            _matmul2_refl_3d(Gp_comp, sig2, h2, 'sig'))

        # E_refl = ik*A - grad(phi)
        e_refl = _sub_safe(ik_A, grad_phi)

        # H_refl = curl(Gp @ h1) + curl(Gp @ h2)
        h_refl = _add_safe(
            _cross_refl_3d(Gp_comp, sig1, h1),
            _cross_refl_3d(Gp_comp, sig2, h2))

        # Combine direct + reflected
        e_total = field_direct.e
        h_total = field_direct.h
        if not isinstance(e_refl, (int, float)):
            e_total = e_total + e_refl
        if not isinstance(h_refl, (int, float)):
            h_total = h_total + h_refl

        return CompStruct(self.p1, enei, e=e_total, h=h_total)

    def setup_tabulation(self, nr = 30, nz = 20):

        self.gr.setup_tabulation(nr = nr, nz = nz)

    def __repr__(self) -> str:
        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else '?'
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else '?'
        return 'CompGreenRetLayer(p1: {} faces, p2: {} faces)'.format(n1, n2)


def _add_to_attr(obj, attr, refl_val):
    """Add reflected value to an attribute of a CompStruct, handling zeros."""
    direct_val = getattr(obj, attr, None)
    if isinstance(refl_val, (int, float)) and refl_val == 0:
        return direct_val if direct_val is not None else 0
    if direct_val is None:
        return refl_val
    return direct_val + refl_val
