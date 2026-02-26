import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompGreenRetLayer, CompStruct


class BEMRetLayer(object):

    name = 'bemsolver'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: Any,
            layer: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        self.p = p
        self.layer = layer

        self.enei = None
        self.k = None
        self.nvec = None
        self.eps1 = None
        self.eps2 = None

        # BEM matrices
        self.G1i = None
        self.G2i = None
        self.L1 = None
        self.L2 = None
        self.Sigma1 = None
        self.Deltai = None
        self.Sigmai = None

        # Green function with layer
        self.g = None
        self.options = options

        if enei is not None:
            self.init(enei)

    def init(self,
            enei: float) -> 'BEMRetLayer':

        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        self.enei = enei

        # Outer surface normals
        self.nvec = self.p.nvec

        # Wavenumber in vacuum
        self.k = 2 * np.pi / enei

        # Get dielectric functions
        eps_vals = []
        k_vals = []
        for eps_func in self.p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        # Dielectric function values
        eps1_vals = self.p.eps1(enei)
        eps2_vals = self.p.eps2(enei)

        if np.allclose(eps1_vals, eps1_vals[0]) and np.allclose(eps2_vals, eps2_vals[0]):
            self.eps1 = eps1_vals[0]
            self.eps2 = eps2_vals[0]
        else:
            self.eps1 = np.diag(eps1_vals)
            self.eps2 = np.diag(eps2_vals)

        # Create Green function with layer
        if self.g is None:
            self.g = CompGreenRetLayer(self.p, self.p, self.layer, **self.options)

        # Compute Green function matrices with layer contributions
        # MATLAB: Uses structured Green function (ss, hh, p, sh, hs components)
        # For layer: G includes direct + reflected contributions
        G11 = self.g.eval(0, 0, 'G', enei)
        G21 = self.g.eval(1, 0, 'G', enei)
        G22 = self.g.eval(1, 1, 'G', enei)
        G12 = self.g.eval(0, 1, 'G', enei)

        G1 = G11 - G21 if not (isinstance(G21, (int, float)) and G21 == 0) else G11
        G2 = G22 - G12 if not (isinstance(G12, (int, float)) and G12 == 0) else G22

        H11 = self.g.eval(0, 0, 'H1', enei)
        H21 = self.g.eval(1, 0, 'H1', enei)
        H22 = self.g.eval(1, 1, 'H2', enei)
        H12 = self.g.eval(0, 1, 'H2', enei)

        H1_mat = H11 - H21 if not (isinstance(H21, (int, float)) and H21 == 0) else H11
        H2_mat = H22 - H12 if not (isinstance(H12, (int, float)) and H12 == 0) else H22

        # Compute inverses
        self.G1i = np.linalg.inv(G1)
        self.G2i = np.linalg.inv(G2)

        # L matrices
        if np.isscalar(self.eps1):
            self.L1 = self.eps1
            self.L2 = self.eps2
        else:
            self.L1 = G1 @ self.eps1 @ self.G1i
            self.L2 = G2 @ self.eps2 @ self.G2i

        # Sigma matrices
        self.Sigma1 = H1_mat @ self.G1i
        Sigma2 = H2_mat @ self.G2i

        # Inverse Delta matrix
        self.Deltai = np.linalg.inv(self.Sigma1 - Sigma2)

        # Combined Sigma matrix with layer structure
        L = self.L1 - self.L2

        if np.isscalar(L):
            Sigma = self.Sigma1 * self.L1 - Sigma2 * self.L2
            nvec_outer = self.nvec @ self.nvec.T
            Sigma = Sigma + self.k ** 2 * L * (self.Deltai * nvec_outer) * L
        else:
            nvec_outer = self.nvec @ self.nvec.T
            Sigma = (self.Sigma1 @ self.L1 - Sigma2 @ self.L2 +
                self.k ** 2 * ((L @ self.Deltai) * nvec_outer) @ L)

        self.Sigmai = np.linalg.inv(Sigma)

        return self

    def _excitation(self,
            exc: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        enei = exc.enei if hasattr(exc, 'enei') else exc['enei']
        nfaces = self.p.nfaces if hasattr(self.p, 'nfaces') else self.p.n

        def get_field(name: str) -> Any:
            if hasattr(exc, name):
                val = getattr(exc, name)
                if isinstance(val, np.ndarray):
                    return val
                return val
            elif isinstance(exc, dict) and name in exc:
                val = exc[name]
                if isinstance(val, np.ndarray):
                    return val
                return val
            return 0

        phi1 = get_field('phi1')
        phi1p = get_field('phi1p')
        a1 = get_field('a1')
        a1p = get_field('a1p')
        phi2 = get_field('phi2')
        phi2p = get_field('phi2p')
        a2 = get_field('a2')
        a2p = get_field('a2p')

        k = 2 * np.pi / enei

        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)
        nvec = self.nvec

        # Potential jumps
        phi = self._subtract(phi2, phi1)
        a = self._subtract(a2, a1)

        # Eq. (15)
        outer_term2 = self._outer_eps(nvec, phi2, eps2)
        outer_term1 = self._outer_eps(nvec, phi1, eps1)
        alpha = self._subtract(a2p, a1p) - 1j * k * self._subtract(outer_term2, outer_term1)

        # Eq. (18)
        matmul_term2 = self._matmul_eps(eps2, phi2p)
        matmul_term1 = self._matmul_eps(eps1, phi1p)
        inner_term2 = self._inner_eps(nvec, a2, eps2)
        inner_term1 = self._inner_eps(nvec, a1, eps1)

        De = self._subtract(matmul_term2, matmul_term1) - 1j * k * self._subtract(inner_term2, inner_term1)

        return phi, a, alpha, De

    def _subtract(self,
            a: Any,
            b: Any) -> Any:

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a - b
        elif isinstance(a, np.ndarray):
            return a if b == 0 else a - b
        elif isinstance(b, np.ndarray):
            return -b if a == 0 else a - b
        else:
            return a - b

    def _outer_eps(self,
            nvec: np.ndarray,
            phi: Any,
            eps: np.ndarray) -> Any:

        if isinstance(phi, np.ndarray):
            if phi.ndim == 1:
                return nvec * (phi * eps)[:, np.newaxis]
            else:
                npol = phi.shape[1]
                n = len(nvec)
                result = np.zeros((n, 3, npol), dtype = complex)
                for ipol in range(npol):
                    result[:, :, ipol] = nvec * (phi[:, ipol] * eps)[:, np.newaxis]
                return result
        elif phi == 0:
            return 0
        else:
            return nvec * (phi * eps)

    def _inner_eps(self,
            nvec: np.ndarray,
            a: Any,
            eps: np.ndarray) -> Any:

        if isinstance(a, np.ndarray) and a.ndim >= 2:
            if a.ndim == 2:
                dot = np.sum(nvec * a, axis = 1)
                return dot * eps
            else:
                npol = a.shape[2]
                n = len(nvec)
                result = np.zeros((n, npol), dtype = complex)
                for ipol in range(npol):
                    dot = np.sum(nvec * a[:, :, ipol], axis = 1)
                    result[:, ipol] = dot * eps
                return result
        elif not isinstance(a, np.ndarray) and a == 0:
            return 0
        else:
            return 0

    def _matmul_eps(self,
            eps: np.ndarray,
            phi_p: Any) -> Any:

        if isinstance(phi_p, np.ndarray):
            if phi_p.ndim == 1:
                return eps * phi_p
            else:
                return eps[:, np.newaxis] * phi_p
        elif phi_p == 0:
            return 0
        else:
            return eps * phi_p

    def solve(self,
            exc: Any) -> Tuple[CompStruct, 'BEMRetLayer']:

        enei = exc.enei if hasattr(exc, 'enei') else exc['enei']
        self.init(enei)

        phi, a, alpha, De = self._excitation(exc)

        k = self.k
        nvec = self.nvec
        G1i = self.G1i
        G2i = self.G2i
        L1 = self.L1
        L2 = self.L2
        Sigma1 = self.Sigma1
        Deltai = self.Deltai
        Sigmai = self.Sigmai
        nfaces = self.p.nfaces if hasattr(self.p, 'nfaces') else self.p.n

        # Ensure proper shapes
        if not isinstance(phi, np.ndarray) or phi.size == 0:
            phi = np.zeros(nfaces, dtype = complex)
        if not isinstance(a, np.ndarray) or a.size == 0:
            a = np.zeros((nfaces, 3), dtype = complex)
        if not isinstance(alpha, np.ndarray):
            alpha = np.zeros((nfaces, 3), dtype = complex)
        if not isinstance(De, np.ndarray):
            De = np.zeros(nfaces, dtype = complex)

        # Determine number of polarizations
        npol = 1
        if isinstance(a, np.ndarray) and a.ndim == 3:
            npol = a.shape[2]
        elif isinstance(alpha, np.ndarray) and alpha.ndim == 3:
            npol = alpha.shape[2]
        elif isinstance(phi, np.ndarray) and phi.ndim == 2:
            npol = phi.shape[1]
        elif isinstance(De, np.ndarray) and De.ndim == 2:
            npol = De.shape[1]

        if npol == 1:
            if isinstance(a, np.ndarray) and a.ndim == 3:
                a = a[:, :, 0]
            if isinstance(alpha, np.ndarray) and alpha.ndim == 3:
                alpha = alpha[:, :, 0]
            if isinstance(phi, np.ndarray) and phi.ndim == 2:
                phi = phi[:, 0]
            if isinstance(De, np.ndarray) and De.ndim == 2:
                De = De[:, 0]

        if npol == 1:
            # Single polarization
            if np.isscalar(L1):
                L1_phi = L1 * phi
                L1_a = L1 * a
            else:
                L1_phi = L1 @ phi
                L1_a = L1 @ a

            alpha_mod = alpha - (Sigma1 @ a) + 1j * k * (nvec * L1_phi[:, np.newaxis])

            if np.isscalar(L1):
                De_mod = De - Sigma1 @ (L1 * phi) + 1j * k * np.sum(nvec * L1_a, axis = 1)
            else:
                De_mod = De - Sigma1 @ L1 @ phi + 1j * k * np.sum(nvec * L1_a, axis = 1)

            L_diff = L1 - L2
            if np.isscalar(L_diff):
                inner_term = np.sum(nvec * (L_diff * (Deltai @ alpha_mod)), axis = 1)
            else:
                inner_term = np.sum(nvec * (L_diff @ (Deltai @ alpha_mod)), axis = 1)

            sig2 = Sigmai @ (De_mod + 1j * k * inner_term)

            if np.isscalar(L_diff):
                outer_term = nvec * (L_diff * sig2)[:, np.newaxis]
            else:
                outer_term = nvec * (L_diff @ sig2)[:, np.newaxis]
            h2 = Deltai @ (1j * k * outer_term + alpha_mod)

            sig1_all = G1i @ (sig2 + phi)
            h1_all = G1i @ (h2 + a)
            sig2_all = G2i @ sig2
            h2_all = G2i @ h2
        else:
            # Multiple polarizations
            sig1_all = np.zeros((nfaces, npol), dtype = complex)
            sig2_all = np.zeros((nfaces, npol), dtype = complex)
            h1_all = np.zeros((nfaces, 3, npol), dtype = complex)
            h2_all = np.zeros((nfaces, 3, npol), dtype = complex)

            for ipol in range(npol):
                phi_i = phi[:, ipol] if phi.ndim > 1 else phi
                a_i = a[:, :, ipol] if a.ndim > 2 else a
                alpha_i = alpha[:, :, ipol] if alpha.ndim > 2 else alpha
                De_i = De[:, ipol] if De.ndim > 1 else De

                if np.isscalar(L1):
                    L1_phi = L1 * phi_i
                    L1_a = L1 * a_i
                else:
                    L1_phi = L1 @ phi_i
                    L1_a = L1 @ a_i

                alpha_mod = alpha_i - (Sigma1 @ a_i) + 1j * k * (nvec * L1_phi[:, np.newaxis])

                if np.isscalar(L1):
                    De_mod = De_i - Sigma1 @ (L1 * phi_i) + 1j * k * np.sum(nvec * L1_a, axis = 1)
                else:
                    De_mod = De_i - Sigma1 @ L1 @ phi_i + 1j * k * np.sum(nvec * L1_a, axis = 1)

                L_diff = L1 - L2
                if np.isscalar(L_diff):
                    inner_term = np.sum(nvec * (L_diff * (Deltai @ alpha_mod)), axis = 1)
                else:
                    inner_term = np.sum(nvec * (L_diff @ (Deltai @ alpha_mod)), axis = 1)

                sig2 = Sigmai @ (De_mod + 1j * k * inner_term)

                if np.isscalar(L_diff):
                    outer_term = nvec * (L_diff * sig2)[:, np.newaxis]
                else:
                    outer_term = nvec * (L_diff @ sig2)[:, np.newaxis]
                h2 = Deltai @ (1j * k * outer_term + alpha_mod)

                sig1_all[:, ipol] = G1i @ (sig2 + phi_i)
                h1_all[:, :, ipol] = G1i @ (h2 + a_i)
                sig2_all[:, ipol] = G2i @ sig2
                h2_all[:, :, ipol] = G2i @ h2

        sig = CompStruct(self.p, enei, sig1 = sig1_all, sig2 = sig2_all,
            h1 = h1_all, h2 = h2_all)

        return sig, self

    def __truediv__(self,
            exc: Any) -> Tuple[CompStruct, 'BEMRetLayer']:

        return self.solve(exc)

    def __mul__(self,
            sig: Any) -> CompStruct:

        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        return CompStruct(self.p, enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            a1 = pot1.a1, a1p = pot1.a1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p,
            a2 = pot2.a2, a2p = pot2.a2p)

    def potential(self,
            sig: Any,
            inout: int = 2) -> CompStruct:

        return self.g.potential(sig, inout)

    def field(self,
            sig: Any,
            inout: int = 2) -> CompStruct:

        return self.g.field(sig, inout)

    def clear(self) -> 'BEMRetLayer':

        self.G1i = None
        self.G2i = None
        self.L1 = None
        self.L2 = None
        self.Sigma1 = None
        self.Deltai = None
        self.Sigmai = None
        self.enei = None
        return self

    def __call__(self,
            enei: float) -> 'BEMRetLayer':

        return self.init(enei)

    def __repr__(self) -> str:
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        n = self.p.nfaces if hasattr(self.p, 'nfaces') else self.p.n if hasattr(self.p, 'n') else '?'
        return 'BEMRetLayer(p: {} faces, {})'.format(n, status)
