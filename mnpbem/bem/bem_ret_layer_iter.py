import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..greenfun import CompStruct
from .bem_iter import BEMIter


class BEMRetLayerIter(BEMIter):

    # MATLAB: @bemretlayeriter properties (Constant)
    name = 'bemsolver'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: Any,
            layer: Optional[Any] = None,
            enei: Optional[float] = None,
            **options: Any) -> None:

        # Initialize BEMIter base class
        super(BEMRetLayerIter, self).__init__(**options)

        # MATLAB: @bemretlayeriter properties
        self.p = p
        self.layer = layer if layer is not None else options.get('layer', None)
        self.enei = None
        self.g = None

        # MATLAB: @bemretlayeriter properties (Access = private)
        self._op = options
        self._sav = None
        self._k = None
        self._eps1 = None
        self._eps2 = None
        self._nvec = p.nvec
        self._G1 = None
        self._H1 = None
        self._G2 = None  # structured: ss, hh, p, sh, hs components
        self._H2 = None  # structured: ss, hh, p, sh, hs components

        # Green function (with layer structure)
        # MATLAB: obj.g = aca.compgreenretlayer(p, varargin{:}, ...)
        self._init_green(p, **options)

        # Initialize for given wavelength
        if enei is not None:
            self._init_matrices(enei)

    def _init_green(self,
            p: Any,
            **options: Any) -> None:

        # MATLAB: bemretlayeriter/private/init.m
        from ..greenfun import CompGreenRetLayer
        self.g = CompGreenRetLayer(p, p, self.layer, **options)

    def _init_matrices(self,
            enei: float) -> 'BEMRetLayerIter':

        # MATLAB: bemretlayeriter/private/initmat.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 128 (2015)
        if self.enei is not None and self.enei == enei:
            return self

        self.enei = enei

        # Wavenumber
        self._k = 2 * np.pi / enei

        # Dielectric function
        self._eps1 = self.p.eps1(enei)
        self._eps2 = self.p.eps2(enei)

        # Green functions for inner surfaces
        # MATLAB: G1 = g{1,1}.G(enei) - g{2,1}.G(enei)
        G11 = self.g.eval(0, 0, 'G', enei)
        G21 = self.g.eval(1, 0, 'G', enei)
        G1 = G11 - G21 if not (isinstance(G21, (int, float)) and G21 == 0) else G11

        H11 = self.g.eval(0, 0, 'H1', enei)
        H21 = self.g.eval(1, 0, 'H1', enei)
        H1 = H11 - H21 if not (isinstance(H21, (int, float)) and H21 == 0) else H11

        # Green functions for outer surfaces (with layer structure)
        # MATLAB: G2 = g{2,2}.G(enei); g2 = g{1,2}.G(enei)
        G22_full = self.g.eval(1, 1, 'G', enei)
        g2 = self.g.eval(0, 1, 'G', enei)

        H22_full = self.g.eval(1, 1, 'H2', enei)
        h2 = self.g.eval(0, 1, 'H2', enei)

        # For layer structure, G2 is a dict-like with ss, hh, p, sh, hs components
        # MATLAB: G2.ss = G2.ss - g2; G2.hh = G2.hh - g2; G2.p = G2.p - g2
        if isinstance(G22_full, dict):
            G2 = {}
            H2 = {}
            for key in G22_full:
                if key in ('ss', 'hh', 'p'):
                    g2_val = g2 if not isinstance(g2, dict) else g2.get(key, g2)
                    h2_val = h2 if not isinstance(h2, dict) else h2.get(key, h2)
                    G2[key] = G22_full[key] - g2_val
                    H2[key] = H22_full[key] - h2_val
                else:
                    G2[key] = G22_full[key]
                    H2[key] = H22_full[key]
        elif hasattr(G22_full, 'ss'):
            # Object with attributes
            G2 = _LayerGreen()
            H2 = _LayerGreen()

            g2_mat = g2 if not hasattr(g2, 'ss') else g2
            h2_mat = h2 if not hasattr(h2, 'ss') else h2

            G2.ss = G22_full.ss - (g2_mat if np.isscalar(g2_mat) or isinstance(g2_mat, np.ndarray) else g2_mat.ss)
            G2.hh = G22_full.hh - (g2_mat if np.isscalar(g2_mat) or isinstance(g2_mat, np.ndarray) else g2_mat.hh)
            G2.p = G22_full.p - (g2_mat if np.isscalar(g2_mat) or isinstance(g2_mat, np.ndarray) else g2_mat.p)
            G2.sh = G22_full.sh if hasattr(G22_full, 'sh') else np.zeros_like(G22_full.ss)
            G2.hs = G22_full.hs if hasattr(G22_full, 'hs') else np.zeros_like(G22_full.ss)

            H2.ss = H22_full.ss - (h2_mat if np.isscalar(h2_mat) or isinstance(h2_mat, np.ndarray) else h2_mat.ss)
            H2.hh = H22_full.hh - (h2_mat if np.isscalar(h2_mat) or isinstance(h2_mat, np.ndarray) else h2_mat.hh)
            H2.p = H22_full.p - (h2_mat if np.isscalar(h2_mat) or isinstance(h2_mat, np.ndarray) else h2_mat.p)
            H2.sh = H22_full.sh if hasattr(H22_full, 'sh') else np.zeros_like(H22_full.ss)
            H2.hs = H22_full.hs if hasattr(H22_full, 'hs') else np.zeros_like(H22_full.ss)
        else:
            # Fallback: treat as simple matrix (no layer structure difference)
            G2 = G22_full - g2 if not (isinstance(g2, (int, float)) and g2 == 0) else G22_full
            H2 = H22_full - h2 if not (isinstance(h2, (int, float)) and h2 == 0) else H22_full

        # Save Green functions
        self._G1 = G1
        self._H1 = H1
        self._G2 = G2
        self._H2 = H2

        # Initialize preconditioner
        if self.precond is not None:
            self._init_precond(enei)

        return self

    def _init_precond(self,
            enei: float) -> None:

        # MATLAB: bemretlayeriter/private/initprecond.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 128 (2015)
        k = 2 * np.pi / enei
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec

        # Dielectric as diagonal
        if np.isscalar(eps1) or (isinstance(eps1, np.ndarray) and eps1.ndim == 0):
            eps1_diag = eps1 * np.eye(self._G1.shape[0])
            eps2_diag = eps2 * np.eye(self._G1.shape[0])
        else:
            eps1_diag = np.diag(eps1)
            eps2_diag = np.diag(eps2)

        ikdeps = 1j * k * (eps1_diag - eps2_diag)

        G1 = self._G1
        H1 = self._H1
        G2 = self._G2
        H2 = self._H2

        # Get the parallel Green function component
        G2_p = G2.p if hasattr(G2, 'p') else (G2['p'] if isinstance(G2, dict) else G2)
        H2_p = H2.p if hasattr(H2, 'p') else (H2['p'] if isinstance(H2, dict) else H2)

        # Inverse of G1 and of parallel component
        G1i = np.linalg.inv(G1)
        G2pi = np.linalg.inv(G2_p)

        # Sigma matrices [Eq. (21)]
        Sigma1 = H1 @ G1i
        Sigma2p = H2_p @ G2pi

        # Perpendicular component of normal vector
        nperp_diag = np.diag(nvec[:, 3 - 1])  # nvec(:,3)

        # Gamma matrix
        Gamma = np.linalg.inv(Sigma1 - Sigma2p)

        # Gammapar with only parallel normal vector components
        Gammapar = ikdeps @ self._decorate_gamma(Gamma, nvec)

        # Get structured Green function components
        G2_ss = G2.ss if hasattr(G2, 'ss') else (G2['ss'] if isinstance(G2, dict) else G2)
        G2_sh = G2.sh if hasattr(G2, 'sh') else (G2['sh'] if isinstance(G2, dict) else np.zeros_like(G1))
        G2_hs = G2.hs if hasattr(G2, 'hs') else (G2['hs'] if isinstance(G2, dict) else np.zeros_like(G1))
        G2_hh = G2.hh if hasattr(G2, 'hh') else (G2['hh'] if isinstance(G2, dict) else G2)
        H2_ss = H2.ss if hasattr(H2, 'ss') else (H2['ss'] if isinstance(H2, dict) else H2)
        H2_sh = H2.sh if hasattr(H2, 'sh') else (H2['sh'] if isinstance(H2, dict) else np.zeros_like(H1))
        H2_hs = H2.hs if hasattr(H2, 'hs') else (H2['hs'] if isinstance(H2, dict) else np.zeros_like(H1))
        H2_hh = H2.hh if hasattr(H2, 'hh') else (H2['hh'] if isinstance(H2, dict) else H2)

        # Set up full matrix, Eq. (10)
        m11 = (eps1_diag @ Sigma1 - Gammapar @ ikdeps) @ G2_ss - eps2_diag @ H2_ss - (nperp_diag @ ikdeps) @ G2_hs
        m12 = (eps1_diag @ Sigma1 - Gammapar @ ikdeps) @ G2_sh - eps2_diag @ H2_sh - (nperp_diag @ ikdeps) @ G2_hh
        m21 = Sigma1 @ G2_hs - H2_hs - nperp_diag @ ikdeps @ G2_ss
        m22 = Sigma1 @ G2_hh - H2_hh - nperp_diag @ ikdeps @ G2_sh

        # LU decomposition as block inverse
        # L11 * U11 = M11
        im11 = np.linalg.inv(m11)
        # L11 * U12 = M12 -> U12 = inv(L11) * M12
        im12 = im11 @ m12
        # L21 * U11 = M21 -> L21 = M21 * inv(U11)
        im21 = m21 @ im11
        # L22 * U22 = M22 - L21 * U12
        im22 = np.linalg.inv(m22 - im21 @ m12)

        # Save variables
        sav = {}
        sav['k'] = k
        sav['nvec'] = nvec
        sav['eps1'] = eps1_diag
        sav['eps2'] = eps2_diag
        sav['G1i'] = G1i
        sav['G2pi'] = G2pi
        sav['G2'] = G2
        sav['Sigma1'] = Sigma1
        sav['Gamma'] = Gamma
        sav['im'] = [[im11, im12], [im21, im22]]

        self._sav = sav

    @staticmethod
    def _decorate_gamma(
            Gamma: np.ndarray,
            nvec: np.ndarray) -> np.ndarray:

        # MATLAB: fun(Gamma, nvec) in initprecond.m for layer
        # Only uses parallel (x,y) components of normal vector
        # Gamma_decorated = nvec1 * Gamma * nvec1 + nvec2 * Gamma * nvec2
        n = nvec.shape[0]
        result = np.zeros((n, n), dtype = Gamma.dtype)
        for i in range(2):  # only x, y components (parallel)
            nvec_i = np.diag(nvec[:, i])
            result = result + nvec_i @ Gamma @ nvec_i
        return result

    def _pack(self, *args: Any) -> np.ndarray:

        # MATLAB: bemretlayeriter/private/pack.m
        if len(args) == 4:
            phi, a, phip, ap = args
            total_len = phi.size + a.size + phip.size + ap.size
            vec = np.empty(total_len, dtype = complex)
            offset = 0
            for arr in [phi, a, phip, ap]:
                flat = arr.ravel()
                vec[offset:offset + flat.size] = flat
                offset += flat.size
            return vec
        elif len(args) == 6:
            # phi, apar, aperp, phip, appar, apperp
            phi, apar, aperp, phip, appar, apperp = args
            n = phi.shape[0] if isinstance(phi, np.ndarray) else aperp.shape[0]

            # Determine siz from aperp
            if aperp.ndim == 1:
                siz = 1
            else:
                siz = aperp.shape[1]

            # Combine parallel and perpendicular into full 3D vectors
            if siz == 1:
                a = np.empty((n, 3), dtype = complex)
                a[:, :2] = apar.reshape(n, 2) if apar.ndim >= 2 else apar
                a[:, 2] = aperp.ravel()

                ap = np.empty((n, 3), dtype = complex)
                ap[:, :2] = appar.reshape(n, 2) if appar.ndim >= 2 else appar
                ap[:, 2] = apperp.ravel()
            else:
                a = np.empty((n, 3, siz), dtype = complex)
                a[:, :2, :] = apar.reshape(n, 2, siz)
                a[:, 2, :] = aperp.reshape(n, siz)

                ap = np.empty((n, 3, siz), dtype = complex)
                ap[:, :2, :] = appar.reshape(n, 2, siz)
                ap[:, 2, :] = apperp.reshape(n, siz)

            total_len = phi.size + a.size + phip.size + ap.size
            vec = np.empty(total_len, dtype = complex)
            offset = 0
            for arr in [phi, a, phip, ap]:
                flat = arr.ravel()
                vec[offset:offset + flat.size] = flat
                offset += flat.size
            return vec

    def _unpack(self,
            vec: np.ndarray,
            nout: int = 4) -> Tuple:

        # MATLAB: bemretlayeriter/private/unpack.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # Last dimension
        siz = int(vec.size / (8 * n))

        # Reshape vector
        vec_2d = vec.reshape(-1, 8)

        # Extract potentials from vector
        phi = vec_2d[:, 0].reshape(n, siz) if siz > 1 else vec_2d[:, 0].reshape(n)
        a = vec_2d[:, 1:4].reshape(n, 3, siz) if siz > 1 else vec_2d[:, 1:4].reshape(n, 3)
        phip = vec_2d[:, 4].reshape(n, siz) if siz > 1 else vec_2d[:, 4].reshape(n)
        ap = vec_2d[:, 5:8].reshape(n, 3, siz) if siz > 1 else vec_2d[:, 5:8].reshape(n, 3)

        if nout == 4:
            return phi, a, phip, ap
        else:
            # Decompose vectors into parallel and perpendicular components
            if a.ndim == 2:
                apar = a[:, :2]
                aperp = a[:, 2]
            else:
                apar = a[:, :2, :]
                aperp = a[:, 2, :]

            if ap.ndim == 2:
                appar = ap[:, :2]
                apperp = ap[:, 2]
            else:
                appar = ap[:, :2, :]
                apperp = ap[:, 2, :]

            return phi, apar, aperp, phip, appar, apperp

    @staticmethod
    def _outer(
            nvec: np.ndarray,
            val: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretlayeriter/private/outer.m
        if isinstance(val, (int, float)) and val == 0:
            return 0

        if mul is not None:
            if val.ndim == 1:
                val = val * mul
            else:
                val = val * mul[:, np.newaxis] if mul.ndim == 1 else val * mul

        ndim = nvec.shape[1]  # 2 for parallel, 3 for full

        if val.ndim == 1:
            n = val.shape[0]
            result = np.empty((n, ndim), dtype = val.dtype)
            for i in range(ndim):
                result[:, i] = val * nvec[:, i]
            return result
        else:
            n = val.shape[0]
            siz = val.shape[1]
            result = np.empty((n, ndim, siz), dtype = val.dtype)
            for i in range(ndim):
                result[:, i, :] = val * nvec[:, i:i + 1]
            return result

    @staticmethod
    def _inner(
            nvec: np.ndarray,
            a: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretlayeriter/private/inner.m
        if isinstance(a, (int, float)) and a == 0:
            return 0

        ndim = nvec.shape[1]

        if a.ndim == 2:
            result = np.zeros(a.shape[0], dtype = a.dtype)
            for i in range(min(ndim, a.shape[1])):
                result = result + a[:, i] * nvec[:, i]
        elif a.ndim == 3:
            siz = a.shape[2]
            result = np.zeros((a.shape[0], siz), dtype = a.dtype)
            for i in range(min(ndim, a.shape[1])):
                result = result + a[:, i, :] * nvec[:, i:i + 1]
        else:
            result = a

        if mul is not None:
            if result.ndim == 1:
                result = result * mul
            else:
                result = result * mul[:, np.newaxis] if mul.ndim == 1 else result * mul

        return result

    @staticmethod
    def _subtract(
            a: Any,
            b: Any) -> Any:

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a - b
        elif isinstance(a, np.ndarray):
            return a if (isinstance(b, (int, float)) and b == 0) else a - b
        elif isinstance(b, np.ndarray):
            return -b if (isinstance(a, (int, float)) and a == 0) else a - b
        else:
            return a - b

    def _excitation(self,
            exc: CompStruct) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: bemretlayeriter/private/excitation.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        phi1 = getattr(exc, 'phi1', 0)
        phi1p = getattr(exc, 'phi1p', 0)
        a1 = getattr(exc, 'a1', 0)
        a1p = getattr(exc, 'a1p', 0)
        phi2 = getattr(exc, 'phi2', 0)
        phi2p = getattr(exc, 'phi2p', 0)
        a2 = getattr(exc, 'a2', 0)
        a2p = getattr(exc, 'a2p', 0)

        k = self._k
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec

        def _matmul(a_val: Any, x_val: Any) -> Any:
            if isinstance(x_val, (int, float)) and x_val == 0:
                return 0
            if np.isscalar(a_val):
                return a_val * x_val
            return a_val[:, np.newaxis] * x_val if x_val.ndim > 1 else a_val * x_val

        # Eqs. (10, 11)
        phi = self._subtract(phi2, phi1)
        a = self._subtract(a2, a1)

        # Eq. (15)
        alpha = self._subtract(a2p, a1p) - \
            1j * k * self._subtract(
                self._outer(nvec, phi2, eps2),
                self._outer(nvec, phi1, eps1))

        # Eq. (18)
        De = self._subtract(_matmul(eps2, phi2p), _matmul(eps1, phi1p)) - \
            1j * k * self._subtract(
                self._inner(nvec, a2, eps2),
                self._inner(nvec, a1, eps1))

        # Expand arrays
        if isinstance(phi, (int, float)) and phi == 0:
            if isinstance(De, np.ndarray):
                phi = np.zeros_like(De)
            else:
                phi = np.zeros(n, dtype = complex)

        if isinstance(a, (int, float)) and a == 0:
            if isinstance(alpha, np.ndarray):
                a = np.zeros_like(alpha)
            else:
                a = np.zeros((n, 3), dtype = complex)

        return phi, a, alpha, De

    def _afun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretlayeriter/private/afun.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # Split vector array into 6 components
        sig1, h1par, h1perp, sig2, h2par, h2perp = self._unpack(vec, nout = 6)

        k = self._k
        nvec = self._nvec
        eps1 = self._eps1
        eps2 = self._eps2
        npar = nvec[:, :2]
        nperp = nvec[:, 2]

        G1 = self._G1
        H1 = self._H1
        G2 = self._G2
        H2 = self._H2

        # Get layer components
        G2_ss = G2.ss if hasattr(G2, 'ss') else (G2['ss'] if isinstance(G2, dict) else G2)
        G2_sh = G2.sh if hasattr(G2, 'sh') else (G2['sh'] if isinstance(G2, dict) else np.zeros_like(G1))
        G2_hs = G2.hs if hasattr(G2, 'hs') else (G2['hs'] if isinstance(G2, dict) else np.zeros_like(G1))
        G2_hh = G2.hh if hasattr(G2, 'hh') else (G2['hh'] if isinstance(G2, dict) else G2)
        G2_p = G2.p if hasattr(G2, 'p') else (G2['p'] if isinstance(G2, dict) else G2)
        H2_ss = H2.ss if hasattr(H2, 'ss') else (H2['ss'] if isinstance(H2, dict) else H2)
        H2_sh = H2.sh if hasattr(H2, 'sh') else (H2['sh'] if isinstance(H2, dict) else np.zeros_like(H1))
        H2_hs = H2.hs if hasattr(H2, 'hs') else (H2['hs'] if isinstance(H2, dict) else np.zeros_like(H1))
        H2_hh = H2.hh if hasattr(H2, 'hh') else (H2['hh'] if isinstance(H2, dict) else H2)
        H2_p = H2.p if hasattr(H2, 'p') else (H2['p'] if isinstance(H2, dict) else H2)

        def matmul(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            if b.ndim == 1:
                return a_mat @ b
            n_rows = a_mat.shape[0]
            return (a_mat @ b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        def mul(x: Any, y: np.ndarray) -> np.ndarray:
            if np.isscalar(x):
                return x * y
            if y.ndim == 1:
                return x * y
            return x[:, np.newaxis] * y if x.ndim == 1 else x * y

        # Apply Green functions to surface charges
        Gsig1 = G1 @ sig1
        Gsig2 = G2_ss @ sig2 + G2_sh @ h2perp
        Hsig1 = H1 @ sig1
        Hsig2 = H2_ss @ sig2 + H2_sh @ h2perp

        # Apply Green functions to parallel surface currents
        Gh1par = matmul(G1, h1par)
        Gh2par = matmul(G2_p, h2par)
        Hh1par = matmul(H1, h1par)
        Hh2par = matmul(H2_p, h2par)

        # Apply Green functions to perpendicular surface currents
        Gh1perp = G1 @ h1perp
        Gh2perp = G2_hh @ h2perp + G2_hs @ sig2
        Hh1perp = H1 @ h1perp
        Hh2perp = H2_hh @ h2perp + H2_hs @ sig2

        # Eq. (7a)
        phi = Gsig1 - Gsig2
        # Eqs. (7b, c)
        apar = Gh1par - Gh2par
        aperp = Gh1perp - Gh2perp

        # Eqs. (8a, b)
        alphapar = Hh1par - Hh2par - \
            1j * k * (self._outer(npar, Gsig1, eps1) - self._outer(npar, Gsig2, eps2))
        alphaperp = Hh1perp - Hh2perp - \
            1j * k * (mul(Gsig1, eps1 * nperp) - mul(Gsig2, eps2 * nperp))

        # Eq. (9)
        De = mul(Hsig1, eps1) - mul(Hsig2, eps2) - \
            1j * k * (self._inner(npar, Gh1par, eps1) - self._inner(npar, Gh2par, eps2)) - \
            1j * k * (mul(Gh1perp, eps1 * nperp) - mul(Gh2perp, eps2 * nperp))

        return self._pack(phi, apar, aperp, De, alphapar, alphaperp)

    def _mfun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretlayeriter/private/mfun.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)

        # Unpack matrices
        phi, a, De, alpha = self._unpack(vec, nout = 4)

        sav = self._sav
        k = sav['k']
        nvec = sav['nvec']
        G2 = sav['G2']
        G1i = sav['G1i']
        G2pi = sav['G2pi']
        eps1 = sav['eps1']
        eps2 = sav['eps2']
        Sigma1 = sav['Sigma1']
        Gamma = sav['Gamma']
        im = sav['im']

        deps = eps1 - eps2
        npar = nvec[:, :2]

        if a.ndim == 2:
            apar = a[:, :2]
            aperp = a[:, 2]
        else:
            apar = a[:, :2, :]
            aperp = a[:, 2, :]

        def matmul1(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            if b.ndim == 1:
                return a_mat @ b
            n_rows = a_mat.shape[0]
            return (a_mat @ b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        def matmul2(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            # For preconditioner: equivalent to solve(a_mat, b) or a_mat @ b
            if b.ndim == 1:
                return a_mat @ b
            n_rows = a_mat.shape[0]
            return (a_mat @ b.reshape(b.shape[0], -1)).reshape(n_rows, *b.shape[1:])

        # Modify alpha
        alpha = alpha - matmul1(Sigma1, a) + 1j * k * self._outer(nvec, eps1 @ phi if phi.ndim <= 1 else eps1 @ phi)
        if alpha.ndim == 2:
            alphapar = alpha[:, :2]
            alphaperp = alpha[:, 2]
        else:
            alphapar = alpha[:, :2, :]
            alphaperp = alpha[:, 2, :]

        # Modify De
        De = De - eps1 @ Sigma1 @ phi + \
            1j * k * self._inner(nvec, matmul1(eps1, a)) + \
            1j * k * self._inner(npar, matmul1(deps @ Gamma, alphapar))

        # Solve Eq. (10) using block LU
        sig2, h2perp = self._solve_block_lu(im, De, alphaperp)

        # Get G2 components
        G2_ss = G2.ss if hasattr(G2, 'ss') else (G2['ss'] if isinstance(G2, dict) else G2)
        G2_sh = G2.sh if hasattr(G2, 'sh') else (G2['sh'] if isinstance(G2, dict) else np.zeros_like(G1i))
        G2_p = G2.p if hasattr(G2, 'p') else (G2['p'] if isinstance(G2, dict) else G2)
        G2_hh = G2.hh if hasattr(G2, 'hh') else (G2['hh'] if isinstance(G2, dict) else G2)
        G2_hs = G2.hs if hasattr(G2, 'hs') else (G2['hs'] if isinstance(G2, dict) else np.zeros_like(G1i))

        # Parallel component, Eq. (A.1)
        h2par = matmul2(G2pi, matmul1(Gamma, alphapar +
            1j * k * self._outer(npar, deps @ (G2_ss @ sig2 + G2_sh @ h2perp))))

        # Surface charges at inner interface
        sig1 = matmul2(G1i, G2_ss @ sig2 + G2_sh @ h2perp + phi)

        # Surface currents at inner interface
        h1perp = matmul2(G1i, G2_hh @ h2perp + G2_hs @ sig2 + aperp)
        h1par = matmul2(G1i, matmul1(G2_p, h2par) + apar)

        result = self._pack(sig1, h1par, h1perp, sig2, h2par, h2perp)
        return result

    @staticmethod
    def _solve_block_lu(
            im: List[List[np.ndarray]],
            b1: np.ndarray,
            b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB: fun(M, b1, b2) in mfun.m
        # Solve system using LU decomposition:
        # [L11, 0; L21, L22] * [y1; y2] = [b1; b2]
        # [U11, U12; 0, U22] * [x1; x2] = [y1; y2]

        L = im
        U = im

        # Forward solve
        y1 = L[0][0] @ b1
        y2 = L[1][1] @ (b2 - L[1][0] @ y1)

        # Backward solve
        x2 = U[1][1] @ y2
        x1 = U[0][0] @ (y1 - U[0][1] @ x2)

        return x1, x2

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetLayerIter']:

        # MATLAB: bemretlayeriter/solve.m
        # Initialize BEM solver (if needed)
        self._init_matrices(exc.enei)

        # External excitation
        phi, a, alpha, De = self._excitation(exc)

        # Size of excitation arrays
        siz1 = phi.shape
        siz2 = a.shape

        # Pack everything to single vector
        b = self._pack(phi, a, De, alpha)

        # Function for matrix multiplication
        fa = self._afun
        fm = None
        if self.precond is not None:
            fm = self._mfun

        # Iterative solution
        x, self_updated = self._iter_solve(None, b, fa, fm)

        # Unpack and save solution vector
        sig1, h1, sig2, h2 = self._unpack(x, nout = 4)

        # Reshape surface charges and currents
        if len(siz1) > 1:
            sig1 = sig1.reshape(siz1)
            sig2 = sig2.reshape(siz1)
        if len(siz2) > 2:
            h1 = h1.reshape(siz2)
            h2 = h2.reshape(siz2)

        sig = CompStruct(self.p, exc.enei,
            sig1 = sig1, sig2 = sig2, h1 = h1, h2 = h2)

        return sig, self

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetLayerIter']:

        # MATLAB: bemretlayeriter/mldivide.m
        return self.solve(exc)

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        return CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            a1 = pot1.a1, a1p = pot1.a1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p,
            a2 = pot2.a2, a2p = pot2.a2p)

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemretlayeriter/field.m
        # Waxenegger et al., Comp. Phys. Commun. 193, 138 (2015)
        if hasattr(self.g, 'deriv') and self.g.deriv == 'cart':
            return self.g.field(sig, inout)

        # Norm-based derivative approach
        k = 2 * np.pi / sig.enei
        pot = self.potential(sig, inout)

        if hasattr(pot, 'phi1'):
            phi, phip, a, ap = pot.phi1, pot.phi1p, pot.a1, pot.a1p
        else:
            phi, phip, a, ap = pot.phi2, pot.phi2p, pot.a2, pot.a2p

        # Tangential directions via interpolation
        phi1_d, phi2_d = self.p.deriv(self.p.interp(phi))[:2]
        a1_d, a2_d, t1, t2 = self.p.deriv(self.p.interp(a))

        # Normal vector
        nvec = np.cross(t1, t2)
        h = np.sqrt(np.sum(nvec * nvec, axis = 1, keepdims = True))
        nvec = nvec / h

        # Tangential vectors
        tvec1 = np.cross(t2, nvec) / h
        tvec2 = -np.cross(t1, nvec) / h

        # Electric field
        e = 1j * k * a - \
            self._outer(nvec, phip) - \
            self._outer(tvec1, phi1_d) - \
            self._outer(tvec2, phi2_d)

        # Magnetic field
        def _matcross(v: np.ndarray, a_d: np.ndarray) -> np.ndarray:
            if a_d.ndim == 2:
                return np.cross(v, a_d)
            else:
                n_pts = v.shape[0]
                siz = a_d.shape[2]
                result = np.empty((n_pts, 3, siz), dtype = a_d.dtype)
                for s in range(siz):
                    result[:, :, s] = np.cross(v, a_d[:, :, s])
                return result

        h_field = _matcross(tvec1, a1_d) + _matcross(tvec2, a2_d) + _matcross(nvec, ap)

        return CompStruct(self.p, sig.enei, e = e, h = h_field)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemretlayeriter/potential.m
        return self.g.potential(sig, inout)

    def clear(self) -> 'BEMRetLayerIter':

        # MATLAB: bemretlayeriter/clear.m
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None
        self._sav = None
        return self

    def __call__(self,
            enei: float) -> 'BEMRetLayerIter':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMRetLayerIter(p: {} faces, solver={}, {})'.format(n, self.solver, status)


class _LayerGreen(object):

    def __init__(self) -> None:
        self.ss = None
        self.hh = None
        self.p = None
        self.sh = None
        self.hs = None
