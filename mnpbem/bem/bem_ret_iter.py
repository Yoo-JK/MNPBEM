import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..greenfun import CompStruct
from .bem_iter import BEMIter


class BEMRetIter(BEMIter):

    # MATLAB: @bemretiter properties (Constant)
    name = 'bemsolver'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        # Initialize BEMIter base class
        super(BEMRetIter, self).__init__(**options)

        # MATLAB: @bemretiter properties
        self.p = p
        self.enei = None
        self.g = None

        # MATLAB: @bemretiter properties (Access = private)
        self._op = options
        self._sav = None
        self._k = None
        self._eps1 = None
        self._eps2 = None
        self._nvec = p.nvec
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None

        # Green function (with H-matrix / ACA approximation for iterative solver)
        # MATLAB: obj.g = aca.compgreenret(p, varargin{:}, ...)
        self._init_green(p, **options)

        # Initialize for given wavelength
        if enei is not None:
            self._init_matrices(enei)

    def _init_green(self,
            p: Any,
            **options: Any) -> None:

        # MATLAB: bemretiter/private/init.m
        from ..greenfun import CompGreenRet
        self.g = CompGreenRet(p, p, **options)

    def _init_matrices(self,
            enei: float) -> 'BEMRetIter':

        # MATLAB: bemretiter/private/initmat.m
        if self.enei is not None and self.enei == enei:
            return self

        self.enei = enei

        # Wavenumber
        self._k = 2 * np.pi / enei

        # Dielectric function
        self._eps1 = self.p.eps1(enei)
        self._eps2 = self.p.eps2(enei)

        # Green functions and surface derivatives
        # MATLAB: G1 = g{1,1}.G(enei) - g{2,1}.G(enei)
        G11 = self.g.eval(0, 0, 'G', enei)
        G21 = self.g.eval(1, 0, 'G', enei)
        G22 = self.g.eval(1, 1, 'G', enei)
        G12 = self.g.eval(0, 1, 'G', enei)

        self._G1 = G11 - G21 if not (isinstance(G21, (int, float)) and G21 == 0) else G11
        self._G2 = G22 - G12 if not (isinstance(G12, (int, float)) and G12 == 0) else G22

        H11 = self.g.eval(0, 0, 'H1', enei)
        H21 = self.g.eval(1, 0, 'H1', enei)
        H22 = self.g.eval(1, 1, 'H2', enei)
        H12 = self.g.eval(0, 1, 'H2', enei)

        self._H1 = H11 - H21 if not (isinstance(H21, (int, float)) and H21 == 0) else H11
        self._H2 = H22 - H12 if not (isinstance(H12, (int, float)) and H12 == 0) else H22

        # Initialize preconditioner
        if self.precond is not None:
            self._init_precond(enei)

        return self

    def _init_precond(self,
            enei: float) -> None:

        # MATLAB: bemretiter/private/initprecond.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        k = 2 * np.pi / enei
        eps1 = self._eps1
        eps2 = self._eps2
        nvec = self._nvec

        G1 = self._G1
        H1 = self._H1
        G2 = self._G2
        H2 = self._H2

        # Dielectric as diagonal matrices for matrix operations
        if np.isscalar(eps1) or (isinstance(eps1, np.ndarray) and eps1.ndim == 0):
            eps1_diag = eps1
            eps2_diag = eps2
        else:
            eps1_diag = np.diag(eps1)
            eps2_diag = np.diag(eps2)

        # Inverse Green function
        G1i = np.linalg.inv(G1)
        G2i = np.linalg.inv(G2)

        # Sigma matrices [Eq. (21)]
        Sigma1 = H1 @ G1i
        Sigma2 = H2 @ G2i

        # Inverse Delta matrix
        Deltai = np.linalg.inv(Sigma1 - Sigma2)

        # deps = eps1 - eps2
        if np.isscalar(eps1_diag):
            deps = eps1_diag - eps2_diag
        else:
            deps = eps1_diag - eps2_diag

        # Sigma matrix [Eq. (21,22)]
        # MATLAB: Sigma = eps1 * Sigma1 - eps2 * Sigma2 + k^2 * deps * fun(Deltai, nvec) * deps
        # fun(Deltai, nvec) = sum_i nvec_i * Deltai * nvec_i
        Deltai_nvec = self._decorate_deltai(Deltai, nvec)

        if np.isscalar(eps1_diag):
            Sigma_mat = eps1_diag * Sigma1 - eps2_diag * Sigma2 + k ** 2 * deps * Deltai_nvec * deps
        else:
            Sigma_mat = eps1_diag @ Sigma1 - eps2_diag @ Sigma2 + k ** 2 * deps @ Deltai_nvec @ deps

        Sigmai = np.linalg.inv(Sigma_mat)

        # Save variables for preconditioner
        sav = {}
        sav['k'] = k
        sav['nvec'] = nvec
        sav['G1i'] = G1i
        sav['G2i'] = G2i
        sav['eps1'] = eps1_diag
        sav['eps2'] = eps2_diag
        sav['Sigma1'] = Sigma1
        sav['Deltai'] = Deltai
        sav['Sigmai'] = Sigmai

        self._sav = sav

    @staticmethod
    def _decorate_deltai(
            Deltai: np.ndarray,
            nvec: np.ndarray) -> np.ndarray:

        # MATLAB: fun(Deltai, nvec) in initprecond.m
        # Deltai_nvec = nvec1 * Deltai * nvec1 + nvec2 * Deltai * nvec2 + nvec3 * Deltai * nvec3
        n = nvec.shape[0]
        result = np.zeros((n, n), dtype = Deltai.dtype)
        for i in range(3):
            nvec_i = np.diag(nvec[:, i])
            result = result + nvec_i @ Deltai @ nvec_i
        return result

    def _pack(self,
            phi: np.ndarray,
            a: np.ndarray,
            phip: np.ndarray,
            ap: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/pack.m
        total_len = phi.size + a.size + phip.size + ap.size
        vec = np.empty(total_len, dtype = complex)
        offset = 0
        for arr in [phi, a, phip, ap]:
            flat = arr.ravel()
            vec[offset:offset + flat.size] = flat
            offset += flat.size
        return vec

    def _unpack(self,
            vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: bemretiter/private/unpack.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # last dimension
        siz = int(vec.size / (8 * n))

        # reshape vector
        vec_2d = vec.reshape(-1, 8)

        # extract potentials from vector
        phi = vec_2d[:, 0].reshape(n, siz) if siz > 1 else vec_2d[:, 0].reshape(n)
        a = vec_2d[:, 1:4].reshape(n, 3, siz) if siz > 1 else vec_2d[:, 1:4].reshape(n, 3)
        phip = vec_2d[:, 4].reshape(n, siz) if siz > 1 else vec_2d[:, 4].reshape(n)
        ap = vec_2d[:, 5:8].reshape(n, 3, siz) if siz > 1 else vec_2d[:, 5:8].reshape(n, 3)

        return phi, a, phip, ap

    @staticmethod
    def _outer(
            nvec: np.ndarray,
            val: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretiter/private/outer.m
        if isinstance(val, (int, float)) and val == 0:
            return 0

        if mul is not None:
            if val.ndim == 1:
                val = val * mul
            else:
                val = val * mul[:, np.newaxis] if mul.ndim == 1 else val * mul

        if val.ndim == 1:
            # val: (n,), nvec: (n, 3) -> result: (n, 3)
            return nvec * val[:, np.newaxis]
        else:
            # val: (n, siz), nvec: (n, 3) -> result: (n, 3, siz)
            siz = val.shape[1]
            n = val.shape[0]
            result = np.empty((n, 3, siz), dtype = val.dtype)
            for i in range(3):
                result[:, i, :] = val * nvec[:, i:i + 1]
            return result

    @staticmethod
    def _inner(
            nvec: np.ndarray,
            a: Any,
            mul: Optional[np.ndarray] = None) -> Any:

        # MATLAB: bemretiter/private/inner.m
        if isinstance(a, (int, float)) and a == 0:
            return 0

        if a.ndim == 2:
            # a: (n, 3), nvec: (n, 3) -> result: (n,)
            result = np.sum(a * nvec, axis = 1)
        elif a.ndim == 3:
            # a: (n, 3, siz), nvec: (n, 3) -> result: (n, siz)
            result = np.sum(a * nvec[:, :, np.newaxis], axis = 1)
        else:
            result = a

        if mul is not None:
            if result.ndim == 1:
                result = result * mul
            else:
                result = result * mul[:, np.newaxis] if mul.ndim == 1 else result * mul

        return result

    def _excitation(self,
            exc: CompStruct) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: bemretiter/private/excitation.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces

        # Default values for potentials
        phi1 = getattr(exc, 'phi1', 0)
        phi1p = getattr(exc, 'phi1p', 0)
        a1 = getattr(exc, 'a1', 0)
        a1p = getattr(exc, 'a1p', 0)
        phi2 = getattr(exc, 'phi2', 0)
        phi2p = getattr(exc, 'phi2p', 0)
        a2 = getattr(exc, 'a2', 0)
        a2p = getattr(exc, 'a2p', 0)

        k = 2 * np.pi / exc.enei
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

        return phi, a, De, alpha

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

    def _afun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/afun.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        siz = int(vec.size / 2)

        # Split vector array
        vec1 = vec[:siz].reshape(n, -1)
        vec2 = vec[siz:].reshape(n, -1)

        # Multiplication with Green functions
        G1_vec1 = self._G1 @ vec1
        G2_vec2 = self._G2 @ vec2

        # Pack into combined vector for unpack
        combined_g = np.empty(G1_vec1.size + G2_vec2.size, dtype = complex)
        combined_g[:G1_vec1.size] = G1_vec1.ravel()
        combined_g[G1_vec1.size:] = G2_vec2.ravel()
        Gsig1, Gh1, Gsig2, Gh2 = self._unpack(combined_g)

        H1_vec1 = self._H1 @ vec1
        H2_vec2 = self._H2 @ vec2
        combined_h = np.empty(H1_vec1.size + H2_vec2.size, dtype = complex)
        combined_h[:H1_vec1.size] = H1_vec1.ravel()
        combined_h[H1_vec1.size:] = H2_vec2.ravel()
        Hsig1, Hh1, Hsig2, Hh2 = self._unpack(combined_h)

        k = self._k
        nvec = self._nvec
        eps1 = self._eps1
        eps2 = self._eps2

        def _matmul_diag(a_val: Any, b: np.ndarray) -> np.ndarray:
            if np.isscalar(a_val):
                return a_val * b
            if b.ndim == 1:
                return a_val * b
            return a_val[:, np.newaxis] * b if a_val.ndim == 1 else a_val * b

        # Eq. (10)
        phi = Gsig1 - Gsig2
        # Eq. (11)
        a = Gh1 - Gh2

        # Eq. (14)
        alpha = Hh1 - Hh2 - \
            1j * k * self._outer(nvec, _matmul_diag(eps1, Gsig1) - _matmul_diag(eps2, Gsig2))
        # Eq. (17)
        De = _matmul_diag(eps1, Hsig1) - _matmul_diag(eps2, Hsig2) - \
            1j * k * self._inner(nvec, _matmul_diag(eps1, Gh1) - _matmul_diag(eps2, Gh2))

        return self._pack(phi, a, De, alpha)

    def _mfun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemretiter/private/mfun.m
        # Garcia de Abajo and Howie, PRB 65, 115418 (2002)

        # Unpack matrices
        phi, a, De, alpha = self._unpack(vec)

        sav = self._sav
        k = sav['k']
        nvec = sav['nvec']
        G1i = sav['G1i']
        G2i = sav['G2i']
        eps1 = sav['eps1']
        eps2 = sav['eps2']
        Sigma1 = sav['Sigma1']
        Deltai = sav['Deltai']
        Sigmai = sav['Sigmai']

        def matmul1(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            if b.ndim == 1:
                return a_mat @ b
            return a_mat @ b.reshape(b.shape[0], -1)

        def matmul2(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
            # Solve a_mat * x = b (equivalent to inv(a_mat) @ b for LU-based)
            if b.ndim == 1:
                return a_mat @ b
            return a_mat @ b.reshape(b.shape[0], -1)

        # Modify alpha and De
        # MATLAB: alpha = alpha - matmul1(Sigma1, a) + 1i*k*outer(nvec, eps1*phi)
        if np.isscalar(eps1):
            alpha = alpha - matmul1(Sigma1, a) + 1j * k * self._outer(nvec, eps1 * phi)
            De = De - eps1 * matmul1(Sigma1, phi) + 1j * k * eps1 * self._inner(nvec, a)
        else:
            alpha = alpha - matmul1(Sigma1, a) + 1j * k * self._outer(nvec, eps1 @ phi if phi.ndim > 1 else eps1 @ phi)
            De = De - eps1 @ matmul1(Sigma1, phi) + 1j * k * self._inner(nvec, eps1 @ a if a.ndim <= 2 else a)

        # Eq. (19)
        deps = eps1 - eps2
        if np.isscalar(deps):
            inner_alpha = self._inner(nvec, matmul1(Deltai, alpha))
            sig2 = matmul2(Sigmai, De + 1j * k * deps * inner_alpha)
        else:
            inner_alpha = self._inner(nvec, matmul1(Deltai, alpha))
            sig2 = matmul2(Sigmai, De + 1j * k * deps @ inner_alpha if inner_alpha.ndim <= 1 else De + 1j * k * (deps @ inner_alpha))

        # Eq. (20)
        if np.isscalar(deps):
            h2 = matmul1(Deltai, 1j * k * self._outer(nvec, deps * sig2) + alpha)
        else:
            h2 = matmul1(Deltai, 1j * k * self._outer(nvec, deps @ sig2 if sig2.ndim <= 1 else deps @ sig2) + alpha)

        # Surface charges and currents
        sig1 = matmul2(G1i, sig2 + phi)
        h1 = matmul2(G1i, h2 + a)
        sig2_out = matmul2(G2i, sig2)
        h2_out = matmul2(G2i, h2)

        result = self._pack(sig1, h1, sig2_out, h2_out)
        return result

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetIter']:

        # MATLAB: bemretiter/solve.m
        # Initialize BEM solver (if needed)
        self._init_matrices(exc.enei)

        # External excitation
        phi, a, De, alpha = self._excitation(exc)

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
        sig1, h1, sig2, h2 = self._unpack(x)

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
            exc: CompStruct) -> Tuple[CompStruct, 'BEMRetIter']:

        # MATLAB: bemretiter/mldivide.m
        return self.solve(exc)

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        # MATLAB: bemretiter/mtimes.m
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

        # MATLAB: bemretiter/field.m
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

        # MATLAB: bemretiter/potential.m
        return self.g.potential(sig, inout)

    def clear(self) -> 'BEMRetIter':

        # MATLAB: bemretiter/clear.m
        self._G1 = None
        self._H1 = None
        self._G2 = None
        self._H2 = None
        self._sav = None
        return self

    def __call__(self,
            enei: float) -> 'BEMRetIter':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMRetIter(p: {} faces, solver={}, {})'.format(n, self.solver, status)
