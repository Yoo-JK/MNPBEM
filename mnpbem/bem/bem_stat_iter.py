import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..greenfun import CompStruct
from .bem_iter import BEMIter


class BEMStatIter(BEMIter):

    # MATLAB: @bemstatiter properties (Constant)
    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        # Initialize BEMIter base class
        super(BEMStatIter, self).__init__(**options)

        # MATLAB: @bemstatiter properties
        self.p = p
        self.enei = None
        self.F = None

        # MATLAB: @bemstatiter properties (Access = private)
        self._op = options
        self._g = None
        self._lambda = None
        self._mat = None

        # Green function
        # MATLAB: obj.g = aca.compgreenstat(p, varargin{:}, 'htol', ...)
        # For iterative solver, Green function is computed as H-matrix
        self._init_green(p, **options)

        # Initialize for given wavelength
        if enei is not None:
            self._init_matrices(enei)

    def _init_green(self,
            p: Any,
            **options: Any) -> None:

        # MATLAB: bemstatiter/private/init.m
        # In the iterative version, Green function uses H-matrix (ACA) approximation
        # For now, compute full Green function -- H-matrix support can be added later
        from ..greenfun import CompGreenStat
        self._g = CompGreenStat(p, p, **options)

        # Surface derivative of Green function
        # MATLAB: obj.F = eval(obj.g, 'F')
        self.F = self._g.F

    def _init_matrices(self,
            enei: float) -> 'BEMStatIter':

        # MATLAB: bemstatiter/private/initmat.m
        if self.enei is not None and self.enei == enei:
            return self

        self.enei = enei

        # Dielectric functions
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)

        # Lambda function [Garcia de Abajo, Eq. (23)]
        # MATLAB: obj.lambda = 2 * pi * (eps1 + eps2) ./ (eps1 - eps2)
        self._lambda = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        # Initialize preconditioner
        if self.precond is not None:
            F = self.F
            Lambda = np.diag(self._lambda)

            if self.precond == 'hmat':
                # MATLAB: obj.mat = lu(-lambda - F)
                # For Python, store the inverse for solving
                self._mat = np.linalg.inv(-Lambda - F)

            elif self.precond == 'full':
                # MATLAB: obj.mat = inv(-lambda - full(F))
                self._mat = np.linalg.inv(-Lambda - F)

            else:
                raise ValueError('[error] preconditioner not known: <{}>'.format(self.precond))

        return self

    def _afun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemstatiter/private/afun.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        vec_2d = vec.reshape(n, -1)

        # -(lambda + F) * vec
        result = -(self.F @ vec_2d + vec_2d * self._lambda[:, np.newaxis])
        return result.reshape(-1)

    def _mfun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemstatiter/private/mfun.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        vec_2d = vec.reshape(n, -1)

        if self.precond == 'hmat' or self.precond == 'full':
            # MATLAB: vec = solve(obj.mat, vec) or obj.mat * vec
            result = self._mat @ vec_2d
        else:
            result = vec_2d

        return result.reshape(-1)

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatIter']:

        # MATLAB: bemstatiter/solve.m
        # Initialize BEM solver (if needed)
        self._init_matrices(exc.enei)

        # Excitation and size of excitation array
        b = exc.phip.ravel()
        siz = exc.phip.shape

        # Function for matrix multiplication
        fa = self._afun
        fm = None
        if self.precond is not None:
            fm = self._mfun

        # Iterative solution
        x, self_updated = self._iter_solve(None, b, fa, fm)

        # Save everything in single structure
        sig = CompStruct(self.p, exc.enei, sig = x.reshape(siz))

        return sig, self

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatIter']:

        # MATLAB: bemstatiter/mldivide.m
        return self.solve(exc)

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        # MATLAB: bemstatiter/mtimes.m
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemstatiter/field.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        nvec = self.p.nvec

        # Electric field in normal direction
        if inout == 1:
            H = self._g.H1
        else:
            H = self._g.H2

        # MATLAB: e = -outer(obj.p.nvec, matmul(obj.g.H, sig.sig))
        H_sig = H @ sig.sig.reshape(n, -1)
        if H_sig.ndim == 1:
            e = -nvec * H_sig[:, np.newaxis]
        else:
            e = -nvec[:, :, np.newaxis] * H_sig[:, np.newaxis, :]

        # Tangential directions via interpolation
        G_sig = self._g.G @ sig.sig.reshape(n, -1)
        phi = self.p.interp(G_sig)
        phi1, phi2, t1, t2 = self.p.deriv(phi)

        # Normal vector
        nvec_c = np.cross(t1, t2)
        h = np.sqrt(np.sum(nvec_c * nvec_c, axis = 1, keepdims = True))
        nvec_c = nvec_c / h

        # Tangential derivative of PHI
        tvec1 = np.cross(t2, nvec_c) / h
        tvec2 = np.cross(t1, nvec_c) / h

        if phi1.ndim == 1:
            phip = tvec1 * phi1[:, np.newaxis] - tvec2 * phi2[:, np.newaxis]
        else:
            phip = tvec1[:, :, np.newaxis] * phi1[:, np.newaxis, :] - \
                   tvec2[:, :, np.newaxis] * phi2[:, np.newaxis, :]

        e = e - phip

        return CompStruct(self.p, sig.enei, e = e)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemstatiter/potential.m
        return self._g.potential(sig, inout)

    def clear(self) -> 'BEMStatIter':

        # MATLAB: bemstatiter/clear.m
        self._mat = None
        return self

    def __call__(self,
            enei: float) -> 'BEMStatIter':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatIter(p: {} faces, solver={}, {})'.format(n, self.solver, status)
