import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompGreenStatLayer, CompStruct


class BEMStatLayer(object):

    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self,
            p: Any,
            layer: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        self.p = p
        self.layer = layer

        self.enei = None
        self.mat = None

        # Green function with layer
        # MATLAB: obj.g = compgreenstatlayer(p, p, layer, varargin{:})
        self.g = CompGreenStatLayer(p, p, layer, **options)

        # Surface derivative of Green function
        self.F = self.g.F

        if enei is not None:
            self(enei)

    def _init_matrices(self,
            enei: float) -> 'BEMStatLayer':

        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # Get Green function with reflected contribution
        F_total = self.g.eval(enei, 'F')

        # Inside and outside dielectric function
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)

        # Lambda [Garcia de Abajo, Eq. (23)]
        # MATLAB: lambda = 2 * pi * (eps1 + eps2) ./ (eps1 - eps2)
        lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        # BEM resolvent matrix
        # MATLAB: obj.mat = -inv(diag(lambda) + F_total)
        Lambda = np.diag(lambda_diag)
        self.mat = -np.linalg.inv(Lambda + F_total)

        self.enei = enei

        return self

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatLayer']:

        return self.__truediv__(exc)

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatLayer']:

        self._init_matrices(exc.enei)

        sig_result = self.mat @ exc.phip
        sig = CompStruct(self.p, exc.enei, sig = sig_result)

        return sig, self

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        return self.g.field(sig, inout)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        return self.g.potential(sig, inout)

    def clear(self) -> 'BEMStatLayer':

        self.mat = None
        self.enei = None
        return self

    def __call__(self,
            enei: float) -> 'BEMStatLayer':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        return 'BEMStatLayer(p: {} faces, {})'.format(n, status)
