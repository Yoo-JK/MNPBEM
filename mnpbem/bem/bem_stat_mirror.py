import numpy as np
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..greenfun.compgreen_stat_mirror import CompGreenStatMirror
from ..geometry.comparticle_mirror import CompStructMirror


class BEMStatMirror(object):
    """BEM solver for quasistatic approximation with mirror symmetry.

    Given an external excitation, BEMStatMirror computes the surface
    charges such that the boundary conditions of Maxwell's equations
    in the quasistatic approximation are fulfilled.

    MATLAB: @bemstatmirror

    Parameters
    ----------
    p : ComParticleMirror
        Composite particle with mirror symmetry
    enei : float, optional
        Light wavelength in vacuum for pre-initialization
    """

    name = 'bemsolver'
    needs = {'sim': 'stat', 'sym': True}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:
        self.p = p
        self.enei = None  # type: Optional[float]

        # Green function
        self.g = CompGreenStatMirror(p, p, **options)

        # surface derivative of Green function (list, one per symmetry value)
        self.F = self.g.F

        # resolvent matrices
        self.mat = None  # type: Optional[List]

        if enei is not None:
            self._init_matrices(enei)

    def _init_matrices(self, enei: float) -> 'BEMStatMirror':
        """Initialize matrices for BEM solver.

        MATLAB: @bemstatmirror/subsref.m case '()'
        """
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # inside and outside dielectric function
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)

        # Lambda [Garcia de Abajo, Eq. (23)]
        lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        self.mat = []
        for i in range(len(self.F)):
            # BEM resolvent matrix
            self.mat.append(-np.linalg.inv(np.diag(lambda_diag) + self.F[i]))

        self.enei = enei
        return self

    def solve(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatMirror']:
        """Surface charge for given excitation.

        MATLAB: @bemstatmirror/mldivide.m

        Parameters
        ----------
        exc : CompStructMirror
            External excitation with field 'phip'

        Returns
        -------
        sig : CompStructMirror
            Surface charge
        obj : BEMStatMirror
            Updated solver
        """
        self._init_matrices(exc.enei)

        sig = CompStructMirror(self.p, exc.enei, getattr(exc, 'fun', None))

        for i in range(len(exc.val)):
            ind = self.p.symindex(exc.val[i].symval[-1, :])

            sig_val = _matmul(self.mat[ind], exc.val[i].phip)

            val = CompStruct(self.p, exc.enei, sig = sig_val)
            val.symval = exc.val[i].symval
            sig.val.append(val)

        return sig, self

    def __truediv__(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatMirror']:
        return self.solve(exc)

    def __mul__(self, sig: CompStructMirror) -> CompStructMirror:
        """Induced potential for given surface charge.

        MATLAB: @bemstatmirror/mtimes.m
        """
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        result = CompStructMirror(self.p, sig.enei, sig.fun)
        for i in range(len(sig.val)):
            combined = CompStruct(self.p, sig.enei)
            for attr in ('phi1', 'phi1p'):
                v = getattr(pot1.val[i], attr, None)
                if v is not None:
                    setattr(combined, attr, v)
            for attr in ('phi2', 'phi2p'):
                v = getattr(pot2.val[i], attr, None)
                if v is not None:
                    setattr(combined, attr, v)
            combined.symval = sig.val[i].symval
            result.val.append(combined)

        return result

    def potential(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Potentials and surface derivatives inside/outside of particle.

        MATLAB: @bemstatmirror/potential.m
        """
        return self.g.potential(sig, inout)

    def field(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Electric field inside/outside of particle surface.

        MATLAB: @bemstatmirror/field.m
        """
        return self.g.field(sig, inout)

    def __call__(self, enei: float) -> 'BEMStatMirror':
        return self._init_matrices(enei)

    def __repr__(self) -> str:
        status = 'enei={}'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatMirror(p={}, {})'.format(self.p, status)


def _matmul(a: Any, x: Any) -> Any:
    if isinstance(a, (int, float)):
        if a == 0:
            return 0
        return a * x
    if isinstance(x, (int, float)):
        if x == 0:
            return 0
        return a * x
    if np.isscalar(a):
        return a * x
    if isinstance(a, np.ndarray) and isinstance(x, np.ndarray):
        if x.ndim == 1:
            return a @ x
        elif x.ndim == 2:
            return a @ x
    return a @ x
