import numpy as np
from scipy.sparse.linalg import eigs
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..greenfun.compgreen_stat_mirror import CompGreenStatMirror
from ..geometry.comparticle_mirror import CompStructMirror


class BEMStatEigMirror(object):
    """BEM solver for quasistatic approximation and eigenmode expansion
    using mirror symmetry.

    Given an external excitation, BEMStatEigMirror computes the surface
    charges such that the boundary conditions of Maxwell's equations in
    the quasistatic approximation (using eigenmode expansion) are fulfilled.

    MATLAB: @bemstateigmirror

    Parameters
    ----------
    p : ComParticleMirror
        Composite particle with mirror symmetry
    nev : int
        Number of eigenmodes
    enei : float, optional
        Light wavelength in vacuum for pre-initialization
    """

    name = 'bemsolver'
    needs = {'sim': 'stat', 'nev': True, 'sym': True}

    def __init__(self,
            p: Any,
            nev: int = 20,
            enei: Optional[float] = None,
            **options: Any) -> None:
        self.p = p
        self.nev = nev
        self.enei = None  # type: Optional[float]

        # eigenmodes (one set per symmetry value)
        self.ur = []  # type: List[np.ndarray]
        self.ul = []  # type: List[np.ndarray]
        self.ene = []  # type: List[np.ndarray]
        self.unit = []  # type: List[np.ndarray]

        # resolvent matrices
        self.mat = None  # type: Optional[List]

        # Green function
        self.g = CompGreenStatMirror(p, p, **options)

        # surface derivative of Green function (list, one per symmetry value)
        F_list = self.g.F

        # eigenmode expansion
        for i in range(len(F_list)):
            F_i = F_list[i]

            # left and right eigenvectors
            # eigs returns (eigenvalues, eigenvectors) where eigenvectors is (n, k)
            _, ul_i = eigs(F_i.T, k = self.nev, which = 'SR', maxiter = 1000)
            ul_i = ul_i.T  # (nev, n)
            ene_i, ur_i = eigs(F_i, k = self.nev, which = 'SR', maxiter = 1000)
            # ur_i is (n, nev), ene_i is (nev,)
            ene_i = np.diag(ene_i)

            # make eigenvectors orthogonal
            overlap = ul_i @ ur_i  # (nev, nev)
            ul_i = np.linalg.solve(overlap, ul_i)

            # unit matrices
            unit_i = np.zeros((self.nev ** 2, p.np), dtype = complex)
            for ip in range(p.np):
                ind = p.index_func(ip + 1)
                chunk = ul_i[:, ind] @ ur_i[ind, :]  # (nev, nev)
                unit_i[:, ip] = chunk.ravel()

            self.ur.append(ur_i)
            self.ul.append(ul_i)
            self.ene.append(ene_i)
            self.unit.append(unit_i)

        if enei is not None:
            self._init_matrices(enei)

    def _init_matrices(self, enei: float) -> 'BEMStatEigMirror':
        """Initialize matrices for BEM solver.

        MATLAB: @bemstateigmirror/subsref.m case '()'
        """
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # dielectric functions
        eps_vals = [eps_func(enei)[0] for eps_func in self.p.eps]

        # inside and outside dielectric function
        eps1_arr = np.array([eps_vals[int(self.p.inout[j, 0]) - 1]
                            for j in range(self.p.inout.shape[0])])
        eps2_arr = np.array([eps_vals[int(self.p.inout[j, 1]) - 1]
                            for j in range(self.p.inout.shape[0])])

        # Lambda [Garcia de Abajo, Eq. (23)]
        Lambda = 2 * np.pi * (eps1_arr + eps2_arr) / (eps1_arr - eps2_arr)

        self.mat = []
        for i in range(len(self.ur)):
            # BEM resolvent matrix from eigenmodes
            unit_lambda = self.unit[i] @ Lambda[:]  # (nev^2,)
            unit_lambda_mat = unit_lambda.reshape(self.nev, self.nev)
            resolvent = unit_lambda_mat + self.ene[i]
            self.mat.append(-self.ur[i] @ np.linalg.solve(resolvent, self.ul[i]))

        self.enei = enei
        return self

    def solve(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatEigMirror']:
        """Surface charge for given excitation.

        MATLAB: @bemstateigmirror/mldivide.m

        Parameters
        ----------
        exc : CompStructMirror
            External excitation with field 'phip'

        Returns
        -------
        sig : CompStructMirror
            Surface charge
        obj : BEMStatEigMirror
            Updated solver
        """
        self._init_matrices(exc.enei)

        sig = CompStructMirror(self.p, exc.enei, exc.fun)

        for i in range(len(exc.val)):
            ind = self.p.symindex(exc.val[i].symval[-1, :])

            sig_val = _matmul(self.mat[ind], exc.val[i].phip)

            val = CompStruct(self.p, exc.enei, sig = sig_val)
            val.symval = exc.val[i].symval
            sig.val.append(val)

        return sig, self

    def __truediv__(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatEigMirror']:
        return self.solve(exc)

    def __mul__(self, sig: CompStructMirror) -> CompStructMirror:
        """Induced potential for given surface charge.

        MATLAB: @bemstateigmirror/mtimes.m
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

        MATLAB: @bemstateigmirror/potential.m
        """
        return self.g.potential(sig, inout)

    def field(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Electric field inside/outside of particle surface.

        MATLAB: @bemstateigmirror/field.m
        """
        return self.g.field(sig, inout)

    def __call__(self, enei: float) -> 'BEMStatEigMirror':
        return self._init_matrices(enei)

    def __repr__(self) -> str:
        status = 'enei={}'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatEigMirror(p={}, nev={}, {})'.format(self.p, self.nev, status)


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
        return a @ x
    return a @ x
