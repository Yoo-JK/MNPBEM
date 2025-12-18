"""
Retarded Green function for composite particles with mirror symmetry.

MATLAB: Greenfun/@compgreenretmirror/
100% identical to MATLAB MNPBEM implementation.
"""

import numpy as np
from typing import Optional, Tuple, Any, List

from .compgreen_ret import CompGreenRet
from .compgreen_stat import CompStruct
from .compgreen_stat_mirror import CompStructMirror


class CompGreenRetMirror:
    """
    Retarded Green function for composite particles with mirror symmetry.

    MATLAB: @compgreenretmirror

    For particles with x, y, or xy mirror symmetry, the Green function
    is computed between the particle and its full (expanded) version,
    then contracted using the symmetry table.

    Properties
    ----------
    name : str
        'greenfunction' (constant)
    needs : dict
        {'sim': 'ret', 'sym': True} (constant)
    p : ComParticleMirror
        Particle with mirror symmetry
    g : CompGreenRet
        Green function connecting p and full(p)

    Methods
    -------
    __init__(p, dum, **options)
        Constructor - initialize Green functions for mirror symmetry
    eval(enei, *keys)
        Evaluate Green function at wavelength (G, F, H1, H2, Gp, H1p, H2p)
    field(sig, inout=1)
        Electric and magnetic field inside/outside of particle surface
    potential(sig, inout=1)
        Potentials and surface derivatives inside/outside of particle
    """

    # Class constants
    name = 'greenfunction'
    needs = {'sim': 'ret', 'sym': True}

    def __init__(self, p, dum=None, **options):
        """
        Initialize Green functions for composite object and mirror symmetry.

        MATLAB: @compgreenretmirror/compgreenretmirror.m

        Parameters
        ----------
        p : ComParticleMirror
            Particle with mirror symmetry
        dum : any, optional
            Dummy argument to match CompGreenRet calling sequence
        **options : dict
            Options passed to CompGreenRet

        Examples
        --------
        >>> from mnpbem import trisphere, EpsConst, ComParticleMirror
        >>> from mnpbem.greenfun import CompGreenRetMirror
        >>>
        >>> eps = [EpsConst(1.0), EpsConst(2.0)]
        >>> p = trisphere(144, 10.0)
        >>> cp = ComParticleMirror(eps, [p], [[2, 1]], sym='x')
        >>> g = CompGreenRetMirror(cp)
        """
        self.p = p
        # Initialize Green function between p and full(p)
        self.g = CompGreenRet(p, p.full(), **options)

    @property
    def deriv(self):
        """Return derivative type from underlying Green function."""
        return self.g.deriv

    def eval(self, enei, *keys, **kwargs):
        """
        Evaluate retarded Green function with mirror symmetry.

        MATLAB: @compgreenretmirror/eval.m

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum (nm)
        *keys : str
            Keys for Green function components:
            - 'G'   : Green function
            - 'F'   : Surface derivative of Green function
            - 'H1'  : F + 2π (on diagonal)
            - 'H2'  : F - 2π (on diagonal)
            - 'Gp'  : Cartesian derivative of Green function
            - 'H1p' : Gp + 2π (on diagonal)
            - 'H2p' : Gp - 2π (on diagonal)

        Returns
        -------
        g : list
            Green function matrices contracted for each symmetry value
        """
        # Get Green function from compgreenret
        mat = self.g.eval(enei, *keys, **kwargs)

        # Handle case when mat is 0 (no result)
        if isinstance(mat, (int, float)) and mat == 0:
            tab = self.p.symtable
            return [0 for _ in range(tab.shape[0])]

        # Symmetry table
        tab = self.p.symtable

        # Handle single vs multiple outputs
        if not isinstance(mat, tuple):
            mats = [mat]
            single_output = True
        else:
            mats = list(mat)
            single_output = False

        # Process each output
        results = []
        for mat_single in mats:
            if isinstance(mat_single, (int, float)) and mat_single == 0:
                results.append([0 for _ in range(tab.shape[0])])
                continue

            siz = mat_single.shape

            # Decompose Green matrix into sub-matrices
            if len(siz) == 2:
                # G, F, H1, H2: shape (n1, n2)
                n_sub = siz[1] // siz[0]
                mat_parts = [mat_single[:, i*siz[0]:(i+1)*siz[0]]
                             for i in range(n_sub)]
            else:
                # Gp, H1p, H2p: shape (n1, 3, n2)
                n_sub = siz[2] // siz[0]
                mat_parts = [mat_single[:, :, i*siz[0]:(i+1)*siz[0]]
                             for i in range(n_sub)]

            # Contract Green function for different symmetry values
            g_contracted = []
            for i in range(tab.shape[0]):
                g_i = np.zeros_like(mat_parts[0])
                for j in range(tab.shape[1]):
                    g_i = g_i + tab[i, j] * mat_parts[j]
                g_contracted.append(g_i)
            results.append(g_contracted)

        if single_output:
            return results[0]
        else:
            # Reorganize: list of lists -> list of tuples
            return [tuple(r[i] for r in results) for i in range(tab.shape[0])]

    def field(self, sig, inout=1):
        """
        Electric and magnetic field inside/outside of particle surface.

        MATLAB: @compgreenretmirror/field.m

        Parameters
        ----------
        sig : CompStructMirror
            Surface charges and currents with mirror symmetry
        inout : int
            Fields inside (1, default) or outside (2) of particle surface

        Returns
        -------
        field : CompStructMirror
            Electric and magnetic field with mirror symmetry

        Notes
        -----
        Requires deriv='cart' (Cartesian derivatives).
        """
        # Cannot compute fields from just normal surface derivative
        assert self.g.deriv == 'cart', "Field computation requires deriv='cart'"

        # Wavelength and wavenumber
        enei = sig.enei
        k = 2 * np.pi / enei

        # Allocate output
        field = CompStructMirror(sig.p, enei, sig.fun)

        # Get Green functions for both symmetry configurations
        G1 = self.eval(enei, 'G')  # {inout, 1}
        G2 = self.eval(enei, 'G')  # {inout, 2}

        if inout == 1:
            H1p = self.eval(enei, 'H1p')  # {inout, 1}
            H2p = self.eval(enei, 'H1p')  # {inout, 2}
        else:
            H1p = self.eval(enei, 'H2p')
            H2p = self.eval(enei, 'H2p')

        # Loop over symmetry values
        for i, isig in enumerate(sig.val):
            # Index of symmetry values within symmetry table
            x = self.p.symindex(isig.symval[0, :])
            y = self.p.symindex(isig.symval[1, :])
            z = self.p.symindex(isig.symval[2, :])
            ind = [x, y, z]

            # Electric field: E = ik*A - grad(V)
            e = (1j * k * self._indmul(G1, isig.h1, ind) -
                 self._matmul(H1p[z], isig.sig1) +
                 1j * k * self._indmul(G2, isig.h2, ind) -
                 self._matmul(H2p[z], isig.sig2))

            # Magnetic field: H = curl(A)
            h = self._indcross(H1p, isig.h1, ind) + self._indcross(H2p, isig.h2, ind)

            # Set output
            field.val.append(CompStruct(sig.p, enei, e=e, h=h))
            field.val[i].symval = isig.symval

        return field

    def potential(self, sig, inout=1):
        """
        Potentials and surface derivatives inside/outside of particle.

        MATLAB: @compgreenretmirror/potential.m

        Parameters
        ----------
        sig : CompStructMirror
            Surface charges and currents with mirror symmetry
        inout : int
            Potentials inside (1, default) or outside (2) of particle

        Returns
        -------
        pot : CompStructMirror
            Potentials with mirror symmetry
        """
        # Wavelength and wavenumber
        enei = sig.enei
        k = 2 * np.pi / enei

        # Get Green function and surface derivative
        H_key = 'H1' if inout == 1 else 'H2'
        G = self.eval(enei, 'G')
        H = self.eval(enei, H_key)

        # Allocate output
        pot = CompStructMirror(sig.p, enei, sig.fun)

        # Loop over symmetry values
        for i, isig in enumerate(sig.val):
            # Index of symmetry values
            x = self.p.symindex(isig.symval[0, :])
            y = self.p.symindex(isig.symval[1, :])
            z = self.p.symindex(isig.symval[2, :])
            ind = [x, y, z]

            # Scalar potential
            phi1 = self._matmul(G[z], isig.sig1)
            phi2 = self._matmul(G[z], isig.sig2)
            phi1p = self._matmul(H[z], isig.sig1)
            phi2p = self._matmul(H[z], isig.sig2)

            # Vector potential
            a1 = self._indmul(G, isig.h1, ind)
            a2 = self._indmul(G, isig.h2, ind)
            a1p = self._indmul(H, isig.h1, ind) if self.deriv == 'cart' else None
            a2p = self._indmul(H, isig.h2, ind) if self.deriv == 'cart' else None

            # Set output
            if inout == 1:
                pot.val.append(CompStruct(sig.p, enei,
                                          phi1=phi1, phi2=phi2,
                                          phi1p=phi1p, phi2p=phi2p,
                                          a1=a1, a2=a2, a1p=a1p, a2p=a2p))
            else:
                pot.val.append(CompStruct(sig.p, enei,
                                          phi1=phi1, phi2=phi2,
                                          phi1p=phi1p, phi2p=phi2p,
                                          a1=a1, a2=a2, a1p=a1p, a2p=a2p))
            pot.val[i].symval = isig.symval

        return pot

    def _matmul(self, a, x):
        """Matrix multiplication helper."""
        if isinstance(a, (int, float)) and a == 0:
            return 0
        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
            return a * x
        return a @ x

    def _indmul(self, mat, v, ind):
        """
        Indexed matrix multiplication.

        MATLAB: indmul(mat, v, ind)
        """
        if isinstance(mat[0], (int, float)) and mat[0] == 0:
            return 0

        siz = list(v.shape)
        siz[1] = 1

        result = np.concatenate([
            self._matmul(mat[ind[0]], v[:, 0:1, :].reshape(siz)),
            self._matmul(mat[ind[1]], v[:, 1:2, :].reshape(siz)),
            self._matmul(mat[ind[2]], v[:, 2:3, :].reshape(siz))
        ], axis=1)

        return result

    def _indcross(self, mat, v, ind):
        """
        Indexed cross product.

        MATLAB: indcross(mat, v, ind)
        """
        if isinstance(mat[0], (int, float)) and mat[0] == 0:
            return 0

        siz = list(v.shape)
        siz[1] = 1

        # Matrix element accessor
        def imat(k, i):
            return np.squeeze(mat[ind[k]][:, i, :])

        # Vector element accessor
        def ivec(i):
            return v[:, i:i+1, :].reshape(siz)

        # Cross product components
        result = np.concatenate([
            self._matmul(imat(2, 1), ivec(2)) - self._matmul(imat(1, 2), ivec(1)),
            self._matmul(imat(0, 2), ivec(0)) - self._matmul(imat(2, 0), ivec(2)),
            self._matmul(imat(1, 0), ivec(1)) - self._matmul(imat(0, 1), ivec(0))
        ], axis=1)

        return result

    def __repr__(self):
        """String representation."""
        return f"CompGreenRetMirror(p: {self.p}, g: {self.g})"

    def __str__(self):
        """Detailed string representation."""
        return (
            f"compgreenretmirror:\n"
            f"  p: {self.p}\n"
            f"  g: {self.g}"
        )
