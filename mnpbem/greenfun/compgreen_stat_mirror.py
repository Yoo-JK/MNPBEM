"""
Quasistatic Green function for composite particles with mirror symmetry.

MATLAB: Greenfun/@compgreenstatmirror/
100% identical to MATLAB MNPBEM implementation.
"""

import numpy as np
from typing import Optional, Tuple, Any, List

from .compgreen_stat import CompGreenStat, CompStruct


class CompGreenStatMirror:
    """
    Quasistatic Green function for composite particles with mirror symmetry.

    MATLAB: @compgreenstatmirror

    For particles with x, y, or xy mirror symmetry, the Green function
    is computed between the particle and its full (expanded) version,
    then contracted using the symmetry table.

    Properties
    ----------
    name : str
        'greenfunction' (constant)
    needs : dict
        {'sim': 'stat', 'sym': True} (constant)
    p : ComParticleMirror
        Particle with mirror symmetry
    g : CompGreenStat
        Green function connecting p and full(p)

    Methods
    -------
    __init__(p, dum, **options)
        Constructor - initialize Green functions for mirror symmetry
    eval(*keys, ind=None)
        Evaluate Green function (G, F, H1, H2, Gp, H1p, H2p)
    field(sig, inout=1)
        Electric field inside/outside of particle surface
    potential(sig, inout=1)
        Potentials and surface derivatives inside/outside of particle
    """

    # Class constants
    name = 'greenfunction'
    needs = {'sim': 'stat', 'sym': True}

    def __init__(self, p, dum=None, **options):
        """
        Initialize Green functions for composite object and mirror symmetry.

        MATLAB: @compgreenstatmirror/compgreenstatmirror.m

        Parameters
        ----------
        p : ComParticleMirror
            Particle with mirror symmetry
        dum : any, optional
            Dummy argument to match CompGreenStat calling sequence
        **options : dict
            Options passed to CompGreenStat

        Examples
        --------
        >>> from mnpbem import trisphere, EpsConst, ComParticleMirror
        >>> from mnpbem.greenfun import CompGreenStatMirror
        >>>
        >>> eps = [EpsConst(1.0), EpsConst(2.0)]
        >>> p = trisphere(144, 10.0)
        >>> cp = ComParticleMirror(eps, [p], [[2, 1]], sym='x')
        >>> g = CompGreenStatMirror(cp)
        """
        self.p = p
        # Initialize Green function between p and full(p)
        self.g = CompGreenStat(p, p.full(), **options)

    def eval(self, *keys, **kwargs):
        """
        Evaluate quasistatic Green function with mirror symmetry.

        MATLAB: @compgreenstatmirror/eval.m

        Parameters
        ----------
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
        # Get Green function from compgreenstat
        mat = self.g.eval(*keys, **kwargs)

        # Symmetry table
        tab = self.p.symtable

        # Allocate output
        n_sym = tab.shape[0]
        g = [np.zeros_like(mat) if not isinstance(mat, tuple) else
             [np.zeros_like(m) for m in mat] for _ in range(n_sym)]

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
            return [tuple(r[i] for r in results) for i in range(n_sym)]

    def field(self, sig, inout=1):
        """
        Electric field inside/outside of particle surface.

        MATLAB: @compgreenstatmirror/field.m

        Parameters
        ----------
        sig : CompStructMirror
            Surface charges with mirror symmetry
        inout : int
            Fields inside (1, default) or outside (2) of particle surface

        Returns
        -------
        field : CompStructMirror
            Electric field with mirror symmetry
        """
        # Get derivative of Green function
        if inout == 1:
            Hp = self.eval('H1p')
        else:
            Hp = self.eval('H2p')

        # Allocate output
        field = CompStructMirror(sig.p, sig.enei, sig.fun)

        # Loop over symmetry values
        for i, isig in enumerate(sig.val):
            # Index of symmetry values within symmetry table
            x = self.p.symindex(isig.symval[0, :])
            y = self.p.symindex(isig.symval[1, :])
            z = self.p.symindex(isig.symval[2, :])

            # Electric field: e = -Hp @ sig
            e = -np.stack([
                self._matmul(Hp[x], isig.sig),
                self._matmul(Hp[y], isig.sig),
                self._matmul(Hp[z], isig.sig)
            ], axis=1)

            # Set output
            field.val[i] = CompStruct(sig.p, sig.enei, e=e)
            field.val[i].symval = isig.symval

        return field

    def potential(self, sig, inout=1):
        """
        Potentials and surface derivatives inside/outside of particle.

        MATLAB: @compgreenstatmirror/potential.m

        Parameters
        ----------
        sig : CompStructMirror
            Surface charges with mirror symmetry
        inout : int
            Potentials inside (1, default) or outside (2) of particle

        Returns
        -------
        pot : CompStructMirror
            Potentials with mirror symmetry
        """
        # Get Green function and surface derivative
        H_key = 'H1' if inout == 1 else 'H2'
        G = self.eval('G')
        H = self.eval(H_key)

        # Allocate output
        pot = CompStructMirror(sig.p, sig.enei, sig.fun)

        # Loop over symmetry values
        for i, isig in enumerate(sig.val):
            # Index of z-symmetry
            z = self.p.symindex(isig.symval[2, :])

            # Potential and surface derivative
            phi = self._matmul(G[z], isig.sig)
            phip = self._matmul(H[z], isig.sig)

            # Set output
            if inout == 1:
                pot.val[i] = CompStruct(sig.p, sig.enei, phi1=phi, phi1p=phip)
            else:
                pot.val[i] = CompStruct(sig.p, sig.enei, phi2=phi, phi2p=phip)
            pot.val[i].symval = isig.symval

        return pot

    def _matmul(self, a, x):
        """Matrix multiplication helper."""
        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
            return a * x
        return a @ x

    def __repr__(self):
        """String representation."""
        return f"CompGreenStatMirror(p: {self.p}, g: {self.g})"

    def __str__(self):
        """Detailed string representation."""
        return (
            f"compgreenstatmirror:\n"
            f"  p: {self.p}\n"
            f"  g: {self.g}"
        )


class CompStructMirror:
    """
    Structure for compound of particles with mirror symmetry.

    MATLAB: @compstructmirror

    Contains:
    - a reference to the particle with mirror symmetry
    - the light wavelength in vacuum
    - values for each symmetry configuration
    """

    def __init__(self, p, enei=None, fun=None):
        """
        Initialize compstructmirror.

        Parameters
        ----------
        p : ComParticleMirror
            Particle with mirror symmetry
        enei : float, optional
            Light wavelength in vacuum
        fun : callable, optional
            Function for symmetry operations
        """
        self.p = p
        self.enei = enei
        self.fun = fun
        self.val = []

    def __repr__(self):
        """String representation."""
        return f"CompStructMirror(p={self.p}, enei={self.enei}, n_val={len(self.val)})"
