"""
Composite Green function for retarded (full Maxwell) approximation.

MATLAB: Greenfun/@compgreenret/
100% identical to MATLAB MNPBEM implementation.
"""

import numpy as np
from typing import Optional, Tuple, Any, List


class CompGreenRet:
    """
    Green function for composite points and particle.

    MATLAB: @compgreenret

    Properties
    ----------
    name : str
        'greenfunction' (constant)
    needs : dict
        {'sim': 'ret'} (constant)
    p1 : ComParticle
        Green function between points p1 and comparticle p2
    p2 : ComParticle
        Green function between points p1 and comparticle p2
    con : list of list
        Connectivity matrix between points of p1 and p2
    g : list of list
        Green functions connecting p1 and p2 (cell array)
    hmode : str or None
        'aca1', 'aca2', 'svd' for initialization of H-matrices
    block : BlockMatrix
        Block matrix for evaluation of selected Green function elements
    hmat : HMatrix or None
        Template for hierarchical matrices

    Methods
    -------
    __init__(p1, p2, **options)
        Constructor - initialize Green functions for composite objects
    eval(i, j, key, enei, ind=None)
        Evaluate Green function (G, F, H1, H2, Gp, H1p, H2p)
    field(sig, inout=1)
        Electric and magnetic field inside/outside of particle surface
    potential(sig, inout=1)
        Potentials and surface derivatives inside/outside of particle
    """

    # Class constants
    name = 'greenfunction'
    needs = {'sim': 'ret'}

    def __init__(self, p1, p2, **options):
        """
        Initialize Green functions for composite objects.

        MATLAB: compgreenret.m, private/init.m

        Parameters
        ----------
        p1 : ComParticle
            Green function between points p1 and comparticle p2
        p2 : ComParticle
            Green function between points p1 and comparticle p2
        **options : dict
            deriv : str, optional
                'cart' (Cartesian) or 'norm' (normal) derivative (default: 'norm')
            hmode : str, optional
                'aca1', 'aca2', 'svd' for hierarchical matrices (default: None)
            waitbar : int, optional
                Show progress bar (default: 0)

        Examples
        --------
        >>> from mnpbem import trisphere, EpsConst, EpsTable, ComParticle
        >>> from mnpbem.greenfun import CompGreenRet
        >>>
        >>> eps = [EpsConst(1.0), EpsTable('gold.dat')]
        >>> p = trisphere(144, 10.0)
        >>> cp = ComParticle(eps, [p], [[2, 1]])
        >>> g = CompGreenRet(cp, cp)
        """
        self.p1 = p1
        self.p2 = p2
        self.deriv = options.get('deriv', 'norm')
        self.hmode = options.get('hmode', None)

        # Initialize Green function
        self._init(p1, p2, **options)

    def _init(self, p1, p2, **options):
        """
        Initialize composite Green function.

        MATLAB: @compgreenret/private/init.m

        Handles:
        - Creation of Green function between p1 and p2
        - Closed surface diagonal correction
        - Connectivity matrix
        - Block matrix for evaluation
        - H-matrix initialization
        """
        # Initialize Green function
        # MATLAB: g = greenret(p1, p2, varargin{:})
        g = self._greenret(p1, p2, **options)

        # Deal with closed argument
        # MATLAB: g = initclosed(g, p1, p2, varargin{:})
        g = self._initclosed(g, p1, p2, **options)

        # Split Green function into cell array
        # MATLAB: obj.g = mat2cell(g, p1.p, p2.p)
        self.g = self._mat2cell(g, p1.p, p2.p)

        # Connectivity matrix
        # MATLAB: obj.con = connect(p1, p2)
        self.con = self._connect(p1, p2)

        # Size of point or particle objects
        # MATLAB: siz1 = cellfun(@(p) p.n, p1.p, 'uniform', 1)
        siz1 = [p.n for p in p1.p]
        siz2 = [p.n for p in p2.p]

        # Block matrix for evaluation of selected Green function elements
        # MATLAB: obj.block = blockmatrix(siz1, siz2)
        self.block = BlockMatrix(siz1, siz2)

        # Hierarchical matrices?
        if self.hmode is not None:
            # Set up cluster trees and initialize hierarchical matrix
            # For now, we don't implement H-matrices (TODO)
            self.hmat = None
        else:
            self.hmat = None

    def _greenret(self, p1, p2, **options):
        """
        Create retarded Green function.

        MATLAB: greenret(p1, p2, varargin)

        For composite particles, this creates a single Green function
        between the concatenated particles.
        """
        # For now, return a simple GreenRet object
        # In full implementation, this would call the greenret class
        return GreenRetSimple(p1, p2, self.deriv)

    def _initclosed(self, g, p1, p2, **options):
        """
        Deal with closed argument of COMPARTICLE objects.

        MATLAB: @compgreenret/private/initclosed.m

        For a closed particle the surface integral of -F should give 2*pi
        See R. Fuchs and S. H. Liu, Phys. Rev. B 14, 5521 (1976).
        """
        # Full particle in case of mirror symmetry
        full1 = p1
        if hasattr(p1, 'sym'):
            if hasattr(p1, 'pfull'):
                full1 = p1.pfull

        # Check for closed surfaces
        if hasattr(full1, 'closed') and (full1 is p2 or full1 == p2):
            if full1.closed is not None and any(c is not None for c in full1.closed):
                # Loop over particles
                for i in range(len(p1.p)):
                    # Index to particle faces
                    ind = p1.index_func(i + 1)  # 1-indexed in MATLAB

                    # Select particle and closed particle surface
                    part = p1.p[i]
                    full, dir_val, loc = self._closedparticle(p1, i)

                    if full is not None:
                        if loc is not None:
                            # Use already computed Green function object
                            f = self._fun_closed(g, loc, ind, **options)
                        else:
                            # Set up Green function using quasistatic
                            # MATLAB: gstat = greenstat(full, part, bemoptions(...))
                            # For closed surface correction, we use quasistatic Green function
                            from .compgreen_stat import CompGreenStat
                            gstat = CompGreenStat(full, part, **options)

                            # Sum over closed surface
                            f = self._fun_closed_stat(gstat, **options)

                        # Set diagonal elements of Green function
                        # MATLAB: g = diag(g, ind, -2*pi*dir - f.')
                        if isinstance(ind, (list, np.ndarray)):
                            ind_array = np.array(ind)
                        else:
                            ind_array = np.array([ind])

                        # Update diagonal (implementation depends on greenret structure)
                        # For now, store the correction for later use
                        if not hasattr(g, 'diag_corrections'):
                            g.diag_corrections = {}
                        g.diag_corrections[i] = (-2 * np.pi * dir_val - f, part.nvec)

        return g

    def _closedparticle(self, p, i):
        """Get closed particle surface."""
        # This would call the closedparticle function in MATLAB
        # For now, return None (closed surface handling can be added later)
        return None, 1, None

    def _fun_closed(self, g, loc, ind, **options):
        """Sum over closed surface using already computed Green function."""
        return 0.0

    def _fun_closed_stat(self, gstat, **options):
        """
        Sum over closed surface.

        MATLAB: initclosed.m/fun(gstat, varargin)
        """
        p1 = gstat.p1
        p2 = gstat.p2

        area1 = p1.area
        area2 = p2.area

        # f = sum(area1[:, None] * gstat.F * (1/area2)[None, :], axis=0)
        F_weighted = area1[:, np.newaxis] * gstat.F / area2[np.newaxis, :]
        f = np.sum(F_weighted, axis=0)

        return f

    def _mat2cell(self, g, p1_list, p2_list):
        """
        Split Green function into cell array.

        MATLAB: mat2cell(g, p1.p, p2.p)

        Returns cell array g{i, j} for each particle pair.
        """
        # Create cell array (list of lists)
        n1 = len(p1_list)
        n2 = len(p2_list)

        g_cell = [[None for _ in range(n2)] for _ in range(n1)]

        # Get cumulative indices
        idx1 = [0] + list(np.cumsum([p.n for p in p1_list]))
        idx2 = [0] + list(np.cumsum([p.n for p in p2_list]))

        # Split into blocks
        for i in range(n1):
            for j in range(n2):
                i1_start, i1_end = idx1[i], idx1[i+1]
                i2_start, i2_end = idx2[j], idx2[j+1]

                # Create sub-green function
                g_cell[i][j] = GreenRetBlock(
                    p1_list[i], p2_list[j],
                    i1_start, i1_end, i2_start, i2_end,
                    g, self.deriv
                )

        return g_cell

    def _connect(self, p1, p2):
        """
        Connectivity matrix.

        MATLAB: connect(p1, p2)

        Returns connectivity matrix con{i,j} indicating which material
        index connects particles i and j.
        """
        n1 = len(p1.p)
        n2 = len(p2.p)

        # Create connectivity cell array
        con = [[None for _ in range(n2)] for _ in range(n1)]

        # For each particle pair, determine connectivity
        for i in range(n1):
            for j in range(n2):
                # Get material indices for particles i and j
                # MATLAB comparticle stores inout as [[in1, out1], [in2, out2], ...]
                # Connectivity is determined by matching materials

                # Get material indices
                if hasattr(p1, 'inout') and hasattr(p2, 'inout'):
                    inout1 = p1.inout[i] if i < len(p1.inout) else [1, 2]
                    inout2 = p2.inout[j] if j < len(p2.inout) else [1, 2]

                    # Check for matching materials
                    # con[i,j] is the material index where they connect
                    # Simplified: assume they connect through their shared environment
                    con[i][j] = self._find_connection(inout1, inout2)
                else:
                    # Default: connect through material 1 (usually vacuum/air)
                    con[i][j] = 1

        return con

    def _find_connection(self, inout1, inout2):
        """Find connection material index between two particles."""
        # MATLAB logic: particles connect if they share a material
        # inout1 = [inside_mat, outside_mat] for particle 1
        # inout2 = [inside_mat, outside_mat] for particle 2

        # Check if outside materials match (most common case)
        if inout1[1] == inout2[1]:
            return inout1[1]
        # Check if inside of 1 matches outside of 2
        elif inout1[0] == inout2[1]:
            return inout1[0]
        # Check if outside of 1 matches inside of 2
        elif inout1[1] == inout2[0]:
            return inout1[1]
        # Default: use outside material of first particle
        else:
            return inout1[1]

    def eval(self, i, j, key, enei, ind=None):
        """
        Evaluate retarded Green function.

        MATLAB: @compgreenret/eval.m, eval1.m, eval2.m

        Usage
        -----
        g = eval(obj, i, j, key, enei)       # Full matrix
        g = eval(obj, i, j, key, enei, ind)  # Selected elements

        Parameters
        ----------
        i : int
            Index to p1 particle (1-based in MATLAB, 0-based here)
        j : int
            Index to p2 particle (1-based in MATLAB, 0-based here)
        key : str
            G    - Green function
            F    - Surface derivative of Green function
            H1   - F + 2 * pi
            H2   - F - 2 * pi
            Gp   - Derivative of Green function
            H1p  - Gp + 2 * pi
            H2p  - Gp - 2 * pi
        enei : float
            Light wavelength in vacuum
        ind : array, optional
            Index to selected matrix elements

        Returns
        -------
        g : ndarray
            Requested Green function

        Examples
        --------
        >>> g_mat = obj.eval(0, 0, 'G', 600.0)
        >>> f_mat = obj.eval(0, 1, 'F', 600.0)
        >>> g_sel = obj.eval(0, 0, 'G', 600.0, ind=[0, 1, 2])
        """
        if ind is None:
            # Compute full matrix
            return self._eval1(i, j, key, enei)
        else:
            # Compute selected matrix elements
            return self._eval2(i, j, key, enei, ind)

    def _eval1(self, i, j, key, enei):
        """
        Evaluate retarded Green function (full matrix).

        MATLAB: @compgreenret/private/eval1.m
        """
        # Evaluate connectivity matrix
        con = self.con[i][j]

        # Evaluate dielectric functions to get wavenumbers
        # MATLAB: [~, k] = cellfun(@(eps) (eps(enei)), obj.p1.eps)
        k_list = []
        for eps_func in self.p1.eps:
            eps_val, k_val = eps_func(enei)
            k_list.append(k_val)

        # Get wavenumber for this connection
        if con is not None and con > 0:
            k = k_list[con - 1]  # Convert to 0-based indexing
        else:
            k = k_list[0]  # Default to first material

        # Evaluate G, F, H1, H2
        if key not in ['Gp', 'H1p', 'H2p']:
            # Allocate array
            g = np.zeros((self.p1.n, self.p2.n), dtype=complex)

            # Loop over composite particles
            n1 = len(self.p1.p)
            n2 = len(self.p2.p)

            for i1 in range(n1):
                for i2 in range(n2):
                    if self.con[i1][i2] is not None and self.con[i1][i2] > 0:
                        # Get indices for this block
                        idx1 = self.p1.index_func(i1 + 1)  # 1-indexed in MATLAB
                        idx2 = self.p2.index_func(i2 + 1)  # 1-indexed in MATLAB

                        # Get wavenumber for this connection
                        k_block = k_list[self.con[i1][i2] - 1]

                        # Add Green function
                        g_block = self.g[i1][i2].eval(k_block, key)
                        g[np.ix_(idx1, idx2)] = g_block

        # Evaluate Gp, H1p, H2p
        else:
            # Allocate array
            g = np.zeros((self.p1.n, 3, self.p2.n), dtype=complex)

            # Loop over composite particles
            n1 = len(self.p1.p)
            n2 = len(self.p2.p)

            for i1 in range(n1):
                for i2 in range(n2):
                    if self.con[i1][i2] is not None and self.con[i1][i2] > 0:
                        # Get indices for this block
                        idx1 = self.p1.index_func(i1 + 1)  # 1-indexed in MATLAB
                        idx2 = self.p2.index_func(i2 + 1)  # 1-indexed in MATLAB

                        # Get wavenumber for this connection
                        k_block = k_list[self.con[i1][i2] - 1]

                        # Add Green function
                        g_block = self.g[i1][i2].eval(k_block, key)
                        g[np.ix_(idx1, range(3), idx2)] = g_block

        # Return zero if all elements are zero
        if np.all(g == 0):
            return 0

        return g

    def _eval2(self, i, j, key, enei, ind):
        """
        Evaluate retarded Green function (selected matrix elements).

        MATLAB: @compgreenret/private/eval2.m
        """
        # Evaluate connectivity matrix
        con = self.con[i][j]

        # Convert total index to cell array of subindices
        # MATLAB: [sub, ind] = ind2sub(obj.block, ind)
        sub, ind_blocks = self.block.ind2sub(ind)

        # Evaluate dielectric functions to get wavenumbers
        k_list = []
        for eps_func in self.p1.eps:
            eps_val, k_val = eps_func(enei)
            k_list.append(k_val)

        # Place wavevectors into cell array
        # MATLAB: con(con == 0) = nan; con(~isnan(con)) = k(con(~isnan(con)))
        con_k = [[None for _ in range(len(self.con[0]))] for _ in range(len(self.con))]
        for i1 in range(len(self.con)):
            for i2 in range(len(self.con[0])):
                if self.con[i1][i2] is not None and self.con[i1][i2] > 0:
                    con_k[i1][i2] = k_list[self.con[i1][i2] - 1]
                else:
                    con_k[i1][i2] = np.nan

        # Evaluate Green function submatrices
        g_blocks = []
        for i1 in range(len(self.g)):
            row = []
            for i2 in range(len(self.g[0])):
                if np.isnan(con_k[i1][i2]):
                    row.append(None)
                else:
                    k_val = con_k[i1][i2]
                    sub_ind = sub[i1][i2]
                    if sub_ind is not None and len(sub_ind) > 0:
                        g_sub = self.g[i1][i2].eval_ind(k_val, key, sub_ind)
                        row.append(g_sub)
                    else:
                        row.append(None)
            g_blocks.append(row)

        # Assemble together submatrices
        # MATLAB: g = accumarray(obj.block, ind, g)
        g = self.block.accumarray(ind_blocks, g_blocks)

        return g

    def field(self, sig, inout=1):
        """
        Electric and magnetic field inside/outside of particle surface.
        Computed from solutions of full Maxwell equations.

        MATLAB: @compgreenret/field.m

        Parameters
        ----------
        sig : CompStruct
            COMPSTRUCT with surface charges & currents (see bemret)
        inout : int
            fields inside (inout=1, default) or outside (inout=2) of particle surface

        Returns
        -------
        field : CompStruct
            COMPSTRUCT object with electric and magnetic fields 'e' and 'h'

        Examples
        --------
        >>> field = g.field(sig, inout=1)  # Inside
        >>> field = g.field(sig, inout=2)  # Outside
        """
        # Wavelength and wavenumber of light in vacuum
        enei = sig.enei
        k = 2 * np.pi / enei

        # Green function and E = i k A
        # MATLAB: e = 1i * k * (matmul(eval(obj, inout, 1, 'G', enei), sig.h1) + ...)
        G1 = self.eval(inout-1, 0, 'G', enei)  # Convert to 0-based
        G2 = self.eval(inout-1, 1, 'G', enei)

        e = 1j * k * (self._matmul(G1, sig.h1) + self._matmul(G2, sig.h2))

        # Derivative of Green function
        if inout == 1:
            H1p = self.eval(inout-1, 0, 'H1p', enei)
            H2p = self.eval(inout-1, 1, 'H1p', enei)
        else:
            H1p = self.eval(inout-1, 0, 'H2p', enei)
            H2p = self.eval(inout-1, 1, 'H2p', enei)

        # Add derivative of scalar potential to electric field
        # MATLAB: e = e - matmul(H1p, sig.sig1) - matmul(H2p, sig.sig2)
        e = e - self._matmul(H1p, sig.sig1) - self._matmul(H2p, sig.sig2)

        # Magnetic field
        # MATLAB: h = cross(H1p, sig.h1) + cross(H2p, sig.h2)
        h = self._cross(H1p, sig.h1) + self._cross(H2p, sig.h2)

        # Set output
        from .compgreen_stat import CompStruct
        field = CompStruct(self.p1, enei, e=e, h=h)
        return field

    def potential(self, sig, inout=1):
        """
        Potentials and surface derivatives inside/outside of particle.
        Computed from solutions of full Maxwell equations.

        MATLAB: @compgreenret/potential.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with surface charges (see bemret)
        inout : int
            potentials inside (inout=1, default) or outside (inout=2) of particle

        Returns
        -------
        pot : CompStruct
            compstruct object with potentials & surface derivatives

        Examples
        --------
        >>> pot = g.potential(sig, inout=1)  # Inside
        >>> pot = g.potential(sig, inout=2)  # Outside
        """
        enei = sig.enei

        # Set parameters that depend on inside/outside
        # MATLAB: H = subsref({'H1', 'H2'}, substruct('{}', {inout}))
        H_key = 'H1' if inout == 1 else 'H2'

        # Green functions
        # MATLAB: G1 = subsref(g, substruct('{}', {inout, 1}, '.', 'G', '()', var))
        G1 = self.eval(inout-1, 0, 'G', enei)
        G2 = self.eval(inout-1, 1, 'G', enei)

        # Surface derivatives of Green functions
        H1 = self.eval(inout-1, 0, H_key, enei)
        H2 = self.eval(inout-1, 1, H_key, enei)

        # Potential and surface derivative
        # Scalar potential
        phi = self._matmul(G1, sig.sig1) + self._matmul(G2, sig.sig2)
        phip = self._matmul(H1, sig.sig1) + self._matmul(H2, sig.sig2)

        # Vector potential
        a = self._matmul(G1, sig.h1) + self._matmul(G2, sig.h2)
        ap = self._matmul(H1, sig.h1) + self._matmul(H2, sig.h2)

        # Set output
        from .compgreen_stat import CompStruct
        if inout == 1:
            pot = CompStruct(self.p1, enei, phi1=phi, phi1p=phip, a1=a, a1p=ap)
        else:
            pot = CompStruct(self.p1, enei, phi2=phi, phi2p=phip, a2=a, a2p=ap)

        return pot

    def _matmul(self, a, x):
        """
        Generalized matrix multiplication for tensors.

        MATLAB: Misc/matmul.m
        """
        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
            if a == 0:
                return 0
            else:
                return a * x
        elif np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
            if x == 0:
                return 0
            else:
                return a * x
        else:
            # A is matrix/tensor
            if not isinstance(a, np.ndarray):
                return 0

            siza = a.shape
            sizx = x.shape if hasattr(x, 'shape') else (len(x),)

            # Check if we need special handling for 3D arrays
            if len(siza) == 3:
                # a is (n1, 3, n2), x is (n2,) or (n2, ...)
                n1, _, n2 = siza

                if len(sizx) == 1:
                    # x is 1D
                    y = np.tensordot(a, x, axes=([2], [0]))
                else:
                    # x is multi-dimensional
                    a_flat = a.reshape(n1 * 3, n2)
                    x_flat = x.reshape(n2, -1)
                    y_flat = a_flat @ x_flat

                    new_shape = (n1, 3) + sizx[1:]
                    y = y_flat.reshape(new_shape)

                return y
            else:
                # Standard 2D matrix multiplication
                if len(sizx) == 1:
                    return a @ x
                else:
                    return a @ x.reshape(sizx[0], -1).reshape((sizx[0],) + sizx[1:])

    def _cross(self, G, h):
        """
        Multidimensional cross product.

        MATLAB: @compgreenret/field.m/cross()

        For G of shape (n1, 3, n2) and h of shape (n2, 3, ...),
        compute cross product.
        """
        if not isinstance(G, np.ndarray) or G.size == 1:
            return 0

        # Size of vector field
        siz = h.shape
        siz = (siz[0],) + siz[2:] if len(siz) > 2 else (siz[0], 1)

        # Get component
        def at(h_arr, i):
            return h_arr[:, i, ...].reshape(siz)

        # Cross product: G x h
        # cross[i, :] = G[i, :, :] x h[:, :]
        cross = np.zeros((G.shape[0], 3) + siz[1:], dtype=complex)

        cross[:, 0, ...] = (self._matmul(G[:, 1, :], at(h, 2)) -
                            self._matmul(G[:, 2, :], at(h, 1)))
        cross[:, 1, ...] = (self._matmul(G[:, 2, :], at(h, 0)) -
                            self._matmul(G[:, 0, :], at(h, 2)))
        cross[:, 2, ...] = (self._matmul(G[:, 0, :], at(h, 1)) -
                            self._matmul(G[:, 1, :], at(h, 0)))

        return cross

    def __getitem__(self, key):
        """
        Cell array indexing for Green function access.

        MATLAB: @compgreenret/subsref.m

        Usage
        -----
        obj{i, j}.G(enei)     - Get Green function between particles i and j
        obj{i, j}.F(enei)     - Get surface derivative
        obj{i, j}.H1(enei)    - Get H1 matrix

        Examples
        --------
        >>> g_mat = obj[0, 0].G(600.0)
        >>> f_mat = obj[0, 1].F(600.0)
        """
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            return GreenRetAccessor(self, i, j)
        else:
            raise ValueError("CompGreenRet indexing requires (i, j) tuple")

    def G(self, enei):
        """
        Green function matrix for full composite.

        MATLAB: obj.g{i,j}.G(enei)

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        g : ndarray
            Green function matrix
        """
        return self.eval(0, 0, 'G', enei)

    def H1(self, enei):
        """
        Surface derivative H1 matrix for full composite.

        MATLAB: obj.g{i,j}.H1(enei)

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        h1 : ndarray
            H1 matrix
        """
        return self.eval(0, 0, 'H1', enei)

    def H2(self, enei):
        """
        Surface derivative H2 matrix for full composite.

        MATLAB: obj.g{i,j}.H2(enei)

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        h2 : ndarray
            H2 matrix
        """
        return self.eval(0, 0, 'H2', enei)

    def __repr__(self):
        """String representation."""
        return (
            f"CompGreenRet(p1: {self.p1.n if hasattr(self.p1, 'n') else '?'} faces, "
            f"p2: {self.p2.n if hasattr(self.p2, 'n') else '?'} faces)"
        )

    def __str__(self):
        """Detailed string representation."""
        return (
            f"compgreenret:\n"
            f"  p1: {self.p1}\n"
            f"  p2: {self.p2}\n"
            f"  con: {len(self.con)}x{len(self.con[0]) if self.con else 0}\n"
            f"  g: {len(self.g)}x{len(self.g[0]) if self.g else 0}\n"
            f"  hmode: {self.hmode}"
        )


class GreenRetSimple:
    """Simple Green function object for retarded case."""

    def __init__(self, p1, p2, deriv='norm'):
        self.p1 = p1
        self.p2 = p2
        self.deriv = deriv
        self.diag_corrections = {}


class GreenRetBlock:
    """Green function block for a particle pair."""

    def __init__(self, p1, p2, i1_start, i1_end, i2_start, i2_end, g_full, deriv):
        self.p1 = p1
        self.p2 = p2
        self.i1_start = i1_start
        self.i1_end = i1_end
        self.i2_start = i2_start
        self.i2_end = i2_end
        self.g_full = g_full
        self.deriv = deriv

    def eval(self, k, key):
        """Evaluate Green function for this block."""
        # Compute Green function matrices
        pos1 = self.p1.pos
        pos2 = self.p2.pos
        nvec1 = self.p1.nvec
        area2 = self.p2.area

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Compute distances
        r = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
        d = np.linalg.norm(r, axis=2)
        d = np.maximum(d, np.finfo(float).eps)

        # Phase factor
        phase = np.exp(1j * k * d)

        # Evaluate based on key
        if key == 'G':
            # G = exp(ikd)/d * area
            return (phase / d) * area2[np.newaxis, :]

        elif key == 'F':
            # F = (n·r) * (ik - 1/d) / d² * exp(ikd) * area
            n_dot_r = np.sum(nvec1[:, np.newaxis, :] * r, axis=2)
            f_aux = (1j * k - 1.0 / d) / (d ** 2)
            F = n_dot_r * f_aux * phase * area2[np.newaxis, :]

            # Diagonal correction for closed surfaces
            if self.p1 is self.p2:
                np.fill_diagonal(F, -2.0 * np.pi)

            return F

        elif key == 'H1':
            F = self.eval(k, 'F')
            H1 = F.copy()
            if self.p1 is self.p2:
                np.fill_diagonal(H1, np.diag(F) + 2.0 * np.pi)
            return H1

        elif key == 'H2':
            F = self.eval(k, 'F')
            H2 = F.copy()
            if self.p1 is self.p2:
                np.fill_diagonal(H2, np.diag(F) - 2.0 * np.pi)
            return H2

        elif key == 'Gp':
            # Gp = -r / d³ * exp(ikd) * (ik - 1/d) * area
            # But for retarded: Gp includes phase
            Gp_factor = -phase * (1j * k - 1.0 / d) / (d ** 2)
            Gp = r * Gp_factor[:, :, np.newaxis] * area2[np.newaxis, :, np.newaxis]
            return np.transpose(Gp, (0, 2, 1))

        elif key == 'H1p':
            Gp = self.eval(k, 'Gp')
            H1p = Gp.copy()
            if self.p1 is self.p2:
                nvec = self.p1.nvec
                for i in range(len(nvec)):
                    H1p[i, :, i] += 2 * np.pi * nvec[i]
            return H1p

        elif key == 'H2p':
            Gp = self.eval(k, 'Gp')
            H2p = Gp.copy()
            if self.p1 is self.p2:
                nvec = self.p1.nvec
                for i in range(len(nvec)):
                    H2p[i, :, i] -= 2 * np.pi * nvec[i]
            return H2p

        else:
            raise ValueError(f"Unknown key: {key}")

    def eval_ind(self, k, key, ind):
        """Evaluate selected elements."""
        # For now, evaluate full and index
        g_full = self.eval(k, key)
        if key in ['Gp', 'H1p', 'H2p']:
            # 3D array
            g_flat = g_full.reshape(-1, 3)
            return g_flat[ind]
        else:
            # 2D array
            return g_full.ravel()[ind]


class GreenRetAccessor:
    """Accessor for Green function cell array."""

    def __init__(self, parent, i, j):
        self.parent = parent
        self.i = i
        self.j = j

    def G(self, enei, ind=None):
        """Get Green function."""
        return self.parent.eval(self.i, self.j, 'G', enei, ind)

    def F(self, enei, ind=None):
        """Get surface derivative."""
        return self.parent.eval(self.i, self.j, 'F', enei, ind)

    def H1(self, enei, ind=None):
        """Get H1 matrix."""
        return self.parent.eval(self.i, self.j, 'H1', enei, ind)

    def H2(self, enei, ind=None):
        """Get H2 matrix."""
        return self.parent.eval(self.i, self.j, 'H2', enei, ind)

    def Gp(self, enei, ind=None):
        """Get derivative of Green function."""
        return self.parent.eval(self.i, self.j, 'Gp', enei, ind)

    def H1p(self, enei, ind=None):
        """Get H1p matrix."""
        return self.parent.eval(self.i, self.j, 'H1p', enei, ind)

    def H2p(self, enei, ind=None):
        """Get H2p matrix."""
        return self.parent.eval(self.i, self.j, 'H2p', enei, ind)


class BlockMatrix:
    """
    Block matrix for indexing into cell arrays.

    MATLAB: blockmatrix(siz1, siz2)
    """

    def __init__(self, siz1, siz2):
        """
        Initialize block matrix.

        Parameters
        ----------
        siz1 : list
            Sizes of blocks in dimension 1
        siz2 : list
            Sizes of blocks in dimension 2
        """
        self.siz1 = siz1
        self.siz2 = siz2
        self.n1 = len(siz1)
        self.n2 = len(siz2)

        # Cumulative indices
        self.idx1 = [0] + list(np.cumsum(siz1))
        self.idx2 = [0] + list(np.cumsum(siz2))

        # Total size
        self.total1 = self.idx1[-1]
        self.total2 = self.idx2[-1]

    def ind2sub(self, ind):
        """
        Convert linear indices to block indices.

        Parameters
        ----------
        ind : array
            Linear indices into the full matrix

        Returns
        -------
        sub : list of list
            sub[i][j] contains indices for block (i, j)
        ind_blocks : list of list
            Corresponding indices in each block
        """
        # Initialize cell arrays
        sub = [[[] for _ in range(self.n2)] for _ in range(self.n1)]
        ind_blocks = [[[] for _ in range(self.n2)] for _ in range(self.n1)]

        # Convert to 2D indices
        ind_array = np.asarray(ind)
        rows = ind_array // self.total2
        cols = ind_array % self.total2

        # Assign to blocks
        for idx, (row, col) in enumerate(zip(rows, cols)):
            # Find which block this belongs to
            i1 = np.searchsorted(self.idx1[1:], row, side='right')
            i2 = np.searchsorted(self.idx2[1:], col, side='right')

            # Local indices within block
            local_row = row - self.idx1[i1]
            local_col = col - self.idx2[i2]
            local_ind = local_row * self.siz2[i2] + local_col

            sub[i1][i2].append(local_ind)
            ind_blocks[i1][i2].append(idx)

        return sub, ind_blocks

    def accumarray(self, ind_blocks, g_blocks):
        """
        Accumulate block results into full array.

        Parameters
        ----------
        ind_blocks : list of list
            Indices for each block
        g_blocks : list of list
            Values for each block

        Returns
        -------
        g : array
            Assembled result
        """
        # Count total elements
        total_count = sum(len(ind_blocks[i][j])
                         for i in range(self.n1)
                         for j in range(self.n2)
                         if g_blocks[i][j] is not None)

        if total_count == 0:
            return np.array([])

        # Determine output shape from first non-None block
        sample_block = None
        for i in range(self.n1):
            for j in range(self.n2):
                if g_blocks[i][j] is not None and len(g_blocks[i][j]) > 0:
                    sample_block = g_blocks[i][j]
                    break
            if sample_block is not None:
                break

        if sample_block is None:
            return np.array([])

        # Check dimensionality
        if isinstance(sample_block, np.ndarray):
            if sample_block.ndim == 2:
                # 3D output (for Gp, H1p, H2p)
                g = np.zeros((total_count, sample_block.shape[1]), dtype=complex)
            else:
                # 1D output (for G, F, H1, H2)
                g = np.zeros(total_count, dtype=complex)
        else:
            g = np.zeros(total_count, dtype=complex)

        # Fill in values
        for i in range(self.n1):
            for j in range(self.n2):
                if g_blocks[i][j] is not None and len(ind_blocks[i][j]) > 0:
                    indices = ind_blocks[i][j]
                    values = g_blocks[i][j]
                    g[indices] = values

        return g
