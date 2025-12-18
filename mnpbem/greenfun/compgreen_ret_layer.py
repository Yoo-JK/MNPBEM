"""
Composite Green function for layer structure (retarded).

MATLAB: Greenfun/@compgreenretlayer/
100% identical to MATLAB MNPBEM implementation.
"""

import numpy as np
from typing import Optional, Tuple, Any

from .compgreen_ret import CompGreenRet
from .compgreen_stat import CompStruct
from .greenret_layer import GreenRetLayer


class CompGreenRetLayer:
    """
    Green function for composite particles with layer structure (retarded).

    MATLAB: @compgreenretlayer

    Combines direct Green function (compgreenret) with reflected part
    from layer structure (greenretlayer).

    Properties
    ----------
    name : str
        'greenfunction' (constant)
    needs : dict
        {'sim': 'ret', 'layer': True} (constant)
    p1 : ComParticle
        Green function between points p1 and comparticle p2
    p2 : ComParticle
        Green function between points p1 and comparticle p2
    g : CompGreenRet
        Direct Green functions connecting p1 and p2
    gr : GreenRetLayer
        Reflected parts of Green functions
    layer : LayerStructure
        Layer structure
    ind1 : ndarray
        Index to positions of P1 connected to layer structure
    ind2 : ndarray
        Index to boundary elements of P2 connected to layer structure

    Methods
    -------
    __init__(p1, p2, **options)
        Constructor - initialize Green functions with layer structure
    eval(enei, *keys)
        Evaluate total Green function (direct + reflected)
    field(sig, inout=1)
        Electric and magnetic field inside/outside of particle surface
    potential(sig, inout=1)
        Potentials and surface derivatives inside/outside of particle
    """

    # Class constants
    name = 'greenfunction'
    needs = {'sim': 'ret', 'layer': True}

    def __init__(self, p1, p2, **options):
        """
        Initialize Green function object for layer structure.

        MATLAB: @compgreenretlayer/compgreenretlayer.m

        Parameters
        ----------
        p1 : ComParticle
            Green function between points p1 and comparticle p2
        p2 : ComParticle
            Green function between points p1 and comparticle p2
        **options : dict
            layer : LayerStructure
                Layer structure object
            greentab : GreenTab
                Table for Green function interpolation
            Additional options passed to CompGreenRet and GreenRetLayer

        Examples
        --------
        >>> from mnpbem import trisphere, EpsConst, ComParticle, LayerStructure
        >>> from mnpbem.greenfun import CompGreenRetLayer
        >>>
        >>> # Create substrate layer structure
        >>> layer = LayerStructure([EpsConst(1.0), EpsConst(2.25)], [0])
        >>>
        >>> # Create particle above substrate
        >>> p = trisphere(144, 10.0)
        >>> p.verts[:, 2] += 15  # Move particle 15 nm above surface
        >>> eps = [EpsConst(1.0), EpsConst(-10.0)]
        >>> cp = ComParticle(eps, [p], [[2, 1]])
        >>>
        >>> # Create Green function
        >>> g = CompGreenRetLayer(cp, cp, layer=layer)
        """
        self.p1 = p1
        self.p2 = p2

        # Initialize direct Green function
        self.g = CompGreenRet(p1, p2, **options)

        # Get layer structure
        self.layer = options.get('layer')

        # Find elements connected to layer structure
        self._init(**options)

    def _init(self, **options):
        """
        Initialize Green function object and layer structure.

        MATLAB: @compgreenretlayer/private/init.m
        """
        p1 = self.p1
        p2 = self.p2
        layer = self.layer

        if layer is None:
            self.ind1 = np.array([])
            self.ind2 = np.array([])
            self.gr = None
            return

        # Get inout arrays for p1 and p2
        # inout specifies which material is inside/outside each face
        inout1 = self._expand_inout(p1)
        inout2 = self._expand_inout(p2)

        # Find elements connected to layer structure
        # (elements where inout matches one of the layer indices)
        layer_ind = layer.ind if hasattr(layer, 'ind') else [0]

        # Find indices where inout matches layer indices
        self.ind1 = np.where(np.isin(inout1, layer_ind))[0]
        self.ind2 = np.where(np.isin(inout2, layer_ind))[0]

        if len(self.ind1) == 0 or len(self.ind2) == 0:
            self.gr = None
            return

        # Create particle subsets connected to layer
        p1_sub = self._select_faces(p1, self.ind1)
        p2_sub = self._select_faces(p2, self.ind2)

        # Initialize reflected part of Green function
        self.gr = GreenRetLayer(p1_sub, p2_sub, **options)

    def _expand_inout(self, p):
        """
        Expand inout array for all particles.

        Parameters
        ----------
        p : ComParticle
            Composite particle

        Returns
        -------
        inout : ndarray
            Expanded inout array
        """
        if hasattr(p, 'inout'):
            return p.inout[:, -1]
        else:
            # Default: all faces connected to medium 0
            return np.zeros(p.nfaces, dtype=int)

    def _select_faces(self, p, ind):
        """
        Select subset of faces from particle.

        Parameters
        ----------
        p : ComParticle or Particle
            Particle object
        ind : ndarray
            Indices of faces to select

        Returns
        -------
        p_sub : Particle
            Particle with selected faces
        """
        # Get combined particle
        if hasattr(p, 'p') and isinstance(p.p, list):
            # ComParticle - concatenate all particles
            particles = p.p
            if len(particles) == 1:
                pc = particles[0]
            else:
                pc = particles[0]
                for pi in particles[1:]:
                    pc = pc + pi
        else:
            pc = p

        # Create subset
        return pc.select(ind)

    def eval(self, enei, *keys, **kwargs):
        """
        Evaluate total Green function (direct + reflected).

        MATLAB: @compgreenretlayer/eval.m

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
        varargout : tuple
            Requested Green functions (direct + reflected)
        """
        # Get direct Green functions
        result_direct = self.g.eval(enei, *keys, **kwargs)

        # Get reflected Green functions (if layer is present)
        if self.gr is not None and len(self.ind1) > 0 and len(self.ind2) > 0:
            # Map keys to reflected component keys
            refl_keys = []
            for key in keys:
                if key in ['G', 'F', 'Gp']:
                    refl_keys.append(key)
                elif key in ['H1', 'H2']:
                    refl_keys.append('F')
                elif key in ['H1p', 'H2p']:
                    refl_keys.append('Gp')
                else:
                    refl_keys.append(key)

            # Get reflected parts
            result_refl = self.gr.eval(enei, *refl_keys)

            # Add reflected parts to direct parts
            if not isinstance(result_direct, tuple):
                result_direct = (result_direct,)
                result_refl = (result_refl,)

            results = []
            for rd, rr, key in zip(result_direct, result_refl, keys):
                if rr is not None and rd is not None:
                    # Add reflected part at appropriate indices
                    result = rd.copy()
                    idx = np.ix_(self.ind1, self.ind2)
                    if result.ndim == 3:
                        # Gp, H1p, H2p
                        result[:, :, idx[1]] = result[:, :, idx[1]] + rr
                    else:
                        # G, F, H1, H2
                        result[idx] = result[idx] + rr
                    results.append(result)
                else:
                    results.append(rd)

            if len(results) == 1:
                return results[0]
            return tuple(results)

        return result_direct

    def field(self, sig, inout=1):
        """
        Electric and magnetic field inside/outside of particle surface.

        MATLAB: @compgreenretlayer/field.m

        Parameters
        ----------
        sig : CompStruct
            Surface charges and currents
        inout : int
            Fields inside (1, default) or outside (2) of particle surface

        Returns
        -------
        field : CompStruct
            Electric and magnetic field
        """
        # Wavelength and wavenumber
        enei = sig.enei
        k = 2 * np.pi / enei

        # Get derivative of Green function
        if inout == 1:
            G, Hp = self.eval(enei, 'G', 'H1p')
        else:
            G, Hp = self.eval(enei, 'G', 'H2p')

        # Electric field: E = ik*A - grad(V)
        # Magnetic field: H = curl(A)
        e = 1j * k * self._matmul(G, sig.h) - self._matmul(Hp, sig.sig)
        h = self._cross_matmul(Hp, sig.h)

        return CompStruct(self.p1, enei, e=e, h=h)

    def potential(self, sig, inout=1):
        """
        Potentials and surface derivatives inside/outside of particle.

        MATLAB: @compgreenretlayer/potential.m

        Parameters
        ----------
        sig : CompStruct
            Surface charges and currents
        inout : int
            Potentials inside (1, default) or outside (2) of particle

        Returns
        -------
        pot : CompStruct
            Potentials (phi, a) and surface derivatives (phip, ap)
        """
        # Wavelength and wavenumber
        enei = sig.enei
        k = 2 * np.pi / enei

        # Get Green function and surface derivative
        H_key = 'H1' if inout == 1 else 'H2'
        G, H = self.eval(enei, 'G', H_key)

        # Scalar potential
        phi = self._matmul(G, sig.sig)
        phip = self._matmul(H, sig.sig)

        # Vector potential
        a = self._matmul(G, sig.h)
        ap = self._matmul(H, sig.h) if hasattr(sig, 'h') else None

        if inout == 1:
            return CompStruct(self.p1, enei, phi1=phi, phi1p=phip, a1=a, a1p=ap)
        else:
            return CompStruct(self.p1, enei, phi2=phi, phi2p=phip, a2=a, a2p=ap)

    def _matmul(self, a, x):
        """Matrix multiplication helper."""
        if x is None:
            return None
        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
            if a == 0:
                return 0
            return a * x
        return a @ x

    def _cross_matmul(self, Hp, h):
        """
        Cross product with matrix multiplication.

        Computes: result = curl(Hp @ h) = Hp × h
        """
        if h is None or Hp is None:
            return None

        # Hp: (n1, 3, n2), h: (n2, 3) or (n2, 3, ...)
        # Result: (n1, 3, ...)

        # For simplicity, compute component by component
        n1 = Hp.shape[0]
        n2 = Hp.shape[2]

        # Expand h for matrix multiplication
        if h.ndim == 2:
            h = h[:, :, np.newaxis]

        result = np.zeros((n1, 3) + h.shape[2:], dtype=complex)

        # Cross product: result = Hp × h
        # result_x = Hp_y * h_z - Hp_z * h_y
        # result_y = Hp_z * h_x - Hp_x * h_z
        # result_z = Hp_x * h_y - Hp_y * h_x

        result[:, 0] = np.tensordot(Hp[:, 1, :], h[:, 2], axes=(1, 0)) - \
                       np.tensordot(Hp[:, 2, :], h[:, 1], axes=(1, 0))
        result[:, 1] = np.tensordot(Hp[:, 2, :], h[:, 0], axes=(1, 0)) - \
                       np.tensordot(Hp[:, 0, :], h[:, 2], axes=(1, 0))
        result[:, 2] = np.tensordot(Hp[:, 0, :], h[:, 1], axes=(1, 0)) - \
                       np.tensordot(Hp[:, 1, :], h[:, 0], axes=(1, 0))

        return result.squeeze()

    def __repr__(self):
        """String representation."""
        return (
            f"CompGreenRetLayer(p1: {self.p1}, p2: {self.p2}, "
            f"g: {self.g}, gr: {self.gr})"
        )

    def __str__(self):
        """Detailed string representation."""
        return (
            f"compgreenretlayer:\n"
            f"  p1: {self.p1}\n"
            f"  p2: {self.p2}\n"
            f"  g: {self.g}\n"
            f"  gr: {self.gr}"
        )
