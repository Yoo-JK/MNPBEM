"""
LayerStructure class - Dielectric layer structures.

MATLAB: Particles/@layerstructure

The outer surface normals of the layers must point upwards.
We assume a geometry of the following form:

          eps{ 1 }
--------------------------  z( 1 )
          eps{ 2 }
--------------------------  z( 2 )
           ...
--------------------------  z( end )
          eps{ end }
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Union, Any, Callable
from scipy.integrate import solve_ivp
from scipy.special import jv, hankel1
from dataclasses import dataclass, field


@dataclass
class LayerOptions:
    """Options for layer structure.

    MATLAB: layerstructure.options
    """
    ztol: float = 2e-2      # tolerance for detecting points in layer
    rmin: float = 1e-2      # minimum radial distance for Green function
    zmin: float = 1e-2      # minimum distance to layers for Green function
    semi: float = 0.1       # imaginary part of semiellipse for complex integration
    ratio: float = 2.0      # z : r ratio which determines integration path
    # ODE integration options
    atol: float = 1e-6
    initial_step: float = 1e-3


@dataclass
class PositionStruct:
    """Position structure for Green function calculations."""
    r: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    ind1: np.ndarray = None
    ind2: np.ndarray = None
    zmin: np.ndarray = None


class LayerStructure:
    """
    Dielectric layer structures.

    MATLAB: @layerstructure

    Properties
    ----------
    eps : list
        List of dielectric functions
    z : ndarray
        z-positions of layers
    ind : ndarray
        Index to table of dielectric functions
    ztol : float
        Tolerance for detecting points in layer
    rmin : float
        Minimum radial distance for Green function
    zmin : float
        Minimum distance to layer for Green function
    semi : float
        Imaginary part of semiellipse for complex integration
    ratio : float
        z : r ratio which determines integration path

    Examples
    --------
    >>> from mnpbem.material import Drude
    >>> eps1 = lambda enei: (1.0, 2*np.pi/enei)  # Air
    >>> eps2 = lambda enei: (2.25, 2*np.pi/enei * 1.5)  # Glass
    >>> layer = LayerStructure([eps1, eps2], [1, 2], [0.0])
    """

    def __init__(self, epstab: List, ind: Union[List, np.ndarray],
                 z: Union[List, np.ndarray], **kwargs):
        """
        Initialize layer structure.

        MATLAB: layerstructure.m

        Parameters
        ----------
        epstab : list
            Cell array (list) of dielectric functions
        ind : array-like
            Index to dielectric functions of layer structure (1-indexed in MATLAB, 0-indexed here)
        z : array-like
            z-position(s) of layers
        **kwargs : dict
            Additional options:
            - ztol : tolerance for detecting points in layer
            - rmin : minimum radial distance for Green function
            - zmin : minimum distance to layers for Green function
            - semi : imaginary part of semiellipse for complex integration
            - ratio : z : r ratio which determines integration path
        """
        ind = np.atleast_1d(ind)
        z = np.atleast_1d(z).astype(float)

        # Store dielectric functions for each layer
        # In MATLAB: obj.eps = epstab(ind), we need to handle 0/1-indexed
        self.eps = [epstab[i] for i in ind]
        self.ind = ind
        self.z = z

        # Default options
        self.ztol = kwargs.get('ztol', 2e-2)
        self.rmin = kwargs.get('rmin', 1e-2)
        self.zmin = kwargs.get('zmin', 1e-2)
        self.semi = kwargs.get('semi', 0.1)
        self.ratio = kwargs.get('ratio', 2.0)
        self.atol = kwargs.get('atol', 1e-6)
        self.initial_step = kwargs.get('initial_step', 1e-3)

    @property
    def n(self) -> int:
        """Number of layers."""
        return len(self.z)

    def indlayer(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find index of layer within which z is embedded.

        MATLAB: indlayer.m

        Parameters
        ----------
        z : ndarray
            z-values of points

        Returns
        -------
        ind : ndarray
            Index to layer within which z is embedded (0-indexed)
        in_layer : ndarray
            Boolean array: is point located in layer?
        """
        z = np.atleast_1d(z)

        # Layers are ordered with decreasing z-values
        # Use histogram to find bin indices
        bins = np.concatenate([[-np.inf], -self.z, [np.inf]])
        ind = np.digitize(-z, bins) - 1

        # Is point located in layer?
        zmin_dist, _ = self.mindist(z)
        in_layer = np.abs(zmin_dist) < self.ztol

        return ind, in_layer

    def mindist(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find minimal distance of z-values to layers.

        MATLAB: mindist.m

        Parameters
        ----------
        z : ndarray
            z-values of points

        Returns
        -------
        zmin : ndarray
            Minimal distance to layer
        ind : ndarray
            Index to nearest layer (0-indexed)
        """
        z = np.atleast_1d(z)
        original_shape = z.shape
        z_flat = z.flatten()

        # Difference to each layer
        diff = z_flat[:, np.newaxis] - self.z[np.newaxis, :]

        # Find minimal distance
        abs_diff = np.abs(diff)
        ind = np.argmin(abs_diff, axis=1)
        zmin = abs_diff[np.arange(len(z_flat)), ind]

        # Reshape to original shape
        zmin = zmin.reshape(original_shape)
        ind = ind.reshape(original_shape)

        return zmin, ind

    def round_z(self, *z_arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Round z-values to achieve minimal distance to layers.

        MATLAB: round.m

        Parameters
        ----------
        *z_arrays : ndarray
            z-values of points

        Returns
        -------
        tuple of ndarray
            z-values with a minimal distance to layers
        """
        results = []

        for z in z_arrays:
            z = np.atleast_1d(z).copy()

            # Minimal distance to layer
            zmin_dist, ind = self.mindist(z)

            # z-value of nearest layer
            ztab = self.z[ind]

            # Shift direction
            direction = np.sign(z - ztab)

            # Shift points that are too close to layer
            mask = zmin_dist <= self.zmin
            z[mask] = ztab[mask] + direction[mask] * self.zmin

            results.append(z)

        return tuple(results) if len(results) > 1 else results[0]

    def fresnel(self, enei: float, kpar: float, pos: PositionStruct) -> Dict[str, np.ndarray]:
        """
        Fresnel reflection and transmission coefficients for potentials.

        MATLAB: fresnel.m

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum
        kpar : float
            Parallel wavevector
        pos : PositionStruct
            Position structure

        Returns
        -------
        r : dict
            Structure with reflection and transmission coefficients
        """
        # Wavenumber in media
        k = np.array([self._get_wavenumber(eps, enei) for eps in self.eps])

        # Perpendicular component of wavevector
        kz = np.sqrt(k**2 - kpar**2 + 0j) + 1e-10j
        kz = kz * np.sign(np.imag(kz))

        # Perpendicular components of wavevector
        k1z = kz[pos.ind1]
        k2z = kz[pos.ind2]

        # Ratio of z-component of wavevector
        if np.array(pos.z1).shape == np.array(pos.z2).shape:
            ratio = k2z / k1z
        else:
            ratio = np.outer(1.0 / k1z, k2z)

        # Reflection and transmission coefficients
        r, _ = self.reflection(enei, kpar, pos)

        # Correct reflection coefficients
        for key in r:
            r[key] = r[key] * ratio

        return r

    def efresnel(self, pol: np.ndarray, direction: np.ndarray,
                 enei: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Reflected and transmitted electric fields.

        MATLAB: efresnel.m

        Parameters
        ----------
        pol : ndarray, shape (n, 3)
            Polarization vector
        direction : ndarray, shape (n, 3)
            Light propagation direction
        enei : float
            Wavelength of light in vacuum

        Returns
        -------
        e : dict
            Structure with reflected and transmitted electric fields
        k : dict
            Wavevectors for reflection and transmission
        """
        pol = np.atleast_2d(pol)
        direction = np.atleast_2d(direction)

        # Wavenumber of light in vacuum
        k0 = 2 * np.pi / enei

        # Wavenumber in media
        k = np.array([self._get_wavenumber(eps, enei) for eps in self.eps])

        # Upper and lower layers
        z1, ind1 = self.z[0] + 1e-10, 0
        z2, ind2 = self.z[-1] - 1e-10, len(self.z)

        # Allocate output arrays
        ei = np.zeros_like(pol, dtype=complex)
        er = np.zeros_like(pol, dtype=complex)
        et = np.zeros_like(pol, dtype=complex)
        ki = np.zeros_like(pol, dtype=complex)
        kr = np.zeros_like(pol, dtype=complex)
        kt = np.zeros_like(pol, dtype=complex)

        # Loop over propagation directions
        for i in range(pol.shape[0]):
            # Position structures for reflection and transmission
            if direction[i, 2] < 0:
                # Excitation through upper medium
                posr = PositionStruct(r=np.array([0.0]), z1=np.array([z1]),
                                      z2=np.array([z1]), ind1=np.array([ind1]),
                                      ind2=np.array([ind1]))
                post = PositionStruct(r=np.array([0.0]), z1=np.array([z2]),
                                      z2=np.array([z1]), ind1=np.array([ind2]),
                                      ind2=np.array([ind1]))
            else:
                # Excitation through lower medium
                posr = PositionStruct(r=np.array([0.0]), z1=np.array([z2]),
                                      z2=np.array([z2]), ind1=np.array([ind2]),
                                      ind2=np.array([ind2]))
                post = PositionStruct(r=np.array([0.0]), z1=np.array([z1]),
                                      z2=np.array([z2]), ind1=np.array([ind1]),
                                      ind2=np.array([ind2]))

            # Parallel component of wavevector
            kpar = k[post.ind2[0]] * direction[i, :2]
            kpar_norm = np.linalg.norm(kpar)

            # Perpendicular components of reflected and transmitted waves
            kzr = np.sqrt(k[posr.ind1[0]]**2 - np.dot(kpar, kpar) + 0j)
            kzr = kzr * np.sign(np.imag(kzr + 1e-10j))
            kzt = np.sqrt(k[post.ind1[0]]**2 - np.dot(kpar, kpar) + 0j)
            kzt = kzt * np.sign(np.imag(kzt + 1e-10j))

            # Wavevectors for incident, reflected, and transmitted waves
            ki[i, :] = np.array([kpar[0], kpar[1], np.sign(direction[i, 2]) * kzr])
            kr[i, :] = np.array([kpar[0], kpar[1], -np.sign(direction[i, 2]) * kzr])
            kt[i, :] = np.array([kpar[0], kpar[1], np.sign(direction[i, 2]) * kzt])

            # Reflection and transmission coefficients
            r = self.fresnel(enei, kpar_norm, posr)
            t = self.fresnel(enei, kpar_norm, post)

            # Incoming electric field
            ei[i, :] = pol[i, :]

            # Reflected and transmitted electric field
            er[i, :2] = r['p'] * pol[i, :2]
            er[i, 2] = r['hh'] * pol[i, 2] - kr[i, 2] / k0 * r['sh'] * pol[i, 2]

            et[i, :2] = t['p'] * pol[i, :2]
            et[i, 2] = t['hh'] * pol[i, 2] - kt[i, 2] / k0 * t['sh'] * pol[i, 2]

            # Add phase factors
            er[i, :] *= np.exp(-1j * kr[i, 2] * posr.z2[0] - 1j * kr[i, 2] * posr.z1[0])
            et[i, :] *= np.exp(-1j * kr[i, 2] * post.z2[0] - 1j * kt[i, 2] * post.z1[0])

        e = {'i': ei, 'r': er, 't': et}
        k_out = {'i': ki, 'r': kr, 't': kt}

        return e, k_out

    def reflection(self, enei: float, kpar: float,
                   pos: PositionStruct) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Reflection coefficients for surface charges and currents.

        MATLAB: reflection.m

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum
        kpar : float
            Parallel wavevector
        pos : PositionStruct
            Position structure

        Returns
        -------
        r : dict
            Structure with reflection and transmission coefficients
        rz : dict
            Derivative of R wrt z-value
        """
        # Use simpler equations for substrate in case of single interface
        if len(self.z) == 1:
            return self._reflectionsubs(enei, kpar, pos)

        # Wavenumbers in media
        k = np.array([self._get_wavenumber(eps, enei) for eps in self.eps])

        # Perpendicular component of wavevector
        kz = np.sqrt(k**2 - kpar**2 + 0j) + 1e-10j
        kz = kz * np.sign(np.imag(kz))

        # Index to layers
        ind1 = pos.ind1
        ind2 = pos.ind2

        # Perpendicular components of wavevectors in different media
        k1z = kz[ind1]
        k2z = kz[ind2]

        # Extended z-values for boundary conditions
        z_ext_lo = np.concatenate([self.z, [-1e100]])
        z_ext_up = np.concatenate([[1e100], self.z])

        # Distance to lower interfaces
        z1_flat = np.atleast_1d(pos.z1).flatten()
        z2_flat = np.atleast_1d(pos.z2).flatten()

        dn1 = np.abs(z1_flat - z_ext_lo[ind1])
        dn2 = np.abs(z2_flat - z_ext_lo[ind2])

        # Distance to upper interfaces
        up1 = np.abs(z1_flat - z_ext_up[ind1])
        up2 = np.abs(z2_flat - z_ext_up[ind2])

        # Size of excitation matrix
        n_layers = len(self.z)
        n_ind2 = len(np.atleast_1d(ind2))

        # Prefactor for Green function
        fac = 2j * np.pi / k2z

        # Excitation matrix
        exc = np.zeros((2 * n_layers + 2, n_ind2), dtype=complex)
        for j in range(n_ind2):
            idx2 = ind2[j] if np.isscalar(ind2) is False else ind2
            exc[2 * idx2, j] = fac[j] * np.exp(1j * k2z[j] * dn2[j])
            exc[2 * idx2 - 1, j] = fac[j] * np.exp(1j * k2z[j] * up2[j])

        # Remove layers at infinity
        exc = exc[1:-1, :]

        # Matrices for solution of BEM equations
        par, perp = self._bemsolve(enei, kpar)

        # Initialize output dictionaries
        r = {}
        rz = {}

        # Parallel surface current
        y = par @ exc

        # Surface currents above and below interface
        h1 = np.vstack([np.zeros((1, n_ind2)), y[1::2, :]])
        h2 = np.vstack([y[0::2, :], np.zeros((1, n_ind2))])

        # Reflection and transmission coefficients
        r['p'] = self._multiply(pos, h2, np.exp(1j * k1z * dn1)) + \
                 self._multiply(pos, h1, np.exp(1j * k1z * up1))
        rz['p'] = self._multiply(pos, h2, np.exp(1j * k1z * dn1)) - \
                  self._multiply(pos, h1, np.exp(1j * k1z * up1))

        # Surface charge - extend excitation matrix
        exc2 = np.zeros((2 * exc.shape[0], exc.shape[1]), dtype=complex)
        exc2[0::2, :] = exc

        # Solve BEM equations
        y = perp @ exc2

        # Surface charge above and below interface
        sig1 = np.vstack([np.zeros((1, n_ind2)), y[2::4, :]])
        sig2 = np.vstack([y[0::4, :], np.zeros((1, n_ind2))])

        r['ss'] = self._multiply(pos, sig2, np.exp(1j * k1z * dn1)) + \
                  self._multiply(pos, sig1, np.exp(1j * k1z * up1))
        rz['ss'] = self._multiply(pos, sig2, np.exp(1j * k1z * dn1)) - \
                   self._multiply(pos, sig1, np.exp(1j * k1z * up1))

        # Surface currents
        h1 = np.vstack([np.zeros((1, n_ind2)), y[3::4, :]])
        h2 = np.vstack([y[1::4, :], np.zeros((1, n_ind2))])

        r['hs'] = self._multiply(pos, h2, np.exp(1j * k1z * dn1)) + \
                  self._multiply(pos, h1, np.exp(1j * k1z * up1))
        rz['hs'] = self._multiply(pos, h2, np.exp(1j * k1z * dn1)) - \
                   self._multiply(pos, h1, np.exp(1j * k1z * up1))

        # Perpendicular surface current - extend excitation matrix
        exc2 = np.zeros((2 * exc.shape[0], exc.shape[1]), dtype=complex)
        exc2[1::2, :] = exc

        # Solve BEM equations
        y = perp @ exc2

        # Surface charge
        sig1 = np.vstack([np.zeros((1, n_ind2)), y[2::4, :]])
        sig2 = np.vstack([y[0::4, :], np.zeros((1, n_ind2))])

        r['sh'] = self._multiply(pos, sig2, np.exp(1j * k1z * dn1)) + \
                  self._multiply(pos, sig1, np.exp(1j * k1z * up1))
        rz['sh'] = self._multiply(pos, sig2, np.exp(1j * k1z * dn1)) - \
                   self._multiply(pos, sig1, np.exp(1j * k1z * up1))

        # Surface currents
        h1 = np.vstack([np.zeros((1, n_ind2)), y[3::4, :]])
        h2 = np.vstack([y[1::4, :], np.zeros((1, n_ind2))])

        r['hh'] = self._multiply(pos, h2, np.exp(1j * k1z * dn1)) + \
                  self._multiply(pos, h1, np.exp(1j * k1z * up1))
        rz['hh'] = self._multiply(pos, h2, np.exp(1j * k1z * dn1)) - \
                   self._multiply(pos, h1, np.exp(1j * k1z * up1))

        return r, rz

    def _reflectionsubs(self, enei: float, kpar: float,
                        pos: PositionStruct) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Reflection coefficients for substrate (single interface).

        MATLAB: reflectionsubs.m
        """
        # Dielectric functions and wavenumbers in media
        eps_vals = np.array([self._get_eps(eps, enei) for eps in self.eps])
        k = np.array([self._get_wavenumber(eps, enei) for eps in self.eps])

        # z-component of wavevector
        kz = np.sqrt(k**2 - kpar**2 + 0j)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        # eps1 (eps2) label the medium above (below) the interface
        eps1, k1z = eps_vals[0], kz[0]
        eps2, k2z = eps_vals[1], kz[1]

        # Parallel surface current
        rr = (k1z - k2z) / (k2z + k1z)
        p_mat = np.array([[rr, 1 + rr], [1 - rr, -rr]])

        # Wavenumber of light in vacuum
        k0 = 2 * np.pi / enei

        # Auxiliary quantity
        Delta = (k2z + k1z) * (eps1 * k2z + eps2 * k1z)

        # Helper functions for matrices
        def mat1(r1, r2):
            return np.array([[r1, k1z/k2z * (r2 + 1)],
                           [k2z/k1z * (r1 + 1), r2]])

        def mat2(r1, r2):
            return np.array([[r1, -k1z/k2z * r2],
                           [k2z/k1z * r1, -r2]])

        # Induced surface charge, from surface charge source
        ss_mat = mat1((k1z + k2z) * (2*eps1*k1z - eps2*k1z - eps1*k2z) / Delta,
                      (k2z + k1z) * (2*eps2*k2z - eps1*k2z - eps2*k1z) / Delta)

        # Induced surface current, from surface charge source
        hs_mat = mat2(-2*k0*(eps2 - eps1)*eps1*k1z / Delta,
                      -2*k0*(eps1 - eps2)*eps2*k2z / Delta)

        # Induced surface charge, from surface current source
        sh_mat = mat2(-2*k0*(eps2 - eps1)*k1z / Delta,
                      -2*k0*(eps1 - eps2)*k2z / Delta)

        # Induced surface current, from surface charge source
        hh_mat = mat1((k1z - k2z) * (2*eps1*k1z - eps2*k1z + eps1*k2z) / Delta,
                      (k2z - k1z) * (2*eps2*k2z - eps1*k2z + eps2*k1z) / Delta)

        r_mats = {'p': p_mat, 'ss': ss_mat, 'hs': hs_mat, 'sh': sh_mat, 'hh': hh_mat}

        # Green functions
        ind1 = np.atleast_1d(pos.ind1)
        ind2 = np.atleast_1d(pos.ind2)
        z1 = np.atleast_1d(pos.z1)
        z2 = np.atleast_1d(pos.z2)

        g1 = np.exp(1j * kz[ind1] * np.abs(z1 - self.z[0]))
        g2 = np.exp(1j * kz[ind2] * np.abs(z2 - self.z[0]))

        # Derivative of Green function wrt z-value
        g1z = g1 * np.sign(z1 - self.z[0])

        r = {}
        rz = {}

        for name, rr in r_mats.items():
            if ind1.shape == ind2.shape:
                r[name] = g1 * rr[ind1, ind2] * g2
                rz[name] = g1z * rr[ind1, ind2] * g2
            else:
                r[name] = rr[ind1, :][:, ind2] * np.outer(g1, g2)
                rz[name] = rr[ind1, :][:, ind2] * np.outer(g1z, g2)

        return r, rz

    def green(self, enei: float, r: np.ndarray, z1: np.ndarray,
              z2: np.ndarray) -> Tuple[Dict, Dict, Dict, PositionStruct]:
        """
        Compute reflected potential and derivatives for layer structure.

        MATLAB: green.m

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum
        r : ndarray
            Radial distance between points
        z1 : ndarray
            z-values for position where potential is computed
        z2 : ndarray
            z-values for position of exciting charge or current

        Returns
        -------
        G : dict
            Reflected Green function
        Fr : dict
            Derivative of Green function in radial direction
        Fz : dict
            Derivative of Green function in z-direction
        pos : PositionStruct
            Expanded arrays for radii and heights
        """
        # Round radii and z-values
        r = np.maximum(r, self.rmin)
        z1, z2 = self.round_z(z1, z2)

        # Save positions in structure
        pos = PositionStruct(r=r, z1=z1, z2=z2)

        # Indices to layers
        pos.ind1, _ = self.indlayer(z1)
        pos.ind2, _ = self.indlayer(z2)

        # Positions using multiplication function
        r_exp = self._mul(r, self._mul(np.ones_like(z1), np.ones_like(z2)))
        z1_exp = self._mul(np.ones_like(r), self._mul(z1, np.ones_like(z2)))
        z2_exp = self._mul(np.ones_like(r), self._mul(np.ones_like(z1), z2))

        # Minimal distance to layers
        zmin = (self.mindist(z1_exp.flatten())[0] +
                self.mindist(z2_exp.flatten())[0]).reshape(r_exp.shape)

        # Size of integrand
        n1 = r_exp.size

        # Initial vector
        y1 = np.zeros(15 * n1)

        # Solve differential equation (semi-ellipse in complex kr-plane)
        t_span = [0, np.pi]
        t_eval = np.array([0, 1e-3, np.pi])

        sol = solve_ivp(
            lambda t, y: self._green_fun(enei, t, pos, 1, r_exp, z1_exp, z2_exp),
            t_span, y1, t_eval=t_eval, atol=self.atol
        )
        y1 = sol.y[:, -1]

        # Determine integration path in complex plane
        zmin_flat = zmin.flatten()
        r_flat = r_exp.flatten()

        ind2 = np.where(zmin_flat >= r_flat / self.ratio)[0]
        ind3 = np.where(zmin_flat < r_flat / self.ratio)[0]

        # Integration along real axis
        n2 = len(ind2)
        y2 = np.zeros(15 * n2) if n2 > 0 else np.array([])

        if n2 > 0:
            sol = solve_ivp(
                lambda t, y: self._green_fun(enei, t, pos, 2, r_exp, z1_exp, z2_exp, ind2),
                [1, 1e-10], y2, t_eval=[1, 1e-3, 1e-10], atol=self.atol
            )
            y2 = sol.y[:, -1]

        # Integration along imaginary axis
        n3 = len(ind3)
        y3 = np.zeros(15 * n3) if n3 > 0 else np.array([])

        if n3 > 0:
            sol = solve_ivp(
                lambda t, y: self._green_fun(enei, t, pos, 3, r_exp, z1_exp, z2_exp, ind3),
                [1, 1e-10], y3, t_eval=[1, 1e-3, 1e-10], atol=self.atol
            )
            y3 = sol.y[:, -1]

        # Get field names from reflection coefficients
        names = list(self.reflection(enei, 0, pos)[0].keys())

        # Index functions
        def num1(iname, i):
            return slice((iname * 3 + i) * n1, (iname * 3 + i + 1) * n1)

        def num2(iname, i):
            return slice((iname * 3 + i) * n2, (iname * 3 + i + 1) * n2)

        def num3(iname, i):
            return slice((iname * 3 + i) * n3, (iname * 3 + i + 1) * n3)

        # Save expanded arrays
        pos_out = PositionStruct(r=r_exp, z1=z1_exp, z2=z2_exp, zmin=zmin)

        G = {}
        Fr = {}
        Fz = {}

        # Loop over field names
        for iname, name in enumerate(names):
            # Green functions from semi-ellipse
            g = y1[num1(iname, 0)]
            fr = y1[num1(iname, 1)]
            fz = y1[num1(iname, 2)]

            # Green functions from integration along real axis
            if n2 > 0:
                g[ind2] += y2[num2(iname, 0)]
                fr[ind2] += y2[num2(iname, 1)]
                fz[ind2] += y2[num2(iname, 2)]

            # Green functions from integration along imaginary axis
            if n3 > 0:
                g[ind3] += y3[num3(iname, 0)]
                fr[ind3] += y3[num3(iname, 1)]
                fz[ind3] += y3[num3(iname, 2)]

            # Reshape and store
            G[name] = np.squeeze(g.reshape(r_exp.shape))
            Fr[name] = np.squeeze(fr.reshape(r_exp.shape))
            Fz[name] = np.squeeze(fz.reshape(r_exp.shape))

        return G, Fr, Fz, pos_out

    def _green_fun(self, enei: float, x: float, pos: PositionStruct, key: int,
                   r: np.ndarray, z1: np.ndarray, z2: np.ndarray,
                   ind: np.ndarray = None) -> np.ndarray:
        """
        Integration path function.

        MATLAB: green.m - fun()
        """
        # Wavenumber of light in vacuum
        k0 = 2 * np.pi / enei

        # Wavenumbers in media
        k = np.array([self._get_wavenumber(eps, enei) for eps in self.eps])

        # Large half-axis
        k1max = np.max(np.real(k)) + k0

        if key == 1:
            # Semi-ellipse
            kr = k1max * (1 - np.cos(x) - 1j * self.semi * np.sin(x))
            deriv = k1max * (np.sin(x) - 1j * self.semi * np.cos(x))
            return deriv * self._intbessel(enei, kr, pos, r, z1, z2, ind)
        elif key == 2:
            # Real kr-axis
            kr = 2 * k1max / x
            return -2 * k1max * self._intbessel(enei, kr, pos, r, z1, z2, ind) / x**2
        elif key == 3:
            # Imaginary kr-axis
            kr = 2 * k1max * (1 - 1j + 1j / x)
            return -2j * k1max * self._inthankel(enei, kr, pos, r, z1, z2, ind) / x**2

    def _intbessel(self, enei: float, kpar: complex, pos: PositionStruct,
                   r: np.ndarray, z1: np.ndarray, z2: np.ndarray,
                   ind: np.ndarray = None) -> np.ndarray:
        """
        Integration for given k-parallel with Bessel functions.

        MATLAB: private/intbessel.m
        """
        if ind is None:
            ind = np.arange(r.size)

        # Wavenumber in media
        k = np.array([self._get_wavenumber(eps, enei) for eps in self.eps])

        # Perpendicular component of wavevector
        kz = np.sqrt(k**2 - kpar**2 + 0j)
        kz = kz * np.sign(np.imag(kz + 1e-10j))
        kz = kz[pos.ind1]

        # Reflection and transmission coefficients
        refl, reflz = self.reflection(enei, np.abs(kpar), pos)

        # Bessel functions
        r_flat = r.flatten()
        j0 = jv(0, kpar * r_flat)
        j1 = jv(1, kpar * r_flat)

        n = len(ind)
        y = np.zeros(15 * n, dtype=complex)

        # Loop over field names
        for iname, name in enumerate(refl.keys()):
            rr = np.atleast_1d(refl[name]).flatten()
            rrz = np.atleast_1d(reflz[name]).flatten()

            kz_flat = np.atleast_1d(kz).flatten()

            idx = slice(iname * 3 * n, iname * 3 * n + n)
            y[idx] = 1j * (j0[ind] * rr[ind] * kpar / kz_flat[ind])

            idx = slice(iname * 3 * n + n, iname * 3 * n + 2*n)
            y[idx] = 1j * (j1[ind] * rr[ind] * (-kpar**2) / kz_flat[ind])

            idx = slice(iname * 3 * n + 2*n, iname * 3 * n + 3*n)
            y[idx] = 1j * (j0[ind] * 1j * rrz[ind] * kpar)

        return np.real(y)

    def _inthankel(self, enei: float, kpar: complex, pos: PositionStruct,
                   r: np.ndarray, z1: np.ndarray, z2: np.ndarray,
                   ind: np.ndarray = None) -> np.ndarray:
        """
        Integration for given k-parallel with Hankel functions.

        MATLAB: private/inthankel.m
        """
        if ind is None:
            ind = np.arange(r.size)

        # Wavenumber in media
        k = np.array([self._get_wavenumber(eps, enei) for eps in self.eps])

        # Parallel components
        kpar1 = kpar
        kpar2 = np.conj(kpar)

        # Perpendicular component of wavevector
        kz1 = np.sqrt(k**2 - kpar1**2 + 0j)
        kz1 = kz1 * np.sign(np.imag(kz1 + 1e-10j))
        kz2 = np.sqrt(k**2 - kpar2**2 + 0j)
        kz2 = kz2 * np.sign(np.imag(kz2 + 1e-10j))

        kz1 = kz1[pos.ind1]
        kz2 = kz2[pos.ind1]

        # Reflection and transmission coefficients
        refl1, refl1z = self.reflection(enei, np.abs(kpar1), pos)
        refl2, refl2z = self.reflection(enei, np.abs(kpar2), pos)

        # Hankel functions
        r_flat = r.flatten()
        h0 = hankel1(0, kpar * r_flat)
        h1 = hankel1(1, kpar * r_flat)

        n = len(ind)
        y = np.zeros(15 * n, dtype=complex)

        # Loop over field names
        for iname, name in enumerate(refl1.keys()):
            rr1 = np.atleast_1d(refl1[name]).flatten()
            rr1z = np.atleast_1d(refl1z[name]).flatten()
            rr2 = np.atleast_1d(refl2[name]).flatten()
            rr2z = np.atleast_1d(refl2z[name]).flatten()

            kz1_flat = np.atleast_1d(kz1).flatten()
            kz2_flat = np.atleast_1d(kz2).flatten()

            idx = slice(iname * 3 * n, iname * 3 * n + n)
            y[idx] = (0.5j * h0[ind] * rr1[ind] * kpar1 / kz1_flat[ind] -
                      0.5j * np.conj(h0[ind]) * rr2[ind] * kpar2 / kz2_flat[ind])

            idx = slice(iname * 3 * n + n, iname * 3 * n + 2*n)
            y[idx] = (0.5j * h1[ind] * rr1[ind] * (-kpar1**2) / kz1_flat[ind] -
                      0.5j * np.conj(h1[ind]) * rr2[ind] * (-kpar2**2) / kz2_flat[ind])

            idx = slice(iname * 3 * n + 2*n, iname * 3 * n + 3*n)
            y[idx] = (0.5j * h0[ind] * 1j * rr1z[ind] * kpar1 -
                      0.5j * np.conj(h0[ind]) * 1j * rr2z[ind] * kpar2)

        return np.real(y)

    def _bemsolve(self, enei: float, kpar: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve BEM equation for layer structure.

        MATLAB: bemsolve.m
        """
        # Wavenumber of light in vacuum
        k0 = 2 * np.pi / enei

        # Dielectric function and wavenumber
        eps = np.array([self._get_eps(e, enei) for e in self.eps])
        k = np.array([self._get_wavenumber(e, enei) for e in self.eps])

        # Perpendicular component of wavevector
        kz = np.sqrt(k**2 - kpar**2 + 0j)
        kz = kz * np.sign(np.imag(kz + 1e-10j))

        # Number of interfaces
        n = len(self.z)

        # Intralayer Green function
        G0 = 2j * np.pi / kz

        # Interlayer Green function
        if n > 1:
            G = 2j * np.pi / kz[1:-1] * np.exp(1j * kz[1:-1] * np.abs(np.diff(self.z)))
        else:
            G = np.array([])

        # Parallel surface current
        siz = (2 * n, 2 * n)
        lhs = np.zeros(siz, dtype=complex)
        rhs = np.zeros(siz, dtype=complex)

        # Indices for [h2(mu), h1(mu+1), ...]
        i1 = np.arange(1, 2*n, 2)  # 0-indexed: 1, 3, 5, ...
        i2 = np.arange(0, 2*n, 2)  # 0-indexed: 0, 2, 4, ...

        # Indices for equations
        ind1 = np.arange(0, 2*n, 2)
        ind2 = np.arange(1, 2*n, 2)

        # Continuity of vector potential [Eq. (13a)]
        lhs[ind1, i1] = G0[1:]
        lhs[ind1, i2] = -G0[:-1]

        if n > 1:
            lhs[ind1[1:], i1[:-1]] = -G
            lhs[ind1[:-1], i2[1:]] = G

        rhs[ind1, i1] = -1
        rhs[ind1, i2] = 1

        # Continuity of derivative of vector potential [Eq. (13b)]
        lhs[ind2, i1] = 2j * np.pi
        lhs[ind2, i2] = 2j * np.pi

        if n > 1:
            lhs[ind2[1:], i1[:-1]] = -kz[1:-1] * G
            lhs[ind2[:-1], i2[1:]] = -kz[1:-1] * G

        rhs[ind2, i1] = kz[1:]
        rhs[ind2, i2] = kz[:-1]

        # Matrix for solution of BEM equation (parallel)
        par = np.linalg.solve(lhs, rhs)

        # Perpendicular surface current and surface charge
        siz = (4 * n, 4 * n)
        lhs = np.zeros(siz, dtype=complex)
        rhs = np.zeros(siz, dtype=complex)

        # Indices for surface current and charge
        i1 = np.arange(2, 4*n, 4)  # sig1
        i2 = np.arange(0, 4*n, 4)  # sig2
        j1 = np.arange(3, 4*n, 4)  # h1
        j2 = np.arange(1, 4*n, 4)  # h2

        # Indices for equations
        ind1 = np.arange(0, 4*n, 4)
        ind2 = np.arange(1, 4*n, 4)
        ind3 = np.arange(2, 4*n, 4)
        ind4 = np.arange(3, 4*n, 4)

        # Continuity of scalar potential [Eq. (14a)]
        lhs[ind1, i1] = G0[1:]
        lhs[ind1, i2] = -G0[:-1]

        if n > 1:
            lhs[ind1[1:], i1[:-1]] = -G
            lhs[ind1[:-1], i2[1:]] = G

        rhs[ind1, i1] = -1
        rhs[ind1, i2] = 1

        # Continuity of vector potential [Eq. (14b)]
        lhs[ind2, j1] = G0[1:]
        lhs[ind2, j2] = -G0[:-1]

        if n > 1:
            lhs[ind2[1:], j1[:-1]] = -G
            lhs[ind2[:-1], j2[1:]] = G

        rhs[ind2, j1] = -1
        rhs[ind2, j2] = 1

        # Continuity of dielectric displacement [Eq. (14c)]
        lhs[ind3, i1] = 2j * np.pi * eps[1:]
        lhs[ind3, i2] = 2j * np.pi * eps[:-1]
        lhs[ind3, j1] = k0 * G0[1:] * eps[1:]
        lhs[ind3, j2] = -k0 * G0[:-1] * eps[:-1]

        if n > 1:
            lhs[ind3[1:], i1[:-1]] = -kz[1:-1] * eps[1:-1] * G
            lhs[ind3[:-1], i2[1:]] = -kz[1:-1] * eps[1:-1] * G
            lhs[ind3[1:], j1[:-1]] = -k0 * eps[1:-1] * G
            lhs[ind3[:-1], j2[1:]] = k0 * eps[1:-1] * G

        rhs[ind3, i1] = kz[1:] * eps[1:]
        rhs[ind3, i2] = kz[:-1] * eps[:-1]
        rhs[ind3, j1] = -k0 * eps[1:]
        rhs[ind3, j2] = k0 * eps[:-1]

        # Continuity of derivative of vector potential [Eq. (14d)]
        lhs[ind4, j1] = 2j * np.pi
        lhs[ind4, j2] = 2j * np.pi
        lhs[ind4, i1] = k0 * G0[1:] * eps[1:]
        lhs[ind4, i2] = -k0 * G0[:-1] * eps[:-1]

        if n > 1:
            lhs[ind4[1:], j1[:-1]] = -kz[1:-1] * G
            lhs[ind4[:-1], j2[1:]] = -kz[1:-1] * G
            lhs[ind4[1:], i1[:-1]] = -k0 * eps[1:-1] * G
            lhs[ind4[:-1], i2[1:]] = k0 * eps[1:-1] * G

        rhs[ind4, j1] = kz[1:]
        rhs[ind4, j2] = kz[:-1]
        rhs[ind4, i1] = -k0 * eps[1:]
        rhs[ind4, i2] = k0 * eps[:-1]

        # Matrix for solution of BEM equation (perpendicular)
        perp = np.linalg.solve(lhs, rhs)

        return par, perp

    def tabspace(self, *args, **kwargs):
        """
        Compute suitable grids for tabulated r and z-values.

        MATLAB: tabspace.m

        Parameters
        ----------
        r : array-like
            Radial values [rmin, rmax, nr]
        z1 : array-like
            z-values [zmin, zmax, nz]
        z2 : array-like
            z-values [zmin, zmax, nz]

        OR

        p : Particle
            Particle or list of particles
        pt : Point, optional
            Points or list of points

        Returns
        -------
        tab : dict
            Dictionary with tabulated r, z1, z2 values
        """
        if len(args) > 0 and isinstance(args[0], (int, float, np.ndarray, list)):
            return self._tabspace1(*args, **kwargs)
        else:
            return self._tabspace2(*args, **kwargs)

    def _tabspace1(self, r, z1, z2, rmod='log', zmod='log'):
        """
        Compute grids for tabulated r and z-values (manual specification).

        MATLAB: private/tabspace1.m
        """
        tab = {}

        # Table for radii
        r = np.atleast_1d(r)
        if len(r) >= 3:
            rmin, rmax, nr = max(r[0], self.rmin), r[1], int(r[2])
            tab['r'] = self._linlogspace(rmin, rmax, nr, rmod)
        else:
            tab['r'] = r

        # Table for z1 values
        z1 = np.atleast_1d(z1)
        if len(z1) >= 3:
            z1_sorted = np.sort(self.round_z(z1[:2]))
            tab['z1'] = self._zlinlogspace(z1_sorted[0], z1_sorted[1], int(z1[2]), zmod)
        else:
            tab['z1'] = z1

        # Table for z2 values
        z2 = np.atleast_1d(z2)
        if len(z2) >= 3:
            z2_sorted = np.sort(self.round_z(z2[:2]))
            tab['z2'] = self._zlinlogspace(z2_sorted[0], z2_sorted[1], int(z2[2]), zmod)
        else:
            tab['z2'] = z2

        return tab

    def _tabspace2(self, p, pt=None, **kwargs):
        """
        Compute grids for tabulated r and z-values (from particles).

        MATLAB: private/tabspace2.m
        """
        scale = kwargs.get('scale', 1.05)
        nr = kwargs.get('nr', 30)
        nz = kwargs.get('nz', 30)

        if not isinstance(p, list):
            p = [p]

        # Get positions from particles
        pos_list = []
        for particle in p:
            if hasattr(particle, 'verts'):
                pos_list.append(particle.verts)
            elif hasattr(particle, 'pos'):
                pos_list.append(particle.pos)

        pos1 = np.vstack(pos_list)
        pos2 = pos1.copy()

        # Handle additional point argument
        if pt is not None:
            pt_pos = [pt.pos] if not isinstance(pt, list) else [x.pos for x in pt]
            pos1 = np.vstack([pos1] + pt_pos)

        # Get limits for radii and z-values
        from scipy.spatial.distance import cdist
        r_dist = cdist(pos1[:, :2], pos2[:, :2])

        z1 = pos1[:, 2]
        z2 = pos2[:, 2]
        ind1, _ = self.indlayer(z1)
        ind2, _ = self.indlayer(z2)

        n_layers = len(self.z) + 1
        tabs = []

        # nz can be scalar or array
        nz_arr = np.atleast_1d(nz)
        if len(nz_arr) == 1:
            nz_arr = np.repeat(nz_arr, n_layers)

        # Loop over layers
        for i1 in range(n_layers):
            for i2 in range(n_layers):
                mask1 = (ind1 == i1)
                mask2 = (ind2 == i2)

                if np.any(mask1) and np.any(mask2):
                    r_sub = r_dist[np.ix_(mask1, mask2)]
                    z1_sub = z1[mask1]
                    z2_sub = z2[mask2]

                    # Limits
                    ir = [r_sub.min(), r_sub.max()]
                    iz1 = [z1_sub.min(), z1_sub.max()]
                    iz2 = [z2_sub.min(), z2_sub.max()]

                    # Scale
                    ir = [max(np.mean(ir) - 0.5 * scale * (ir[1] - ir[0]), 0),
                          np.mean(ir) + 0.5 * scale * (ir[1] - ir[0])]

                    # Create tabspace
                    r_spec = [ir[0], ir[1], nr]
                    z1_spec = [iz1[0], iz1[1], nz_arr[i1]]
                    z2_spec = [iz2[0], iz2[1], nz_arr[i2]]

                    tab = self._tabspace1(r_spec, z1_spec, z2_spec, **kwargs)
                    tabs.append(tab)

        return tabs if len(tabs) > 1 else tabs[0] if tabs else {}

    def _linlogspace(self, xmin, xmax, n, key, x0=0):
        """
        Make table with linear or logarithmic spacing.

        MATLAB: private/tabspace1.m - linlogspace
        """
        if key == 'lin':
            return np.linspace(xmin, xmax, n)
        else:  # log
            return x0 + np.logspace(np.log10(xmin - x0), np.log10(xmax - x0), n)

    def _zlinlogspace(self, zmin, zmax, n, key):
        """
        Make table for heights.

        MATLAB: private/tabspace1.m - zlinlogspace
        """
        if key == 'lin':
            return np.linspace(zmin, zmax, n)

        # Logarithmic scale
        medium, _ = self.indlayer(zmin)

        if medium == 0:
            # Upper layer
            return self.z[0] + np.logspace(np.log10(zmin - self.z[0]),
                                           np.log10(zmax - self.z[0]), n)
        elif medium == len(self.z):
            # Lower medium
            z = self.z[-1] - np.logspace(np.log10(self.z[-1] - zmax),
                                         np.log10(self.z[-1] - zmin), n)
            return z[::-1]
        else:
            # Intermediate layer
            zup = self.z[medium - 1]
            zlo = self.z[medium]

            # z-value scaled to interval [-1, 1]
            zmin_scaled = 2 * (zmin - zlo) / (zup - zlo) - 1
            zmax_scaled = 2 * (zmax - zlo) / (zup - zlo) - 1

            # Table that is logarithmic at both ends
            z = np.tanh(np.linspace(np.arctanh(zmin_scaled), np.arctanh(zmax_scaled), n))

            # Scale to interval
            return 0.5 * (zup + zlo) + 0.5 * z * (zup - zlo)

    @staticmethod
    def options(**kwargs) -> LayerOptions:
        """
        Set options or use default options for layer structure.

        MATLAB: options.m

        Parameters
        ----------
        **kwargs : dict
            Options to override defaults

        Returns
        -------
        LayerOptions
            Options dataclass
        """
        return LayerOptions(**kwargs)

    def _get_eps(self, eps_func, enei: float) -> complex:
        """Get dielectric function value."""
        if callable(eps_func):
            result = eps_func(enei)
            if isinstance(result, tuple):
                return result[0]
            return result
        return eps_func

    def _get_wavenumber(self, eps_func, enei: float) -> complex:
        """Get wavenumber in medium."""
        if callable(eps_func):
            result = eps_func(enei)
            if isinstance(result, tuple):
                return result[1]
            # If only eps is returned, compute k = k0 * sqrt(eps)
            k0 = 2 * np.pi / enei
            return k0 * np.sqrt(result)
        k0 = 2 * np.pi / enei
        return k0 * np.sqrt(eps_func)

    def _mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Multiplication between arrays.

        MATLAB: private/mul.m
        """
        a = np.atleast_1d(a)
        b = np.atleast_1d(b)

        if a.shape == b.shape:
            return a * b
        else:
            # Outer product
            return np.outer(a, b).reshape(list(a.shape) + list(b.shape))

    def _multiply(self, pos: PositionStruct, a: np.ndarray,
                  b: np.ndarray) -> np.ndarray:
        """
        Multiply arrays based on position structure.

        MATLAB: reflection.m - multiply
        """
        z1 = np.atleast_1d(pos.z1)
        z2 = np.atleast_1d(pos.z2)
        ind1 = np.atleast_1d(pos.ind1)

        if z1.shape == z2.shape:
            # Direct product
            indices = (ind1, np.arange(a.shape[1]))
            return a[indices] * b
        else:
            # Outer product
            return a[ind1, :] * b[:, np.newaxis]

    def __repr__(self) -> str:
        """Command window display."""
        return f"LayerStructure(n_layers={self.n}, z={self.z})"
