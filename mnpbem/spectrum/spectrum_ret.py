"""
Spectrum class for computing far-fields and scattering cross sections.

Matches MATLAB MNPBEM @spectrumret implementation.
"""

import numpy as np


def trisphere_unit(n_faces=144):
    """
    Create unit sphere mesh for far-field integration.

    Parameters
    ----------
    n_faces : int
        Approximate number of faces (will be adjusted for icosahedron subdivision)

    Returns
    -------
    verts : ndarray, shape (nverts, 3)
        Vertex positions on unit sphere
    faces : ndarray, shape (nfaces, 3)
        Face connectivity
    nvec : ndarray, shape (nfaces, 3)
        Outward normal vectors (same as face centroids for unit sphere)
    area : ndarray, shape (nfaces,)
        Solid angle of each face (area on unit sphere)
    """
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    verts = np.array([
        [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [phi, 0, -1], [phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)

    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # Icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=int)

    # Subdivide until we have approximately n_faces
    while len(faces) < n_faces:
        verts, faces = _subdivide_sphere(verts, faces)

    # Compute face properties
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Face centroids (normalized to unit sphere = outward normal)
    centroids = (v0 + v1 + v2) / 3
    nvec = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Solid angles (area on unit sphere)
    # For spherical triangle: area = |cross(v1-v0, v2-v0)| / 2, projected to sphere
    cross_prod = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(cross_prod, axis=1) / 2

    return verts, faces, nvec, area


def _subdivide_sphere(verts, faces):
    """Subdivide icosphere by splitting each face into 4."""
    edge_midpoints = {}
    new_verts = list(verts)

    def get_midpoint(i1, i2):
        key = tuple(sorted([i1, i2]))
        if key not in edge_midpoints:
            mid = (verts[i1] + verts[i2]) / 2
            mid = mid / np.linalg.norm(mid)  # Project to sphere
            edge_midpoints[key] = len(new_verts)
            new_verts.append(mid)
        return edge_midpoints[key]

    new_faces = []
    for f in faces:
        a, b, c = f
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.extend([
            [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
        ])

    return np.array(new_verts), np.array(new_faces, dtype=int)


class SpectrumRet:
    """
    Compute far-fields and scattering cross sections.

    Uses a discretized unit sphere to compute total and differential
    scattering cross sections by integrating the Poynting vector.

    Parameters
    ----------
    pinfty : Particle or ndarray or int, optional
        - Particle: Use its nvec and area properties
        - ndarray (ndir, 3): Light propagation directions
        - int: Number of faces for unit sphere (creates trisphere)
        - None: Creates trisphere(256, 2) like MATLAB default
    medium : int, optional
        Index of embedding medium (1-indexed, default: 1)

    Attributes
    ----------
    pinfty : object
        Discretized unit sphere (stores nvec, area)
    nvec : ndarray, shape (ndir, 3)
        Directions on unit sphere (outward normals)
    area : ndarray, shape (ndir,)
        Solid angles for each direction
    medium : int
        Embedding medium index

    Notes
    -----
    MATLAB equivalent: @spectrumret class

    Scattering cross section is computed from Poynting vector:
        dsca = 0.5 * real(nvec · (E × conj(H)))
        sca = sum(area * dsca)

    Examples
    --------
    >>> spec = SpectrumRet(144)
    >>> sca, dsca = spec.scattering(sig)
    """

    def __init__(self, pinfty=None, medium=1):
        """
        Initialize spectrum calculator.

        MATLAB: spectrumret(pinfty, op, PropertyPair)

        Parameters
        ----------
        pinfty : Particle, ndarray, or int, optional
            Unit sphere specification (see class docstring)
        medium : int
            Embedding medium index (1-indexed)
        """
        self.medium = medium

        # Handle different input types like MATLAB init.m
        if pinfty is None:
            # Default: create trisphere(256, 2)
            _, _, nvec, area = trisphere_unit(256)
            self.pinfty = _PinftyStruct(nvec, area)
        elif isinstance(pinfty, int):
            # Integer: create unit sphere with given number of faces
            _, _, nvec, area = trisphere_unit(pinfty)
            self.pinfty = _PinftyStruct(nvec, area)
        elif isinstance(pinfty, np.ndarray):
            # Numeric array: treat as direction vectors
            nvec = np.atleast_2d(pinfty)
            # For direction vectors, compute area as uniform solid angles
            area = np.full(nvec.shape[0], 4 * np.pi / nvec.shape[0])
            self.pinfty = _PinftyStruct(nvec, area)
        elif hasattr(pinfty, 'nvec') and hasattr(pinfty, 'area'):
            # Particle or struct with nvec and area properties
            self.pinfty = pinfty
        else:
            # Try to use as particle
            _, _, nvec, area = trisphere_unit(256)
            self.pinfty = _PinftyStruct(nvec, area)

        # Expose nvec and area directly for convenience
        self.nvec = self.pinfty.nvec if hasattr(self.pinfty, 'nvec') else self.pinfty['nvec']
        self.area = self.pinfty.area if hasattr(self.pinfty, 'area') else self.pinfty['area']
        self.ndir = len(self.nvec)


class _PinftyStruct:
    """Simple struct to hold pinfty data."""

    def __init__(self, nvec, area):
        self.nvec = nvec
        self.area = area


# Add methods to SpectrumRet class
def _farfield(self, sig, direction=None):
    """
    Compute far-field amplitudes from surface charges and currents.

    MATLAB: Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010), Eq. (50)

    Parameters
    ----------
    sig : dict
        Solution containing:
        - 'sig1', 'sig2': surface charges
        - 'h1', 'h2': surface currents
        - 'p': particle
        - 'enei': wavelength
    direction : ndarray, optional
        Light propagation directions. Defaults to self.nvec.

    Returns
    -------
    field : dict
        Far-field with 'e' and 'h' arrays
    """
    if direction is None:
        direction = self.nvec

    p = sig['p']
    enei = sig['enei']

    # Wavenumber
    _, k = p.eps[self.medium - 1](enei)
    k0 = 2 * np.pi / enei

    pos = p.pos
    area = p.area
    inout_faces = p.inout_faces

    # Get charges and currents
    sig1 = sig.get('sig1', np.zeros(p.nfaces))
    sig2 = sig.get('sig2', np.zeros(p.nfaces))
    h1 = sig.get('h1', np.zeros((p.nfaces, 3)))
    h2 = sig.get('h2', np.zeros((p.nfaces, 3)))

    # Ensure proper shape
    if sig1.ndim == 1:
        sig1 = sig1[:, np.newaxis]
        sig2 = sig2[:, np.newaxis]
    if h1.ndim == 2:
        h1 = h1[:, :, np.newaxis]
        h2 = h2[:, :, np.newaxis]

    npol = sig1.shape[1] if sig1.ndim > 1 else 1
    ndir = len(direction)

    # Initialize far-field
    e = np.zeros((ndir, 3, npol), dtype=complex)
    h = np.zeros((ndir, 3, npol), dtype=complex)

    # Phase factor: exp(-i*k*dir·pos) * area
    # MATLAB: phase = exp(-1i * k * dir * p.pos') * spdiag(p.area)
    phase = np.exp(-1j * k * np.dot(direction, pos.T)) * area  # (ndir, nfaces)

    # Find faces connected to medium
    # MATLAB: ind = p.index(find(p.inout(:, 1) == obj.medium)')
    # Inside contribution: faces where inside == medium
    ind1 = np.where(inout_faces[:, 0] == self.medium)[0]
    # Outside contribution: faces where outside == medium
    ind2 = np.where(inout_faces[:, 1] == self.medium)[0]

    for ipol in range(npol):
        # Inside surface contribution
        # MATLAB: e = 1i*k0 * matmul(phase(:,ind), sig.h1(ind,:,:)) -
        #             1i*k * outer(dir, matmul(phase(:,ind), sig.sig1(ind,:)))
        if len(ind1) > 0:
            phase1 = phase[:, ind1]  # (ndir, nind1)
            # Current term: i*k0 * phase @ h
            h_term = 1j * k0 * np.dot(phase1, h1[ind1, :, ipol])  # (ndir, 3)
            # Charge term: -i*k * dir * (phase @ sig)
            sig_term = np.dot(phase1, sig1[ind1, ipol])  # (ndir,)
            e_term = -1j * k * direction * sig_term[:, np.newaxis]

            e[:, :, ipol] += h_term + e_term
            # Magnetic field: H = i*k * cross(dir, matmul(phase, h))
            h[:, :, ipol] += 1j * k * np.cross(
                direction, np.dot(phase1, h1[ind1, :, ipol])
            )

        # Outside surface contribution
        if len(ind2) > 0:
            phase2 = phase[:, ind2]
            h_term = 1j * k0 * np.dot(phase2, h2[ind2, :, ipol])
            sig_term = np.dot(phase2, sig2[ind2, ipol])
            e_term = -1j * k * direction * sig_term[:, np.newaxis]

            e[:, :, ipol] += h_term + e_term
            h[:, :, ipol] += 1j * k * np.cross(
                direction, np.dot(phase2, h2[ind2, :, ipol])
            )

    return {
        'e': e,
        'h': h,
        'nvec': direction,
        'area': self.area if direction is self.nvec else np.full(ndir, 4*np.pi/ndir),
        'enei': enei,
        'k': k
    }


def _scattering(self, sig):
    """
    Compute scattering cross section.

    MATLAB:
        dsca = 0.5 * real(inner(nvec, cross(e, conj(h))))
        sca = matmul(area, dsca)

    Parameters
    ----------
    sig : dict
        Solution from BEM solver

    Returns
    -------
    sca : ndarray
        Total scattering cross section
    dsca : ndarray
        Differential scattering (per solid angle)
    """
    # Get far-field
    field = self.farfield(sig)
    e = field['e']  # (ndir, 3, npol)
    h = field['h']  # (ndir, 3, npol)

    if e.ndim == 2:
        e = e[:, :, np.newaxis]
        h = h[:, :, np.newaxis]

    npol = e.shape[2]

    # Poynting vector component in radial direction
    # dsca = 0.5 * real(nvec · (E × conj(H)))
    dsca = np.zeros((self.ndir, npol))

    for ipol in range(npol):
        # Cross product E × conj(H)
        poynting = np.cross(e[:, :, ipol], np.conj(h[:, :, ipol]))  # (ndir, 3)
        # Dot with nvec
        dsca[:, ipol] = 0.5 * np.real(np.sum(self.nvec * poynting, axis=1))

    # Total scattering: integrate over sphere
    sca = np.dot(self.area, dsca)  # (npol,)

    if npol == 1:
        sca = sca[0]
        dsca = dsca[:, 0]

    return sca, dsca


def _spectrumret_repr(self):
    return f"SpectrumRet(ndir={self.ndir}, medium={self.medium})"


# Attach methods to SpectrumRet class
SpectrumRet.farfield = _farfield
SpectrumRet.scattering = _scattering
SpectrumRet.__repr__ = _spectrumret_repr
