"""
Utility functions for particle shape generation.
"""

import numpy as np
from typing import Tuple


def fvgrid(
    u: np.ndarray,
    v: np.ndarray,
    periodic_u: bool = False,
    periodic_v: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create face-vertex grid from parameter arrays.

    Parameters
    ----------
    u : ndarray
        First parameter values.
    v : ndarray
        Second parameter values.
    periodic_u : bool
        Whether u is periodic (wraps around).
    periodic_v : bool
        Whether v is periodic.

    Returns
    -------
    verts : ndarray
        Vertices as (u, v) pairs, shape (nu*nv, 2).
    faces : ndarray
        Quadrilateral faces, shape (n_faces, 4).
    """
    nu, nv = len(u), len(v)

    # Create vertex grid
    uu, vv = np.meshgrid(u, v, indexing='ij')
    verts = np.column_stack([uu.ravel(), vv.ravel()])

    # Create faces
    faces = []
    nu_faces = nu if periodic_u else nu - 1
    nv_faces = nv if periodic_v else nv - 1

    for i in range(nu_faces):
        for j in range(nv_faces):
            # Vertex indices for this face
            i1 = i * nv + j
            i2 = i * nv + (j + 1) % nv if periodic_v else i * nv + j + 1
            i3 = ((i + 1) % nu) * nv + (j + 1) % nv if periodic_u else (i + 1) * nv + j + 1
            i4 = ((i + 1) % nu) * nv + j if periodic_u else (i + 1) * nv + j

            if not periodic_v and j == nv - 1:
                continue
            if not periodic_u and i == nu - 1:
                continue

            faces.append([i1, i2, i3, i4])

    return verts, np.array(faces)


def cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : ndarray
        Cartesian coordinates.

    Returns
    -------
    r : ndarray
        Radial distance.
    theta : ndarray
        Polar angle (from z-axis).
    phi : ndarray
        Azimuthal angle.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / np.where(r > 0, r, 1), -1, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph2cart(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical to Cartesian coordinates.

    Parameters
    ----------
    r : ndarray
        Radial distance.
    theta : ndarray
        Polar angle.
    phi : ndarray
        Azimuthal angle.

    Returns
    -------
    x, y, z : ndarray
        Cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
