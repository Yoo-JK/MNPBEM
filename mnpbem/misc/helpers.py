"""
Helper functions for vector and matrix operations.
"""

import numpy as np
from scipy import sparse
from typing import Union


def inner(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute inner product of two arrays along specified axis.

    Parameters
    ----------
    a : ndarray
        First array.
    b : ndarray
        Second array.
    axis : int, optional
        Axis along which to compute inner product. Default is -1.

    Returns
    -------
    ndarray
        Inner product result.
    """
    return np.sum(a * b, axis=axis)


def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute outer product of two arrays.

    Parameters
    ----------
    a : ndarray
        First array of shape (..., n).
    b : ndarray
        Second array of shape (..., m).

    Returns
    -------
    ndarray
        Outer product of shape (..., n, m).
    """
    return np.einsum('...i,...j->...ij', a, b)


def matcross(a: np.ndarray) -> np.ndarray:
    """
    Create cross product matrix from vector.

    For a vector [ax, ay, az], returns the matrix M such that
    M @ v = a x v (cross product).

    Parameters
    ----------
    a : ndarray
        Vector(s) of shape (..., 3).

    Returns
    -------
    ndarray
        Cross product matrix of shape (..., 3, 3).
    """
    if a.shape[-1] != 3:
        raise ValueError("Input must have last dimension of size 3")

    shape = a.shape[:-1] + (3, 3)
    result = np.zeros(shape, dtype=a.dtype)

    result[..., 0, 1] = -a[..., 2]
    result[..., 0, 2] = a[..., 1]
    result[..., 1, 0] = a[..., 2]
    result[..., 1, 2] = -a[..., 0]
    result[..., 2, 0] = -a[..., 1]
    result[..., 2, 1] = a[..., 0]

    return result


def matmul(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Matrix-vector multiplication with broadcasting.

    Parameters
    ----------
    mat : ndarray
        Matrix of shape (..., m, n) or sparse matrix.
    vec : ndarray
        Vector of shape (..., n).

    Returns
    -------
    ndarray
        Result of shape (..., m).
    """
    if sparse.issparse(mat):
        return mat @ vec
    return np.einsum('...ij,...j->...i', mat, vec)


def spdiag(diag: np.ndarray) -> sparse.csr_matrix:
    """
    Create sparse diagonal matrix.

    Parameters
    ----------
    diag : ndarray
        Diagonal elements.

    Returns
    -------
    sparse.csr_matrix
        Sparse diagonal matrix.
    """
    n = len(diag)
    return sparse.diags(diag, 0, shape=(n, n), format='csr')


def vecnorm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute vector norm along specified axis.

    Parameters
    ----------
    v : ndarray
        Input vectors.
    axis : int, optional
        Axis along which to compute norm. Default is -1.

    Returns
    -------
    ndarray
        Vector norms.
    """
    return np.sqrt(np.sum(v * v, axis=axis))


def vecnormalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalize vectors along specified axis.

    Parameters
    ----------
    v : ndarray
        Input vectors.
    axis : int, optional
        Axis along which to normalize. Default is -1.

    Returns
    -------
    ndarray
        Normalized vectors.
    """
    norms = vecnorm(v, axis=axis)
    # Keep dimensions for broadcasting
    norms = np.expand_dims(norms, axis=axis)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return v / norms


def bdist2(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared distances between two sets of points.

    Parameters
    ----------
    pos1 : ndarray
        First set of points, shape (n1, d).
    pos2 : ndarray
        Second set of points, shape (n2, d).

    Returns
    -------
    ndarray
        Squared distances, shape (n1, n2).
    """
    # Use broadcasting: (n1, 1, d) - (1, n2, d) -> (n1, n2, d)
    diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
    return np.sum(diff * diff, axis=-1)


def pdist2(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between two sets of points.

    Parameters
    ----------
    pos1 : ndarray
        First set of points, shape (n1, d).
    pos2 : ndarray
        Second set of points, shape (n2, d).

    Returns
    -------
    ndarray
        Distances, shape (n1, n2).
    """
    return np.sqrt(bdist2(pos1, pos2))
