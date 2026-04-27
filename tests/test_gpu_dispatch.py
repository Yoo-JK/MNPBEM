"""Unit tests for GPU dispatch (CPU vs GPU LU consistency)."""

import os

import numpy as np
import pytest


def _solve(A, b, env_value):
    os.environ["MNPBEM_GPU"] = env_value
    import importlib
    import mnpbem.utils.gpu as gmod

    importlib.reload(gmod)
    piv = gmod.lu_factor_dispatch(A.copy())
    x = gmod.lu_solve_dispatch(piv, b.copy())
    return x, piv[0]


def test_gpu_vs_cpu_lu_solve_2000x2000():
    rng = np.random.default_rng(0)
    N = 2000
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    b = rng.standard_normal((N, 4)) + 1j * rng.standard_normal((N, 4))

    x_cpu, tag_cpu = _solve(A, b, "0")
    assert tag_cpu == "cpu"

    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("cupy not installed")

    x_gpu, tag_gpu = _solve(A, b, "1")
    if tag_gpu == "cpu":
        pytest.skip("GPU threshold not crossed or cupy disabled")

    rel = np.max(np.abs(x_gpu - x_cpu)) / np.max(np.abs(x_cpu))
    assert rel < 1e-12, f"rel error {rel:.3e} exceeds 1e-12"


def test_threshold_uses_cpu_for_small_matrix():
    os.environ["MNPBEM_GPU"] = "1"
    os.environ["MNPBEM_GPU_THRESHOLD"] = "1500"
    import importlib
    import mnpbem.utils.gpu as gmod

    importlib.reload(gmod)
    rng = np.random.default_rng(1)
    N = 200
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    piv = gmod.lu_factor_dispatch(A)
    assert piv[0] == "cpu"
