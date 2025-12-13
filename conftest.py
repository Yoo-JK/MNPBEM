
"""
Pytest configuration for MNPBEM tests
"""

import numpy as np
import pytest


@pytest.fixture
def matlab_reference_dir():
    """Path to MATLAB reference data directory"""
    return 'tests/references'


@pytest.fixture
def rtol():
    """Relative tolerance for numerical comparisons"""
    return 1e-10


@pytest.fixture
def atol():
    """Absolute tolerance for numerical comparisons"""
    return 1e-12


def assert_allclose_complex(a, b, rtol=1e-10, atol=1e-12):
    """Assert two complex arrays are close"""
    np.testing.assert_allclose(a.real, b.real, rtol=rtol, atol=atol, err_msg="Real parts differ")
    np.testing.assert_allclose(a.imag, b.imag, rtol=rtol, atol=atol, err_msg="Imaginary parts differ")


def compare_with_matlab(python_result, matlab_file, var_name='result', rtol=1e-10):
    """
    Generic comparison function

    Args:
        python_result: Result from Python code
        matlab_file: Path to .mat file with MATLAB results
        var_name: Variable name in .mat file
        rtol: Relative tolerance
    """
    import scipy.io

    matlab_data = scipy.io.loadmat(matlab_file)
    matlab_result = matlab_data[var_name]

    # Handle complex arrays
    if np.iscomplexobj(python_result) or np.iscomplexobj(matlab_result):
        assert_allclose_complex(python_result, matlab_result, rtol=rtol)
    else:
        np.testing.assert_allclose(python_result, matlab_result, rtol=rtol)

    # Statistical report
    relative_error = np.abs((python_result - matlab_result) / (matlab_result + 1e-16))
    print(f"  Max relative error: {np.max(relative_error):.2e}")
    print(f"  Mean relative error: {np.mean(relative_error):.2e}")

    return True
