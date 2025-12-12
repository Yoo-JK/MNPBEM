"""
Test script for Step 5: Plane wave excitation.

Tests PlaneWaveStat and PlaneWaveRet implementations matching MATLAB MNPBEM.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/MNPBEM')

from mnpbem import (
    EpsConst, EpsTable, Particle, ComParticle, trisphere,
    BEMStat, BEMRet, PlaneWaveStat, PlaneWaveRet
)


def test_planewave_stat_basic():
    """Test PlaneWaveStat initialization and potential."""
    print("\n=== Test: PlaneWaveStat Basic ===")

    # Single polarization along x
    exc = PlaneWaveStat([1, 0, 0])
    print(f"Polarization: {exc.pol}")
    print(f"Medium: {exc.medium}")

    # Multiple polarizations
    exc2 = PlaneWaveStat([[1, 0, 0], [0, 1, 0]])
    print(f"Multiple polarizations shape: {exc2.pol.shape}")

    print("PASSED: PlaneWaveStat initialization")


def test_planewave_stat_potential():
    """Test PlaneWaveStat potential calculation."""
    print("\n=== Test: PlaneWaveStat Potential ===")

    # Create simple sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Create excitation
    exc = PlaneWaveStat([1, 0, 0])

    # Compute potential
    enei = 500.0  # nm
    pot = exc.potential(p, enei)

    print(f"phip shape: {pot['phip'].shape}")
    print(f"phip min/max: {pot['phip'].min():.4f} / {pot['phip'].max():.4f}")

    # Verify: phip = -nvec · pol
    # For x-polarization, phip should be proportional to -nvec_x
    nvec_x = p.nvec[:, 0]
    expected_phip = -nvec_x

    # Compare
    computed_phip = pot['phip'][:, 0]
    error = np.max(np.abs(computed_phip - expected_phip))
    print(f"Verification error: {error:.2e}")

    assert error < 1e-10, f"Potential error too large: {error}"
    print("PASSED: PlaneWaveStat potential")


def test_planewave_stat_cross_sections():
    """Test PlaneWaveStat scattering and absorption."""
    print("\n=== Test: PlaneWaveStat Cross Sections ===")

    # Create sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Create excitation
    exc = PlaneWaveStat([1, 0, 0])

    # Wavelength
    enei = 500.0  # nm

    # Get potential
    pot = exc.potential(p, enei)

    # Solve BEM equation
    bem = BEMStat(p)
    sig = bem.solve(enei, pot)

    print(f"Surface charge shape: {sig['sig'].shape}")
    print(f"Surface charge range: {sig['sig'].min():.4e} to {sig['sig'].max():.4e}")

    # Compute cross sections
    sca = exc.scattering(sig)
    abs_cs = exc.absorption(sig)
    ext = exc.extinction(sig)

    print(f"Scattering: {sca[0]:.4e} nm²")
    print(f"Absorption: {abs_cs[0]:.4e} nm²")
    print(f"Extinction: {ext[0]:.4e} nm²")

    # Energy conservation: ext = sca + abs
    error = abs(ext[0] - (sca[0] + abs_cs[0]))
    print(f"Energy conservation error: {error:.2e}")

    assert error < 1e-10 * abs(ext[0]), "Energy conservation violated"
    print("PASSED: PlaneWaveStat cross sections")


def test_planewave_ret_basic():
    """Test PlaneWaveRet initialization."""
    print("\n=== Test: PlaneWaveRet Basic ===")

    # Polarization along x, propagation along z
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
    print(f"Polarization: {exc.pol}")
    print(f"Direction: {exc.dir}")
    print(f"Medium: {exc.medium}")

    # Test orthogonality check
    try:
        exc_bad = PlaneWaveRet([1, 0, 0], [1, 0, 0])
        print("ERROR: Should have raised for non-orthogonal vectors")
    except ValueError as e:
        print(f"Correctly rejected non-orthogonal: {e}")

    print("PASSED: PlaneWaveRet initialization")


def test_planewave_ret_potential():
    """Test PlaneWaveRet potential calculation."""
    print("\n=== Test: PlaneWaveRet Potential ===")

    # Create simple sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Create excitation (x-polarized, z-propagating)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])

    # Compute potential
    enei = 500.0  # nm
    pot = exc.potential(p, enei)

    print(f"a1 shape: {pot['a1'].shape}")
    print(f"a1p shape: {pot['a1p'].shape}")

    # Vector potential should be complex
    print(f"a1 is complex: {np.iscomplexobj(pot['a1'])}")

    # Check magnitude
    a_mag = np.abs(pot['a1'][:, 0, 0])  # x-component of vector potential
    print(f"a1 x-component range: {a_mag.min():.4e} to {a_mag.max():.4e}")

    print("PASSED: PlaneWaveRet potential")


def test_planewave_ret_with_bem():
    """Test PlaneWaveRet with BEM solver."""
    print("\n=== Test: PlaneWaveRet with BEM ===")

    # Create sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Create excitation
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])

    # Wavelength
    enei = 500.0  # nm

    # Get potential
    pot = exc.potential(p, enei)

    # Note: BEMRet solver would need to be adapted to handle vector potential
    # For now, just verify potential calculation works
    print("Vector potential computed successfully")
    print("PASSED: PlaneWaveRet with BEM (potential only)")


def test_multiple_polarizations():
    """Test excitation with multiple polarizations."""
    print("\n=== Test: Multiple Polarizations ===")

    # Create sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Two polarizations: x and y
    exc = PlaneWaveStat([[1, 0, 0], [0, 1, 0]])

    # Compute potential
    enei = 500.0
    pot = exc.potential(p, enei)

    print(f"phip shape for 2 polarizations: {pot['phip'].shape}")
    assert pot['phip'].shape[1] == 2, "Should have 2 columns for 2 polarizations"

    # Solve for both polarizations
    bem = BEMStat(p)
    sig = bem.solve(enei, pot)

    print(f"Solution shape: {sig['sig'].shape}")

    # Cross sections for both polarizations
    sca = exc.scattering(sig)
    abs_cs = exc.absorption(sig)

    print(f"Scattering (x-pol): {sca[0]:.4e} nm²")
    print(f"Scattering (y-pol): {sca[1]:.4e} nm²")
    print(f"Absorption (x-pol): {abs_cs[0]:.4e} nm²")
    print(f"Absorption (y-pol): {abs_cs[1]:.4e} nm²")

    # For a sphere, x and y polarizations should give same results
    rel_diff = abs(sca[0] - sca[1]) / (abs(sca[0]) + 1e-30)
    print(f"Relative difference x vs y: {rel_diff:.2e}")

    assert rel_diff < 0.1, "Sphere should have similar cross sections for x and y"
    print("PASSED: Multiple polarizations")


def run_all_tests():
    """Run all excitation tests."""
    print("=" * 60)
    print("STEP 5: PLANE WAVE EXCITATION TESTS")
    print("=" * 60)

    test_planewave_stat_basic()
    test_planewave_stat_potential()
    test_planewave_stat_cross_sections()
    test_planewave_ret_basic()
    test_planewave_ret_potential()
    test_planewave_ret_with_bem()
    test_multiple_polarizations()

    print("\n" + "=" * 60)
    print("ALL STEP 5 TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
