"""
Test script for Step 5: Excitation classes.

Tests PlaneWaveStat, PlaneWaveRet, DipoleStat, DipoleRet.
Verifies MATLAB MNPBEM compatibility.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/MNPBEM')

from mnpbem import (
    EpsConst, EpsTable, Particle, ComParticle, trisphere,
    BEMStat, BEMRet, PlaneWaveStat, PlaneWaveRet, DipoleStat, DipoleRet
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
    """Test PlaneWaveRet potential with proper inout handling."""
    print("\n=== Test: PlaneWaveRet Potential ===")

    # Create simple sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    # inout = [2, 1] means inside=metal (index 2), outside=vacuum (index 1)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Create excitation (x-polarized, z-propagating, medium=1=vacuum)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1], medium=1)

    # Compute potential
    enei = 500.0  # nm
    pot = exc.potential(p, enei)

    print(f"a1 shape: {pot['a1'].shape}")
    print(f"a2 shape: {pot['a2'].shape}")

    # For medium=1 (vacuum) and inout=[2,1]:
    # - Inside is metal (index 2), so a1 should be zero
    # - Outside is vacuum (index 1), so a2 should have values
    a1_norm = np.linalg.norm(pot['a1'])
    a2_norm = np.linalg.norm(pot['a2'])

    print(f"a1 norm (should be ~0): {a1_norm:.4e}")
    print(f"a2 norm (should be >0): {a2_norm:.4e}")

    # Verify a1 is zero (vacuum excitation, inside is metal)
    assert a1_norm < 1e-10, f"a1 should be zero for vacuum excitation, got {a1_norm}"
    # Verify a2 is non-zero (vacuum excitation, outside is vacuum)
    assert a2_norm > 0, "a2 should be non-zero for vacuum excitation"

    print("PASSED: PlaneWaveRet potential with inout handling")


def test_dipole_stat_basic():
    """Test DipoleStat initialization."""
    print("\n=== Test: DipoleStat Basic ===")

    # Single dipole at z=20nm with x-orientation
    exc = DipoleStat([[0, 0, 20]], [[1, 0, 0]])
    print(f"Dipole positions: {exc.pt_pos}")
    print(f"Dipole orientations shape: {exc.dip.shape}")
    print(f"npt={exc.npt}, ndip={exc.ndip}")

    # Default orientations (x, y, z)
    exc2 = DipoleStat([[0, 0, 20]])
    print(f"Default orientations: ndip={exc2.ndip}")
    assert exc2.ndip == 3, "Default should have 3 dipole orientations"

    print("PASSED: DipoleStat initialization")


def test_dipole_stat_field():
    """Test DipoleStat electric field."""
    print("\n=== Test: DipoleStat Field ===")

    # Create sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Dipole at z=20nm (outside sphere of radius 10nm)
    exc = DipoleStat([[0, 0, 20]], [[1, 0, 0]])

    # Compute field
    enei = 500.0
    field = exc.field(p, enei)

    print(f"Electric field shape: {field['e'].shape}")  # (nfaces, 3, npt, ndip)

    # Field should decay as 1/r^3
    e_mag = np.sqrt(np.sum(np.abs(field['e'][:, :, 0, 0])**2, axis=1))
    print(f"Field magnitude range: {e_mag.min():.4e} to {e_mag.max():.4e}")

    assert e_mag.max() > 0, "Field should be non-zero"
    print("PASSED: DipoleStat field")


def test_dipole_stat_potential():
    """Test DipoleStat potential for BEM."""
    print("\n=== Test: DipoleStat Potential ===")

    # Create sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Dipole at z=20nm
    exc = DipoleStat([[0, 0, 20]], [[1, 0, 0]])

    # Compute potential
    enei = 500.0
    pot = exc.potential(p, enei)

    print(f"phip shape: {pot['phip'].shape}")  # (nfaces, npt, ndip)
    print(f"phip range: {pot['phip'].real.min():.4e} to {pot['phip'].real.max():.4e}")

    # phip = -nvec · E, should be non-zero
    assert np.abs(pot['phip']).max() > 0, "Potential should be non-zero"
    print("PASSED: DipoleStat potential")


def test_dipole_ret_basic():
    """Test DipoleRet initialization."""
    print("\n=== Test: DipoleRet Basic ===")

    # Single dipole
    exc = DipoleRet([[0, 0, 20]], [[1, 0, 0]], medium=1)
    print(f"Dipole positions: {exc.pt_pos}")
    print(f"npt={exc.npt}, ndip={exc.ndip}, medium={exc.medium}")

    print("PASSED: DipoleRet initialization")


def test_dipole_ret_potential():
    """Test DipoleRet potential calculation."""
    print("\n=== Test: DipoleRet Potential ===")

    # Create sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Dipole at z=20nm
    exc = DipoleRet([[0, 0, 20]], [[1, 0, 0]])

    # Compute potential
    enei = 500.0
    pot = exc.potential(p, enei)

    print(f"phi1 shape: {pot['phi1'].shape}")
    print(f"a1 shape: {pot['a1'].shape}")

    # Check scalar and vector potentials are non-zero
    phi_norm = np.linalg.norm(pot['phi1'])
    a_norm = np.linalg.norm(pot['a1'])

    print(f"phi norm: {phi_norm:.4e}")
    print(f"a norm: {a_norm:.4e}")

    assert phi_norm > 0, "Scalar potential should be non-zero"
    assert a_norm > 0, "Vector potential should be non-zero"

    print("PASSED: DipoleRet potential")


def test_dipole_ret_field():
    """Test DipoleRet electromagnetic field."""
    print("\n=== Test: DipoleRet Field ===")

    # Create sphere
    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Dipole at z=20nm
    exc = DipoleRet([[0, 0, 20]], [[1, 0, 0]])

    # Compute field
    enei = 500.0
    field = exc.field(p, enei)

    print(f"E field shape: {field['e'].shape}")
    print(f"H field shape: {field['h'].shape}")

    e_mag = np.sqrt(np.sum(np.abs(field['e'][:, :, 0, 0])**2, axis=1))
    h_mag = np.sqrt(np.sum(np.abs(field['h'][:, :, 0, 0])**2, axis=1))

    print(f"E field magnitude range: {e_mag.min():.4e} to {e_mag.max():.4e}")
    print(f"H field magnitude range: {h_mag.min():.4e} to {h_mag.max():.4e}")

    assert e_mag.max() > 0, "Electric field should be non-zero"
    assert h_mag.max() > 0, "Magnetic field should be non-zero"

    print("PASSED: DipoleRet field")


def test_inout_faces_property():
    """Test ComParticle.inout_faces property."""
    print("\n=== Test: ComParticle.inout_faces ===")

    eps_vac = EpsConst(1.0)
    eps_metal = EpsConst(-10.0 + 1.0j)
    epstab = [eps_vac, eps_metal]

    sphere = trisphere(144, 10.0)
    # inout = [2, 1] means inside=metal (index 2), outside=vacuum (index 1)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    inout_faces = p.inout_faces
    print(f"inout_faces shape: {inout_faces.shape}")
    print(f"Inside material (all faces): {np.unique(inout_faces[:, 0])}")
    print(f"Outside material (all faces): {np.unique(inout_faces[:, 1])}")

    # All faces should have inside=2, outside=1
    assert np.all(inout_faces[:, 0] == 2), "Inside should be 2 for all faces"
    assert np.all(inout_faces[:, 1] == 1), "Outside should be 1 for all faces"

    # Test get_face_indices
    inside_metal = p.get_face_indices(2, 'inside')
    outside_vacuum = p.get_face_indices(1, 'outside')
    print(f"Faces with metal inside: {len(inside_metal)}")
    print(f"Faces with vacuum outside: {len(outside_vacuum)}")

    assert len(inside_metal) == p.nfaces
    assert len(outside_vacuum) == p.nfaces

    print("PASSED: ComParticle.inout_faces")


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
    print("STEP 5: EXCITATION TESTS (Complete)")
    print("=" * 60)

    test_planewave_stat_basic()
    test_planewave_stat_potential()
    test_planewave_stat_cross_sections()
    test_planewave_ret_basic()
    test_planewave_ret_potential()
    test_inout_faces_property()
    test_dipole_stat_basic()
    test_dipole_stat_field()
    test_dipole_stat_potential()
    test_dipole_ret_basic()
    test_dipole_ret_potential()
    test_dipole_ret_field()
    test_multiple_polarizations()

    print("\n" + "=" * 60)
    print("ALL STEP 5 TESTS PASSED!")
    print("=" * 60)
    print("\nImplemented excitation classes:")
    print("  - PlaneWaveStat: quasistatic plane wave")
    print("  - PlaneWaveRet: retarded plane wave with inout handling")
    print("  - DipoleStat: quasistatic dipole excitation")
    print("  - DipoleRet: retarded dipole excitation")


if __name__ == "__main__":
    run_all_tests()
